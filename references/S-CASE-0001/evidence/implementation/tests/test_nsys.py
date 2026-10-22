"""Behavioral checks for partial traces, overlap and process-scoped attribution."""

from contextlib import closing, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from infraagents.cli import main
from infraagents.nsys import analyze_nsys


PROCESS = (1 << 48) | (42 << 24)
OTHER_PROCESS = (1 << 48) | (99 << 24)


def fixture(path):
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE META_DATA_EXPORT(name TEXT, value TEXT);
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
            INSERT INTO StringIds VALUES(1, 'kernel-a'),(2, 'kernel-b'),(3, 'launch');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, globalPid INTEGER,
                deviceId INTEGER, streamId INTEGER, correlationId INTEGER, demangledName INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER, globalPid INTEGER,
                deviceId INTEGER, streamId INTEGER, correlationId INTEGER, bytes INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER,
                correlationId INTEGER, nameId INTEGER);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT);
            INSERT INTO NVTX_EVENTS VALUES(100, NULL, 'point'),(110, 140, 'range'),(400, 300, 'invalid');
        """)
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?)", [
            (100, 200, PROCESS, 0, 7, 7, 1), (150, 250, PROCESS, 0, 8, 8, 2),
            (400, 450, PROCESS, 0, 7, 9, 1), (110, 120, OTHER_PROCESS, 0, 7, 7, 1),
        ])
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?,?,?,?,?)", (250, 300, PROCESS, 0, 7, 10, 128))
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)", [
            (80, 90, PROCESS | 5, 7, 3), (100, 160, PROCESS | 5, 8, 3),
            (390, 395, PROCESS | 5, 9, 3), (105, 115, OTHER_PROCESS | 5, 7, 3),
            # Same PID and correlationId on another VM must not match.
            (80, 90, (2 << 48) | (42 << 24) | 5, 7, 3),
        ])


class NsysTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "capture.sqlite"
        fixture(self.source)

    def analyze(self, name="analysis", **kwargs):
        result = analyze_nsys(self.source, self.directory / name, **kwargs)
        self.assertEqual(result["status"], "ok", result.get("failure"))
        return result

    def test_overlap_union_is_separate_from_cumulative_time_and_other_processes(self):
        report = self.analyze()
        groups = report["device_intervals"]["groups"]
        self.assertEqual(len(groups), 2)
        group = next(item for item in groups if item["globalPid"] == PROCESS)
        self.assertEqual(group["summed_event_duration_ns"], 300)
        self.assertEqual(group["observed_interval_union_ns"], 250)
        self.assertEqual(group["span_ns"], 350)
        self.assertEqual(group["uncovered_within_span_ns"], 100)
        self.assertEqual(group["overlap_excess_ns"], 50)
        self.assertEqual(group["largest_uncovered_intervals"], [{"start_ns": 300, "end_ns": 400, "duration_ns": 100}])

    def test_window_clips_aggregates_but_correlates_api_outside_window(self):
        report = self.analyze(start_ns=175, end_ns=425)
        group = report["device_intervals"]["groups"][0]
        self.assertEqual(group["observed_interval_union_ns"], 150)
        self.assertEqual(group["summed_event_duration_ns"], 175)
        kernels = report["signals"]["kernel"]
        self.assertEqual(kernels["selected_interval_rows"], 3)
        first = kernels["events"][0]
        self.assertEqual(first["start"], 100)  # Original event, not clipped provenance.
        self.assertEqual(first["host_api"]["status"], "unique")
        self.assertEqual(first["host_api"]["candidates"][0]["start"], 80)
        self.assertEqual(first["host_api"]["candidate_count"], 1)

    def test_ambiguous_correlations_preserve_candidates_and_negative_time_deltas(self):
        with closing(sqlite3.connect(self.source)) as db, db:
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(100,120,?,?,3)", (PROCESS | 6, 7))
        report = self.analyze()
        events = report["signals"]["kernel"]["events"]
        self.assertEqual(events[0]["host_api"]["status"], "ambiguous")
        self.assertEqual(events[0]["host_api"]["candidate_count"], 2)
        second = next(event for event in events if event["correlationId"] == 8)
        self.assertEqual(second["host_api"]["gpu_start_minus_api_end_ns"], -10)

    def test_absent_empty_unsupported_and_incomplete_are_distinct(self):
        with closing(sqlite3.connect(self.source)) as db, db:
            db.executescript("""CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER, end INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(timestamp INTEGER);""")
        report = self.analyze()
        signals = report["signals"]
        self.assertEqual(signals["synchronization"]["status"], "table_absent")
        self.assertEqual(signals["cuda_driver"]["status"], "no_completed_intervals_in_selection")
        self.assertEqual(signals["memset"]["status"], "unsupported_schema")
        self.assertEqual(signals["nvtx"]["rows_excluded_from_interval_analysis"], 2)
        self.assertEqual(signals["nvtx"]["selected_interval_rows"], 1)
        self.assertIn("capture_completeness", report["unverified"])

    def test_limits_do_not_truncate_counts_or_interval_coverage(self):
        report = self.analyze(limit=1)
        kernel = report["signals"]["kernel"]
        self.assertEqual(kernel["selected_interval_rows"], 4)
        self.assertEqual(len(kernel["events"]), 1)
        self.assertTrue(kernel["events_truncated"])
        self.assertTrue(kernel["hotspots_truncated"])
        self.assertEqual(report["device_intervals"]["groups"][0]["observed_interval_union_ns"], 250)

    def test_source_is_unchanged_and_queries_reproduce_rows(self):
        before = self.source.read_bytes()
        report = self.analyze()
        self.assertEqual(before, self.source.read_bytes())
        self.assertEqual(report["source"]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual({path.name for path in self.directory.iterdir()}, {"capture.sqlite", "analysis"})
        queries = [json.loads(line) for line in (self.directory / "analysis/queries.jsonl").read_text().splitlines()]
        with closing(sqlite3.connect(self.source.as_uri() + "?mode=ro", uri=True)) as db:
            for query in queries:
                db.execute(query["sql"], query["parameters"]).fetchall()
        first = report["signals"]["kernel"]["events"][0]
        self.assertEqual(first["source_rowid"], 1)

    def test_corrupt_source_failure_and_previous_output_are_retained(self):
        self.source.write_bytes(b"broken SQLite export")
        result = analyze_nsys(self.source, self.directory / "failed")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure"]["type"], "DatabaseError")
        self.assertIn("Traceback", result["failure"]["traceback"])
        saved = (self.directory / "failed/report.json").read_bytes()
        with self.assertRaises(FileExistsError):
            analyze_nsys(self.source, self.directory / "failed")
        self.assertEqual(saved, (self.directory / "failed/report.json").read_bytes())
        self.assertEqual(self.source.read_bytes(), b"broken SQLite export")

    def test_invalid_requests_make_no_output(self):
        for options in ({"start_ns": 5, "end_ns": 5}, {"limit": 0}, {"limit": 201}, {"start_ns": 2**63}, {"start_ns": True}):
            with self.assertRaises(ValueError):
                analyze_nsys(self.source, self.directory / "invalid", **options)
            self.assertFalse((self.directory / "invalid").exists())
        context = self.directory / "context.json"
        context.write_text("null")
        with self.assertRaises(ValueError):
            analyze_nsys(self.source, self.directory / "invalid", context=context)

    def test_unknown_process_is_not_merged_and_live_sidecars_are_rejected(self):
        with closing(sqlite3.connect(self.source)) as db, db:
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100,900,NULL,0,0,1,1)")
        report = self.analyze()
        self.assertEqual(len(report["device_intervals"]["groups"]), 2)
        self.assertEqual(report["device_intervals"]["unidentified_rows_excluded"]["CUPTI_ACTIVITY_KIND_KERNEL"], 1)
        sidecar = Path(str(self.source) + "-wal")
        sidecar.write_bytes(b"preserve")
        result = analyze_nsys(self.source, self.directory / "sidecar")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(sidecar.read_bytes(), b"preserve")

    def test_no_trace_cli_reports_absence_without_zero_work_claim(self):
        source = self.directory / "no-trace.sqlite"
        with closing(sqlite3.connect(source)) as db, db:
            db.execute("CREATE TABLE META_DATA_EXPORT(name TEXT, value TEXT)")
        context = self.directory / "context.json"
        context.write_text(json.dumps({"capture": {"signals_requested": ["none"]}}))
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = main(["analyze-nsys", str(source), "--context", str(context), "--output-dir", str(self.directory / "cli"), "--json"])
        self.assertEqual(code, 0)
        report = json.loads(stream.getvalue())
        self.assertEqual(report["context"]["status"], "declared_not_verified")
        self.assertEqual(report["signals"]["kernel"]["status"], "table_absent")
        self.assertNotIn("selected_interval_rows", report["signals"]["kernel"])
        self.assertEqual(report["device_intervals"]["groups"], [])
        self.assertTrue(Path(report["report_path"]).is_file())
