"""Read native Nsight SQLite exports; collection stays with the native CLI.

Only observed, completed intervals are analyzed. Missing tables, unsupported
schemas and empty selections are evidence states, never zero-work assertions.
"""

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import traceback


TABLES = {
    "kernel": "CUPTI_ACTIVITY_KIND_KERNEL",
    "memcpy": "CUPTI_ACTIVITY_KIND_MEMCPY",
    "memset": "CUPTI_ACTIVITY_KIND_MEMSET",
    "cuda_api": "CUPTI_ACTIVITY_KIND_RUNTIME",
    "cuda_driver": "CUPTI_ACTIVITY_KIND_DRIVER",
    "synchronization": "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
    "nvtx": "NVTX_EVENTS",
}
DEVICE_KINDS = ("kernel", "memcpy", "memset")
EVENT_COLUMNS = (
    "start", "end", "deviceId", "contextId", "streamId", "globalPid", "globalTid",
    "correlationId", "bytes", "copyKind", "value", "returnValue", "graphNodeId",
    "gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ", "registersPerThread",
    "staticSharedMemory", "dynamicSharedMemory", "text", "textId", "endGlobalTid",
)


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _fingerprint(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


class _Reader:
    def __init__(self, db, log, start_ns, end_ns, limit):
        self.db, self.log = db, log
        self.start_ns, self.end_ns, self.limit = start_ns, end_ns, limit
        self.schema = {}
        for row in self.query("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"):
            self.schema[row["name"]] = {
                "sql": row["sql"],
                "columns": [item["name"] for item in self.query(f"PRAGMA table_info({_quote(row['name'])})")],
            }

    def query(self, sql, parameters=()):
        self.log.write(json.dumps({"sql": sql, "parameters": list(parameters)}, ensure_ascii=False) + "\n")
        self.log.flush()
        return self.db.execute(sql, parameters)

    def columns(self, table):
        return self.schema.get(table, {}).get("columns", [])

    def valid(self):
        return "typeof(e.start)='integer' AND typeof(e.end)='integer' AND e.end > e.start"

    def where(self):
        parts, parameters = [self.valid()], []
        if self.start_ns is not None:
            parts.append("e.end > ?")
            parameters.append(self.start_ns)
        if self.end_ns is not None:
            parts.append("e.start < ?")
            parameters.append(self.end_ns)
        return " AND ".join(parts), parameters

    def clipped(self):
        # Bounds were validated as signed 64-bit integers before making artifacts.
        start = "e.start" if self.start_ns is None else f"MAX(e.start, {self.start_ns})"
        end = "e.end" if self.end_ns is None else f"MIN(e.end, {self.end_ns})"
        return start, end

    def name(self, kind):
        columns = self.columns(TABLES[kind])
        name_column = next((name for name in ("demangledName", "shortName", "nameId", "textId") if name in columns), None)
        expression = "NULL"
        if name_column and {"id", "value"}.issubset(self.columns("StringIds")):
            expression = f"(SELECT value FROM StringIds WHERE id=e.{_quote(name_column)})"
        if "text" in columns:
            expression = f"COALESCE(e.text, {expression})"
        return f"COALESCE({expression}, '{kind}:unresolved')"

    def events(self, kind, where=None, parameters=None, limit=None):
        table = TABLES[kind]
        if where is None:
            where, parameters = self.where()
        selected = ", ".join("e." + _quote(column) for column in EVENT_COLUMNS if column in self.columns(table))
        sql = f"SELECT e.rowid AS source_rowid, {self.name(kind)} AS name, {selected} FROM {_quote(table)} e WHERE {where} ORDER BY e.start, e.rowid LIMIT ?"
        return [dict(row) for row in self.query(sql, [*parameters, limit or self.limit])]

    def signal(self, kind):
        table = TABLES[kind]
        result = {"table": table}
        if table not in self.schema:
            return {**result, "status": "table_absent", "reason": "未在导出中观察到该表；不能据此判定信号未启用或工作未发生。"}
        result["rows"] = self.query(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0]
        missing = sorted({"start", "end"} - set(self.columns(table)))
        if missing:
            return {**result, "status": "unsupported_schema", "missing_columns": missing}
        where, parameters = self.where()
        valid_rows = self.query(f"SELECT COUNT(*) FROM {_quote(table)} e WHERE {self.valid()}").fetchone()[0]
        result["completed_interval_rows"] = valid_rows
        result["rows_excluded_from_interval_analysis"] = result["rows"] - valid_rows
        result["selected_interval_rows"] = self.query(f"SELECT COUNT(*) FROM {_quote(table)} e WHERE {where}", parameters).fetchone()[0]
        result["status"] = "observed" if result["selected_interval_rows"] else "no_completed_intervals_in_selection"
        start, end = self.clipped()
        keys = [f"{self.name(kind)} AS name"]
        groups = ["name"]
        for column in ("globalPid", "deviceId", "globalTid"):
            if column in self.columns(table):
                keys.append("e." + _quote(column))
                groups.append("e." + _quote(column))
        sql = (f"SELECT {', '.join(keys)}, COUNT(*) AS count, SUM({end}-{start}) AS summed_duration_ns "
               f"FROM {_quote(table)} e WHERE {where} GROUP BY {', '.join(groups)} "
               "ORDER BY summed_duration_ns DESC LIMIT ?")
        hotspots = [dict(row) for row in self.query(sql, [*parameters, self.limit + 1])]
        result["hotspots"] = hotspots[:self.limit]
        result["hotspots_truncated"] = len(hotspots) > self.limit
        result["events"] = self.events(kind)
        result["events_truncated"] = result["selected_interval_rows"] > len(result["events"])
        return result

    def correlate(self, event):
        if type(event.get("globalPid")) is not int or type(event.get("correlationId")) is not int or event["correlationId"] == 0:
            return {"status": "unavailable", "reason": "缺少可用的进程身份或 correlationId。"}
        candidates, eligible, total = [], [], 0
        for kind in ("cuda_api", "cuda_driver"):
            table = TABLES[kind]
            if not {"globalTid", "correlationId", "start", "end"}.issubset(self.columns(table)):
                continue
            eligible.append(table)
            # Preserve the full encoded process scope, including VM bits.
            # Do not match a reused correlationId from another process.
            where = "(e.globalTid >> 24)=? AND e.correlationId=?"
            parameters = [event["globalPid"] >> 24, event["correlationId"]]
            total += self.query(f"SELECT COUNT(*) FROM {_quote(table)} e WHERE {where}", parameters).fetchone()[0]
            for row in self.events(kind, where, parameters, self.limit):
                candidates.append({"table": table, **row})
        status = "unavailable" if not eligible else "unmatched" if total == 0 else "unique" if total == 1 else "ambiguous"
        result = {"status": status, "candidate_count": total, "candidates": candidates[:self.limit],
                  "candidates_truncated": total > self.limit,
                  "basis": "same full encoded process scope and correlationId; API lookup is not time-window filtered"}
        if total == 1 and isinstance(candidates[0].get("end"), int):
            result["gpu_start_minus_api_end_ns"] = event["start"] - candidates[0]["end"]
        return result

    def coverage(self):
        selects, parameters, included, omitted, unidentified = [], [], [], [], {}
        where, bounds = self.where()
        start, end = self.clipped()
        for kind in DEVICE_KINDS:
            table = TABLES[kind]
            if not {"start", "end", "deviceId", "globalPid"}.issubset(self.columns(table)):
                omitted.append(table)
                continue
            identity = "typeof(e.globalPid)='integer' AND typeof(e.deviceId)='integer'"
            unidentified[table] = self.query(f"SELECT COUNT(*) FROM {_quote(table)} e WHERE {where} AND NOT ({identity})", bounds).fetchone()[0]
            selects.append(f"SELECT e.globalPid, e.deviceId, {start} AS start_ns, {end} AS end_ns FROM {_quote(table)} e WHERE {where} AND {identity}")
            parameters.extend(bounds)
            included.append(table)
        groups, current = [], None
        if selects:
            sql = " UNION ALL ".join(selects) + " ORDER BY globalPid, deviceId, start_ns, end_ns"
            for row in self.query(sql, parameters):
                key = row["globalPid"], row["deviceId"]
                if current is None or (current["globalPid"], current["deviceId"]) != key:
                    current = {"globalPid": key[0], "deviceId": key[1], "start_ns": row["start_ns"],
                               "end_ns": row["start_ns"], "summed_event_duration_ns": 0,
                               "observed_interval_union_ns": 0, "largest_uncovered_intervals": []}
                    groups.append(current)
                start_value, end_value = row["start_ns"], row["end_ns"]
                current["summed_event_duration_ns"] += end_value - start_value
                if start_value > current["end_ns"]:
                    current["largest_uncovered_intervals"].append({"start_ns": current["end_ns"], "end_ns": start_value, "duration_ns": start_value-current["end_ns"]})
                    current["largest_uncovered_intervals"].sort(key=lambda gap: gap["duration_ns"], reverse=True)
                    del current["largest_uncovered_intervals"][5:]
                current["observed_interval_union_ns"] += max(0, end_value-max(start_value, current["end_ns"]))
                current["end_ns"] = max(current["end_ns"], end_value)
        for group in groups:
            group["span_ns"] = group["end_ns"] - group["start_ns"]
            group["uncovered_within_span_ns"] = group["span_ns"] - group["observed_interval_union_ns"]
            group["overlap_excess_ns"] = group["summed_event_duration_ns"] - group["observed_interval_union_ns"]
        return {"included_tables": included, "omitted_tables": omitted, "unidentified_rows_excluded": unidentified, "groups": groups,
                "meaning": "按进程与设备合并已观察到的 kernel/copy/memset 区间；仅覆盖首尾事件之间。未覆盖区间不是设备全局空闲时间，区间并集不是 SM 利用率。"}


def analyze_nsys(source: Path, output_dir: Path, *, context: Path | None = None,
                 start_ns: int | None = None, end_ns: int | None = None, limit: int = 20) -> dict:
    """Analyze a completed SQLite export without modifying it or invoking nsys.

    Output directories are immutable per attempt. Failed reads preserve the
    exception and query log in the new directory. Large raw traces stay in place.
    """
    for bound in (start_ns, end_ns):
        if bound is not None and (type(bound) is not int or not -(2**63) <= bound < 2**63):
            raise ValueError("Time bounds must be signed 64-bit integer nanoseconds.")
    if start_ns is not None and end_ns is not None and start_ns >= end_ns:
        raise ValueError("start_ns must be less than end_ns.")
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError("limit must be an integer from 1 to 200.")
    source = source.resolve(strict=True)
    declared_context = json.loads(context.read_text(encoding="utf-8")) if context else None
    if context and not isinstance(declared_context, dict):
        raise ValueError("Context must be a JSON object.")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "nsys_completed_intervals", "source": {"path": str(source)},
        "context": {"status": "declared_not_verified" if context else "not_provided", "value": declared_context},
        "selection": {"start_ns": start_ns, "end_ns": end_ns, "limit_per_signal": limit,
                      "rule": "positive-duration intervals intersecting [start_ns, end_ns); aggregate durations clipped to bounds"},
        "evidence": {"schema": "schema.json", "queries": "queries.jsonl"},
        "unverified": ["capture_completeness", "context_fidelity", "profiling_overhead", "model_correctness", "request_SLOs", "bottleneck_causality"],
        "notes": [
            "原生采集与导出仍由 Coding Agent 使用 nsys 完成；本工具只读已结束的 SQLite 导出。",
            "表缺失可能来自未采集、无事件或过滤导出；空表也不能证明目标没有执行。",
            "只分析已完成的正时长区间；NVTX 点事件、未闭合范围、异常时间戳保留在原始库并计入排除行数。",
            "热点累计时长可能重叠；CPU、GPU、不同线程或设备的累计时长不能相加解释端到端延迟。",
            "事件保留原始时间戳，聚合时长按选择窗口裁剪。关联时间差可为负，不直接表示排队或提交开销。",
            "返回数量有限；根据时间窗口再次查询或使用 schema、SQL 与 source_rowid 回查原始库。",
            "只有 kernel/copy/memset 进入设备区间并集；整图活动、其他进程及其他引擎可能未被覆盖。",
        ],
    }
    try:
        if any(Path(str(source) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
            raise ValueError("SQLite sidecars present; use a closed, completed native export.")
        before = source.stat()
        report["source"] = _fingerprint(source)
        if context:
            report["context"]["source"] = _fingerprint(context)
        # mode=ro prevents writes; do not use immutable=1 on a potentially live DB.
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as db, (output_dir / "queries.jsonl").open("w", encoding="utf-8") as log:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("PRAGMA trusted_schema=OFF")
            db.execute("BEGIN")
            reader = _Reader(db, log, start_ns, end_ns, limit)
            _write(output_dir / "schema.json", reader.schema)
            if not (set(TABLES.values()) | {"META_DATA_EXPORT"}).intersection(reader.schema):
                raise ValueError("No recognized Nsight Systems tables in this SQLite export.")
            report["signals"] = {kind: reader.signal(kind) for kind in TABLES}
            for kind in DEVICE_KINDS:
                for event in report["signals"][kind].get("events", []):
                    event["host_api"] = reader.correlate(event)
            report["device_intervals"] = reader.coverage()
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source changed while analyzing; use a completed, stable export.")
        report["status"] = "ok"
    except (OSError, sqlite3.Error, ValueError) as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    _write(output_dir / "report.json", report)
    return report
