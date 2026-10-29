"""Registration invariants with isolated, exact-byte temporary baselines."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knowledge_release import ReleaseError, digest, encoded, finalize_verification, read, release


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "docs/evidence/run-one"
        self.before = self.run / "before"
        self.put("knowledge/algorithms/a.md", b"# K-ALG-001: old\r\n\r\nExact old bytes.\r\n")
        self.put("knowledge/algorithms/b.md", b"# K-ALG-002: untouched\n")
        self.put("knowledge/sources/S-ONE.md", b"# Source note\n")
        self.put("references/S-ONE/original.txt", b"Fixed original archive.\n")
        self.put("references/S-ONE/manifest.json", encoded({"files": [{
            "path": "original.txt", "sha256": digest(self.root / "references/S-ONE/original.txt")}]}))
        entries = []
        for key, name in [("K-ALG-001", "a"), ("K-ALG-002", "b")]:
            entries.append({"id": key, "path": f"algorithms/{name}.md", "category": "algorithms",
                            "revision": 1, "sha256": digest(self.root / f"knowledge/algorithms/{name}.md"),
                            "title": name, "summary": name, "source_ids": ["S-ONE"],
                            "status": "seed", "updated_at": "2026-01-01"})
        self.catalog = {"schema_version": 1, "entries": entries}
        self.source = {"id": "S-ONE", "title": "Fixed source", "version": "v1",
                       "local_path": "knowledge/sources/S-ONE.md",
                       "local_sha256": digest(self.root / "knowledge/sources/S-ONE.md"),
                       "archive_manifest": "references/S-ONE/manifest.json",
                       "archive_sha256": digest(self.root / "references/S-ONE/manifest.json")}
        self.put("knowledge/catalog.json", encoded(self.catalog))
        self.put("knowledge/sources.json", encoded({"schema_version": 3, "sources": [self.source]}))
        self.put("knowledge/temporal-policy.json", encoded({"profile_id": "base-v1"}))
        self.put("VERSION.json", encoded({"schema_version": 1, "version_id": "base-v1", "counts": {},
                                         "entries": [], "current_runtime_validation": "old-reader.json"}))
        self.put("docs/expansion-status.json", encoded({"version_id": "base-v1", "entries": [
            {"id": e["id"], "path": e["path"], "review": "pending", "revision": 1,
             "sha256": e["sha256"]} for e in entries]}))
        self.put("README.md", b"Version base-v1.\n")
        self.put("knowledge/README.md", b"Version v1.\n")
        paths = [str(p.relative_to(self.root).as_posix()) for p in self.root.rglob("*")
                 if p.is_file() and not p.is_relative_to(self.run)]
        for name in paths:
            target = self.before / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((self.root / name).read_bytes())
        self.put("docs/evidence/run-one/before/snapshot.json", encoded({
            name: digest(self.before / name) for name in paths}))
        self.put("docs/evidence/run-one/workflow.json", encoded({"jobs": {
            "K-ALG-001": {"round": 2}, "K-ALG-002": {"round": 1}}}))
        self.put("docs/evidence/run-one/K-ALG-001/round-2/review.json", encoded({"decision": "accept"}))
        self.put("docs/evidence/run-one/K-ALG-002/round-1/review.json", encoded({"decision": "accept"}))
        self.render = patch("knowledge_release._render", return_value=(
            {"knowledge/INDEX.md": b"Rendered index\n"}, {"profile_id": "base-v2", "valid": True}))
        self.render.start()
        self.addCleanup(self.render.stop)

    def put(self, name, body):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)

    def changed(self):
        self.put("knowledge/algorithms/a.md", b"# K-ALG-001: revised\n")

    def invoke(self, **kwargs):
        return release(self.root, self.run, ["K-ALG-001"], "base-v2", updated_at="2026-10-28", **kwargs)

    def registration_bytes(self):
        return {(p.relative_to(self.root)).as_posix(): p.read_bytes()
                for p in self.root.rglob("*") if p.is_file() and not p.is_relative_to(self.run)}

    def test_exact_history_metadata_and_source_note_hash(self):
        old = (self.root / "knowledge/algorithms/a.md").read_bytes()
        self.changed()
        self.put("knowledge/sources/S-ONE.md", b"# Improved source interpretation\n")
        result = self.invoke()
        self.assertEqual(result["revised_ids"], ["K-ALG-001"])
        self.assertEqual(result["source_note_changes"], ["S-ONE"])
        self.assertEqual((self.root / "knowledge/history/K-ALG-001/0001.md").read_bytes(), old)
        self.assertEqual(read(self.root / "knowledge/history/K-ALG-001/0001.json"), self.catalog["entries"][0])
        current = read(self.root / "knowledge/catalog.json")["entries"]
        self.assertEqual(current[0]["revision"], 2)
        self.assertEqual(current[1], self.catalog["entries"][1])
        source = read(self.root / "knowledge/sources.json")["sources"][0]
        self.assertEqual(source["local_sha256"], digest(self.root / source["local_path"]))
        self.assertEqual({k: v for k, v in source.items() if k != "local_sha256"},
                         {k: v for k, v in self.source.items() if k != "local_sha256"})
        version = read(self.root / "VERSION.json")
        self.assertEqual(version["counts"]["active_history_versions"], 1)
        self.assertIsNone(version["current_runtime_validation"])
        self.assertEqual(version["previous_version_sha256"], digest(self.before / "VERSION.json"))
        status = read(self.root / "docs/expansion-status.json")["entries"]
        self.assertIn("round-2/review.json", status[0]["evidence"])
        self.assertEqual(status[1]["review"], "pending")

    def test_idempotent_replay_preserves_revision_and_history(self):
        self.changed()
        self.invoke()
        before = self.registration_bytes()
        result = self.invoke()
        self.assertTrue(result["already_registered"])
        self.assertEqual(before, self.registration_bytes())

    def test_unselected_change_rejected_before_any_registration_write(self):
        self.changed()
        self.put("knowledge/algorithms/b.md", b"Unexpected other author change\n")
        before = self.registration_bytes()
        with self.assertRaisesRegex(ReleaseError, "Unselected chapter"):
            self.invoke()
        self.assertEqual(before, self.registration_bytes())

    def test_stale_baseline_and_new_run_cannot_rollback_release(self):
        self.changed()
        version = read(self.root / "VERSION.json")
        version["version_id"] = "another-release"
        self.put("VERSION.json", encoded(version))
        before = self.registration_bytes()
        with self.assertRaisesRegex(ReleaseError, "Stale baseline"):
            self.invoke()
        self.assertEqual(before, self.registration_bytes())

    def test_fixed_source_identity_rejected(self):
        registry = read(self.root / "knowledge/sources.json")
        registry["sources"][0]["version"] = "v2"
        self.put("knowledge/sources.json", encoded(registry))
        with self.assertRaisesRegex(ReleaseError, "Fixed source identity"):
            self.invoke()
        self.assertFalse((self.root / "knowledge/history").exists())

    def test_archive_tamper_rejected_even_when_registry_unchanged(self):
        self.put("references/S-ONE/original.txt", b"Changed upstream contents\n")
        with self.assertRaisesRegex(ReleaseError, "Archived source member"):
            self.invoke()

    def test_existing_conflicting_history_rejected(self):
        self.changed()
        self.put("knowledge/history/K-ALG-001/0001.md", b"Wrong historical bytes\n")
        with self.assertRaisesRegex(ReleaseError, "Conflicting history"):
            self.invoke()

    def test_metadata_only_revision_and_no_change_review(self):
        result = self.invoke(entry_updates={"K-ALG-001": {"summary": "A more accurate summary"}})
        self.assertEqual(result["revised_ids"], ["K-ALG-001"])
        self.assertEqual(read(self.root / "knowledge/catalog.json")["entries"][0]["revision"], 2)

    def test_unchanged_body_review_does_not_invent_revision(self):
        result = self.invoke()
        self.assertEqual(result["revised_ids"], [])
        self.assertEqual(read(self.root / "knowledge/catalog.json")["entries"][0]["revision"], 1)

    def test_receipt_detects_late_body_mutation(self):
        self.changed()
        self.invoke()
        self.put("knowledge/algorithms/a.md", b"A late edit after registration\n")
        with self.assertRaisesRegex(ReleaseError, "Registered release has changed"):
            self.invoke()

    def test_missing_review_artifact_fails_without_writes(self):
        self.changed()
        (self.run / "K-ALG-001/round-2/review.json").unlink()
        before = self.registration_bytes()
        with self.assertRaisesRegex(ReleaseError, "Missing accepted review"):
            self.invoke()
        self.assertEqual(before, self.registration_bytes())

    def verification_reports(self):
        entries = read(self.root / "knowledge/catalog.json")["entries"]
        structural = {"result": "PASS", "issues": [], "content_bindings": {
            e["id"]: {"revision": e["revision"], "sha256": e["sha256"]} for e in entries}}
        first = entries[0]
        reader = {"result": "PASS", "canonical_unchanged": True,
                  "modules": [{"id": first["id"], "revision": first["revision"],
                               "sha256": first["sha256"], "complete_read": True}],
                  "documents": {name: digest(self.root / name) for name in
                                ["knowledge/catalog.json", "knowledge/sources.json"]}}
        self.put("docs/evidence/run-one/structural.json", encoded(structural))
        self.put("docs/evidence/run-one/reader.json", encoded(reader))
        return self.run / "structural.json", self.run / "reader.json"

    def test_verification_binding_then_registration_replay(self):
        self.changed()
        self.invoke()
        structural, reader = self.verification_reports()
        result = finalize_verification(self.root, self.run, structural, reader)
        version = read(self.root / "VERSION.json")
        self.assertEqual(version["current_runtime_validation"], "docs/evidence/run-one/reader.json")
        self.assertIn("verification", result)
        self.assertTrue(self.invoke()["already_registered"])

    def test_failed_verification_cannot_bind(self):
        self.invoke()
        structural, reader = self.verification_reports()
        bad = read(reader)
        bad["result"] = "FAIL"
        reader.write_bytes(encoded(bad))
        before = (self.root / "VERSION.json").read_bytes()
        with self.assertRaisesRegex(ReleaseError, "Reader verification"):
            finalize_verification(self.root, self.run, structural, reader)
        self.assertEqual(before, (self.root / "VERSION.json").read_bytes())

    def test_stale_report_revision_cannot_bind(self):
        self.changed()
        self.invoke()
        structural, reader = self.verification_reports()
        bad = read(structural)
        bad["content_bindings"]["K-ALG-001"]["revision"] = 1
        structural.write_bytes(encoded(bad))
        with self.assertRaisesRegex(ReleaseError, "current catalog revisions"):
            finalize_verification(self.root, self.run, structural, reader)


if __name__ == "__main__":
    unittest.main()
