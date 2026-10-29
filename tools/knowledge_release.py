"""Register reviewed local knowledge revisions against an exact run baseline.

The caller decides semantic acceptance. This module preserves identity/history,
renders the existing runtime's navigation/audit, and records local registration.
It neither invokes models nor publishes, changes Tasks, or edits archived sources.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import date
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys


class ReleaseError(ValueError):
    """Registration precondition failed; no automatic repair is attempted."""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def inside(root, relative):
    relative = Path(relative)
    if relative.is_absolute():
        raise ReleaseError(f"Expected repository-relative path: {relative}")
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ReleaseError(f"Path escapes repository: {relative}")
    return result


def _render(root, catalog, registry, policy, old_version, runtime_source):
    """Use the maintained runtime rules instead of a second index implementation."""
    configured = runtime_source or old_version.get("current_runtime_contract", {}).get(
        "validated_runtime", {}).get("source_directory")
    if configured is None:
        raise ReleaseError("runtime_source is required to render navigation and temporal audit")
    source = Path(configured).resolve()
    if not (source / "infra_agent/knowledge_index.py").is_file():
        raise ReleaseError(f"Runtime source is unavailable: {source}")
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(source))
        index = importlib.import_module("infra_agent.knowledge_index")
        temporal = importlib.import_module("infra_agent.temporal")
        if not Path(index.__file__).resolve().is_relative_to(source):
            raise ReleaseError("A different infra_agent runtime is already imported")
        store = importlib.import_module("infra_agent.knowledge_store")
        store.validate_catalog(root, catalog, registry)
        temporal.load_policy(root)  # Validate the existing unchanged source policy.
        view = temporal.temporal_view(catalog, registry, policy, repository=root)
        source_rows = [{"id": s["id"], "title": s["title"],
                        **view["source_statuses"][s["id"]]} for s in registry["sources"]]
        selected = {e["id"]: e for e in view["catalog"]["entries"]}
        entry_rows = []
        for current in catalog["entries"]:
            entry = selected.get(current["id"], current)
            decision = temporal.entry_status(entry, view["source_statuses"])
            entry_rows.append({"id": entry["id"], "revision": entry["revision"],
                               "path": entry["path"], "category": entry["category"],
                               "source_ids": entry.get("source_ids", []), **decision})
        audit = {"valid": True, "profile_id": policy["profile_id"], "model": policy["model"],
                 "knowledge_cutoff": policy["knowledge_cutoff"],
                 "effective_cutoff": policy["effective_cutoff"], "mode": policy["mode"],
                 "entries": entry_rows, "sources": source_rows, "counts": view["counts"],
                 "meaning": "Only eligible entries are available to strict offline role snapshots; blocked material remains for audit and history.",
                 "limits": ["A dated source decision proves temporal eligibility, not semantic correctness or task applicability.",
                            "Editorial updated_at and archive capture timestamps do not establish source availability."]}
        return {"knowledge/" + n: b for n, b in index.indexes(catalog).items()}, audit
    finally:
        sys.path[:] = previous


def _verify_archives(root, sources):
    """An unchanged manifest field does not establish unchanged archived bytes."""
    for source in sources:
        if not source.get("archive_manifest"):
            continue
        path = inside(root, source["archive_manifest"])
        if digest(path) != source["archive_sha256"]:
            raise ReleaseError(f"Archive manifest changed: {source['id']}")
        for member in read(path)["files"]:
            target = (path.parent / member["path"]).resolve()
            if not target.is_relative_to(root) or digest(target) != member["sha256"]:
                raise ReleaseError(f"Archived source member changed: {source['id']} {member['path']}")


def release(root, run_dir, entry_ids, new_version_id, report_paths=(), *,
            runtime_source=None, updated_at=None, entry_updates=None):
    """Register selected reviewed entries; exact same invocation is idempotent.

    ``before/snapshot.json`` is the path-to-SHA map captured before author edits.
    Chapter/source metadata are frozen except explicit selected entry_updates and
    source-note hashes. All checks/rendering happen before any registration write.
    Files are replaced atomically individually; this is not a multi-file filesystem
    transaction. An interrupted registration must be investigated, not guessed away.
    """
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    if not run_dir.is_relative_to(root / "docs/evidence"):
        raise ReleaseError("run_dir must be under the knowledge repository's docs/evidence")
    selected = sorted(set(entry_ids))
    if not selected or not isinstance(new_version_id, str) or not new_version_id.strip():
        raise ReleaseError("Select entries and provide a nonempty new version ID")
    before = run_dir / "before"
    raw_map = read(before / "snapshot.json")
    snapshot = raw_map.get("files", raw_map)
    for name, sha in snapshot.items():
        if digest(inside(before, name)) != sha:
            raise ReleaseError(f"Baseline snapshot changed: {name}")
    required = {"VERSION.json", "knowledge/catalog.json", "knowledge/sources.json",
                "knowledge/temporal-policy.json", "docs/expansion-status.json"}
    if not required <= snapshot.keys():
        raise ReleaseError(f"Baseline is incomplete: {sorted(required - snapshot.keys())}")
    old_version = read(before / "VERSION.json")
    old_catalog = read(before / "knowledge/catalog.json")
    old_registry = read(before / "knowledge/sources.json")
    entries = {x["id"]: x for x in old_catalog["entries"]}
    if set(selected) - entries.keys():
        raise ReleaseError(f"Unknown selected entries: {sorted(set(selected) - entries.keys())}")
    receipt_path = run_dir / "release.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt.get("version_id") != new_version_id or receipt.get("selected_ids") != selected:
            raise ReleaseError("This run already registered a different release")
        for name, sha in receipt["registered_files"].items():
            if digest(inside(root, name)) != sha:
                raise ReleaseError(f"Registered release has changed; refusing to replay: {name}")
        _verify_archives(root, old_registry["sources"])
        return {**receipt, "already_registered": True}
    if root.joinpath("VERSION.json").read_bytes() != before.joinpath("VERSION.json").read_bytes():
        raise ReleaseError("Stale baseline: VERSION has changed since run initialization")
    if old_version["version_id"] == new_version_id:
        raise ReleaseError("The new version ID must differ from the baseline")
    if read(root / "knowledge/catalog.json") != old_catalog:
        raise ReleaseError("Catalog changed since run initialization")
    if read(root / "knowledge/temporal-policy.json") != read(before / "knowledge/temporal-policy.json"):
        raise ReleaseError("Temporal policy changed since run initialization")
    if "knowledge/source-reading.json" in snapshot and digest(root / "knowledge/source-reading.json") != snapshot["knowledge/source-reading.json"]:
        raise ReleaseError("Source reading policy changed since run initialization")
    if read(root / "docs/expansion-status.json") != read(before / "docs/expansion-status.json"):
        raise ReleaseError("Expansion accounting changed since run initialization")
    current_registry = read(root / "knowledge/sources.json")
    prior_sources = {s["id"]: s for s in old_registry["sources"]}
    current_sources = {s["id"]: s for s in current_registry["sources"]}
    if len(current_sources) != len(current_registry["sources"]) or current_sources.keys() != prior_sources.keys():
        raise ReleaseError("Source set changed; this workflow does not add or withdraw sources")
    if {k: v for k, v in current_registry.items() if k != "sources"} != {k: v for k, v in old_registry.items() if k != "sources"}:
        raise ReleaseError("Source registry identity changed")
    for sid, source in prior_sources.items():
        if {k: v for k, v in source.items() if k != "local_sha256"} != {k: v for k, v in current_sources[sid].items() if k != "local_sha256"}:
            raise ReleaseError(f"Fixed source identity changed: {sid}")
    _verify_archives(root, old_registry["sources"])
    updates = entry_updates or {}
    if set(updates) - set(selected):
        raise ReleaseError("Metadata updates must belong to selected entries")
    protected = {"id", "revision", "sha256", "path", "category", "status", "origin", "evidence"}
    day = updated_at or date.today().isoformat()
    date.fromisoformat(day)
    catalog, registry = deepcopy(old_catalog), deepcopy(old_registry)
    writes, changed_ids, note_changes, changes = {}, [], [], []
    for entry in catalog["entries"]:
        name = "knowledge/" + entry["path"]
        if name not in snapshot or snapshot[name] != entry["sha256"]:
            raise ReleaseError(f"Baseline chapter is missing or mismatched: {entry['id']}")
        actual = digest(inside(root, name))
        body_changed = actual != entry["sha256"]
        supplied = updates.get(entry["id"], {})
        if protected & supplied.keys():
            raise ReleaseError(f"Cannot directly update protected metadata: {entry['id']}")
        metadata_changed = any(entry.get(k) != v for k, v in supplied.items())
        if entry["id"] not in selected and body_changed:
            raise ReleaseError(f"Unselected chapter changed: {entry['id']}")
        if not body_changed and not metadata_changed:
            continue
        old = deepcopy(entry)
        history = f"knowledge/history/{entry['id']}/{entry['revision']:04d}"
        for target_name, body in [(history + ".md", (before / name).read_bytes()),
                                  (history + ".json", encoded(old))]:
            target = inside(root, target_name)
            if target.exists():
                equal = read(target) == old if target.suffix == ".json" else target.read_bytes() == body
                if not equal:
                    raise ReleaseError(f"Conflicting history already exists: {target_name}")
            else:
                writes[target_name] = body
        entry.update(supplied)
        entry.update(revision=old["revision"] + 1, sha256=actual, updated_at=day)
        changed_ids.append(entry["id"])
        changes.append({"entry_id": entry["id"], "previous_revision": old["revision"],
                        "revision": entry["revision"], "path": name, "summary": entry["summary"]})
    for source in registry["sources"]:
        name = source["local_path"]
        if name not in snapshot or snapshot[name] != source["local_sha256"]:
            raise ReleaseError(f"Baseline source note is missing or mismatched: {source['id']}")
        actual = digest(inside(root, name))
        if actual != source["local_sha256"]:
            source["local_sha256"] = actual
            note_changes.append(source["id"])
    reports = []
    for value in report_paths:
        path = Path(value)
        if path.is_absolute():
            path = path.resolve().relative_to(root)
        name = path.as_posix()
        reports.append({"path": name, "sha256": digest(inside(root, name))})
    policy = read(before / "knowledge/temporal-policy.json")
    policy["profile_id"] = new_version_id
    navigation, audit = _render(root, catalog, registry, policy, old_version, runtime_source)
    writes.update(navigation)
    writes["knowledge/catalog.json"] = encoded(catalog)
    writes["knowledge/sources.json"] = encoded(registry)
    writes["knowledge/temporal-policy.json"] = encoded(policy)
    writes["knowledge/temporal-audit.json"] = encoded(audit)
    accounting = read(before / "docs/expansion-status.json")
    records = {e["id"]: e for e in accounting["entries"]}
    if records.keys() != entries.keys():
        raise ReleaseError("Expansion accounting does not cover the baseline catalog exactly")
    workflow = read(run_dir / "workflow.json") if (run_dir / "workflow.json").exists() else {}
    for entry in catalog["entries"]:
        record = records[entry["id"]]
        record.update(revision=entry["revision"], sha256=entry["sha256"])
        if entry["id"] in selected:
            job = workflow.get("jobs", {}).get(entry["id"], {})
            review = run_dir / entry["id"] / f"round-{job.get('round', 1)}" / "review.json"
            if not review.is_file():
                raise ReleaseError(f"Missing accepted review artifact: {entry['id']}")
            record.update(review="workflow_content_reviewed_with_limits", evidence=review.relative_to(root).as_posix())
    accounting.update(version_id=new_version_id, updated_at=day, current_batch=selected,
                      counts_by_review=dict(Counter(x["review"] for x in accounting["entries"])))
    writes["docs/expansion-status.json"] = encoded(accounting)
    writes["docs/actionable-knowledge/change-manifest.json"] = encoded({
        "schema_version": 1, "version_id": new_version_id, "changes": changes,
        "reviewed_ids": selected, "source_note_changes": note_changes,
        "source_changes": {"added": [], "removed": []}})
    version = deepcopy(old_version)
    version.update(version_id=new_version_id, status="local_workflow_content_revision", updated_at=day,
                   source_changes={"added": [], "removed": []}, verification_reports=reports,
                   verification_scope="The recorded workflow reviewed the selected entries. Reports retain their own scope; registration does not certify GPU behavior or controlled knowledge benefit.",
                   publication_status="Local workflow registration; no Git commit, remote publication or active Task switch.",
                   supersedes=old_version["version_id"])
    version["validation"] = (run_dir / "structural.json").relative_to(root).as_posix()
    version["current_runtime_validation"] = None
    if "current_runtime_contract" in version:
        version["current_runtime_contract"]["validation"] = None
        version["current_runtime_contract"]["validation_status"] = "Current revision reader validation has not been run by registration."
    for key, name in [("previous_catalog", "knowledge/catalog.json"),
                      ("previous_sources", "knowledge/sources.json"), ("previous_version", "VERSION.json")]:
        version[key] = (before / name).relative_to(root).as_posix()
        version[key + "_sha256"] = digest(before / name)
    version.setdefault("counts", {}).update(
        chapters=len(catalog["entries"]), sources=len(registry["sources"]), new_chapters=0,
        revised_existing_chapters=len(changed_ids), unchanged_chapters=len(entries) - len(changed_ids),
        active_history_versions=sum(e["revision"] - 1 for e in catalog["entries"]),
        delta_basis=old_version["version_id"])
    version["entries"] = [{"entry_id": e["id"], "path": e["path"], "revision": e["revision"],
                           "sha256": e["sha256"]} for e in catalog["entries"]]
    for name in ["README.md", "knowledge/README.md"]:
        target = root / name
        if target.is_file():
            text = target.read_text(encoding="utf-8")
            text = text.replace(old_version["version_id"], new_version_id)
            # Existing repository README uses the short suffix in its version line.
            old_short = re.search(r"(?:^|-)(v\d[^\s]*)$", old_version["version_id"])
            new_short = re.search(r"(?:^|-)(v\d[^\s]*)$", new_version_id)
            if old_short and new_short:
                text = text.replace(old_short.group(1), new_short.group(1))
            old_histories = old_version.get("counts", {}).get("active_history_versions")
            if old_histories is not None:
                text = text.replace(f"{old_histories}份活动历史", f"{version['counts']['active_history_versions']}份活动历史")
            writes[name] = text.replace("\r\n", "\n").encode("utf-8")
    # VERSION is intentionally replaced after the dependent registration files.
    writes["VERSION.json"] = encoded(version)
    for name, body in writes.items():
        target = inside(root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".release-writing")
        temporary.write_bytes(body)
        temporary.replace(target)
    registered = {name: digest(inside(root, name)) for name in writes}
    registered.update({"knowledge/" + e["path"]: digest(root / "knowledge" / e["path"]) for e in catalog["entries"]})
    registered.update({s["local_path"]: digest(root / s["local_path"]) for s in registry["sources"]})
    registered.update({r["path"]: r["sha256"] for r in reports})
    receipt = {"schema_version": 1, "version_id": new_version_id,
               "baseline_version": old_version["version_id"], "selected_ids": selected,
               "revised_ids": changed_ids, "source_note_changes": note_changes,
               "registered_files": registered, "already_registered": False,
               "meaning": "Local identity/history registration; semantic acceptance is recorded in the per-entry reviews. Run structural and reader checks against these bytes separately."}
    temporary = receipt_path.with_name(receipt_path.name + ".writing")
    temporary.write_bytes(encoded(receipt))
    temporary.replace(receipt_path)
    return receipt


def finalize_verification(root, run_dir, structural_path, reader_path):
    """Bind actually executed structural/reader reports to a registered release.

    Calling code runs the tools and retains their execution evidence. This checks
    the resulting report scopes and byte/revision bindings; it does not fabricate
    either report or infer semantic, model-effectiveness, or target-device success.
    """
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    if not run_dir.is_relative_to(root / "docs/evidence"):
        raise ReleaseError("run_dir must be under docs/evidence")
    receipt_path = run_dir / "release.json"
    receipt = read(receipt_path)
    for name, sha in receipt["registered_files"].items():
        if digest(inside(root, name)) != sha:
            raise ReleaseError(f"Registered release changed before verification binding: {name}")
    paths = []
    for value in [structural_path, reader_path]:
        path = Path(value)
        path = path.resolve() if path.is_absolute() else inside(root, path)
        if not path.is_relative_to(run_dir):
            raise ReleaseError("Verification reports must belong to this run")
        paths.append(path)
    structural, reader = (read(path) for path in paths)
    if structural.get("result") != "PASS" or structural.get("issues") != []:
        raise ReleaseError("Structural verification did not pass")
    if reader.get("result") != "PASS" or reader.get("canonical_unchanged") is not True:
        raise ReleaseError("Reader verification did not pass")
    catalog = read(root / "knowledge/catalog.json")
    entries = {e["id"]: e for e in catalog["entries"]}
    expected = {key: {"revision": e["revision"], "sha256": e["sha256"]} for key, e in entries.items()}
    if structural.get("content_bindings") != expected:
        raise ReleaseError("Structural report does not bind the current catalog revisions")
    modules = {m["id"]: m for m in reader.get("modules", [])}
    if len(modules) != len(reader.get("modules", [])) or not set(receipt["selected_ids"]) <= modules.keys():
        raise ReleaseError("Reader report does not cover every reviewed entry")
    for key, item in modules.items():
        if key not in entries or any(item.get(field) != entries[key][field] for field in ["revision", "sha256"]):
            raise ReleaseError(f"Stale reader module binding: {key}")
        if item.get("complete_read") is not True:
            raise ReleaseError(f"Reader did not completely read: {key}")
    for name in ["knowledge/catalog.json", "knowledge/sources.json"]:
        if reader.get("documents", {}).get(name) != digest(root / name):
            raise ReleaseError(f"Reader input binding changed: {name}")
    for name, sha in reader.get("documents", {}).items():
        if digest(inside(root, name)) != sha:
            raise ReleaseError(f"Stale reader input: {name}")
    version = read(root / "VERSION.json")
    if version["version_id"] != receipt["version_id"]:
        raise ReleaseError("Verification version differs from the registered release")
    reports = {x["path"]: x for x in version.get("verification_reports", [])}
    for path in paths:
        name = path.relative_to(root).as_posix()
        reports[name] = {"path": name, "sha256": digest(path)}
    structural_name, reader_name = [p.relative_to(root).as_posix() for p in paths]
    version.update(validation=structural_name, current_runtime_validation=reader_name,
                   verification_reports=list(reports.values()), status="local_workflow_verified_content_revision")
    if "current_runtime_contract" in version:
        version["current_runtime_contract"].update(
            validation=reader_name,
            validation_status="Actual current-revision frozen-reader fixture passed; this verifies retrieval, not model or target-device effectiveness.")
    target = root / "VERSION.json"
    temporary = target.with_name(target.name + ".release-writing")
    temporary.write_bytes(encoded(version))
    temporary.replace(target)
    receipt["registered_files"]["VERSION.json"] = digest(target)
    for report in reports.values():
        receipt["registered_files"][report["path"]] = report["sha256"]
    receipt["verification"] = {"structural": reports[structural_name], "reader": reports[reader_name],
                               "scope": "Structural identity/history and actual frozen retrieval for the reviewed entries only."}
    temporary = receipt_path.with_name(receipt_path.name + ".writing")
    temporary.write_bytes(encoded(receipt))
    temporary.replace(receipt_path)
    return {"version_id": receipt["version_id"], "verification": receipt["verification"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--entries", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--runtime-source", type=Path)
    parser.add_argument("--report", action="append", default=[])
    args = parser.parse_args()
    result = release(args.root, args.run, args.entries.split(","), args.version,
                     args.report, runtime_source=args.runtime_source)
    print(json.dumps({k: v for k, v in result.items() if k != "registered_files"}, ensure_ascii=False))
