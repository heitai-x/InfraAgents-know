#!/usr/bin/env python3
"""Validate a knowledge checkout without importing or modifying its runtime.

This checks identities, local reading paths, archival continuity and declared
optional question examples. It does not certify technical semantics, freedom
from exploration bias, or target-system results.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_bytes().decode("utf-8-sig"))


def validate(root: Path, baseline: str | None = None) -> dict:
    issues: list[dict] = []

    def issue(kind: str, **details):
        issues.append({"kind": kind, **details})

    def local(relative: str, parent: Path | None = None) -> Path:
        path = ((parent or root) / unquote(relative)).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Path escapes repository: {relative}")
        return path

    catalog = read_json(root / "knowledge/catalog.json")
    registry = read_json(root / "knowledge/sources.json")
    policy = read_json(root / "knowledge/temporal-policy.json")
    version = read_json(root / "VERSION.json")
    entries = catalog["entries"]
    sources = registry["sources"]
    by_id = {entry["id"]: entry for entry in entries}
    by_path = {local("knowledge/" + e["path"]): e["id"] for e in entries}
    source_ids = {source["id"] for source in sources}
    reading_file=root/'knowledge/source-reading.json'
    reading=read_json(reading_file).get('sources',{}) if reading_file.is_file() else None
    if reading is not None and set(reading)!=source_ids:
        issue('reading_source_set')
    for kind, values in (("entry", entries), ("source", sources)):
        for identifier, count in Counter(x["id"] for x in values).items():
            if count != 1:
                issue("duplicate_id", object=kind, id=identifier, count=count)
    if len(by_path) != len(entries):
        issue("duplicate_entry_path")

    links = []
    history_versions = 0
    fingerprints = {}

    def inspect_links(path: Path, text: str) -> set[str]:
        linked = set()
        for destination in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            destination = destination.split("#", 1)[0]
            if not destination or "://" in destination or destination.startswith("mailto:"):
                continue
            try:
                target = local(destination, path.parent)
                exists = target.is_file()
            except ValueError:
                exists = False
                target = None
            links.append({"from": str(path.relative_to(root)), "to": destination,
                          "exists": exists})
            if not exists:
                issue("missing_local_link", file=str(path.relative_to(root)), target=destination)
            if reading is not None and target is not None and target.is_relative_to(root/'references'):
                relative=target.relative_to(root).as_posix();parts=Path(relative).parts
                if len(parts)>2 and parts[1] in reading and relative not in reading[parts[1]]['paths']:
                    issue('link_to_audit_only_member',file=str(path.relative_to(root)),target=relative)
            if target in by_path:
                linked.add(by_path[target])
        return linked

    for entry in entries:
        identifier = entry["id"]
        path = local("knowledge/" + entry["path"])
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        fingerprints[identifier] = {"revision": entry["revision"], "sha256": sha(raw)}
        if sha(raw) != entry["sha256"]:
            issue("body_hash", id=identifier)
        if not text.startswith("# " + identifier + "："):
            issue("heading_id", id=identifier)
        prose = re.sub(r'(?m)^```pseudocode[ \t]*\r?\n([\s\S]*?)^```[ \t]*(?:\r?\n|$)', r'\1', text)
        if "```" in prose or "~~~" in prose or re.search(r"https?://", text):
            issue("noncode_local_prose", id=identifier)
        headings = re.findall(r"^## (.+)$", text, re.MULTILINE)
        if len(headings) != len(set(headings)):
            issue("duplicate_revision_heading", id=identifier)
        if not entry.get("source_ids") or set(entry["source_ids"]) - source_ids:
            issue("invalid_sources", id=identifier)
        for field in ('questions','assumes','symptoms'):
            value=entry.get(field,[])
            if not isinstance(value,list) or not all(isinstance(x,str) and x.strip() for x in value):
                issue('invalid_navigation_metadata',id=identifier,field=field)
        if set(entry.get("related", [])) - by_id.keys():
            issue("invalid_related", id=identifier)
        inspect_links(path, text)
        for revision in range(1, entry["revision"]):
            meta_path = root / f"knowledge/history/{identifier}/{revision:04d}.json"
            body_path = meta_path.with_suffix(".md")
            if not meta_path.is_file() or not body_path.is_file():
                issue("missing_history", id=identifier, revision=revision)
                continue
            meta = read_json(meta_path)
            if (meta.get("id") != identifier or meta.get("revision") != revision
                    or sha(body_path.read_bytes()) != meta.get("sha256")):
                issue("history_identity", id=identifier, revision=revision)
            history_versions += 1

    archive_files = 0
    archive_bytes = 0
    cutoff = date.fromisoformat(version["source_cutoff"])
    for source in sources:
        sid = source["id"]
        for path_key, hash_key in (("local_path", "local_sha256"),
                                   ("archive_manifest", "archive_sha256")):
            if sha(local(source[path_key]).read_bytes()) != source[hash_key]:
                issue("source_identity", id=sid, field=path_key)
        note = local(source["local_path"])
        inspect_links(note, note.read_text(encoding="utf-8-sig"))
        manifest_path = local(source["archive_manifest"])
        manifest = read_json(manifest_path)
        if reading is not None:
            registered={source['archive_manifest']} | {(manifest_path.parent/x['path']).relative_to(root).as_posix() for x in manifest['files']}
            selected=reading.get(sid,{})
            if set(selected.get('paths',[]))-registered or selected.get('primary') is not None and selected['primary'] not in selected.get('paths',[]):
                issue('invalid_reading_members',id=sid)
            implementation=[x['path'] for x in manifest['files'] if x['path'].startswith('upstream/') and Path(x['path']).suffix in {'.py','.cu','.cuh','.cpp','.cc','.c','.h','.hpp','.hip','.sh'}]
            blocked=implementation+(['reader.html','fulltext.txt','original.tar.gz'] if implementation else [])
            if any((manifest_path.parent/name).relative_to(root).as_posix() in selected.get('paths',[]) for name in blocked):
                issue('implementation_exposed_as_reading',id=sid)
        for member in manifest["files"]:
            path = local(member["path"], manifest_path.parent)
            if not path.is_file():
                issue("missing_archive_member", id=sid, path=member["path"])
                continue
            data = path.read_bytes()
            if sha(data) != member["sha256"]:
                issue("archive_member_hash", id=sid, path=member["path"])
            archive_files += 1
            archive_bytes += len(data)
        qualification = policy["sources"].get(sid, {})
        try:
            available = date.fromisoformat(source["available_at"])
            policy_available = date.fromisoformat(qualification["available_at"])
            if available > cutoff or policy_available != available:
                issue("temporal_boundary", id=sid)
        except (KeyError, ValueError):
            issue("missing_source_date", id=sid)

    # The checked-in report is an audit of this catalog/profile, not a spare
    # historical report. Old reports remain under docs/evidence/.
    audit = read_json(root / "knowledge/temporal-audit.json")
    if audit.get("profile_id") != policy.get("profile_id") or policy.get("profile_id") != version["version_id"]:
        issue("temporal_audit_profile")
    audit_entries = {entry["id"]: entry for entry in audit.get("entries", [])}
    if len(audit_entries) != len(audit.get("entries", [])) or set(audit_entries) != set(by_id):
        issue("temporal_audit_entry_set")
    for entry in entries:
        record = audit_entries.get(entry["id"], {})
        if any(record.get(key) != entry[key] for key in ("revision", "path", "category", "source_ids")):
            issue("temporal_audit_entry_binding", id=entry["id"])
    if {source["id"] for source in audit.get("sources", [])} != source_ids:
        issue("temporal_audit_source_set")
    if audit.get("effective_cutoff") != str(cutoff):
        issue("temporal_audit_cutoff")

    navigation = {}
    for relative in ["knowledge/INDEX.md", "knowledge/DESIGN.md", "knowledge/DIAGNOSIS.md"]:
        path = root / relative
        linked = inspect_links(path, path.read_text(encoding="utf-8-sig"))
        navigation[relative] = {"linked_entries": len(linked),
                                "missing_ids": sorted(by_id.keys() - linked)}
        if by_id.keys() - linked:
            issue("incomplete_navigation", file=relative, missing=sorted(by_id.keys() - linked))
    for category in sorted({entry["category"] for entry in entries}):
        path = root / "knowledge" / category / "README.md"
        linked = inspect_links(path, path.read_text(encoding="utf-8-sig"))
        expected = {entry["id"] for entry in entries if entry["category"] == category}
        if linked != expected:
            issue("category_navigation", category=category)

    routes = read_json(root / "docs/actionable-knowledge/use-cases.json")
    if (routes.get("schema_version") != 2 or routes.get("ordering") != "none"
            or routes.get("mandatory_entry") is not None
            or routes.get("candidate_space_closed") is not False
            or routes.get("must_use_knowledge") is not False):
        issue("question_examples_misdeclared_as_policy")
    route_ids = set()
    for route in routes["uses"]:
        if route["id"] in route_ids:
            issue("duplicate_use_case", id=route["id"])
        route_ids.add(route["id"])
        if set(route["knowledge_ids"]) - by_id.keys():
            issue("use_case_unknown_entry", id=route["id"])
        for field in ("question", "context_examples", "possible_observations", "applicability", "retrieval_queries"):
            if not route.get(field):
                issue("incomplete_use_case", id=route["id"], field=field)
        if any(field in route for field in ("required_validation", "expected_outputs", "roles", "next_stage", "runtime_entry")):
            issue("question_example_contains_workflow_obligation", id=route["id"])
        if route.get("evidence_status") != "illustrative_not_behavior_validated":
            issue("unjustified_use_case_status", id=route["id"])
    for relative in ["README.md", "docs/knowledge-use-cases.md", "docs/inference-coverage.md",
                     "docs/BUILD-STATUS.md", "docs/USAGE.md"]:
        path = root / relative
        inspect_links(path, path.read_text(encoding="utf-8-sig"))

    version_entries = {entry["entry_id"]: entry for entry in version["entries"]}
    for entry in entries:
        record = version_entries.get(entry["id"], {})
        if record.get("revision") != entry["revision"] or record.get("sha256") != entry["sha256"]:
            issue("version_binding", id=entry["id"])
    if set(version_entries) != set(by_id):
        issue("version_entry_set")
    if version["counts"]["chapters"] != len(entries) or version["counts"]["sources"] != len(sources):
        issue("version_counts")
    if version["counts"].get("active_history_versions") != history_versions:
        issue("version_history_count")

    # A delivered local proposal may already have been frozen by its consumer.
    # Keep its identities even when its topics are subsequently withdrawn.
    retired = read_json(local(version["withdrawn_chapters"]))
    withdrawal_catalog = read_json(local(version["withdrawal_catalog"]))
    withdrawal_entries = {entry["id"]: entry for entry in withdrawal_catalog["entries"]}
    for key in ("withdrawal_catalog", "withdrawal_version"):
        if sha(local(version[key]).read_bytes()) != version[key + "_sha256"]:
            issue("withdrawal_origin_snapshot_hash", file=version[key])
    retired_by_id = {entry["id"]: entry for entry in retired["chapters"]}
    if len(retired_by_id) != len(retired["chapters"]):
        issue("duplicate_withdrawn_id")
    if (retired.get("not_in_active_catalog") is not True
            or retired.get("previous_local_proposal_tree") != version.get("previous_local_proposal_tree")):
        issue("withdrawal_register_identity")
    withdrawn_versions = 0
    for identifier, record in retired_by_id.items():
        if identifier in by_id or local("knowledge/" + record["path"]).exists():
            issue("withdrawn_topic_still_active", id=identifier)
        if (record.get("status") != "withdrawn_from_active_catalog"
                or record.get("id_reserved_do_not_reuse") is not True):
            issue("withdrawn_topic_not_reserved", id=identifier)
        if set(record.get("merged_into", [])) - by_id.keys():
            issue("withdrawal_unknown_destination", id=identifier)
        expected = f"knowledge/history/{identifier}/{record['revision']:04d}"
        if (record["body_archive"] != expected + ".md"
                or record["metadata_archive"] != expected + ".json"):
            issue("withdrawn_archive_path", id=identifier)
        body_path = local(record["body_archive"])
        meta_path = local(record["metadata_archive"])
        if not body_path.is_file() or not meta_path.is_file():
            issue("withdrawn_archive_missing", id=identifier)
            continue
        meta = read_json(meta_path)
        if meta != withdrawal_entries.get(identifier):
            issue("withdrawn_origin_metadata_not_exact", id=identifier)
        if (meta.get("id") != identifier or meta.get("revision") != record["revision"]
                or meta.get("path") != record["path"]
                or meta.get("sha256") != record["sha256"]
                or sha(body_path.read_bytes()) != record["sha256"]):
            issue("withdrawn_archive_identity", id=identifier)
        withdrawn_versions += 1
    if version["counts"].get("withdrawn_proposal_chapters") != withdrawn_versions:
        issue("version_withdrawn_count")

    previous_catalog = read_json(local(version["previous_catalog"]))
    for key in ("previous_catalog", "previous_version"):
        if sha(local(version[key]).read_bytes()) != version.get(key + "_sha256"):
            issue("previous_delivery_snapshot_hash", file=version[key])
    previous_entries = {entry["id"]: entry for entry in previous_catalog["entries"]}
    previous_delta = {"catalog": version["previous_catalog"], "new_ids": [],
                      "revised_ids": [], "unchanged_ids": [], "withdrawn_ids": []}
    for identifier, old in previous_entries.items():
        entry = by_id.get(identifier)
        if entry is None:
            previous_delta["withdrawn_ids"].append(identifier)
            record = retired_by_id.get(identifier)
            if record is None:
                issue("previous_topic_removed_without_archive", id=identifier)
            elif read_json(local(record["metadata_archive"])) != old:
                issue("withdrawn_previous_metadata_not_exact", id=identifier)
            continue
        if entry == old:
            previous_delta["unchanged_ids"].append(identifier)
            continue
        previous_delta["revised_ids"].append(identifier)
        if entry["revision"] != old["revision"] + 1:
            issue("immediate_previous_revision_step", id=identifier)
        history_path = root / f"knowledge/history/{identifier}/{old['revision']:04d}"
        if (not history_path.with_suffix(".md").is_file()
                or sha(history_path.with_suffix(".md").read_bytes()) != old["sha256"]):
            issue("immediate_previous_body_not_exact", id=identifier)
        if (not history_path.with_suffix(".json").is_file()
                or read_json(history_path.with_suffix(".json")) != old):
            issue("immediate_previous_metadata_not_exact", id=identifier)
    for identifier in sorted(by_id.keys() - previous_entries.keys()):
        previous_delta["new_ids"].append(identifier)
        if by_id[identifier]["revision"] != 1:
            issue("new_since_previous_revision", id=identifier)
    if not set(previous_delta["withdrawn_ids"]) <= set(retired_by_id):
        issue("removal_missing_withdrawal_record")
    if set(withdrawal_entries) - set(by_id) != set(retired_by_id):
        issue("withdrawal_set_disagrees_with_origin_catalog")

    # Source-note maintenance does not change the fixed upstream source.
    # Retain the previous note bytes and reject changes to source identity.
    previous_sources_path = local(version["previous_sources"])
    if sha(previous_sources_path.read_bytes()) != version["previous_sources_sha256"]:
        issue("previous_source_registry_hash")
    previous_sources = {source["id"]: source for source in read_json(previous_sources_path)["sources"]}
    added_sources=source_ids-set(previous_sources)
    removed_sources=set(previous_sources)-source_ids
    declared=version.get('source_changes',{})
    if (removed_sources or added_sources!=set(declared.get('added',[]))
            or declared.get('removed',[])!=[]):
        issue("source_set_changed",added=sorted(added_sources),removed=sorted(removed_sources))
    for source in sources:
        if source['id'] in added_sources:
            continue  # New source is fully checked above and explicitly declared.
        prior = previous_sources.get(source["id"], {})
        if {k: v for k, v in source.items() if k != "local_sha256"} != {
                k: v for k, v in prior.items() if k != "local_sha256"}:
            issue("fixed_source_changed", id=source["id"])
        if source["local_sha256"] != prior.get("local_sha256"):
            note = local(source["local_path"], previous_sources_path.parent.parent)
            if not note.is_file() or sha(note.read_bytes()) != prior.get("local_sha256"):
                issue("previous_source_note_not_preserved", id=source["id"])

    delta = {"baseline": baseline, "new_ids": [], "revised_ids": [], "unchanged_ids": []}
    previous_tree_check = "not_requested; portable snapshot and archive hashes checked"
    if baseline:
        def original(relative: str) -> bytes:
            return subprocess.check_output(["git", "show", f"{baseline}:{relative}"], cwd=root)

        old_entries = {e["id"]: e for e in json.loads(original("knowledge/catalog.json"))["entries"]}
        for entry in entries:
            old = old_entries.get(entry["id"])
            if old is None:
                delta["new_ids"].append(entry["id"])
                if entry["id"] not in previous_entries and entry["revision"] != 1:
                    issue("new_entry_revision", id=entry["id"])
                continue
            if entry == old:
                delta["unchanged_ids"].append(entry["id"])
                if local("knowledge/" + old["path"]).read_bytes() != original("knowledge/" + old["path"]):
                    issue("unregistered_body_change", id=entry["id"])
                continue
            delta["revised_ids"].append(entry["id"])
            # The immediate prior delivery is checked above. More than one
            # revision may legitimately separate this base from that delivery.
            if entry["revision"] <= old["revision"]:
                issue("revision_not_advanced_from_baseline", id=entry["id"])
            history_path = root / f"knowledge/history/{entry['id']}/{old['revision']:04d}"
            if history_path.with_suffix(".md").read_bytes() != original("knowledge/" + old["path"]):
                issue("previous_body_not_exact", id=entry["id"])
            if read_json(history_path.with_suffix(".json")) != old:
                issue("previous_metadata_not_exact", id=entry["id"])
        if old_entries.keys() - by_id.keys():
            issue("removed_entry_ids", ids=sorted(old_entries.keys() - by_id.keys()))
        prior_tree = version["previous_local_proposal_tree"]
        tree_available = subprocess.run(
            ["git", "cat-file", "-e", prior_tree + "^{tree}"], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if tree_available:
            previous_tree_check = "available; exact snapshot and withdrawn body bytes checked"
            for archived, old_path in ((version["withdrawal_catalog"], "knowledge/catalog.json"),
                                       (version["withdrawal_version"], "VERSION.json")):
                exact = subprocess.check_output(["git", "show", f"{prior_tree}:{old_path}"], cwd=root)
                if local(archived).read_bytes() != exact:
                    issue("previous_delivery_snapshot_not_exact", file=archived)
            for identifier, record in retired_by_id.items():
                exact = subprocess.check_output(
                    ["git", "show", f"{prior_tree}:knowledge/{record['path']}"], cwd=root)
                if local(record["body_archive"]).read_bytes() != exact:
                    issue("withdrawn_body_not_exact_previous_tree", id=identifier)
        else:
            previous_tree_check = "unavailable; portable snapshot and archive hashes checked"
        for relative in ("knowledge/sources.json", "knowledge/temporal-policy.json"):
            old = json.loads(original(relative))
            current = read_json(root / relative)
            if relative.endswith("temporal-policy.json"):
                old = {k: v for k, v in old.items() if k != "profile_id"}
                current = {k: v for k, v in current.items() if k != "profile_id"}
            else:
                old = {**old, "sources": [{k: v for k, v in item.items() if k != "local_sha256"}
                                          for item in old["sources"]]}
                current = {**current, "sources": [{k: v for k, v in item.items() if k != "local_sha256"}
                                                  for item in current["sources"]]}
            if old != current:
                issue("source_or_policy_changed", file=relative)

    return {"result": "PASS" if not issues else "FAIL", "scope": "Structural, identity and optional-question-example checks only; no certification of model exploration behavior, semantic completeness or engine performance.",
            "chapter_count": len(entries), "source_count": len(sources), "source_cutoff": str(cutoff),
            "archive_files_checked": archive_files, "archive_bytes_checked": archive_bytes,
            "history_versions_checked": history_versions,
            "withdrawn_revisions_checked": withdrawn_versions,
            "previous_delivery_delta": previous_delta, "previous_tree_check": previous_tree_check,
            "local_links_checked": len(links),
            "use_case_count": len(route_ids), "navigation": navigation, "delta": delta,
            "content_bindings": fingerprints, "issues": issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline", help="Git commit to verify exact archival continuity against")
    parser.add_argument("--report", type=Path, help="Write the full reproducible check result here")
    args = parser.parse_args()
    result = validate(args.repository.resolve(), args.baseline)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"content_bindings", "navigation"}},
                     ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
