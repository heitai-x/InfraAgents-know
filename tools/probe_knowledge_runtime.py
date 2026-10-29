#!/usr/bin/env python3
"""Exercise an explicitly supplied InfraAgents reader against an isolated bank.

The probe calls the real RepositoryKnowledge API. It does not run an Agent,
model, GPU, publication, or performance experiment. The supplied runtime is
identified by its imported module bytes, independently of recommended branches.
Only the JSON report is retained; temporary bank and Records copies are removed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import traceback


ROLES = ("planner", "executor", "experimenter", "diagnostician", "evolution")
RESTRICTED = ("S-VLLM-CODE", "S-SGLANG-CODE", "S-TRTLLM-CODE")
NAVIGATION = ("README.md", "DESIGN.md", "DIAGNOSIS.md", "INDEX.md",
              "SOURCES.md", "COVERAGE.md")
RUNTIME_CONTRACT = {
    "read": "Exact catalog ID or path in the current frozen view; ID@revision is rejected.",
    "public_use": "KnowledgeUse contains entry_id and purpose only.",
    "binding": "The supplied runtime's bind_handoff supplies frozen revisions before validate_uses.",
    "scope": "Validates the explicitly supplied runtime; no network fetch or historical-runtime compatibility claim.",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_bytes().decode("utf-8-sig"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def git_identity(root: Path) -> dict:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                            text=True, capture_output=True, check=False)
    if result.returncode or Path(result.stdout.strip()).resolve() != root:
        return {"available": False, "reason": "Supplied directory is not a Git checkout root"}
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                          capture_output=True, check=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], text=True,
                            capture_output=True, check=True).stdout.splitlines()
    return {"available": True, "head": head, "dirty": bool(status)}


def runtime_source(path: Path) -> Path:
    for candidate in (path / "src", path, path.parent if path.name == "infra_agent" else path):
        if (candidate / "infra_agent/knowledge.py").is_file():
            return candidate.resolve()
    raise ValueError("--runtime must contain src/infra_agent or name its src directory")


def input_inventory(root: Path, extra_paths: list[str]) -> dict:
    files = {}
    for name in ("knowledge", "references"):
        for path in sorted((root / name).rglob("*")):
            if path.is_symlink():
                raise ValueError("Probe requires independent regular files: " + str(path))
            if path.is_file():
                files[path.relative_to(root).as_posix()] = file_sha(path)
    for relative in extra_paths:
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and path.is_file(), "Missing local input: " + relative)
        files[relative] = file_sha(path)
    return files


def inventory_summary(files: dict) -> dict:
    encoded = json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {"files": len(files), "inventory_sha256": sha(encoded),
            "catalog_sha256": files["knowledge/catalog.json"],
            "sources_sha256": files["knowledge/sources.json"]}


def paged_read(bank, name: str) -> tuple[str, dict]:
    chunks, offset, ranges = [], 0, []
    identity = None
    for _ in range(10000):
        result = bank.read(name, offset)
        stable = {key: result.get(key) for key in
                  ("path", "entry_id", "revision", "sha256", "total_characters")}
        require(identity is None or stable == identity, "Reader identity changed during pagination: " + name)
        identity = stable
        content = result["content"]
        require(isinstance(content, str), "Reader returned nontext: " + name)
        chunks.append(content)
        end = offset + len(content)
        ranges.append([offset, end])
        require(end <= result["total_characters"], "Reader crossed advertised length: " + name)
        following = result.get("next_offset")
        if following is None or following == result["total_characters"]:
            require(end == result["total_characters"], "Reader stopped before complete text: " + name)
            joined = "".join(chunks)
            require(len(joined) == result["total_characters"], "Reader omitted a range: " + name)
            require(sha(joined.encode("utf-8")) == result["sha256"], "Decoded body differs from read SHA: " + name)
            return joined, {**stable, "pages": len(chunks), "ranges": ranges, "complete": True}
        require(type(following) is int and following == end and following > offset,
                "Reader offset did not advance contiguously: " + name)
        offset = following
    raise AssertionError("Unbounded reader pagination: " + name)


def rejection(action, description: str) -> dict:
    try:
        action()
    except ValueError as error:
        return {"check": description, "rejected": True, "error_type": type(error).__name__,
                "message": str(error)}
    raise AssertionError("Expected rejection was accepted: " + description)


def revision_body(root: Path, entry: dict, current: dict) -> bytes:
    if entry["revision"] == current[entry["id"]]["revision"]:
        return (root / "knowledge" / current[entry["id"]]["path"]).read_bytes()
    return (root / "knowledge/history" / entry["id"] / f'{entry["revision"]:04d}.md').read_bytes()


def validate_public_uses(bank, role, uses, api):
    """Exercise public parsing and the Controller's existing version binding."""
    public = {"knowledge_used": [api["KnowledgeUse"].from_mapping(use).to_mapping() for use in uses]}
    bound = api["bind_handoff"](bank, public, role=role, purpose="engineering", work_id="probe",
                                access=None, visible=[])
    require(public["knowledge_used"] == uses, "Binding changed the public declaration")
    require([{key: value for key, value in item.items() if key != "revision"}
             for item in bound["knowledge_used"]] == uses, "Binding changed declared content")
    return bank.validate_uses(role, bound["knowledge_used"])


def question_examples(document: dict, catalog: dict) -> list[dict]:
    """Validate example identities, never turn their contents into Task gates."""
    require(document.get("schema_version") == 2, "Expected the open-question example schema (version 2)")
    uses = document.get("uses")
    require(isinstance(uses, list), "Question examples must be a list")
    known = {entry["id"] for entry in catalog["entries"]}
    seen = set()
    for example in uses:
        identifier = example.get("id")
        require(isinstance(identifier, str) and identifier and identifier not in seen,
                "Question example identity is missing or repeated")
        seen.add(identifier)
        require(isinstance(example.get("question"), str) and example["question"].strip(),
                "Question example needs its question text")
        require(isinstance(example.get("knowledge_ids"), list)
                and set(example["knowledge_ids"]) <= known, "Question example names an unknown chapter")
        require(isinstance(example.get("retrieval_queries"), list)
                and all(isinstance(query, str) and query.strip() for query in example["retrieval_queries"]),
                "Example queries must be nonempty strings")
    return uses


def probe_queries(bank, uses: list[dict], allowed: dict, *, empty: bool) -> list[dict]:
    results = []
    for use in uses:
        # Preserve the earlier probe's retrieval modes for these two examples.
        # This is a test setting, not runtime routing or an execution sequence.
        mode = "diagnosis" if use["id"] in {"U04", "U06"} else "design"
        for query in use["retrieval_queries"]:
            result = bank.search(query, mode=mode, limit=6)
            matches = result["matches"]
            require(len(matches) <= 6, "Search ignored requested limit")
            for item in matches:
                require(item["id"] in allowed, "Search exposed an excluded entry")
                require(item["revision"] == allowed[item["id"]]["revision"]
                        and item["sha256"] == allowed[item["id"]]["sha256"],
                        "Search returned a different revision identity")
            if empty:
                require(not matches, "No-knowledge search exposed a chapter")
            found = {item["id"] for item in matches}
            results.append({"question_example": use["id"], "question": use["question"],
                            "query": query, "mode": mode,
                            "example_entry_ids": use["knowledge_ids"],
                            "example_entry_hits": sorted(found & set(use["knowledge_ids"])),
                            "matches": [{key: item[key] for key in
                                         ("id", "revision", "sha256", "score", "matched_terms")}
                                        for item in matches],
                            "interpretation": "Optional-entry observation only. A missing listed hit is not Task failure or grounds to exclude a design."})
    return results


def withdrawn_archive_probe(root, catalog, document, read_revision) -> dict:
    """Bind retained archives without treating them as active runtime history."""
    require(document.get("schema_version") == 1, "Unknown withdrawn-chapter registry schema")
    chapters = document.get("chapters")
    require(isinstance(chapters, list), "Withdrawn registry must contain a chapter list")
    active = {entry["id"] for entry in catalog["entries"]}
    active_paths = {entry["path"] for entry in catalog["entries"]}
    seen, records = set(), []
    for chapter in chapters:
        identifier, revision = chapter["id"], chapter["revision"]
        require(identifier not in active and identifier not in seen,
                "Withdrawn identity is active or repeated: " + identifier)
        seen.add(identifier)
        require(type(revision) is int and revision > 0, "Invalid withdrawn revision")
        require(chapter["status"] == "withdrawn_from_active_catalog", "Unknown withdrawal status")
        require(chapter["path"] not in active_paths, "Withdrawn original path is active")
        require(set(chapter["merged_into"]) <= active, "Withdrawal names an unknown merge target")
        original = (root / "knowledge" / chapter["path"]).resolve()
        require(original.is_relative_to(root / "knowledge") and not original.exists(),
                "Withdrawn original body still exists: " + chapter["path"])
        archived = {}
        for key, suffix in (("body_archive", ".md"), ("metadata_archive", ".json")):
            expected = f"knowledge/history/{identifier}/{revision:04d}{suffix}"
            require(chapter[key] == expected, "Withdrawal has an unexpected archive path")
            path = root / chapter[key]
            require(path.is_file(), "Missing withdrawn archive: " + chapter[key])
            archived[key] = {"path": chapter[key], "sha256": file_sha(path),
                             "bytes": path.stat().st_size}
        metadata = read_json(root / chapter["metadata_archive"])
        require(all(metadata[key] == chapter[key] for key in ("id", "path", "revision", "sha256")),
                "Withdrawn metadata differs from its retained identity: " + identifier)
        require(archived["body_archive"]["sha256"] == chapter["sha256"],
                "Withdrawn body differs from its retained identity: " + identifier)
        rejected = rejection(lambda: read_revision(root, identifier, revision),
                             "Active history API must reject withdrawn " + identifier)
        records.append({"entry_id": identifier, "revision": revision,
                        "sha256": chapter["sha256"], "original_path": chapter["path"],
                        "merged_into": chapter["merged_into"], **archived,
                        "active_history_api_rejection": rejected})
    return {"result": "PASS", "chapter_count": len(records), "chapters": records,
            "declared_previous_proposal_tree": document.get("previous_local_proposal_tree"),
            "scope": "Retained body bytes match the withdrawal registry and archived metadata; metadata file hashes are recorded, without independently comparing the declared previous Git tree. The active read_revision API rejects these catalog-absent IDs; they are not counted as active historical revisions."}


def probe_view(root, records_root, name, config, models, uses, catalog, registry, withdrawn, api):
    records = api["Records"](records_root / name)
    bank = api["RepositoryKnowledge"](root, records, selection=config, models=models)
    snapshot = bank.prepare_work("probe")
    bank.begin_invocation("probe", "planner", "probe", conversation_id="probe-" + name, generation="one")
    result = {"policy": config, "selection_sha256": bank.selection["sha256"],
              "date_basis": bank.selection["date_basis"], "model_binding": models,
              "model_binding_is_fixture_only": True, "index": bank.index()}
    try:
        current = {entry["id"]: entry for entry in catalog["entries"]}
        entries = snapshot["catalog"]["entries"]
        allowed = {entry["id"]: entry for entry in entries}
        empty = config["condition"] == "none"
        if empty:
            require(not allowed and not snapshot["sources"]["sources"], "none view is not empty")
        else:
            # This particular base contains seed chapters only. Report that
            # limitation instead of inventing a learned entry or trial.
            require(set(allowed) == set(current), "This seed base was unexpectedly filtered")
        require(bank.index()["entry_count"] == len(entries), "Index count differs from frozen view")
        require(bank.case_index()["count"] == 0, "This seed-only base unexpectedly exposes cases")
        result["chapter_count"] = len(entries)
        result["source_count"] = len(snapshot["sources"]["sources"])
        result["negative_checks"] = []
        withdrawn_checks = []
        for chapter in withdrawn:
            identifier = chapter["id"]
            paths = [identifier, f'{identifier}@{chapter["revision"]}', chapter["path"],
                     "knowledge/" + chapter["path"], chapter["body_archive"], chapter["metadata_archive"]]
            prefix = "knowledge/history/" + identifier + "/"
            require(identifier not in allowed, "Withdrawn ID entered a selected catalog")
            require("knowledge/" + chapter["path"] not in snapshot["text_sha256"]
                    and not any(path.startswith(prefix) for path in snapshot["text_sha256"]),
                    "Withdrawn original/history path entered a frozen view")
            blocked = [rejection(lambda path=path: bank.read(path), path) for path in paths]
            use_rejection = rejection(lambda: validate_public_uses(bank, "planner", [{
                "entry_id": identifier,
                "purpose": "withdrawn-entry negative control"
            }], api), "Declare withdrawn " + identifier)
            withdrawn_checks.append({"entry_id": identifier, "excluded_from_catalog": True,
                                     "excluded_from_snapshot": True, "read_rejections": blocked,
                                     "use_declaration_rejection": use_rejection})
        result["withdrawn_chapters"] = withdrawn_checks
        if entries:
            first = entries[0]
            result["negative_checks"].append(rejection(lambda: validate_public_uses(bank, "planner", [
                {"entry_id": first["id"], "purpose": "unread negative control"}
            ], api), "Declare an unread current chapter"))
            structure=bank.read(first['id'],view='structure')
            require(structure['recorded_as_chapter_read'] is False,'Structure was counted as a chapter read')
            result['negative_checks'].append(rejection(lambda: validate_public_uses(bank,'planner',[
                {'entry_id':first['id'],'purpose':'structure-only negative control'}],api),
                'Structure navigation alone must not permit adoption'))
            if structure['sections']:
                title=structure['sections'][0]['title']
                section=bank.read(first['id'],section=title)
                expected=revision_body(root,first,current).decode('utf-8')
                bounds=section['chapter_range']
                require(section['content']==expected[bounds[0]:bounds[1]],'Section returned incorrect frozen character range')
                result['section_navigation']={'entry_id':first['id'],'section':title,'chapter_range':bounds}
            require(len(bank.index(limit=1)['chapters'])==1,'Catalog paging ignored the requested limit')
        reads = []
        for entry in entries:
            identifier = entry["id"]
            body, read = paged_read(bank, identifier)
            expected = revision_body(root, entry, current)
            require(body == expected.decode("utf-8") and read["sha256"] == entry["sha256"],
                    "Frozen chapter differs from registered bytes: " + identifier)
            require(read["entry_id"] == entry["id"] and read["revision"] == entry["revision"],
                    "Reader lost exact revision identity")
            aliases = [entry["path"], "knowledge/" + entry["path"]]
            for alias in aliases:
                alternate = bank.read(alias)
                require(all(alternate[key] == read[key] for key in ("entry_id", "revision", "sha256")),
                        "ID/path alias changed version binding: " + alias)
                require(alternate["content"] == body[:20000], "ID/path alias returned different text")
            reads.append({**read, "requested": identifier, "verified_aliases": aliases})
        result["chapter_reads"] = reads
        declarations = [{"entry_id": entry["id"],
                         "purpose": "Reader/identity probe; no model adoption or engineering execution claim"}
                        for entry in entries]
        frozen_uses = validate_public_uses(bank, "planner", declarations, api)
        for item in frozen_uses:
            require(item["sha256"] == allowed[item["entry_id"]]["sha256"], "Use declaration changed body identity")
            require(file_sha(records.directory / item["path"]) == item["sha256"], "Frozen use body differs")
        result["validated_uses"] = frozen_uses
        result["public_use_fields"] = ["entry_id", "purpose"]
        if entries:
            first = entries[0]
            result["negative_checks"].append(rejection(
                lambda: bank.read(f'{first["id"]}@{first["revision"]}'), "Read using removed ID@revision syntax"))
            result["negative_checks"].append(rejection(lambda: validate_public_uses(bank, "planner", [
                {"entry_id": first["id"], "revision": first["revision"], "purpose": "obsolete public field"}
            ], api), "Public declaration must reject a caller-supplied revision"))
            result["negative_checks"].append(rejection(lambda: validate_public_uses(
                bank, "planner", [declarations[0], declarations[0]], api), "Duplicate public adoption"))
            result["negative_checks"].append(rejection(lambda: bank.validate_uses("planner", [
                {"entry_id": first["id"], "revision": first["revision"] + 1, "purpose": "wrong revision control"}
            ]), "Internal bound declaration must reject a nonselected revision"))
        navigation = {}
        for filename in NAVIGATION:
            body, read = paged_read(bank, "knowledge/" + filename)
            linked = set(re.findall(r"K-[A-Z0-9]+-[0-9]+", body))
            require(linked <= set(allowed), "Navigation leaked an excluded ID: " + filename)
            if filename in {"DESIGN.md", "DIAGNOSIS.md", "INDEX.md"}:
                require(linked == set(allowed), "Navigation omits a selected entry: " + filename)
            navigation[filename] = {**read, "entry_ids": sorted(linked)}
        result["navigation"] = navigation
        for relative, expected in (("knowledge/catalog.json", snapshot["catalog"]),
                                   ("knowledge/sources.json", snapshot["sources"])):
            body, _ = paged_read(bank, relative)
            require(json.loads(body) == expected, "Runtime JSON exposes a different view: " + relative)
        result["queries"] = probe_queries(bank, uses, allowed, empty=empty)
        exposed_sources = {source["id"]: source for source in snapshot["sources"]["sources"]}
        restrictions = []
        archive_index, _ = paged_read(bank, "references/index.html")
        for sid in config.get("restricted_originals", []):
            source = exposed_sources[sid]
            _, card = paged_read(bank, source["local_path"])
            require(card["sha256"] == source["local_sha256"], "Restricted source card differs")
            require("<td>" + sid + "</td>" not in archive_index, "Restricted source remains in archive index")
            prefix = "references/" + sid + "/"
            require(not any(path.startswith(prefix) for path in snapshot["text_sha256"]),
                    "Restricted archive members leaked into the snapshot")
            manifest = read_json(root / source["archive_manifest"])
            members = [item["path"] for item in manifest["files"]
                       if "/" in item["path"] and Path(item["path"]).suffix in {".py", ".cu", ".cpp", ".h", ".rst"}]
            paths = [prefix + leaf for leaf in ("manifest.json", "fulltext.txt", "reader.html")]
            if members:
                paths.append(prefix + members[0])
            blocked = [rejection(lambda path=path: bank.read(path), path) for path in paths]
            restrictions.append({"source_id": sid, "card_read": card, "blocked_reads": blocked,
                                 "all_archive_members_excluded_from_snapshot": True})
        result["restricted_originals"] = restrictions
        reading_path=root/'knowledge/source-reading.json'
        if reading_path.is_file():
            reading=read_json(reading_path)['sources']
            checks=[]
            for sid,source in exposed_sources.items():
                if sid in config.get('restricted_originals',[]):continue
                prefix='references/'+sid+'/'
                if prefix+'fulltext.txt' in reading[sid]['paths']:continue
                _,note=paged_read(bank,source['local_path'])
                require(note['sha256']==source['local_sha256'],'Algorithm source note lost its identity')
                manifest=read_json(root/source['archive_manifest'])
                names=[prefix+'reader.html',prefix+'fulltext.txt']
                implementation=next((prefix+x['path'] for x in manifest['files'] if x['path'].startswith('upstream/')
                    and Path(x['path']).suffix in {'.py','.h','.cu','.cpp','.c','.hpp','.sh'}),None)
                if implementation:names.append(implementation)
                blocked=[rejection(lambda name=name:bank.read(name),name) for name in names]
                primary=reading[sid]['primary']
                if primary:paged_read(bank,primary)
                checks.append({'source_id':sid,'blocked':blocked,'readable_document':primary})
            result['curated_source_reading']=checks
        if empty:
            exemplar = catalog["entries"][0]
            source = registry["sources"][0]
            attempts = [exemplar["id"], f'{exemplar["id"]}@{exemplar["revision"]}',
                        "knowledge/" + exemplar["path"], source["local_path"],
                        "references/" + source["id"] + "/fulltext.txt"]
            result["negative_checks"].extend(rejection(lambda path=path: bank.read(path), path) for path in attempts)
            require("<td>S-" not in archive_index, "none archive index exposes source entries")
        elif name == "attributable_seed":
            source = exposed_sources["S-CUDA-BEST"]
            original_path = "references/S-CUDA-BEST/fulltext.txt"
            body, read = paged_read(bank, original_path)
            require(body == (root / original_path).read_bytes().decode("utf-8"), "Allowed original was not read exactly")
            manifest = read_json(root / source["archive_manifest"])
            expected = next(item["sha256"] for item in manifest["files"] if item["path"] == "fulltext.txt")
            require(read["sha256"] == expected, "Allowed original is not manifest-bound")
            require(read["pages"] > 1, "Chosen allowed original did not exercise continuation pagination")
            result["allowed_original"] = {"source_id": source["id"], **read}
        result["result"] = "PASS"
        return result
    finally:
        bank.end_invocation()


def history_probe(root, catalog, read_revision) -> dict:
    versions = []
    current = {entry["id"]: entry for entry in catalog["entries"]}
    for entry in catalog["entries"]:
        # Current bodies have already passed the actual frozen reader in every
        # nonempty view. Exercise the history API only for prior revisions.
        for revision in range(1, entry["revision"]):
            metadata, body = read_revision(root, entry["id"], revision)
            require(metadata["id"] == entry["id"] and metadata["revision"] == revision,
                    "History reader changed identity")
            require(sha(body) == metadata["sha256"], "History body does not match registered SHA")
            require(body == revision_body(root, metadata, current), "History reader returned different bytes")
            versions.append({"entry_id": entry["id"], "revision": revision, "sha256": sha(body),
                             "bytes": len(body), "historical": True})
    return {"result": "PASS", "versions": versions, "total_versions_read": len(versions),
            "historical_versions_read": sum(item["historical"] for item in versions),
            "scope": "All prior revisions via read_revision; current revisions via the frozen reader above. Frozen role views reject nonselected revisions."}


def imported_modules(source: Path) -> list[dict]:
    modules = []
    for name, module in sorted(sys.modules.items()):
        if name != "infra_agent" and not name.startswith("infra_agent."):
            continue
        location = getattr(module, "__file__", None)
        if not location:
            continue
        path = Path(location).resolve()
        require(path.is_relative_to(source), "An infra_agent module came from outside --runtime: " + str(path))
        modules.append({"module": name, "path": str(path),
                        "relative_path": path.relative_to(source).as_posix(), "sha256": file_sha(path)})
    return modules


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True, help="InfraAgents checkout or its src directory")
    parser.add_argument("--repository", type=Path, required=True, help="Knowledge repository to copy and inspect")
    parser.add_argument("--report", type=Path, required=True, help="Output JSON report (outside knowledge/references)")
    args = parser.parse_args()
    root, report_path = args.repository.resolve(), args.report.resolve()
    for forbidden in (root / "knowledge", root / "references"):
        if report_path.is_relative_to(forbidden):
            parser.error("--report must not overwrite knowledge or source assets")
    if report_path.suffix != ".json":
        parser.error("--report must name a JSON file")
    report = {"schema_version": 3, "result": "FAIL", "started_at": datetime.now(timezone.utc).isoformat(),
              "repository": str(root), "repository_git": git_identity(root),
              "script_sha256": file_sha(Path(__file__).resolve()), "python": sys.version,
              "runtime_contract": RUNTIME_CONTRACT, "views": {}, "errors": [],
              "limits": [
                  "Real local reader API calls only; no Agent process, model request, GPU execution, publication or serving benchmark.",
                  "Declared role model identities are selection fixtures, not evidence of a model call or its training data.",
                  "All input chapters are seed; the seed_and_learned view checks this corpus but does not test promotion of real learned assets.",
                  "Complete reads here do not establish that validate_uses enforces full reading or that a role adopted the content correctly.",
                  "Listed-entry hits describe the supplied example queries, not general retrieval quality, scientific benefit or task success. A missing example hit is not an API failure.",
                  "History checks verify registered bytes; restricted-original checks concern runtime access, not external filesystem authorization.",
              ]}
    source = None
    before = None
    extra = []
    try:
        source = runtime_source(args.runtime.resolve())
        report["runtime"] = {"source_directory": str(source), "git": git_identity(source.parent)}
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source))
        api = {"RepositoryKnowledge": importlib.import_module("infra_agent.knowledge").RepositoryKnowledge,
               "Records": importlib.import_module("infra_agent.records").Records,
               "read_revision": importlib.import_module("infra_agent.knowledge_store").read_revision,
               "KnowledgeUse": importlib.import_module("infra_agent.role_contracts").KnowledgeUse,
               "bind_handoff": importlib.import_module("infra_agent.knowledge_bindings").bind_handoff}
        policy = read_json(root / "knowledge/temporal-policy.json")
        extra = ["docs/actionable-knowledge/use-cases.json",
                 "docs/actionable-knowledge/withdrawn-chapters.json", policy["model"]["metadata_evidence"]]
        before = input_inventory(root, extra)
        report["canonical_before"] = inventory_summary(before)
        with tempfile.TemporaryDirectory(prefix="infraagents-knowledge-probe-") as scratch:
            scratch_path = Path(scratch)
            bank_root = scratch_path / "bank"
            bank_root.mkdir()
            for directory in ("knowledge", "references", "docs"):
                shutil.copytree(root / directory, bank_root / directory)
            require(input_inventory(bank_root, extra) == before, "Copied bank does not match the original input snapshot")
            report["isolation"] = {"strategy": "Independent file copies; no symlinks or hardlinks to canonical bank",
                                   "temporary_directory_removed_on_exit": True}
            catalog = read_json(bank_root / "knowledge/catalog.json")
            registry = read_json(bank_root / "knowledge/sources.json")
            question_document = read_json(bank_root / "docs/actionable-knowledge/use-cases.json")
            uses = question_examples(question_document, catalog)
            withdrawn_document = read_json(bank_root / "docs/actionable-knowledge/withdrawn-chapters.json")
            report["withdrawn_archives"] = withdrawn_archive_probe(
                bank_root, catalog, withdrawn_document, api["read_revision"])
            report["withdrawn_archives"]["registry_sha256"] = before["docs/actionable-knowledge/withdrawn-chapters.json"]
            require(all(entry["status"] == "seed" and not entry.get("origin", {}).get("run_id")
                        for entry in catalog["entries"]), "Probe expectations are for the declared seed-only base")
            require(set(RESTRICTED) <= {source["id"] for source in registry["sources"]}, "Missing restricted source fixtures")
            models = {role: policy["model"]["id"] for role in ROLES}
            common = {"freeze_initial": True, "publication": "deferred"}
            configurations = {
                "full_seed": {**common, "view": "full", "condition": "seed"},
                "full_seed_and_learned": {**common, "view": "full", "condition": "seed_and_learned"},
                "attributable_seed": {**common, "view": "attributable", "condition": "seed",
                                      "profile": "knowledge/temporal-policy.json", "restricted_originals": list(RESTRICTED)},
                "none": {**common, "view": "full", "condition": "none"},
            }
            for name, config in configurations.items():
                try:
                    report["views"][name] = probe_view(bank_root, scratch_path / "records", name,
                                                      config, models, uses, catalog, registry,
                                                      withdrawn_document["chapters"], api)
                except Exception as error:
                    report["errors"].append({"stage": name, "type": type(error).__name__,
                                             "message": str(error), "traceback": traceback.format_exc()})
                    report["views"][name] = {"result": "FAIL", "policy": config}
                print(json.dumps({"phase": name, "result": report["views"][name]["result"]}),
                      file=sys.stderr, flush=True)
            try:
                report["history"] = history_probe(bank_root, catalog, api["read_revision"])
            except Exception as error:
                report["errors"].append({"stage": "history", "type": type(error).__name__,
                                         "message": str(error), "traceback": traceback.format_exc()})
            query_cases = [query for use in uses for query in use["retrieval_queries"]]
            report["query_fixture"] = {"schema_version": question_document["schema_version"],
                                       "example_count": len(uses), "query_case_count": len(query_cases),
                                       "distinct_query_count": len(set(query_cases)),
                                       "sha256": before["docs/actionable-knowledge/use-cases.json"],
                                       "authority": "Examples are non-exhaustive navigation material, not stage definitions, candidate restrictions or a Task grading rubric."}
            report["retrieval_route_observations"] = {
                name: {"query_count": len(view.get("queries", [])),
                       "queries_with_listed_entry_hit": sum(bool(item["example_entry_hits"]) for item in view.get("queries", [])),
                       "queries_without_listed_entry_hit": [item["query"] for item in view.get("queries", []) if not item["example_entry_hits"]]}
                for name, view in report["views"].items() if name != "none"
            }
            report["retrieval_route_observations_note"] = "Reported independently of API correctness. No requirement to use listed chapters, no relevance threshold, no forced design diversity and no task-quality inference."
    except Exception as error:
        report["errors"].append({"stage": "setup_or_copy", "type": type(error).__name__,
                                 "message": str(error), "traceback": traceback.format_exc()})
    finally:
        if source is not None:
            try:
                report["imported_runtime_modules"] = imported_modules(source)
            except Exception as error:
                report["errors"].append({"stage": "runtime_identity", "message": str(error)})
        if before is not None:
            try:
                after = input_inventory(root, extra)
                report["canonical_after"] = inventory_summary(after)
                report["canonical_unchanged"] = before == after
                if before != after:
                    report["errors"].append({"stage": "canonical_identity", "message": "Canonical input changed during probe",
                                             "changed_paths": sorted(path for path in before.keys() | after.keys()
                                                                     if before.get(path) != after.get(path))})
            except Exception as error:
                report["errors"].append({"stage": "canonical_identity", "message": str(error)})
        complete = (len(report["views"]) == 4
                    and all(view["result"] == "PASS" for view in report["views"].values())
                    and report.get("history", {}).get("result") == "PASS"
                    and report.get("withdrawn_archives", {}).get("result") == "PASS"
                    and report.get("canonical_unchanged") is True)
        if not complete and not report["errors"]:
            report["errors"].append({"stage": "completion", "message": "Probe did not complete every required stage"})
        report["result"] = "PASS" if complete and not report["errors"] else "FAIL"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"result": report["result"], "report": str(report_path),
               "views": {name: {"result": view["result"], "chapters": view.get("chapter_count"),
                                "queries": len(view.get("queries", []))}
                         for name, view in report["views"].items()},
               "history_versions": report.get("history", {}).get("historical_versions_read"),
               "withdrawn_chapters": report.get("withdrawn_archives", {}).get("chapter_count"),
               "canonical_unchanged": report.get("canonical_unchanged"), "errors": report["errors"]}
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
