"""Resumable author/reviewer workflow; semantic decisions remain Agent decisions.

Only the coordinator mutates queue state. Authors own separate chapter files.
No model API, upstream code distribution, Task switch or remote publication.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ["definition", "legality", "algorithm", "cost", "adaptation", "diagnosis"]
LABELS = ["动机、数学、输入输出和状态", "合法条件与方案选择", "完整算法、数据流和状态变化",
          "计算、访存、通信、准备、同步和驻留成本", "条件变化后的保留、调整与放弃", "竞争解释、区分实验与反证修正"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside knowledge repository: {relative}")
    return path


def initialize(root, run_dir, selected="all", max_rounds=3):
    root, run_dir = root.resolve(), run_dir.resolve()
    if not run_dir.is_relative_to(root / "docs/evidence"):
        raise ValueError("Run directory must be under docs/evidence")
    if (run_dir / "workflow.json").exists() or (run_dir / "before").exists():
        raise ValueError("Run already exists; use status/next to resume")
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    catalog, sources = read(root / "knowledge/catalog.json"), read(root / "knowledge/sources.json")
    status_file = root / "docs/expansion-status.json"
    accounting = {x["id"]: x for x in read(status_file)["entries"]} if status_file.exists() else {}
    entries = {x["id"]: x for x in catalog["entries"]}
    if selected == "all":
        ids = list(entries)
    elif selected == "pending":
        ids = [key for key in entries if "pending" in accounting.get(key, {}).get("review", "pending")]
    else:
        ids = list(dict.fromkeys(selected.split(",")))
    if not ids or set(ids) - entries.keys():
        raise ValueError(f"Empty or unknown selection: {ids}")
    # Preflight every registered byte before creating a baseline.
    for entry in entries.values():
        if digest(root / "knowledge" / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Unregistered chapter modification: {entry['id']}")
    for source in sources["sources"]:
        if digest(inside(root, source["local_path"])) != source["local_sha256"]:
            raise ValueError(f"Unregistered source-note modification: {source['id']}")
    paths = {"VERSION.json", "knowledge/catalog.json", "knowledge/sources.json", "knowledge/temporal-policy.json"}
    paths.update("knowledge/" + e["path"] for e in entries.values())
    paths.update(s["local_path"] for s in sources["sources"])
    for relative in ["knowledge/source-reading.json", "docs/expansion-status.json", "README.md", "knowledge/README.md",
                     "docs/knowledge-method.md", "docs/knowledge-review.md", "docs/BUILD-STATUS.md",
                     "knowledge/templates/entry.md", "docs/actionable-knowledge/change-manifest.json"]:
        if (root / relative).is_file():
            paths.add(relative)
    baseline = {}
    for name in sorted(paths):
        destination = run_dir / "before" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(inside(root, name).read_bytes())
        baseline[name] = digest(destination)
    write(run_dir / "before/snapshot.json", baseline)
    jobs = {}
    for key in ids:
        entry = entries[key]
        jobs[key] = {"id": key, "path": "knowledge/" + entry["path"], "baseline_revision": entry["revision"],
                     "baseline_sha256": entry["sha256"], "prior_review": accounting.get(key, {}).get("review", "unknown"),
                     "source_ids": entry.get("source_ids", []), "related": entry.get("related", []),
                     "state": "queued", "round": 0, "author": None, "reviews": [], "checks": [],
                     "agent_use": "unassessed", "issues": []}
    state = {"schema_version": 1, "created_at": now(), "baseline_version": read(root / "VERSION.json")["version_id"],
             "baseline_sha256": digest(root / "VERSION.json"), "max_rounds": max_rounds,
             "questions": QUESTIONS, "scope": "Whole selected queue, not a certificate of semantic completeness or device performance.",
             "jobs": jobs}
    write(run_dir / "workflow.json", state)
    return summary(state)


def summary(state):
    return {"baseline_version": state["baseline_version"], "total": len(state["jobs"]),
            "counts": dict(Counter(x["state"] for x in state["jobs"].values())),
            "all_content_accepted": all(x["state"] == "verified" for x in state["jobs"].values()),
            "agent_use_counts": dict(Counter(x["agent_use"] for x in state["jobs"].values()))}


def claim(run_dir, owner, limit=1, entry_id=None):
    state = read(run_dir / "workflow.json")
    if limit < 1 or not owner.strip():
        raise ValueError("A named author and positive limit are required")
    choices = [state["jobs"][entry_id]] if entry_id else list(state["jobs"].values())
    claimed = []
    for job in choices:
        if job["state"] not in {"queued", "revise", "checks_failed"}:
            continue
        if job["round"] >= job.get("max_rounds", state["max_rounds"]):
            job["state"] = "needs_attention"
            continue
        job.update(state="working", author=owner, round=job["round"] + 1)
        job["checks"] = []
        claimed.append(job["id"])
        if len(claimed) == limit:
            break
    write(run_dir / "workflow.json", state)
    return {"claimed": claimed, **summary(state)}


def make_prompt(root, run_dir, entry_id, role):
    state = read(run_dir / "workflow.json")
    job = state["jobs"][entry_id]
    sources = {x["id"]: x for x in read(root / "knowledge/sources.json")["sources"]}
    lines = [f"# {role}: {entry_id}", "", f"知识库根目录：{root}", f"当前章节：{job['path']}",
             f"本轮产物目录：{run_dir.relative_to(root).as_posix()}/{entry_id}/round-{job['round']}",
             "先读 docs/knowledge-method.md、docs/knowledge-review.md、正文和必要来源说明。", "",
             "六项标准（保持不变）："]
    lines += [f"- {key}: {label}" for key, label in zip(QUESTIONS, LABELS)]
    lines += ["", "来源说明："] + [f"- {key}: {sources[key]['local_path']}" for key in job["source_ids"]]
    lines += ["", "相关条目：" + ", ".join(job["related"]), "",
              "来源事实服从当前 cutoff；论文与文档可读，源码由作者核验后转成算法/伪代码。",
              "缺口必须来自当前文本；已修复的问题关闭，不增加统一章节标题、例子数或无关验收条件。"]
    if role == "author":
        lines += ["只修改本章及自己的检查/证据文件；共享来源说明先由协调者分配单一责任方。",
                  "不要修改 catalog、VERSION、历史、队列或其他作者正文。",
                  "给出核心算法、边界、条件化成本、反例和区分实验；需要硬件映射或组合轨迹时完整展开。",
                  "检查应有独立参考或能触发错误的负对照。报告模型范围，不能把CPU检查写成GPU实测。",
                  "提交文件清单、真实检查命令和具体遗留边界。"]
    elif role == "reviewer":
        lines += ["由非本章作者复核完整阅读链，实际检查算法与成本，不按文风或标题打分。",
                  "使用 docs/knowledge-build-workflow.md 的 review JSON 格式；六项分别引用可核对的原文。",
                  "blocking findings 未解决则 decision=revise；正确且声明清楚的验证边界可保留为 limitation。",
                  "核对 author 判断与测试范围，不自行把未执行的检查记成PASS。"]
    elif role == "learner":
        lines += ["只使用允许的知识材料完成正文未演示的设计/实现/诊断任务。",
                  "不要阅读作者检查脚本、报告、其他答卷和上游实现；保留初始答案后再接收反证并另存修正。",
                  "如无操作系统隔离或共享父上下文，应明确记录；该练习不等同受控知识增益或GPU证明。"]
    if job["issues"]:
        lines += ["", "本轮待处理问题：", json.dumps(job["issues"], ensure_ascii=False, indent=2)]
    return "\n".join(lines) + "\n"


def validate_review(root, job, review):
    if review.get("entry_id") != job["id"] or review.get("body_sha256") != digest(root / job["path"]):
        raise ValueError("Review does not bind the current chapter bytes")
    if not review.get("reviewer") or review["reviewer"] == job["author"]:
        raise ValueError("A reviewer distinct from the recorded author is required")
    if review.get("decision") not in {"accept", "revise"}:
        raise ValueError("decision must be accept or revise")
    criteria = review.get("criteria", {})
    if set(criteria) != set(QUESTIONS):
        raise ValueError("Review must answer the original six criteria exactly")
    bindings = review.get("documents", {})
    if job["path"] not in bindings:
        raise ValueError("Review must record actual chapter reading")
    for name, sha in bindings.items():
        if digest(inside(root, name)) != sha:
            raise ValueError(f"Stale reviewed file: {name}")
    for key, item in criteria.items():
        if item.get("assessment") not in {"sufficient", "partial", "not_applicable"} or not item.get("reason"):
            raise ValueError(f"Missing scoped assessment: {key}")
        if item["assessment"] == "partial" and review["decision"] == "accept":
            raise ValueError(f"Unresolved content criterion: {key}")
        evidence = item.get("evidence", [])
        if item["assessment"] != "not_applicable" and not evidence:
            raise ValueError(f"Missing concrete evidence: {key}")
        for point in evidence:
            path, quote = point["path"], point["quote"]
            if path not in bindings or len(quote.strip()) < 8 or quote not in inside(root, path).read_text(encoding="utf-8-sig"):
                raise ValueError(f"Evidence is absent or unbound: {key}")
    if any(x.get("blocking") for x in review.get("findings", [])) and review["decision"] == "accept":
        raise ValueError("Blocking findings prevent acceptance")
    for field in ["findings", "limitations", "checks"]:
        if not isinstance(review.get(field), list):
            raise ValueError(f"Review requires list field: {field}")
    if not review["checks"] and not review.get("no_checks_reason"):
        raise ValueError("Declare meaningful checks or explain why none are appropriate")


def record_review(root, run_dir, entry_id, review_path):
    state = read(run_dir / "workflow.json")
    job = state["jobs"][entry_id]
    if job["state"] not in {"working", "reviewed"}:
        raise ValueError(f"Cannot review state {job['state']}")
    review_path = inside(root, str(review_path))
    review = read(review_path)
    validate_review(root, job, review)
    target = run_dir / entry_id / f"round-{job['round']}" / "review.json"
    if target.exists() and target.read_bytes() != review_path.read_bytes():
        raise ValueError("Review already retained; start a revision round instead of overwriting it")
    if target.exists() and job["state"] == "reviewed":
        return {"entry_id": entry_id, "state": job["state"]}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(review_path.read_bytes())
    job["reviews"].append({"path": target.relative_to(root).as_posix(), "sha256": digest(target)})
    job["issues"] = review["findings"]
    job["state"] = "reviewed" if review["decision"] == "accept" else "revise"
    job["checks"] = []
    write(run_dir / "workflow.json", state)
    return {"entry_id": entry_id, "state": job["state"]}


def execute_checks(root, run_dir, entry_id, timeout=300, *, runner=None):
    state = read(run_dir / "workflow.json")
    job = state["jobs"][entry_id]
    if job["state"] != "reviewed":
        raise ValueError("Accepted, current review required before checks")
    reference = job["reviews"][-1]
    review_path = inside(root, reference["path"])
    if digest(review_path) != reference["sha256"]:
        raise ValueError("Retained review changed")
    review = read(review_path)
    validate_review(root, job, review)
    results = []
    for index, specification in enumerate(review["checks"]):
        script = inside(root, specification["script"])
        if not script.is_file() or script.suffix != ".py" or not script.is_relative_to(root / "tools"):
            raise ValueError("Checks must be repository Python tools")
        arguments = specification.get("args", [])
        if not isinstance(arguments, list) or not all(isinstance(x, str) for x in arguments):
            raise ValueError("Check arguments must be a string list")
        script_sha = digest(script)
        argv = [sys.executable, "-X", "utf8", str(script), *arguments]
        execution = {"backend": "local_subprocess", "actual_argv": argv}
        try:
            if runner is None:
                output = subprocess.run(argv, cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
                code, stdout, stderr = output.returncode, output.stdout, output.stderr
            else:
                output_dir = run_dir / entry_id / f"round-{job['round']}" / f"execution-{index}"
                output = runner(root, script.relative_to(root).as_posix(), arguments, output_dir, timeout=timeout)
                code, stdout, stderr = output["returncode"], output["stdout"], output["stderr"]
                execution = output["execution"]
                if execution.get("script_sha256") != script_sha:
                    raise ValueError("Executed script does not match the reviewed local script")
        except subprocess.TimeoutExpired as exc:
            code, stdout, stderr = -1, str(exc.stdout or ""), "Check timed out; inspect before retrying"
        log = run_dir / entry_id / f"round-{job['round']}" / f"check-{index}.json"
        result = {"argv": argv, "exit_code": code, "stdout": stdout, "stderr": stderr,
                  "script_path": script.relative_to(root).as_posix(), "script_sha256": script_sha,
                  "execution": execution, "documents": review["documents"], "executed_at": now()}
        if digest(script) != script_sha:
            result["exit_code"] = code = -2
            result["stderr"] += "\nCheck script changed during its own execution"
        write(log, result)
        results.append({"path": log.relative_to(root).as_posix(), "sha256": digest(log), "exit_code": code})
        if code:
            break
    # The author tools cannot silently change their reviewed inputs during checks.
    validate_review(root, job, review)
    job["checks"] = results
    job["state"] = "ready" if len(results) == len(review["checks"]) and all(x["exit_code"] == 0 for x in results) else "checks_failed"
    write(run_dir / "workflow.json", state)
    return {"entry_id": entry_id, "state": job["state"], "checks": results}


def reopen(run_dir, entry_id, reason):
    state = read(run_dir / "workflow.json")
    job = state["jobs"][entry_id]
    if job["state"] in {"registered", "verified"} or not reason.strip():
        raise ValueError("Registered chapters need a new run; revision requires a reason")
    job.setdefault("revision_reasons", []).append({"round": job["round"], "reason": reason, "at": now()})
    job["state"] = "revise"
    write(run_dir / "workflow.json", state)
    return {"entry_id": entry_id, "state": "revise"}


def record_exercise(root, run_dir, entry_id, path):
    state = read(run_dir / "workflow.json")
    job = state["jobs"][entry_id]
    artifact = inside(root, str(path))
    result = read(artifact)
    if result.get("entry_id") != entry_id or result.get("body_sha256") != digest(root / job["path"]):
        raise ValueError("Exercise must bind current chapter")
    for key in ["participant", "task", "initial_answer", "feedback", "revision", "limitations", "observed_outcome"]:
        if not result.get(key):
            raise ValueError(f"Exercise must record {key}")
    if result["participant"] == job["author"]:
        raise ValueError("Exercise participant must differ from the author")
    for name, sha in result.get("documents", {}).items():
        if digest(inside(root, name)) != sha:
            raise ValueError(f"Stale exercise artifact: {name}")
    for key in ["task", "initial_answer", "feedback", "revision"]:
        if result[key] not in result.get("documents", {}):
            raise ValueError(f"Unbound exercise artifact: {key}")
    job["exercise"] = {"path": artifact.relative_to(root).as_posix(), "sha256": digest(artifact)}
    job["agent_use"] = "bounded_exercise_recorded"
    write(run_dir / "workflow.json", state)
    return {"entry_id": entry_id, "agent_use": job["agent_use"]}


def register(root, run_dir, version_id, runtime_source=None):
    from knowledge_release import release
    state = read(run_dir / "workflow.json")
    ready = [j["id"] for j in state["jobs"].values() if j["state"] in {"ready", "registered", "verified"}]
    if not ready:
        raise ValueError("No reviewed and checked entries ready")
    report_paths = []
    for identifier in ready:
        job = state["jobs"][identifier]
        ref = job["reviews"][-1]
        path = inside(root, ref["path"])
        if digest(path) != ref["sha256"]:
            raise ValueError("Review changed after acceptance")
        validate_review(root, job, read(path))
        report_paths.append(ref["path"])
        for check in job["checks"]:
            log = inside(root, check["path"])
            if digest(log) != check["sha256"] or check["exit_code"]:
                raise ValueError("Check log changed or failed")
            record = read(log)
            checked_script = inside(root, record["script_path"]) if "script_path" in record else Path(record["argv"][3])
            if digest(checked_script) != record["script_sha256"]:
                raise ValueError("Check script changed after execution")
            report_paths.append(check["path"])
        if job.get("exercise"):
            ref = job["exercise"]
            if digest(inside(root, ref["path"])) != ref["sha256"]:
                raise ValueError("Exercise evidence changed")
            report_paths.append(ref["path"])
    result = release(root, run_dir, ready, version_id, report_paths=report_paths, runtime_source=runtime_source)
    for identifier in ready:
        if state["jobs"][identifier]["state"] != "verified":
            state["jobs"][identifier]["state"] = "registered"
    state["registered_version"] = version_id
    write(run_dir / "workflow.json", state)
    return {key: result[key] for key in ["version_id", "selected_ids", "revised_ids", "source_note_changes", "already_registered"]}


def continue_run(root, run_dir, next_name):
    state = read(run_dir / "workflow.json")
    if not state.get("registered_version"):
        raise ValueError("Register the current completed batch before advancing its baseline")
    if any(j["state"] in {"working", "reviewed", "ready", "checks_failed", "registered"} for j in state["jobs"].values()):
        raise ValueError("Finish or explicitly revise active jobs before advancing")
    remaining = [j["id"] for j in state["jobs"].values() if j["state"] != "verified"]
    if not remaining:
        return {"all_content_accepted": True, "remaining": 0}
    destination = inside(root, "docs/evidence/" + next_name)
    if state.get("successor_run"):
        if state["successor_run"] != destination.relative_to(root).as_posix():
            raise ValueError("This batch already has a different successor")
        return summary(read(destination / "workflow.json"))
    initialize(root, destination, ",".join(remaining), state["max_rounds"])
    successor = read(destination / "workflow.json")
    successor["predecessor_run"] = run_dir.relative_to(root).as_posix()
    for identifier in remaining:
        old = state["jobs"][identifier]
        new = successor["jobs"][identifier]
        new.update(round=old["round"], state=old["state"], issues=old["issues"])
        for field in ["max_rounds", "budget_adjustments", "manager_notes"]:
            if field in old:
                new[field] = old[field]
    write(destination / "workflow.json", successor)
    state["successor_run"] = destination.relative_to(root).as_posix()
    write(run_dir / "workflow.json", state)
    return {"next_run": destination.relative_to(root).as_posix(), **summary(successor)}


def verify_release(root, run_dir, runtime_source):
    from knowledge_release import finalize_verification
    state = read(run_dir / "workflow.json")
    identifiers = [j["id"] for j in state["jobs"].values() if j["state"] in {"registered", "verified"}]
    if not identifiers or not state.get("registered_version"):
        raise ValueError("Register a reviewed batch first")
    runtime_source = runtime_source.resolve()
    if runtime_source.name != "src" or not (runtime_source / "infra_agent").is_dir():
        raise ValueError("runtime-source must identify the existing runtime src directory")
    specifications = [
        [sys.executable, "-X", "utf8", str(root / "tools/validate_knowledge.py"), "--repository", str(root),
         "--report", str(run_dir / "structural.json")],
        [sys.executable, "-X", "utf8", str(root / "tools/probe_mechanism_expansion.py"),
         "--runtime", str(runtime_source.parent), "--entries", ",".join(identifiers),
         "--report", str(run_dir / "reader.json")],
    ]
    for index, argv in enumerate(specifications):
        script_sha = digest(Path(argv[3]))
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=300)
        log = run_dir / f"release-check-{index}.json"
        write(log, {"argv": argv, "exit_code": result.returncode, "stdout": result.stdout,
                    "stderr": result.stderr, "script_sha256": script_sha, "executed_at": now()})
        if result.returncode:
            raise ValueError(f"Release verification failed; inspect {log.relative_to(root)}")
        if digest(Path(argv[3])) != script_sha:
            raise ValueError("Release verification script changed during execution")
    result = finalize_verification(root, run_dir, run_dir / "structural.json", run_dir / "reader.json")
    for identifier in identifiers:
        state["jobs"][identifier]["state"] = "verified"
    state["verified_at"] = now()
    write(run_dir / "workflow.json", state)
    return {**result, **summary(state)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run", required=True, help="Name under docs/evidence")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--entries", default="all")
    init.add_argument("--max-rounds", type=int, default=3)
    sub.add_parser("status")
    nxt = sub.add_parser("next")
    nxt.add_argument("--owner", required=True)
    nxt.add_argument("--limit", type=int, default=1)
    nxt.add_argument("--entry")
    prompt = sub.add_parser("prompt")
    prompt.add_argument("--entry", required=True)
    prompt.add_argument("--role", choices=["author", "reviewer", "learner"], required=True)
    review = sub.add_parser("review")
    review.add_argument("--entry", required=True)
    review.add_argument("--file", type=Path, required=True)
    checks = sub.add_parser("check")
    checks.add_argument("--entry", required=True)
    checks.add_argument("--timeout", type=int, default=300)
    exercise = sub.add_parser("exercise")
    exercise.add_argument("--entry", required=True)
    exercise.add_argument("--file", type=Path, required=True)
    revision = sub.add_parser("reopen")
    revision.add_argument("--entry", required=True)
    revision.add_argument("--reason", required=True)
    release_command = sub.add_parser("register")
    release_command.add_argument("--version", required=True)
    release_command.add_argument("--runtime-source", type=Path)
    advance = sub.add_parser("continue")
    advance.add_argument("--next-run", required=True)
    verify_command = sub.add_parser("verify-release")
    verify_command.add_argument("--runtime-source", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    run_dir = inside(root, "docs/evidence/" + args.run)
    if args.command == "init":
        result = initialize(root, run_dir, args.entries, args.max_rounds)
    elif args.command == "status":
        result = summary(read(run_dir / "workflow.json"))
    elif args.command == "next":
        result = claim(run_dir, args.owner, args.limit, args.entry)
    elif args.command == "prompt":
        print(make_prompt(root, run_dir, args.entry, args.role))
        return
    elif args.command == "review":
        result = record_review(root, run_dir, args.entry, args.file)
    elif args.command == "check":
        result = execute_checks(root, run_dir, args.entry, args.timeout)
    elif args.command == "exercise":
        result = record_exercise(root, run_dir, args.entry, args.file)
    elif args.command == "reopen":
        result = reopen(run_dir, args.entry, args.reason)
    elif args.command == "continue":
        result = continue_run(root, run_dir, args.next_run)
    elif args.command == "verify-release":
        result = verify_release(root, run_dir, args.runtime_source)
    else:
        result = register(root, run_dir, args.version, args.runtime_source)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
