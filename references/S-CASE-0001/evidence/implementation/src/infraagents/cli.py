"""Task inspection and evidence analysis; no model invocation."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from .environment import inspect_environment
from .nsys import analyze_nsys
from .roles import ROLES
from .task_spec import validate_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="InfraAgents 设计骨架与任务结构检查")
    commands = parser.add_subparsers(dest="command", required=True)
    roles = commands.add_parser("roles", help="显示四个逻辑角色")
    roles.add_argument("--json", action="store_true")
    validate = commands.add_parser("validate-task", help="只校验 TOML 任务结构")
    validate.add_argument("path", type=Path)
    validate.add_argument("--json", action="store_true")
    inspect = commands.add_parser("inspect-environment", help="探查现有工具并保存原始反馈")
    inspect.add_argument("--output-dir", type=Path)
    inspect.add_argument("--timeout", type=float, default=10, help="每条探查命令的超时秒数")
    inspect.add_argument("--codex", help="可选 Codex 可执行文件位置；默认从 PATH 查找")
    inspect.add_argument("--json", action="store_true")
    nsys = commands.add_parser("analyze-nsys", help="只读分析原生 Nsight SQLite 导出")
    nsys.add_argument("source", type=Path)
    nsys.add_argument("--output-dir", type=Path)
    nsys.add_argument("--context", type=Path, help="记录模型、硬件、workload、运行点和采集条件的 JSON")
    nsys.add_argument("--start-ns", type=int)
    nsys.add_argument("--end-ns", type=int)
    nsys.add_argument("--limit", type=int, default=20)
    nsys.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "roles":
        if args.json:
            print(json.dumps([asdict(role) for role in ROLES], ensure_ascii=False, indent=2))
        else:
            for role in ROLES:
                print(f"{role.name} / {role.label}: {role.responsibility} [{role.prompt}]")
        return 0

    if args.command == "analyze-nsys":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = args.output_dir or Path("runs") / f"nsys-analysis-{stamp}-{uuid4().hex[:6]}"
        try:
            report = analyze_nsys(args.source, output_dir, context=args.context,
                                  start_ns=args.start_ns, end_ns=args.end_ns, limit=args.limit)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if args.json:
            print(json.dumps({"report_path": str(output_dir.resolve() / "report.json"), **report}, ensure_ascii=False, indent=2))
        else:
            print(f"Nsight 分析记录：{output_dir.resolve() / 'report.json'}")
            print(f"分析状态：{report['status']}")
            for name, signal in report.get("signals", {}).items():
                print(f"- {name}: {signal['status']}; selected_intervals={signal.get('selected_interval_rows', 'unknown')}")
            if report.get("failure"):
                print(report["failure"]["message"])
            print("仅解释已观察到的区间；采集完整性、性能归因、正确性与 SLO 仍需独立验证。")
        return 0 if report["status"] == "ok" else 1

    if args.command == "inspect-environment":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = args.output_dir or Path("runs") / f"environment-{stamp}-{uuid4().hex[:6]}"
        try:
            report = inspect_environment(output_dir, timeout=args.timeout, codex=args.codex)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if args.json:
            print(json.dumps({"report_path": str(output_dir.resolve() / "report.json"), **report}, ensure_ascii=False, indent=2))
        else:
            print(f"环境探查记录：{output_dir.resolve() / 'report.json'}")
            for tool in report["tools"]:
                print(f"- {tool['name']}: {tool['status']}")
                preview = tool.get("stdout_preview") or tool.get("stderr_preview") or ""
                if preview.strip():
                    print(f"  {preview.strip().splitlines()[0]}")
            print("本次只检查工具与环境，模型执行、实际 profiler 采集和 SLO 仍需后续验证。")
        return 0

    result = validate_task(args.path)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print("任务配置结构有效。" if result.valid else "任务配置结构无效。")
        for error in result.errors:
            print(f"- {error}")
        for note in result.notes:
            print(note)
    return 0 if result.valid else 2
