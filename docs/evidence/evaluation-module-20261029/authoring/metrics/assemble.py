"""Assemble authoring artifacts without executing the metric check program."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
KB = RUN.parents[2]
AUTHOR = '/root/evaluation_metrics_audit'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    current = KB / 'knowledge/workloads/metric-events.md'
    baseline_path = RUN / 'before/knowledge/workloads/metric-events.md'
    old_raw = current.read_bytes()
    if baseline_path.is_file() and baseline_path.read_bytes() != old_raw:
        raise ValueError('Canonical metric chapter differs from the initialized baseline')
    old = old_raw.decode('utf-8-sig')
    additions = (HERE / 'additions.md').read_text(encoding='utf-8')
    parts = {name: body.strip() + '\n' for name, body in re.findall(
        r'<!-- PART:([^ ]+) -->\n([\s\S]*?)(?=<!-- PART:|\Z)', additions)}
    if set(parts) != {'metric-map', 'trace', 'quality'}:
        raise ValueError('Missing addition section')
    body = old.replace('## Token、传输 chunk 与日志分组', parts['metric-map'] + '\n## Token、传输 chunk 与日志分组', 1)
    body = body.replace('## 时钟区间如何限制解释', parts['trace'] + '\n' + parts['quality'] + '\n## 时钟区间如何限制解释', 1)
    check_scope = (
        '新增 `tools/check_evaluation_metrics.py` 将八请求表与逐请求端点、有理数手算账、'
        '资源侧位置身份枚举及未决资格的有限赋值枚举核对。它覆盖实际发送/计划起点、'
        '协议尾部与TPOT、三种吞吐人口、半开内容窗口、迟到接受的两本账、质量证据截止、'
        '空与单token及chunk-only的未定义值，并让错误算法分支消费同一输入。'
        '例如协议终点代替内容终点、发出起点改成计划、chunk补造token时间、未知资格算通过或失败、'
        '把offered输入当真实工作和将draft候选并入输出，都交给相同参考判据。'
        '这些是有限合成检查的覆盖定义，实际是否通过由绑定脚本和输入的执行记录承担；'
        '不执行真实tokenizer、服务、网络、负载生成器、GPU或质量评价器，也不证明总体SLO概率。'
    )
    body = body.replace('这些检查不测 GPU、真实服务', check_scope + '\n\n这些检查不测 GPU、真实服务', 1)
    derivation = (
        '2026-10-29新增的指标对照、八请求轨迹、质量双截止、覆盖范围及工作计数为独立构造。'
        '来源核对使用固定GenAI-Perf指标表、measurement interval与goodput段、LoadGen责任/准确率评价段，'
        '以及本库当前指标、到达与联合资格说明；没有将这些构造归称为工具的完整实现或实测。'
        '当前来源说明由协调者统一维护，来源归档字节与时间资格不因本次提炼改变。'
        '前一修订的阅读范围记录如下。'
    )
    body = body.replace('本轮阅读范围为任务提供的完整方法', derivation + '\n\n本轮阅读范围为任务提供的完整方法', 1)
    old_iterator = iter(body.splitlines())
    for line in old.splitlines():
        if not any(candidate == line for candidate in old_iterator):
            raise ValueError('An existing chapter line was removed or reordered')
    headings = re.findall(r'^## (.*)$', body, re.M)
    if len(headings) != len(set(headings)):
        raise ValueError('Duplicate H2')
    script_path = HERE / 'check_evaluation_metrics.py'
    script = script_path.read_text(encoding='utf-8')
    ast.parse(script, filename=str(script_path))  # Parse only; do not execute generated checks on this host.
    (HERE / 'metric-events.md').write_text(body, encoding='utf-8', newline='')

    documents = {}
    readings = []
    full = [
        'AGENTS.md', 'docs/knowledge-method.md', 'docs/knowledge-review.md',
        'docs/knowledge-build-workflow.md', 'knowledge/templates/entry.md',
        'knowledge/workloads/metric-events.md', 'knowledge/workloads/slo-goodput.md',
        'knowledge/workloads/arrival-and-concurrency.md',
        'knowledge/sources/S-GENAI-PERF.md', 'knowledge/sources/S-MLPERF-LLM.md',
        'knowledge/sources/S-MLPERF-SCENARIOS.md',
    ]
    for relative in full:
        raw = (KB / relative).read_bytes()
        documents[relative] = sha(raw)
        readings.append({'path': relative, 'sha256': sha(raw), 'scope': 'full current text'})
    excerpted = {
        'references/S-GENAI-PERF/fulltext.txt': [(486, 503), (833, 843), (1007, 1025)],
        'references/S-MLPERF-LLM/upstream/loadgen/README.md': [(1, 110)],
    }
    for relative, ranges in excerpted.items():
        raw = (KB / relative).read_bytes()
        documents[relative] = sha(raw)
        lines = raw.decode('utf-8-sig').splitlines()
        excerpts = []
        for start, stop in ranges:
            text = '\n'.join(lines[start - 1:stop]) + '\n'
            excerpts.append({'line_start': start, 'line_end_inclusive': stop,
                             'text_sha256': sha(text.encode('utf-8'))})
        readings.append({'path': relative, 'sha256': sha(raw), 'scope': 'specified excerpts only', 'excerpts': excerpts})
    notes = (
        '实际作者为 /root/evaluation_metrics_audit，当前协作子代理；没有调用DS或另一个独立作者，'
        '也未充当本候选的独立审查者。保留基线全部原始行及顺序，仅插入受影响的完整二级节、'
        '新检查覆盖和来源推导边界。新增常用metric与窗口人口映射；p/r/s起点及实际接受决定/有效业务准入两本账；'
        '协议E2E误作token终点的10对17.5ms反例；八请求表贯通130/90/62.5三种token吞吐、'
        '7/6准入计数、迟发、取消、超时、未完成、chunk覆盖及质量J/H双截止。'
        'J=160到200只改变证据，不移动H=160；未决范围是有限数据的识别范围，不是统计置信区间。'
        '按照协调者边界反馈，质量标签允许在完整内容确定后、协议FINAL前产生；'
        '新增最后内容30/质量35/FINAL45正例、相同证据幂等、不同截止与矛盾/身份冲突拒绝。'
        '原初稿在 before-quality-boundary-correction/ 保留；该修订没有执行算法或冒充独立审查。'
        '新增检查源已由AST解析，不曾在本机执行算法检查；CPU/GPU、真实服务和Agent效果均未执行。'
        '请协调者用既有隔离执行器运行 tools/check_evaluation_metrics.py，并按独立审查需要复跑未改动的 '
        'tools/check_k_work_001_full.py。不能将AST解析或此作者说明当作测试通过。'
        '实际全文/摘录阅读与当前来源说明SHA见 author-reading.json；上游源码归档未读取。'
    )
    candidate = {'entry_id': 'K-WORK-001', 'body': body,
                 'checks': [{'script': 'tools/check_evaluation_metrics.py', 'body': script, 'args': []}],
                 'author_notes': notes}
    (HERE / 'candidate.json').write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    ledger = {'author': AUTHOR, 'created_at': datetime.now(timezone.utc).isoformat(),
              'entry_id': 'K-WORK-001', 'baseline_body_sha256': sha(old_raw),
              'body_sha256': sha(body.encode('utf-8')), 'documents': documents,
              'readings': readings, 'limitations': [
                  'No upstream implementation originals or unprovided images read.',
                  'No semantic check program executed on this host; AST parsing only.',
                  'No independent review, actual service/device measurement or Agent-benefit claim.',
              ]}
    (HERE / 'author-reading.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (HERE / 'author-notes.txt').write_text(notes + '\n', encoding='utf-8')
    manifest = {'entry_id': 'K-WORK-001', 'author': AUTHOR, 'result': 'AUTHOR_DELIVERY_PENDING_INDEPENDENT_REVIEW_AND_ISOLATED_EXECUTION',
                'canonical_unchanged': sha(current.read_bytes()) == sha(old_raw),
                'baseline_lines_preserved_in_order': len(old.splitlines()),
                'new_lines': len(body.splitlines()), 'script_ast_parse': 'PASS_NOT_EXECUTION',
                'files': {name: sha((HERE / name).read_bytes()) for name in [
                    'metric-events.md', 'check_evaluation_metrics.py', 'candidate.json', 'author-reading.json', 'author-notes.txt']}}
    (HERE / 'delivery.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
