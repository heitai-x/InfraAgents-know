"""Author artifact assembly only. Generated check programs are parsed, not run."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pprint
import re

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
KB = RUN.parents[2]
METRICS = HERE / 'metrics-r2'
GOODPUT = HERE / 'goodput'
AUTHOR = '/root/evaluation_metrics_audit'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def common_rows(script):
    tree = ast.parse(script)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Request')
    fields = [n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)]
    defaults = {n.target.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign) and n.value is not None}
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ROWS' for t in n.targets))
    rows = []
    for call in assignment.value.elts:
        row = dict(defaults)
        row.update(zip(fields, (ast.literal_eval(arg) for arg in call.args)))
        row.update({kw.arg: ast.literal_eval(kw.value) for kw in call.keywords})
        if set(row) != set(fields):
            raise ValueError('Incomplete common fixture schema')
        rows.append(row)
    return rows


def patch_metrics_script(raw):
    text = raw.replace('from dataclasses import dataclass, replace', 'from dataclasses import asdict, dataclass, replace')
    text = text.replace('import json\n', 'import json\nimport hashlib\n', 1)
    text = text.replace("    quality_events: tuple[tuple[str, int, bool, str, str], ...] = ()\n", "    quality_events: tuple[tuple[str, int, bool, str, str], ...] = ()\n    material_ready_at: int | None = None\n    material_token_count: int | None = None\n", 1)
    material = {'A': (30, 3), 'B': (60, 3), 'C': (50, 3), 'F': (90, 3), 'H': (100, 1)}
    lines = []
    for line in text.splitlines():
        match = re.match(r"    Request\('([A-H])',", line)
        if match and match.group(1) in material:
            ready, count = material[match.group(1)]
            if not line.endswith('),'):
                raise ValueError('Unexpected fixture constructor layout')
            line = line[:-2] + f', material_ready_at={ready}, material_token_count={count}),' 
        lines.append(line)
    text = '\n'.join(lines) + '\n'
    text = text.replace("    if row.quality is not None:\n", "    if (row.material_ready_at is None) != (row.material_token_count is None):\n        raise ValueError('incomplete immutable-output descriptor')\n    if row.material_ready_at is not None:\n        if type(row.material_ready_at) is not int or type(row.material_token_count) is not int or row.material_token_count != sum(size for _, size in row.content):\n            raise ValueError('immutable-output descriptor count/type')\n    if row.quality is not None:\n", 1)
    text = text.replace("        complete_content_at = max((when for when, _ in row.content), default=row.sent)\n        if row.quality[0] < complete_content_at:\n            raise ValueError('quality before complete content')", "        if row.material_ready_at is not None and row.quality[0] < row.material_ready_at:\n            raise ValueError('quality before immutable material readiness')", 1)
    text = text.replace("    complete_content_at = max((when for when, _ in row.content), default=row.sent)\n", '', 1)
    text = text.replace("        if when < complete_content_at or output_version != row.output_version or rule != 'task-quality-v1':", "        if (row.material_ready_at is not None and when < row.material_ready_at) or output_version != row.output_version or rule != 'task-quality-v1':", 1)
    text = text.replace("    return result\n\n\ndef report", "    return result if row.material_ready_at is not None else None\n\n\ndef report", 1)
    text = text.replace("terminal=(165, 'FINAL'), quality=(175, True))", "terminal=(165, 'FINAL'), quality=(175, True), material_ready_at=160)", 1)
    extra = """    producer_early = replace(ROWS[0], material_ready_at=25, quality=(28, True))
    producer_early_rows = tuple(producer_early if row.name == 'A' else row for row in ROWS)
    same(report(producer_early_rows, evidence=160), observed, 'quality may precede client last content after producer SEAL')
    same(metrics(producer_early, 160), metrics(ROWS[0], 160), 'producer-side quality never changes client visibility')
    missing_descriptor = replace(ROWS[0], material_ready_at=None, material_token_count=None)
    missing_rows = tuple(missing_descriptor if row.name == 'A' else row for row in ROWS)
    same((report(missing_rows)['known_good'], report(missing_rows)['unknown_good']), (2, 2), 'no immutable output proof stays unknown')
    checks += 3
"""
    text = text.replace("    rejected = []\n", extra + "    rejected = []\n", 1)
    rows = common_rows(text)
    fingerprint = sha(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    text = text.replace("def main() -> None:\n    checks = 0", "COMMON_FIXTURE_SHA256 = '" + fingerprint + "'\n\n\ndef main() -> None:\n    encoded_fixture = json.dumps([asdict(row) for row in ROWS], sort_keys=True, separators=(',', ':')).encode('utf-8')\n    same(hashlib.sha256(encoded_fixture).hexdigest(), COMMON_FIXTURE_SHA256, 'common fixture fingerprint')\n    checks = 1", 1)
    text = text.replace("    print(json.dumps({'result': 'PASS', 'checks': checks,", "    summary_keys = ['offered', 'confirmed_accepts', 'effective_admitted', 'normal', 'known_good', 'unknown_good', 'cohort_output', 'goodput_lower', 'goodput_upper', 'token_goodput_lower']\n    common_summary = {key: str(observed[key]) if isinstance(observed[key], F) else observed[key] for key in summary_keys}\n    print(json.dumps({'result': 'PASS', 'checks': checks,\n                      'common_fixture_sha256': COMMON_FIXTURE_SHA256, 'common_fixture_summary': common_summary,", 1)
    ast.parse(text)
    return text, rows, fingerprint


def main():
    METRICS.mkdir(exist_ok=True)
    GOODPUT.mkdir(exist_ok=True)
    metric_path = KB / 'knowledge/workloads/metric-events.md'
    goodput_path = KB / 'knowledge/workloads/slo-goodput.md'
    metric_input = metric_path.read_bytes()
    goodput_input = goodput_path.read_bytes()
    if sha(metric_input) != '6bd512aa5f3b62e494cfe0dd23f07a559a23ee4b6fa581eae0c3bcb7c727e528':
        raise ValueError('Metrics input is not the retained first candidate')
    if sha(goodput_input) != 'e15792741cc4d911e201218dd05bc59a7e37108703d7879ad3dde68dacdf53a2':
        raise ValueError('Goodput input unexpectedly changed')
    script_input = (KB / 'tools/check_evaluation_metrics.py').read_bytes()
    if sha(script_input) != '6fd3e7537f23defd1fab99bbef99a784fc42fb350ee8174a115dcc1b48c23689':
        raise ValueError('Metrics check input is not the retained first script')
    metrics_script, rows, fingerprint = patch_metrics_script(script_input.decode('utf-8'))
    (METRICS / 'check_evaluation_metrics.py').write_text(metrics_script, encoding='utf-8')
    compact_fixture = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode('utf-8')
    (METRICS / 'common-fixture.json').write_bytes(compact_fixture)
    (GOODPUT / 'common-fixture.json').write_bytes(compact_fixture)
    goodput_script_path = GOODPUT / 'check_evaluation_goodput.py'
    template = goodput_script_path.read_text(encoding='utf-8')
    if '__COMMON_FIXTURE_LITERAL__' not in template:
        template = (GOODPUT / 'check-template.py.txt').read_text(encoding='utf-8')
    else:
        (GOODPUT / 'check-template.py.txt').write_text(template, encoding='utf-8')
    goodput_script = template.replace('__COMMON_FIXTURE_LITERAL__', pprint.pformat(rows, sort_dicts=True, width=110))
    goodput_script = goodput_script.replace('__COMMON_FIXTURE_SHA256__', fingerprint)
    ast.parse(goodput_script)
    goodput_script_path.write_text(goodput_script, encoding='utf-8')
    body = metric_input.decode('utf-8-sig')
    before = '完整内容身份确定之后，审核可以在协议尾部尚未结束时完成，例如最后内容30、标签35、FINAL45均可合法。'
    after = ('完整不可变输出身份成立后，审核可以在协议尾部尚未结束时完成，例如最后内容30、标签35、FINAL45均可合法。'
             '该就绪事实属于生产侧完整输出，不等于客户端末内容可见：生产侧SEAL25、证书28、客户端最后内容30、FINAL45也可以合法。'
             'SEAL在这里表示完整内容身份及其生产就绪依据，不要求某个同名API；不能从最后一条已观测内容或shape推断它成立。')
    if before not in body:
        raise ValueError('Metrics quality boundary anchor absent')
    body = body.replace(before, after, 1)
    body = body.replace('A 的 TTFT 为 10、TPOT 为 10、最后内容延迟为 30、协议延迟为 45。',
                        '本例另给生产侧完整输出描述：A/B/C/F/H的SEAL时刻分别为30、60、50、90、100，封存token数为3、3、3、3、1；D/E/G没有完整输出描述。它们是独立输入，部分数值与client末内容时刻相同不表示可以由后者推断SEAL。\n\nA 的 TTFT 为 10、TPOT 为 10、最后内容延迟为 30、协议延迟为 45。', 1)
    body = body.replace('评价器修正或规则变化需要明确的新证据/规则身份和报表修订，不能冒充未改变的判定。',
                        '评价器修正或规则变化需要明确的新证据/规则身份和报表修订，不能冒充未改变的判定。随机原始评审标签须由明确的采样/聚合规则产生证书，不能直接用后来一个PASS覆盖一个既有固定规则FAIL。', 1)
    adapter = (
        '这份结算与[Goodput](slo-goodput.md)共用明确接口：请求事件归约提供客户端内容事实；'
        'CohortGoodput先冻结H前的业务终态与完整输出描述，再按J归一目标输出/规则的证据。'
        '早期证据可以等待业务FINAL；若FINAL在H之后，证据不能使它成为H前完成。'
        'SettlePerformanceEvidence返回PASS、FAIL、UNKNOWN或CONFLICT及原因，Goodput的最后阶段只计显式PASS；'
        '旧CohortReport的good是已证真计数，仍有UNKNOWN时须连同潜在资格和范围报告，CONFLICT阻止确定资格结算。'
        '不能把UNKNOWN强转为最终失败，也不能在下游再运行一遍“最后质量标签覆盖”的分支。'
        '完整描述若作为晚到证据补齐，绑定的仍须是同一个H前输出，不能改变服务事件或内容身份。'
    )
    body = body.replace('```pseudocode\n算法 SettlePerformanceEvidence', adapter + '\n\n```pseudocode\n算法 SettlePerformanceEvidence', 1)
    body = body.replace('固定service_snapshot的人口、截止H、计时边界和内容事实，不以J重算服务历时',
                        '输入采用CohortGoodput冻结的H业务快照、客户端内容事实、完整生产输出描述与J证据视图\n固定其人口、截止H、计时边界和内容事实，不以J重算服务历时', 1)
    body = body.replace('逐个对象核对证书的请求、完整输出版本、评价规则及证据发生时间',
                        '逐个对象核对证书的请求、完整生产输出身份、评价规则及证据可用时间\n完整输出绑定事实尚缺则保留UNKNOWN；证据早于FINAL可以挂起，不要求晚于client末内容', 1)
    (METRICS / 'metric-events.md').write_text(body, encoding='utf-8')

    goodput_body = goodput_input.decode('utf-8-sig')
    start = goodput_body.index('## 窗口账与可复算的资格归约')
    stop = goodput_body.index('## 准入必须区分硬合法性和时延预测', start)
    replacement = (GOODPUT / 'window-section.md').read_text(encoding='utf-8').strip() + '\n\n'
    goodput_body = goodput_body[:start] + replacement + goodput_body[stop:]
    old_origin = next(line for line in goodput_body.splitlines() if line.startswith('以下核心算法以非空文本响应为例。'))
    new_origin = ('以下核心算法以非空文本响应为例。请求i的外部到达/计划为aᵢ，实际客户端发送为sᵢ，首个有效内容为fᵢ，正常业务完成为cᵢ，'
                  '逐token可见时刻为tᵢ,1…tᵢ,n。cohort按aᵢ纳入；时延另固定合同起点oᵢ，可选择实际发送sᵢ或外部到达aᵢ，'
                  '不能混用。TTFT＝fᵢ−oᵢ；n>1时TPOT＝(tᵢ,n−tᵢ,1)/(n−1)，最大间隔为maxₖ(tᵢ,k−tᵢ,k−1)。'
                  '客户端持续时间使用同一客户端时钟，跨机拆分另需映射。只有chunk时间时可以评价明确chunk指标，不能伪装逐token ITL。')
    goodput_body = goodput_body.replace(old_origin, new_origin, 1)
    old_quality = next(line for line in goodput_body.splitlines() if line.startswith('质量尚未判定时，Cᵢ不能记为真。') or line.startswith('质量尚未判定时，Cᵢ 不能记为真。'))
    goodput_body = goodput_body.replace(old_quality, old_quality + ' 下文将证书归一后的PASS、FAIL、UNKNOWN和CONFLICT分别保留；Qᵢ只在所选J视图显式PASS时为真，固定规则矛盾证书不能按最后值覆盖。', 1)
    goodput_body = goodput_body.replace('首内容预算为 D_first−(now−aᵢ)', '首内容预算为 D_first−(now−oᵢ)', 1)
    coverage = ('新增 `tools/check_evaluation_goodput.py` 自包含一份与指标检查按规范JSON指纹绑定的共同八请求数据。'
                '独立的业务事件状态机与声明式终态/证据集合参考分别计算H快照、J质量视图和资格；'
                '它们不共享资格函数，也不导入其他检查脚本。覆盖早于FINAL的质量、早于客户端末内容但晚于生产SEAL的质量、'
                '固定规则FAIL→PASS冲突、身份载荷冲突、相同证据幂等、H后J前证书、H后FINAL不纳入及取消不复活。'
                '故障分支实际执行迟到要求、后写覆盖、J改H、证据复活业务和以client末内容猜SEAL，交给同一声明式参考。'
                '原检查继续承担其历史事件模型与资源/流量回归，不把它的严格证书到达顺序当作通用质量接口。'
                '实际检查结果须由当前绑定执行记录承担；共同数据的指纹和摘要可以跨两个独立脚本核对，不能代替独立参考。')
    goodput_body = goodput_body.replace('这些检查不执行模型、tokenizer', coverage + '\n\n这些检查不执行模型、tokenizer', 1)
    provenance = ('2026-10-29的质量证据归一、生产侧完整输出身份、H/J快照、跨归约接口和共同fixture为本库独立构造，'
                  '用于修复模块审阅发现的证据责任不一致。它们不是GenAI-Perf、MLPerf或DistServe已经实现的完整证书协议；'
                  '原来源字节、截止日期与既有理论条件不因本次修订改变。')
    goodput_body = goodput_body.replace('来源核验使用三份来源说明', provenance + '\n\n来源核验使用三份来源说明', 1)
    (GOODPUT / 'slo-goodput.md').write_text(goodput_body, encoding='utf-8')
    finding_path = RUN / 'semantic-review/round-1/bundle.json'
    finding = json.loads(finding_path.read_text(encoding='utf-8-sig'))['module_findings'][0]
    for directory in [METRICS, GOODPUT]:
        json_write(directory / 'finding-M-001.json', finding)
    common_note = (
        '实际作者 /root/evaluation_metrics_audit，处理原始M-001的语义问题；原review有引用格式缺陷，不作为accepted。'
        '完整阅读原bundle及当前指标/Goodput正文；本次不修补原review。'
        '证据绑定生产侧完整不可变输出，不强制quality晚于client末token或FINAL。'
        'H服务快照、J证据视图、终态吸收、证据幂等/冲突和随机评分聚合责任分别展开。'
        '新脚本自包含，无sibling导入；共同纯数据按标准JSON编码SHA绑定，两脚本输出common_fixture_sha256与common_fixture_summary，'
        '协调者真实执行后可据此比较实际指标归约与Goodput输出。Goodput另有独立状态机/声明式集合参考。'
        '只作AST解析与数据/产物组装，未在主机执行检查程序，未编造测试通过或独立审查。'
        '旧check_k_work_001_full.py和check_k_work_005_full.py未修改，需由协调者隔离回归。'
    )
    for entry, directory, chapter_name, chapter_body, script_name, script_body, input_bytes in [
        ('K-WORK-001', METRICS, 'metric-events.md', body, 'check_evaluation_metrics.py', metrics_script, metric_input),
        ('K-WORK-005', GOODPUT, 'slo-goodput.md', goodput_body, 'check_evaluation_goodput.py', goodput_script, goodput_input),
    ]:
        notes = common_note + (' 指标R1保留不动，本轮为metrics-r2。' if entry == 'K-WORK-001' else
                               ' Goodput保留其他容量/准入/资源/前沿机制；同步明确cohort起点与TTFT起点，预算用同一oᵢ。')
        candidate = dict(entry_id=entry, body=chapter_body,
                         checks=[dict(script='tools/' + script_name, body=script_body, args=[])], author_notes=notes)
        json_write(directory / 'candidate.json', candidate)
        docs = ['AGENTS.md', 'docs/knowledge-method.md', 'docs/knowledge-review.md', 'docs/evaluation-module.md',
                'knowledge/workloads/metric-events.md', 'knowledge/workloads/slo-goodput.md',
                'knowledge/sources/S-GENAI-PERF.md', 'knowledge/sources/S-MLPERF-LLM.md',
                'docs/evidence/evaluation-module-20261029/semantic-review/round-1/bundle.json']
        readings = [{'path': rel, 'sha256': sha((KB / rel).read_bytes()), 'scope': 'full current text'} for rel in docs]
        previous = json.loads((HERE / 'metrics/author-reading.json').read_text(encoding='utf-8'))
        ledger = dict(author=AUTHOR, entry_id=entry, created_at=datetime.now(timezone.utc).isoformat(),
                      revision_input_sha256=sha(input_bytes), body_sha256=sha(chapter_body.encode('utf-8')),
                      original_reading_ledger='docs/evidence/evaluation-module-20261029/authoring/metrics/author-reading.json',
                      original_reading_ledger_sha256=sha((HERE / 'metrics/author-reading.json').read_bytes()),
                      documents={r['path']: r['sha256'] for r in readings}, readings=readings,
                      prior_original_excerpt_ranges=previous['readings'][-2:],
                      finding=dict(path='docs/evidence/evaluation-module-20261029/semantic-review/round-1/bundle.json',
                                   sha256=sha(finding_path.read_bytes()), id='M-001', role='semantic feedback, not accepted review'),
                      common_fixture_sha256=fingerprint,
                      limitations=['No check program run on this host; AST parsing only.',
                                   'No new source acquisition, device/service results or independent acceptance claimed.'])
        json_write(directory / 'author-reading.json', ledger)
        (directory / 'author-notes.txt').write_text(notes + '\n', encoding='utf-8')
        manifest = dict(entry_id=entry, author=AUTHOR, status='PENDING_ISOLATED_EXECUTION_AND_FRESH_REVIEW',
                        common_fixture_sha256=fingerprint, script_ast='PASS_NOT_EXECUTION',
                        canonical_body_unchanged=(KB / ('knowledge/workloads/' + chapter_name)).read_bytes() == input_bytes,
                        files={n: sha((directory / n).read_bytes()) for n in [chapter_name, script_name, 'candidate.json',
                                                                          'author-reading.json', 'author-notes.txt', 'common-fixture.json']})
        json_write(directory / 'delivery.json', manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
