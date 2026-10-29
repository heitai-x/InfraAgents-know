"""One scoped coordinator using the maintained workflow and sandbox contracts."""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
KB = RUN.parents[2]
WORKSPACE = KB.parents[1]
sys.path.insert(0, str(WORKSPACE / 'knowledge-workflow' / 'src'))
from knowledge_workflow import build
from knowledge_workflow.config import load_config
from knowledge_workflow.cli import repository_writer
from knowledge_workflow.check_runner import SandboxRunner, _snapshot
from knowledge_workflow.model_clients import ModelClient

CONFIG = load_config(WORKSPACE / 'knowledge-workflow' / 'workflow.evaluation.local.toml')
JOBS = {
    'K-WORK-001': ('metrics', 'knowledge/workloads/metric-events.md', 'tools/check_evaluation_metrics.py', 'tools/check_k_work_001_full.py'),
    'K-WORK-004': ('design', 'knowledge/workloads/measurement-comparability.md', 'tools/check_evaluation_design.py', 'tools/check_k_work_004_full.py'),
    'K-TOOLS-007': ('statistics', 'knowledge/tools/statistical-experiments.md', 'tools/check_evaluation_statistics.py', 'tools/check_k_tools_007_full.py'),
    'K-WORK-005': ('goodput', 'knowledge/workloads/slo-goodput.md', 'tools/check_evaluation_goodput.py', 'tools/check_k_work_005_full.py'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f'Preserve existing evidence, choose a fresh attempt: {path}')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def apply_candidates():
    with repository_writer(CONFIG):
        state = read(RUN / 'workflow.json')
        receipt = {}
        pending = []
        for identifier, (folder, body_path, script_path, _) in JOBS.items():
            source = RUN / 'authoring' / folder
            candidate_path = source / 'candidate.json'
            candidate = read(candidate_path)
            if 'chapter' in candidate:
                body, script = candidate['chapter'], candidate['test_script']
            else:
                body = candidate['body']
                matching = [item for item in candidate['checks'] if item['script'] == script_path]
                if len(matching) != 1:
                    raise RuntimeError(f'Unexpected candidate script: {identifier}')
                script = matching[0]['body']
            if not body.startswith('# ' + identifier + '：'):
                raise RuntimeError(f'Wrong chapter identity: {identifier}')
            ast.parse(script, filename=script_path)
            job = state['jobs'][identifier]
            if job['state'] != 'working' or sha(KB / body_path) != job['baseline_sha256']:
                raise RuntimeError(f'Unexpected current author state/body: {identifier}')
            receipt[identifier] = {
                'actual_author': job['author'], 'candidate_path': candidate_path.relative_to(KB).as_posix(),
                'candidate_sha256': sha(candidate_path),
                'body_sha256': hashlib.sha256(body.encode('utf-8')).hexdigest(),
                'script_sha256': hashlib.sha256(script.encode('utf-8')).hexdigest(),
                'identity_scope': 'Actual collaboration author; no native DS/Sol author call is claimed.',
            }
            pending.append((body_path, body, script_path, script))
        for body_path, body, script_path, script in pending:
            (KB / body_path).write_text(body, encoding='utf-8', newline='')
            (KB / script_path).write_text(script, encoding='utf-8', newline='')
        save(RUN / 'author-handoff.json', receipt)
    print(json.dumps({'applied': list(receipt)}, ensure_ascii=False))


def preflight(attempt, previous_attempt=None):
    runner = SandboxRunner(CONFIG.adapters)
    def execute(item):
        identifier, (_, _, script, _) = item
        reused = None
        if previous_attempt:
            prior_path = RUN / 'preflight' / previous_attempt / identifier / 'result.json'
            if prior_path.is_file():
                prior = read(prior_path)
                inputs = {name: item['sha256'] for name, item in _snapshot(KB, script).items()}
                if prior['returncode'] == 0 and prior['execution']['input_digests'] == inputs:
                    result, path, reused = prior, prior_path, previous_attempt
        if reused is None:
            directory = RUN / 'preflight' / attempt / identifier
            result = runner(KB, script, [], directory, timeout=300)
            path = directory / 'result.json'
            save(path, result)
        return {'entry_id': identifier, 'script': script, 'path': path.relative_to(KB).as_posix(),
                'sha256': sha(path), 'returncode': result['returncode'], 'stdout': result['stdout'],
                'stderr': result['stderr'], 'backend': result['execution']['backend'],
                'script_sha256': result['execution']['script_sha256'], 'reused_exact_inputs_from': reused}
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(execute, JOBS.items()))
    save(RUN / 'preflight' / attempt / 'summary.json', results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if any(item['returncode'] for item in results):
        raise SystemExit(2)
    if 'K-WORK-005' in JOBS:
        outputs = {item['entry_id']: json.loads(item['stdout']) for item in results}
        left, right = outputs['K-WORK-001'], outputs['K-WORK-005']
        for field in ['common_fixture_sha256', 'common_fixture_summary']:
            if field not in left or field not in right or left[field] != right[field]:
                raise RuntimeError('Cross-chapter execution disagrees on ' + field)
        save(RUN / 'preflight' / attempt / 'cross-chapter.json', {
            'result': 'PASS', 'scope': 'Actual finite metrics and Goodput outputs on the identical declared fixture; not production integration.',
            'common_fixture_sha256': left['common_fixture_sha256'],
            'common_fixture_summary': left['common_fixture_summary']})


def apply_quality_revision():
    with repository_writer(CONFIG):
        state = read(RUN / 'workflow.json')
        prior = read(RUN / 'author-handoff.json')
        receipt = {}
        pending = []
        for identifier, folder in [('K-WORK-001', 'metrics-r2'), ('K-WORK-005', 'goodput')]:
            _, body_path, script_path, _ = JOBS[identifier]
            path = RUN / 'authoring' / folder / 'candidate.json'
            candidate = read(path)
            if 'chapter' in candidate:
                body, script = candidate['chapter'], candidate['test_script']
            else:
                body = candidate['body']
                checks = [item for item in candidate['checks'] if item['script'] == script_path]
                if len(checks) != 1:
                    raise RuntimeError('Unexpected quality revision script')
                script = checks[0]['body']
            expected = prior[identifier]['body_sha256'] if identifier in prior else state['jobs'][identifier]['baseline_sha256']
            if state['jobs'][identifier]['state'] != 'working' or sha(KB / body_path) != expected:
                raise RuntimeError('Unexpected pre-repair chapter bytes')
            if not body.startswith('# ' + identifier + '：'):
                raise RuntimeError('Wrong quality-repair identity')
            ast.parse(script, filename=script_path)
            receipt[identifier] = {'actual_author': state['jobs'][identifier]['author'],
                'candidate_path': path.relative_to(KB).as_posix(), 'candidate_sha256': sha(path),
                'body_sha256': hashlib.sha256(body.encode('utf-8')).hexdigest(),
                'script_sha256': hashlib.sha256(script.encode('utf-8')).hexdigest(),
                'reason': 'Bounded M-001 cross-chapter certificate repair; prior candidates and raw review retained.'}
            pending.append((body_path, body, script_path, script))
        for body_path, body, script_path, script in pending:
            (KB / body_path).write_text(body, encoding='utf-8', newline='')
            (KB / script_path).write_text(script, encoding='utf-8', newline='')
        save(RUN / 'author-handoff-quality-revision.json', receipt)
    print(json.dumps({'applied_quality_revision': list(receipt)}))


def review(attempt, preflight_attempt):
    state = read(RUN / 'workflow.json')
    paths = ['AGENTS.md', 'docs/knowledge-method.md', 'docs/knowledge-review.md', 'docs/evaluation-module.md']
    for _, body, script, old_script in JOBS.values():
        paths.extend([body, script, old_script])
    paths += ['knowledge/workloads/arrival-and-concurrency.md', 'knowledge/workloads/slo-goodput.md',
              'knowledge/sources/S-GENAI-PERF.md', 'knowledge/sources/S-STATS.md',
              'knowledge/sources/S-MLPERF-SCENARIOS.md', 'knowledge/sources/S-MLPERF-LLM.md',
              'knowledge/sources/S-TORCH-REPRO.md', 'knowledge/sources/S-ASGI.md',
              'references/S-GENAI-PERF/fulltext.txt']
    registry = {item['id']: item for item in read(KB / 'knowledge/sources.json')['sources']}
    for identifier in JOBS:
        paths.extend(registry[sid]['local_path'] for sid in state['jobs'][identifier]['source_ids'])
    files = [{'path': path, 'sha256': sha(KB / path), 'reading_scope': 'full',
              'content': (KB / path).read_text(encoding='utf-8-sig')} for path in dict.fromkeys(paths)]
    if attempt != 'round-1':
        # Reserve full reading for four current chapters and every check. The
        # fixed tool's metric/window passages are the relevant original scope.
        genai = next(item for item in files if item['path'] == 'references/S-GENAI-PERF/fulltext.txt')
        lines = genai['content'].splitlines(keepends=True)
        genai_ranges = [[484, 555], [810, 857], [1007, 1027]]
        genai.update(reading_scope='excerpt', range_unit='one-based inclusive lines', ranges=genai_ranges,
            content='\n'.join(f'LINES {lo}-{hi}\n' + ''.join(lines[lo-1:hi]) for lo, hi in genai_ranges))
    original = 'references/S-STATS/fulltext.txt'
    raw = (KB / original).read_text(encoding='utf-8-sig').splitlines(keepends=True)
    ranges = [[696, 870], [1381, 1690], [2033, 2160], [2307, 2407]]
    files.append({'path': original, 'sha256': sha(KB / original), 'reading_scope': 'excerpt',
                  'range_unit': 'one-based inclusive lines', 'ranges': ranges,
                  'content': '\n'.join(f'LINES {lo}-{hi}\n' + ''.join(raw[lo-1:hi]) for lo, hi in ranges)})
    documents = {item['path']: item['sha256'] for item in files if item['reading_scope'] == 'full'}
    schemas = []
    for identifier, (_, body, script, old_script) in JOBS.items():
        schemas.append({'entry_id': identifier, 'body_sha256': documents[body],
            'reviewer': 'sol-reviewer:evaluation-module:' + attempt, 'decision': 'accept|revise',
            'documents': documents,
            'criteria': {key: {'assessment': 'sufficient|partial|not_applicable', 'reason': 'specific assessment',
                              'evidence': [{'path': body, 'quote': 'literal current quote, minimum 12 characters; copy exact spelling and spaces'}]} for key in build.QUESTIONS},
            'findings': [], 'limitations': [],
            'checks': [{'script': old_script, 'args': []}, {'script': script, 'args': []}],
            'reading_scope': 'State actual full/excerpt reading for each supplied original; do not claim full originals from excerpts.',
            'test_assessment': 'Inspect actual independent references, negative controls and limits.'})
    author_provenance = read(RUN / 'author-handoff.json')
    if (RUN / 'author-handoff-quality-revision.json').exists():
        author_provenance.update(read(RUN / 'author-handoff-quality-revision.json'))
    payload = {
        'task': 'Fresh non-author review of one integrated evaluation knowledge module. Return only strict JSON with module_findings and reviews, one review per required schema. Challenge formulas, assumptions, experimental identifiability, stopping rules and actual checks. Current quality uses the original six criteria, not title counts.',
        'requirements': [
            'Copy entry IDs, body hashes, documents maps and executable check recipes exactly. Never repair or invent hashes, claims of reading, test outcomes or quotations.',
            'Judge existing retained mechanisms and new additions together. A limitation in real-device or causal Agent-effectiveness evidence is not automatically a defect in a scoped mathematical chapter.',
            'Check client/plan/server boundaries, token/chunk/end-of-protocol distinctions, event/quality cutoffs, actual versus effective admission responsibilities and window denominators.',
            'Check estimand and experimental unit, randomization/carryover, factorial interaction scale, workload standardization limits, complete service examples, uncertainty and honest decisions.',
            'Check sample planning assumptions, strict binomial boundaries, multiple endpoints versus conjunctive promotion, planned looks, optional stopping and no postselection reuse.',
            'Do not replace content judgment with test count. Accept may not contain partial criteria or blocking findings. Revise findings need concrete evidence and bounded repairs.',
            'module_findings lists only current unresolved findings and must use an explicit blocking boolean. Describe a repaired M-001 under optional closed_findings; do not leave a resolved historical issue marked blocking.',
            'The first raw review is retained but did not pass the quote contract. Every new quotation must be copied verbatim and be at least 12 characters. Do not paraphrase inside quotation fields or use short fragments.',
            'You received full check source and actual new-check preflight summaries. Unchanged old checks will also be truly rerun after acceptance. No GPU/service measurement or knowledge-benefit claim follows from software models.',
        ],
        'actual_authors': {key: state['jobs'][key]['author'] for key in JOBS},
        'actual_author_provenance': author_provenance,
        'schemas': schemas, 'materials': files,
        'actual_preflight': read(RUN / 'preflight' / preflight_attempt / 'summary.json'),
    }
    if attempt != 'round-1':
        previous = read(RUN / 'semantic-review' / 'round-1' / 'bundle.json')
        payload['prior_feedback'] = {'status': 'Raw prior invocation retained; failed literal quote validation and was not accepted.',
            'module_findings': previous.get('module_findings', []),
            'task': 'Determine whether the specific cross-chapter defect is actually closed in the current algorithms and shared fixtures; also review the full current module.'}
    characters = sum(len(item['content']) for item in files)
    if characters > 350000:
        raise RuntimeError(f'Material cap exceeded: {characters}')
    directory = RUN / 'semantic-review' / attempt
    save(directory / 'materials.json', {'files': files, 'documents': documents, 'characters': characters})
    record = ModelClient(CONFIG.adapters)('reviewer', json.dumps(payload, ensure_ascii=False), directory / 'invocation',
        engine=CONFIG.models['reviewer'], effort=CONFIG.models['effort'], timeout=CONFIG.execution['model_timeout'])
    if record['status'] != 'completed' or not isinstance(record.get('parsed'), dict):
        raise RuntimeError('Actual reviewer did not complete a strict JSON response; retained invocation is authoritative')
    bundle = record['parsed']
    save(directory / 'bundle.json', bundle)
    if {r['entry_id'] for r in bundle.get('reviews', [])} != set(JOBS):
        raise RuntimeError('Reviewer did not return every selected entry review')
    if any(f.get('blocking') or f.get('severity') == 'blocking' for f in bundle.get('module_findings', [])):
        raise RuntimeError('Integrated module has blocking findings; inspect retained bundle')
    for item in bundle['reviews']:
        if item.get('documents') != documents:
            raise RuntimeError('Reviewer changed the supplied literal documents map')
        build.validate_review(KB, state['jobs'][item['entry_id']], item)
        save(directory / (item['entry_id'] + '.json'), item)
    print(json.dumps({'review_attempt': attempt, 'decisions': {r['entry_id']: r['decision'] for r in bundle['reviews']}}, ensure_ascii=False))


def record_reviews(attempt):
    directory = RUN / 'semantic-review' / attempt
    with repository_writer(CONFIG):
        state = read(RUN / 'workflow.json')
        reviews = [read(directory / (key + '.json')) for key in JOBS]
        for item in reviews:
            build.validate_review(KB, state['jobs'][item['entry_id']], item)
        for item in reviews:
            print(json.dumps(build.record_review(KB, RUN, item['entry_id'],
                (directory / (item['entry_id'] + '.json')).relative_to(KB).as_posix()), ensure_ascii=False))


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'apply':
        apply_candidates()
    elif command == 'apply-quality-revision':
        apply_quality_revision()
    elif command == 'preflight':
        preflight(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif command == 'review':
        review(sys.argv[2], sys.argv[3])
    elif command == 'record':
        record_reviews(sys.argv[2])
    else:
        raise ValueError('Unknown scoped operation')
