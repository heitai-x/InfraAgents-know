"""One-time prose-only registration and real sharded frozen-reader verification."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2]
sys.path.insert(0, str(PACKAGE / 'src'))
from knowledge_workflow import build
from knowledge_workflow.cli import current_run, repository_writer
from knowledge_workflow.config import load_config
from knowledge_workflow.campaign import next_version
from knowledge_workflow.release import finalize_verification

CONFIG = load_config(PACKAGE / 'workflow.local.toml')
ROOT = CONFIG.repository
read = build.read
write = build.write
digest = build.digest


def apply():
    with repository_writer(CONFIG):
        run = current_run(CONFIG)
        state = read(run / 'workflow.json')
        assert not state.get('registered_version'), 'Use the untouched successor baseline'
        assert all(j['state'] in ('queued', 'revise', 'needs_attention') or
                   (j.get('revision_scope') == 'editorial_only' and j['state'] in ('working','reviewed','ready'))
                   for j in state['jobs'].values())
        assert state['baseline_version'] == read(ROOT / 'VERSION.json')['version_id']
        assert digest(ROOT / 'VERSION.json') == state['baseline_sha256']
        manifest = read(HERE / 'manifest.json')
        independent = read(HERE / 'independent-editorial-review.json')
        assert independent['manifest_sha256'] == digest(HERE / 'manifest.json')
        entries = manifest['entries']
        assert len(entries) == 40
        catalog = {e['id']: e for e in read(ROOT / 'knowledge/catalog.json')['entries']}
        accounting = {e['id']: e for e in read(ROOT / 'docs/expansion-status.json')['entries']}
        reviews = {}
        for e in entries:
            assert e['id'] not in state['jobs'] or state['jobs'][e['id']].get('revision_scope') == 'editorial_only'
            assert e['baseline_sha256'] == catalog[e['id']]['sha256']
            assert digest(ROOT / e['path']) in (e['baseline_sha256'], e['body_sha256'])
            assert digest(Path(e['staged_path'])) == e['body_sha256']
            review = read(HERE / 'reviews' / (e['id'] + '.json'))
            assert review['body_sha256'] == e['body_sha256']
            assert review['review_scope'] == 'editorial_only' and review['decision'] == 'accept'
            assert review['checks'] == [] and review['no_checks_reason']
            reviews[e['id']] = review
            for check in e['prior_checks']:
                assert digest(ROOT / check['path']) == check['sha256']
                record = read(ROOT / check['path'])
                assert record['exit_code'] == 0
                assert digest(ROOT / record['script_path']) == record['script_sha256']
        convention = read(HERE / 'convention-plan.json')
        for e in convention['entries']:
            text = (ROOT / e['path']).read_text(encoding='utf-8-sig')
            for op in e['replacements']:
                assert text.count(op['old']) == 1
                text = text.replace(op['old'], op['new'], 1)
        evidence = run / 'editorial'
        evidence.mkdir(exist_ok=True)
        pending_write = run / 'workflow.json.writing'
        if pending_write.exists():
            pending = read(pending_write)
            differences = [k for k,j in pending['jobs'].items() if j != state['jobs'][k]]
            assert differences == ['K-SYS-001'] and state['jobs']['K-SYS-001']['state'] == 'reviewed'
            assert pending['jobs']['K-SYS-001'] == {**state['jobs']['K-SYS-001'], 'state':'ready'}
            (evidence / 'interrupted-workflow-write.json').write_bytes(pending_write.read_bytes())
        for name in ['manifest.json', 'algorithms-plan.json', 'hardware-plan.json', 'systems-plan.json',
                     'latest-plan.json', 'convention-plan.json', 'independent-editorial-review.json',
                     'convention-check.json', 'publish.py']:
            target = evidence / name
            data = (HERE / name).read_bytes()
            if target.exists() and target.read_bytes() != data:
                assert name == 'publish.py'
                target = evidence / 'publish-resume.py'
            if not target.exists():
                target.write_bytes(data)
        for e in entries:
            c = catalog[e['id']]
            if e['id'] in state['jobs']:
                continue
            state['jobs'][e['id']] = {
                'id': e['id'], 'path': e['path'], 'baseline_revision': c['revision'],
                'baseline_sha256': c['sha256'], 'prior_review': accounting[e['id']]['review'],
                'source_ids': c.get('source_ids', []), 'related': c.get('related', []),
                'state': 'queued', 'round': 0, 'author': None, 'reviews': [], 'checks': [],
                'agent_use': 'unassessed', 'issues': [], 'revision_scope': 'editorial_only',
                'prior_semantic_review': e['prior_review'], 'prior_algorithm_checks': e['prior_checks']}
        write(run / 'workflow.json', state)
        for e in entries:
            job = read(run / 'workflow.json')['jobs'][e['id']]
            if job['state'] == 'queued':
                build.claim(run, e['author'], entry_id=e['id'])
                job = read(run / 'workflow.json')['jobs'][e['id']]
            if job['state'] == 'working':
                (ROOT / e['path']).write_bytes(Path(e['staged_path']).read_bytes())
                review_path = evidence / 'reviews' / (e['id'] + '.json')
                write(review_path, reviews[e['id']])
                build.record_review(ROOT, run, e['id'], review_path.relative_to(ROOT).as_posix())
                job = read(run / 'workflow.json')['jobs'][e['id']]
            if job['state'] == 'reviewed':
                result = build.execute_checks(ROOT, run, e['id'])
                assert result['state'] == 'ready' and result['checks'] == []
            else:
                assert job['state'] == 'ready'
            assert digest(ROOT / e['path']) == e['body_sha256']
            time.sleep(0.15)
        for e in convention['entries']:
            path = ROOT / e['path']
            text = path.read_text(encoding='utf-8-sig')
            for op in e['replacements']:
                text = text.replace(op['old'], op['new'], 1)
            path.write_text(text, encoding='utf-8', newline='\n')
        version = next_version(read(ROOT / 'VERSION.json')['version_id'])
        result = build.register(ROOT, run, version, CONFIG.runtime_source)
        write(HERE / 'publication.json', {'run': run.relative_to(ROOT).as_posix(), **result})
        print(json.dumps(result, ensure_ascii=False), flush=True)


def execute(argv, log, report, script):
    before = digest(script)
    started = build.now()
    clock = time.perf_counter()
    try:
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                                encoding='utf-8', timeout=1800)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        code = -1
        stdout = exc.stdout.decode('utf-8', errors='replace') if isinstance(exc.stdout, bytes) else exc.stdout or ''
        stderr = 'Timed out after 1800 seconds; inspect retained partial output.'
    write(log, {'argv': argv, 'exit_code': code, 'stdout': stdout, 'stderr': stderr,
                'script_path': script.relative_to(ROOT).as_posix(), 'script_sha256': before,
                'runtime_source': str(CONFIG.runtime_source), 'started_at': started,
                'executed_at': build.now(), 'elapsed_seconds': time.perf_counter()-clock})
    assert digest(script) == before, 'Verification tool changed'
    assert code == 0 and report.is_file(), str(log)
    return read(report)


def verify():
    with repository_writer(CONFIG):
        publication = read(HERE / 'publication.json')
        run = ROOT / publication['run']
        state = read(run / 'workflow.json')
        ids = publication['selected_ids']
        assert len(ids) == 40 and all(state['jobs'][i]['state'] == 'registered' for i in ids)
        for n in range(1, 100):
            attempt = run / 'verification' / f'editorial-{n:03d}'
            if not attempt.exists():
                attempt.mkdir(parents=True)
                break
        validator = ROOT / 'tools/validate_knowledge.py'
        probe = ROOT / 'tools/probe_mechanism_expansion.py'
        structural = attempt / 'structural.json'
        execute([sys.executable,'-X','utf8',str(validator),'--repository',str(ROOT),'--report',str(structural)],
                attempt / 'structural-execution.json', structural, validator)
        print('Structural validation passed; starting five independent frozen-reader probes.', flush=True)
        chunks = [ids[i:i+8] for i in range(0, len(ids), 8)]
        reports = {}
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {}
            for i, chunk in enumerate(chunks):
                report = attempt / f'reader-part-{i+1}.json'
                argv = [sys.executable,'-X','utf8',str(probe),'--runtime',str(CONFIG.runtime_source.parent),
                        '--entries',','.join(chunk),'--report',str(report)]
                future = pool.submit(execute, argv, attempt / f'reader-part-{i+1}-execution.json', report, probe)
                futures[future] = (i, report)
            for future in as_completed(futures):
                i, path = futures[future]
                report = future.result()
                assert report['result'] == 'PASS' and report['canonical_unchanged'] is True
                assert {m['id'] for m in report['modules']} == set(chunks[i])
                reports[i] = (path, report)
                print(f'Frozen-reader part {i+1}/5 passed: {len(report["modules"])} chapters.', flush=True)
        sources, originals, rejected, modules, parts = {}, {}, set(), [], []
        documents = None
        for i in sorted(reports):
            path, report = reports[i]
            if documents is None:
                documents = report['documents']
            assert documents == report['documents']
            modules.extend(report['modules'])
            for k,v in report['sources'].items():
                assert k not in sources or sources[k] == v
                sources[k] = v
            for item in report['readable_originals']:
                assert item['id'] not in originals or originals[item['id']] == item
                originals[item['id']] = item
            rejected.update(report['implementation_paths_rejected'])
            parts.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path),
                          'execution': path.with_name(path.stem+'-execution.json').relative_to(ROOT).as_posix()})
        assert len(modules) == len({m['id'] for m in modules}) == 40
        aggregate = attempt / 'reader.json'
        write(aggregate, {'result':'PASS','modules':modules,'source_count':len(sources),'sources':sources,
            'readable_originals':list(originals.values()),'implementation_paths_rejected':sorted(rejected),
            'canonical_unchanged':True,'documents':documents,'parts':parts,
            'scope':'Union of five actual isolated frozen-reader invocations; each selected body, all H2 sections, sources and adoption identities read by the unchanged repository probe.',
            'limits':['Editorial revision only; no algorithm checks rerun, no model invocation, device result or Agent-effectiveness evidence.']})
        result = finalize_verification(ROOT, run, structural, aggregate)
        for i in ids:
            state['jobs'][i]['state'] = 'verified'
        state['verified_at'] = build.now()
        write(run / 'workflow.json', state)
        following = re.sub(r'batch(\d+)$', lambda m:'batch'+str(int(m[1])+1), run.name)
        assert following != run.name
        continuation = build.continue_run(ROOT, run, following)
        write(HERE / 'completion.json', {**result, 'editorial_entries':ids, 'continuation':continuation,
            'algorithm_checks_rerun':False,'prior_algorithm_evidence_preserved':True,
            'structural_execution':(attempt/'structural-execution.json').relative_to(ROOT).as_posix(),
            'reader_parts':parts})
        print(json.dumps({'verification': result, 'continuation': continuation},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    {'apply':apply, 'verify':verify}[sys.argv[1]]()
