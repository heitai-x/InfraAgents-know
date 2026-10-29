#!/usr/bin/env python3
"""Check numeric outputs of three actual cross-reading transfer exercises.

The answers are produced separately by sub-agents. Qualitative diagnosis and
revision are reviewed by the primary agent, not inferred from keyword counts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'docs/evidence/parallel-mechanisms-20261027'


def run():
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def read(name):
        return json.loads((EVIDENCE/'transfers'/name).read_bytes())

    paged, scheduling, pipeline = [read(x+'-answer.json') for x in ['paged', 'scheduling', 'pipeline']]
    check('four independent pool byte addresses', paged['offsets'] == {
        'NHD_K': 94, 'NHD_V': 158, 'HND_K': 100, 'HND_V': 168})
    check('owners and two immutable snapshot readers after cancel', paged['after_cancel'] == {
        '5': {'owners': 1, 'pins': 2}, '2': {'owners': 1, 'pins': 1}, '7': {'owners': 0, 'pins': 1}})
    check('COW bytes count valid prefix separately from append', paged['cow'] == {
        'read_bytes': 64, 'write_bytes': 64, 'append_bytes': 32})
    for page in [3, 8]:
        entries = 13*math.ceil(5/page)
        allocated = entries*page*96*1024
        expected = {'allocated_bytes': allocated, 'wasted_bytes': allocated-13*5*96*1024,
                    'table_entries': entries}
        check(f'capacity P={page}', all(paged['capacity'][f'P{page}'][k] == v for k, v in expected.items()))

    rejected = scheduling['rejected']
    check('token-only proposed batch fails time bound', rejected == {
        'Q': 5, 'pairs': 20, 'time_us': 40.0, 'incremental_pages': 2})
    states = {'A': (5, 3, 1), 'B': (8, 7, 3), 'C': (4, 0, 0)}
    choices = set()
    check('at least two legal alternative batches', len(scheduling['legal_candidates']) >= 2)
    for i, answer in enumerate(scheduling['legal_candidates']):
        alloc = answer['alloc']
        check(f'candidate {i} identity set', set(alloc) == set(states))
        for name, q in alloc.items():
            check(f'candidate {i} known-input dependency {name}',
                  isinstance(q, int) and 0 <= q <= states[name][0]-states[name][1])
        qsum = sum(alloc.values())
        pairs = sum(q*states[name][1]+q*(q+1)//2 for name, q in alloc.items())
        pages = sum(max(0, math.ceil((states[name][1]+q)/3)-states[name][2])
                    for name, q in alloc.items() if q)
        predicted = 20+2*qsum+0.5*pairs
        check(f'candidate {i} work and time',
              answer['Q'] == qsum and answer['pairs'] == pairs and answer['time_us'] == predicted)
        check(f'candidate {i} hard constraints', 0 < qsum <= 5 and predicted <= 36 and 4+pages <= 7)
        check(f'candidate {i} incremental pages', answer['incremental_pages'] == pages)
        choices.add(tuple(alloc[name] for name in sorted(states)))
    check('alternative allocations actually differ', len(choices) >= 2)
    check('packed output rows retain position and sample eligibility', scheduling['selected_rows'] == [
        {'request': 'A', 'position': 3, 'sample': False},
        {'request': 'A', 'position': 4, 'sample': True},
        {'request': 'B', 'position': 7, 'sample': True}])
    check('sampled token still lacks KV', scheduling['A_after'] == {'n': 6, 'k': 5})

    check('changed shape shared and logical accumulator bytes',
          pipeline['shared_per_stage_bytes'] == 640 and pipeline['accumulator_bytes'] == 768)
    # Independently recompute the task's finite recurrence from its parameters.
    times = {}
    for slots in [1, 2, 3]:
        copies, finished, previous_issue = [], [], -2
        for index in range(4):
            issued = max(previous_issue+2, finished[index-slots] if index >= slots else 0)
            copies.append(issued+7)
            finished.append(max(copies[-1], finished[-1] if finished else 0)+4)
            previous_issue = issued
        times[str(slots)] = finished[-1]
    check('finite four-stage makespan', pipeline['cta_times'] == times)
    for limit in [2048, 4096]:
        residency = {str(d): min(2, limit//(d*640)) for d in [1, 2, 3]}
        total = {str(d): math.ceil(4/residency[str(d)])*times[str(d)] for d in [1, 2, 3]}
        check(f'capacity {limit} residency and wave reversal', pipeline['grid'][str(limit)] == {
            'residency': residency, 'times': total})

    documents = {}
    for name, answer in [('paged', paged), ('scheduling', scheduling), ('pipeline', pipeline)]:
        check(f'{name} recorded more than one competing explanation', len(answer['hypotheses']) >= 2)
        check(f'{name} actual knowledge reads are recorded', bool(answer['documents']))
        for path, sha in answer['documents'].items():
            check(f'{name} read identity: {path}', hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == sha)
            documents[path] = sha
        for suffix in ['answer.md', 'answer.json', 'solution.py', 'revision.md']:
            path = EVIDENCE/'transfers'/f'{name}-{suffix}'
            check(f'{name} retained {suffix}', path.is_file())
            documents[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        preserved = read(name+'-initial-sha.json')
        for path, sha in preserved.items():
            check(f'{name} initial artifact unchanged after feedback: {path}',
                  hashlib.sha256((EVIDENCE/'transfers'/path).read_bytes()).hexdigest() == sha)
        for suffix in ['transfer.md', 'feedback.md']:
            task = EVIDENCE/'tasks'/f'{name}-{suffix}'
            documents[task.relative_to(ROOT).as_posix()] = hashlib.sha256(task.read_bytes()).hexdigest()
        identity_path = EVIDENCE/'transfers'/f'{name}-initial-sha.json'
        documents[identity_path.relative_to(ROOT).as_posix()] = hashlib.sha256(identity_path.read_bytes()).hexdigest()
    return {'result': 'PASS', 'count': len(checks), 'checks': checks, 'documents': documents,
            'scope': 'Numeric results and retained artifacts from three actual sub-agent cross-reading exercises.',
            'limits': ['Agents share a parent conversation and repository; this is not a blind or controlled before/after study.',
                       'File access restrictions were task instructions, not an operating-system sandbox.',
                       'Qualitative diagnosis and feedback updates require the separate primary review.',
                       'Synthetic CPU calculations do not establish target GPU behavior or complete engine construction capability.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'result': report['result'], 'count': report['count']}))
