"""Independent pipeline transfer exercise; imports no author-side check code."""
from pathlib import Path
import hashlib
import json
import math


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def schedule(depth, stages):
    events = []
    for j in range(stages):
        previous_copy = events[j-1]['s']+2 if j else 0
        reused_slot = events[j-depth]['e'] if j >= depth else 0
        s = max(previous_copy, reused_slot)
        f = s+7
        b = max(f, events[j-1]['e'] if j else 0)
        events.append({'j': j, 'slot': j % depth, 'generation': j//depth,
                       's': s, 'f': f, 'b': b, 'e': b+4})
    return events


def tile_model(a, b, m, n, k, bm, bn):
    accumulators = [[0.0]*3 for _ in range(64)]
    seen_outputs = set()
    for stage in range(math.ceil(k/5)):
        shared_a = [None]*40
        shared_b = [None]*120
        written_a, written_b = set(), set()
        for tid in range(64):
            if tid < 40:
                r, kk = divmod(tid, 5)
                row, reduction = bm*8+r, stage*5+kk
                # Address formation is guarded by these same bounds.
                shared_a[tid] = a[row][reduction] if row < m and reduction < k else 0.0
                written_a.add(tid)
            for u in range(2):
                index = tid+64*u
                if index < 120:
                    kk, c = divmod(index, 24)
                    reduction, col = stage*5+kk, bn*24+c
                    shared_b[index] = b[reduction][col] if reduction < k and col < n else 0.0
                    written_b.add(index)
        assert written_a == set(range(40)) and written_b == set(range(120))
        # A scalar phase here models data readiness and all-reader completion;
        # it makes no assertion about CUDA synchronization instructions.
        for tid in range(64):
            for t in range(3):
                row, col = divmod(tid+64*t, 24)
                for kk in range(5):
                    accumulators[tid][t] += shared_a[row*5+kk]*shared_b[kk*24+col]
    result = {}
    for tid in range(64):
        for t in range(3):
            row, col = divmod(tid+64*t, 24)
            assert (row, col) not in seen_outputs
            seen_outputs.add((row, col))
            global_row, global_col = 8*bm+row, 24*bn+col
            if global_row < m and global_col < n:
                result[global_row, global_col] = accumulators[tid][t]
    assert seen_outputs == {(r, c) for r in range(8) for c in range(24)}
    return result


def mapping_checks():
    cases = 0
    for m in (1, 7, 8, 11):
        for n in (1, 23, 24, 29):
            for k in (0, 1, 4, 5, 6, 17):
                a = [[(3*r-2*q+1)/16 for q in range(k)] for r in range(m)]
                b = [[(q+2*c-7)/32 for c in range(n)] for q in range(k)]
                observed = {}
                for bm in range(math.ceil(m/8)):
                    for bn in range(math.ceil(n/24)):
                        tile = tile_model(a, b, m, n, k, bm, bn)
                        assert not observed.keys() & tile.keys()
                        observed.update(tile)
                expected = {(r, c): math.fsum(a[r][q]*b[q][c] for q in range(k))
                            for r in range(m) for c in range(n)}
                assert observed == expected
                cases += 1
    for m, n in ((0, 3), (3, 0), (0, 0)):
        assert math.ceil(m/8)*math.ceil(n/24) == 0
    # Demonstrate that using global strides for shared buffers does not work.
    assert (2*17+3) != (2*5+3) and (3*29+4) != (3*24+4)
    return cases


def main():
    shared = 4*(8*5+5*24)
    accumulator = 64*3*4
    tables = {str(d): schedule(d, 4) for d in (1, 2, 3)}
    times = {d: table[-1]['e'] for d, table in tables.items()}
    assert shared == 640 and accumulator == 768
    assert times == {'1': 44, '2': 26, '3': 23}
    for d, table in tables.items():
        for j, event in enumerate(table):
            assert event['f'] == event['s']+7
            assert event['b'] >= event['f'] and event['e'] == event['b']+4
            if j:
                assert event['s'] >= table[j-1]['s']+2 and event['b'] >= table[j-1]['e']
            if j >= int(d):
                assert event['s'] >= table[j-int(d)]['e']
    boundary = {str(n): {str(d): (schedule(d, n)[-1]['e'] if n else 0)
                         for d in (1, 2, 3)} for n in (0, 1, 2)}
    assert boundary == {'0': {'1': 0, '2': 0, '3': 0},
                        '1': {'1': 11, '2': 11, '3': 11},
                        '2': {'1': 22, '2': 15, '3': 15}}
    grid = {}
    for capacity in (2048, 4096):
        residency = {str(d): min(2, capacity//(d*shared)) for d in (1, 2, 3)}
        grid[str(capacity)] = {'residency': residency,
            'times': {d: math.ceil(4/r)*times[d] for d, r in residency.items()}}
    assert grid['2048'] == {'residency': {'1': 2, '2': 1, '3': 1},
                            'times': {'1': 88, '2': 104, '3': 92}}
    assert grid['4096'] == {'residency': {'1': 2, '2': 2, '3': 2},
                            'times': {'1': 88, '2': 52, '3': 46}}
    documents = ['knowledge/hardware/gpu-kernel-pipelines.md',
                 'knowledge/sources/S-CUDA-BEST.md', 'knowledge/sources/S-CUDA-ASYNC.md',
                 'docs/evidence/parallel-mechanisms-20261027/tasks/pipeline-transfer.md']
    report = {'status': 'PASS', 'phase': 'initial_answer_before_counterevidence',
        'date': '2026-10-28', 'shared_per_stage_bytes': shared, 'accumulator_bytes': accumulator,
        'cta_times': times, 'events': tables, 'boundary_times': boundary, 'grid': grid,
        'mapping_cases': mapping_checks(),
        'hypotheses': [
            {'name': 'fill_drain', 'prediction': 'Fixed extra cost is amortized for longer K; long-stage slope should not worsen if this is the only cause.',
             'intervention': 'Sweep stages at fixed tile, compiler path and timing boundary; fit intercept and slope.'},
            {'name': 'residency', 'prediction': 'Single CTA can improve while the grid gets slower near a resource threshold.',
             'intervention': 'Record actual allocated resources and active CTAs; compare one CTA with many, and vary an isolated capacity budget.'},
            {'name': 'spill', 'prediction': 'Additional local-memory traffic persists per stage and may worsen the long-stage slope.',
             'intervention': 'Inspect generated program/local traffic; shorten unrelated live ranges while recording all changed resources.'},
            {'name': 'unrealized_overlap', 'prediction': 'Async source expressions do not shorten actual exposed copy/compute gaps.',
             'intervention': 'Inspect generated instructions and hardware timelines with the same logical work.'},
            {'name': 'preparation_or_contention', 'prediction': 'Full-call cost or multi-CTA copy/compute rates differ despite the single-CTA model.',
             'intervention': 'Separate preparation from kernel and measure L/I/C as a function of residency.'}],
        'documents': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in documents},
        'scope': 'Cross-agent reading and independent changed-shape CPU exercise with inherited parent context; no OS isolation, fixed budget, GPU execution or controlled knowledge-gain estimate.',
        'limits': ['Python scalar operations do not prove FP32 SIMT rounding or compiled instruction correctness.',
                   'The declared event recurrence and nonoverlapping wave model are synthetic assumptions, not measured GPU behavior.',
                   'No author-side checker, report, other answer, or upstream implementation was read/imported.']}
    (HERE/'pipeline-answer.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'mapping_cases': report['mapping_cases'],
                      'shared': shared, 'accumulator': accumulator, 'cta_times': times, 'grid': grid}))


if __name__ == '__main__':
    main()
