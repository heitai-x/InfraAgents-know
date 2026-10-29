#!/usr/bin/env python3
"""K-TOOLS-004: standalone deterministic logical checks, without device measurements."""

import itertools
import random
from fractions import Fraction


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def rejects(action, label):
    try:
        action()
    except AssertionError:
        return
    raise AssertionError('negative control escaped: ' + label)


def clip(intervals, window):
    lo, hi = window
    check(lo <= hi, 'reversed window')
    result = []
    for start, end in intervals:
        check(start <= end, 'reversed interval')
        left, right = max(start, lo), min(end, hi)
        if left < right:
            result.append((left, right))
    return result


def union_length(intervals, window):
    total = 0
    current = None
    for start, end in sorted(clip(intervals, window)):
        if current is None:
            current = (start, end)
        elif start <= current[1]:
            current = (current[0], max(current[1], end))
        else:
            total += current[1] - current[0]
            current = (start, end)
    if current is not None:
        total += current[1] - current[0]
    return total


def tick_union(intervals, window):
    # Independent reference for integer endpoints: occupied unit cells.
    lo, hi = window
    return sum(any(start <= t < end for start, end in intervals)
               for t in range(lo, hi))


def test_intervals():
    window = (-1, 4)
    intervals = [(s, e) for s in range(-2, 4) for e in range(s, 5)]
    count = 0
    for first, second in itertools.product(intervals, repeat=2):
        sample = [first, second]
        check(union_length(sample, window) == tick_union(sample, window),
              'interval sweep differs from occupied cells')
        count += 1
    rng = random.Random(20261028)
    for _ in range(200):
        sample = []
        for _ in range(rng.randrange(9)):
            start = rng.randrange(-8, 45)
            sample.append((start, start + rng.randrange(12)))
        check(union_length(sample, (0, 40)) == tick_union(sample, (0, 40)),
              'random clipped union differs from reference')
    check(union_length([], (0, 40)) == 0, 'empty union')
    check(union_length([(0, 8)], (3, 3)) == 0, 'empty window')
    check(union_length([(0, 1), (1, 2)], (0, 2)) == 2, 'adjacency')
    rejects(lambda: union_length([(3, 2)], (0, 4)), 'reversed interval')
    overlap = [(0, 4), (2, 6), (5, 8)]
    naive = sum(e - s for s, e in overlap)
    rejects(lambda: check(naive == tick_union(overlap, (0, 8)),
                          'overlap sum is not occupied time'), 'summed overlap')
    print('intervals: %d exhaustive pairs and 200 random cases match integer-cell reference; overlap sum rejected' % count)


SCOPE = ('run', 'pid', 'dev', 'queue')


def full_key(record):
    return tuple(record[field] for field in SCOPE) + (record['corr'],)


def exact_join(apis, dispatches):
    check(len({a['id'] for a in apis}) == len(apis), 'duplicate API identity')
    check(len({d['id'] for d in dispatches}) == len(dispatches),
          'duplicate dispatch identity')
    index = {}
    output = {a['id']: [] for a in apis}
    for api in apis:
        if api['corr'] is None:
            continue
        key = full_key(api)
        check(key not in index, 'ambiguous submission key requires an epoch')
        index[key] = api['id']
    unresolved = []
    for dispatch in dispatches:
        if dispatch['corr'] is None:
            unresolved.append(dispatch['id'])
            continue
        owner = index.get(full_key(dispatch))
        if owner is None:
            unresolved.append(dispatch['id'])
        else:
            output[owner].append(dispatch['id'])
    return {name: sorted(ids) for name, ids in output.items()}, unresolved


def reference_join(apis, dispatches):
    # Relational nested-loop reference; does not use the indexed join.
    return {
        api['id']: sorted(
            dispatch['id'] for dispatch in dispatches
            if api['corr'] is not None
            and all(api[field] == dispatch[field]
                    for field in SCOPE + ('corr',)))
        for api in apis
    }


def validate_join(apis, dispatches, edges, unresolved):
    check(edges == reference_join(apis, dispatches), 'incorrect exact relation')
    matched = {name for ids in edges.values() for name in ids}
    unknown = set(unresolved)
    all_ids = {d['id'] for d in dispatches}
    check(len(unknown) == len(unresolved), 'duplicate unresolved identity')
    check(not matched & unknown, 'matched and unresolved overlap')
    check(matched | unknown == all_ids, 'dispatch missing from partition')


def record(name, pid=1, dev=0, queue=0, corr='c1', run='r'):
    return dict(id=name, pid=pid, dev=dev, queue=queue, corr=corr, run=run)


def test_attribution():
    apis = [record('L1'), record('L2', corr='c2'),
            record('L3', pid=2, dev=1), record('L4', corr=None)]
    dispatches = [record('K1'), record('K5'), record('K7', queue=1),
                  record('K2', corr='c2'), record('K3', pid=2, dev=1),
                  record('K4', corr=None)]
    edges, unresolved = exact_join(apis, dispatches)
    validate_join(apis, dispatches, edges, unresolved)
    check(edges == {'L1': ['K1', 'K5'], 'L2': ['K2'],
                    'L3': ['K3'], 'L4': []}, 'base relation changed')
    check(unresolved == ['K7', 'K4'],
          'unmatched queue and missing correlation must both remain unresolved')
    rejects(lambda: validate_join(apis, dispatches, edges, ['K4']),
            'discarding unmatched K7')
    bad_edges = {
        a['id']: sorted(d['id'] for d in dispatches
                        if a['corr'] is not None and a['corr'] == d['corr'])
        for a in apis}
    rejects(lambda: validate_join(apis, dispatches, bad_edges, ['K4']),
            'correlation-only cross-scope join')
    forced = {name: list(ids) for name, ids in edges.items()}
    forced['L4'] = ['K4']
    rejects(lambda: validate_join(apis, dispatches, forced, ['K7']),
            'invented ownership for missing correlation')
    for field in SCOPE:
        mutated = record('different')
        mutated[field] = 'other' if field == 'run' else 99
        out, unknown = exact_join([apis[0]], [mutated])
        validate_join([apis[0]], [mutated], out, unknown)
        check(out['L1'] == [] and unknown == ['different'],
              'one scope dimension was ignored: ' + field)
    for values in itertools.product(range(2), repeat=4):
        dispatch = record('grid')
        dispatch.update(dict(zip(SCOPE, values)))
        api = record('grid_api')
        api.update(dict(zip(SCOPE, (0, 0, 0, 0))))
        out, unknown = exact_join([api], [dispatch])
        validate_join([api], [dispatch], out, unknown)
    for corr in (0, ''):
        out, unknown = exact_join([record('a', corr=corr)],
                                  [record('k', corr=corr)])
        check(out == {'a': ['k']} and unknown == [],
              'valid false-valued correlation was treated as missing')
    check(exact_join([], dispatches)[1] == [d['id'] for d in dispatches],
          'empty API set lost dispatches')
    check(exact_join(apis, [])[1] == [], 'empty dispatch set')
    rejects(lambda: exact_join([record('a'), record('b')], []),
            'reused submission key without epoch')
    rejects(lambda: exact_join([record('a'), record('a', corr='c2')], []),
            'duplicate API identity')
    rejects(lambda: exact_join([], [record('k'), record('k')]),
            'duplicate dispatch identity')
    rng = random.Random(4004)
    for _ in range(30):
        rng.shuffle(apis)
        rng.shuffle(dispatches)
        out, unknown = exact_join(apis, dispatches)
        validate_join(apis, dispatches, out, unknown)
        check(set(unknown) == {'K7', 'K4'}, 'input order affected unresolved set')
    print('attribution: K7 and K4 remain unresolved; partition, scope, split dispatch and negative controls checked')


def allocated_costs(kernels, weights):
    costs = {}
    check(set(weights) == set(kernels), 'allocation kernel coverage differs')
    for name, (start, end) in kernels.items():
        check(start <= end, 'invalid kernel interval')
        shares = weights[name]
        check(shares and all(w >= 0 for w in shares.values()),
              'missing or negative shares')
        check(sum(shares.values()) == 1, 'kernel shares must sum to one')
        for owner, weight in shares.items():
            costs[owner] = costs.get(owner, Fraction(0)) + (end - start) * weight
    return costs


def test_overlap_and_sharing():
    host = [(0, 2), (2, 3), (3, 4), (4, 24), (24, 28)]
    kernels = {'K1': (6, 14), 'K2': (14, 22)}
    device = list(kernels.values())
    check(union_length(host, (0, 40)) == 28, 'host range union')
    check(union_length(device, (0, 40)) == 16, 'kernel union')
    check(union_length(host + device, (0, 40)) == 28, 'combined range union')
    rejects(lambda: check(28 + 16 == tick_union(host + device, (0, 40)),
                          'host wait overlaps device execution'), 'host/device addition')
    weights = {'K1': {'gemm': Fraction(1, 2), 'bias': Fraction(1, 2)},
               'K2': {'attention': Fraction(1)}}
    costs = allocated_costs(kernels, weights)
    check(costs == {'gemm': 4, 'bias': 4, 'attention': 8}, 'shared allocation')
    bad = dict(weights)
    bad['K1'] = {'gemm': Fraction(1), 'bias': Fraction(1)}
    rejects(lambda: allocated_costs(kernels, bad), 'full fused cost counted twice')
    bad['K1'] = {'gemm': Fraction(-1), 'bias': Fraction(2)}
    rejects(lambda: allocated_costs(kernels, bad), 'negative share with unit sum')
    concurrent = dict(kernels, K3=(10, 18))
    expanded = dict(weights, K3={'copy': Fraction(1)})
    total = sum(allocated_costs(concurrent, expanded).values())
    check(total == 24 and union_length(list(concurrent.values()), (0, 40)) == 16,
          'concurrent allocation conserves sum, not union')
    print('time/shared: host-device naive sum 44 versus union 28; fused allocation 16; concurrent sum 24 versus union 16')


def storage_segments(rows, window):
    # Each row is a protection lease: (storage identity, capacity, start, end).
    sizes, changes = {}, {}
    for identity, size, start, end in rows:
        check(size >= 0 and start <= end, 'invalid storage lease')
        check(identity not in sizes or sizes[identity] == size,
              'storage capacity changed without a new identity')
        sizes[identity] = size
        for left, right in clip([(start, end)], window):
            for t, delta in ((left, 1), (right, -1)):
                bucket = changes.setdefault(t, {})
                bucket[identity] = bucket.get(identity, 0) + delta
    # Validate an empty input window too.
    check(window[0] <= window[1], 'reversed memory window')
    refs, active, segments = {}, 0, []
    times = sorted(changes)
    for index, t in enumerate(times):
        for identity, delta in changes[t].items():
            before = refs.get(identity, 0)
            after = before + delta
            check(after >= 0, 'negative protection count')
            if before == 0 and after > 0:
                active += sizes[identity]
            elif before > 0 and after == 0:
                active -= sizes[identity]
            refs[identity] = after
        check(active >= 0, 'negative active capacity')
        if index + 1 < len(times):
            segments.append((t, times[index + 1], active))
    check(active == 0 and all(count == 0 for count in refs.values()),
          'clipped leases did not drain')
    return segments


def storage_peak(rows, window):
    return max((value for _, _, value in storage_segments(rows, window)), default=0)


def sweep_active(segments, t):
    return sum(value for start, end, value in segments if start <= t < end)


def reference_active(rows, t):
    # Independent reference: actual live identities, without endpoint events.
    live = {identity: size for identity, size, start, end in rows
            if start <= t < end}
    return sum(live.values())


def reference_peak(rows, window):
    return max((reference_active(rows, t) for t in range(*window)), default=0)


def faulty_active(rows, t):
    # Intentionally wrong: counts every lease/view as a separate allocation.
    return sum(size for _, size, start, end in rows if start <= t < end)


def faulty_peak(rows, window):
    return max((faulty_active(rows, t) for t in range(*window)), default=0)


def validate_storage_curve(rows, window):
    segments = storage_segments(rows, window)
    for t in range(*window):
        check(sweep_active(segments, t) == reference_active(rows, t),
              'storage curve differs from independent live-identity reference')
    check(storage_peak(rows, window) == reference_peak(rows, window),
          'storage peak differs from independent reference')


def validate_reuse(reuse, consumers):
    for identity, completion in consumers:
        check(identity in reuse, 'missing reuse boundary')
        check(reuse[identity] >= completion, 'storage reused before reader completion')


def validate_levels(rows, reserved, device, window):
    for t in range(*window):
        check(0 <= reference_active(rows, t) <= reserved <= device,
              'incompatible nested memory levels')


def test_memory():
    rows = [('weights', 6, 0, 40), ('compile', 3, 0, 5),
            ('temp', 2, 6, 14), ('temp', 2, 8, 14), ('kv', 4, 14, 40)]
    window = (0, 40)
    validate_storage_curve(rows, window)
    check(storage_peak(rows, window) == 10, 'base storage peak')
    check(reference_active(rows, 20) == 10, 'active sample')
    validate_levels(rows, 14, 16, window)
    rejects(lambda: validate_levels(rows, 9, 16, window), 'reserved below active')

    # Preserve the concrete round-2 fixture. Its global peak masks the bug:
    # the wrong 10 during [8,14) equals the later correct 10 during [14,40).
    check(faulty_peak(rows, window) == reference_peak(rows, window) == 10,
          'masked-peak counterexample changed')
    check(reference_active(rows, 8) == 8 and faulty_active(rows, 8) == 10,
          'alias pointwise counterexample setup')
    rejects(lambda: check(faulty_active(rows, 8) == reference_active(rows, 8),
                          'view double counting at t=8'), 'view double counting, pointwise')
    alias_window = (8, 14)
    validate_storage_curve(rows, alias_window)
    check(storage_peak(rows, alias_window) == reference_peak(rows, alias_window) == 8,
          'alias-window reference peak')
    check(faulty_peak(rows, alias_window) == 10, 'faulty alias-window peak')
    rejects(lambda: check(faulty_peak(rows, alias_window)
                          == reference_peak(rows, alias_window),
                          'view double counting in alias window'),
            'view double counting, local peak')

    consumers = [('temp', 12), ('temp', 14)]
    validate_reuse({'temp': 14}, consumers)
    rejects(lambda: validate_reuse({'temp': 13}, consumers), 'early storage reuse')
    retained = [(identity, size, start, 18 if identity == 'temp' else end)
                for identity, size, start, end in rows]
    validate_storage_curve(retained, window)
    check(storage_peak(retained, window) == 12, 'retention peak')
    check(storage_peak(rows, (14, 14)) == 0, 'empty memory window')
    check(storage_peak([], (0, 4)) == 0, 'empty storage set')
    check(storage_peak([('s', 3, -5, 5)], (0, 3)) == 3,
          'object alive before window was lost')
    check(storage_peak([('s', 3, 0, 2), ('s', 3, 2, 4)], (0, 4)) == 3,
          'same storage adjacent leases')
    check(storage_peak([('a', 2, 0, 2), ('b', 4, 2, 4)], (0, 4)) == 4,
          'simultaneous release/start created a false peak')
    check(storage_peak([('s', 3, 2, 2)], (0, 4)) == 0, 'zero-duration lease')
    check(storage_peak([('s', 0, 0, 4)], (0, 4)) == 0, 'zero capacity')
    rejects(lambda: storage_peak([('s', 2, 0, 2), ('s', 3, 1, 3)], (0, 4)),
            'capacity conflict without generation change')
    rejects(lambda: storage_peak([('s', 2, 3, 2)], (0, 4)), 'reversed lease')
    rejects(lambda: storage_peak([], (4, 0)), 'reversed memory window')
    rng = random.Random(104004)
    for _ in range(100):
        sample = []
        for identity in range(rng.randrange(6)):
            size = rng.randrange(6)
            for _ in range(rng.randrange(1, 4)):
                start = rng.randrange(-4, 25)
                sample.append((identity, size, start, start + rng.randrange(10)))
        validate_storage_curve(sample, (0, 30))
    print('memory: full correct/faulty peaks both 10 GiB mask alias bug; at t=8 and in [8,14), correct 8 versus faulty 10; both alias negative controls rejected')
    print('memory: endpoint sweep matches independent curves and peaks in 100 random cases; retention peak 12; level and early-reuse controls rejected')


def record_fragments(events, schedule):
    windows = []
    start = None
    for step in range(len(schedule) + 1):
        recording = step < len(schedule) and schedule[step] == 'RECORD'
        if recording and start is None:
            start = step
        elif not recording and start is not None:
            windows.append((start, step))
            start = None
    output = []
    for identity, left, right in events:
        for window in windows:
            for a, b in clip([(left, right)], window):
                output.append((identity, a, b))
    return output


def validate_fragments(events, schedule, fragments):
    expected = {(identity, t) for identity, start, end in events
                for t, state in enumerate(schedule)
                if state == 'RECORD' and start <= t < end}
    actual = [(identity, t) for identity, start, end in fragments
              for t in range(start, end)]
    check(len(actual) == len(set(actual)), 'duplicate event coverage')
    check(set(actual) == expected, 'record coverage includes missing or extra cells')


def test_window():
    schedule = ['WAIT', 'WAIT', 'WARMUP', 'RECORD', 'RECORD', 'RECORD', 'SAVE', 'WAIT']
    events = [('a', -2, 4), ('b', 4, 7), ('c', 6, 8), ('zero', 3, 3)]
    fragments = record_fragments(events, schedule)
    check(fragments == [('a', 3, 4), ('b', 4, 6)], 'clipped RECORD fragments')
    validate_fragments(events, schedule, fragments)
    rejects(lambda: validate_fragments(events, schedule, events),
            'WAIT/WARMUP and window tails included')
    rejects(lambda: validate_fragments(events, schedule,
                                      [('a', 3, 4), ('b', 4, 7)]),
            'API-start-only filtering retained an outside tail')
    check(record_fragments(events, []) == [], 'empty schedule')
    check(record_fragments(events, ['WAIT', 'WARMUP']) == [], 'no RECORD state')
    separated = ['RECORD', 'WAIT', 'RECORD']
    spanning = [('span', -1, 4)]
    pieces = record_fragments(spanning, separated)
    check(pieces == [('span', 0, 1), ('span', 2, 3)], 'separate record windows')
    validate_fragments(spanning, separated, pieces)
    print('window: RECORD coverage and boundary clipping checked; warmup and outside-tail negative controls rejected')


def main():
    test_intervals()
    test_attribution()
    test_overlap_and_sharing()
    test_memory()
    test_window()
    print('scope: deterministic integer-time logical accounting only; no profiler capture, CPU performance measurement, GPU measurement or Agent outcome experiment')


if __name__ == '__main__':
    main()
