#!/usr/bin/env python3
# K-WORK-003: deterministic, read-only, offline CPU mechanism models.
# No project imports, subprocesses, network, filesystem access or GPU claims.
from collections import Counter
from itertools import product
from math import isclose


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_value_error(call, label):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('expected rejection: ' + label)


def nonnegative_integer(value):
    return type(value) is int and value >= 0


def legal_record(record):
    x, y = record.get('x'), record.get('y')
    if not nonnegative_integer(x) or x < 1:
        return False
    if not nonnegative_integer(y) or record.get('artifact') != 'M':
        return False
    if 'media' in record:
        h, w = record['media']
        if not all(nonnegative_integer(v) and v > 0 and v % 28 == 0
                   for v in (h, w)):
            return False
    return True


def graph_order(nodes):
    by_id = {node['id']: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError('duplicate node identity')
    for node in nodes:
        parents = [parent for parent, gap in node['deps']]
        if len(parents) != len(set(parents)):
            raise ValueError('duplicate dependency')
        if any(parent not in by_id for parent in parents):
            raise ValueError('missing dependency')
        if any(not nonnegative_integer(gap) for parent, gap in node['deps']):
            raise ValueError('invalid dependency delay')
    done, order = set(), []
    while len(done) < len(nodes):
        progressed = False
        for node in nodes:
            if node['id'] in done:
                continue
            if all(parent in done for parent, gap in node['deps']):
                done.add(node['id'])
                order.append(node)
                progressed = True
        if not progressed:
            raise ValueError('dependency cycle')
    return order


def canonical_units(records):
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate raw identity')
    traces = {r['id']: ['RAW'] for r in records}
    staged, rejected, blocked = {}, set(), set()
    groups = {}
    for record in records:
        rid = record['id']
        groups.setdefault(record['unit'], []).append(record)
        if not legal_record(record):
            traces[rid].append('REJECTED')
            rejected.add(rid)
            continue
        staged[rid] = dict(record)
        traces[rid].append('VALIDATED')
    accepted = []
    for group in groups.values():
        bad = any(r['id'] not in staged for r in group)
        if not bad:
            try:
                graph_order(group)
            except ValueError:
                bad = True
        if bad:
            for record in group:
                rid = record['id']
                if rid in staged:
                    traces[rid].append('BLOCKED_DEPENDENCY')
                    blocked.add(rid)
            continue
        for record in group:
            traces[record['id']].append('ELIGIBLE')
            accepted.append(staged[record['id']])
    return accepted, rejected, blocked, traces


def replay_unit(nodes, copy_id, base, ignore_dependencies=False):
    if not nonnegative_integer(base):
        raise ValueError('invalid replay base')
    for node in nodes:
        if not all(nonnegative_integer(node[key])
                   for key in ('offset', 'duration')):
            raise ValueError('invalid timing')
    order = graph_order(nodes)
    ends, result = {}, []
    for node in order:
        start = base + node['offset']
        if not ignore_dependencies:
            for parent, gap in node['deps']:
                start = max(start, ends[parent] + gap)
        end = start + node['duration']
        ends[node['id']] = end
        result.append({
            'id': (copy_id, node['id']), 'start': start, 'end': end,
            'parents': [((copy_id, parent), gap)
                        for parent, gap in node['deps']],
        })
    return result


def replay_invariant(events):
    by_id = {event['id']: event for event in events}
    if len(by_id) != len(events):
        return False
    for event in events:
        if event['end'] < event['start']:
            return False
        for parent, gap in event['parents']:
            if parent not in by_id:
                return False
            if event['start'] < by_id[parent]['end'] + gap:
                return False
    return True


def prefill_formula(x, h):
    if not (type(x) is int and type(h) is int and x >= 1 and 0 <= h < x):
        raise ValueError('invalid KV-only generation boundary')
    q = x - h
    return q * h + q * (q + 1) // 2


def decode_formula(x, y):
    if not (type(x) is int and x >= 1 and nonnegative_integer(y)):
        raise ValueError('invalid selected-token boundary')
    m = max(y - 1, 0)
    return m * x + m * (m + 1) // 2


def event_pair_reference(x, h, y):
    # Enumerate individual legal edges, independently of the closed forms.
    prompt_edges = {(query, key) for query in range(h + 1, x + 1)
                    for key in range(1, query + 1)}
    consumed = x
    selected = 0
    decode_edges = set()
    while selected < y:
        selected += 1
        if selected == y:
            break
        consumed += 1
        for key in range(1, consumed + 1):
            decode_edges.add((consumed, key))
    return len(prompt_edges), len(decode_edges), consumed


def select_until_stop(proposals, cap):
    if not nonnegative_integer(cap):
        raise ValueError('invalid cap')
    selected = []
    for token in proposals:
        if len(selected) == cap:
            break
        selected.append(token)
        if token == 'EOS':
            break
    return selected


def old_decode_control(x, y):
    # Faulty: recharges a first-output step after prefill.
    return sum(x + t for t in range(y))


def physical_sweep(sizes, events, end, mode='physical'):
    # refs order: request A, cache C, reader R, writer/handoff W.
    refs = {page: [0, 0, 0, 0] for page in sizes}
    cuts = sorted(set(events) | {0, end})
    if cuts[0] < 0 or cuts[-1] > end or end <= 0:
        raise ValueError('invalid observation interval')
    segments = []
    for index, time in enumerate(cuts):
        for page, kind, delta in events.get(time, []):
            refs[page][kind] += delta
            if refs[page][kind] < 0:
                raise ValueError('reference underflow')
        if index + 1 == len(cuts):
            break
        total = 0
        for page, owners in refs.items():
            if mode == 'request_sum':
                total += sizes[page] * owners[0]
            elif mode == 'ignore_reader':
                total += sizes[page] * int(owners[0] + owners[1] + owners[3] > 0)
            elif mode == 'physical':
                total += sizes[page] * int(sum(owners) > 0)
            else:
                raise ValueError('unknown accounting mode')
        segments.append((time, cuts[index + 1], total))
    integral = sum((right - left) * size for left, right, size in segments)
    peak = max((size for left, right, size in segments), default=0)
    return segments, peak, integral


def owner_interval_reference(sizes, holds, cuts):
    # Per-object union of independently supplied ownership intervals.
    result = []
    for left, right in zip(cuts, cuts[1:]):
        midpoint = (left + right) / 2
        live = {page for page, intervals in holds.items()
                if any(a <= midpoint < b for a, b in intervals)}
        result.append((left, right, sum(sizes[page] for page in live)))
    return result


def lru_misses(sequence, capacity):
    if not nonnegative_integer(capacity):
        raise ValueError('invalid capacity')
    cache, misses = [], 0
    for block in sequence:
        if block in cache:
            cache.remove(block)
        else:
            misses += 1
        cache.insert(0, block)
        if len(cache) > capacity:
            cache.pop()
    return misses


def stack_misses(sequence, capacity):
    if not nonnegative_integer(capacity):
        raise ValueError('invalid capacity')
    misses = 0
    for index, block in enumerate(sequence):
        previous = index - 1
        while previous >= 0 and sequence[previous] != block:
            previous -= 1
        if previous < 0 or len(set(sequence[previous + 1:index])) >= capacity:
            misses += 1
    return misses


def vision_formula(height, width):
    if not all(nonnegative_integer(v) and v > 0 and v % 28 == 0
               for v in (height, width)):
        raise ValueError('processed dimensions must be positive multiples of 28')
    rows, cols = height // 14, width // 14
    window_pairs = 0
    for row in range(0, rows, 8):
        for col in range(0, cols, 8):
            n = min(8, rows - row) * min(8, cols - col)
            window_pairs += n * n
    return rows * cols, (height // 28) * (width // 28), window_pairs


def vision_coordinate_reference(height, width):
    groups, windows = set(), Counter()
    for row in range(height // 14):
        for col in range(width // 14):
            groups.add((row // 2, col // 2))
            windows[(row // 8, col // 8)] += 1
    return len(groups), sum(count * count for count in windows.values())


def covariance(records):
    mx = sum(x for x, y in records) / len(records)
    my = sum(y for x, y in records) / len(records)
    return sum((x - mx) * (y - my) for x, y in records) / len(records)


def main():
    raw = [
        {'id': 'g', 'unit': 'g', 'x': 2, 'y': 1, 'artifact': 'M', 'deps': []},
        {'id': 'bad', 'unit': 'bad', 'x': None, 'y': 1, 'artifact': 'M', 'deps': []},
        {'id': 'p', 'unit': 'session', 'x': 2, 'y': 1, 'artifact': 'wrong', 'deps': []},
        {'id': 'c', 'unit': 'session', 'x': 3, 'y': 1, 'artifact': 'M', 'deps': [('p', 3)]},
    ]
    accepted, rejected, blocked, traces = canonical_units(raw)
    require([r['id'] for r in accepted] == ['g'], 'only independent legal record enters output')
    require(rejected == {'bad', 'p'} and blocked == {'c'}, 'reject and closure classification')
    require(traces['bad'] == ['RAW', 'REJECTED'], 'reject is terminal')
    require(traces['c'] == ['RAW', 'VALIDATED', 'BLOCKED_DEPENDENCY'], 'dependent blocks')
    expected_ids = {'g'}  # Hand-classified fixture; no candidate state consulted.
    faulty_fallthrough_ids = {r['id'] for r in raw}
    require(faulty_fallthrough_ids != expected_ids, 'rejected fallthrough control detected')
    require(not ({r['id'] for r in accepted} & (rejected | blocked)), 'disjoint state invariant')
    require(canonical_units([])[0] == [], 'empty input')
    perturbed = dict(raw[0], id='copy-g', unit='copy-g', x=-1)
    pa, pr, pb, pt = canonical_units([perturbed])
    require(pa == [] and pr == {'copy-g'} and pt['copy-g'][-1] == 'REJECTED',
            'invalid perturbation cannot enter output')
    orphan = dict(raw[3], unit='orphan')
    require(canonical_units([orphan])[2] == {'c'}, 'missing ancestor blocks unit')
    expect_value_error(lambda: canonical_units([raw[0], raw[0]]), 'duplicate raw identity')

    nodes = [
        {'id': 't2', 'deps': [('t1', 3)], 'offset': 0, 'duration': 2},
        {'id': 't1', 'deps': [], 'offset': 0, 'duration': 5},
    ]
    c0 = replay_unit(nodes, 'C0', 100)
    c1 = replay_unit(nodes, 'C1', 101)
    require([(r['start'], r['end']) for r in c0] == [(100, 105), (108, 110)],
            'independent hand timeline C0')
    require([(r['start'], r['end']) for r in c1] == [(101, 106), (109, 111)],
            'independent hand timeline C1')
    require(replay_invariant(c0 + c1), 'copies have unique identities and local edges')
    require(not replay_invariant(c0 + c0), 'duplicate copy identity detected')
    require(not replay_invariant(replay_unit(nodes, 'fault', 100, True)),
            'premature dependent release control detected')
    expect_value_error(lambda: replay_unit([nodes[0]], 'orphan', 100), 'isolated second turn')
    slow = [dict(node, duration=12) if node['id'] == 't1' else dict(node) for node in nodes]
    require(replay_unit(slow, 'slow', 100)[1]['start'] == 115, 'completion feedback shifts child')
    join = [
        {'id': 'j', 'deps': [('a', 1), ('b', 3)], 'offset': 0, 'duration': 1},
        {'id': 'a', 'deps': [], 'offset': 0, 'duration': 2},
        {'id': 'b', 'deps': [], 'offset': 0, 'duration': 4},
    ]
    jr = replay_unit(join, 'join', 0)
    require(jr[-1]['start'] == 7 and replay_invariant(jr), 'join waits for both parents')
    cycle = [dict(nodes[1], deps=[('t2', 0)]), nodes[0]]
    expect_value_error(lambda: replay_unit(cycle, 'cycle', 0), 'cycle')
    require(replay_unit([], 'empty', 0) == [], 'empty replay')

    cases = 0
    for x in range(1, 9):
        for h in range(x):
            for y in range(7):
                prompt, decode, consumed = event_pair_reference(x, h, y)
                require(prompt == prefill_formula(x, h), 'prefill legal-edge reference')
                require(decode == decode_formula(x, y), 'decode event reference')
                require(consumed == x + max(y - 1, 0), 'selected versus consumed frontier')
                cases += 1
    require(decode_formula(10, 0) == 0 and decode_formula(10, 1) == 0,
            'zero and first-token boundaries')
    require(decode_formula(10, 3) == 23, 'multi-token hand result')
    require(old_decode_control(10, 1) != decode_formula(10, 1), 'first-token double charge detected')
    for proposals, cap, expected in [
        (['EOS', 'unused'], 5, ['EOS']),
        (['a', 'b', 'EOS', 'unused'], 5, ['a', 'b', 'EOS']),
        (['a', 'b', 'EOS'], 1, ['a']),
        (['a'], 0, []),
    ]:
        selected = select_until_stop(proposals, cap)
        require(selected == expected, 'EOS and cap selection reference')
        y = len(selected)
        require(event_pair_reference(10, 0, y)[1] == decode_formula(10, y),
                'EOS included in selected count without consuming terminal token')
    require(decode_formula(10, 3) + 13 == 36, 'explicit final-token state supplement')
    require(decode_formula(10, 3) + 13 != decode_formula(10, 3),
            'unrequested terminal consumption control detected')
    expect_value_error(lambda: prefill_formula(3, 3), 'KV-only missing-logits boundary')

    sizes = {'P@0': 4096, 'A@0': 4096, 'B@0': 4096}
    events = {
        0: [('P@0', 0, 2), ('P@0', 1, 1), ('A@0', 0, 1),
            ('A@0', 2, 1), ('B@0', 0, 1)],
        1: [('P@0', 0, -2), ('A@0', 0, -1), ('B@0', 0, -1)],
        2: [('A@0', 2, -1)],
        4: [('P@0', 1, -1)],
    }
    holds = {
        'P@0': [(0, 1), (0, 1), (0, 4)],
        'A@0': [(0, 1), (0, 2)],
        'B@0': [(0, 1)],
    }
    cuts = [0, 1, 2, 4, 5]
    segments, peak, integral = physical_sweep(sizes, events, 5)
    require(segments == owner_interval_reference(sizes, holds, cuts), 'owner-interval union reference')
    require([n // 1024 for a, b, n in segments] == [12, 8, 4, 0], 'hand physical trajectory')
    require(peak == 12 * 1024 and integral == 28 * 1024, 'peak and byte-second integral')
    require(isclose(integral / 5 / 1024, 5.6), 'time-average capacity')
    rs, rp, ri = physical_sweep(sizes, events, 5, 'request_sum')
    require(rp == 16 * 1024 and rs[1][2] == 0 and ri != integral,
            'request-sum double-count/retention negative control detected')
    ir, ip, ii = physical_sweep(sizes, events, 5, 'ignore_reader')
    require(ir[1][2] == 4 * 1024 and ii != integral, 'last-reader omission detected')
    empty_segments, empty_peak, empty_integral = physical_sweep({}, {}, 5)
    require(empty_peak == 0 and empty_integral == 0, 'empty physical ledger')
    expect_value_error(lambda: physical_sweep({'p': 4}, {0: [('p', 0, -1)]}, 1),
                       'reference underflow')
    pool = 5 * 4096
    require((pool - peak) + peak == pool and pool + peak != pool,
            'pool decomposition is not additive with its live subset')

    comparisons = 0
    for length in range(7):
        for sequence in product('ABC', repeat=length):
            for capacity in range(5):
                require(lru_misses(sequence, capacity) == stack_misses(sequence, capacity),
                        'independent LRU/stack-distance comparison')
                comparisons += 1
    burst, interleaved = tuple('AABBCCDDEEFF'), tuple('ABCDEFABCDEF')
    require(Counter(burst) == Counter(interleaved), 'same access multiset')
    require(lru_misses(burst, 3) == 6 and lru_misses(interleaved, 3) == 12,
            'reuse distance hand counterexample')
    require(lru_misses(interleaved, 6) == 6, 'capacity threshold')
    faulty_seen_hits = len(interleaved) - len(set(interleaved))
    require(faulty_seen_hits != len(interleaved) - lru_misses(interleaved, 3),
            'ever-seen hit control detected')
    expect_value_error(lambda: lru_misses([], -1), 'negative cache capacity')

    for h, w in [(28, 28), (56, 28), (56, 224), (112, 112), (448, 672)]:
        patches, merged, wpairs = vision_formula(h, w)
        ref_merged, ref_wpairs = vision_coordinate_reference(h, w)
        require((merged, wpairs) == (ref_merged, ref_wpairs), 'coordinate merge/window reference')
        require(patches == 4 * merged, 'spatial merge conservation')
    require(vision_formula(448, 672) == (1536, 384, 98304), 'declared image geometry')
    require(vision_formula(56, 224) == (64, 16, 2048), 'wide edge windows')
    require(vision_formula(112, 112) == (64, 16, 4096), 'square full window')
    require(vision_formula(448, 672)[0] != vision_coordinate_reference(448, 672)[0],
            'raw-patch-as-decoder-token control detected')
    expect_value_error(lambda: vision_formula(30, 28), 'invalid processed dimension')
    expect_value_error(lambda: vision_formula(0, 28), 'empty image')
    kv_bytes = 2 * 28 * 4 * 128 * 2
    require(kv_bytes == 57344 and 896 * kv_bytes == 49 * 1024 ** 2,
            'input KV hand budget')
    require(1152 * kv_bytes == 63 * 1024 ** 2, 'reservation bound')
    require(896 + 256 - 1 == 1151, 'reservation differs from consumed frontier')

    w0, w1 = [(100, 100), (100, 100)], [(10, 190), (190, 10)]
    require(sum(x for x, y in w0) == sum(x for x, y in w1) == 200, 'same input mean')
    require(sum(y for x, y in w0) == sum(y for x, y in w1) == 200, 'same output mean')
    require(covariance(w1) == -8100, 'joint covariance')
    require(2 * max(x for x, y in w1) == 380, 'padded linear work')
    require(2 * max(x for x, y in w1) ** 2 == 72200, 'padded square work')
    require(sum(x * x for x, y in w1) == 36200, 'effective square work')
    j0, j1 = [(1, 2), (2, 3), (3, 1)], [(1, 3), (2, 1), (3, 2)]
    require(Counter(x for x, y in j0) == Counter(x for x, y in j1), 'equal x marginal')
    require(Counter(y for x, y in j0) == Counter(y for x, y in j1), 'equal y marginal')
    require(isclose(covariance(j0), covariance(j1)) and Counter(j0) != Counter(j1),
            'moment-only joint-equality control detected')
    outcomes = [(True, True, True, True), (True, True, False, True),
                (True, False, True, True), (False, False, False, False)]
    valid_ids = [i for i, flags in enumerate(outcomes) if all(flags)]
    require(valid_ids == [0], 'independent joint-SLO fixture')
    goodput = len(valid_ids) / 10
    completed_rate = sum(flags[0] for flags in outcomes) / 10
    require(isclose(goodput, 0.1) and isclose(completed_rate, 0.3), 'goodput hand values')
    require(not isclose(goodput, completed_rate), 'completion-only goodput control detected')

    print('scope: standalone CPU logic models; no tokenizer, processor, model forward or GPU execution')
    print('reject: invalid record stays REJECTED; dependent unit blocks; invalid perturbation excluded')
    print('replay: C0=100..105,108..110 ms; C1=101..106,109..111 ms; slow parent shifts child to 115 ms')
    print('decode: independent legal-edge cases=%d; x=10 y=0/1/3 gives 0/0/23 pairs' % cases)
    print('physical: 12/8/4/0 KiB; peak=12 KiB; integral=28 KiB*s; average=5.6 KiB')
    print('cache: independent LRU/stack-distance comparisons=%d; burst/interleaved misses=6/12 at C=3' % comparisons)
    print('vision: 1536 patches/384 decoder tokens; equal-area window pairs=2048 versus 4096')
    print('joint: equal low-order moments do not prove joint equality; goodput=0.1 versus completion rate=0.3')
    print('negative controls detected: reject fallthrough, early/orphan turn, duplicate identity, first-token charge, terminal consumption, request-sum capacity, omitted reader, ever-seen hit, raw patches, moment-only equality, completion-only goodput')
    print('all scoped mechanism checks passed')


if __name__ == '__main__':
    main()
