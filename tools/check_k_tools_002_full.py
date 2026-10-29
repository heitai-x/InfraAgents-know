#!/usr/bin/env python3
'''Read-only, offline finite-model checks for K-TOOLS-002; no device measurements.'''
from collections import deque
from fractions import Fraction


class CheckFailure(Exception):
    pass


def require(condition, message):
    if not condition:
        raise CheckFailure(message)


def number(value):
    return isinstance(value, (int, Fraction)) and not isinstance(value, bool)


def identity(value):
    require(value is not None, 'missing identity')
    try:
        hash(value)
    except TypeError:
        raise CheckFailure('unhashable identity') from None


def rejected(action, label):
    try:
        action()
    except CheckFailure:
        return
    raise CheckFailure('invalid input accepted: ' + label)


class LCG:
    def __init__(self, seed):
        self.state = seed & 0xffffffff

    def pick(self, n):
        self.state = (1664525 * self.state + 1013904223) & 0xffffffff
        return self.state % n


def interval_ledger(raw, window):
    w0, w1 = window
    require(number(w0) and number(w1) and w0 <= w1, 'invalid window')
    seen, work = {}, []
    points = 0
    for s, e, pid in raw:
        identity(pid)
        require(number(s) and number(e) and s <= e, 'invalid raw interval')
        if pid in seen:
            require(seen[pid] == (s, e), 'conflicting physical identity')
            continue
        seen[pid] = (s, e)
        # Input contract: equal endpoints represent schema-confirmed point events.
        if s == e:
            points += int(w0 <= s < w1)
            continue
        a, b = max(s, w0), min(e, w1)
        if a < b:
            work.append((a, b))
    total = sum(b - a for a, b in work)
    merged = []
    for a, b in sorted(work):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    events = sorted([(a, 1) for a, b in work] + [(b, -1) for a, b in work])
    active = peak = 0
    for _, delta in events:
        active += delta
        require(active >= 0, 'negative active count')
        peak = max(peak, active)
    require(active == 0, 'activity did not drain')
    covered = sum(b - a for a, b in merged)
    require(covered <= total <= peak * covered, 'interval integral inequality')
    require(covered <= w1 - w0, 'window coverage overflow')
    return dict(sum=total, union=covered, peak=peak, points=points, merged=merged)


def reference_interval(raw, window):
    # Independent partition/integration reference, without a sweep counter or union merge.
    w0, w1 = window
    require(number(w0) and number(w1) and w0 <= w1, 'reference window')
    records = {}
    for s, e, pid in raw:
        identity(pid)
        require(number(s) and number(e) and s <= e, 'reference raw interval')
        if pid in records:
            require(records[pid] == (s, e), 'reference identity conflict')
        else:
            records[pid] = (s, e)
    cuts = {w0, w1}
    points = 0
    for s, e in records.values():
        if s == e and w0 <= s < w1:
            points += 1
        for t in (s, e):
            if w0 < t < w1:
                cuts.add(t)
    cuts = sorted(cuts)
    total = covered = peak = 0
    cells = []
    for a, b in zip(cuts, cuts[1:]):
        middle = (Fraction(a) + Fraction(b)) / 2
        count = sum(1 for s, e in records.values() if s <= middle < e)
        total += count * (b - a)
        peak = max(peak, count)
        if count:
            covered += b - a
            cells.append((a, b))
    runs = []
    for a, b in cells:
        if runs and runs[-1][1] == a:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    return dict(sum=total, union=covered, peak=peak, points=points, merged=runs)


def earliest(nodes, edges, target=None):
    pred = {v: [] for v in nodes}
    succ = {v: [] for v in nodes}
    for v, (duration, release) in nodes.items():
        identity(v)
        require(number(duration) and number(release), 'invalid node number')
        require(duration >= 0 and release >= 0, 'negative node number')
    require(target is None or target in nodes, 'missing target')
    pairs = set()
    for u, v, lag in edges:
        require(u in nodes and v in nodes, 'missing edge endpoint')
        require(number(lag) and lag >= 0, 'invalid edge lag')
        require((u, v) not in pairs, 'duplicate ordered edge')
        pairs.add((u, v))
        pred[v].append((u, lag))
        succ[u].append((v, lag))
    indeg = {v: len(pred[v]) for v in nodes}
    ready = deque(v for v in nodes if indeg[v] == 0)
    order = []
    while ready:
        u = ready.popleft()
        order.append(u)
        for v, _ in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    require(len(order) == len(nodes), 'cycle detected')
    if not nodes:
        return dict(finish=0, path=[], ES={}, EF={})
    es, ef, parent = {}, {}, {}
    for v in order:
        duration, release = nodes[v]
        best, par = release, None
        for u, lag in pred[v]:
            candidate = ef[u] + lag
            if candidate > best:
                best, par = candidate, u
        es[v], ef[v], parent[v] = best, best + duration, par
        require(es[v] >= release, 'release constraint violated')
        for u, lag in pred[v]:
            require(es[v] >= ef[u] + lag, 'predecessor constraint violated')
    end = target if target is not None else max(order, key=lambda v: ef[v])
    path, x = [], end
    while x is not None:
        path.append(x)
        x = parent[x]
    path.reverse()
    lag_by_pair = {(u, v): lag for u, v, lag in edges}
    score = nodes[path[0]][1] + sum(nodes[v][0] for v in path)
    score += sum(lag_by_pair[(u, v)] for u, v in zip(path, path[1:]))
    require(score == ef[end], 'witness path score mismatch')
    return dict(finish=ef[end], path=path, ES=es, EF=ef)


def enumerate_reference(nodes, edges, roots_only=False):
    # Every node is a legal starting point because every node has a release constraint.
    # roots_only intentionally reproduces the previous broken reference.
    succ = {v: [] for v in nodes}
    incoming = set()
    for u, v, lag in edges:
        succ[u].append((v, lag))
        incoming.add(v)
    scores = {v: 0 for v in nodes}

    def visit(v, score, route):
        require(v not in route, 'reference cycle')
        scores[v] = max(scores[v], score)
        route = route + (v,)
        for nxt, lag in succ[v]:
            visit(nxt, score + lag + nodes[nxt][0], route)

    for v, (duration, release) in nodes.items():
        if not roots_only or v not in incoming:
            visit(v, release + duration, ())
    return scores


def associate(submits, dispatches):
    index = {}
    submit_ids = set()
    for sid, key in submits:
        identity(sid)
        require(sid not in submit_ids, 'duplicate submit identity')
        submit_ids.add(sid)
        if all(part is not None for part in key):
            index.setdefault(key, []).append(sid)
    exact, unresolved = {}, {}
    dispatch_ids = set()
    for did, key in dispatches:
        identity(did)
        require(did not in dispatch_ids, 'duplicate dispatch identity')
        dispatch_ids.add(did)
        if any(part is None for part in key):
            unresolved[did] = 'missing'
            continue
        matches = index.get(key, [])
        if len(matches) == 1:
            exact[did] = matches[0]
        else:
            unresolved[did] = 'ambiguous' if matches else 'absent'
    require(set(exact).isdisjoint(unresolved), 'association partition overlap')
    require(set(exact) | set(unresolved) == dispatch_ids, 'association partition incomplete')
    return exact, unresolved


def reference_association(submits, dispatches):
    # Deliberately uses a nested join rather than the indexed implementation.
    out = {}
    for did, key in dispatches:
        if None in key:
            out[did] = ('unresolved', 'missing')
        else:
            candidates = [sid for sid, other in submits if other == key]
            if len(candidates) == 1:
                out[did] = ('exact', candidates[0])
            else:
                out[did] = ('unresolved', 'ambiguous' if candidates else 'absent')
    return out


def allocate(duration, raw_weights):
    require(number(duration) and duration >= 0, 'invalid physical duration')
    if not raw_weights:
        require(duration == 0, 'positive work without readers')
        return {}
    require(all(number(w) and w >= 0 for w in raw_weights.values()), 'invalid weights')
    total = sum(raw_weights.values())
    require(total > 0, 'zero weight sum')
    return {r: Fraction(duration) * w / total for r, w in raw_weights.items()}


def clock_bounds(a, b, delta_lo, delta_hi):
    require(all(number(v) for v in (a, b, delta_lo, delta_hi)), 'invalid clock number')
    require(delta_lo <= delta_hi, 'invalid offset interval')
    lo, hi = max(0, b - a - delta_hi), b - a - delta_lo
    return None if hi < lo else (lo, hi)


def combination(g=7, p2_release=16, p1_duration=3):
    nodes = {
        'h1': (2, 0), 'h2': (2, 1), 'batch': (1, 0), 'submit': (1, 0),
        'copy': (2, 0), 'G': (g, 0), 'P1': (p1_duration, 0),
        'P2': (3, p2_release), 'F1': (1, 0), 'F2': (1, 0),
    }
    edges = [
        ('h1', 'batch', 0), ('h2', 'batch', 0), ('batch', 'submit', 0),
        ('submit', 'copy', 0), ('copy', 'G', 0), ('G', 'P1', 0),
        ('G', 'P2', 0), ('P1', 'F1', 0), ('P2', 'F2', 0),
    ]
    result = earliest(nodes, edges)
    require(result['EF'] == enumerate_reference(nodes, edges), 'combination reference')
    return result


def pipeline(blocks, host_slots, device_slots=2):
    nodes, constraints = {}, set()
    for i in range(blocks):
        for kind, duration in (('read', 10), ('copy', 4), ('compute', 6)):
            nodes[(kind, i)] = (duration, 0)
        constraints.add((('read', i), ('copy', i), 0))
        constraints.add((('copy', i), ('compute', i), 0))
        if i:
            for kind in ('read', 'copy', 'compute'):
                constraints.add(((kind, i - 1), (kind, i), 0))
        if i >= host_slots:
            constraints.add((('copy', i - host_slots), ('read', i), 0))
        if i >= device_slots:
            constraints.add((('compute', i - device_slots), ('copy', i), 0))
    # Sorting affects repeatable traversal only; earliest() itself uses a FIFO queue.
    return earliest(nodes, sorted(constraints))['finish']


def main():
    overlap = [(0, 10, 'a'), (5, 15, 'b')]
    adjacent = [(0, 10, 'a'), (10, 20, 'b')]
    fixed = [
        ([], (0, 0)), (overlap, (0, 20)), (adjacent, (0, 20)),
        ([(0, 10, 'a'), (0, 10, 'a')], (0, 20)),
        ([(0, 10, 'a'), (0, 10, 'b')], (0, 20)),
        ([(-2, 2, 'a'), (8, 12, 'b')], (0, 10)),
        ([(0, 0, 'left'), (10, 10, 'right')], (0, 10)),
        ([(0, 3, 'a'), (1, 2, 'b')], (1, 1)),
        ([(Fraction(1, 2), Fraction(5, 2), 'a'), (1, 3, 'b')], (0, 4)),
    ]
    for raw, window in fixed:
        require(interval_ledger(raw, window) == reference_interval(raw, window), 'fixed interval')
    obs = interval_ledger(overlap, (0, 20))
    require((obs['sum'], obs['union'], obs['peak']) == (20, 15, 2), 'overlap constants')
    require(interval_ledger(adjacent, (0, 20))['peak'] == 1, 'half-open adjacency')
    require(interval_ledger(fixed[6][0], (0, 10))['points'] == 1, 'point window endpoints')
    rejected(lambda: interval_ledger([(9, 8, 'bad')], (0, 1)), 'outside reversed span')
    rejected(lambda: interval_ledger([(0, 2, 'a'), (0, 3, 'a')], (0, 5)), 'conflicting duplicate')
    rejected(lambda: interval_ledger([(0, 2, None)], (0, 5)), 'missing physical identity')
    rejected(lambda: interval_ledger([(0, float('nan'), 'a')], (0, 5)), 'nonfinite endpoint')
    rejected(lambda: interval_ledger([], (2, 1)), 'reversed window')

    rng = LCG(20261028)
    for case in range(300):
        raw = []
        for i in range(rng.pick(10)):
            s = rng.pick(13) - 4
            e = s + rng.pick(6)
            item = (s, e, (case, i))
            raw.append(item)
            if i % 4 == 0:
                raw.append(item)
        w0 = rng.pick(5) - 2
        window = (w0, w0 + rng.pick(9))
        require(interval_ledger(raw, window) == reference_interval(raw, window),
                'random interval mismatch at ' + str(case))
    faulty_union = sum(e - s for s, e, _ in overlap)
    require(faulty_union == 20 and faulty_union != obs['union'], 'sum-as-union mutant survived')
    events = sorted([(0, 1), (10, -1), (10, 1), (20, -1)], key=lambda x: (x[0], -x[1]))
    active = wrong_peak = 0
    for _, delta in events:
        active += delta
        wrong_peak = max(wrong_peak, active)
    require(wrong_peak == 2 and wrong_peak != interval_ledger(adjacent, (0, 20))['peak'],
            'start-before-end mutant survived')
    print('intervals: 9 fixed + 300 random; overlap S=20 U=15 K=2; two faulty controls detected')

    release_nodes = {0: (2, 0), 1: (3, 10)}
    release_edges = [(0, 1, 4)]
    result = earliest(release_nodes, release_edges, 1)
    corrected = enumerate_reference(release_nodes, release_edges)
    broken = enumerate_reference(release_nodes, release_edges, roots_only=True)
    require(result['finish'] == corrected[1] == 13, 'internal release regression')
    require(result['path'] == [1], 'witness must start at internal release')
    require(broken[1] == 9 and broken[1] != corrected[1], 'old reference mutant survived')
    dropped = {0: (2, 0), 1: (3, 0)}
    require(earliest(dropped, release_edges, 1)['finish'] == 9, 'drop-release mutant setup')
    require(earliest(dropped, release_edges, 1)['finish'] != corrected[1], 'drop-release mutant survived')

    join_nodes = {'a': (2, 0), 'b': (6, 0), 'c': (1, 0)}
    join_edges = [('a', 'c', 0), ('b', 'c', 0)]
    require(earliest(join_nodes, join_edges, 'c')['finish'] == 7, 'join must wait for both')
    wrong_join = min(2, 6) + 1
    require(wrong_join == 3 and wrong_join != enumerate_reference(join_nodes, join_edges)['c'],
            'minimum-predecessor mutant survived')
    tie_nodes = {'a': (4, 0), 'b': (4, 0), 'f': (0, 0)}
    tie_edges = [('a', 'f', 0), ('b', 'f', 0)]
    require(earliest(tie_nodes, tie_edges, 'f')['finish'] == 4, 'equal paths')
    tie_nodes['a'] = (1, 0)
    require(earliest(tie_nodes, tie_edges, 'f')['finish'] == 4, 'other critical path must remain')
    fractional = {'a': (Fraction(1, 3), 0), 'b': (Fraction(2, 3), 0)}
    fractional_edges = [('a', 'b', Fraction(1, 7))]
    require(earliest(fractional, fractional_edges, 'b')['finish'] == Fraction(8, 7), 'exact rational DAG')
    require(earliest({}, [])['finish'] == 0, 'empty graph')
    rejected(lambda: earliest({}, [], 'x'), 'empty target')
    rejected(lambda: earliest({}, [('x', 'y', 0)]), 'edge in empty graph')
    rejected(lambda: earliest({0: (1, 0)}, [], 9), 'missing target')
    rejected(lambda: earliest({0: (1, 0)}, [(0, 0, 0)]), 'self cycle')
    rejected(lambda: earliest({0: (1, 0), 1: (1, 0)}, [(0, 1, 0), (1, 0, 0)]), 'cycle')
    rejected(lambda: earliest(release_nodes, [(0, 1, -1)]), 'negative lag')
    rejected(lambda: earliest(release_nodes, [(0, 1, 0), (0, 1, 2)]), 'duplicate edge')
    rejected(lambda: earliest({0: (-1, 0)}, []), 'negative duration')
    rejected(lambda: earliest({0: (1, -1)}, []), 'negative release')

    rng = LCG(20261029)
    targets = 0
    for case in range(200):
        labels = [91, 7, 104, 2, 55, 0][:1 + rng.pick(6)]
        nodes = {v: (rng.pick(5), rng.pick(10)) for v in labels}
        edges = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if rng.pick(100) < 35:
                    edges.append((labels[i], labels[j], rng.pick(4)))
        reference = enumerate_reference(nodes, edges)
        got = earliest(nodes, edges)
        require(got['EF'] == reference, 'all node finishes mismatch at ' + str(case))
        require(got['finish'] == max(reference.values()), 'global finish mismatch')
        for target in labels:
            require(earliest(nodes, edges, target)['finish'] == reference[target], 'target finish mismatch')
            targets += 1
        reverse_nodes = dict(reversed(list(nodes.items())))
        reverse_edges = list(reversed(edges))
        require(earliest(reverse_nodes, reverse_edges)['EF'] == reference, 'insertion order dependence')
    require(targets == 744, 'deterministic generator drift')
    print('DAGs: 200 random, 744 targets; old roots-only oracle=9, corrected oracle=13; release/join mutants detected')

    q = ('run', 'pid', 'dev', 'queue0', 0, 1)
    submits = [('S', q)]
    dispatches = [
        ('K1', q), ('K5', q),
        ('K7', ('run', 'pid', 'dev', 'queue1', 0, 1)),
        ('K4', ('run', 'pid', 'dev', 'queue0', None, 1)),
        ('K8', ('run', 'pid', 'dev', 'queue0', 0, 2)),
    ]
    exact, unresolved = associate(submits, dispatches)
    require(exact == {'K1': 'S', 'K5': 'S'}, 'exact association')
    require(unresolved == {'K7': 'absent', 'K4': 'missing', 'K8': 'absent'}, 'unresolved reasons')
    flattened = {k: ('exact', v) for k, v in exact.items()}
    flattened.update({k: ('unresolved', v) for k, v in unresolved.items()})
    require(flattened == reference_association(submits, dispatches), 'independent association join')
    wrong = {did: 'S' for did, key in dispatches if key[4] == q[4]}
    require(wrong['K7'] == 'S' and 'K7' not in exact, 'correlation-only mutant survived')
    ambiguous = associate([('S1', q), ('S2', q)], [('K', q)])
    require(ambiguous == ({}, {'K': 'ambiguous'}), 'ambiguous submit key')
    rejected(lambda: associate(submits, [('K', q), ('K', q)]), 'duplicate dispatch identity')
    print('association: indexed and nested joins agree; queue/generation mismatch retained; faulty correlation join detected')

    allocation = allocate(7, {'R1': 1, 'R2': 1})
    require(allocation == {'R1': Fraction(7, 2), 'R2': Fraction(7, 2)}, 'equal allocation')
    weighted = allocate(9, {'R1': 1, 'R2': 2, 'R3': 0})
    require(weighted == {'R1': 3, 'R2': 6, 'R3': 0}, 'weighted allocation')
    require(sum(allocation.values()) == 7 and sum(weighted.values()) == 9, 'allocation conservation')
    wrong_shared = {reader: 7 for reader in allocation}
    require(sum(wrong_shared.values()) == 14 and sum(wrong_shared.values()) != 7,
            'full-charge-per-reader mutant survived')
    require(allocate(0, {}) == {}, 'zero physical work')
    rejected(lambda: allocate(1, {}), 'no readers')
    rejected(lambda: allocate(1, {'r': 0}), 'zero total weight')
    rejected(lambda: allocate(1, {'r': -1}), 'negative weight')
    print('shared work: exact charge conservation; full-charge-per-reader=14 versus physical=7 detected')

    clocks = 0
    for lower in range(-3, 4):
        for width in range(4):
            for observed in range(-4, 8):
                upper = lower + width
                feasible = [observed - offset for offset in range(lower, upper + 1)
                            if observed - offset >= 0]
                reference = None if not feasible else (min(feasible), max(feasible))
                require(clock_bounds(0, observed, lower, upper) == reference, 'clock bound mismatch')
                clocks += 1
    require(clock_bounds(10, 15, -2, 4) == (1, 7), 'clock example')
    require(clock_bounds(10, 15, 6, 8) is None, 'incompatible clock evidence')
    require(clock_bounds(10, 15, -2, 4) != (5, 5), 'raw-subtraction mutant survived')
    rejected(lambda: clock_bounds(0, 1, 3, 2), 'reversed offset bound')
    print('clock bounds: ' + str(clocks) + ' integer offset-enumeration cases; raw subtraction detected')

    base = combination()
    require((base['EF']['P1'], base['EF']['P2'], base['EF']['F1'], base['EF']['F2']) == (17, 19, 18, 20),
            'device versus output completion')
    device = interval_ledger([(5, 7, 'copy'), (7, 14, 'G'), (14, 17, 'P1'), (16, 19, 'P2')], (0, 20))
    require((device['sum'], device['union'], device['peak']) == (15, 14, 2), 'combination resource ledger')
    fixed_release = combination(g=5)
    moved_release = combination(g=5, p2_release=14)
    short_p1 = combination(p1_duration=1)
    require((fixed_release['EF']['F1'], fixed_release['EF']['F2']) == (16, 20), 'fixed release counterfactual')
    require((moved_release['EF']['F1'], moved_release['EF']['F2']) == (16, 18), 'moving release counterfactual')
    require((short_p1['EF']['F1'], short_p1['EF']['F2']) == (16, 20), 'P1 target counterfactual')
    require(pipeline(0, 2) == 0, 'empty pipeline')
    for blocks in range(1, 6):
        require(pipeline(blocks, 2) == 20 + (blocks - 1) * 10, 'two host slot pipeline')
        require(pipeline(blocks, 1) == 20 + (blocks - 1) * 14, 'one host slot pipeline')
    print('combination: device=17/19, output=18/20; fixed/moving release distinguished; slot pipeline formulas checked')
    print('PASS: exact finite timeline models and independent references agree; intentional faulty controls detected')
    print('Scope: Python CPU model only; no GPU, DMA, profiler, storage, RDMA, service performance or learner-effect measurement')


if __name__ == '__main__':
    main()
