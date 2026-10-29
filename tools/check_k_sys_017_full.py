#!/usr/bin/env python3
"""K-SYS-017: deterministic, read-only, offline CPU logic checks.
Inputs are embedded below. No project/upstream imports or external I/O.
"""
import heapq
import itertools
import math
from dataclasses import dataclass
from fractions import Fraction


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_error(fn, error=ValueError):
    try:
        fn()
    except error:
        return
    raise AssertionError('expected ' + error.__name__)


def dot(a, b):
    if len(a) != len(b):
        raise ValueError('dimension mismatch')
    return sum((x * y for x, y in zip(a, b)), Fraction(0))


def distance_key(q, x, metric):
    if len(q) != len(x):
        raise ValueError('dimension mismatch')
    if metric == 'ip':
        return -dot(q, x)
    return sum(((a - b) ** 2 for a, b in zip(q, x)), Fraction(0))


def validate_query(q, k, metric):
    if type(k) is not int or k < 0:
        raise ValueError('k must be a nonnegative integer')
    if metric not in ('ip', 'l2') or not q:
        raise ValueError('query contract')
    if not all(math.isfinite(x) for x in q):
        raise ValueError('nonfinite query')


@dataclass(frozen=True)
class Row:
    loc: str
    ident: str
    ver: int
    commit: int
    tomb: bool
    vector: object
    material: str
    allowed: bool


def row(loc, ident, ver, value=None, tomb=False, commit=None,
        material=None, allowed=True):
    vector = None if tomb else tuple(Fraction(str(x)) for x in value)
    if material is None:
        material = '' if tomb else ident + '@' + str(ver)
    return Row(loc, ident, ver, ver if commit is None else commit,
               tomb, vector, material, allowed)


def fingerprint(r):
    return (r.commit, r.tomb, r.vector, r.material, r.allowed)


def build_visible(routes, cut):
    """Streaming metadata resolution, before any score or quota."""
    versions = {}
    latest = {}
    locations = set()
    for records in routes.values():
        for r in records:
            if r.loc in locations:
                raise ValueError('duplicate physical identity')
            locations.add(r.loc)
            if r.commit > cut:
                continue
            if r.ver < 0 or r.commit < 0:
                raise ValueError('invalid version or commit')
            if r.tomb:
                if r.vector is not None:
                    raise ValueError('tombstone vector')
            elif not r.vector or not all(math.isfinite(x) for x in r.vector):
                raise ValueError('invalid live vector')
            key = (r.ident, r.ver)
            previous = versions.get(key)
            if previous is not None and fingerprint(previous) != fingerprint(r):
                raise ValueError('same-version conflict')
            if previous is None or r.loc < previous.loc:
                versions[key] = r
            old = latest.get(r.ident)
            if old is None or r.ver > old:
                latest[r.ident] = r.ver
    return {ident: versions[(ident, ver)] for ident, ver in latest.items()}


class Candidate:
    """Reversed order makes heap[0] the worst retained candidate."""
    def __init__(self, r, penalty):
        self.row = r
        self.penalty = penalty
        self.key = (penalty, r.ident)

    def __lt__(self, other):
        return self.key > other.key


def scan_route(records, visible, q, k, allowed, metric):
    validate_query(q, k, metric)
    if k == 0:
        return []
    heap = []
    for r in records:
        current = visible.get(r.ident)
        if current is None or r != current or current.tomb:
            continue
        if not current.allowed or (allowed is not None and r.ident not in allowed):
            continue
        candidate = Candidate(r, distance_key(q, r.vector, metric))
        if len(heap) < k:
            heapq.heappush(heap, candidate)
        elif candidate.key < heap[0].key:
            heapq.heapreplace(heap, candidate)
    return sorted(heap, key=lambda c: c.key)


def merge(receipts, expected, visible, k):
    if set(receipts) != set(expected):
        raise ValueError('missing or unexpected route receipt')
    candidates = []
    seen = set()
    for records in receipts.values():
        for c in records:
            r = c.row
            if r != visible.get(r.ident) or r.tomb:
                raise ValueError('stale or non-owner result')
            if r.ident in seen:
                raise ValueError('duplicate logical result')
            seen.add(r.ident)
            candidates.append(c)
    ordered = sorted(candidates, key=lambda c: c.key)[:k]
    return [(c.row.ident, c.row.ver, c.penalty) for c in ordered]


def search(routes, cut, q, k, allowed=None, metric='ip'):
    validate_query(q, k, metric)
    visible = build_visible(routes, cut)
    receipts = {
        name: scan_route(records, visible, q, k, allowed, metric)
        for name, records in routes.items()
    }
    return merge(receipts, set(routes), visible, k)


def reference(routes, cut, q, k, allowed=None, metric='ip'):
    """Independent group-all-records, resolve-all-versions, full-sort oracle.
    Does not call build_visible, scan_route, merge, fingerprint or distance_key.
    Fractions preserve the exact finite arithmetic of the embedded inputs.
    """
    if type(k) is not int or k < 0:
        raise ValueError('reference invalid k')
    if metric not in ('ip', 'l2') or not q:
        raise ValueError('reference query contract')
    groups = {}
    for records in routes.values():
        for r in records:
            if r.commit <= cut:
                groups.setdefault(r.ident, {}).setdefault(r.ver, []).append(r)
    answer = []
    for ident in sorted(groups):
        by_version = groups[ident]
        for copies in by_version.values():
            first = copies[0]
            for other in copies[1:]:
                if (other.commit != first.commit or other.tomb != first.tomb
                        or other.vector != first.vector
                        or other.material != first.material
                        or other.allowed != first.allowed):
                    raise ValueError('reference same-version conflict')
        newest = by_version[max(by_version)][0]
        if newest.tomb or not newest.allowed:
            continue
        if allowed is not None and ident not in allowed:
            continue
        if len(newest.vector) != len(q):
            raise ValueError('reference dimension mismatch')
        total = Fraction(0)
        for j in range(len(q)):
            if metric == 'ip':
                total -= q[j] * newest.vector[j]
            else:
                total += (q[j] - newest.vector[j]) ** 2
        answer.append((ident, newest.ver, total))
    answer.sort(key=lambda x: (x[2], x[0]))
    return answer[:k]


def compare(actual, expected, label):
    require(actual == expected,
            label + ': actual=' + repr(actual) + ', expected=' + repr(expected))


def detect_wrong_result(fn, expected, label):
    """The faulty path must produce a result rejected by the normal comparator."""
    actual = fn()
    try:
        compare(actual, expected, label)
    except AssertionError:
        print('negative detected:', label, 'actual=', actual, 'reference=', expected)
        return
    raise AssertionError('negative control escaped: ' + label)


def bad_truncate_then_override(routes, cut, q, k):
    """Even a correct final authority check cannot refill a lost local slot."""
    visible = build_visible(routes, cut)
    truncated = []
    for records in routes.values():
        raw = [r for r in records if r.commit <= cut and not r.tomb]
        raw.sort(key=lambda r: (-dot(q, r.vector), r.ident, r.loc))
        truncated.extend(raw[:k])
    survivors = [r for r in truncated if r == visible.get(r.ident) and not r.tomb]
    survivors.sort(key=lambda r: (-dot(q, r.vector), r.ident))
    return [(r.ident, r.ver, -dot(q, r.vector)) for r in survivors[:k]]


def bad_any_tombstone_wins(routes, cut, q, k):
    answer = reference(routes, cut, q, k)
    deleted_ever = {r.ident for records in routes.values() for r in records
                    if r.commit <= cut and r.tomb}
    return [item for item in answer if item[0] not in deleted_ever]


def bad_last_write_accepts_conflict(routes, cut):
    latest = {}
    for records in routes.values():
        for r in records:
            if r.commit <= cut:
                old = latest.get(r.ident)
                if old is None or r.ver >= old.ver:
                    latest[r.ident] = r
    return latest


def check_version_trajectories():
    q = (Fraction(1),)
    base = [row('main/0', 'a', 1, [1]), row('main/1', 'b', 1, ['0.9'])]
    update = {'main': base, 'delta': [row('delta/0', 'a', 2, ['0.1'])]}
    deletion = {'main': base, 'delta': [row('delta/0', 'a', 2, tomb=True)]}
    expected = [('b', 1, -Fraction(9, 10))]
    for label, routes in [('downgrade', update), ('deletion', deletion)]:
        oracle = reference(routes, 2, q, 1)
        compare(oracle, expected, label + ' hand result')
        compare(search(routes, 2, q, 1), oracle, label + ' corrected search')
        detect_wrong_result(lambda r=routes: bad_truncate_then_override(r, 2, q, 1),
                            oracle, label + ' truncate-before-visibility')
    compare(search(update, 1, q, 1), [('a', 1, Fraction(-1))], 'old snapshot')
    reinsert = {
        'main': [row('main/0', 'a', 1, tomb=True)],
        'delta': [row('delta/0', 'a', 2, ['0.8'])]
    }
    compare(search(reinsert, 1, q, 1), [], 'snapshot before reinsert')
    oracle = reference(reinsert, 2, q, 1)
    compare(oracle, [('a', 2, -Fraction(4, 5))], 'reinsert hand result')
    compare(search(reinsert, 2, q, 1), oracle, 'authorized newer live version')
    detect_wrong_result(lambda: bad_any_tombstone_wins(reinsert, 2, q, 1),
                        oracle, 'old tombstone cannot suppress newer live version')
    denied = {
        'main': base,
        'delta': [row('delta/0', 'a', 2, ['0.1'], allowed=False)]
    }
    compare(search(denied, 2, q, 2), reference(denied, 2, q, 2), 'latest denied')
    compare(search(denied, 2, q, 2), expected, 'no fallback to older allowed a')
    print('versions: downgrade, deletion, reinsertion, snapshot cut and latest metadata checked')


def check_conflicts_and_replicas():
    q = (Fraction(1),)
    first = row('main/0', 'a', 1, [1])
    conflicts = [
        row('delta/0', 'a', 1, ['0.5']),
        row('delta/0', 'a', 1, tomb=True),
        row('delta/0', 'a', 1, [1], material='different-material'),
        row('delta/0', 'a', 1, [1], allowed=False),
        row('delta/0', 'a', 1, [1], commit=2)
    ]
    checked = 0
    for conflict in conflicts:
        for ordered in [(first, conflict), (conflict, first)]:
            routes = {'one': [ordered[0]], 'two': [ordered[1]]}
            expect_error(lambda: reference(routes, 2, q, 1))
            expect_error(lambda: search(routes, 2, q, 1))
            accepted = bad_last_write_accepts_conflict(routes, 2)
            require('a' in accepted, 'faulty overwrite did not exercise acceptance')
            checked += 1
    shadowed = {'main': [first, conflicts[0]],
                'delta': [row('delta/1', 'a', 2, [2])]}
    expect_error(lambda: reference(shadowed, 2, q, 1))
    expect_error(lambda: search(shadowed, 2, q, 1))
    replicas = {
        'main': [first, row('main/1', 'b', 1, [1])],
        'delta': [row('delta/0', 'a', 1, [1])]
    }
    compare(search(replicas, 1, q, 3), reference(replicas, 1, q, 3), 'identical replicas')
    compare(search(replicas, 1, q, 3),
            [('a', 1, Fraction(-1)), ('b', 1, Fraction(-1))], 'ID tie and dedup')
    reversed_routes = {name: list(reversed(records))
                       for name, records in reversed(list(replicas.items()))}
    compare(search(reversed_routes, 1, q, 3), search(replicas, 1, q, 3), 'permutation')
    print('conflicts:', checked, 'ordered conflicts rejected; faulty overwrite accepts them')
    print('replicas: identical copies accepted once; ID ties and permutations checked')


def make_state(ident, state, routes):
    m = 'main/' + ident
    d = 'delta/' + ident
    if state == 0:
        return
    if state == 1:
        routes['main'].append(row(m, ident, 1, [1]))
    elif state == 2:
        routes['main'].append(row(m, ident, 1, [1]))
        routes['delta'].append(row(d, ident, 2, ['0.1']))
    elif state == 3:
        routes['main'].append(row(m, ident, 1, [1]))
        routes['delta'].append(row(d, ident, 2, tomb=True))
    elif state == 4:
        routes['main'].append(row(m, ident, 1, tomb=True))
        routes['delta'].append(row(d, ident, 2, ['0.5']))
    elif state == 5:
        routes['main'].append(row(m, ident, 1, [1]))
        routes['delta'].append(row(d, ident, 1, [1]))
    elif state == 6:
        routes['main'].append(row(m, ident, 1, ['0.5']))
        routes['delta'].append(row(d, ident, 3, [2]))


def check_finite_state_combinations():
    q = (Fraction(1),)
    comparisons = 0
    for states in itertools.product(range(7), repeat=3):
        routes = {'main': [], 'delta': []}
        for ident, state in zip(('a', 'b', 'c'), states):
            make_state(ident, state, routes)
        for cut in (0, 1, 2, 3):
            for k in range(5):
                for metric in ('ip', 'l2'):
                    for allowed in (None, {'b'}):
                        compare(search(routes, cut, q, k, allowed, metric),
                                reference(routes, cut, q, k, allowed, metric),
                                'finite-state ' + repr((states, cut, k, metric, allowed)))
                        comparisons += 1
    require(comparisons == 27440, 'finite-state enumeration count')
    print('finite-state:', comparisons,
          'exact comparisons; seven per-ID states, three IDs, cuts, k, metrics and filters')


def check_query_and_merge_boundaries():
    q = (Fraction(1),)
    empty = {'main': [], 'delta': []}
    compare(search(empty, 0, q, 9), [], 'empty library')
    routes = {'main': [row('main/a', 'a', 1, [1])], 'delta': []}
    compare(search(routes, 1, q, 0), [], 'k zero')
    compare(search(routes, 1, q, 9, set()), [], 'empty legal set')
    expect_error(lambda: search(routes, 1, q, -1))
    expect_error(lambda: search(routes, 1, (Fraction(1), Fraction(2)), 1))
    expect_error(lambda: search(routes, 1, (float('nan'),), 1))
    visible = build_visible(routes, 1)
    receipt = scan_route(routes['main'], visible, q, 1, None, 'ip')
    expect_error(lambda: merge({'main': receipt}, {'main', 'delta'}, visible, 1))
    stale = Candidate(row('other/a', 'a', 0, [5]), Fraction(-5))
    expect_error(lambda: merge({'main': [stale], 'delta': []}, set(routes), visible, 1))
    expect_error(lambda: merge({'main': receipt, 'delta': receipt}, set(routes), visible, 1))
    print('boundaries: empty, k=0, negative k, dimensions, NaN and receipt violations checked')


def check_distance_filter_and_norm():
    q = (Fraction(1), Fraction(0))
    vectors = [(Fraction(1), Fraction(0)),
               (Fraction(4, 5), Fraction(3, 5)),
               (Fraction(0), Fraction(1))]
    routes = {'main': [row('main/' + ident, ident, 1, vector)
                       for ident, vector in zip(('a', 'b', 'c'), vectors)], 'delta': []}
    expected = reference(routes, 1, q, 1, {'b', 'c'})
    compare(search(routes, 1, q, 1, {'b', 'c'}), expected, 'legal distance top1')
    post = [item for item in reference(routes, 1, q, 1) if item[0] in {'b', 'c'}]
    detect_wrong_result(lambda: post, expected, 'fixed top-k then permission filter')
    for vector in vectors:
        require(distance_key(q, vector, 'l2') == 2 - 2 * dot(q, vector),
                'unit-vector IP/L2 identity')
    q2 = (Fraction(1), Fraction(1))
    a, b = (Fraction(10), Fraction(0)), (Fraction(1), Fraction(1))
    def cosine(v):
        return float(dot(q2, v)) / math.sqrt(float(dot(q2, q2) * dot(v, v)))
    require(dot(q2, a) > dot(q2, b) and cosine(a) < cosine(b), 'norm counterexample')
    print('distance: exact unit identity and unnormalized IP/cosine reversal checked')


def check_shard_budget():
    q = (Fraction(1),)
    routes = {
        's1': [row('s1/a', 'a', 1, ['0.99']), row('s1/b', 'b', 1, ['0.98'])],
        's2': [row('s2/c', 'c', 1, ['0.97']), row('s2/d', 'd', 1, ['0.10'])]
    }
    oracle = reference(routes, 1, q, 2)
    compare(search(routes, 1, q, 2), oracle, 'local k merges exactly')
    visible = build_visible(routes, 1)
    short = {name: scan_route(records, visible, q, 1, None, 'ip')
             for name, records in routes.items()}
    detect_wrong_result(lambda: merge(short, set(routes), visible, 2),
                        oracle, 'local budget 1 for global k=2')


class VersionStore:
    def __init__(self):
        self.data = {}
        self.current = None
        self.readers = {}

    def publish(self, version, payload):
        if version in self.data:
            raise ValueError('duplicate version')
        self.data[version] = payload
        self.current = version

    def pin(self, version):
        if version not in self.data:
            raise KeyError(version)
        self.readers[version] = self.readers.get(version, 0) + 1

    def release(self, version):
        if self.readers.get(version, 0) <= 0:
            raise ValueError('no reader')
        self.readers[version] -= 1

    def gc(self):
        for version in list(self.data):
            if version != self.current and self.readers.get(version, 0) == 0:
                del self.data[version]


def check_snapshot():
    store = VersionStore()
    store.publish('v1', 1)
    store.pin('v1')
    store.publish('v2', 2)
    store.gc()
    require(store.data['v1'] == 1, 'pinned snapshot was collected')
    store.release('v1')
    store.gc()
    require('v1' not in store.data, 'unreferenced old snapshot not collected')
    bad = VersionStore()
    bad.publish('v1', 1)
    bad.pin('v1')
    bad.publish('v2', 2)
    for version in list(bad.data):
        if version != bad.current:
            del bad.data[version]
    try:
        require(bad.data['v1'] == 1, 'bad old value')
    except KeyError:
        print('negative detected: naive GC removes pinned snapshot')
    else:
        raise AssertionError('naive GC escaped read invariant')
    print('snapshot: sequential read protection and last-release collection checked')


def check_cost_and_timing():
    n, d, lists, m, ksub = 1000000, 768, 4096, 96, 256
    require(n * d * 4 == 3072000000, 'Flat vector bytes')
    count16 = Fraction(lists) + Fraction(16 * n, lists)
    count32 = Fraction(lists) + Fraction(32 * n, lists)
    require(count16 == Fraction(32009, 4), 'uniform IVF p16')
    require(count32 == Fraction(47634, 4), 'uniform IVF p32')
    centers, sizes = (0, 100, 101, 102), (10, 20, 30, 40)
    chosen = sorted(range(4), key=lambda i: (abs(centers[i]), i))[:2]
    require(sum(sizes[i] for i in chosen) == 30, 'skew list scan')
    codebook = m * ksub * (d // m) * 4
    centroid_bytes = lists * d * 4
    total = n * m + n * 8 + codebook + centroid_bytes
    require(codebook == 786432 and centroid_bytes == 12582912, 'PQ components')
    require(total == 117369344, 'PQ capacity total')
    stages = [5, 8, 12, 30, 2, 80, 5]
    require(sum(stages) == 142, 'serial timing')
    faster = stages[:]
    faster[1] = 4
    require(sum(stages) - sum(faster) == 4, 'search-only improvement')
    require(sum(stages) + 10 == 152, 'queue timing')
    recall = Fraction(len({0, 2, 5} & {0, 1, 5}), len({0, 1, 5}))
    require(recall == Fraction(2, 3), 'ANN recall arithmetic')
    print('cost: Flat 3072000000 B; listed IVF/PQ components 117369344 B')
    print('IVF uniform logical counts:', count16, count32, '; no index execution')
    print('timing: serial 142 ms; queue gives 152 ms; search improvement saves 4 ms')


def main():
    check_version_trajectories()
    check_conflicts_and_replicas()
    check_finite_state_combinations()
    check_query_and_merge_boundaries()
    check_distance_filter_and_norm()
    check_shard_budget()
    check_snapshot()
    check_cost_and_timing()
    print('scope: exact-rational CPU record/selection models and fixed arithmetic only')
    print('not measured: real ANN/PQ/graph, model quality, concurrency, GPU, I/O or service latency')
    print('all embedded assertions completed')


if __name__ == '__main__':
    main()
