from collections import deque
from fractions import Fraction
from itertools import combinations, permutations, product

GiB = 1024 ** 3
MiB = 1024 ** 2
KiB = 1024
INF = float('inf')
checks = 0
negative_controls = []


def require(condition, name):
    global checks
    checks += 1
    if not condition:
        raise AssertionError(name)


def differs(name, actual, reference):
    require(actual != reference, 'negative control not detected: ' + name)
    negative_controls.append(name)


def raises(name, exception, function, *args):
    try:
        function(*args)
    except exception:
        require(True, name)
    else:
        require(False, name)


def physical_usage(records, caps):
    sizes = {}
    used = {device: 0 for device in caps}
    for device, backing, size in records:
        if device not in caps or not isinstance(size, int) or size < 0:
            raise ValueError('invalid device or size')
        key = (device, backing)
        if key in sizes:
            if sizes[key] != size:
                raise ValueError('conflicting size for physical backing')
            continue
        sizes[key] = size
        used[device] += size
    return used


def fits(records, caps):
    used = physical_usage(records, caps)
    return all(used[d] <= caps[d] for d in caps)


def slot_reference(records, devices):
    occupied = set()
    for device, backing, size in records:
        occupied.update((device, backing, slot) for slot in range(size))
    return {d: sum(cell[0] == d for cell in occupied) for d in devices}


def reserve_atomic(resources, additions, capacity, fail_at=None):
    original = dict(resources)
    for index, (key, size) in enumerate(additions):
        if key in resources and resources[key] != size:
            resources.clear()
            resources.update(original)
            raise ValueError('conflicting reservation')
        resources[key] = size
        if sum(resources.values()) > capacity or index == fail_at:
            resources.clear()
            resources.update(original)
            return False
    return True


# A candidate is (name, queue_ms, load_ms, compute_ms, plan_legal, ready).
def route(candidates):
    best = None
    for name, queue, load, compute, legal, ready in candidates:
        if legal:
            key = (queue + load + compute, name)
            if best is None or key < best:
                best = key
    return best


def route_reference(candidates):
    feasible = [(sum(c[1:4]), c[0]) for c in candidates if c[4]]
    return sorted(feasible)[0] if feasible else None


def faulty_queue_only(candidates):
    best = None
    for name, queue, load, compute, legal, ready in candidates:
        if best is None or queue < best[0]:
            best = (queue, name)
    return best


def executable(candidate):
    return candidate[4] and candidate[5]


def partition(layers, stages, memory=None, stage_cap=None):
    n = len(layers)
    if n == 0:
        return (0, ()) if stages == 0 else (INF, ())
    if stages < 1 or stages > n:
        return INF, ()
    if memory is None:
        memory = [0] * n
    if len(memory) != n:
        raise ValueError('memory length')
    times = [0]
    sizes = [0]
    for time, size in zip(layers, memory):
        times.append(times[-1] + time)
        sizes.append(sizes[-1] + size)
    best = [[INF] * (n + 1) for _ in range(stages + 1)]
    parent = [[-1] * (n + 1) for _ in range(stages + 1)]
    best[0][0] = 0
    for s in range(1, stages + 1):
        for k in range(s, n + 1):
            for i in range(s - 1, k):
                if best[s - 1][i] == INF:
                    continue
                if stage_cap is not None and sizes[k] - sizes[i] > stage_cap:
                    continue
                value = max(best[s - 1][i], times[k] - times[i])
                if value < best[s][k]:
                    best[s][k] = value
                    parent[s][k] = i
    if best[stages][n] == INF:
        return INF, ()
    segments = []
    k = n
    for s in range(stages, 0, -1):
        i = parent[s][k]
        segments.append((i, k))
        k = i
    return best[stages][n], tuple(reversed(segments))


def partition_reference(layers, stages, memory=None, stage_cap=None):
    n = len(layers)
    if n == 0:
        return 0 if stages == 0 else INF
    if stages < 1 or stages > n:
        return INF
    if memory is None:
        memory = [0] * n
    scores = []
    for cuts in combinations(range(1, n), stages - 1):
        ends = (0,) + cuts + (n,)
        segments = list(zip(ends, ends[1:]))
        if stage_cap is not None and any(
            sum(memory[a:b]) > stage_cap for a, b in segments
        ):
            continue
        scores.append(max(sum(layers[a:b]) for a, b in segments))
    return min(scores) if scores else INF


def round_robin(arrivals, capacity):
    counts = {}
    ring = deque()
    service = []
    for row in arrivals:
        for tenant in row:
            if counts.get(tenant, 0) == 0:
                ring.append(tenant)
            counts[tenant] = counts.get(tenant, 0) + 1
        selected = []
        for _ in range(capacity):
            if not ring:
                break
            tenant = ring.popleft()
            selected.append(tenant)
            counts[tenant] -= 1
            if counts[tenant]:
                ring.append(tenant)
        service.append(selected)
    return service, counts


def faulty_hot_first(arrivals, capacity):
    waiting = []
    service = []
    for row in arrivals:
        waiting.extend(row)
        waiting.sort(key=lambda tenant: (tenant != 'A', tenant))
        service.append(waiting[:capacity])
        del waiting[:capacity]
    return service, waiting


class LeaseStore:
    def __init__(self):
        self.slots = {}
        self.generations = {}
        self.operations = {}
        self.done = set()

    def create(self, page, version, value, request, ready, published=False):
        if page in self.slots:
            raise RuntimeError('occupied page')
        generation = self.generations.get(page, 0) + 1
        self.generations[page] = generation
        handle = (page, generation, version)
        self.slots[page] = {
            'handle': handle, 'value': value, 'refs': {request},
            'pending': set(), 'ready': ready, 'published': published,
        }
        return handle

    def state(self, handle):
        state = self.slots.get(handle[0])
        if state is None or state['handle'] != handle:
            raise RuntimeError('stale handle')
        return state

    def submit(self, operation, handle, kind='execute'):
        state = self.state(handle)
        if operation in self.operations or operation in self.done:
            raise RuntimeError('operation ID reused')
        if kind not in ('execute', 'copy'):
            raise ValueError('kind')
        if kind == 'execute' and not state['ready']:
            raise RuntimeError('object not ready')
        state['pending'].add(operation)
        self.operations[operation] = (handle, kind)

    def read(self, operation):
        handle, kind = self.operations[operation]
        if kind != 'execute':
            raise RuntimeError('not an execution reader')
        return self.state(handle)['value']

    def release(self, handle, request):
        self.state(handle)['refs'].discard(request)

    def unpublish(self, handle):
        self.state(handle)['published'] = False

    def complete(self, operation):
        if operation in self.done:
            return False
        handle, kind = self.operations[operation]
        state = self.state(handle)
        state['pending'].remove(operation)
        if kind == 'copy':
            state['ready'] = True
        del self.operations[operation]
        self.done.add(operation)
        return True

    def reclaim(self, handle):
        state = self.state(handle)
        if state['refs'] or state['pending'] or state['published']:
            raise RuntimeError('protected page')
        del self.slots[handle[0]]


def main():
    tiny_caps = {'d0': 8, 'd1': 8}
    tiny = [('d0', 'base', 3), ('d0', 'a', 2),
            ('d1', 'base', 3), ('d0', 'base-copy', 3)]
    for mask in product((0, 1), repeat=len(tiny)):
        records = [record for record, include in zip(tiny, mask) if include]
        records += records[:1]
        got = physical_usage(records, tiny_caps)
        ref = slot_reference(records, tiny_caps)
        require(got == ref, 'physical backing versus explicit slot set')
    require(fits([('d0', 'x', 8)], tiny_caps), 'capacity equality')
    require(not fits([('d0', 'x', 9)], tiny_caps), 'capacity overflow')
    raises('unknown device rejected', ValueError, physical_usage,
           [('missing', 'x', 1)], tiny_caps)
    raises('conflicting backing sizes rejected', ValueError, physical_usage,
           [('d0', 'x', 2), ('d0', 'x', 3)], tiny_caps)
    shared = [('d0', 'base', 6), ('d0', 'base', 6)]
    differs('double count shared backing', sum(x[2] for x in shared),
            slot_reference(shared, tiny_caps)['d0'])
    require(physical_usage([('d0', 'base', 3), ('d0', 'copy', 3)], tiny_caps)['d0'] == 6,
            'equal content with separate backing counts twice')

    baseline = {'base': 6}
    for fail_at, capacity in ((None, 10), (0, 20), (1, 20)):
        resources = dict(baseline)
        require(not reserve_atomic(resources, [('adapter', 2), ('kv', 3)],
                                   capacity, fail_at), 'reservation fails')
        require(resources == baseline, 'failure restores exact prior reservations')
    resources = dict(baseline)
    require(reserve_atomic(resources, [('adapter', 2), ('kv', 3)], 11),
            'exact capacity reservation succeeds')
    faulty_resources = dict(baseline)
    faulty_resources['adapter'] = 2
    differs('partial allocation without rollback', faulty_resources, baseline)

    budget = 24 * GiB - 12 * GiB - 3 * GiB - GiB
    per_token = 128 * KiB
    one_adapter = 64 * 16 * (4096 + 4096) * 2
    require(one_adapter == 16 * MiB, 'adapter geometry')
    require(128 * one_adapter == 2 * GiB, 'host adapter main term')
    require(16 * (2048 * per_token + one_adapter) == Fraction(17, 4) * GiB,
            'sixteen histories and adapters')
    require(16 * 512 * per_token == GiB, 'future growth for sixteen histories')
    remaining = budget - 30 * (2048 * per_token + one_adapter)
    need31 = 2048 * per_token + one_adapter
    require(budget == 8 * GiB and remaining == 32 * MiB, 'thirty histories budget')
    require(need31 == 272 * MiB and need31 > remaining, 'reject thirty-first request')
    differs('ignore KV during admission', one_adapter <= remaining, need31 <= remaining)
    require(8 * 30 * per_token <= remaining < 9 * 30 * per_token,
            'existing requests exhaust remaining growth after eight rounds')

    legacy = [('A', 100, 0, 10, True, True),
              ('B', 10, 40, 10, True, False),
              ('C', 10, 0, 10, False, False)]
    require(faulty_queue_only(legacy) == (10, 'B'),
            'recorded failure reproduced: queue tie preserves earlier B')
    tie_results = {faulty_queue_only(p) for p in permutations(legacy)}
    require(tie_results == {(10, 'B'), (10, 'C')}, 'tie depends on input order')
    for ordering in permutations(legacy):
        require(route(ordering) == (60, 'B'), 'legal routing independent of input order')
    strict = legacy[:2] + [('C', 0, 0, 10, False, False)]
    require(route(strict) == (60, 'B'), 'strict fixture selects feasible cold-load plan B')
    require(faulty_queue_only(strict) == (0, 'C'), 'strict negative actually selects illegal C')
    differs('queue-only admits illegal shortest candidate', faulty_queue_only(strict),
            (10, 'B'))
    require(not executable(legacy[1]), 'accepted cold plan cannot execute yet')
    ready_b = legacy[1][:-1] + (True,)
    require(executable(ready_b), 'loaded legal plan can execute')
    require(not executable(('C', 0, 0, 1, False, True)), 'readiness does not override capacity')
    require(route([]) is None, 'empty candidate set')
    variants = list(product(range(3), range(2), (False, True)))
    route_cases = 0
    for settings in product(variants, repeat=3):
        candidates = [(name, q, load, 1, legal, True)
                      for name, (q, load, legal) in zip(('A', 'B', 'C'), settings)]
        require(route(candidates) == route_reference(candidates),
                'routing versus independently sorted feasible set')
        route_cases += 1

    partition_cases = 0
    for n in range(1, 6):
        for layers in product((1, 2, 3), repeat=n):
            for stages in range(1, n + 1):
                score, segments = partition(layers, stages)
                ref = partition_reference(layers, stages)
                require(score == ref, 'DP versus exhaustive cut enumeration')
                require(segments[0][0] == 0 and segments[-1][1] == n,
                        'partition covers both boundaries')
                require(all(a < b for a, b in segments), 'nonempty stages')
                require(all(segments[i][1] == segments[i + 1][0]
                            for i in range(len(segments) - 1)), 'no missing or duplicated layer')
                require(max(sum(layers[a:b]) for a, b in segments) == score,
                        'reconstructed partition objective')
                partition_cases += 1
    require(partition([], 0) == (0, ()), 'empty model zero stages')
    require(partition([], 1)[0] == INF, 'empty model nonzero stages')
    require(partition([1], 0)[0] == INF, 'nonempty model zero stages')
    require(partition([1], 2)[0] == INF, 'more stages than layers')
    layers = [6, 1, 1, 4]
    require(partition(layers, 2)[0] == 6, 'uneven layer optimum')
    differs('equal layer count treated as best stage balance',
            max(sum(layers[:2]), sum(layers[2:])), partition_reference(layers, 2))
    memory = [3, 3, 1, 1]
    for stages in (1, 2, 3, 4):
        require(partition(layers, stages, memory, 4)[0] ==
                partition_reference(layers, stages, memory, 4), 'stage memory feasibility')
    require(partition(layers, 2, memory, 4)[0] == INF, 'time optimum cannot override memory')

    one_burst = [Fraction(11) + j * Fraction(11, 2) for j in range(4)]
    correlated = [Fraction(11) + j * Fraction(11, 2) for j in range(8)]
    independent_mean = Fraction(sum((10, 20, 30, 40)), 4)
    require(sum(one_burst) / 4 == Fraction(77, 4) < independent_mean,
            'idle other model permits burst multiplexing')
    require(sum(correlated) / 8 == Fraction(121, 4) > independent_mean,
            'simultaneous bursts reverse choice')
    require(Fraction(1, 1) / Fraction(11, 2) < Fraction(2, 10),
            'saturated pipeline service rate lower in declared model')

    arrivals = [['A'] * 4 + (['B'] if t % 3 == 0 else []) for t in range(12)]
    rr, waiting = round_robin(arrivals, 4)
    hot, hot_waiting = faulty_hot_first(arrivals, 4)
    b_rr = sum(row.count('B') for row in rr)
    b_hot = sum(row.count('B') for row in hot)
    require(b_rr == 4 and waiting.get('B', 0) == 0, 'all four cold requests progress')
    require(b_hot == 0 and hot_waiting.count('B') == 4, 'hot-priority starvation witnessed')
    differs('hot-priority cold progress', b_hot, b_rr)
    continuous = [['A'] * 20 + ['B'] * 20 + ['C'] * 20] + [[] for _ in range(11)]
    fair, _ = round_robin(continuous, 1)
    sequence = [row[0] for row in fair]
    require(sequence == ['A', 'B', 'C'] * 4, 'persistent cursor across twelve rounds')
    for start in range(len(sequence) - 2):
        require(set(sequence[start:start + 3]) == {'A', 'B', 'C'},
                'every continuous-ready tenant progresses within M opportunities')
    reset, _ = faulty_hot_first(continuous, 1)
    differs('reset priority cursor each round', [row[0] for row in reset], sequence)
    require(round_robin([[], []], 4)[0] == [[], []], 'empty queues')

    rank_work = sum([16] * 8 + [64] * 8)
    padded_work = 16 * 64
    require(Fraction(padded_work, rank_work) == Fraction(8, 5), 'rank padding ratio')
    one_ms = Fraction(one_adapter * 1000, 16 * GiB)
    require(one_ms == Fraction(125, 128), 'adapter transfer units')
    require(24 + 16 * one_ms == Fraction(317, 8), 'serial transfer critical path')
    slow_ms = 24 + Fraction(16 * one_adapter * 1000, 8 * GiB)
    require(slow_ms == Fraction(221, 4) and slow_ms > 40, 'bandwidth change reverses SLO estimate')
    differs('ignore transfer at slow bandwidth', 24 <= 40, slow_ms <= 40)

    store = LeaseStore()
    v1 = store.create(0, 'v1', 3, 'r1', True, True)
    v2 = store.create(1, 'v2', 7, 'r2', False, True)
    store.submit('old-execution', v1)
    store.submit('copy-v2', v2, 'copy')
    raises('FETCHING object cannot execute', RuntimeError, store.submit,
           'too-early', v2)
    store.release(v1, 'r1')
    store.unpublish(v1)
    raises('cancelled old request still has reader', RuntimeError, store.reclaim, v1)
    require(store.read('old-execution') == 3, 'old reader sees immutable v1 value')
    require(store.complete('copy-v2'), 'load completion publishes readiness')
    store.submit('new-execution', v2)
    require(store.read('new-execution') == 7, 'new reader sees v2 value')
    require(store.complete('old-execution'), 'late old completion releases its reader')
    require(not store.complete('old-execution'), 'duplicate completion is idempotent')
    store.reclaim(v1)
    v3 = store.create(0, 'v3', 11, 'r3', True)
    require(v3[1] > v1[1], 'physical page reuse increments generation')
    store.submit('generation-three-execution', v3)
    require(not store.complete('old-execution'), 'old replay ignored after page reuse')
    require(store.state(v3)['pending'] == {'generation-three-execution'},
            'old completion does not reduce new reader')
    raises('new generation remains protected', RuntimeError, store.reclaim, v3)
    raises('stale handle rejected', RuntimeError, store.state, v1)
    store.complete('generation-three-execution')
    store.release(v3, 'r3')
    store.reclaim(v3)
    store.complete('new-execution')
    store.release(v2, 'r2')
    store.unpublish(v2)
    store.reclaim(v2)
    require(not store.slots and not store.operations, 'all completed resources reclaim')

    copying = LeaseStore()
    copy_handle = copying.create(0, 'v2', 7, 'cancelled', False)
    copying.submit('inflight-copy', copy_handle, 'copy')
    copying.release(copy_handle, 'cancelled')
    raises('cancelled DMA destination remains protected', RuntimeError,
           copying.reclaim, copy_handle)
    copying.complete('inflight-copy')
    copying.reclaim(copy_handle)

    faulty = LeaseStore()
    old = faulty.create(0, 'v1', 3, 'old-request', True)
    faulty.submit('old-reader', old)
    faulty.release(old, 'old-request')
    faulty.slots[0]['value'] = 7
    differs('overwrite cancelled but inflight old reader', faulty.read('old-reader'), 3)
    expected_pending = set(faulty.state(old)['pending'])
    faulty.slots[0]['pending'].clear()
    differs('unscoped completion erases current reader',
            set(faulty.state(old)['pending']), expected_pending)

    print('scope: CPU rational arithmetic and finite software state models; no GPU measurements')
    print('legacy queue tie: faulty implementation picks B, not necessarily C')
    print('strict illegal shortest queue: faulty C=0 ms; legal plan B=60 ms')
    print('capacity budget GiB:', budget // GiB)
    print('remaining MiB after 30 histories:', remaining // MiB)
    print('new request MiB:', need31 // MiB, '; common growth rounds before reallocation: 8')
    print('pipeline burst means ms:', Fraction(77, 4), Fraction(121, 4))
    print('rank padding ratio:', Fraction(padded_work, rank_work))
    print('one adapter transfer ms:', one_ms, '; slow serial path ms:', slow_ms)
    print('cold B served: round robin =', b_rr, '; hot priority =', b_hot)
    print('routing enumerations:', route_cases, '; partition enumerations:', partition_cases)
    print('detected negative controls:', ', '.join(negative_controls))
    print('PASS:', checks, 'assertions; declared independent references and negative controls agree')


if __name__ == '__main__':
    main()
