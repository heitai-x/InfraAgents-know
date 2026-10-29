#!/usr/bin/env python3
'''K-SYS-019: deterministic stdlib CPU symbolic checks.

Offline and read-only: no files, processes, threads, network, credentials,
project imports or devices are used. Inputs are the fixtures below.
This is not a Linux, CUDA, plugin, MPS, MIG or serving measurement.
'''

from collections import deque
from fractions import Fraction
from itertools import combinations, product

GIB = 1 << 30
CHECKS = 0
NEGATIVES = []


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def caught(name, faulty, expected):
    check(faulty != expected, 'negative control escaped: ' + name)
    NEGATIVES.append(name)


def raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def powerset(items):
    values = tuple(sorted(items))
    for k in range(len(values) + 1):
        for part in combinations(values, k):
            yield frozenset(part)


def fold_set(online, requests):
    effective = frozenset(online)
    for requested in requests:
        overlap = effective & requested
        if requested and overlap:
            effective = overlap
    return effective


def set_reference(online, requests, universe):
    # Boolean state per resource; decide whether a level has any grantable
    # requested resource before changing individual membership flags.
    flags = [c in online for c in universe]
    for requested in requests:
        eligible = [flags[i] and c in requested
                    for i, c in enumerate(universe)]
        if any(eligible):
            flags = eligible
    return frozenset(c for i, c in enumerate(universe) if flags[i])


def old_intersection(online, requests):
    effective = frozenset(online)
    for requested in requests:
        if requested:
            effective &= requested
    return effective


def path(hierarchy, node):
    result = []
    while node is not None:
        result.append(node)
        node = hierarchy[node]['parent']
    return tuple(reversed(result))


def hierarchy():
    return {
        'root': dict(parent=None, quota=None, mem=128 * GIB,
                     cpus=frozenset(range(4))),
        'svc': dict(parent='root', quota=Fraction(4), mem=48 * GIB,
                    cpus=frozenset()),
        'a': dict(parent='svc', quota=Fraction(3), mem=32 * GIB,
                  cpus=frozenset((0, 1))),
        'b': dict(parent='svc', quota=Fraction(3), mem=32 * GIB,
                  cpus=frozenset((2, 3))),
        'tiny': dict(parent='svc', quota=Fraction(8), mem=32 * GIB,
                     cpus=frozenset((0,))),
        'fallback': dict(parent='a', quota=None, mem=32 * GIB,
                         cpus=frozenset((9,))),
    }


def effective(h, node):
    ancestors = path(h, node)
    cpus = fold_set(h['root']['cpus'],
                    [h[a]['cpus'] for a in ancestors[1:]])
    quotas = [h[a]['quota'] for a in ancestors
              if h[a]['quota'] is not None]
    cpu = min([Fraction(len(cpus))] + quotas)
    memory = min(h[a]['mem'] for a in ancestors
                 if h[a]['mem'] is not None)
    return cpu, memory, cpus


def fractional_flow(consumers):
    # consumers: (unique id, nonnegative Fraction demand, allowed CPU set).
    # Fractional bipartite max flow with exact residual arithmetic.
    source, sink = ('source',), ('sink',)
    residual = {}
    adjacency = {}

    def edge(u, v, capacity):
        residual[u, v] = Fraction(capacity)
        residual[v, u] = Fraction(0)
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    ids = [cid for cid, _d, _cpus in consumers]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate consumer identity')
    total = sum((d for _cid, d, _cpus in consumers), Fraction(0))
    if any(d < 0 for _cid, d, _cpus in consumers):
        raise ValueError('negative demand')
    all_cpus = set()
    for cid, demand, cpus in consumers:
        vertex = ('consumer', cid)
        edge(source, vertex, demand)
        for cpu in sorted(cpus):
            edge(vertex, ('cpu', cpu), total)
            all_cpus.add(cpu)
    for cpu in sorted(all_cpus):
        edge(('cpu', cpu), sink, 1)

    amount = Fraction(0)
    while True:
        previous = {source: None}
        queue = deque((source,))
        while queue and sink not in previous:
            u = queue.popleft()
            for v in adjacency.get(u, ()):
                if v not in previous and residual[u, v] > 0:
                    previous[v] = u
                    queue.append(v)
        if sink not in previous:
            return amount
        increment = total
        v = sink
        while previous[v] is not None:
            u = previous[v]
            increment = min(increment, residual[u, v])
            v = u
        v = sink
        while previous[v] is not None:
            u = previous[v]
            residual[u, v] -= increment
            residual[v, u] += increment
            v = u
        amount += increment


def subset_capacity_reference(consumers):
    # Weighted Hall condition: every consumer subset must fit the union of
    # its allowed unit-capacity CPUs. No flow or residual graph is used.
    for k in range(1, len(consumers) + 1):
        for selected in combinations(consumers, k):
            cpus = set()
            demand = Fraction(0)
            for _cid, d, allowed in selected:
                demand += d
                cpus.update(allowed)
            if demand > len(cpus):
                return False
    return True


def cpu_ok(h, consumers, use_cpuset=True):
    # consumers are (id, cgroup, CPU seconds/second), including existing
    # reservations. This is a static long-run continuous-share model.
    flow_input = []
    for cid, node, demand in consumers:
        if demand < 0:
            raise ValueError('negative CPU demand')
        cpu, _memory, cpus = effective(h, node)
        if not use_cpuset:
            quotas = [h[a]['quota'] for a in path(h, node)
                      if h[a]['quota'] is not None]
            cpus = h['root']['cpus']
            cpu = min([Fraction(len(cpus))] + quotas)
        if demand > cpu:
            return False
        flow_input.append((cid, demand, cpus))
    for ancestor, values in h.items():
        quota = values['quota']
        if quota is not None:
            demand = sum((d for _cid, n, d in consumers
                          if ancestor in path(h, n)), Fraction(0))
            if demand > quota:
                return False
    return fractional_flow(flow_input) == sum(
        (d for _cid, d, _cpus in flow_input), Fraction(0))


def memory_totals(h, records):
    # A record is (allocation-or-unmaterialized-reservation identity,
    # actual charge owner, bytes). Occupancy and reservations must not
    # cover the same storage twice. Identical aliases are deduplicated.
    unique = {}
    for identity, owner, size in records:
        if owner not in h or size < 0:
            raise ValueError('invalid memory record')
        metadata = owner, size
        if identity in unique and unique[identity] != metadata:
            raise ValueError('conflicting memory identity')
        unique[identity] = metadata
    totals = {a: 0 for a in h}
    for owner, size in unique.values():
        for ancestor in path(h, owner):
            totals[ancestor] += size
    return totals


def memory_ok(h, records):
    totals = memory_totals(h, records)
    return all(values['mem'] is None or totals[a] <= values['mem']
               for a, values in h.items())


def admission(h, consumers, records, complete):
    if not cpu_ok(h, consumers) or not memory_ok(h, records):
        return 'REJECT'
    return 'MODEL_FEASIBLE' if complete else 'UNKNOWN'


def rank_only_fault(h, consumers, records):
    # Wrong: each consumer is independently granted the full path budget;
    # no shared ancestor totals and no common physical CPU allocation.
    cpu_pass = all(d <= effective(h, n)[0] for _cid, n, d in consumers)
    memory_pass = all(size <= effective(h, owner)[1]
                      for _identity, owner, size in records)
    return cpu_pass and memory_pass


def resolve(grants, required, observed):
    authorized = set(grants)
    required = set(required)
    if not required <= authorized:
        return 'REJECT', None
    if observed is None:
        return 'WAIT_ENUMERATION', None
    ordinals = [i for i, _p in observed]
    identities = [p for _i, p in observed]
    if (len(ordinals) != len(set(ordinals)) or
            sorted(ordinals) != list(range(len(observed))) or
            len(identities) != len(set(identities)) or
            set(identities) != required):
        return 'REJECT', None
    return 'READY', dict(observed)


def map_matches_observation(candidate, observed):
    # Independent fixture comparison by ordinal lookup, not grant order.
    if candidate is None or len(candidate) != len(observed):
        return False
    return all(i in candidate and candidate[i] == physical
               for i, physical in observed)


def gpu_memory_ok(capacity, demands, mode, shares=1):
    if capacity < 0 or any(d < 0 for d in demands) or shares < 1:
        raise ValueError('invalid GPU budget')
    if sum(demands) > capacity:
        return False
    if mode == 'mps':
        return all(d <= Fraction(capacity, shares) for d in demands)
    if mode == 'time-slicing':
        return True
    raise ValueError('unsupported sharing model')


def survivors(groups, failed):
    return {name for name, (dependencies, _capacity) in groups.items()
            if dependencies.isdisjoint(failed)}


def fault_ok(groups, scenarios):
    for failed, required_capacity in scenarios:
        live = survivors(groups, failed)
        if sum(groups[name][1] for name in live) < required_capacity:
            return False
    return True


def scope_check(scope, cert_epoch, current_epoch, request):
    model, dtype, modality, context = request
    if cert_epoch != current_epoch:
        return 'STALE_EPOCH'
    if (model not in scope['model'] or dtype not in scope['dtype'] or
            modality not in scope['modality'] or
            not 0 <= context <= scope['max_ctx']):
        return 'OUT_OF_SCOPE'
    return 'ADMIT'


def model_only_fault(scope, cert_epoch, current_epoch, request):
    return 'ADMIT' if request[0] in scope['model'] else 'OUT_OF_SCOPE'


def peak_of_snapshots(snapshots):
    totals = []
    for snapshot in snapshots:
        unique = {}
        for identity, size in snapshot:
            if size < 0 or (identity in unique and unique[identity] != size):
                raise ValueError('invalid storage snapshot')
            unique[identity] = size
        totals.append(sum(unique.values()))
    return max(totals, default=0), totals


def dag_finish(tasks):
    # Resource order must already be represented in predecessor edges.
    finish = {}
    remaining = dict(tasks)
    while remaining:
        progressed = False
        for name in sorted(tuple(remaining)):
            duration, predecessors = remaining[name]
            if duration < 0:
                raise ValueError('negative task duration')
            if all(p in finish for p in predecessors):
                finish[name] = max((finish[p] for p in predecessors),
                                   default=0) + duration
                del remaining[name]
                progressed = True
        if not progressed:
            raise ValueError('cycle or missing predecessor')
    return max(finish.values(), default=0)


def main():
    print('K-SYS-019: offline CPU symbolic model; no device measurements')

    universe = (0, 1, 2)
    sets = tuple(powerset(universe))
    set_cases = 0
    old_hits = 0
    own_hits = 0
    for online in sets:
        for requests in product(sets, repeat=3):
            got = fold_set(online, requests)
            expected = set_reference(online, requests, universe)
            check(got == expected, 'set fold/reference mismatch')
            parent = online
            for requested in requests:
                child = fold_set(parent, (requested,))
                check(child <= parent, 'effective escaped parent')
                if parent & requested:
                    check(child <= requested, 'granted branch escaped request')
                else:
                    check(child == parent, 'fallback did not inherit')
                parent = child
            old_hits += old_intersection(online, requests) != expected
            own = (requests[-1] & online) if requests[-1] else online
            own_hits += own != expected
            set_cases += 1
    check(fold_set({0, 1}, (frozenset((2,)),)) == {0, 1},
          'required disjoint-request fallback missing')
    caught('old nonempty intersection', old_hits, 0)
    caught('own-only set', own_hits, 0)
    print('sets: cases=%d; old-intersection disagreements=%d; own-only=%d'
          % (set_cases, old_hits, own_hits))

    flow_cases = 0
    for left, right in product(sets, repeat=2):
        for d0, d1 in product((Fraction(0), Fraction(1, 2), Fraction(1)),
                              repeat=2):
            consumers = [('x', d0, left), ('y', d1, right)]
            actual = fractional_flow(consumers) == d0 + d1
            check(actual == subset_capacity_reference(consumers),
                  'flow disagrees with subset capacity reference')
            flow_cases += 1
    print('CPU assignment: %d flow/subset-reference comparisons' % flow_cases)

    h = hierarchy()
    check(effective(h, 'fallback')[2] == {0, 1}, 'hierarchy fallback wrong')
    check(cpu_ok(h, [('x', 'tiny', Fraction(1))]), 'CPU equality rejected')
    check(not cpu_ok(h, [('x', 'tiny', Fraction(2))]), 'cpuset bound missed')
    caught('quota-only CPU admission',
           cpu_ok(h, [('x', 'tiny', Fraction(2))], use_cpuset=False), False)

    h_cpu = {n: dict(v) for n, v in h.items()}
    h_cpu['svc']['quota'] = Fraction(2)
    shared_cpu = [('x', 'a', Fraction(3, 2)), ('y', 'b', Fraction(3, 2))]
    check(not cpu_ok(h_cpu, shared_cpu), 'shared CPU ancestor overcommitted')
    caught('per-rank-only shared CPU', rank_only_fault(h_cpu, shared_cpu, []), False)
    h_overlap = {n: dict(v) for n, v in h.items()}
    h_overlap['a']['cpus'] = frozenset((0,))
    h_overlap['b']['cpus'] = frozenset((0,))
    overlapping = [('x', 'a', Fraction(1)), ('y', 'b', Fraction(1))]
    check(not cpu_ok(h_overlap, overlapping), 'shared physical CPU missed')
    caught('per-rank-only physical CPU',
           rank_only_fault(h_overlap, overlapping, []), False)

    over_memory = [('x', 'a', 30 * GIB), ('y', 'b', 30 * GIB)]
    check(memory_totals(h, over_memory)['svc'] == 60 * GIB,
          'ancestor memory total wrong')
    check(not memory_ok(h, over_memory), '48GiB parent overcommit missed')
    caught('per-rank-only shared memory', rank_only_fault(h, [], over_memory), False)
    existing = [('existing-object', 'b', 20 * GIB)]
    new = [('new-reservation', 'a', 30 * GIB)]
    check(not memory_ok(h, existing + new), 'existing occupancy omitted')
    check(memory_ok(h, [('x', 'a', 24 * GIB), ('y', 'b', 24 * GIB)]),
          'ancestor equality rejected')
    aliases = [('shared', 'a', 30 * GIB), ('shared', 'a', 30 * GIB)]
    check(memory_ok(h, aliases), 'shared aliases counted twice')
    check(memory_totals(h, aliases)['svc'] == 30 * GIB, 'alias accounting wrong')
    caught('sum memory aliases', sum(r[2] for r in aliases) <= 32 * GIB, True)
    check(raises_value_error(lambda: memory_totals(
        h, [('same', 'a', GIB), ('same', 'b', GIB)])),
        'conflicting charge owners accepted')
    feasible = [('r', 'a', 8 * GIB)]
    check(admission(h, [], feasible, True) == 'MODEL_FEASIBLE', 'model feasible case')
    check(admission(h, [], feasible, False) == 'UNKNOWN', 'missing ledger not unknown')
    check(admission(h, [], over_memory, False) == 'REJECT', 'known violation lost')
    print('budgets: cpuset, shared ancestors, physical CPU, ownership and UNKNOWN checked')

    grants = ['GPU-a', 'GPU-b']
    container_observed = [(0, 'GPU-b'), (1, 'GPU-a')]
    final_observed = [(0, 'GPU-b')]
    check(resolve(grants, {'GPU-b'}, None) == ('WAIT_ENUMERATION', None),
          'missing enumeration continued')
    status, mu = resolve(grants, {'GPU-b'}, final_observed)
    check(status == 'READY' and mu == {0: 'GPU-b'}, 'final observed map wrong')
    check(map_matches_observation(mu, final_observed), 'observed mapping failed')
    grant_order_fault = {i: p for i, p in enumerate(grants)}
    caught('grant-order mapping',
           map_matches_observation(grant_order_fault, container_observed), True)
    ignore_filter_fault = dict(container_observed)
    caught('ignore final filtering',
           map_matches_observation(ignore_filter_fault, final_observed), True)
    check(resolve(grants, set(grants), container_observed)[0] == 'READY',
          'actual order differing from grant order rejected')
    bad_records = [
        [(0, 'GPU-z')],
        [],
        [(1, 'GPU-b')],
        [(0, 'GPU-b'), (0, 'GPU-a')],
        [(0, 'GPU-b'), (1, 'GPU-b')],
    ]
    for observed in bad_records:
        check(resolve(grants, {'GPU-b'}, observed)[0] == 'REJECT',
              'invalid enumeration accepted: %r' % (observed,))
    shared_grants = ['GPU-x', 'GPU-x']
    for _process in ('client-A', 'client-B'):
        check(resolve(shared_grants, {'GPU-x'}, [(0, 'GPU-x')]) ==
              ('READY', {0: 'GPU-x'}), 'shared slots fabricated ordinals')
    check(resolve([], set(), []) == ('READY', {}), 'explicit empty CPU-only map')
    check(not gpu_memory_ok(80, [45, 45], 'time-slicing'), 'shared GPU overcommit')
    caught('replicas multiply capacity', 45 + 45 <= len(shared_grants) * 80, False)
    check(gpu_memory_ok(80, [40, 40], 'time-slicing'), 'shared equality rejected')
    check(not gpu_memory_ok(80, [60, 20], 'mps', 2), 'MPS client cap missed')
    check(gpu_memory_ok(80, [40, 40], 'mps', 2), 'MPS boundary rejected')
    caught('MPS total-only admission', sum((60, 20)) <= 80, False)
    print('devices: explicit enumeration, WAIT, final filter, slots and MPS caps checked')

    single = {'single': (frozenset(('node-A',)), 40)}
    check(fault_ok(single, []), 'no-target single instance incorrectly rejected')
    tp = {'tp': (frozenset(('node-A', 'node-B')), 40)}
    scenario = [(frozenset(('node-A',)), 1)]
    check(survivors(tp, scenario[0][0]) == set(), 'partial TP group counted alive')
    check(not fault_ok(tp, scenario), 'spread TP group treated as replica')
    spread_rank_fault = len(tp['tp'][0]) > 1
    caught('rank dispersion implies survival', spread_rank_fault, False)
    replicas = {
        'A': (frozenset(('node-A', 'rack-X')), 40),
        'B': (frozenset(('node-B', 'rack-X')), 40),
        'C': (frozenset(('node-C', 'rack-Y')), 40),
    }
    one_node = [(frozenset((node,)), 70)
                for node in ('node-A', 'node-B', 'node-C')]
    check(fault_ok(replicas, one_node), 'independent-node capacity fixture wrong')
    check(not fault_ok(replicas, [(frozenset(('rack-X',)), 70)]),
          'common domain capacity loss missed')
    print('faults: complete groups, absent target, replicas and common domains checked')

    scope = dict(model={'m1'}, dtype={'fp16'}, modality={'text', 'image'}, max_ctx=4096)
    narrow = dict(scope, modality={'text'})
    cases = [
        (scope, 7, 7, ('m1', 'fp16', 'text', 4096), 'ADMIT'),
        (scope, 7, 7, ('m1', 'bf16', 'text', 2048), 'OUT_OF_SCOPE'),
        (narrow, 7, 7, ('m1', 'fp16', 'image', 2048), 'OUT_OF_SCOPE'),
        (scope, 7, 7, ('m1', 'fp16', 'text', 4097), 'OUT_OF_SCOPE'),
        (scope, 7, 8, ('m1', 'fp16', 'text', 2048), 'STALE_EPOCH'),
        (narrow, 8, 8, ('m1', 'fp16', 'text', 2048), 'ADMIT'),
        (scope, 7, 7, ('other', 'fp16', 'text', 2048), 'OUT_OF_SCOPE'),
        (scope, 7, 7, ('m1', 'fp16', 'text', -1), 'OUT_OF_SCOPE'),
    ]
    for index, (sc, cert, current, request, expected) in enumerate(cases):
        check(scope_check(sc, cert, current, request) == expected, 'scope fixture %d' % index)
        if expected in ('OUT_OF_SCOPE', 'STALE_EPOCH') and request[0] == 'm1':
            caught('model-only scope fixture %d' % index,
                   model_only_fault(sc, cert, current, request), expected)
    print('scope: explicit dtype, modality, context, epoch and narrowing fixtures checked')

    snapshots = [
        [('source', 24), ('staging', 4)],
        [('source', 24), ('target', 24), ('staging', 4)],
        [('target', 24), ('graph-pool', 3)],
    ]
    peak, totals = peak_of_snapshots(snapshots)
    # Independent reference: explicit unique (storage, GiB-granule) tokens.
    reference_totals = []
    for snapshot in snapshots:
        tokens = set()
        for storage, size in snapshot:
            tokens.update((storage, offset) for offset in range(size))
        reference_totals.append(len(tokens))
    check(totals == reference_totals == [28, 52, 27] and peak == 52,
          'unique-storage peak mismatch')
    check(peak_of_snapshots([]) == (0, []), 'empty peak wrong')
    caught('sum phase totals as actual peak', sum(totals), peak)

    tasks = {
        'read-A': (4, ()),
        'read-B': (4, ('read-A',)),
        'verify-A': (2, ('read-A',)),
        'verify-B': (2, ('read-B',)),
    }
    check(dag_finish(tasks) == 10, 'shared SSD DAG finish wrong')
    intervals = {
        'read-A': (0, 4, 'SSD'),
        'read-B': (4, 8, 'SSD'),
        'verify-A': (4, 6, 'GPU-A'),
        'verify-B': (8, 10, 'GPU-B'),
    }
    # Independent schedule witness: audit durations, dependencies and
    # resource interval overlap without the DAG recurrence.
    for name, (start, end, _resource) in intervals.items():
        duration, predecessors = tasks[name]
        check(end - start == duration, 'schedule duration wrong')
        check(all(intervals[p][1] <= start for p in predecessors),
              'schedule dependency violated')
    for left, right in combinations(intervals, 2):
        s0, e0, r0 = intervals[left]
        s1, e1, r1 = intervals[right]
        check(r0 != r1 or e0 <= s1 or e1 <= s0, 'exclusive resource overlap')
    check(max(v[1] for v in intervals.values()) == 10, 'schedule witness finish')
    resource_lower_bound = 4 + 4
    isolated_rank_estimate = max(4 + 2, 4 + 2)
    caught('isolated max as shared-resource upper bound',
           isolated_rank_estimate >= resource_lower_bound, True)
    check(sum(d for d, _pred in tasks.values()) == 12 and 12 >= 10,
          'legal serial upper estimate invalid')
    check(raises_value_error(lambda: dag_finish({'x': (1, ('x',))})),
          'DAG cycle not detected')
    print('cost: peak=52GiB; shared SSD witness=10s; serial=12s; isolated 6s rejected')

    print('negative controls detected: ' + '; '.join(NEGATIVES))
    print('completed scoped assertions=%d; negative witnesses=%d' % (CHECKS, len(NEGATIVES)))
    print('limit: static CPU models only; no kernel, GPU, runtime, SLO or Agent-effect evidence')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
