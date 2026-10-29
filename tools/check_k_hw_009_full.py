#!/usr/bin/env python3
"""Offline CPU-serving arithmetic and dependency checks; no I/O or hardware calls."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction as F
from itertools import permutations, product
from math import lcm


def require(name, condition):
    if not condition:
        raise AssertionError(name)


def rejected_mutant(name, action):
    try:
        action()
    except AssertionError:
        print('DETECTED: ' + name)
        return
    raise AssertionError('undetected negative control: ' + name)


def invalid(name, action):
    try:
        action()
    except ValueError:
        print('REJECTED INPUT: ' + name)
        return
    raise AssertionError('accepted invalid input: ' + name)


def effective_cores(cpuset, quota=None, period=None):
    if type(cpuset) is not int or cpuset < 0:
        raise ValueError('cpuset must be a nonnegative integer')
    if quota is None:
        return F(cpuset)
    if type(quota) is not int or quota < 0:
        raise ValueError('quota must be a nonnegative integer')
    if type(period) is not int or period <= 0:
        raise ValueError('period must be a positive integer')
    return min(F(cpuset), F(quota, period))


def budget_flags(pe, host, workers, threads, arrival, costs, reservations,
                 data, bandwidth):
    if type(workers) is not int or type(threads) is not int:
        raise ValueError('worker and thread counts must be integers')
    if workers < 0 or threads < 0 or len(costs) != len(reservations):
        raise ValueError('invalid worker counts or stage vectors')
    pe, host, arrival, data, bandwidth = map(F, (pe, host, arrival, data, bandwidth))
    costs = tuple(map(F, costs))
    reservations = tuple(map(F, reservations))
    if any(x < 0 for x in (pe, host, arrival, data, bandwidth) + costs + reservations):
        raise ValueError('negative quantity')
    flags = set()
    if host > pe:
        flags.add('host')
    if workers * threads + host > pe:
        flags.add('runnable')
    if sum(reservations, F(0)) + host > pe:
        flags.add('reservation')
    if arrival > 0 and (workers == 0 or threads == 0):
        flags.add('workers')
    for i, (cost, reserve) in enumerate(zip(costs, reservations)):
        demand = arrival * cost
        if demand > 0 and demand >= reserve:
            flags.add('stage:' + str(i))
    total = arrival * sum(costs, F(0))
    if total > 0 and total >= pe - host:
        flags.add('cpu')
    if arrival * data > bandwidth:
        flags.add('memory')
    return flags


def reservation_slot_reference(pe, host, reservations):
    """Allocate distinct interchangeable CPU-time quanta, independently of demand."""
    values = tuple(map(F, (pe, host) + tuple(reservations)))
    scale = lcm(*(x.denominator for x in values))
    available = set(range(int(values[0] * scale)))
    for amount in values[1:]:
        for _ in range(int(amount * scale)):
            if not available:
                return False
            available.remove(min(available))
    return True


def test_budget():
    require('quota/cpuset minimum', effective_cores(8, 400, 100) == 4)
    require('larger quota cannot enlarge cpuset', effective_cores(4, 800, 100) == 4)
    require('unlimited quota', effective_cores(6) == 6)
    require('fractional quota', effective_cores(8, 150, 100) == F(3, 2))
    args = (4, 1, 3, 1, 1, (1, 1), (2, 2), 1, 100)
    flags = budget_flags(*args)
    require('reported counterexample fails only reservation', flags == {'reservation'})
    require('independent slot allocation rejects it', not reservation_slot_reference(4, 1, (2, 2)))
    rejected_mutant('delete stage total reservation check',
                    lambda: require('candidate must be rejected', bool(flags - {'reservation'})))
    repaired = budget_flags(4, 1, 3, 1, 1, (1, 1), (F(3, 2), F(3, 2)), 1, 100)
    require('fractional reservation boundary fits', not repaired)
    require('independent fractional slots fit', reservation_slot_reference(4, 1, (F(3, 2), F(3, 2))))
    for pe2, h2, p2, q2 in product(range(5), range(5), range(6), range(6)):
        pe, h, p, q = (F(x, 2) for x in (pe2, h2, p2, q2))
        flags = budget_flags(pe, h, 0, 0, 0, (0, 0), (p, q), 0, 0)
        require('reservation agrees with distinct slot reference',
                ('reservation' not in flags) == reservation_slot_reference(pe, h, (p, q)))
    require('active stage equality is saturation',
            'stage:0' in budget_flags(4, 1, 1, 1, 1, (1,), (1,), 0, 0))
    require('zero cost and zero reservation allowed',
            not budget_flags(4, 1, 1, 1, 1, (0,), (0,), 0, 0))
    require('idle plan may have zero workers', not budget_flags(0, 0, 0, 0, 0, (), (), 0, 0))
    require('idle does not conceal host over-reservation',
            'host' in budget_flags(0, 1, 0, 0, 0, (), (), 0, 0))
    require('positive arrival needs worker',
            'workers' in budget_flags(4, 0, 0, 1, 1, (), (), 0, 0))
    require('runnable cap separate from service demand',
            'runnable' in budget_flags(4, 1, 2, 2, 1, (1,), (2,), 0, 0))
    require('bandwidth equality is non-exceeding only',
            'memory' not in budget_flags(4, 1, 1, 1, 1, (1,), (2,), 100, 100))
    require('positive bytes cannot use zero bandwidth',
            'memory' in budget_flags(4, 1, 1, 1, 1, (1,), (2,), 1, 0))
    invalid('negative cpuset', lambda: effective_cores(-1))
    invalid('zero quota period', lambda: effective_cores(8, 100, 0))
    invalid('mismatched stage vectors', lambda: budget_flags(4, 0, 1, 1, 1, (1,), (), 0, 0))
    invalid('negative stage cost', lambda: budget_flags(4, 0, 1, 1, 1, (-1,), (1,), 0, 0))
    print('OBSERVED: quota budget and stage reservation checks agree with slot allocation; counterexample rejected')


REQUESTS = ('A', 'B')
LAYERS = range(4)


def build_layout(mode, weight=F(20), kv_each=F(15), auxiliary=(F(2), F(2))):
    if mode not in ('full', 'shared', 'layer'):
        raise ValueError('unknown layout')
    weight, kv_each = F(weight), F(kv_each)
    if weight < 0 or kv_each < 0 or len(auxiliary) != 2 or any(x < 0 for x in auxiliary):
        raise ValueError('invalid residency input')
    objects, ops = {}, []
    for layer in LAYERS:
        shard_domain = 0 if layer < 2 else 1
        owners = (0, 1) if mode == 'full' else ((0,) if mode == 'shared' else (shard_domain,))
        for owner in owners:
            wid = ('w', owner, layer)
            objects[wid] = ('weight', layer, owner, weight / 4)
        for ridx, request in enumerate(REQUESTS):
            compute = shard_domain if mode == 'layer' else ridx
            weight_owner = compute if mode == 'full' else (0 if mode == 'shared' else shard_domain)
            kid = ('kv', request, layer)
            objects[kid] = ('kv', (request, layer), compute, kv_each / 4)
            ops.append((request, layer, compute, ('w', weight_owner, layer), kid))
    return {'objects': objects, 'ops': ops, 'aux': tuple(map(F, auxiliary))}


def validate_workload(plan, weight=F(20), kv_each=F(15)):
    """Reference is defined by the fixed requests, not by layout branch totals."""
    expected = {(r, layer) for r in REQUESTS for layer in LAYERS}
    actual = Counter((r, layer) for r, layer, _, _, _ in plan['ops'])
    require('every required request-layer exactly once', actual == Counter(expected))
    kv_objects = [obj for obj in plan['objects'].values() if obj[0] == 'kv']
    require('KV identities exactly conserved', Counter(obj[1] for obj in kv_objects) == Counter(expected))
    for kind, logical, owner, amount in plan['objects'].values():
        require('physical owner exists', owner in (0, 1))
        if kind == 'weight':
            require('weight layer and size', logical in LAYERS and amount == F(weight) / 4)
        else:
            require('KV layer and size', kind == 'kv' and logical in expected and amount == F(kv_each) / 4)
    require('total historical KV invariant', sum((o[3] for o in kv_objects), F(0)) == 2 * kv_each)
    for request, layer, compute, wid, kid in plan['ops']:
        require('compute domain exists', compute in (0, 1))
        require('referenced physical objects exist', wid in plan['objects'] and kid in plan['objects'])
        w, k = plan['objects'][wid], plan['objects'][kid]
        require('weight belongs to executed layer', w[0] == 'weight' and w[1] == layer)
        require('KV belongs to request and layer', k[0] == 'kv' and k[1] == (request, layer))


def layout_facts(plan):
    resident = list(plan['aux'])
    rw, rk = [[F(0), F(0)], [F(0), F(0)]], [[F(0), F(0)], [F(0), F(0)]]
    for _, _, owner, amount in plan['objects'].values():
        resident[owner] += amount
    for _, _, compute, wid, kid in plan['ops']:
        for oid, matrix in ((wid, rw), (kid, rk)):
            obj = plan['objects'][oid]
            matrix[compute][obj[2]] += obj[3]
    return tuple(resident), rw, rk


def traffic_unit_reference(mode, weight=F(20), kv_each=F(15)):
    """Enumerate canonical payload units for every logical request-layer read."""
    rw, rk = [[F(0), F(0)], [F(0), F(0)]], [[F(0), F(0)], [F(0), F(0)]]
    wpart, kpart = F(weight) / 4, F(kv_each) / 4
    scale = lcm(wpart.denominator, kpart.denominator)
    unit = F(1, scale)
    seen = set()
    for ridx, request in enumerate(REQUESTS):
        reader_map = (0, 0, 1, 1) if mode == 'layer' else (ridx,) * 4
        owner_map = {'full': (ridx,) * 4, 'shared': (0,) * 4, 'layer': (0, 0, 1, 1)}[mode]
        for layer in LAYERS:
            reader, owner = reader_map[layer], owner_map[layer]
            for atom in range(int(wpart * scale)):
                identity = (request, layer, 'weight', atom)
                require('reference weight unit visited once', identity not in seen)
                seen.add(identity)
                rw[reader][owner] += unit
            for atom in range(int(kpart * scale)):
                identity = (request, layer, 'kv', atom)
                require('reference KV unit visited once', identity not in seen)
                seen.add(identity)
                rk[reader][reader] += unit
    return rw, rk


def matrix_total(matrix):
    return sum((sum(row, F(0)) for row in matrix), F(0))


def test_numa():
    expected_residents = {'full': (37, 37), 'shared': (37, 17), 'layer': (27, 27)}
    weight_totals, kv_totals = set(), set()
    for mode in expected_residents:
        plan = build_layout(mode)
        validate_workload(plan)
        resident, rw, rk = layout_facts(plan)
        ref_w, ref_k = traffic_unit_reference(mode)
        require(mode + ' residency', resident == expected_residents[mode])
        require(mode + ' independent traffic reference', rw == ref_w and rk == ref_k)
        require(mode + ' same historical KV matrix', rk == [[15, 0], [0, 15]])
        weight_totals.add(matrix_total(rw))
        kv_totals.add(matrix_total(rk))
        require(mode + ' capacity at 32 GiB', all(m <= 32 for m in resident) == (mode == 'layer'))
        require(mode + ' capacity at 40 GiB', all(m <= 40 for m in resident))
        require(mode + ' exact capacity boundary', all(m <= max(resident) for m in resident))
    require('cross-layout logical weight reads invariant', weight_totals == {F(40)})
    require('cross-layout historical KV reads invariant', kv_totals == {F(30)})
    shared = build_layout('shared')
    bad_kv = deepcopy(shared)
    for oid, obj in list(bad_kv['objects'].items()):
        if obj[0] == 'kv':
            bad_kv['objects'][oid] = obj[:3] + (obj[3] / 2,)
    rejected_mutant('shared layout halves total KV', lambda: validate_workload(bad_kv))
    bad_work = deepcopy(shared)
    bad_work['ops'] = [op for op in bad_work['ops'] if op[0] == 'A']
    rejected_mutant('shared layout performs only one request sweep', lambda: validate_workload(bad_work))
    repeated = deepcopy(shared)
    repeated['ops'] = [op for op in repeated['ops'] if op[0] == 'A'] * 2
    rejected_mutant('same operation count repeats A instead of completing B', lambda: validate_workload(repeated))
    _, rw, rk = layout_facts(shared)
    bad_all_local = [[20, 0], [0, 20]]
    require('all-local mutant preserves branch internal byte sum', matrix_total(bad_all_local) == matrix_total(rw))
    rejected_mutant('shared remote reads relabeled local',
                    lambda: require('traffic must match owner reference', bad_all_local == traffic_unit_reference('shared')[0]))
    full_resident = layout_facts(build_layout('full'))[0]
    require('sum of capacity can hide a per-domain failure', sum(full_resident) <= 80 and full_resident[0] > 32)
    for mode in expected_residents:
        empty_history = build_layout(mode, kv_each=F(0))
        validate_workload(empty_history, kv_each=F(0))
        _, w0, k0 = layout_facts(empty_history)
        require('zero historical KV contributes no read', matrix_total(k0) == 0 and matrix_total(w0) == 40)
    require('fixed history byte identity', F(122880 * 128, 1024 * 1024) == 15)
    full_lb = F(35, 100)
    shared_remote_lb = F(20, 25)
    shared_owner_lb = F(55, 100)
    require('full pair memory lower bound', full_lb == F(7, 20))
    require('shared remote lower bound is 0.8 s', shared_remote_lb == F(4, 5))
    require('shared owner controller remains constrained', shared_owner_lb == F(11, 20))
    invalid('unknown layout', lambda: build_layout('tensor_without_axis'))
    print('OBSERVED: historical KV=30 GiB, weight reads=40 GiB, KV reads=30 GiB across all layouts')
    print('OBSERVED: resident GiB full=(37,37), shared=(37,17), layer=(27,27); auxiliary=2 GiB/domain')
    print('OBSERVED: full read bound=0.35 s, shared remote bound=0.8 s; these are model bounds')


def pipeline_finish(jobs, stage_time, message_time):
    if type(jobs) is not int or jobs < 0 or stage_time < 0 or message_time < 0:
        raise ValueError('invalid pipeline input')
    a, delta = F(stage_time), F(message_time)
    free0 = free_link = free1 = F(0)
    for _ in range(jobs):
        free0 += a
        free_link = max(free0, free_link) + delta
        free1 = max(free_link, free1) + a
    return free1


def exact_resource_order_reference(jobs, stage_time, message_time):
    """Exhaust all resource orders, then evaluate each acyclic precedence graph."""
    if jobs == 0:
        return F(0)
    durations = (F(stage_time), F(message_time), F(stage_time))
    orders = tuple(permutations(range(jobs)))
    best = None
    for resource_orders in product(orders, repeat=3):
        predecessors = {(r, s): set() for r in range(jobs) for s in range(3)}
        for r in range(jobs):
            predecessors[r, 1].add((r, 0))
            predecessors[r, 2].add((r, 1))
        for stage, order in enumerate(resource_orders):
            for before, after in zip(order, order[1:]):
                predecessors[after, stage].add((before, stage))
        end = {}
        while len(end) < len(predecessors):
            ready = [v for v, pred in predecessors.items() if v not in end and pred <= end.keys()]
            if not ready:
                break
            for v in ready:
                start = max((end[u] for u in predecessors[v]), default=F(0))
                end[v] = start + durations[v[1]]
        if len(end) == len(predecessors):
            finish = max(end.values())
            best = finish if best is None else min(best, finish)
    require('reference found a legal schedule', best is not None)
    return best


def test_pipeline():
    for jobs, a, delta in product(range(4), (F(0), F(1), F(2)), (F(0), F(1), F(3))):
        require('pipeline equals independently enumerated schedule',
                pipeline_finish(jobs, a, delta) == exact_resource_order_reference(jobs, a, delta))
    for a, delta in product((F(0), F(7, 40), F(1)), (F(0), F(1, 10), F(1, 2))):
        require('two-job formula includes both link regimes',
                pipeline_finish(2, a, delta) == 2 * a + delta + max(a, delta))
    a = F(7, 40)
    require('one request has serial layer halves', pipeline_finish(1, a, 0) == F(7, 20))
    require('pair completion with KV is 0.525 s', pipeline_finish(2, a, 0) == F(21, 40))
    require('weights-only pair completion is 0.3 s', pipeline_finish(2, F(1, 10), 0) == F(3, 10))
    rejected_mutant('aggregate two-domain bandwidth drops layer dependency',
                    lambda: require('must equal exact pair completion', F(35, 100) == pipeline_finish(2, a, 0)))
    rejected_mutant('single request receives twice the serial bandwidth',
                    lambda: require('must equal exact single completion', a == pipeline_finish(1, a, 0)))
    invalid('negative message time', lambda: pipeline_finish(2, 1, -1))
    print('OBSERVED: layer pair DAG bound=0.525 s with KV, 0.3 s weights-only, before messages and compute')


def pages(byte_count, page_size, offset=0):
    if type(byte_count) is not int or type(page_size) is not int or type(offset) is not int:
        raise ValueError('page model requires integers')
    if byte_count < 0 or page_size <= 0 or not 0 <= offset < page_size:
        raise ValueError('invalid byte range')
    return 0 if byte_count == 0 else (offset + byte_count + page_size - 1) // page_size


def byte_page_reference(byte_count, page_size, offset=0):
    return len({address // page_size for address in range(offset, offset + byte_count)})


def panel_tlb(mc, kc, nr, elem, page):
    a, b, c = mc * kc * elem, kc * nr * elem, mc * nr * elem
    return pages(a, page) + 2 * (pages(b, page) + pages(c, page))


def independent_panel_pages(mc, kc, nr, elem, page):
    footprints = (mc * kc * elem, kc * nr * elem, mc * nr * elem,
                  kc * nr * elem, mc * nr * elem)
    touched, base = set(), 0
    for amount in footprints:
        for address in range(base, base + amount):
            touched.add(address // page)
        base += (amount // page + 2) * page
    return len(touched)


def packed_bytes(count, bits, group_size, metadata_bytes):
    if count < 0 or bits <= 0 or group_size <= 0 or metadata_bytes < 0:
        raise ValueError('invalid storage input')
    return (count * bits + 7) // 8 + ((count + group_size - 1) // group_size) * metadata_bytes


def stage_lower_bound(flops, peak, data, bandwidth, exposed_sync):
    if any(x < 0 for x in (flops, peak, data, bandwidth, exposed_sync)):
        raise ValueError('negative stage quantity')
    if (flops > 0 and peak == 0) or (data > 0 and bandwidth == 0):
        raise ValueError('positive work needs positive capacity')
    compute = F(flops) / peak if flops else F(0)
    memory = F(data) / bandwidth if data else F(0)
    return max(compute, memory) + F(exposed_sync)


def test_panels_storage_and_sync():
    for p in (1, 4, 16):
        for offset in range(p):
            for amount in range(2 * p + 1):
                require('page count agrees with byte-address set', pages(amount, p, offset) == byte_page_reference(amount, p, offset))
    args = (240, 256, 4, 8, 4096)
    count = panel_tlb(*args)
    require('panel reference page set', count == independent_panel_pages(*args) == 128)
    a_bytes = 240 * 256 * 8
    require('A only fits 125 entries', pages(a_bytes, 4096) == 120)
    rejected_mutant('TLB counts A but omits current and next B/C',
                    lambda: require('same TLB feasibility', (pages(a_bytes, 4096) <= 125) == (count <= 125)))
    cache_bytes = a_bytes + 256 * 4 * 8 + 240 * 4 * 8
    require('cache body sum', cache_bytes == 507392)
    rejected_mutant('cache counts only A',
                    lambda: require('same cache feasibility', (a_bytes <= 500000) == (cache_bytes <= 500000)))
    require('unaligned exact page payload spans two pages', pages(4096, 4096, 1) == 2)
    invalid('zero page size', lambda: pages(1, 0))
    require('4-bit payload plus group metadata', packed_bytes(4096 * 4096, 4, 128, 4) == 8912896)
    for count in (0, 1, 127, 128, 129, 255, 256, 257):
        groups = list(range(0, count, 128))
        reference_body = len(range(0, count, 2))
        require('tail groups and packed nibbles', packed_bytes(count, 4, 128, 4) == reference_body + 4 * len(groups))
    require('serial exposed sync after overlapping compute/memory', stage_lower_bound(2, 1, 4, 1, 3) == 7)
    rejected_mutant('max of all terms hides serial synchronization',
                    lambda: require('same serial-stage bound', max(F(2), F(4), F(3)) == stage_lower_bound(2, 1, 4, 1, 3)))
    rejected_mutant('sum compute and memory double counts overlap',
                    lambda: require('same overlap bound', F(2 + 4 + 3) == stage_lower_bound(2, 1, 4, 1, 3)))
    require('zero body with exposed sync', stage_lower_bound(0, 0, 0, 0, 2) == 2)
    invalid('positive bytes with zero bandwidth', lambda: stage_lower_bound(0, 0, 1, 0, 0))
    print('OBSERVED: panel TLB=128 pages; cache body=507392 bytes; INT4 example=8.5 MiB')
    print('OBSERVED: serial exposed synchronization uses max(compute,memory)+sync')


def main():
    print('SCOPE: deterministic offline CPU-serving arithmetic, residency and dependency models')
    print('SCOPE: no project imports, files, subprocesses, network, hardware counters or GPU measurements')
    test_budget()
    test_numa()
    test_pipeline()
    test_panels_storage_and_sync()
    print('RESULT: PASS (scoped model checks only)')


if __name__ == '__main__':
    main()
