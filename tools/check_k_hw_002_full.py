from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction as F
from itertools import product


def legal(condition, message):
    if not condition:
        raise ValueError(message)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


@dataclass(frozen=True)
class Record:
    holder: int
    chunk: int
    version: int
    covers: frozenset
    values: tuple
    made: int

    @property
    def key(self):
        return self.holder, self.chunk, self.version


def expand_ring(lengths, element_bytes=1):
    legal(len(lengths) > 0, 'empty communicator')
    legal(all(type(x) is int and x >= 0 for x in lengths), 'integer nonnegative lengths required')
    legal(type(element_bytes) is int and element_bytes > 0, 'invalid element width')
    p = len(lengths)
    n = sum(lengths)
    inputs = tuple(tuple(1000 * r + i + 1 for i in range(n)) for r in range(p))
    offsets = [sum(lengths[:j]) for j in range(p)]
    records = {}
    current = {}
    for r in range(p):
        for j, size in enumerate(lengths):
            rec = Record(r, j, 0, frozenset((r,)), inputs[r][offsets[j]:offsets[j] + size], -1)
            records[rec.key] = rec
            current[r, j] = rec.key
    phases = []
    sent = [0] * p
    for kind in ('RS', 'AG'):
        if kind == 'AG':
            current = {(r, r): current[r, r] for r in range(p)}
        for s in range(p - 1):
            k = len(phases)
            snapshot = dict(current)
            updates = {}
            messages = []
            for r in range(p):
                j = (r - s - (1 if kind == 'RS' else 0)) % p
                d = (r + 1) % p
                source_id = snapshot[r, j]
                source = records[source_id]
                local_id = snapshot[d, j] if kind == 'RS' else None
                if local_id is not None:
                    local = records[local_id]
                    legal(not (source.covers & local.covers), 'duplicate contribution')
                    legal(local.covers == frozenset((d,)), 'unexpected local contribution')
                    covers = source.covers | local.covers
                    legal(len(source.values) == len(local.values), 'reduction extent')
                    values = tuple(a + b for a, b in zip(source.values, local.values))
                else:
                    legal(source.covers == frozenset(range(p)), 'incomplete gather source')
                    covers = source.covers
                    values = source.values
                out = Record(d, j, k + 1, covers, values, k)
                legal(out.key not in records, 'duplicate producer')
                records[out.key] = out
                messages.append({'source': source_id, 'local': local_id, 'dest': d,
                                 'chunk': j, 'bytes': lengths[j] * element_bytes, 'output': out.key})
                updates[d, j] = out.key
                sent[r] += lengths[j] * element_bytes
            current.update(updates)
            phases.append({'index': k, 'kind': kind, 'step': s,
                           'deps': () if k == 0 else (k - 1,), 'messages': messages})
        if kind == 'RS':
            legal(all(records[current[r, r]].covers == frozenset(range(p)) for r in range(p)),
                  'reduce-scatter did not finish')
    return {'inputs': inputs, 'records': records, 'phases': phases,
            'final': current, 'sent': tuple(sent)}


def audit_ring(plan, lengths, element_bytes=1):
    # Independent reference: walk one packet per chunk around the ring.
    # This does not reuse the rank/step send-index formula of the producer.
    p = len(lengths)
    pool = plan['records']
    inputs = plan['inputs']
    check(len(inputs) == p and all(len(row) == sum(lengths) for row in inputs), 'input extent')
    offsets = [sum(lengths[:j]) for j in range(p)]
    known = set()
    for r in range(p):
        for j, size in enumerate(lengths):
            key = (r, j, 0)
            expected = Record(r, j, 0, frozenset((r,)), inputs[r][offsets[j]:offsets[j] + size], -1)
            check(pool[key] == expected, 'initial record differs from raw input')
            known.add(key)
    check(len(plan['phases']) == 2 * (p - 1), 'round count')
    locations = [(j + 1) % p for j in range(p)]
    packet_ids = [(locations[j], j, 0) for j in range(p)]
    contributors = [frozenset((locations[j],)) for j in range(p)]
    expected_final = {(r, r): (r, r, p - 1) for r in range(p)}
    sent = [0] * p
    reductions = 0
    for k, phase in enumerate(plan['phases']):
        gathering = k >= p - 1
        step = k - (p - 1) if gathering else k
        if gathering and step == 0:
            locations = list(range(p))
            packet_ids = [(j, j, p - 1) for j in range(p)]
            contributors = [frozenset(range(p)) for _ in range(p)]
        check(phase['index'] == k, 'stage index')
        check(phase['kind'] == ('AG' if gathering else 'RS') and phase['step'] == step, 'stage identity')
        check(phase['deps'] == (() if k == 0 else (k - 1,)), 'missing round dependency')
        check(len(phase['messages']) == p, 'missing or extra message')
        by_sender = {}
        for message in phase['messages']:
            sender = message['source'][0]
            check(sender not in by_sender, 'duplicate sender')
            by_sender[sender] = message
        for j, size in enumerate(lengths):
            sender = locations[j]
            dest = (sender + 1) % p
            message = by_sender[sender]
            check(message['source'] == packet_ids[j], 'wrong source chunk or version')
            check(message['dest'] == dest and message['chunk'] == j, 'wrong destination or chunk')
            check(type(message['bytes']) is int and message['bytes'] == size * element_bytes, 'payload extent')
            source = pool[message['source']]
            check(source.made < k, 'source not produced before consumption')
            check(source.covers == contributors[j], 'source coverage')
            expected_source = tuple(sum(inputs[o][offsets[j] + i] for o in contributors[j]) for i in range(size))
            check(source.values == expected_source, 'source value differs from direct sum')
            if gathering:
                check(message['local'] is None, 'gather must copy rather than reduce')
                after = contributors[j]
            else:
                check(message['local'] == (dest, j, 0), 'local must be original and unused')
                local = pool[message['local']]
                check(local.made < k and local.covers == frozenset((dest,)), 'invalid local contribution')
                check(dest not in contributors[j], 'contribution counted twice')
                after = contributors[j] | frozenset((dest,))
                reductions += size
            output_id = (dest, j, k + 1)
            check(message['output'] == output_id and output_id not in known, 'output producer identity')
            expected_values = tuple(sum(inputs[o][offsets[j] + i] for o in after) for i in range(size))
            check(pool[output_id] == Record(dest, j, k + 1, after, expected_values, k), 'invalid produced record')
            known.add(output_id)
            packet_ids[j] = output_id
            locations[j] = dest
            contributors[j] = after
            sent[sender] += size * element_bytes
            if gathering:
                expected_final[dest, j] = output_id
        if not gathering and step == p - 2:
            check(locations == list(range(p)), 'wrong reduce-scatter owners')
            check(all(a == frozenset(range(p)) for a in contributors), 'incomplete reduction')
    check(set(pool) == known, 'unexplained records')
    check(plan['final'] == expected_final, 'final ownership table')
    for r in range(p):
        for j, size in enumerate(lengths):
            final = pool[plan['final'][r, j]]
            direct = tuple(sum(inputs[o][offsets[j] + i] for o in range(p)) for i in range(size))
            check(final.covers == frozenset(range(p)) and final.values == direct, 'all-reduce result')
    byte_lengths = [x * element_bytes for x in lengths]
    nbytes = sum(byte_lengths)
    closed_sent = tuple(0 if p == 1 else 2 * nbytes - byte_lengths[r] - byte_lengths[(r + 1) % p]
                        for r in range(p))
    check(tuple(sent) == plan['sent'] == closed_sent, 'per-rank byte conservation')
    check(sum(sent) == 2 * (p - 1) * nbytes, 'global byte conservation')
    check(reductions == (p - 1) * sum(lengths), 'reduction element count')


def capacity_table(arcs, groups, outgoing, incoming):
    caps = {('arc', a): F(v[2]) for a, v in arcs.items()}
    caps.update({('group', g): F(v[0]) for g, v in groups.items()})
    caps.update({('out', r): F(c) for r, c in enumerate(outgoing)})
    caps.update({('in', r): F(c) for r, c in enumerate(incoming)})
    return caps


def window_account(demand, caps, window):
    legal(window >= 0 and all(demand[k] <= caps[k] * window for k in caps), 'window too short')
    utilization = {k: demand[k] / (caps[k] * window) if window else F(0) for k in caps}
    headroom = {k: caps[k] - demand[k] / window if window else caps[k] for k in caps}
    slack = {k: caps[k] * window - demand[k] for k in caps}
    return utilization, headroom, slack


def price_phase(flows, arcs, groups, outgoing, incoming, alpha=F(0)):
    caps = capacity_table(arcs, groups, outgoing, incoming)
    legal(all(c > 0 for c in caps.values()) and alpha >= 0, 'invalid capacity or alpha')
    legal(all(v[3] >= 0 for v in arcs.values()), 'negative path latency')
    for cap, weights in groups.values():
        legal(all(a in arcs and h >= 0 for a, h in weights.items()), 'invalid group incidence')
    demand = {k: F(0) for k in caps}
    latency = F(0)
    for flow in flows:
        b = flow['bytes']
        legal(type(b) is int and b >= 0, 'integer payload bytes required')
        route = flow['path']
        legal(len(route) > 0 and all(a in arcs for a in route), 'missing route arc')
        cursor = flow['src']
        path_latency = F(0)
        for a in route:
            u, v, cap, ell = arcs[a]
            legal(cursor == u, 'disconnected or wrong-direction route')
            cursor = v
            path_latency += ell
            demand['arc', a] += b
            for g, (group_cap, weights) in groups.items():
                demand['group', g] += b * weights.get(a, 0)
        legal(cursor == flow['dst'], 'wrong route endpoint')
        legal(('out', flow['src']) in caps and ('in', flow['dst']) in caps, 'unknown endpoint')
        demand['out', flow['src']] += b
        demand['in', flow['dst']] += b
        latency = max(latency, path_latency)
    seconds = {k: demand[k] / cap for k, cap in caps.items()}
    tau = max([latency, F(0)] + list(seconds.values()))
    utilization, headroom, slack = window_account(demand, caps, tau)
    return {'caps': caps, 'demand': demand, 'seconds': seconds, 'latency': latency,
            'tau': tau, 'w': alpha + tau, 'utilization': utilization,
            'headroom': headroom, 'slack': slack}


def audit_price(result, flows, arcs, groups, outgoing, incoming, alpha=F(0)):
    # Independent byte-token reference, deliberately different from weighted
    # message aggregation. Tests stay small so each payload byte can be visited.
    capacities = {}
    for a in arcs:
        capacities['arc', a] = F(arcs[a][2])
    for g in groups:
        capacities['group', g] = F(groups[g][0])
    for r, cap in enumerate(outgoing):
        capacities['out', r] = F(cap)
    for r, cap in enumerate(incoming):
        capacities['in', r] = F(cap)
    tokens = {k: F(0) for k in capacities}
    longest_latency = F(0)
    for flow in flows:
        check(type(flow['bytes']) is int and flow['bytes'] >= 0, 'reference payload extent')
        cursor = flow['src']
        elapsed = F(0)
        for a in flow['path']:
            check(a in arcs and arcs[a][0] == cursor, 'reference route continuity')
            cursor = arcs[a][1]
            elapsed += arcs[a][3]
        check(cursor == flow['dst'], 'reference route destination')
        longest_latency = max(longest_latency, elapsed)
        for _byte in range(flow['bytes']):
            tokens['out', flow['src']] += 1
            tokens['in', flow['dst']] += 1
            for occurrence in flow['path']:
                tokens['arc', occurrence] += 1
                for g in groups:
                    tokens['group', g] += groups[g][1].get(occurrence, 0)
    check(result['caps'] == capacities, 'capacity schema mismatch')
    check(result['demand'] == tokens, 'demand differs from byte-token reference')
    for k, cap in capacities.items():
        check(tokens[k] <= cap * result['tau'], 'resource budget exceeded: ' + str(k))
        check(result['seconds'][k] == tokens[k] / cap, 'resource seconds')
        expected_util = tokens[k] / (cap * result['tau']) if result['tau'] else F(0)
        expected_headroom = cap - tokens[k] / result['tau'] if result['tau'] else cap
        check(result['utilization'][k] == expected_util, 'utilization')
        check(result['headroom'][k] == expected_headroom, 'headroom')
        check(result['slack'][k] == cap * result['tau'] - tokens[k], 'window slack')
    exact = max([F(0), longest_latency] + [tokens[k] / cap for k, cap in capacities.items()])
    check(result['tau'] == exact and result['w'] == alpha + exact, 'stage budget or startup')
    check(result['latency'] == longest_latency, 'latency bound')


def ring_resources(capacities):
    p = len(capacities)
    arcs = {'a' + str(r): (r, (r + 1) % p, F(c), F(0)) for r, c in enumerate(capacities)}
    return arcs, [F(1000)] * p, [F(1000)] * p


def phase_flows(phase):
    return [{'src': m['source'][0], 'dst': m['dest'], 'bytes': m['bytes'],
             'path': ('a' + str(m['source'][0]),)} for m in phase['messages']]


def price_ring(plan, capacities, groups=None, alpha=F(0)):
    arcs, outgoing, incoming = ring_resources(capacities)
    groups = {} if groups is None else groups
    stages = []
    for phase in plan['phases']:
        flows = phase_flows(phase)
        result = price_phase(flows, arcs, groups, outgoing, incoming, alpha)
        audit_price(result, flows, arcs, groups, outgoing, incoming, alpha)
        stages.append(result)
    return sum((stage['w'] for stage in stages), F(0)), stages


def rejected(label, operation):
    try:
        operation()
    except (ValueError, AssertionError):
        return label
    raise AssertionError('negative control escaped: ' + label)


def byte_rows(plan):
    return tuple(tuple(m['bytes'] for m in phase['messages']) for phase in plan['phases'])


def main():
    plans = 0
    for p in range(1, 6):
        for lengths in product(range(3), repeat=p):
            width = 1 + sum(lengths) % 2
            plan = expand_ring(lengths, width)
            audit_ring(plan, lengths, width)
            capacities = [1 + r % 3 for r in range(p)]
            groups = {'fabric': (F(3), {a: F(1) for a in ring_resources(capacities)[0]})} if sum(lengths) % 2 == 0 else {}
            total, stages = price_ring(plan, capacities, groups, F(1, 7))
            if not any(lengths):
                check(total == F(2 * (p - 1), 7), 'zero-payload startup convention')
            if p == 1:
                check(total == 0 and not stages and plan['sent'] == (0,), 'single rank')
            plans += 1
    print('CPU ring plans audited:', plans, '; P=1..5, integer lengths 0..2, widths 1/2, heterogeneous capacities')

    uniform = expand_ring((4, 4, 4, 4))
    audit_ring(uniform, (4, 4, 4, 4))
    uniform_time, _ = price_ring(uniform, (40, 40, 40, 40), alpha=F(1, 200000))
    check(uniform_time == 6 * F(1, 200000) + F(24, 40), 'uniform closed form')
    check(uniform['sent'] == (24, 24, 24, 24), 'uniform send ledger')

    uneven = expand_ring((2, 1))
    audit_ring(uneven, (2, 1))
    uneven_time, uneven_stages = price_ring(uneven, (1, 1))
    check(byte_rows(uneven) == ((1, 2), (2, 1)), 'uneven message matrix')
    check(tuple(x['tau'] for x in uneven_stages) == (F(2), F(2)), 'uneven stage maxima')
    check(uneven_time == 4 and uneven['sent'] == (3, 3), 'uneven total and ledger')
    print('uneven blocks [2,1]: rows =', byte_rows(uneven), '; stage seconds =',
          [str(x['tau']) for x in uneven_stages], '; total =', str(uneven_time))

    heterogeneous = expand_ring((3, 1, 2))
    audit_ring(heterogeneous, (3, 1, 2))
    hetero_time, hetero_stages = price_ring(heterogeneous, (1, 2, 3))
    expected_rows = ((2, 3, 1), (1, 2, 3), (3, 1, 2), (2, 3, 1))
    check(byte_rows(heterogeneous) == expected_rows, 'heterogeneous message matrix')
    check(tuple(x['tau'] for x in hetero_stages) == (F(2), F(1), F(3), F(2)), 'heterogeneous stage maxima')
    check(hetero_time == F(8), 'heterogeneous total must be 8, not 17/2')
    check(heterogeneous['sent'] == (8, 9, 7), 'heterogeneous per-rank ledger')
    # Explicit service arithmetic for the formerly miscomputed second round.
    check(tuple(F(b, c) for b, c in zip(expected_rows[1], (1, 2, 3))) == (F(1), F(1), F(1)),
          'second round independently has three unit service times')
    print('blocks [3,1,2], capacities [1,2,3]: rows =', expected_rows,
          '; stage seconds =', [str(x['tau']) for x in hetero_stages],
          '; total =', str(hetero_time), '; sent =', heterogeneous['sent'])

    arcs = {'ab': (0, 1, F(100), F(0)), 'ba': (1, 0, F(100), F(0))}
    flows = [{'src': 0, 'dst': 1, 'bytes': 100, 'path': ('ab',)},
             {'src': 1, 'dst': 0, 'bytes': 100, 'path': ('ba',)}]
    endpoints = [F(100), F(100)]
    half_groups = {'wire': (F(100), {'ab': F(1), 'ba': F(1)})}
    full_groups = {'forward': (F(100), {'ab': F(1)}), 'reverse': (F(100), {'ba': F(1)})}
    half = price_phase(flows, arcs, half_groups, endpoints, endpoints)
    full = price_phase(flows, arcs, full_groups, endpoints, endpoints)
    audit_price(half, flows, arcs, half_groups, endpoints, endpoints)
    audit_price(full, flows, arcs, full_groups, endpoints, endpoints)
    check(half['tau'] == 2 and full['tau'] == 1, 'duplex common pricing path')
    check(half['demand']['group', 'wire'] == 200, 'half-duplex service bytes')
    check(half['utilization']['group', 'wire'] == 1 and half['headroom']['group', 'wire'] == 0, 'shared occupancy')
    util, head, slack = window_account(half['demand'], half['caps'], F(4))
    check(util['group', 'wire'] == F(1, 2) and head['group', 'wire'] == 50
          and slack['group', 'wire'] == 200, 'explicit longer-window accounting')
    print('common pricing path: full-duplex seconds =', str(full['tau']),
          '; half-duplex seconds =', str(half['tau']), '; shared service bytes = 200')

    path_arcs = {'first': (0, 'switch', F(1000), F(1, 10)),
                 'second': ('switch', 1, F(1000), F(1, 5))}
    one_flow = [{'src': 0, 'dst': 1, 'bytes': 100, 'path': ('first', 'second')}]
    repeated = {'bus': (F(100), {'first': F(1), 'second': F(1)})}
    canonical = {'bus': (F(100), {'first': F(1), 'second': F(0)})}
    for groups, expected in ((repeated, F(2)), (canonical, F(1))):
        result = price_phase(one_flow, path_arcs, groups, endpoints, endpoints)
        audit_price(result, one_flow, path_arcs, groups, endpoints, endpoints)
        check(result['tau'] == expected, 'resource-use incidence')
    fractional = {'bus': (F(100), {'first': F(1, 2), 'second': F(1, 2)})}
    weighted = price_phase(one_flow, path_arcs, fractional, endpoints, endpoints)
    audit_price(weighted, one_flow, path_arcs, fractional, endpoints, endpoints)
    check(weighted['demand']['group', 'bus'] == 100, 'fractional service coefficients')
    latency_only = price_phase([{'src': 0, 'dst': 1, 'bytes': 1, 'path': ('first', 'second')}],
                               path_arcs, {}, endpoints, endpoints)
    audit_price(latency_only, [{'src': 0, 'dst': 1, 'bytes': 1, 'path': ('first', 'second')}],
                path_arcs, {}, endpoints, endpoints)
    check(latency_only['tau'] == F(3, 10), 'path latency bound')
    empty = price_phase([], arcs, half_groups, endpoints, endpoints)
    audit_price(empty, [], arcs, half_groups, endpoints, endpoints)
    check(empty['tau'] == 0, 'empty stage')
    print('one flow: two shared services = 2 seconds; one canonical service = 1 second; latency-only bound = 3/10 second')

    negatives = []
    # Retain the complete true schema and service demand. Only the estimator
    # incorrectly omits group constraints, so rejection must expose underbudget.
    no_groups = price_phase(flows, arcs, {}, endpoints, endpoints)
    faulty_shared = deepcopy(half)
    faulty_shared['tau'] = no_groups['tau']
    faulty_shared['w'] = no_groups['w']
    check(faulty_shared['tau'] == 1, 'ignored-group mutant did not reproduce underpricing')
    negatives.append(rejected('ignored shared group violates real resource budget',
                              lambda: audit_price(faulty_shared, flows, arcs, half_groups, endpoints, endpoints)))

    average_tau = F(3, 2)
    check(2 * average_tau == 3 < uneven_time, 'average mutant did not reproduce 3 versus 4')
    faulty_average = deepcopy(uneven_stages[0])
    faulty_average['tau'] = average_tau
    faulty_average['w'] = average_tau
    uneven_arcs, out_caps, in_caps = ring_resources((1, 1))
    negatives.append(rejected('average block time violates real message budget',
                              lambda: audit_price(faulty_average, phase_flows(uneven['phases'][0]),
                                                  uneven_arcs, {}, out_caps, in_caps)))
    negatives.append(rejected('old heterogeneous total 17/2',
                              lambda: check(hetero_time == F(17, 2), 'stale total')))

    missing = deepcopy(heterogeneous)
    missing['phases'][0]['messages'].pop()
    negatives.append(rejected('missing ring message', lambda: audit_ring(missing, (3, 1, 2))))
    corrupt = deepcopy(heterogeneous)
    key = corrupt['phases'][0]['messages'][0]['output']
    corrupt['records'][key] = replace(corrupt['records'][key], covers=frozenset((0,)))
    negatives.append(rejected('corrupt contribution coverage', lambda: audit_ring(corrupt, (3, 1, 2))))
    premature = deepcopy(heterogeneous)
    source_key = premature['phases'][0]['messages'][0]['source']
    premature['records'][source_key] = replace(premature['records'][source_key], made=0)
    negatives.append(rejected('unproduced source version', lambda: audit_ring(premature, (3, 1, 2))))

    for lengths, width in (((), 1), ((1, -1), 1), ((1, F(1, 2)), 1), ((1,), 0)):
        negatives.append(rejected('invalid ring input',
                                  lambda lengths=lengths, width=width: expand_ring(lengths, width)))
    negatives.append(rejected('missing physical arc', lambda: price_phase(
        [{'src': 0, 'dst': 1, 'bytes': 1, 'path': ('absent',)}], arcs, {}, endpoints, endpoints)))
    negatives.append(rejected('zero group capacity', lambda: price_phase(
        flows, arcs, {'wire': (F(0), {'ab': F(1)})}, endpoints, endpoints)))
    negatives.append(rejected('wrong route direction', lambda: price_phase(
        [{'src': 0, 'dst': 1, 'bytes': 1, 'path': ('ba',)}], arcs, {}, endpoints, endpoints)))
    negatives.append(rejected('undersized service window',
                              lambda: window_account(half['demand'], half['caps'], F(1))))
    print('negative controls rejected:', '; '.join(negatives))
    print('scope: deterministic CPU integer/Fraction model only; no GPU, NCCL, CUDA, NIC, file, network, or subprocess operations')


if __name__ == '__main__':
    main()
