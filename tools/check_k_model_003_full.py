from collections import defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction as F
from math import comb


G = (17, 5, 7, 9)
OLD_G = (16, 5, 7, 8)


class ProtocolError(Exception):
    pass


class Incomplete(ProtocolError):
    pass


def require(ok, message):
    if not ok:
        raise ProtocolError(message)


def destination(place, uid, expert):
    key = (uid, expert)
    return place[key] if key in place else place[expert]


def version(expert):
    return 100 + expert


def expert_value(expert, x):
    # Synthetic function, deliberately not a SwiGLU implementation.
    d = len(x)
    return tuple((expert + 1) * value + j * x[(j + 1) % d]
                 for j, value in enumerate(x))


def shared_value(x, enabled):
    return tuple(2 * value + F(j, 3) if enabled else F(0)
                 for j, value in enumerate(x))


def direct_reference(tokens, routes, shared=False):
    # Independent token-major formula: no placement, buckets or messages.
    result = {}
    for uid, x in tokens.items():
        q = shared_value(x, shared)
        d = len(x)
        result[uid] = tuple(x[j] + q[j] + sum(
            (gate * ((e + 1) * x[j] + j * x[(j + 1) % d])
             for e, gate in routes[uid]), F(0)) for j in range(d))
    return result


@dataclass(frozen=True)
class Item:
    generation: tuple
    uid: tuple
    expert: int
    gate: F
    target: int
    weight_version: int


@dataclass(frozen=True)
class Reply:
    generation: tuple
    uid: tuple
    sender: int
    covered: frozenset
    versions: tuple
    value: tuple


def simulate(tokens, routes, place, ranks, dedup, premerge,
             shared=False, reverse=False, fault=None, b_h=2, b_y=2):
    require(ranks > 0, 'empty rank group')
    require(b_h > 0 and b_y > 0, 'invalid element bytes')
    require(set(tokens) == set(routes), 'token-route mismatch')
    dimensions = {len(x) for x in tokens.values()}
    require(len(dimensions) <= 1 and all(d > 0 for d in dimensions),
            'invalid dimensions')
    require(all(0 <= o < ranks and t >= 0 for o, t in tokens),
            'invalid token identity')
    for o in range(ranks):
        ids = sorted(t for origin, t in tokens if origin == o)
        require(ids == list(range(len(ids))), 'local token scope')

    expected = {uid: {} for uid in tokens}
    planned_counts = defaultdict(int)
    items = []
    for uid in sorted(tokens):
        for e, gate in routes[uid]:
            require(e not in expected[uid], 'duplicate selected expert')
            r = destination(place, uid, e)
            require(0 <= r < ranks, 'invalid destination')
            expected[uid][e] = (r, version(e))
            planned_counts[uid[0], r] += 1
            items.append(Item(G, uid, e, F(gate), r, version(e)))

    # Build complete packets before enqueuing each packet once.
    packets = {}
    for item in items:
        key = ((item.uid, item.target) if dedup else
               (item.uid, item.target, item.expert))
        if key not in packets:
            packets[key] = [item.uid, item.target, tokens[item.uid], []]
        packets[key][3].append(item)
    arrivals = list(packets.values())
    ledger = []
    for uid, r, x, descriptors in arrivals:
        if uid[0] != r:
            ledger.append(('dispatch', uid[0], r, uid, len(x) * b_h))

    if fault in ('stale_dispatch', 'drop_descriptor'):
        remote = [p for p in arrivals if p[0][0] != p[1]]
        require(bool(remote), 'fault fixture requires remote dispatch')
        if fault == 'stale_dispatch':
            p = remote[0]
            p[3][0] = replace(p[3][0], generation=OLD_G)
        else:
            p = next(p for p in remote if len(p[3]) > 1)
            p[3].pop()
    if reverse:
        arrivals.reverse()

    buckets = [defaultdict(list) for _ in range(ranks)]
    received = set()
    received_counts = defaultdict(int)
    recv_plan = defaultdict(set)
    for uid, r, x, descriptors in arrivals:
        for item in descriptors:
            require(item.generation == G, 'stale dispatch generation')
            require(item.uid == uid, 'descriptor-hidden identity mismatch')
            require(item.target == r, 'wrong receiving rank')
            require(item.weight_version == version(item.expert),
                    'wrong dispatch weight version')
            identity = (item.generation, item.uid, item.expert)
            require(identity not in received, 'duplicate bucket insertion')
            received.add(identity)
            received_counts[uid[0], r] += 1
            recv_plan[r, uid].add(item.expert)
            buckets[r][item.expert].append((item, x))
    require(dict(received_counts) == dict(planned_counts),
            'dispatch item count incomplete')

    # Remote and local inputs execute through the same expert-major buckets.
    groups = defaultdict(list)
    executed = set()
    for r in range(ranks):
        for e in sorted(buckets[r]):
            for item, x in sorted(buckets[r][e], key=lambda row: row[0].uid):
                identity = (item.generation, item.uid, item.expert)
                require(identity not in executed, 'duplicate expert execution')
                executed.add(identity)
                out = expert_value(e, x)
                weighted = tuple(item.gate * v for v in out)
                groups[r, item.uid].append((e, weighted))
    require(executed == received, 'expert execution coverage')

    replies = []
    for (r, uid), group in sorted(groups.items()):
        group.sort(key=lambda row: row[0])
        if premerge:
            covered = frozenset(e for e, v in group)
            require(covered == frozenset(recv_plan[r, uid]), 'partial group')
            vector = tuple(sum((v[j] for e, v in group), F(0))
                           for j in range(len(tokens[uid])))
            versions = tuple((e, version(e)) for e in sorted(covered))
            replies.append(Reply(G, uid, r, covered, versions, vector))
        else:
            for e, vector in group:
                replies.append(Reply(G, uid, r, frozenset([e]),
                                     ((e, version(e)),), vector))

    if fault == 'wrong_origin':
        require(not premerge, 'origin fault requires item return mode')
        swapped = 0
        for i, reply in enumerate(replies):
            if (reply.sender == 2 and reply.uid in ((0, 0), (1, 0))
                    and reply.covered == frozenset([1])):
                replies[i] = replace(reply, uid=(1 - reply.uid[0], 0))
                swapped += 1
        require(swapped == 2, 'origin collision fixture missing')
    if fault in ('drop_return', 'duplicate_return', 'stale_return', 'bad_version'):
        indices = [i for i, m in enumerate(replies) if m.sender != m.uid[0]]
        require(bool(indices), 'fault fixture requires remote return')
        i = indices[0]
        if fault == 'drop_return':
            replies.pop(i)
        elif fault == 'duplicate_return':
            replies.append(replies[i])
        elif fault == 'stale_return':
            replies[i] = replace(replies[i], generation=OLD_G)
        else:
            bad = tuple((e, v + 1) for e, v in replies[i].versions)
            replies[i] = replace(replies[i], versions=bad)
    if reverse:
        replies.reverse()

    seen = {uid: set() for uid in tokens}
    sums = {uid: [F(0)] * len(x) for uid, x in tokens.items()}
    for reply in replies:
        uid, r = reply.uid, reply.sender
        require(uid in expected, 'unknown return token')
        if r != uid[0]:
            ledger.append(('return', r, uid[0], uid, len(reply.value) * b_y))
        require(reply.generation == G, 'stale return generation')
        coverage = reply.covered
        require(bool(coverage), 'empty return coverage')
        require(all(e in expected[uid] and expected[uid][e][0] == r
                    for e in coverage), 'unexpected contributor')
        require(reply.versions == tuple((e, expected[uid][e][1])
                                        for e in sorted(coverage)),
                'wrong return weight version')
        if premerge:
            whole = frozenset(e for e, (target, v) in expected[uid].items()
                              if target == r)
            require(coverage == whole, 'incomplete premerged coverage')
        else:
            require(len(coverage) == 1, 'item mode coverage size')
        require(not (seen[uid] & coverage), 'duplicate combine contribution')
        require(len(reply.value) == len(sums[uid]), 'return dimension')
        if fault == 'overwrite':
            sums[uid] = list(reply.value)
        else:
            for j, value in enumerate(reply.value):
                sums[uid][j] += value
        seen[uid].update(coverage)

    missing = {uid: set(expected[uid]) - seen[uid] for uid in tokens
               if set(expected[uid]) != seen[uid]}
    if missing:
        raise Incomplete('origin cannot publish; missing=' + repr(missing))
    output = {}
    for uid, x in tokens.items():
        q = shared_value(x, shared)
        output[uid] = tuple(x[j] + q[j] + sums[uid][j]
                            for j in range(len(x)))
    return output, ledger


def network_reference(tokens, routes, place, ranks, dedup, premerge,
                      b_h=2, b_y=2):
    # Route-set formulas, independent of packets and simulated send events.
    d = len(next(iter(tokens.values()))) if tokens else 0
    remote_items = sum(1 for uid in tokens for e, gate in routes[uid]
                       if destination(place, uid, e) != uid[0])
    remote_pairs = {(uid, r) for uid in tokens for r in range(ranks)
                    if r != uid[0] and any(destination(place, uid, e) == r
                                           for e, gate in routes[uid])}
    sends = len(remote_pairs) if dedup else remote_items
    returns = len(remote_pairs) if premerge else remote_items
    return {'dispatch': sends * d * b_h, 'return': returns * d * b_y}, (
        remote_items, len(remote_pairs))


def check_case(name, tokens, routes, place, ranks, shared=False, b_h=2, b_y=2):
    reference = direct_reference(tokens, routes, shared)
    for dedup in (False, True):
        for premerge in (False, True):
            predicted, counts = network_reference(
                tokens, routes, place, ranks, dedup, premerge, b_h, b_y)
            for reverse in (False, True):
                got, ledger = simulate(tokens, routes, place, ranks,
                                       dedup, premerge, shared, reverse,
                                       b_h=b_h, b_y=b_y)
                assert got == reference, (name, dedup, premerge, reverse)
                assert all(src != dst for phase, src, dst, uid, size in ledger)
                actual = {phase: sum(size for p, src, dst, uid, size in ledger
                                     if p == phase)
                          for phase in ('dispatch', 'return')}
                assert actual == predicted, (name, actual, predicted)
    print(name + ': exact token formula and send ledger matched; '
          'remote_items=' + str(counts[0]) + ', remote_pairs=' + str(counts[1]))
    return reference


def must_reject(label, operation, exception=ProtocolError):
    try:
        operation()
    except exception as error:
        print(label + ': detected (' + str(error) + ')')
    else:
        raise AssertionError('negative control escaped: ' + label)


def check_distributed():
    place = {0: 0, 1: 2, 2: 2, 3: 1, 4: 2}
    tokens = {(0, 0): (F(1),), (1, 0): (F(2),)}
    routes = {(0, 0): ((1, F(1, 4)), (2, F(3, 4))),
              (1, 0): ((1, F(1, 2)), (3, F(1, 2)))}
    ref = check_case('two origins have token 0; destination rank has B_o=0',
                     tokens, routes, place, 3)
    assert ref == {(0, 0): (F(15, 4),), (1, 0): (F(8),)}
    bad, unused = simulate(tokens, routes, place, 3, True, False,
                           fault='wrong_origin')
    assert bad == {(0, 0): (F(21, 4),), (1, 0): (F(13, 2),)}
    assert bad != ref
    print('wrong origin: complete expert tickets still produced wrong values; reference detected fault')
    bad, unused = simulate(tokens, routes, place, 3, False, False,
                           fault='overwrite')
    assert bad != ref
    print('overwrite combine: independent token formula detected fault')
    for fault in ('drop_return', 'duplicate_return', 'stale_dispatch',
                  'stale_return', 'bad_version', 'drop_descriptor'):
        exception = Incomplete if fault == 'drop_return' else ProtocolError
        must_reject(fault, lambda fault=fault: simulate(
            tokens, routes, place, 3, True, True, fault=fault), exception)

    mixed = {(0, 0): (F(-1), F(3)), (0, 1): (F(2), F(5)),
             (1, 0): (F(4), F(-2))}
    mixed_routes = {(0, 0): ((1, F(1, 3)), (2, F(2, 3))),
                    (0, 1): ((0, F(1, 2)), (3, F(1, 2))),
                    (1, 0): ((0, F(2, 5)), (1, F(3, 5)))}
    check_case('mixed placement, vector experts, shared and b_h!=b_y',
               mixed, mixed_routes, place, 3, shared=True, b_h=2, b_y=4)
    replicas = dict(place)
    replicas[((0, 0), 1)] = 0
    assert check_case('same-version replica changes one item destination',
                      tokens, routes, replicas, 3) == ref
    hotspot = {(o, 0): (F(o + 1), F(2 - o)) for o in range(3)}
    check_case('hotspot includes local and remote origins', hotspot,
               {uid: ((1, F(1)),) for uid in hotspot}, place, 3)
    check_case('k=0 retains residual and shared', mixed,
               {uid: () for uid in mixed}, place, 3, shared=True)
    check_case('empty global batch', {}, {}, place, 3)
    check_case('k=E and empty unused origin ranks',
               {(0, 0): (F(2), F(-3))},
               {(0, 0): tuple((e, F(1, 4)) for e in range(4))}, place, 3)
    must_reject('duplicate selected expert', lambda: simulate(
        {(0, 0): (F(1),)},
        {(0, 0): ((1, F(1, 2)), (1, F(1, 2)))}, place, 3, True, True))

    wide = {uid: (x[0],) * 4096 for uid, x in tokens.items()}
    check_case('d=4096 mixed-placement payload example', wide, routes, place, 3)
    totals = []
    for dedup, premerge in ((False, False), (False, True),
                           (True, False), (True, True)):
        predicted, counts = network_reference(wide, routes, place, 3,
                                               dedup, premerge)
        assert counts == (3, 2)
        totals.append(sum(predicted.values()))
    assert totals == [49152, 40960, 40960, 32768]
    print('modeled payload totals: 48,40,40,32 KiB; metadata excluded')

    local = {(0, 0): (F(1),) * 4096}
    local_routes = {(0, 0): ((0, F(2, 5)), (1, F(3, 5)))}
    local_place = {0: 0, 1: 0}
    check_case('single-rank all-local: zero network',
               local, local_routes, local_place, 1)
    check_case('all-local with empty peer origins: zero network',
               local, local_routes, local_place, 3)
    predicted, counts = network_reference(local, local_routes, local_place,
                                           3, True, True)
    assert predicted == {'dispatch': 0, 'return': 0} and counts == (0, 0)
    all_pairs = {(uid, destination(local_place, uid, e))
                 for uid in local for e, gate in local_routes[uid]}
    wrong_bytes = len(all_pairs) * 4096 * 2
    assert wrong_bytes == 8192 and wrong_bytes != predicted['dispatch']
    print('local-as-network negative control: erroneous 8192 bytes versus send ledger 0 detected')


def topk(scores, k, bias=None):
    if not 0 <= k <= len(scores):
        raise ValueError('k outside expert range')
    if bias is None:
        bias = [F(0)] * len(scores)
    if len(bias) != len(scores):
        raise ValueError('bias dimension')
    return sorted(range(len(scores)),
                  key=lambda e: (-(scores[e] + bias[e]), e))[:k]


def gates(scores, selected):
    if not selected:
        return {}
    total = sum((scores[e] for e in selected), F(0))
    if total == 0:
        raise ValueError('zero selected affinity')
    return {e: scores[e] / total for e in selected}


def check_routing():
    scores = [F(1, 2), F(2, 5), F(1, 10)]
    bias = [F(0), F(0), F(1)]
    selected = topk(scores, 2, bias)
    assert selected == [2, 0]
    correct = gates(scores, selected)
    assert correct == {2: F(1, 6), 0: F(5, 6)}
    faulty = gates([s + b for s, b in zip(scores, bias)], selected)
    tokens = {(0, 0): (F(1),)}
    assert direct_reference(tokens, {(0, 0): tuple(correct.items())}) != (
        direct_reference(tokens, {(0, 0): tuple(faulty.items())}))
    assert topk([F(1)] * 3, 2) == [0, 1]
    assert gates(scores, topk(scores, 0)) == {}
    for invalid_k in (-1, 4):
        try:
            topk(scores, invalid_k)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid k accepted')
    try:
        gates([F(0)], [0])
    except ValueError:
        pass
    else:
        raise AssertionError('zero normalization accepted')
    print('routing: supplied affinity, ties and k boundaries checked; biased-gating fault detected')


def expected_unique_dp(experts, batch, k):
    if experts < 0 or batch < 0 or not 0 <= k <= experts:
        raise ValueError('invalid expert-union parameters')
    if experts == 0:
        return F(0)
    choices = comb(experts, k)
    distribution = {0: F(1)}
    for step in range(batch):
        nxt = defaultdict(F)
        for touched, probability in distribution.items():
            for overlap in range(k + 1):
                fresh = k - overlap
                if overlap <= touched and fresh <= experts - touched:
                    ways = comb(touched, overlap) * comb(experts - touched, fresh)
                    nxt[touched + fresh] += probability * F(ways, choices)
        assert sum(nxt.values(), F(0)) == 1
        distribution = nxt
    return sum((count * p for count, p in distribution.items()), F(0))


def check_costs():
    for experts, batch, k in ((8, 16, 2), (8, 0, 2), (8, 3, 8),
                              (4, 3, 1), (4, 3, 0), (0, 5, 0)):
        exact = expected_unique_dp(experts, batch, k)
        formula = (F(experts) * (1 - F(experts - k, experts) ** batch)
                   if experts else F(0))
        assert exact == formula
    assert expected_unique_dp(8, 16, 2) != min(8, 16 * 2)
    ns = [16, 4, 3, 3, 2, 2, 1, 1]
    slots = [(e, base + lane) for e, n in enumerate(ns)
             for base in range(0, n, 4) for lane in range(4)]
    formula_slots = sum(((n + 3) // 4) * 4 for n in ns)
    assert len(slots) == formula_slots == 44
    assert sum(ns) == 32 and F(len(slots), sum(ns)) == F(11, 8)
    wrong_candidate_slots = len(ns) * max(ns)
    assert wrong_candidate_slots == 128 and wrong_candidate_slots != formula_slots
    for counts in ([], [0, 0], [0, 0, 1]):
        explicit = sum(4 for n in counts for base in range(0, n, 4))
        assert explicit == sum(((n + 3) // 4) * 4 for n in counts)
    assert 3 * 4096 * 14336 * 2 == 336 * 1024 ** 2
    assert F(8 * 336, 1024) == F(21, 8)
    assert 6 * 4096 * 14336 * 32 == 11274289152
    print('cost: union probability DP matched closed form; min(Bk,E) fault detected')
    print('padding: explicit tile slots 44 distinguished from full-max candidate 128; empty groups checked')


def main():
    print('scope: deterministic CPU exact-rational synthetic MoE message model and payload ledger')
    print('not measured: GPU, NCCL, real network traffic, HBM/cache, floating-point kernels, device visibility or physical cancellation drain')
    check_distributed()
    check_routing()
    check_costs()
    print('all scoped independent references and invariants matched; faulty controls detected')


if __name__ == '__main__':
    main()
