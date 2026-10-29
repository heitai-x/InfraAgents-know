import math
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import permutations, product

TOL = 1e-12


def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('negative control was accepted')


def matmul(a, b):
    assert len(a[0]) == len(b)
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def ranges(n, p):
    if p < 1 or n < p or n % p:
        raise ValueError('uniform split unsupported')
    step = n // p
    return [(r * step, (r + 1) * step) for r in range(p)]


def check_tp():
    x = [[1, 2, 3, 4, 5, 6], [0, -1, 2, 1, 0, 1]]
    w = [[1, 0, 2, -1, 0, 1], [0, 1, 1, 2, -1, 0],
         [2, -1, 0, 1, 1, 0], [1, 1, 1, 0, 0, 2],
         [0, 2, -1, 1, 2, 0], [3, 0, 1, -2, 1, 1]]
    ref = [[29, 13, 9, -1, 17, 15], [8, -2, 1, -2, 4, 3]]
    assert matmul(x, w) == ref
    for p in (1, 2, 3, 6):
        cuts = ranges(6, p)
        cols = [matmul(x, [row[lo:hi] for row in w]) for lo, hi in cuts]
        assert [sum((part[i] for part in cols), []) for i in range(2)] == ref
        parts = [matmul([row[lo:hi] for row in x], w[lo:hi])
                 for lo, hi in cuts]
        got = [[sum(part[i][j] for part in parts) for j in range(6)]
               for i in range(2)]
        assert got == ref
        if p > 1:
            assert parts[0] != ref
            residual = [[1] * 6 for _ in x]
            correct = [[got[i][j] + residual[i][j] for j in range(6)]
                       for i in range(2)]
            wrong = [[got[i][j] + p * residual[i][j] for j in range(6)]
                     for i in range(2)]
            assert wrong != correct
    rejects(lambda: ranges(6, 4))
    rejects(lambda: ranges(6, 0))
    print('TP: exact output, column order, row sum and residual checked; missing-reduce and repeated-residual faults detected')


def ring_simulation(p, chunk=2, element_bytes=4):
    n = p * chunk
    state = [[[100 * r + c * chunk + j for j in range(chunk)]
              for c in range(p)] for r in range(p)]
    sent = [0] * p
    for step in range(p - 1):
        messages = [((r + 1) % p, (r - step) % p,
                     state[r][(r - step) % p][:]) for r in range(p)]
        for r, (dst, c, values) in enumerate(messages):
            state[dst][c] = [a + b for a, b in zip(state[dst][c], values)]
            sent[r] += len(values) * element_bytes
    for step in range(p - 1):
        messages = [((r + 1) % p, (r + 1 - step) % p,
                     state[r][(r + 1 - step) % p][:]) for r in range(p)]
        for r, (dst, c, values) in enumerate(messages):
            state[dst][c] = values
            sent[r] += len(values) * element_bytes
    ref = [sum(100 * r + j for r in range(p)) for j in range(n)]
    assert all(sum(blocks, []) == ref for blocks in state)
    assert all(v == Fraction(2 * (p - 1), p) * n * element_bytes for v in sent)
    return sent


def check_ring():
    for p in (1, 2, 3, 4):
        ring_simulation(p)
    a = 16 * 1024 * 1024
    one = Fraction(2 * 3, 4) * a
    assert one == 25165824 and 2 * one == 50331648
    assert Fraction(2, 4) * a != one
    assert ring_simulation(1) == [0]
    print('Ring: message simulation reconstructs all values for p=1,2,3,4; one/two 16-MiB reductions send 25165824/50331648 bytes per rank')


def pipeline(costs, m):
    if not costs or any(c <= 0 for c in costs) or m < 0:
        raise ValueError('invalid forward pipeline')
    if m == 0:
        return 0
    finish = [[0] * m for _ in costs]
    for s, cost in enumerate(costs):
        for j in range(m):
            finish[s][j] = max(finish[s - 1][j] if s else 0,
                               finish[s][j - 1] if j else 0) + cost
    return finish[-1][-1]


def check_pp():
    for p in range(1, 6):
        for m in range(1, 9):
            assert pipeline([2] * p, m) == (m + p - 1) * 2
    for costs in ([1, 3, 2], [5, 1], [2, 2, 2, 2]):
        for m in range(1, 7):
            assert pipeline(costs, m) == sum(costs) + (m - 1) * max(costs)
    assert pipeline([1, 3, 2], 4) == 15
    assert pipeline([1, 1, 1], 4) == 6
    assert pipeline([1, 1, 1], 4) != 4 + 3
    assert pipeline([1, 2], 0) == 0
    rejects(lambda: pipeline([], 1))
    rejects(lambda: pipeline([0], 1))
    print('PP: forward-only recurrence agrees with equal/unequal-stage invariants and empty work; off-by-one fault detected')


def validate_partition(parts, n):
    flat = [j for part in parts for j in part]
    if not parts or any(type(j) is not int for j in flat):
        raise ValueError('invalid partition identity')
    if Counter(flat) != Counter(range(n)):
        raise ValueError('missing, duplicate or foreign key')


def empty_stats(d):
    return (-math.inf, 0.0, (0.0,) * d)


def local_stats(scores, values, ids, allowed):
    m, total, acc = empty_stats(len(values[0]))
    for j in ids:
        if j not in allowed:
            continue
        score = scores[j]
        if score > m:
            carry = 0.0 if total == 0 else math.exp(m - score)
            total *= carry
            acc = tuple(v * carry for v in acc)
            m = score
        weight = math.exp(score - m)
        total += weight
        acc = tuple(a + weight * v for a, v in zip(acc, values[j]))
    return (m, total, acc)


def merge_stats(a, b):
    if a[1] == 0:
        return b
    if b[1] == 0:
        return a
    m = max(a[0], b[0])
    ca, cb = math.exp(a[0] - m), math.exp(b[0] - m)
    return (m, ca * a[1] + cb * b[1],
            tuple(ca * x + cb * y for x, y in zip(a[2], b[2])))


def normalize(stats):
    if stats[1] == 0:
        return None
    return tuple(v / stats[1] for v in stats[2])


def split_attention(scores, values, parts, allowed, checked=True):
    if checked:
        validate_partition(parts, len(scores))
    result = empty_stats(len(values[0]))
    for part in parts:
        result = merge_stats(result, local_stats(scores, values, part, allowed))
    return normalize(result)


def dense_decimal(scores, values, allowed):
    ids = sorted(allowed)
    if not ids:
        return None
    with localcontext() as ctx:
        ctx.prec = 70
        ds = [Decimal(str(scores[j])) for j in ids]
        maximum = max(ds)
        weights = [(s - maximum).exp() for s in ds]
        denominator = sum(weights, Decimal(0))
        return tuple(float(sum((weights[i] * Decimal(str(values[j][c]))
                                for i, j in enumerate(ids)), Decimal(0)) / denominator)
                     for c in range(len(values[0])))


def equal_output(got, ref):
    if got is None or ref is None:
        return got is None and ref is None
    return len(got) == len(ref) and all(abs(a - b) <= TOL for a, b in zip(got, ref))


def check_cp():
    q = [0.6, -0.2, 0.4]
    keys = [[0.2, -0.4, 0.7], [1.0, 0.1, -0.3], [-0.5, 0.8, 0.2],
            [0.3, 0.3, -0.6], [0.9, -0.7, 0.1]]
    values = [[1.5, -2], [-0.5, 3], [2, 1], [0.25, -4], [-1, 2]]
    scores = [sum(a * b for a, b in zip(q, k)) / math.sqrt(len(q)) for k in keys]
    masks = []
    for pos in range(-1, 5):
        for window in (None, 2):
            allowed = {j for j in range(5)
                       if j <= pos and (window is None or j > pos - window)}
            masks.append((allowed, dense_decimal(scores, values, allowed)))
    comparisons = 0
    for p in (1, 2, 3):
        for assignment in product(range(p), repeat=5):
            parts = [[j for j, r in enumerate(assignment) if r == rank]
                     for rank in range(p)]
            for allowed, ref in masks:
                for order in (parts, list(reversed(parts))):
                    assert equal_output(split_attention(scores, values, order, allowed), ref)
                    comparisons += 1
    assert comparisons == 6624
    allowed = set(range(5))
    ref = dense_decimal(scores, values, allowed)
    legacy_bad = [[[0, 1], [2, 3], []], [[0, 2, 4], [1, 4], [3]]]
    bad_outputs = []
    for parts in legacy_bad:
        rejects(lambda parts=parts: split_attention(scores, values, parts, allowed))
        unchecked = split_attention(scores, values, parts, allowed, checked=False)
        assert any(abs(a - b) > 1e-6 for a, b in zip(unchecked, ref))
        bad_outputs.append(unchecked)
    rejects(lambda: split_attention(scores, values, legacy_bad[1], {0}))
    rejects(lambda: validate_partition([[0, 1, 2, 3, 5]], 5))
    locals_ = [normalize(local_stats(scores, values, ids, allowed))
               for ids in ([0], [1], [2, 3, 4])]
    wrong_mean = tuple(sum(v[c] for v in locals_) / len(locals_)
                       for c in range(2))
    assert any(abs(a - b) > 1e-6 for a, b in zip(wrong_mean, ref))
    st = local_stats(scores, values, range(5), allowed)
    identity = empty_stats(2)
    assert merge_stats(identity, st) == st == merge_stats(st, identity)
    assert merge_stats(identity, identity) == identity
    extreme_scores = [-1000.0, 1000.0, 999.0]
    extreme_values = [[1, 2], [-1, 3], [2, -4]]
    ext_ref = dense_decimal(extreme_scores, extreme_values, {0, 1, 2})
    for order in permutations(range(3)):
        parts = [[j] for j in order] + [[]]
        got = split_attention(extreme_scores, extreme_values, parts, {0, 1, 2})
        assert equal_output(got, ext_ref)
        assert all(math.isfinite(v) for v in got)
    sts = [local_stats(extreme_scores, extreme_values, [j], {0, 1, 2})
           for j in range(3)]
    wrong_unscaled = tuple(sum(s[2][c] for s in sts) / sum(s[1] for s in sts)
                           for c in range(2))
    assert not equal_output(wrong_unscaled, ext_ref)
    assert split_attention(scores, values, [[], list(range(5))], set()) is None
    print('CP:', comparisons, 'comparisons with independent 70-digit Decimal dense softmax; absolute tolerance', TOL)
    print('CP: legacy missing/duplicate partitions rejected; unchecked outputs', bad_outputs,
          'differ from full output', ref)
    print('CP: global causal/window masks, empty identity and extreme scores checked; unweighted and unscaled merge faults detected')


def expert(x, e):
    return tuple((e + 1) * v + (i + 1) * e for i, v in enumerate(x))


def ep_reference(tokens, routes):
    return {key: tuple(sum((g * expert(x, e)[i] for e, g in routes[key]), Fraction(0))
                       for i in range(len(x))) for key, x in tokens.items()}


def ep_records(tokens, routes, owner, premerge):
    buckets = {}
    for key, x in tokens.items():
        for e, g in routes[key]:
            buckets.setdefault((owner[e], e), []).append((key, x, g))
    records = []
    for (dst, e), items in sorted(buckets.items()):
        for key, x, g in items:
            records.append((key, dst, frozenset((e,)), tuple(g * v for v in expert(x, e))))
    if not premerge:
        return records
    groups = {}
    for key, dst, covered, value in records:
        identity = (key, dst)
        if identity not in groups:
            groups[identity] = (covered, value)
        else:
            old, acc = groups[identity]
            assert not old.intersection(covered)
            groups[identity] = (old.union(covered), tuple(a + b for a, b in zip(acc, value)))
    return [(key, dst, covered, value)
            for (key, dst), (covered, value) in groups.items()]


def ep_collect(tokens, routes, records):
    expected = {key: {e for e, _ in routes[key]} for key in tokens}
    if any(len(expected[key]) != len(routes[key]) for key in tokens):
        raise ValueError('duplicate selected expert')
    seen = {key: set() for key in tokens}
    out = {key: tuple(Fraction(0) for _ in x) for key, x in tokens.items()}
    for key, dst, covered, value in records:
        if key not in expected or not covered or not covered.issubset(expected[key]):
            raise ValueError('foreign contribution')
        if seen[key].intersection(covered) or len(value) != len(out[key]):
            raise ValueError('duplicate or malformed contribution')
        seen[key].update(covered)
        out[key] = tuple(a + b for a, b in zip(out[key], value))
    if any(seen[key] != expected[key] for key in tokens):
        raise ValueError('missing contribution')
    return out


def check_ep():
    f = Fraction
    tokens = {(0, 0): (f(1), f(2)), (0, 1): (f(2), f(-1)),
              (1, 0): (f(-1), f(3)), (2, 0): (f(1, 2), f(2))}
    routes = {(0, 0): [(1, f(1, 3)), (2, f(2, 3))],
              (0, 1): [(0, f(1, 4)), (3, f(3, 4))],
              (1, 0): [(0, f(1, 2)), (2, f(1, 2))],
              (2, 0): [(1, f(2, 5)), (3, f(3, 5))]}
    owner = {0: 0, 1: 1, 2: 1, 3: 2}
    ref = ep_reference(tokens, routes)
    assert ref[(0, 0)] == (f(13, 3), f(26, 3))
    assert ref[(1, 0)] == (f(-1), f(8))
    observed_bytes = []
    raw = ep_records(tokens, routes, owner, False)
    remote = [r for r in raw if r[0][0] != r[1]]
    assert len(remote) == 5
    pairs = {(key, dst) for key, dst, _, _ in remote}
    assert len(pairs) == 4
    for premerge in (False, True):
        records = ep_records(tokens, routes, owner, premerge)
        assert ep_collect(tokens, routes, records) == ref
        assert ep_collect(tokens, routes, list(reversed(records))) == ref
        rejects(lambda records=records: ep_collect(tokens, routes, records[1:]))
        rejects(lambda records=records: ep_collect(tokens, routes, records + [records[0]]))
        for dedup in (False, True):
            dispatch_count = len(pairs) if dedup else len(remote)
            return_count = sum(key[0] != dst for key, dst, _, _ in records)
            observed_bytes.append((dispatch_count + return_count) * 2 * 2)
    assert observed_bytes == [40, 36, 36, 32]
    assert ep_collect({}, {}, []) == {}
    assert ep_reference({}, {}) == {}
    print('EP: exact Fraction reference preserves origin/token/expert coverage; missing and duplicate returns rejected')
    print('EP: fixed-route two-channel toy payloads for plain/dedup/premerge/both:', observed_bytes,
          'bytes; router, real FFN kernels and transport are outside this model')


def check_combination():
    kib, mib, gib = 1024, 1024 ** 2, 1024 ** 3
    per_token = 32 * (512 + 64) * 2
    assert per_token == 36 * kib
    prompt = 16 * 8192 * per_token
    growth = 16 * 128 * per_token
    assert prompt == Fraction(9, 2) * gib and growth == 72 * mib
    ab = Fraction(12 + 2) + Fraction(prompt + growth, gib)
    c = Fraction(12 + 2) + Fraction(prompt + growth, 2 * gib)
    assert ab == Fraction('18.5703125') and c == Fraction('16.28515625')
    assert ab + 2 == Fraction('20.5703125')
    items = []
    for t in range(16):
        origin = t % 4
        destinations = [origin, (origin + 1) % 4] if t < 8 else [(origin + 1) % 4, (origin + 2) % 4]
        items.extend((t, origin, dst) for dst in destinations)
    assert len(items) == 32
    remote = [item for item in items if item[1] != item[2]]
    assert len(remote) == 24
    assert Counter(origin for _, origin, _ in remote) == Counter({r: 6 for r in range(4)})
    ep_bytes = len(remote) * 4096 * 2 * 2
    gather_bytes = 4 * 3 * 4 * 4096 * 2
    assert ep_bytes == gather_bytes == 384 * kib
    gathered_ids = [t for r in range(4) for t in range(16) if t % 4 == r]
    assert gathered_ids != list(range(16))
    inverse = {token: slot for slot, token in enumerate(gathered_ids)}
    assert [gathered_ids[inverse[t]] for t in range(16)] == list(range(16))
    assert ep_bytes + gather_bytes == 768 * kib
    a_block = 4 * 2 * Fraction(2 * 3, 4) * (16 * 4096 * 2)
    b_block = 4 * Fraction(2 * 3, 4) * (16 * 4096 * 2) + ep_bytes + gather_bytes
    assert a_block == b_block == Fraction(3, 2) * mib
    print('Combination: per-rank growth and capacity checked; EP plus TP-output restoration sends 768 KiB group-wide')
    print('Combination: rank-ordered gathered rows require an explicit token mapping; treating rank order as token order detected')


def main():
    check_tp()
    check_ring()
    check_pp()
    check_cp()
    check_ep()
    check_combination()
    print('Scope: deterministic CPU algebra/message/coverage models only; no GPU measurements, upstream imports, network, subprocesses or file writes')


if __name__ == '__main__':
    main()
