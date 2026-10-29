import math
from copy import deepcopy
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import product


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def uniform_map(hq, hkv):
    if hq <= 0 or hkv <= 0 or hq % hkv:
        raise ValueError('invalid uniform head geometry')
    return tuple(h // (hq // hkv) for h in range(hq))


def validate_tensor(name, tensor, rows, heads, width):
    if len(tensor) != rows:
        raise ValueError(name + ': token axis mismatch')
    for row in tensor:
        if len(row) != heads:
            raise ValueError(name + ': head axis mismatch')
        for vector in row:
            if len(vector) != width:
                raise ValueError(name + ': channel axis mismatch')
            if not all(math.isfinite(x) for x in vector):
                raise ValueError(name + ': nonfinite input')


def merge(left, right, faulty=False):
    m1, l1, u1 = left
    m2, l2, u2 = right
    if l1 == 0:
        return m2, l2, list(u2)
    if l2 == 0:
        return m1, l1, list(u1)
    new_m = max(m1, m2)
    r1 = 1.0 if faulty else math.exp(m1 - new_m)
    r2 = 1.0 if faulty else math.exp(m2 - new_m)
    return new_m, r1 * l1 + r2 * l2, [r1 * a + r2 * b for a, b in zip(u1, u2)]


def grouped_online(q, kh, vh, kn, vn, mapping, hkv, dk, dv,
                   visible, bias, br=2, bc=2, scale=0.5, fault=None):
    hq = len(mapping)
    mcount = len(q)
    n = len(kh)
    if min(hq, hkv, dk, dv, br, bc) <= 0:
        raise ValueError('positive widths and tile sizes required')
    if not math.isfinite(scale):
        raise ValueError('nonfinite scale')
    if any(not isinstance(m, int) or m < 0 or m >= hkv for m in mapping):
        raise ValueError('mapping outside KV axis')
    validate_tensor('Q', q, mcount, hq, dk)
    validate_tensor('Khist', kh, n, hkv, dk)
    validate_tensor('Vhist', vh, n, hkv, dv)
    validate_tensor('Knew', kn, mcount, hkv, dk)
    validate_tensor('Vnew', vn, mcount, hkv, dv)
    k = kh + kn
    v = vh + vn
    groups = [[] for _ in range(hkv)]
    for h, m in enumerate(mapping):
        groups[m].append(h)
    out = [[[0.0] * dv for _ in range(hq)] for _ in q]
    empty = [[True] * hq for _ in q]
    flops = [0] * hq
    loads = 0

    def seen(i, h, j, kvhead):
        if fault == 'shared_mask':
            h = groups[kvhead][0]
        return visible(i, h, j)

    for q0 in range(0, mcount, br):
        rows = range(q0, min(q0 + br, mcount))
        states = {(i, h): (-math.inf, 0.0, [0.0] * dv)
                  for i in rows for h in range(hq)}
        for k0 in range(0, len(k), bc):
            tile = range(k0, min(k0 + bc, len(k)))
            for kvhead, heads in enumerate(groups):
                needed = [j for j in tile
                          if any(seen(i, h, j, kvhead) for i in rows for h in heads)]
                if not needed:
                    continue
                ktile = {j: tuple(k[j][kvhead]) for j in needed}
                vtile = {j: tuple(v[j][kvhead]) for j in needed}
                loads += len(needed)
                for h in heads:
                    for i in rows:
                        chosen = [j for j in tile if seen(i, h, j, kvhead)]
                        block = (-math.inf, 0.0, [0.0] * dv)
                        if chosen:
                            scores = []
                            for j in chosen:
                                dot = 0.0
                                for c in range(dk):
                                    dot += q[i][h][c] * ktile[j][c]
                                    flops[h] += 2
                                z = scale * dot + bias(i, h, j)
                                if not math.isfinite(z):
                                    raise ValueError('nonfinite visible score')
                                scores.append(z)
                            mb = max(scores)
                            weights = [math.exp(z - mb) for z in scores]
                            ub = [0.0] * dv
                            for t, j in enumerate(chosen):
                                for c in range(dv):
                                    ub[c] += weights[t] * vtile[j][c]
                                    flops[h] += 2
                            block = (mb, math.fsum(weights), ub)
                        states[i, h] = merge(states[i, h], block, fault == 'unscaled_merge')
        for i in rows:
            for h in range(hq):
                _, den, numerator = states[i, h]
                if den:
                    out[i][h] = [x / den for x in numerator]
                    empty[i][h] = False
    return out, empty, tuple(flops), loads


def dense_decimal_reference(q, kh, vh, kn, vn, mapping, dv,
                            visible, bias, scale=0.5):
    # Independent path: expand by the intended mapping, then normalize a whole row.
    # It does not call merge or the grouped kernel's visibility adapter.
    k = kh + kn
    v = vh + vn
    expanded_k = [[list(row[mapping[h]]) for h in range(len(mapping))] for row in k]
    expanded_v = [[list(row[mapping[h]]) for h in range(len(mapping))] for row in v]
    out = [[[0.0] * dv for _ in mapping] for _ in q]
    empty = [[True] * len(mapping) for _ in q]
    with localcontext() as ctx:
        ctx.prec = 65
        a = Decimal(str(scale))
        for i, row in enumerate(q):
            for h, query in enumerate(row):
                indices = [j for j in range(len(k)) if visible(i, h, j)]
                if not indices:
                    continue
                logits = []
                for j in indices:
                    dot = sum((Decimal(str(x)) * Decimal(str(y))
                               for x, y in zip(query, expanded_k[j][h])), Decimal(0))
                    logits.append(a * dot + Decimal(str(bias(i, h, j))))
                maximum = max(logits)
                weights = [(z - maximum).exp() for z in logits]
                den = sum(weights, Decimal(0))
                for c in range(dv):
                    numerator = sum((w * Decimal(str(expanded_v[j][h][c]))
                                     for w, j in zip(weights, indices)), Decimal(0))
                    out[i][h][c] = float(numerator / den)
                empty[i][h] = False
    return out, empty


def compare_outputs(actual, expected, tolerance=3e-11):
    require(len(actual) == len(expected), 'output row count mismatch')
    maximum = 0.0
    for ar, er in zip(actual, expected):
        require(len(ar) == len(er), 'output head count mismatch')
        for av, ev in zip(ar, er):
            require(len(av) == len(ev), 'output width mismatch')
            for a, e in zip(av, ev):
                require(math.isfinite(a) and math.isfinite(e), 'nonfinite comparison')
                delta = abs(a - e)
                maximum = max(maximum, delta)
                require(delta <= tolerance * max(1.0, abs(e)), 'attention reference mismatch')
    return maximum


def compare_result(actual, reference):
    require(actual[1] == reference[1], 'empty-row contract mismatch')
    return compare_outputs(actual[0], reference[0])


def expect_rejected(action, error=AssertionError):
    try:
        action()
    except error:
        return
    raise AssertionError('negative control was not rejected')


def independent_work(q, key_count, dv, visible):
    # Enumerate semantic scalar operations, without a FLOP product formula.
    counts = [0] * (len(q[0]) if q else 4)
    for i, row in enumerate(q):
        for h, query in enumerate(row):
            for j in range(key_count):
                if visible(i, h, j):
                    for _ in query:
                        counts[h] += 2
                    for _ in range(dv):
                        counts[h] += 2
    return tuple(counts)


def independent_loads(mcount, key_count, mapping, hkv, br, visible):
    # A logical vector pair is needed if any consumer in this query block sees it.
    events = set()
    for i in range(mcount):
        for h, kvhead in enumerate(mapping):
            for j in range(key_count):
                if visible(i, h, j):
                    events.add((i // br, kvhead, j))
    require(all(m < hkv for _, m, _ in events), 'load oracle invalid head')
    return len(events)


def make_data(mcount, history, hkv, dk, dv):
    q = [[[(3 * i + 2 * h + c) % 11 / 4.0 - 1.25
           for c in range(dk)] for h in range(4)] for i in range(mcount)]
    k = [[[(5 * j + 3 * h + 2 * c) % 13 / 8.0 - 0.75
           for c in range(dk)] for h in range(hkv)] for j in range(history + mcount)]
    v = [[[((j + 1) * (h + 2) * (c + 1)) / 8.0 + (-1) ** (j + c) / 4.0
           for c in range(dv)] for h in range(hkv)] for j in range(history + mcount)]
    return q, k[:history], v[:history], k[history:], v[history:]


def scalar_matmul_count(rows, inner, columns):
    # Actual small dense matrices and actual multiplication/accumulator events.
    a = [[((i + 2 * k) % 5 - 2) / 4.0 for k in range(inner)] for i in range(rows)]
    w = [[((3 * k + j) % 7 - 3) / 8.0 for j in range(columns)] for k in range(inner)]
    result = [[0.0] * columns for _ in range(rows)]
    multiplies = 0
    additions = 0
    for i in range(rows):
        for j in range(columns):
            for k in range(inner):
                term = a[i][k] * w[k][j]
                multiplies += 1
                result[i][j] += term
                additions += 1
    require(all(math.isfinite(x) for row in result for x in row), 'projection nonfinite')
    return multiplies + additions


def projection_path(mcount, hidden, hq, hkv, dk, dv, fault=None):
    qheads = hkv if fault == 'Q_uses_Hkv' else hq
    oheads = hkv if fault == 'O_uses_Hkv' else hq
    geometries = {
        'Q': (mcount, hidden, qheads * dk),
        'K': (mcount, hidden, hkv * dk),
        'V': (mcount, hidden, hkv * dv),
        'O': (mcount, oheads * dv, hidden),
    }
    counts = {name: scalar_matmul_count(*geometry) for name, geometry in geometries.items()}
    return geometries, counts


def projection_axis_oracle(mcount, hidden, hq, hkv, dk, dv):
    # Independent semantic axes: token/head/channel/hidden, not matrix dimension products.
    counts = {}
    for name, heads, channels in (('Q', hq, dk), ('K', hkv, dk),
                                  ('V', hkv, dv), ('O', hq, dv)):
        count = 0
        for _ in product(range(mcount), range(heads), range(channels), range(hidden)):
            count += 2
        counts[name] = count
    qchannels = list(product(range(hq), range(dk)))
    kchannels = list(product(range(hkv), range(dk)))
    vchannels = list(product(range(hkv), range(dv)))
    ochannels = list(product(range(hq), range(dv)))
    geometries = {
        'Q': (mcount, hidden, len(qchannels)),
        'K': (mcount, hidden, len(kchannels)),
        'V': (mcount, hidden, len(vchannels)),
        'O': (mcount, len(ochannels), hidden),
    }
    return geometries, counts


def check_projection(candidate, reference):
    geometry, counts = candidate
    expected_geometry, expected_counts = reference
    # Check counts first: faults must alter the same executed counting path.
    require(counts == expected_counts, 'per-projection scalar work mismatch')
    require(geometry == expected_geometry, 'projection semantic axes mismatch')


def local_kv_sets(hq, hkv, tp, faulty=False):
    if hq % tp:
        raise ValueError('query partition must be divisible in this model')
    if faulty:
        width = hkv // tp
        return tuple(tuple(range(r * width, (r + 1) * width)) for r in range(tp))
    mapping = uniform_map(hq, hkv)
    local_queries = hq // tp
    return tuple(tuple(sorted({mapping[h]
                               for h in range(r * local_queries, (r + 1) * local_queries)}))
                 for r in range(tp))


def main():
    masks = (
        ({0, 2}, {1}, set(), {0, 4, 6}),
        (set(), {0, 1, 3, 4}, {2}, {5}),
        ({1, 4, 6}, {0, 2}, {6}, set()),
    )
    visible = lambda i, h, j: j in masks[i][h]
    bias = lambda i, h, j: (i - 2 * h + 3 * j) / 8.0
    require(uniform_map(4, 2) == (0, 0, 1, 1), 'continuous grouping mismatch')
    cases = (
        ('GQA_dK3_dV2', (0, 0, 1, 1), 2, 3, 2),
        ('GQA_dK2_dV3', (0, 0, 1, 1), 2, 2, 3),
        ('MQA', (0, 0, 0, 0), 1, 3, 2),
        ('MHA', (0, 1, 2, 3), 4, 3, 2),
        ('explicit_unequal_groups', (1, 0, 1, 1), 2, 3, 2),
    )
    largest_delta = 0.0
    executions = 0
    for name, mapping, hkv, dk, dv in cases:
        data = make_data(3, 4, hkv, dk, dv)
        frozen = deepcopy(data)
        reference = dense_decimal_reference(*data, mapping, dv, visible, bias)
        work_reference = independent_work(data[0], 7, dv, visible)
        pairs = [sum(1 for i in range(3) for j in range(7) if visible(i, h, j))
                 for h in range(4)]
        require(tuple(2 * (dk + dv) * p for p in pairs) == work_reference,
                'head-dependent attention formula mismatch')
        for br in (1, 2, 4):
            for bc in (1, 2, 4, 9):
                actual = grouped_online(*data, mapping, hkv, dk, dv, visible, bias, br, bc)
                largest_delta = max(largest_delta, compare_result(actual, reference))
                require(actual[2] == work_reference, name + ': executed work mismatch')
                require(actual[3] == independent_loads(3, 7, mapping, hkv, br, visible),
                        name + ': logical vector-pair loads mismatch')
                executions += 1
        require(data == frozen, 'attention mutated caller state')

    # Empty history with real new candidates, all-empty rows, and zero query.
    for mcount, history, vis in (
        (2, 0, lambda i, h, j: j == 0 and h % 2 == 0),
        (2, 2, lambda i, h, j: False),
        (0, 0, lambda i, h, j: False),
    ):
        data = make_data(mcount, history, 2, 3, 2)
        actual = grouped_online(*data, (0, 0, 1, 1), 2, 3, 2, vis, bias, 2, 3)
        reference = dense_decimal_reference(*data, (0, 0, 1, 1), 2, vis, bias)
        compare_result(actual, reference)
        require(actual[3] == independent_loads(mcount, history + mcount,
                                             (0, 0, 1, 1), 2, 2, vis),
                'edge-case load accounting mismatch')

    # Finite large logits test stable normalization and online rescaling.
    data = make_data(3, 4, 2, 3, 2)
    mapping = (0, 0, 1, 1)
    extreme_bias = lambda i, h, j: 900.0 if j % 2 else -900.0
    reference_extreme = dense_decimal_reference(*data, mapping, 2, visible, extreme_bias)
    compare_result(grouped_online(*data, mapping, 2, 3, 2, visible, extreme_bias, 2, 1),
                   reference_extreme)

    reference = dense_decimal_reference(*data, mapping, 2, visible, bias)
    bad_map = grouped_online(*data, (0, 1, 0, 1), 2, 3, 2, visible, bias)
    expect_rejected(lambda: compare_result(bad_map, reference))
    bad_mask = grouped_online(*data, mapping, 2, 3, 2, visible, bias, fault='shared_mask')
    expect_rejected(lambda: compare_result(bad_mask, reference))
    bad_merge = grouped_online(*data, mapping, 2, 3, 2, visible, extreme_bias,
                               bc=1, fault='unscaled_merge')
    expect_rejected(lambda: compare_result(bad_merge, reference_extreme))

    # Both history and new K must use dK, not dV. Faults enter the real validator.
    for which in (1, 3):
        malformed = list(deepcopy(data))
        malformed[which] = [[vector[:2] for vector in row] for row in malformed[which]]
        expect_rejected(lambda malformed=malformed: grouped_online(
            *malformed, mapping, 2, 3, 2, visible, bias), ValueError)
    nonfinite = list(deepcopy(data))
    nonfinite[0][0][0][0] = math.nan
    expect_rejected(lambda: grouped_online(*nonfinite, mapping, 2, 3, 2, visible, bias), ValueError)
    expect_rejected(lambda: compare_outputs([[[math.nan]]], [[[0.0]]]))

    projection_observations = []
    for spec in ((2, 5, 4, 2, 3, 2), (1, 4, 4, 1, 2, 3),
                 (1, 3, 4, 4, 2, 2), (0, 5, 4, 2, 3, 2)):
        candidate = projection_path(*spec)
        oracle = projection_axis_oracle(*spec)
        check_projection(candidate, oracle)
        mcount, hidden, hq, hkv, dk, dv = spec
        formula = 2 * mcount * hidden * (hq + hkv) * (dk + dv)
        require(sum(candidate[1].values()) == formula, 'general projection formula mismatch')
        if dk == dv:
            require(sum(candidate[1].values()) == 4 * mcount * hidden * dk * (hq + hkv),
                    'equal-width projection formula mismatch')
        projection_observations.append(candidate[1])
    spec = (2, 5, 4, 2, 3, 2)
    projection_reference = projection_axis_oracle(*spec)
    for fault in ('Q_uses_Hkv', 'O_uses_Hkv'):
        bad_projection = projection_path(*spec, fault=fault)
        expect_rejected(lambda bad_projection=bad_projection:
                        check_projection(bad_projection, projection_reference))

    sets24 = local_kv_sets(24, 6, 4)
    expected24 = ((0, 1), (1, 2), (3, 4), (4, 5))
    require(sets24 == expected24, 'global-to-local TP mapping mismatch')
    require(sum(map(len, sets24)) == 8, 'TP24 physical copies mismatch')
    faulty_sets = local_kv_sets(24, 6, 4, faulty=True)
    expect_rejected(lambda: require(faulty_sets == expected24, 'floor TP allocation mismatch'))
    sets32 = local_kv_sets(32, 8, 16)
    require(sets32 == tuple((r // 2,) for r in range(16)), 'TP16 mapping mismatch')

    # Resource arithmetic only: no corresponding buffers or device transactions.
    head_bytes = 32 * (128 + 128) * 2
    logical_token_bytes = 8 * head_bytes
    physical_token_bytes = sum(len(s) * head_bytes for s in sets32)
    require(logical_token_bytes == 131072, 'logical KV arithmetic mismatch')
    require(logical_token_bytes * 4096 == 536870912, '512 MiB arithmetic mismatch')
    require(head_bytes == 16384, 'MQA or per-rank head bytes mismatch')
    require(physical_token_bytes == 262144, 'TP16 physical bytes mismatch')
    effective_bits = Fraction(2 * 32 + 8 * 4, 32)
    require(effective_bits == 3, 'effective-bit arithmetic mismatch')

    print('attention configurations executed:', executions, 'maximum Decimal-reference delta:', largest_delta)
    print('unequal per-head visible pairs:', (5, 7, 2, 4), 'GQA dK3/dV2 main FLOPs:', 180)
    print('projection Q/K/V/O scalar-event counts:', projection_observations)
    print('negative controls rejected: wrong mapping, shared mask, unscaled merge, history/new K width, NaN, Q width, O width, floor TP allocation')
    print('TP24 actual local KV sets:', sets24)
    print('resource arithmetic only: logical bytes/token, bytes at 4096, TP16 physical bytes/token:',
          logical_token_bytes, logical_token_bytes * 4096, physical_token_bytes)
    print('scope: deterministic CPU binary64/Decimal arithmetic; no GPU/Neuron, device visibility, physical traffic, collective, quantization format or model-quality measurement')


if __name__ == '__main__':
    main()
