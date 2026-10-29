#!/usr/bin/env python3
"""K-ALG-021: deterministic, read-only, offline stdlib mechanism models.
No project imports, files, subprocesses, network, GPU, or performance measurements.
"""
from fractions import Fraction as F
from itertools import combinations
from math import exp, isfinite

NEGATIVES = []


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def equal(actual, expected):
    require(actual == expected, 'independent reference mismatch')


def detect_difference(name, actual, expected):
    try:
        equal(actual, expected)
    except AssertionError:
        NEGATIVES.append(name)
    else:
        raise AssertionError('negative control survived: ' + name)


def must_raise(name, function, exception):
    try:
        function()
    except exception:
        NEGATIVES.append(name)
    else:
        raise AssertionError('expected rejection missing: ' + name)


def rational(x):
    if not isinstance(x, (int, F)):
        raise ValueError('model accepts exact int/Fraction inputs only')
    return F(x)


def group(values, qmin=0, qmax=3, wrong_offset=False):
    if not values or qmin >= qmax:
        raise ValueError('empty group or invalid code range')
    v = [rational(x) for x in values]
    low, high = min(v), max(v)
    if low == high:
        return tuple(qmin for _ in v), F(1), low - qmin
    scale = (high - low) / (qmax - qmin)
    bias = low if wrong_offset else low - scale * qmin
    codes = []
    for x in v:
        q = ((x - bias) / scale + F(1, 2)).__floor__()
        codes.append(max(qmin, min(qmax, q)))
    return tuple(codes), scale, bias


def decode_group(record):
    codes, scale, bias = record
    return [bias + scale * q for q in codes]


def grid_reference(values, x, qmin=0, qmax=3):
    # Enumerate reconstructed endpoint grids, without encoder scale/bias/rounding.
    low, high = min(values), max(values)
    candidates = [(low + (high - low) * F(i, qmax - qmin), qmin + i)
                  for i in range(qmax - qmin + 1)]
    return min(candidates, key=lambda item: (abs(x - item[0]), -item[1]))[0]


def new_state(d, G, R):
    if d < 1 or G < 1 or R < 1 or R % G:
        raise ValueError('invalid geometry')
    return dict(d=d, G=G, R=R, n=0, kb=0, vb=0,
                Kq={}, Vq={}, Kr=[], Vr=[])


def append(state, Krow, Vrow):
    d, G, R = state['d'], state['G'], state['R']
    if len(Krow) != d or len(Vrow) != d:
        raise ValueError('row width mismatch')
    state['Kr'].append([rational(x) for x in Krow])
    state['Vr'].append([rational(x) for x in Vrow])
    if len(state['Kr']) == R:
        for u in range(0, R, G):
            for c in range(d):
                state['Kq'][((state['kb'] + u) // G, c)] = group(
                    [state['Kr'][i][c] for i in range(u, u + G)])
        state['kb'] += R
        state['Kr'] = []
    if len(state['Vr']) > R:
        row = state['Vr'].pop(0)
        for h in range((d + G - 1) // G):
            state['Vq'][(state['vb'], h)] = group(row[h * G:min(d, (h + 1) * G)])
        state['vb'] += 1
    state['n'] += 1


def kivi_quant(K, V, G, R, d=None):
    if len(K) != len(V):
        raise ValueError('K/V lengths differ')
    width = d if d is not None else (len(K[0]) if K else 1)
    state = new_state(width, G, R)
    for k, v in zip(K, V):
        append(state, k, v)
    return state  # One dictionary, including for empty input.


def rebuild(state, wrong_v_boundary=False):
    K, V = [], []
    d, G = state['d'], state['G']
    for j in range(state['n']):
        kr, vr = [], []
        for c in range(d):
            if j < state['kb']:
                codes, scale, bias = state['Kq'][(j // G, c)]
                kr.append(bias + scale * codes[j % G])
            else:
                kr.append(state['Kr'][j - state['kb']][c])
            boundary = state['kb'] if wrong_v_boundary else state['vb']
            if j < boundary:
                codes, scale, bias = state['Vq'][(j, c // G)]
                vr.append(bias + scale * codes[c % G])
            else:
                vr.append(state['Vr'][j - state['vb']][c])
        K.append(kr)
        V.append(vr)
    return K, V


def reference_kivi(K, V, G, R):
    L = len(K)
    if not L:
        return [], []
    d, kb, vb = len(K[0]), L - L % R, max(0, L - R)
    Kr, Vr = [], []
    for j in range(L):
        kr, vr = [], []
        for c in range(d):
            kg = [K[i][c] for i in range((j // G) * G, (j // G + 1) * G)] if j < kb else []
            vg = V[j][(c // G) * G:min(d, (c // G + 1) * G)]
            kr.append(grid_reference(kg, K[j][c]) if j < kb else K[j][c])
            vr.append(grid_reference(vg, V[j][c]) if j < vb else V[j][c])
        Kr.append(kr)
        Vr.append(vr)
    return Kr, Vr


def check_quant():
    for qmin, qmax in ((0, 3), (-2, 1)):
        for seed in range(12):
            values = [F((i * i + seed * 3 + i * seed) % 19 - 9, 4) for i in range(6)]
            equal(decode_group(group(values, qmin, qmax)),
                  [grid_reference(values, x, qmin, qmax) for x in values])
    values = [F(-1), F(0), F(2)]
    detect_difference('signed min-offset clipping',
                      decode_group(group(values, -2, 1, True)), values)
    equal(decode_group(group([F(3)] * 3, -2, 1)), [F(3)] * 3)
    must_raise('empty scalar group', lambda: group([]), ValueError)
    must_raise('invalid residual geometry', lambda: new_state(5, 3, 4), ValueError)
    Q0 = kivi_quant([], [], 2, 2)
    require(isinstance(Q0, dict), 'empty input must return one state dictionary')
    equal(rebuild(Q0), ([], []))
    require(not any(Q0[key] for key in ('Kq', 'Vq', 'Kr', 'Vr')), 'nonempty zero state')
    states = 0
    for G, R in ((2, 4), (3, 6), (1, 2)):
        state, K, V = new_state(5, G, R), [], []
        equal(rebuild(state), reference_kivi(K, V, G, R))
        states += 1
        for t in range(13):
            K.append([F((3 * t + 2 * c + t * c) % 17 - 8, 4) for c in range(5)])
            V.append([F((t * t + 3 * c + t * c) % 23 - 11, 5) for c in range(5)])
            append(state, K[-1], V[-1])
            equal(rebuild(state), reference_kivi(K, V, G, R))
            equal((state['kb'], state['vb']),
                  (len(K) - len(K) % R, max(0, len(K) - R)))
            equal(len(state['Kq']), state['kb'] // G * 5)
            equal(len(state['Vq']), state['vb'] * ((5 + G - 1) // G))
            states += 1
    state = kivi_quant(K, V, 2, 4)
    reference = rebuild(state)
    record = state['Kq'][(0, 0)]
    state['Kq'][(0, 0)] = (record[0], record[1], record[2] + 1)
    detect_difference('published parameter overwritten', rebuild(state), reference)
    state = kivi_quant(K[:8], V[:8], 2, 4)
    equal((state['kb'], state['vb']), (8, 4))
    must_raise('K boundary incorrectly used for V',
               lambda: rebuild(state, True), KeyError)
    print('observation: exact-grid quantization, empty-state return contract, axes, tails and', states, 'stream states checked')


def dense_attention(q, K, V):
    if not K:
        return [0.0] * len(q)
    weights = [exp(sum(float(a * b) for a, b in zip(q, k))) for k in K]
    total = sum(weights)
    return [sum(w * float(v[c]) for w, v in zip(weights, V)) / total for c in range(len(V[0]))]


def online_attention(q, K, V, partitions):
    mass, maximum, U = 0.0, None, [0.0] * len(q)
    for ids in partitions:
        ids = list(ids)
        if not ids:
            continue
        scores = [sum(float(a * b) for a, b in zip(q, K[j])) for j in ids]
        local_max = max(scores)
        weights = [exp(z - local_max) for z in scores]
        local_mass = sum(weights)
        local_U = [sum(w * float(V[j][c]) for w, j in zip(weights, ids)) for c in range(len(q))]
        merged_max = local_max if maximum is None else max(maximum, local_max)
        old_factor = 0.0 if maximum is None else exp(maximum - merged_max)
        new_factor = exp(local_max - merged_max)
        U = [old_factor * a + new_factor * b for a, b in zip(U, local_U)]
        mass = old_factor * mass + new_factor * local_mass
        maximum = merged_max
    return [x / mass for x in U] if mass else U


def close(a, b):
    require(len(a) == len(b), 'output width differs')
    require(all(isfinite(x) and isfinite(y) and abs(x - y) <= 1e-12 * (1 + abs(y))
                for x, y in zip(a, b)), 'attention reference mismatch')


def check_attention():
    K = [[F((t + 2 * c) % 7 - 3, 5) for c in range(5)] for t in range(13)]
    V = [[F((3 * t + c) % 11 - 5, 7) for c in range(5)] for t in range(13)]
    q = [F(c - 2, 6) for c in range(5)]
    for L in (0, 1, 3, 4, 5, 8, 9, 13):
        state = kivi_quant(K[:L], V[:L], 2, 4, d=5)
        k, v = rebuild(state)
        reference = dense_attention(q, k, v)
        partitions = [[], range(state['kb']), [], range(state['kb'], L)]
        close(online_attention(q, k, v, partitions), reference)
        close(online_attention(q, k, v, [range(i, min(L, i + 3)) for i in range(0, L, 3)]), reference)
    k, v, q1 = [[F(0)]] * 3, [[F(0)], [F(2)], [F(4)]], [F(1)]
    good = dense_attention(q1, k, v)
    bad = [dense_attention(q1, k[:1], v[:1])[0] + dense_attention(q1, k[1:], v[1:])[0]]
    equal(good, [2.0])
    detect_difference('separately normalized partitions added', bad, good)
    print('observation: online attention agrees with independent whole-row formula on differing K/V boundaries and empty partitions')


def select(A, scores, t, BH, BR):
    recent = set(range(max(0, t - BR + 1), t + 1))
    if not recent <= A:
        raise ValueError('required recent history missing')
    candidates, heavy = set(A) - recent, set()
    for _ in range(min(BH, len(candidates))):
        j = max(candidates, key=lambda x: (scores[x], -x))
        heavy.add(j)
        candidates.remove(j)
    return recent | heavy


def enumerate_selection(A, scores, t, BH, BR):
    recent = {j for j in range(t + 1) if j >= t - BR + 1}
    candidates = sorted(A - recent)
    alternatives = combinations(candidates, min(BH, len(candidates)))
    best = max(alternatives, key=lambda H: (sum((scores[j] for j in H), F(0)), tuple(-j for j in H)))
    return recent | set(best)


def h2o_step(S, scores, probability, t, BH, BR):
    if BH < 0 or BR < 0 or len(S) > BH + BR or not S <= set(range(t)):
        raise ValueError('invalid old state or budget')
    A = set(S) | {t}
    if set(probability) != A or any(x < 0 for x in probability.values()) or sum(probability.values()) != 1:
        raise ValueError('invalid probability row')
    updated = {j: scores.get(j, F(0)) + probability[j] for j in A}
    chosen = select(A, updated, t, BH, BR)
    return chosen, {j: updated[j] for j in chosen}, updated


def budget(S, B):
    if len(S) > B:
        raise ValueError('physical union exceeds budget')


def check_h2o():
    steps = 0
    for BH, BR in ((0, 0), (1, 0), (0, 1), (2, 1), (0, 4)):
        S, scores = set(), {}
        for t in range(13):
            A = S | {t}
            weights = {j: F(((j + 2) * (t + 1)) % 7 + 1) for j in A}
            total = sum(weights.values())
            probability = {j: w / total for j, w in weights.items()}
            chosen, new_scores, updated = h2o_step(S, scores, probability, t, BH, BR)
            equal(chosen, enumerate_selection(A, updated, t, BH, BR))
            budget(chosen, BH + BR)
            require(chosen <= A, 'evicted history resurrected')
            equal(set(new_scores), chosen)
            S, scores = chosen, new_scores
            steps += 1
    p = {0: F(1, 6), 1: F(2, 6), 2: F(3, 6)}
    chosen, _, _ = h2o_step({0, 1}, {}, p, 2, 1, 1)
    equal(chosen, {1, 2})
    def consume(slots):
        return sum((p[j] * (F(40) if j == 2 else slots[j]) for j in (0, 1, 2)), F(0))
    correct = consume({0: F(10), 1: F(20)})
    overwritten = consume({0: F(40), 1: F(20)})
    equal(correct, F(85, 3))
    detect_difference('evicted slot overwritten before current consumption', overwritten, correct)
    p4 = {j: F(1, 4) for j in range(4)}
    S1, _, _ = h2o_step({0, 1, 2}, {0: F(9), 1: F(8), 2: F(0)}, p4, 3, 2, 1)
    S2, _, _ = h2o_step({0, 1, 2}, {0: F(0), 1: F(8), 2: F(9)}, p4, 3, 2, 1)
    equal((S1, S2), ({0, 1, 3}, {1, 2, 3}))
    must_raise('GQA union silently counted as B rows', lambda: budget(S1 | S2, 3), ValueError)
    print('observation: H2O subset-enumeration reference checked', steps, 'steps; staging and GQA capacity faults detected')


def ac_encode(symbols, counts):
    if not counts or any(not isinstance(x, int) or x <= 0 for x in counts):
        raise ValueError('invalid frequency table')
    M, low, high = sum(counts), F(0), F(1)
    for symbol in symbols:
        if not isinstance(symbol, int) or not 0 <= symbol < len(counts):
            raise ValueError('symbol out of range')
        cumulative = sum(counts[:symbol])
        width, old_low = high - low, low
        low = old_low + width * F(cumulative, M)
        high = old_low + width * F(cumulative + counts[symbol], M)
    n = 0
    while True:
        scale = 1 << n
        k = (low.numerator * scale + low.denominator - 1) // low.denominator
        if F(k, scale) < high:
            return k, n, len(symbols)
        n += 1


def ac_decode(tag, counts):
    k, n, length = tag
    if n < 0 or length < 0 or not 0 <= k < (1 << n) or not counts or any(x <= 0 for x in counts):
        raise ValueError('invalid tag or table')
    x, M, result = F(k, 1 << n), sum(counts), []
    for _ in range(length):
        cumulative = 0
        for symbol, frequency in enumerate(counts):
            if F(cumulative, M) <= x < F(cumulative + frequency, M):
                result.append(symbol)
                x = (M * x - cumulative) / frequency
                break
            cumulative += frequency
        else:
            raise ValueError('no matching interval')
    return result


AS, AB, DS, DB = F(1, 10), F(1, 4), F(1, 20), F(-1, 4)
ACOUNTS, DCOUNTS = (1, 2, 3, 1), tuple(1 + i % 3 for i in range(16))


def fixed_code(x, scale, bias, maximum):
    return max(0, min(maximum, ((x - bias) / scale + F(1, 2)).__floor__()))


def fixed_reference(x, scale, bias, maximum):
    return min((bias + scale * q for q in range(maximum + 1)),
               key=lambda y: (abs(x - y), -y))


def cache_encode(values, C, wrong_anchor=False):
    if C <= 0:
        raise ValueError('invalid chunk length')
    anchors, deltas = [], []
    for start in range(0, len(values), C):
        anchor = values[start]
        qa = fixed_code(anchor, AS, AB, 3)
        anchors.append(qa)
        rebuilt_anchor = AB + AS * qa
        base = anchor if wrong_anchor else rebuilt_anchor
        for x in values[start + 1:start + C]:
            deltas.append(fixed_code(x - base, DS, DB, 15))
    return dict(N=len(values), C=C, version=1,
                anchors=ac_encode(anchors, ACOUNTS),
                deltas=ac_encode(deltas, DCOUNTS))


def cache_decode(packet):
    if packet['version'] != 1 or packet['C'] <= 0 or packet['N'] < 0:
        raise ValueError('format mismatch')
    anchors = ac_decode(packet['anchors'], ACOUNTS)
    deltas = ac_decode(packet['deltas'], DCOUNTS)
    result, ai, di = [], 0, 0
    for start in range(0, packet['N'], packet['C']):
        anchor = AB + AS * anchors[ai]
        ai += 1
        result.append(anchor)
        for _ in range(start + 1, min(packet['N'], start + packet['C'])):
            result.append(anchor + DB + DS * deltas[di])
            di += 1
    if ai != len(anchors) or di != len(deltas):
        raise ValueError('symbol count mismatch')
    return result


def cache_reference(values, C):
    result = []
    for start in range(0, len(values), C):
        anchor = fixed_reference(values[start], AS, AB, 3)
        result.append(anchor)
        for x in values[start + 1:start + C]:
            result.append(anchor + fixed_reference(x - anchor, DS, DB, 15))
    return result


def check_cachegen():
    # Independent hand interval: symbols 1,0 under counts 1,3 yield [1/4,7/16).
    equal(ac_encode([1, 0], (1, 3)), (1, 2, 2))
    equal(ac_decode((1, 2, 2), (1, 3)), [1, 0])
    for symbols in ([], [0], [1], [1, 0, 1, 1, 0], [0, 0, 0, 1]):
        equal(ac_decode(ac_encode(symbols, (1, 3)), (1, 3)), symbols)
    detect_difference('wrong entropy table', ac_decode((1, 2, 2), (3, 1)), [1, 0])
    values = [F(30, 100), F(40, 100), F(50, 100), F(61, 100), F(64, 100)]
    for C in (1, 2, 3, 8):
        for L in range(len(values) + 1):
            equal(cache_decode(cache_encode(values[:L], C)), cache_reference(values[:L], C))
    expected = cache_reference(values, 3)
    detect_difference('delta formed from original rather than reconstructed anchor',
                      cache_decode(cache_encode(values, 3, True)), expected)
    packet = cache_encode(values, 3)
    packet['version'] = 2
    must_raise('unsupported transport format', lambda: cache_decode(packet), ValueError)
    print('observation: rational interval codec, known interval, empty/short chunks and independent anchor reconstruction checked')


def transmission(D_bytes, B_GiB_s, ratio, extra_ms):
    if D_bytes < 0 or B_GiB_s <= 0 or not 0 <= ratio <= 1 or extra_ms < 0:
        raise ValueError('invalid cost model input')
    raw = F(D_bytes) / (B_GiB_s * (1 << 30)) * 1000
    return raw, extra_ms + ratio * raw


def check_cost():
    D = 512 * (1 << 20)
    for bandwidth, expected in ((F(20), (F(25), F(73, 4))),
                                (F(100), (F(5), F(53, 4))),
                                (F(125, 4), (F(16), F(16)))):
        raw, compressed = transmission(D, bandwidth, F(1, 4), F(12))
        equal((raw, compressed), expected)
        equal(raw, F(512) / (bandwidth * 1024) * 1000)
    raw, compressed = transmission(D, F(100), F(1, 4), F(12))
    good_choice = 'compressed' if compressed < raw else 'raw'
    bad_choice = 'compressed' if raw / 4 < raw else 'raw'
    detect_difference('controller ignores decode time', bad_choice, good_choice)
    detect_difference('MiB incorrectly treated as GiB', F(512, 100) * 1000, raw)
    candidates = [('raw', F(25), True), ('compressed', F(73, 4), True),
                  ('aggressive', F(10), False)]
    feasible = [x for x in candidates if x[2] and x[1] <= 10]
    equal(feasible, [])
    # Aggregate d represents all layer/head channels; no large arrays allocated.
    L, d, R, G, w, m = 4096, 131072, 128, 32, 2, 4
    nk, nv = L - L % R, max(0, L - R)
    code = F((nk + nv) * d * w, 8)
    meta = m * ((nk // G) * d + nv * ((d + G - 1) // G))
    residual = 2 * d * ((L - nk) + (L - nv))
    equal((code + meta + residual) / (1 << 20), F(410))
    upper = 2 * d * (2 * R) + F(2 * (L - R) * d * w, 8) + 2 * (L - R) * d // G * m
    equal(upper / (1 << 20), F(436))
    print('observation: 20/100/31.25 GiB/s costs, infeasible deadline, and 410/436 MiB capacity conditions checked')


def main():
    check_quant()
    check_attention()
    check_h2o()
    check_cachegen()
    check_cost()
    require(len(NEGATIVES) >= 10, 'missing required active negative controls')
    print('negative controls detected:', '; '.join(NEGATIVES))
    print('scope: exact-rational quantization/codec and small binary64 attention software models only')
    print('not measured: target kernels, finite-width codec, low-precision hardware rounding, GPU visibility/transactions, network, model quality, or Agent benefit')


if __name__ == '__main__':
    main()
