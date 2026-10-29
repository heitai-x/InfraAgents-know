import math
from collections import Counter, defaultdict
from itertools import combinations


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(x, y):
    return x == y or math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10)


def reference_state(Q, K, V, visible, scale, bias, row, end, dv):
    # Independent whole-prefix reference: no LDS, lane mapping, or online update.
    keys = [j for j in range(end) if visible[row][j]]
    if not keys:
        return -math.inf, 0.0, [0.0] * dv
    scores = [scale * math.fsum(Q[row][t] * K[j][t] for t in range(len(Q[row]))) + bias(row, j) for j in keys]
    maximum = max(scores)
    weights = [math.exp(z - maximum) for z in scores]
    mass = math.fsum(weights)
    numerator = [math.fsum(w * V[j][u] for w, j in zip(weights, keys)) for u in range(dv)]
    return maximum, mass, numerator


def dense_reference(Q, K, V, visible, scale, bias, dv):
    output, empty = [], []
    for row in range(len(Q)):
        _, mass, numerator = reference_state(Q, K, V, visible, scale, bias, row, len(K), dv)
        output.append([x / mass for x in numerator] if mass else [0.0] * dv)
        empty.append(mass == 0.0)
    return output, empty


def make_pages(K, V, d, dv, page_size=7):
    count = (len(K) + page_size - 1) // page_size
    table = list(reversed(range(count)))
    # Poison unused physical tail elements. A masked logical load must not read them.
    pk = [[[math.nan] * d for _ in range(page_size)] for _ in range(count)]
    pv = [[[math.nan] * dv for _ in range(page_size)] for _ in range(count)]
    for j in range(len(K)):
        physical = table[j // page_size]
        offset = j % page_size
        pk[physical][offset] = list(K[j])
        pv[physical][offset] = list(V[j])
    return table, pk, pv


def wave_model(Q, K, V, visible, scale, bias, d, dv, fault=None):
    require(d > 0 and dv > 0, 'positive feature dimensions required')
    L, S = len(Q), len(K)
    output = [[0.0] * dv for _ in range(L)]
    empty = [None] * L
    audit = {'bindings': [], 'trace': [], 'O_writers': [], 'empty_writers': [],
             'reads': [], 'exponents': [], 'global_reads': []}
    table, pk, pv = make_pages(K, V, d, dv)

    def paged_read(kind, pos, feature):
        require(0 <= pos < S, 'out-of-range logical global read')
        page = table[pos // 7]
        offset = pos % 7
        data = pk if kind == 'K' else pv
        value = data[page][offset][feature]
        require(math.isfinite(value), 'poison physical tail was consumed')
        audit['global_reads'].append((kind, pos, feature))
        return value

    for r0 in range(0, L, 4):
        qs = [[math.nan] * d for _ in range(4)]
        q_writes = [0] * (4 * d)
        last_load_row = [None] * 64
        # Execute the actual cooperative load and retain each lane's last temporary.
        for lane in range(64):
            for a in range(lane, 4 * d, 64):
                g2, t = divmod(a, d)
                load_row = r0 + g2
                last_load_row[lane] = load_row
                qs[g2][t] = Q[load_row][t] if load_row < L else 0.0
                q_writes[a] += 1
        require(all(n == 1 for n in q_writes), 'Q slot ownership')
        # This binding is deliberately after the cooperative load, as in the chapter.
        query_rows = [r0 + lane // 16 for lane in range(64)]
        if fault == 'legacy_row':
            query_rows = list(last_load_row)
        audit['bindings'].append((r0, list(last_load_row), list(query_rows)))
        require(all(row is not None for row in query_rows), 'uninitialized row reached score phase')
        m = [-math.inf] * 64
        ell = [0.0] * 64
        U = [{u: 0.0 for u in range(lane % 16, dv, 16)} for lane in range(64)]

        for k0 in range(0, S, 16):
            ks = [[math.nan] * d for _ in range(16)]
            vs = [[math.nan] * dv for _ in range(16)]
            kw, vw = [0] * (16 * d), [0] * (16 * dv)
            for lane in range(64):
                for a in range(lane, 16 * d, 64):
                    j, t = divmod(a, d)
                    ks[j][t] = paged_read('K', k0 + j, t) if k0 + j < S else 0.0
                    kw[a] += 1
                for a in range(lane, 16 * dv, 64):
                    j, u = divmod(a, dv)
                    vs[j][u] = paged_read('V', k0 + j, u) if k0 + j < S else 0.0
                    vw[a] += 1
            require(all(n == 1 for n in kw + vw), 'K/V slot ownership')
            z = [[-math.inf] * 16 for _ in range(4)]
            reads = Counter()
            for lane in range(64):
                g, l = divmod(lane, 16)
                query_row, pos = query_rows[lane], k0 + l
                if query_row < L and pos < S:
                    if visible[query_row][pos]:
                        acc = 0.0
                        for t in range(d):
                            reads['Q'] += 1
                            reads['K'] += 1
                            acc += qs[g][t] * ks[l][t]
                        z[g][l] = scale * acc + bias(query_row, pos)
                    elif fault == 'mask_zero':
                        z[g][l] = 0.0
            # Phase boundaries serialize publication in this CPU model only.
            maxima = [max(z[g]) for g in range(4)]
            probs = [[0.0] * 16 for _ in range(4)]
            new_m, rescale = list(m), [0.0] * 64
            probability_exp, rescale_exp = 0, 0
            for lane in range(64):
                g, l = divmod(lane, 16)
                if maxima[g] > -math.inf:
                    new_m[lane] = max(m[lane], maxima[g])
                    if ell[lane] > 0.0:
                        difference = m[lane] - new_m[lane]
                        if fault == 'reverse_rescale':
                            difference = -difference
                        rescale[lane] = math.exp(difference)
                        rescale_exp += 1
                    if z[g][l] > -math.inf:
                        probs[g][l] = math.exp(z[g][l] - new_m[lane])
                        probability_exp += 1
            sums = [sum(probs[g]) for g in range(4)]
            for lane in range(64):
                g, l = divmod(lane, 16)
                if maxima[g] > -math.inf:
                    for u in U[lane]:
                        add = 0.0
                        for q in range(16):
                            reads['P'] += 1
                            prob = probs[g][q]
                            if prob > 0.0:
                                reads['V'] += 1
                                add += prob * vs[q][u]
                        U[lane][u] = rescale[lane] * U[lane][u] + add
                    ell[lane] = rescale[lane] * ell[lane] + sums[g]
                    m[lane] = new_m[lane]
                canonical_row = r0 + g
                if canonical_row < L:
                    audit['trace'].append((canonical_row, min(S, k0 + 16), m[lane], ell[lane], dict(U[lane])))
            audit['reads'].append((r0, k0, dict(reads)))
            audit['exponents'].append((r0, k0, probability_exp, rescale_exp))

        for lane in range(64):
            g, l = divmod(lane, 16)
            query_row = r0 + g
            if query_row >= L:
                continue
            for u in U[lane]:
                output[query_row][u] = U[lane][u] / ell[lane] if ell[lane] else 0.0
                audit['O_writers'].append((query_row, u, lane))
                if fault == 'duplicate_empty':
                    empty[query_row] = ell[lane] == 0.0
                    audit['empty_writers'].append((query_row, lane))
            if fault != 'duplicate_empty' and l == 0:
                empty[query_row] = ell[lane] == 0.0
                audit['empty_writers'].append((query_row, lane))
    return output, empty, audit


def verify(result, Q, K, V, visible, scale, bias, dv):
    output, empty, audit = result
    expected, expected_empty = dense_reference(Q, K, V, visible, scale, bias, dv)
    require(empty == expected_empty, 'empty-row value contract')
    require(len(output) == len(expected), 'output row count')
    error = 0.0
    for r in range(len(Q)):
        require(len(output[r]) == dv, 'output feature count')
        for u in range(dv):
            require(close(output[r][u], expected[r][u]), 'dense output mismatch at row=%d dim=%d' % (r, u))
            error = max(error, abs(output[r][u] - expected[r][u]))
    ow = Counter((r, u) for r, u, lane in audit['O_writers'])
    ew = Counter(r for r, lane in audit['empty_writers'])
    require(ow == Counter({(r, u): 1 for r in range(len(Q)) for u in range(dv)}), 'O writer uniqueness')
    require(ew == Counter({r: 1 for r in range(len(Q))}), 'empty writer uniqueness')
    require(all(lane % 16 == 0 for r, lane in audit['empty_writers']), 'empty writer must be l=0')
    require(all(u % 16 == lane % 16 and r % 4 == lane // 16 for r, u, lane in audit['O_writers']), 'O lane ownership')
    cache = {}
    for row, end, m, ell, numerator in audit['trace']:
        if (row, end) not in cache:
            cache[row, end] = reference_state(Q, K, V, visible, scale, bias, row, end, dv)
        rm, re, ru = cache[row, end]
        require(close(m, rm) and close(ell, re), 'prefix m/ell invariant')
        require(all(close(value, ru[u]) for u, value in numerator.items()), 'prefix U invariant')
    return error


def reject(label, operation):
    try:
        operation()
    except AssertionError as exc:
        print('REJECT %s: %s' % (label, exc))
    else:
        raise AssertionError('negative control escaped: ' + label)


def scratch_owners(stride):
    owners = defaultdict(list)
    for g in range(4):
        for l in range(16):
            owners[g * stride + l].append((g, l))
    return dict(owners)


def scratch_pairs(owners):
    pairs = defaultdict(list)
    for slot, writers in sorted(owners.items()):
        for (g1, l1), (g2, l2) in combinations(writers, 2):
            pairs[g1, g2].append(slot)
    return dict(pairs)


def fixture(L, S, d, dv):
    Q = [[(((r + 1) * (t + 3) * 5) % 13 - 6) / 6.0 for t in range(d)] for r in range(L)]
    K = [[(((j + 3) * (t + 2) * 7) % 19 - 9) / 9.0 for t in range(d)] for j in range(S)]
    V = [[(((j + 5) * (u + 1) * 11) % 23 - 11) / 11.0 for u in range(dv)] for j in range(S)]
    visible = []
    for r in range(L):
        visible.append([r == 0 or (r == 2 and j < 2) or (r == 3 and j >= 32) or (r >= 4 and j % 3 == r % 3) for j in range(S)])
    return Q, K, V, visible


def main():
    bias = lambda r, j: 0.04 * ((r + 2 * j) % 7 - 3) + j / 50.0
    worst = 0.0
    cases = [(5, 39, 4, 17), (5, 23, 128, 33), (1, 1, 1, 1),
             (1, 0, 4, 16), (0, 0, 4, 16), (3, 7, 4, 16)]
    for L, S, d, dv in cases:
        Q, K, V, visible = fixture(L, S, d, dv)
        scale = 1.0 / math.sqrt(d)
        result = wave_model(Q, K, V, visible, scale, bias, d, dv)
        error = verify(result, Q, K, V, visible, scale, bias, dv)
        worst = max(worst, error)
        print('OBSERVE scalar lane trace L=%d S=%d d=%d dv=%d error=%.6g empty=%s' % (L, S, d, dv, error, result[1]))
        for r0, last, bound in result[2]['bindings']:
            require(bound == [r0 + lane // 16 for lane in range(64)], 'post-load query binding')
            if d == 4:
                require(sum(x is None for x in last) == 48, 'd=4 load lifecycle enumeration')
            if d == 128:
                require(last == [r0 + 3] * 64, 'd=128 load lifecycle enumeration')

    zero_bias = lambda r, j: 0.0
    Q = [[0.0] * 128 for _ in range(4)]
    K = [[0.0] * 128 for _ in range(2)]
    V = [[1.0], [3.0]]
    visible = [[True, False], [True, True], [True, True], [False, True]]
    good = wave_model(Q, K, V, visible, 1.0, zero_bias, 128, 1)
    verify(good, Q, K, V, visible, 1.0, zero_bias, 1)
    bad = wave_model(Q, K, V, visible, 1.0, zero_bias, 128, 1, 'legacy_row')
    require(good[0][0][0] == 1.0 and bad[0][0][0] == 3.0, 'specific legacy row counterexample')
    reject('legacy load row d=128', lambda: verify(bad, Q, K, V, visible, 1.0, zero_bias, 1))
    print('OBSERVE row-mask counterexample correct=1 legacy=3')
    Q4, K4 = [[0.0] * 4 for _ in range(4)], [[0.0] * 4 for _ in range(2)]
    reject('uninitialized legacy row d=4', lambda: wave_model(Q4, K4, V, visible, 1.0, zero_bias, 4, 1, 'legacy_row'))

    all_visible = [[True, True] for _ in range(4)]
    row_bias = lambda r, j: math.log(3.0) if (r == 0 and j == 0) or (r == 3 and j == 1) else 0.0
    correct_bias = wave_model(Q, K, V, all_visible, 1.0, row_bias, 128, 1)
    verify(correct_bias, Q, K, V, all_visible, 1.0, row_bias, 1)
    require(close(correct_bias[0][0][0], 1.5) and close(correct_bias[0][3][0], 2.5), 'bias reference values')
    wrong_bias = wave_model(Q, K, V, all_visible, 1.0, row_bias, 128, 1, 'legacy_row')
    reject('legacy row with bias only', lambda: verify(wrong_bias, Q, K, V, all_visible, 1.0, row_bias, 1))

    for dv in (1, 16, 17, 33):
        q, k, v, vis = fixture(4, 2, 4, dv)
        correct = wave_model(q, k, v, vis, 0.5, bias, 4, dv)
        verify(correct, q, k, v, vis, 0.5, bias, dv)
        require(Counter(r for r, lane in correct[2]['empty_writers']) == Counter({r: 1 for r in range(4)}), 'one marker per row')
        wrong = wave_model(q, k, v, vis, 0.5, bias, 4, dv, 'duplicate_empty')
        counts = Counter(r for r, lane in wrong[2]['empty_writers'])
        require(counts == Counter({r: dv for r in range(4)}), 'legacy marker write count')
        if dv > 1:
            reject('duplicate empty dv=%d' % dv, lambda: verify(wrong, q, k, v, vis, 0.5, bias, dv))
        else:
            verify(wrong, q, k, v, vis, 0.5, bias, dv)
        print('OBSERVE marker writers dv=%d correct_per_row=1 legacy_per_row=%d' % (dv, dv))

    q, k, v, vis = [[1.0]], [[-2.0], [0.0]], [[1.0], [0.0]], [[True, False]]
    bad = wave_model(q, k, v, vis, 1.0, zero_bias, 1, 1, 'mask_zero')
    require(close(bad[0][0][0], 1.0 / (1.0 + math.exp(2.0))), 'invalid-key numerical counterexample')
    reject('masked zero enters denominator', lambda: verify(bad, q, k, v, vis, 1.0, zero_bias, 1))

    for stride in (16, 17):
        good_slots = scratch_owners(stride)
        require(len(good_slots) == 64 and all(len(w) == 1 for w in good_slots.values()), 'good scratch ownership')
    aliases = scratch_owners(8)
    expected_pairs = {(0, 1): list(range(8, 16)), (1, 2): list(range(16, 24)), (2, 3): list(range(24, 32))}
    actual_pairs = scratch_pairs(aliases)
    require(actual_pairs == expected_pairs, 'exact scratch intersection pairs and slots')
    for g in range(3):
        for t in range(8):
            require(aliases[8 * (g + 1) + t] == [(g, 8 + t), (g + 1, t)], 'slot-to-logical-writer identity')
    require(len(aliases) == 40 and sum(len(w) - 1 for w in aliases.values()) == 24, 'alias totals')
    reject('stride8 uniqueness', lambda: require(all(len(w) == 1 for w in aliases.values()), 'multiple scratch writers'))
    reject('incorrect 0/2 and 1/3 conflict claim', lambda: require(set(actual_pairs) == {(0, 2), (1, 3)}, 'claimed conflict pairs disagree with addresses'))
    print('OBSERVE scratch pairs=%s; proof covers slot aliases, not GPU overwrite order' % actual_pairs)

    # Full two-block case: 1000/1002 tests stable exponent shifts and rescaling.
    q = [[0.0] * 128 for _ in range(4)]
    k = [[0.0] * 128 for _ in range(32)]
    v = [[float(j // 16)] * 16 for j in range(32)]
    vis = [[True] * 32 for _ in range(4)]
    large_bias = lambda r, j: 1000.0 if j < 16 else 1002.0
    full = wave_model(q, k, v, vis, 1.0, large_bias, 128, 16)
    verify(full, q, k, v, vis, 1.0, large_bias, 16)
    require(close(full[0][0][0], 1.0 / (1.0 + math.exp(-2.0))), 'two-block rescale output')
    for r0, k0, counts in full[2]['reads']:
        byte_counts = {name: 4 * counts.get(name, 0) for name in ('Q', 'K', 'P', 'V')}
        require(byte_counts == {'Q': 32768, 'K': 32768, 'P': 4096, 'V': 4096}, 'enumerated LDS main read budget')
        reject('Q/K budget missing K factor', lambda: require(byte_counts['Q'] + byte_counts['K'] == 4 * 16 * 128 * 4, 'Q/K sum is 65536 bytes'))
    require(full[2]['exponents'] == [(0, 0, 64, 0), (0, 16, 64, 64)], 'probability versus replicated rescale exponent counts')
    capacity = 4 * (4 * 128 + 16 * 128 + 16 * 16 + 2 * 4 * 17 + 2 * 4)
    require(capacity == 11840, 'FP32 LDS capacity including padded score rows and scalar arrays')
    require(len(full[2]['global_reads']) == 32 * (128 + 16), 'one global K/V element load per group')
    reverse = wave_model(q, k, v, vis, 1.0, large_bias, 128, 16, 'reverse_rescale')
    reject('reverse online rescale', lambda: verify(reverse, q, k, v, vis, 1.0, large_bias, 16))
    print('OBSERVE full block main LDS reads Q=32768 K=32768 P=4096 V=4096 bytes; capacity=11840 bytes')
    print('OBSERVE probability/rescale exponent calls first=(64,0) next=(64,64) in lane-replicated model')
    print('OBSERVE checked prefix states, paged logical loads, output ownership; worst regular-case error=%.6g' % worst)
    print('SCOPE: deterministic scalar CPU model only; Python arithmetic is not FP32 GPU execution.')
    print('SCOPE: no evidence for GPU barriers, bank cycles, MFMA fragments, async overlap, dispatch, HBM transactions, compiled registers, occupancy, timing, model quality, or agent benefit.')


if __name__ == '__main__':
    main()
