import math
from decimal import Decimal, localcontext

CHECKS = 0


def require(name, condition):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(name)


def near(a, b, tol=2e-11):
    if math.isnan(a) or math.isnan(b):
        return False
    if not math.isfinite(a) or not math.isfinite(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def near_vec(a, b):
    return len(a) == len(b) and all(near(x, y) for x, y in zip(a, b))


def empty(dv):
    return (-math.inf, 0.0, (0.0,) * dv)


def same_state(a, b):
    return near(a[0], b[0]) and near(a[1], b[1]) and near_vec(a[2], b[2])


def merge(x, y, base_two=False, fault=None):
    m1, l1, u1 = x
    m2, l2, u2 = y
    if l1 == 0.0 and l2 == 0.0:
        return empty(len(u1))
    if l1 == 0.0:
        return (m2, l2, tuple(u2))
    if l2 == 0.0:
        return (m1, l1, tuple(u1))
    m = max(m1, m2)
    exponent = (lambda t: 2.0 ** t) if base_two and fault != 'base' else math.exp
    a = exponent(m1 - m)
    b = exponent(m2 - m)
    au = a
    if fault == 'old_both':
        a = exponent(m - m1)
        au = a
    if fault == 'old_u_only':
        au = exponent(m - m1)
    return (m, a * l1 + b * l2,
            tuple(au * v1 + b * v2 for v1, v2 in zip(u1, u2)))


def block_state(q, K, V, scale, bias, visible, lo, hi, dv, base_two=False):
    # Every invocation initializes ALL components, including after a valid block.
    mb, lb, ub = -math.inf, 0.0, [0.0] * dv
    pairs = []
    for j in range(lo, hi):
        if visible[j]:
            z = scale * sum(q[c] * K[j][c] for c in range(len(q))) + bias[j]
            if not math.isfinite(z) or not all(math.isfinite(v) for v in V[j]):
                raise ValueError('outside finite-score/value model')
            pairs.append((j, z / math.log(2.0) if base_two else z))
    if pairs:
        mb = max(z for _, z in pairs)
        for j, z in pairs:
            w = 2.0 ** (z - mb) if base_two else math.exp(z - mb)
            lb += w
            for c in range(dv):
                ub[c] += w * V[j][c]
    return (mb, lb, tuple(ub))


def online(q, K, V, scale, bias, visible, blocks, dv, base_two=False, fault=None):
    state = empty(dv)
    trace = []
    previous_local = empty(dv)
    for lo, hi in blocks:
        local = block_state(q, K, V, scale, bias, visible, lo, hi, dv, base_two)
        if fault == 'stale' and local[1] == 0.0:
            local = previous_local
        else:
            previous_local = local
        # No continue: empty local states execute the documented merge path.
        state = merge(state, local, base_two, fault)
        trace.append(state)
    return state, trace


def normalized(state):
    _, ell, u = state
    return (0.0,) * len(u) if ell == 0.0 else tuple(v / ell for v in u)


def lse(state, base_two=False):
    m, ell, _ = state
    if ell == 0.0:
        return -math.inf
    return math.log(2.0) * (m + math.log2(ell)) if base_two else m + math.log(ell)


def reference(q, K, V, scale, bias, visible, indices, dv):
    # Independent whole-set Decimal oracle: no block_state, online, or merge calls.
    with localcontext() as ctx:
        ctx.prec = 80
        D = Decimal.from_float
        entries = []
        for j in indices:
            if visible[j]:
                score = D(scale) * sum((D(q[c]) * D(K[j][c]) for c in range(len(q))), Decimal(0)) + D(bias[j])
                entries.append((j, score))
        if not entries:
            return (-math.inf, 0.0, (0.0,) * dv), (0.0,) * dv, -math.inf
        m = max(z for _, z in entries)
        weights = [(j, (z - m).exp()) for j, z in entries]
        ell = sum((w for _, w in weights), Decimal(0))
        u = tuple(sum((w * D(V[j][c]) for j, w in weights), Decimal(0)) for c in range(dv))
        state = (float(m), float(ell), tuple(float(v) for v in u))
        return state, tuple(float(v / ell) for v in u), float(m + ell.ln())


def two_pass(q, K, V, scale, bias, visible, blocks, dv):
    m, ell = -math.inf, 0.0
    for lo, hi in blocks:
        scores = [scale * sum(q[c] * K[j][c] for c in range(len(q))) + bias[j]
                  for j in range(lo, hi) if visible[j]]
        if scores:
            mb = max(scores)
            lb = sum(math.exp(z - mb) for z in scores)
            if ell == 0.0:
                m, ell = mb, lb
            else:
                mn = max(m, mb)
                ell = math.exp(m - mn) * ell + math.exp(mb - mn) * lb
                m = mn
    if ell == 0.0:
        return (0.0,) * dv
    u = [0.0] * dv
    for lo, hi in blocks:
        for j in range(lo, hi):
            if visible[j]:
                z = scale * sum(q[c] * K[j][c] for c in range(len(q))) + bias[j]
                for c in range(dv):
                    u[c] += math.exp(z - m) * V[j][c]
    return tuple(v / ell for v in u)


def run_case(name, q, K, V, scale, bias, visible, dv):
    _, expected, expected_lse = reference(q, K, V, scale, bias, visible, range(len(K)), dv)
    maximum_error = 0.0
    empty_identity_events = 0
    for width in (1, 2, 3, 4, 8):
        blocks = [(lo, min(lo + width, len(K))) for lo in range(0, len(K), width)]
        state, trace = online(q, K, V, scale, bias, visible, blocks, dv)
        processed = []
        before = empty(dv)
        for (lo, hi), actual in zip(blocks, trace):
            processed.extend(range(lo, hi))
            ref_state, _, _ = reference(q, K, V, scale, bias, visible, processed, dv)
            require(name + ':prefix_state', same_state(actual, ref_state))
            if not any(visible[lo:hi]):
                require(name + ':empty_identity_exact', actual == before)
                empty_identity_events += 1
            before = actual
        out = normalized(state)
        require(name + ':finite_output', all(math.isfinite(v) for v in out))
        require(name + ':whole_output', near_vec(out, expected))
        require(name + ':natural_lse', near(lse(state), expected_lse))
        require(name + ':two_pass', near_vec(two_pass(q, K, V, scale, bias, visible, blocks, dv), expected))
        parts = [block_state(q, K, V, scale, bias, visible, lo, hi, dv) for lo, hi in blocks]
        acc = empty(dv)
        for part in [empty(dv)] + list(reversed(parts)) + [empty(dv)]:
            acc = merge(acc, part)
        require(name + ':reverse_split_state', same_state(acc, state))
        require(name + ':reverse_split_output', near_vec(normalized(acc), expected))
        state2, _ = online(q, K, V, scale, bias, visible, blocks, dv, base_two=True)
        require(name + ':base2_output', near_vec(normalized(state2), expected))
        require(name + ':base2_lse', near(lse(state2, True), expected_lse))
        maximum_error = max(maximum_error, max((abs(x - y) for x, y in zip(out, expected)), default=0.0))
    return maximum_error, empty_identity_events


def main():
    K = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5], [0.5, -1.0], [2.0, 3.0], [0.0, 0.0], [-2.0, 1.0]]
    V = [[1.0, 0.0, -1.0], [0.0, 2.0, 1.0], [0.5, 0.5, 0.0], [-1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [3.0, -2.0, 1.0], [0.25, 0.5, -0.5]]
    bias = [0.25, -0.5, 0.75, 0.0, -1.0, 0.6, 0.1]
    rows = [([0.7, -0.3], [True, True, False, False, True, False, False]),
            ([1.0, 0.5], [False] * 7),
            ([-0.2, 0.8], [False, False, True, True, True, True, True])]
    errors, identities = [], 0
    for i, (q, visible) in enumerate(rows):
        err, count = run_case('row' + str(i), q, K, V, 0.9, bias, visible, 3)
        errors.append(err)
        identities += count
    for name, q, keys, vals, biases, vis, dv in [
        ('no_keys', [1.0], [], [], [], [], 2),
        ('single_visible', [1.0], [[-3.0], [2.0]], [[7.0], [9.0]], [0.0, 0.0], [True, False], 1),
        ('large_finite', [1.0], [[1000.0], [-1000.0], [1001.0]], [[1.0], [5.0], [-2.0]], [0.0] * 3, [True] * 3, 1),
        ('negative_scores', [1.0], [[-20.0], [-30.0], [-10.0]], [[1.0], [3.0], [8.0]], [0.0] * 3, [True, False, True], 1)]:
        err, count = run_case(name, q, keys, vals, 1.0, biases, vis, dv)
        errors.append(err)
        identities += count

    q, keys, vals = [1.0], [[0.0], [2.0], [99.0]], [[1.0], [0.0], [100.0]]
    visible, biases = [True, True, False], [0.0] * 3
    blocks = [(0, 1), (1, 2), (2, 3)]
    _, expected, _ = reference(q, keys, vals, 1.0, biases, visible, range(3), 1)
    correct, trace = online(q, keys, vals, 1.0, biases, visible, blocks, 1)
    stale, bad_trace = online(q, keys, vals, 1.0, biases, visible, blocks, 1, fault='stale')
    require('valid_then_empty_exact_state', trace[2] == trace[1])
    require('stale_local_state_detected', not same_state(bad_trace[2], trace[2]))
    require('stale_local_output_detected', not near_vec(normalized(stale), expected))
    require('stale_counterexample_value', near(normalized(stale)[0], 1.0 / (1.0 + 2.0 * math.exp(2.0))))
    wrong, _ = online(q, keys, vals, 1.0, biases, visible, blocks, 1, fault='old_both')
    wrong_u, _ = online(q, keys, vals, 1.0, biases, visible, blocks, 1, fault='old_u_only')
    require('wrong_both_scaling_detected', not near_vec(normalized(wrong), expected))
    require('wrong_both_still_in_hull', 0.0 <= normalized(wrong)[0] <= 1.0)
    require('wrong_u_only_outside_hull', normalized(wrong_u)[0] > 1.0)
    wrong_base, _ = online(q, keys, vals, 1.0, biases, visible, blocks, 1, base_two=True, fault='base')
    require('base_mismatch_detected', not near_vec(normalized(wrong_base), expected))
    partials = [normalized(block_state(q, keys, vals, 1.0, biases, visible, lo, hi, 1))[0] for lo, hi in blocks[:2]]
    require('equal_partial_average_detected', not near(sum(partials) / 2.0, expected[0]))
    _, biased, _ = reference(rows[0][0], K, V, 0.9, bias, rows[0][1], range(7), 3)
    omitted, _ = online(rows[0][0], K, V, 0.9, [0.0] * 7, rows[0][1], [(0, 7)], 3)
    require('omitted_bias_detected', not near_vec(normalized(omitted), biased))
    unsafe = math.exp(-math.inf - (-math.inf))
    require('unsafe_empty_subtraction_is_nan', math.isnan(unsafe))
    require('comparison_rejects_nan', not near_vec([unsafe], [0.0]))
    require('empty_merge_is_defined', merge(empty(2), empty(2)) == empty(2))

    L, S, d, dv, Br = 4096, 4096, 64, 64, 128
    groups = (L + Br - 1) // Br
    qkvo = 2 * (L * d + groups * S * (d + dv) + L * dv)
    bias16, bias32 = 2 * L * S, 4 * L * S
    require('qkvo_bytes', qkvo == 34603008)
    require('fp16_bias_bytes', bias16 == 33554432)
    require('fp32_bias_bytes', bias32 == 67108864)
    require('qkvo_plus_fp16_bias', qkvo + bias16 == 68157440)
    require('qkvo_plus_fp32_bias', qkvo + bias32 == 101711872)
    # Independent address enumeration on a small shape includes the tail query block.
    nL, nS, nd, nv, nr = 5, 7, 2, 3, 2
    issued = [('Q', r, c) for r in range(nL) for c in range(nd)]
    issued += [('O', r, c) for r in range(nL) for c in range(nv)]
    explicit_bias_reads = []
    for lo in range(0, nL, nr):
        issued += [('K', j, c) for j in range(nS) for c in range(nd)]
        issued += [('V', j, c) for j in range(nS) for c in range(nv)]
        explicit_bias_reads += [(r, j) for r in range(lo, min(lo + nr, nL)) for j in range(nS)]
    expected_main = nL * nd + ((nL + nr - 1) // nr) * nS * (nd + nv) + nL * nv
    require('main_access_enumeration', len(issued) == expected_main)
    require('bias_each_pair_once', len(explicit_bias_reads) == len(set(explicit_bias_reads)) == nL * nS)
    require('omitted_explicit_input_account_detected', len(issued) + len(explicit_bias_reads) != expected_main)
    require('row_bitpack_tail', 3 * ((9 + 7) // 8) == 6)

    print('K-ALG-004: CPU scalar binary64 model; independent Decimal oracle at 80 digits')
    print('scope: finite scores/values, no dropout; no GPU, FP16/FP8 emulation, races, or timing')
    print('maximum_online_output_abs_error', max(errors))
    print('empty_identity_events', identities)
    print('correct_two_visible_then_empty', normalized(correct)[0])
    print('stale_local_negative_control', normalized(stale)[0])
    print('wrong_both_scaling_negative_control', normalized(wrong)[0])
    print('wrong_u_only_negative_control', normalized(wrong_u)[0])
    print('wrong_base_negative_control', normalized(wrong_base)[0])
    print('qkvo_bytes', qkvo)
    print('additional_fp16_bias_bytes', bias16)
    print('additional_fp32_bias_bytes', bias32)
    print('checks_completed', CHECKS)


if __name__ == '__main__':
    main()
