#!/usr/bin/env python3
'''K-FW-006: deterministic, read-only, offline CPU scalar checks.

Only Python standard library; all inputs are embedded. No project imports,
file access, network, subprocesses, GPU execution, or performance measurements.
Decimal references independently reconstruct each processed legal prefix.
The coded-input case checks scale binding and singleton P reconstruction;
it is not a general FP8 quantizer or hardware rounding simulation.
'''
import math
from decimal import Decimal, localcontext

NEG_INF = float('-inf')
TOL = 5e-12


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b):
    if math.isnan(a) or math.isnan(b):
        return False
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= TOL * max(1.0, abs(a), abs(b))


def close_vec(a, b):
    return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))


def same_state(a, b):
    return close(a[0], b[0]) and close(a[1], b[1]) and close_vec(a[2], b[2])


def dense_reference(q, keys, vals, allowed, indices, dv, scale=1.0, bias=None):
    # Does not call the online recurrence, its score function, or the merger.
    # Each invocation starts over from raw inputs for this entire prefix.
    with localcontext() as ctx:
        ctx.prec = 70
        D = lambda x: Decimal(str(x))
        entries = []
        for j in indices:
            if allowed[j]:
                z = D(scale) * sum((D(q[c]) * D(keys[j][c])
                                    for c in range(len(q))), Decimal(0))
                if bias is not None:
                    z += D(bias[j])
                entries.append((j, z))
        if not entries:
            return (NEG_INF, 0.0, [0.0] * dv), ([0.0] * dv, NEG_INF)
        m = max(z for _, z in entries)
        weights = [(j, (z - m).exp()) for j, z in entries]
        ell = sum((w for _, w in weights), Decimal(0))
        U = [sum((w * D(vals[j][c]) for j, w in weights), Decimal(0))
             for c in range(dv)]
        state = float(m), float(ell), [float(u) for u in U]
        result = [float(u / ell) for u in U], float(m + ell.ln())
        return state, result


def online_row(q, keys, vals, allowed, blocks, dv, scale=1.0, bias=None, bug=None):
    m, ell, U = NEG_INF, 0.0, [0.0] * dv
    trace = [(m, ell, U[:])]
    for block_index, block in enumerate(blocks):
        pairs = []
        for j in block:
            if allowed[j]:
                z = scale * math.fsum(q[c] * keys[j][c] for c in range(len(q)))
                if bias is not None:
                    z += bias[j]
                pairs.append((j, z))
        if bug == 'last_block_only' and block_index != len(blocks) - 1:
            pairs = []
        if not pairs:
            if bug == 'reset_empty':
                m, ell, U = NEG_INF, 0.0, [0.0] * dv
            trace.append((m, ell, U[:]))
            continue
        m_new = max(m, max(z for _, z in pairs))
        r = 0.0 if ell == 0.0 else math.exp(m - m_new)
        ru = r
        if bug in ('inverse_u_only', 'inverse_both') and ell != 0.0:
            ru = math.exp(m_new - m)
        p = [(j, math.exp(z - m_new)) for j, z in pairs]
        ell_factor = ru if bug == 'inverse_both' else r
        ell = ell_factor * ell + math.fsum(w for _, w in p)
        U = [ru * U[c] + math.fsum(w * vals[j][c] for j, w in p)
             for c in range(dv)]
        m = m_new
        trace.append((m, ell, U[:]))
    result = ([u / ell for u in U], m + math.log(ell)) if ell else ([0.0] * dv, NEG_INF)
    return result, trace


def check_trace(q, keys, vals, allowed, blocks, dv, trace, scale=1.0, bias=None):
    require(len(trace) == len(blocks) + 1, 'missing state observations')
    seen = []
    for prefix in range(len(trace)):
        if prefix:
            seen.extend(blocks[prefix - 1])
        expected, _ = dense_reference(q, keys, vals, allowed, seen, dv, scale, bias)
        require(same_state(trace[prefix], expected),
                'prefix %d m/ell/U mismatch: %r != %r' % (prefix, trace[prefix], expected))


def checked_row(q, keys, vals, allowed, blocks, dv, scale=1.0, bias=None):
    result, trace = online_row(q, keys, vals, allowed, blocks, dv, scale, bias)
    check_trace(q, keys, vals, allowed, blocks, dv, trace, scale, bias)
    indices = [j for block in blocks for j in block]
    _, expected = dense_reference(q, keys, vals, allowed, indices, dv, scale, bias)
    require(close_vec(result[0], expected[0]), 'final O mismatch')
    require(close(result[1], expected[1]), 'final natural LSE mismatch')
    return result, trace


def require_trace_rejection(args, trace, label):
    try:
        check_trace(*args, trace)
    except AssertionError:
        return
    raise AssertionError('negative control was accepted: ' + label)


def merge_states(states, dv, bug=None):
    live = [(o, lse) for o, lse in states if lse != NEG_INF]
    if not live:
        return [0.0] * dv, NEG_INF
    top = max(lse for _, lse in live)
    if bug == 'average':
        weights = [1.0] * len(live)
    elif bug == 'base_mix':
        weights = [2.0 ** (lse - top) for _, lse in live]
    else:
        weights = [math.exp(lse - top) for _, lse in live]
    denom = math.fsum(weights)
    out = [math.fsum(w * state[0][c] for w, state in zip(weights, live)) / denom
           for c in range(dv)]
    return out, top + math.log(denom)


def test_prefixes_and_edges():
    q = [0.5, -0.25, 1.0]
    keys = [[0.1, 0.2, -0.3], [0.4, -0.1, 0.2], [0.0, 0.5, 0.1],
            [0.7, 0.2, -0.2], [-0.3, 0.4, 0.2], [0.2, -0.6, 0.5], [0.9, 0.1, 0.0]]
    vals = [[1.0, 0.0], [0.5, 1.0], [-0.2, 0.3], [2.0, -1.0],
            [0.0, 0.7], [1.5, 0.2], [-0.5, 0.4]]
    allowed = [True, False, True, True, False, True, True]
    bias = [0.1, 0.2, -0.3, 0.4, 0.0, 0.5, -0.2]
    blocks = [[], [0, 1], [4], [2, 3], [], [5, 6], []]
    checked_row(q, keys, vals, allowed, blocks, 2, 0.7, bias)
    for width in (1, 2, 3, 4, 8):
        partition = [list(range(start, min(start + width, len(keys))))
                     for start in range(0, len(keys), width)]
        checked_row(q, keys, vals, allowed, partition, 2, 0.7, bias)
    checked_row([1.0], [], [], [], [], 2)
    checked_row([1.0], [], [], [], [[], []], 2)
    checked_row([1.0], [[0.0], [1.0]], [[3.0, 4.0], [5.0, 6.0]],
                [False, False], [[0], [], [1]], 2)
    checked_row([1.0], [[1000.0], [1001.0]], [[1.0], [0.0]],
                [True, True], [[0], [1]], 1)
    require(not close(float('nan'), float('nan')), 'comparator accepted NaN')
    print('CPU prefixes: initial and every block m/ell/U compared to independent Decimal reconstruction; final O/LSE compared; masks, bias, dk!=dv, tails, S=0 and large finite scores exercised')


def test_two_block_and_state_controls():
    args = ([1.0], [[0.0], [2.0]], [[1.0], [0.0]], [True, True], [[0], [1]], 1)
    result, trace = checked_row(*args)
    require(same_state(trace[1], (0.0, 1.0, [1.0])), 'first block trace')
    require(same_state(trace[2], (2.0, 1.0 + math.exp(-2), [math.exp(-2)])), 'second block trace')
    require(close(result[0][0], 1.0 / (1.0 + math.exp(2))), 'lost early contribution')
    for bug in ('last_block_only', 'inverse_u_only', 'inverse_both'):
        wrong, wrong_trace = online_row(*args, bug=bug)
        require_trace_rejection(args, wrong_trace, bug)
        require(not close_vec(wrong[0], result[0]), 'final reference missed ' + bug)
        if bug == 'last_block_only':
            require(wrong[0] == [0.0], 'last-block control did not reproduce failure')
        if bug == 'inverse_u_only':
            require(wrong[0][0] > 1.0, 'U-only control did not leave convex hull')
        if bug == 'inverse_both':
            require(0.0 <= wrong[0][0] <= 1.0, 'double-inverse hull counterexample lost')
    empty_args = args[:4] + ([[0], [], [1], []], 1)
    checked_row(*empty_args)
    wrong, wrong_trace = online_row(*empty_args, bug='reset_empty')
    require_trace_rejection(empty_args, wrong_trace, 'reset_empty')
    require(not close_vec(wrong[0], result[0]), 'empty-reset control missed')
    # A consistent but false state scale can preserve BOTH final O and LSE.
    shifted = []
    for m, ell, U in trace:
        shifted.append((m + 1.0, ell / math.e, [u / math.e for u in U]) if ell else (m, ell, U[:]))
    m, ell, U = shifted[-1]
    require(close_vec([u / ell for u in U], result[0]), 'shift did not preserve O')
    require(close(m + math.log(ell), result[1]), 'shift did not preserve LSE')
    require_trace_rejection(args, shifted, 'common_state_scale')
    print('CPU two-block trace: O=%.15f; last-block-only, inverse U, inverse U/ell and reset-empty rejected; false common state scale rejected despite preserved final O/LSE' % result[0][0])


def test_decoded_scale_singleton_path():
    # Q=2, decoded K=(0,1), decoded V=(1,0), logits=(0,2).
    # These are exactly representable codes; general FP8 rounding is outside scope.
    q8, sQ = [4.0], 0.5
    k8, sK = [[0.0], [4.0]], [0.25, 0.25]
    v8, sV = [[2.0], [0.0]], [0.5, 0.5]
    q = [sQ * x for x in q8]
    keys = [[sK[j] * x for x in k8[j]] for j in range(2)]
    vals = [[sV[j] * x for x in v8[j]] for j in range(2)]
    # Singleton visible P is 1; maximal finite E4M3FN code has value 448.
    p8, sP = 448.0, 1.0 / 448.0
    require(close(p8 * sP, 1.0), 'singleton P reconstruction')
    vals = [[sP * p8 * x for x in row] for row in vals]
    args = (q, keys, vals, [True, True], [[0], [1]], 1)
    expected, _ = checked_row(*args)
    wrong, bad_trace = online_row(*args, bug='last_block_only')
    require_trace_rejection(args, bad_trace, 'coded last-block-only')
    require(wrong[0] == [0.0] and expected[0][0] > 0.0, 'coded early mass lost')
    wrong_keys = k8  # Omit Q/K dequantization units while retaining the same logical fixture.
    wrong_scale, _ = online_row(q8, wrong_keys, vals, [True, True], [[0], [1]], 1)
    require(not close_vec(wrong_scale[0], expected[0]), 'missing Q/K scale control missed')
    wrong_vals = v8
    wrong_value, _ = online_row(q, keys, wrong_vals, [True, True], [[0], [1]], 1)
    require(not close_vec(wrong_value[0], expected[0]), 'missing V scale control missed')
    print('CPU coded singleton case: nonunit Q/K/V scales and exactly reconstructed P=1 retained early mass; omitted scales and last-block-only rejected; no general FP8 rounding claim')


def test_partition_controls():
    q, keys, vals, allowed = [1.0], [[0.0], [2.0]], [[1.0], [0.0]], [True, True]
    _, ref = dense_reference(q, keys, vals, allowed, [0, 1], 1)
    parts = [checked_row(q, keys, vals, allowed, [[j]], 1)[0] for j in range(2)]
    empty = ([0.0], NEG_INF)
    for states in (parts, parts[::-1], [empty] + parts + [empty]):
        out, lse = merge_states(states, 1)
        require(close_vec(out, ref[0]) and close(lse, ref[1]), 'partition merge mismatch')
    require(merge_states([empty, empty], 1) == empty, 'all-empty merge')
    for bug in ('average', 'base_mix'):
        wrong, _ = merge_states(parts, 1, bug)
        require(not close_vec(wrong, ref[0]), 'merge control missed: ' + bug)
    duplicate, _ = merge_states([ref, parts[0]], 1)
    require(not close_vec(duplicate, ref[0]), 'duplicate mass control missed')
    print('CPU merge: disjoint and empty partitions matched dense reference; direct average, natural/base-2 mixing and duplicate mass rejected')


def test_mask_denominator_control():
    q, keys, vals = [0.0], [[0.0], [0.0]], [[2.0], [100.0]]
    correct, _ = checked_row(q, keys, vals, [True, False], [[0, 1]], 1)
    # Actual faulty path: include ALL keys in softmax, then zero illegal weights.
    scores = [math.fsum(q[c] * k[c] for c in range(len(q))) for k in keys]
    weights = [math.exp(z - max(scores)) for z in scores]
    denom = math.fsum(weights)
    bad = math.fsum(weights[j] / denom * vals[j][0] * int(j == 0) for j in range(2))
    require(close(correct[0][0], 2.0) and close(bad, 1.0), 'mask fixture')
    require(not close(bad, correct[0][0]), 'bad denominator not rejected')
    print('CPU mask: illegal key excluded before normalization; normalize-all then zero-mask produced 1 instead of 2 and was rejected')


def test_causal_and_gqa_numerical_paths():
    keys = [[0.0] for _ in range(5)]
    vals = [[float(j * j + 1)] for j in range(5)]
    expected = [[True, True, True, True, False], [True, True, True, True, True]]
    for row, position in enumerate((3, 4)):
        allowed = [j <= position for j in range(5)]
        require(allowed == expected[row], 'absolute causal set')
        good, _ = checked_row([1.0], keys, vals, allowed, [[0, 1], [2, 3], [4]], 1)
        bad, _ = online_row([1.0], keys, vals, [j <= row for j in range(5)], [[0, 1], [2, 3], [4]], 1)
        require(not close_vec(good[0], bad[0]), 'wrong causal alignment not numerically detected')
    mapping = [0, 0, 0, 1, 1, 1]
    kv_keys = [[[0.0], [1.0]], [[1.0], [-1.0]]]
    kv_vals = [[[1.0], [3.0]], [[10.0], [20.0]]]
    mismatch = False
    for h in range(6):
        owner = h // 3
        require(owner == mapping[h], 'uniform GQA owner')
        q = [0.5 + h * 0.25]
        good, _ = checked_row(q, kv_keys[owner], kv_vals[owner], [True, True], [[0], [1]], 1)
        _, independent = dense_reference(q, kv_keys[mapping[h]], kv_vals[mapping[h]], [True, True], [0, 1], 1)
        require(close_vec(good[0], independent[0]), 'GQA numerical output')
        bad_owner = h % 2
        bad, _ = online_row(q, kv_keys[bad_owner], kv_vals[bad_owner], [True, True], [[0], [1]], 1)
        mismatch = mismatch or not close_vec(bad[0], independent[0])
    require(mismatch, 'modulo-head negative control not detected')
    print('CPU numerical integration: causal positions 3/4 and six-query/two-KV GQA matched independent expectations; left-top causal and modulo-head controls rejected')


def main():
    test_prefixes_and_edges()
    test_two_block_and_state_controls()
    test_decoded_scale_singleton_path()
    test_partition_controls()
    test_mask_denominator_control()
    test_causal_and_gqa_numerical_paths()
    print('Completed only the declared deterministic CPU scalar checks; no CUDA/HIP, GPU measurement, memory-visibility, performance, model-quality or Agent-effect evidence.')


if __name__ == '__main__':
    main()
