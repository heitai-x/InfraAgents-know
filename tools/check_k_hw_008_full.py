import math
from decimal import Decimal, localcontext


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_geometry(L, S, Bq, Bk, d, dv, D):
    for name, value in (('L', L), ('S', S)):
        if type(value) is not int or value < 0:
            raise ValueError(name + ' must be a nonnegative integer')
    for name, value in (('Bq', Bq), ('Bk', Bk), ('d', d), ('dv', dv), ('D', D)):
        if type(value) is not int or value <= 0:
            raise ValueError(name + ' must be a positive integer')


def decimal_state(row, values, mask, stop, dv):
    chosen = [j for j in range(stop) if mask[j]]
    if not chosen:
        return (-math.inf, 0.0, tuple(0.0 for _ in range(dv)))
    with localcontext() as ctx:
        ctx.prec = 70
        z = [Decimal(str(row[j])) for j in chosen]
        m = max(z)
        weights = [(x - m).exp() for x in z]
        den = sum(weights, Decimal(0))
        num = tuple(sum((w * Decimal(str(values[j][t]))
                         for w, j in zip(weights, chosen)), Decimal(0))
                    for t in range(dv))
        return (float(m), float(den), tuple(float(x) for x in num))


def dense_reference(scores, values, valid, dv):
    outputs = []
    lses = []
    with localcontext() as ctx:
        ctx.prec = 70
        for row, mask in zip(scores, valid):
            chosen = [j for j in range(len(values)) if mask[j]]
            if not chosen:
                outputs.append([0.0] * dv)
                lses.append(-math.inf)
                continue
            z = [Decimal(str(row[j])) for j in chosen]
            m = max(z)
            w = [(x - m).exp() for x in z]
            den = sum(w, Decimal(0))
            outputs.append([float(sum((weight * Decimal(str(values[j][t]))
                                       for weight, j in zip(w, chosen)), Decimal(0)) / den)
                            for t in range(dv)])
            lses.append(float(m + den.ln()))
    return outputs, lses


def tiled_attention(scores, values, valid, Bq, Bk, dv, d=1, D=1, faulty=None):
    L, S = len(scores), len(values)
    validate_geometry(L, S, Bq, Bk, d, dv, D)
    if len(valid) != L or any(len(row) != S for row in scores):
        raise ValueError('score or mask shape mismatch')
    if any(len(row) != S for row in valid) or any(len(v) != dv for v in values):
        raise ValueError('value or mask shape mismatch')
    for i in range(L):
        for j in range(S):
            if valid[i][j] and (not math.isfinite(scores[i][j]) or
                               any(not math.isfinite(x) for x in values[j])):
                raise ValueError('nonfinite participating input')
    out = [[0.0] * dv for _ in range(L)]
    lse = [-math.inf] * L
    traces = [[] for _ in range(L)]
    for q0 in range(0, L, Bq):
        for i in range(q0, min(q0 + Bq, L)):
            m, den, u = -math.inf, 0.0, [0.0] * dv
            for k0 in range(0, S, Bk):
                stop = min(k0 + Bk, S)
                chosen = [j for j in range(k0, stop)
                          if faulty == 'include_padding' or valid[i][j]]
                old = (m, den, tuple(u))
                if chosen:
                    mt = max(scores[i][j] for j in chosen)
                    mn = max(m, mt)
                    alpha = 0.0 if den == 0.0 else math.exp(m - mn)
                    if faulty == 'no_rescale' and den > 0.0:
                        alpha = 1.0
                    weights = [math.exp(scores[i][j] - mn) for j in chosen]
                    den_new = alpha * den + math.fsum(weights)
                    tile_u = [math.fsum(w * values[j][t]
                                        for w, j in zip(weights, chosen))
                              for t in range(dv)]
                    u_new = [alpha * u[t] + tile_u[t] for t in range(dv)]
                    m, den, u = (m if faulty == 'missing_m_writeback' else mn), den_new, u_new
                else:
                    require((m, den, tuple(u)) == old, 'empty block changed state')
                traces[i].append((stop, m, den, tuple(u)))
            if den > 0.0:
                out[i] = [x / den for x in u]
                lse[i] = m + math.log(den)
    return out, lse, traces


def close_scalar(actual, expected, label):
    if actual == expected:
        return 0.0
    require(math.isfinite(actual) and math.isfinite(expected), label + ': nonfinite mismatch')
    error = abs(actual - expected)
    require(error <= 2e-11 * max(1.0, abs(expected)), label + ': value mismatch')
    return error


def output_difference(a, b):
    require(len(a) == len(b), 'output row count mismatch')
    result = 0.0
    for xrow, yrow in zip(a, b):
        require(len(xrow) == len(yrow), 'output width mismatch')
        for x, y in zip(xrow, yrow):
            require(math.isfinite(x) and math.isfinite(y), 'nonfinite output')
            result = max(result, abs(x - y))
    return result


def run_case(name, scores, values, valid, dv):
    reference, ref_lse = dense_reference(scores, values, valid, dv)
    largest = 0.0
    prefix_count = 0
    for Bq in (1, 2, 4):
        for Bk in (1, 2, 3, 7):
            out, lse, traces = tiled_attention(scores, values, valid, Bq, Bk, dv)
            for i in range(len(scores)):
                for t in range(dv):
                    largest = max(largest, close_scalar(out[i][t], reference[i][t], name + '/O'))
                close_scalar(lse[i], ref_lse[i], name + '/LSE')
                for stop, m, den, u in traces[i]:
                    rm, rd, ru = decimal_state(scores[i], values, valid[i], stop, dv)
                    close_scalar(m, rm, name + '/prefix_m')
                    close_scalar(den, rd, name + '/prefix_l')
                    for x, y in zip(u, ru):
                        close_scalar(x, y, name + '/prefix_U')
                    prefix_count += 1
    print('correct ' + name + ': max_output_error=' + format(largest, '.3e') +
          ', checked_prefix_states=' + str(prefix_count))
    return reference


def expect_fault(name, scores, values, valid, dv, threshold):
    reference, _ = dense_reference(scores, values, valid, dv)
    out, _, _ = tiled_attention(scores, values, valid, 1, 1, dv, faulty=name)
    diff = output_difference(out, reference)
    require(diff > threshold, 'negative control not detected: ' + name)
    print('negative ' + name + ': output=' + str(out) +
          ', reference=' + str(reference) + ', max_error=' + format(diff, '.3e'))


def check_validation():
    base = dict(L=1, S=2, Bq=1, Bk=1, d=1, dv=2, D=1)
    count = 0
    for field in base:
        bad = (-1, 1.5, True) if field in ('L', 'S') else (0, -1, 1.5, True)
        for value in bad:
            args = dict(base)
            args[field] = value
            try:
                validate_geometry(**args)
            except ValueError:
                count += 1
            else:
                raise AssertionError('illegal geometry accepted: ' + field)
    for field in ('Bq', 'Bk', 'd', 'dv', 'D'):
        args = dict(Bq=1, Bk=1, dv=2, d=1, D=1)
        args[field] = 0
        for scores, values, valid in (([[0, 0]], [[1, 0], [0, 1]], [[True, True]]),
                                      ([], [], [])):
            try:
                tiled_attention(scores, values, valid, **args)
            except ValueError:
                count += 1
            else:
                raise AssertionError('zero parameter accepted before loop: ' + field)
    print('legality: rejected ' + str(count) + ' invalid geometry calls, including L=0 with zero parameters')


def logical_matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def check_coordinates():
    q = [[1, 2, -1], [3, -2, 4]]
    k = [[2, 1, 3], [-1, 4, 2], [5, -2, 1], [0, 3, -4]]
    v = [[1, 2], [3, -1], [-2, 4], [5, 3]]
    q_stationary = [[q[r][p] for r in range(2)] for p in range(3)]
    k_moving = [[k[c][p] for c in range(4)] for p in range(3)]
    mapped_qk = [[sum(q_stationary[p][r] * k_moving[p][c] for p in range(3))
                  for c in range(4)] for r in range(2)]
    dense_qk = logical_matmul(q, [list(column) for column in zip(*k)])
    require(mapped_qk == dense_qk, 'QK coordinate mapping mismatch')
    weights = [[1, 2, 3, 4], [5, 1, 0, 2]]
    probability_stationary = [[weights[r][c] for r in range(2)] for c in range(4)]
    mapped_pv = [[sum(probability_stationary[c][r] * v[c][t] for c in range(4))
                  for t in range(2)] for r in range(2)]
    dense_pv = logical_matmul(weights, v)
    require(mapped_pv == dense_pv, 'PV coordinate mapping mismatch')
    faulty_pv = [[sum(probability_stationary[3-c][r] * v[c][t] for c in range(4))
                  for t in range(2)] for r in range(2)]
    require(faulty_pv != dense_pv, 'wrong key pairing negative control escaped')
    print('coordinates: integer QK/PV match dense logical matrices; wrong key pairing detected')


def check_dependency_trace(trace):
    prerequisites = {
        'weights_ready': set(),
        'pv_layout_done': {'weights_ready'},
        'pv_inputs_visible': {'pv_layout_done'},
        'pv_issue': {'pv_inputs_visible'},
        'pv_matrix_done': {'pv_issue'},
        'psum_to_sbuf_done': {'pv_matrix_done'},
        'tileU_visible': {'psum_to_sbuf_done'},
        'state_commit': {'tileU_visible'},
    }
    seen = set()
    for event in trace:
        if event not in prerequisites or event in seen:
            raise ValueError('unknown or duplicate event')
        if not prerequisites[event].issubset(seen):
            raise ValueError('consumer before required producer: ' + event)
        seen.add(event)
    if seen != set(prerequisites):
        raise ValueError('incomplete trace')


def check_dependencies():
    trace = ['weights_ready', 'pv_layout_done', 'pv_inputs_visible', 'pv_issue',
             'pv_matrix_done', 'psum_to_sbuf_done', 'tileU_visible', 'state_commit']
    check_dependency_trace(trace)
    faults = [
        ['weights_ready', 'pv_issue', 'pv_layout_done', 'pv_inputs_visible',
         'pv_matrix_done', 'psum_to_sbuf_done', 'tileU_visible', 'state_commit'],
        ['weights_ready', 'pv_layout_done', 'pv_issue', 'pv_inputs_visible',
         'pv_matrix_done', 'psum_to_sbuf_done', 'tileU_visible', 'state_commit'],
        ['weights_ready', 'pv_layout_done', 'pv_inputs_visible', 'pv_issue',
         'psum_to_sbuf_done', 'pv_matrix_done', 'tileU_visible', 'state_commit'],
        ['weights_ready', 'pv_layout_done', 'pv_inputs_visible', 'pv_issue',
         'pv_matrix_done', 'state_commit', 'psum_to_sbuf_done', 'tileU_visible'],
        trace[:6] + ['state_commit'],
    ]
    for bad in faults:
        try:
            check_dependency_trace(bad)
        except ValueError:
            continue
        raise AssertionError('unsafe PV dependency trace accepted')
    print('dependencies: valid abstract trace accepted; ' + str(len(faults)) +
          ' producer/visibility/completion/handoff faults rejected')


def main():
    print('K-HW-008: deterministic CPU binary64/Decimal and integer models; no Neuron or GPU measurement')
    check_validation()
    scores = [[((i + 1) * (j + 2)) % 7 - 3 for j in range(5)] for i in range(5)]
    values = [[(j + 1) % 3, (j + 2) % 3] for j in range(5)]
    causal = [[j <= i for j in range(5)] for i in range(5)]
    run_case('causal_tail', scores, values, causal, 2)
    pad_scores = [[1, 2, 3, 20, 21] for _ in range(3)]
    pad_values = [[1, 0], [0, 1], [1, 1], [100, -100], [-100, 100]]
    pad_valid = [[j < 3 for j in range(5)] for _ in range(3)]
    pad_ref = run_case('padding_poison', pad_scores, pad_values, pad_valid, 2)
    changed_scores = [[1, 2, 3, -1000, 1000] for _ in range(3)]
    changed_values = [[1, 0], [0, 1], [1, 1], [999, 999], [-999, -999]]
    changed, _, _ = tiled_attention(changed_scores, changed_values, pad_valid, 2, 2, 2)
    require(output_difference(changed, pad_ref) < 1e-12, 'masked poison changed output')
    expect_fault('include_padding', pad_scores, pad_values, pad_valid, 2, 1.0)
    equal_scores, unit_v, valid = [[0, 0]], [[1, 0], [0, 1]], [[True, True]]
    equal_ref = run_case('equal_scores_two_blocks', equal_scores, unit_v, valid, 2)
    require(equal_ref == [[0.5, 0.5]], 'equal-score closed form mismatch')
    expect_fault('missing_m_writeback', equal_scores, unit_v, valid, 2, 0.49)
    run_case('rising_maximum', [[0, 10]], unit_v, valid, 2)
    expect_fault('no_rescale', [[0, 10]], unit_v, valid, 2, 0.49)
    run_case('empty_query', [], [], [], 2)
    zero_keys = run_case('empty_keys', [[], []], [], [[], []], 3)
    require(zero_keys == [[0.0] * 3, [0.0] * 3], 'S=0 lost explicit output width')
    run_case('all_masked_and_empty_middle',
             [[0, 100, 200, 2], [0, 1, 2, 3], [1, 2, 3, 4]],
             [[1, 0], [7, 9], [-4, 8], [0, 1]],
             [[True, False, False, True], [False, False, False, False],
              [False, True, True, False]], 2)
    run_case('extreme_finite', [[-1000, 1000, 999, -999]],
             [[1], [2], [-3], [4]], [[True] * 4], 1)
    check_coordinates()
    check_dependencies()
    print('scope: online state/O/LSE, input legality, logical coordinates, abstract PV dependencies')
    print('unmeasured: NKI compilation, NEFF, real layouts/banks, DMA visibility, concurrent reuse, FP8 encoding and performance')


if __name__ == '__main__':
    main()
