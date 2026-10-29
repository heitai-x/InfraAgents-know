import math
import json
from collections import Counter
from decimal import Decimal, localcontext
from itertools import product


class CheckFailure(Exception):
    pass


class ContractError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise CheckFailure(message)


def ensure(condition, message):
    if not condition:
        raise ContractError(message)


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-11 + 1e-10 * abs(b)


def compare(got, expected, label):
    require(len(got) == len(expected), label + ': length')
    for i, (a, b) in enumerate(zip(got, expected)):
        require(close(a, b), label + ': element ' + str(i))


def reject(action, label, exception=ContractError):
    try:
        action()
    except exception as exc:
        return str(exc)
    raise CheckFailure(label + ': faulty path was accepted')


def test_layout_and_indices():
    original = list(range(48))
    reordered = [original[b * 24 + l * 8 + h * 4 + d]
                 for b, h, l, d in product(range(2), range(2), range(3), range(4))]
    mismatches = 0
    for b, h, l, d in product(range(2), range(2), range(3), range(4)):
        view_address = b * 24 + h * 4 + l * 8 + d
        new_address = b * 24 + h * 12 + l * 4 + d
        require(original[view_address] == reordered[new_address], 'axis reorder reference')
        mismatches += original[new_address] != reordered[new_address]
    require(mismatches > 0, 'flat copy negative control not exposed')
    reject(lambda: compare(original, reordered, 'flat copy'), 'flat copy', CheckFailure)

    table = [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]
    ids = [2, 0, 2]

    def gather(indices):
        ensure(all(isinstance(i, int) and 0 <= i < len(table) for i in indices), 'gather bounds')
        return [list(table[i]) for i in indices]

    expected = [[30.0, 31.0], [10.0, 11.0], [30.0, 31.0]]
    require(gather(ids) == expected, 'gather multiplicity/order')
    require(gather([]) == [], 'empty gather')
    reject(lambda: gather([-1]), 'negative gather index')
    reject(lambda: gather([3]), 'large gather index')
    bad = gather(sorted(set(ids)))
    reject(lambda: require(bad == expected, 'duplicate output lost'), 'deduplicated gather', CheckFailure)

    def scatter(mode):
        out = [0.0] * 3
        for i, value in zip([1, 1, 2], [2.0, 3.0, 4.0]):
            if mode == 'add':
                out[i] += value
            else:
                out[i] = value
        return out

    compare(scatter('add'), [0.0, 5.0, 4.0], 'scatter add')
    compare(scatter('assign'), [0.0, 3.0, 4.0], 'ordered scatter assign')
    reject(lambda: compare(scatter('assign'), [0.0, 5.0, 4.0], 'wrong mode'), 'scatter mode', CheckFailure)
    return {'axis_positions': 48, 'flat_copy_mismatches': mismatches,
            'gather': 'duplicates retained; invalid indices rejected',
            'scatter': 'assign/add distinguished'}


def gemm_tiled(x, w, bias, bm, bn, bk, bias_per_chunk=False):
    m, k, n = len(x), len(w), len(bias)
    ensure(bm > 0 and bn > 0 and bk > 0, 'positive tiles')
    ensure(all(len(row) == k for row in x), 'X shape')
    ensure(all(len(row) == n for row in w), 'W shape')
    y = [[0.0] * n for _ in range(m)]
    for m0 in range(0, m, bm):
        for n0 in range(0, n, bn):
            acc = [[0.0] * bn for _ in range(bm)]
            for k0 in range(0, k, bk):
                for i in range(bm):
                    for j in range(bn):
                        if m0 + i < m and n0 + j < n:
                            for kk in range(bk):
                                if k0 + kk < k:
                                    acc[i][j] += x[m0 + i][k0 + kk] * w[k0 + kk][n0 + j]
                            if bias_per_chunk:
                                acc[i][j] += bias[n0 + j]
            for i in range(bm):
                for j in range(bn):
                    if m0 + i < m and n0 + j < n:
                        y[m0 + i][n0 + j] = acc[i][j] + (0.0 if bias_per_chunk else bias[n0 + j])
    return y


def test_gemm_and_sizes():
    x = [[1.0, 2.0, 3.0, 4.0, 5.0],
         [0.5, -1.0, 2.0, -2.0, 1.0],
         [-3.0, 0.25, 1.0, 2.0, -1.0]]
    w = [[0.1, -0.2, 0.3, 0.4], [0.5, 0.6, -0.7, 0.8],
         [-0.9, 1.0, 1.1, -1.2], [0.2, -0.3, 0.4, -0.5],
         [0.6, 0.7, -0.8, 0.9]]
    bias = [0.1, -0.2, 0.3, 0.4]
    ref = [[math.fsum(xv * wv for xv, wv in zip(row, column)) + b
            for column, b in zip(zip(*w), bias)] for row in x]
    for tiles in [(2, 3, 2), (4, 5, 7), (1, 1, 1)]:
        got = gemm_tiled(x, w, bias, *tiles)
        for actual, expected in zip(got, ref):
            compare(actual, expected, 'tiled GEMM')
    bad = gemm_tiled(x, w, bias, 2, 3, 2, True)
    reject(lambda: compare(sum(bad, []), sum(ref, []), 'bias repeated'), 'bias per K tile', CheckFailure)
    require(gemm_tiled([], w, bias, 2, 3, 2) == [], 'M=0')
    require(gemm_tiled([[], []], [], [0.5, -0.25], 2, 3, 2) == [[0.5, -0.25], [0.5, -0.25]], 'K=0 epilogue')
    require(gemm_tiled([[1.0]], [[]], [], 1, 1, 1) == [[]], 'N=0')
    file_weight = [[3.0, 4.0], [5.0, 6.0]]
    math_weight = [list(c) for c in zip(*file_weight)]
    correct = gemm_tiled([[1.0, 2.0]], math_weight, [0.0, 0.0], 1, 2, 1)[0]
    compare(correct, [11.0, 17.0], 'specified file axes')
    wrong = gemm_tiled([[1.0, 2.0]], file_weight, [0.0, 0.0], 1, 2, 1)[0]
    reject(lambda: compare(wrong, correct, 'file axes swapped'), 'weight axes', CheckFailure)
    m, k, n, s = 32, 4096, 4096, 2
    flops = 2 * m * k * n
    data_bytes = s * (m * k + k * n + m * n)
    require(flops == 1073741824 and data_bytes == 34078720, 'linear arithmetic')
    q_width, kv_width = 32 * 128, 8 * 128
    width = q_width + 2 * kv_width
    require(width == 6144, 'GQA projection width')
    require(m * width * s == 393216 and k * width * s == 50331648, 'QKV bytes')
    return {'tail_tiles': 3, 'zero_dimensions': 'M/N/K covered',
            'weight_axis_example': correct, 'intensity_M32': flops / data_bytes,
            'qkv_width': width, 'size_scope': 'analytic arithmetic, not transactions'}


def rms_scalar(values, gamma, epsilon, divisor=None):
    ensure(len(values) > 0 and len(values) == len(gamma), 'RMS shape')
    ensure(epsilon > 0 and all(math.isfinite(x) for x in values + gamma), 'RMS finite domain')
    denominator = len(values) if divisor is None else divisor
    scale = math.sqrt(sum(x * x for x in values) / denominator + epsilon)
    return [g * x / scale for g, x in zip(gamma, values)]


def rms_reference(values, gamma, epsilon):
    with localcontext() as ctx:
        ctx.prec = 60
        dv = [Decimal.from_float(v) for v in values]
        dg = [Decimal.from_float(g) for g in gamma]
        q = sum((v * v for v in dv), Decimal(0)) / Decimal(len(dv))
        scale = (q + Decimal.from_float(epsilon)).sqrt()
        return [float(g * v / scale) for g, v in zip(dg, dv)]


def test_rms():
    values = [0.25, -2.0, 3.5, 0.0, 1.25]
    gamma = [1.0, 0.5, -0.75, 2.0, 1.5]
    epsilon = 1e-5
    compare(rms_scalar(values, gamma, epsilon), rms_reference(values, gamma, epsilon), 'RMS Decimal')
    compare(rms_scalar([0.0] * 5, gamma, epsilon), [0.0] * 5, 'zero RMS row')
    h, padded = 3072, 4096
    good = rms_scalar([1.0] * h, [1.0] * h, epsilon)
    bad = rms_scalar([1.0] * h, [1.0] * h, epsilon, padded)
    compare(good, [1.0 / math.sqrt(1.0 + epsilon)] * h, 'all-one closed form')
    reject(lambda: compare(bad, good, 'padded divisor'), 'RMS padding', CheckFailure)
    reject(lambda: rms_scalar([], [], epsilon), 'empty hidden axis')
    return {'H': h, 'P': padded, 'wrong_divisor_ratio': bad[0] / good[0],
            'reference': 'Decimal and all-one closed form'}


class BufferModel:
    def __init__(self, x, r, stale_u=False):
        self.data = {'x': list(x), 'r': list(r), 'u': [None] * len(x), 'y': [None] * len(x)}
        if stale_u:
            self.data['u'] = [123.0 + i for i in range(len(x))]
        self.events = []

    def read(self, stage, name, index):
        ensure(0 <= index < len(self.data[name]), 'buffer read bounds')
        self.events.append((stage, 'read', name, index))
        value = self.data[name][index]
        ensure(value is not None, 'read-before-write:' + name + '[' + str(index) + ']')
        return value

    def write(self, stage, name, index, value):
        ensure(0 <= index < len(self.data[name]), 'buffer write bounds')
        self.events.append((stage, 'write', name, index))
        self.data[name][index] = value


def residual_execute(x, r, rows, hidden, gamma, epsilon, fused, active, omit_u=False, stale_u=False):
    ensure(len(x) == len(r) == rows * hidden and hidden > 0, 'residual shape')
    memory = BufferModel(x, r, stale_u)
    if not fused:
        for i in range(len(x)):
            value = memory.read('add', 'x', i) + memory.read('add', 'r', i)
            memory.write('add', 'u', i, value)
        for row in range(rows):
            indices = list(range(row * hidden, (row + 1) * hidden))
            local = [memory.read('norm', 'u', i) for i in indices]
            values = rms_scalar(local, gamma, epsilon)
            for i, value in zip(indices, values):
                memory.write('norm', 'y', i, value)
    else:
        for row in range(rows):
            indices = list(range(row * hidden, (row + 1) * hidden))
            local = [memory.read('fused', 'x', i) + memory.read('fused', 'r', i) for i in indices]
            values = rms_scalar(local, gamma, epsilon)
            for i, u_value, y_value in zip(indices, local, values):
                if active and not omit_u:
                    memory.write('fused', 'u', i, u_value)
                memory.write('fused', 'y', i, y_value)
    return memory


def residual_consumer(memory, expected_u, expected_y):
    result = []
    for i in range(len(expected_u)):
        u = memory.read('consumer', 'u', i)
        y = memory.read('consumer', 'y', i)
        require(close(u, expected_u[i]), 'consumer saw stale residual at ' + str(i))
        result.append(u + y)
    compare(result, [u + y for u, y in zip(expected_u, expected_y)], 'downstream residual branch')
    return result


def test_residual_events():
    rows, hidden = 2, 5
    x = [0.5, -1.0, 2.0, 0.0, 3.0, -0.25, 1.5, -2.0, 4.0, 0.75]
    r = [1.0, 0.25, -0.5, 2.0, 0.5, 0.75, -0.5, 1.0, -1.0, 2.0]
    gamma, epsilon = [1.0, 0.5, 1.5, -0.75, 2.0], 1e-5
    expected_u = [a + b for a, b in zip(x, r)]
    expected_y = []
    for row in range(rows):
        expected_y.extend(rms_reference(expected_u[row * hidden:(row + 1) * hidden], gamma, epsilon))
    counts, summaries = {}, {}
    cases = [('independent', False, True), ('fused_active', True, True), ('fused_dead', True, False)]
    for name, fused, active in cases:
        memory = residual_execute(x, r, rows, hidden, gamma, epsilon, fused, active)
        kernel_events = tuple(memory.events)
        expected_events = []
        for i in range(len(x)):
            if fused:
                expected_events.extend([('fused', 'read', 'x', i), ('fused', 'read', 'r', i), ('fused', 'write', 'y', i)])
                if active:
                    expected_events.append(('fused', 'write', 'u', i))
            else:
                expected_events.extend([('add', 'read', 'x', i), ('add', 'read', 'r', i),
                                        ('add', 'write', 'u', i), ('norm', 'read', 'u', i),
                                        ('norm', 'write', 'y', i)])
        require(Counter(kernel_events) == Counter(expected_events), name + ': access coverage/multiplicity')
        compare(memory.data['y'], expected_y, name + ': y')
        if active:
            residual_consumer(memory, expected_u, expected_y)
        else:
            require(all(v is None for v in memory.data['u']), 'dead residual was materialized')
        counts[name] = len(kernel_events)
        grouped = Counter((stage, op, buf) for stage, op, buf, index in kernel_events)
        summaries[name] = {'/'.join(key): value for key, value in sorted(grouped.items())}

    missing = residual_execute(x, r, rows, hidden, gamma, epsilon, True, True, omit_u=True)
    compare(missing.data['y'], expected_y, 'faulty path still has correct y')
    missing_error = reject(lambda: residual_consumer(missing, expected_u, expected_y), 'missing active u')
    require(missing_error == 'read-before-write:u[0]', 'unexpected missing-u failure')
    stale = residual_execute(x, r, rows, hidden, gamma, epsilon, True, True, omit_u=True, stale_u=True)
    compare(stale.data['y'], expected_y, 'stale path still has correct y')
    stale_error = reject(lambda: residual_consumer(stale, expected_u, expected_y), 'stale active u', CheckFailure)
    return {'N': len(x), 'event_counts': counts, 'events_by_buffer': summaries,
            'consumer_reads_excluded_from_kernel_counts': True,
            'missing_write_negative': missing_error, 'stale_write_negative': stale_error,
            'scope': 'logical external element accesses; gamma and local temporaries excluded'}


def rotate_candidate(vector, position, frequencies, pairing, wrong_direction=False):
    out = list(vector)
    half = len(frequencies)
    for i, frequency in enumerate(frequencies):
        a, b = (2 * i, 2 * i + 1) if pairing == 'adjacent' else (i, i + half)
        angle = position * frequency * (-1 if wrong_direction else 1)
        c, s = math.cos(angle), math.sin(angle)
        va, vb = vector[a], vector[b]
        out[a], out[b] = va * c - vb * s, va * s + vb * c
    return out


def rotate_reference(vector, position, frequencies, pairing):
    result = list(vector)
    for i in range(len(frequencies)):
        if pairing == 'adjacent':
            real_index, imag_index = 2 * i, 2 * i + 1
        else:
            real_index, imag_index = i, len(frequencies) + i
        value = complex(vector[real_index], vector[imag_index])
        angle = position * frequencies[i]
        rotated = value * complex(math.cos(angle), math.sin(angle))
        result[real_index], result[imag_index] = rotated.real, rotated.imag
    return result


def make_cache(request, start):
    return {'request': request, 'ready_len': start,
            'rows': {p: ('old', request, p) for p in range(start)}}


def rotate_chunk(q, k, q_keys, k_keys, request, start, width, frequencies, pairing, cache, capacity,
                 wrong_direction=False, reset_position=False):
    ensure(len(q) == len(k), 'Q/K chunk lengths differ')
    total = len(q)
    ensure(width > 0 and 2 * len(frequencies) <= width, 'rotary dimension')
    ensure(pairing in ('adjacent', 'half'), 'pairing')
    ensure(isinstance(start, int) and start >= 0, 'position domain')
    ensure(all(math.isfinite(f) for f in frequencies), 'finite frequencies')
    ensure(all(len(row) == width for row in q + k), 'row width')
    ensure(all(math.isfinite(v) for row in q + k for v in row), 'finite vectors')
    expected_keys = [(request, start + j) for j in range(total)]
    ensure(q_keys == expected_keys and k_keys == expected_keys, 'Q/K row identities differ')
    ensure(cache['request'] == request and cache['ready_len'] == start, 'cache request/start')
    ensure(set(cache['rows']) == set(range(start)), 'cache prefix coverage')
    ensure(start + total <= capacity, 'cache capacity')
    qo, ko = [], []
    for j in range(total):
        position = j if reset_position else start + j
        qr = rotate_candidate(q[j], position, frequencies, pairing, wrong_direction)
        kr = rotate_candidate(k[j], position, frequencies, pairing, wrong_direction)
        qo.append(qr)
        ko.append(kr)
        cache['rows'][start + j] = tuple(kr)
    cache['ready_len'] = start + total
    return qo, ko


def test_rope_chunk():
    q = [[0.9, -0.4, 0.7, 1.3, 8.0], [1.0, 2.0, -0.5, 0.25, 9.0], [-0.25, 0.75, 2.0, -1.0, 10.0]]
    k = [[-0.2, 1.1, 0.5, -0.8, 11.0], [0.75, -0.5, 1.25, 2.0, 12.0], [2.0, 0.5, -1.0, 0.75, 13.0]]
    request, start, width = 'A', 7, 5
    frequencies = [0.3, 0.7]
    keys = [(request, start + j) for j in range(3)]
    for pairing in ('adjacent', 'half'):
        cache = make_cache(request, start)
        old = dict(cache['rows'])
        qo, ko = rotate_chunk(q, k, keys, keys, request, start, width, frequencies, pairing, cache, 10)
        for j in range(3):
            compare(qo[j], rotate_reference(q[j], start + j, frequencies, pairing), 'Q complex reference')
            compare(ko[j], rotate_reference(k[j], start + j, frequencies, pairing), 'K complex reference')
            require(qo[j][-1] == q[j][-1] and ko[j][-1] == k[j][-1], 'unrotated tail')
            require(cache['rows'][start + j] == tuple(ko[j]), 'cache append content')
        require(all(cache['rows'][p] == value for p, value in old.items()), 'old K modified')
        require(cache['ready_len'] == 10 and len(cache['rows']) == 10, 'append coverage')
        split = make_cache(request, start)
        qa, ka = rotate_chunk(q[:1], k[:1], keys[:1], keys[:1], request, start, width, frequencies, pairing, split, 10)
        qb, kb = rotate_chunk(q[1:], k[1:], keys[1:], keys[1:], request, start + 1, width, frequencies, pairing, split, 10)
        compare(sum(qa + qb, []), sum(qo, []), 'cross-chunk Q')
        compare(sum(ka + kb, []), sum(ko, []), 'cross-chunk K')
        require(split == cache, 'cross-chunk cache state')
        empty_before = (cache['ready_len'], dict(cache['rows']))
        require(rotate_chunk([], [], [], [], request, 10, width, frequencies, pairing, cache, 10) == ([], []), 'empty chunk')
        require(empty_before == (cache['ready_len'], cache['rows']), 'empty chunk changed cache')
        for wrong_direction, reset_position, label in [(True, False, 'both directions reversed'), (False, True, 'positions reset')]:
            bad_cache = make_cache(request, start)
            bq, bk = rotate_chunk(q, k, keys, keys, request, start, width, frequencies, pairing,
                                  bad_cache, 10, wrong_direction, reset_position)
            reject(lambda: compare(sum(bq + bk, []), sum(qo + ko, []), label), label, CheckFailure)

    compare(rotate_candidate([1.0, 2.0, 9.0], 1, [math.pi / 2], 'adjacent'), [-2.0, 1.0, 9.0], 'fixed positive quarter turn')
    compare(rotate_candidate(q[0], 999, [], 'adjacent'), q[0], 'R=0 identity')
    rotated = rotate_candidate(q[0], 5, frequencies, 'adjacent')
    require(close(math.fsum(v * v for v in rotated), math.fsum(v * v for v in q[0])), 'rotation norm')
    qp = rotate_candidate(q[0], 5, frequencies, 'adjacent')
    kr = rotate_candidate(k[0], 2, frequencies, 'adjacent')
    relative = rotate_reference(k[0], -3, frequencies, 'adjacent')
    require(close(math.fsum(a * b for a, b in zip(qp, kr)), math.fsum(a * b for a, b in zip(q[0], relative))), 'relative inner product')

    def invalid_call(qin, kin, qids, kids, capacity=10):
        cache = make_cache(request, start)
        before = (cache['ready_len'], dict(cache['rows']))
        error = reject(lambda: rotate_chunk(qin, kin, qids, kids, request, start, width, frequencies, 'adjacent', cache, capacity), 'invalid chunk')
        require(before == (cache['ready_len'], cache['rows']), 'invalid call partially wrote cache')
        return error

    length_error = invalid_call(q[:1], k[:2], keys[:1], keys[:2])
    other_request = [('B', p) for req, p in keys]
    identity_error = invalid_call(q, k, keys, other_request)
    invalid_call(q, k, keys, list(reversed(keys)))
    invalid_call(q, k, keys, keys, 9)

    def faulty_common_loop(qin, kin):
        qo, ko = [], []
        for j, (qrow, krow) in enumerate(zip(qin, kin)):
            qo.append(rotate_candidate(qrow, start + j, frequencies, 'adjacent'))
            ko.append(rotate_candidate(krow, start + j, frequencies, 'adjacent'))
        return qo, ko

    bad_q, bad_k = faulty_common_loop(q[:1], k[:2])
    reject(lambda: require(len(bad_q) == 1 and len(bad_k) == 2, 'common loop omitted a K row'), 'truncated common loop', CheckFailure)
    return {'positions': [7, 8, 9], 'pairings': ['adjacent', 'half'],
            'cross_chunk': 'same outputs and cache', 'length_rejection': length_error,
            'identity_rejection': identity_error,
            'negative_controls': ['truncated common loop', 'both directions reversed', 'positions reset'],
            'scope': 'single request/head scalar rotations and synchronous K append'}


def softmax_state(z, values, visible):
    ensure(len(z) == len(values) == len(visible), 'softmax row lengths')
    width = len(values[0]) if values else 2
    ensure(all(len(v) == width for v in values), 'value width')
    indices = [i for i, use in enumerate(visible) if use]
    ensure(all(math.isfinite(z[i]) and all(math.isfinite(x) for x in values[i]) for i in indices), 'finite visible softmax domain')
    if not indices:
        return (-math.inf, 0.0, [0.0] * width)
    maximum = max(z[i] for i in indices)
    weights = [math.exp(z[i] - maximum) for i in indices]
    mass = math.fsum(weights)
    weighted = [math.fsum(w * values[i][c] for w, i in zip(weights, indices)) for c in range(width)]
    return maximum, mass, weighted


def merge_states(first, second, omit_u_rescale=False):
    m1, l1, u1 = first
    m2, l2, u2 = second
    if l1 == 0.0:
        return second
    if l2 == 0.0:
        return first
    maximum = max(m1, m2)
    a, b = math.exp(m1 - maximum), math.exp(m2 - maximum)
    mass = a * l1 + b * l2
    weighted = [x + y for x, y in zip(u1, u2)] if omit_u_rescale else [a * x + b * y for x, y in zip(u1, u2)]
    return maximum, mass, weighted


def softmax_reference(z, values, visible):
    indices = [i for i, use in enumerate(visible) if use]
    if not indices:
        return [0.0] * len(z), [0.0, 0.0], -math.inf
    with localcontext() as ctx:
        ctx.prec = 70
        # Direct unshifted Decimal exponentials are independent of the candidate state/merge.
        weights = {i: Decimal.from_float(z[i]).exp() for i in indices}
        total = sum(weights.values(), Decimal(0))
        probs = [float(weights[i] / total) if i in weights else 0.0 for i in range(len(z))]
        output = [float(sum((weights[i] * Decimal.from_float(values[i][c]) for i in indices), Decimal(0)) / total) for c in range(2)]
        return probs, output, float(total.ln())


def check_softmax_state(state, z, values, visible):
    probs, output, lse = softmax_reference(z, values, visible)
    maximum, mass, weighted = state
    if not any(visible):
        require(maximum == -math.inf and mass == 0.0 and weighted == [0.0, 0.0], 'empty state')
        return
    actual = [math.exp(value - maximum) / mass if use else 0.0 for value, use in zip(z, visible)]
    compare(actual, probs, 'softmax probabilities')
    compare([v / mass for v in weighted], output, 'weighted U/mass')
    require(close(maximum + math.log(mass), lse), 'natural LSE')
    require(close(math.fsum(actual), 1.0), 'probability mass')


def test_softmax():
    z = [1000.0, 999.0, 998.0, 1001.0, -20.0]
    values = [[1.0, -2.0], [99.0, 99.0], [3.0, 4.0], [-1.0, 0.5], [7.0, -3.0]]
    visible = [True, False, True, True, True]
    whole = softmax_state(z, values, visible)
    check_softmax_state(whole, z, values, visible)
    empty = softmax_state([], [], [])
    parts = [softmax_state(z[:2], values[:2], visible[:2]), empty,
             softmax_state(z[2:3], values[2:3], visible[2:3]),
             softmax_state(z[3:], values[3:], visible[3:]), empty]
    state = empty
    for part in parts:
        state = merge_states(state, part)
    check_softmax_state(state, z, values, visible)
    reversed_state = empty
    for part in reversed(parts):
        reversed_state = merge_states(reversed_state, part)
    check_softmax_state(reversed_state, z, values, visible)
    require(merge_states(whole, empty) == whole and merge_states(empty, whole) == whole, 'empty identity')
    check_softmax_state(softmax_state([0.0, 0.0], [[1.0, 2.0], [3.0, 4.0]], [False, False]),
                        [0.0, 0.0], [[1.0, 2.0], [3.0, 4.0]], [False, False])
    check_softmax_state(empty, [], [], [])
    bad = empty
    for part in parts:
        bad = merge_states(bad, part, True)
    reject(lambda: check_softmax_state(bad, z, values, visible), 'U rescale omitted', CheckFailure)
    probs, output, lse = softmax_reference(z, values, visible)
    concatenated = []
    for lo, hi in [(0, 2), (2, 5)]:
        local_probs, unused_output, unused_lse = softmax_reference(z[lo:hi], values[lo:hi], visible[lo:hi])
        concatenated.extend(local_probs)
    reject(lambda: compare(concatenated, probs, 'independent chunk normalization'), 'chunk concat', CheckFailure)
    finite_state = softmax_state([-1e30] * 3, [[0.0, 0.0]] * 3, [True] * 3)
    finite_probs = [math.exp(-1e30 - finite_state[0]) / finite_state[1]] * 3
    reject(lambda: compare(finite_probs, [0.0] * 3, 'finite constant empty mask'), 'finite mask', CheckFailure)
    stored_log2 = lse / math.log(2.0)
    require(close(stored_log2 * math.log(2.0), lse), 'LSE base conversion')
    reject(lambda: require(close(stored_log2, lse), 'log2 interpreted as natural LSE'), 'LSE base', CheckFailure)
    reject(lambda: softmax_state([math.inf], [[1.0, 2.0]], [True]), 'positive infinity')
    reject(lambda: softmax_state([math.nan], [[1.0, 2.0]], [True]), 'NaN')
    return {'reference': '70-digit Decimal direct whole-row exponentials',
            'dimensions': '5 scores, 2 value channels', 'natural_LSE': lse,
            'empty_and_merge_orders': 'covered',
            'negative_controls': ['U rescale omitted', 'chunk concat', 'finite empty mask', 'LSE base mismatch']}


def main():
    observations = {
        'layout_indices': test_layout_and_indices(),
        'gemm_sizes': test_gemm_and_sizes(),
        'rms': test_rms(),
        'residual_events': test_residual_events(),
        'rope_chunk': test_rope_chunk(),
        'softmax': test_softmax(),
    }
    print('K-ALG-016 scoped observations')
    print(json.dumps(observations, sort_keys=True, indent=2, allow_nan=False))
    print('Model range: CPU binary64, Decimal references, and sequential logical buffer events only.')
    print('No PyTorch/CUDA/Triton execution, GPU measurements, actual memory transactions, low-precision simulation, concurrency proof, or model-quality evidence.')


if __name__ == '__main__':
    main()
