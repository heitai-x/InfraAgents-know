import math


class CheckFailure(Exception):
    pass


class AccessFault(Exception):
    def __init__(self, name, row, col):
        self.name = name
        self.row = row
        self.col = col
        super().__init__('out-of-shape access: %s[%d,%d]' % (name, row, col))


def require(condition, message):
    if not condition:
        raise CheckFailure(message)


def close(a, b, tol=1e-10):
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def vector_equal(actual, expected, label):
    require(len(actual) == len(expected), label + ': vector length')
    for index, (a, b) in enumerate(zip(actual, expected)):
        require(close(a, b), '%s[%d]: %r != %r' % (label, index, a, b))


def matrix_equal(actual, expected, label):
    require(len(actual) == len(expected), label + ': row count')
    for index, (a, b) in enumerate(zip(actual, expected)):
        vector_equal(a, b, label + ' row ' + str(index))


def rejected(check, label):
    try:
        check()
    except CheckFailure:
        return
    raise CheckFailure('negative control was accepted: ' + label)


GUARD = object()


class StridedMatrix:
    def __init__(self, name, rows, cols, row_stride, col_stride, offset, values=None):
        self.name = name
        self.rows = rows
        self.cols = cols
        self.rs = row_stride
        self.cs = col_stride
        self.offset = offset
        last = offset
        if rows and cols:
            last += (rows - 1) * row_stride + (cols - 1) * col_stride
        self.data = [GUARD] * (last + 5)
        self.valid_addresses = set()
        for r in range(rows):
            for c in range(cols):
                address = offset + r * row_stride + c * col_stride
                require(address not in self.valid_addresses, 'overlapping model storage')
                self.valid_addresses.add(address)
                if values is not None:
                    self.data[address] = values[r][c]

    def location(self, row, col):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise AccessFault(self.name, row, col)
        return self.offset + row * self.rs + col * self.cs

    def load(self, row, col, trace):
        address = self.location(row, col)
        trace.append((self.name, row, col, 4 * address))
        require(self.data[address] is not GUARD, 'uninitialized input consumed')
        return self.data[address]

    def store(self, row, col, value, trace):
        address = self.location(row, col)
        trace.append((self.name, row, col, 4 * address))
        require(self.data[address] is GUARD, 'duplicate output writer')
        self.data[address] = value

    def dense(self):
        return [[self.data[self.location(r, c)] for c in range(self.cols)] for r in range(self.rows)]

    def check_guards(self):
        for address, value in enumerate(self.data):
            if address not in self.valid_addresses:
                require(value is GUARD, 'padding was written: ' + self.name)


def make_gemm_inputs(m, n, k):
    a = [[((3 * r + 2 * t) % 9) - 4 for t in range(k)] for r in range(m)]
    b = [[((5 * t + 3 * c) % 11) - 5 for c in range(n)] for t in range(k)]
    return a, b


def direct_gemm(a, b, m, n, k, bias):
    result = []
    for r in range(m):
        row = []
        for c in range(n):
            value = sum(a[r][t] * b[t][c] for t in range(k))
            row.append(value if bias is None else value + bias[c])
        result.append(row)
    return result


def tiled_gemm(a_values, b_values, m, n, k, bm, bn, bk, bias=None, bad_mask=None, double_epilogue=False):
    require(min(m, n, k) >= 0 and min(bm, bn, bk) > 0, 'invalid GEMM domain')
    a = StridedMatrix('A', m, k, 2 * k + 3, 2, 3, a_values)
    b = StridedMatrix('B', k, n, 1, k + 2, 5, b_values)
    c = StridedMatrix('C', m, n, 2 * n + 3, 2, 7)
    reads, writes = [], []
    for m0 in range(0, m, bm):
        for n0 in range(0, n, bn):
            acc = [[0.0] * bn for _ in range(bm)]
            for k0 in range(0, k, bk):
                at = [[0.0] * bk for _ in range(bm)]
                bt = [[0.0] * bn for _ in range(bk)]
                for i in range(bm):
                    for t in range(bk):
                        valid = (m0 + i < m) and (k0 + t < k)
                        if bad_mask == 'A':
                            valid = (i < m0 + bm) and (k0 + t < k)
                        if valid:
                            at[i][t] = a.load(m0 + i, k0 + t, reads)
                for t in range(bk):
                    for j in range(bn):
                        valid = (k0 + t < k) and (n0 + j < n)
                        if bad_mask == 'B':
                            valid = (k0 + t < k) and (j < n0 + bn)
                        if valid:
                            bt[t][j] = b.load(k0 + t, n0 + j, reads)
                for i in range(bm):
                    for j in range(bn):
                        for t in range(bk):
                            acc[i][j] += at[i][t] * bt[t][j]
            for i in range(bm):
                for j in range(bn):
                    if m0 + i < m and n0 + j < n:
                        value = acc[i][j]
                        if bias is not None:
                            value += bias[n0 + j]
                            if double_epilogue:
                                value += bias[n0 + j]
                        c.store(m0 + i, n0 + j, value, writes)
    matrices = {'A': a, 'B': b, 'C': c}
    for name, row, col, byte_offset in reads + writes:
        matrix = matrices[name]
        require(0 <= row < matrix.rows and 0 <= col < matrix.cols, 'invalid traced coordinate')
        require(byte_offset == 4 * (matrix.offset + row * matrix.rs + col * matrix.cs), 'stride address mismatch')
    require(len(writes) == m * n, 'output coverage')
    require(len({(r, col) for _, r, col, _ in writes}) == m * n, 'unique output ownership')
    for matrix in matrices.values():
        matrix.check_guards()
    result = c.dense()
    require(all(value is not GUARD for row in result for value in row), 'missing output')
    return result, reads, writes


def test_gemm():
    cases = [(5, 5, 3), (1, 7, 5), (7, 1, 1), (0, 5, 3), (5, 0, 3), (5, 5, 0), (3, 2, 4)]
    for m, n, k in cases:
        a, b = make_gemm_inputs(m, n, k)
        for bias in (None, [c + 1.0 for c in range(n)]):
            actual, reads, writes = tiled_gemm(a, b, m, n, k, 4, 4, 2, bias)
            matrix_equal(actual, direct_gemm(a, b, m, n, k, bias), 'GEMM reference')
            if k == 0 or m == 0 or n == 0:
                require(not reads, 'empty multiplication performed input loads')
            if (m, n, k) == (5, 5, 3):
                require(sum(item[0] == 'A' for item in reads) == 30, 'A tail trace count')
                require(sum(item[0] == 'B' for item in reads) == 30, 'B tail trace count')
                require(len(writes) == 25, 'C tail trace count')
    a, b = make_gemm_inputs(5, 5, 3)
    for faulty in ('A', 'B'):
        try:
            tiled_gemm(a, b, 5, 5, 3, 4, 4, 2, bad_mask=faulty)
        except AccessFault as error:
            require(error.name == faulty, 'wrong input mask detected')
            require((error.row == 5 if faulty == 'A' else error.col == 5), 'unexpected fault coordinate')
        else:
            raise CheckFailure('original ' + faulty + ' mask was accepted')
    bias = [1.0, -2.0, 3.0, -4.0, 5.0]
    faulty, _, _ = tiled_gemm(a, b, 5, 5, 3, 4, 4, 2, bias, double_epilogue=True)
    expected = direct_gemm(a, b, 5, 5, 3, bias)
    rejected(lambda: matrix_equal(faulty, expected, 'double epilogue'), 'double epilogue')
    print('observed CPU model: 5x5x3 GEMM traces 30 A loads, 30 B loads, 25 unique stores; strided addresses and guards checked')
    print('observed CPU model: identity, single nonzero bias, K=0 and empty outputs match direct GEMM; old A/B masks and double bias rejected')


def empty_state(dv):
    return -math.inf, 0.0, [0.0] * dv


def direct_stats(scores, values, visible, end, dv):
    ids = [j for j in range(end) if visible[j]]
    if not ids:
        return empty_state(dv)
    top = max(scores[j] for j in ids)
    weights = [math.exp(scores[j] - top) for j in ids]
    mass = math.fsum(weights)
    weighted = [math.fsum(weight * values[j][t] for weight, j in zip(weights, ids)) for t in range(dv)]
    return top, mass, weighted


def direct_attention(scores, values, visible, dv):
    top, mass, weighted = direct_stats(scores, values, visible, len(scores), dv)
    if mass == 0.0:
        return [0.0] * dv, True, -math.inf
    return [value / mass for value in weighted], False, top + math.log(mass)


def local_state(scores, values, visible, start, end, block, base, dv, bad_padding=False):
    ys, vs = [], []
    for j in range(start, start + block):
        if j < end and visible[j]:
            ys.append(scores[j] / math.log(base))
            vs.append(values[j])
        elif bad_padding:
            ys.append(0.0)
            vs.append([0.0] * dv)
    if not ys:
        return empty_state(dv)
    top = max(ys)
    weights = [base ** (y - top) for y in ys]
    mass = sum(weights)
    weighted = [sum(weight * value[t] for weight, value in zip(weights, vs)) for t in range(dv)]
    return top, mass, weighted


def merge(x, y, base, bad_base=False):
    m1, l1, u1 = x
    m2, l2, u2 = y
    if l1 == 0.0 and l2 == 0.0:
        return empty_state(len(u1))
    if l1 == 0.0:
        return m2, l2, list(u2)
    if l2 == 0.0:
        return m1, l1, list(u1)
    top = max(m1, m2)
    if bad_base:
        r1, r2 = math.exp(m1 - top), math.exp(m2 - top)
    else:
        r1, r2 = base ** (m1 - top), base ** (m2 - top)
    return top, r1 * l1 + r2 * l2, [r1 * a + r2 * b for a, b in zip(u1, u2)]


def block_attention(scores, values, visible, block, base, dv, bad_base=False, bad_padding=False):
    state = empty_state(dv)
    history = []
    for start in range(0, len(scores), block):
        end = min(start + block, len(scores))
        local = local_state(scores, values, visible, start, end, block, base, dv, bad_padding)
        before = state
        state = merge(state, local, base, bad_base)
        if not bad_base and not bad_padding and local[1] == 0.0:
            require(state == before, 'empty block changed accumulated state')
        history.append((end, state))
    if state[1] == 0.0:
        result = ([0.0] * dv, True, -math.inf)
    else:
        result = ([value / state[1] for value in state[2]], False, math.log(base) * state[0] + math.log(state[1]))
    return result, history


def result_equal(actual, expected, label):
    out, empty, lse = actual
    ref_out, ref_empty, ref_lse = expected
    require(empty is ref_empty, label + ': empty flag')
    vector_equal(out, ref_out, label + ': output')
    if ref_empty:
        require(lse == -math.inf, label + ': empty natural LSE must be negative infinity')
        require(out == [0.0] * len(ref_out), label + ': empty output must be zero')
    else:
        require(close(lse, ref_lse), label + ': natural LSE')


def check_prefix_states(history, scores, values, visible, base, dv):
    for end, (top, mass, weighted) in history:
        ref_top, ref_mass, ref_weighted = direct_stats(scores, values, visible, end, dv)
        if ref_mass == 0.0:
            require((top, mass, weighted) == empty_state(dv), 'empty prefix state')
        else:
            require(close(top * math.log(base), ref_top), 'prefix max scale')
            require(close(mass, ref_mass), 'prefix normalization mass')
            vector_equal(weighted, ref_weighted, 'prefix weighted sum')


def check_attention_case(scores, values, visible, dv):
    expected = direct_attention(scores, values, visible, dv)
    for base in (2.0, math.e):
        for block in (1, 2, 4, 7):
            actual, history = block_attention(scores, values, visible, block, base, dv)
            result_equal(actual, expected, 'attention base=%r block=%d' % (base, block))
            check_prefix_states(history, scores, values, visible, base, dv)
            if len(scores) > 0 and not any(visible) and block in (2, 4):
                require(len(history) > 1, 'all-masked test did not traverse multiple blocks')
                require(all(state == empty_state(dv) for _, state in history), 'all-masked state transition')


def test_attention():
    scores = [-3.2, -1.5, 0.25, -0.75, 2.0, -4.0, -0.1, 1.3, -0.6]
    values = [[1.0, -2.0], [0.5, 3.0], [-1.0, 0.0], [2.0, 1.0], [0.25, -0.5], [-2.0, 2.0], [1.5, 0.5], [-0.25, -1.0], [0.75, 1.25]]
    masks = [
        [True] * 9,
        [True, True, False, False, True, True, True, True, True],
        [False, False, True, True, True, True, False, False, False],
        [False] * 9,
    ]
    for visible in masks:
        check_attention_case(scores, values, visible, 2)
    check_attention_case([], [], [], 2)
    extreme = [-700.0, 0.0, 700.0]
    extreme_values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    check_attention_case(extreme, extreme_values, [True] * 3, 2)

    negative_scores = [-2.0, 2.0]
    negative_values = [[1.0, 0.0], [0.0, 1.0]]
    expected = direct_attention(negative_scores, negative_values, [True, True], 2)
    faulty, _ = block_attention(negative_scores, negative_values, [True, True], 1, 2.0, 2, bad_base=True)
    rejected(lambda: result_equal(faulty, expected, 'natural exponent in base-2 merge'), 'base mismatch')

    masked_scores = [-5.0, -3.0, 100.0, -100.0, -4.0]
    masked_values = [[1.0, 2.0], [2.0, 1.0], [99.0, 99.0], [-99.0, -99.0], [3.0, 4.0]]
    visible = [True, True, False, False, True]
    expected = direct_attention(masked_scores, masked_values, visible, 2)
    for base in (2.0, math.e):
        good, history = block_attention(masked_scores, masked_values, visible, 4, base, 2)
        result_equal(good, expected, 'masked finite scores')
        check_prefix_states(history, masked_scores, masked_values, visible, base, 2)
        faulty, _ = block_attention(masked_scores, masked_values, visible, 4, base, 2, bad_padding=True)
        rejected(lambda: result_equal(faulty, expected, 'invalid slots entered normalization'), 'invalid normalization mass')
        rejected(lambda: vector_equal(faulty[0], expected[0], 'invalid slots output'), 'invalid slots must change output')

    ids = [j for j in range(len(masked_scores)) if visible[j]]
    shifted_weights = [math.exp(masked_scores[j] - 0.0) for j in ids]
    shifted_out = [math.fsum(w * masked_values[j][t] for w, j in zip(shifted_weights, ids)) / math.fsum(shifted_weights) for t in range(2)]
    vector_equal(shifted_out, expected[0], 'changing shift alone with valid-only normalization')
    shifted_lse = math.log(math.fsum(shifted_weights))
    require(close(shifted_lse, expected[2]), 'shift-only natural LSE')

    p2 = 2.0 * (2.0 ** 2) / ((2.0 ** 0) + (2.0 ** 2))
    pe = 2.0 * math.exp(2.0) / (math.exp(0.0) + math.exp(2.0))
    require(close(p2, 1.6), 'base-2 mass example')
    require(close(pe, 1.7615941559557646), 'natural-base misuse example')
    print('observed CPU binary64 model: base 2/e outputs, natural LSE and every prefix state match independent whole-prefix references')
    print('observed CPU binary64 model: S=9 all-False rows traverse multiple empty blocks; zero output, empty=True, LSE=-inf checked separately from S=0')
    print('observed CPU binary64 model: empty leading/middle/trailing blocks, nondivisible tails and finite scores [-700,0,700] checked')
    print('observed CPU binary64 model: wrong exponential base and invalid zero-score normalization mass rejected; shift-only valid normalization remains correct on the stated input')


def ceil_div(a, b):
    return (a + b - 1) // b


def resource_model(m, n, k, bm, bn, bk):
    blocks = ceil_div(m, bm) * ceil_div(n, bn)
    stages = ceil_div(k, bk)
    stage_bytes = 2 * (bm * bk + bk * bn)
    return blocks, stages, stage_bytes, 2 * stage_bytes, 4 * bm * bn, bm * bn / 128


def test_derived_accounts():
    a = resource_model(128, 256, 96, 32, 64, 32)
    b = resource_model(128, 256, 96, 64, 128, 32)
    require(a == (16, 3, 6144, 12288, 8192, 16.0), 'resource example A')
    require(b == (4, 3, 12288, 24576, 32768, 64.0), 'resource example B')
    require(a[0] * a[1] * a[2] == 288 * 1024, 'request bytes A')
    require(b[0] * b[1] * b[2] == 144 * 1024, 'request bytes B')
    require(2 * (128 * 96 + 96 * 256) == 72 * 1024, 'unique input bytes')
    offset = 2 ** 31
    wrapped = (offset + 2 ** 31) % (2 ** 32) - 2 ** 31
    require(offset > 2 ** 31 - 1 and wrapped == -offset, 'signed index boundary')
    require(2 * offset == 4 * 1024 ** 3, 'FP16 byte offset')
    cost_us, saving_us = 1200000, 2
    minimum_calls = cost_us // saving_us + 1
    require(minimum_calls == 600001, 'strict amortization threshold')
    require(minimum_calls * saving_us > cost_us, 'amortization inequality')
    require((minimum_calls - 1) * saving_us <= cost_us, 'amortization minimality')
    require(minimum_calls / 32 > 18750, 'decode-step amortization')
    print('observed analytic model: tile capacity/request accounts, signed 2^31 offset and strict amortization boundary checked; no device timing')


def main():
    rejected(lambda: vector_equal([math.nan], [0.0], 'NaN comparator'), 'NaN comparator')
    test_gemm()
    test_attention()
    test_derived_accounts()
    print('PASS: scoped CPU scalar and analytic checks only; GPU execution, synchronization, low precision and performance are untested')


if __name__ == '__main__':
    main()
