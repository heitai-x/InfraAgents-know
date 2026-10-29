#!/usr/bin/env python3
"""K-HW-010: deterministic, read-only, offline stdlib software checks.
No project imports, filesystem access, subprocesses, credentials or network.
"""
from fractions import Fraction


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def nonnegative(value):
    if type(value) is not int or value < 0:
        raise ValueError('expected nonnegative integer')


def positive(value):
    if type(value) is not int or value <= 0:
        raise ValueError('expected positive integer')


def ceildiv(a, b):
    return (a + b - 1) // b


class View:
    def __init__(self, matrix, rows, cols, transposed=False):
        nonnegative(rows)
        nonnegative(cols)
        if len(matrix) != rows or any(len(row) != cols for row in matrix):
            raise ValueError('matrix shape mismatch')
        self.rows, self.cols = rows, cols
        self.rs = 2 if transposed else 2 * (cols + 1) + 1
        self.cs = 2 * (rows + 1) + 1 if transposed else 2
        self.offset = 3
        size = self.offset + max(0, rows - 1) * self.rs + max(0, cols - 1) * self.cs + 5
        self.data = [99991] * size
        self.reads = 0
        for i in range(rows):
            for j in range(cols):
                self.data[self.offset + i * self.rs + j * self.cs] = matrix[i][j]

    def get(self, i, j):
        if not (0 <= i < self.rows and 0 <= j < self.cols):
            raise AssertionError('out-of-domain source load')
        self.reads += 1
        return self.data[self.offset + i * self.rs + j * self.cs]


def finish(value, bias, old, beta, relu):
    result = value + bias + (old if beta else 0)
    return max(0, result) if relu else result


def dense_reference(M, N, K, X, W, bias, old, beta, relu):
    # Uses dense logical inputs, never the packed buffers or View.get.
    result = [[0] * N for _ in range(M)]
    for m in range(M):
        for n in range(N):
            products = [X[m][k] * W[k][n] for k in range(K)]
            result[m][n] = finish(sum(products), bias[n], old[m][n] if beta else 0, beta, relu)
    return result


def packed_gemm(M, N, K, X, W, tile, bias, old, beta, relu, padding=0):
    for value in (M, N, K):
        nonnegative(value)
    for value in tile:
        positive(value)
    if (X.rows, X.cols, W.rows, W.cols) != (M, K, K, N):
        raise ValueError('view shape mismatch')
    if beta not in (0, 1):
        raise ValueError('unsupported beta')
    mc, nc, kc, mr, nr = tile
    cm, cn = ceildiv(mc, mr) * mr, ceildiv(nc, nr) * nr
    result = [[0] * N for _ in range(M)]
    owners = [[0] * N for _ in range(M)]
    products = 0
    for n0 in range(0, N, nc):
        for m0 in range(0, M, mc):
            acc = [[0] * cn for _ in range(cm)]
            for k0 in range(0, K, kc):
                xp = [padding] * (cm * kc)
                wp = [padding] * (kc * cn)
                for i in range(mc):
                    for q in range(kc):
                        if m0 + i < M and k0 + q < K:
                            xp[((i // mr) * kc + q) * mr + i % mr] = X.get(m0 + i, k0 + q)
                for j in range(nc):
                    for q in range(kc):
                        if n0 + j < N and k0 + q < K:
                            wp[((j // nr) * kc + q) * nr + j % nr] = W.get(k0 + q, n0 + j)
                for mi in range(0, mc, mr):
                    for nj in range(0, nc, nr):
                        registers = [[acc[mi + r][nj + c] for c in range(nr)] for r in range(mr)]
                        for q in range(kc):
                            for r in range(mr):
                                xv = xp[((mi // mr) * kc + q) * mr + r]
                                for c in range(nr):
                                    wv = wp[((nj // nr) * kc + q) * nr + c]
                                    registers[r][c] += xv * wv
                                    products += 1
                        for r in range(mr):
                            for c in range(nr):
                                acc[mi + r][nj + c] = registers[r][c]
            for i in range(min(mc, M - m0)):
                for j in range(min(nc, N - n0)):
                    m, n = m0 + i, n0 + j
                    result[m][n] = finish(acc[i][j], bias[n], old[m][n] if beta else 0, beta, relu)
                    owners[m][n] += 1
    return result, owners, products


def partition(length, count):
    nonnegative(length)
    positive(count)
    return [(p * length // count, (p + 1) * length // count) for p in range(count)]


def covered_once(length, intervals):
    # Independent invariant: actual emitted coordinates, without clipping bad ranges.
    counts = [0] * length
    for a, b in intervals:
        if not (0 <= a <= b <= length):
            return False
        for k in range(a, b):
            counts[k] += 1
    return counts == [1] * length


def split_run(x, w, count, emitted=None):
    positive(count)
    if len(x) != len(w):
        raise ValueError('dot length mismatch')
    K = len(x)
    visits, tickets, partials = [0] * K, [], []
    # emitted injects faulty producer behavior; the external invariant rejects it.
    ids = list(range(count)) if emitted is None else list(emitted)
    for p in ids:
        if type(p) is not int or not 0 <= p < count:
            raise ValueError('invalid segment identity')
        a, b = p * K // count, (p + 1) * K // count
        value = 0
        for k in range(a, b):
            value += x[k] * w[k]
            visits[k] += 1
        tickets.append(p)
        partials.append(value)
    return {'value': sum(partials), 'visits': visits, 'tickets': tickets, 'partials': partials}


def split_complete(record, K, count):
    return record['visits'] == [1] * K and sorted(record['tickets']) == list(range(count))


def lane_dot(x, w, width):
    positive(width)
    if len(x) != len(w):
        raise ValueError('dot length mismatch')
    lanes = [0] * width
    for a in range(0, len(x), width):
        for lane in range(width):
            k = a + lane
            if k < len(x):
                lanes[lane] += x[k] * w[k]
    return lanes


def pages_contiguous(size, page, start=0):
    nonnegative(size)
    nonnegative(start)
    positive(page)
    if size == 0:
        return 0
    return (start + size - 1) // page - start // page + 1


def expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('expected ValueError was not raised')


def main():
    print('scope: exact integer/Fraction CPU software model; no ISA, hardware timing, PMU or GPU evidence')
    gemm_cases = 0
    tiles = [(2, 3, 2, 1, 2), (3, 2, 4, 2, 3), (1, 1, 1, 2, 2)]
    for M in range(4):
        for N in range(5):
            for K in range(6):
                X = [[(3 * m + 2 * k) % 7 - 3 for k in range(K)] for m in range(M)]
                W = [[(2 * k + 3 * n) % 9 - 4 for n in range(N)] for k in range(K)]
                bias = [n - 2 for n in range(N)]
                old = [[m - n + 1 for n in range(N)] for m in range(M)]
                beta, relu = (M + N + K) % 2, bool((M + K) % 2)
                want = dense_reference(M, N, K, X, W, bias, old, beta, relu)
                for tile in tiles:
                    mc, nc, kc, mr, nr = tile
                    for transposed in (False, True):
                        xv = View(X, M, K, transposed)
                        wv = View(W, K, N, not transposed)
                        got, owners, work = packed_gemm(M, N, K, xv, wv, tile, bias, old, beta, relu)
                        require(got == want, 'packed GEMM or epilogue mismatch')
                        require(all(value == 1 for row in owners for value in row), 'output ownership mismatch')
                        require(xv.reads == ceildiv(N, nc) * M * K, 'X source-read formula mismatch')
                        require(wv.reads == ceildiv(M, mc) * K * N, 'W source-read formula mismatch')
                        expected_work = (ceildiv(M, mc) * ceildiv(N, nc) * ceildiv(K, kc)
                                         * ceildiv(mc, mr) * mr * ceildiv(nc, nr) * nr * kc)
                        require(work == expected_work, 'padded execution-work formula mismatch')
                        gemm_cases += 1
    print('observation: packed strided GEMM, ownership, source reads and padded products matched independent references on', gemm_cases, 'cases')

    X, W = [[2, 3, 4]], [[1], [2], [3]]
    want = dense_reference(1, 1, 3, X, W, [0], [[0]], 0, False)
    faulty, _, _ = packed_gemm(1, 1, 3, View(X, 1, 3), View(W, 3, 1, True),
                               (2, 2, 2, 1, 2), [0], [[0]], 0, False, padding=1)
    require(want == [[20]] and faulty == [[21]], 'nonzero K-padding negative control missed')
    print('negative control detected: nonzero K padding changed 20 to 21')

    for N in range(9):
        for T in range(1, 12):
            require(covered_once(N, partition(N, T)), 'column coverage including empty threads failed')
    require(not covered_once(10, [(0, 4), (4, 8), (7, 10)]), 'overlap negative control missed')
    require(not covered_once(10, [(0, 3), (4, 10)]), 'missing-column negative control missed')
    require(not covered_once(10, [(-1, 10)]), 'out-of-domain partition accepted')
    print('observation: column ownership includes empty threads; overlap, holes and invalid bounds rejected')

    x = [1, -2, 3, 4, -5, 6, 7]
    w = [2, 1, -1, 3, 2, -2, 1]
    reference = sum(a * b for a, b in zip(x, w))
    correct = split_run(x, w, 3)
    omitted = split_run(x, w, 3, [1, 2])
    require(reference == -6 and correct['partials'] == [0, 9, -15], 'original regression identity changed')
    require(split_complete(correct, 7, 3), 'complete original split rejected')
    require(omitted['value'] == reference, 'zero-contribution cancellation counterexample lost')
    require(not split_complete(omitted, 7, 3), 'zero-contribution omitted segment missed')
    require(omitted['visits'][:2] == [0, 0], 'missing coordinates not observed')
    nonzero = split_run(list(range(1, 8)), [1] * 7, 3, [1, 2])
    require(nonzero['value'] == 25 and nonzero['value'] != sum(range(1, 8)), 'nonzero omitted-segment witness missed')
    print('regression: parts [0,9,-15] total -6; omitting first still gives -6 but misses k=0,1; nonzero witness changes 28 to 25')

    split_cases = 0
    for K in range(9):
        for P in range(1, 12):
            x, w = list(range(1, K + 1)), [1] * K
            record = split_run(x, w, P)
            require(record['value'] == sum(a * b for a, b in zip(x, w)), 'split dot mismatch')
            require(split_complete(record, K, P), 'complete split/tickets rejected')
            for k in range(K):
                basis = [int(j == k) for j in range(K)]
                require(split_run(basis, w, P)['value'] == 1, 'basis-vector split coverage failed')
            ids = list(range(P))
            for p in range(P):
                dropped = split_run(x, w, P, [i for i in ids if i != p])
                duplicated = split_run(x, w, P, ids + [p])
                require(not split_complete(dropped, K, P), 'missing segment/ticket missed')
                require(not split_complete(duplicated, K, P), 'duplicate segment/ticket missed')
            split_cases += 1
    empty_ticket = split_run([1, 2], [3, 4], 5, [1, 2, 3, 4])
    require(empty_ticket['value'] == 11 and empty_ticket['visits'] == [1, 1], 'empty-segment witness invalid')
    require(not split_complete(empty_ticket, 2, 5), 'missing empty completion ticket missed')
    print('observation:', split_cases, 'split shapes matched direct dot and basis coverage; missing/duplicate segments including empty tickets rejected')

    lane_cases = 0
    for K in range(9):
        x, w = [k - 3 for k in range(K)], [2 - k for k in range(K)]
        for width in (1, 2, 4, 8):
            lanes = lane_dot(x, w, width)
            require(sum(lanes) == sum(a * b for a, b in zip(x, w)), 'lane dot or tail mismatch')
            lane_cases += 1
    lanes = lane_dot([1, 2, 3], [1, 1, 1], 2)
    require(lanes == [4, 2] and lanes[0] != 6, 'omitted horizontal merge negative control missed')
    expected_epilogue = max(0, sum([2, -3]) + 1)
    early_epilogue = sum(max(0, part + 1) for part in [2, -3])
    require(expected_epilogue == 0 and early_epilogue == 3, 'early epilogue negative control missed')
    print('observation:', lane_cases, 'lane dots matched direct sums; omitted horizontal merge and early epilogue rejected')

    page_cases = 0
    for start in range(19):
        for size in range(21):
            independent = len({address // 8 for address in range(start, start + size)})
            require(pages_contiguous(size, 8, start) == independent, 'continuous-page count mismatch')
            page_cases += 1
    require(pages_contiguous(4096, 4096, 1) == 2, 'offset boundary wrong')
    faulty_floor = (1 + 4096) // 4096
    require(faulty_floor != pages_contiguous(4096, 4096, 1), 'floor-page negative control missed')
    strided = {address // 4096
               for i in range(3) for j in range(2)
               for address in range(4 * (1024 * i + j), 4 * (1024 * i + j) + 4)}
    require(len(strided) == 3 and pages_contiguous(24, 4096) == 1, 'strided versus packed page witness failed')
    A = pages_contiguous(240 * 256 * 8, 4096)
    B = pages_contiguous(256 * 4 * 8, 4096)
    C = pages_contiguous(240 * 4 * 8, 4096)
    require((A, B, C) == (120, 2, 2), 'Goto panel pages wrong')
    complete, naive = A + 2 * (B + C), A + B + C
    require(complete == 128 and naive == 124 and naive <= 125 < complete, 'TLB coexistence negative control missed')
    print('observation:', page_cases, 'page ranges matched byte enumeration; stride and TLB factor omission distinguished (128 versus 124 at limit 125)')

    F1 = 2 * 4096 * 4096
    D1 = 4 * (4096 + 4096 * 4096 + 4096)
    F32 = 2 * 32 * 4096 * 4096
    D32 = 4 * (32 * 4096 + 4096 * 4096 + 32 * 4096)
    require((F1, D1) == (33554432, 67141632), 'M=1 arithmetic mismatch')
    require((F32, D32) == (1073741824, 68157440), 'M=32 arithmetic mismatch')
    require(Fraction(F1, D1) == Fraction(1024, 2049), 'M=1 intensity wrong')
    require(Fraction(F32, D32) == Fraction(1024, 65), 'M=32 intensity wrong')
    weight_only = Fraction(F32, 4 * 4096 * 4096)
    require(weight_only == 16 and weight_only != Fraction(F32, D32), 'weight-only negative control missed')
    require(Fraction(64 * 2**20, 80 * 2**30) * 1000 == Fraction(25, 32), 'bandwidth lower-bound units wrong')
    require(not (600000 * 2 > 1200000), 'amortization equality accepted')
    require(600001 * 2 > 1200000, 'amortization above boundary rejected')
    require((600000 * 2 >= 1200000) != (600000 * 2 > 1200000), 'nonstrict amortization negative control missed')
    print('observation: intensities 1024/2049 and 1024/65; weight-only bytes and nonstrict amortization rejected')

    expect_value_error(lambda: partition(3, 0))
    expect_value_error(lambda: split_run([1], [1], 0))
    expect_value_error(lambda: split_run([1], [], 1))
    expect_value_error(lambda: lane_dot([], [], 0))
    expect_value_error(lambda: pages_contiguous(-1, 4096))
    expect_value_error(lambda: pages_contiguous(1, 0))
    expect_value_error(lambda: packed_gemm(1, 1, 1, View([[1]], 1, 1), View([[2]], 1, 1),
                                         (1, 1, 0, 1, 1), [0], [[0]], 0, False))
    print('observation: invalid counts, lengths, pages and tiles raise ValueError')
    print('completed: software-model references and intentional negative controls; target hardware remains unmeasured')


if __name__ == '__main__':
    main()
