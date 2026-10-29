#!/usr/bin/env python3
"""K-FW-012: deterministic offline, read-only arithmetic model.

No file I/O, project imports, subprocesses, network or hardware execution.
Logical-code dense references are independent of tiled compensation paths.
Fraction arithmetic does not model full IEEE floating-point execution.
"""
from fractions import Fraction as F
from itertools import product
from math import gcd

CHECKS = 0
NEGATIVES = []


def check(condition, message):
    global CHECKS
    if not condition:
        raise AssertionError(message)
    CHECKS += 1


def detected(name, wrong, reference):
    check(wrong != reference, 'undetected fault: ' + name)
    check(name not in NEGATIVES, 'duplicate fault name')
    NEGATIVES.append(name)


def rejects(kind, operation, message):
    try:
        operation()
    except kind:
        check(True, message)
    else:
        raise AssertionError('expected rejection: ' + message)


def cd(a, b):
    return (a + b - 1) // b


def pack(codes):
    if any(type(q) is not int or not -8 <= q <= 7 for q in codes):
        raise ValueError('signed INT4 code required')
    result = []
    for j in range(0, len(codes), 2):
        low = codes[j] % 16
        high = codes[j + 1] % 16 if j + 1 < len(codes) else 0
        result.append(low + 16 * high)
    return bytes(result)


def unpack_at(row, j):
    nib = (row[j // 2] >> (4 * (j % 2))) & 15
    return nib if nib < 8 else nib - 16


def validate_rows(rows, n, k):
    if len(rows) != n:
        raise ValueError('packed row count')
    for row in rows:
        if len(row) != cd(k, 2):
            raise ValueError('packed row size')
        if k % 2 and row[-1] // 16:
            raise ValueError('noncanonical unused high nibble')


def validate_table(scales, zeros, rows, groups, domain):
    if len(scales) != rows or len(zeros) != rows:
        raise ValueError('parameter row count')
    for ss, zz in zip(scales, zeros):
        if len(ss) != groups or len(zz) != groups:
            raise ValueError('parameter group count')
        if any(F(s) <= 0 for s in ss):
            raise ValueError('positive rational scales required')
        if any(type(z) is not int or not domain[0] <= z <= domain[1]
               for z in zz):
            raise ValueError('zero point domain')


def validate_shape(n, k, m, positive):
    if any(type(v) is not int or v < 0 for v in (n, k, m)):
        raise ValueError('nonnegative integer shape required')
    if any(type(v) is not int or v <= 0 for v in positive):
        raise ValueError('positive tile and group sizes required')


def w4_reference(codes, x, scales, zeros, n, k, m, g, bias):
    # Reconstruct the entire logical real matrix before dense multiplication.
    w = [[F(scales[r][j // g]) * (codes[r][j] - zeros[r][j // g])
          for j in range(k)] for r in range(n)]
    return [[sum((w[r][j] * F(x[j][c]) for j in range(k)), F(0))
             + F(bias[r]) for c in range(m)] for r in range(n)]


def w4_tile(rows, x, scales, zeros, n, k, m, g, bn, bk, bm, bias, fault=None):
    validate_shape(n, k, m, (g, bn, bk, bm))
    validate_rows(rows, n, k)
    validate_table(scales, zeros, n, cd(k, g), (-8, 7))
    if len(bias) != n or len(x) < k or any(len(row) != m for row in x):
        raise ValueError('activation or bias shape')
    # Faulty path traverses packed physical capacity rather than valid K.
    limit = 2 * cd(k, 2) if fault == 'tail' else k
    if len(x) < limit:
        raise ValueError('physical activation padding missing')
    y = [[F(0) for _ in range(m)] for _ in range(n)]
    updates, epilogues, reads = 0, 0, []
    for n0 in range(0, n, bn):
        n1 = min(n0 + bn, n)
        for m0 in range(0, m, bm):
            m1 = min(m0 + bm, m)
            for k0 in range(0, limit, bk):
                k1 = min(k0 + bk, limit)
                panel = [row[m0:m1] for row in x[k0:k1]]
                for r in range(n0, n1):
                    a = k0
                    while a < k1:
                        group = a // g
                        end = min(k1, (group + 1) * g)
                        qs = [F(0) for _ in range(m1 - m0)]
                        xs = [F(0) for _ in range(m1 - m0)]
                        for j in range(a, end):
                            q = unpack_at(rows[r], j)
                            reads.append((r, m0, j))
                            for c in range(m0, m1):
                                value = F(panel[j - k0][c - m0])
                                qs[c - m0] += q * value
                                xs[c - m0] += value
                        for c in range(m0, m1):
                            scale = scales[r][0 if fault == 'scale' else group]
                            z = 0 if fault == 'zero' else zeros[r][group]
                            y[r][c] += F(scale) * (qs[c - m0] - z * xs[c - m0])
                            updates += 1
                        a = end
    for r in range(n):
        for c in range(m):
            y[r][c] += F(bias[r])
            epilogues += 1
    return y, updates, epilogues, reads


def raw_plan(length, wrange, xrange, initial=0, bits=32, split=False):
    if type(length) is not int or length < 0 or type(bits) is not int or bits < 2:
        raise ValueError('invalid length or width')
    for domain in (wrange, xrange):
        if len(domain) != 2 or any(type(v) is not int for v in domain):
            raise ValueError('integer endpoint domain required')
        if domain[0] > domain[1]:
            raise ValueError('reversed domain')
    lower, upper = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    if type(initial) is not int or not lower <= initial <= upper:
        return 'reject', 0
    ps = [w * x for w in wrange for x in xrange]
    lo, hi = min(ps), max(ps)
    limits = []
    if hi > 0:
        limits.append((upper - initial) // hi)
    if lo < 0:
        limits.append((initial - lower) // (-lo))
    cap = min(limits) if limits else max(length, 1)
    if length <= cap:
        return 'admit', max(length, 1)
    if split and initial == 0 and cap >= 1:
        return 'split', cap
    return 'reject', 0


def wrong_bound(length, wrange, xrange, initial=0, bits=32, split=False):
    return raw_plan(length, (-127, 127), (-127, 127), initial, bits, split)


def no_rejection(length, wrange, xrange, initial=0, bits=32, split=False):
    return 'admit', max(length, 1)


def execute_raw(qw, qx, wrange, xrange, split=False, policy=raw_plan, wrap=False):
    if len(qw) != len(qx):
        raise ValueError('raw shape mismatch')
    mode, cap = policy(len(qw), wrange, xrange, split=split)
    if mode == 'reject':
        raise ValueError('raw dot rejected before reading operands')
    total, lengths = 0, []
    for start in range(0, len(qw), cap):
        end = min(start + cap, len(qw))
        acc = 0
        for j in range(start, end):
            w, x = qw[j], qx[j]
            if not wrange[0] <= w <= wrange[1] or not xrange[0] <= x <= xrange[1]:
                raise ValueError('operand outside real contract')
            acc += w * x
            if wrap:
                acc = (acc + 2**31) % 2**32 - 2**31
            elif not -(2**31) <= acc <= 2**31 - 1:
                raise OverflowError('admitted int32 accumulation overflow')
        total += acc  # Widened merge, never written back to narrow accumulator.
        lengths.append(end - start)
    return total, lengths, mode


def w8_reference(qw, qx, sw, zw, sx, zx, n, k, m, gw, gx, bias):
    # No tile loops, joint spans or zero-point compensation expansion.
    w = [[F(sw[r][j // gw]) * (qw[r][j] - zw[r][j // gw])
          for j in range(k)] for r in range(n)]
    x = [[F(sx[c][j // gx]) * (qx[j][c] - zx[c][j // gx])
          for c in range(m)] for j in range(k)]
    return [[sum((w[r][j] * x[j][c] for j in range(k)), F(0))
             + F(bias[r]) for c in range(m)] for r in range(n)]


def w8_tile(qw, qx, sw, zw, sx, zx, n, k, m, gw, gx,
            bn, bk, bm, bias, fault=None):
    validate_shape(n, k, m, (gw, gx, bn, bk, bm))
    if len(qw) != n or any(len(row) != k for row in qw):
        raise ValueError('weight shape')
    if len(qx) != k or any(len(row) != m for row in qx) or len(bias) != n:
        raise ValueError('activation or bias shape')
    if any(type(q) is not int or not -128 <= q <= 127
           for matrix in (qw, qx) for row in matrix for q in row):
        raise ValueError('signed INT8 contract')
    validate_table(sw, zw, n, cd(k, gw), (-128, 127))
    validate_table(sx, zx, m, cd(k, gx), (-128, 127))
    y = [[F(7 if fault == 'initial' else 0) for _ in range(m)] for _ in range(n)]
    updates, epilogues = 0, 0
    for n0 in range(0, n, bn):
        n1 = min(n0 + bn, n)
        for m0 in range(0, m, bm):
            m1 = min(m0 + bm, m)
            for k0 in range(0, k, bk):
                k1 = min(k0 + bk, k)
                a = k0
                while a < k1:
                    wg, xg = a // gw, a // gx
                    end = min(k1, (wg + 1) * gw, (xg + 1) * gx)
                    if raw_plan(end - a, (-128, 127), (-128, 127))[0] != 'admit':
                        raise ValueError('requires a different arithmetic plan')
                    acc = [[0 for _ in range(m1 - m0)] for _ in range(n1 - n0)]
                    sumw = [0 for _ in range(n1 - n0)]
                    sumx = [0 for _ in range(m1 - m0)]
                    for j in range(a, end):
                        w = [qw[r][j] for r in range(n0, n1)]
                        x = [qx[j][c] for c in range(m0, m1)]
                        for r in range(len(w)):
                            sumw[r] += w[r]
                        for c in range(len(x)):
                            sumx[c] += x[c]
                        for r in range(len(w)):
                            for c in range(len(x)):
                                wp = w[-1] if fault == 'scalar' else w[r]
                                xp = x[-1] if fault == 'scalar' else x[c]
                                acc[r][c] += wp * xp
                    length = end - a
                    for r in range(n0, n1):
                        for c in range(m0, m1):
                            rr, cc = r - n0, c - m0
                            corr = acc[rr][cc]
                            if fault != 'zero':
                                corr -= zw[r][wg] * sumx[cc]
                                corr -= zx[c][xg] * sumw[rr]
                                corr += length * zw[r][wg] * zx[c][xg]
                            # Python integers stand for the proved-wide correction.
                            ws = sw[r][0 if fault == 'scale' else wg]
                            xs = sx[c][0 if fault == 'scale' else xg]
                            y[r][c] += F(ws) * F(xs) * corr
                            updates += 1
                    a = end
    if not (fault == 'empty-epilogue' and k == 0):
        for r in range(n):
            for c in range(m):
                y[r][c] += F(bias[r])
                epilogues += 1
    return y, updates, epilogues


def segment_count(k, steps):
    # Set-of-boundaries reference is independent of kernel's advancement loop.
    if not k:
        return 0
    boundaries = {0, k}
    for step in steps:
        boundaries.update(range(step, k, step))
    return len(boundaries) - 1


def enumerate_intersections(k, g, bk):
    return sum(max(t, a) < min(t + bk, a + g, k)
               for t in range(0, k, bk) for a in range(0, k, g))


def test_format():
    check(pack([]) == b'', 'empty pack')
    known = [-8, 7, -1, 0, 3]
    known_bytes = bytes([0x78, 0x0F, 0x03])
    check(pack(known) == known_bytes, 'known asymmetric bytes')
    check([unpack_at(known_bytes, j) for j in range(5)] == known, 'known decode')
    seen = set()
    for lo, hi in product(range(-8, 8), repeat=2):
        row = pack([lo, hi])
        expected = (lo if lo >= 0 else lo + 16)
        expected += 16 * (hi if hi >= 0 else hi + 16)
        check(row == bytes([expected]), 'arithmetic byte construction')
        check([unpack_at(row, j) for j in range(2)] == [lo, hi], 'all INT4 pairs')
        seen.add(row)
    check(len(seen) == 256, 'unique representation for all pairs')
    row = pack([-1, 7])
    swapped = bytes([16 * (row[0] % 16) + row[0] // 16])
    detected('nibble order', [unpack_at(swapped, j) for j in range(2)], [-1, 7])
    detected('unsigned decode', [row[0] % 16, row[0] // 16], [-1, 7])
    rejects(ValueError, lambda: pack([8]), 'illegal positive code')
    rejects(ValueError, lambda: pack([-9]), 'illegal negative code')
    rejects(ValueError, lambda: validate_rows([bytes([0xF3])], 1, 1), 'bad padding')
    rejects(ValueError, lambda: validate_rows([b''], 1, 1), 'short packed row')


def test_w4():
    for n, k, m in ((2, 0, 3), (2, 1, 3), (3, 5, 4), (2, 7, 2),
                    (0, 5, 2), (2, 5, 0)):
        codes = [[(r * 5 + j * 7) % 16 - 8 for j in range(k)] for r in range(n)]
        rows = [pack(row) for row in codes]
        x = [[F((j * 3 + c * 5) % 13 - 6, 3) for c in range(m)] for j in range(k)]
        bias = [F(r + 1, 7) for r in range(n)]
        for g, bk in product((1, 2, 3, 8), (1, 2, 4, 9)):
            s = [[F(r + h + 1, 4) for h in range(cd(k, g))] for r in range(n)]
            z = [[(r + h) % 5 - 2 for h in range(cd(k, g))] for r in range(n)]
            ref = w4_reference(codes, x, s, z, n, k, m, g, bias)
            for bn, bm in ((1, 1), (2, 3), (5, 7)):
                y, updates, epi, reads = w4_tile(rows, x, s, z, n, k, m, g,
                                                bn, bk, bm, bias)
                check(y == ref, 'W4 packed tile versus decoded dense reference')
                check(updates == n * m * segment_count(k, (g, bk)), 'W4 actual updates')
                check(epi == n * m, 'W4 epilogue once including empty K')
                check(all(j < k for _, _, j in reads), 'only valid positions consumed')
                check(len(reads) == n * cd(m, bm) * k, 'actual repeated decode count')
    codes = [[-8, -1, 0, 7, 3]]
    x = [[F(j)] for j in (1, 2, 3, 4, 5)]
    s, z = [[F(1, 4), F(1, 2)]], [[1, -2]]
    args = ([pack(codes[0])], x, s, z, 1, 5, 1, 3, 1, 2, 1, [F(1, 2)])
    ref = w4_reference(codes, x, s, z, 1, 5, 1, 3, [F(1, 2)])
    check(w4_tile(*args)[0] == ref == [[F(27)]], 'chapter composition trace')
    detected('W4 zero point omitted', w4_tile(*args, fault='zero')[0], ref)
    detected('W4 global scale', w4_tile(*args, fault='scale')[0], ref)
    padcodes = [[3] * 5]
    physical_x = [[F(j)] for j in (1, 2, 3, 4, 5, 1)]
    ps, pz = [[F(1, 2)] * 3], [[3] * 3]
    pargs = ([pack(padcodes[0])], physical_x, ps, pz,
             1, 5, 1, 2, 1, 2, 1, [F(0)])
    good = w4_tile(*pargs)
    bad = w4_tile(*pargs, fault='tail')
    ref = w4_reference(padcodes, physical_x, ps, pz, 1, 5, 1, 2, [F(0)])
    check(good[0] == ref == [[F(0)]], 'packed valid-length actual reduction')
    check([j for _, _, j in good[3]] == list(range(5)), 'valid consumption trace')
    check([j for _, _, j in bad[3]] == list(range(6)), 'fault reads physical tail')
    check(bad[0] == [[F(-3, 2)]], 'tail fault propagates through reducer')
    detected('packed tail consumed', bad[0], ref)
    return good[0], bad[0]


def test_w8():
    args = ([[1], [2]], [[3, 5]], [[1], [1]], [[0], [0]],
            [[1], [1]], [[0], [0]], 2, 1, 2, 1, 1, 2, 1, 2, [0, 0])
    ref = w8_reference(*args[:11], args[-1])
    expected = [[F(3), F(5)], [F(6), F(10)]]
    check(ref == expected, 'independent review counterexample reference')
    check(w8_tile(*args)[0] == expected, 'indexed matrix operands')
    bad = w8_tile(*args, fault='scalar')[0]
    check(bad == [[F(10)] * 2] * 2, 'reproduce last-scalar broadcast')
    detected('W8 stale scalar', bad, expected)
    detected('W8 stale output initialization', w8_tile(*args, fault='initial')[0], expected)
    for n, k, m in ((2, 0, 3), (2, 1, 2), (3, 5, 4), (2, 7, 3),
                    (3, 10, 2), (0, 5, 2), (2, 5, 0)):
        qw = [[(r * 19 + j * 13) % 256 - 128 for j in range(k)] for r in range(n)]
        qx = [[(j * 11 + c * 17) % 256 - 128 for c in range(m)] for j in range(k)]
        bias = [F(r + 1, 3) for r in range(n)]
        for gw, gx in ((1, 1), (2, 3), (3, 2), (4, 6), (8, 1)):
            sw = [[F(r + h + 1, 5) for h in range(cd(k, gw))] for r in range(n)]
            zw = [[(r + h) % 5 - 2 for h in range(cd(k, gw))] for r in range(n)]
            sx = [[F(c + h + 2, 7) for h in range(cd(k, gx))] for c in range(m)]
            zx = [[(c + h) % 7 - 3 for h in range(cd(k, gx))] for c in range(m)]
            ref = w8_reference(qw, qx, sw, zw, sx, zx, n, k, m, gw, gx, bias)
            for bn, bk, bm in ((1, 1, 1), (2, 2, 3), (4, 4, 2), (9, 11, 9)):
                y, updates, epi = w8_tile(qw, qx, sw, zw, sx, zx, n, k, m,
                                          gw, gx, bn, bk, bm, bias)
                check(y == ref, 'W8 matrix tile versus independently decoded product')
                check(updates == n * m * segment_count(k, (bk, gw, gx)), 'W8 joint segments')
                check(epi == n * m, 'W8 exactly one epilogue per output')
    empty = ([[], []], [], [[], []], [[], []], [[], [], []], [[], [], []],
             2, 0, 3, 2, 3, 2, 2, 2, [F(1, 2), F(-3, 4)])
    expected_empty = [[F(1, 2)] * 3, [F(-3, 4)] * 3]
    check(w8_tile(*empty)[0] == expected_empty, 'K=0 produces bias in every column')
    detected('empty K skips epilogue', w8_tile(*empty, fault='empty-epilogue')[0], expected_empty)
    mixed = ([[2, 3, 4], [-1, 5, 7]], [[5, 7], [6, -2], [9, 11]],
             [[F(1, 2), 2], [F(3, 2), F(1, 4)]], [[1, -2], [0, 2]],
             [[1, 2, 3], [F(1, 2), 1, 2]], [[1, -1, 2], [-1, 1, 0]],
             2, 3, 2, 2, 1, 2, 2, 2, [F(1, 3), F(-2, 3)])
    ref = w8_reference(*mixed[:11], mixed[-1])
    check(w8_tile(*mixed)[0] == ref, 'independent source group indices')
    detected('W8 compensation omitted', w8_tile(*mixed, fault='zero')[0], ref)
    detected('W8 global scales', w8_tile(*mixed, fault='scale')[0], ref)
    invalid = list(mixed)
    invalid[11] = 0
    rejects(ValueError, lambda: w8_tile(*invalid), 'zero BN rejected')
    return expected, bad


def test_cost():
    for k, g, bk in product(range(41), range(1, 13), range(1, 13)):
        explicit = enumerate_intersections(k, g, bk)
        common = bk * g // gcd(bk, g)
        formula = 0 if not k else 1 + (k - 1) // bk + (k - 1) // g - (k - 1) // common
        check(explicit == formula, 'intersection enumeration versus boundary-union formula')
    check(enumerate_intersections(128, 128, 1) == 128, 'review J counterexample')
    detected('logical groups substituted for J', cd(128, 128), enumerate_intersections(128, 128, 1))
    check(3 * cd(3, 2) == 6 and cd(3 * 3, 2) == 5, 'row versus continuous capacity')
    code = 4096 * cd(4096, 2)
    meta = 4096 * cd(4096, 128) * 4
    check((code, meta, code + meta) == (8388608, 524288, 8912896), 'weight byte account')
    check(F(8 * (code + meta), 4096**2) == F(17, 4), 'effective bit width')
    check(F(2) / (F(4, 8) + F(4, 128)) == F(64, 17), 'M=1 intensity')
    check(F(64) / (F(4, 8) + F(4, 128)) == F(2048, 17), 'M=32 intensity')
    check(131071 * 65536 == 8589869056 < 2**63, 'wide correction bound')


class ConstantCodes:
    def __init__(self, length, value):
        self.length, self.value, self.reads = length, value, 0

    def __len__(self):
        return self.length

    def __getitem__(self, j):
        if not 0 <= j < self.length:
            raise IndexError(j)
        self.reads += 1
        return self.value


def test_width():
    # Independent exhaustive reachable-prefix-sum oracle for a 4-bit accumulator.
    # It uses all integer products, not endpoint bounds or quotient formulas.
    for wr, xr in (((-2, 1), (-1, 2)), ((0, 2), (1, 2)),
                   ((-2, -1), (1, 2)), ((0, 0), (-2, 2))):
        ps = {w * x for w in range(wr[0], wr[1] + 1) for x in range(xr[0], xr[1] + 1)}
        for initial in (-2, 0, 2):
            reachable, unsafe = {initial}, False
            for length in range(9):
                if length:
                    reachable = {s + p for s in reachable for p in ps}
                unsafe = unsafe or any(s < -8 or s > 7 for s in reachable)
                check((raw_plan(length, wr, xr, initial, bits=4)[0] == 'admit') == (not unsafe),
                      'planner versus exhaustive prefix sums')
    domain = (-128, 127)
    decisions = {k: raw_plan(k, domain, domain)[0] for k in (0, 4096, 131071, 131072)}
    check(decisions == {0: 'admit', 4096: 'admit', 131071: 'admit', 131072: 'reject'},
          'actual boundary decisions')
    check(raw_plan(131072, (-127, 127), (-127, 127))[0] == 'admit', 'restricted real code domain')
    check(raw_plan(131071, domain, domain, initial=16384)[0] == 'reject', 'initial value accounted')
    check(raw_plan(0, domain, domain, initial=2**31)[0] == 'reject', 'invalid initial even for empty K')
    probes = ConstantCodes(131072, -128), ConstantCodes(131072, -128)
    rejects(ValueError, lambda: execute_raw(*probes, domain, domain), 'dangerous reduction rejected')
    check(probes[0].reads == probes[1].reads == 0, 'rejection precedes operand consumption')
    w, x = [-128] * 131071, [-128] * 131071
    reference = sum(a * b for a, b in zip(w, x))
    check(execute_raw(w, x, domain, domain)[0] == reference == 2147467264, 'last admitted dot executes')
    w, x = [-128] * 131072, [-128] * 131072
    reference = sum(a * b for a, b in zip(w, x))
    total, lengths, mode = execute_raw(w, x, domain, domain, split=True)
    check(mode == 'split' and lengths == [131071, 1], 'safe split actually executed')
    check(total == reference == 2**31, 'widened merge preserves result')
    detected('127 bound admits forbidden extreme', wrong_bound(131072, domain, domain)[0], 'reject')
    rejects(OverflowError, lambda: execute_raw(w, x, domain, domain, policy=wrong_bound),
            'wrong bound reaches overflow in admitted execution')
    detected('removed rejection decision', no_rejection(131072, domain, domain)[0], 'reject')
    wrong = execute_raw(w, x, domain, domain, policy=no_rejection, wrap=True)[0]
    check(wrong == -(2**31), 'faulty admitted int32 path really wraps')
    detected('removed rejection corrupts executed dot', wrong, reference)
    rejects(ValueError, lambda: raw_plan(1, (2, 1), domain), 'reversed domain')
    return decisions, lengths, total


def main():
    test_format()
    padding = test_w4()
    matrix = test_w8()
    test_cost()
    decisions, lengths, total = test_width()
    print('scope: exact rational/integer software model; no CPU/GPU kernel or timing')
    print('matrix reference: %s; stale-scalar fault: %s' % matrix)
    print('packed valid-length result: %s; tail-consumption fault: %s' % padding)
    print('K=128 G=128 BK=1: 128 actual compensation/update segments')
    print('raw int32 admission decisions: %s' % decisions)
    print('executed split lengths: %s; widened total: %s' % (lengths, total))
    print('checks passed: %d' % CHECKS)
    print('negative controls detected: %d' % len(NEGATIVES))
    for name in NEGATIVES:
        print('detected: ' + name)
    print('limits: rational scales, exact arithmetic, linear bias epilogue; no complete IEEE rounding, ISA, dispatch, concurrency or quality proof')


if __name__ == '__main__':
    main()
