import math

EPS = 1e-9


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_detection(check, name):
    try:
        check()
    except AssertionError:
        return
    raise AssertionError('negative control escaped: ' + name)


def close(A, B, tol=EPS):
    return (len(A) == len(B) and all(
        len(a) == len(b) and all(abs(x-y) <= tol for x, y in zip(a, b))
        for a, b in zip(A, B)))


def half_up(x):
    return math.floor(x + 0.5)


def encode(x, parameter, lo, hi):
    s, z = parameter
    return max(lo, min(hi, half_up(x / s) + z))


def freeze(W, n, G, lo, hi, supplied=None):
    require(G > 0 and lo < hi and hi > 0, 'invalid grouping/code domain')
    require(all(len(row) == n for row in W), 'weight shape')
    require(all(math.isfinite(v) for row in W for v in row), 'nonfinite weight')
    groups = (n + G - 1) // G
    if supplied is None:
        require(lo < 0, 'symmetric builder needs signed domain')
        supplied = []
        for row in W:
            params = []
            for start in range(0, n, G):
                a = max(abs(v) for v in row[start:start+G])
                params.append((a / hi if a else 1.0, 0))
            supplied.append(params)
    require(len(supplied) == len(W), 'parameter row count')
    require(all(len(row) == groups for row in supplied), 'parameter group count')
    for row in supplied:
        for s, z in row:
            require(math.isfinite(s) and s > 0, 'invalid scale')
            require(isinstance(z, int) and lo <= z <= hi, 'invalid zero point')
    return tuple(tuple((float(s), z) for s, z in row) for row in supplied)


def independent_decode(Q, parameters, G, n):
    # Enumerate group intervals instead of using the encoder's column//G lookup.
    require(len(Q) == len(parameters), 'decode rows')
    output = []
    for codes, row_params in zip(Q, parameters):
        require(len(codes) == n, 'decode columns')
        row = []
        for g, (s, z) in enumerate(row_params):
            for j in range(g * G, min((g + 1) * G, n)):
                row.append(s * (codes[j] - z))
        require(len(row) == n, 'decode coverage')
        output.append(row)
    return output


def transpose(A):
    return [list(c) for c in zip(*A)]


def multiply(A, B):
    return [[sum(a * b for a, b in zip(row, col))
             for col in zip(*B)] for row in A]


def chol(A):
    n = len(A)
    require(all(len(row) == n for row in A), 'square matrix required')
    require(all(math.isfinite(v) for row in A for v in row), 'nonfinite matrix')
    require(all(abs(A[i][j] - A[j][i]) < EPS
                for i in range(n) for j in range(n)), 'nonsymmetric matrix')
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            v = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                require(v > 0 and math.isfinite(v), 'not positive definite')
                L[i][j] = math.sqrt(v)
            else:
                L[i][j] = v / L[j][j]
    return L


def chol_solve(L, b):
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][k]*y[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[k][i]*x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def inverse_factor(H):
    L = chol(H)
    n = len(H)
    columns = [chol_solve(L, [float(i == j) for i in range(n)])
               for j in range(n)]
    Hinv = transpose(columns)
    U = transpose(chol(Hinv))
    require(close(multiply(transpose(U), U), Hinv), 'inverse-factor invariant')
    return U


def solve_reference(A, b):
    # Independent pivoted Gauss-Jordan solve; no Cholesky or inverse-factor rows.
    n = len(A)
    M = [list(row) + [rhs] for row, rhs in zip(A, b)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        require(abs(M[p][c]) > 1e-14, 'singular reference system')
        M[c], M[p] = M[p], M[c]
        pivot = M[c][c]
        M[c] = [v / pivot for v in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                M[r] = [a - f*b for a, b in zip(M[r], M[c])]
    return [row[-1] for row in M]


def blocked(W, H, parameters, G, B, lo, hi, wrong_factor=False):
    n = len(H)
    require(B > 0, 'invalid block size')
    parameters = freeze(W, n, G, lo, hi, parameters)
    Q = [[0] * n for _ in W]
    used_values = [[0.0] * n for _ in W]
    if not n or not W:
        return Q, parameters, used_values
    U = transpose(chol(H)) if wrong_factor else inverse_factor(H)
    work = [row[:] for row in W]
    writes = [[0] * n for _ in W]
    for start in range(0, n, B):
        end = min(start + B, n)
        width = end - start
        E = [[0.0] * width for _ in W]
        for j in range(start, end):
            for r in range(len(W)):
                s, z = parameters[r][j // G]
                q = encode(work[r][j], (s, z), lo, hi)
                h = s * (q - z)
                Q[r][j] = q
                used_values[r][j] = h
                writes[r][j] += 1
                error = (work[r][j] - h) / U[j][j]
                E[r][j - start] = error
                for k in range(j, end):
                    work[r][k] -= error * U[j][k]
        for r in range(len(W)):
            for k in range(end, n):
                work[r][k] -= sum(E[r][t] * U[start+t][k]
                                  for t in range(width))
    require(all(v == 1 for row in writes for v in row), 'code write coverage')
    return Q, parameters, used_values


def constrained_reference(W, H, parameters, G, lo, hi):
    # Recompute the continuous minimizer from original W and fixed prefix each time.
    # Quantization uses exhaustive nearest-grid search, with ties toward larger code.
    n = len(H)
    Q = [[0] * n for _ in W]
    recon = [[0.0] * n for _ in W]
    for r, original in enumerate(W):
        grid_by_column = []
        for g, (s, z) in enumerate(parameters[r]):
            grid = [(q, s*(q-z)) for q in range(lo, hi + 1)]
            grid_by_column.extend([grid] * min(G, n - g*G))
        for j in range(n):
            F = list(range(j, n))
            A = [[H[a][b] for b in F] for a in F]
            rhs = [sum(H[a][k] * (original[k] - recon[r][k])
                       for k in range(j)) for a in F]
            delta = solve_reference(A, rhs)
            current = original[j] + delta[0]
            q, h = min(grid_by_column[j], key=lambda item: (abs(item[1]-current), -item[0]))
            Q[r][j], recon[r][j] = q, h
    return Q, recon


def objective(W, reconstructed, H):
    result = 0.0
    for a, b in zip(W, reconstructed):
        e = [x-y for x, y in zip(a, b)]
        result += sum(e[i]*H[i][j]*e[j]
                      for i in range(len(e)) for j in range(len(e)))
    return result


def test_frozen_group_regression():
    W = [[1.0, 4.0]]
    H = [[1.0, 0.0], [0.0, 1.0]]
    P = freeze(W, 2, 2, -2, 1)
    require(P == (((4.0, 0),),), 'full-group preparation failed')
    for B in (1, 2, 5):
        Q, exported, used = blocked(W, H, P, 2, B, -2, 1)
        decoded = independent_decode(Q, exported, 2, 2)
        require(Q == [[0, 1]] and decoded == [[0.0, 4.0]], 'INT2 frozen regression')
        require(close(decoded, used) and exported == P, 'artifact/compensation mismatch')
    # Deliberately faulty lifecycle: refit from one column and overwrite one group entry.
    bad_codes, bad_used = [], []
    mutable_group = [1.0, 0]
    for value in W[0]:
        mutable_group[0] = abs(value) or 1.0
        q = encode(value, mutable_group, -2, 1)
        bad_codes.append(q)
        bad_used.append(mutable_group[0] * (q - mutable_group[1]))
    bad_final = independent_decode([bad_codes], ((tuple(mutable_group),),), 2, 2)
    require(bad_codes == [1, 1] and bad_used == [1.0, 4.0], 'fault not exercised')
    require(bad_final == [[4.0, 4.0]], 'fault counterexample changed')
    expect_detection(lambda: require(close(bad_final, [bad_used]), 'overwritten metadata'),
                     'old code reinterpreted by last scale')
    print('frozen-group: INT2 G=2 s=4 codes=[0,1], decode=[0,4]; overwrite [1,1] -> [4,4] detected')


def test_grouped_constrained_reference():
    H = [[2.0, .4, .05, .1, 0.0],
         [.4, 1.0, -.1, 0.0, .05],
         [.05, -.1, .8, .1, .02],
         [.1, 0.0, .1, 1.5, .2],
         [0.0, .05, .02, .2, 1.2]]
    W = [[1.0, 4.0, -2.0, .55, 3.1], [-.3, .55, .2, -4.2, .31]]
    cases = 0
    for G in (1, 3, 7):
        P = freeze(W, 5, G, -8, 7)
        require(any(abs(s - 1.0) > EPS for row in P for s, z in row), 'nonunit scales absent')
        ref_Q, ref_values = constrained_reference(W, H, P, G, -8, 7)
        for B in (1, 2, 3, 8):
            Q, exported, used = blocked(W, H, P, G, B, -8, 7)
            decoded = independent_decode(Q, exported, G, 5)
            require(Q == ref_Q, 'grouped block codes differ from constrained reference')
            require(close(decoded, ref_values) and close(decoded, used), 'independent decode mismatch')
            require(exported == P, 'frozen table changed')
            cases += 1
    # Explicit nonzero zero-points, two rows, G=3 crossing B=2, short final group.
    custom = (((.5, 1), (2.0, -1)), ((.25, -2), (1.5, 2)))
    P = freeze(W, 5, 3, -8, 7, custom)
    ref_Q, ref_values = constrained_reference(W, H, P, 3, -8, 7)
    Q, exported, used = blocked(W, H, P, 3, 2, -8, 7)
    require(Q == ref_Q, 'nonzero-zero-point reference mismatch')
    decoded = independent_decode(Q, exported, 3, 5)
    require(close(decoded, ref_values) and close(decoded, used), 'nonzero zero decode mismatch')
    print('grouped-GPTQ: %d block/group combinations plus nonzero-zero-point case; independent constrained reference agrees' % cases)


def test_factor_negative():
    H2 = [[2.0, .4], [.4, 1.0]]
    U = inverse_factor(H2)
    wrong = transpose(chol(H2))
    require(abs(U[0][1]/U[0][0] + .4) < EPS, 'inverse-factor row ratio')
    require(abs(wrong[0][1]/wrong[0][0] - .2) < EPS, 'wrong-factor row ratio')
    H = [[2.0, .4, .05], [.4, 1.0, -.1], [.05, -.1, .8]]
    W = [[-.3, .55, .2]]
    P = (((1.0, 0),),)
    Q, exported, used = blocked(W, H, P, 3, 2, -8, 7)
    ref_Q, ref_values = constrained_reference(W, H, P, 3, -8, 7)
    bad_Q, bad_P, bad_used = blocked(W, H, P, 3, 2, -8, 7, wrong_factor=True)
    require(Q == ref_Q == [[0, 0, 0]], 'correct short-block result')
    require(bad_Q == [[0, 1, 0]], 'wrong-factor path not exercised')
    bad_values = independent_decode(bad_Q, bad_P, 3, 3)
    expect_detection(lambda: require(bad_Q == ref_Q, 'wrong decomposition object'), 'factor')
    correct_cost = objective(W, ref_values, H)
    wrong_cost = objective(W, bad_values, H)
    require(abs(correct_cost - .3545) < EPS and abs(wrong_cost - .5345) < EPS, 'controlled objective')
    require(wrong_cost > correct_cost, 'wrong factor not separated')
    print('factor: actual short-block wrong-factor codes detected; eHe %.4f vs %.4f' % (correct_cost, wrong_cost))


def masked_dot(codes, weights, valid, s, b):
    require(len(codes) == len(weights) == len(valid), 'mask shape')
    total = 0.0
    for q, w, enabled in zip(codes, weights, valid):
        if enabled:
            require(0 <= q <= 15, 'invalid active code')
            total += (b + s*q) * w
    return total


def test_rounding_range_and_padding():
    require([half_up(x) for x in (.5, -.5, 1.5, -1.5)] == [1, 0, 2, -1], 'tie rule')
    require(encode(.5, (1.0, 1), 0, 3) == 2, 'canonical zero-point placement')
    require(round(.5) + 1 == 1 and round(1.5) == 2, 'nearest-even negative control')
    for i in range(-80, 71):
        x = i / 20.0
        q = encode(x, (.5, 0), -8, 7)
        require(abs(.5*q - x) <= .25 + EPS, 'rounding error bound')
    require(abs(.5*encode(10.0, (.5, 0), -8, 7) - 10.0) > .25, 'saturation absent')
    s, z = .5, 3
    base = sum(s*(q-z)*w for q, w in zip([1, 5], [2, -3]))
    padded = base + s*(z-z)*7
    require(base == padded, 'exact-zero pad changes reduction')
    expect_detection(lambda: require(base + s*(0-z)*7 == base, 'byte-zero padding'), 'zero point')
    step, b = 8.0/15.0, -2.0
    nearest = half_up(-b / step)
    contribution = (b + step*nearest)*15
    require(nearest == 4 and abs(contribution - 2.0) < EPS, 'real-grid counterexample')
    codes, weights = [1, nearest, 5], [2.0, 15.0, -3.0]
    expected = (b + step)*2.0 + (b + step*5)*-3.0
    masked = masked_dot(codes, weights, [True, False, True], step, b)
    unmasked = masked_dot(codes, weights, [True, True, True], step, b)
    require(abs(masked - expected) < EPS, 'actual masked loop failed')
    expect_detection(lambda: require(abs(unmasked-expected) < EPS, 'near-zero consumed'), 'REAL padding')
    require(abs(masked_dot([1, 999, 5], weights, [True, False, True], step, b)-expected) < EPS,
            'invalid masked position was consumed')
    print('numeric: half-integer placement, saturation, exact ZERO padding and actual REAL mask traversal checked')


def test_edges():
    zero_W = [[0.0, 0.0, 0.0]]
    P = freeze(zero_W, 3, 2, -8, 7)
    require(P == (((1.0, 0), (1.0, 0)),), 'all-zero/short group rule')
    H = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    Q, exported, used = blocked(zero_W, H, P, 2, 2, -8, 7)
    require(independent_decode(Q, exported, 2, 3) == zero_W, 'zero group reconstruction')
    P = freeze([[]], 0, 2, -8, 7)
    require(blocked([[]], [], P, 2, 3, -8, 7)[0] == [[]], 'empty input dimension')
    require(blocked([], [[1.0]], (), 2, 3, -8, 7)[0] == [], 'empty output dimension')
    expect_detection(lambda: freeze([[1.0]], 1, 1, -8, 7, [[(0.0, 0)]]), 'zero scale')
    expect_detection(lambda: freeze([[1.0]], 1, 1, -8, 7, [[(1.0, 8)]]), 'illegal zero point')
    expect_detection(lambda: freeze([[float('nan')]], 1, 1, -8, 7), 'nonfinite input')
    expect_detection(lambda: chol([[0.0]]), 'nonpositive Hessian')
    print('edges: zero groups, short groups, empty dimensions and invalid inputs checked')


def test_signed4_capacity_and_reduction():
    original = [-8, -1, 0, 7, 3]
    packed = []
    for i in range(0, len(original), 2):
        low = original[i] & 15
        high = (original[i+1] & 15) if i+1 < len(original) else 0
        packed.append(low | high << 4)
    decoded, unsigned = [], []
    for index in range(len(original)):
        nibble = (packed[index // 2] >> (4*(index % 2))) & 15
        unsigned.append(nibble)
        decoded.append(nibble - 16 if nibble >= 8 else nibble)
    require(decoded == original and len(packed) == 3, 'signed INT4 round trip')
    expect_detection(lambda: require(unsigned == original, 'unsigned nibble interpretation'), 'signedness')
    rows, cols, G = 4096, 4096, 128
    groups = rows*((cols + G - 1)//G)
    total = rows*cols//2 + groups*4
    require(groups == 131072 and total == 8912896, 'capacity model')
    require(2*((129+127)//128) == 4, 'per-row tail group count')
    correct = .5*(1+2) + 2.0*(3+4)
    bad = .5*(1+2+3+4)
    require(correct == 15.5 and bad == 5.0, 'group reduction example')
    expect_detection(lambda: require(correct == bad, 'global scale'), 'group units')
    print('format/cost: signed INT4 odd tail, 8.5 MiB capacity and group units checked')


def main():
    test_frozen_group_regression()
    test_grouped_constrained_reference()
    test_factor_negative()
    test_rounding_range_and_padding()
    test_edges()
    test_signed4_capacity_and_reduction()
    print('scope: deterministic Python binary64 software model, logical Q/metadata and small signed4 packing only')
    print('not measured: target CPU/GPU kernels, FP16 metadata conversion, device synchronization, performance or model quality')
    print('PASS: K-ALG-005 frozen-group lifecycle and independent-reference checks')


if __name__ == '__main__':
    main()
