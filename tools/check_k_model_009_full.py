#!/usr/bin/env python3
"""K-MODEL-009: deterministic offline CPU mechanism checks; no file writes."""
import math
from fractions import Fraction


class HarnessError(AssertionError):
    pass


class Mismatch(AssertionError):
    pass


class DomainError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise HarnessError(message)


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-11)


def verify_number(got, expected, label):
    if not close(got, expected):
        raise Mismatch(label + ': got ' + repr(got) + ', expected ' + repr(expected))


def expect_mismatch(label, operation):
    try:
        operation()
    except Mismatch as exc:
        print('negative control rejected: ' + label + '; ' + str(exc))
        return
    raise HarnessError('defect was accepted: ' + label)


def expect_domain(label, operation):
    try:
        operation()
    except DomainError:
        return
    raise HarnessError('invalid input was accepted: ' + label)


def test_harness():
    def fail():
        raise HarnessError('deliberate framework failure')
    propagated = False
    try:
        expect_mismatch('framework propagation probe', fail)
    except HarnessError as exc:
        require(str(exc) == 'deliberate framework failure', 'wrong framework failure propagated')
        propagated = True
    require(propagated, 'framework failure was swallowed')
    accepted_defect_failed = False
    try:
        expect_mismatch('accepted-object probe', lambda: None)
    except HarnessError:
        accepted_defect_failed = True
    require(accepted_defect_failed, 'accepting a negative control must fail the harness')
    print('framework: verifier rejection, invalid domain and framework failures remain distinct')


def convert(raw, x, alpha, kind):
    if not 0.0 < alpha < 1.0:
        raise DomainError('conversion needs alpha in (0,1)')
    s, n = math.sqrt(alpha), math.sqrt(1.0 - alpha)
    if kind == 'epsilon':
        return (x - n * raw) / s, raw
    if kind == 'x0':
        return raw, (x - s * raw) / n
    if kind == 'v':
        return s * x - n * raw, s * raw + n * x
    raise DomainError('unknown prediction type')


def wrong_v(raw, x, alpha, kind):
    if kind != 'v':
        return convert(raw, x, alpha, kind)
    s, n = math.sqrt(alpha), math.sqrt(1.0 - alpha)
    return s * x - n * raw, s * raw - n * x


def verify_predictions(converter):
    for alpha in (0.02, 0.5, 0.8, 0.99):
        for x0, eps in ((0.7, -0.2), (-1.2, 0.4), (0.0, 0.0)):
            s, n = math.sqrt(alpha), math.sqrt(1.0 - alpha)
            x = s * x0 + n * eps
            v = s * eps - n * x0
            for kind, raw in (('epsilon', eps), ('x0', x0), ('v', v)):
                got_x0, got_eps = converter(raw, x, alpha, kind)
                verify_number(got_x0, x0, kind + ' x0')
                verify_number(got_eps, eps, kind + ' epsilon')
                verify_number(s * got_x0 + n * got_eps, x, kind + ' forward reconstruction')


def test_predictions():
    verify_predictions(convert)
    expect_mismatch('wrong v sign', lambda: verify_predictions(wrong_v))
    verify_predictions(convert)
    expect_domain('alpha=1 conversion', lambda: convert(0.0, 0.0, 1.0, 'x0'))
    print('prediction: known x0/epsilon reference accepted, wrong v rejected, restoration accepted')


def validate_eta(eta):
    if isinstance(eta, bool) or not isinstance(eta, (int, float)):
        raise DomainError('eta must be numeric')
    value = float(eta)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise DomainError('eta must be finite and in [0,1]')
    return value


def validate_alpha(a_t, a_s):
    if not (math.isfinite(a_t) and math.isfinite(a_s) and 0.0 < a_t < a_s <= 1.0):
        raise DomainError('need finite 0 < a_t < a_s <= 1')


def sigma_sq(eta, a_t, a_s):
    eta = validate_eta(eta)
    validate_alpha(a_t, a_s)
    return eta * eta * (1.0 - a_s) / (1.0 - a_t) * (1.0 - a_t / a_s)


def zero_variance_mutant(eta, a_t, a_s):
    validate_eta(eta)
    validate_alpha(a_t, a_s)
    return 0.0


def make_step(variance_function):
    def step(a_t, a_s, x0, eps, eta, noise):
        variance = variance_function(eta, a_t, a_s)
        direction = 1.0 - a_s - variance
        if not math.isfinite(variance) or variance < 0.0 or direction < -1e-12:
            raise DomainError('invalid variance or direction')
        state = (math.sqrt(a_s) * x0 + math.sqrt(max(0.0, direction)) * eps
                 + math.sqrt(variance) * noise)
        return state, variance, direction
    return step


def verify_ddim(variance_function, step):
    # Independent fixed oracle: posterior variance is 3/20 for alpha 1/2 -> 4/5.
    # eta=2/5 gives (4/25)*(3/20)=3/125; direction is 22/125.
    variance_ref = float(Fraction(3, 125))
    direction_ref = float(Fraction(22, 125))
    verify_number(variance_function(0.4, 0.5, 0.8), variance_ref, 'independent sigma squared')
    x0, eps = 0.7, -0.2
    for noise in (-0.2, 0.3, 0.8):
        got, variance, direction = step(0.5, 0.8, x0, eps, 0.4, noise)
        expected = math.sqrt(0.8) * x0 + math.sqrt(direction_ref) * eps + math.sqrt(variance_ref) * noise
        verify_number(variance, variance_ref, 'returned variance')
        verify_number(direction, direction_ref, 'independent direction variance')
        verify_number(got, expected, 'complete next state')
    low = step(0.5, 0.8, x0, eps, 0.4, -0.2)[0]
    high = step(0.5, 0.8, x0, eps, 0.4, 0.8)[0]
    verify_number(high - low, math.sqrt(variance_ref), 'noise perturbation delta')
    for eta, v_ref, d_ref in ((0.0, 0.0, 0.2), (1.0, 0.15, 0.05)):
        got, variance, direction = step(0.5, 0.8, x0, eps, eta, 0.3)
        verify_number(variance, v_ref, 'eta endpoint variance')
        verify_number(direction, d_ref, 'eta endpoint direction')
        expected = math.sqrt(0.8) * x0 + math.sqrt(d_ref) * eps + math.sqrt(v_ref) * 0.3
        verify_number(got, expected, 'eta endpoint state')
    got, variance, direction = step(0.5, 1.0, x0, eps, 0.4, 9.0)
    verify_number(got, x0, 'clean target endpoint')
    verify_number(variance, 0.0, 'clean endpoint variance')
    verify_number(direction, 0.0, 'clean endpoint direction')


def verify_domains(variance_function):
    for bad in (-0.4, float('nan'), float('inf'), -float('inf'), 1.2, True, '0.4'):
        expect_domain('invalid eta ' + repr(bad), lambda value=bad: variance_function(value, 0.5, 0.8))
    for a_t, a_s in ((0.0, 0.8), (0.8, 0.5), (0.8, 0.8), (0.5, 1.1), (float('nan'), 0.8)):
        expect_domain('invalid alpha pair', lambda t=a_t, s=a_s: variance_function(0.4, t, s))


def test_ddim():
    step = make_step(sigma_sq)
    verify_domains(sigma_sq)
    verify_ddim(sigma_sq, step)
    verify_domains(zero_variance_mutant)
    bad_step = make_step(zero_variance_mutant)
    expect_mismatch('domain-preserving zero variance', lambda: verify_ddim(zero_variance_mutant, bad_step))
    # Keep the correct public variance result but corrupt only the step path.
    expect_mismatch('step-only zero variance', lambda: verify_ddim(sigma_sq, bad_step))
    verify_ddim(sigma_sq, step)
    print('DDIM: independent sigma^2=0.024, direction=0.176, full state and noise delta checked')
    print('DDIM: zero-variance mutants rejected; correct implementation accepted again; domains retained')


def matrix_shape(matrix):
    if not matrix or not matrix[0]:
        raise Mismatch('empty matrix outside this reference scope')
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise Mismatch('ragged matrix')
    return len(matrix), width


def transpose(matrix):
    rows, cols = matrix_shape(matrix)
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]


def matmul(left, right):
    rows, inner = matrix_shape(left)
    right_rows, cols = matrix_shape(right)
    if inner != right_rows:
        raise Mismatch('matrix inner axes disagree: ' + str(inner) + ' vs ' + str(right_rows))
    return [[sum(left[i][k] * right[k][j] for k in range(inner))
             for j in range(cols)] for i in range(rows)]


def gelu(value):
    return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))


def activate(matrix):
    return [[gelu(value) for value in row] for row in matrix]


def row_ffn(u, w1, w2):
    return matmul(activate(matmul(u, w1)), w2)


def wrong_left_ffn(u, w1, w2):
    # Column-vector weights incorrectly left-multiply the whole token matrix.
    return matmul(transpose(w2), activate(matmul(transpose(w1), u)))


def scalar_ffn_reference(u, w1, w2):
    # Independent per-token construction, not a call to matrix multiplication.
    result = []
    for token in u:
        hidden = []
        for j in range(len(w2)):
            total = 0.0
            for channel, value in enumerate(token):
                total += value * w1[channel][j]
            hidden.append(gelu(total))
        row = []
        for channel in range(len(token)):
            total = 0.0
            for j, value in enumerate(hidden):
                total += value * w2[j][channel]
            row.append(total)
        result.append(row)
    return result


def verify_matrix(got, expected, label):
    if matrix_shape(got) != matrix_shape(expected):
        raise Mismatch(label + ' shape mismatch')
    for i, row in enumerate(expected):
        for j, value in enumerate(row):
            verify_number(got[i][j], value, label + ' at ' + str((i, j)))


def verify_ffn(implementation):
    w1 = [[0.2, -0.3, 0.7, 0.1], [0.4, 0.8, -0.2, 0.5]]
    w2 = [[0.6, 0.1], [-0.4, 0.3], [0.2, 0.9], [0.5, -0.7]]
    for tokens in (3, 1, 2, 7):
        u = [[(i + 1) * 0.2, (i % 3 - 1) * 0.3] for i in range(tokens)]
        reference = scalar_ffn_reference(u, w1, w2)
        got = implementation(u, w1, w2)
        verify_matrix(got, reference, 'FFN')
        require(matrix_shape(got) == (tokens, 2), 'residual output must remain T x d')
        residual = [[u[i][j] + got[i][j] for j in range(2)] for i in range(tokens)]
        require(matrix_shape(residual) == (tokens, 2), 'residual shape changed')
        reordered = list(reversed(u))
        verify_matrix(implementation(reordered, w1, w2), list(reversed(reference)), 'token permutation')


def test_ffn():
    verify_ffn(row_ffn)
    expect_mismatch('left-multiplication of token matrix', lambda: verify_ffn(wrong_left_ffn))
    verify_ffn(row_ffn)
    print('FFN: T=1/2/3/7, d=2, d_ff=4; scalar token reference and residual axes agree')


def cross_projection_mac(source_length, source_width, output_width):
    return source_length * source_width * output_width


def wrong_cross_mac(source_length, source_width, output_width):
    return source_length * output_width * output_width


def verify_cross_cost(counter):
    # Enumerate scalar products for a small rectangular projection.
    independent_count = 0
    for source_row in range(3):
        for out_channel in range(5):
            for source_channel in range(2):
                independent_count += 1
    require(independent_count == 30, 'reference operation enumeration failed')
    for shape, expected in (((3, 2, 5), independent_count),
                            ((77, 768, 1536), 90832896),
                            ((2, 3, 3), 18)):
        got = counter(*shape)
        if got != expected:
            raise Mismatch('cross projection MAC mismatch for ' + str(shape))


def test_cross_cost():
    verify_cross_cost(cross_projection_mac)
    expect_mismatch('source width replaced by output width', lambda: verify_cross_cost(wrong_cross_mac))
    verify_cross_cost(cross_projection_mac)
    each = cross_projection_mac(77, 768, 1536)
    require((30 - 1) * 2 * each == 5268307968, 'cached source projection saving mismatch')
    require(2 * 77 * 1536 * 2 == 473088, 'K/V cache output bytes mismatch')
    print('cross cost: each K/V projection 90832896 MAC; 30-call saving 5268307968 MAC per layer/branch')


def ceil_div(n, d):
    return (n + d - 1) // d


def verify_bytes(h, w, channels, element_bytes, proposed):
    # Direct logical grid reference; deliberately independent of patch count.
    expected = h * w * channels * element_bytes
    if proposed != expected:
        raise Mismatch('logical latent byte mismatch: ' + str(proposed) + ' vs ' + str(expected))


def test_capacity_and_scenario():
    h = w = 128
    logical = 131072
    for p in (1, 2, 4, 8):
        tokens = ceil_div(h, p) * ceil_div(w, p)
        verify_bytes(h, w, 4, 2, p * p * tokens * 4 * 2)
    expect_mismatch('missing patch area', lambda: verify_bytes(h, w, 4, 2, 4096 * 4 * 2))
    verify_bytes(h, w, 4, 2, logical)
    h2, w2, p2 = 130, 127, 4
    padded = ceil_div(h2, p2) * p2 * ceil_div(w2, p2) * p2 * 4 * 2
    require(padded == 135168, 'padded working bytes mismatch')
    expect_mismatch('padding treated as logical output', lambda: verify_bytes(h2, w2, 4, 2, padded))
    verify_bytes(h2, w2, 4, 2, 132080)
    mib = 1024 ** 2
    require(4096 * 1536 * 2 == 12 * mib, 'hidden bytes mismatch')
    require(16 * 4096 * 4096 * 2 == 512 * mib, 'scores bytes mismatch')
    require(8 + 6 == 14 and 8 + 2 * 6 == 20 and 20 <= 24, 'scenario budget mismatch')
    require(40 + 30 * (2 * 30 + 1) + 120 == 1990, 'sequential scenario timing mismatch')
    require(40 + 30 * (45 + 1) + 120 == 1540, 'merged scenario timing mismatch')
    print('capacity/scenario: logical prediction 128 KiB independent of divisible p; hidden 12 MiB; scores 512 MiB')
    print('scenario arithmetic only: 14/20 GiB budgets and 1990/1540 ms configured serial paths')


def validate_schedule(times):
    if len(times) < 2:
        raise DomainError('need at least two boundaries')
    if any(times[i] <= times[i + 1] for i in range(len(times) - 1)):
        raise DomainError('boundaries must strictly decrease')


def verify_next(times, index, chosen):
    validate_schedule(times)
    if not 0 <= index < len(times) - 1:
        raise DomainError('no update at this index')
    if chosen != times[index + 1]:
        raise Mismatch('next boundary is not the selected entry')


def test_schedule_and_cfg():
    times = [999, 750, 500, 250, 0]
    verify_next(times, 1, 500)
    expect_mismatch('raw t-1', lambda: verify_next(times, 1, 749))
    verify_next(times, 1, 500)
    expect_domain('one boundary', lambda: validate_schedule([0]))
    expect_domain('nondecreasing boundaries', lambda: validate_schedule([999, 750, 751]))
    expect_domain('past final update', lambda: verify_next(times, 4, 0))
    require(len(times) - 1 == 4, 'boundary/update count mismatch')
    u, c = 0.1, 0.7
    for weight in (0.0, 1.0, 2.0):
        got = u + weight * (c - u)
        verify_number(got, (1.0 - weight) * u + weight * c, 'CFG affine reference')
    verify_number(u + 0.0 * (c - u), u, 'CFG unconditional endpoint')
    verify_number(u + 1.0 * (c - u), c, 'CFG conditional endpoint')
    print('schedule/CFG: selected boundary and count checked; affine endpoints checked; no upstream scheduler execution')


def conv_valid(signal, kernel):
    return [sum(signal[i + j] * value for j, value in enumerate(kernel))
            for i in range(len(signal) - len(kernel) + 1)]


def required_range(start, stop, kernel_length):
    return start, stop + kernel_length - 1


def missing_right_input(start, stop, kernel_length):
    return start, stop + kernel_length - 2


def verify_tiles(reader, signal, kernel, width):
    reference = conv_valid(signal, kernel)
    observed = [None] * len(reference)
    for start in range(0, len(reference), width):
        stop = min(start + width, len(reference))
        left, right = reader(start, stop, len(kernel))
        patch = signal[left:right]
        local = conv_valid(patch, kernel)
        for position in range(start, stop):
            offset = position - left
            if not 0 <= offset < len(local):
                raise Mismatch('missing required input for output ' + str(position))
            observed[position] = local[offset]
    for position, expected in enumerate(reference):
        if observed[position] is None:
            raise Mismatch('unwritten output position')
        verify_number(observed[position], expected, 'tile output ' + str(position))


def test_tiling():
    signal = [float((i * 7 + 3) % 11) for i in range(31)]
    kernel = [0.2, 0.3, 0.3, 0.1, 0.1]
    for width in (1, 7, 40):
        verify_tiles(required_range, signal, kernel, width)
    expect_mismatch('one required right input omitted', lambda: verify_tiles(missing_right_input, signal, kernel, 7))
    verify_tiles(required_range, signal, kernel, 7)
    verify_tiles(required_range, signal, [1.0], 7)
    print('synthetic 1D convolution: identical output coordinates checked; one missing input rejected; restoration accepted')


def main():
    print('scope: Python stdlib CPU algebra, small FFN matrices, operation counts, logical bytes and synthetic convolution')
    print('not measured: real checkpoint, GPU numerics/timing/HBM/synchronization, real VAE or image quality')
    test_harness()
    test_predictions()
    test_ddim()
    test_ffn()
    test_cross_cost()
    test_capacity_and_scenario()
    test_schedule_and_cfg()
    test_tiling()
    print('all scoped checks completed; injected defects rejected and restored implementations accepted')


if __name__ == '__main__':
    main()
