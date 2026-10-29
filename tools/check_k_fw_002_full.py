from decimal import Decimal, localcontext
from fractions import Fraction as F
import itertools
import json
import math
import struct


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def pow2(e):
    return F(2 ** e) if e >= 0 else F(1, 2 ** (-e))


def round_binary(value, precision=24):
    value = F(value)
    if value == 0:
        return F(0)
    sign = -1 if value < 0 else 1
    value = abs(value)
    e = value.numerator.bit_length() - value.denominator.bit_length()
    if value < pow2(e):
        e -= 1
    if value >= pow2(e + 1):
        e += 1
    step = pow2(e - precision + 1)
    scaled = value / step
    whole, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    if twice > scaled.denominator or (twice == scaled.denominator and whole % 2):
        whole += 1
    return sign * whole * step


def sequential(values):
    total = F(0)
    for value in values:
        total = round_binary(total + value)
    return total


def pairwise(values):
    if not values:
        return F(0)
    if len(values) == 1:
        return round_binary(values[0])
    middle = len(values) // 2
    return round_binary(pairwise(values[:middle]) + pairwise(values[middle:]))


def compare(got, reference, atol, rtol):
    require(math.isfinite(atol) and math.isfinite(rtol), 'nonfinite budget')
    require(atol >= 0 and rtol >= 0, 'negative budget')
    if len(got) != len(reference):
        return 'shape'
    if not got:
        return 'empty'
    if any(not math.isfinite(v) for v in list(got) + list(reference)):
        return 'nonfinite'
    for value, target in zip(got, reference):
        error = abs(F(value) - F(target))
        budget = F(atol) + F(rtol) * abs(F(target))
        if error > budget:
            return 'numeric'
    return 'pass'


def argmax(values):
    return max(range(len(values)), key=lambda i: (values[i], -i))


def stable_softmax(values):
    require(bool(values), 'softmax requires a nonempty finite domain')
    require(all(math.isfinite(x) for x in values), 'invalid logits')
    maximum = max(values)
    weights = [math.exp(x - maximum) for x in values]
    denominator = math.fsum(weights)
    return [x / denominator for x in weights]


def decimal_softmax(values):
    with localcontext() as context:
        context.prec = 90
        logits = [Decimal.from_float(x) for x in values]
        maximum = max(logits)
        weights = [(x - maximum).exp() for x in logits]
        denominator = sum(weights, Decimal(0))
        return [float(x / denominator) for x in weights]


def affine_reference(groups):
    total = F(0)
    for xs, ws, zx, zw, sx, sw in groups:
        require(len(xs) == len(ws), 'invalid group lengths')
        for x, w in zip(xs, ws):
            real_x = sx * (x - zx)
            real_w = sw * (w - zw)
            total += real_x * real_w
    return total


def compensated(groups, wrong_global_scale=False):
    partials = []
    for xs, ws, zx, zw, sx, sw in groups:
        raw = sum(x * w for x, w in zip(xs, ws))
        integer_sum = raw - zw * sum(xs) - zx * sum(ws) + len(xs) * zx * zw
        partials.append((integer_sum, sx * sw))
    if wrong_global_scale:
        return sum(p for p, _ in partials) * partials[0][1]
    return sum((p * scale for p, scale in partials), F(0))


def paged_trace(codes, scales, chunk_size, fault=None):
    pages = []
    result = []
    for start in range(0, len(codes), chunk_size):
        for position in range(start, min(start + chunk_size, len(codes))):
            page = position // 2
            if page == len(pages):
                pages.append([])
            pages[page].append(codes[position])
            decoded = []
            for page_id, values in enumerate(pages):
                if fault == 'omit_old_page' and len(pages) > 1 and page_id == 0:
                    continue
                scale_id = max(0, page_id - 1) if fault == 'stale_scale' else page_id
                for code in values:
                    decoded.append(scales[scale_id] * (code - 3))
            result.append((('request-A', position + 1, 'attention'), sum(decoded, F(0)) / len(decoded)))
    return result


def record_map(records):
    output = {}
    for key, value in records:
        if key in output:
            raise ValueError('duplicate logical identity')
        output[key] = value
    return output


def first_mismatch(reference, candidate):
    expected = record_map(reference)
    actual = record_map(candidate)
    if set(expected) != set(actual):
        return ('identity-set',)
    for key, value in reference:
        if actual[key] != value:
            return key
    return None


def main():
    observations = {}
    negatives = []
    require(compare([0.125], [0.0], 0.125, 0.0) == 'pass', 'inclusive tolerance')
    require(compare([0.25], [0.0], 0.125, 0.0) == 'numeric', 'missed numeric error')
    require(compare([0.0], [0.0], 0.0, 0.0) == 'pass', 'zero budget')
    require(compare([], [], 0.0, 0.0) == 'empty', 'empty evidence classification')
    for values in ([float('nan')], [float('inf')], [float('-inf')]):
        require(compare(values, [0.0], 1.0, 1.0) == 'nonfinite', 'accepted nonfinite')
    require(compare([float('inf')], [float('inf')], 1.0, 1.0) == 'nonfinite', 'inf matched under finite contract')
    require(compare([], [0.0], 1.0, 0.0) == 'shape', 'missing value accepted')
    require(compare([1.0], [2.0], 0.0, 0.5) == 'pass', 'reference orientation')
    require(compare([2.0], [1.0], 0.0, 0.5) == 'numeric', 'unexpected symmetric tolerance')
    negatives.extend(['nonfinite_payload', 'missing_value', 'out_of_budget'])
    observations['comparator'] = 'finite contract, zero and inclusive budgets, empty and missing, reference orientation'

    vectors = [[], [F(0)], [F(1), F(-1)], [F(2 ** 24), F(1), F(-(2 ** 24))], [F(1), pow2(-24), F(-1)], [F(1, 8), F(2, 8), F(-4, 8), F(8, 8), F(1, 16)]]
    u = pow2(-24)
    for vector in vectors:
        exact = sum(vector, F(0))
        count = max(0, len(vector) - 1)
        gamma = count * u / (1 - count * u)
        scale = sum((abs(x) for x in vector), F(0))
        require(abs(sequential(vector) - exact) <= gamma * scale, 'sequential error exceeds stated normal-range bound')
        depth = (len(vector) - 1).bit_length() if vector else 0
        tree_gamma = depth * u / (1 - depth * u)
        require(abs(pairwise(vector) - exact) <= tree_gamma * scale, 'tree error exceeds depth bound')
    require(round_binary(1 + pow2(-24)) == 1, 'even tie at one')
    require(round_binary(1 + 3 * pow2(-24)) == 1 + pow2(-22), 'odd tie rounds to even')
    for value in [F(0), F(1), F(-1), 1 + pow2(-24), 1 + 3 * pow2(-24), F(2 ** 24 + 1), F(1, 8)]:
        packed = struct.unpack('>f', struct.pack('>f', float(value)))[0]
        require(round_binary(value) == F(packed), 'selected binary32 normal-range cross-check')
    cancellation = vectors[3]
    require(sequential(cancellation) == 0 and pairwise(cancellation) == 1, 'cancellation witness')
    condition = sum(abs(x) for x in cancellation) / abs(sum(cancellation))
    require(condition == 33554433, 'condition scale')
    negatives.append('claim_of_order_independent_bits')
    observations['reduction'] = {'exact': '1', 'sequential': str(sequential(cancellation)), 'tree': str(pairwise(cancellation)), 'condition': str(condition)}

    certified = 0
    disagreements = 0
    for gap in [F(1, 4), F(1), F(4)]:
        for epsilon in [F(0), F(1, 16), F(1, 8), F(1, 4), F(1, 2)]:
            reference = [F(0), gap]
            for e0, e1 in itertools.product([-epsilon, F(0), epsilon], repeat=2):
                output = [e0, gap + e1]
                if 2 * epsilon < gap:
                    require(argmax(output) == argmax(reference), 'invalid margin certificate')
                    certified += 1
                elif argmax(output) != argmax(reference):
                    disagreements += 1
    require(argmax([F(0), F(1, 4)]) != argmax([F(1, 8), F(1, 8)]), 'tie boundary not exercised')
    bad_epsilon = F(1, 5)
    require(bad_epsilon < F(1, 4), 'wrong certificate setup')
    require(argmax([bad_epsilon, F(1, 4) - bad_epsilon]) != 1, 'wrong epsilon < margin rule was not exposed')
    negatives.extend(['nonstrict_margin_boundary', 'missing_factor_two'])
    observations['margin'] = {'certified_grid_cases': certified, 'uncertified_disagreements': disagreements}

    max_softmax_error = 0.0
    for values in [[-1000.0, 0.0, 1000.0], [0.0], [0.0, 0.25, -0.5], [-20.0, -20.25, -19.75]]:
        got = stable_softmax(values)
        reference = decimal_softmax(values)
        error = max(abs(a - b) for a, b in zip(got, reference))
        max_softmax_error = max(max_softmax_error, error)
        require(error < 2e-15, 'stable softmax differs from Decimal reference')
        require(abs(math.fsum(got) - 1.0) < 2e-15, 'probability mass')
    logits = [0.0, 0.25, -0.5]
    base = stable_softmax(logits)
    shifted = stable_softmax([x + 1024.0 for x in logits])
    require(max(abs(a - b) for a, b in zip(base, shifted)) < 2e-15, 'common-shift invariance')
    epsilon = 0.125
    maximum_delta = 0.0
    for errors in itertools.product([-epsilon, 0.0, epsilon], repeat=3):
        perturbed = stable_softmax([x + e for x, e in zip(logits, errors)])
        delta = max(abs(a - b) for a, b in zip(base, perturbed))
        require(delta <= epsilon / 2 + 2e-15, 'Jacobian infinity bound')
        maximum_delta = max(maximum_delta, delta)
    overflow_detected = False
    try:
        weights = [math.exp(x) for x in [-1000.0, 0.0, 1000.0]]
        denominator = sum(weights)
        bad = [x / denominator for x in weights]
        overflow_detected = compare(bad, stable_softmax([-1000.0, 0.0, 1000.0]), 1e-12, 1e-12) != 'pass'
    except OverflowError:
        overflow_detected = True
    require(overflow_detected, 'unstable-exp negative not detected')
    negatives.append('unshifted_extreme_softmax')
    observations['softmax'] = {'maximum_reference_error': max_softmax_error, 'maximum_grid_delta': maximum_delta, 'scope': 'binary64 versus 90-digit Decimal, finite logits'}

    with localcontext() as context:
        context.prec = 80
        exact_values = [Decimal(100000001), Decimal(99999999)]
        mean = sum(exact_values) / 2
        variance = sum((x - mean) ** 2 for x in exact_values) / 2
    values = [100000001.0, 99999999.0]
    mean_float = sum(values) / 2
    raw_variance = sum(x * x for x in values) / 2 - mean_float * mean_float
    centered_variance = sum((x - mean_float) ** 2 for x in values) / 2
    require(variance == 1 and centered_variance == 1.0 and raw_variance == 0.0, 'variance cancellation witness')
    negatives.append('raw_moment_variance_on_offset_input')
    observations['variance'] = {'Decimal_reference': str(variance), 'binary64_raw_moment': raw_variance, 'binary64_centered': centered_variance}

    groups = [([5, 7], [2, 0], 5, 1, F(1, 2), F(2)), ([3], [6], 1, 2, F(1, 4), F(1, 2))]
    direct = affine_reference(groups)
    require(direct == -1 and compensated(groups) == direct, 'affine compensation')
    wrong = compensated(groups, wrong_global_scale=True)
    require(wrong == 6 and wrong != direct, 'wrong group-unit merge not detected')
    padded = [(xs + [zx], ws + [zw], zx, zw, sx, sw) for xs, ws, zx, zw, sx, sw in groups]
    require(affine_reference(padded) == direct and compensated(padded) == direct, 'quantized zero padding')
    bad_padding = [(xs + [0], ws + [0], zx, zw, sx, sw) for xs, ws, zx, zw, sx, sw in groups]
    require(compensated(bad_padding) != direct, 'byte-zero padding negative not detected')
    negatives.extend(['global_scale_for_different_groups', 'byte_zero_padding'])
    observations['affine'] = {'direct': str(direct), 'compensated': str(compensated(groups)), 'faulty_global_scale': str(wrong)}

    codes = [4, 7, 2, 6, 5]
    scales = [F(1, 4), F(1, 2), F(1, 4)]
    canonical_values = [F(1, 4), F(1), F(-1, 2), F(3, 2), F(1, 2)]
    reference = [(('request-A', t, 'attention'), sum(canonical_values[:t], F(0)) / t) for t in range(1, 6)]
    require([x for _, x in reference] == [F(1, 4), F(5, 8), F(1, 4), F(9, 16), F(11, 20)], 'independent dense means')
    for chunk in [1, 2, 3, 5, 8]:
        require(first_mismatch(reference, paged_trace(codes, scales, chunk)) is None, 'legal chunk changed function')
    require(paged_trace([], [], 1) == [], 'empty cache')
    stale = first_mismatch(reference, paged_trace(codes, scales, 2, 'stale_scale'))
    missing = first_mismatch(reference, paged_trace(codes, scales, 3, 'omit_old_page'))
    require(stale == ('request-A', 3, 'attention'), 'wrong stale-scale divergence location')
    require(missing == ('request-A', 3, 'attention'), 'wrong omitted-page divergence location')
    require(first_mismatch(reference, reference[:-1]) == ('identity-set',), 'missing boundary accepted')
    duplicate_detected = False
    try:
        first_mismatch(reference, reference + [reference[0]])
    except ValueError:
        duplicate_detected = True
    require(duplicate_detected, 'duplicate identity accepted')
    originals = [F(3, 10), F(11, 10), F(-2, 5), F(7, 5), F(9, 20)]
    representation_loss = [abs(sum(originals[:t], F(0)) / t - reference[t - 1][1]) for t in range(1, 6)]
    require(any(x > 0 for x in representation_loss), 'representation and execution errors conflated')
    negatives.extend(['stale_page_scale', 'omitted_old_page', 'missing_boundary', 'duplicate_identity'])
    observations['cache'] = {'quantized_reference_means': [str(v) for _, v in reference], 'stale_scale_first_step': stale[1], 'omitted_page_first_step': missing[1], 'original_model_mean_errors': [str(x) for x in representation_loss]}

    observations['negative_controls_detected'] = negatives
    observations['scope'] = 'Deterministic standalone stdlib arithmetic checks; no files, network, subprocesses, project imports, real model, GPU, timing, or concurrency tests. Binary rounding uses an unbounded exponent model; selected normal-range outputs are cross-checked with struct binary32.'
    print(json.dumps(observations, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
