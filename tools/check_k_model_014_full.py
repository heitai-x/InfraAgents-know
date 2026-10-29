#!/usr/bin/env python3
'''K-MODEL-014: deterministic, read-only, offline stdlib mechanism checks.'''

from fractions import Fraction as F
from itertools import product


def equal(actual, expected):
    if actual != expected:
        raise AssertionError('actual=' + repr(actual) + ' expected=' + repr(expected))


def observation(name, actual, expected):
    equal(actual, expected)
    print('OBS', name, repr(actual))


def negative(name, actual, expected):
    try:
        equal(actual, expected)
    except AssertionError:
        print('DETECTED', name, 'actual=' + repr(actual), 'reference=' + repr(expected))
        return
    raise AssertionError('negative control escaped: ' + name)


def rejects(name, fn):
    try:
        fn()
    except ValueError:
        print('OBS', name, 'rejected')
        return
    raise AssertionError('expected ValueError: ' + name)


def dimensions(x, w, spec):
    sh, sw, ph, pw, dh, dw, groups = spec
    b, ci, h, width = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
    co, kh, kw = len(w), len(w[0][0]), len(w[0][0][0])
    if min(sh, sw, dh, dw, groups) <= 0 or min(ph, pw) < 0:
        raise ValueError('invalid convolution parameters')
    if ci % groups or co % groups or len(w[0]) != ci // groups:
        raise ValueError('invalid channel groups')
    return b, ci, h, width, co, kh, kw


def zeros(b, co, oh, ow, bias):
    return [[[[0 if bias is None else bias[c] for _ in range(ow)]
              for _ in range(oh)] for c in range(co)] for _ in range(b)]


def conv_gather(x, w, bias, spec, faulty_dilation=False, faulty_groups=False):
    batch, cin, h, width, cout, kh, kw = dimensions(x, w, spec)
    sh, sw, ph, pw, dh, dw, groups = spec
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if min(oh, ow) <= 0:
        return []
    y = zeros(batch, cout, oh, ow, bias)
    sample_dh, sample_dw = (1, 1) if faulty_dilation else (dh, dw)
    for b in range(batch):
        for co in range(cout):
            group = 0 if faulty_groups else co // (cout // groups)
            for oy in range(oh):
                for ox in range(ow):
                    for cl in range(cin // groups):
                        ci = group * (cin // groups) + cl
                        for ky in range(kh):
                            iy = oy * sh - ph + ky * sample_dh
                            for kx in range(kw):
                                ix = ox * sw - pw + kx * sample_dw
                                if 0 <= iy < h and 0 <= ix < width:
                                    y[b][co][oy][ox] += x[b][ci][iy][ix] * w[co][cl][ky][kx]
    return y


def conv_scatter_reference(x, w, bias, spec):
    # Independent organization: enumerate legal origins, then distribute each
    # input contribution to output positions. No shared output-size helper.
    batch, cin, h, width, cout, kh, kw = dimensions(x, w, spec)
    sh, sw, ph, pw, dh, dw, groups = spec
    origins_y = list(range(-ph, h + ph - dh * (kh - 1), sh))
    origins_x = list(range(-pw, width + pw - dw * (kw - 1), sw))
    if not origins_y or not origins_x:
        return []
    row_of = {origin: i for i, origin in enumerate(origins_y)}
    col_of = {origin: i for i, origin in enumerate(origins_x)}
    y = zeros(batch, cout, len(origins_y), len(origins_x), bias)
    for b in range(batch):
        for ci in range(cin):
            group, cl = divmod(ci, cin // groups)
            for iy in range(h):
                for ix in range(width):
                    for ky in range(kh):
                        oy = row_of.get(iy - ky * dh)
                        if oy is None:
                            continue
                        for kx in range(kw):
                            ox = col_of.get(ix - kx * dw)
                            if ox is None:
                                continue
                            for co in range(group * (cout // groups), (group + 1) * (cout // groups)):
                                y[b][co][oy][ox] += x[b][ci][iy][ix] * w[co][cl][ky][kx]
    return y


def make_x(b, c, h, width):
    return [[[[((ib * 7 + ic * 5 + iy * 3 + ix * 11 + 1) % 9) - 4
               for ix in range(width)] for iy in range(h)] for ic in range(c)] for ib in range(b)]


def make_w(co, ci_local, kh, kw):
    return [[[[((oc * 3 + ic * 7 + ky * 5 + kx * 2 + 2) % 7) - 3
               for kx in range(kw)] for ky in range(kh)] for ic in range(ci_local)] for oc in range(co)]


def test_convolution():
    cases = [
        (2, 4, 5, 6, 6, 3, 3, (1, 1, 1, 1, 1, 1, 2)),
        (1, 3, 4, 5, 6, 2, 3, (2, 2, 1, 1, 2, 2, 3)),
        (1, 2, 3, 4, 2, 1, 1, (1, 2, 0, 0, 1, 1, 2)),
    ]
    for index, (b, ci, h, width, co, kh, kw, spec) in enumerate(cases):
        x, w = make_x(b, ci, h, width), make_w(co, ci // spec[-1], kh, kw)
        bias = [i - 2 for i in range(co)]
        equal(conv_gather(x, w, bias, spec), conv_scatter_reference(x, w, bias, spec))
        if index == 0:
            negative('conv_group_mapping', conv_gather(x, w, bias, spec, faulty_groups=True),
                     conv_scatter_reference(x, w, bias, spec))
        if index == 1:
            reference = conv_scatter_reference(x, w, bias, spec)
            wrong = conv_gather(x, w, bias, spec, faulty_dilation=True)
            # The fault changes sample spacing only; output dimensions stay fixed.
            equal((len(wrong[0][0]), len(wrong[0][0][0])), (2, 2))
            equal((len(reference[0][0]), len(reference[0][0][0])), (2, 2))
            negative('conv_dilation_same_output_shape', wrong, reference)
    print('OBS conv_gather_vs_input_scatter cases=3; grouped, batch=2, dilation, depthwise multiplier')
    observation('conv_hand_boundary',
                conv_gather([[[[1, 2, 3]]]], [[[[1, 2, 3]]]], None, (1, 1, 0, 1, 1, 1, 1)),
                [[[[8, 14, 8]]]])
    x, w = make_x(1, 2, 1, 1), make_w(2, 2, 2, 2)
    spec = (1, 1, 0, 0, 1, 1, 1)
    observation('conv_empty', conv_gather(x, w, None, spec), [])
    equal(conv_scatter_reference(x, w, None, spec), [])
    rejects('conv_illegal_group', lambda: conv_gather(x, w, None, (1, 1, 0, 0, 1, 1, 3)))


def mean_at(values, indices):
    selected = [values[i] for i in indices]
    if not selected:
        raise ValueError('empty pooling domain')
    return sum((F(v) for v in selected), F(0)) / len(selected)


def line_conv(values, kernel, bias=0):
    radius = len(kernel) // 2
    out = []
    for position in range(len(values)):
        value = bias
        for k, weight in enumerate(kernel):
            source = position - radius + k
            if 0 <= source < len(values):
                value += values[source] * weight
        out.append(value)
    return out


def line_reference(values, kernel, bias=0):
    # Independently materialize zero boundaries and complete sliding windows.
    radius = len(kernel) // 2
    padded = [0] * radius + list(values) + [0] * radius
    return [bias + sum(a * b for a, b in zip(padded[start:start + len(kernel)], kernel))
            for start in range(len(values))]


def bucket_layers(values, capacity, layers, isolate_each_layer):
    n = len(values)
    if n == 0 or capacity < n:
        raise ValueError('invalid logical domain or capacity')
    current = list(values) + [0] * (capacity - n)
    for kernel, bias in layers:
        current = line_conv(current, kernel, bias)
        if isolate_each_layer:
            current[n:] = [0] * (capacity - n)
    return current


def test_padding_contract():
    # Immutable processor output: position 1 is MODEL padding, not an extra slot.
    processor_output = [1, 0]
    identity = [0, 1, 0]
    model_features = line_conv(processor_output, identity)
    observation('model_letterbox_original_domain', mean_at(model_features, range(2)), F(1, 2))
    negative('source_pixel_mask_changes_checkpoint_function',
             mean_at(model_features, [0]), F(1, 2))
    extended = bucket_layers(processor_output, 4, [(identity, 0)], True)
    observation('extra_slots_excluded_but_model_padding_retained', mean_at(extended, range(2)), F(1, 2))
    negative('physical_capacity_used_as_denominator', mean_at(extended, range(4)), F(1, 2))
    negative('source_mask_reused_as_execution_domain', mean_at(extended, [0]), F(1, 2))

    kernel = [1, 1, 1]
    first = line_conv([1, 0], kernel)
    second = line_conv(first, kernel)
    observation('boundary_first_layer_full_trace', first, [1, 1])
    observation('boundary_second_layer_full_trace', second, [2, 2])
    independent = line_reference(line_reference([1], kernel), kernel)
    observation('independent_two_layer_output', independent, [1])
    negative('final_mask_does_not_remove_boundary_return', mean_at(second, [0]), F(1))
    isolated = bucket_layers([1], 2, [(kernel, 0), (kernel, 0)], True)
    observation('per_layer_logical_boundary_restored', isolated, [1, 0])
    equal(mean_at(isolated, [0]), F(1))

    layer_sets = [
        [(identity, 0), (identity, 0)],
        [(kernel, 0), (kernel, 0)],
        [([2, -1, 3], 1), ([1, 2, -1], -1)],
    ]
    count = 0
    for n in range(1, 5):
        for values in product((-1, 0, 1), repeat=n):
            for capacity in range(n, n + 3):
                for layers in layer_sets:
                    reference = list(values)
                    for weights, bias in layers:
                        reference = line_reference(reference, weights, bias)
                    result = bucket_layers(values, capacity, layers, True)
                    equal(result[:n], reference)
                    equal(result[n:], [0] * (capacity - n))
                    equal(mean_at(result, range(n)), mean_at(reference, range(n)))
                    count += 1
    print('OBS per_layer_isolation_enumeration cases=' + str(count) + '; width=1..4, extra=0..2, signed inputs, bias')
    rejects('empty_pooling_domain', lambda: mean_at([], []))
    rejects('bucket_capacity_overflow', lambda: bucket_layers([1, 0], 1, layer_sets[0], True))


def test_bn_residual():
    gamma, beta, mean, denominator = F(4), F(1, 2), F(1), F(2)
    weight, bias = F(2), F(1, 4)
    scale = gamma / denominator
    folded_weight = scale * weight
    folded_bias = scale * (bias - mean) + beta
    for x in (F(-3), F(0), F(2), F(7, 3)):
        reference = gamma * (weight * x + bias - mean) / denominator + beta
        equal(folded_weight * x + folded_bias, reference)
    print('OBS fixed_bn_fold rational_affine_reference inputs=4; supplied sqrt denominator=2')
    x = F(2)
    reference = gamma * (weight * x + bias - mean) / denominator + beta
    negative('bn_missing_mean', folded_weight * x + scale * bias + beta, reference)
    original = [1, 2, 3]
    correct = [value + (value + 1) for value in original]
    overwritten = [value + 1 for value in original]
    negative('residual_overwrites_original', [2 * value for value in overwritten], correct)
    equal(correct, [3, 5, 7])


def test_coordinates():
    # Ideal unrounded geometry only; no image decode or interpolation.
    scale = min(F(8, 6), F(8, 4))
    px, py = (8 - 6 * scale) / 2, (8 - 4 * scale) / 2
    point = (F(3), F(2))
    mapped = (px + scale * point[0], py + scale * point[1])
    observation('letterbox_fixed_center', mapped, (F(4), F(4)))
    restored = ((mapped[0] - px) / scale, (mapped[1] - py) / scale)
    observation('letterbox_inverse', restored, point)
    negative('letterbox_inverse_omits_offset', (mapped[0] / scale, mapped[1] / scale), point)
    # A separately specified rounded geometry has unequal axis scales.
    sx, sy, left, top = F(7, 5), F(4, 3), F(0), F(2)
    original = (F(5), F(3))
    target = (left + sx * original[0], top + sy * original[1])
    observation('rounded_geometry_edge', target, (F(7), F(6)))
    equal(((target[0] - left) / sx, (target[1] - top) / sy), original)
    negative('rounded_geometry_guessed_uniform_scale',
             ((target[0] - left) / sx, (target[1] - top) / sx), original)


def area(box):
    return max(F(0), box[2] - box[0]) * max(F(0), box[3] - box[1])


def iou(a, b):
    intersection = max(F(0), min(a[2], b[2]) - max(a[0], b[0])) * max(F(0), min(a[3], b[3]) - max(a[1], b[1]))
    union = area(a) + area(b) - intersection
    return intersection / union if union else F(0)


def nms(boxes, scores, threshold, classes=None, nonstrict=False):
    if len(boxes) != len(scores) or not F(0) <= threshold <= F(1):
        raise ValueError('NMS contract')
    if classes is not None and len(classes) != len(boxes):
        raise ValueError('NMS class length')
    if any(area(box) <= 0 for box in boxes):
        raise ValueError('nonpositive box area')
    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))
    keep, suppressed = [], set()
    for position, i in enumerate(order):
        if i in suppressed:
            continue
        keep.append(i)
        for j in order[position + 1:]:
            if j in suppressed or (classes is not None and classes[i] != classes[j]):
                continue
            overlap = iou(boxes[i], boxes[j])
            if (overlap >= threshold) if nonstrict else (overlap > threshold):
                suppressed.add(j)
    return keep


def clip(box, width, height):
    return tuple(max(F(0), min(value, F(width if i % 2 == 0 else height)))
                 for i, value in enumerate(box))


def test_nms():
    boxes = [(F(0), F(0), F(2), F(2)), (F(1), F(0), F(3), F(2))]
    scores = [F(9, 10), F(8, 10)]
    observation('iou_hand_reference', iou(*boxes), F(1, 3))
    observation('nms_strict_threshold', nms(boxes, scores, F(1, 3)), [0, 1])
    negative('nms_nonstrict_threshold', nms(boxes, scores, F(1, 3), nonstrict=True), [0, 1])
    raw = [(F(-2), F(0), F(2), F(2)), (F(0), F(0), F(4), F(2))]
    clipped = [clip(box, 3, 3) for box in raw]
    equal(iou(*raw), F(1, 3))
    equal(iou(*clipped), F(2, 3))
    observation('nms_clip_first', nms(clipped, scores, F(1, 2)), [0])
    negative('nms_before_clipping', nms(raw, scores, F(1, 2)), [0])
    same = [boxes[0], boxes[0]]
    observation('nms_tie_original_index', nms(same, [F(1), F(1)], F(1, 2)), [0])
    observation('nms_class_aware', nms(same, scores, F(1, 2), classes=[0, 1]), [0, 1])
    negative('nms_ignores_class_contract', nms(same, scores, F(1, 2)), [0, 1])
    observation('nms_empty', nms([], [], F(1, 2)), [])
    rejects('nms_zero_area', lambda: nms([(F(0), F(0), F(0), F(2))], [F(1)], F(1, 2)))


def test_cost():
    observation('C12C_MAC', 56 * 56 * 64 * 64 * 9, 115605504)
    observation('C12C_output_bytes', 56 * 56 * 64 * 2, 401408)
    observation('C12C_im2col_bytes', 56 * 56 * 64 * 9 * 2, 3612672)


def main():
    test_convolution()
    test_padding_contract()
    test_bn_residual()
    test_coordinates()
    test_nms()
    test_cost()
    print('SCOPE deterministic integer/Fraction software models; no image processing, target kernels, GPU measurements, performance or task-quality evidence')


if __name__ == '__main__':
    main()
