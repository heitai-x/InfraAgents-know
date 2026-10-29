from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import combinations
from math import exp, isclose, isfinite


class ModelError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare(actual, expected, label):
    require(actual == expected, label + ': actual=' + repr(actual) + ', expected=' + repr(expected))


def must_reject(fn, token):
    try:
        fn()
    except ModelError as exc:
        require(token in str(exc), 'unexpected rejection: ' + str(exc))
        return
    raise AssertionError('missing rejection: ' + token)


def detect_difference(label, actual, expected, detected):
    try:
        compare(actual, expected, label)
    except AssertionError:
        detected.append(label)
        return
    raise AssertionError('negative control escaped: ' + label)


def validate_bag(rows, ids, weights, mode):
    if mode not in ('SUM', 'MEAN_UNWEIGHTED', 'MEAN_WEIGHTED'):
        raise ModelError('unknown mode')
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ModelError('row shape')
    if len(ids) != len(weights):
        raise ModelError('length mismatch')
    for idx in ids:
        if type(idx) is not int or not 0 <= idx < len(rows):
            raise ModelError('row out of range')
    for weight in weights:
        if not isinstance(weight, (int, float, Fraction)):
            raise ModelError('weight type')
        if isinstance(weight, float) and not isfinite(weight):
            raise ModelError('nonfinite weight')
        if mode == 'MEAN_UNWEIGHTED' and weight != 1:
            raise ModelError('unit weights required')
        if mode == 'MEAN_WEIGHTED' and weight < 0:
            raise ModelError('negative weight')


def reference_bag(rows, ids, weights, mode):
    validate_bag(rows, ids, weights, mode)
    count = len(ids)
    mass = sum((Fraction(w) for w in weights), Fraction(0))
    # Independently enumerate every occurrence for every output channel.
    total = tuple(sum((Fraction(weights[k]) * rows[ids[k]][j]
                       for k in range(count)), Fraction(0))
                  for j in range(len(rows[0])))
    if mode == 'SUM':
        return total, count, mass
    if not count:
        raise ModelError('empty mean')
    denominator = Fraction(count) if mode == 'MEAN_UNWEIGHTED' else mass
    if denominator <= 0:
        raise ModelError('nonpositive mass')
    return tuple(v / denominator for v in total), count, mass


def aggregate_sharded(rows, ids, weights, mode, owner, completed, omit_vector_shard=None):
    validate_bag(rows, ids, weights, mode)
    grouped = {}
    for idx, weight in zip(ids, weights):
        shard = owner(idx)
        if shard is None:
            raise ModelError('no owner')
        entry = grouped.setdefault(shard, {}).setdefault(idx, [0, Fraction(0)])
        entry[0] += 1
        entry[1] += Fraction(weight)
    if not set(grouped).issubset(completed):
        raise ModelError('incomplete shard')
    partials = {}
    for shard, records in sorted(grouped.items()):
        vector = [Fraction(0)] * len(rows[0])
        count, mass = 0, Fraction(0)
        for idx, (mult, wsum) in sorted(records.items()):
            for j, value in enumerate(rows[idx]):
                vector[j] += wsum * value
            count += mult
            mass += wsum
        partials[shard] = tuple(vector), count, mass
    merged = [Fraction(0)] * len(rows[0])
    count, mass = 0, Fraction(0)
    for shard, (vector, local_count, local_mass) in sorted(partials.items()):
        # Fault injection executes the merge but drops a completed vector.
        # Accounting still includes that shard, so count invariants alone pass.
        if shard != omit_vector_shard:
            for j, value in enumerate(vector):
                merged[j] += value
        count += local_count
        mass += local_mass
    if count != len(ids) or mass != sum(map(Fraction, weights), Fraction(0)):
        raise ModelError('accounting mismatch')
    if mode == 'SUM':
        return tuple(merged), count, mass
    if not count:
        raise ModelError('empty mean')
    denominator = count if mode == 'MEAN_UNWEIGHTED' else mass
    if denominator <= 0:
        raise ModelError('nonpositive mass')
    return tuple(v / denominator for v in merged), count, mass


def bad_semantic_dedup(rows, ids, weights, mode):
    first = {}
    for idx, weight in zip(ids, weights):
        first.setdefault(idx, weight)
    return reference_bag(rows, list(first), list(first.values()), mode)


def bad_mean_of_means(rows, ids, owner):
    groups = {}
    for idx in ids:
        groups.setdefault(owner(idx), []).append(idx)
    means = [reference_bag(rows, group, [1] * len(group), 'MEAN_UNWEIGHTED')[0]
             for group in groups.values()]
    return tuple(sum(v[j] for v in means) / len(means) for j in range(len(rows[0])))


def reference_pairs(vectors):
    # Independent pair enumeration and whole dot expression.
    return tuple(sum((Fraction(a) * b for a, b in zip(vectors[i], vectors[j])), Fraction(0))
                 for i, j in combinations(range(len(vectors)), 2))


def interactions(vectors, omit_dense=False, zero_dense=False, reverse=False):
    if not vectors or not vectors[0] or any(len(v) != len(vectors[0]) for v in vectors):
        raise ModelError('interaction shape')
    pairs = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            if omit_dense and i == 0:
                continue
            pairs.append((i, j))
    if reverse:
        pairs.reverse()
    result = []
    for i, j in pairs:
        value = Fraction(0)
        for k in range(len(vectors[0])):
            if not (zero_dense and i == 0):
                value += vectors[i][k] * vectors[j][k]
        result.append(value)
    return tuple(result)


def mlp(x, layers, fault=None):
    if not x or not layers:
        raise ModelError('mlp shape')
    width = len(x)
    for weight, bias, activation in layers:
        if not weight or len(weight) != len(bias) or any(len(row) != width for row in weight):
            raise ModelError('mlp shape')
        if activation not in ('IDENTITY', 'RELU'):
            raise ModelError('activation')
        width = len(bias)
    a = tuple(map(Fraction, x))
    trace = []
    for layer_index, (weight, bias, activation) in enumerate(layers):
        q, next_a = [], []
        if fault == 'final_relu' and layer_index == len(layers) - 1:
            activation = 'RELU'
        for j in range(len(bias)):
            acc = Fraction(0)
            for i in range(len(a)):
                acc += weight[j][i] * a[i]
            acc += 0 if fault == 'skip_bias' else bias[j]
            q.append(acc)
            next_a.append(max(Fraction(0), acc) if activation == 'RELU' else acc)
        a = tuple(next_a)
        trace.append((tuple(q), a))
    return a, tuple(trace)


def reference_mlp(x, layers):
    # Bias is an extra coefficient in an augmented input, independent of
    # the candidate's bias epilogue and explicit inner accumulation loop.
    a, trace = tuple(map(Fraction, x)), []
    for weight, bias, activation in layers:
        augmented = a + (Fraction(1),)
        affine = tuple(sum((c * v for c, v in zip(tuple(row) + (b,), augmented)), Fraction(0))
                       for row, b in zip(weight, bias))
        a = tuple(max(Fraction(0), v) if activation == 'RELU' else v for v in affine)
        trace.append((affine, a))
    return a, tuple(trace)


def sigmoid(logit):
    value = float(logit)
    if not isfinite(value):
        raise ModelError('head nonfinite')
    if value >= 0:
        return 1.0 / (1.0 + exp(-value))
    e = exp(value)
    return e / (1.0 + e)


def reference_sigmoid(logit):
    with localcontext() as context:
        context.prec = 70
        value = Decimal(str(logit))
        return float(Decimal(1) / (Decimal(1) + (-value).exp()))


def placement_flow(rows, ids, weights, mode, preaggregate):
    validate_bag(rows, ids, weights, mode)
    d = len(rows[0])
    weighted = mode == 'MEAN_WEIGHTED' or any(w != 1 for w in weights)
    events = []

    def accumulate(items, item_weights, site):
        total = [Fraction(0)] * d
        for row, weight in zip(items, item_weights):
            for j in range(d):
                contribution = Fraction(row[j])
                if weighted:
                    contribution *= weight
                    events.append(site + '_multiply')
                total[j] += contribution
                events.append(site + '_add')
        return tuple(total)

    if preaggregate:
        source_sum = accumulate((rows[idx] for idx in ids), weights, 'source')
        wire = (source_sum,)
        total = wire[0]
    else:
        # This specific raw-row candidate materializes every occurrence.
        wire = tuple(tuple(rows[idx]) for idx in ids)
        total = accumulate(wire, weights, 'target')
    count = len(ids)
    mass = sum(map(Fraction, weights), Fraction(0))
    if mode != 'SUM':
        if not count:
            raise ModelError('empty mean')
        denominator = count if mode == 'MEAN_UNWEIGHTED' else mass
        if denominator <= 0:
            raise ModelError('nonpositive mass')
        normalized = []
        for value in total:
            normalized.append(value / denominator)
            events.append('target_divide')
        total = tuple(normalized)
    metrics = {name: events.count(name) for name in
               ('source_add', 'source_multiply', 'target_add', 'target_multiply', 'target_divide')}
    metrics['payload_elements'] = sum(len(row) for row in wire)
    return (tuple(total), count, mass), metrics


def split_csr(indices, offsets):
    if not offsets or any(type(x) is not int for x in offsets):
        raise ModelError('offset type')
    if offsets[0] != 0 or offsets[-1] != len(indices):
        raise ModelError('coverage')
    if any(a < 0 or a > b for a, b in zip(offsets, offsets[1:])):
        raise ModelError('monotonic')
    return tuple(tuple(indices[a:b]) for a, b in zip(offsets, offsets[1:]))


def same_artifact(model, table, dense):
    if model != table or model != dense:
        raise ModelError('artifact mismatch')
    return model


def main():
    detected = []
    rows = ((1, 2, 0), (3, 4, 1), (5, 6, 2), (7, 8, 3))
    owner = lambda idx: 0 if idx < 2 else 1
    ids = [0, 0, 1, 2, 3]
    completed = {0, 1}
    compare(tuple(sum(owner(idx) == shard for idx in ids) for shard in (0, 1)), (3, 2), 'shard counts')
    cases = (
        ('SUM', [1, 1, 1, 1, 1], (17, 22, 6)),
        ('SUM', [1, 2, 3, 4, 0], (32, 42, 11)),
        ('MEAN_UNWEIGHTED', [1, 1, 1, 1, 1], (Fraction(17, 5), Fraction(22, 5), Fraction(6, 5))),
        ('MEAN_WEIGHTED', [1, 2, 3, 4, 0], (Fraction(16, 5), Fraction(21, 5), Fraction(11, 10))),
    )
    observations = []
    for case_index, (mode, weights, anchor) in enumerate(cases):
        expected = reference_bag(rows, ids, weights, mode)
        compare(expected[0], anchor, 'hand anchor')
        actual = aggregate_sharded(rows, ids, weights, mode, owner, completed)
        compare(actual, expected, 'completed multishard ' + mode)
        bad = aggregate_sharded(rows, ids, weights, mode, owner, completed, omit_vector_shard=1)
        compare(bad[1:], expected[1:], 'fault preserves accounting')
        detect_difference('omitted_completed_vector_' + str(case_index), bad, expected, detected)
        observations.append((mode, actual))

    for mode in ('SUM', 'MEAN_UNWEIGHTED', 'MEAN_WEIGHTED'):
        weights = [1, 2, 3, 4, 0] if mode == 'MEAN_WEIGHTED' else [1] * len(ids)
        expected = reference_bag(rows, ids, weights, mode)
        source_result, source_cost = placement_flow(rows, ids, weights, mode, True)
        target_result, target_cost = placement_flow(rows, ids, weights, mode, False)
        compare(source_result, expected, 'source aggregation ' + mode)
        compare(target_result, expected, 'target aggregation ' + mode)
        compare((source_cost['source_add'], source_cost['target_add']), (15, 0), 'source additions')
        compare((target_cost['source_add'], target_cost['target_add']), (0, 15), 'target additions')
        compare((source_cost['payload_elements'], target_cost['payload_elements']), (3, 15), 'wire materialization')
        expected_multiply = 15 if mode == 'MEAN_WEIGHTED' else 0
        compare(source_cost['source_multiply'], expected_multiply, 'source multiplications')
        compare(target_cost['target_multiply'], expected_multiply, 'target multiplications')
        expected_divide = 0 if mode == 'SUM' else 3
        compare(source_cost['target_divide'], expected_divide, 'source-route normalization')
        compare(target_cost['target_divide'], expected_divide, 'raw-route normalization')
        faulty_cost = dict(target_cost)
        faulty_cost['target_add'] = 0
        detect_difference('raw_rows_do_not_remove_aggregation_' + mode, faulty_cost, target_cost, detected)

    compare(aggregate_sharded(rows, [], [], 'SUM', owner, set()), ((0, 0, 0), 0, 0), 'empty SUM')
    for mode in ('MEAN_UNWEIGHTED', 'MEAN_WEIGHTED'):
        must_reject(lambda mode=mode: aggregate_sharded(rows, [], [], mode, owner, set()), 'empty')
    must_reject(lambda: aggregate_sharded(rows, ids, [1] * 5, 'SUM', owner, {0}), 'incomplete')
    must_reject(lambda: aggregate_sharded(rows, [9], [1], 'SUM', owner, completed), 'range')
    must_reject(lambda: aggregate_sharded(rows, [-1], [1], 'SUM', owner, completed), 'range')
    must_reject(lambda: aggregate_sharded(rows, [0], [], 'SUM', owner, completed), 'length')
    must_reject(lambda: aggregate_sharded(rows, [0], [1], 'OTHER', owner, completed), 'mode')
    must_reject(lambda: aggregate_sharded(rows, [0], [float('nan')], 'SUM', owner, completed), 'nonfinite')
    must_reject(lambda: aggregate_sharded(rows, [0], [-1], 'MEAN_WEIGHTED', owner, completed), 'negative')
    must_reject(lambda: aggregate_sharded(rows, [0], [2], 'MEAN_UNWEIGHTED', owner, completed), 'unit')
    must_reject(lambda: aggregate_sharded(rows, [0, 2], [0, 0], 'MEAN_WEIGHTED', owner, completed), 'mass')
    must_reject(lambda: aggregate_sharded(rows, [0], [1], 'SUM', lambda idx: None, completed), 'owner')
    compare(aggregate_sharded(rows, [0, 2], [-1, 2], 'SUM', owner, completed),
            reference_bag(rows, [0, 2], [-1, 2], 'SUM'), 'signed SUM')
    for mode, weights in (('SUM', [1] * 5), ('MEAN_WEIGHTED', [1, 2, 3, 4, 0])):
        detect_difference('semantic_dedup_' + mode,
                          bad_semantic_dedup(rows, ids, weights, mode),
                          reference_bag(rows, ids, weights, mode), detected)
    detect_difference('mean_of_means', bad_mean_of_means(rows, ids, owner),
                      reference_bag(rows, ids, [1] * 5, 'MEAN_UNWEIGHTED')[0], detected)

    vectors = ((2, 1, 3), (1, 4, 2), (3, 2, 5), (2, 3, 7))
    expected_pairs = reference_pairs(vectors)
    compare(expected_pairs, (12, 23, 28, 21, 28, 47), 'pair anchors')
    compare(interactions(vectors), expected_pairs, 'pair implementation')
    detect_difference('omitted_dense_category_pairs', interactions(vectors, omit_dense=True), expected_pairs, detected)
    detect_difference('zeroed_dense_pairs_same_shape', interactions(vectors, zero_dense=True), expected_pairs, detected)
    detect_difference('reversed_pair_order', interactions(vectors, reverse=True), expected_pairs, detected)
    compare(interactions(((1, 2),)), (), 'no pairs')
    must_reject(lambda: interactions(((1, 2), (3,))), 'shape')

    bottom = (
        (((1, -2), (2, 1), (-1, 1)), (1, -3, -2), 'RELU'),
        (((1, -1, 2), (-2, 1, 1)), (1, -1), 'IDENTITY'),
    )
    top = (
        (((1, 2, -1, 1, 0), (-1, 0, 1, -1, 1)), (1, -2), 'RELU'),
        (((2, 1),), (-33,), 'IDENTITY'),
    )
    bottom_expected = reference_mlp((2, 1), bottom)
    bottom_actual = mlp((2, 1), bottom)
    compare(bottom_actual, bottom_expected, 'bottom every layer')
    compare(bottom_actual[0], (0, -1), 'bottom affine final')
    compare(bottom_actual[1][0], ((1, 2, -3), (1, 2, 0)), 'bottom hidden activation')
    detect_difference('missing_mlp_bias', mlp((2, 1), bottom, 'skip_bias'), bottom_expected, detected)
    detect_difference('incorrect_bottom_final_relu', mlp((2, 1), bottom, 'final_relu'), bottom_expected, detected)
    must_reject(lambda: mlp((2,), bottom), 'shape')
    must_reject(lambda: mlp((2, 1), ((((1, 2),), (), 'IDENTITY'),)), 'shape')

    tiny_rows = ((1, 2), (3, 4), (5, -1))
    bag1 = aggregate_sharded(tiny_rows, [0, 0, 2], [1, 1, 1], 'SUM', owner, completed)[0]
    bag2 = aggregate_sharded(tiny_rows, [1], [1], 'SUM', owner, completed)[0]
    z = bottom_actual[0]
    h = z + interactions((z, bag1, bag2))
    ref_z = bottom_expected[0]
    ref_bag1 = reference_bag(tiny_rows, [0, 0, 2], [1, 1, 1], 'SUM')[0]
    ref_bag2 = reference_bag(tiny_rows, [1], [1], 'SUM')[0]
    ref_h = ref_z + reference_pairs((ref_z, ref_bag1, ref_bag2))
    compare(h, ref_h, 'composed top input')
    compare(h, (0, -1, -3, -4, 33), 'top input anchor')
    top_actual = mlp(h, top)
    top_expected = reference_mlp(ref_h, top)
    compare(top_actual, top_expected, 'top every layer')
    compare(top_actual[0], (-1,), 'negative final logit')
    detect_difference('incorrect_top_final_relu', mlp(h, top, 'final_relu'), top_expected, detected)
    for logit in (-1000, -1, 0, 1, 1000):
        require(isclose(sigmoid(logit), reference_sigmoid(logit), rel_tol=1e-15, abs_tol=1e-15), 'head reference')
    probability = sigmoid(top_actual[0][0])
    require(isclose(probability, reference_sigmoid(-1), rel_tol=1e-15, abs_tol=1e-15), 'composed probability')
    require(not isclose(sigmoid(probability), reference_sigmoid(-1), rel_tol=1e-15, abs_tol=1e-15), 'double sigmoid escaped')
    detected.append('double_sigmoid')

    compare(split_csr((0, 0, 1, 2, 3), (0, 3, 3, 5)), ((0, 0, 1), (), (2, 3)), 'CSR empty bag')
    must_reject(lambda: split_csr((0, 0, 1, 2, 3, 4), (0, 3, 3, 5)), 'coverage')
    must_reject(lambda: split_csr((0, 1), (0, 2, 1, 2)), 'monotonic')
    bags = ((0, 0), (1, 2, 3))
    per_request = tuple(reference_bag(rows, bag, [1] * len(bag), 'MEAN_UNWEIGHTED')[0] for bag in bags)
    all_ids = [idx for bag in bags for idx in bag]
    global_mean = reference_bag(rows, all_ids, [1] * len(all_ids), 'MEAN_UNWEIGHTED')[0]
    detect_difference('cross_request_global_mean', (global_mean, global_mean), per_request, detected)
    compare(same_artifact('v1', 'v1', 'v1'), 'v1', 'artifact identity')
    must_reject(lambda: same_artifact('v1', 'v1', 'v2'), 'artifact')
    left = (1e16 + -1e16) + 1.0
    right = 1e16 + (-1e16 + 1.0)
    compare((left, right), (1.0, 0.0), 'binary64 nonassociativity')

    costs = (
        26 * 1000000 * 64 * 2,
        26 * 256 * 4 * 64 * 2,
        26 * 256 * 64 * 2,
        351 * 256 * 64,
        256 * (415 * 512 + 512 * 256 + 256),
        26 * 256 * 4 * 64,
        26 * 256 * 64,
        256 * 13 * 64 * 2,
    )
    compare(costs, (3328000000, 3407872, 851968, 5750784, 88014848, 1703936, 425984, 425984), 'C12-D costs')
    compare((27 * 26 // 2, 64 + 351), (351, 415), 'interaction dimensions')
    print('scope: deterministic CPU Fraction/Decimal/binary64 and logical operation/payload model')
    print('completed shards: {0,1}; occurrence counts: (3,2); ids: [0,0,1,2,3]')
    for mode, result in observations:
        print('bag observation:', mode, result)
    print('pair observation:', expected_pairs)
    print('composed forward: bottom=(0,-1), top_input=(0,-1,-3,-4,33), logit=-1, probability=', probability)
    print('placement observations: both routes aggregate; payload elements 3 versus 15 for this materialized model')
    print('negative controls detected:', len(detected), ', '.join(detected))
    print('C12-D integer ledger:', costs)
    print('checks completed; no torch/CUDA/GPU, device timing, production quality, or asynchronous protocol validation')


if __name__ == '__main__':
    main()
