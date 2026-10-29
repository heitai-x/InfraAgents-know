#!/usr/bin/env python3
"""K-MODEL-016: offline, deterministic, read-only stdlib mechanism checks.
Inputs are embedded below. No project/upstream imports, files, network or processes.
Model: scalar Fraction mean/self-concat, ReLU, normalization disabled.
"""
from collections import Counter
from fractions import Fraction as F
from itertools import permutations, product


def require(condition, detail):
    if not condition:
        raise AssertionError(detail)


def mean(values):
    return sum(values, F(0)) / len(values) if values else F(0)


def selected(graph, node, level, limits=None):
    row = tuple(graph.get(node, ()))
    limit = None if limits is None else limits.get(level)
    if limit is not None:
        if not isinstance(limit, int) or limit < 0:
            raise ValueError('fanout must be a nonnegative integer')
        row = row[:limit]
    return row


def build(graph, targets, depth, limits=None, omit_self=False):
    if not isinstance(depth, int) or depth < 0:
        raise ValueError('depth must be nonnegative integer')
    layers = [None] * (depth + 1)
    layers[depth] = tuple(sorted(set(targets)))
    edges = {}
    for level in range(depth, 0, -1):
        previous = set() if omit_self else set(layers[level])
        for node in layers[level]:
            row = selected(graph, node, level, limits)
            edges[level, node] = row
            previous.update(row)
        layers[level - 1] = tuple(sorted(previous))
    return layers, edges


def path_nodes(graph, node, level, stop, limits=None):
    # Independent recursive path enumeration, no deduplicated layer construction.
    if level == stop:
        return {node}
    result = path_nodes(graph, node, level - 1, stop, limits)
    for neighbor in selected(graph, node, level, limits):
        result |= path_nodes(graph, neighbor, level - 1, stop, limits)
    return result


def closure_reference(graph, targets, depth, stop, limits=None):
    result = set()
    for node in targets:
        result |= path_nodes(graph, node, depth, stop, limits)
    return result


def tree_value(graph, features, node, level, limits=None, ws=F(1), wn=F(1)):
    # Direct expansion of the mathematical computation tree; no plan or buffers.
    if level == 0:
        return F(features[node])
    neighbors = selected(graph, node, level, limits)
    own = tree_value(graph, features, node, level - 1, limits, ws, wn)
    if neighbors:
        contribution = sum((tree_value(graph, features, u, level - 1,
                                       limits, ws, wn) for u in neighbors), F(0))
        contribution /= len(neighbors)
    else:
        contribution = F(0)
    return max(F(0), ws * own + wn * contribution)


def execute(features, targets, depth, plan, ws=F(1), wn=F(1),
            dedup=False, early_relu=False):
    layers, edges = plan
    previous = {v: F(features[v]) for v in layers[0]}
    trace = [dict(previous)]
    for level in range(1, depth + 1):
        current = {}
        for node in layers[level]:
            row = edges[level, node]
            if dedup:
                row = tuple(dict.fromkeys(row))
            values = [previous[u] for u in row]
            if early_relu:
                values = [max(F(0), x) for x in values]
            value = ws * previous[node] + wn * mean(values)
            current[node] = max(F(0), value)
        previous = current
        trace.append(dict(previous))
    return tuple(previous[v] for v in targets), trace


def check_model(graph, features, targets, depth, limits=None, ws=F(1), wn=F(1)):
    plan = build(graph, targets, depth, limits)
    for level, layer in enumerate(plan[0]):
        expected = closure_reference(graph, targets, depth, level, limits)
        require(set(layer) == expected, ('closure', level, layer, expected))
    output, trace = execute(features, targets, depth, plan, ws, wn)
    expected = tuple(tree_value(graph, features, v, depth, limits, ws, wn)
                     for v in targets)
    require(output == expected, ('forward', output, expected))
    for level, values in enumerate(trace):
        for node, value in values.items():
            reference = tree_value(graph, features, node, level, limits, ws, wn)
            require(value == reference, ('intermediate', level, node))
    return plan, output, trace


class Draws:
    def __init__(self, values):
        self.values = iter(values)

    def draw(self, low, high):
        value = next(self.values)
        require(low <= value <= high, ('draw range', low, high, value))
        return value


def sample_occurrences(row, mode, count, draws):
    if not isinstance(count, int) or count < 0:
        raise ValueError('negative/noninteger fanout')
    if mode == 'full':
        return tuple(row)
    if mode not in ('without', 'with'):
        raise ValueError('unknown mode')
    size = len(row)
    if size == 0 or count == 0:
        return ()
    if mode == 'with':
        return tuple(row[draws.draw(0, size - 1)] for _ in range(count))
    if count >= size:
        return tuple(row)
    swaps = {}
    output = []
    for index in range(count):
        chosen = draws.draw(index, size - 1)
        left = swaps.get(index, index)
        right = swaps.get(chosen, chosen)
        swaps[index] = right
        swaps[chosen] = left
        output.append(row[right])
    return tuple(output)


def invalidation(plan, changed):
    layers, edges = plan
    affected = [set(changed)]
    for level in range(1, len(layers)):
        current = set()
        for node in layers[level]:
            sources = {node} | set(edges[level, node])
            if sources & affected[-1]:
                current.add(node)
        affected.append(current)
    return affected


def touches_feature(graph, node, level, changed, limits=None):
    # Independent existence of a path from one output to a changed leaf.
    if level == 0:
        return node in changed
    if touches_feature(graph, node, level - 1, changed, limits):
        return True
    return any(touches_feature(graph, u, level - 1, changed, limits)
               for u in selected(graph, node, level, limits))


def rejected_variant(label, observed, expected, detected):
    require(observed != expected, ('negative control escaped', label))
    detected.append(label)


def main():
    graph = {0: (1, 2), 1: (3, 3, 4), 2: (4, 5),
             3: (), 4: (), 5: (), 6: ()}
    features = {0: 10, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    targets = (0,)
    plan, output, trace = check_model(graph, features, targets, 2)
    require(output == (F(203, 12),), 'hard-coded two-layer result')
    require(tuple(trace[1][v] for v in (0, 1, 2)) ==
            (F(23, 2), F(13, 3), F(13, 2)), 'hand intermediate values')
    print('full_target=' + str(output[0]))
    limited, sampled_output, _ = check_model(graph, features, targets, 2, {2: 1})
    require(sampled_output == (F(95, 6),), 'hard-coded sampled result')
    require(5 not in limited[0][0], 'unused sampled dependency')
    print('sampled_layer2_target=' + str(sampled_output[0]))
    detected = []

    bad, _ = execute(features, targets, 2, plan, dedup=True)
    require(bad == (F(17),), 'dedup concrete value')
    rejected_variant('parallel_edge_dedup', bad, output, detected)
    incomplete = build(graph, targets, 2, omit_self=True)
    rejected_variant('missing_self_closure', set(incomplete[0][0]),
                     closure_reference(graph, targets, 2, 0), detected)

    reversed_graph = {v: [] for v in graph}
    for node, row in graph.items():
        for neighbor in row:
            reversed_graph[neighbor].append(node)
    reversed_plan = build(reversed_graph, targets, 2)
    wrong_direction, _ = execute(features, targets, 2, reversed_plan)
    rejected_variant('reversed_direction', wrong_direction, output, detected)

    signed_graph = {0: (1, 2), 1: (), 2: ()}
    signed_features = {0: 0, 1: -2, 2: 2}
    signed_plan, correct, _ = check_model(signed_graph, signed_features, (0,), 1)
    wrong_activation, _ = execute(signed_features, (0,), 1, signed_plan,
                                  early_relu=True)
    require(correct == (F(0),) and wrong_activation == (F(1),), 'nonlinear example')
    rejected_variant('relu_before_mean', wrong_activation, correct, detected)

    for changes in ({5}, {6}, {0, 4}, set()):
        affected = invalidation(plan, changes)
        for level in range(1, 3):
            reference = {v for v in plan[0][level]
                         if touches_feature(graph, v, level, changes)}
            require(affected[level] == reference, ('invalidation', level, changes))
    affected = invalidation(plan, {5})
    require(2 in affected[1] and 0 in affected[2], 'reverse consumer chain')
    bad_final = {v for v in plan[0][2]
                 if ({v} | set(plan[1][2, v])) & {5}}
    rejected_variant('one_hop_invalidation', bad_final, affected[2], detected)
    inside = dict(features)
    inside[5] = 100
    _, changed_output, _ = check_model(graph, inside, targets, 2)
    require(changed_output != output, 'depth-two feature actually changes output')
    outside = dict(features)
    outside[6] = 999
    _, outside_output, _ = check_model(graph, outside, targets, 2)
    require(outside_output == output, 'outside feature independence')

    # Exhaust all 3-node directed simple graphs, including self-loops.
    # The explicit graph above additionally supplies parallel edges.
    cases = 0
    for mask in range(1 << 9):
        small = {v: tuple(u for u in range(3) if mask & (1 << (3 * v + u)))
                 for v in range(3)}
        for depth in range(3):
            for batch in ((), (0,), (0, 0, 2)):
                check_model(small, {0: -2, 1: 0, 2: 3}, batch, depth,
                            ws=F(-1, 2), wn=F(3, 2))
                cases += 1
    _, zero_fanout, _ = check_model(graph, features, (0,), 2, {1: 0, 2: 0})
    require(zero_fanout == (F(10),), 'zero fanout keeps self')
    _, below_fanout, _ = check_model(graph, features, (0,), 2, {1: 100, 2: 100})
    require(below_fanout == output, 'insufficient neighbors take all')
    _, isolated, _ = check_model(graph, features, (6,), 2)
    require(isolated == (F(6),), 'empty-neighbor zero contract')
    _, duplicated, _ = check_model(graph, features, (0, 0), 2)
    require(duplicated == output * 2, 'target restoration')
    print('enumerated_scalar_graph_cases=' + str(cases))

    for size in range(1, 5):
        for count in range(size):
            frequencies = Counter()
            ranges = [range(index, size) for index in range(count)]
            for choices in product(*ranges):
                sample = sample_occurrences(tuple(range(size)), 'without', count,
                                            Draws(choices))
                frequencies[sample] += 1
            expected = Counter({p: 1 for p in permutations(range(size), count)})
            require(frequencies == expected, ('uniform ordered samples', size, count))
    with_replacement = Counter()
    for choices in product(range(2), repeat=3):
        sample = sample_occurrences((0, 1), 'with', 3, Draws(choices))
        with_replacement[sample] += 1
    require(with_replacement == Counter({p: 1 for p in product(range(2), repeat=3)}),
            'with-replacement distribution')
    require(sample_occurrences((), 'with', 5, Draws(())) == (), 'empty with replacement')
    require(sample_occurrences((0, 1), 'without', 5, Draws(())) == (0, 1), 'take all')
    require(sample_occurrences((0, 1), 'full', 0, Draws(())) == (0, 1), 'full ignores limit')
    parallel_positions = sample_occurrences((0, 1, 2), 'without', 2, Draws((0, 1)))
    ids = (3, 3, 4)
    require(tuple(ids[i] for i in parallel_positions) == (3, 3),
            'unique occurrences may share node identity')
    require(sample_occurrences((0, 1), 'with', 3, Draws((0, 0, 0))) == (0, 0, 0),
            'repeated occurrence legal with replacement')
    print('small_sampler_ordered_frequencies=uniform_under_uniform_draws')

    values = (F(3), F(3), F(4))
    require(all(mean(p) == F(10, 3) for p in permutations(values)), 'exact mean permutation')
    require(mean((F(4), F(8), F(0), F(0))) == F(3), 'padding denominator counterexample')
    self_value, neighbors = F(3), (F(-2), F(4), F(7))
    a, b = F(2, 3), F(-5, 4)
    direct = a * self_value + b * mean(neighbors)
    projected = a * self_value + mean(tuple(b * x for x in neighbors))
    require(direct == projected, 'linear projection distribution')

    n2, s2, s1, width, byte = 64, 10, 5, 128, 4
    n1, n0 = n2 * (1 + s2), n2 * (1 + s2) * (1 + s1)
    m2, m1 = n2 * s2, n1 * s1
    cost = (n1, n0, m1 + m2, n0 * width * byte,
            m1 * width * byte, m2 * width * byte,
            2 * (n1 + n2) * (width + width) * width,
            (n0 + n1) * width * byte, (n1 + n2) * width * byte,
            n1 * width * byte, (m1 + m2 + n1 + 1 + n2 + 1) * 4)
    expected_cost = (704, 4224, 4160, 2162688, 1802240, 327680,
                     50331648, 2523136, 393216, 360448, 19720)
    require(cost == expected_cost, ('C12-E independent numeric ledger', cost))
    print('C12_E_n1_n0_edges_unique_bytes=' + str(cost[:4]))
    print('negative_controls_detected=' + ','.join(detected))
    print('scope=host_fraction_scalar_graph_finite_sampler_analytic_cost')
    print('normalization_lstm_device_quality_measurements=false')


if __name__ == '__main__':
    main()
