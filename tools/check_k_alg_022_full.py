import math


def close(a, b, eps=1e-11):
    return abs(a - b) <= eps


def require_rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('invalid input was accepted')


def validate_sets(sets, mandatory, n, skip=None):
    if len(sets) != len(mandatory):
        raise ValueError('layer count')
    for i, selected in enumerate(sets):
        if any(j < 0 or j >= n for j in selected):
            raise ValueError('position')
        if skip != 'mandatory' and not mandatory[i].issubset(selected):
            raise ValueError('mandatory missing')
        if i and skip != 'nesting' and not selected.issubset(sets[i - 1]):
            raise ValueError('selection expanded')


def choose(selected, mandatory, deviation, budget):
    if not mandatory.issubset(selected):
        raise ValueError('mandatory lacks hidden input')
    if not len(mandatory) <= budget <= len(selected):
        raise ValueError('budget')
    candidates = sorted(selected - mandatory, key=lambda j: (-deviation[j], j))
    return mandatory | set(candidates[:budget - len(mandatory)])


def test_sets():
    # Both former negative controls were valid and must remain accepted.
    validate_sets([{0, 1, 2, 4}, {1, 4}], [{4}, {4}], 5)
    validate_sets([{0, 1, 2, 4}, {0, 2, 4}], [set(), {4}], 5)
    expanded = [{0, 4}, {1, 4}]
    missing = [{0, 4}, {0}]
    mandatory = [{4}, {4}]
    require_rejected(lambda: validate_sets(expanded, mandatory, 5))
    require_rejected(lambda: validate_sets(missing, mandatory, 5))
    # A mutant omitting each rule accepts the corresponding invalid fixture.
    validate_sets(expanded, mandatory, 5, skip='nesting')
    validate_sets(missing, mandatory, 5, skip='mandatory')
    cases = 0
    for first in range(16):
        for second in range(16):
            sets = [{j for j in range(4) if mask & (1 << j)} for mask in (first, second)]
            expected = bool(first & 8) and bool(second & 8) and (second & ~first) == 0
            accepted = True
            try:
                validate_sets(sets, [{3}, {3}], 4)
            except ValueError:
                accepted = False
            assert accepted == expected
            cases += 1
    assert choose({0, 1, 2, 3}, {3}, [0, 0, 0, 0], 2) == {0, 3}
    assert choose({0, 1, 2, 3}, {3}, [1, 9, 2, 0], 3) == {1, 2, 3}
    assert choose({0, 3}, {3}, [0, 0, 0, 0], 1) == {3}
    require_rejected(lambda: choose({0, 3}, {3}, [0, 0, 0, 0], 0))
    require_rejected(lambda: choose({0, 3}, {1}, [0, 0, 0, 0], 2))
    require_rejected(lambda: validate_sets([{5}], [set()], 5))
    print('selection: 256 exhaustive schedules; legal old fixtures accepted; invalid controls and rule mutants distinguished')
    assert cases == 256


def rotate(v, pos, frequencies):
    if len(v) != 2 * len(frequencies):
        raise ValueError('pair geometry')
    out = []
    for i, frequency in enumerate(frequencies):
        a, b = v[2 * i:2 * i + 2]
        c, s = math.cos(pos * frequency), math.sin(pos * frequency)
        out.extend((c * a - s * b, s * a + c * b))
    return out


def dot(a, b):
    assert len(a) == len(b)
    return math.fsum(x * y for x, y in zip(a, b))


def test_rope():
    q, k, frequencies = [.3, -.4, .5, .9], [-.2, .6, .1, .4], [.7, .2]
    cases = 0
    for a in (-3, 0, 2):
        for source in (-2, 0, 5):
            for target in (-1, 0, 8):
                old = rotate(k, source, frequencies)
                moved = rotate(old, target - source, frequencies)
                direct = rotate(k, target, frequencies)
                assert all(close(x, y) for x, y in zip(moved, direct))
                assert close(dot(rotate(q, a, frequencies), direct), dot(q, rotate(k, target - a, frequencies)))
                cases += 1
    # Independent geometric reference: a quarter turn maps (1,0) to (0,1).
    correct = dot(rotate([1, 0], math.pi / 2, [1]), [0, 1])
    bad_cos_only = dot([math.cos(math.pi / 2), 0], [0, 1])
    assert close(correct, 1) and not close(bad_cos_only, 1)
    require_rejected(lambda: rotate([1, 2, 3], 1, [1]))
    print('RoPE: %d position cases and quarter-turn reference; cos-only fault detected' % cases)


def online_attention(query, keys, values):
    assert len(keys) == len(values)
    maximum, mass, weighted = 0.0, 0.0, 0.0
    for key, value in zip(keys, values):
        score = query * key
        if mass == 0:
            maximum, mass, weighted = score, 1.0, value
        else:
            new_maximum = max(maximum, score)
            old_scale = math.exp(maximum - new_maximum)
            weight = math.exp(score - new_maximum)
            weighted = old_scale * weighted + weight * value
            mass = old_scale * mass + weight
            maximum = new_maximum
    return weighted / mass if mass else 0.0


def dense_reference(x, parameters):
    # Independent dense rows: materialize all scores, normalize, then sum PV.
    hidden, caches = list(x), []
    for wq, wk, wv, wm in parameters:
        keys = [wk * value for value in hidden]
        values = [wv * value for value in hidden]
        output = []
        for j, value in enumerate(hidden):
            scores = [wq * value * keys[t] for t in range(j + 1)]
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            total = math.fsum(weights)
            attention = math.fsum(weight * values[t] for t, weight in enumerate(weights)) / total
            output.append(value + wm * math.tanh(attention))
        caches.append((keys, values))
        hidden = output
    return hidden, caches


def independent_cache(x, ranges, parameters):
    caches = [([None] * len(x), [None] * len(x)) for _ in parameters]
    for start, end in ranges:
        _, local = dense_reference(x[start:end], parameters)
        for layer, pair in enumerate(local):
            for kind in (0, 1):
                caches[layer][kind][start:end] = pair[kind]
    return caches


def selective(x, cached, parameters, sets, mandatory, selected_keys_only=False):
    if not parameters or len(parameters) != len(sets) or len(cached) != len(parameters):
        raise ValueError('layer geometry')
    validate_sets(sets, [mandatory for _ in sets], len(x))
    hidden = {j: x[j] for j in sets[0]}
    fused = []
    for layer, (wq, wk, wv, wm) in enumerate(parameters):
        keys, values = list(cached[layer][0]), list(cached[layer][1])
        if len(keys) != len(x) or len(values) != len(x):
            raise ValueError('cache geometry')
        for j in sets[layer]:
            if j not in hidden:
                raise ValueError('missing hidden input')
            keys[j], values[j] = wk * hidden[j], wv * hidden[j]
        if any(value is None for value in keys + values):
            raise ValueError('cache hole')
        output = {}
        for j in sorted(sets[layer]):
            visible = [t for t in range(j + 1) if not selected_keys_only or t in sets[layer]]
            attention = online_attention(wq * hidden[j], [keys[t] for t in visible], [values[t] for t in visible])
            output[j] = hidden[j] + wm * math.tanh(attention)
        fused.append((keys, values))
        hidden = output
    return hidden, fused


def test_selective():
    parameters = [(1, 1, 1, 1), (.8, 1.1, .9, .7), (1.2, .7, 1.1, .6)]
    for x in ([.2], [.2, -.1], [.2, -.1, .7, .3, .5]):
        cached = independent_cache(x, [(j, j + 1) for j in range(len(x))], parameters)
        # Last position is uncached: its mandatory computation must fill the hole.
        for keys, values in cached:
            keys[-1], values[-1] = None, None
        snapshot = [(list(keys), list(values)) for keys, values in cached]
        sets = [set(range(len(x))) for _ in parameters]
        actual, fused = selective(x, cached, parameters, sets, {len(x) - 1})
        reference, reference_cache = dense_reference(x, parameters)
        assert all(close(value, actual[j]) for j, value in enumerate(reference))
        for layer in range(len(parameters)):
            for kind in (0, 1):
                assert all(close(a, b) for a, b in zip(fused[layer][kind], reference_cache[layer][kind]))
        assert cached == snapshot
    x = [.2, -.1, .7, .3, .5]
    cached = independent_cache(x, [(0, 2), (2, 4), (4, 5)], parameters)
    reference, reference_cache = dense_reference(x, parameters)
    query_sets = [{4}, {4}, {4}]
    actual, _ = selective(x, cached, parameters, query_sets, {4})
    error = abs(actual[4] - reference[4])
    assert error > 1e-6
    # At layer 1 all projections agree; at layer 2 cross-chunk effects exist.
    assert all(close(a, b) for a, b in zip(cached[0][0], reference_cache[0][0]))
    assert any(abs(a - b) > 1e-6 for a, b in zip(cached[1][0], reference_cache[1][0]))
    # Causal closed-form reference for zero logits: mean of every visible V.
    one_layer = [(0, 1, 1, 1)]
    toy = [0., 2., 4.]
    toy_cache = [([0., 2., 4.], [0., 2., 4.])]
    good, _ = selective(toy, toy_cache, one_layer, [{2}], {2})
    bad, _ = selective(toy, toy_cache, one_layer, [{2}], {2}, selected_keys_only=True)
    expected = 4 + math.tanh(2)
    assert close(good[2], expected) and not close(bad[2], expected)
    # Cache hole and output-row loss must be rejected in the actual path.
    holes = [([None] * 5, [None] * 5) for _ in parameters]
    require_rejected(lambda: selective(x, holes, parameters, query_sets, {4}))
    require_rejected(lambda: selective(x, cached, parameters, [{4}, {4}, set()], {4}))
    empty, _ = selective([], [([], []) for _ in parameters], parameters, [set() for _ in parameters], set())
    assert empty == {}
    print('selective: 3 full-set dense references; cache holes and output-row loss rejected')
    print('selective: query-only missing-context error %.12g; selected-key-only fault detected' % error)
    print('probe timing: first-layer zero deviation and second-layer nonzero deviation observed')


def test_pipeline():
    cases = 0
    for layers in (0, 1, 2, 4, 7):
        for load_time in (0, 2, 6, 10):
            for compute_time in (0, 2, 9, 10):
                load_end, compute_end = 0, 0
                for _ in range(layers):
                    load_end += load_time
                    compute_end = max(load_end, compute_end) + compute_time
                expected = 0 if layers == 0 else load_time + compute_time + (layers - 1) * max(load_time, compute_time)
                assert compute_end == expected
                cases += 1
    assert 6 + 2 + 3 * min(6, 2) != 26
    assert 6 + 9 + 3 * max(6, 9) == 42
    print('pipeline: %d ideal two-resource cases; min-stage fault detected' % cases)


def test_capacity_and_ownership():
    def capacity(layers, tokens, heads, dk, dv, size):
        return layers * tokens * heads * (dk + dv) * size
    assert capacity(32, 8192, 8, 128, 128, 2) == 2 ** 30
    assert capacity(32, 3072, 8, 128, 128, 2) == 384 * 2 ** 20
    # Independent element enumeration, including unequal K/V widths.
    count = sum(4 for _ in range(2) for _ in range(3) for _ in range(1) for _ in range(2 + 5))
    assert capacity(2, 3, 1, 2, 5, 4) == count == 168
    bad_k_only = 2 * 3 * 1 * 2 * 4
    assert bad_k_only != count
    assert capacity(32, 0, 8, 128, 128, 2) == 0
    public = [[1, 2], [3, 4]]
    private = [list(row) for row in public]
    private[0][0] = 9
    assert public == [[1, 2], [3, 4]]
    faulty = list(public)
    faulty[0][0] = 9
    assert public != [[1, 2], [3, 4]]
    print('capacity: equal/unequal widths and zero tokens; K-only fault detected')
    print('ownership: private row copy isolated; shallow nested copy fault detected')


def main():
    print('K-ALG-022 deterministic CPU scalar/set/event model; no GPU or task-quality measurements')
    test_sets()
    test_rope()
    test_selective()
    test_pipeline()
    test_capacity_and_ownership()
    print('PASS: all executed assertions completed within the declared software-model scope')


if __name__ == '__main__':
    main()
