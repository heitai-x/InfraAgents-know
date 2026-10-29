import copy
import math

D = 4
LAYERS = 2
VOCAB = 6
FREQUENCIES = (0.9, 0.37)
EMBEDDINGS = (
    (0.8, -0.4, 0.6, 0.2),
    (-0.5, 0.9, -0.3, 0.7),
    (0.4, 0.6, -0.8, -0.2),
    (-0.9, -0.2, 0.3, 0.8),
    (0.2, -1.0, 0.1, 0.5),
    (0.6, 0.3, -0.7, -0.8),
)
TOL = 1e-11


class ContractError(Exception):
    pass


class InjectedFailure(Exception):
    pass


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def difference(a, b):
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
            return float('inf')
        if len(a) != len(b):
            return float('inf')
        return max([0.0] + [difference(x, y) for x, y in zip(a, b)])
    if a is None or b is None:
        return 0.0 if a is b else float('inf')
    return abs(a - b)


def close(a, b, label):
    error = difference(a, b)
    require(math.isfinite(error) and error <= TOL, label + ': ' + repr(error))
    return error


def matrix(seed, amplitude=0.45, output_width=D):
    return [[amplitude * math.sin(seed + 1.1 * i + 0.7 * j)
             for j in range(D)] for i in range(output_width)]


class Model:
    def __init__(self):
        self.q = [matrix(1.0), matrix(1.7)]
        self.k = [matrix(2.0), matrix(2.6)]
        self.v = [matrix(3.0), matrix(3.5)]
        self.o = [matrix(4.0), matrix(4.4)]
        self.f = [matrix(5.0, 0.25), matrix(5.4, 0.25)]
        self.head = matrix(7.0, 0.4, VOCAB)


def matvec(w, x):
    return [sum(a * b for a, b in zip(row, x)) for row in w]


def reference_matvec(w, x):
    return [math.fsum(row[j] * x[j] for j in range(len(x))) for row in w]


def norm(x):
    denominator = math.sqrt(math.fsum(v * v for v in x) / D + 0.05)
    return [v / denominator for v in x]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def rope(x, position):
    out = []
    for pair, frequency in enumerate(FREQUENCIES):
        a, b = x[2 * pair:2 * pair + 2]
        c, s = math.cos(position * frequency), math.sin(position * frequency)
        out.extend((a * c - b * s, a * s + b * c))
    return out


def relative_dot_reference(q, k, delta):
    terms = []
    for pair, frequency in enumerate(FREQUENCIES):
        a, b = q[2 * pair:2 * pair + 2]
        c, d = k[2 * pair:2 * pair + 2]
        angle = delta * frequency
        terms.append((a * c + b * d) * math.cos(angle)
                     + (b * c - a * d) * math.sin(angle))
    return math.fsum(terms)


def position_embedding(p):
    return [0.31 * math.sin(0.71 * p + 0.2),
            0.23 * math.cos(0.53 * p + 0.7),
            0.19 * math.sin(0.37 * p + 1.1),
            0.17 * math.cos(0.29 * p + 1.7)]


def softmax(scores):
    require(bool(scores), 'test model has an empty attention domain')
    maximum = max(scores)
    masses = [math.exp(v - maximum) for v in scores]
    total = sum(masses)
    return [v / total for v in masses]


def finish_layer(model, hidden, attention, layer, multiply=matvec):
    projected = multiply(model.o[layer], attention)
    residual = [a + b for a, b in zip(hidden, projected)]
    feed = multiply(model.f[layer], norm(residual))
    return [residual[i] + 0.2 * math.tanh(feed[i]) for i in range(D)]


def full_reference(model, tokens, mode='rope', window=None):
    # No cache, no append/commit path, and no explicit Q/K rotation helper.
    hidden = []
    for p, token in enumerate(tokens):
        x = list(EMBEDDINGS[token])
        if mode == 'additive':
            x = [a + b for a, b in zip(x, position_embedding(p))]
        hidden.append(x)
    for layer in range(LAYERS):
        inputs = [norm(x) for x in hidden]
        queries = [reference_matvec(model.q[layer], x) for x in inputs]
        keys = [reference_matvec(model.k[layer], x) for x in inputs]
        values = [reference_matvec(model.v[layer], x) for x in inputs]
        next_hidden = []
        for p in range(len(tokens)):
            first = 0 if window is None else max(0, p - window + 1)
            visible = list(range(first, p + 1))
            scores = []
            for j in visible:
                if mode == 'rope':
                    score = relative_dot_reference(queries[p], keys[j], j - p)
                else:
                    score = math.fsum(queries[p][z] * keys[j][z] for z in range(D))
                score /= math.sqrt(D)
                if mode == 'bias':
                    score += 0.12 * (j - p)
                scores.append(score)
            maximum = max(scores)
            masses = [math.exp(v - maximum) for v in scores]
            total = math.fsum(masses)
            attention = [math.fsum(masses[i] * values[j][z]
                                   for i, j in enumerate(visible)) / total
                         for z in range(D)]
            next_hidden.append(finish_layer(model, hidden[p], attention, layer,
                                            reference_matvec))
        hidden = next_hidden
    return [reference_matvec(model.head, norm(x)) for x in hidden]


def new_state(mode='rope', storage='rot', window=None, requests=('A',)):
    require(mode in ('rope', 'additive', 'bias'), 'mode')
    require(storage in ('raw', 'rot'), 'storage')
    require(window is None or window >= 1, 'window')
    return {'mode': mode, 'storage': storage, 'window': window,
            'root': {'version': 0,
                     'requests': {r: {'n': 0, 'kv': [[] for _ in range(LAYERS)]}
                                  for r in requests}}}


def make_rows(tokens, start=0, request='A', want=True):
    return [{'request': request, 'token': token, 'pos': start + i,
             'valid': True, 'want': want} for i, token in enumerate(tokens)]


def consume(model, state, rows, fail=None, fault=None):
    root = state['root']
    valid_indices = [i for i, row in enumerate(rows) if row.get('valid', True)]
    result = [None] * len(rows)
    if not valid_indices:
        return result
    counts = {}
    ordinal = {}
    for i in valid_indices:
        row = rows[i]
        rid = row['request']
        if rid not in root['requests']:
            raise ContractError('unknown request')
        j = counts.get(rid, 0)
        if row['pos'] != root['requests'][rid]['n'] + j:
            raise ContractError('noncontiguous logical position')
        if not 0 <= row['token'] < VOCAB:
            raise ContractError('token')
        ordinal[i] = j
        counts[rid] = j + 1
    if fail == 'reserve':
        raise InjectedFailure('reservation')
    # This small CPU model deliberately deep-copies private state. It is not
    # a performance model for a persistent directory or device allocation.
    candidate = copy.deepcopy(root['requests'])
    hidden = {}
    for i in valid_indices:
        row = rows[i]
        x = list(EMBEDDINGS[row['token']])
        if state['mode'] == 'additive':
            x = [a + b for a, b in zip(x, position_embedding(row['pos']))]
        hidden[i] = x
    for layer in range(LAYERS):
        if fail == 'layer' + str(layer):
            raise InjectedFailure('layer ' + str(layer))
        staged = {r: [] for r in counts}
        queries = {}
        for i in valid_indices:
            row = rows[i]
            p = row['pos']
            transform_position = ordinal[i] if fault == 'local' else p
            u = norm(hidden[i])
            q = matvec(model.q[layer], u)
            k = matvec(model.k[layer], u)
            v = matvec(model.v[layer], u)
            if state['mode'] == 'rope' and fault != 'missing':
                q = rope(q, transform_position)
                if state['storage'] == 'rot':
                    k = rope(k, transform_position)
            queries[i] = q
            staged[row['request']].append((p, k, v))
        next_hidden = {}
        for i in valid_indices:
            row = rows[i]
            rid, p = row['request'], row['pos']
            entries = root['requests'][rid]['kv'][layer] + staged[rid]
            visible = [e for e in entries
                       if e[0] <= p and (state['window'] is None
                                        or p - e[0] < state['window'])]
            scores = []
            for ep, key, value in visible:
                key_use = key
                if state['mode'] == 'rope' and state['storage'] == 'raw':
                    key_use = rope(key_use, ep)
                if state['mode'] == 'rope' and fault == 'double':
                    key_use = rope(key_use, ep)
                score = dot(queries[i], key_use) / math.sqrt(D)
                if state['mode'] == 'bias':
                    score += 0.12 * (ep - p)
                scores.append(score)
            weights = softmax(scores)
            attention = [sum(weights[j] * entry[2][z]
                             for j, entry in enumerate(visible)) for z in range(D)]
            next_hidden[i] = finish_layer(model, hidden[i], attention, layer)
        hidden = next_hidden
        for rid in counts:
            new_n = root['requests'][rid]['n'] + counts[rid]
            entries = root['requests'][rid]['kv'][layer] + staged[rid]
            if state['window'] is not None:
                entries = [e for e in entries if e[0] >= new_n - state['window']]
            candidate[rid]['kv'][layer] = entries
            if fault == 'early':
                # Intentionally wrong: publish a layer and retire old entries.
                root['requests'][rid]['kv'][layer] = entries
    if fail == 'logits':
        raise InjectedFailure('logits')
    for i in valid_indices:
        if rows[i].get('want', True):
            result[i] = matvec(model.head, norm(hidden[i]))
    for rid, count in counts.items():
        candidate[rid]['n'] = root['requests'][rid]['n'] + count
    state['root'] = {'version': root['version'] + 1, 'requests': candidate}
    return result


def recover(model, state, token, request='A', faulty_append=False):
    saved = state['root']['requests'][request]
    p = saved['n'] - 1
    if p < 0:
        raise ContractError('empty prefix recovery')
    first = 0 if state['window'] is None else max(0, p - state['window'] + 1)
    expected_positions = list(range(first, p + 1))
    hidden = list(EMBEDDINGS[token])
    if state['mode'] == 'additive':
        hidden = [a + b for a, b in zip(hidden, position_embedding(p))]
    for layer in range(LAYERS):
        visible = [e for e in saved['kv'][layer] if first <= e[0] <= p]
        if [e[0] for e in visible] != expected_positions:
            raise ContractError('recovery dependency missing or duplicated')
        q = matvec(model.q[layer], norm(hidden))
        if state['mode'] == 'rope':
            q = rope(q, p)
        scores = []
        for ep, key, value in visible:
            if state['mode'] == 'rope' and state['storage'] == 'raw':
                key = rope(key, ep)
            score = dot(q, key) / math.sqrt(D)
            if state['mode'] == 'bias':
                score += 0.12 * (ep - p)
            scores.append(score)
        weights = softmax(scores)
        attention = [sum(weights[j] * entry[2][z]
                         for j, entry in enumerate(visible)) for z in range(D)]
        hidden = finish_layer(model, hidden, attention, layer)
    result = matvec(model.head, norm(hidden))
    if faulty_append:
        for layer in range(LAYERS):
            saved['kv'][layer].append(copy.deepcopy(saved['kv'][layer][-1]))
    return result


def require_unchanged(before, after):
    require(before == after, 'committed state changed')


def expect_error(error_type, function, label):
    try:
        function()
    except error_type:
        return
    raise AssertionError(label + ': expected ' + error_type.__name__)


def check_cache(state):
    for saved in state['root']['requests'].values():
        n = saved['n']
        first = 0 if state['window'] is None else max(0, n - state['window'])
        expected = list(range(first, n))
        for entries in saved['kv']:
            require([e[0] for e in entries] == expected,
                    'layer positions/uniqueness/window do not match committed length')


def analytic_position_control():
    # Four legal positions 0,1,2,3; keys at 1 and 2 are zero.
    # Only position 0 has a nonzero scalar value. New q/k at position 3
    # are (1,0,0,0); the deliberately wrong path resets only new q/k to 0.
    one = [1.0, 0.0, 0.0, 0.0]
    zero = [0.0] * D
    old_keys = [rope(one, 0), zero, zero]
    good_q, good_current = rope(one, 3), rope(one, 3)
    bad_q, bad_current = rope(one, 0), rope(one, 0)
    good_scores = [dot(good_q, key) / 2 for key in old_keys + [good_current]]
    bad_scores = [dot(bad_q, key) / 2 for key in old_keys + [bad_current]]
    good = softmax(good_scores)[0]
    bad = softmax(bad_scores)[0]
    mass = math.exp(math.cos(3 * FREQUENCIES[0]) / 2)
    independent = mass / (mass + 2 + math.exp(0.5))
    close(good, independent, 'closed-form position witness')
    require(abs(good - bad) > 0.1, 'analytic local-reset fault not detected')
    q = [0.8, -0.3, 0.4, 0.6]
    k = [-0.7, 0.2, 0.9, -0.5]
    expected = relative_dot_reference(q, k, 3)
    close(dot(rope(q, 5), rope(k, 8)), expected, 'relative rotation identity')
    close(dot(rope(q, 18), rope(k, 21)), expected, 'common shift invariant')
    print('analytic_local_reset correct=%.9f faulty=%.9f gap=%.9f detected'
          % (good, bad, abs(good - bad)))
    print('standard_rope_common_shift invariant checked separately')


def main():
    print('K-MODEL-001; stdlib only; deterministic CPU arithmetic')
    print('Scope: two layers, one head, D=4, toy RMS/FFN; no GPU or checkpoint measurements')
    model = Model()
    sequence = [0, 1, 2, 3, 4, 5, 2, 0]
    worst = 0.0
    cases = 0
    for mode in ('rope', 'additive', 'bias'):
        for storage in (('raw', 'rot') if mode == 'rope' else ('raw',)):
            for window in (None, 1, 2, 3):
                state = new_state(mode, storage, window)
                actual = []
                for start, end in ((0, 2), (2, 5), (5, 8)):
                    actual.extend(consume(model, state,
                                          make_rows(sequence[start:end], start)))
                    check_cache(state)
                expected = full_reference(model, sequence, mode, window)
                worst = max(worst, close(actual, expected, 'chunk/full'))
                require(state['root']['requests']['A']['n'] == len(sequence), 'length')
                sequential = new_state(mode, storage, window)
                by_one = []
                for p, token in enumerate(sequence):
                    by_one.extend(consume(model, sequential, make_rows([token], p)))
                worst = max(worst, close(by_one, expected, 'one-row/full'))
                close(actual[:2], full_reference(model, sequence[:2], mode, window),
                      'empty-state two-row input')
                cases += 1
    print('full_vs_chunks_and_steps cases=%d max_abs=%.3e' % (cases, worst))

    state = new_state()
    before = copy.deepcopy(state)
    require(consume(model, state, []) == [], 'empty rows')
    require(consume(model, state, [{'valid': False}]) == [None], 'padding-only rows')
    require_unchanged(before, state)
    for positions in ((0, 0), (0, 2), (1, 0)):
        bad = make_rows([0, 1])
        for row, p in zip(bad, positions):
            row['pos'] = p
        expect_error(ContractError, lambda: consume(model, state, bad), 'invalid sequence')
        require_unchanged(before, state)
    expect_error(ContractError, lambda: recover(model, state, 0), 'empty recovery')
    print('empty_padding_duplicate_gap_reversal rejected without mutation')

    base = new_state(requests=('A', 'B'))
    prefix_a, prefix_b = [0, 1], [3, 4, 5]
    consume(model, base, make_rows(prefix_a, request='A')
            + make_rows(prefix_b, request='B'))
    a_rows = make_rows([2, 3], 2, 'A')
    b_rows = make_rows([0, 1], 3, 'B')
    orders = ([b_rows[0], a_rows[0], b_rows[1], {'valid': False}, a_rows[1]],
              [a_rows[0], b_rows[0], a_rows[1], b_rows[1], {'valid': False}])
    expected_a = full_reference(model, prefix_a + [2, 3])
    expected_b = full_reference(model, prefix_b + [0, 1])
    final_states = []
    for order in orders:
        state = copy.deepcopy(base)
        result = consume(model, state, order)
        for row, output in zip(order, result):
            if not row.get('valid', True):
                require(output is None, 'padding output')
                continue
            reference = expected_a if row['request'] == 'A' else expected_b
            close(output, reference[row['pos']], 'interleaved request output')
        check_cache(state)
        final_states.append(state)
    require(final_states[0] == final_states[1], 'cross-request order changed state')
    print('interleaved_requests unequal_history and cross-request permutation checked')

    expected = full_reference(model, sequence[:5])[2:]
    for fault in ('local', 'missing', 'double'):
        state = new_state()
        consume(model, state, make_rows(sequence[:2]))
        faulty = consume(model, state, make_rows(sequence[2:5], 2), fault=fault)
        error = difference(faulty, expected)
        require(math.isfinite(error) and error > 1e-6,
                'negative position control not detected: ' + fault)
        print('negative_%s max_abs=%.6e detected' % (fault, error))
    analytic_position_control()

    def naive_position_validator():
        frozen = 2
        for row in make_rows([2, 3], 2):
            if row['pos'] != frozen:
                raise ContractError('faulty validator rejects second valid row')
    expect_error(ContractError, naive_position_validator, 'old entry-length contract')
    print('negative_old_position_contract rejects valid second row detected')

    initial = new_state(window=3)
    consume(model, initial, make_rows(sequence[:3], want=False))
    for failure in ('reserve', 'layer1', 'logits'):
        state = copy.deepcopy(initial)
        before = copy.deepcopy(state)
        expect_error(InjectedFailure,
                     lambda: consume(model, state, make_rows(sequence[3:5], 3),
                                     fail=failure), failure)
        require_unchanged(before, state)
        check_cache(state)
    faulty = copy.deepcopy(initial)
    before = copy.deepcopy(faulty)
    expect_error(InjectedFailure,
                 lambda: consume(model, faulty, make_rows(sequence[3:5], 3),
                                 fail='layer1', fault='early'), 'early publication')
    expect_error(AssertionError, lambda: require_unchanged(before, faulty),
                 'early publication invariant')
    require(faulty['root']['requests']['A']['n'] == 3, 'fault length should remain old')
    require([e[0] for e in faulty['root']['requests']['A']['kv'][0]] == [2, 3, 4],
            'fault did not exercise old-window retirement')
    state = copy.deepcopy(initial)
    peer = {'mode': state['mode'], 'storage': state['storage'],
            'window': state['window'], 'root': state['root']}
    peer_snapshot = copy.deepcopy(peer)
    output = consume(model, state, make_rows(sequence[3:5], 3))
    close(output, full_reference(model, sequence[:5], window=3)[3:], 'retry after failure')
    check_cache(state)
    require_unchanged(peer_snapshot, peer)
    print('transaction reserve/layer1/logits failures preserve old state; retry and shared old root checked')
    print('negative_early_publish old length with mutated layer/window detected')

    prompt = [0, 1, 3, 2, 4]
    recovery_worst = 0.0
    recovery_cases = 0
    for mode in ('rope', 'additive', 'bias'):
        for storage in (('raw', 'rot') if mode == 'rope' else ('raw',)):
            for window in (None, 1, 3):
                state = new_state(mode, storage, window)
                absent = consume(model, state, make_rows(prompt, want=False))
                require(all(value is None for value in absent), 'logits must actually be absent')
                before = copy.deepcopy(state)
                recovered = recover(model, state, prompt[-1])
                reference = full_reference(model, prompt, mode, window)[-1]
                recovery_worst = max(recovery_worst, close(recovered, reference, 'prefix recovery'))
                require_unchanged(before, state)
                check_cache(state)
                # The first pending token is selected only after distribution recovery.
                pending = max(range(VOCAB), key=lambda t: recovered[t])
                require(pending == max(range(VOCAB), key=lambda t: reference[t]), 'first token')
                start = state['root']['requests']['A']['n']
                end = start
                good_work, bad_work = end - start + 1, end - start
                require(good_work == 1 and bad_work == 0, 'inclusive pending interval')
                # This separate path has a known pending token and needs no old logits.
                output = consume(model, state, make_rows([pending], start))
                close(output[-1], full_reference(model, prompt + [pending], mode, window)[-1],
                      'known pending consumption')
                require(state['root']['requests']['A']['n'] == len(prompt) + 1, 'pending length')
                check_cache(state)
                recovery_cases += 1
    print('prefix_without_logits recovery_cases=%d max_abs=%.3e; no append; recovered distribution selects first token'
          % (recovery_cases, recovery_worst))
    print('known_pending work_rows=1 faulty_coordinate_difference=0 detected; next distribution checked')

    state = new_state(window=3)
    consume(model, state, make_rows(prompt, want=False))
    before = copy.deepcopy(state)
    recover(model, state, prompt[-1], faulty_append=True)
    expect_error(AssertionError, lambda: require_unchanged(before, state), 'recovery append')
    expect_error(AssertionError, lambda: check_cache(state), 'duplicate position')
    missing = copy.deepcopy(before)
    missing['root']['requests']['A']['kv'][0].pop(0)
    expect_error(ContractError, lambda: recover(model, missing, prompt[-1]),
                 'recovery with W-1 retained entries')
    print('negative_recovery_duplicate_append and missing_oldest_dependency detected')

    for history in range(5):
        for queries in range(6):
            pairs = queries * history + queries * (queries + 1) // 2
            independent_pairs = sum(1 for p in range(history, history + queries)
                                    for k in range(p + 1))
            require(pairs == independent_pairs, 'pair count')
            for heads in (1, 2, 4):
                counted = sum(2 * 3 + 2 * 3 for head in range(heads)
                              for p in range(history, history + queries)
                              for k in range(p + 1))
                require(counted == 4 * heads * 3 * pairs, 'head axis FLOPs')
            for window in (1, 3):
                masked = sum(1 for p in range(history, history + queries)
                             for k in range(p + 1) if p - k < window)
                formula = sum(min(window, history + i) for i in range(1, queries + 1))
                require(masked == formula, 'window pairs')
    chunks = [(0, 2), (2, 3), (5, 3)]
    chunk_pairs = sum(q * h + q * (q + 1) // 2 for h, q in chunks)
    require(chunk_pairs == 8 * 9 // 2, 'chunk pair conservation')
    require(sum(h + 1 for h in (3, 0)) == 5, 'decode pair count')
    correct = 4 * 2 * 3 * 3
    incorrect = 4 * 2 * 3 * (2 * 3)
    require(correct == 72 and incorrect == 144, 'double-head control')
    print('cost_axis explicit per-head operations=72 faulty_double_head=144 detected; chunk_pairs=36')
    print('PASS within stated CPU model scope; standalone Python execution is the coordinator check')


if __name__ == '__main__':
    main()
