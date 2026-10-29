'''Read-only offline CPU mechanism model; not a Qwen/GPU implementation.
Dense explicit-mask and cached online attention share primitive linear/norm/FFN
operations. No files, project imports, subprocesses, network or writes.
'''
import hashlib
import json
import math

D, HEADS, DK, VOCAB, LAYERS = 4, 2, 6, 7, 2
GROUPS = 0


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label='reference'):
    if isinstance(expected, (tuple, list)):
        require(isinstance(actual, (tuple, list)), label + ': type')
        require(len(actual) == len(expected), label + ': length')
        for a, b in zip(actual, expected):
            close(a, b, label)
    else:
        require(math.isfinite(actual), label + ': finite')
        require(abs(actual - expected) <= 2e-10 * (1 + abs(expected)), label + ': value')


def passed(name):
    global GROUPS
    GROUPS += 1
    print('PASS ' + name)


def rejected(name, operation):
    try:
        operation()
    except AssertionError as error:
        passed('detected: ' + name + ' [' + str(error) + ']')
        return
    raise AssertionError('negative control escaped: ' + name)


def matrix(rows, cols, seed):
    return tuple(tuple(0.23 * math.sin((i + 1) * 1.7 + (j + 1) * 0.9 + seed)
                       for j in range(cols)) for i in range(rows))


def linear(row, weights):
    return tuple(sum(row[i] * weights[i][j] for i in range(len(row)))
                 for j in range(len(weights[0])))


def norm(row):
    denominator = math.sqrt(sum(x * x for x in row) / len(row) + 1e-5)
    return tuple(x / denominator for x in row)


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


PARAMS = tuple({
    'q': matrix(D, HEADS * DK, 1 + l),
    'k': matrix(D, DK, 4 + l),
    'v': matrix(D, DK, 7 + l),
    'o': matrix(HEADS * DK, D, 10 + l),
    'g': matrix(D, 5, 13 + l),
    'u': matrix(D, 5, 16 + l),
    'down': matrix(5, D, 19 + l)
} for l in range(LAYERS))
HEAD = matrix(D, VOCAB, 25)
MERGE1, MERGE2 = matrix(4 * D, 5, 28), matrix(5, D, 31)


def rotate(row, position):
    result = []
    for axis in range(3):
        angle = 0.17 * (axis + 1) * position[axis]
        a, b = row[2 * axis:2 * axis + 2]
        result.extend((a * math.cos(angle) - b * math.sin(angle),
                       a * math.sin(angle) + b * math.cos(angle)))
    return tuple(result)


def projections(rows, positions, layer):
    require(len(rows) == len(positions), 'position length')
    p = PARAMS[layer]
    queries, keys, values = [], [], []
    for original, position in zip(rows, positions):
        row = norm(original)
        raw = linear(row, p['q'])
        queries.append(tuple(rotate(raw[h * DK:(h + 1) * DK], position)
                             for h in range(HEADS)))
        keys.append(rotate(linear(row, p['k']), position))
        values.append(linear(row, p['v']))
    return tuple(queries), tuple(keys), tuple(values)


def finish(row, attention, layer):
    p = PARAMS[layer]
    y = add(row, linear(attention, p['o']))
    u = norm(y)
    gate, up = linear(u, p['g']), linear(u, p['u'])
    activated = tuple(a / (1 + math.exp(-a)) * b for a, b in zip(gate, up))
    return add(y, linear(activated, p['down']))


def logits(pre_final_norm_hidden):
    return linear(norm(pre_final_norm_hidden), HEAD)


def dense_attention(query_heads, keys, values, legal):
    require(bool(legal), 'empty dense visibility')
    joined = []
    for query in query_heads:
        scores = [sum(query[d] * keys[j][d] for d in range(DK)) / math.sqrt(DK)
                  for j in legal]
        maximum = max(scores)
        weights = [math.exp(score - maximum) for score in scores]
        total = math.fsum(weights)
        joined.extend(math.fsum(weight * values[j][d]
                                for weight, j in zip(weights, legal)) / total
                      for d in range(DK))
    return tuple(joined)


def dense(rows, positions):
    '''Full-sequence reference, explicit causal scores and probability vectors.'''
    require(bool(rows), 'dense requires input')
    hidden, caches = tuple(rows), []
    for layer in range(LAYERS):
        q, k, v = projections(hidden, positions, layer)
        hidden = tuple(finish(row, dense_attention(q[i], k, v, list(range(i + 1))), layer)
                       for i, row in enumerate(hidden))
        caches.append(tuple(zip(k, v)))
    return tuple(logits(row) for row in hidden), tuple(caches), hidden


def online(query_heads, key_values):
    '''Stable streaming reduction; no score/probability vector.'''
    require(bool(key_values), 'empty online visibility')
    joined = []
    for query in query_heads:
        maximum, denominator = -math.inf, 0.0
        numerator = [0.0] * DK
        for key, value in key_values:
            score = math.fsum(a * b for a, b in zip(query, key)) / math.sqrt(DK)
            new_maximum = max(maximum, score)
            scale = 0.0 if maximum == -math.inf else math.exp(maximum - new_maximum)
            weight = math.exp(score - new_maximum)
            denominator = denominator * scale + weight
            numerator = [acc * scale + weight * val for acc, val in zip(numerator, value)]
            maximum = new_maximum
        joined.extend(x / denominator for x in numerator)
    return tuple(joined)


def empty_cache():
    return tuple(() for _ in range(LAYERS))


def cache_length(cache):
    require(len(cache) == LAYERS, 'layer count')
    require(len({len(layer) for layer in cache}) == 1, 'complete-layer boundary')
    return len(cache[0])


def consume(history, rows, positions, omit_current=False):
    require(bool(rows), 'consume requires new input')
    start, hidden, staged = cache_length(history), tuple(rows), []
    for layer in range(LAYERS):
        q, k, v = projections(hidden, positions, layer)
        combined = history[layer] + tuple(zip(k, v))
        output = []
        for offset, row in enumerate(hidden):
            end = start + offset + (0 if omit_current else 1)
            output.append(finish(row, online(q[offset], combined[:end]), layer))
        staged.append(combined)
        hidden = tuple(output)
    complete = tuple(staged)
    require(cache_length(complete) == start + len(rows), 'commit length')
    return complete, logits(hidden[-1])


def embedding(token):
    return tuple(0.4 * math.cos((token + 1) * (i + 1) * 0.63) for i in range(D))


def next_position(positions):
    value = max(max(position) for position in positions) + 1
    return (value, value, value)


def greedy(values):
    return max(range(len(values)), key=lambda i: (values[i], -i))


def reference_generate(rows, positions, limit, eos=None):
    known, pos, outputs = list(rows), list(positions), []
    for _ in range(limit):
        all_logits, _, _ = dense(known, pos)
        token = greedy(all_logits[-1])
        outputs.append(token)
        known.append(embedding(token))
        pos.append(next_position(pos))
        if token == eos:
            break
    return tuple(outputs)


def select_action(n, c, have_logits, remaining, omit_recovery=False):
    '''Fault mutation genuinely yields a state with no successor; it does not throw.'''
    if remaining == 0:
        return 'done'
    if c < n:
        return 'consume_suffix'
    if c == n and have_logits:
        return 'sample'
    if c == n and n > 0 and not omit_recovery:
        return 'recover_last'
    return None


def generate(rows, positions, prefix, saved=None, limit=4, eos=None, fault=None):
    require(bool(rows), 'canonical input requires BOS or a real row')
    require(0 <= prefix <= len(rows), 'prefix range')
    require(limit >= 0, 'output budget')
    if prefix:
        prefix_logits, shared, prefix_hidden = dense(rows[:prefix], positions[:prefix])
    else:
        prefix_logits, shared, prefix_hidden = (), empty_cache(), ()
    snapshot, state = repr(shared), shared
    current = None
    if prefix == len(rows) and saved == 'logits':
        current = prefix_logits[-1]
    elif prefix == len(rows) and saved == 'pre_final_hidden':
        current = logits(prefix_hidden[-1])
    known, pos, outputs, trace = list(rows), list(positions), [], []
    remaining = limit
    while remaining:
        n, c = len(known), cache_length(state)
        action = select_action(n, c, current is not None, remaining,
                               omit_recovery=(fault == 'omit_recovery'))
        require(action is not None, 'active request has no successor')
        if action == 'consume_suffix':
            trace.append(action)
            state, current = consume(state, known[c:], pos[c:])
        elif action == 'recover_last':
            trace.append(action)
            history = state if fault == 'duplicate_last' else tuple(layer[:-1] for layer in state)
            state, current = consume(history, known[-1:], pos[-1:])
        require(cache_length(state) == len(known), 'sampling boundary equals known length')
        require(current is not None, 'sampling requires logits')
        expected_logits, expected_cache, _ = dense(known, pos)
        close(current, expected_logits[-1], 'last logits')
        close(state, expected_cache, 'all-layer KV')
        token = greedy(current)
        trace.append('sample')
        outputs.append(token)
        known.append(embedding(token))
        pos.append(next_position(pos))
        current = None
        remaining -= 1
        require(cache_length(state) == len(known) - 1, 'sampled token has no KV yet')
        require(repr(shared) == snapshot, 'published shared cache changed')
        if token == eos:
            break
    require(repr(shared) == snapshot, 'shared cache changed at termination')
    return tuple(outputs), cache_length(state), len(known), tuple(trace)


def vision(rows, positions, height, width, faulty_scatter=False):
    require(len(rows) == height * width, 'visual grid coverage')
    hidden = tuple(rows)
    for layer in range(LAYERS):
        q, k, v = projections(hidden, positions, layer)
        if layer == 0:
            groups = [[r * width + c
                       for r in range(top, min(height, top + 2))
                       for c in range(left, min(width, left + 2))]
                      for top in range(0, height, 2) for left in range(0, width, 2)]
            groups.reverse()
        else:
            groups = [list(reversed(range(len(rows))))]
        scattered, gathered = [None] * len(rows), []
        for group in groups:
            source = tuple((k[j], v[j]) for j in group)
            for i in group:
                output = finish(hidden[i], online(q[i], source), layer)
                require(scattered[i] is None, 'duplicate scatter destination')
                scattered[i] = output
                gathered.append(output)
        require(all(row is not None for row in scattered), 'scatter coverage')
        hidden = tuple(gathered if faulty_scatter else scattered)
    return hidden


def dense_vision(rows, positions, height, width):
    '''Dense coordinate-mask reference; no gather groups or inverse mapping.'''
    require(len(rows) == height * width, 'reference grid coverage')
    hidden = tuple(rows)
    for layer in range(LAYERS):
        q, k, v = projections(hidden, positions, layer)
        result = []
        for i, row in enumerate(hidden):
            legal = [j for j in range(len(rows))
                     if layer == 1 or
                     ((i // width) // 2 == (j // width) // 2 and
                      (i % width) // 2 == (j % width) // 2)]
            result.append(finish(row, dense_attention(q[i], k, v, legal), layer))
        hidden = tuple(result)
    return hidden


def merge_projection(flat):
    return linear(tuple(math.tanh(x) for x in linear(flat, MERGE1)), MERGE2)


def spatial_merge(features, height, width, contiguous_fault=False):
    require(height > 0 and width > 0 and height % 2 == width % 2 == 0,
            'merge divisibility')
    require(len(features) == height * width, 'merge coverage')
    if contiguous_fault:
        groups = [list(range(i, i + 4)) for i in range(0, len(features), 4)]
    else:
        groups = [[(2 * r + dr) * width + 2 * c + dc
                   for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1))]
                  for r in range(height // 2) for c in range(width // 2)]
    return tuple(merge_projection(tuple(x for i in indices for x in features[i]))
                 for indices in groups)


def geometry(height, width):
    require(height > 0 and width > 0 and height % 28 == width % 28 == 0,
            'positive 28-multiple geometry')
    h, w = height // 14, width // 14
    return h, w, h * w, h * w // 4


def window_pairs(height, width):
    return sum((min(8, height - r) * min(8, width - c)) ** 2
               for r in range(0, height, 8) for c in range(0, width, 8))


def enumerate_pairs(height, width):
    return sum(1 for i in range(height * width) for j in range(height * width)
               if (i // width) // 8 == (j // width) // 8
               and (i % width) // 8 == (j % width) // 8)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(',', ':')).encode('ascii')).hexdigest()


def validate_identity(expected, supplied):
    require(expected == supplied, 'prefix identity mismatch')


def main():
    h, w, patches, merged = geometry(448, 672)
    require((h, w, patches, merged) == (32, 48, 1536, 384), 'geometry reference')
    for sh, sw in ((1, 1), (2, 2), (6, 10), (8, 8), (10, 18)):
        require(window_pairs(sh, sw) == enumerate_pairs(sh, sw), 'window enumeration')
    require(window_pairs(h, w) == 98304, 'large window count')
    require(4 * patches ** 2 + 28 * window_pairs(h, w) == 12189696, 'mixed pair count')
    require(32 * patches ** 2 == 75497472, 'dense pair count')
    bkv = 2 * 28 * 4 * 128 * 2
    require(bkv == 57344 and bkv * 896 == 49 * 1024 ** 2, 'input KV bytes')
    require(bkv * 1152 == 63 * 1024 ** 2, 'request reservation')
    require(bkv * 1152 * 8 == 504 * 1024 ** 2, 'eight reservations')
    require(1536 * 1280 * 2 + 384 * 3584 * 2 + 448 * 672 * 3 * 4 == 10297344,
            'three simultaneous objects')
    passed('geometry, independently enumerated tail windows and resource arithmetic')
    rejected('invalid 434-pixel height', lambda: geometry(434, 672))
    faulty_merged = patches
    rejected('forgotten spatial merge', lambda: require(faulty_merged == merged, 'token count'))

    vh, vw = 2, 4
    visual_input = tuple(tuple(0.13 * (i + 1) + math.sin((d + 1) * (i + 2))
                               for d in range(D)) for i in range(vh * vw))
    visual_positions = tuple((3, i // vw, i % vw) for i in range(vh * vw))
    reference_features = dense_vision(visual_input, visual_positions, vh, vw)
    features = vision(visual_input, visual_positions, vh, vw)
    close(features, reference_features, 'visual window/full layers')
    for sh, sw in ((1, 1), (1, 3), (3, 3)):
        sample = tuple(embedding(i) for i in range(sh * sw))
        pos = tuple((0, i // sw, i % sw) for i in range(sh * sw))
        close(vision(sample, pos, sh, sw), dense_vision(sample, pos, sh, sw), 'partial windows')
    passed('visual gather/scatter against independent dense coordinate mask')
    rejected('missing inverse window mapping', lambda: close(
        vision(visual_input, visual_positions, vh, vw, True), reference_features))
    coordinates = {(i // vw, i % vw): reference_features[i] for i in range(vh * vw)}
    reference_z = tuple(merge_projection(tuple(value for r in (0, 1)
                          for c in (left, left + 1) for value in coordinates[(r, c)]))
                        for left in (0, 2))
    z = spatial_merge(features, vh, vw)
    close(z, reference_z, 'coordinate merge reference')
    passed('spatial 2x2 grouping and two-layer bridge')
    rejected('contiguous rows mistaken for spatial neighbors', lambda: close(
        spatial_merge(reference_features, vh, vw, True), reference_z))
    rejected('odd merge grid', lambda: spatial_merge(visual_input[:6], 2, 3))

    rows = (embedding(1), z[0], z[1], embedding(3))
    positions = ((0, 0, 0), (1, 0, 0), (1, 0, 1), (3, 3, 3))
    expected_logits, expected_cache, _ = dense(rows, positions)
    for chunk in (1, 2, len(rows)):
        state = empty_cache()
        for start in range(0, len(rows), chunk):
            end = min(len(rows), start + chunk)
            state, last = consume(state, rows[start:end], positions[start:end])
        close(state, expected_cache, 'chunked KV')
        close(last, expected_logits[-1], 'chunked logits')
    passed('full, chunked and tokenwise decoder against dense reference')
    _, prefix_cache, _ = dense(rows[:2], positions[:2])
    rejected('current KV omitted', lambda: close(
        consume(prefix_cache, rows[2:], positions[2:], True)[1], expected_logits[-1]))
    rejected('partially committed layers', lambda: cache_length((expected_cache[0], ())))

    expected_tokens = reference_generate(rows, positions, 4)
    traces = []
    for prefix in range(len(rows) + 1):
        outputs, c, n, trace = generate(rows, positions, prefix)
        require(outputs == expected_tokens, 'generated token reference')
        require(c == n - 1, 'terminal token need not have KV')
        traces.append(trace)
    for saved in ('logits', 'pre_final_hidden'):
        outputs, c, n, trace = generate(rows, positions, len(rows), saved=saved)
        require(outputs == expected_tokens and trace[0] == 'sample', 'saved output branch')
    require(traces[0][0] == 'consume_suffix', 'cold branch coverage')
    require(traces[-1][0] == 'recover_last', 'recovery branch coverage')
    passed('all prefix boundaries, saved logits/hidden and complete hit without output')

    stalled_action = select_action(len(rows), len(rows), False, 4, omit_recovery=True)
    require(stalled_action is None, 'fault must actually produce no successor')
    rejected('omitted recovery causes genuine no-successor state', lambda: generate(
        rows, positions, len(rows), fault='omit_recovery'))
    rejected('duplicate last input on complete hit', lambda: generate(
        rows, positions, len(rows), fault='duplicate_last'))
    eos = expected_tokens[0]
    outputs, c, n, trace = generate(rows, positions, len(rows), eos=eos)
    require(outputs == (eos,) and c == len(rows) and n == len(rows) + 1, 'EOS boundary')
    for prefix in (0, len(rows)):
        outputs, c, n, trace = generate(rows, positions, prefix, limit=0)
        require(not outputs and not trace and c == prefix and n == len(rows), 'zero budget')
    singleton, singleton_pos = (embedding(2),), ((0, 0, 0),)
    require(generate(singleton, singleton_pos, 1)[0]
            == reference_generate(singleton, singleton_pos, 4), 'single-row recovery')
    require(generate(rows[:-1], positions[:-1], len(rows) - 1)[0]
            == reference_generate(rows[:-1], positions[:-1], 4), 'last row is visual')
    rejected('empty canonical input without BOS', lambda: generate((), (), 0))
    passed('liveness oracle, EOS, zero budget, singleton and final visual row')

    feature_a = identity(['content-A', 'processor-v1', 'encoder-v1', 'merger-v1', [448, 672]])
    feature_b = identity(['content-B', 'processor-v1', 'encoder-v1', 'merger-v1', [448, 672]])
    require(feature_a != feature_b, 'content identity separation')
    descriptor = [[feature_a], [1, 3], positions, 'causal', 'template-v1']
    correct = identity(descriptor)
    validate_identity(correct, correct)
    flat_positions = tuple((i, i, i) for i in range(len(rows)))
    wrong_position = identity([[feature_a], [1, 3], flat_positions, 'causal', 'template-v1'])
    rejected('flattened MRoPE reused', lambda: validate_identity(correct, wrong_position))
    wrong_content = identity([[feature_b], [1, 3], positions, 'causal', 'template-v1'])
    rejected('same shape/locator with changed content', lambda: validate_identity(correct, wrong_content))
    wrong_text = identity([[feature_a], [1, 4], positions, 'causal', 'template-v1'])
    rejected('same text length with changed token', lambda: validate_identity(correct, wrong_text))
    ordered, swapped = identity([feature_a, feature_b]), identity([feature_b, feature_a])
    rejected('media binding swapped', lambda: validate_identity(ordered, swapped))
    def faulty_position_key(value):
        return identity([value[0], value[1], len(value[2]), value[3], value[4]])
    altered = [[feature_a], [1, 3], flat_positions, 'causal', 'template-v1']
    rejected('position-ignoring key collision', lambda: require(
        faulty_position_key(descriptor) != faulty_position_key(altered), 'faulty key collision'))
    passed('prefix reuse validator rejects changed positions, content, text and media order')

    print('observations: CPU toy layers=2 query_heads=2 KV_heads=1 head_dim=6')
    print('observations: visual grid=2x4; window/full layers; merged visual rows=2')
    print('observations: no-logits complete-hit trace=' + repr(traces[-1]))
    print('observations: omitted-recovery successor=' + repr(stalled_action))
    print('observations: independent full-forward generated IDs=' + repr(expected_tokens))
    print('observations: geometry arithmetic patches=1536 visual_rows=384 input_KV=49MiB')
    print('PASS groups=' + str(GROUPS))
    print('scope: shared primitive math is not independently verified; no real checkpoint,')
    print('scope: processor, GPU, DMA, asynchronous cancellation, physical COW, performance,')
    print('scope: image quality, stochastic sampling, grammar, stop strings, video policy or Agent benefit')


if __name__ == '__main__':
    main()
