#!/usr/bin/env python3
"""Offline CPU model; no project imports, device execution, processes or writes."""
import itertools
import math


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def bucket_plan(requests, suffix=(512, 1024, 2048), prefix=(0, 512, 1024), fault=False):
    if (not suffix or not prefix or
            any(type(v) is not int or v <= 0 for v in suffix) or
            any(type(v) is not int or v < 0 for v in prefix) or
            tuple(sorted(set(suffix))) != tuple(suffix) or
            tuple(sorted(set(prefix))) != tuple(prefix) or prefix[0] != 0):
        raise ValueError('config')
    ready, active = [], []
    for index, row in enumerate(requests):
        if len(row) != 3:
            raise ValueError('request')
        length, hit, logits = row
        if (type(length) is not int or type(hit) is not int or
                type(logits) is not bool or length <= 0 or not 0 <= hit <= length):
            raise ValueError('request')
        if hit == length and logits:
            ready.append(index)
            continue
        tail = int(hit == length)
        upper = hit - tail
        active.append((index, length, hit, upper, tail))
    if not active:
        return (0, 0, tuple(ready), ())
    need = max(length if fault else length - upper
               for index, length, hit, upper, tail in active)
    bs = next((b for b in suffix if b >= need), None)
    if bs is None:
        raise ValueError('suffix')
    work = []
    for index, length, hit, upper, tail in active:
        moved = min(upper, bs - (length - upper))
        kept = upper - moved
        work.append((index, kept, length - kept, moved, tail))
    bp = next((b for b in prefix if all(row[1] <= b for row in work)), None)
    if bp is None:
        raise ValueError('prefix')
    return (bs, bp, tuple(ready), tuple(work))


def enumerated_reference(requests, suffix, prefix):
    # Independent reference: enumerate legal retained-prefix lengths.
    ready, active = [], []
    for index, (length, hit, logits) in enumerate(requests):
        if hit == length and logits:
            ready.append(index)
        else:
            tail = int(hit == length)
            active.append((index, length, hit - tail, tail))
    if not active:
        return (0, 0, tuple(ready), ())
    candidates = [b for b in suffix
                  if all(length - upper <= b for _, length, upper, _ in active)]
    if not candidates:
        raise ValueError('suffix')
    bs = min(candidates)
    work = []
    for index, length, upper, tail in active:
        feasible = [a for a in range(upper + 1) if length - a <= bs]
        kept = min(feasible)
        work.append((index, kept, length - kept, upper - kept, tail))
    candidates = [p for p in prefix if all(a <= p for _, a, _, _, _ in work)]
    if not candidates:
        raise ValueError('prefix')
    return (bs, min(candidates), tuple(ready), tuple(work))


def outcome(fn, *args, **kwargs):
    try:
        return ('ok', fn(*args, **kwargs))
    except ValueError as exc:
        return ('error', str(exc))


def inputs(length):
    return [(t % 7 - 3) / 10.0 for t in range(length)]


def project(value, position, layer):
    return (0.7 * value + 0.01 * (position + 1),
            0.4 * value - 0.005 * position + 0.02 * layer,
            0.6 * value + 0.03 * layer)


def dense_weighted(query, keys, values):
    scores = [query * key for key in keys]
    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    return math.fsum(w * value for w, value in zip(weights, values)) / math.fsum(weights)


def dense_reference(length, layers=2):
    # Full sequence, layer by layer; no cache, block table, plan or padding.
    hidden = inputs(length)
    history = []
    for layer in range(layers):
        projected = [project(value, t, layer) for t, value in enumerate(hidden)]
        queries = [item[0] for item in projected]
        keys = [item[1] for item in projected]
        values = [item[2] for item in projected]
        output = [0.5 * hidden[t] + dense_weighted(queries[t], keys[:t + 1], values[:t + 1])
                  for t in range(length)]
        history.append((output, keys, values))
        hidden = output
    return history


def online_weighted(query, flat, indices):
    # Streaming reduction differs from the dense normalized reference.
    peak, denominator, numerator = None, 0.0, 0.0
    for index in indices:
        key, value = flat[index]
        score = query * key
        if peak is None:
            peak, denominator, numerator = score, 1.0, value
        else:
            new_peak = max(peak, score)
            old_scale = math.exp(peak - new_peak)
            new_scale = math.exp(score - new_peak)
            denominator = denominator * old_scale + new_scale
            numerator = numerator * old_scale + new_scale * value
            peak = new_peak
    if peak is None:
        raise ValueError('empty attention')
    return numerator / denominator


def cached_run(length, kept, bs, bp, reference, fault=None):
    count = length - kept
    require(0 <= kept < length and count <= bs and kept <= bp, 'illegal cached plan')
    hidden = inputs(length)[kept:]
    history = []
    page_size = 3
    page_count = (kept + page_size - 1) // page_size
    table = list(reversed(range(page_count)))
    for layer, (_, reference_keys, reference_values) in enumerate(reference):
        # Cache contains only the retained prefix. Physical pages are reversed.
        pages = {}
        for t in range(kept):
            physical = table[t // page_size]
            pages[(physical, t % page_size)] = (reference_keys[t], reference_values[t])
        flat = [(20.0, 500.0)] * (bp + bs)
        for t in range(kept):
            flat[t] = pages[(table[t // page_size], t % page_size)]
        projected = []
        for j, value in enumerate(hidden):
            position = bp + j if fault == 'position' else kept + j
            projected.append(project(value, position, layer))
        output, keys, values = [], [], []
        for j, (query, key, value) in enumerate(projected):
            flat[bp + j] = (key, value)
            if fault == 'mask':
                visible = range(bp + j + 1)
            else:
                visible = itertools.chain(range(kept), range(bp, bp + j + 1))
            output.append(0.5 * hidden[j] + online_weighted(query, flat, visible))
            keys.append(key)
            values.append(value)
        history.append((output, keys, values))
        hidden = output
    return history


def state_error(candidate, reference, kept):
    require(len(candidate) == len(reference), 'layer count')
    maximum = 0.0
    for actual_layer, expected_layer in zip(candidate, reference):
        for actual, expected_full in zip(actual_layer, expected_layer):
            expected = expected_full[kept:]
            require(len(actual) == len(expected), 'state length')
            for left, right in zip(actual, expected):
                require(math.isfinite(left), 'nonfinite state')
                maximum = max(maximum, abs(left - right))
    return maximum


def main():
    suffix, prefix = (4, 8, 16, 32), (0, 4, 8, 16, 32)
    checked = 0
    for length in range(1, 21):
        for hit in range(length + 1):
            for logits in (False, True):
                requests = [(length, hit, logits)]
                got = outcome(bucket_plan, requests, suffix, prefix)
                expected = outcome(enumerated_reference, requests, suffix, prefix)
                require(got == expected, 'single enumeration mismatch: ' + repr(requests))
                checked += 1
    choices = [(5, 0, False), (7, 6, False), (6, 6, True), (8, 8, False)]
    for batch in itertools.product(choices, repeat=2):
        require(outcome(bucket_plan, batch, suffix, prefix) ==
                outcome(enumerated_reference, batch, suffix, prefix), 'batch enumeration mismatch')
        checked += 1
    require(bucket_plan([]) == (0, 0, (), ()), 'empty batch is not no-op')
    require(bucket_plan([(8, 8, True)], suffix, prefix) == (0, 0, (0,), ()), 'full-hit output')
    require(bucket_plan([(8, 8, False)], suffix, prefix) ==
            (4, 4, (), ((0, 4, 4, 3, 1),)), 'full-hit replay did not enter bucket algorithm')
    mixed = [(6, 6, True), (7, 7, False), (5, 2, False)]
    require(bucket_plan(mixed, suffix, prefix) ==
            (4, 4, (0,), ((1, 3, 4, 3, 1), (2, 1, 4, 1, 0))), 'request identity changed')
    examples = [
        ([(1000, 0, False)], (1024, 0, (), ((0, 0, 1000, 0, 0),))),
        ([(800, 128, False)], (1024, 0, (), ((0, 0, 800, 128, 0),))),
        ([(900, 640, False)], (512, 512, (), ((0, 388, 512, 252, 0),))),
        ([(800, 128, False), (900, 640, False)],
         (1024, 0, (), ((0, 0, 800, 128, 0), (1, 0, 900, 640, 0))))]
    for requests, expected in examples:
        require(bucket_plan(requests) == expected, 'document example mismatch')
    require(outcome(bucket_plan, [(1600, 1280, False)]) == ('error', 'prefix'), 'prefix overflow')
    retry = bucket_plan([(1600, 1280, False)], (1024, 2048), (0, 512, 1024))
    require(retry == (1024, 1024, (), ((0, 576, 1024, 704, 0),)), 'explicit larger-suffix retry')
    for request in ([(0, 0, False)], [(2, 3, False)], [(2, -1, False)]):
        require(outcome(bucket_plan, request) == ('error', 'request'), 'invalid request accepted')
    require(outcome(bucket_plan, [(4097, 0, False)]) == ('error', 'suffix'), 'suffix overflow')
    require(outcome(bucket_plan, [(2, 0, False)], (8, 4), (0, 4)) ==
            ('error', 'config'), 'unordered config accepted')
    require(outcome(bucket_plan, [(2, 0, False)], (4, 8), (4, 8)) ==
            ('error', 'config'), 'missing zero prefix accepted')
    print('OBS bucket model: %d independent single/batch enumerations; explicit full-hit, identity and overflow edges' % checked)

    pair_cases = 0
    for kept in range(10):
        for count in range(10):
            formula = kept * count + count * (count + 1) // 2
            enumerated = sum(len(tuple(range(kept + j + 1))) for j in range(count))
            require(formula == enumerated, 'causal pair formula')
            pair_cases += 1
    require(388 * 512 + 512 * 513 // 2 == 329984, 'moved pairs')
    require(640 * 260 + 260 * 261 // 2 == 200330, 'unmoved pairs')
    require(512 * (512 + 512) == 524288 and 512 * (1024 + 512) == 786432,
            'rectangular work')
    wrong_pairs = 0 * 1 + 1 * (1 - 1) // 2
    require(wrong_pairs != len(tuple(range(1))), 'negative triangular control escaped')
    bad_selector = outcome(bucket_plan, [(900, 640, False)], fault=True)
    oracle = outcome(enumerated_reference, [(900, 640, False)],
                     (512, 1024, 2048), (0, 512, 1024))
    require(bad_selector != oracle, 'negative total-prompt selector escaped')
    print('OBS cost model: %d pair-domain enumerations; moved=329984, retained=200330; rectangular=524288/786432' % pair_cases)

    maximum_error = 0.0
    numeric_cases = [(1, 0), (1, 1), (8, 0), (17, 12), (20, 19), (900, 640), (800, 128)]
    for length, hit in numeric_cases:
        config_s = suffix if length <= 20 else (512, 1024, 2048)
        config_p = prefix if length <= 20 else (0, 512, 1024)
        bs, bp, ready, work = bucket_plan([(length, hit, False)], config_s, config_p)
        require(not ready and len(work) == 1, 'missing replay successor')
        _, kept, count, moved, tail = work[0]
        require(kept + count == length and hit - kept == moved + tail, 'length conservation')
        reference = dense_reference(length)
        error = state_error(cached_run(length, kept, bs, bp, reference), reference, kept)
        require(error < 5e-12, 'cached multi-layer state mismatch: ' + repr((length, hit, error)))
        maximum_error = max(maximum_error, error)
    reference = dense_reference(17)
    bs, bp, _, work = bucket_plan([(17, 12, False)], suffix, prefix)
    kept = work[0][1]
    require(bp > kept, 'negative controls need actual prefix padding')
    mask_error = state_error(cached_run(17, kept, bs, bp, reference, 'mask'), reference, kept)
    position_error = state_error(cached_run(17, kept, bs, bp, reference, 'position'), reference, kept)
    require(mask_error > 1e-6, 'poison-padding negative control escaped')
    require(position_error > 1e-6, 'physical-position negative control escaped')
    require(outcome(online_weighted, 1.0, [], []) == ('error', 'empty attention'), 'empty reduction')
    print('OBS FP64 CPU attention: %d two-layer dense/cache comparisons, max_abs_error=%.3g' %
          (len(numeric_cases), maximum_error))
    print('OBS negative controls detected: total-prompt selector, missing causal diagonal, padding mask, physical position')
    print('OBS faulty attention errors: mask=%.6g position=%.6g' % (mask_error, position_error))

    for position, expected in ((1000, 1024), (1023, 1024), (1024, 2048), (2047, 2048), (2048, None)):
        selected = next((b for b in (1024, 2048) if b >= position + 1), None)
        legal = [b for b in (1024, 2048) if position in range(b)]
        require(selected == expected and selected == (min(legal) if legal else None), 'generation boundary')
    print('OBS generation capacity: positions 1023/1024 and maximum overflow checked')
    print('SCOPE CPU algebra, explicit block/flat ABI and scalar two-layer attention only; no NxD, NEFF, GPU/Neuron, visibility or performance claims')
    print('RESULT PASS')


if __name__ == '__main__':
    main()
