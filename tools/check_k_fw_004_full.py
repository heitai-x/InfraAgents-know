#!/usr/bin/env python3
"""K-FW-004: deterministic, standalone, read-only CPU mechanism checks."""
from fractions import Fraction
from itertools import product
from math import exp, isclose


DETECTED = []


def equal(got, expected):
    if got != expected:
        raise AssertionError((got, expected))


def reject_fault(name, operation):
    try:
        operation()
    except AssertionError:
        DETECTED.append(name)
    else:
        raise AssertionError('fault survived: ' + name)


def validate(n, h, p):
    if min(n, h, p) < 0 or h > n:
        raise ValueError('invalid lengths')


def migration(n, h, p):
    validate(n, h, p)
    suffix = n - h
    if suffix > p:
        return None
    moved = min(h, p - suffix)
    return h - moved, suffix + moved, moved, p - suffix - moved


def token_oracle(n, h, p):
    validate(n, h, p)
    prefix = list(range(h))
    suffix = list(range(h, n))
    if len(suffix) > p:
        return None
    moved = 0
    while prefix and len(suffix) < p:
        suffix.insert(0, prefix.pop())
        moved += 1
    equal(prefix + suffix, list(range(n)))
    return len(prefix), len(suffix), moved, p - len(suffix)


def plan(batch, p_buckets, k_buckets, use_oracle=False):
    for n, h in batch:
        validate(n, h, 0)
    if not batch:
        return 'empty', None, ()
    ps = sorted(set(p_buckets))
    ks = sorted(set(k_buckets))
    if any(x < 0 for x in ps + ks):
        raise ValueError('negative bucket')
    if use_oracle:
        viable = []
        for p in ps:
            parts = [token_oracle(n, h, p) for n, h in batch]
            if all(part is not None for part in parts):
                viable.append((p, parts))
        if not viable:
            return 'suffix_overflow', None, ()
        p, parts = viable[0]
        fitting = [k for k in ks if all(part[0] <= k for part in parts)]
    else:
        p = next((p for p in ps if p >= max(n - h for n, h in batch)), None)
        if p is None:
            return 'suffix_overflow', None, ()
        parts = [migration(n, h, p) for n, h in batch]
        required = max(part[0] for part in parts)
        fitting = [k for k in ks if k >= required]
    if not fitting:
        return 'prefix_overflow', None, ()
    return 'ok', (p, fitting[0]), tuple(parts)


def score(position, token):
    return (((position + 1) * (token + 2)) % 11 - 5) / 3.0


def weighted(items):
    maximum = max(s for s, value in items)
    weights = [exp(s - maximum) for s, value in items]
    return sum(w * value for w, (s, value) in zip(weights, items)) / sum(weights)


def attention_reference(n, start):
    # Single-layer visibility reference, not a multilayer model.
    return tuple(weighted([(score(pos, t), 2 * t + 1) for t in range(pos + 1)])
                 for pos in range(start, n))


def attention_physical(n, h, p, k, include_padding=False):
    start, q, moved, pad = migration(n, h, p)
    if k < start:
        raise ValueError('prefix capacity')
    slots = list(range(start)) + [None] * (k - start)
    slots += list(range(start, n)) + [None] * pad
    result = []
    pairs = 0
    for row in range(p):
        if row >= q:
            result.append(None)
            continue
        items = []
        for index, token in enumerate(slots):
            visible = index < start or (k <= index < k + q and index - k <= row)
            if visible:
                if token is None:
                    raise AssertionError('visible padding')
                items.append((score(start + row, token), 2 * token + 1))
                pairs += 1
            elif include_padding and token is None:
                items.append((0.0, 0.0))
        result.append(weighted(items))
    return tuple(result), pairs


def close_rows(got, expected):
    equal(len(got), len(expected))
    for actual, reference in zip(got, expected):
        if reference is None:
            equal(actual, None)
        elif actual is None or not isclose(actual, reference, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError((actual, reference))


def close_nested(got, expected):
    if isinstance(expected, (tuple, list)):
        equal(len(got), len(expected))
        for actual, reference in zip(got, expected):
            close_nested(actual, reference)
    elif not isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError((got, expected))


def guard(artifact, request, omitted=()):
    return all(request[field] == value for field, value in artifact.items()
               if field not in omitted)


def compiled_integer_view(artifact, storage):
    rows, cols = artifact['shape']
    sr, sc = artifact['stride']
    constant = artifact['constant']
    return tuple(tuple(storage[i * sr + j * sc] * constant for j in range(cols))
                 for i in range(rows))


def branch_reference(n):
    values = tuple(range(1, n + 1))
    if n == 10:
        return tuple(x * 10 for x in values)
    if n <= 30:
        return tuple(x * 200 for x in values)
    return tuple(x * n for x in values)


def break_even(cost, saving, hit_fraction):
    phi = Fraction(hit_fraction)
    if cost < 0 or saving <= 0 or not 0 < phi <= 1:
        raise ValueError('invalid amortization domain')
    threshold = Fraction(cost, saving) / phi
    return -(-threshold.numerator // threshold.denominator)


def verify_break_even(count, cost, saving, phi):
    if count == 0:
        equal(cost, 0)
        return
    if not (Fraction(count - 1) * phi * saving < cost <= Fraction(count) * phi * saving):
        raise AssertionError('not the first cost-covering count')


def lifetime(ignore_graph=False):
    model = {'state': 'ready', 'runs': set(), 'graphs': set()}
    events = [('capture', 'g1'), ('launch', 'r1'), ('drain', ''),
              ('finish', 'r1'), ('release', ''), ('destroy', 'g1'), ('release', '')]
    history = []
    for action, ticket in events:
        history.append((action, ticket))
        if action == 'capture':
            model['graphs'].add(ticket)
        elif action == 'launch':
            equal(model['state'], 'ready')
            model['runs'].add(ticket)
        elif action == 'drain':
            model['state'] = 'draining'
        elif action == 'finish':
            model['runs'].discard(ticket)
        elif action == 'destroy':
            model['graphs'].discard(ticket)
        elif action == 'release':
            if (model['state'] == 'draining' and not model['runs']
                    and (ignore_graph or not model['graphs'])):
                model['state'] = 'released'
        # Reconstruct external owners independently from the event history.
        owners = set()
        for event, identity in history:
            if event == 'capture':
                owners.add(('graph', identity))
            elif event == 'launch':
                owners.add(('run', identity))
            elif event == 'finish':
                owners.discard(('run', identity))
            elif event == 'destroy':
                owners.discard(('graph', identity))
        if model['state'] == 'released' and owners:
            raise AssertionError('released with external readers')
    equal(model['state'], 'released')
    return len(events)


# A separate toy causal decoder. Scalar normalization is identity by definition;
# affine projections, positional terms, attention, two residuals and the head
# make layer identity and hidden-state forwarding observable. This is not a
# checkpoint implementation, RoPE implementation or low-precision simulation.
def embedding(token, position):
    return (token + 1) / 8.0 + position / 32.0


def toy_reference(tokens, depth=3):
    if not tokens:
        raise ValueError('empty prompt has no last-row logits')
    hidden = tuple(embedding(token, pos) for pos, token in enumerate(tokens))
    caches = []
    layers = []
    for layer in range(depth):
        # Full-sequence reference creates every layer from original logical rows.
        queries = tuple(x * (layer + 1) / 16.0 + (pos + 1) / 32.0
                        for pos, x in enumerate(hidden))
        keys = tuple(x * (layer + 2) / 32.0 + (pos + 1) / 64.0
                     for pos, x in enumerate(hidden))
        values = tuple(x * (layer + 3) / 8.0 + (layer + 1) / 4.0 for x in hidden)
        caches.append(tuple(zip(keys, values)))
        next_hidden = []
        for pos, x in enumerate(hidden):
            scores = [queries[pos] * keys[t] for t in range(pos + 1)]
            maximum = max(scores)
            weights = [exp(s - maximum) for s in scores]
            attention = sum(weights[t] * values[t] for t in range(pos + 1)) / sum(weights)
            residual = x + attention / (layer + 2)
            next_hidden.append(residual + residual / (layer + 3) + 1.0 / 8.0)
        hidden = tuple(next_hidden)
        layers.append(hidden)
    logits = (2.0 * hidden[-1], -hidden[-1] + 0.25)
    return tuple(caches), tuple(layers), logits


def toy_suffix(tokens, start, old_cache, depth=3, fault=''):
    if not 0 <= start < len(tokens):
        raise ValueError('suffix must contain an effective last row')
    hidden = tuple(embedding(tokens[pos], pos) for pos in range(start, len(tokens)))
    deltas = []
    layers = []
    for layer in range(depth):
        if fault == 'reset_hidden' and layer > 0:
            hidden = tuple(embedding(tokens[pos], pos) for pos in range(start, len(tokens)))
        qkv = []
        for row, x in enumerate(hidden):
            pos = start + row
            qkv.append((x * (layer + 1) / 16.0 + (pos + 1) / 32.0,
                        x * (layer + 2) / 32.0 + (pos + 1) / 64.0,
                        x * (layer + 3) / 8.0 + (layer + 1) / 4.0))
        delta = tuple((k, v) for query, k, v in qkv)
        history_layer = 0 if fault == 'wrong_history_layer' else layer
        history = old_cache[history_layer][:start]
        equal(len(history), start)
        visible_view = history + delta
        next_hidden = []
        for row, x in enumerate(hidden):
            items = [(qkv[row][0] * k, v) for k, v in visible_view[:start + row + 1]]
            attention = weighted(items)
            residual = x + attention / (layer + 2)
            next_hidden.append(residual + residual / (layer + 3) + 1.0 / 8.0)
        hidden = tuple(next_hidden)
        deltas.append(delta)
        layers.append(hidden)
    final_row = layers[0][-1] if fault == 'early_head' else hidden[-1]
    return tuple(deltas), tuple(layers), (2.0 * final_row, -final_row + 0.25)


def old_descriptor(full_cache, committed):
    return (7, committed, tuple(layer[:committed] for layer in full_cache), None)


def transaction(old, start, n, deltas, logits, fail_after=None, early_publish=False):
    # Each staged layer is private. No actual allocator or asynchronous device.
    published = old
    candidate_layers = []
    physical_writes = []
    for layer, delta in enumerate(deltas):
        candidate_layers.append(old[2][layer][:start] + delta)
        physical_writes.append((layer, start, n))
        if early_publish:
            # Fault: publish after one physical layer, before whole-request success.
            published = (old[0] + 1, n, tuple(candidate_layers), None)
        if layer == fail_after:
            return published, tuple(physical_writes), 'quarantined'
    equal(len(candidate_layers), len(old[2]))
    for cache in candidate_layers:
        equal(len(cache), n)
    published = (old[0] + 1, n, tuple(candidate_layers), logits)
    return published, tuple(physical_writes), 'committed'


def check_failed_transaction(old, outcome):
    published, writes, state = outcome
    equal(state, 'quarantined')
    equal(published, old)
    if not writes:
        raise AssertionError('failure injection did not follow a physical layer write')


def multilayer_checks():
    cases = 0
    for n in range(1, 7):
        tokens = tuple((pos * 3 + 1) % 7 for pos in range(n))
        full_cache, full_layers, full_logits = toy_reference(tokens)
        for h in range(n + 1):
            # Actual cached prefix, independently computed without the suffix.
            if h:
                prefix_cache, unused_layers, unused_logits = toy_reference(tokens[:h])
                for layer in range(3):
                    close_nested(prefix_cache[layer], full_cache[layer][:h])
            else:
                prefix_cache = ((), (), ())
            for p in range(1, 8):
                split = migration(n, h, p)
                if split is None:
                    continue
                start, q, moved, pad = split
                deltas, layers, logits = toy_suffix(tokens, start, prefix_cache)
                close_nested(logits, full_logits)
                for layer in range(3):
                    close_nested(deltas[layer], full_cache[layer][start:])
                    close_nested(layers[layer], full_layers[layer][start:])
                cases += 1
    equal(cases, 154)

    tokens = (1, 4, 0, 3, 6)
    full_cache, full_layers, full_logits = toy_reference(tokens)
    prefix_cache, unused_layers, unused_logits = toy_reference(tokens[:4])
    for fault in ('reset_hidden', 'wrong_history_layer', 'early_head'):
        reject_fault('multilayer ' + fault,
                     lambda fault=fault: close_nested(
                         toy_suffix(tokens, 2, prefix_cache, fault=fault)[2], full_logits))

    # Old commit has length 4; migrated execution starts at 2 and computes to 5.
    old = old_descriptor(prefix_cache, 4)
    frozen_old = (old[0], old[1], tuple(tuple(layer) for layer in old[2]), old[3])
    deltas, layers, logits = toy_suffix(tokens, 2, old[2])
    for cut in range(3):
        outcome = transaction(old, 2, len(tokens), deltas, logits, fail_after=cut)
        check_failed_transaction(old, outcome)
        equal(len(outcome[1]), cut + 1)
        equal(old, frozen_old)
        # Retry obtains only the old published cache, never the aborted deltas.
        retry = toy_suffix(tokens, 2, outcome[0][2])
        close_nested(retry[0], tuple(layer[2:] for layer in full_cache))
        close_nested(retry[2], full_logits)
    reject_fault('layer write publishes partial request',
                 lambda: check_failed_transaction(
                     old, transaction(old, 2, len(tokens), deltas, logits,
                                      fail_after=1, early_publish=True)))
    committed, writes, state = transaction(old, 2, len(tokens), deltas, logits)
    equal(state, 'committed')
    equal(committed[:2], (8, len(tokens)))
    close_nested(committed[2], full_cache)
    close_nested(committed[3], full_logits)
    equal(old, frozen_old)

    # Full KV hit without logits: plan can retreat while old commit remains n.
    old_full = old_descriptor(full_cache, len(tokens))
    recovered = toy_suffix(tokens, len(tokens) - 1, old_full[2])
    recovery, unused_writes, recovery_state = transaction(
        old_full, len(tokens) - 1, len(tokens), recovered[0], recovered[2])
    equal(old_full[1], len(tokens))
    equal(recovery_state, 'committed')
    close_nested(recovery[2], full_cache)
    close_nested(recovery[3], full_logits)
    try:
        toy_reference(())
    except ValueError:
        pass
    else:
        raise AssertionError('empty prompt produced last-row logits')
    return cases, 3


def main():
    valid = rejected = attention_cases = 0
    for n in range(9):
        for h in range(n + 1):
            for p in range(9):
                result = migration(n, h, p)
                equal(result, token_oracle(n, h, p))
                if result is None:
                    rejected += 1
                    continue
                valid += 1
                start, q, moved, pad = result
                equal(start + q, n)
                physical, pairs = attention_physical(n, h, p, start + 2)
                reference = attention_reference(n, start) + (None,) * pad
                close_rows(physical, reference)
                oracle_pairs = sum(1 for pos in range(start, n) for key in range(pos + 1))
                equal(pairs, oracle_pairs)
                equal(pairs, q * start + q * (q + 1) // 2)
                attention_cases += 1
    equal((valid, rejected), (285, 120))
    for bad in [(-1, 0, 1), (2, 3, 4), (2, 1, -1)]:
        try:
            migration(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid length accepted')

    requests = [(n, h) for n in range(5) for h in range(n + 1)]
    batches = [()] + [(r,) for r in requests] + list(product(requests, repeat=2))
    p_sets = [(), (0,), (1, 3, 5), (0, 2, 4), (2, 4, 8)]
    k_sets = [(), (0,), (0, 2, 4), (2, 4), (0, 1, 3, 8)]
    batch_cases = 0
    for batch, ps, ks in product(batches, p_sets, k_sets):
        equal(plan(batch, ps, ks), plan(batch, ps, ks, True))
        batch_cases += 1
    equal(plan(((8, 1),), (4,), (0, 8))[0], 'suffix_overflow')
    equal(plan(((800, 128),), (512, 1024), (0, 512, 1024)),
          ('ok', (1024, 0), ((0, 800, 128, 224),)))
    equal(plan(((1600, 1280),), (512, 1024), (0, 512, 1024))[0], 'prefix_overflow')
    equal(plan(((1600, 1280),), (1024,), (0, 512, 1024)),
          ('ok', (1024, 1024), ((576, 1024, 704, 0),)))

    expected = token_oracle(6, 4, 4)
    reject_fault('omitted migration violates chosen policy',
                 lambda: equal((4, 2, 0, 2), expected))
    prefix, suffix = list(range(4)), list(range(4, 6))
    while prefix and len(suffix) < 4:
        suffix.insert(0, prefix.pop(0))
    reject_fault('front migration loses token order',
                 lambda: equal(prefix + suffix, list(range(6))))
    bad_moved = min(4, 4 - (6 - 4) + 1)
    reject_fault('overfill', lambda: equal((4 - bad_moved, 2 + bad_moved,
                                          bad_moved, 4 - 2 - bad_moved), expected))
    reject_fault('prefix capacity ignored',
                 lambda: equal(('ok', (512, 1024), ((1088, 512, 192, 0),)),
                               plan(((1600, 1280),), (512,), (0, 512, 1024), True)))
    good, unused_pairs = attention_physical(3, 1, 4, 2)
    ref = attention_reference(3, 0) + (None,)
    close_rows(good, ref)
    reject_fault('padding enters softmax denominator',
                 lambda: close_rows(attention_physical(3, 1, 4, 2, True)[0], ref))
    reject_fault('physical last row used as logits',
                 lambda: equal(good[-1], attention_reference(3, 0)[-1]))

    artifact = {'shape': (2, 3), 'stride': (3, 1), 'dtype': 'f16', 'constant': 4}
    storage = tuple(range(1, 19))
    equal(guard(artifact, dict(artifact)), True)
    equal(compiled_integer_view(artifact, storage), ((4, 8, 12), (16, 20, 24)))
    stride_request = dict(artifact, stride=(6, 1))
    constant_request = dict(artifact, constant=5)
    equal(guard(artifact, stride_request), False)
    equal(guard(artifact, constant_request), False)
    equal(guard(artifact, dict(artifact, dtype='f32')), False)
    equal(guard(artifact, stride_request, ('stride',)), True)
    equal(guard(artifact, constant_request, ('constant',)), True)
    reject_fault('stride guard omitted and old program executed',
                 lambda: equal(compiled_integer_view(artifact, storage),
                               ((4, 8, 12), (28, 32, 36))))
    reject_fault('folded constant guard omitted and old program executed',
                 lambda: equal(compiled_integer_view(artifact, storage),
                               ((5, 10, 15), (20, 25, 30))))
    branch_cases = 0
    for n in range(42):
        domains = [n == 10, n != 10 and n <= 30, n > 30]
        equal(sum(domains), 1)
        coefficient = (10, 200, n)[domains.index(True)]
        equal(tuple(x * coefficient for x in range(1, n + 1)), branch_reference(n))
        branch_cases += 1
    reject_fault('branch exclusion guard omitted',
                 lambda: equal(tuple(x * 200 for x in range(1, 11)), branch_reference(10)))
    reject_fault('physical shape replaces logical branch input',
                 lambda: equal(tuple(x * 32 for x in range(1, 21)), branch_reference(20)))

    cost_cases = 0
    for b in range(9):
        for n in range(b + 1):
            large = {(i, j) for i in range(b) for j in range(b)}
            small = {(i, j) for i in range(n) for j in range(n)}
            equal(b * b - n * n, len(large - small))
            cost_cases += 1
    reject_fault('quadratic cross terms omitted', lambda: equal((4 - 2) ** 2, 12))
    equal(512 * 388 + 512 * 513 // 2, 329984)
    equal(260 * 640 + 260 * 261 // 2, 200330)
    equal(329984 - 200330, 129654)
    equal(512 * 512 + 512 * 513 // 2, 393472)
    equal(393472 - 329984, 63488)

    accumulated, reference_count = Fraction(0), 0
    while accumulated < 31:
        accumulated += Fraction(2, 5) * 3
        reference_count += 1
    equal(break_even(31, 3, Fraction(2, 5)), reference_count)
    equal(break_even(0, 50, Fraction(1, 10)), 0)
    equal(break_even(3000000, 50, 1), 60000)
    offered = break_even(3000000, 50, Fraction(1, 10))
    equal(offered, 600000)
    verify_break_even(offered, 3000000, 50, Fraction(1, 10))
    reject_fault('hit fraction omitted',
                 lambda: verify_break_even(60000, 3000000, 50, Fraction(1, 10)))
    for args in [(3, 0, 1), (3, 2, 0), (3, 2, 2)]:
        try:
            break_even(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid amortization accepted')
    lifecycle_events = lifetime()
    reject_fault('graph reader ignored during release', lambda: lifetime(True))
    multilayer_cases, failure_cuts = multilayer_checks()

    print('scope=CPU integer/binary64/rational, toy causal decoder and finite sequential event models only')
    print('no project imports, compiler, GPU/Neuron, FP16 simulation, network, file writes or timing measurements')
    print('migration_valid=', valid, 'suffix_overflow=', rejected)
    print('batch_plans=', batch_cases, 'single_layer_visibility_cases=', attention_cases)
    print('branch_cases=', branch_cases, 'rectangular_cost_cases=', cost_cases)
    print('lifecycle_events=', lifecycle_events, 'offered_break_even=', offered)
    print('toy_three_layer_cases=', multilayer_cases, 'failure_cuts=', failure_cuts)
    print('commit_scope=immutable CPU descriptors and private tuples; no device visibility, allocator or asynchronous drain proof')
    print('detected_negative_controls=', DETECTED)


if __name__ == '__main__':
    main()
