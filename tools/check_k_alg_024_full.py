import copy
import itertools
import math
from fractions import Fraction as F

EOS = 'EOS'
LIMIT = 'LIMIT'
DETECTED = []


def must_detect(label, operation):
    try:
        operation()
    except (AssertionError, ValueError):
        DETECTED.append(label)
    else:
        raise AssertionError('negative control escaped: ' + label)


def assert_close(a, b, tolerance=1e-10):
    assert math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance), (a, b)


def beam_distribution(path):
    if not path:
        return {'a': F(4, 10), 'b': F(5, 10), EOS: F(1, 10)}
    if len(path) == 1:
        if path[0] == 'a':
            return {'a': F(5, 10), 'b': F(2, 10), EOS: F(3, 10)}
        return {'a': F(2, 10), 'b': F(4, 10), EOS: F(4, 10)}
    return {'a': F(2, 10), 'b': F(3, 10), EOS: F(5, 10)}


def path_probability(path, reason=None):
    probability = F(1)
    for position, token in enumerate(path):
        probability *= beam_distribution(path[:position])[token]
    if reason == EOS:
        probability *= beam_distribution(path)[EOS]
    return probability


def retain_finished(records, width):
    unique = {}
    for path, probability, reason in records:
        identity = (path, reason)
        if identity not in unique or probability > unique[identity]:
            unique[identity] = probability
    result = [(path, probability, reason)
              for (path, reason), probability in unique.items()]
    return sorted(result, key=lambda r: (-r[1], r[0], r[2]))[:width]


def beam_step(active, finished, width, cap, faulty_finished=False):
    if width < 1 or cap < 0:
        raise ValueError('invalid width or cap')
    live = []
    work = [] if faulty_finished else list(finished)
    ends = []
    for path, probability in active:
        if len(path) >= cap:
            if not faulty_finished:
                work.append((path, probability, LIMIT))
            continue
        distribution = beam_distribution(path)
        for token in ('a', 'b'):
            if distribution[token] == 0:
                continue
            child = path + (token,)
            candidate = (child, probability * distribution[token])
            if len(child) == cap:
                if not faulty_finished:
                    work.append((child, candidate[1], LIMIT))
            else:
                live.append(candidate)
        if distribution[EOS] > 0:
            ends.append((path, probability * distribution[EOS], EOS))
    work.extend(ends)
    live.sort(key=lambda r: (-r[1], r[0]))
    unique_live = []
    for record in live:
        if all(record[0] != old[0] for old in unique_live):
            unique_live.append(record)
    return unique_live[:width], retain_finished(work, width)


def reference_beam(active, finished, width, cap):
    # Recompute every candidate from its full conditional path, ignoring stored scores.
    candidates = []
    completions = [(path, path_probability(path, reason), reason)
                   for path, _, reason in finished]
    for path, _ in active:
        if len(path) >= cap:
            completions.append((path, path_probability(path), LIMIT))
            continue
        for event in ('a', 'b', EOS):
            if event == EOS:
                completions.append((path, path_probability(path, EOS), EOS))
            else:
                child = path + (event,)
                probability = path_probability(child)
                if len(child) == cap:
                    completions.append((child, probability, LIMIT))
                else:
                    candidates.append((child, probability))
    candidates.sort(key=lambda r: (-r[1], r[0]))
    completions.sort(key=lambda r: (-r[1], r[0], r[2]))
    answer = []
    identities = []
    for record in completions:
        identity = (record[0], record[2])
        if identity not in identities:
            identities.append(identity)
            answer.append(record)
    live = []
    for record in candidates:
        if record[0] not in [old[0] for old in live]:
            live.append(record)
    return live[:width], answer[:width]


def check_beam():
    active = [(path, path_probability(path))
              for path in [('a',), ('b',), ('b', 'b', 'b')]]
    finished = [((), path_probability((), EOS), EOS)]
    expected = reference_beam(active, finished, 4, 3)
    actual = beam_step(active, finished, 4, 3)
    assert actual == expected
    assert ((), F(1, 10), EOS) in actual[1]
    assert (('b', 'b', 'b'), F(6, 100), LIMIT) in actual[1]
    assert (('b',), F(2, 10), EOS) in actual[1]
    must_detect('beam drops old and length-finished records',
                lambda: assert_equal(beam_step(active, finished, 4, 3, True), expected))
    for width in (1, 2, 4):
        for cap in range(4):
            live, done = [((), F(1))], []
            for _ in range(cap + 2):
                expected = reference_beam(live, done, width, cap)
                actual = beam_step(live, done, width, cap)
                assert actual == expected
                live, done = actual
                assert len(live) <= width and len(done) <= width
                assert len({(r[0], r[2]) for r in done}) == len(done)
                if not live:
                    break
            assert not live
    repeated = finished + finished
    assert beam_step([], repeated, 4, 3)[1] == finished
    assert beam_step([], [], 1, 0) == ([], [])
    must_detect('invalid beam width', lambda: beam_step([], [], 0, 3))
    root = {'a': F(4, 10), 'b': F(6, 10)}
    following = {'a': {'a': F(95, 100), 'b': F(5, 100)},
                 'b': {'a': F(1, 2), 'b': F(1, 2)}}
    all_paths = [(a + b, root[a] * following[a][b])
                 for a, b in itertools.product(('a', 'b'), repeat=2)]
    all_paths.sort(key=lambda r: (-r[1], r[0]))
    first = sorted(root, key=lambda t: (-root[t], t))[0]
    second = sorted(following[first], key=lambda t: (-following[first][t], t))[0]
    assert first + second == 'ba'
    assert all_paths[0] == ('aa', F(38, 100))
    assert root[first] * following[first][second] == F(3, 10)
    local = {'a': {'a': F(9, 10), 'b': F(1, 10)},
             'b': {'a': F(1, 2), 'b': F(1, 2)}}
    initial = {'a': F(1, 100), 'b': F(99, 100)}
    cumulative = sorted(((a + b, initial[a] * local[a][b])
                         for a, b in itertools.product(('a', 'b'), repeat=2)),
                        key=lambda r: (-r[1], r[0]))[:2]
    faulty = sorted(((a + b, local[a][b])
                     for a, b in itertools.product(('a', 'b'), repeat=2)),
                    key=lambda r: (-r[1], r[0]))[:2]
    must_detect('local-last-token ranking',
                lambda: assert_equal([r[0] for r in faulty], [r[0] for r in cumulative]))
    print('beam: old/EOS/LIMIT merge, cross-round enumeration, ties, cap=0; K=1 ba=0.3 < aa=0.38')


def assert_equal(actual, expected):
    assert actual == expected, (actual, expected)


ROOT = (1, 2)
NODES = {'A': (None, 3), 'B': (None, 7), 'C': ('A', 4),
         'D': ('A', 6), 'E': ('B', 8)}


def node_chain(node):
    result = []
    while node is not None:
        result.append(node)
        node = NODES[node][0]
    return tuple(reversed(result))


def attention_output(value, rows, layer):
    scores = [0.07 * layer * value * key for key, _ in rows]
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    total = sum(weights)
    weighted = sum(weight * row[1] for weight, row in zip(weights, rows)) / total
    return value + weighted + 0.01 * value * value


def logits(value):
    return (0.2 * value, -0.1 * value, 0.03 * value + 0.1)


def dense_forward(tokens):
    # Independent full-sequence causal evaluation; no tree, parent or row map.
    hidden = [token + 0.1 * position for position, token in enumerate(tokens)]
    cache = []
    for layer in (1, 2):
        rows = tuple((0.11 * value, 0.13 * layer * value) for value in hidden)
        cache.append(rows)
        hidden = [attention_output(value, rows[:position + 1], layer)
                  for position, value in enumerate(hidden)]
    return {'kv': tuple(cache), 'logits': logits(hidden[-1])}


def tree_forward(fault=None, order=None):
    order = tuple(NODES) if order is None else tuple(order)
    root_cache = dense_forward(ROOT)['kv']
    hidden = {}
    for physical_row, node in enumerate(order):
        position = len(ROOT) + len(node_chain(node)) - 1
        if fault == 'flat_position':
            position = len(ROOT) + physical_row
        hidden[node] = NODES[node][1] + 0.1 * position
    candidate_layers = []
    for layer in (1, 2):
        rows = {node: (0.11 * hidden[node], 0.13 * layer * hidden[node])
                for node in order}
        candidate_layers.append(rows)
        next_hidden = {}
        for node in order:
            visible = list(node_chain(node))
            if fault == 'sibling' and node == 'C':
                visible.append('B')
            inputs = root_cache[layer - 1] + tuple(rows[ancestor] for ancestor in visible)
            next_hidden[node] = attention_output(hidden[node], inputs, layer)
        hidden = next_hidden
    bundles = {}
    for node in order:
        path = node_chain(node)
        cache = tuple(root_cache[layer] + tuple(candidate_layers[layer][a] for a in path)
                      for layer in range(2))
        bundles[node] = {'kv': cache, 'logits': logits(hidden[node]), 'ready': True}
    return bundles


def compare_cache(actual, expected):
    assert len(actual) == len(expected)
    for actual_layer, expected_layer in zip(actual, expected):
        assert len(actual_layer) == len(expected_layer)
        for actual_row, expected_row in zip(actual_layer, expected_layer):
            for a, b in zip(actual_row, expected_row):
                assert_close(a, b)


def check_tree_against_paths(bundles):
    for node in NODES:
        tokens = ROOT + tuple(NODES[a][1] for a in node_chain(node))
        reference = dense_forward(tokens)
        compare_cache(bundles[node]['kv'], reference['kv'])
        for actual, expected in zip(bundles[node]['logits'], reference['logits']):
            assert_close(actual, expected)
        assert bundles[node]['ready']


def check_tree():
    correct = tree_forward()
    check_tree_against_paths(correct)
    check_tree_against_paths(tree_forward(order=reversed(tuple(NODES))))
    must_detect('sibling KV changes tree logits',
                lambda: check_tree_against_paths(tree_forward('sibling')))
    must_detect('flat row used as logical position',
                lambda: check_tree_against_paths(tree_forward('flat_position')))
    assert node_chain('C') == ('A', 'C')
    print('tree: two-layer scalar causal attention matches all five independent paths and row permutation')
    return correct


ORDER = (3, 4, 6, 7, 8, 9, EOS)


def target(prefix):
    # Separate finite-prefix probability model, not logits of the scalar attention model.
    table = {
        ROOT: {3: F(6, 10), 7: F(2, 10), EOS: F(2, 10)},
        ROOT + (3,): {4: F(1, 2), 6: F(1, 4), EOS: F(1, 4)},
        ROOT + (7,): {8: F(2, 5), EOS: F(3, 5)},
        ROOT + (3, 4): {9: F(3, 4), EOS: F(1, 4)},
    }
    return table.get(prefix, {3: F(1, 5), 4: F(3, 10), EOS: F(1, 2)})


def validate_distribution(distribution):
    assert distribution and all(v >= 0 for v in distribution.values())
    assert sum(distribution.values(), F(0)) == 1


def choose(distribution, value):
    validate_distribution(distribution)
    assert 0 <= value < 1
    cumulative = F(0)
    for token in ORDER:
        cumulative += distribution.get(token, F(0))
        if value < cumulative:
            return token
    raise AssertionError('categorical mass missing')


def verify(old, proposals, tape, budget):
    if budget < 0:
        raise ValueError('negative budget')
    cursor = old['rng']
    selected = []
    prefix = old['tokens']
    node = None
    score = old['score']
    count, checksum = old['grammar']

    def draw():
        nonlocal cursor
        if cursor >= len(tape):
            raise ValueError('exhausted deterministic tape')
        value = tape[cursor]
        cursor += 1
        return value

    def receipt(event):
        return {'selected': tuple(selected), 'event': event, 'rng': cursor,
                'score': score, 'grammar': (count, checksum),
                'base': old['tokens'], 'generation': old['generation']}

    while len(selected) < budget:
        p = target(prefix)
        validate_distribution(p)
        if node in proposals:
            q = proposals[node]
            validate_distribution(q)
            children = {token: name for name, (parent, token) in NODES.items()
                        if parent == node}
            assert all(token == EOS or token in children for token in q)
            candidate = choose(q, draw())
            uniform = draw()
            acceptance = min(F(1), p.get(candidate, F(0)) / q[candidate])
            if uniform < acceptance:
                if candidate != EOS:
                    selected.append(children[candidate])
                    score += math.log(float(p[candidate]))
                    count += 1
                    checksum += candidate
                    prefix += (candidate,)
                    node = children[candidate]
                    continue
                event = EOS
            else:
                residual = {token: max(F(0), p.get(token, F(0)) - q.get(token, F(0)))
                            for token in ORDER}
                total = sum(residual.values(), F(0))
                assert total > 0, 'unreachable zero-residual rejection'
                residual = {token: mass / total for token, mass in residual.items()}
                event = choose(residual, draw())
        else:
            event = choose(p, draw())
        score += math.log(float(p[event]))
        if event != EOS:
            count += 1
            checksum += event
        return receipt(event)
    return receipt(LIMIT)


def make_old():
    return {'tokens': ROOT, 'kv': dense_forward(ROOT)['kv'], 'k': len(ROOT),
            'pending': None, 'rng': 0, 'score': 0.0, 'grammar': (0, 0),
            'generation': 0, 'status': 'ACTIVE'}


def prepare(old, result, bundles, fault=None):
    if result['base'] != old['tokens'] or result['generation'] != old['generation']:
        raise ValueError('stale receipt')
    selected = result['selected']
    parent = None
    for node in selected:
        if NODES[node][0] != parent or not bundles[node]['ready']:
            raise ValueError('invalid path or incomplete state')
        parent = node
    ordinary = tuple(NODES[node][1] for node in selected)
    event = result['event']
    tokens = old['tokens'] + ordinary
    pending = None
    if event not in (EOS, LIMIT):
        tokens += (event,)
        pending = event
    cache = old['kv'] if not selected else bundles[selected[-1]]['kv']
    answer = dict(old)
    answer.update(tokens=tokens, kv=cache, k=len(old['tokens']) + len(selected),
                  pending=pending, rng=result['rng'], score=result['score'],
                  grammar=result['grammar'], generation=old['generation'] + 1,
                  status='ACTIVE' if pending is not None else event)
    if fault == 'consume_bonus' and pending is not None:
        # Actually consume the pending token and publish the wrong state boundary.
        answer['kv'] = dense_forward(tokens)['kv']
        answer['k'] = len(tokens)
        answer['pending'] = None
    if fault == 'wrong_branch':
        answer['kv'] = bundles['E']['kv']
    if fault == 'old_rng':
        answer['rng'] = old['rng']
    if fault == 'proposal_score':
        answer['score'] = old['score']
    if fault == 'old_grammar':
        answer['grammar'] = old['grammar']
    return answer


def transaction(old, result, bundles, fail_at=None, fault=None):
    try:
        if fail_at == 'allocation':
            raise ValueError('injected allocation failure')
        candidate = prepare(old, result, bundles, fault)
        if fail_at == 'ready':
            raise ValueError('injected pre-publication failure')
        return candidate
    except ValueError:
        return old


def check_commit_reference(old, result, actual, expected_rng):
    # Dense replay builds expected state solely from the selected logical token path.
    generated = tuple(NODES[node][1] for node in result['selected'])
    computed_tokens = old['tokens'] + generated
    event = result['event']
    output_tokens = computed_tokens
    if event not in (EOS, LIMIT):
        output_tokens += (event,)
    assert actual['tokens'] == output_tokens
    assert actual['k'] == len(computed_tokens)
    compare_cache(actual['kv'], dense_forward(computed_tokens)['kv'])
    assert actual['pending'] == (event if event not in (EOS, LIMIT) else None)
    assert actual['rng'] == expected_rng
    expected_grammar = (old['grammar'][0] + len(output_tokens) - len(old['tokens']),
                        old['grammar'][1] + sum(output_tokens[len(old['tokens']):]))
    assert actual['grammar'] == expected_grammar
    prefix = old['tokens']
    probability = F(1)
    for token in generated:
        probability *= target(prefix)[token]
        prefix += (token,)
    if event != LIMIT:
        probability *= target(prefix)[event]
    assert_close(actual['score'], old['score'] + math.log(float(probability)))
    assert actual['status'] == ('ACTIVE' if event not in (EOS, LIMIT) else event)
    assert actual['generation'] == old['generation'] + 1


def check_verify_and_commit(bundles):
    old = make_old()
    frozen = copy.deepcopy(old)
    proposals = {None: {3: F(1)}, 'A': {4: F(1)}}
    tape = tuple(map(F, ('0.1', '0.1', '0.3', '0.1', '0.2')))
    result = verify(old, proposals, tape, 3)
    assert result['selected'] == ('A', 'C') and result['event'] == 9
    assert result['rng'] == 5
    committed = transaction(old, result, bundles)
    check_commit_reference(old, result, committed, 5)
    assert_close(math.exp(committed['score']), 0.225)
    for fault in ('consume_bonus', 'wrong_branch', 'old_rng', 'proposal_score', 'old_grammar'):
        must_detect('commit ' + fault,
                    lambda fault=fault: check_commit_reference(
                        old, result, transaction(old, result, bundles, fault=fault), 5))
    for stage in ('allocation', 'ready'):
        assert transaction(old, result, bundles, fail_at=stage) is old
        assert old == frozen
    incomplete = copy.deepcopy(bundles)
    incomplete['C']['ready'] = False
    assert transaction(old, result, incomplete) is old
    stale = dict(result, generation=-1)
    assert transaction(old, stale, bundles) is old
    invalid = dict(result, selected=('A', 'E'))
    assert transaction(old, invalid, bundles) is old
    cases = [
        (tuple(map(F, ('0.1', '0.9', '0.1'))), 3, (), 7, 3),
        (tuple(map(F, ('0.1', '0.1', '0.3', '0.1', '0.9'))), 3, ('A', 'C'), EOS, 5),
        (tape, 1, ('A',), LIMIT, 2),
        ((), 0, (), LIMIT, 0),
    ]
    for values, budget, selected, event, cursor in cases:
        receipt = verify(old, proposals, values, budget)
        assert receipt['selected'] == selected and receipt['event'] == event
        answer = transaction(old, receipt, bundles)
        check_commit_reference(old, receipt, answer, cursor)
        assert old == frozen
    leaf = verify(old, {}, (F(95, 100),), 3)
    assert leaf['event'] == EOS and leaf['selected'] == ()
    check_commit_reference(old, leaf, transaction(old, leaf, bundles), 1)
    direct_eos = verify(old, {None: {EOS: F(1)}}, (F(1, 10), F(1, 10)), 3)
    assert direct_eos['event'] == EOS
    check_commit_reference(old, direct_eos, transaction(old, direct_eos, bundles), 2)
    same = {None: target(ROOT)}
    same_result = verify(old, same, (F(1, 10), F(99, 100)), 1)
    assert same_result['selected'] == ('A',) and same_result['event'] == LIMIT
    check_commit_reference(old, same_result, transaction(old, same_result, bundles), 2)
    print('verification/commit: AC+9 gives probability 0.225, k=4, output=5, RNG cursor=5')
    print('verification/commit: rejection-to-precomputed-sibling stays pending; EOS, zero budget, exact cap, p=Q, stale/incomplete and pre-publish failure covered')


def speculative_mass(p, q, omit_residual=False):
    validate_distribution(p)
    validate_distribution(q)
    vocabulary = set(p) | set(q)
    accepted = {token: F(0) for token in vocabulary}
    rejected = F(0)
    for token, probability in q.items():
        if probability == 0:
            continue
        acceptance = min(F(1), p.get(token, F(0)) / probability)
        accepted[token] += probability * acceptance
        rejected += probability * (1 - acceptance)
    if rejected and not omit_residual:
        residual = {token: max(F(0), p.get(token, F(0)) - q.get(token, F(0)))
                    for token in vocabulary}
        total = sum(residual.values(), F(0))
        assert total > 0
        for token in vocabulary:
            accepted[token] += rejected * residual[token] / total
    return accepted


def compare_mass(actual, expected):
    assert sum(actual.values(), F(0)) == 1
    for token in set(actual) | set(expected):
        assert actual.get(token, F(0)) == expected.get(token, F(0))


def check_mass():
    cases = [({'A': F(9, 10), 'B': F(1, 10)}, {'A': F(1, 10), 'B': F(9, 10)}),
             ({'A': F(2, 5), 'B': F(3, 5)}, {'A': F(1), 'B': F(0)}),
             ({'A': F(1, 5), 'B': F(3, 10), 'C': F(1, 2)},
              {'A': F(1, 5), 'B': F(3, 10), 'C': F(1, 2)}),
             ({'A': F(1), 'B': F(0)}, {'A': F(0), 'B': F(1)})]
    for p, q in cases:
        compare_mass(speculative_mass(p, q), p)
    p, q = cases[0]
    argmax = {'A': F(1), 'B': F(0)}
    must_detect('argmax target replaces sampling', lambda: compare_mass(argmax, p))
    must_detect('discard rejection mass', lambda: compare_mass(speculative_mass(p, q, True), p))
    print('probability: exact Fraction mass equals p for reverse bias, q=0, p=q and disjoint support')


def check_random_contract():
    assignments = list(itertools.product((0, 1), repeat=2))
    independent = {(a, b): F(assignments.count((a, b)), 4)
                   for a, b in assignments}
    assert all(mass == F(1, 4) for mass in independent.values())
    cloned = {(0, 0): F(1, 2), (1, 1): F(1, 2)}
    must_detect('cloned random position correlates samples',
                lambda: assert_equal(cloned, independent))
    tape = (F(1, 10), F(8, 10))
    row_ab = dict(zip(('A', 'B'), tape))
    row_ba = dict(zip(('B', 'A'), tape))
    assert row_ab['A'] != row_ba['A']
    keyed = {'A': tape[0], 'B': tape[1]}
    assert {name: keyed[name] for name in ('B', 'A')} == keyed
    print('RNG scope: finite independent-draw allocation versus cloned position; row reassignment changes seeded trajectory, not a statistical PRNG-quality proof')


def physical_pages(length, page, tails):
    # Synthetic immutable-root page contents, with a private copy of each appended tail.
    root_pages = tuple(('root', index) for index in range((length + page - 1) // page))
    live = set(root_pages)
    root_tail = tuple(range((length // page) * page, length))
    full_count = length // page
    views = []
    for branch, tail in enumerate(tails):
        if tail == 0:
            view = root_pages
        else:
            payload = root_tail + tuple(('new', branch, index) for index in range(tail))
            private = tuple(('branch', branch, offset // page)
                            for offset in range(0, len(payload), page))
            view = root_pages[:full_count] + private
        live.update(view)
        views.append(view)
    return live, views


def check_capacity():
    page, length, branches, tail = 4, 1024, 4, 16
    full = branches * ((length + tail + page - 1) // page)
    physical, _ = physical_pages(length, page, [tail] * branches)
    shared = len(physical)
    assert full == 1040 and shared == 272
    token_bytes = 128 * 1024
    assert full * page * token_bytes == 520 * 1024 * 1024
    assert shared * page * token_bytes == 136 * 1024 * 1024
    faulty = branches * ((length + page - 1) // page + (tail + page - 1) // page)
    must_detect('shared root counted per branch', lambda: assert_equal(faulty, shared))
    for length in range(10):
        for tails in ((0, 0), (1, 3), (0, 5), (4, 8)):
            objects, _ = physical_pages(length, page, tails)
            quotient, remainder = divmod(length, page)
            if remainder:
                expected = quotient + 1 + sum((remainder + e + page - 1) // page
                                               for e in tails if e > 0)
            else:
                expected = quotient + sum((e + page - 1) // page for e in tails)
            assert len(objects) == expected
    objects, _ = physical_pages(6, 4, (1, 3))
    assert len(objects) == 5
    faulty_unaligned = (6 + 3) // 4 + (1 + 3) // 4 + (3 + 3) // 4
    must_detect('unaligned COW tail omitted', lambda: assert_equal(faulty_unaligned, len(objects)))
    print('capacity: aligned full/shared=1040/272 pages, 520/136 MiB; L=6 P=4 tails=1,3 with old root retained needs 5 unique pages')


def main():
    check_beam()
    bundles = check_tree()
    check_mass()
    check_verify_and_commit(bundles)
    check_random_contract()
    check_capacity()
    print('detected negative controls:', len(DETECTED), ', '.join(DETECTED))
    print('scope: deterministic CPU logical models only; no real checkpoint, GPU, asynchronous page lifetime, throughput, empirical sampling-frequency or Agent-effect claim')


if __name__ == '__main__':
    main()
