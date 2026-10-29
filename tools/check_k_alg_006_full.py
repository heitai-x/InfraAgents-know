from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from itertools import permutations
import math

T = (0, 1, 2)
ZERO = F(0)
ONE = F(1)


def validate(p):
    assert len(p) == len(T)
    assert all(x >= 0 for x in p)
    assert sum(p) == ONE


def residual(p, q):
    raw = tuple(max(ZERO, a - b) for a, b in zip(p, q))
    z = sum(raw)
    assert z > 0, 'unreachable or inconsistent rejection branch'
    return tuple(a / z for a in raw)


def single_step(p, q, recorded=None, fault=None):
    validate(p)
    validate(q)
    used = q if recorded is None else recorded
    validate(used)
    out = defaultdict(F)
    rejected = ZERO
    residual_calls = 0
    for x in T:
        if q[x] == 0:
            continue
        assert used[x] > 0
        a = min(ONE, p[x] / used[x])
        if fault == 'threshold':
            a = ONE if p[x] >= used[x] else ZERO
        out[x] += q[x] * a
        if a < 1:
            rejected += q[x] * (1 - a)
            r = p if fault == 'direct_residual' else residual(p, used)
            residual_calls += 1
            for y in T:
                out[y] += q[x] * (1 - a) * r[y]
    return tuple(out[x] for x in T), rejected, residual_calls


def target(h):
    code = sum((i + 1) * (x + 1) for i, x in enumerate(h))
    w = (1 + (len(h) + code) % 5,
         1 + (2 * code + 1) % 7,
         1 + (3 * len(h) + code + 2) % 6)
    return tuple(F(x, sum(w)) for x in w)


def proposal(h):
    if len(h) % 3 == 0:
        return (ONE, ZERO, ZERO)
    if len(h) % 3 == 1:
        return (F(1, 2), ZERO, F(1, 2))
    return (F(1, 6), F(1, 2), F(1, 3))


@lru_cache(None)
def round_distribution(prefix, gamma, fault=None):
    out = defaultdict(F)

    def walk(accepted, weight):
        h = prefix + accepted
        if len(accepted) == gamma:
            for t, pt in enumerate(target(h)):
                out[accepted + (t,)] += weight * pt
            return
        p = target(prefix) if fault == 'wrong_prefix' and accepted else target(h)
        q = proposal(h)
        for x in T:
            if q[x] == 0:
                continue
            a = min(ONE, p[x] / q[x])
            if fault == 'threshold_second' and len(accepted) == 1:
                a = ONE if p[x] >= q[x] else ZERO
            if a < 1:
                r = p if fault == 'direct_residual' else residual(p, q)
                for t, rt in enumerate(r):
                    if rt:
                        out[accepted + (t,)] += weight * q[x] * (1 - a) * rt
            if a:
                walk(accepted + (x,), weight * q[x] * a)

    walk((), ONE)
    return dict(out)


def direct_sequences(prefix, horizon):
    if horizon == 0:
        return {(): ONE}
    out = {}
    for x, px in enumerate(target(prefix)):
        for tail, prob in direct_sequences(prefix + (x,), horizon - 1).items():
            out[(x,) + tail] = px * prob
    return out


def speculative_sequences(prefix, horizon, gamma, fault=None):
    @lru_cache(None)
    def advance(h, remaining):
        if remaining == 0:
            return {(): ONE}
        out = defaultdict(F)
        for chunk, pc in round_distribution(h, gamma, fault).items():
            if len(chunk) >= remaining:
                out[chunk[:remaining]] += pc
            else:
                for tail, pt in advance(h + chunk, remaining - len(chunk)).items():
                    out[chunk + tail] += pc * pt
        return dict(out)
    return advance(prefix, horizon)


def acceptance_survival(prefix, gamma):
    frontier = {(): ONE}
    survival = []
    for _ in range(gamma):
        nxt = defaultdict(F)
        for path, weight in frontier.items():
            p = target(prefix + path)
            q = proposal(prefix + path)
            for x in T:
                mass = min(p[x], q[x])
                if mass:
                    nxt[path + (x,)] += weight * mass
        frontier = dict(nxt)
        survival.append(sum(frontier.values()))
    return survival


def ordered_children(p, children, fault=False):
    r = p
    alive = ONE
    out = defaultdict(F)
    for c in children:
        a = r[c]
        out[c] += alive * a
        alive *= 1 - a
        if alive == 0:
            break
        if not fault:
            delta = tuple(ONE if x == c else ZERO for x in T)
            r = residual(r, delta)
    if alive:
        for x, rx in enumerate(r):
            out[x] += alive * rx
    return tuple(out[x] for x in T)


def probability_checks():
    grid = [tuple(F(x, 4) for x in (a, b, 4 - a - b))
            for a in range(5) for b in range(5 - a)]
    pairs = 0
    for p in grid:
        for q in grid:
            got, reject, calls = single_step(p, q)
            assert got == p, (p, q, got)
            assert reject == 1 - sum(min(a, b) for a, b in zip(p, q))
            if p == q:
                assert reject == 0 and calls == 0
            pairs += 1
    boundary = [
        ('missing_proposal_support', (F(1, 2), F(1, 2), ZERO), (ONE, ZERO, ZERO)),
        ('disjoint_support', (ZERO, ONE, ZERO), (ONE, ZERO, ZERO)),
        ('equal_distributions', (F(1, 2), F(1, 2), ZERO), (F(1, 2), F(1, 2), ZERO))
    ]
    for name, p, q in boundary:
        got, reject, calls = single_step(p, q)
        assert got == p
        if name == 'equal_distributions':
            assert calls == 0
        print(name, 'output=', got, 'rejection_mass=', reject, 'residual_calls=', calls)
    p = (F(1, 2), F(1, 2), ZERO)
    q = (ONE, ZERO, ZERO)
    recorded = (F(1, 2), F(1, 2), ZERO)
    assert single_step(p, q, recorded=recorded)[0] != p
    assert single_step(p, q, fault='direct_residual')[0] != p
    assert single_step(p, q, fault='threshold')[0] != p
    cases = 0
    for prefix in ((), (1,), (2, 0)):
        reference = direct_sequences(prefix, 3)
        for gamma in range(4):
            got = speculative_sequences(prefix, 3, gamma)
            assert got == reference, ('complete_sequence', prefix, gamma)
            chunks = round_distribution(prefix, gamma)
            assert sum(chunks.values()) == 1
            expected = sum(F(len(chunk)) * mass for chunk, mass in chunks.items())
            independent_length = 1 + sum(acceptance_survival(prefix, gamma))
            assert expected == independent_length
            cases += 1
    detected = []
    reference = direct_sequences((), 3)
    for fault in ('threshold_second', 'direct_residual', 'wrong_prefix'):
        bad = speculative_sequences((), 3, 2, fault)
        assert bad != reference, ('undetected_probability_fault', fault)
        assert sum(bad.values()) == 1
        detected.append(fault)
    tree_cases = 0
    for p in grid:
        for width in range(4):
            for children in permutations(T, width):
                assert ordered_children(p, children) == p
                tree_cases += 1
    assert ordered_children((F(1, 2), F(1, 3), F(1, 6)), (0, 1), True) != (F(1, 2), F(1, 3), F(1, 6))
    print('exact rational single-step pairs=', pairs, 'complete-sequence cases=', cases)
    print('ordered deterministic tree-child cases=', tree_cases)
    print('probability negative controls detected=', detected + ['recorded_q_mismatch', 'unchanged_tree_residual'])


def softmax(values):
    peak = max(values)
    e = [math.exp(x - peak) for x in values]
    total = sum(e)
    return tuple(x / total for x in e)


def embed(token, position):
    return (token * 0.19 + position * 0.07,
            math.sin(token * 0.4) + position * 0.03)


def project(h, layer):
    a, b = h
    return ((a * 0.13 + b * 0.07 + layer * 0.01, b * 0.11 - a * 0.03),
            (a * 0.09 - b * 0.02, b * 0.08 + a * 0.04),
            (a * 0.30 + b * 0.10, b * 0.25 - a * 0.07))


def attend_and_update(h, visible_kv, layer):
    q = project(h, layer)[0]
    weights = softmax([(q[0] * k[0] + q[1] * k[1]) / math.sqrt(2)
                       for k, v in visible_kv])
    av = [sum(w * kv[1][d] for w, kv in zip(weights, visible_kv)) for d in range(2)]
    u = (h[0] + av[0], h[1] + av[1])
    return (u[0] + 0.17 * math.tanh(u[1]), u[1] + 0.11 * math.tanh(u[0]))


def head(h):
    a, b = h
    return softmax((0.7 * a - 0.2 * b, -0.3 * a + 0.8 * b, 0.1 * a + 0.2 * b))


def serial_forward(tokens):
    # Independent full-sequence causal reference: no tree masks or packed row map.
    h = [embed(t, i) for i, t in enumerate(tokens)]
    layers = []
    for layer in range(2):
        projections = [project(x, layer) for x in h]
        kv = [(p[1], p[2]) for p in projections]
        nxt = [attend_and_update(x, kv[:i + 1], layer) for i, x in enumerate(h)]
        layers.append({'kv': kv, 'h': nxt})
        h = nxt
    return {'layers': layers, 'dist': [head(x) for x in h]}


PREFIX = (2, 3)
PARENT = {'A': None, 'B': None, 'C': 'A', 'D': 'C'}
TOKEN = {'A': 7, 'B': 13, 'C': 5, 'D': 11}


def node_path(v):
    path = []
    while v is not None:
        path.append(v)
        v = PARENT[v]
    return tuple(reversed(path))


def packed_forward(order, fault=None):
    base = serial_forward(PREFIX)
    row = {v: i for i, v in enumerate(order)}
    h = [embed(TOKEN[v], len(PREFIX) + len(node_path(v)) - 1) for v in order]
    visibility = {}
    for v in order:
        ids = list(range(len(PREFIX))) + [len(PREFIX) + row[a] for a in node_path(v)]
        if fault == 'omit_self':
            ids.remove(len(PREFIX) + row[v])
        if fault == 'sibling_leak' and v == 'C':
            ids.append(len(PREFIX) + row['B'])
        visibility[v] = tuple(ids)
    layers = []
    for layer in range(2):
        projections = [project(x, layer) for x in h]
        kv = [(p[1], p[2]) for p in projections]
        pool = base['layers'][layer]['kv'] + kv
        nxt = [attend_and_update(h[row[v]], [pool[i] for i in visibility[v]], layer)
               for v in order]
        layers.append({'kv': kv, 'h': nxt})
        h = nxt
    return {'row': row, 'layers': layers, 'dist': [head(x) for x in h],
            'root': base['dist'][-1], 'visibility': visibility}


def max_difference(a, b):
    if isinstance(a, (tuple, list)):
        assert isinstance(b, (tuple, list)) and len(a) == len(b)
        return max((max_difference(x, y) for x, y in zip(a, b)), default=0.0)
    return abs(a - b)


def tree_errors(result, wrong_validation_row=False):
    errors = []
    row = result['row']
    for v in row:
        path = node_path(v)
        tokens = PREFIX + tuple(TOKEN[x] for x in path)
        reference = serial_forward(tokens)
        end = len(tokens) - 1
        if max_difference(result['dist'][row[v]], reference['dist'][-1]) > 1e-12:
            errors.append(('node_output', v))
        for layer in range(2):
            observed = (result['layers'][layer]['kv'][row[v]], result['layers'][layer]['h'][row[v]])
            expected = (reference['layers'][layer]['kv'][end], reference['layers'][layer]['h'][end])
            if max_difference(observed, expected) > 1e-12:
                errors.append(('layer_state', v, layer))
        if wrong_validation_row:
            verification = result['dist'][row[v]]
        elif PARENT[v] is None:
            verification = result['root']
        else:
            verification = result['dist'][row[PARENT[v]]]
        parent_tokens = PREFIX + tuple(TOKEN[x] for x in path[:-1])
        expected_before = serial_forward(parent_tokens)['dist'][-1]
        if max_difference(verification, expected_before) > 1e-12:
            errors.append(('verification_distribution', v))
    return errors


def tree_forward_checks():
    for order in permutations(('A', 'B', 'C', 'D')):
        result = packed_forward(order)
        assert not tree_errors(result), (order, tree_errors(result))
    order = ('D', 'B', 'A', 'C')
    good = packed_forward(order)
    observations = []
    for fault in ('sibling_leak', 'omit_self'):
        bad = packed_forward(order, fault)
        errors = tree_errors(bad)
        assert errors, ('undetected_tree_fault', fault)
        delta = max(max_difference(good['dist'][good['row'][v]], bad['dist'][bad['row'][v]])
                    for v in order)
        assert delta > 1e-6, (fault, delta)
        observations.append((fault, len(errors), delta))
    alignment_errors = tree_errors(good, wrong_validation_row=True)
    assert alignment_errors and all(e[0] == 'verification_distribution' for e in alignment_errors)
    observations.append(('node_row_instead_of_parent', len(alignment_errors)))
    # Collect the accepted A,C path from every packed layer, then compare to a fresh serial model.
    accepted = ('A', 'C')
    expected = serial_forward(PREFIX + tuple(TOKEN[x] for x in accepted))
    base = serial_forward(PREFIX)
    for layer in range(2):
        committed = base['layers'][layer]['kv'] + [good['layers'][layer]['kv'][good['row'][v]] for v in accepted]
        assert max_difference(committed, expected['layers'][layer]['kv']) < 1e-12
        wrong = base['layers'][layer]['kv'] + [good['layers'][layer]['kv'][good['row'][v]] for v in ('A', 'B')]
        assert max_difference(wrong, expected['layers'][layer]['kv']) > 1e-6
    print('packed CPU toy forward: 24 physical row permutations match independent serial paths')
    print('tree output/state negative controls=', observations)
    print('all-layer accepted-path KV collection matches fresh serial reference; A,B substitution detected')


def closed_state(tokens):
    # Closed-form reference, independent of incremental checkpoint construction.
    return tuple(F((layer + 1) * (base ** len(tokens)) +
                   sum(token * (base ** (len(tokens) - 1 - i)) for i, token in enumerate(tokens)))
                 for layer, base in enumerate((5, 7)))


def checkpoints(prefix, candidates):
    states = [F(1), F(2)]
    for t in prefix:
        states = [base * s + t for base, s in zip((5, 7), states)]
    result = [tuple(states)]
    for t in candidates:
        states = [base * s + t for base, s in zip((5, 7), states)]
        result.append(tuple(states))
    return result


def state_errors(prefix, candidates, accepted, extra, terminal, observed):
    errors = []
    kept = prefix + candidates[:accepted]
    known = kept if terminal else kept + (extra,)
    if observed['known'] != known:
        errors.append('known')
    if observed['kv_tokens'] != kept:
        errors.append('kv_visibility')
    if observed['K'] != len(kept):
        errors.append('kv_boundary')
    if observed['state'] != closed_state(kept):
        errors.append('recursive_checkpoint')
    if observed['grammar'] != known:
        errors.append('grammar_path')
    if not terminal and len(known) != observed['K'] + 1:
        errors.append('pending_final_token')
    return errors


def state_checks():
    count = 0
    mutations = set()
    for m in (1, 2, 5):
        prefix = tuple(range(2, 2 + m))
        for gamma in range(4):
            candidates = tuple(range(101, 101 + gamma))
            snaps = checkpoints(prefix, candidates)
            for n in range(gamma + 1):
                for terminal in ((False, True) if n else (False,)):
                    extra = 999
                    kept = prefix + candidates[:n]
                    known = kept if terminal else kept + (extra,)
                    valid = {'known': known, 'kv_tokens': kept, 'K': len(kept),
                             'state': snaps[n], 'grammar': known}
                    assert not state_errors(prefix, candidates, n, extra, terminal, valid)
                    count += 1
                    if not terminal:
                        bad = dict(valid, K=len(known), kv_tokens=known)
                        assert 'kv_boundary' in state_errors(prefix, candidates, n, extra, terminal, bad)
                        mutations.add('final_token_counted_as_KV')
                    if n < gamma:
                        bad = dict(valid, state=snaps[-1])
                        assert 'recursive_checkpoint' in state_errors(prefix, candidates, n, extra, terminal, bad)
                        mutations.add('final_checkpoint_used_after_partial_acceptance')
                        bad_known = prefix + candidates + (() if terminal else (extra,))
                        bad = dict(valid, known=bad_known, grammar=bad_known)
                        assert 'known' in state_errors(prefix, candidates, n, extra, terminal, bad)
                        mutations.add('rejected_candidates_published')
                    bad = dict(valid, grammar=known + (777,))
                    assert 'grammar_path' in state_errors(prefix, candidates, n, extra, terminal, bad)
                    mutations.add('grammar_not_rolled_back')
    print('state boundary cases=', count, 'including zero candidates, zero acceptance, full acceptance and accepted terminal token')
    print('state negative controls detected=', sorted(mutations))


class UndefinedTargetDistribution(ValueError):
    def __init__(self, history, reason):
        self.history = history
        self.reason = reason
        super().__init__((history, reason))


def direct_grammar_trajectory(words, initial, count, target_stops=()):
    # Independent direct-target reference: scans complete words at each step.
    # It never uses proposal records, packed rows, truncation or acceptance.
    h = initial
    out = []
    for _ in range(count):
        if 2 in h or h in target_stops:
            break
        completions = [w for w in words if w[:len(h)] == h]
        options = set()
        for w in completions:
            options.add(2 if len(w) == len(h) else w[len(h)])
        if not options:
            raise UndefinedTargetDistribution(h, 'direct_dead_prefix')
        chosen = min(options)
        out.append(chosen)
        h += (chosen,)
    return tuple(out)


def grammar_round(words, draft_tokens, gamma, budget, initial=(),
                  draft_stop_tokens=(), target_stops=(), fault=None):
    # This is a control-flow/state model. Uniform zero selects one specific
    # acceptance/sampling branch; probability correctness is checked separately.
    assert gamma >= 0 and budget >= 0
    assert all(all(t in (0, 1) for t in word) for word in words)
    prefix = (9,) + initial
    trace = {'draft_calls': [], 'target_calls': [], 'raw_inputs': [],
             'path_checks': [], 'residual_calls': []}
    next_table = defaultdict(set)
    for word in words:
        for j, token in enumerate(word):
            next_table[word[:j]].add(token)
        next_table[word].add(2)

    def terminal(h):
        return 2 in h or h in target_stops

    def target_dist(h):
        trace['target_calls'].append(h)
        if terminal(h):
            raise UndefinedTargetDistribution(h, 'terminal_prefix')
        allowed = next_table.get(h, set())
        if not allowed:
            raise UndefinedTargetDistribution(h, 'dead_prefix')
        p = tuple(F(1, len(allowed)) if t in allowed else ZERO for t in T)
        validate(p)
        return p

    def transition_legal(h, token):
        # Independent path check, using whole-word prefix viability instead of
        # the oracle's next-token table.
        if token == 2:
            return h in words
        extended = h + (token,)
        return any(word[:len(extended)] == extended for word in words)

    if budget == 0 or terminal(initial):
        return {'output': (), 'known': prefix, 'kv_tokens': prefix,
                'K': len(prefix), 'state': closed_state(prefix), 'grammar': prefix,
                'grammar_history': initial, 'n': 0, 'g': 0, 'c': 0, 's': 0,
                'extra': None, 'trace': trace, 'finished': True}

    p1 = target_dist(initial)
    candidates = []
    q_records = []
    for token in draft_tokens[:min(gamma, budget)]:
        trace['draft_calls'].append(initial + tuple(candidates))
        q = tuple(ONE if t == token else ZERO for t in T)
        validate(q)
        assert q[token] > 0
        candidates.append(token)
        q_records.append(q)
        if token == 2 or token in draft_stop_tokens:
            break
    candidates = tuple(candidates)
    g = len(candidates)

    if fault == 'eager_suffix':
        # Deliberately restore the old eager processing of every suffix.
        # The strict independent oracle must actually reject a dead/terminal h.
        trace['raw_inputs'] = [initial + candidates[:i] for i in range(1, g + 1)]
        for i in range(1, g + 1):
            target_dist(initial + candidates[:i])

    h = initial
    c = 0
    s = 0
    invalid = False
    for i, token in enumerate(candidates, 1):
        c = i
        trace['path_checks'].append(h)
        if not transition_legal(h, token):
            invalid = True
            break
        h += (token,)
        s = i
        if terminal(h) or i == budget:
            break
    trace['raw_inputs'] = [initial + candidates[:i] for i in range(1, s + 1)]
    assert s == c - 1 if invalid else s == c

    def consume_parent_distribution(h):
        if h == initial:
            return p1
        assert h in trace['raw_inputs'], ('missing_parent_row', h)
        return target_dist(h)

    def pick_first_positive(p):
        validate(p)
        return next(t for t, mass in enumerate(p) if mass > 0)

    accepted = ()
    extra = None
    accepted_terminal = False
    for i in range(c):
        h = initial + accepted
        p = consume_parent_distribution(h)
        token = candidates[i]
        q = q_records[i]
        a = min(ONE, p[token] / q[token])
        u = ZERO
        if u < a:
            accepted += (token,)
            if terminal(initial + accepted) or len(accepted) == budget:
                accepted_terminal = True
                break
        else:
            r = residual(p, q)
            trace['residual_calls'].append(h)
            extra = pick_first_positive(r)
            break
    if extra is None and not accepted_terminal:
        extra = pick_first_positive(consume_parent_distribution(initial + accepted))

    n = len(accepted)
    output = accepted + (() if extra is None else (extra,))
    kept = prefix + accepted
    known = prefix + output
    result = {'output': output, 'known': known, 'kv_tokens': kept,
              'K': len(kept), 'state': checkpoints(prefix, candidates[:s])[n],
              'grammar': known, 'grammar_history': initial + output,
              'n': n, 'g': g, 'c': c, 's': s, 'extra': extra, 'trace': trace,
              'finished': terminal(initial + output) or len(output) == budget}
    errors = state_errors(prefix, candidates, n, extra, extra is None, result)
    assert not errors, errors
    check_h = initial
    for token in output:
        assert not terminal(check_h), ('published_after_terminal', check_h)
        assert transition_legal(check_h, token), ('published_illegal_token', check_h, token)
        check_h += (token,)
    assert result['grammar_history'] == check_h
    assert len(trace['draft_calls']) == g
    assert len(trace['raw_inputs']) == s
    assert n <= s <= g
    assert all(not terminal(hh) for hh in trace['target_calls'])
    return result


def grammar_boundary_checks():
    # 0=a, 1=b, 2=EOS. At the empty generated prefix the first case is
    # exactly the two-token p=(1,0), q=(0,1) counterexample, padded by zero EOS.
    words = ((0,),)
    good = grammar_round(words, (1, 1), 2, 9)
    reference = direct_grammar_trajectory(words, (), 1)
    assert reference == (0,)
    assert good['output'] == reference
    assert (good['g'], good['c'], good['s'], good['n']) == (2, 1, 0, 0)
    assert good['known'] == (9, 0) and good['K'] == 1
    assert good['kv_tokens'] == (9,)
    assert good['state'] == closed_state((9,))
    assert good['grammar_history'] == (0,)
    assert good['trace']['target_calls'] == [()]
    assert good['trace']['raw_inputs'] == []
    assert good['trace']['residual_calls'] == [()]
    assert good['trace']['draft_calls'] == [(), (1,)]
    print('grammar unreachable suffix: output=a; g,c,s,n=',
          (good['g'], good['c'], good['s'], good['n']),
          'K=', good['K'], 'target_distribution_histories=', good['trace']['target_calls'])

    detected = []
    try:
        grammar_round(words, (1, 1), 2, 9, fault='eager_suffix')
    except UndefinedTargetDistribution as exc:
        assert exc.history == (1,) and exc.reason == 'dead_prefix'
        detected.append(('eager_dead_suffix', exc.history))
    else:
        raise AssertionError('eager dead-suffix negative control was not detected')

    late = grammar_round(((0, 1),), (0, 0, 1), 3, 9)
    assert late['output'] == direct_grammar_trajectory(((0, 1),), (), 2) == (0, 1)
    assert (late['g'], late['c'], late['s'], late['n']) == (3, 2, 1, 1)
    assert late['trace']['target_calls'] == [(), (0,)]
    assert late['trace']['raw_inputs'] == [(0,)]
    assert late['known'] == (9, 0, 1) and late['K'] == 2
    assert late['state'] == closed_state((9, 0))
    assert late['grammar_history'] == (0, 1)

    # Draft EOS stops preparation immediately; its accepted target EOS has a
    # materialized input row, but its post-EOS distribution is never consumed.
    eos = grammar_round(words, (0, 2, 1), 3, 9)
    assert eos['output'] == direct_grammar_trajectory(words, (), 9) == (0, 2)
    assert (eos['g'], eos['s'], eos['n']) == (2, 2, 2)
    assert eos['trace']['draft_calls'] == [(), (0,)]
    assert eos['trace']['target_calls'] == [(), (0,)]
    assert eos['trace']['raw_inputs'] == [(0,), (0, 2)]
    assert eos['extra'] is None and eos['K'] == len(eos['known']) == 3

    # The same EOS as a bonus has not been consumed by the model.
    bonus_eos = grammar_round(words, (0,), 1, 9)
    assert bonus_eos['output'] == (0, 2)
    assert bonus_eos['extra'] == 2 and bonus_eos['n'] == 1
    assert bonus_eos['K'] == len(bonus_eos['known']) - 1 == 2
    assert bonus_eos['trace']['raw_inputs'] == [(0,)]
    assert bonus_eos['trace']['target_calls'] == [(), (0,)]

    # A draft-only ordinary stop ends proposals, not target generation.
    draft_stop = grammar_round(((0, 1),), (0, 1, 1), 3, 9, draft_stop_tokens=(0,))
    assert draft_stop['g'] == draft_stop['s'] == 1
    assert draft_stop['output'] == (0, 1) and draft_stop['extra'] == 1
    assert draft_stop['trace']['draft_calls'] == [()]
    assert draft_stop['trace']['target_calls'] == [(), (0,)]
    assert not draft_stop['finished']

    # Target policy stops after a legal ordinary token; pre-generated draft
    # suffixes are charged as g=3, but neither input nor probability consumes them.
    target_stop = grammar_round(words, (0, 1, 1), 3, 9, target_stops=((0,),))
    assert target_stop['output'] == direct_grammar_trajectory(words, (), 9, ((0,),)) == (0,)
    assert (target_stop['g'], target_stop['c'], target_stop['s']) == (3, 1, 1)
    assert target_stop['trace']['target_calls'] == [()]
    assert target_stop['trace']['path_checks'] == [()]
    assert target_stop['trace']['raw_inputs'] == [(0,)]
    assert target_stop['extra'] is None and target_stop['K'] == len(target_stop['known'])
    try:
        grammar_round(words, (0, 1, 1), 3, 9, target_stops=((0,),), fault='eager_suffix')
    except UndefinedTargetDistribution as exc:
        assert exc.history == (0,) and exc.reason == 'terminal_prefix'
        detected.append(('eager_terminal_suffix', exc.history))
    else:
        raise AssertionError('terminal-suffix negative control was not detected')

    # Illegal EOS is rejected before it can terminate the target request.
    illegal_eos = grammar_round(words, (2, 1), 2, 9)
    assert illegal_eos['g'] == 1 and illegal_eos['s'] == 0
    assert illegal_eos['output'] == (0,) and illegal_eos['extra'] == 0
    assert not illegal_eos['finished']
    assert illegal_eos['trace']['target_calls'] == [()]

    # Accepting is not terminal: after a, b and EOS are both allowed.
    expandable_words = ((0,), (0, 1))
    expandable = grammar_round(expandable_words, (0, 1), 2, 9)
    assert expandable['output'] == direct_grammar_trajectory(expandable_words, (), 9) == (0, 1, 2)
    assert expandable['n'] == 2 and expandable['extra'] == 2
    assert expandable['trace']['target_calls'] == [(), (0,), (0, 1)]
    assert expandable['K'] == len(expandable['known']) - 1

    plain = grammar_round(words, (1, 1), 0, 9)
    empty_draft = grammar_round(words, (), 3, 9)
    for result in (plain, empty_draft):
        assert result['output'] == (0,)
        assert result['g'] == result['s'] == result['n'] == 0
        assert result['trace']['target_calls'] == [()]
        assert result['K'] == 1

    limited = grammar_round(((0, 1),), (0, 1, 0), 3, 1)
    assert limited['output'] == (0,) and limited['g'] == limited['s'] == 1
    assert limited['extra'] is None and limited['finished']
    assert limited['trace']['target_calls'] == [()]
    assert limited['K'] == len(limited['known'])

    zero = grammar_round(words, (1, 1), 2, 0)
    already_done = grammar_round(words, (1, 1), 2, 9, initial=(0, 2))
    for result in (zero, already_done):
        assert result['output'] == ()
        assert all(not value for value in result['trace'].values())
        assert result['K'] == len(result['known'])

    try:
        grammar_round(words, (0, 0), 2, 9, initial=(1,))
    except UndefinedTargetDistribution as exc:
        assert exc.history == (1,) and exc.reason == 'dead_prefix'
        detected.append(('confirmed_initial_dead_prefix_is_real_error', exc.history))
    else:
        raise AssertionError('confirmed dead initial prefix must not silently recover')

    bad_state = dict(good, K=len(good['known']), kv_tokens=good['known'])
    errors = state_errors((9,), (1, 1), 0, 0, False, bad_state)
    assert 'kv_boundary' in errors and 'kv_visibility' in errors
    bad_published = dict(good, known=(9, 1, 1, 0), grammar=(9, 1, 1, 0))
    assert 'known' in state_errors((9,), (1, 1), 0, 0, False, bad_published)
    print('grammar flow edges: second-position rejection, accepted/bonus EOS, draft-only stop, target stop,')
    print('illegal EOS, accepting-and-expandable state, zero candidates, exhausted draft, limit, zero budget, initial terminal/dead prefix')
    print('strict grammar negative controls and initial-error distinction=', detected)
    print('grammar state mutations detected: pending correction counted as KV; rejected b,b published')
    print('Grammar scope: finite token language and deterministic branch traversal, not arbitrary CFG compilation or RNG statistics.')


def main():
    probability_checks()
    tree_forward_checks()
    state_checks()
    grammar_boundary_checks()
    print('Scope: deterministic offline CPU checks; exact rational probability enumeration and two-layer single-head 2D floating toy attention.')
    print('No real-model, CUDA, DMA, graph replay, distributed sampling, GPU timing, asynchronous lifetime or RNG implementation measurement.')


if __name__ == '__main__':
    main()
