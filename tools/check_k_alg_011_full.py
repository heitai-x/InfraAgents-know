#!/usr/bin/env python3
"""K-ALG-011: deterministic stdlib CPU checks; offline and read-only."""
import copy
import itertools
import math
from dataclasses import dataclass, field, replace
from fractions import Fraction


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_error(operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def left_fold(values):
    # Deliberately explicit binary64 addition; never replace with builtin sum.
    total = 0.0
    for value in values:
        total = total + value
    return total


def normalize(weights):
    vals = [float(x) for x in weights]
    if not vals or any(not math.isfinite(x) or x < 0.0 for x in vals):
        raise ValueError('invalid mass')
    try:
        total = math.fsum(vals)
    except OverflowError as exc:
        raise ValueError('invalid total') from exc
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError('invalid total')
    return [x / total for x in vals]


def endpoints(vals, order):
    cumulative = 0.0
    for j, token in enumerate(order):
        cumulative = min(1.0, cumulative + vals[token])
        yield token, (1.0 if j == len(order) - 1 else cumulative)


def inverse_cdf(weights, u):
    if not 0.0 <= u < 1.0:
        raise ValueError('invalid uniform')
    vals = normalize(weights)
    order = [i for i, x in enumerate(vals) if x > 0.0]
    for token, endpoint in endpoints(vals, order):
        if u < endpoint:
            return token
    raise AssertionError('final positive endpoint must be one')


def prefix_until(weights, rho):
    if not math.isfinite(rho) or not 0.0 < rho <= 1.0:
        raise ValueError('invalid top-p')
    vals = normalize(weights)
    order = sorted((i for i, x in enumerate(vals) if x > 0.0),
                   key=lambda i: (-vals[i], i))
    if rho == 1.0:
        return tuple(order)
    selected = []
    for token, endpoint in endpoints(vals, order):
        selected.append(token)
        if endpoint >= rho:
            return tuple(selected)
    raise AssertionError('unreachable prefix')


def top_k_set(weights, k):
    if type(k) is not int or k < 1:
        raise ValueError('invalid top-k')
    vals = normalize(weights)
    order = sorted((i for i, x in enumerate(vals) if x > 0.0),
                   key=lambda i: (-vals[i], i))
    return set(order[:k])


def truncate(weights, k, rho, mode):
    vals = normalize(weights)
    chosen_k = top_k_set(vals, k)
    if mode == 'k_then_p':
        restricted = [x if i in chosen_k else 0.0 for i, x in enumerate(vals)]
        support = set(prefix_until(restricted, rho))
    elif mode == 'p_then_k':
        support = set(prefix_until(vals, rho)) & chosen_k
    else:
        raise ValueError('unknown mode')
    q = normalize([x if i in support else 0.0 for i, x in enumerate(vals)])
    return q, tuple(sorted(support))


def legal_scores(logits, legal):
    if not logits:
        raise ValueError('empty logits')
    indices = sorted(set(legal))
    if not indices:
        raise ValueError('empty legal set')
    if any(type(i) is not int or i < 0 or i >= len(logits) for i in indices):
        raise ValueError('invalid legal id')
    scores = {i: float(logits[i]) for i in indices}
    if any(math.isnan(x) or x == math.inf for x in scores.values()):
        raise ValueError('invalid score')
    if not any(math.isfinite(x) for x in scores.values()):
        raise ValueError('no finite legal mass')
    return scores


def softmax_logits(logits, temperature, legal):
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError('positive finite temperature required')
    scores = legal_scores(logits, legal)
    maximum = max(scores.values())
    weights = [0.0] * len(logits)
    for i, score in scores.items():
        weights[i] = math.exp((score - maximum) / temperature)
    return normalize(weights)


def prepare_row(logits, temperature, k, rho, mode, legal):
    try:
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError('invalid temperature')
        if type(k) is not int or k < 1:
            raise ValueError('invalid top-k')
        if not math.isfinite(rho) or not 0.0 < rho <= 1.0:
            raise ValueError('invalid top-p')
        if mode not in ('k_then_p', 'p_then_k'):
            raise ValueError('unknown mode')
        scores = legal_scores(logits, legal)
        if temperature == 0.0:
            token = min(scores, key=lambda i: (-scores[i], i))
            q = [0.0] * len(logits)
            q[token] = 1.0
            return dict(status='ok', token=token, q=q,
                        support=(token,), draws=0)
        p = softmax_logits(logits, temperature, legal)
        q, support = truncate(p, k, rho, mode)
        return dict(status='ok', token=None, q=q, support=support, draws=1)
    except ValueError as exc:
        return dict(status='error', reason=str(exc))


def rational_sample(weights, u):
    # Independent exact arithmetic: no normalize or endpoints calls.
    vals = [Fraction(x) for x in weights]
    target = Fraction.from_float(u) * sum(vals)
    accumulated = Fraction(0)
    for i, value in enumerate(vals):
        accumulated += value
        if value > 0 and target < accumulated:
            return i
    raise AssertionError('rational sample out of range')


def rational_prefix(weights, rho):
    vals = [Fraction(x) for x in weights]
    order = sorted((i for i, x in enumerate(vals) if x > 0),
                   key=lambda i: (-vals[i], i))
    if rho == 1.0:
        return tuple(order)
    target = Fraction.from_float(rho) * sum(vals)
    accumulated = Fraction(0)
    selected = []
    for i in order:
        selected.append(i)
        accumulated += vals[i]
        if accumulated >= target:
            return tuple(selected)
    raise AssertionError('rational prefix out of range')


def faulty_inverse_tail(weights, u):
    vals = normalize(weights)
    cumulative = 0.0
    for i, value in enumerate(vals):
        cumulative = cumulative + value
        if u < cumulative:
            return i
    return len(vals) - 1


def faulty_tolerant_prefix(weights, rho):
    vals = normalize(weights)
    order = sorted(range(len(vals)), key=lambda i: (-vals[i], i))
    selected = []
    cumulative = 0.0
    for i in order:
        if vals[i] > 0.0:
            selected.append(i)
            cumulative = cumulative + vals[i]
            if cumulative >= rho - 1e-12:
                break
    return tuple(selected)


MASK = (1 << 64) - 1


def mix64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return (value ^ (value >> 31)) & MASK


def unit(seed, *parts):
    # Test-only keyed mapping; distinct keys do not prove PRNG independence.
    value = mix64(seed)
    for index, part in enumerate(parts):
        value = mix64(value ^ ((int(part) + (index + 1) *
                              0x9E3779B97F4A7C15) & MASK))
    return (value >> 11) / 9007199254740992.0


@dataclass
class Branch:
    generation: int
    version: int = 0
    tokens: list = field(default_factory=list)
    grammar: tuple = ()
    draws: int = 0
    terminal: bool = False
    reason: str = ''


@dataclass
class Request:
    generation: int
    branches: dict
    done: bool = False
    cancelled: bool = False
    max_length: int = 100


@dataclass(frozen=True)
class Descriptor:
    beta: int
    request: int
    branch: int
    request_generation: int
    branch_generation: int
    version: int
    position: int
    ordinal: int
    source_row: int
    temperature: float = 1.0
    k: int = 100
    rho: float = 1.0
    mode: str = 'k_then_p'
    legal: tuple = (0, 1)


def descriptor(state, request, branch, source_row, **changes):
    req = state[request]
    br = req.branches[branch]
    d = Descriptor(77, request, branch, req.generation, br.generation,
                   br.version, len(br.tokens), 0, source_row)
    return replace(d, **changes)


def current(d, state, beta):
    req = state.get(d.request)
    if req is None or d.beta != beta or req.done or req.cancelled:
        return False
    br = req.branches.get(d.branch)
    return (br is not None and not br.terminal and
            d.request_generation == req.generation and
            d.branch_generation == br.generation and
            d.version == br.version and d.position == len(br.tokens) and
            len(br.tokens) < req.max_length)


def stage(beta, descriptors, logits, state, trace, fault=None,
          successor_errors=frozenset()):
    targets = [(d.request, d.branch) for d in descriptors]
    if len(set(targets)) != len(targets):
        raise ValueError('duplicate target branch in snapshot')
    results = []
    previous_support = None
    for d in descriptors:
        if fault == 'common_generation' and d.request_generation != beta:
            results.append(dict(d=d, status='discarded'))
            continue
        if not current(d, state, beta):
            results.append(dict(d=d, status='discarded'))
            continue
        if not 0 <= d.source_row < len(logits):
            results.append(dict(d=d, status='error', reason='invalid source row'))
            continue
        br = state[d.request].branches[d.branch]
        prepared = prepare_row(logits[d.source_row], d.temperature, d.k,
                               d.rho, d.mode, d.legal)
        if prepared['status'] == 'error':
            if fault == 'advance_on_error':
                br.draws += 1
            results.append(dict(d=d, status='error', reason=prepared['reason']))
            continue
        support = prepared['support']
        if fault == 'reuse_greedy_support' and prepared['draws'] == 0:
            support = previous_support
        previous_support = prepared['support']
        if prepared['draws'] == 0:
            token = prepared['token']
        else:
            key = (d.request, d.branch, d.position, d.ordinal, 0)
            trace.append(key)
            try:
                token = inverse_cdf(prepared['q'], unit(12345, *key))
            except ValueError as exc:
                results.append(dict(d=d, status='error', reason=str(exc)))
                continue
        if (d.request, d.branch) in successor_errors:
            results.append(dict(d=d, status='error', reason='successor failed'))
            continue
        results.append(dict(d=d, status='ready', token=token,
                            draws=prepared['draws'], support=support,
                            next_grammar=br.grammar + (token,)))
    return results


def commit(beta, staged, state, fault=None):
    # Serial model of atomic check/publish, not a lock or memory-model test.
    outcomes = []
    touched = set()
    for original in staged:
        item = dict(original)
        if item['status'] != 'ready':
            outcomes.append(item)
            continue
        d = item['d']
        if not current(d, state, beta):
            item['status'] = 'discarded'
            outcomes.append(item)
            continue
        req = state[d.request]
        br = req.branches[d.branch]
        br.tokens.append(item['token'])
        br.grammar = item['next_grammar']
        br.draws += item['draws']
        br.version += 1
        if item['token'] == 0:
            br.terminal = True
            br.reason = 'eos'
            if fault == 'request_eos':
                req.done = True
        elif len(br.tokens) >= req.max_length:
            br.terminal = True
            br.reason = 'length'
        item['status'] = 'committed'
        touched.add((d.request, d.request_generation))
        outcomes.append(item)
    for request, generation in touched:
        req = state[request]
        if req.generation != generation or req.cancelled:
            continue
        req.done = req.done or (bool(req.branches) and
                               all(br.terminal for br in req.branches.values()))
    return outcomes


def build_row_map(segments, events):
    # Segment: request, source branch, query count, valid count, logical start.
    source_rows = []
    offsets = {}
    counts = {}
    visits = 0
    for request, branch, count, valid, start in segments:
        path = (request, branch)
        if path in offsets or not 0 <= valid <= count:
            raise ValueError('duplicate segment or invalid lengths')
        offsets[path] = len(source_rows)
        counts[path] = count
        for k in range(count):
            source_rows.append((request, branch, start + k, k < valid))
            visits += 1
    result = []
    targets = set()
    for request, source_branch, source_offset, target in events:
        owner = (target[0], target[1])
        path = (request, source_branch)
        if owner in targets:
            raise ValueError('duplicate target')
        targets.add(owner)
        if path not in offsets or not 0 <= source_offset < counts[path]:
            raise ValueError('invalid segment-local offset')
        row = offsets[path] + source_offset
        source = source_rows[row]
        if source[:2] != path or not source[3]:
            raise ValueError('invalid source identity or padding')
        if target[0] != request or target[2] != source[2] + 1:
            raise ValueError('prediction position mismatch')
        result.append(row)
    return source_rows, tuple(result), visits


def numerical_checks():
    vectors = 0
    selections = 0
    for weights in itertools.product(range(3), repeat=3):
        if sum(weights) not in (1, 2, 4):
            continue
        vectors += 1
        for j in range(16):
            u = j / 16.0
            got = inverse_cdf(weights, u)
            require(got == rational_sample(weights, u),
                    'dyadic CDF differs from independent Fraction oracle')
            require(weights[got] > 0, 'zero mass selected')
            selections += 1
        for rho in (0.25, 0.5, 0.75, 1.0):
            require(prefix_until(weights, rho) == rational_prefix(weights, rho),
                    'dyadic top-p differs from independent Fraction oracle')
    print('exact rational oracle: vectors =', vectors, 'CDF selections =', selections)

    weights = [0.1] * 10 + [0.0]
    u = math.nextafter(1.0, 0.0)
    folded = left_fold(weights)
    require(0.0 <= u < 1.0, 'counterexample uniform must be valid')
    require(folded == u, 'explicit binary64 left-fold probe failed')
    require(inverse_cdf(weights, u) == 9, 'tail must map to last positive mass')
    bad_token = faulty_inverse_tail(weights, u)
    require(bad_token == 10 and weights[bad_token] == 0.0,
            'old tail fallback negative control not detected')
    # This observation is intentionally not an assertion about builtin sum.
    print('binary64 probe: builtin sum =', repr(sum(weights)),
          'explicit left fold =', repr(folded), 'fixed uniform =', repr(u))
    print('tail control: old token =', bad_token, 'fixed token =', inverse_cdf(weights, u))
    require(inverse_cdf([0.0, 1.0, 0.0], 0.0) == 1,
            'leading zero mass selected')
    tiny_tail = [1.0 - 5e-13, 5e-13]
    for rho in (1.0, 1.0 - 2.5e-13):
        require(prefix_until(tiny_tail, rho) == (0, 1),
                'strict top-p dropped required positive tail')
        require(faulty_tolerant_prefix(tiny_tail, rho) == (0,),
                'tolerant-prefix negative control not detected')
    print('top-p controls: rho=1 and near-one retain positive tail; tolerance fault detected')

    for boundary, expected in ((0.5, 1), (0.8, 2)):
        require(inverse_cdf([0.5, 0.3, 0.2], boundary) == expected,
                'strict CDF boundary failed')
    require(prefix_until([0.5, 0.3, 0.2], 0.8) == (0, 1),
            'top-p equality boundary failed')
    require(inverse_cdf([0.9, 0.05, 0.05], 0.5) == 0 and int(0.5 * 3) == 1,
            'equal-width inverse-CDF fault not distinguished')
    for weights in ([], [0.0, 0.0], [math.nan], [math.inf], [-0.1, 1.1]):
        expect_error(lambda w=weights: inverse_cdf(w, 0.5))
    for invalid_u in (-0.1, 1.0, math.nan):
        expect_error(lambda u=invalid_u: inverse_cdf([1.0], u))
    for invalid_rho in (0.0, 1.1, math.nan):
        expect_error(lambda p=invalid_rho: prefix_until([1.0], p))

    p1 = softmax_logits([1.0, 2.0, -0.5], 1.0, (0, 1, 2))
    p2 = softmax_logits([101.0, 102.0, 99.5], 1.0, (0, 1, 2))
    require(max(abs(a - b) for a, b in zip(p1, p2)) <= 1e-15,
            'softmax additive invariant failed')
    greedy = prepare_row([1.0, 1.0, 0.0], 0.0, 3, 1.0,
                         'k_then_p', (0, 1, 2))
    require(greedy['status'] == 'ok' and greedy['token'] == 0 and
            greedy['q'] == [1.0, 0.0, 0.0] and
            greedy['support'] == (0,) and greedy['draws'] == 0,
            'greedy fields or tie rule failed')
    for temperature in (0.0, 1.0):
        for logits, legal in (([0.0, 1.0], ()),
                              ([-math.inf, -math.inf], (0, 1)),
                              ([math.nan, 1.0], (0, 1)),
                              ([math.inf, 1.0], (0, 1)),
                              ([0.0, 1.0], (2,))):
            require(prepare_row(logits, temperature, 2, 1.0,
                                'k_then_p', legal)['status'] == 'error',
                    'invalid input accepted on greedy/random path')
    for temperature in (-1.0, math.nan, math.inf):
        require(prepare_row([0.0, 1.0], temperature, 2, 1.0,
                            'k_then_p', (0, 1))['status'] == 'error',
                'invalid temperature accepted')
    for k, rho, mode in ((0, 1.0, 'k_then_p'), (True, 1.0, 'k_then_p'),
                         (2, 0.0, 'k_then_p'), (2, 1.0, 'unknown')):
        require(prepare_row([0.0, 1.0], 0.0, k, rho, mode,
                            (0, 1))['status'] == 'error',
                'greedy skipped parameter validation')

    _, kp = truncate([0.5, 0.3, 0.2], 2, 0.6, 'k_then_p')
    _, pk = truncate([0.5, 0.3, 0.2], 2, 0.6, 'p_then_k')
    require(kp == (0,) and pk == (0, 1), 'truncation-order reference failed')
    legal = softmax_logits([math.log(0.4), math.log(0.35), math.log(0.25)],
                           1.0, (1, 2))
    require(prefix_until(legal, 0.4) == (1,), 'legal-first support failed')
    wrong = set(prefix_until([0.4, 0.35, 0.25], 0.4)) & {1, 2}
    require(wrong == set(), 'late-mask negative control not detected')
    global_support = set(prefix_until([0.6, 0.3, 0.1], 0.5))
    local_union = set(prefix_until([0.6, 0.3], 0.5)) | {
        2 + i for i in prefix_until([0.1], 0.5)}
    require(global_support == {0} and local_union == {0, 2},
            'shard-local normalization fault not distinguished')
    print('greedy fields, failures, boundaries, truncation order, late mask and shard controls checked')


def mapping_and_random_checks():
    segments = [(10, 0, 4, 3, 0), (11, 0, 2, 1, 0)]
    events = [(10, 0, 2, (10, 0, 3, 0)),
              (10, 0, 2, (10, 1, 3, 1)),
              (11, 0, 0, (11, 0, 1, 0))]
    rows, mapping, visits = build_row_map(segments, events)
    require(len(rows) == 6 and visits == 6 and mapping == (2, 2, 4),
            'query/output mapping failed')
    wrong = (3, 3, 5)
    require(mapping != wrong and not rows[3][3] and not rows[5][3],
            'query-count-last-row fault not distinguished')
    for bad in ([(10, 0, 3, (10, 0, 4, 0))],
                [(10, 0, 2, (10, 0, 2, 0))],
                [(10, 0, -1, (10, 0, 0, 0))],
                [(10, 0, 4, (10, 0, 5, 0))],
                [events[0], events[0]]):
        expect_error(lambda e=bad: build_row_map(segments, e))
    larger = [(10, 0, 40, 3, 0), (11, 0, 2, 1, 0)]
    larger_rows, larger_map, larger_visits = build_row_map(larger, events)
    require(larger_visits == len(larger_rows) == 42 and
            larger_map == (2, 2, 40) and len(larger_map) == len(mapping),
            'fixed-B row-map preparation did not account for Q')
    print('row map: B=3; Q=6/42 visits=6/42; padding, offset and position faults rejected')

    probs = ([0.9, 0.1], [0.1, 0.9], [0.5, 0.5])
    logical_rows = [(10, 0, 3, 0, 0), (11, 0, 1, 0, 1),
                    (10, 1, 3, 1, 2)]
    permutation = [logical_rows[2], logical_rows[0], logical_rows[1]]

    def sample_order(ordered, seed, mode):
        out = {}
        for index, row in enumerate(ordered):
            r, b, pos, ordinal, source = row
            parts = (r, b, pos, ordinal, 0) if mode == 'logical' else (index, 0)
            out[(r, b, pos, ordinal)] = inverse_cdf(probs[source], unit(seed, *parts))
        return out

    require(sample_order(logical_rows, 17, 'logical') ==
            sample_order(permutation, 17, 'logical'), 'logical permutation invariant failed')
    bad_seed = next((seed for seed in range(2048)
                     if sample_order(logical_rows, seed, 'cursor') !=
                     sample_order(permutation, seed, 'cursor')), None)
    require(bad_seed is not None, 'row-cursor fault not triggered')
    divergent_seed = next((seed for seed in range(2048)
                           if inverse_cdf(probs[2], unit(seed, 10, 0, 3, 0, 0)) !=
                           inverse_cdf(probs[2], unit(seed, 10, 1, 3, 1, 0))), None)
    require(divergent_seed is not None, 'shared-draw fault not exposed')
    shared = inverse_cdf(probs[2], unit(divergent_seed, 2, 0))
    distinct = (inverse_cdf(probs[2], unit(divergent_seed, 10, 0, 3, 0, 0)),
                inverse_cdf(probs[2], unit(divergent_seed, 10, 1, 3, 1, 0)))
    require((shared, shared) != distinct, 'shared-row RNG reuse not distinguished')
    print('logical-key permutation invariant; cursor/shared-draw faults at seeds',
          bad_seed, divergent_seed)


def initial_state():
    return {10: Request(2, {0: Branch(3), 1: Branch(4)}),
            11: Request(9, {0: Branch(6)})}


def state_checks():
    beta = 77
    logits = [[3.0, 1.0], [0.0, 3.0], [-math.inf, 0.0]]
    state = initial_state()
    ds = [descriptor(state, 10, 0, 0, temperature=0.0),
          descriptor(state, 10, 1, 1, temperature=0.0, ordinal=1),
          descriptor(state, 11, 0, 2)]
    before = copy.deepcopy(state)
    trace = []
    staged = stage(beta, ds, logits, state, trace)
    require(state == before, 'staging changed official state')
    require(len(trace) == 1, 'greedy consumed randomness')
    require([x['support'] for x in staged] == [(0,), (1,), (1,)],
            'first/mixed greedy support uninitialized')
    outcomes = commit(beta, staged, state)
    require(all(x['status'] == 'committed' for x in outcomes),
            'valid mixed generations dropped')
    require(state[10].branches[0].terminal and not state[10].branches[1].terminal,
            'EOS affected sibling')
    require(not state[10].done, 'request ended before required sibling')
    require(state[10].branches[0].tokens == [0] and
            state[10].branches[1].tokens == [1] and
            state[11].branches[0].tokens == [1], 'unexpected branch tokens')
    require(state[10].branches[0].grammar == (0,) and
            state[10].branches[1].grammar == (1,), 'grammar crossed branches')
    require(state[10].branches[0].draws == 0 and
            state[10].branches[1].draws == 0 and
            state[11].branches[0].draws == 1, 'wrong committed random consumption')
    expected = copy.deepcopy(state)
    for perm in itertools.permutations(staged):
        trial = initial_state()
        commit(beta, perm, trial)
        require(trial == expected, 'staged commit order changed final state')
    for perm in itertools.permutations(ds):
        trial = initial_state()
        commit(beta, stage(beta, perm, logits, trial, []), trial)
        require(trial == expected, 'preparation order changed final state')
    trial = initial_state()
    commit(beta, stage(beta, ds, logits, trial, []), trial, fault='request_eos')
    require(trial != expected and trial[10].branches[1].tokens == [],
            'request-wide EOS fault not detected')
    trial = initial_state()
    pending = stage(beta, ds, logits, trial, [], fault='common_generation')
    commit(beta, pending, trial)
    require(trial == initial_state() and trial != expected,
            'common-generation fault not detected')
    print('beta=77, request generations=2/9; six preparation and six commit orders agree')
    print('negative controls detected: request-wide EOS and common-generation comparison')

    trial = initial_state()
    bad = stage(beta, [ds[0]], logits, trial, [], fault='reuse_greedy_support')
    require(bad[0]['support'] != (0,), 'first-greedy support fault not detected')
    bad = stage(beta, [ds[2], ds[0]], logits, trial, [], fault='reuse_greedy_support')
    require(bad[1]['support'] != (0,), 'mixed-greedy support fault not detected')
    for temperature in (0.0, 1.0):
        for bad_logits, bad_legal in (([0.0, 1.0], ()),
                                      ([-math.inf, -math.inf], (0, 1))):
            trial = initial_state()
            invalid = replace(ds[0], temperature=temperature, legal=bad_legal)
            frozen = copy.deepcopy(trial)
            reads = []
            pending = stage(beta, [invalid], [bad_logits], trial, reads)
            require(pending[0]['status'] == 'error', 'row error not propagated')
            commit(beta, pending, trial)
            require(trial == frozen and reads == [],
                    'failed preparation advanced state or read randomness')
        trial = initial_state()
        frozen = copy.deepcopy(trial)
        invalid = replace(ds[0], temperature=temperature, legal=())
        stage(beta, [invalid], logits, trial, [], fault='advance_on_error')
        require(trial != frozen, 'failure-advances-ledger fault not detected')
    trial = initial_state()
    frozen = copy.deepcopy(trial)
    reads = []
    pending = stage(beta, [ds[2]], logits, trial, reads,
                    successor_errors=frozenset({(11, 0)}))
    commit(beta, pending, trial)
    require(pending[0]['status'] == 'error' and len(reads) == 1 and trial == frozen,
            'post-PEEK successor error advanced official state')
    trial = initial_state()
    invalid = replace(ds[0], legal=())
    mixed = stage(beta, [invalid, ds[2]], logits, trial, [])
    result = commit(beta, mixed, trial)
    require([x['status'] for x in result] == ['error', 'committed'] and
            trial[10].branches[0].tokens == [] and trial[11].branches[0].tokens == [1],
            'per-branch failure incorrectly blocked or advanced another branch')
    print('greedy/error controls checked; preparation errors read no RNG; successor errors do not commit')

    for change in ('cancel', 'request_generation', 'branch_generation', 'version', 'terminal'):
        trial = initial_state()
        pending = stage(beta, [ds[2]], logits, trial, [])
        req = trial[11]
        br = req.branches[0]
        if change == 'cancel':
            req.cancelled = True
        elif change == 'request_generation':
            req.generation += 1
        elif change == 'branch_generation':
            br.generation += 1
        elif change == 'version':
            br.version += 1
        else:
            br.terminal = True
        frozen = copy.deepcopy(trial)
        result = commit(beta, pending, trial)
        require(result[0]['status'] == 'discarded' and trial == frozen,
                'late result mutated changed authority: ' + change)
    for stale in (replace(ds[2], request_generation=8), replace(ds[2], beta=78)):
        trial = initial_state()
        reads = []
        pending = stage(beta, [stale], logits, trial, reads)
        require(pending[0]['status'] == 'discarded' and not reads and trial == initial_state(),
                'stale preparation changed state or consumed randomness')
    trial = initial_state()
    pending = stage(beta, [ds[2]], logits, trial, [])
    commit(beta, pending, trial)
    frozen = copy.deepcopy(trial)
    result = commit(beta, pending, trial)
    require(result[0]['status'] == 'discarded' and trial == frozen,
            'same result committed twice')
    reads = []
    expect_error(lambda: stage(beta, [ds[0], ds[0]], logits, initial_state(), reads))
    require(not reads, 'duplicate targets read RNG before rejection')
    print('cancel, generation, version, terminal, snapshot and duplicate-commit guards checked')

    trial = {10: Request(2, {0: Branch(3), 1: Branch(4)})}
    shared = [descriptor(trial, 10, 0, 0),
              descriptor(trial, 10, 1, 0, ordinal=1)]
    reads = []
    commit(beta, stage(beta, shared, [[-math.inf, 0.0]], trial, reads), trial)
    b0, b1 = trial[10].branches[0], trial[10].branches[1]
    require(len(set(reads)) == 2 and b0.draws == b1.draws == 1,
            'shared probability row reused random identity')
    require(b0.tokens == b1.tokens == [1] and b0.tokens is not b1.tokens,
            'child mutable tails alias')
    b0.tokens.append(7)
    require(b1.tokens == [1] and b1.grammar == (1,), 'child successors alias')
    finishing = descriptor(state, 10, 1, 0, temperature=0.0, ordinal=1)
    commit(beta, stage(beta, [finishing], [[3.0, 0.0]], state, []), state)
    require(state[10].done and state[10].branches[1].tokens == [1, 0] and
            state[10].branches[0].tokens == [0], 'aggregate completion failed')
    trial = {12: Request(4, {0: Branch(1)}, max_length=1)}
    last = descriptor(trial, 12, 0, 0, temperature=0.0)
    commit(beta, stage(beta, [last], [[0.0, 3.0]], trial, []), trial)
    require(trial[12].done and trial[12].branches[0].reason == 'length',
            'length-limit completion failed')
    print('shared-row children have private successors; EOS/length request completion is aggregated')


def speculative_mass_checks():
    # Algebra only: no multi-position speculative implementation is claimed.
    p = [Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)]
    q = [Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)]
    accepted = [min(a, b) for a, b in zip(p, q)]
    rejection = 1 - sum(accepted)
    residual = [max(a - b, 0) for a, b in zip(p, q)]
    require(rejection == sum(residual) and rejection > 0,
            'residual normalization identity failed')
    result = [a + rejection * (r / sum(residual))
              for a, r in zip(accepted, residual)]
    require(result == p, 'speculative correction did not recover target mass')
    faulty = [a + rejection * target for a, target in zip(accepted, p)]
    require(faulty != p, 'unmodified-target fallback fault not distinguished')
    require(sum(min(a, b) for a, b in zip(p, p)) == 1,
            'identical distributions should never reject')
    print('Fraction single-step speculative mass identity; unmodified-target fallback fault detected')


def main():
    print('scope: deterministic Python stdlib CPU model; read-only and offline; no GPU or timing claims')
    numerical_checks()
    mapping_and_random_checks()
    state_checks()
    speculative_mass_checks()
    print('PASS: scoped numerical, mapping, serial branch-commit and algebra checks; faulty controls detected')
    print('unmeasured: GPU arithmetic/memory/synchronization, PRNG quality, performance, real grammar, beam and multi-position speculation')


if __name__ == '__main__':
    main()
