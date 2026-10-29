from dataclasses import dataclass, replace
from fractions import Fraction as F
from itertools import permutations
from math import exp, isclose


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def detected(label, verifier):
    try:
        verifier()
    except AssertionError:
        print('negative_control: %s detected=yes' % label)
        return
    raise AssertionError('negative control was accepted: ' + label)


@dataclass(frozen=True)
class Request:
    p: int
    tokens: tuple
    c: int
    ell: object = None
    identity: bool = True
    flight: bool = False
    stopped: bool = False
    cap: int = 64

    @property
    def a(self):
        return len(self.tokens)


@dataclass(frozen=True)
class Segment:
    owner: int
    mode: str
    h: int
    q: int
    cow: int = 0


def invalid(r):
    if not all(type(x) is int for x in (r.p, r.c, r.cap)):
        return True
    if not (1 <= r.p <= r.a <= r.cap and 0 <= r.c <= r.a):
        return True
    if r.c < r.p:
        return r.a != r.p or r.ell is not None
    if r.a - r.c not in (0, 1):
        return True
    return r.ell is not None and (r.a != r.c or r.ell != r.c - 1)


def classify_all(requests):
    kinds, excluded = {}, {}
    for i, r in enumerate(requests):
        if not r.identity:
            excluded[i] = 'stale'
            continue
        if invalid(r):
            excluded[i] = 'invalid'
            continue
        if r.stopped:
            excluded[i] = 'terminal'
            continue
        if r.flight:
            excluded[i] = 'wait_completion'
            continue
        if r.c < r.p:
            kinds[i] = 'prefill'
        elif r.a == r.c + 1:
            kinds[i] = 'decode'
        elif r.ell is not None:
            kinds[i] = 'sample'
        else:
            kinds[i] = 'recovery'
    return kinds, excluded


def kind(r):
    return classify_all((r,))[0].get(0)


def make_segment(r, owner=0, q=1):
    mode = kind(r)
    require(mode in ('prefill', 'decode', 'recovery'), 'no executable input')
    if mode == 'prefill':
        require(0 < q <= r.p - r.c, 'prefill extent')
        return Segment(owner, mode, r.c, q)
    return Segment(owner, mode, r.c if mode == 'decode' else r.c - 1, 1)


def verify_rows(requests, view):
    kinds, _ = classify_all(requests)
    for s in view:
        r = requests[s.owner]
        require(kinds.get(s.owner) == s.mode, 'excluded or unready request emitted a row')
        require(s.q > 0 and 0 <= s.h and s.h + s.q <= r.a, 'unknown input or negative row')
        if s.mode == 'prefill':
            require(s.h == r.c and s.h + s.q <= r.p, 'prefill writes wrong range')
        elif s.mode == 'decode':
            require(s.h == r.c and s.q == 1 and r.a == r.c + 1, 'decode input boundary')
        else:
            require(s.h == r.c - 1 and s.q == 1, 'recovery last row')


def commit(r, s, success=True):
    verify_rows((r,), (replace(s, owner=0),))
    if not success:
        return r
    if s.mode == 'recovery':
        return replace(r, ell=r.c - 1)
    end = r.c + s.q
    ell = end - 1 if s.mode == 'decode' or end == r.p else None
    return replace(r, c=end, ell=ell)


def accept(r, token, eos=False):
    require(kind(r) == 'sample' and r.a < r.cap, 'sampling event has no valid ticket')
    return replace(r, tokens=r.tokens + (token,), ell=None, stopped=eos)


def verify_decode_callback(before, after):
    require(after.c == before.c + 1, 'successful decode failed to advance KV')
    require(after.a == before.a and after.tokens == before.tokens, 'decode appended an input')
    require(after.ell == before.c, 'decode logits attached to wrong row')


def test_state_trajectory():
    r = Request(8, tuple(range(8)), 5)
    r = commit(r, make_segment(r, q=3))
    require((r.p, r.a, r.c, r.ell) == (8, 8, 8, 7), 'last prompt row must be P-1')
    ready = r
    detected('logits_are_not_a_known_decode_input', lambda: verify_rows((ready,), (Segment(0, 'decode', 8, 1),)))
    trace = []
    for token in (42, 43):
        r = accept(r, token)
        s = make_segment(r)
        before = r
        require(commit(r, s, success=False) == before, 'failed decode changed confirmed state')
        detected('successful_decode_without_boundary_advance', lambda b=before: verify_decode_callback(b, b))
        r = commit(r, s)
        verify_decode_callback(before, r)
        trace.append((token, s.h, r.p, r.a, r.c, r.ell))
    require(trace == [(42, 8, 8, 9, 9, 8), (43, 9, 8, 10, 10, 9)], 'event ledger differs from two-round oracle')
    require(kind(r) == 'sample', 'c>P must remain legal')
    terminal = accept(r, 99, eos=True)
    require(kind(terminal) is None and terminal.a == terminal.c + 1, 'EOS should not require its own KV')
    recovery = Request(8, tuple(range(8)), 8)
    s = make_segment(recovery)
    require((s.mode, s.h) == ('recovery', 7), 'full hit recovery row')
    require(commit(recovery, s).c == 8 and commit(recovery, s).ell == 7, 'recovery appended KV')
    one = Request(1, (7,), 0)
    require(commit(one, make_segment(one)).ell == 0, 'P=1 boundary')
    bad = [Request(0, (), 0), Request(8, tuple(range(8)), -1), Request(8, tuple(range(8)), 9), Request(8, tuple(range(10)), 8), replace(ready, identity=False), replace(ready, flight=True)]
    snapshot = tuple(bad)
    kinds, excluded = classify_all(bad)
    require(not kinds and len(excluded) == len(bad), 'invalid classification fell through')
    require(tuple(bad) == snapshot, 'classification changed committed state')
    for i, invalid_request in enumerate(bad):
        detected('invalid_fallthrough_%d' % i, lambda r=invalid_request: verify_rows((r,), (Segment(0, 'recovery', r.c - 1, 1),)))
    print('state_model: prompt_last=7 decode_rows=8,9 P=8 A=10 C=10; failure retains input; invalid rows excluded')


def full_attention_reference(tokens):
    values = [0.13 * (t + 1) + 0.07 * p for p, t in enumerate(tokens)]
    for _ in range(2):
        next_values = []
        for p, x in enumerate(values):
            scores = [(0.7 * x + 0.02 * p) * (0.5 * values[k] + 0.01 * k) for k in range(p + 1)]
            maximum = max(scores)
            weights = [exp(z - maximum) for z in scores]
            weighted = sum(w * (1.2 * values[k] - 0.03 * k) for k, w in enumerate(weights))
            next_values.append(x + 0.3 * weighted / sum(weights))
        values = next_values
    return values


def chunk_attention(tokens, start, q, caches, omit_self=False):
    values = [0.13 * (tokens[p] + 1) + 0.07 * p for p in range(start, start + q)]
    for layer in range(2):
        require(len(caches[layer]) == start, 'cache endpoint mismatch')
        current = [(0.5 * x + 0.01 * (start + j), 1.2 * x - 0.03 * (start + j)) for j, x in enumerate(values)]
        joined = caches[layer] + current
        output = []
        for j, x in enumerate(values):
            p = start + j
            query = 0.7 * x + 0.02 * p
            visible = joined[:p if omit_self else p + 1]
            require(bool(visible), 'empty attention row in fixture')
            scores = [query * kv[0] for kv in visible]
            shift = max(scores)
            weights = [exp(z - shift) for z in scores]
            output.append(x + 0.3 * sum(w * kv[1] for w, kv in zip(weights, visible)) / sum(weights))
        caches[layer].extend(current)
        values = output
    return values


def verify_close(got, expected):
    require(len(got) == len(expected), 'output length')
    require(all(isclose(x, y, rel_tol=1e-12, abs_tol=1e-12) for x, y in zip(got, expected)), 'scalar attention differs from independent full reference')


def test_attention_partition():
    tokens = (2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4)
    reference = full_attention_reference(tokens)
    caches = [[], []]
    prefix = chunk_attention(tokens, 0, 5, caches)
    verify_close(prefix, reference[:5])
    outputs, pairs, start = [], set(), 5
    for q in (2, 3, 1):
        outputs.extend(chunk_attention(tokens, start, q, caches))
        for p in range(start, start + q):
            pairs.update((p, k) for k in range(p + 1))
        start += q
    verify_close(outputs, reference[5:])
    direct_pairs = {(p, k) for p in range(len(tokens)) for k in range(len(tokens)) if p >= 5 and k <= p}
    require(pairs == direct_pairs and len(pairs) == 51, 'nonzero-prefix visibility relation')
    before = tuple(tuple(layer) for layer in caches)
    scratch = [list(layer[:-1]) for layer in caches]
    recovered = chunk_attention(tokens, 10, 1, scratch)
    verify_close(recovered, reference[-1:])
    require(tuple(tuple(layer) for layer in caches) == before, 'recovery modified published KV')
    bad_cache = [list(layer[:5]) for layer in caches]
    bad = chunk_attention(tokens, 5, 6, bad_cache, omit_self=True)
    detected('attention_omits_self', lambda: verify_close(bad, reference[5:]))
    print('attention_model: two scalar softmax layers; H=5 N=6 P=11 pairs=51; nonuniform chunks and scratch recovery agree')


def predicted_time(view, pad=16):
    if not view:
        return F(0)
    rows = sum(s.q for s in view)
    padded = ((rows + pad - 1) // pad) * pad
    pairs = sum(s.q * s.h + s.q * (s.q + 1) // 2 for s in view)
    return F(2) + F(padded, 100) + F(pairs, 1000000)


def reference_time(view, pad=16):
    positions = [p for s in view for p in range(s.h, s.h + s.q)]
    if not positions:
        return F(0)
    slots = len(positions)
    while slots % pad:
        slots += 1
    visible_keys = sum(len(range(p + 1)) for p in positions)
    return F(2) + F(slots, 100) + F(visible_keys, 1000000)


def page_need(view, page=4):
    return sum(0 if s.mode == 'recovery' else ((s.h + s.q + page - 1) // page - (s.h + page - 1) // page + s.cow) for s in view)


def reference_pages(view, page=4):
    new_physical = set()
    for s in view:
        if s.mode == 'recovery':
            continue
        old = {p // page for p in range(s.h)}
        touched = {p // page for p in range(s.h, s.h + s.q)}
        new_physical.update((s.owner, 'new', p) for p in touched - old)
        new_physical.update((s.owner, 'cow', i) for i in range(s.cow))
    return len(new_physical)


class Tickets:
    def __init__(self, capacity):
        self.free = capacity
        self.held = 0

    def attempt(self, total, fail=False):
        delta = total - self.held
        require(delta >= 0, 'fixture only adds segments')
        if delta > self.free:
            return False
        self.free -= delta
        self.held += delta
        if fail:
            self.free += delta
            self.held -= delta
            return False
        return True


def choose(decodes, candidates, budget, free_pages, fail_owner=None):
    view = ()
    tickets = Tickets(free_pages)
    for d in decodes:
        proposed = view + (d,)
        if predicted_time(proposed) <= budget and tickets.attempt(page_need(proposed)):
            view = proposed
    for s in sorted(candidates, key=lambda item: item.q, reverse=True):
        proposed = view + (s,)
        if predicted_time(proposed) > budget:
            continue
        if tickets.attempt(page_need(proposed), fail=s.owner == fail_owner):
            view = proposed
            break
    return view, tickets


def verify_feasible(view, budget, capacity):
    require(reference_time(view) <= budget, 'full mixed view exceeds time budget')
    require(reference_pages(view) <= capacity, 'full mixed view exceeds physical pages')


def verify_decode_preserved(view, expected):
    require(expected in view, 'feasible selected decode lost when no prefill fits')


def test_mixed_budget_and_fallback():
    decodes = tuple(Segment(i, 'decode', 7000, 1) for i in range(4))
    suffix = lambda q: Segment(10, 'prefill', 1000, q)
    for q in range(1, 801):
        require(predicted_time(decodes + (suffix(q),)) == reference_time(decodes + (suffix(q),)), 'mixed time arithmetic disagrees with row enumeration')
    single_max = max(q for q in range(1, 801) if reference_time((suffix(q),)) <= 10)
    mixed_max = max(q for q in range(1, 801) if reference_time(decodes + (suffix(q),)) <= 10)
    require((single_max, mixed_max) == (704, 692), 'maximal legal q changed')
    require(reference_time(decodes + (suffix(692),)) == F(9999782, 1000000), 'mixed boundary time')
    detected('ignore_decode_and_padding_budget', lambda: verify_feasible(decodes + (suffix(single_max),), F(10), 1000))
    detected('accept_next_boundary_q', lambda: verify_feasible(decodes + (suffix(693),), F(10), 1000))
    d = Segment(0, 'decode', 8, 1)
    for candidates, capacity, label in (((), 1, 'decode_only'), ((Segment(1, 'prefill', 0, 800),), 1000, 'prefill_time_blocked'), ((Segment(1, 'prefill', 0, 1),), 1, 'prefill_page_blocked')):
        view, tickets = choose((d,), candidates, F(10), capacity)
        require(view == (d,), label)
        verify_feasible(view, F(10), capacity)
        require(tickets.held == 1 and tickets.free == capacity - 1, 'reservation conservation')
    detected('return_empty_when_prefill_blocked', lambda: verify_decode_preserved((), d))
    view, tickets = choose((d,), (Segment(1, 'prefill', 0, 1),), F(10), 2, fail_owner=1)
    require(view == (d,) and tickets.held == 1 and tickets.free == 1, 'failed partial reservation damaged decode ticket')
    empty, tickets = choose((), (), F(10), 2)
    require(not empty and tickets.held == 0 and tickets.free == 2, 'idle leaked resources')
    blocked, tickets = choose((), (Segment(1, 'prefill', 0, 800),), F(10), 2)
    require(not blocked and tickets.held == 0, 'empty blocked view leaked resources')
    for h in (0, 1, 4, 5, 8, 1000):
        for q in (1, 2, 4, 17, 72):
            for cow in (0, 1):
                v = (Segment(20, 'prefill', h, q, cow),)
                require(page_need(v, 16) == reference_pages(v, 16), 'page formula differs from physical identity sets')
    for free, cow, expected in ((4, 0, 72), (2, 0, 40), (2, 1, 24)):
        maximum = max(q for q in range(1, 100) if reference_pages((Segment(20, 'prefill', 1000, q, cow),), 16) <= free)
        require(maximum == expected, 'tail and COW capacity boundary')
    print('mixed_model: independent row reference q_alone=704 q_with_4_decodes=692; q=693 rejected; decode-only/time/page fallbacks and reservation rollback hold')


def test_combination():
    requests = (Request(11, tuple(range(11)), 5), Request(8, tuple(range(9)), 8), Request(4, tuple(range(4)), 4))
    view = (replace(make_segment(requests[0], 0, 2), cow=1), make_segment(requests[1], 1), make_segment(requests[2], 2))
    verify_rows(requests, view)
    require([p for s in view for p in range(s.h, s.h + s.q)] == [5, 6, 8, 3], 'compressed row ownership')
    require(sum(p + 1 for s in view for p in range(s.h, s.h + s.q)) == 26, 'first mixed pair count')
    require(page_need(view) == reference_pages(view) == 2, 'COW plus decode page')
    after = [commit(r, s) for r, s in zip(requests, view)]
    after[1] = accept(after[1], 42)
    after[2] = accept(after[2], 43)
    next_view = tuple(make_segment(r, i, 4 if i == 0 else 1) for i, r in enumerate(after))
    verify_rows(after, next_view)
    require(sum(p + 1 for s in next_view for p in range(s.h, s.h + s.q)) == 53, 'second mixed pair count')
    require(page_need(next_view) == reference_pages(next_view) == 2, 'second mixed physical pages')
    final = [commit(r, s) for r, s in zip(after, next_view)]
    require((final[0].c, final[0].ell) == (11, 10), 'prefill final output identity')
    print('combination: two views pairs=26,53 new_pages=2,2; prompt last row=10; decode and recovery handoffs remain distinct')


class Flight:
    def __init__(self):
        self.accepted = {'cow_copy', 'layer0', 'layer1_partial', 'output_copy'}
        self.done = set()
        self.refs = {'old_shared': 1, 'cow_target': 0, 'scratch': 0}
        self.released = set()
        self.cancelled = False

    def release(self, name):
        require(self.done == self.accepted, 'accepted physical work has not drained')
        require(self.refs[name] == 0, 'shared owner still references page')
        self.released.add(name)


def independent_release_invariant(flight):
    for name in flight.released:
        require(not (flight.accepted - flight.done), 'logical cancellation released in-flight addresses')
        require(flight.refs[name] == 0, 'shared page released with live owner')


def test_drain_contract():
    mutant = Flight()
    mutant.cancelled = True
    mutant.released.add('cow_target')
    detected('generation_or_cancel_without_physical_drain', lambda: independent_release_invariant(mutant))
    orders = 0
    for order in permutations(sorted(Flight().accepted)):
        flight = Flight()
        flight.cancelled = True
        before = Request(8, tuple(range(9)), 8)
        s = make_segment(before)
        require(commit(before, s, success=False) == before, 'cancelled transaction published progress')
        for event in order:
            require(flight.done != flight.accepted, 'fixture completed early')
            try:
                flight.release('cow_target')
            except AssertionError:
                pass
            else:
                raise AssertionError('COW target reused before full drain')
            flight.done.add(event)
        flight.release('cow_target')
        flight.release('scratch')
        try:
            flight.release('old_shared')
        except AssertionError:
            pass
        else:
            raise AssertionError('old shared page lost its remaining owner')
        flight.refs['old_shared'] = 0
        flight.release('old_shared')
        independent_release_invariant(flight)
        orders += 1
    require(orders == 24, 'completion order coverage')
    print('drain_event_model: 24 abstract completion orders; partial-layer/COW/output work retained until join; shared owner separately blocks reuse')


def main():
    print('scope: deterministic offline CPU scalar, rational-cost, state and event models; no GPU, allocator-runtime or agent-effectiveness measurements')
    test_state_trajectory()
    test_attention_partition()
    test_mixed_budget_and_fallback()
    test_combination()
    test_drain_contract()
    print('PASS: scoped model invariants and intentionally faulty negative controls')


if __name__ == '__main__':
    main()
