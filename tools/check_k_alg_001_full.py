from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
import copy
import hashlib
import json

PAGE = 4
POOL = 8


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class Ref:
    slot: int
    epoch: int


@dataclass
class Page:
    epoch: int = 0
    allocated: bool = False
    owner: object = None
    users: set = field(default_factory=set)


@dataclass
class Request:
    key: tuple
    tokens: list
    k: int
    pages: list
    limit: int
    nmax: int
    flight: object = None
    status: str = 'live'
    generated: list = field(default_factory=list)


@dataclass(frozen=True)
class Segment:
    key: tuple
    start: int
    end: int
    n0: int
    inputs: tuple
    refs: tuple
    row0: int
    row1: int


@dataclass(frozen=True)
class View:
    ticket: str
    segments: tuple
    rows: tuple

    @property
    def refs(self):
        return frozenset(ref for s in self.segments for ref in s.refs)


def page_count(n):
    return (n + PAGE - 1) // PAGE


class Engine:
    def __init__(self, bug=None, pool=POOL):
        self.bug = bug
        self.pages = [Page() for _ in range(pool)]
        self.requests = {}
        self.prompts = {}
        self.views = {}
        self.pending = set()
        self.done = {}
        self.outputs = []

    def get_page(self, ref):
        p = self.pages[ref.slot]
        if not p.allocated or p.epoch != ref.epoch:
            raise ProtocolError('stale physical page identity')
        return p

    def free_slots(self):
        return [i for i, p in enumerate(self.pages) if not p.allocated]

    def allocate(self, key, count):
        slots = self.free_slots()
        if len(slots) < count:
            raise ProtocolError('allocation without sufficient reservation')
        refs = []
        for i in slots[:count]:
            p = self.pages[i]
            p.allocated = True
            p.owner = key
            refs.append(Ref(i, p.epoch))
        return refs

    def sweep(self):
        for p in self.pages:
            if p.allocated and p.owner is None and not p.users:
                p.allocated = False
                p.epoch += 1

    def retire_owner(self, r):
        for ref in r.pages:
            p = self.get_page(ref)
            if p.owner != r.key:
                raise ProtocolError('retirement does not own page')
            p.owner = None
        r.pages = []

    def arrive(self, key, tokens, cached=0, limit=99, nmax=100):
        if key in self.requests:
            raise ValueError('request lifetime already exists')
        if not tokens or len(tokens) > nmax or not 0 <= cached <= len(tokens):
            raise ValueError('invalid prompt or cache range')
        if limit < 0:
            raise ValueError('negative generation limit')
        k = min(cached, len(tokens) - 1)
        if self.bug == 'recovery_stall' and cached == len(tokens):
            k = cached
        if limit == 0:
            k = 0
        if page_count(k) > len(self.free_slots()):
            raise ValueError('initial state capacity unavailable')
        refs = self.allocate(key, page_count(k))
        r = Request(key, list(tokens), k, refs, limit, nmax)
        if limit == 0:
            r.status = 'ended'
        self.requests[key] = r
        self.prompts[key] = tuple(tokens)

    def submit(self, ticket, specs, token_budget=128):
        if not specs:
            return False
        if ticket in self.views:
            raise ValueError('submission credential reused')
        if len({key for key, q in specs}) != len(specs):
            raise ValueError('request appears twice in view')
        if sum(q for key, q in specs) > token_budget:
            raise ValueError('token budget exceeded')
        demands = []
        for key, q in specs:
            r = self.requests[key]
            if r.status != 'live' or r.flight is not None:
                raise ValueError('request not executable')
            if not isinstance(q, int) or not 1 <= q <= len(r.tokens) - r.k:
                raise ValueError('illegal known-input range')
            end = r.k + q
            need = max(0, page_count(end) - len(r.pages))
            demands.append((r, q, end, need))
        if sum(item[3] for item in demands) > len(self.free_slots()):
            return False
        segments = []
        rows = []
        for r, q, end, need in demands:
            r.pages.extend(self.allocate(r.key, need))
            row0 = len(rows)
            for pos in range(r.k, end):
                rows.append((r.key, pos, r.tokens[pos], end == len(r.tokens) and pos == end - 1))
            segments.append(Segment(r.key, r.k, end, len(r.tokens),
                                    tuple(r.tokens[r.k:end]), tuple(r.pages),
                                    row0, len(rows)))
        view = View(ticket, tuple(segments), tuple(rows))
        for ref in view.refs:
            if self.bug != 'missing_execution_pin':
                self.get_page(ref).users.add(ticket)
        self.views[ticket] = view
        self.pending.add(ticket)
        for s in view.segments:
            r = self.requests[s.key]
            r.flight = ticket
            if self.bug == 'advance_on_submit':
                r.k = s.end
        return True

    def cancel(self, key):
        r = self.requests[key]
        if r.status != 'live':
            return False
        refs = tuple(r.pages)
        r.status = 'canceled'
        self.retire_owner(r)
        if self.bug == 'cancel_drops_execution' and r.flight is not None:
            for ref in refs:
                self.get_page(ref).users.discard(r.flight)
        self.sweep()
        return True

    def complete(self, ticket, results, drained=True):
        if ticket not in self.views:
            raise ValueError('unknown completion credential')
        payload = (tuple(results), drained)
        if ticket in self.done:
            if self.done[ticket] != payload:
                raise ValueError('conflicting repeated completion')
            if self.bug != 'no_dedup':
                return False
        if not drained:
            raise ValueError('completion does not prove physical drain')
        view = self.views[ticket]
        if len(results) != len(view.rows):
            raise ValueError('result shape does not match frozen view')
        for ref in view.refs:
            if ticket not in self.get_page(ref).users:
                raise ProtocolError('completion has no matching outstanding execution use')
        for s in view.segments:
            r = self.requests[s.key]
            if r.status == 'live' and r.flight == ticket:
                if self.bug != 'skip_k_commit':
                    r.k = s.start + 1 if self.bug == 'use_start_plus_one' else s.end
                if s.end == s.n0:
                    if len(r.tokens) >= r.nmax or len(r.generated) >= r.limit:
                        r.status = 'ended'
                    else:
                        row = s.row0 if self.bug == 'first_row_sample' else s.row1 - 1
                        token = results[row]
                        if token == 'EOS':
                            r.status = 'ended'
                        elif self.bug != 'skip_append':
                            seq = len(r.generated)
                            r.tokens.append(token)
                            r.generated.append(token)
                            self.outputs.append((r.key, seq, token))
                            if len(r.generated) >= r.limit or len(r.tokens) >= r.nmax:
                                r.status = 'ended'
                if r.status == 'ended':
                    self.retire_owner(r)
            if r.flight == ticket:
                r.flight = None
        for ref in view.refs:
            self.get_page(ref).users.remove(ticket)
        self.pending.discard(ticket)
        self.done[ticket] = payload
        self.sweep()
        return True


# All numerical endpoints, generated tokens and page totals below are oracle data.
# Engine never reads these expected states.
def S(n, k, generated=(), flight=None, status='live', owners=0):
    return (n, k, tuple(generated), flight, status, owners)


def E(specs, uses=(), occupied=0, done=()):
    return (tuple(specs), tuple(uses), occupied, tuple(done))


A, B, C, BNEW = ('A', 0), ('B', 0), ('C', 0), ('B', 1)
PA = ('a0', 'a1', 'a2', 'a3', 'a4', 'a5')
PB = ('b0', 'b1', 'b2', 'b3', 'b4')
PC = ('c0', 'c1', 'c2')
PU = ('u0', 'u1', 'u2', 'u3', 'u4')
R0 = ('wrong_a4', 'a6', 'discard_b5')
R1 = ('wrong_c0', 'wrong_c1', 'c3', 'a7')
A0 = S(6, 4, owners=1)
B0 = S(5, 4, owners=1)
AF0 = S(6, 4, flight='t0', owners=2)
BF0 = S(5, 4, flight='t0', owners=2)
BC = S(5, 4, flight='t0', status='canceled')
BD = S(5, 4, status='canceled')
C0 = S(3, 0)
A1 = S(7, 6, ('a6',), owners=2)
AF1 = S(7, 6, ('a6',), 't1', owners=2)
CF1 = S(3, 0, flight='t1', owners=1)
A2 = S(8, 7, ('a6', 'a7'), owners=2)
C1 = S(4, 3, ('c3',), owners=1)
U0 = S(5, 4, owners=1)
MAIN_INITIAL = ((A, PA, 4, 99, 100), (B, PB, 4, 99, 100))
MAIN_TRACE = (
    ('submit0', 'submit', ('t0', ((A, 2), (B, 1)), 3)),
    ('arrive_C', 'arrive', (C, PC, 0, 99, 100)),
    ('cancel_B', 'cancel', (B,)),
    ('repeat_cancel_B', 'cancel', (B,)),
    ('complete0', 'complete', ('t0', R0, True)),
    ('submit1', 'submit', ('t1', ((C, 3), (A, 1)), 4)),
    ('repeat0_while_t1_pending', 'complete', ('t0', R0, True)),
    ('complete1', 'complete', ('t1', R1, True)),
    ('reuse_B_lifetime', 'arrive', (BNEW, PU, 4, 99, 100)),
    ('repeat0_after_reuse', 'complete', ('t0', R0, True)),
    ('repeat1_after_reuse', 'complete', ('t1', R1, True)),
)
MAIN_EXPECTED = {
    'initial': E(((A, A0), (B, B0)), occupied=2),
    'submit0': E(((A, AF0), (B, BF0)), (('t0', 4),), 4),
    'arrive_C': E(((A, AF0), (B, BF0), (C, C0)), (('t0', 4),), 4),
    'cancel_B': E(((A, AF0), (B, BC), (C, C0)), (('t0', 4),), 4),
    'repeat_cancel_B': E(((A, AF0), (B, BC), (C, C0)), (('t0', 4),), 4),
    'complete0': E(((A, A1), (B, BD), (C, C0)), occupied=2, done=('t0',)),
    'submit1': E(((A, AF1), (B, BD), (C, CF1)), (('t1', 3),), 3, ('t0',)),
    'repeat0_while_t1_pending': E(((A, AF1), (B, BD), (C, CF1)), (('t1', 3),), 3, ('t0',)),
    'complete1': E(((A, A2), (B, BD), (C, C1)), occupied=3, done=('t0', 't1')),
    'reuse_B_lifetime': E(((A, A2), (B, BD), (C, C1), (BNEW, U0)), occupied=4, done=('t0', 't1')),
    'repeat0_after_reuse': E(((A, A2), (B, BD), (C, C1), (BNEW, U0)), occupied=4, done=('t0', 't1')),
    'repeat1_after_reuse': E(((A, A2), (B, BD), (C, C1), (BNEW, U0)), occupied=4, done=('t0', 't1')),
}
MAIN_ROWS = {
    't0': ((A, 4, 'a4', False), (A, 5, 'a5', True), (B, 4, 'b4', True)),
    't1': ((C, 0, 'c0', False), (C, 1, 'c1', False), (C, 2, 'c2', True), (A, 6, 'a6', True)),
}

D = ('D', 0)
PARTIAL = (
    'partial_prefill',
    ((D, ('d0', 'd1', 'd2'), 0, 3, 10),),
    (
        ('part_submit', 'submit', ('p0', ((D, 2),), 2)),
        ('part_done', 'complete', ('p0', ('must_not_sample0', 'must_not_sample1'), True)),
        ('part_repeat', 'complete', ('p0', ('must_not_sample0', 'must_not_sample1'), True)),
        ('last_submit', 'submit', ('p1', ((D, 1),), 1)),
        ('last_done', 'complete', ('p1', ('d3',), True)),
    ),
    {
        'initial': E(((D, S(3, 0)),)),
        'part_submit': E(((D, S(3, 0, flight='p0', owners=1)),), (('p0', 1),), 1),
        'part_done': E(((D, S(3, 2, owners=1)),), occupied=1, done=('p0',)),
        'part_repeat': E(((D, S(3, 2, owners=1)),), occupied=1, done=('p0',)),
        'last_submit': E(((D, S(3, 2, flight='p1', owners=1)),), (('p1', 1),), 1, ('p0',)),
        'last_done': E(((D, S(4, 3, ('d3',), owners=1)),), occupied=1, done=('p0', 'p1')),
    },
    {
        'p0': ((D, 0, 'd0', False), (D, 1, 'd1', False)),
        'p1': ((D, 2, 'd2', True),),
    },
)

R = ('R', 0)
RECOVERY = (
    'complete_cache_hit',
    ((R, ('r0', 'r1', 'r2'), 3, 3, 10),),
    (
        ('recover_submit', 'submit', ('r0', ((R, 1),), 1)),
        ('recover_done', 'complete', ('r0', ('r3',), True)),
    ),
    {
        'initial': E(((R, S(3, 2, owners=1)),), occupied=1),
        'recover_submit': E(((R, S(3, 2, flight='r0', owners=1)),), (('r0', 1),), 1),
        'recover_done': E(((R, S(4, 3, ('r3',), owners=1)),), occupied=1, done=('r0',)),
    },
    {'r0': ((R, 2, 'r2', True),)},
)


def stopping_suite(name, token, limit, nmax, final_n, final_generated):
    key = (name, 0)
    initial = ((key, ('input',), 1, limit, nmax),)
    results = (token,)
    trace = (
        ('stop_submit', 'submit', ('stop', ((key, 1),), 1)),
        ('stop_done', 'complete', ('stop', results, True)),
        ('stop_repeat', 'complete', ('stop', results, True)),
    )
    final = S(final_n, 1, final_generated, status='ended')
    oracle = {
        'initial': E(((key, S(1, 0)),)),
        'stop_submit': E(((key, S(1, 0, flight='stop', owners=1)),), (('stop', 1),), 1),
        'stop_done': E(((key, final),), done=('stop',)),
        'stop_repeat': E(((key, final),), done=('stop',)),
    }
    return (name, initial, trace, oracle, {'stop': ((key, 0, 'input', True),)})


SUITES = (
    ('mixed_cancel_reuse', MAIN_INITIAL, MAIN_TRACE, MAIN_EXPECTED, MAIN_ROWS),
    PARTIAL,
    RECOVERY,
    stopping_suite('eos', 'EOS', 3, 10, 1, ()),
    stopping_suite('generation_limit', 'accepted', 1, 10, 2, ('accepted',)),
    stopping_suite('position_limit', 'must_not_accept', 3, 1, 1, ()),
)
FAULTS = (
    'advance_on_submit', 'skip_k_commit', 'use_start_plus_one',
    'skip_append', 'first_row_sample', 'no_dedup',
    'cancel_drops_execution', 'missing_execution_pin', 'recovery_stall',
)


def observe(engine):
    states = []
    for key, r in sorted(engine.requests.items()):
        states.append((key, len(r.tokens), r.k, tuple(r.generated),
                       r.flight, r.status, len(r.pages)))
    uses = Counter(t for p in engine.pages for t in p.users)
    occupied = sum(p.allocated for p in engine.pages)
    return (tuple(states), tuple(sorted(uses.items())), occupied, tuple(sorted(engine.done)))


def audit(engine, expected, expected_rows):
    specs, expected_uses, expected_occupied, expected_done = expected
    errors = []

    def require(ok, message):
        if not ok:
            errors.append(message)

    wanted = dict(specs)
    require(set(engine.requests) == set(wanted), 'request lifetime set differs from independent reference')
    expected_outputs = []
    owners = {}
    for key, state in specs:
        if key not in engine.requests:
            continue
        r = engine.requests[key]
        n, k, generated, flight, status, owned = state
        actual = (len(r.tokens), r.k, tuple(r.generated), r.flight, r.status, len(r.pages))
        require(actual == state, 'request state mismatch for ' + repr(key) + ': ' + repr(actual) + ' expected ' + repr(state))
        require(tuple(r.tokens) == engine.prompts[key] + generated,
                'actual appended token sequence differs from supplied prompt plus oracle samples')
        require(0 <= r.k <= len(r.tokens), 'KV boundary outside confirmed tokens')
        if r.status == 'live':
            require(len(r.pages) * PAGE >= r.k, 'live KV lacks owned capacity')
        require(len(set(r.pages)) == len(r.pages), 'duplicate request page mapping')
        for ref in r.pages:
            require(ref not in owners, 'private page has two logical owners')
            owners[ref] = key
        expected_outputs.extend((key, i, token) for i, token in enumerate(generated))
    require(sorted(engine.outputs) == sorted(expected_outputs), 'output identity, sequence or token differs from reference')
    require(len({(key, i) for key, i, token in engine.outputs}) == len(engine.outputs), 'duplicate output commit')
    require(set(engine.done) == set(expected_done), 'completed credential receipts differ from reference')
    require(engine.pending == {t for t, count in expected_uses}, 'pending credential set differs from reference')
    require({r.flight for r in engine.requests.values() if r.flight is not None} == engine.pending,
            'request flight and physical pending set disagree')
    require(set(engine.views) == set(expected_done) | {t for t, count in expected_uses},
            'submitted credential history differs from reference')
    for ticket, view in engine.views.items():
        require(ticket in expected_rows and view.rows == expected_rows[ticket], 'frozen row identity or terminal sampling target differs')
        for s in view.segments:
            require(s.start < s.end <= s.n0 and s.row1 - s.row0 == s.end - s.start,
                    'frozen segment range malformed')
            r = engine.requests[s.key]
            require(tuple(r.tokens[s.start:s.end]) == s.inputs, 'frozen input token range changed')
            if ticket in engine.pending:
                require(r.flight == ticket, 'pending segment lost execution authority record')
                require(r.k == s.start, 'confirmed KV advanced before completion')
    for i, p in enumerate(engine.pages):
        ref = Ref(i, p.epoch)
        desired_users = {t for t in engine.pending if ref in engine.views[t].refs}
        require(p.owner == owners.get(ref), 'physical owner differs from actual request ownership')
        require(p.users == desired_users, 'physical execution responsibilities differ from outstanding frozen views')
        require(p.allocated == bool(p.owner is not None or p.users), 'page allocation and last-use gate disagree')
    for ref in owners:
        p = engine.pages[ref.slot]
        require(p.allocated and p.epoch == ref.epoch, 'request owns stale or recycled page')
    for t in engine.pending:
        for ref in engine.views[t].refs:
            p = engine.pages[ref.slot]
            require(p.allocated and p.epoch == ref.epoch, 'pending execution refers to recycled page')
    uses = Counter(t for p in engine.pages for t in p.users)
    require(dict(uses) == dict(expected_uses), 'execution page counts differ from hand reference')
    require(sum(p.allocated for p in engine.pages) == expected_occupied,
            'unique physical page union differs from hand reference')
    require(sum(p.allocated for p in engine.pages) <= len(engine.pages), 'page pool capacity exceeded')
    return errors


def run_trace(suite, bug=None):
    name, initial, trace, oracle, rows = suite
    engine = Engine(bug)
    errors = []
    observations = []
    for args in initial:
        engine.arrive(*args)
    errors.extend(name + '/initial: ' + e for e in audit(engine, oracle['initial'], rows))
    observations.append(('initial', observe(engine)))
    for label, action, args in trace:
        try:
            result = getattr(engine, action)(*args)
            if action == 'submit' and result is not True:
                errors.append(name + '/' + label + ': executable reference submission was blocked')
        except (ValueError, KeyError, IndexError) as exc:
            errors.append(name + '/' + label + ': unexpected event rejection: ' + str(exc))
        errors.extend(name + '/' + label + ': ' + e for e in audit(engine, oracle[label], rows))
        observations.append((label, observe(engine)))
    return errors, observations


def boundary_checks():
    errors = []
    engine = Engine()
    key = ('boundary', 0)
    engine.arrive(key, ('x0', 'x1'))
    rejected = (
        ('empty_prompt', 'arrive', (('empty', 0), ())),
        ('overlong_prompt', 'arrive', (('long', 0), ('x', 'y'), 0, 1, 1)),
        ('zero_q', 'submit', ('bad0', ((key, 0),), 2)),
        ('unknown_input_q', 'submit', ('bad1', ((key, 3),), 3)),
        ('duplicate_request', 'submit', ('bad2', ((key, 1), (key, 1)), 2)),
        ('unknown_completion', 'complete', ('unknown', ('x',), True)),
    )
    for label, action, args in rejected:
        before = copy.deepcopy(engine.__dict__)
        try:
            getattr(engine, action)(*args)
        except ValueError:
            pass
        else:
            errors.append(label + ' was not rejected')
        if engine.__dict__ != before:
            errors.append(label + ' changed state on rejection')
    if not engine.submit('valid', ((key, 1),), 1):
        errors.append('valid partial view unexpectedly blocked')
    before = copy.deepcopy(engine.__dict__)
    for results, drained in ((('x',), False), ((), True)):
        try:
            engine.complete('valid', results, drained)
        except ValueError:
            pass
        else:
            errors.append('insufficient completion evidence was accepted')
        if engine.__dict__ != before:
            errors.append('rejected completion changed state or references')
    engine.complete('valid', ('unused_partial_row',), True)
    before = copy.deepcopy(engine.__dict__)
    try:
        engine.complete('valid', ('conflicting_payload',), True)
    except ValueError:
        pass
    else:
        errors.append('conflicting repeated payload accepted')
    if engine.__dict__ != before:
        errors.append('conflicting completion changed state')
    small = Engine(pool=1)
    small.arrive(('full', 0), ('f0', 'f1', 'f2', 'f3', 'f4'), 4)
    before = copy.deepcopy(small.__dict__)
    if small.submit('capacity', ((('full', 0), 1),), 1) is not False:
        errors.append('capacity exhaustion was not blocked')
    if small.__dict__ != before:
        errors.append('capacity failure changed ownership or progress')
    if small.submit('empty_plan', (), 0) is not False or small.__dict__ != before:
        errors.append('empty plan submitted work or changed state')
    zero = Engine()
    zero.arrive(('zero', 0), ('z0',), 0, 0, 10)
    zr = zero.requests[('zero', 0)]
    if zr.status != 'ended' or zr.pages or zero.views or zero.outputs:
        errors.append('zero generation quota created work or retained state')
    return errors


def cost_checks():
    def enumerated(q, h):
        return sum(1 for p in range(h, h + q) for key in range(p + 1))

    errors = []
    for h in (0, 1, 4, 16):
        for q in (1, 2, 3, 7):
            formula = q * h + q * (q + 1) // 2
            if formula != enumerated(q, h):
                errors.append('causal pair formula differs from explicit visibility enumeration')
    if enumerated(2, 4) + enumerated(1, 4) != 16:
        errors.append('round0 attention reference mismatch')
    if enumerated(3, 0) + enumerated(1, 6) != 13:
        errors.append('round1 attention reference mismatch')
    return errors


def main():
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    inputs = {'page': PAGE, 'pool': POOL, 'suites': SUITES, 'mutations': FAULTS,
              'additional_boundaries': 'deterministic literals in bound source'}
    input_bytes = json.dumps(inputs, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode('utf-8')
    print('script_sha256=' + source_sha)
    print('input_sha256=' + hashlib.sha256(input_bytes).hexdigest())
    print('scope=CPU symbolic request transitions, explicit sampled tokens, private page identities and execution responsibilities; not GPU or engine measurement')
    failures = []
    for suite in SUITES:
        errors, observations = run_trace(suite)
        failures.extend(errors)
        if suite[0] == 'mixed_cancel_reuse':
            for label, value in observations:
                print('observation event=' + label + ' state=' + repr(value))
        else:
            print('observation suite=' + suite[0] + ' final=' + repr(observations[-1][1]) + ' discrepancies=' + str(len(errors)))
    failures.extend(boundary_checks())
    failures.extend(cost_checks())
    for bug in FAULTS:
        mutant_errors = []
        for suite in SUITES:
            errors, observations = run_trace(suite, bug)
            mutant_errors.extend(errors)
        detected = bool(mutant_errors)
        print('negative_control bug=' + bug + ' detected=' + str(detected) +
              ' first_discrepancy=' + repr(mutant_errors[0] if detected else None))
        if not detected:
            failures.append('mutation survived shared trace and auditor: ' + bug)
    if failures:
        print('check_status=failed discrepancies=' + repr(failures))
        raise SystemExit(1)
    print('check_status=ok within_declared_symbolic_scope')


if __name__ == '__main__':
    main()
