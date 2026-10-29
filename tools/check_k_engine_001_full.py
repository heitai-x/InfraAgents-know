# K-ENGINE-001: deterministic, standalone CPU symbolic check.
# Standard library only; no file access, network, processes, or device operations.
from dataclasses import dataclass, replace
from itertools import product


class AuditError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit_equal(actual, expected):
    if actual != expected:
        raise AuditError('actual=' + repr(actual) + ' expected=' + repr(expected))


def pages(tokens, size):
    return (tokens + size - 1) // size


@dataclass(frozen=True)
class Ticket:
    tid: tuple
    gen: int
    ver: int
    pos: int
    eta: str = 'model-v1'


@dataclass(frozen=True)
class Req:
    rid: str
    gen: int
    x: tuple
    c: int
    s: int
    a: int
    enc: tuple
    ver: int = 0
    active: object = None
    log: object = None
    ready: bool = True
    cancelled: bool = False
    done: bool = False
    draining: bool = False
    out: tuple = ()
    consumed: tuple = ()
    limit: int = 3
    model_limit: int = 12
    eta: str = 'model-v1'

    @property
    def n(self):
        return len(self.x)


@dataclass(frozen=True)
class Action:
    rid: str
    gen: int
    view: int
    ver: int
    n: int
    old_c: int
    old_s: int
    old_a: int
    start: int
    q: int
    kind: str
    enc: int
    new_pages: int
    eta: str = 'model-v1'

    @property
    def key(self):
        return (self.rid, self.gen, self.view)


@dataclass(frozen=True)
class Plan:
    actions: tuple
    samples: tuple
    states: tuple
    free: int
    reasons: tuple


@dataclass(frozen=True)
class Event:
    action: Action
    status: str = 'OK'
    coverage: bool = True
    ended: bool = True


def make(rid='a', n=3, c=0, a=None, enc=0, **kw):
    if a is None:
        a = c
    return Req(rid, 1, tuple(range(n)), c, c, a, (enc,) * n, **kw)


def valid(r):
    return (r.n >= 1 and len(r.enc) == r.n and
            all(e >= 0 for e in r.enc) and
            0 <= r.c <= r.s <= r.n <= r.model_limit and r.s <= r.a and
            (r.active is not None or r.s == r.c))


def ticket_valid(r):
    t = r.log
    return (t is not None and
            (t.gen, t.ver, t.pos, t.eta) ==
            (r.gen, r.ver, r.n - 1, r.eta))


def schedule(reqs, budget, free, size=2, cap=3, view=1, mutant=''):
    require(all(b >= 0 for b in budget) and free >= 0 and size > 0 and cap >= 0,
            'invalid scheduler configuration')
    require(len({r.rid for r in reqs}) == len(reqs), 'duplicate candidate request')
    bt, bs, be = budget
    ut = us = ue = 0
    actions, samples, states, reasons = [], [], list(reqs), []
    for index, r in enumerate(reqs):
        if not valid(r):
            reasons.append((r.rid, 'invalid'))
            continue
        if r.cancelled or r.done:
            reasons.append((r.rid, 'terminal'))
            continue
        if len(r.out) >= r.limit or r.n == r.model_limit:
            states[index] = replace(r, done=True)
            continue
        if r.active is not None or r.draining or not r.ready:
            reasons.append((r.rid, 'wait'))
            if mutant == 'drop_existing_active' and r.active is not None:
                states[index] = replace(r, active=None)
            continue
        if r.c == r.n and ticket_valid(r):
            samples.append((r.rid, r.gen, r.log.tid))
            continue
        if r.log is not None:
            reasons.append((r.rid, 'invalid_ticket'))
            continue
        recovery = r.c == r.n
        kind = 'RECOVER_LAST' if recovery else 'COMPUTE'
        start = r.n - 1 if recovery else r.s
        upper = min(cap, 1 if recovery else r.n - r.s)
        if mutant == 'capacity_cap' and not recovery:
            upper = min(upper, r.a - r.s)
        if mutant == 'omit_recovery' and recovery:
            reasons.append((r.rid, 'no_work'))
            continue
        accepted = False
        quantities = (1,) if recovery and upper >= 1 else range(upper, 0, -1)
        for q in quantities:
            e = sum(r.enc[start:start + q])
            seq_need = 0 if mutant == 'recover_no_seq' and recovery else 1
            if mutant != 'ignore_tok' and ut + q > bt:
                continue
            if mutant != 'ignore_seq' and us + seq_need > bs:
                continue
            if mutant != 'ignore_enc' and ue + e > be:
                continue
            need = 1 if recovery else max(0, pages(start + q, size) - pages(r.a, size))
            if mutant == 'no_reserve':
                need = 0
            if need > free:
                continue
            action = Action(r.rid, r.gen, view, r.ver, r.n, r.c, r.s,
                            r.a, start, q, kind, e, need, r.eta)
            new_s = r.s if recovery else start + q
            new_a = r.a if recovery else max(r.a, pages(start + q, size) * size)
            new_c = new_s if mutant == 'confirm_on_schedule' else r.c
            states[index] = replace(r, s=new_s, a=new_a, c=new_c, active=action.key)
            actions.append(action)
            free -= need
            ut += q
            us += seq_need
            ue += e
            accepted = True
            break
        if not accepted:
            reasons.append((r.rid, 'budget_or_capacity'))
        if accepted and mutant == 'zero_encoder_break' and ue == be:
            break
    return Plan(tuple(actions), tuple(samples), tuple(states), free, tuple(reasons))


def signature(a):
    return (a.rid, a.gen, a.view, a.start, a.q, a.kind, a.enc, a.new_pages, a.eta)


def oracle(reqs, budget, free, size=2, cap=3, view=1):
    # Independent reference: enumerate lengths and page-index sets from raw input.
    # Does not call schedule, valid, ticket_valid, or pages.
    bt, bs, be = budget
    rows = sequences = encoder = 0
    expected, samples, state = [], [], []
    for r in reqs:
        state.append((r.c, r.s, r.a, r.active, r.done))
        sane = (len(r.x) > 0 and len(r.enc) == len(r.x) and
                all(e >= 0 for e in r.enc) and
                0 <= r.c <= r.s <= len(r.x) <= r.model_limit and r.s <= r.a and
                (r.active is not None or r.s == r.c))
        if not sane or r.cancelled or r.done:
            continue
        if len(r.out) >= r.limit or len(r.x) == r.model_limit:
            state[-1] = (r.c, r.s, r.a, r.active, True)
            continue
        if r.active is not None or r.draining or not r.ready:
            continue
        n = len(r.x)
        if r.c == n and r.log is not None:
            if (r.log.gen, r.log.ver, r.log.pos, r.log.eta) == (r.gen, r.ver, n - 1, r.eta):
                samples.append((r.rid, r.gen, r.log.tid))
            continue
        if r.log is not None:
            continue
        replay = r.c == n
        candidates = []
        for length in range(1, cap + 1):
            if replay:
                if length != 1:
                    continue
                start, kind = n - 1, 'RECOVER_LAST'
                count_pages = 1
            else:
                if length > n - r.c:
                    continue
                start, kind = r.c, 'COMPUTE'
                held = {p // size for p in range(r.a)}
                required_pages = {p // size for p in range(start + length)}
                count_pages = len(required_pages - held)
            encoder_need = 0
            for position in range(start, start + length):
                encoder_need += r.enc[position]
            if (rows + length <= bt and sequences + 1 <= bs and
                    encoder + encoder_need <= be and count_pages <= free):
                candidates.append((length, start, kind, encoder_need, count_pages))
        if not candidates:
            continue
        length, start, kind, e, p = max(candidates)
        expected.append((r.rid, r.gen, view, start, length, kind, e, p, r.eta))
        if replay:
            end_s, end_a = r.s, r.a
        else:
            end_s = start + length
            page_positions = {k // size for k in range(end_s)}
            end_a = max(r.a, len(page_positions) * size)
        state[-1] = (r.c, end_s, end_a, (r.rid, r.gen, view), r.done)
        rows += length
        sequences += 1
        encoder += e
        free -= p
    return (tuple(expected), tuple(samples), tuple(state), free)


def audit_plan(plan, expected, before):
    actual = (tuple(signature(a) for a in plan.actions), plan.samples,
              tuple((r.c, r.s, r.a, r.active, r.done) for r in plan.states), plan.free)
    audit_equal(actual, expected)
    audit_equal(len(plan.states), len(before))
    audit_equal(len({a.key for a in plan.actions}), len(plan.actions))
    for old, current in zip(before, plan.states):
        newly_selected = [a for a in plan.actions if a.rid == old.rid]
        if not newly_selected:
            # An existing active belongs to an earlier view, not this round's actions.
            # Preserve every field except an independently expected terminal flag.
            audit_equal(current, replace(old, done=current.done))
            audit_equal(current.active, old.active)
            continue
        audit_equal(len(newly_selected), 1)
        a = newly_selected[0]
        audit_equal(old.active, None)
        audit_equal((a.gen, a.ver, a.n, a.old_c, a.old_s, a.old_a, a.eta),
                    (old.gen, old.ver, old.n, old.c, old.s, old.a, old.eta))
        audit_equal(current.active, a.key)
        audit_equal(current.c, old.c)
        audit_equal((current.x, current.enc, current.ver, current.log, current.consumed),
                    (old.x, old.enc, old.ver, old.log, old.consumed))
        audit_equal(a.q > 0, True)


def checked(reqs, budget, free, size=2, cap=3, view=1):
    expected = oracle(reqs, budget, free, size, cap, view)
    plan = schedule(reqs, budget, free, size, cap, view)
    audit_plan(plan, expected, reqs)
    return plan, expected


def resolve(r, a, event, settled=frozenset(), mutant=''):
    # Certificates are external symbols; no concrete physical allocator is simulated.
    if event.action != a:
        return r, settled, 'REJECT'
    if mutant != 'event_only':
        if a.key in settled:
            return r, settled, 'DUPLICATE'
        authorized = (r.rid == a.rid and r.gen == a.gen and r.ver == a.ver and
                      r.eta == a.eta and r.active == a.key and r.n == a.n and
                      r.c == a.old_c and not r.cancelled and not r.done)
        if a.kind == 'COMPUTE':
            authorized = authorized and a.start == a.old_c and r.s == a.start + a.q
        else:
            authorized = authorized and r.c == r.s == r.n and a.start == r.n - 1 and a.q == 1
        if not authorized:
            return r, settled, 'STALE'
    if event.status == 'FAIL':
        rollback_s = a.start if mutant == 'rollback_all' else a.old_s
        out = replace(r, s=rollback_s, a=a.old_a, active=None, ver=r.ver + 1,
                      log=None, draining=not event.ended)
        return out, settled | {a.key}, 'FAILED'
    if event.status != 'OK' or not event.coverage:
        return r, settled, 'WAIT'
    new_c = a.start + a.q if a.kind == 'COMPUTE' else r.c
    new_ver = r.ver + 1
    ticket = None
    if new_c == r.n and len(r.out) < r.limit and r.n < r.model_limit:
        ticket = Ticket(a.key, r.gen, new_ver, r.n - 1, r.eta)
    out = replace(r, c=new_c, s=new_c, active=None, ver=new_ver, log=ticket,
                  draining=not event.ended)
    return out, settled | {a.key}, 'OK'


def accept(r, ticket, token, mutant=''):
    if any(tid == ticket.tid for tid, _ in r.consumed):
        return r, 'REPLAY'
    if (r.cancelled or r.done or r.active is not None or r.draining or
            r.log != ticket or not ticket_valid(r) or r.c != r.s or r.s != r.n):
        return r, 'REJECT'
    if len(r.out) >= r.limit or r.n == r.model_limit:
        return replace(r, done=True), 'STOP'
    record = r.consumed + ((ticket.tid, token),)
    if token == 'EOS':
        return replace(r, done=True, log=None, ver=r.ver + 1, consumed=record), 'STOP'
    new_x = r.x if mutant == 'omit_append' else r.x + (token,)
    new_enc = r.enc if mutant == 'omit_append' else r.enc + (0,)
    new_out = r.out + (token,)
    done = len(new_out) >= r.limit or len(new_x) == r.model_limit
    return replace(r, x=new_x, enc=new_enc, out=new_out, log=None,
                   ver=r.ver + 1, consumed=record, done=done), 'ACCEPTED'


def cancel(r, a, ended=False):
    return replace(r, cancelled=True, ver=r.ver + 1, s=a.old_s, a=a.old_a,
                   active=None, log=None, draining=not ended)


def detect(name, audit):
    # Setup/reference/normal checks happen outside this capture domain.
    # Only a mismatch produced by auditing a preconstructed mutant is expected.
    try:
        audit()
    except AuditError:
        print('negative_control_detected=' + name)
        return 1
    raise RuntimeError('negative control survived: ' + name)


def main():
    cases = 0
    detected = 0
    for n in range(1, 5):
        for c in range(n + 1):
            for rate, extra, free, bt, bs, be in product(
                    (0, 1), (0, 2), range(3), range(5), range(2), range(3)):
                r = make(n=n, c=c, a=c + extra, enc=rate)
                checked((r,), (bt, bs, be), free)
                cases += 1
    recs = (make('a', 2, 2), make('b', 3, 3))
    mixed = (make('a', 3, 0, enc=1), make('b', 2, 0), make('c', 2, 2))
    for reqs in (recs, mixed):
        for bt, bs, be, free in product(range(4), range(3), range(4), range(4)):
            checked(reqs, (bt, bs, be), free)
            cases += 1
    ticket = Ticket(('cached', 1), 1, 0, 1)
    sample = make(n=2, c=2, log=ticket)
    p, _ = checked((sample,), (0, 0, 0), 0)
    require(len(p.samples) == 1 and not p.actions, 'sample notification lost at zero budget')
    for r in (make(ready=False), make(draining=True), make(cancelled=True),
              replace(make(), active=('a', 1, 99))):
        checked((r,), (3, 2, 3), 3)
    checked((make(),), (3, 2, 0), 3, cap=0)
    for r in (make(n=0), replace(make(), s=1), replace(make(), c=-1)):
        p, _ = checked((r,), (3, 2, 3), 3)
        require(p.states == (r,) and not p.actions, 'invalid state changed')

    # Regression for round-2 actual=0 expected=1: old active is not a new action.
    old_compute = replace(make(n=3, c=1, a=2), s=2, active=('a', 1, 90))
    old_recovery = replace(make(n=2, c=2, a=2), active=('a', 1, 91))
    for old in (old_compute, old_recovery):
        p, _ = checked((old,), (3, 2, 3), 3, view=100)
        require(p.states == (old,) and not p.actions, 'old view was not preserved')
        p2, _ = checked(p.states, (3, 2, 3), p.free, view=101)
        require(p2.states == (old,) and not p2.actions, 'repeat scan duplicated old work')
        paired, _ = checked((old, make('b', n=1)), (3, 2, 0), 3, view=102)
        require(paired.states[0] == old and len(paired.actions) == 1 and
                paired.actions[0].rid == 'b', 'mixed old/new view regression')
    print('existing_views=compute_and_recovery_preserved_across_repeated_and_mixed_scans')

    growing = make(n=3, c=2, a=2)
    p, _ = checked((growing,), (1, 1, 0), 1)
    require((p.states[0].a, p.states[0].s, p.states[0].c) == (4, 3, 2),
            'capacity growth did not precede scheduled advance')
    p, _ = checked((growing,), (1, 1, 0), 0)
    require(p.states == (growing,) and p.free == 0, 'reservation failure changed state')
    print('scheduler_reference_cases=' + str(cases))

    mutant_cases = (
        ('ignore_tok', recs, (0, 2, 0), 2),
        ('ignore_seq', (make(n=2),), (2, 0, 0), 2),
        ('ignore_enc', (make(n=2, enc=1),), (2, 1, 0), 2),
        ('zero_encoder_break', (make('a', 1), make('b', 1)), (2, 2, 0), 2),
        ('capacity_cap', (growing,), (1, 1, 0), 1),
        ('recover_no_seq', recs, (2, 1, 0), 2),
        ('omit_recovery', (make(n=2, c=2),), (1, 1, 0), 1),
        ('no_reserve', (growing,), (1, 1, 0), 0),
        ('confirm_on_schedule', (make(),), (2, 1, 0), 2),
        ('drop_existing_active', (old_compute,), (3, 2, 3), 3),
    )
    for name, reqs, budget, free in mutant_cases:
        _, expected = checked(reqs, budget, free)
        bad = schedule(reqs, budget, free, mutant=name)
        detected += detect(name, lambda b=bad, e=expected, r=reqs: audit_plan(b, e, r))

    original = make(n=3, c=1, a=2)
    p, _ = checked((original,), (1, 1, 0), 1)
    a = p.actions[0]
    flight = p.states[0]
    event = Event(a)
    good, ledger, status = resolve(flight, a, event)
    require((good.c, good.s, good.n, good.active, status) == (2, 2, 3, None, 'OK'),
            'normal completion frontier')
    replay, ledger2, status = resolve(good, a, event, ledger)
    require(replay == good and ledger2 == ledger and status == 'DUPLICATE', 'duplicate commit')
    require(resolve(flight, a, Event(a, coverage=False)) == (flight, frozenset(), 'WAIT'),
            'incomplete evidence installed result')
    require(resolve(flight, a, Event(a, status='PENDING')) == (flight, frozenset(), 'WAIT'),
            'pending result fell through')
    require(resolve(flight, a, Event(replace(a, view=99))) == (flight, frozenset(), 'REJECT'),
            'identity mismatch fell through')
    failed, _, _ = resolve(flight, a, Event(a, status='FAIL', ended=False))
    require((failed.c, failed.s, failed.a, failed.draining) == (1, 1, 2, True),
            'compute failure rollback')
    blocked, _ = checked((failed,), (1, 1, 0), 1, view=2)
    require(not blocked.actions, 'retry overlapped unresolved accesses')
    drained = replace(failed, draining=False)
    retry, _ = checked((drained,), (1, 1, 0), 1, view=2)
    require(len(retry.actions) == 1, 'compute failure not retryable after drain')

    changed_states = (
        replace(original, gen=2),
        replace(flight, active=('a', 1, 99)),
        replace(flight, ver=flight.ver + 1),
        replace(flight, eta='model-v2'),
        cancel(flight, a),
    )
    for i, current in enumerate(changed_states):
        # Old event and old entry match each other in every case.
        normal = resolve(current, a, event)[0]
        audit_equal(normal, current)
        bad = resolve(current, a, event, mutant='event_only')[0]
        detected += detect('event_only_authority_' + str(i),
                           lambda b=bad, e=current: audit_equal(b, e))

    cached = make(n=4, c=4, a=4)
    p, _ = checked((cached,), (1, 1, 0), 1, view=10)
    recovery = p.actions[0]
    require(recovery.kind == 'RECOVER_LAST' and p.states[0].active == recovery.key,
            'recovery missing frozen active item')
    recovering = p.states[0]
    failed, _, _ = resolve(recovering, recovery, Event(recovery, status='FAIL'))
    expected_fail = replace(recovering, active=None, ver=1)
    audit_equal(failed, expected_fail)
    bad = resolve(recovering, recovery, Event(recovery, status='FAIL'), mutant='rollback_all')[0]
    detected += detect('recovery_rollback', lambda: audit_equal(bad, expected_fail))
    require(valid(failed), 'recovery failure broke frontier invariant')
    again, _ = checked((failed,), (1, 1, 0), 1, view=11)
    require(len(again.actions) == 1, 'recovery failure not retryable')
    ar = again.actions[0]
    restored, _, _ = resolve(again.states[0], ar, Event(ar))
    require((restored.c, restored.s, restored.n, restored.log.pos) == (4, 4, 4, 3),
            'recovery should only install last-row ticket')
    ready, _ = checked((restored,), (0, 0, 0), 1, view=12)
    require(ready.samples == (('a', 1, restored.log.tid),), 'recovery sampling notification missing')
    accepted, status = accept(restored, restored.log, 'z')
    expected_accept = replace(restored, x=restored.x + ('z',), enc=restored.enc + (0,),
                              out=('z',), log=None, ver=restored.ver + 1,
                              consumed=((restored.log.tid, 'z'),))
    audit_equal(accepted, expected_accept)
    require(status == 'ACCEPTED' and (accepted.c, accepted.s, accepted.n, accepted.a) == (4, 4, 5, 4),
            'sampling did not create known decode input')
    bad_accept = accept(restored, restored.log, 'z', mutant='omit_append')[0]
    detected += detect('sample_omit_append', lambda: audit_equal(bad_accept, expected_accept))
    duplicate, status = accept(accepted, restored.log, 'different_candidate')
    require(duplicate == accepted and status == 'REPLAY', 'ticket consumed twice')
    rejected, status = accept(replace(restored, cancelled=True), restored.log, 'z')
    require(rejected == replace(restored, cancelled=True) and status == 'REJECT',
            'cancelled sample was accepted')
    decode, _ = checked((accepted,), (1, 1, 0), 1, view=13)
    require((decode.actions[0].start, decode.actions[0].q, decode.states[0].a) == (4, 1, 6),
            'recovery-sampling-decode chain disconnected')
    decoded, _, _ = resolve(decode.states[0], decode.actions[0], Event(decode.actions[0]))
    stopped, status = accept(decoded, decoded.log, 'EOS')
    require(stopped.done and stopped.x == decoded.x and status == 'STOP', 'EOS contract')
    final_quota = replace(restored, out=('u', 'v'))
    final, _ = accept(final_quota, final_quota.log, 'z')
    require(final.done and len(final.out) == 3, 'output quota stop')
    edge = make(n=11, c=11, a=12, log=Ticket(('edge', 1), 1, 0, 10))
    edge_done, _ = accept(edge, edge.log, 'z')
    require(edge_done.done and edge_done.n == 12 and edge_done.c == 11, 'model length stop')
    print('protocol=recovery_failure_retry_sample_decode_authority_duplicates_evidence')

    for h, q in product(range(6), range(1, 5)):
        enumerated_pairs = sum(len(tuple(range(p + 1))) for p in range(h, h + q))
        audit_equal(q * h + q * (q + 1) // 2, enumerated_pairs)
    for size, held, end in ((16, 32, 33), (16, 32, 49), (2, 4, 5)):
        expected_pages = len({p // size for p in range(end)} - {p // size for p in range(held)})
        audit_equal(max(0, pages(end, size) - pages(held, size)), expected_pages)
    bytes_by_components = sum(2 for _layer in range(32) for _head in range(8)
                              for _component in range(128 + 128))
    audit_equal(32 * 8 * (128 + 128) * 2, bytes_by_components)
    audit_equal(bytes_by_components, 131072)
    wrong_units = 32 * bytes_by_components
    detected += detect('kv_double_layer_count', lambda: audit_equal(wrong_units, bytes_by_components))
    print('cost_scope=causal_pairs_page_sets_all_layer_kv_units')
    print('negative_controls_detected=' + str(detected))
    print('scope=cpu_symbolic_scheduler_and_serial_logical_protocol')
    print('encoder_scope=integer_costs_without_shared_blocks_or_feature_execution')
    print('physical_scope=external_completion_symbols_and_conflict_gate_only')
    print('sampling_scope=externally_supplied_tokens_without_rng_or_probability_model')
    print('device_measurements=none')


if __name__ == '__main__':
    main()
