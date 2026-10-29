#!/usr/bin/env python3
"""K-HW-003: deterministic CPU arithmetic and protocol models only.
Standard library, standalone, offline, read-only; no CUDA or DMA execution.
"""
from fractions import Fraction
from itertools import product


END = 'END_UNCONFIRMED'
ACCEPTANCE = 'ACCEPTANCE_UNKNOWN'
IDENTITY = 'IDENTITY_REVIEW'


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def rejected(action):
    try:
        action()
    except AssertionError:
        return True
    return False


def negative(label, action):
    check(rejected(action), 'negative control escaped: ' + label)
    print('negative_control_detected:', label)


def calendar(times, batches):
    available = [0] * len(times)
    for _ in range(batches):
        predecessor = 0
        for stage, service in enumerate(times):
            finish = max(available[stage], predecessor) + service
            available[stage] = finish
            predecessor = finish
    return available[-1]


def formula(times, batches):
    return 0 if batches == 0 else sum(times) + (batches - 1) * max(times)


def interval_peak(length, period, batches):
    intervals = [(i * period, i * period + length) for i in range(batches)]
    return max([0] + [sum(start <= t < end for start, end in intervals)
                      for t, _ in intervals])


def arithmetic_checks():
    count = 0
    for times in product(range(5), repeat=4):
        for batches in range(10):
            check(calendar(times, batches) == formula(times, batches),
                  ('calendar/formula', times, batches))
            count += 1
    check(calendar((3, 2, 8, 1), 8) == 70, 'eight-batch example')
    print('observed independent_pipeline_cases:', count)
    slot_cases = 0
    for length in range(33):
        for period in range(1, 13):
            required = (length + period - 1) // period
            check(interval_peak(length, period, 40) == required,
                  ('half-open occupancy', length, period))
            slot_cases += 1
    check(interval_peak(10, 8, 8) == 2, 'device input slots')
    check(interval_peak(9, 8, 8) == 2, 'device output slots')
    check(interval_peak(100, 1, 2) == 2, 'finite workload caps occupancy')
    print('observed half_open_slot_cases:', slot_cases)
    negative('floor instead of ceiling',
             lambda: check(10 // 4 >= interval_peak(10, 4, 40),
                           'two slots cannot cover three simultaneous users'))
    shared_copy_lower_bound = 1 + 5 * (8 + 4)
    wrong = formula((1, 8, 2, 4), 5)
    check((shared_copy_lower_bound, wrong) == (61, 47), 'copy example')
    negative('independent directions on one copy engine',
             lambda: check(wrong >= shared_copy_lower_bound,
                           'schedule violates total copy service demand'))
    bound_ms = Fraction(128, 8 * 1024) * 1000
    check(bound_ms == Fraction(125, 8), 'DRAM units')
    check(Fraction(8) < bound_ms, '8 ms period excluded by DRAM demand')
    print('observed copy_engine_lower_bound_ms:', shared_copy_lower_bound)
    print('observed shared_dram_lower_bound_ms:', str(bound_ms))


class Buffer:
    """Local range model for publication, holds, and named consumers."""
    def __init__(self):
        self.owner = None
        self.gen = 0
        self.serial = 0
        self.hold = False
        self.writer = None
        self.ready = False
        self.canceled = False
        self.readers = set()
        self.done = set()
        self.data = None

    def acquire(self, request):
        check(self.owner is None and not self.readers and self.writer is None,
              'range is still owned or accessed')
        self.gen += 1
        self.owner = request
        self.hold = True
        self.ready = False
        self.canceled = False
        self.writer = (self.gen, request, 'writer')
        return self.writer

    def publish(self, writer, values, full=True, ended=True):
        check(writer == self.writer, 'wrong writer identity')
        if not ended:
            return False
        self.writer = None
        if full and not self.canceled:
            self.data = tuple(values)
            self.ready = True
            self.hold = True
            return True
        self.data = None
        self.ready = False
        self.hold = False
        return False

    def start_readers(self, names):
        check(self.ready and self.hold and self.writer is None
              and not self.canceled, 'source is not available for publication')
        check(names and len(set(names)) == len(names), 'invalid consumer group')
        tickets = []
        for name in names:
            self.serial += 1
            ticket = (self.gen, self.owner, self.serial, name)
            tickets.append(ticket)
        self.readers.update(tickets)
        self.hold = False
        return tuple(tickets)

    def finish(self, ticket, ended=True):
        if ticket in self.done:
            return None
        if not ended:
            return None
        check(ticket in self.readers, 'unknown live responsibility')
        check(ticket[:2] == (self.gen, self.owner), 'live access met a reused range')
        payload = self.data
        self.readers.remove(ticket)
        self.done.add(ticket)
        return payload

    def cancel(self):
        self.canceled = True
        self.hold = False
        self.ready = False

    def release(self):
        check(self.owner is not None and not self.hold
              and self.writer is None and not self.readers,
              'release before all holds and accesses ended')
        self.owner = None
        self.ready = False
        self.data = None


def ready_buffer(values, request='q'):
    buf = Buffer()
    writer = buf.acquire(request)
    check(buf.publish(writer, values), 'full publication')
    return buf


def pipeline_payload(raw):
    h, d, o, r = Buffer(), Buffer(), Buffer(), Buffer()
    hw = h.acquire('q')
    check(h.publish(hw, [x + 1 for x in raw]), 'prepare')
    check(rejected(h.release), 'READY hold must protect unclaimed data')
    ht, = h.start_readers(['H2D'])
    check(rejected(lambda: h.start_readers(['duplicate H2D'])),
          'source must leave the publishable state at submission')
    check(rejected(lambda: h.acquire('next')), 'in-flight source protection')
    h.finish(ht, ended=False)
    check(ht in h.readers, 'missing end certificate retained')
    dw = d.acquire('q')
    copied = h.finish(ht)
    check(d.publish(dw, copied), 'H2D completion')
    h.release()
    kt, = d.start_readers(['kernel'])
    ow = o.acquire('q')
    device_input = d.finish(kt)
    computed = [3 * x + i for i, x in enumerate(device_input)]
    check(o.publish(ow, computed), 'kernel completion')
    d.release()
    ot, = o.start_readers(['D2H'])
    rw = r.acquire('q')
    check(r.publish(rw, o.finish(ot)), 'D2H completion')
    o.release()
    cpu, archive = r.start_readers(['CPU', 'archive'])
    response = tuple(r.finish(cpu))
    check(rejected(r.release), 'archive must protect result range')
    r.finish(archive)
    r.release()
    check(all(buf.owner is None for buf in (h, d, o, r)), 'path drained')
    return response


def lifecycle_checks():
    inputs = [(), (0,), (-7, 0, 4, 19), tuple(range(17))]
    for raw in inputs:
        expected = tuple(3 * (x + 1) + i for i, x in enumerate(raw))
        check(pipeline_payload(raw) == expected, ('independent payload reference', raw))
    print('observed payload_reference_cases:', len(inputs))
    buf = ready_buffer([4, 5], 'A')
    copy, archive = buf.start_readers(['copy', 'archive'])
    buf.finish(copy)
    buf.finish(copy)
    check(buf.readers == {archive}, 'duplicate cannot acknowledge archive')
    check(rejected(buf.release), 'second reader still protects the range')
    buf.finish(archive)
    buf.release()
    old_gen = buf.gen
    writer = buf.acquire('B')
    buf.publish(writer, [8, 9])
    live, = buf.start_readers(['copy'])
    buf.finish(copy)
    check(buf.gen == old_gen + 1 and buf.readers == {live},
          'old completion cannot mutate the new generation')
    buf.cancel()
    check(rejected(buf.release), 'cancellation cannot remove an active reader')
    check(buf.finish(live) == (8, 9), 'canceled physical access still completes')
    buf.release()
    partial = Buffer()
    pw = partial.acquire('partial')
    check(not partial.publish(pw, [1], full=False), 'partial range not published')
    check(rejected(lambda: partial.start_readers(['kernel'])),
          'incomplete object cannot be consumed')
    partial.release()
    unknown = Buffer()
    uw = unknown.acquire('unknown')
    check(not unknown.publish(uw, [3], ended=False), 'unknown write not ended')
    unknown.cancel()
    check(rejected(unknown.release), 'unknown write remains protected after cancel')
    check(unknown.writer == uw, 'writer responsibility retained')
    unknown.publish(uw, [3], ended=True)
    check(not unknown.ready, 'canceled destination cannot be republished')
    unknown.release()
    for order in (('copy', 'archive'), ('archive', 'copy')):
        group = ready_buffer([6])
        tickets = dict(zip(('copy', 'archive'), group.start_readers(['copy', 'archive'])))
        group.finish(tickets[order[0]])
        check(rejected(group.release), 'either first reader leaves another responsibility')
        group.finish(tickets[order[1]])
        group.release()
    print('observed lifecycle: READY hold, two completion orders, duplicate, stale generation, cancellation, partial and unknown completion protected')
    faulty_source = ready_buffer([1, 2])
    dma, = faulty_source.start_readers(['DMA'])
    independent_expected = (1, 2)
    faulty_source.data = (9, 9)
    negative('overwrite source after API return',
             lambda: check(faulty_source.finish(dma) == independent_expected,
                           'DMA observed the next payload'))
    faulty_ledger = ready_buffer([1])
    first, second = faulty_ledger.start_readers(['copy', 'archive'])
    def faulty_ack(_ticket):
        if faulty_ledger.readers:
            faulty_ledger.readers.remove(min(faulty_ledger.readers))
    faulty_ack(first)
    faulty_ack(first)
    negative('duplicate notification consumes another reader',
             lambda: check(second in faulty_ledger.readers,
                           'archive was never acknowledged but lost protection'))


class Pipeline:
    """Event-driven four-pool model with one abstract capacity domain.

    Isolation entries are (request, uid, reason), independent of pending.
    ended and acceptance_confirmed are supplied abstract proof inputs.
    IDENTITY_REVIEW holds the operation endpoints until separately resolved;
    this model tests its retention, not real identity-repair procedures.
    Capacity restoration is an explicit external input, not a real concurrent
    subscription/epoch handshake or retry timer.
    """
    def __init__(self, inputs, capacity=1, outcomes=None, policy='retry', bug=''):
        self.raw = [tuple(x) for x in inputs]
        self.live = [True] * len(inputs)
        self.terminal = [None] * len(inputs)
        self.phase = ['WAIT_INPUT'] * len(inputs)
        self.isolated = set()
        self.isolation_refs = {}
        self.borrowed = set()
        self.next = 0
        self.serial = 0
        self.pending = {}
        self.acceptance_wait = {}
        self.resolution_done = {}
        self.queue = []
        self.blocked = set()
        self.outcomes = {k: list(v) for k, v in (outcomes or {}).items()}
        self.policy = policy
        self.bug = bug
        self.results = {}
        self.events = []
        self.slots = []
        for pool in range(4):
            for slot in range(capacity):
                self.slots.append(dict(pool=pool, slot=slot, gen=0, owner=None,
                                       hold=False, writer=None, readers=set(),
                                       ready=False, data=None))

    def emit(self, *event):
        self.events.append(event)

    def ref(self, slot):
        return (slot['pool'], slot['slot'], slot['gen'], slot['owner'])

    def quarantine(self, request, uid, reason):
        op = self.pending[uid]
        check(op['q'] == request, 'isolation request does not match uid')
        key = (request, uid, reason)
        if key not in self.isolated:
            self.isolated.add(key)
            self.isolation_refs[key] = op['refs']
            self.emit('isolate', request, uid, reason, op['refs'])

    def clear_reason(self, request, uid, reason, proof):
        allowed = {END: ('all_accesses_ended', 'never_accepted'),
                   ACCEPTANCE: ('whole_operation_accepted', 'never_accepted')}
        check(proof in allowed.get(reason, ()), 'proof cannot clear this reason')
        key = (request, uid, reason)
        if key in self.isolated:
            self.isolated.remove(key)
            del self.isolation_refs[key]
            self.emit('reason_clear', request, uid, reason, proof)

    def eligible(self, request):
        return (self.live[request] and self.terminal[request] is None
                and not any(q == request for q, _, _ in self.isolated))

    def range_isolated(self, ref):
        # End/acceptance reasons retain actual access responsibilities.
        # An unresolved identity review additionally retains endpoint leases.
        return any(key[2] == IDENTITY and ref in refs
                   for key, refs in self.isolation_refs.items())

    def free_slot(self, pool):
        return next((s for s in self.slots
                     if s['pool'] == pool and s['owner'] is None), None)

    def acquire(self, pool, request):
        check(self.eligible(request) or self.bug == 'admit_dead',
              'acquisition without execution eligibility')
        slot = self.free_slot(pool)
        check(slot is not None, 'pool exhausted')
        slot.update(gen=slot['gen'] + 1, owner=request, hold=True,
                    writer=None, readers=set(), ready=False, data=None)
        self.emit('reserve', self.ref(slot))
        return slot

    def release(self, slot):
        if (slot is not None and slot['owner'] is not None
                and not slot['hold'] and slot['writer'] is None
                and not slot['readers'] and not self.range_isolated(self.ref(slot))):
            self.emit('free', self.ref(slot))
            slot.update(owner=None, ready=False, data=None)

    def terminate(self, request, status):
        if self.terminal[request] is None:
            self.terminal[request] = status
            self.emit('terminal', request, status)
        self.live[request] = False
        self.blocked = {key for key in self.blocked if key[0] != request}
        for slot in self.slots:
            if slot['owner'] == request:
                slot['hold'] = False
                slot['ready'] = False
                self.release(slot)

    def end_borrow(self, uid):
        if uid in self.borrowed:
            self.borrowed.remove(uid)
            self.emit('unborrow', uid)

    def rollback(self, uid):
        if uid not in self.pending:
            return
        op = self.pending[uid]
        check(op['outcome'] == 'NO', 'rollback requires non-acceptance proof')
        if self.bug != 'leave_pending':
            del self.pending[uid]
        self.emit('settle', uid, 'not_accepted')
        self.end_borrow(uid)
        source, dest = op['source'], op['dest']
        if source is not None:
            source['readers'].discard(uid)
            source['hold'] = self.eligible(op['q'])
            source['ready'] = source['hold']
            self.release(source)
        if dest is not None:
            if dest['writer'] == uid:
                dest['writer'] = None
            dest['hold'] = False
            dest['ready'] = False
            self.release(dest)

    def start(self, request, stage, source, dest):
        if not self.eligible(request) and self.bug != 'admit_dead':
            if dest is not None:
                dest['hold'] = False
                self.release(dest)
            return 'INELIGIBLE', None
        if source is not None:
            check(source['ready'] and source['hold']
                  and source['writer'] is None, 'source not READY')
        self.serial += 1
        uid = self.serial
        options = self.outcomes.get((request, stage), [])
        outcome = options.pop(0) if options else 'YES'
        check(outcome in ('YES', 'NO', 'UNKNOWN'), 'invalid submission result')
        refs = tuple(self.ref(s) for s in (source, dest) if s is not None)
        self.pending[uid] = dict(q=request, stage=stage, source=source,
                                 dest=dest, refs=refs, outcome=outcome)
        self.emit('intent', uid, request, stage, refs)
        if stage == 0:
            self.borrowed.add(uid)
            self.emit('borrow', uid, request)
        if source is not None:
            source['readers'].add(uid)
            source['hold'] = False
            source['ready'] = False
        if dest is not None:
            dest['writer'] = uid
        self.emit('submission', uid, outcome)
        if outcome == 'NO':
            self.rollback(uid)
            if self.policy == 'fail':
                self.terminate(request, 'FAILED')
            else:
                self.blocked.add((request, stage))
                if stage == 0:
                    self.phase[request] = 'WAIT_CAPACITY'
        elif outcome == 'UNKNOWN':
            self.quarantine(request, uid, ACCEPTANCE)
        else:
            self.queue.append(uid)
        return outcome, uid

    def capacity_restored(self):
        self.blocked.clear()
        self.emit('capacity')
        self.advance()

    def resolve_acceptance(self, uid, accepted, refs=None):
        check(type(accepted) is bool, 'classification proof must be complete')
        if uid in self.resolution_done:
            self.emit('resolution_ignored', uid)
            return False
        retained = self.acceptance_wait.get(uid)
        op = self.pending.get(uid)
        record = retained if retained is not None else op
        if record is None:
            self.emit('resolution_ignored', uid)
            return False
        request = record['q']
        proof_refs = record['refs'] if refs is None else refs
        if proof_refs != record['refs'] or (request, uid, ACCEPTANCE) not in self.isolated:
            self.emit('resolution_ignored', uid)
            return False
        # The refs are immutable original identities, never current mutable slots.
        self.emit('acceptance_proof', uid, accepted, proof_refs)
        proof = 'whole_operation_accepted' if accepted else 'never_accepted'
        self.clear_reason(request, uid, ACCEPTANCE, proof)
        if retained is not None:
            del self.acceptance_wait[uid]
            self.resolution_done[uid] = (accepted, request, record['stage'], record['refs'])
            self.emit('acceptance_closed', uid, accepted)
            # Physical settlement already happened. Do not touch old slot objects,
            # roll back a second time, recreate output, or revive terminal requests.
        elif accepted:
            op['outcome'] = 'YES'
            self.emit('submission', uid, 'YES')
            if uid not in self.queue:
                self.queue.append(uid)
        else:
            self.clear_reason(request, uid, END, 'never_accepted')
            op['outcome'] = 'NO'
            self.emit('submission', uid, 'NO')
            self.rollback(uid)
            self.terminate(request, 'FAILED')
            self.resolution_done[uid] = (False, request, op['stage'], op['refs'])
        self.capacity_restored()
        return True

    def resolve_unaccepted(self, uid, refs=None):
        return self.resolve_acceptance(uid, False, refs)

    def resolve_accepted(self, uid, refs=None):
        return self.resolve_acceptance(uid, True, refs)

    def advance(self):
        # One owner runs to quiescence. NO establishes a wait, so it cannot spin.
        while True:
            changed = False
            for stage in range(4, 0, -1):
                for source in self.slots:
                    if source['pool'] != stage - 1 or not source['ready']:
                        continue
                    request = source['owner']
                    if not self.eligible(request) or (request, stage) in self.blocked:
                        continue
                    if stage < 4 and self.free_slot(stage) is None:
                        continue
                    dest = self.acquire(stage, request) if stage < 4 else None
                    self.start(request, stage, source, dest)
                    changed = True
            while self.next < len(self.raw):
                request = self.next
                if self.terminal[request] is not None and self.bug != 'admit_dead':
                    self.emit('skip', request)
                    self.next += 1
                    changed = True
                    continue
                if ((not self.eligible(request) and self.bug != 'admit_dead')
                        or (request, 0) in self.blocked or self.free_slot(0) is None):
                    break
                dest = self.acquire(0, request)
                outcome, _ = self.start(request, 0, None, dest)
                changed = True
                if outcome in ('YES', 'UNKNOWN'):
                    self.phase[request] = 'ACTIVE'
                    self.next += 1
                elif self.terminal[request] is not None:
                    self.next += 1
                elif self.bug == 'lose_cursor':
                    self.next += 1
                    break
                else:
                    break
            if not changed:
                break

    def complete(self, uid, success=True, full=True, ended=True,
                 acceptance_confirmed=True):
        if uid not in self.pending:
            if uid in self.acceptance_wait and acceptance_confirmed:
                self.resolve_accepted(uid)
            else:
                self.emit('ignored', uid)
            return
        op = self.pending[uid]
        check(op['outcome'] != 'NO', 'unaccepted operation cannot complete')
        source, dest = op['source'], op['dest']
        request, stage = op['q'], op['stage']
        if tuple(self.ref(s) for s in (source, dest) if s is not None) != op['refs']:
            self.quarantine(request, uid, IDENTITY)
            return
        self.emit('evidence', uid, ended, acceptance_confirmed)
        if not ended:
            self.quarantine(request, uid, END)
            return
        # The sole new recovery mutant omits this cause only for ACCEPTED.
        # It changes state updates, rather than raising a synthetic failure.
        if not (self.bug == 'omit_accepted_end_clear' and op['outcome'] == 'YES'):
            self.clear_reason(request, uid, END, 'all_accesses_ended')
        if op['outcome'] == 'UNKNOWN' and acceptance_confirmed:
            self.clear_reason(request, uid, ACCEPTANCE, 'whole_operation_accepted')
        values = None
        if success and full:
            if stage == 0:
                values = tuple(x + 1 for x in self.raw[request])
            elif stage == 2:
                values = tuple(3 * x + i for i, x in enumerate(source['data']))
            else:
                values = tuple(source['data'])
        if (request, uid, ACCEPTANCE) in self.isolated:
            snapshot = dict(q=request, stage=stage, refs=op['refs'], ended=True)
            self.emit('acceptance_wait', uid, request, stage, op['refs'])
            if self.bug != 'drop_post_end_record':
                self.acceptance_wait[uid] = snapshot
        del self.pending[uid]
        if uid in self.queue:
            self.queue.remove(uid)
        self.emit('settle', uid, 'ended')
        self.end_borrow(uid)
        if source is not None:
            source['readers'].remove(uid)
        if dest is not None:
            check(dest['writer'] == uid, 'wrong destination writer')
            dest['writer'] = None
        if (not success or not full
                or any(q == request for q, _, _ in self.isolated)):
            # This reference discards an unpublishable completed result and fails
            # conservatively, while leaving every unresolved evidence cause intact.
            self.terminate(request, 'FAILED')
        if dest is not None:
            dest['hold'] = self.eligible(request) and success and full
            dest['ready'] = dest['hold']
            dest['data'] = values if dest['hold'] else None
            if dest['ready']:
                self.emit('publish', uid, self.ref(dest))
        if stage == 4 and success and full and self.eligible(request):
            check(request not in self.results, 'duplicate response')
            self.results[request] = tuple(values)
            self.emit('commit', request, tuple(values))
            self.terminate(request, 'SUCCESS')
        self.release(source)
        self.release(dest)
        # This model uses one abstract capacity domain for executor and pools.
        self.capacity_restored()

    def drive(self, reverse=False):
        self.advance()
        steps = 0
        while self.queue:
            self.complete(max(self.queue) if reverse else min(self.queue))
            steps += 1
            check(steps < 1000, 'event model failed to quiesce')

    def drained(self):
        return (all(s is not None for s in self.terminal)
                and not self.pending and not self.acceptance_wait
                and not self.blocked and not self.isolated
                and not self.borrowed and not self.isolation_refs
                and all(s['owner'] is None for s in self.slots))


def observe(model, expected):
    # Independent replay uses public event order, not hold/writer/free flags.
    owned, active, submissions, terminal, committed = {}, {}, {}, {}, {}
    requests, stages, evidence, reasons, borrows = {}, {}, {}, {}, set()
    original_refs, classifications, awaiting = {}, {}, {}
    for event in model.events:
        kind = event[0]
        if kind == 'reserve':
            ref = event[1]
            check(ref[3] not in terminal, 'observer: terminal request acquired a slot')
            check(not any(key[0] == ref[3] for key in reasons),
                  'observer: isolated request acquired a slot')
            check(ref[:2] not in owned, 'observer: overlapping physical leases')
            owned[ref[:2]] = ref
        elif kind == 'intent':
            _, uid, request, stage, refs = event
            check(request not in terminal, 'observer: new work after terminal')
            check(not any(key[0] == request for key in reasons),
                  'observer: new work during isolation')
            check(uid not in active and uid not in requests, 'observer: repeated uid')
            check(all(owned.get(ref[:2]) == ref for ref in refs),
                  'observer: intent references an unowned lease')
            active[uid] = refs
            requests[uid] = request
            stages[uid] = stage
            original_refs[uid] = refs
        elif kind == 'borrow':
            _, uid, request = event
            check(uid in active and requests[uid] == request and stages[uid] == 0,
                  'observer: external borrow lacks prepare intent')
            check(uid not in borrows, 'observer: repeated external borrow')
            borrows.add(uid)
        elif kind == 'unborrow':
            uid = event[1]
            check(uid in borrows and uid not in active,
                  'observer: borrow released without access settlement')
            borrows.remove(uid)
        elif kind == 'submission':
            submissions[event[1]] = event[2]
        elif kind == 'evidence':
            _, uid, ended, accepted = event
            check(uid in active, 'observer: proof for non-pending operation')
            evidence[uid] = (ended, accepted)
        elif kind == 'acceptance_wait':
            _, uid, request, stage, refs = event
            check(uid in active and evidence.get(uid) == (True, False),
                  'observer: deferred classification lacks ended operation')
            check(requests[uid] == request and stages[uid] == stage
                  and original_refs[uid] == refs, 'observer: deferred identity mismatch')
            check((request, uid, ACCEPTANCE) in reasons and uid not in awaiting,
                  'observer: unexpected deferred classification')
            awaiting[uid] = dict(q=request, stage=stage, refs=refs, ended=True)
        elif kind == 'acceptance_proof':
            _, uid, accepted, refs = event
            check(uid in requests and original_refs[uid] == refs,
                  'observer: classification proof identity mismatch')
            check((requests[uid], uid, ACCEPTANCE) in reasons,
                  'observer: classification has no unresolved reason')
            classifications[uid] = accepted
        elif kind == 'acceptance_closed':
            _, uid, accepted = event
            check(uid in awaiting and uid not in active
                  and classifications.get(uid) is accepted,
                  'observer: closing unknown or active classification record')
            check((requests[uid], uid, ACCEPTANCE) not in reasons,
                  'observer: record removed before classification reason')
            del awaiting[uid]
        elif kind == 'isolate':
            _, request, uid, reason, refs = event
            check(uid in active and requests[uid] == request,
                  'observer: isolation identity mismatch')
            check(refs == active[uid], 'observer: isolation range mismatch')
            key = (request, uid, reason)
            check(key not in reasons, 'observer: duplicate isolation registration')
            reasons[key] = refs
        elif kind == 'reason_clear':
            _, request, uid, reason, proof = event
            key = (request, uid, reason)
            check(key in reasons and requests[uid] == request,
                  'observer: clearing an unrelated isolation')
            if proof == 'all_accesses_ended':
                check(reason == END and evidence.get(uid, (False, False))[0],
                      'observer: end cause cleared without end proof')
            elif proof == 'whole_operation_accepted':
                check(reason == ACCEPTANCE and (evidence.get(uid) == (True, True)
                      or classifications.get(uid) is True),
                      'observer: acceptance cause cleared without matching proof')
            elif proof == 'never_accepted':
                # This explicit resolver input is separate from ordinary completion.
                check(reason in (END, ACCEPTANCE) and submissions[uid] == 'UNKNOWN'
                      and classifications.get(uid) is False,
                      'observer: invalid non-acceptance resolution')
            else:
                check(False, 'observer: unknown reason-clear proof')
            del reasons[key]
        elif kind == 'settle':
            _, uid, reason = event
            check(uid in active, 'observer: duplicate responsibility settlement')
            if reason == 'ended':
                check(submissions[uid] != 'NO', 'observer: invented completion')
                check(evidence.get(uid, (False, False))[0],
                      'observer: settlement without end proof')
            else:
                check(submissions[uid] == 'NO', 'observer: rollback of accepted work')
            del active[uid]
        elif kind == 'publish':
            _, uid, ref = event
            check(uid not in active and evidence.get(uid, (False, False))[0],
                  'observer: publication before writer ended')
            check(ref[3] not in terminal
                  and not any(key[0] == ref[3] for key in reasons),
                  'observer: publication without execution eligibility')
            check(owned.get(ref[:2]) == ref, 'observer: publication to wrong lease')
        elif kind == 'free':
            ref = event[1]
            check(owned.get(ref[:2]) == ref, 'observer: wrong free identity')
            check(not any(ref in refs for refs in active.values()),
                  'observer: free with outstanding physical responsibility')
            check(not any(key[2] == IDENTITY and ref in refs
                          for key, refs in reasons.items()),
                  'observer: free during unresolved identity review')
            del owned[ref[:2]]
        elif kind == 'terminal':
            check(event[1] not in terminal, 'observer: changed terminal status')
            terminal[event[1]] = event[2]
        elif kind == 'commit':
            check(event[1] not in terminal and event[1] not in committed,
                  'observer: commit after terminal or duplicate')
            check(not any(key[0] == event[1] for key in reasons),
                  'observer: commit during isolation')
            committed[event[1]] = event[2]
    check(set(active) == set(model.pending), 'observer: unsettled pending mismatch')
    check(borrows == model.borrowed, 'observer: external borrow mismatch')
    check(awaiting == model.acceptance_wait, 'observer: post-end classification record lost')
    check(set(reasons) == model.isolated, 'observer: isolation responsibility lost')
    check(reasons == model.isolation_refs, 'observer: isolated ranges changed')
    check(committed == expected, ('independent direct output reference', committed, expected))
    if model.drained():
        check(not owned and not active and not reasons and not borrows and not awaiting,
              'observer: drain omitted a lease, operation, borrow or isolation')
    return owned, active


def direct(raw):
    return tuple(3 * x + 3 + i for i, x in enumerate(raw))


def scheduler_checks():
    for raw in ((), (0,), (-7, 0, 4), tuple(range(17))):
        m = Pipeline([raw], outcomes={(0, 0): ['NO']})
        m.advance()
        check(m.next == 0 and not m.pending and not m.queue,
              'prepare rejection must retain cursor and settle all intent')
        check(m.free_slot(0) is not None and m.blocked == {(0, 0)},
              'rejection returns destination and retains a wakeable retry')
        before = (m.next, len(m.events), m.serial)
        m.advance()
        check(before == (m.next, len(m.events), m.serial), 'no busy retry')
        m.rollback(1)
        m.complete(1)
        m.capacity_restored()
        m.drive()
        check(m.drained() and m.terminal == ['SUCCESS'], 'retry did not drain')
        observe(m, {0: direct(raw)})
    print('observed admission: N=1 first prepare rejected; cursor retained; idempotent rollback; capacity wake restores complete output for 4 payloads')

    failed = Pipeline([(2,)], outcomes={(0, 0): ['NO']}, policy='fail')
    failed.drive()
    check(failed.drained() and failed.terminal == ['FAILED'] and not failed.queue,
          'synchronous terminal failure waited for nonexistent completion')
    observe(failed, {})

    for occupied in (False, True):
        m = Pipeline([(1,), (2,), (3,)])
        if occupied:
            m.advance()
            canceled = 1
        else:
            canceled = 0
        m.terminate(canceled, 'CANCELED')
        m.drive(reverse=True)
        m.advance()
        m.capacity_restored()
        check(m.drained(), 'canceled admission prevented neighbor progress')
        check(not any(e[0] == 'reserve' and e[1][3] == canceled for e in m.events),
              'unadmitted canceled request acquired H')
        check(not any(e[0] == 'intent' and e[2] == canceled for e in m.events),
              'unadmitted canceled request created pending')
        observe(m, {q: direct(raw) for q, raw in enumerate(m.raw) if q != canceled})

    m = Pipeline([(1,)])
    dest = m.acquire(0, 0)
    m.terminate(0, 'CANCELED')
    outcome, uid = m.start(0, 0, None, dest)
    check((outcome, uid) == ('INELIGIBLE', None) and not m.pending,
          'start must not register canceled work')
    m.advance()
    check(m.drained(), 'pre-start cancellation did not drain')
    observe(m, {})
    print('observed cancellation: before first H, behind occupied H, repeated wakes, start recheck; neighbors complete')

    for stage in range(1, 5):
        m = Pipeline([(2, 4)], outcomes={(0, stage): ['NO']})
        m.drive()
        check(m.blocked == {(0, stage)}, 'downstream retry needs a capacity event')
        m.capacity_restored()
        m.drive()
        check(m.drained(), ('downstream rejection failed to recover', stage))
        observe(m, {0: direct(m.raw[0])})

    m = Pipeline([(5,)], outcomes={(0, 0): ['UNKNOWN']})
    m.advance()
    check(m.next == 1 and len(m.pending) == 1 and not m.queue,
          'unknown acceptance must retain ownership without retry')
    m.terminate(0, 'CANCELED')
    check(not m.drained() and m.free_slot(0) is None, 'unknown access freed on cancel')
    m.complete(1, ended=False)
    check(not m.drained(), 'unknown ended certificate discarded')
    m.complete(1, ended=True)
    check(m.drained(), 'confirmed unknown access did not drain')
    observe(m, {})

    resumed = Pipeline([(5,)], outcomes={(0, 0): ['UNKNOWN']})
    resumed.advance()
    resumed.complete(1, ended=True)
    resumed.drive()
    check(resumed.drained(), 'resolved acceptance did not resume')
    observe(resumed, {0: direct((5,))})

    unaccepted = Pipeline([(5,)], outcomes={(0, 0): ['UNKNOWN']})
    unaccepted.advance()
    unaccepted.resolve_unaccepted(1)
    check(unaccepted.drained() and unaccepted.terminal == ['FAILED'],
          'resolved non-acceptance waited for a nonexistent completion')
    observe(unaccepted, {})

    for stage in range(5):
        for partial in (False, True):
            m = Pipeline([(1, 3)])
            m.advance()
            while m.pending and not any(op['stage'] == stage for op in m.pending.values()):
                m.complete(min(m.queue))
            uid = next(u for u, op in m.pending.items() if op['stage'] == stage)
            m.complete(uid, success=partial, full=not partial)
            m.drive()
            check(m.drained() and m.terminal == ['FAILED'], 'failure cleanup')
            observe(m, {})
    empty = Pipeline([])
    empty.drive()
    check(empty.drained(), 'zero request drain')
    observe(empty, {})

    for count in range(1, 5):
        for capacity in (1, 2):
            for reverse in (False, True):
                raw = [tuple(range(q)) for q in range(count)]
                m = Pipeline(raw, capacity=capacity)
                m.drive(reverse)
                check(m.drained(), 'finite four-pool path failed to drain')
                observe(m, {q: direct(x) for q, x in enumerate(raw)})
    print('observed scheduler: downstream rollback, unknown quarantine/resolution, all-stage errors/partial ranges, zero requests, finite pool reuse and completion order')

    def faulty_retry(bug):
        m = Pipeline([(7,)], outcomes={(0, 0): ['NO']}, bug=bug)
        m.advance()
        m.capacity_restored()
        m.drive()
        observe(m, {0: direct((7,))})
        check(m.drained(), 'lost cursor or residual pending prevents draining')

    negative('unconditionally advance cursor after prepare rejection',
             lambda: faulty_retry('lose_cursor'))
    negative('rollback leaves unaccepted pending entry',
             lambda: faulty_retry('leave_pending'))

    def faulty_admission():
        m = Pipeline([(1,), (2,)], bug='admit_dead')
        m.terminate(0, 'CANCELED')
        m.drive()
        observe(m, {1: direct((2,))})
    negative('canceled input admitted after wake', faulty_admission)


def accepted_end_recovery(bug='', raw=(2, -1, 5)):
    """Same actual trajectory and observer for the correct model and mutant."""
    m = Pipeline([raw], bug=bug)
    m.advance()
    check(m.next == 1 and len(m.pending) == 1, 'accepted prepare admission')
    uid = next(iter(m.pending))
    op = m.pending[uid]
    check(op['stage'] == 0 and op['outcome'] == 'YES',
          'regression must start with unequivocal ACCEPTED')
    h = op['dest']
    lease = m.ref(h)
    m.complete(uid, ended=False)
    check(m.pending.get(uid) is op and h['writer'] == uid,
          'insufficient end proof lost pending or writer')
    check(uid in m.borrowed and m.ref(h) == lease and h['hold'],
          'insufficient end proof lost external borrow or destination lease')
    check(m.isolated == {(0, uid, END)} and not m.eligible(0),
          'ACCEPTED operation must be isolated specifically for its missing end')
    check(not h['ready'] and not m.results
          and not any(e[0] == 'publish' for e in m.events),
          'insufficient end proof published a destination')
    observe(m, {})
    # Repeating the insufficient notification remains idempotent and protective.
    m.complete(uid, ended=False)
    check(h['writer'] == uid and uid in m.borrowed and uid in m.pending,
          'repeated insufficient notification removed protection')
    m.complete(uid, success=True, full=True, ended=True)
    # No manual READY, next, isolation or responsibility repair is performed.
    m.drive()
    m.complete(uid)
    observe(m, {0: direct(raw)})
    check(m.terminal == ['SUCCESS'] and m.results == {0: direct(raw)},
          'accepted end recovery has no successful terminal/output')
    check([e[3] for e in m.events if e[0] == 'intent'] == list(range(5)),
          'recovery did not automatically execute prepare/H2D/kernel/D2H/CPU')
    check(sum(e[0] == 'commit' for e in m.events) == 1,
          'duplicate completion committed twice')
    check(m.drained() and not m.queue and not m.pending and not m.borrowed,
          'accepted end recovery did not drain every responsibility')
    check(all(s['owner'] is None and s['writer'] is None
              and not s['readers'] and not s['hold'] for s in m.slots),
          'accepted end recovery left a pool lease or access')
    return m


def quarantine_recovery_checks():
    accepted_end_recovery()
    # Retain the exact N=1 nonempty regression, then cover empty payload as an edge.
    accepted_end_recovery(raw=())
    print('observed accepted_end_recovery: N=1 ACCEPTED prepare; ended=false retains pending/writer/borrow/lease; sufficient end resumes full path; direct output, one commit and complete drain checked')

    # Same uid, different cause: end proof does not certify identity.
    other = Pipeline([(2, -1, 5)])
    other.advance()
    uid = next(iter(other.pending))
    op = other.pending[uid]
    h = op['dest']
    lease = other.ref(h)
    other.complete(uid, ended=False)
    other.quarantine(0, uid, IDENTITY)
    other.complete(uid, ended=True)
    other.drive()
    other.complete(uid)
    check(other.isolated == {(0, uid, IDENTITY)},
          'end proof cleared another isolation cause')
    check(not other.eligible(0) and other.terminal == ['FAILED'] and not other.results,
          'remaining identity cause incorrectly restored the request')
    check(not other.pending and not other.borrowed and h['writer'] is None,
          'proved-ended access was not individually settled')
    check(other.ref(h) == lease and not h['ready'] and other.free_slot(0) is None,
          'identity-isolated endpoint was published or reused')
    check(not other.drained(), 'unresolved cause was hidden by pending removal')
    observe(other, {})

    # UNKNOWN may have two causes; ended alone must not certify whole acceptance.
    unknown = Pipeline([(4,)], outcomes={(0, 0): ['UNKNOWN']})
    unknown.advance()
    uid = next(iter(unknown.pending))
    unknown.complete(uid, ended=False, acceptance_confirmed=False)
    check(unknown.isolated == {(0, uid, ACCEPTANCE), (0, uid, END)},
          'unknown acceptance and end must be separate causes')
    unknown.complete(uid, ended=True, acceptance_confirmed=False)
    check(unknown.isolated == {(0, uid, ACCEPTANCE)} and not unknown.eligible(0)
          and not unknown.results and not unknown.pending,
          'end proof implicitly cleared unknown acceptance')
    observe(unknown, {})

    # Different uid/request: completing one must preserve the other's cause/access.
    pair = Pipeline([(1,), (7,)], capacity=2)
    pair.advance()
    by_request = {op['q']: u for u, op in pair.pending.items()}
    first, second = by_request[0], by_request[1]
    pair.complete(first, ended=False)
    pair.complete(second, ended=False)
    second_op = pair.pending[second]
    pair.complete(first, ended=True)
    check(pair.isolated == {(1, second, END)} and pair.pending[second] is second_op,
          'settling one uid cleared a different uid')
    check(second_op['dest']['writer'] == second and second in pair.borrowed,
          'other uid lost its physical protection')
    pair.drive(reverse=True)
    check(pair.drained(), 'independent requests did not resume after own evidence')
    observe(pair, {q: direct(raw) for q, raw in enumerate(pair.raw)})

    # Clearing an isolation cause cannot revoke a prior terminal decision.
    for status in ('CANCELED', 'FAILED'):
        m = Pipeline([(3, 6)])
        m.advance()
        uid = next(iter(m.pending))
        m.complete(uid, ended=False)
        m.terminate(0, status)
        m.complete(uid, ended=True)
        m.drive()
        m.complete(uid)
        check(m.terminal == [status] and not m.results and m.drained(),
              'late sufficient end resurrected a canceled/failed request')
        check(not any(e[0] in ('publish', 'commit') for e in m.events),
              'terminal request published after cause settlement')
        observe(m, {})
    print('observed isolation controls: identity-isolated completed request fails with endpoint lease retained; unknown acceptance retained without acceptance proof; other uid untouched; cancellation/failure veto publication')

    negative('omit only ACCEPTED END_UNCONFIRMED settlement; same recovery path loses required success',
             lambda: accepted_end_recovery(bug='omit_accepted_end_clear'))


def late_acceptance_case(accepted, bug=''):
    m = Pipeline([(4,)], outcomes={(0, 0): ['UNKNOWN']}, bug=bug)
    m.advance()
    uid = next(iter(m.pending))
    refs = m.pending[uid]['refs']
    m.complete(uid, ended=False, acceptance_confirmed=False)
    m.complete(uid, ended=True, acceptance_confirmed=False)
    check(m.terminal == ['FAILED'] and not m.results and not m.pending
          and not m.borrowed and all(s['owner'] is None for s in m.slots),
          'post-end unresolved request must fail and settle physical resources')
    check(m.isolated == {(0, uid, ACCEPTANCE)} and not m.drained(),
          'post-end unresolved classification was hidden')
    before = (len(m.events), set(m.isolated), dict(m.acceptance_wait))
    wrong = tuple((p, s, g + 1, q) for p, s, g, q in refs)
    m.resolve_acceptance(uid, accepted, wrong)
    check(set(m.isolated) == before[1] and m.acceptance_wait == before[2],
          'wrong-generation classification changed original evidence')
    # Both the correct model and loss mutant receive the actual late proof.
    if accepted:
        m.complete(uid, ended=True, acceptance_confirmed=True)
    else:
        m.resolve_unaccepted(uid)
    m.drive()
    check(m.drained() and m.terminal == ['FAILED'] and not m.results,
          'late classification failed to settle without resurrecting request')
    observe(m, {})
    check(sum(e[0] == 'settle' and e[1] == uid for e in m.events) == 1,
          'late classification settled physical access twice')
    check(not any(e[0] in ('publish', 'commit') for e in m.events),
          'late classification republished failed output')
    snapshot = (tuple(m.terminal), dict(m.results), dict(m.acceptance_wait),
                set(m.isolated), tuple(m.ref(s) for s in m.slots))
    m.resolve_acceptance(uid, accepted)
    m.resolve_acceptance(uid, not accepted)
    m.resolve_acceptance(uid + 1000, accepted)
    m.complete(uid)
    check(snapshot == (tuple(m.terminal), dict(m.results), dict(m.acceptance_wait),
                       set(m.isolated), tuple(m.ref(s) for s in m.slots)),
          'duplicate/unknown late classification changed settled state')
    observe(m, {})
    return m


def post_end_acceptance_checks():
    for accepted in (True, False):
        late_acceptance_case(accepted)
        negative('lost post-end record with late ' + ('accepted' if accepted else 'never-accepted'),
                 lambda accepted=accepted: late_acceptance_case(accepted, 'drop_post_end_record'))
    for status in ('CANCELED', 'FAILED'):
        for accepted in (True, False):
            m = Pipeline([(2,)], outcomes={(0, 0): ['UNKNOWN']})
            m.advance()
            m.terminate(0, status)
            m.complete(1, ended=True, acceptance_confirmed=False)
            m.resolve_acceptance(1, accepted)
            m.drive()
            check(m.terminal == [status] and not m.results and m.drained(),
                  'late classification changed an earlier terminal decision')
            observe(m, {})
    # Old immutable refs must remain usable after the physical slot changes owner.
    for accepted in (True, False):
        m = Pipeline([(4,), (7,)], outcomes={(0, 0): ['UNKNOWN']})
        m.advance()
        old_uid = next(iter(m.pending))
        old_refs = m.pending[old_uid]['refs']
        m.complete(old_uid, ended=True, acceptance_confirmed=False)
        fresh_uid = next(iter(m.pending))
        fresh = m.pending[fresh_uid]
        check(fresh['q'] == 1 and fresh['refs'] != old_refs,
              'reuse control did not obtain a new physical identity')
        fresh_ref, fresh_writer = m.ref(fresh['dest']), fresh['dest']['writer']
        m.resolve_acceptance(old_uid, accepted, old_refs)
        check(m.ref(fresh['dest']) == fresh_ref and fresh['dest']['writer'] == fresh_writer,
              'old classification mutated a new allocation')
        m.drive()
        check(m.terminal == ['FAILED', 'SUCCESS'] and m.drained(),
              'old evidence blocked or resurrected the wrong request')
        observe(m, {1: direct((7,))})
    # Classification can also precede end; it must not publish before end.
    m = Pipeline([(3,)], outcomes={(0, 0): ['UNKNOWN']})
    m.advance()
    m.resolve_accepted(1)
    check(1 in m.pending and not m.results and not any(e[0] == 'publish' for e in m.events),
          'acceptance classification was mistaken for physical completion')
    m.drive()
    check(m.terminal == ['SUCCESS'] and m.drained(), 'pre-end acceptance did not resume')
    observe(m, {0: direct((3,))})
    print('observed post_end_acceptance: late accepted and never-accepted both settle immutable records, preserve FAILED and fully drain; loss mutants detected; reused slots and duplicate/wrong/unknown proofs unchanged')


def main():
    arithmetic_checks()
    lifecycle_checks()
    scheduler_checks()
    quarantine_recovery_checks()
    post_end_acceptance_checks()
    print('scope: deterministic CPU resource calendars, interval occupancy, local buffers and finite four-pool event model only; ended/acceptance/identity causes are abstract evidence inputs; no CUDA, DMA, GPU timing, real identity repair, concurrent notification implementation or service measurement')


if __name__ == '__main__':
    main()
