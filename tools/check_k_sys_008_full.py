import itertools


CHUNKS = [b'aaa', b'bbb', b'cc', b'd', b'e', b'fff', b'gg']


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Stream:
    def __init__(self, capacity=8, low=4, items=8, item_low=4, fault=''):
        require(capacity > 0 and 0 <= low < capacity, 'invalid byte limits')
        require(items >= 1 and 0 <= item_low < items, 'invalid item limits')
        self.capacity = capacity
        self.low = low
        self.items = items
        self.item_low = item_low
        self.fault = fault
        self.pending = []
        self.inflight = None
        self.accepted = []
        self.accepted_bytes = 0
        self.committed_bytes = 0
        self.discarded_bytes = 0
        self.uncertain_bytes = 0
        self.paused = False
        self.done = False
        self.terminal = None
        self.end_state = 'NOT_STARTED'
        self.broken = False
        self.next_id = 0
        self.settled = set()
        self.log = []
        # Fake adapter queue: only _launch may append operations.
        self.jobs = []
        self.resumes = 0
        self.used = False

    def occupancy(self):
        events = list(self.pending)
        if self.inflight is not None and self.inflight[1] == 'DATA':
            events.append((self.inflight[2], self.inflight[3]))
        return sum(len(data) for _, data in events), len(events)

    def summary(self):
        return (
            tuple(seq for seq, _ in self.pending), self.inflight,
            self.accepted_bytes, self.committed_bytes,
            self.discarded_bytes, self.uncertain_bytes,
            self.paused, self.done, self.terminal, self.end_state,
            self.broken, self.resumes,
        )

    def boundary(self):
        self.log.append(('boundary', self.summary()))

    def offer(self, data):
        seq = len(self.accepted)
        occupied, count = self.occupancy()
        if self.terminal is not None:
            result = 'terminal'
        elif self.done:
            result = 'closed'
        elif len(data) > self.capacity:
            result = 'too_large'
        else:
            if (self.paused and occupied <= self.low
                    and count <= self.item_low):
                self.paused = False
            room = (occupied + len(data) <= self.capacity
                    and count + 1 <= self.items)
            force = (self.fault == 'overflow' and not self.used
                     and occupied + len(data) > self.capacity)
            ignore = (self.fault == 'ignore_hysteresis'
                      and self.paused and room)
            if (self.paused or not room) and not force and not ignore:
                self.paused = True
                result = 'paused'
            else:
                if force:
                    self.used = True
                if ignore:
                    self.paused = False
                self.accepted.append(data)
                self.accepted_bytes += len(data)
                if self.fault == 'drop' and not self.used and seq == 2:
                    self.used = True
                else:
                    self.pending.append((seq, data))
                result = 'accepted'
        self.log.append(('offer', seq, data, result))
        if result == 'accepted' and self.fault != 'omit_offer_drive':
            self._drive()
        self.boundary()
        return result

    def _launch(self, kind, seq, value):
        require(self.inflight is None, 'parallel launch in subject')
        operation = (self.next_id, kind, seq, value)
        self.next_id += 1
        self.inflight = operation
        if kind == 'END':
            self.end_state = 'SENDING'
        self.log.append(('launch',) + operation)
        self.jobs.append(operation)

    def _decide(self, reason):
        if self.terminal is not None:
            return
        if reason == 'FINAL' and self.fault != 'early_final':
            require(self.done and not self.pending and self.inflight is None,
                    'subject FINAL barrier')
        self.terminal = reason
        self.paused = False
        if reason != 'FINAL':
            self.discarded_bytes += sum(len(data) for _, data in self.pending)
            self.pending = []
        self.log.append(('terminal', reason))

    def _drive(self):
        if self.inflight is not None:
            return
        if self.terminal is None:
            if self.pending:
                seq, data = self.pending.pop(0)
                self._launch('DATA', seq, data)
                return
            if self.done:
                self._decide('FINAL')
        if (self.terminal is not None and self.end_state == 'NOT_STARTED'
                and not self.broken):
            self._launch('END', None, self.terminal)

    def producer_done(self):
        if self.terminal is None and not self.done:
            self.done = True
            self.log.append(('done',))
            if self.fault == 'early_final':
                self.used = True
                self._decide('FINAL')
        else:
            self.log.append(('done_ignored',))
        self._drive()
        self.boundary()

    def stop(self, reason):
        self.log.append(('stop', reason))
        if reason == 'PEER_CLOSED':
            self.broken = True
        self._decide(reason)
        self._drive()
        self.boundary()

    def on_send_complete(self, ticket, success=True):
        if (ticket in self.settled or self.inflight is None
                or self.inflight[0] != ticket):
            self.log.append(('ignored', ticket))
            self.boundary()
            return
        operation = self.inflight
        self.inflight = None
        self.settled.add(ticket)
        self.log.append(('complete', ticket, success))
        if operation[1] == 'DATA':
            if success:
                self.committed_bytes += len(operation[3])
                if self.fault == 'duplicate_complete' and not self.used:
                    self.committed_bytes += len(operation[3])
                    self.used = True
            else:
                self.uncertain_bytes += len(operation[3])
                self.broken = True
                self._decide('PEER_CLOSED')
        else:
            self.end_state = 'SUBMITTED' if success else 'FAILED'
            if not success:
                self.broken = True
        occupied, count = self.occupancy()
        if (self.terminal is None and self.paused
                and occupied <= self.low and count <= self.item_low):
            self.paused = False
            self.resumes += 1
        if self.fault != 'omit_complete_drive':
            self._drive()
        self.boundary()


def deliver_next(stream, success=True):
    # Deliver a completion only for an operation already launched by the subject.
    # This driver never consumes pending and never calls _drive or a flush API.
    require(bool(stream.jobs), 'no previously launched operation can complete')
    operation = stream.jobs.pop(0)
    stream.on_send_complete(operation[0], success)
    return operation[0]


def audit(stream, expected, terminal_expected):
    # Reference state is reconstructed from flat history and disjoint sets.
    # Subject pending, occupancy and counters are not used to decide expectations.
    errors = []
    accepted = []
    started = set()
    committed = set()
    discarded = set()
    uncertain = set()
    active = None
    paused = False
    done = False
    terminal = None
    end_state = 'NOT_STARTED'
    broken = False
    resumes = 0
    next_id = 0
    due = None

    def need(condition, message):
        if not condition:
            errors.append(message)

    def pending():
        return [(seq, data) for seq, data in enumerate(accepted)
                if seq not in started and seq not in discarded]

    def live():
        return [(seq, data) for seq, data in enumerate(accepted)
                if seq not in committed and seq not in discarded
                and seq not in uncertain]

    def occupancy():
        events = live()
        return sum(len(data) for _, data in events), len(events)

    def total(indices):
        return sum(len(accepted[seq]) for seq in indices
                   if 0 <= seq < len(accepted))

    for event in stream.log:
        kind = event[0]
        occupied, count = occupancy()
        if kind == 'offer':
            seq, data, result = event[1:]
            if terminal is not None:
                wanted = 'terminal'
            elif done:
                wanted = 'closed'
            elif len(data) > stream.capacity:
                wanted = 'too_large'
            else:
                if occupied <= stream.low and count <= stream.item_low:
                    paused = False
                blocked = (paused or occupied + len(data) > stream.capacity
                           or count + 1 > stream.items)
                wanted = 'paused' if blocked else 'accepted'
                if wanted == 'paused':
                    paused = True
            need(result == wanted, 'offer_result')
            if result == 'accepted':
                need(seq == len(accepted), 'accept_sequence')
                need(seq < len(expected) and data == expected[seq],
                     'accept_payload')
                accepted.append(data)
        elif kind == 'launch':
            operation = tuple(event[1:])
            ticket, op_kind, seq, value = operation
            need(active is None, 'parallel_launch')
            need(ticket == next_id, 'ticket_sequence')
            next_id += 1
            if op_kind == 'DATA':
                waiting = pending()
                need(terminal is None, 'data_after_terminal')
                need(bool(waiting) and seq == waiting[0][0], 'launch_order')
                need(0 <= seq < len(accepted) and accepted[seq] == value,
                     'launch_payload')
                started.add(seq)
            elif op_kind == 'END':
                need(terminal is not None and end_state == 'NOT_STARTED'
                     and not broken, 'end_launch')
                need(value == terminal, 'end_reason')
                end_state = 'SENDING'
            else:
                need(False, 'unknown_operation')
            active = operation
        elif kind == 'done':
            need(not done and terminal is None, 'done_state')
            need(len(accepted) == len(expected), 'producer_missing')
            done = True
        elif kind == 'done_ignored':
            need(done or terminal is not None, 'unexpected_done_ignore')
        elif kind == 'stop':
            if event[1] == 'PEER_CLOSED':
                broken = True
            if terminal is None:
                due = event[1]
        elif kind == 'terminal':
            reason = event[1]
            need(terminal is None, 'duplicate_terminal')
            if reason == 'FINAL':
                need(done, 'final_before_done')
                need(not pending() and active is None, 'final_before_drain')
                need(due is None, 'terminal_race')
            else:
                need(due == reason, 'terminal_reason')
                discarded.update(seq for seq, _ in pending())
            terminal = reason
            due = None
            paused = False
        elif kind == 'complete':
            ticket, success = event[1:]
            need(active is not None and active[0] == ticket, 'completion_ticket')
            if active is None:
                continue
            operation = active
            active = None
            if operation[1] == 'DATA':
                if success:
                    committed.add(operation[2])
                else:
                    uncertain.add(operation[2])
                    broken = True
                    if terminal is None:
                        due = 'PEER_CLOSED'
            else:
                end_state = 'SUBMITTED' if success else 'FAILED'
                if not success:
                    broken = True
            occupied, count = occupancy()
            if (terminal is None and paused and occupied <= stream.low
                    and count <= stream.item_low):
                paused = False
                resumes += 1
        elif kind == 'ignored':
            need(active is None or event[1] != active[0],
                 'ignored_current_ticket')
        elif kind == 'boundary':
            need(due is None, 'missing_decision')
            need(not (terminal is None and pending() and active is None),
                 'idle_pending')
            need(not (terminal is None and done and not pending()
                      and active is None), 'missing_final')
            need(not (terminal is not None and active is None
                      and end_state == 'NOT_STARTED' and not broken),
                 'missing_end')
            wanted = (
                tuple(seq for seq, _ in pending()), active,
                sum(len(data) for data in accepted), total(committed),
                total(discarded), total(uncertain), paused, done,
                terminal, end_state, broken, resumes,
            )
            need(event[1] == wanted, 'state_snapshot')
        else:
            need(False, 'unknown_event')
        occupied, count = occupancy()
        need(occupied <= stream.capacity and count <= stream.items, 'capacity')

    need(terminal == terminal_expected, 'final_reason')
    return errors


def checked(stream, chunks, terminal, label):
    errors = audit(stream, chunks, terminal)
    require(not errors, '%s: %r' % (label, errors))


def run_to_final(stream, chunks):
    for data in chunks:
        attempts = 0
        while True:
            result = stream.offer(data)
            if result == 'accepted':
                break
            require(result == 'paused', 'unexpected offer: ' + result)
            attempts += 1
            require(attempts <= stream.items + 2, 'retry bound exceeded')
            deliver_next(stream)
    stream.producer_done()
    steps = 0
    while stream.jobs:
        steps += 1
        require(steps <= len(chunks) + 2, 'completion bound exceeded')
        deliver_next(stream)
    require(stream.end_state == 'SUBMITTED', 'END was not submitted')


def main():
    liveness = Stream()
    require(liveness.offer(b'aaa') == 'accepted', 'first acceptance failed')
    require(not liveness.done and liveness.inflight is not None
            and liveness.inflight[1] == 'DATA' and len(liveness.jobs) == 1,
            'first Offer did not launch DATA before producer completion')
    for data in [b'bbb', b'cc']:
        require(liveness.offer(data) == 'accepted', 'liveness setup failed')
    require(liveness.occupancy() == (8, 3), 'inflight lost capacity ownership')
    require(liveness.offer(b'd') == 'paused', 'full queue did not pause')
    deliver_next(liveness)
    require(liveness.occupancy()[0] == 5, 'first completion occupancy')
    require(liveness.offer(b'd') == 'paused', 'hysteresis released too early')
    deliver_next(liveness)
    require(not liveness.paused and liveness.resumes == 1,
            'completion did not publish low-water recovery')
    require(liveness.offer(b'd') == 'accepted', 'retained block retry failed')
    liveness.producer_done()
    while liveness.jobs:
        deliver_next(liveness)
    checked(liveness, [b'aaa', b'bbb', b'cc', b'd'], 'FINAL', 'Offer liveness')
    require(liveness.committed_bytes == 9, 'finite output did not drain')
    print('liveness: first Offer launches DATA before done; 8->5->2 B; retry and FINAL complete')

    regression = Stream()
    run_to_final(regression, CHUNKS)
    checked(regression, CHUNKS, 'FINAL', 'seven-block regression')
    observed = [(event[1], event[3]) for event in regression.log
                if event[0] == 'offer']
    fixed_reference = [
        (0, 'accepted'), (1, 'accepted'), (2, 'accepted'),
        (3, 'paused'), (3, 'paused'), (3, 'accepted'),
        (4, 'accepted'), (5, 'accepted'),
        (6, 'paused'), (6, 'paused'), (6, 'accepted'),
    ]
    require(observed == fixed_reference, 'fixed hysteresis trace differs')
    launched = [event[4] for event in regression.log
                if event[0] == 'launch' and event[2] == 'DATA']
    require(launched == CHUNKS, 'adapter launch sequence differs from reference')
    require(regression.committed_bytes == 15 and regression.resumes == 2,
            'regression byte or recovery count differs')
    print('regression: 7 DATA launches; 15 B; 4 PAUSED returns; 2 recoveries; FINAL')

    count = 0
    for capacity in range(1, 5):
        for low in range(capacity):
            for length in range(5):
                for sizes in itertools.product(range(4), repeat=length):
                    if any(size > capacity for size in sizes):
                        continue
                    chunks = [bytes([65 + seq]) * size
                              for seq, size in enumerate(sizes)]
                    candidate = Stream(capacity, low, 3, 1)
                    run_to_final(candidate, chunks)
                    checked(candidate, chunks, 'FINAL',
                            'enumerated %r' % ((capacity, low, sizes),))
                    count += 1
    require(count == 2660, 'enumeration coverage changed')
    print('finite CPU model: capacity/low-water/chunk combinations=%d' % count)

    cancel = Stream()
    for data in CHUNKS[:3]:
        require(cancel.offer(data) == 'accepted', 'cancel setup failed')
    cancel.stop('CANCELLED')
    require(cancel.discarded_bytes == 5 and cancel.occupancy() == (3, 1),
            'cancel released inflight data or retained pending data')
    deliver_next(cancel)
    deliver_next(cancel)
    cancel.on_send_complete(0)
    cancel.stop('TIMEOUT')
    require(cancel.offer(b'x') == 'terminal', 'post-cancel acceptance')
    checked(cancel, CHUNKS, 'CANCELLED', 'cancel before completion')
    require(cancel.committed_bytes == 3 and cancel.discarded_bytes == 5,
            'cancel conservation differs')
    print('cancel: pending 5 B discarded; inflight 3 B retained until completion; reason stable')

    for cancel_first in [False, True]:
        for success in [False, True]:
            candidate = Stream()
            candidate.offer(b'a')
            candidate.offer(b'b')
            if cancel_first:
                candidate.stop('CANCELLED')
            deliver_next(candidate, success)
            if not cancel_first:
                candidate.stop('CANCELLED')
            while candidate.jobs:
                deliver_next(candidate)
            reason = 'CANCELLED' if cancel_first or success else 'PEER_CLOSED'
            checked(candidate, [b'a', b'b'], reason, 'cancel/completion order')
    print('event orders: cancel before/after DATA success/failure preserve first decision')

    tickets = Stream()
    tickets.offer(b'a')
    tickets.offer(b'b')
    deliver_next(tickets)
    original = tickets.summary()
    tickets.on_send_complete(0)
    tickets.on_send_complete(999)
    require(tickets.summary() == original, 'stale callback changed active ownership')
    tickets.producer_done()
    tickets.producer_done()
    while tickets.jobs:
        deliver_next(tickets)
    checked(tickets, [b'a', b'b'], 'FINAL', 'ticket identity')
    print('tickets: duplicate old and wrong identities cannot settle current DATA')

    failed = Stream()
    for data in CHUNKS[:3]:
        failed.offer(data)
    deliver_next(failed, False)
    require(failed.uncertain_bytes == 3 and failed.discarded_bytes == 5
            and failed.committed_bytes == 0 and not failed.jobs,
            'failed DATA ownership classification')
    failed.stop('CANCELLED')
    checked(failed, CHUNKS, 'PEER_CLOSED', 'DATA failure')
    print('DATA failure: unknown coverage 3 B; discarded 5 B; no END on broken transport')

    empty = Stream()
    empty.producer_done()
    require(empty.terminal == 'FINAL' and empty.end_state == 'SENDING',
            'empty output did not launch END')
    deliver_next(empty, False)
    empty.stop('PEER_CLOSED')
    checked(empty, [], 'FINAL', 'END failure')
    require(empty.end_state == 'FAILED' and empty.broken,
            'END failure lost delivery status')
    zero = Stream(1, 0, 2, 0)
    run_to_final(zero, [b'', b'', b''])
    checked(zero, [b'', b'', b''], 'FINAL', 'zero-byte item capacity')
    require(any(event[0] == 'offer' and event[3] == 'paused'
                for event in zero.log), 'empty events never exercised item limit')
    print('empty/zero-byte: END failure remains business FINAL; event capacity exercised')

    barrier = Stream()
    barrier.offer(b'a')
    barrier.producer_done()
    require(barrier.terminal is None, 'FINAL crossed inflight barrier')
    require(barrier.offer(b'b') == 'closed', 'post-done acceptance')
    deliver_next(barrier)
    require(barrier.terminal == 'FINAL', 'drained done stream did not finish')
    deliver_next(barrier)
    checked(barrier, [b'a'], 'FINAL', 'FINAL barrier')
    oversize = Stream()
    require(oversize.offer(b'123456789') == 'too_large', 'oversize accepted')
    oversize.stop('FAILED')
    deliver_next(oversize)
    checked(oversize, [], 'FAILED', 'oversize')
    print('boundaries: producer closed after done; FINAL waits for DATA; 9 B rejected at Q=8 B')

    for fault, target in [
        ('overflow', 'capacity'),
        ('ignore_hysteresis', 'offer_result'),
        ('drop', 'final_before_drain'),
        ('duplicate_complete', 'state_snapshot'),
        ('early_final', 'final_before_drain'),
    ]:
        faulty = Stream(fault=fault)
        run_to_final(faulty, CHUNKS)
        errors = audit(faulty, CHUNKS, 'FINAL')
        require(target in errors, 'negative target missed: %s %r' % (fault, errors))
        if fault != 'ignore_hysteresis':
            require(faulty.used, 'fault injection not reached: ' + fault)
        print('negative %s detected: %s' % (fault, target))

    for fault in ['omit_offer_drive', 'omit_complete_drive']:
        faulty = Stream(fault=fault)
        faulty.offer(b'aaa')
        if fault == 'omit_complete_drive':
            faulty.offer(b'bbb')
            deliver_next(faulty)
        errors = audit(faulty, [b'aaa', b'bbb'], None)
        require('idle_pending' in errors, 'missing Drive escaped: ' + fault)
        require(not faulty.jobs, 'unexpected adapter operation masks missing Drive')
        print('negative %s detected: idle_pending; no external flush used' % fault)

    print('PASS: finite single-stream CPU event model; no ASGI, network, GPU or performance measurement')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
