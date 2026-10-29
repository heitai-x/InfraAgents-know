#!/usr/bin/env python3
# K-ALG-017: deterministic, standalone, read-only, offline CPU model.
# Logical positions occupy one capacity unit. No hardware measurements.
from collections import namedtuple
from copy import deepcopy
from fractions import Fraction


class CheckError(AssertionError):
    pass


class ProtocolError(ValueError):
    pass


def check(condition, message):
    if not condition:
        raise CheckError(message)


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def rejects(action, reason):
    try:
        action()
    except ProtocolError as exc:
        check(str(exc) == reason, 'unexpected rejection: ' + str(exc))
    else:
        raise CheckError('expected rejection: ' + reason)


Obj = namedtuple('Obj', 'identity start end')
OBJ = Obj(('model', 'representation', 1), 0, 32)


def reference(obj, begin, end):
    # Logical coordinate oracle, independent of copies and simulated memory.
    return [obj.identity[2] * 1000 + i for i in range(begin, end)]


class Machine:
    def __init__(self, limits=None, fault=''):
        self.limits = dict({'H': 100, 'D': 100} if limits is None else limits)
        self.fault = fault
        self.copies = {}
        self.live = {}
        self.memory = {}
        self.epochs = {}
        self.ops = {}
        self.views = {}
        self.reads = {}
        self.ended_reads = set()
        self.closed = set()
        self.waiting = {}
        self.admitted = {}
        self.events = []

    def bytes(self, tier):
        return sum(self.copies[k]['end'] - self.copies[k]['begin']
                   for k in self.live.values()
                   if self.copies[k]['tier'] == tier)

    def validate_specs(self, specs):
        slots = set()
        totals = {t: self.bytes(t) for t in self.limits}
        for obj, begin, end, tier, slot, owner in specs:
            require(owner not in self.closed, 'closed owner')
            require(isinstance(begin, int) and isinstance(end, int)
                    and 0 <= obj.start <= begin < end <= obj.end, 'range')
            require(slot not in self.live and slot not in slots, 'busy')
            require(tier in totals, 'capacity')
            slots.add(slot)
            totals[tier] += end - begin
            require(totals[tier] <= self.limits[tier], 'capacity')

    def reserve(self, obj, begin, end, tier, slot, owner='build'):
        self.validate_specs([(obj, begin, end, tier, slot, owner)])
        epoch = self.epochs.get(slot, -1) + 1
        key = (slot, epoch)
        self.epochs[slot] = epoch
        self.live[slot] = key
        self.memory[slot] = [None] * (end - begin)
        self.copies[key] = dict(obj=obj, begin=begin, end=end, tier=tier,
                                slot=slot, state='RESERVED',
                                owners={owner}, uses=set())
        self.events.append(('reserve', key, owner))
        return key

    def reserve_many(self, specs):
        self.validate_specs(specs)
        return [self.reserve(*spec) for spec in specs]

    def queue(self, request_id, owner, specs):
        require(owner not in self.closed, 'closed owner')
        require(request_id not in self.waiting
                and request_id not in self.admitted, 'request reuse')
        require(bool(specs) and all(s[5] == owner for s in specs), 'request owner')
        self.waiting[request_id] = (owner, tuple(specs))
        self.wake()

    def wake(self):
        # No hidden gc: missing retirement/reclamation cannot be repaired here.
        self.events.append(('wake',))
        for request_id, (owner, specs) in list(self.waiting.items()):
            if owner in self.closed:
                del self.waiting[request_id]
                continue
            try:
                self.validate_specs(specs)
            except ProtocolError as exc:
                if str(exc) not in ('capacity', 'busy'):
                    raise
                continue
            keys = self.reserve_many(specs)
            del self.waiting[request_id]
            self.admitted[request_id] = keys
            self.events.append(('admit', request_id))

    def seed(self, obj, begin, end, tier, slot):
        key = self.reserve(obj, begin, end, tier, slot, 'cache')
        self.memory[slot] = reference(obj, begin, end)
        self.copies[key]['state'] = 'READY'
        return key

    def attach(self, key, owner):
        require(owner not in self.closed, 'closed owner')
        require(self.copies[key]['state'] in ('READY', 'LOADING'), 'attach')
        self.copies[key]['owners'].add(owner)

    def withdraw_owner_raw(self, key, owner):
        # Only withdraw L/views: no retirement, gc or wake can mask a mutant.
        rec = self.copies[key]
        if owner not in rec['owners']:
            return False
        rec['owners'].remove(owner)
        self.views = {v: w for v, w in self.views.items()
                      if not (w['key'] == key and w['owner'] == owner)}
        self.events.append(('remove', key, owner))
        return True

    def retire_unowned(self, key):
        rec = self.copies[key]
        if not rec['owners'] and rec['state'] not in ('RETIRING', 'GONE'):
            rec['state'] = 'RETIRING'
            self.events.append(('retire', key))

    def withdraw_holds(self, pairs, retire=True):
        # Chapter WITHDRAW_HOLDS: raw withdrawal, then explicit retirement.
        pairs = tuple(pairs)
        affected = tuple(dict.fromkeys(key for key, owner in pairs))
        for key, owner in pairs:
            self.withdraw_owner_raw(key, owner)
        if retire:
            for key in affected:
                self.retire_unowned(key)
        return affected

    def remove_owner(self, key, owner):
        require(owner in self.copies[key]['owners'], 'owner')
        self.withdraw_holds(((key, owner),))

    def drop(self, key, owner):
        self.remove_owner(key, owner)
        self.gc(key)
        self.wake()

    def cancel(self, owner):
        if owner in self.closed:
            return 'DUPLICATE'
        self.closed.add(owner)
        self.events.append(('close', owner))
        self.waiting = {r: w for r, w in self.waiting.items() if w[0] != owner}
        affected = [k for k, rec in self.copies.items()
                    if owner in rec['owners']]
        for key in affected:
            self.remove_owner(key, owner)
        if self.fault != 'cancel_no_gc':
            for key in affected:
                self.gc(key)
        if self.fault != 'cancel_no_wake':
            self.wake()
        return 'CANCELED'

    def gc(self, key):
        rec = self.copies[key]
        if (rec['state'] == 'RETIRING' and not rec['owners']
                and (not rec['uses'] or self.fault == 'early_free')):
            require(self.live.get(rec['slot']) == key, 'slot identity')
            del self.live[rec['slot']]
            rec['state'] = 'GONE'
            self.events.append(('free', key))
            return True
        return False

    def start(self, op_id, source, destination, owner='build'):
        require(owner not in self.closed, 'closed owner')
        require(op_id not in self.ops, 'operation reuse')
        src, dst = self.copies[source], self.copies[destination]
        require(src['state'] == 'READY' and dst['state'] == 'RESERVED'
                and owner in dst['owners'] and src['obj'] == dst['obj']
                and src['begin'] <= dst['begin'] < dst['end'] <= src['end'],
                'source')
        ticket = (op_id, source, destination, dst['obj'],
                  dst['begin'], dst['end'])
        self.ops[op_id] = dict(source=source, destination=destination,
                               ticket=ticket, covered=set(), ended=False)
        src['uses'].add(('copy', op_id))
        dst['uses'].add(('copy', op_id))
        dst['state'] = 'LOADING'
        return ticket

    def progress(self, op_id, ticket, begin, end):
        op = self.ops.get(op_id)
        require(op is not None and not op['ended'] and ticket == op['ticket'],
                'ticket')
        src, dst = self.copies[op['source']], self.copies[op['destination']]
        require(dst['begin'] <= begin < end <= dst['end'], 'progress range')
        require(self.live.get(src['slot']) == op['source']
                and self.live.get(dst['slot']) == op['destination'],
                'physical lifetime')
        for i in range(begin, end):
            self.memory[dst['slot']][i - dst['begin']] = (
                self.memory[src['slot']][i - src['begin']])
            op['covered'].add(i)

    def finish(self, op_id, ticket, access_ended=True, success=True, settle=True):
        op = self.ops.get(op_id)
        require(op is not None and ticket == op['ticket'], 'certificate')
        if op['ended']:
            return 'DUPLICATE'
        require(access_ended, 'no last-access proof')
        src, dst = self.copies[op['source']], self.copies[op['destination']]
        op['ended'] = True
        src['uses'].remove(('copy', op_id))
        dst['uses'].remove(('copy', op_id))
        complete = all(i in op['covered']
                       for i in range(dst['begin'], dst['end']))
        if (dst['state'] == 'LOADING' and success and complete
                and dst['owners']):
            dst['state'] = 'READY'
        elif dst['state'] == 'LOADING':
            self.withdraw_holds((op['destination'], owner)
                                for owner in tuple(dst['owners']))
            self.retire_unowned(op['destination'])
        if settle:
            self.gc(op['source'])
            self.gc(op['destination'])
            self.wake()
        return dst['state']

    def publish(self, view_id, key, obj, begin, end, owner):
        require(owner not in self.closed, 'closed owner')
        rec = self.copies[key]
        require(view_id not in self.views, 'view reuse')
        require((rec['state'] == 'READY' or self.fault == 'early_publish')
                and rec['obj'] == obj
                and rec['begin'] <= begin < end <= rec['end']
                and owner in rec['owners'], 'publish')
        self.views[view_id] = dict(key=key, obj=obj, begin=begin, end=end,
                                   owner=owner)

    def acquire(self, view_id, reader_id):
        require(view_id in self.views, 'read')
        require(reader_id not in self.reads
                and reader_id not in self.ended_reads, 'reader reuse')
        view = self.views[view_id]
        rec = self.copies[view['key']]
        require(rec['state'] == 'READY' and view['owner'] in rec['owners']
                and view['owner'] not in self.closed, 'read')
        self.reads[reader_id] = view['key']
        rec['uses'].add(('reader', reader_id))
        return self.memory[rec['slot']][view['begin'] - rec['begin']:
                                        view['end'] - rec['begin']]

    def release(self, reader_id, access_ended=True):
        if reader_id in self.ended_reads:
            return 'DUPLICATE'
        require(reader_id in self.reads, 'unknown reader')
        require(access_ended, 'no last-access proof')
        key = self.reads.pop(reader_id)
        self.ended_reads.add(reader_id)
        self.copies[key]['uses'].remove(('reader', reader_id))
        self.gc(key)
        self.wake()
        return 'ENDED'


class SubmissionMachine(Machine):
    # Extension preserves the old model/regressions and exposes the interval
    # after actual U registration/LOADING but before acceptance is decided.
    def __init__(self, limits=None, fault=''):
        super().__init__(limits, fault)
        self.stage_plans = {}
        self.retry_pending = set()
        self.replacement_pending = {}

    def start(self, op_id, source, destination, owner='build', plan_id=None,
              workspaces=(), plan_holds=None):
        pid = op_id if plan_id is None else plan_id
        workspaces = tuple(workspaces)
        accesses = (source, destination) + workspaces
        require(len(set(accesses)) == len(accesses), 'access alias')
        holds = ((destination,) + workspaces if plan_holds is None
                 else tuple(plan_holds))
        require(len(set(holds)) == len(holds)
                and set((destination,) + workspaces) <= set(holds)
                and set(holds) <= set(accesses), 'plan holds')
        descriptor = (source, destination, owner, workspaces, holds)
        previous = self.stage_plans.get(pid)
        if previous is not None:
            require(previous['state'] == 'RETRY_WAIT'
                    and previous['open'], 'closed plan')
            require(previous['descriptor'] == descriptor, 'plan identity')
        for key in holds:
            require(owner in self.copies[key]['owners'], 'plan owner')
        for key in workspaces:
            rec = self.copies[key]
            require(rec['state'] == 'RESERVED' and not rec['uses']
                    and self.live.get(rec['slot']) == key, 'workspace')
        # This is the real base registration, not an anticipated rejection.
        base_ticket = super().start(op_id, source, destination, owner)
        ticket = base_ticket + (pid, workspaces, holds)
        op = self.ops[op_id]
        op.update(ticket=ticket, accesses=accesses, phase='REGISTERED',
                  outcome=None, plan_id=pid)
        for key in workspaces:
            self.copies[key]['uses'].add(('copy', op_id))
            self.copies[key]['state'] = 'LOADING'
        self.stage_plans[pid] = dict(descriptor=descriptor, state='REGISTERED',
                                     open=True, op_id=op_id)
        self.retry_pending.discard(pid)
        for key in (destination,) + workspaces:
            self.replacement_pending.pop(key, None)
        self.events.append(('registered', op_id, accesses))
        self.events.append(('loading', op_id))
        return ticket

    def gc(self, key):
        freed = super().gc(key)
        if freed:
            self.replacement_pending.pop(key, None)
        return freed

    def accept(self, op_id, ticket):
        op = self.ops.get(op_id)
        require(op is not None and ticket == op['ticket'], 'certificate')
        require(not op['ended'], 'settled operation')
        require(op['phase'] == 'REGISTERED', 'acceptance')
        op['phase'] = 'ACCEPTED'
        self.events.append(('accepted', op_id))

    def progress(self, op_id, ticket, begin, end):
        op = self.ops.get(op_id)
        require(op is not None and not op['ended']
                and op['phase'] == 'ACCEPTED', 'not accepted')
        return super().progress(op_id, ticket, begin, end)

    def not_started(self, op_id, ticket, proof, decision='abandon'):
        op = self.ops.get(op_id)
        require(op is not None and ticket == op['ticket'], 'certificate')
        require(proof == ('ALL_NOT_STARTED', ticket), 'unstarted evidence')
        if op['ended']:
            require(op['outcome'] == 'NOT_STARTED', 'terminal conflict')
            return 'DUPLICATE'
        require(op['phase'] == 'REGISTERED' and not op['covered'],
                'not unstarted')
        require(decision in ('abandon', 'retry'), 'decision')
        pid = op['plan_id']
        plan = self.stage_plans[pid]
        source, destination, owner, workspaces, holds = plan['descriptor']
        writes = (destination,) + workspaces
        if decision == 'retry':
            require(owner not in self.closed and plan['open'], 'retry eligibility')
            require(self.copies[source]['state'] == 'READY'
                    and all(owner in self.copies[k]['owners'] for k in holds)
                    and all(self.copies[k]['state'] == 'LOADING'
                            and self.copies[k]['uses'] == {('copy', op_id)}
                            for k in writes), 'retry resources')
        # A trusted whole-operation rejection settles exactly this intent.
        op['ended'] = True
        op['outcome'] = 'NOT_STARTED'
        op['phase'] = 'TERMINAL'
        self.events.append(('not_started', op_id))
        for key in op['accesses']:
            self.copies[key]['uses'].remove(('copy', op_id))
        if decision == 'retry':
            for key in writes:
                rec = self.copies[key]
                rec['state'] = 'RESERVED'
                self.memory[rec['slot']] = [None] * (rec['end'] - rec['begin'])
            plan['state'] = 'RETRY_WAIT'
            self.retry_pending.add(pid)
            self.events.append(('retry_wait', pid))
            self.wake()
            return 'RETRY_WAIT'
        plan['open'] = False
        plan['state'] = 'FAILED'
        self.retry_pending.discard(pid)
        self.events.append(('close_plan', pid))
        # Both the real path and mutant use raw withdrawal. Only the explicit
        # retirement step is omitted by the old NOT_STARTED negative control.
        self.withdraw_holds(((key, owner) for key in holds),
                            retire=self.fault != 'unstarted_no_retire')
        for key in holds:
            rec = self.copies[key]
            if rec['owners'] and key in writes and rec['state'] == 'LOADING':
                rec['state'] = 'RESERVED'
                self.memory[rec['slot']] = [None] * (rec['end'] - rec['begin'])
                self.replacement_pending[key] = set(rec['owners'])
        for key in dict.fromkeys(op['accesses'] + holds):
            self.gc(key)
        self.wake()
        return 'FAILED'

    def retry_stage(self, plan_id, new_op_id):
        plan = self.stage_plans.get(plan_id)
        require(plan is not None and plan_id in self.retry_pending
                and plan['state'] == 'RETRY_WAIT' and plan['open'], 'retry plan')
        source, destination, owner, workspaces, holds = plan['descriptor']
        return self.start(new_op_id, source, destination, owner,
                          plan_id, workspaces, holds)

    def finish(self, op_id, ticket, access_ended=True, success=True):
        op = self.ops.get(op_id)
        require(op is not None and ticket == op['ticket'], 'certificate')
        if op['ended']:
            return 'DUPLICATE'
        require(op['phase'] == 'ACCEPTED', 'not accepted')
        # Base settles data/U only here. Do not wake before auxiliary cleanup.
        result = super().finish(op_id, ticket, access_ended, success, settle=False)
        plan = self.stage_plans[op['plan_id']]
        source, destination, owner, workspaces, holds = plan['descriptor']
        for key in workspaces:
            self.copies[key]['uses'].remove(('copy', op_id))
        op['phase'] = 'TERMINAL'
        op['outcome'] = 'SUCCESS' if result == 'READY' else 'FAILED'
        plan['state'] = 'DONE' if result == 'READY' else 'FAILED'
        plan['open'] = False
        self.retry_pending.discard(op['plan_id'])
        self.events.append(('close_plan', op['plan_id']))
        auxiliary = tuple((key, owner) for key in holds if key != destination)
        self.withdraw_holds(auxiliary, retire=self.fault != 'end_aux_no_retire')
        for key in workspaces:
            rec = self.copies[key]
            if rec['owners'] and rec['state'] == 'LOADING':
                rec['state'] = 'RESERVED'
                self.replacement_pending[key] = set(rec['owners'])
        for key in dict.fromkeys(op['accesses'] + holds):
            self.gc(key)
        self.events.append(('end', op_id))
        self.wake()
        return result

    def cancel(self, owner):
        result = super().cancel(owner)
        for pid, plan in self.stage_plans.items():
            if plan['descriptor'][2] == owner:
                plan['open'] = False
                if plan['state'] == 'RETRY_WAIT':
                    plan['state'] = 'CANCELED'
                self.retry_pending.discard(pid)
        return result


def audit(m):
    # Reconstruct expected protection from operation/read tables, not uses.
    errors = []
    expected = {k: set() for k in m.copies}
    for op_id, op in m.ops.items():
        if not op['ended']:
            accesses = op.get('accesses', (op['source'], op['destination']))
            for key in accesses:
                expected[key].add(('copy', op_id))
                if m.live.get(m.copies[key]['slot']) != key:
                    errors.append('inflight allocation missing')
    for reader_id, key in m.reads.items():
        expected[key].add(('reader', reader_id))
        if m.live.get(m.copies[key]['slot']) != key:
            errors.append('reader allocation missing')
    for key, rec in m.copies.items():
        if rec['uses'] != expected[key]:
            errors.append('use ledger mismatch')
        if rec['state'] != 'GONE' and m.live.get(rec['slot']) != key:
            errors.append('live copy allocation missing')
        if rec['state'] == 'GONE' and (rec['owners'] or expected[key]):
            errors.append('gone copy still protected')
        if rec['owners'] & m.closed:
            errors.append('closed owner retained')
        if (rec['state'] == 'RETIRING' and not rec['owners']
                and not expected[key]):
            errors.append('idle retired allocation retained')
        if (rec['state'] == 'LOADING' and not rec['owners']
                and not expected[key]):
            errors.append('unowned loading allocation retained')
    for slot, key in m.live.items():
        rec = m.copies[key]
        if rec['slot'] != slot or rec['state'] == 'GONE':
            errors.append('allocation identity')
        if rec['state'] == 'READY':
            if m.memory[slot] != reference(rec['obj'], rec['begin'], rec['end']):
                errors.append('ready content')
    for view in m.views.values():
        rec = m.copies[view['key']]
        if rec['state'] != 'READY':
            errors.append('unready view')
        if view['owner'] not in rec['owners'] or view['owner'] in m.closed:
            errors.append('unauthorized view')
        if not (rec['obj'] == view['obj']
                and rec['begin'] <= view['begin'] < view['end'] <= rec['end']):
            errors.append('view identity or range')
    for tier, limit in m.limits.items():
        if m.bytes(tier) > limit:
            errors.append('capacity')
    return errors


def healthy(m):
    errors = audit(m)
    check(not errors, 'audit: ' + repr(errors))


def empty_access(obj, begin, end):
    require(obj.start <= begin <= end <= obj.end, 'range')
    require(begin == end, 'not empty')
    return ()


def test_ranges_and_reservation():
    cases = [(0, 16, True), (0, 32, True), (31, 32, True),
             (-1, 16, False), (0, 33, False), (32, 33, False),
             (4, 4, False), (8, 4, False)]
    for begin, end, valid in cases:
        m = Machine()
        before = deepcopy(m.__dict__)
        if valid:
            m.reserve(OBJ, begin, end, 'H', 'x')
            check(m.bytes('H') == end - begin, 'legal subrange')
        else:
            rejects(lambda: m.reserve(OBJ, begin, end, 'H', 'x'), 'range')
            check(m.__dict__ == before, 'rejection mutated state')
    small = Obj(OBJ.identity, 0, 4)
    legal = {(a, b) for a in range(4) for b in range(a + 1, 5)}
    for begin in range(-1, 6):
        for end in range(-1, 6):
            m = Machine()
            if (begin, end) in legal:
                m.reserve(small, begin, end, 'H', 'x')
            else:
                before = deepcopy(m.__dict__)
                rejects(lambda: m.reserve(small, begin, end, 'H', 'x'), 'range')
                check(m.__dict__ == before, 'range rejection leaked allocation')
    m = Machine({'H': 20, 'D': 100})
    check(empty_access(OBJ, 32, 32) == (), 'empty access')
    check(not m.live and not m.ops, 'empty access generated work')
    before = deepcopy(m.__dict__)
    rejects(lambda: m.reserve_many([(OBJ, 0, 16, 'H', 'a', 'build'),
                                    (OBJ, 16, 32, 'H', 'b', 'build')]),
            'capacity')
    check(m.__dict__ == before, 'bundle failure left partial reservation')
    keys = m.reserve_many([(OBJ, 0, 8, 'H', 'a', 'build'),
                           (OBJ, 8, 16, 'H', 'b', 'build')])
    check(len(keys) == 2 and m.bytes('H') == 16, 'bundle allocation')
    rejects(lambda: m.reserve(OBJ, 0, 1, 'D', 'a'), 'busy')
    healthy(m)
    return 57


def test_event_traces():
    count = 0
    for cancellation in ('none', 'before', 'partial', 'after'):
        for reverse in (False, True):
            for failure in (False, True):
                m = Machine()
                source = m.seed(OBJ, 0, 32, 'D', 'source')
                destination = m.reserve(OBJ, 0, 16, 'H', 'destination')
                ticket = m.start('copy', source, destination)
                rejects(lambda: m.publish('early', destination, OBJ, 0, 16,
                                          'build'), 'publish')
                if cancellation == 'before':
                    m.cancel('build')
                first = (8, 16) if reverse else (0, 8)
                second = (0, 8) if reverse else (8, 16)
                m.progress('copy', ticket, *first)
                if cancellation == 'partial':
                    m.cancel('build')
                # Independent reconstruction authorizes this disposable source;
                # this does not authorize deleting arbitrary last KV copies.
                m.drop(source, 'cache')
                healthy(m)
                m.progress('copy', ticket, *second)
                m.progress('copy', ticket, 0, 8)
                rejects(lambda: m.progress('copy', ticket, 0, 17),
                        'progress range')
                bad = (ticket[0], ('wrong', 9)) + ticket[2:]
                rejects(lambda: m.finish('copy', bad), 'certificate')
                rejects(lambda: m.finish('copy', ticket, False),
                        'no last-access proof')
                healthy(m)
                m.finish('copy', ticket, True, not failure)
                if cancellation == 'after':
                    m.cancel('build')
                if m.copies[destination]['state'] == 'READY':
                    m.publish('v', destination, OBJ, 0, 16, 'build')
                    check(m.acquire('v', 'r1') == reference(OBJ, 0, 16),
                          'copied contents differ from coordinate reference')
                    m.acquire('v', 'r2')
                    m.cancel('build')
                    rejects(lambda: m.release('r1', False), 'no last-access proof')
                    m.release('r1')
                    check(m.release('r1') == 'DUPLICATE', 'reader deduplication')
                    check(m.copies[destination]['state'] == 'RETIRING',
                          'duplicate r1 released r2')
                    healthy(m)
                    m.release('r2')
                check(m.copies[source]['state'] == 'GONE'
                      and m.copies[destination]['state'] == 'GONE', 'drain')
                newer = Obj(('model', 'representation', 2), 0, 32)
                current = m.reserve(newer, 0, 16, 'H', 'destination', 'new')
                check(m.finish('copy', ticket) == 'DUPLICATE', 'old completion')
                check(m.live['destination'] == current
                      and m.copies[current]['state'] == 'RESERVED',
                      'old completion changed new allocation')
                healthy(m)
                count += 1
    m = Machine()
    s = m.seed(OBJ, 0, 32, 'D', 's')
    d = m.reserve(OBJ, 0, 16, 'H', 'd')
    t = m.start('partial', s, d)
    m.progress('partial', t, 0, 8)
    check(m.copies[d]['state'] == 'LOADING', 'partial progress published')
    m.finish('partial', t)
    check(m.copies[d]['state'] == 'GONE'
          and m.copies[s]['state'] == 'READY', 'retained source lost')
    wrong = Obj(('model', 'different-representation', 1), 0, 32)
    x = m.reserve(wrong, 0, 16, 'H', 'x')
    before = deepcopy(m.__dict__)
    rejects(lambda: m.start('wrong', s, x), 'source')
    check(m.__dict__ == before, 'identity rejection changed state')
    healthy(m)
    return count


def ready_fixture(fault=''):
    m = Machine({'H': 16, 'D': 64}, fault)
    s = m.seed(OBJ, 0, 32, 'D', 's')
    d = m.reserve(OBJ, 0, 16, 'H', 'd', 'A')
    ticket = m.start('copy', s, d, 'A')
    m.progress('copy', ticket, 0, 16)
    m.finish('copy', ticket)
    check(not m.reads and all(o['ended'] for o in m.ops.values()),
          'fixture must have no future copy or reader completion')
    return m, s, d


def queue_waiter(m):
    m.queue('waiter', 'N', [(OBJ, 0, 16, 'H', 'next', 'N')])
    check('waiter' in m.waiting and 'waiter' not in m.admitted,
          'fixture must genuinely block on capacity')


def event_index(events, event, message):
    check(event in events, message)
    return events.index(event)


def assert_idle_cancel(m, s, d, waiter):
    # No gc, END, release or extra wake is invoked by this observer.
    ops_before = deepcopy(m.ops)
    source_before = deepcopy(m.copies[s])
    at = len(m.events)
    m.cancel('A')
    check(m.ops == ops_before and not m.reads and not m.ended_reads,
          'cancel invented or consumed a completion')
    check(m.copies[s] == source_before, 'cancel changed independent cached source')
    check(m.copies[d]['state'] == 'GONE' and m.live.get('d') != d,
          'idle cancel did not return old allocation')
    tail = m.events[at:]
    check(bool(tail) and tail[0] == ('close', 'A'), 'cancel did not close first')
    free_index = event_index(tail, ('free', d), 'cancel omitted reclamation event')
    remove_index = event_index(tail, ('remove', d, 'A'),
                               'cancel omitted owner withdrawal event')
    wake_index = event_index(tail, ('wake',), 'cancel omitted capacity wake')
    check(remove_index < free_index, 'physical free preceded owner withdrawal')
    check(free_index < wake_index, 'wake preceded reclamation')
    if waiter:
        check('waiter' in m.admitted and 'waiter' not in m.waiting,
              'capacity waiter was not advanced')
        admit_index = event_index(tail, ('admit', 'waiter'),
                                  'capacity waiter admission event missing')
        check(wake_index < admit_index, 'admission did not follow wake')
        check(m.bytes('H') == 16, 'waiter reservation charge')
    else:
        check(m.bytes('H') == 0, 'idle cancel retained capacity')
    healthy(m)


def test_cancel_progress():
    for waiter in (False, True):
        m, s, d = ready_fixture()
        if waiter:
            queue_waiter(m)
        assert_idle_cancel(m, s, d, waiter)
        before = deepcopy(m.__dict__)
        check(m.cancel('A') == 'DUPLICATE', 'cancel deduplication')
        check(m.__dict__ == before, 'duplicate cancel mutated state')
        rejects(lambda: m.attach(s, 'A'), 'closed owner')
        rejects(lambda: m.reserve(OBJ, 0, 1, 'H', 'forbidden', 'A'),
                'closed owner')
    m = Machine({'H': 16, 'D': 64})
    d = m.reserve(OBJ, 0, 16, 'H', 'd', 'A')
    queue_waiter(m)
    m.cancel('A')
    check(not m.ops and not m.reads and m.copies[d]['state'] == 'GONE'
          and 'waiter' in m.admitted, 'unstarted cancellation stalled')
    healthy(m)
    m = Machine({'H': 16, 'D': 64})
    keys = m.reserve_many([(OBJ, 0, 8, 'H', 'target', 'A'),
                           (OBJ, 8, 16, 'H', 'workspace', 'A')])
    queue_waiter(m)
    m.queue('own-pending', 'A', [(OBJ, 0, 1, 'H', 'pending', 'A')])
    m.cancel('A')
    check(all(m.copies[k]['state'] == 'GONE' for k in keys),
          'cancel skipped an affected allocation')
    check('waiter' in m.admitted and 'own-pending' not in m.waiting
          and 'own-pending' not in m.admitted, 'cancel wait queue handling')
    tail = m.events[m.events.index(('close', 'A')):]
    check(max(tail.index(('free', k)) for k in keys) < tail.index(('wake',)),
          'cancel woke before completing affected reclamation pass')
    healthy(m)
    for loading in (False, True):
        if loading:
            m = Machine({'H': 16, 'D': 64})
            s = m.seed(OBJ, 0, 32, 'D', 's')
            d = m.reserve(OBJ, 0, 16, 'H', 'd', 'A')
            t = m.start('copy', s, d, 'A')
        else:
            m, s, d = ready_fixture()
            t = m.ops['copy']['ticket']
        m.attach(d, 'B')
        if not loading:
            m.publish('B-view', d, OBJ, 0, 16, 'B')
        queue_waiter(m)
        uses_before = deepcopy(m.copies[d]['uses'])
        m.cancel('A')
        check(m.copies[d]['owners'] == {'B'}
              and m.copies[d]['uses'] == uses_before,
              'cancel removed shared owner or accepted use')
        check(m.copies[d]['state'] == ('LOADING' if loading else 'READY')
              and m.bytes('H') == 16 and 'waiter' in m.waiting,
              'shared allocation incorrectly reclaimed')
        if loading:
            m.progress('copy', t, 0, 16)
            m.finish('copy', t)
            m.publish('B-view', d, OBJ, 0, 16, 'B')
        check(m.acquire('B-view', 'reader-B') == reference(OBJ, 0, 16),
              'surviving owner lost correct data')
        m.cancel('B')
        check(m.copies[d]['state'] == 'RETIRING'
              and 'reader-B' in m.reads and 'waiter' in m.waiting,
              'cancel discarded existing reader responsibility')
        healthy(m)
        m.release('reader-B')
        check(m.copies[d]['state'] == 'GONE' and 'waiter' in m.admitted,
              'last reader did not free and wake')
        healthy(m)
    return 6


def submission_fixture(split=False, fault=''):
    m = SubmissionMachine({'H': 16, 'D': 64}, fault)
    s = m.seed(OBJ, 0, 32, 'D', 's')
    stable_source = deepcopy(m.copies[s])
    m.attach(s, 'A')
    d = m.reserve(OBJ, 0, 8 if split else 16, 'H', 'd', 'A')
    w = m.reserve(OBJ, 0, 8 if split else 4,
                  'H' if split else 'D', 'workspace', 'A')
    t = m.start('rejected', s, d, 'A', plan_id='plan-A',
                workspaces=(w,), plan_holds=(s, d, w))
    check(m.bytes('H') == 16, 'submission fixture must fill device pool')
    check(all(('copy', 'rejected') in m.copies[k]['uses'] for k in (s, d, w))
          and all(m.copies[k]['state'] == 'LOADING' for k in (d, w)),
          'fixture bypassed registration/LOADING')
    check(m.ops['rejected']['phase'] == 'REGISTERED'
          and not m.ops['rejected']['ended'], 'submission already settled')
    healthy(m)
    return m, s, d, w, t, stable_source


def queue_same_slot_waiter(m):
    newer = Obj(('model', 'representation', 2), 0, 32)
    m.queue('waiter', 'N', [(newer, 0, 16, 'H', 'd', 'N')])
    check('waiter' in m.waiting and 'waiter' not in m.admitted,
          'rejection fixture waiter was not blocked')


def assert_unstarted_abandon(m, s, d, w, t, stable_source, waiter):
    # Both correct and faulty entry points are judged by these same external
    # facts. This observer injects no END, READER_END, gc or extra wake.
    before_ids = set(m.ops)
    at = len(m.events)
    result = m.not_started('rejected', t, ('ALL_NOT_STARTED', t), 'abandon')
    check(result == 'FAILED' and set(m.ops) == before_ids
          and not m.reads and not m.ended_reads,
          'unstarted abandonment invented work or completion')
    check(m.ops['rejected']['ended']
          and m.ops['rejected']['outcome'] == 'NOT_STARTED',
          'unstarted result did not settle the registered attempt')
    check(all(not m.copies[k]['uses'] for k in (s, d, w)),
          'unstarted intent responsibility retained')
    check(all(m.copies[k]['state'] == 'GONE' for k in (d, w)),
          'unstarted abandonment retained allocation')
    check(m.copies[s] == stable_source
          and m.memory['s'] == reference(OBJ, 0, 32),
          'unstarted abandonment destroyed stable source')
    check(m.stage_plans['plan-A']['state'] == 'FAILED'
          and not m.stage_plans['plan-A']['open']
          and 'plan-A' not in m.retry_pending, 'failed plan retained entry')
    tail = m.events[at:]
    close_at = event_index(tail, ('close_plan', 'plan-A'), 'plan close missing')
    wake_at = event_index(tail, ('wake',), 'unstarted wake missing')
    for key in (d, w):
        remove_at = event_index(tail, ('remove', key, 'A'), 'plan L removal missing')
        retire_at = event_index(tail, ('retire', key), 'unstarted retirement missing')
        free_at = event_index(tail, ('free', key), 'unstarted free missing')
        check(close_at < remove_at < retire_at < free_at < wake_at,
              'unstarted close/withdraw/retire/free/wake order')
    check(not any(e[0] == 'end' for e in tail), 'NOT_STARTED synthesized END')
    if waiter:
        check('waiter' in m.admitted and 'waiter' not in m.waiting,
              'unstarted abandonment did not advance waiter')
        current = m.admitted['waiter'][0]
        check(current[0] == d[0] and current[1] > d[1]
              and m.live['d'] == current and m.bytes('H') == 16,
              'waiter did not obtain a new allocation lifetime')
        check(wake_at < event_index(tail, ('admit', 'waiter'), 'admit missing'),
              'waiter preceded wake')
    else:
        check(m.bytes('H') == 0, 'unstarted capacity not returned')
    healthy(m)


def test_unstarted_submission():
    cases = 0
    for split in (False, True):
        for waiter in (False, True):
            m, s, d, w, t, stable = submission_fixture(split)
            if waiter:
                queue_same_slot_waiter(m)
            before = deepcopy(m.__dict__)
            rejects(lambda: m.not_started('rejected', t, ('UNPROVEN', t)),
                    'unstarted evidence')
            check(m.__dict__ == before, 'unproven rejection changed resources')
            assert_unstarted_abandon(m, s, d, w, t, stable, waiter)
            after = deepcopy(m.__dict__)
            check(m.not_started('rejected', t, ('ALL_NOT_STARTED', t))
                  == 'DUPLICATE', 'rejection deduplication')
            check(m.__dict__ == after, 'old rejection changed new lifetime')
            rejects(lambda: m.retry_stage('plan-A', 'forbidden'), 'retry plan')
            rejects(lambda: m.accept('rejected', t), 'settled operation')
            cases += 1
    # Abandon only plan A: B keeps the unproduced reservation and a real todo.
    m, s, d, w, t, stable = submission_fixture()
    m.attach(d, 'B')
    queue_same_slot_waiter(m)
    m.not_started('rejected', t, ('ALL_NOT_STARTED', t), 'abandon')
    check(m.copies[d]['owners'] == {'B'}
          and m.copies[d]['state'] == 'RESERVED' and not m.copies[d]['uses']
          and m.replacement_pending[d] == {'B'}
          and m.copies[w]['state'] == 'GONE'
          and m.copies[s] == stable and 'waiter' in m.waiting,
          'rejection discarded shared responsibility')
    healthy(m)
    t2 = m.start('B-copy', s, d, 'B', plan_id='plan-B')
    check(d not in m.replacement_pending, 'replacement todo not consumed')
    m.accept('B-copy', t2)
    m.progress('B-copy', t2, 0, 16)
    m.finish('B-copy', t2)
    m.publish('B-view', d, OBJ, 0, 16, 'B')
    check(m.acquire('B-view', 'B-reader') == reference(OBJ, 0, 16),
          'surviving holder could not produce correct contents')
    m.drop(d, 'B')
    check('waiter' in m.waiting, 'reader protection was ignored')
    m.release('B-reader')
    check('waiter' in m.admitted, 'surviving holder drain did not advance waiter')
    healthy(m)
    cases += 1
    # Retry retains precisely the old reservations/L, but no old U or LOADING.
    m, s, d, w, t, stable = submission_fixture()
    queue_same_slot_waiter(m)
    owners = {k: set(m.copies[k]['owners']) for k in (s, d, w)}
    live = dict(m.live)
    m.not_started('rejected', t, ('ALL_NOT_STARTED', t), 'retry')
    check(m.live == live and m.bytes('H') == 16
          and m.retry_pending == {'plan-A'} and 'waiter' in m.waiting,
          'retry lost reservations or todo')
    check(all(m.copies[k]['owners'] == owners[k] and not m.copies[k]['uses']
              for k in (s, d, w))
          and all(m.copies[k]['state'] == 'RESERVED' for k in (d, w)),
          'retry incorrectly retained active production')
    healthy(m)
    before = deepcopy(m.__dict__)
    rejects(lambda: m.retry_stage('plan-A', 'rejected'), 'operation reuse')
    check(m.__dict__ == before, 'reused operation rejection mutated retry')
    t2 = m.retry_stage('plan-A', 'second')
    check(t2 != t and not m.retry_pending
          and all(('copy', 'second') in m.copies[k]['uses'] for k in (s, d, w)),
          'retry failed to register fresh operation')
    before = deepcopy(m.__dict__)
    check(m.not_started('rejected', t, ('ALL_NOT_STARTED', t)) == 'DUPLICATE',
          'old rejection was not duplicate')
    check(m.__dict__ == before, 'old rejection removed new attempt responsibility')
    m.accept('second', t2)
    m.progress('second', t2, 0, 16)
    m.finish('second', t2)
    check(m.copies[d]['state'] == 'READY'
          and m.copies[w]['state'] == 'GONE' and m.copies[s] == stable,
          'successful retry failed stage settlement')
    healthy(m)
    m.drop(d, 'A')
    check('waiter' in m.admitted, 'retry result retirement did not wake waiter')
    healthy(m)
    cases += 1
    # Cancel while RETRY_WAIT has no future physical completion to wait for.
    m, s, d, w, t, stable = submission_fixture()
    queue_same_slot_waiter(m)
    m.not_started('rejected', t, ('ALL_NOT_STARTED', t), 'retry')
    m.cancel('A')
    check(not m.retry_pending and m.copies[d]['state'] == 'GONE'
          and m.copies[w]['state'] == 'GONE' and m.copies[s] == stable
          and 'waiter' in m.admitted, 'retry cancellation leaked reservations')
    rejects(lambda: m.retry_stage('plan-A', 'third'), 'retry plan')
    healthy(m)
    cases += 1
    # Accepted work is not eligible for the whole-operation unstarted path.
    m, s, d, w, t, stable = submission_fixture()
    m.accept('rejected', t)
    m.progress('rejected', t, 0, 8)
    before = deepcopy(m.__dict__)
    rejects(lambda: m.not_started('rejected', t, ('ALL_NOT_STARTED', t)),
            'not unstarted')
    check(m.__dict__ == before, 'accepted access was incorrectly withdrawn')
    healthy(m)
    cases += 1
    return cases


def sweep_peak(jobs, fixed=0):
    events = {}
    for begin, end, size in jobs:
        require(begin < end and size > 0, 'interval')
        events[begin] = events.get(begin, 0) + size
        events[end] = events.get(end, 0) - size
    used = peak = fixed
    for time in sorted(events):
        used += events[time]
        peak = max(peak, used)
    return peak


def enumerated_peak(jobs, fixed=0):
    return max([fixed] + [fixed + sum(size for a, b, size in jobs
                                     if a <= Fraction(t, 2) < b)
                          for t in range(11)])


def physical_bytes(allocations, views, faulty=False):
    return (sum(allocations[p] for p in views) if faulty
            else sum(allocations.values()))


def pipeline(n, copy_time, compute_time, slots):
    require(n >= 0 and slots >= 1 and copy_time >= 0 and compute_time >= 0,
            'pipeline input')
    copy_end = compute_end = 0
    consumer_ends = []
    for j in range(n):
        reusable_at = consumer_ends[j - slots] if j >= slots else 0
        copy_end = max(copy_end, reusable_at) + copy_time
        compute_end = max(copy_end, compute_end) + compute_time
        consumer_ends.append(compute_end)
    return compute_end


def test_capacity_and_pipeline():
    plans = 0
    for a in range(4):
        for b in range(a + 1, 5):
            for c in range(4):
                for d in range(c + 1, 5):
                    jobs = [(a, b, 3), (c, d, 5)]
                    check(sweep_peak(jobs, 2) == enumerated_peak(jobs, 2),
                          'capacity sweep differs from enumeration')
                    plans += 1
    check(sweep_peak([(0, 2, 30), (2, 4, 30)], 20) == 50,
          'half-open boundary double charged')
    check(sweep_peak([(0, 3, 50), (1, 4, 50)], 20) == 120,
          'overlap or fixed capacity omitted')
    cases = 0
    for n in range(6):
        for tm in (0, 1, 3):
            for tc in (0, 2, 4):
                for slots in (1, 2, 3):
                    expected = (0 if n == 0 else n * (tm + tc) if slots == 1
                                else tm + tc + (n - 1) * max(tm, tc))
                    check(pipeline(n, tm, tc, slots) == expected,
                          'finite pipeline differs from closed-form reference')
                    cases += 1
    return plans, cases


def test_negative_controls():
    m = Machine(fault='early_publish')
    d = m.reserve(OBJ, 0, 16, 'H', 'd')
    m.publish('v', d, OBJ, 0, 16, 'build')
    check('unready view' in audit(m), 'early-publication mutant escaped')
    m = Machine(fault='early_free')
    s = m.seed(OBJ, 0, 32, 'D', 's')
    d = m.reserve(OBJ, 0, 16, 'H', 'd')
    t = m.start('old-copy', s, d)
    m.drop(d, 'build')
    newer = Obj(('model', 'representation', 2), 0, 32)
    current = m.reserve(newer, 0, 16, 'H', 'd', 'new')
    check(m.live['d'] == current, 'mutant did not actually reuse slot')
    check('inflight allocation missing' in audit(m), 'early-free mutant escaped')
    rejects(lambda: m.progress('old-copy', t, 0, 16), 'physical lifetime')
    allocations = {'shared': 100, 'private': 40}
    views = ['shared', 'shared', 'private']
    expected = len({('shared', i) for i in range(100)}
                   | {('private', i) for i in range(40)})
    check(physical_bytes(allocations, views) == expected == 140,
          'physical byte reference')
    check(physical_bytes(allocations, views, True) == 240 != expected,
          'view-count mutant escaped')
    for fault in ('cancel_no_gc', 'cancel_no_wake'):
        m, s, d = ready_fixture(fault)
        queue_waiter(m)
        try:
            assert_idle_cancel(m, s, d, True)
        except CheckError as exc:
            expected_reason = ('idle cancel did not return old allocation'
                               if fault == 'cancel_no_gc'
                               else 'cancel omitted capacity wake')
            check(str(exc) == expected_reason,
                  'cancellation mutant failed for unexpected reason: ' + str(exc))
        else:
            raise CheckError('cancellation mutant escaped: ' + fault)
        check('A' in m.closed and not m.copies[d]['owners'],
              'mutant failed before cancellation withdrawal')
        check(not m.reads and all(o['ended'] for o in m.ops.values()),
              'mutant depended on future completion')
        if fault == 'cancel_no_gc':
            check(m.copies[d]['state'] == 'RETIRING'
                  and m.live.get('d') == d and m.bytes('H') == 16
                  and 'waiter' in m.waiting and 'waiter' not in m.admitted
                  and 'idle retired allocation retained' in audit(m),
                  'missing-gc mutant not distinguished')
        else:
            check(m.copies[d]['state'] == 'GONE' and m.live.get('d') != d
                  and m.bytes('H') == 0
                  and 'waiter' in m.waiting and 'waiter' not in m.admitted,
                  'missing-wake mutant not distinguished')
            healthy(m)
    # Only omission: no transition into RETIRING after NOT_STARTED withdrawal.
    # gc and wake still execute. No helper supplies retirement implicitly.
    m, s, d, w, t, stable = submission_fixture(fault='unstarted_no_retire')
    queue_same_slot_waiter(m)
    try:
        assert_unstarted_abandon(m, s, d, w, t, stable, True)
    except CheckError as exc:
        check(str(exc) == 'unstarted abandonment retained allocation',
              'unstarted mutant failed for unexpected reason: ' + str(exc))
    else:
        raise CheckError('missing unstarted retirement mutant escaped')
    check(all(m.copies[k]['state'] == 'LOADING'
              and not m.copies[k]['owners'] and not m.copies[k]['uses']
              for k in (d, w)), 'mutant did not actually leave orphan LOADING')
    check(m.live['d'] == d and m.bytes('H') == 16
          and 'waiter' in m.waiting and 'waiter' not in m.admitted
          and m.ops['rejected']['ended']
          and m.ops['rejected']['outcome'] == 'NOT_STARTED'
          and m.copies[s] == stable
          and 'unowned loading allocation retained' in audit(m),
          'missing-retirement capacity failure not observed')
    return 6


def test_costs():
    gib, mib = 1024 ** 3, 1024 ** 2
    h2d, ssd = 16 * gib, 2 * gib
    check(Fraction(10 * gib * 1000, h2d) == 625, '625 ms transfer term')
    restore = Fraction(512 * mib * 1000, h2d)
    check(restore == Fraction(125, 4), '31.25 ms restore term')
    check(Fraction(512 * mib * 1000, ssd) == 250, '250 ms SSD term')
    check(20 < restore < 200, 'restore/recompute reversal')
    per_position = sum(2 for layer in range(32) for kv in range(2)
                       for head in range(8) for dim in range(128))
    check(per_position == 128 * 1024, 'KV coordinate byte enumeration')
    check(Fraction(128 * 160 * per_position, gib) == Fraction(5, 2),
          '2.5 GiB KV main term')
    layer_copy = Fraction(10 * gib * 1000, 32 * h2d)
    check(layer_copy == Fraction(625, 32), 'layer copy term')
    for compute in (Fraction(2), Fraction(30)):
        expected = layer_copy + compute + 31 * max(layer_copy, compute)
        check(pipeline(32, layer_copy, compute, 2) == expected,
              'fractional finite pipeline')
    check(Fraction(128, 1) / Fraction(625, 1000) == Fraction(1024, 5),
          '204.8 token/s transfer-only bound')


def successful_aux_fixture(temporary_source=True, extra_reader=False, workspace_owner=False,
                           fault=''):
    m = SubmissionMachine({'S': 16, 'D': 16, 'W': 8}, fault)
    source = m.seed(OBJ, 0, 16, 'S', 'source')
    m.attach(source, 'A')
    if temporary_source:
        m.drop(source, 'cache')
    destination = m.reserve(OBJ, 0, 16, 'D', 'destination', 'A')
    workspace = m.reserve(OBJ, 0, 8, 'W', 'workspace', 'A')
    ticket = m.start('success', source, destination, 'A', plan_id='successful-stage',
                     workspaces=(workspace,), plan_holds=(source, destination, workspace))
    if extra_reader:
        m.publish('source-view', source, OBJ, 0, 16, 'A')
        m.acquire('source-view', 'old-reader')
    if workspace_owner:
        m.attach(workspace, 'B')
    newer = Obj(('model', 'representation', 2), 0, 32)
    specs = [(newer, 0, 8, 'W', 'workspace', 'N')]
    if temporary_source:
        specs.append((newer, 0, 16, 'S', 'source', 'N'))
    m.queue('aux-waiter', 'N', specs)
    check('aux-waiter' in m.waiting, 'auxiliary waiter must initially be blocked')
    m.accept('success', ticket)
    m.progress('success', ticket, 0, 16)
    healthy(m)
    return m, source, destination, workspace, ticket


def assert_successful_aux_settlement(m, source, destination, workspace, ticket,
                                     temporary_source=True):
    # The observer calls exactly one genuine END and adds no rescue gc/wake/end.
    at = len(m.events)
    ids = set(m.ops)
    check(m.finish('success', ticket) == 'READY', 'successful result not ready')
    check(set(m.ops) == ids and not m.reads and not m.ended_reads,
          'successful settlement invented another operation or reader end')
    check(m.copies[destination]['owners'] == {'A'}
          and m.copies[destination]['state'] == 'READY'
          and m.memory['destination'] == reference(OBJ, 0, 16),
          'auxiliary cleanup damaged the successful destination')
    check(m.copies[workspace]['state'] == 'GONE',
          'successful END retained the auxiliary workspace allocation')
    if temporary_source:
        check(m.copies[source]['state'] == 'GONE',
              'successful END retained the ownerless temporary source')
    else:
        check(m.copies[source]['state'] == 'READY'
              and m.copies[source]['owners'] == {'cache'}
              and m.memory['source'] == reference(OBJ, 0, 16),
              'successful cleanup removed an independent cached source')
    check('aux-waiter' in m.admitted and 'aux-waiter' not in m.waiting,
          'successful END did not advance the auxiliary capacity waiter')
    tail = m.events[at:]
    close_at = event_index(tail, ('close_plan', 'successful-stage'), 'stage close missing')
    wake_at = event_index(tail, ('wake',), 'END wake missing')
    check(sum(event == ('wake',) for event in tail) == 1,
          'END woke before completing all auxiliary cleanup')
    targets = (source, workspace) if temporary_source else (workspace,)
    for key in targets:
        removal = event_index(tail, ('remove', key, 'A'), 'auxiliary L removal missing')
        retired = event_index(tail, ('retire', key), 'auxiliary retirement missing')
        freed = event_index(tail, ('free', key), 'auxiliary free missing')
        check(close_at < removal < retired < freed < wake_at,
              'END close/withdraw/retire/free/wake order')
    check(wake_at < event_index(tail, ('admit', 'aux-waiter'), 'auxiliary admission missing'),
          'auxiliary admission preceded cleanup wake')
    healthy(m)


def test_successful_auxiliary_retirement():
    for temporary_source in (False, True):
        m, source, destination, workspace, ticket = successful_aux_fixture(temporary_source)
        assert_successful_aux_settlement(m, source, destination, workspace, ticket,
                                         temporary_source)
        after = deepcopy(m.__dict__)
        check(m.finish('success', ticket) == 'DUPLICATE', 'successful END deduplication')
        check(m.__dict__ == after, 'duplicate END touched new auxiliary allocations')
    bad, source, destination, workspace, ticket = successful_aux_fixture(fault='end_aux_no_retire')
    try:
        assert_successful_aux_settlement(bad, source, destination, workspace, ticket)
    except CheckError as exc:
        check(str(exc) == 'successful END retained the auxiliary workspace allocation',
              'unexpected failure in missing-auxiliary-retirement control')
    else:
        raise CheckError('missing END auxiliary retirement escaped')
    check(not bad.copies[workspace]['owners'] and not bad.copies[workspace]['uses']
          and bad.copies[workspace]['state'] == 'LOADING'
          and 'aux-waiter' in bad.waiting,
          'auxiliary mutant was silently retired or rescued')
    check(('remove', workspace, 'A') in bad.events
          and ('retire', workspace) not in bad.events,
          'negative control did not isolate raw withdrawal from retirement')
    print('negative_control_detected: successful END raw L withdrawal without auxiliary retirement')
    # U remains after the stage owner leaves: retire immediately, but await the
    # already registered reader's real end before physical reclamation.
    m, source, destination, workspace, ticket = successful_aux_fixture(extra_reader=True)
    m.finish('success', ticket)
    check(m.copies[source]['state'] == 'RETIRING' and not m.copies[source]['owners']
          and m.copies[source]['uses'] == {('reader', 'old-reader')}
          and m.copies[workspace]['state'] == 'GONE'
          and 'aux-waiter' in m.waiting, 'existing reader protection lost')
    healthy(m)
    m.release('old-reader')
    check(m.copies[source]['state'] == 'GONE' and 'aux-waiter' in m.admitted,
          'last genuine reader failed to reclaim and wake')
    healthy(m)
    # Another L remains: retain allocation and make the ended workspace a real
    # RESERVED handoff, rather than claiming its old producer is still active.
    m, source, destination, workspace, ticket = successful_aux_fixture(workspace_owner=True)
    m.finish('success', ticket)
    check(m.copies[workspace]['owners'] == {'B'}
          and m.copies[workspace]['state'] == 'RESERVED'
          and m.replacement_pending[workspace] == {'B'}
          and not m.copies[workspace]['uses'] and 'aux-waiter' in m.waiting,
          'remaining workspace holder was discarded or left LOADING')
    healthy(m)
    m.drop(workspace, 'B')
    check('aux-waiter' in m.admitted and workspace not in m.replacement_pending,
          'remaining owner retirement did not wake or retained a stale handoff')
    healthy(m)
    print('observed successful END auxiliaries: private workspace and temporary source retire/free/wake in one entry; cached L and existing U preserved; duplicate END cannot affect new epochs')


def main():
    ranges = test_ranges_and_reservation()
    traces = test_event_traces()
    cancellations = test_cancel_progress()
    submissions = test_unstarted_submission()
    test_successful_auxiliary_retirement()
    plans, pipelines = test_capacity_and_pipeline()
    mutants = test_negative_controls()
    test_costs()
    print('range decisions:', ranges, '; legal subranges and rejection immutability')
    print('event trajectories:', traces, '; partial, failure, identity and lifetime edges')
    print('dedicated cancellation scenarios:', cancellations,
          '; no-future-event reclamation, wake, full affected set and sharing')
    print('registered submission scenarios:', submissions,
          '; LOADING then NOT_STARTED, workspace, sharing, retry, dedup and cancellation')
    print('capacity plans:', plans, '; finite pipeline cases:', pipelines)
    print('legacy detected negative controls:', mutants,
          '; early publication, early free, shared-view count, missing cancel gc, missing wake, missing rejection retirement')
    print('scope: deterministic CPU model; no GPU, SSD, PCIe, CUDA or performance measurements')
    print('limits: contiguous logical units; serial control; trusted NOT_STARTED input; explicit retry scheduling; no real forward, compression, persistence or unknown-submit recovery')
    print('check_k_alg_017_full: PASS')


if __name__ == '__main__':
    main()
