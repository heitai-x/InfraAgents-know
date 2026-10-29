#!/usr/bin/env python3
"""K-SYS-015: deterministic, offline CPU symbolic checks; stdlib only.

All fixtures are embedded. No file, network, project or subprocess access.
One buffer denotes one fixed request attempt/storage slot, with explicit epochs.
Certificates explicitly state whether the whole registered access has ended.
"""
from copy import deepcopy
from fractions import Fraction
from itertools import permutations, product
import json


class Violation(Exception):
    pass


def need(condition, reason):
    if not condition:
        raise Violation(reason)


def rejected(check, expected):
    try:
        check()
    except Violation as exc:
        need(expected in str(exc), 'WRONG_REJECTION:' + str(exc))
        return str(exc)
    raise Violation('NEGATIVE_CONTROL_ESCAPED:' + expected)


class BufferModel:
    """Reducer under test, including a mutation that forgets a read grant."""
    def __init__(self, omit_read_settlement=False):
        self.s = dict(phase='FREE', epoch=0, cap=0, n=0, owners=set(),
                      uses={}, readers=set(), writer=None, ready=False,
                      eligible=False, quarantine=set(), entry_credit=(0, 0))
        self.omit_read_settlement = omit_read_settlement
        self.trace = []

    def _settle_read(self, uid):
        # One state-owner transaction: both removals precede its snapshot.
        del self.s['uses'][uid]
        if not self.omit_read_settlement:
            self.s['readers'].remove(uid)
        if self.s['phase'] == 'READING' and not self.s['readers']:
            self.s['phase'] = 'PUBLISHED'
        # In particular, RETIRING and eligible are not changed here.

    def _on_done(self, a):
        s = self.s
        uid = a['uid']
        if a['epoch'] != s['epoch'] or uid not in s['uses']:
            return
        if not a['ended']:
            s['quarantine'].add(uid)
            return
        mode = s['uses'][uid]
        if mode == 'R':
            self._settle_read(uid)
        elif mode == 'W':
            del s['uses'][uid]
            s['writer'] = None
            s['ready'], s['n'] = a['success'], a['n']
        else:
            raise ValueError(mode)
        s['quarantine'].discard(uid)

    def event(self, op, **a):
        s = self.s
        if op == 'reserve':
            s.update(phase='RESERVED', cap=a['cap'], n=0,
                     owners={'producer'}, ready=False, eligible=True,
                     entry_credit=(1, a['cap']))
        elif op == 'begin':
            s['phase'] = 'FILLING'
            s['writer'] = a['uid']
            s['uses'][a['uid']] = 'W'
        elif op == 'submit':
            pass
        elif op == 'publish':
            s['phase'] = 'PUBLISHED'
        elif op == 'read':
            s['phase'] = 'READING'
            s['uses'][a['uid']] = 'R'
            s['readers'].add(a['uid'])
        elif op == 'done':
            self._on_done(a)  # Unified completion entry; explicit mode dispatch.
        elif op == 'release':
            # READ adapter only; it cannot finish a registered WRITE.
            if a['epoch'] == s['epoch'] and s['uses'].get(a['uid']) == 'R':
                self._on_done(a)
        elif op == 'timeout':
            if a['epoch'] == s['epoch'] and a['uid'] in s['uses']:
                s['quarantine'].add(a['uid'])
        elif op in ('retire', 'cancel'):
            s['phase'] = 'RETIRING'
            s['owners'].clear()
            if op == 'cancel':
                s['eligible'] = False
        elif op == 'recycle':
            # The auditor must reject this operation if historical debt exists,
            # even when an incorrect reducer erases its local records.
            s.update(phase='FREE', epoch=s['epoch'] + 1, cap=0, n=0,
                     owners=set(), uses={}, readers=set(), writer=None,
                     ready=False, eligible=False, quarantine=set(),
                     entry_credit=(0, 0))
        else:
            raise ValueError(op)
        self.trace.append((op, deepcopy(a), deepcopy(s)))
        return self

    def certificate(self, uid, epoch=0, ended=True, success=True, n=0, via='done'):
        need(via in ('done', 'release'), 'BAD_CERTIFICATE_ENTRY')
        return self.event(via, uid=uid, epoch=epoch, ended=ended,
                          success=success, n=n)

    def can_recycle(self):
        s = self.s
        return (s['phase'] == 'RETIRING' and not s['owners'] and
                not s['uses'] and not s['readers'] and
                not s['quarantine'] and s['writer'] is None)


def audit_buffer(trace):
    """Independent history reference, not the reducer's completion helper.

issued remembers every registration. receipts records whole-access evidence.
The expected read grants are independently issued READs minus full receipts.
A duplicate receipt never pays another use's debt.
"""
    phase, epoch, cap, n = 'FREE', 0, 0, 0
    owners, issued, receipts, quarantine = set(), {}, set(), set()
    ready, eligible = False, False
    for op, a, snapshot in trace:
        key = (a.get('epoch', epoch), a.get('uid'))
        pending = {k: mode for k, mode in issued.items()
                   if k[0] == epoch and k not in receipts}
        if op == 'reserve':
            need(phase == 'FREE' and not pending and not owners, 'RESERVE_LIVE')
            need(a['cap'] >= 0, 'NEGATIVE_CAPACITY')
            cap, n, phase = a['cap'], 0, 'RESERVED'
            owners, ready, eligible = {'producer'}, False, True
        elif op == 'begin':
            need(phase == 'RESERVED' and eligible, 'BEGIN_REQUIRES_RESERVED')
            need(key not in issued and not pending, 'DUPLICATE_OR_CONFLICTING_WRITE')
            issued[key], phase = 'W', 'FILLING'
        elif op == 'submit':
            need(phase == 'FILLING' and pending.get(key) == 'W', 'SUBMIT_WITHOUT_WRITE')
        elif op == 'publish':
            need(phase == 'FILLING', 'PUBLISH_REQUIRES_FILLING')
            need(eligible and ready and 'W' not in pending.values(),
                 'PUBLISH_WITHOUT_CERTIFICATE')
            phase = 'PUBLISHED'
        elif op == 'read':
            need(phase in ('PUBLISHED', 'READING') and eligible and ready,
                 'READ_BEFORE_READY')
            need(key not in issued and 'W' not in pending.values(), 'READ_CONFLICT')
            issued[key], phase = 'R', 'READING'
        elif op == 'done':
            if key in pending:
                if a['ended']:
                    receipts.add(key)
                    quarantine.discard(key)
                    if issued[key] == 'W':
                        ready, n = a['success'], a['n']
                    elif phase == 'READING':
                        remaining_reads = [k for k, mode in issued.items()
                                           if k[0] == epoch and mode == 'R'
                                           and k not in receipts]
                        if not remaining_reads:
                            phase = 'PUBLISHED'
                else:
                    quarantine.add(key)
        elif op == 'release':
            # Independently interpret the READ-only adapter's contract.
            if pending.get(key) == 'R':
                if not a['ended']:
                    quarantine.add(key)
                else:
                    receipts.add(key)
                    quarantine.discard(key)
                    if phase == 'READING' and not any(
                            k[0] == epoch and mode == 'R' and k not in receipts
                            for k, mode in issued.items()):
                        phase = 'PUBLISHED'
        elif op == 'timeout':
            if key in pending:
                quarantine.add(key)
        elif op in ('retire', 'cancel'):
            need(phase != 'FREE', 'RETIRE_FREE')
            phase, owners = 'RETIRING', set()
            if op == 'cancel':
                eligible = False
        elif op == 'recycle':
            need(phase == 'RETIRING', 'RECYCLE_REQUIRES_RETIRING')
            need(not pending and not owners and not quarantine, 'EARLY_REUSE')
            epoch += 1
            phase, cap, n, ready, eligible = 'FREE', 0, 0, False, False
        else:
            raise ValueError(op)
        need(0 <= n <= cap, 'WRITE_CAPACITY')
        current = {uid: mode for (generation, uid), mode in issued.items()
                   if generation == epoch and (generation, uid) not in receipts}
        expected_reads = {uid for uid, mode in current.items() if mode == 'R'}
        need(not (snapshot['readers'] - expected_reads),
             'READ_AUTHORIZATION_RESIDUE:' + op)
        need(not (expected_reads - snapshot['readers']),
             'READ_AUTHORIZATION_MISSING:' + op)
        writers = [uid for uid, mode in current.items() if mode == 'W']
        need(len(writers) <= 1, 'MULTIPLE_WRITERS')
        expected = dict(phase=phase, epoch=epoch, cap=cap, n=n,
                        owners=owners, uses=current, readers=expected_reads,
                        writer=writers[0] if writers else None,
                        ready=ready, eligible=eligible,
                        quarantine={uid for generation, uid in quarantine
                                    if generation == epoch},
                        entry_credit=(0, 0) if phase == 'FREE' else (1, cap))
        need(snapshot == expected, 'LIFECYCLE_SNAPSHOT_MISMATCH:' + op)
    need(bool(trace), 'EMPTY_BUFFER_TRACE')
    return dict(epoch=epoch, phase=phase,
                outstanding=sum(1 for k in issued if k[0] == epoch and k not in receipts),
                readers=len(expected_reads), entry_credit=list(expected['entry_credit']))


def produced(omit_read_settlement=False):
    b = BufferModel(omit_read_settlement)
    b.event('reserve', cap=4).event('begin', uid='w')
    b.certificate('w', n=3).event('publish')
    return b


def check_buffers():
    # Synchronous input completion enters done, without a second release event.
    synchronous = produced().event('read', uid='input')
    synchronous.certificate('input')
    audit_buffer(synchronous.trace)
    need(synchronous.s['phase'] == 'PUBLISHED' and not synchronous.s['readers']
         and not synchronous.s['uses'], 'SYNC_DONE_DID_NOT_SETTLE_READ')
    need(synchronous.s['entry_credit'] == (1, 4), 'SYNC_DONE_DROPPED_OWNER_CREDIT')
    saved = deepcopy(synchronous.s)
    synchronous.certificate('input').certificate('input', via='release')
    need(saved == synchronous.s, 'DUPLICATE_READ_CHANGED_STATE')
    synchronous.event('retire')
    need(synchronous.can_recycle(), 'SYNC_INPUT_CANNOT_RECYCLE')
    synchronous.event('recycle')
    sync_result = audit_buffer(synchronous.trace)

    # Cancellation, partial evidence and two independent readers.
    late = produced().event('read', uid='r1').event('read', uid='r2').event('cancel')
    late.event('timeout', uid='r1', epoch=0).event('timeout', uid='r2', epoch=0)
    late.certificate('r1', ended=False)
    audit_buffer(late.trace)
    need(late.s['uses'] == {'r1': 'R', 'r2': 'R'} and
         late.s['readers'] == {'r1', 'r2'}, 'PARTIAL_READ_RELEASED_RESPONSIBILITY')
    incomplete = deepcopy(late).event('recycle')
    errors = [rejected(lambda: audit_buffer(incomplete.trace), 'EARLY_REUSE')]
    saved = deepcopy(late.s)
    late.certificate('unknown').certificate('r1', epoch=9)
    need(saved == late.s, 'UNKNOWN_CERTIFICATE_CHANGED_STATE')
    late.certificate('r1')
    audit_buffer(late.trace)
    need(late.s['phase'] == 'RETIRING' and not late.s['eligible'], 'READ_REVIVED_CANCELLED_INPUT')
    need(late.s['readers'] == {'r2'} and late.s['uses'] == {'r2': 'R'}
         and late.s['quarantine'] == {'r2'}, 'READ_COMPLETION_PAID_OTHER_READER')
    saved = deepcopy(late.s)
    late.certificate('r1').certificate('r1', via='release')
    need(saved == late.s, 'DUPLICATE_PAID_ANOTHER_READER')
    late.certificate('r2', success=False)
    need(late.can_recycle(), 'LATE_INPUT_DONE_CANNOT_RECYCLE')
    late.event('recycle')
    late_result = audit_buffer(late.trace)
    need(late.s['entry_credit'] == (0, 0), 'LATE_READ_LEAKED_ENTRY_CREDIT')

    # Reuse uid in a new allocation epoch; an old certificate cannot settle it.
    late.event('reserve', cap=4).event('begin', uid='w')
    late.certificate('w', epoch=1, n=4).event('publish').event('read', uid='r1')
    saved = deepcopy(late.s)
    late.certificate('r1', epoch=0)
    need(saved == late.s, 'OLD_READ_CERTIFICATE_HIT_NEW_EPOCH')
    late.event('retire').certificate('r1', epoch=1).event('recycle')
    audit_buffer(late.trace)

    # Enumerate read completion order, normal/cancelled retirement and adapters.
    enumeration = 0
    for cancelled in (False, True):
        for order in permutations(('r1', 'r2')):
            for routes in product(('done', 'release'), repeat=2):
                b = produced().event('read', uid='r1').event('read', uid='r2')
                b.event('cancel' if cancelled else 'retire')
                for uid, route in zip(order, routes):
                    b.certificate(uid, ended=False, via=route)
                    audit_buffer(b.trace)
                    b.certificate(uid, via=route)
                    snapshot = deepcopy(b.s)
                    b.certificate(uid, via='done').certificate(uid, via='release')
                    need(snapshot == b.s, 'ENUM_DUPLICATE_CHANGED_STATE')
                    audit_buffer(b.trace)
                need(b.can_recycle(), 'ENUM_ALL_READS_ENDED_BUT_NOT_RECYCLABLE')
                need(b.s['phase'] == 'RETIRING', 'RETIRING_STATE_LOST')
                b.event('recycle')
                audit_buffer(b.trace)
                enumeration += 1

    # Required negative control: real done execution drops U but leaves R.
    # The exact same auditor must reject the mutated snapshot immediately.
    residue_observations = []
    for cancelled in (False, True):
        faulty = produced(True).event('read', uid='input')
        if cancelled:
            faulty.event('cancel')
        faulty.certificate('input')
        if not cancelled:
            faulty.event('retire')
        need(not faulty.s['uses'] and faulty.s['readers'] == {'input'},
             'READ_RESIDUE_MUTATION_NOT_TRIGGERED')
        need(not faulty.can_recycle() and faulty.s['entry_credit'] == (1, 4),
             'READ_RESIDUE_DID_NOT_BLOCK_RECYCLE')
        faulty.certificate('input').certificate('input', via='release')
        reason = rejected(lambda: audit_buffer(faulty.trace), 'READ_AUTHORIZATION_RESIDUE:done')
        errors.append(reason)
        residue_observations.append(dict(cancelled=cancelled, uses=[], readers=['input'],
                                         recyclable=faulty.can_recycle(),
                                         entry_credit=list(faulty.s['entry_credit']),
                                         rejection=reason))

    delayed = BufferModel().event('reserve', cap=4).event('begin', uid='dma')
    delayed.event('submit', uid='dma')
    # A read-release adapter cannot accidentally complete the output WRITE.
    delayed.certificate('dma', via='release', n=4)
    audit_buffer(delayed.trace)
    need(delayed.s['phase'] == 'FILLING' and delayed.s['writer'] == 'dma',
         'SUBMISSION_OR_READ_ADAPTER_FINISHED_WRITE')
    delayed.certificate('dma', n=4).event('publish').event('retire').event('recycle')
    audit_buffer(delayed.trace)

    cancelled_write = BufferModel().event('reserve', cap=4).event('begin', uid='dma')
    cancelled_write.event('submit', uid='dma').event('cancel')
    cancelled_write.event('timeout', uid='dma', epoch=0).certificate('dma', ended=False)
    audit_buffer(cancelled_write.trace)
    bad = deepcopy(cancelled_write).event('recycle')
    errors.append(rejected(lambda: audit_buffer(bad.trace), 'EARLY_REUSE'))
    cancelled_write.certificate('dma', n=4).certificate('dma', n=4).event('recycle')
    audit_buffer(cancelled_write.trace)

    early = produced().event('read', uid='reader').event('retire').event('recycle')
    errors.append(rejected(lambda: audit_buffer(early.trace), 'EARLY_REUSE'))
    premature = BufferModel().event('reserve', cap=4).event('begin', uid='w')
    premature.event('submit', uid='w').event('publish')
    errors.append(rejected(lambda: audit_buffer(premature.trace), 'PUBLISH_WITHOUT_CERTIFICATE'))
    missing = BufferModel().event('reserve', cap=4).event('publish')
    errors.append(rejected(lambda: audit_buffer(missing.trace), 'PUBLISH_REQUIRES_FILLING'))
    failed = BufferModel().event('reserve', cap=4).event('begin', uid='w')
    failed.certificate('w', success=False, n=2).event('publish')
    errors.append(rejected(lambda: audit_buffer(failed.trace), 'PUBLISH_WITHOUT_CERTIFICATE'))
    empty = BufferModel().event('reserve', cap=0).event('begin', uid='empty')
    empty.certificate('empty', n=0).event('publish').event('read', uid='empty_read')
    empty.certificate('empty_read').event('retire').event('recycle')
    audit_buffer(empty.trace)
    return dict(synchronous=sync_result, cancelled_input=late_result,
                completion_combinations=enumeration, read_residue_mutations=residue_observations,
                negative_reasons=errors)


class Pipeline:
    """A->B capacity/ownership model; accesses have ended before handoff."""
    def __init__(self):
        self.limits = {'A': (1, 4), 'B': (1, 4), 'O': (1, 4)}
        self.entries = {'A': [], 'B': []}
        self.private = []
        self.objects = {}
        self.accounts = {key: [0, 0] for key in self.limits}
        self.physical = 0
        self.control = 0
        self.trace = []

    def record(self, event):
        self.trace.append(deepcopy(dict(event=event, entries=self.entries,
            private=self.private, objects=self.objects, accounts=self.accounts,
            physical=self.physical, control=self.control)))

    def fits(self, key, cap):
        count, size = self.accounts[key]
        max_count, max_size = self.limits[key]
        return count + 1 <= max_count and size + cap <= max_size

    def charge(self, key, cap, sign):
        self.accounts[key][0] += sign
        self.accounts[key][1] += sign * cap

    def admit(self, stage, oid, cap):
        need(cap >= 0 and oid not in self.objects, 'BAD_ADMISSION')
        if not self.fits(stage, cap) or self.physical + cap > 12:
            self.record('admit_reject')
            return False
        self.entries[stage].append(oid)
        self.objects[oid] = dict(cap=cap, n=cap, phase='PUBLISHED')
        self.charge(stage, cap, 1)
        self.physical += cap
        self.record('admit_ready_input')
        return True

    def reserve_output(self, cap, faulty_account=False):
        need('a' in self.entries['A'] and 'o' not in self.objects, 'OUTPUT_WITHOUT_INPUT')
        if cap < 0 or cap > 4 or not self.fits('O', cap) or self.physical + cap > 12:
            self.record('output_reject')
            return False
        self.objects['o'] = dict(cap=cap, n=0, phase='RESERVED')
        self.private.append('o')
        self.charge('O', cap, 1)
        if faulty_account:
            self.accounts['O'][1] -= cap
        self.physical += cap
        self.record('reserve_output')
        return True

    def begin(self):
        self.objects['o']['phase'] = 'FILLING'
        self.record('begin_output')

    def produce(self, n):
        o = self.objects['o']
        if n < 0 or n > o['cap']:
            self.record('write_rejected_before_access')
            return False
        o.update(n=n, phase='PUBLISHED')
        self.record('production_done_and_publish')
        return True

    def remove(self, key, oid):
        cap = self.objects[oid]['cap']
        if key == 'O':
            self.private.remove(oid)
        else:
            self.entries[key].remove(oid)
        self.charge(key, cap, -1)
        self.physical -= cap
        del self.objects[oid]

    def handoff(self, unsafe=False):
        need('o' in self.private and self.objects['o']['phase'] == 'PUBLISHED',
             'HANDOFF_BEFORE_PUBLISH')
        cap = self.objects['o']['cap']
        if not unsafe and not self.fits('B', cap):
            self.record('WAIT_HANDOFF')
            return False
        self.charge('B', cap, 1)
        self.entries['B'].append('o')
        self.private.remove('o')
        self.charge('O', cap, -1)
        self.remove('A', 'a')
        self.record('handoff_commit')
        return True

    def control_tick(self):
        self.control += 1
        self.record('control_serviced')

    def cancel(self):
        self.control += 1
        if 'o' in self.private:
            self.remove('O', 'o')
        if 'a' in self.entries['A']:
            self.remove('A', 'a')
        self.record('cancel_or_failed_access_ended')

    def consume(self, oid):
        self.remove('B', oid)
        self.record('consumer_last_use_and_release')


def audit_pipeline(trace):
    limits = {'A': (1, 4), 'B': (1, 4), 'O': (1, 4)}
    peak = 0
    for s in trace:
        groups = dict(s['entries'])
        groups['O'] = s['private']
        all_ids = [oid for ids in groups.values() for oid in ids]
        need(len(all_ids) == len(set(all_ids)), 'MULTIPLE_LOGICAL_LOCATIONS')
        need(set(all_ids) == set(s['objects']), 'ORPHANED_STORAGE')
        for key, ids in groups.items():
            actual = (len(ids), sum(s['objects'][oid]['cap'] for oid in ids))
            bound = limits[key]
            need(actual[0] <= bound[0] and actual[1] <= bound[1], 'STAGE_CAPACITY:' + key)
            need(tuple(s['accounts'][key]) == actual, 'LEDGER_MISMATCH:' + key)
        physical = sum(obj['cap'] for obj in s['objects'].values())
        need(physical == s['physical'] and physical <= 12, 'PHYSICAL_CAPACITY')
        for key, ids in groups.items():
            for oid in ids:
                obj = s['objects'][oid]
                need(0 <= obj['n'] <= obj['cap'], 'OUTPUT_CAPACITY')
                if key != 'O':
                    need(obj['phase'] == 'PUBLISHED', 'ENTRY_NOT_READABLE')
        peak = max(peak, physical)
    return dict(peak_bytes=peak, final_bytes=trace[-1]['physical'])


def full_pipeline(output_cap=4):
    p = Pipeline()
    need(p.admit('A', 'a', 4) and p.admit('B', 'b', 4), 'FIXTURE_ADMISSION')
    need(p.reserve_output(output_cap), 'FIXTURE_OUTPUT_RESERVATION')
    p.begin()
    need(p.produce(output_cap), 'FIXTURE_PRODUCTION')
    return p


def check_handoff():
    p = full_pipeline()
    need(not p.handoff(), 'FULL_DOWNSTREAM_MUST_WAIT')
    before = deepcopy((p.entries, p.private, p.accounts, p.physical))
    need(not p.handoff(), 'REPEATED_WAIT_RESERVED_AGAIN')
    need(before == (p.entries, p.private, p.accounts, p.physical), 'WAIT_CHANGED_OWNERSHIP')
    p.control_tick()
    need(p.control == 1, 'CONTROL_NOT_REACHABLE')
    p.consume('b')
    need(p.handoff(), 'HANDOFF_AFTER_CREDIT_RELEASE')
    p.consume('o')
    normal = audit_pipeline(p.trace)
    need(p.physical == 0 and all(v == [0, 0] for v in p.accounts.values()), 'NORMAL_LEAK')

    cancelled = full_pipeline()
    need(not cancelled.handoff(), 'CANCEL_FIXTURE_MUST_WAIT')
    cancelled.cancel()
    need(cancelled.entries['B'] == ['b'] and cancelled.physical == 4, 'CANCEL_HARMED_NEIGHBOR')
    cancelled.consume('b')
    audit_pipeline(cancelled.trace)
    need(cancelled.physical == 0, 'CANCEL_LEAK')

    failed = Pipeline()
    need(failed.admit('A', 'a', 4), 'FAILED_FIXTURE_INPUT')
    need(not failed.reserve_output(5), 'OVERSIZE_NOT_REJECTED')
    need(failed.reserve_output(4), 'RESERVE_AFTER_REJECTION')
    failed.begin()
    need(not failed.produce(5) and failed.objects['o']['n'] == 0, 'OVERRUN_WROTE_FIRST')
    failed.cancel()
    audit_pipeline(failed.trace)
    need(failed.physical == 0, 'FAILED_PRODUCTION_LEAK')

    shortage = Pipeline()
    need(shortage.admit('A', 'a', 4) and shortage.admit('B', 'b', 4), 'SHORTAGE_INPUT')
    # Explicit external occupancy fixture for pre-reservation denial only.
    # It is not a closed-object ledger snapshot and is excluded below.
    shortage.physical = 12
    need(not shortage.reserve_output(4), 'PHYSICAL_SHORTAGE_NOT_REJECTED')
    shortage.physical = 8
    shortage.cancel()
    shortage.consume('b')
    audit_pipeline([s for s in shortage.trace if s['event'] != 'output_reject'])
    need(shortage.physical == 0, 'RESERVATION_FAILURE_LEAK')

    overfull = full_pipeline(0)
    overfull.handoff(unsafe=True)
    reasons = [rejected(lambda: audit_pipeline(overfull.trace), 'STAGE_CAPACITY:B')]
    missing = Pipeline()
    missing.admit('A', 'a', 4)
    missing.reserve_output(4, faulty_account=True)
    reasons.append(rejected(lambda: audit_pipeline(missing.trace), 'LEDGER_MISMATCH:O'))
    orphan = full_pipeline()
    orphan.private.clear()
    orphan.record('fault_drop_owner_before_target_reservation')
    reasons.append(rejected(lambda: audit_pipeline(orphan.trace), 'ORPHANED_STORAGE'))
    return dict(normal=normal, cancellation_final_bytes=cancelled.physical,
                failed_final_bytes=failed.physical, negative_reasons=reasons,
                physical_shortage_scope='reservation denial; external occupancy snapshot not audited')


class ChildModel:
    """Finite pipe; environment exit is independent of the kill request."""
    def __init__(self):
        self.s = dict(written=0, drained=0, running=True, timeout=False,
                      kill_requested=False, confirmed=False, returncode=None,
                      eof=False, clean=False)
        self.trace = []

    def event(self, op, faulty_clean=False):
        s = self.s
        if op == 'write':
            if s['running'] and s['written'] - s['drained'] < 4:
                s['written'] += 1
        elif op == 'drain':
            s['drained'] = s['written']
        elif op == 'timeout':
            s['timeout'] = True
        elif op == 'kill_request':
            s['kill_requested'] = True
        elif op == 'os_exit':
            s['running'] = False
        elif op == 'observe_exit':
            if not s['running']:
                s['confirmed'], s['returncode'] = True, 0
        elif op == 'observe_eof':
            if not s['running'] and s['written'] == s['drained']:
                s['eof'] = True
        elif op == 'claim_clean':
            s['clean'] = faulty_clean or (s['confirmed'] and s['eof'] and
                                         s['written'] == s['drained'])
        elif op != 'poll_without_exit':
            raise ValueError(op)
        self.trace.append((op, deepcopy(s)))
        return self


def audit_child(trace):
    exited, confirmed, eof = False, False, False
    for op, s in trace:
        if op == 'os_exit':
            exited = True
        if op == 'observe_exit' and exited:
            confirmed = True
        if op == 'observe_eof' and exited and s['written'] == s['drained']:
            eof = True
        need(s['running'] == (not exited), 'KILL_REQUEST_IS_NOT_EXIT')
        need(s['confirmed'] == confirmed and s['eof'] == eof, 'INVENTED_EXIT_EVIDENCE')
        need(0 <= s['written'] - s['drained'] <= 4, 'PIPE_CAPACITY')
        need((s['returncode'] is not None) == confirmed, 'RETURN_CODE_WITHOUT_CONFIRMATION')
        if s['clean']:
            need(confirmed and eof and s['written'] == s['drained'],
                 'CLEAN_WITHOUT_EXIT_AND_DRAIN')
    return dict(written=trace[-1][1]['written'], confirmed=confirmed, eof=eof)


def check_child():
    blocked = ChildModel()
    for _ in range(10):
        blocked.event('write')
    audit_child(blocked.trace)
    need(blocked.s['written'] == 4 and blocked.s['running'], 'PIPE_DID_NOT_BLOCK')
    blocked.event('timeout').event('kill_request')
    blocked.event('poll_without_exit').event('poll_without_exit').event('claim_clean')
    audit_child(blocked.trace)
    need(blocked.s['running'] and not blocked.s['clean'], 'KILL_IMPLIED_CLEANUP')
    bad = deepcopy(blocked).event('claim_clean', faulty_clean=True)
    reasons = [rejected(lambda: audit_child(bad.trace), 'CLEAN_WITHOUT_EXIT_AND_DRAIN')]
    blocked.event('os_exit').event('observe_exit').event('claim_clean')
    need(not blocked.s['clean'], 'EXIT_CONFIRMATION_IMPLIED_DRAIN')
    blocked.event('drain').event('observe_eof').event('claim_clean')
    need(blocked.s['clean'], 'CONFIRMED_EXIT_AND_EOF_NOT_CLEAN')
    audit_child(blocked.trace)

    normal = ChildModel()
    for _ in range(10):
        normal.event('write').event('drain')
    normal.event('os_exit').event('observe_exit').event('observe_eof').event('claim_clean')
    result = audit_child(normal.trace)
    need(normal.s['written'] == 10 and normal.s['clean'], 'DRAINING_DID_NOT_PROGRESS')
    timeout_only = ChildModel().event('timeout').event('claim_clean', faulty_clean=True)
    reasons.append(rejected(lambda: audit_child(timeout_only.trace), 'CLEAN_WITHOUT_EXIT_AND_DRAIN'))
    return dict(normal=result, delayed_exit='modeled', negative_reasons=reasons)


def quota_cap(cpuset, quotas):
    caps = [Fraction(len(set(cpuset)))]
    for amount, period in quotas:
        need(period > 0, 'INVALID_PERIOD')
        if amount is not None:
            need(amount >= 0, 'NEGATIVE_QUOTA')
            caps.append(Fraction(amount, period))
    return min(caps)


def same(actual, expected, label):
    need(actual == expected, label)


def physical_bytes(objects):
    sizes = {}
    for identity, size in objects:
        need(size >= 0, 'NEGATIVE_PHYSICAL_SIZE')
        need(identity not in sizes or sizes[identity] == size, 'CONFLICTING_PHYSICAL_SIZE')
        sizes[identity] = size
    return sum(sizes.values())


def check_arithmetic():
    same(quota_cap(range(8), [(400000, 100000)]), Fraction(4), 'QUOTA_REFERENCE')
    same(quota_cap(range(8), [(400000, 100000), (200000, 100000)]),
         Fraction(2), 'ANCESTOR_REFERENCE')
    same(quota_cap([], [(None, 100000)]), Fraction(0), 'EMPTY_CPUSET')
    same(quota_cap(range(2), [(None, 100000)]), Fraction(2), 'UNLIMITED_QUOTA')
    demand = lambda rate: Fraction(15, 1000) * rate + Fraction(12, 10)
    same(demand(200), Fraction(21, 5), 'CPU_DEMAND_200')
    same(demand(180), Fraction(39, 10), 'CPU_DEMAND_180')
    need(demand(200) > 4 and demand(180) < 4, 'QUOTA_BOUNDARY_ARITHMETIC')
    reasons = [rejected(lambda: same(64, Fraction(4), 'QUOTA_REFERENCE'), 'QUOTA_REFERENCE')]
    objects = [('ipc', 10), ('ipc', 10), ('copy', 5)]
    byte_set_reference = len({(identity, offset) for identity, size in objects
                              for offset in range(size)})
    same(physical_bytes(objects), byte_set_reference, 'PHYSICAL_UNION_REFERENCE')
    reasons.append(rejected(lambda: same(sum(size for _, size in objects), byte_set_reference,
                                         'PHYSICAL_UNION_REFERENCE'), 'PHYSICAL_UNION_REFERENCE'))
    same(physical_bytes([]), 0, 'EMPTY_PHYSICAL_SET')
    grouped = 512 * 65536 + 512 * 4096 * 4 + 128 * 65536
    same(grouped, 48 * 1024 * 1024, 'RESIDENT_EXAMPLE')
    need(20 * 5 > 90, 'QUEUE_DEADLINE_EXAMPLE')
    return dict(configured_core_cap='4', cpu_demand_200=str(demand(200)),
                resident_example_bytes=grouped, shared_bytes=byte_set_reference,
                negative_reasons=reasons)


def main():
    for name, check in [('buffer_lifecycle', check_buffers),
                        ('bounded_handoff', check_handoff),
                        ('child_evidence', check_child),
                        ('resource_arithmetic', check_arithmetic)]:
        observation = check()
        print(json.dumps(dict(check=name, observation=observation), sort_keys=True))
    print('SCOPE: deterministic CPU symbolic checks only; no real threads, cgroups, subprocesses, network, DMA or GPU measurements')


if __name__ == '__main__':
    main()
