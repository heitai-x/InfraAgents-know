#!/usr/bin/env python3
'''K-SYS-004: deterministic, offline, read-only standard-library CPU models.
No file access, project imports, processes, network, credentials or device APIs.
Fixed decisions and accepted-history audits are independent of mutable models.
'''
from itertools import permutations


def Y(op, *args):
    return ((op,) + args, True)


def N(op, *args):
    return ((op,) + args, False)


def lifetime_facts(history):
    f = dict(objects=set(), current={}, epoch={}, closed=set(), freed=set(),
             tokens=set(), issued=set(), requests={}, commits=set())
    for ev, accepted in history:
        if not accepted:
            continue
        op, *a = ev
        if op in ('start', 'advance'):
            r, g = a
            f['requests'][r] = (g, True)
        elif op in ('cancel', 'timeout'):
            r = a[0]
            f['requests'][r] = (f['requests'][r][0], False)
        elif op == 'alloc':
            x, r = a
            assert x[0] not in f['current']
            assert x[1] == f['epoch'].get(x[0], -1) + 1
            assert not any(t[1][0] == x[0] for t in f['tokens'])
            f['objects'].add(x)
            f['current'][x[0]] = x
            f['epoch'][x[0]] = x[1]
            f['tokens'].add(('L', x, r))
        elif op == 'owner_add':
            x, r = a
            f['tokens'].add(('L', x, r))
        elif op == 'owner_drop':
            x, r = a
            f['tokens'].discard(('L', x, r))
        elif op == 'begin':
            x, r, g, uid, lo, hi = a
            t = ('U', x, r, g, uid, lo, hi)
            f['tokens'].add(t)
            f['issued'].add(t)
        elif op == 'end':
            x, r, g, uid, lo, hi, stopped = a
            assert stopped
            f['tokens'].remove(('U', x, r, g, uid, lo, hi))
        elif op in ('consumer_add', 'bind_add'):
            x, ident = a
            f['tokens'].add(('O' if op == 'consumer_add' else 'B', x, ident))
        elif op in ('consumer_drop', 'bind_drop'):
            x, ident = a
            f['tokens'].discard(('O' if op == 'consumer_drop' else 'B', x, ident))
        elif op == 'retire':
            f['closed'].add(a[0])
        elif op == 'release':
            x = a[0]
            if x not in f['freed']:
                assert x in f['closed']
                assert not any(t[1] == x for t in f['tokens'])
                assert f['current'].get(x[0]) == x
                f['freed'].add(x)
                del f['current'][x[0]]
        elif op == 'commit':
            r, g = a
            assert f['requests'].get(r) == (g, True)
            assert (r, g) not in f['commits']
            f['commits'].add((r, g))
        else:
            raise AssertionError(('unknown lifetime fact', ev))
    return f


class Lifetime:
    def __init__(self, fault=''):
        self.fault = fault
        self.obj = {}
        self.current = {}
        self.epoch = {}
        self.requests = {}
        self.commits = set()
        self.issued = set()

    def apply(self, ev):
        op, *a = ev
        if op == 'start':
            r, g = a
            if r in self.requests or g != 0:
                return False
            self.requests[r] = (g, True)
            return True
        if op == 'advance':
            r, g = a
            old = self.requests.get(r)
            if old is None or not old[1] or g != old[0] + 1:
                return False
            self.requests[r] = (g, True)
            return True
        if op in ('cancel', 'timeout'):
            r = a[0]
            if r not in self.requests:
                return False
            self.requests[r] = (self.requests[r][0], False)
            if op == 'timeout' and self.fault == 'timeout_is_completion':
                for obj in self.obj.values():
                    for key in list(obj['U']):
                        if key[0] == r:
                            del obj['U'][key]
            return True
        if op == 'commit':
            r, g = a
            old = self.requests.get(r)
            ok = old == (g, True)
            if self.fault == 'cancel_commit' and old is not None:
                ok = old[0] == g
            if not ok or (r, g) in self.commits:
                return False
            self.commits.add((r, g))
            return True
        x = a[0]
        obj = self.obj.get(x)
        alive = obj is not None and not obj['free'] and self.current.get(x[0]) == x
        if op == 'alloc':
            x, r = a
            old = self.current.get(x[0])
            blocked = old is not None
            historical = any(y[0] == x[0] and any(v[tag] for tag in ('L', 'U', 'O', 'B'))
                             for y, v in self.obj.items())
            if self.fault == 'reuse_with_old_use':
                blocked = old is not None and bool(self.obj[old]['L'])
                historical = False
            if (r not in self.requests or not self.requests[r][1] or x in self.obj
                    or x[1] != self.epoch.get(x[0], -1) + 1 or blocked or historical):
                return False
            self.obj[x] = dict(L={r}, U={}, O=set(), B=set(), closed=False, free=False)
            self.current[x[0]] = x
            self.epoch[x[0]] = x[1]
            return True
        if op == 'owner_add':
            x, r = a
            if not alive or obj['closed'] or r not in self.requests or not self.requests[r][1]:
                return False
            obj['L'].add(r)
            return True
        if op in ('owner_drop', 'consumer_drop', 'bind_drop'):
            if obj is None:
                return False
            tag = {'owner_drop': 'L', 'consumer_drop': 'O', 'bind_drop': 'B'}[op]
            obj[tag].discard(a[1])
            return True
        if op in ('consumer_add', 'bind_add'):
            if not alive or obj['closed']:
                return False
            obj['O' if op == 'consumer_add' else 'B'].add(a[1])
            return True
        if op == 'begin':
            x, r, g, uid, lo, hi = a
            key = (r, g, uid)
            prefix = ('U', x, r, g, uid)
            if (not alive or self.requests.get(r) != (g, True) or r not in obj['L']
                    or not 0 <= lo < hi <= 8
                    or any(t[:5] == prefix for t in self.issued)):
                return False
            obj['U'][key] = (lo, hi)
            self.issued.add(prefix + (lo, hi))
            return True
        if op == 'end':
            x, r, g, uid, lo, hi, stopped = a
            key = (r, g, uid)
            pending = obj is not None and key in obj['U']
            ok = pending and stopped and obj['U'][key] == (lo, hi)
            if pending and self.fault == 'accept_invalid_completion':
                ok = True
            if not ok:
                if self.fault == 'stale_end_clears_new_use':
                    new = self.obj.get(self.current.get(x[0]))
                    if new is not None:
                        for k in list(new['U']):
                            if k[2] == uid:
                                del new['U'][k]
                return False
            del obj['U'][key]
            if self.fault == 'drop_other_reader':
                obj['U'].clear()
            return True
        if op == 'retire':
            if not alive:
                return False
            obj['closed'] = True
            return True
        if op == 'release':
            if obj is None:
                return False
            if obj['free']:
                if self.fault == 'stale_release_clears_new_slot':
                    self.current.pop(x[0], None)
                return True
            if not alive or not obj['closed']:
                return False
            ignore = {'ignore_uses_on_release': 'U', 'ignore_shared_refs': 'L',
                      'ignore_output': 'O', 'ignore_binding': 'B'}.get(self.fault)
            if any(obj[tag] for tag in ('L', 'U', 'O', 'B') if tag != ignore):
                return False
            obj['free'] = True
            del self.current[x[0]]
            return True
        raise AssertionError(('unknown lifetime operation', ev))

    def snapshot(self):
        tokens = set()
        for x, obj in self.obj.items():
            for tag in ('L', 'O', 'B'):
                tokens.update((tag, x, ident) for ident in obj[tag])
            for (r, g, uid), (lo, hi) in obj['U'].items():
                tokens.add(('U', x, r, g, uid, lo, hi))
        return dict(objects=set(self.obj), current=dict(self.current), epoch=dict(self.epoch),
                    closed={x for x, o in self.obj.items() if o['closed']},
                    freed={x for x, o in self.obj.items() if o['free']},
                    tokens=tokens, issued=set(self.issued), requests=dict(self.requests),
                    commits=set(self.commits))


P = ('prefix', 0)
S0, S1, S2 = ('s', 0), ('s', 1), ('s', 2)


def lifetime_cases():
    yield 'shared_ref', [
        Y('start', 'A', 0), Y('start', 'B', 0), Y('alloc', P, 'A'),
        Y('owner_add', P, 'B'), Y('begin', P, 'A', 0, 'ua', 0, 8),
        Y('cancel', 'A'), Y('owner_drop', P, 'A'), Y('retire', P),
        N('release', P), Y('end', P, 'A', 0, 'ua', 0, 8, True),
        N('release', P), Y('owner_drop', P, 'B'), Y('release', P)]
    yield 'cancel_timeout_commit', [
        Y('start', 'A', 0), Y('start', 'B', 0), Y('alloc', S0, 'A'),
        Y('begin', S0, 'A', 0, 'ua', 0, 8), Y('cancel', 'A'),
        Y('owner_drop', S0, 'A'), Y('retire', S0), N('release', S0),
        N('commit', 'A', 0), Y('timeout', 'A'), N('release', S0),
        Y('end', S0, 'A', 0, 'ua', 0, 8, True), Y('release', S0), Y('alloc', S1, 'B')]
    yield 'slot_reuse_old_completion', [
        Y('start', 'A', 0), Y('start', 'B', 0), Y('start', 'C', 0),
        Y('alloc', S0, 'A'), Y('begin', S0, 'A', 0, 'u', 0, 8),
        Y('retire', S0), N('release', S0), N('alloc', S1, 'B'),
        Y('owner_drop', S0, 'A'), N('release', S0),
        Y('end', S0, 'A', 0, 'u', 0, 8, True), Y('release', S0),
        Y('alloc', S1, 'B'), Y('begin', S1, 'B', 0, 'u', 0, 8),
        Y('retire', S1), Y('owner_drop', S1, 'B'), N('release', S1),
        Y('timeout', 'B'), N('release', S1),
        N('end', S0, 'A', 0, 'u', 0, 8, True), Y('release', S0), N('alloc', S2, 'C'),
        Y('end', S1, 'B', 0, 'u', 0, 8, True), Y('release', S1), Y('alloc', S2, 'C')]
    yield 'invalid_completion_unmasked', [
        Y('start', 'C', 0), Y('alloc', S0, 'C'), Y('begin', S0, 'C', 0, 'uc', 0, 8),
        Y('owner_drop', S0, 'C'), Y('retire', S0),
        N('end', S0, 'C', 0, 'uc', 0, 8, False), N('release', S0),
        N('end', S0, 'C', 0, 'uc', 0, 4, True), N('release', S0),
        Y('end', S0, 'C', 0, 'uc', 0, 8, True),
        N('end', S0, 'C', 0, 'uc', 0, 8, True), Y('release', S0)]
    yield 'output_and_binding', [
        Y('start', 'A', 0), Y('alloc', S0, 'A'), Y('consumer_add', S0, 'c1'),
        Y('consumer_add', S0, 'c2'), Y('bind_add', S0, 'graph'),
        Y('owner_drop', S0, 'A'), Y('retire', S0), N('release', S0),
        Y('consumer_drop', S0, 'c1'), Y('consumer_drop', S0, 'c1'), N('release', S0),
        Y('consumer_drop', S0, 'c2'), N('release', S0),
        Y('bind_drop', S0, 'graph'), Y('release', S0)]
    yield 'commit_and_begin_gates', [
        Y('start', 'A', 0), Y('alloc', S0, 'A'), Y('commit', 'A', 0),
        N('commit', 'A', 0), Y('cancel', 'A'), Y('cancel', 'A'),
        N('begin', S0, 'A', 0, 'denied', 0, 8), N('commit', 'A', 0),
        Y('owner_drop', S0, 'A'), Y('retire', S0), Y('release', S0),
        Y('start', 'B', 0), Y('commit', 'B', 0), Y('advance', 'B', 1),
        N('commit', 'B', 0), Y('commit', 'B', 1), N('commit', 'B', 1)]
    for order in permutations(('u', 'v', 'w')):
        events = [Y('start', 'R', 0), Y('alloc', S0, 'R')]
        events += [Y('begin', S0, 'R', 0, uid, 0, 8) for uid in ('u', 'v', 'w')]
        events += [Y('retire', S0), Y('owner_drop', S0, 'R')]
        for i, uid in enumerate(order):
            events += [Y('end', S0, 'R', 0, uid, 0, 8, True),
                       N('end', S0, 'R', 0, uid, 0, 8, True),
                       (Y if i == 2 else N)('release', S0)]
        yield 'reader_order_' + ''.join(order), events
    for mask in range(16):
        events = [Y('start', 'R', 0), Y('alloc', S0, 'R')]
        if mask & 2:
            events.append(Y('begin', S0, 'R', 0, 'u', 0, 8))
        if mask & 4:
            events.append(Y('consumer_add', S0, 'c'))
        if mask & 8:
            events.append(Y('bind_add', S0, 'b'))
        if not mask & 1:
            events.append(Y('owner_drop', S0, 'R'))
        events += [Y('retire', S0), (Y if mask == 0 else N)('release', S0)]
        yield 'blocker_mask_' + str(mask), events


def reference_regression():
    history = [Y('start', 'A', 0), Y('start', 'B', 0), Y('alloc', S0, 'A'),
               Y('begin', S0, 'A', 0, 'u', 0, 8), Y('retire', S0),
               N('alloc', S1, 'B'), Y('owner_drop', S0, 'A'),
               Y('end', S0, 'A', 0, 'u', 0, 8, True)]
    f = lifetime_facts(history)
    assert f['epoch']['s'] == 0
    assert f['current']['s'] == S0
    assert not any(t[1][0] == 's' for t in f['tokens'])
    old_tokens = f['tokens'] | {('L', S1, 'B')}
    assert any(t[1][0] == 's' for t in old_tokens)
    print('REGRESSION rejected allocation creates no owner or epoch; polluted reference detected')


def migration_facts(history):
    '''Audit public accepted history, never candidate counters or snapshots.'''
    rights = ('src', 0)
    phase, gate, live = 'OPEN', True, True
    tickets, pending, checked = {}, set(), set()
    log, attempt, cut, ready = [], None, None, None
    outcomes, seen = {}, set()
    dst_active, dst_state = False, None
    for ev, accepted in history:
        if not accepted:
            continue
        op, *a = ev
        if op == 'admit':
            who, term, uid, base, cid, value = a
            assert live and gate and phase == 'OPEN' and rights == (who, term)
            assert uid not in tickets and base == len(log)
            assert cid not in {c for c, value in log}
            tickets[uid] = (who, term, base, cid, value)
            pending.add(uid)
        elif op == 'precheck':
            uid = a[0]
            assert live and uid in pending and tickets[uid][:2] == rights
            checked.add(uid)
        elif op == 'install':
            uid = a[0]
            assert live and uid in pending
            who, term, base, cid, value = tickets[uid]
            assert rights == (who, term) and phase in ('OPEN', 'DRAINING')
            assert base == len(log) and cid not in {c for c, value in log}
            log.append((cid, value))
            pending.remove(uid)
            if who == 'dst':
                assert dst_active and dst_state is not None
                dst_state += ((cid, value),)
        elif op == 'revoke':
            assert a[0] in pending
            pending.remove(a[0])
        elif op == 'close':
            ident = a[0]
            assert live and rights == ('src', 0) and phase == 'OPEN' and ident not in seen
            seen.add(ident)
            attempt, phase, gate = ident, 'DRAINING', False
            cut, ready = None, None
        elif op == 'freeze':
            assert live and a[0] == attempt and phase == 'DRAINING' and not pending
            assert not gate
            cut, phase = tuple(log), 'FROZEN'
        elif op == 'ready':
            ident, payload = a
            assert live and ident == attempt and phase == 'FROZEN' and payload == cut
            ready = payload
        elif op == 'transfer':
            ident = a[0]
            if ident in outcomes and outcomes[ident][0] == 'COMMITTED':
                continue
            assert live and ident == attempt and phase == 'FROZEN'
            assert rights == ('src', 0) and not pending and not gate
            assert cut is not None and ready == cut == tuple(log)
            rights, phase, gate = ('dst', 1), 'TAKEN', False
            outcomes[ident] = ('COMMITTED', 1, len(log))
        elif op == 'activate':
            ident, term = a
            assert live and rights == ('dst', term) and term == 1
            assert ident == attempt and outcomes[ident][0] == 'COMMITTED'
            if not dst_active:
                assert phase == 'TAKEN' and ready == tuple(log)
                dst_state, dst_active = ready, True
                phase, gate = 'OPEN', True
        elif op in ('abort', 'timeout_resume'):
            ident = a[0]
            assert live and rights == ('src', 0) and ident == attempt
            assert phase in ('DRAINING', 'FROZEN') and ident not in outcomes
            outcomes[ident] = ('ABORTED', 0, len(log))
            attempt, phase, gate, cut, ready = None, 'OPEN', True, None, None
        elif op == 'query':
            assert a[0] in outcomes
        elif op == 'retry':
            assert a[0] in {cid for cid, value in log}
        elif op == 'lose_ack':
            assert outcomes.get(a[0], ('',))[0] == 'COMMITTED'
        elif op == 'cancel':
            live, gate, phase = False, False, 'TERMINAL'
        else:
            raise AssertionError(('unknown migration fact', ev))
        assert len(log) == len({cid for cid, value in log})
        if phase == 'FROZEN':
            assert not pending and cut == tuple(log) and not gate
        if dst_active:
            assert rights == ('dst', 1) and dst_state == tuple(log)
    return dict(rights=rights, phase=phase, gate=gate, live=live,
                tickets=tickets, pending=pending, checked=checked, log=tuple(log),
                attempt=attempt, cut=cut, ready=ready, outcomes=outcomes, seen=seen,
                dst_active=dst_active, dst_state=dst_state)


class Migration:
    '''One request and one src->dst transfer; aborted attempts may be retried.'''
    def __init__(self, fault=''):
        self.fault = fault
        self.owner, self.term = 'src', 0
        self.phase, self.gate, self.live = 'OPEN', True, True
        self.tickets, self.pending, self.checked = {}, set(), set()
        self.log = []
        self.attempt, self.cut, self.ready = None, None, None
        self.outcomes, self.seen = {}, set()
        self.dst_active, self.dst_state = False, None

    def apply(self, ev):
        op, *a = ev
        if op == 'admit':
            who, term, uid, base, cid, value = a
            allowed = self.gate and self.phase == 'OPEN'
            if self.fault == 'snapshot_progress' and who == 'src':
                allowed = True
            if (not self.live or not allowed or (who, term) != (self.owner, self.term)
                    or uid in self.tickets or base != len(self.log)
                    or cid in {c for c, value in self.log}):
                return False
            self.tickets[uid] = (who, term, base, cid, value)
            self.pending.add(uid)
            return True
        if op == 'precheck':
            uid = a[0]
            if (not self.live or uid not in self.pending
                    or self.tickets[uid][:2] != (self.owner, self.term)):
                return False
            self.checked.add(uid)
            return True
        if op == 'install':
            uid = a[0]
            record = self.tickets.get(uid)
            if record is None:
                return False
            who, term, base, cid, value = record
            ok = (self.live and uid in self.pending and (who, term) == (self.owner, self.term)
                  and self.phase in ('OPEN', 'DRAINING')
                  and cid not in {c for c, value in self.log})
            if self.fault != 'ignore_commit_base':
                ok = ok and base == len(self.log)
            if self.fault == 'cached_authority' and uid in self.checked:
                ok = True
            if not ok:
                return False
            self.log.append((cid, value))
            self.pending.discard(uid)
            if who == 'dst':
                self.dst_state = (self.dst_state or ()) + ((cid, value),)
            return True
        if op == 'revoke':
            uid = a[0]
            if uid not in self.pending:
                return False
            self.pending.remove(uid)
            return True
        if op == 'close':
            ident = a[0]
            if (not self.live or (self.owner, self.term) != ('src', 0)
                    or self.phase != 'OPEN' or ident in self.seen):
                return False
            self.seen.add(ident)
            self.attempt, self.phase, self.gate = ident, 'DRAINING', False
            self.cut, self.ready = None, None
            return True
        if op == 'freeze':
            if (not self.live or a[0] != self.attempt or self.phase != 'DRAINING'
                    or (self.pending and self.fault != 'skip_drain')):
                return False
            self.cut = tuple(self.log)
            if self.fault == 'cut_omits_last_commit' and self.log:
                self.cut = tuple(self.log[:-1])
            self.phase = 'FROZEN'
            return True
        if op == 'ready':
            ident, payload = a
            if (not self.live or ident != self.attempt or self.phase != 'FROZEN'
                    or (payload != self.cut and self.fault != 'accept_wrong_ready')):
                return False
            self.ready = payload
            if self.fault == 'ready_is_authority':
                self.dst_active = True
            return True
        if op == 'transfer':
            ident = a[0]
            if self.outcomes.get(ident, ('',))[0] == 'COMMITTED':
                return True
            if (not self.live or ident != self.attempt or self.phase != 'FROZEN'
                    or (self.owner, self.term) != ('src', 0) or self.gate or self.pending
                    or self.cut is None or self.ready != self.cut or self.cut != tuple(self.log)):
                return False
            self.owner, self.term, self.phase, self.gate = 'dst', 1, 'TAKEN', False
            self.outcomes[ident] = ('COMMITTED', 1, len(self.log))
            return True
        if op == 'activate':
            ident, term = a
            if (not self.live or (self.owner, self.term) != ('dst', term) or term != 1
                    or ident != self.attempt
                    or self.outcomes.get(ident, ('',))[0] != 'COMMITTED'):
                return False
            if not self.dst_active:
                if self.phase != 'TAKEN' or self.ready != tuple(self.log):
                    return False
                self.dst_state, self.dst_active = self.ready, True
                self.phase, self.gate = 'OPEN', True
            return True
        if op in ('abort', 'timeout_resume'):
            ident = a[0]
            if (op == 'timeout_resume' and self.fault == 'timeout_rollback'
                    and self.outcomes.get(ident, ('',))[0] == 'COMMITTED'):
                self.owner, self.term, self.phase, self.gate = 'src', 0, 'OPEN', True
                return True
            if (not self.live or (self.owner, self.term) != ('src', 0)
                    or ident != self.attempt or self.phase not in ('DRAINING', 'FROZEN')
                    or ident in self.outcomes):
                return False
            self.outcomes[ident] = ('ABORTED', 0, len(self.log))
            self.attempt, self.phase, self.gate = None, 'OPEN', True
            self.cut, self.ready = None, None
            return True
        if op == 'query':
            return a[0] in self.outcomes
        if op == 'retry':
            return a[0] in {cid for cid, value in self.log}
        if op == 'lose_ack':
            return self.outcomes.get(a[0], ('',))[0] == 'COMMITTED'
        if op == 'cancel':
            self.live, self.gate, self.phase = False, False, 'TERMINAL'
            return True
        raise AssertionError(('unknown migration operation', ev))

    def snapshot(self):
        return dict(rights=(self.owner, self.term), phase=self.phase, gate=self.gate,
                    live=self.live, tickets=dict(self.tickets), pending=set(self.pending),
                    checked=set(self.checked), log=tuple(self.log), attempt=self.attempt,
                    cut=self.cut, ready=self.ready, outcomes=dict(self.outcomes),
                    seen=set(self.seen), dst_active=self.dst_active, dst_state=self.dst_state)


def migration_cases():
    first = (('c1', 17),)
    yield 'snapshot_then_source_progress_and_lost_ack', [
        Y('admit', 'src', 0, 'u1', 0, 'c1', 17), Y('install', 'u1'),
        Y('close', 'a'), Y('freeze', 'a'),
        N('admit', 'src', 0, 'late', 1, 'late-cid', 99),
        N('ready', 'a', ()), Y('ready', 'a', first),
        N('admit', 'dst', 1, 'early', 1, 'early-cid', 98),
        N('activate', 'a', 1), Y('transfer', 'a'), Y('lose_ack', 'a'),
        N('timeout_resume', 'a'), Y('query', 'a'),
        N('admit', 'src', 0, 'late2', 1, 'late2-cid', 97),
        Y('activate', 'a', 1),
        Y('admit', 'dst', 1, 'u2', 1, 'c2', 23), Y('install', 'u2'),
        Y('activate', 'a', 1), Y('transfer', 'a'), Y('retry', 'c2'),
        N('admit', 'dst', 1, 'u3', 2, 'c2', 23)]
    yield 'prechecked_old_transaction_settles_before_cut', [
        Y('admit', 'src', 0, 'old', 0, 'c1', 17), Y('precheck', 'old'),
        Y('close', 'a'), N('freeze', 'a'), N('transfer', 'a'),
        Y('install', 'old'), Y('freeze', 'a'), Y('ready', 'a', first),
        Y('transfer', 'a'), Y('activate', 'a', 1), N('install', 'old')]
    yield 'prechecked_old_transaction_revoked_then_resumes', [
        Y('admit', 'src', 0, 'old', 0, 'old-cid', 91), Y('precheck', 'old'),
        Y('close', 'a'), N('freeze', 'a'), Y('revoke', 'old'),
        Y('freeze', 'a'), Y('ready', 'a', ()), Y('transfer', 'a'),
        Y('lose_ack', 'a'), N('timeout_resume', 'a'), Y('activate', 'a', 1),
        Y('admit', 'dst', 1, 'new', 0, 'new-cid', 42), Y('install', 'new'),
        N('install', 'old'), N('revoke', 'old'), Y('query', 'a')]
    yield 'abort_invalidates_old_snapshot_and_attempt', [
        Y('close', 'a'), Y('freeze', 'a'), Y('ready', 'a', ()),
        Y('abort', 'a'), Y('query', 'a'),
        Y('admit', 'src', 0, 'u1', 0, 'c1', 17), Y('install', 'u1'),
        N('transfer', 'a'), N('ready', 'a', ()), N('close', 'a'),
        Y('close', 'b'), Y('freeze', 'b'), N('ready', 'b', ()),
        Y('ready', 'b', first), N('transfer', 'a'), Y('transfer', 'b'),
        Y('activate', 'b', 1)]
    yield 'two_entered_commits_require_current_predecessor', [
        Y('admit', 'src', 0, 'u1', 0, 'c1', 17),
        Y('admit', 'src', 0, 'u2', 0, 'c2', 23), Y('precheck', 'u2'),
        Y('close', 'a'), Y('install', 'u1'), N('install', 'u2'),
        N('freeze', 'a'), Y('revoke', 'u2'), Y('freeze', 'a'),
        Y('ready', 'a', first), Y('transfer', 'a'), Y('activate', 'a', 1),
        Y('retry', 'c1'), N('retry', 'c2')]
    yield 'cancel_does_not_restore_service_or_transfer', [
        Y('admit', 'src', 0, 'u1', 0, 'c1', 17), Y('precheck', 'u1'),
        Y('close', 'a'), Y('cancel'), N('install', 'u1'),
        Y('revoke', 'u1'), N('freeze', 'a'), N('abort', 'a'),
        N('timeout_resume', 'a'), N('transfer', 'a'),
        N('admit', 'src', 0, 'u2', 0, 'c2', 23)]
    yield 'unknown_decision_is_not_permission_to_resume', [
        Y('close', 'a'), N('query', 'a'), N('transfer', 'a'),
        N('admit', 'src', 0, 'u1', 0, 'c1', 17),
        Y('freeze', 'a'), Y('timeout_resume', 'a'), Y('query', 'a'),
        N('transfer', 'a'), Y('admit', 'src', 0, 'u1', 0, 'c1', 17),
        Y('install', 'u1')]


def run_case(factory, audit, events, fault=''):
    model, history = factory(fault), []
    for i, (ev, golden) in enumerate(events):
        got = model.apply(ev)
        if got != golden:
            return (i, ev, 'decision', got, golden)
        history.append((ev, golden))
        reference = audit(history)
        observed = model.snapshot()
        if observed != reference:
            fields = tuple(k for k in reference if reference[k] != observed[k])
            return (i, ev, 'state', fields)
    return None


def check_suite(label, factory, audit, cases, faults):
    event_count = 0
    for name, events in cases:
        mismatch = run_case(factory, audit, events)
        assert mismatch is None, ('POSITIVE_FAILURE', label, name, mismatch)
        event_count += len(events)
        print('POSITIVE', label, name, 'events=', len(events), 'decisions_and_history=PASS')
    for fault in faults:
        witness = None
        for name, events in cases:
            mismatch = run_case(factory, audit, events, fault)
            if mismatch is not None:
                witness = (name, mismatch)
                break
        assert witness is not None, ('NEGATIVE_NOT_DETECTED', label, fault)
        print('NEGATIVE', label, fault, 'detected=True', 'witness=', witness)
    print('SUITE', label, 'cases=', len(cases), 'events=', event_count,
          'negative_controls=', len(faults))


def main():
    reference_regression()
    check_suite('lifetime', Lifetime, lifetime_facts, list(lifetime_cases()), (
        'ignore_uses_on_release', 'ignore_shared_refs', 'cancel_commit',
        'timeout_is_completion', 'accept_invalid_completion', 'reuse_with_old_use',
        'stale_release_clears_new_slot', 'stale_end_clears_new_use',
        'ignore_output', 'ignore_binding', 'drop_other_reader'))
    check_suite('migration', Migration, migration_facts, list(migration_cases()), (
        'snapshot_progress', 'skip_drain', 'cached_authority',
        'cut_omits_last_commit', 'accept_wrong_ready', 'ready_is_authority',
        'timeout_rollback', 'ignore_commit_base'))
    print('PASS finite CPU lifetime and authoritative migration models')
    print('scope=explicit golden traces plus accepted-history state and prefix invariants')
    print('not_verified=GPU visibility, CUDA/RDMA/Linux APIs, permanent Q, actual partial submission')
    print('not_verified=distributed consensus, durable storage, real model/RNG transitions, network delivery, performance')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
