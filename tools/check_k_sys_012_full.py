#!/usr/bin/env python3
'''K-SYS-012: deterministic, offline, read-only CPU protocol model.
Python stdlib only; no files, processes, project imports, network, or GPU.
One allocator domain; owner strings denote non-reused request-attempt instances.
All accesses cover [0,G). NEW registrations are immediately accepted by World.
World event history is independent of allocator U and operation terminal records.
'''
from copy import deepcopy
from dataclasses import dataclass, field

G = 64 * 1024


def equal(actual, expected):
    assert actual == expected, (actual, expected)


def rejected(label, check):
    try:
        check()
    except AssertionError:
        print('negative control detected:', label)
        return
    raise AssertionError('oracle accepted faulty control: ' + label)


@dataclass
class Frame:
    fid: int
    tag: str = 'kv'
    epoch: int = 0
    state: str = 'CACHED_FREE'
    mapped: bool = False
    ready: bool = False
    value: object = None
    refs: set = field(default_factory=set)
    pins: set = field(default_factory=set)
    u: set = field(default_factory=set)
    bindings: set = field(default_factory=set)
    quarantine: set = field(default_factory=set)
    tx: object = None


@dataclass
class Transaction:
    tid: int
    owner: str
    old: tuple
    old_exists: bool
    version: int
    sources: set
    new: list = field(default_factory=list)
    status: str = 'PREPARING'
    jobs: set = field(default_factory=set)
    promise: int = 0
    failed: bool = False


@dataclass
class Operation:
    oid: tuple
    desc: tuple
    specs: tuple
    tx: object
    writes: tuple
    copy_pair: object
    status: str = 'PENDING'


class Allocator:
    def __init__(self, capacity_pages, cached_pages=None, forget_done=False):
        if cached_pages is None:
            cached_pages = capacity_pages
        assert 0 <= cached_pages <= capacity_pages
        self.cap = capacity_pages * G
        self.frames = {i: Frame(i) for i in range(cached_pages)}
        self.views = {}
        self.live = {}
        self.versions = {}
        self.transactions = {}
        self.operations = {}
        self.claimed = 0
        self.trace = []
        self.forget_done = forget_done  # Intentionally faulty retention policy.

    def physical(self):
        return len(self.frames) * G

    def plan(self, n, margin=0):
        if type(n) is not int or n < 0 or margin < 0:
            raise ValueError('nonnegative integer page demand required')
        free = sorted(f.fid for f in self.frames.values()
                      if f.state == 'CACHED_FREE' and f.tag == 'kv')
        reuse = free[:n]
        missing = n - len(reuse)
        if self.physical() + self.claimed + missing * G + margin > self.cap:
            return None
        return reuse, missing

    def refresh(self, fid):
        f = self.frames[fid]
        if f.state == 'CACHED_FREE':
            return
        if f.quarantine:
            f.state = 'QUARANTINED'
        elif f.tx is not None:
            t = self.transactions[f.tx]
            if t.status == 'ABORTING':
                f.state = 'DRAINING'
            else:
                f.state = 'MAPPED' if f.mapped else 'RESERVED'
        elif f.refs or f.pins or f.bindings:
            f.state = 'IN_USE' if f.u else 'PUBLISHED'
        elif f.u:
            f.state = 'DRAINING'
        else:
            f.mapped = False
            f.ready = False
            f.value = None
            f.epoch += 1
            f.state = 'CACHED_FREE'

    def reserve(self, owner, n, cow=False, fail_at=None, fault_leak=False,
                margin=0, pause_after_acquire=False):
        if owner in self.live and not self.live[owner]:
            return None
        planned = self.plan(n, margin)
        if planned is None:
            return None
        if pause_after_acquire and n != 1:
            raise ValueError('one-page early preparation fixture required')
        old = self.views.get(owner, ())
        if cow and (len(old) != 1 or n != 1):
            raise ValueError('this model supports one-page COW')
        # Eligibility exists before the first acquired page and before publication.
        self.live.setdefault(owner, True)
        self.versions.setdefault(owner, 0)
        reuse, missing = planned
        tid = len(self.transactions)
        t = Transaction(tid, owner, old, owner in self.views,
                        self.versions[owner], set(old) if cow else set(),
                        promise=missing * G)
        self.transactions[tid] = t
        self.claimed += t.promise
        for fid in t.sources:
            self.frames[fid].pins.add(tid)
        for index, fid in enumerate(list(reuse) + [None] * missing, 1):
            if fid is None:
                fid = max(self.frames, default=-1) + 1
                self.frames[fid] = Frame(fid)
                self.claimed -= G
                t.promise -= G
            f = self.frames[fid]
            assert f.state == 'CACHED_FREE'
            assert not (f.refs or f.u or f.pins or f.bindings or f.quarantine)
            f.state = 'RESERVED'
            f.tx = tid
            t.new.append(fid)
            self.trace.append(('acquire', tid, fid))
            if fail_at == ('acquire', index):
                self.abort(t, fault_leak)
                return t
            if pause_after_acquire:
                return t
        for index, fid in enumerate(t.new, 1):
            self.frames[fid].mapped = True
            self.refresh(fid)
            self.trace.append(('map', tid, fid))
            if fail_at == ('map', index):
                self.abort(t, fault_leak)
                return t
        return t

    @staticmethod
    def descriptor(context, specs, writes=(), copy_pair=None):
        # G makes the fixed whole-page range part of this model's description.
        return (context, G, tuple(specs), tuple(writes), copy_pair)

    def prior(self, oid, desc):
        op = self.operations.get(oid)
        if op is None or op.desc != desc:
            return None
        return op

    def retry(self, oid, desc):
        # Read-only receipt lookup, including after completion or owner closure.
        return self.prior(oid, desc)

    def install_operation(self, oid, desc, specs, tx, writes, copy_pair):
        op = Operation(oid, desc, tuple(specs), tx, tuple(writes), copy_pair)
        self.operations[oid] = op
        for fid, epoch, uid, mode in specs:
            self.frames[fid].u.add((oid, epoch, mode))
            self.refresh(fid)
        if tx is not None:
            self.transactions[tx].jobs.add(oid)
        return op, True

    def register_public(self, owner, uid, specs, writes=()):
        if type(uid) is not int or uid < 0:
            return None, False
        oid = ('public', owner, uid)
        desc = self.descriptor(('view', owner, self.versions.get(owner, 0)),
                               specs, writes)
        if oid in self.operations:
            return self.prior(oid, desc), False
        if not self.live.get(owner, False) or owner not in self.views or not specs:
            return None, False
        current = set(self.views[owner])
        if len({s[0] for s in specs}) != len(specs):
            return None, False
        for fid, epoch, op_uid, mode in specs:
            f = self.frames[fid]
            if (op_uid != uid or mode not in ('read', 'write')
                    or fid not in current or f.epoch != epoch or not f.mapped
                    or not f.ready or owner not in f.refs or f.quarantine
                    or f.tx is not None or f.state not in ('PUBLISHED', 'IN_USE')):
                return None, False
            if mode == 'write':
                if f.refs != {owner} or f.pins or f.bindings or f.u:
                    return None, False
            elif any(k[2] == 'write' for k in f.u):
                return None, False
        return self.install_operation(oid, desc, specs, None, writes, None)

    def register_prepare(self, t, uid, specs, writes=(), copy_pair=None):
        if type(uid) is not int or uid < 0:
            return None, False
        oid = ('prepare', t.tid, uid)
        desc = self.descriptor(('tx', t.tid), specs, writes, copy_pair)
        if oid in self.operations:
            return self.prior(oid, desc), False
        if (t.status != 'PREPARING' or not self.live.get(t.owner, False)
                or not specs or len({s[0] for s in specs}) != len(specs)):
            return None, False
        for fid, epoch, op_uid, mode in specs:
            f = self.frames[fid]
            if (epoch != f.epoch or op_uid != uid or not f.mapped
                    or f.quarantine or mode not in ('read', 'write')):
                return None, False
            if mode == 'write':
                if f.tx != t.tid or f.refs or f.u:
                    return None, False
            elif (fid not in t.sources or t.tid not in f.pins or not f.ready
                  or any(k[2] == 'write' for k in f.u)):
                return None, False
        return self.install_operation(oid, desc, specs, t.tid, writes, copy_pair)

    def complete(self, ev, success=True, last=True, fault_local_only=False):
        if not last:
            return False
        op = self.operations.get(ev['oid'])
        if fault_local_only:
            # Fault: select a current operation by local uid and slot/mode only.
            wanted = tuple((s[0], s[3]) for s in ev['specs'])
            op = next((r for r in self.operations.values()
                       if r.status == 'PENDING' and r.oid[-1] == ev['uid']
                       and tuple((s[0], s[3]) for s in r.specs) == wanted), None)
        else:
            if (op is None or op.desc != ev['desc'] or op.specs != ev['specs']
                    or op.tx != ev['tx']):
                return False
        if op is None or op.status != 'PENDING':
            return False
        keys = []
        for fid, epoch, uid, mode in op.specs:
            f = self.frames[fid]
            key = (op.oid, epoch, mode)
            if f.epoch != epoch or key not in f.u:
                return False
            keys.append((fid, key))
        # Validate every page first, then settle the whole operation atomically.
        for fid, key in keys:
            self.frames[fid].u.remove(key)
            self.frames[fid].quarantine.discard(op.oid)
        op.status = 'DONE'
        if op.tx is not None:
            t = self.transactions[op.tx]
            t.jobs.remove(op.oid)
            if success and t.status == 'PREPARING':
                for fid, _, _, mode in op.specs:
                    if mode == 'write':
                        self.frames[fid].ready = True
            if not success:
                t.failed = True
                self.abort(t)
            elif t.status == 'ABORTING' and not t.jobs:
                self.finalize_abort(t)
        for fid, _ in keys:
            self.refresh(fid)
        if self.forget_done:
            del self.operations[op.oid]
        return True

    def publish(self, t):
        if (t.status != 'PREPARING' or t.failed or t.jobs
                or any(not self.frames[f].ready for f in t.new)):
            return False
        if (not self.live.get(t.owner, False)
                or self.versions.get(t.owner, 0) != t.version
                or self.views.get(t.owner, ()) != t.old
                or (t.owner in self.views) != t.old_exists):
            self.abort(t)
            return False
        for fid in t.new:
            self.frames[fid].refs.add(t.owner)
        self.views[t.owner] = tuple(t.new)
        self.versions[t.owner] = t.version + 1
        # Publication deliberately does not set live=True.
        for fid in t.old:
            self.frames[fid].refs.remove(t.owner)
        for fid in t.sources:
            self.frames[fid].pins.remove(t.tid)
        for fid in t.new:
            self.frames[fid].tx = None
        t.status = 'COMMITTED'
        for fid in set(t.old) | set(t.new) | t.sources:
            self.refresh(fid)
        return True

    def abort(self, t, fault_leak=False):
        if t.status in ('ABORTED', 'COMMITTED'):
            return
        t.status = 'ABORTING'
        self.claimed -= t.promise
        t.promise = 0
        for fid in t.new:
            self.refresh(fid)
        if not t.jobs and not fault_leak:
            self.finalize_abort(t)

    def finalize_abort(self, t):
        assert t.status == 'ABORTING' and not t.jobs
        for fid in t.new:
            self.frames[fid].tx = None
        for fid in t.sources:
            self.frames[fid].pins.remove(t.tid)
        t.status = 'ABORTED'
        for fid in set(t.new) | t.sources:
            self.refresh(fid)

    def share(self, source_owner, new_owner):
        assert self.live.get(source_owner, False)
        assert new_owner not in self.live
        view = self.views[source_owner]
        assert all(not any(k[2] == 'write' for k in self.frames[f].u)
                   for f in view)
        self.live[new_owner] = True
        self.versions[new_owner] = 0
        self.views[new_owner] = view
        for fid in view:
            self.frames[fid].refs.add(new_owner)
            self.refresh(fid)

    def retire(self, owner, fault_skip_unpublished=False):
        if fault_skip_unpublished and owner not in self.views:
            return
        if owner in self.live and not self.live[owner]:
            return
        self.live[owner] = False  # Also creates a tombstone for an unknown owner.
        self.versions[owner] = self.versions.get(owner, 0) + 1
        old = self.views.pop(owner, ())
        for fid in old:
            self.frames[fid].refs.remove(owner)
        for t in self.transactions.values():
            if t.owner == owner and t.status == 'PREPARING':
                self.abort(t)
        for fid in old:
            self.refresh(fid)


class World:
    '''Independent accepted-event history; retries never append an event.'''
    def __init__(self, allocator):
        self.a = allocator
        self.events = []

    def receive(self, op, fresh):
        if op is None:
            return None
        if not fresh:
            return next(e for e in self.events if e['oid'] == op.oid)
        ev = dict(oid=op.oid, uid=op.oid[-1], desc=op.desc, specs=op.specs,
                  tx=op.tx, writes=op.writes, copy_pair=op.copy_pair, pending=True)
        self.events.append(ev)
        return ev

    def access(self, owner, uid, mode='read', value=None):
        view = self.a.views.get(owner, ())
        specs = tuple((fid, self.a.frames[fid].epoch, uid, mode) for fid in view)
        if not specs:
            return None
        writes = tuple((fid, value) for fid in view) if mode == 'write' else ()
        return self.receive(*self.a.register_public(owner, uid, specs, writes))

    def retry(self, ev, desc=None):
        if desc is None:
            desc = ev['desc']
        op = self.a.retry(ev['oid'], desc)
        return self.receive(op, False)

    def prepare(self, t, uid, values=None, copy=False):
        if not t.new:
            return None
        specs = []
        pair = None
        if copy:
            assert len(t.sources) == len(t.new) == 1
            src = next(iter(t.sources))
            pair = (src, t.new[0])
            specs.append((src, self.a.frames[src].epoch, uid, 'read'))
            writes = ()
        else:
            assert values is not None and len(values) == len(t.new)
            writes = tuple(zip(t.new, values))
        specs.extend((fid, self.a.frames[fid].epoch, uid, 'write') for fid in t.new)
        return self.receive(*self.a.register_prepare(t, uid, tuple(specs), writes, pair))

    def finish(self, ev, success=True):
        assert ev is not None and ev['pending']
        # A real physical write does not consult allocator epochs before writing.
        if success:
            if ev['copy_pair'] is not None:
                src, dst = ev['copy_pair']
                self.a.frames[dst].value = self.a.frames[src].value
            for fid, value in ev['writes']:
                self.a.frames[fid].value = value
        ev['pending'] = False
        return self.a.complete(ev, success)


def audit_pending(w):
    '''Reference responsibilities reconstructed only from actual pending events.'''
    expected = {fid: set() for fid in w.a.frames}
    for ev in w.events:
        if ev['pending']:
            for fid, epoch, uid, mode in ev['specs']:
                f = w.a.frames[fid]
                assert f.epoch == epoch, 'reuse while old operation pending'
                assert f.mapped and f.state != 'CACHED_FREE'
                key = (ev['oid'], epoch, mode)
                assert key not in expected[fid], 'duplicate pending identity'
                expected[fid].add(key)
    for fid, f in w.a.frames.items():
        assert f.u == expected[fid], 'event responsibility mismatch'


def audit_values(w, views, values):
    for owner, expected in values.items():
        actual = tuple(w.a.frames[f].value for f in views[owner])
        assert actual == tuple(expected), (owner, actual, expected)


def audit(w, views, values=None, open_owners=None, closed=()):
    a = w.a
    assert a.views == views, (a.views, views)
    actual_open = {o for o, live in a.live.items() if live}
    assert set(views) <= actual_open
    if open_owners is not None:
        assert actual_open == set(open_owners), (actual_open, open_owners)
    for owner in closed:
        assert owner in a.live and a.live[owner] is False, 'owner not CLOSED'
    refs = {fid: set() for fid in a.frames}
    for owner, view in views.items():
        for fid in view:
            refs[fid].add(owner)
    audit_pending(w)
    for fid, f in a.frames.items():
        assert f.refs == refs[fid], 'logical reference mismatch'
        if f.state == 'CACHED_FREE':
            assert not (f.refs or f.pins or f.u or f.bindings or f.quarantine)
            assert f.tx is None and not f.mapped and not f.ready
        if f.refs:
            assert f.ready and f.mapped
            assert f.state in ('PUBLISHED', 'IN_USE', 'QUARANTINED')
    if values is not None:
        assert set(values) == set(views)
        audit_values(w, views, values)
    accepted = {}
    for ev in w.events:
        assert ev['oid'] not in accepted, 'same full identity submitted twice'
        accepted[ev['oid']] = ev
        op = a.operations.get(ev['oid'])
        assert op is not None, 'accepted operation record forgotten'
        assert op.desc == ev['desc'] and op.specs == ev['specs']
        assert (op.status == 'PENDING') == ev['pending'], 'operation terminal mismatch'
    assert set(a.operations) == set(accepted)
    for t in a.transactions.values():
        expected_jobs = {ev['oid'] for ev in w.events
                         if ev['pending'] and ev['tx'] == t.tid}
        assert t.jobs == expected_jobs
        if t.status == 'PREPARING':
            assert a.live.get(t.owner, False), 'closed owner retains preparing transaction'
        if t.status == 'ABORTING':
            assert t.jobs, 'aborted reservation leaked without an outstanding job'
        if t.status in ('ABORTED', 'COMMITTED'):
            assert not t.jobs and t.promise == 0
            assert all(t.tid not in f.pins and f.tx != t.tid for f in a.frames.values())
    assert a.claimed == sum(t.promise for t in a.transactions.values())
    assert a.claimed >= 0 and a.physical() + a.claimed <= a.cap


def create(w, owner, values, uid):
    t = w.a.reserve(owner, len(values))
    assert t is not None and t.status == 'PREPARING'
    ev = w.prepare(t, uid, values=values)
    if ev is not None:
        assert not w.a.publish(t)
        assert w.finish(ev)
    assert w.a.publish(t)
    return tuple(t.new)


def check_arithmetic():
    def waste(regions, g):
        if g <= 0 or any(r < 0 for r in regions):
            raise ValueError('invalid region or granularity')
        return sum(((r + g - 1) // g) * g - r for r in regions)
    cases = 0
    for g in range(1, 17):
        for r in range(65):
            reference = sum(g for _ in range(0, r, g)) - r
            assert waste([r], g) == reference
            cases += 1
    assert waste([], G) == 0 and waste([0, G, 2 * G], G) == 0
    regions = [31 * 2 * 1024] * 64
    assert waste(regions, G) == 128 * 1024
    assert waste(regions, 2 * 1024 * 1024) == 130154496
    rejected('aggregate region rounding',
             lambda: equal(waste([sum(regions)], G), 128 * 1024))
    assert (((31 + 15) // 16) * 16 - 31) * 128 * 1024 == 128 * 1024
    rejected('floor block count', lambda: equal((31 // 16) * 16 >= 31, True))
    print('CPU arithmetic:', cases, 'independent unit-enumeration cases; empty and exact pages')


def check_admission():
    w = World(Allocator(2))
    a = w.a
    assert a.physical() == 128 * 1024 and a.physical() + G > a.cap
    t = a.reserve('A', 1)
    assert t is not None and a.physical() == 2 * G and a.claimed == 0
    assert sum(f.state == 'CACHED_FREE' for f in a.frames.values()) == 1
    ev = w.prepare(t, 1, values=[7])
    assert w.finish(ev) and a.publish(t)
    audit(w, {'A': tuple(t.new)}, {'A': [7]})
    b = Allocator(2)
    for f in b.frames.values():
        f.tag = 'incompatible'
    assert b.reserve('B', 1) is None
    c = Allocator(2, 0)
    cw = World(c)
    tc = c.reserve('C', 2)
    assert tc is not None and c.physical() == 2 * G and c.claimed == 0
    c.abort(tc)
    audit(cw, {}, open_owners={'C'})
    assert all(f.state == 'CACHED_FREE' for f in c.frames.values())
    assert c.reserve('D', 1, margin=1) is None
    z = World(Allocator(0))
    assert create(z, 'empty', [], 1) == ()
    audit(z, {'empty': ()}, {'empty': []})
    print('CPU admission: full cached pool reused; incompatible backing and extra margin rejected; zero demand')


def check_repeated_and_shared():
    w = World(Allocator(2))
    a = w.a
    p = create(w, 'A', [7], 1)
    e1, e2 = w.access('A', 2), w.access('A', 3)
    assert e1 is not None and e2 is not None
    assert w.finish(e1)
    assert a.frames[p[0]].state == 'IN_USE'
    audit(w, {'A': p}, {'A': [7]})
    assert w.finish(e2)
    assert a.frames[p[0]].state == 'PUBLISHED' and not a.complete(e2)
    e3 = w.access('A', 4)
    assert w.finish(e3)
    a.share('A', 'B')
    assert a.physical() == 2 * G
    assert sum(f.state != 'CACHED_FREE' for f in a.frames.values()) * G == G
    a.retire('A')
    assert w.access('A', 5) is None
    eb = w.access('B', 6)
    assert eb is not None
    audit(w, {'B': p}, {'B': [7]}, closed={'A'})
    assert w.finish(eb)
    a.retire('B')
    audit(w, {}, open_owners=set(), closed={'A', 'B'})
    print('CPU ownership: successive fresh operations, multiple readers, shared-owner retirement')


def check_operation_identity():
    w = World(Allocator(1))
    a = w.a
    p = create(w, 'A', [7], 1)
    epoch = a.frames[p[0]].epoch
    old = w.access('A', 42)
    submitted = len(w.events)
    assert w.access('A', 42) is old and w.retry(old) is old
    assert len(w.events) == submitted
    assert not a.complete(old, last=False)
    forged = dict(old, desc=('wrong-description',))
    assert not a.complete(forged)
    forged_specs = dict(old, specs=((p[0], epoch + 1, 42, 'read'),))
    assert not a.complete(forged_specs)
    audit(w, {'A': p}, {'A': [7]})
    assert w.finish(old)
    assert a.frames[p[0]].epoch == epoch and a.live['A']
    assert w.access('A', 42) is old and w.retry(old) is old
    assert w.access('A', 42, 'write', 9) is None
    assert w.retry(old, desc=('changed-input',)) is None
    assert len(w.events) == submitted and not a.complete(old)
    current = w.access('A', 43)
    assert current is not None and current is not old
    a.share('A', 'B')
    same_local = w.access('B', 42)
    assert same_local is not None and same_local['oid'] != old['oid']
    assert not a.complete(old)
    audit(w, {'A': p, 'B': p}, {'A': [7], 'B': [7]})
    a.retire('A')
    assert w.retry(old) is old and not a.complete(old)
    audit(w, {'B': p}, {'B': [7]}, closed={'A'})
    assert w.finish(current) and w.finish(same_local)
    a.retire('B')
    audit(w, {}, closed={'A', 'B'})
    # Reproduce the reviewed same-epoch failure through the faulty retention policy.
    bad = World(Allocator(1, forget_done=True))
    bp = create(bad, 'A', [7], 1)
    be = bad.a.frames[bp[0]].epoch
    old_bad = bad.access('A', 42)
    assert bad.finish(old_bad)
    new_bad = bad.access('A', 42)
    assert new_bad is not old_bad and new_bad['pending']
    assert bad.a.frames[bp[0]].epoch == be
    audit_pending(bad)  # Responsibility exists before replay of the old certificate.
    rejected('forgotten terminal identity permits a second actual submission',
             lambda: audit(bad, {'A': bp}, {'A': [7]}))
    assert bad.a.complete(old_bad)
    assert new_bad['pending']
    rejected('same-epoch old certificate removes a different accepted event obligation',
             lambda: audit_pending(bad))
    bad.a.retire('A')
    rejected('replayed completion permits reclaim with a real pending read',
             lambda: audit_pending(bad))
    print('CPU operation identity: pending/DONE retries submit once; same epoch replay preserves fresh and other-owner operations')


def check_first_publish_retirement():
    for stage in ('acquired', 'initializing', 'ready'):
        w = World(Allocator(1))
        a = w.a
        t = a.reserve('A', 1, pause_after_acquire=(stage == 'acquired'))
        p = t.new[0]
        ev = None
        assert a.live['A'] and 'A' not in a.views
        if stage == 'acquired':
            assert a.frames[p].state == 'RESERVED' and not a.frames[p].mapped
        else:
            ev = w.prepare(t, 10, values=[7])
            assert ev is not None
            if stage == 'ready':
                assert w.finish(ev)
                assert a.frames[p].ready
        bad = deepcopy(w)
        bad_t = bad.a.transactions[t.tid]
        # Same public RetireOwner input, intentionally skipped without a visible view.
        bad.a.retire('A', fault_skip_unpublished=True)
        rejected('RetireOwner skips unpublished owner at ' + stage,
                 lambda: audit(bad, {}, open_owners=set(), closed={'A'}))
        if stage == 'initializing':
            bad_ev = next(e for e in bad.events if e['oid'] == ev['oid'])
            assert bad.finish(bad_ev)
        if stage != 'acquired':
            assert bad.a.publish(bad_t)
            rejected('ignored retirement later publishes at ' + stage,
                     lambda: audit(bad, {}, open_owners=set(), closed={'A'}))
        a.retire('A')
        audit(w, {}, open_owners=set(), closed={'A'})
        assert not a.publish(t) and a.reserve('A', 1) is None
        if stage == 'initializing':
            assert t.status == 'ABORTING'
            assert a.frames[p].mapped and a.frames[p].u
            assert a.frames[p].state == 'DRAINING' and ev['pending']
            assert a.reserve('B', 1) is None
            assert w.finish(ev)
        assert t.status == 'ABORTED'
        assert a.frames[p].state == 'CACHED_FREE' and not a.frames[p].mapped
        assert not a.publish(t) and a.reserve('A', 1) is None
        if ev is not None:
            assert not a.complete(ev)
        a.retire('A')
        audit(w, {}, open_owners=set(), closed={'A'})
        q = create(w, 'B', [5], 20)
        audit(w, {'B': q}, {'B': [5]}, open_owners={'B'}, closed={'A'})
    unknown = World(Allocator(1))
    unknown.a.retire('not-yet-reserved')
    assert unknown.a.reserve('not-yet-reserved', 1) is None
    audit(unknown, {}, open_owners=set(), closed={'not-yet-reserved'})
    print('CPU first publication: acquired, in-flight initialization, ready-before-publish retirement; CLOSED never reopens')


def check_partial_failures():
    for stage in ('acquire', 'map'):
        w = World(Allocator(3))
        p = create(w, 'A', [7], 1)
        t = w.a.reserve('B', 2, fail_at=(stage, 1))
        assert t.status == 'ABORTED' and len(t.new) >= 1
        assert any(x[0] == stage and x[1] == t.tid for x in w.a.trace)
        audit(w, {'A': p}, {'A': [7]})
        assert sum(f.state == 'CACHED_FREE' for f in w.a.frames.values()) == 2
    bad = World(Allocator(3))
    p = create(bad, 'A', [7], 1)
    bad.a.reserve('B', 2, fail_at=('acquire', 1), fault_leak=True)
    rejected('leak after actual first acquisition',
             lambda: audit(bad, {'A': p}, {'A': [7]}))
    w = World(Allocator(3))
    p = create(w, 'A', [7], 1)
    t = w.a.reserve('B', 2)
    ev = w.prepare(t, 2, values=[8, 8])
    w.a.abort(t)
    assert t.status == 'ABORTING'
    assert all(w.a.frames[f].mapped and w.a.frames[f].state == 'DRAINING' for f in t.new)
    assert w.a.reserve('C', 1) is None
    audit(w, {'A': p}, {'A': [7]})
    assert w.finish(ev, success=False) and t.status == 'ABORTED'
    audit(w, {'A': p}, {'A': [7]})
    print('CPU failure injection: first acquired page, first mapped page, accepted initialization')


def check_cow_cancel_publish():
    w = World(Allocator(2))
    a = w.a
    p = create(w, 'A', [7], 1)
    a.share('A', 'B')
    views, values = {'A': p, 'B': p}, {'A': [7], 'B': [7]}
    t = a.reserve('A', 1, cow=True)
    ev = w.prepare(t, 10, copy=True)
    assert w.prepare(t, 10, copy=True) is ev
    assert not a.publish(t)
    audit(w, views, values)
    faulty = deepcopy(w)
    faulty.a.frames[p[0]].refs.remove('A')
    rejected('COW debits old reference before publish', lambda: audit(faulty, views, values))
    a.abort(t)
    assert t.status == 'ABORTING' and a.frames[t.new[0]].mapped
    assert a.frames[p[0]].pins == {t.tid}
    audit(w, views, values)
    assert w.finish(ev) and t.status == 'ABORTED'
    assert w.retry(ev) is ev and not a.complete(ev)
    audit(w, views, values)
    t = a.reserve('A', 1, cow=True)
    ev = w.prepare(t, 11, copy=True)
    assert w.finish(ev)
    a.abort(t)
    assert not a.publish(t)
    audit(w, views, values)
    old_reader = w.access('A', 12)
    t = a.reserve('A', 1, cow=True)
    ev = w.prepare(t, 13, copy=True)
    assert w.finish(ev) and a.publish(t)
    q = tuple(t.new)
    audit(w, {'A': q, 'B': p}, {'A': [7], 'B': [7]})
    write = w.access('A', 14, 'write', 9)
    assert write is not None and w.finish(write)
    audit(w, {'A': q, 'B': p}, {'A': [9], 'B': [7]})
    bad = deepcopy(w)
    bad.a.frames[p[0]].value = 9
    rejected('direct shared-page write corrupts B',
             lambda: audit(bad, {'A': q, 'B': p}, {'A': [9], 'B': [7]}))
    a.retire('B')
    assert a.frames[p[0]].state == 'DRAINING'
    audit(w, {'A': q}, {'A': [9]})
    assert w.finish(old_reader)
    assert a.frames[p[0]].state == 'CACHED_FREE'
    assert w.finish(w.access('A', 15))
    audit(w, {'A': q}, {'A': [9]})
    print('CPU COW: preparation retry, both cancellation stages, atomic commit, isolation and old-reader drain')


def check_version_conflict():
    w = World(Allocator(3))
    a = w.a
    p = create(w, 'A', [7], 1)
    t1, t2 = a.reserve('A', 1, cow=True), a.reserve('A', 1, cow=True)
    e1, e2 = w.prepare(t1, 2, copy=True), w.prepare(t2, 3, copy=True)
    assert w.finish(e1) and w.finish(e2)
    assert a.publish(t1) and not a.publish(t2)
    assert t2.status == 'ABORTED'
    audit(w, {'A': tuple(t1.new)}, {'A': [7]})
    assert a.frames[p[0]].state == 'CACHED_FREE'
    print('CPU publication: only one replacement of a shared base version commits')


def check_stale_cross_epoch():
    w = World(Allocator(1))
    a = w.a
    create(w, 'A', [7], 1)
    old = w.access('A', 42)
    assert w.finish(old)
    a.retire('A')
    p = create(w, 'B', [5], 2)
    current = w.access('B', 42)
    assert current is not None
    good, bad = deepcopy(w), deepcopy(w)
    assert not good.a.complete(old)
    audit(good, {'B': p}, {'B': [5]})
    assert bad.a.complete(old, fault_local_only=True)
    rejected('local-uid slot-only completion ignores operation scope and allocation epoch',
             lambda: audit_pending(bad))
    print('CPU stale completion: same old certificate preserves current event with full identity checks')


def check_late_physical_write():
    w = World(Allocator(1))
    a = w.a
    p = create(w, 'A', [99], 1)
    old_write = w.access('A', 90, 'write', 7)
    assert old_write is not None
    a.retire('A')
    assert a.frames[p[0]].state == 'DRAINING'
    audit(w, {}, closed={'A'})
    assert a.reserve('B', 1) is None
    bad = deepcopy(w)
    f = bad.a.frames[p[0]]
    f.u.clear()  # Fault: discard a real writer's resource responsibility.
    bad.a.refresh(f.fid)
    rejected('early reclaim while a physical writer remains pending', lambda: audit_pending(bad))
    q = create(bad, 'B', [5], 2)
    assert q == p and bad.a.frames[q[0]].value == 5
    old_bad = next(e for e in bad.events if e['oid'] == old_write['oid'])
    assert not bad.finish(old_bad)
    assert bad.a.frames[q[0]].value == 7
    rejected('old physical write actually changes the new contents from 5 to 7',
             lambda: audit_values(bad, {'B': q}, {'B': [5]}))
    assert w.finish(old_write)
    safe_q = create(w, 'B', [5], 2)
    audit(w, {'B': safe_q}, {'B': [5]})
    print('CPU physical-write model: waiting retains new 5; premature reuse yields 7; epoch is no write barrier')


def check_binding_and_quarantine():
    w = World(Allocator(1))
    a = w.a
    p = create(w, 'A', [7], 1)
    f = a.frames[p[0]]
    f.bindings.add('graph-address')
    a.retire('A')
    audit(w, {}, closed={'A'})
    assert a.reserve('B', 1) is None
    f.bindings.clear()
    a.refresh(f.fid)
    create(w, 'B', [5], 2)
    ev = w.access('B', 3)
    a.retire('B')
    f.quarantine.update({ev['oid'], 'independent-cause'})
    a.refresh(f.fid)
    assert f.state == 'QUARANTINED' and a.reserve('C', 1) is None
    audit(w, {}, closed={'A', 'B'})
    assert w.finish(ev)
    assert f.quarantine == {'independent-cause'} and f.state == 'QUARANTINED'
    assert a.reserve('C', 1) is None
    # Explicit external evidence is abstracted as removal of this separate cause.
    f.quarantine.remove('independent-cause')
    a.refresh(f.fid)
    assert f.state == 'CACHED_FREE'
    audit(w, {}, closed={'A', 'B'})
    print('CPU gates: binding retains backing; completion clears only its matching quarantine cause')


def main():
    check_arithmetic()
    check_admission()
    check_repeated_and_shared()
    check_operation_identity()
    check_first_publish_retirement()
    check_partial_failures()
    check_cow_cancel_publish()
    check_version_conflict()
    check_stale_cross_epoch()
    check_late_physical_write()
    check_binding_and_quarantine()
    print('Scope: finite serial CPU model, fixed whole-page scalar accesses, retained terminal records, integer arithmetic; no GPU measurement.')


if __name__ == '__main__':
    main()
