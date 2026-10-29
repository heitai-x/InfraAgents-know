#!/usr/bin/env python3
"""K-ALG-013: deterministic, offline, read-only CPU mechanism checks.

No project imports, files, subprocesses, network, GPU, real attention or timing.
Intervals are tested against an independent cell-set event auditor.
Physical lifetime proofs below are simulated events, not device guarantees.
"""

from itertools import product


class ProtocolError(Exception):
    pass


def cover(ranges, start, end):
    if start < 0 or start > end:
        return False
    ranges = list(ranges)
    if any(a < 0 or a > b for a, b in ranges):
        return False
    cursor = start
    for a, b in sorted(ranges):
        if b <= cursor:
            continue
        if a > cursor:
            return False
        cursor = max(cursor, b)
    return cursor >= end


def reject(label, action, error=ProtocolError):
    try:
        action()
    except error:
        return label
    raise AssertionError('fault was not detected: ' + label)


class Handoff:
    def __init__(self, n, h, a, layers=2, generation=7):
        if not (0 < n and 0 <= h <= n and 0 <= a <= n and layers > 0):
            raise ValueError('invalid bounds')
        self.n, self.h, self.a = n, h, a
        self.layers, self.g = layers, generation
        self.source = {l: [(0, a)] for l in range(layers)}
        self.target = {l: [(0, h)] for l in range(layers)}
        self.sent = {l: [] for l in range(layers)}
        self.token = None
        self.samples = 0
        self.committed = False
        self.visible = {}
        self.events = []

    def token_ready(self, token, sample=False):
        if self.token is not None:
            if self.token != token:
                raise ProtocolError('conflicting confirmed token')
            return
        self.token = token
        self.samples += int(sample)
        self.events.append(('TOKEN', token))

    def prepare(self, b, permitted=True):
        if not permitted:
            raise ProtocolError('no KV preparation path')
        if not self.a <= b <= self.n:
            raise ProtocolError('invalid source acquisition')
        for l in range(self.layers):
            self.source[l].append((self.a, b))
            self.events.append(('LOAD', l, self.a, b))
        for i in range(b, self.n):
            for l in range(self.layers):
                if not cover(self.source[l], 0, i):
                    raise ProtocolError('missing compute history')
                self.events.append(('CALC', l, i))
                self.source[l].append((i, i + 1))
        # No sampling or output is performed by KV preparation.

    def send(self, l, a, b, unsafe=False):
        if not (l in self.source and self.h <= a <= b <= self.n):
            raise ProtocolError('bad send range')
        if not unsafe and not cover(self.source[l], a, b):
            raise ProtocolError('send without source producer')
        self.sent[l].append((a, b))
        self.events.append(('SEND', l, a, b))

    def complete(self, l, a, b, generation=None):
        generation = self.g if generation is None else generation
        if generation != self.g:
            raise ProtocolError('stale completion')
        if l not in self.sent or not cover(self.sent[l], a, b):
            raise ProtocolError('visibility without accepted send')
        self.target[l].append((a, b))
        self.events.append(('VISIBLE', l, a, b))

    def commit(self, unsafe=False):
        if not unsafe and not all(cover(self.target[l], 0, self.n)
                                  for l in range(self.layers)):
            raise ProtocolError('commit missing layer coverage')
        self.committed = True
        self.events.append(('COMMIT',))

    def decode(self, unsafe=False):
        if not unsafe and (self.token is None or not self.committed):
            raise ProtocolError('decode dependency absent')
        self.events.append(('DECODE',))
        return self.token

    def emit(self, policy):
        if self.token is None:
            raise ProtocolError('output without token')
        if policy == 'GATED' and not self.committed:
            raise ProtocolError('gated output before joint readiness')
        if 0 in self.visible:
            return False
        self.visible[0] = self.token
        self.events.append(('OUTPUT', policy, self.token))
        return True


def audit_handoff(m):
    """Independent reference: exact per-layer integer-position sets.

    Does not call cover(), prepare(), commit() or decode().
    Reconstructs provenance and legal visibility from the event trace.
    """
    src = {l: set(range(m.a)) for l in range(m.layers)}
    dst = {l: set(range(m.h)) for l in range(m.layers)}
    sent = {l: set() for l in range(m.layers)}
    all_positions = set(range(m.n))
    token, committed, outputs = None, False, 0
    for e in m.events:
        kind = e[0]
        if kind == 'TOKEN':
            assert token is None or token == e[1], 'different token'
            token = e[1]
        elif kind == 'LOAD':
            _, l, a, b = e
            assert 0 <= a <= b <= m.n, 'bad load'
            src[l].update(range(a, b))
        elif kind == 'CALC':
            _, l, i = e
            assert set(range(i)) <= src[l], 'calculation lacks history'
            if l:
                assert i in src[l - 1], 'previous layer not produced'
            src[l].add(i)
        elif kind == 'SEND':
            _, l, a, b = e
            needed = set(range(a, b))
            assert needed <= src[l], 'send has no source KV'
            sent[l].update(needed)
        elif kind == 'VISIBLE':
            _, l, a, b = e
            needed = set(range(a, b))
            assert needed <= sent[l], 'target data has no send'
            dst[l].update(needed)
        elif kind == 'COMMIT':
            assert all(all_positions <= dst[l] for l in dst), 'partial commit'
            committed = True
        elif kind == 'DECODE':
            assert token is not None and committed, 'premature decode'
            assert all(all_positions <= dst[l] for l in dst), 'missing read'
        elif kind == 'OUTPUT':
            assert token is not None and e[2] == token, 'bad output'
            assert e[1] == 'EARLY' or committed, 'premature gated output'
            outputs += 1
            assert outputs == 1, 'duplicate logical output'
        else:
            raise AssertionError('unknown event')
    return src, dst


class Arena:
    """Range owners plus independently identified accepted physical users."""
    def __init__(self):
        self.owners = {}
        self.pending = {}
        self.done = set()
        self.closed = set()
        self.events = []

    def allocate(self, resource, owner):
        if resource in self.owners:
            raise ProtocolError('resource not free')
        self.owners[resource] = owner
        self.events.append(('ALLOC', resource, owner))

    def accept(self, op, attempt, resources):
        if attempt in self.closed or op in self.pending or op in self.done:
            raise ProtocolError('closed or reused operation identity')
        snapshot = {r: self.owners[r] for r in resources}
        self.pending[op] = (attempt, snapshot)
        self.events.append(('ACCEPT', op, attempt, snapshot))

    def abort(self, attempt):
        self.closed.add(attempt)
        self.events.append(('ABORT', attempt))

    def access(self, op):
        if op not in self.pending:
            raise ProtocolError('operation no longer has access')
        _, snapshot = self.pending[op]
        actual = {r: self.owners.get(r) for r in snapshot}
        self.events.append(('ACCESS', op, actual))

    def settle(self, op):
        if op in self.done:
            return False
        if op not in self.pending:
            raise ProtocolError('unknown operation completion')
        self.pending.pop(op)
        self.done.add(op)
        self.events.append(('TERMINAL', op))
        return True

    def fallback(self, attempt):
        if any(a == attempt for a, _ in self.pending.values()):
            raise ProtocolError('fallback overlaps undrained old attempt')
        self.events.append(('FALLBACK', attempt))

    def release(self, resource, unsafe=False):
        if resource not in self.owners:
            raise ProtocolError('not allocated')
        if not unsafe and any(resource in s for _, s in self.pending.values()):
            raise ProtocolError('physical users still live')
        self.events.append(('FREE', resource, self.owners[resource]))
        del self.owners[resource]


def audit_lifetime(events):
    """Rebuild last-user obligations without using Arena's release guard."""
    owners, accepted, closed = {}, {}, set()
    for e in events:
        if e[0] == 'ALLOC':
            assert e[1] not in owners, 'double allocation'
            owners[e[1]] = e[2]
        elif e[0] == 'ACCEPT':
            _, op, attempt, snapshot = e
            assert attempt not in closed and op not in accepted
            assert all(owners.get(r) == owner for r, owner in snapshot.items())
            accepted[op] = (attempt, dict(snapshot))
        elif e[0] == 'ABORT':
            closed.add(e[1])
        elif e[0] == 'ACCESS':
            _, op, actual = e
            assert op in accepted, 'access after terminal proof'
            assert actual == accepted[op][1], 'old access touches new owner'
        elif e[0] == 'TERMINAL':
            assert e[1] in accepted, 'unmatched completion'
            del accepted[e[1]]
        elif e[0] == 'FREE':
            _, resource, owner = e
            assert owners.get(resource) == owner
            assert not any(resource in s for _, s in accepted.values()), 'free before last user'
            del owners[resource]
        elif e[0] == 'FALLBACK':
            assert not any(a == e[1] for a, _ in accepted.values()), 'fallback before drain'
        else:
            raise AssertionError('unknown lifetime event')
    return owners, accepted


def times(route, *, available=0, prefix=0, queue=0, recover=0,
          sample=0, kv_ready=0, transfer=0, convert=0,
          commit_existing=0, decode_queue=0, decode=10, send=5):
    if route == 'EXISTING_TOKEN':
        first = available
    elif route == 'VALID_LOGITS':
        first = available + sample
    elif route == 'RECOVER_LAST_ROW':
        first = prefix + queue + recover + sample
    elif route == 'PREFILL_SUFFIX':
        first = available + sample
    else:
        raise ValueError('unknown first-token route')
    commit = max(commit_existing, kv_ready + transfer + convert)
    start = max(first, commit) + decode_queue
    early = first + send
    gated = max(first, commit) + send
    end = start + decode
    return {'first': first, 'commit': commit, 'start': start,
            'early': early, 'gated': gated,
            'second_early': max(end, early) + send,
            'second_gated': max(end, gated) + send}


def audit_timed_decode(trace):
    """Independent event prerequisite oracle, not the time formula above."""
    ready = {}
    for name, timestamp in trace:
        if name == 'DECODE':
            assert 'TOKEN' in ready and 'STATE' in ready, 'absent timed prerequisite'
            assert timestamp >= ready['TOKEN'], 'decode before token restoration'
            assert timestamp >= ready['STATE'], 'decode before state commit'
        else:
            ready[name] = timestamp


class Pool:
    def __init__(self, total, active, receiving):
        self.total, self.active, self.receiving = total, active, receiving
        self.reservations = {}

    def estimate(self, need):
        return 0 <= need <= self.total - self.active - self.receiving

    def reserve(self, key, need):
        if key in self.reservations:
            if self.reservations[key] != need:
                raise ProtocolError('reservation conflict')
            return
        if not self.estimate(need):
            raise ProtocolError('capacity exceeded')
        self.receiving += need
        self.reservations[key] = need


def test_dependencies():
    # An independent cell-set reference checks interval union edge cases.
    interval_cases = 0
    for end in range(1, 7):
        candidates = [(a, b) for a in range(end + 1) for b in range(a, end + 1)]
        for left, right in product(candidates, repeat=2):
            ranges = [left, right]
            cells = set(range(*left)) | set(range(*right))
            for start in range(end + 1):
                assert cover(ranges, start, end) == (set(range(start, end)) <= cells)
                interval_cases += 1

    trajectories = 0
    for n in range(1, 9):
        for h in range(n + 1):
            for a in range(n + 1):
                for b in range(a, n + 1):
                    for route in ('EXISTING_TOKEN', 'VALID_LOGITS'):
                        m = Handoff(n, h, a)
                        m.token_ready('confirmed', sample=(route == 'VALID_LOGITS'))
                        before = m.samples
                        if h < n:
                            m.prepare(b)
                            assert m.samples == before
                            for l in range(m.layers):
                                m.send(l, h, n)
                                m.complete(l, h, n)
                        m.commit()
                        assert m.decode() == 'confirmed'
                        m.token_ready('confirmed', sample=True)
                        assert m.samples == before
                        audit_handoff(m)
                        trajectories += 1

    for route in ('EXISTING_TOKEN', 'VALID_LOGITS'):
        m = Handoff(8, 4, 0)
        m.token_ready('y8', sample=(route == 'VALID_LOGITS'))
        reject('ticket does not supply source KV', lambda: m.send(0, 4, 8))
        samples = m.samples
        m.prepare(0)
        for l in range(2):
            m.send(l, 4, 8)
            m.complete(l, 4, 8)
        m.commit()
        assert m.decode() == 'y8' and m.samples == samples
        audit_handoff(m)

    bad = Handoff(8, 4, 0)
    bad.token_ready('y8', sample=True)
    bad.send(0, 4, 8, unsafe=True)  # Faulty producer-dependency omission.
    reject('A05 mutant send audited', lambda: audit_handoff(bad), AssertionError)

    partial = Handoff(8, 4, 8)
    partial.token_ready('y8')
    partial.send(0, 4, 8)
    partial.complete(0, 4, 8)
    partial.send(1, 4, 6)
    partial.complete(1, 4, 6)
    reject('single-layer hole commit', partial.commit)
    partial.commit(unsafe=True)
    reject('partial commit mutant audited', lambda: audit_handoff(partial), AssertionError)

    full = Handoff(8, 8, 0)
    full.commit()
    reject('full KV without token', full.decode)
    reject('gated commit without token', lambda: full.emit('GATED'))
    full.decode(unsafe=True)
    reject('missing token mutant audited', lambda: audit_handoff(full), AssertionError)

    uncommitted = Handoff(8, 4, 8)
    uncommitted.token_ready('y8')
    reject('token without target commit', uncommitted.decode)
    reject('gated output before commit', lambda: uncommitted.emit('GATED'))
    assert uncommitted.emit('EARLY')
    assert not uncommitted.emit('EARLY')
    reject('conflicting token', lambda: uncommitted.token_ready('different'))
    reject('stale generation', lambda: uncommitted.complete(0, 4, 8, 6))
    denied = Handoff(8, 4, 0)
    reject('unpreparable KV candidate', lambda: denied.prepare(0, permitted=False))
    assert denied.events == []
    reject('empty prompt', lambda: Handoff(0, 0, 0), ValueError)
    reject('invalid target prefix', lambda: Handoff(8, 9, 0), ValueError)
    print('dependencies: %d interval comparisons; %d ticket/KV trajectories' %
          (interval_cases, trajectories))
    print('detected: absent source producer, partial-layer commit, missing token, early gated output')


def test_lifetimes():
    arena = Arena()
    for r in ('scratch', 'source', 'target'):
        arena.allocate(r, 'g1/tx1')
    arena.accept('load0', 'tx1', ('scratch',))
    arena.accept('load1', 'tx1', ('scratch',))
    arena.accept('send1', 'tx1', ('source', 'target'))
    arena.accept('convert1', 'tx1', ('target',))
    arena.settle('load0')  # Failed layer, with a terminal physical-access proof.
    arena.abort('tx1')
    reject('new post after abort', lambda: arena.accept('new', 'tx1', ('scratch',)))
    reject('early fallback', lambda: arena.fallback('tx1'))
    for r in ('scratch', 'source', 'target'):
        reject('protected range ' + r, lambda r=r: arena.release(r))
    arena.access('load1')  # Previously accepted write can still run while draining.
    arena.settle('load1')
    arena.release('scratch')
    arena.access('send1')
    arena.settle('send1')
    arena.release('source')
    reject('target still has converter', lambda: arena.release('target'))
    arena.access('convert1')
    arena.settle('convert1')
    arena.fallback('tx1')
    arena.release('target')
    arena.allocate('target', 'g2/tx2')
    arena.accept('send2', 'tx2', ('target',))
    assert not arena.settle('send1')  # Duplicate old completion cannot release send2.
    reject('new owner protected', lambda: arena.release('target'))
    arena.access('send2')
    arena.settle('send2')
    arena.release('target')
    owners, pending = audit_lifetime(arena.events)
    assert owners == {} and pending == {}

    bad = Arena()
    bad.allocate('page', 'g1')
    bad.accept('old_write', 'tx1', ('page',))
    bad.abort('tx1')
    bad.release('page', unsafe=True)  # Fault: logical cancel used as physical proof.
    bad.allocate('page', 'g2')
    bad.access('old_write')  # Old accepted write would touch the new allocation.
    reject('A06 reuse mutant audited', lambda: audit_lifetime(bad.events), AssertionError)
    print('lifetimes: abort retains readers/writers; last-user drain precedes fallback and reuse')
    print('detected: cancel-then-free mutant with delayed old write to reallocated page')


def test_costs_and_capacity():
    pairs = 0
    for n in range(1, 17):
        for b in range(n + 1):
            reference = sum(1 for i in range(b, n) for k in range(i + 1))
            assert n * (n + 1) // 2 - b * (b + 1) // 2 == reference
            pairs += 1
    b_tok = 2 * 32 * 8 * 128 * 2
    assert b_tok == 131072
    assert 4096 * b_tok == 512 * 1024 ** 2
    assert (4096 - 1024) * b_tok == 384 * 1024 ** 2
    assert (384 * 1024 ** 2) / (20 * 1024 ** 3) * 1000 == 18.75

    t = times('PREFILL_SUFFIX', available=90, kv_ready=90,
              transfer=25, convert=15)
    assert (t['first'], t['commit'], t['start']) == (90, 130, 130)
    assert (t['early'], t['gated'], t['second_early'], t['second_gated']) == (95, 135, 145, 145)
    assert t['second_early'] - t['early'] == 50
    assert t['second_gated'] - t['gated'] == 10

    r = times('RECOVER_LAST_ROW', prefix=0, queue=7, recover=13, sample=2)
    assert r['first'] == 22 and r['commit'] == 0 and r['start'] == 22
    audit_timed_decode([('TOKEN', 22), ('STATE', 0), ('DECODE', r['start'])])
    reject('C01 omitted recovery wait', lambda: audit_timed_decode(
        [('TOKEN', 22), ('STATE', 0), ('DECODE', 0)]), AssertionError)
    assert r['gated'] == 27  # Commit alone at t=0 does not permit output.

    for route in ('EXISTING_TOKEN', 'VALID_LOGITS'):
        fast = times(route, available=2, sample=1, kv_ready=30, transfer=5)
        slow = times(route, available=2, sample=1, kv_ready=60, transfer=5)
        assert fast['first'] == slow['first']
        assert fast['start'] == 35 and slow['start'] == 65
        audit_timed_decode([('TOKEN', slow['first']), ('STATE', 65), ('DECODE', slow['start'])])

    # Explicit DAG reference for recovery: resource queue starts after prefix readiness.
    ready = {'prefix': 4, 'queued': 0, 'recover': 0, 'sample': 0}
    ready['queued'] = ready['prefix'] + 7
    ready['recover'] = ready['queued'] + 13
    ready['sample'] = ready['recover'] + 2
    assert times('RECOVER_LAST_ROW', prefix=4, queue=7, recover=13,
                 sample=2)['first'] == ready['sample'] == 26

    pool = Pool(10, 4, 3)
    snapshot = (pool.receiving, dict(pool.reservations))
    assert [pool.estimate(x) for x in (2, 4, 1)] == [True, False, True]
    assert snapshot == (pool.receiving, pool.reservations)
    pool.reserve('chosen', 2)
    pool.reserve('chosen', 2)
    assert pool.receiving == 5 and len(pool.reservations) == 1
    reject('target overcommit', lambda: pool.reserve('overflow', 2))
    assert sum(pool.reservations.values()) == 2
    assert pool.active + pool.receiving <= pool.total
    print('costs: %d explicit QK enumerations; 128 KiB/token; 384 MiB suffix' % pairs)
    print('times: early/gated first=95/135 ms, second=145 ms; full-hit recovery gates decode at 22 ms')
    print('detected: omitted recovery/queue wait; ticket paths retain independent KV preparation time')
    print('capacity: estimates have no effects; chosen reservation is idempotent; overflow rejected')


def main():
    print('K-ALG-013: deterministic CPU dependency/lifetime/cost model; no GPU measurements')
    test_dependencies()
    test_lifetimes()
    test_costs_and_capacity()
    print('PASS: scoped mechanisms and injected faulty traces were checked against independent invariants')
    print('Not covered: real attention/logits, CUDA visibility, NIC cancellation, concurrency, throughput or SLO')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
