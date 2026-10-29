"""K-ENGINE-003: deterministic, standalone CPU symbolic checks.
Standard library only. No project import, subprocess, network or file access.
Execution results must be recorded externally against these exact script bytes.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import permutations, product


@dataclass(frozen=True)
class Request:
    rid: str
    gen: int
    branch: str
    start: int
    q: int
    tokens: tuple


def owner(r):
    return (r.rid, r.gen, r.branch)


def key(r, p):
    return owner(r) + (p,)


def validate_inputs(requests, outputs):
    table = {}
    for r in requests:
        if any(type(v) is not int for v in (r.gen, r.start, r.q)):
            raise ValueError('noninteger request field')
        if r.gen < 0 or r.start < 0 or r.q < 0:
            raise ValueError('negative request field')
        if r.start + r.q > len(r.tokens):
            raise ValueError('query exceeds known input')
        if owner(r) in table:
            raise ValueError('duplicate branch segment')
        table[owner(r)] = r
    events = set()
    for identity, local, role, ordinal in outputs:
        if identity not in table:
            raise ValueError('unknown output owner')
        r = table[identity]
        if type(local) is not int or not 0 <= local < r.q:
            raise ValueError('output outside query segment')
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError('invalid output ordinal')
        p = r.start + local
        if role == 'SAMPLE':
            if local != r.q - 1 or r.start + r.q != len(r.tokens):
                raise ValueError('sample before known frontier')
        elif role == 'LOGPROB':
            if p + 1 >= len(r.tokens):
                raise ValueError('logprob target not known')
        else:
            raise ValueError('unknown output role')
        event = (identity, p + 1, role, ordinal)
        if event in events:
            raise ValueError('duplicate output event')
        events.add(event)
    return table


def build_view(requests, outputs=(), capacity=None):
    table = validate_inputs(requests, outputs)
    active = [r for r in requests if r.q > 0]
    offsets = [0]
    for r in active:
        offsets.append(offsets[-1] + r.q)
    starts = {owner(r): offsets[i] for i, r in enumerate(active)}
    rows = []
    for i, r in enumerate(active):
        for j in range(r.q):
            p = r.start + j
            rows.append((offsets[i] + j, key(r, p), p, r.tokens[p]))
    targets = []
    for identity, local, role, ordinal in outputs:
        r = table[identity]
        p = r.start + local
        targets.append((starts[identity] + local, key(r, p), role,
                        p + 1, ordinal))
    total = offsets[-1]
    cap = total if capacity is None else capacity
    if type(cap) is not int or cap < total:
        raise ValueError('query capacity exceeded')
    return {'rows': rows, 'offsets': offsets, 'Q': total,
            'targets': targets,
            'owners': [row[1] for row in rows] + [None] * (cap - total)}


def reference_view(requests, outputs, capacity):
    # Independent reference: enumerate absolute positions, then locate sources.
    # It never uses candidate offsets, rows, targets or their inverse maps.
    rows = []
    offsets = [0]
    for r in requests:
        for p in range(r.start, r.start + r.q):
            rows.append((len(rows), (r.rid, r.gen, r.branch, p),
                         p, r.tokens[p]))
        if r.q > 0:
            offsets.append(len(rows))
    targets = []
    originals = {owner(r): r for r in requests}
    for identity, local, role, ordinal in outputs:
        r = originals[identity]
        source = identity + (r.start + local,)
        matches = [row for row in rows if row[1] == source]
        if len(matches) != 1:
            raise AssertionError('reference source is not unique')
        row = matches[0]
        targets.append((row[0], source, role, source[-1] + 1, ordinal))
    return {'rows': rows, 'offsets': offsets, 'Q': len(rows),
            'targets': targets,
            'owners': [row[1] for row in rows] +
                      [None] * (capacity - len(rows))}


def audit_view(view, requests, outputs=(), capacity=None):
    cap = sum(r.q for r in requests) if capacity is None else capacity
    expected = reference_view(requests, outputs, cap)
    errors = [name for name in expected if view.get(name) != expected[name]]
    keys = [row[1] for row in view.get('rows', [])]
    if len(keys) != len(set(keys)):
        errors.append('duplicate logical query')
    return errors


def fits(requests, pages, limits, omit_total=False):
    ncap, qcap, qmax, pcap, pmax = limits
    if any(r.q < 0 for r in requests):
        return False
    active = [r for r in requests if r.q > 0]
    if len(pages) != len(active) or any(p < 0 for p in pages):
        return False
    return (len(active) <= ncap and
            all(r.q <= qmax for r in active) and
            all(p <= pmax for p in pages) and
            sum(pages) <= pcap and
            (omit_total or sum(r.q for r in active) <= qcap))


def reference_capacity(requests, pages, limits):
    # Place individual query/page entries into bounded arrays.
    ncap, qcap, qmax, pcap, pmax = limits
    active = [r for r in requests if r.q > 0]
    if any(r.q < 0 for r in requests) or len(pages) != len(active):
        return False
    used_q = used_p = used_n = 0
    for r, count in zip(active, pages):
        if count < 0 or used_n >= ncap:
            return False
        used_n += 1
        for j in range(r.q):
            if j >= qmax or used_q >= qcap:
                return False
            used_q += 1
        for j in range(count):
            if j >= pmax or used_p >= pcap:
                return False
            used_p += 1
    return True


def audit_writes(view, layers, writes, history_cells=(), immutable_pages=()):
    expected = {(ell, row[1]) for ell in range(layers)
                for row in view['rows']}
    actual = [(ell, logical) for ell, logical, page, offset in writes]
    addresses = [(ell, page, offset)
                 for ell, logical, page, offset in writes]
    errors = []
    if set(actual) != expected or len(actual) != len(expected):
        errors.append('incomplete or duplicate layer/query coverage')
    if len(addresses) != len(set(addresses)):
        errors.append('physical write alias')
    if set(addresses).intersection(history_cells):
        errors.append('overwrite historical read cell')
    if any((ell, page) in immutable_pages
           for ell, logical, page, offset in writes):
        errors.append('write immutable shared page')
    return errors


class Slot:
    def __init__(self):
        self.phase = 'FREE'
        self.high = -1
        self.epoch = -1
        self.vid = None
        self.consumers = set()
        self.device = False
        self.completed = False
        self.resolved = False
        self.valid = False
        self.cancelled = False
        self.committed = False
        self.required = frozenset()
        self.snapshot = None

    def publish(self, vid, epoch, consumers, required, snapshot):
        if self.phase != 'FREE':
            return 'BUSY'
        if type(epoch) is not int or epoch < 0:
            return 'INVALID'
        if epoch <= self.high:
            return 'STALE'
        if len(consumers) != len(set(consumers)):
            return 'INVALID'
        self.phase = 'READY'
        self.vid = vid
        self.epoch = self.high = epoch
        self.consumers = set(consumers)
        self.required = frozenset(required)
        self.snapshot = snapshot
        self.device = self.completed = self.resolved = False
        self.valid = self.cancelled = self.committed = False
        return 'OK'

    def launch(self, ready=True):
        if self.phase != 'READY':
            return 'INVALID'
        if not ready:
            return 'WAIT_READY'
        self.device = True
        self.phase = 'RUNNING'
        return 'OK'

    def finish(self, vid, epoch, ended, success, covered,
               ignore_epoch=False):
        if vid != self.vid:
            return 'IGNORE_VIEW'
        if epoch != self.epoch and not ignore_epoch:
            return 'IGNORE_EPOCH'
        if self.completed:
            return 'DUPLICATE'
        if self.phase != 'RUNNING':
            return 'INVALID'
        if not ended:
            return 'DRAIN_WAIT'
        self.device = False
        self.completed = True
        self.valid = success and frozenset(covered) == self.required
        self.phase = 'DRAINING'
        return 'OK'

    def resolve(self, current_snapshot, ignore_guard=False):
        if not self.completed:
            return 'NOT_READY'
        if self.resolved:
            return 'DUPLICATE'
        authorized = (not self.cancelled and
                      current_snapshot == self.snapshot)
        self.committed = self.valid and (authorized or ignore_guard)
        self.resolved = True
        if not self.committed:
            # These are reserved, not yet started output consumers.
            self.consumers.clear()
        self.release()
        return 'COMMITTED' if self.committed else 'REJECTED'

    def ack(self, cid, early=False):
        if not self.resolved or not self.committed:
            return 'NOT_ACTIVE'
        if cid not in self.consumers:
            return 'DUPLICATE'
        self.consumers.remove(cid)
        if early:
            self.phase = 'FREE'
            self.vid = None
        else:
            self.release()
        return 'OK'

    def release(self):
        if (self.completed and self.resolved and not self.device and
                not self.consumers):
            self.phase = 'FREE'
            self.vid = None


def audit_slot(slot, permit_commit=True, expected=None):
    errors = []
    if slot.phase == 'FREE' and (slot.device or slot.consumers):
        errors.append('free with outstanding physical/future use')
    if slot.phase != 'FREE' and slot.vid is None:
        errors.append('live slot missing view identity')
    if slot.committed and not (slot.completed and slot.valid and permit_commit):
        errors.append('commit without complete authorized result')
    if expected is not None and slot.__dict__ != expected:
        errors.append('event violated expected no-op')
    return errors


def rejects(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError('invalid input was accepted')


def sample_outputs(requests):
    return [(owner(r), r.q - 1, 'SAMPLE', 0)
            for r in requests if r.q > 0]


def make_running(required, consumers=('sample', 'copy'), epoch=5):
    s = Slot()
    assert s.publish('view-7', epoch, consumers, required, ('auth', 3)) == 'OK'
    assert s.launch() == 'OK'
    return s


def main():
    base = [Request('A', 1, 'b0', 100, 2, tuple(range(102))),
            Request('B', 7, 'b1', 200, 5, tuple(range(205))),
            Request('C', 2, 'b0', 50, 3, tuple(range(53)))]
    outputs = sample_outputs(base)
    view = build_view(base, outputs, 16)
    assert not audit_view(view, base, outputs, 16)
    assert view['offsets'] == [0, 2, 7, 10]
    assert [t[0] for t in view['targets']] == [1, 6, 9]
    reverse = list(reversed(base))
    reverse_outputs = sample_outputs(reverse)
    rev = build_view(reverse, reverse_outputs)
    assert not audit_view(rev, reverse, reverse_outputs)
    assert rev['offsets'] == [0, 3, 8, 10]
    assert [t[0] for t in rev['targets']] == [2, 7, 9]
    wrong = deepcopy(rev)
    wrong['targets'] = [(i,) + t[1:]
                        for i, t in zip([2, 8, 9], rev['targets'])]
    assert audit_view(wrong, reverse, reverse_outputs)
    print('observed: reversed q=[3,5,2], offsets=[0,3,8,10], last rows=[2,7,9]')
    print('negative control detected: previous erroneous expectation [2,8,9]')

    layout_cases = capacity_cases = 0
    for qs in product(range(4), repeat=3):
        raw = [Request(str(i), i + 1, 'b', a, q,
                       tuple(100 * i + p for p in range(a + q)))
               for i, (a, q) in enumerate(zip((2, 4, 1), qs))]
        for ordered in permutations(raw):
            goals = sample_outputs(ordered)
            cap = sum(qs) + 2
            got = build_view(ordered, goals, cap)
            assert not audit_view(got, ordered, goals, cap)
            layout_cases += 1
        pages = [1 for r in raw if r.q > 0]
        for qcap in range(sum(qs) + 2):
            limits = (3, qcap, 3, 3, 1)
            assert fits(raw, pages, limits) == reference_capacity(raw, pages, limits)
            capacity_cases += 1
    assert layout_cases == 384
    assert not audit_view(build_view([]), [])
    zero = Request('zero', 0, 'b', 0, 0, ())
    assert not audit_view(build_view([zero]), [zero])
    rejects(lambda: build_view([replace(zero, start=-1)]))
    rejects(lambda: build_view([replace(zero, q=-1)]))
    rejects(lambda: build_view([replace(zero, q=1)]))
    rejects(lambda: build_view([replace(zero, q=True)]))
    rejects(lambda: build_view([zero], [(owner(zero), 0, 'SAMPLE', 0)]))
    rejects(lambda: build_view([base[0], base[0]]))
    rejects(lambda: build_view(base, outputs, 9))
    print('observed: %d independently referenced layouts and %d capacity placements' %
          (layout_cases, capacity_cases))

    for field in ('position', 'token', 'identity', 'coherent_identity', 'offset', 'padding'):
        bad = deepcopy(view)
        row = bad['rows'][0]
        if field == 'position':
            bad['rows'][0] = (row[0], row[1], row[0], row[3])
        elif field == 'token':
            bad['rows'][0] = row[:3] + (row[3] + 99,)
        elif field == 'identity':
            bad['rows'][0] = (row[0], ('ghost',) + row[1][1:], row[2], row[3])
        elif field == 'coherent_identity':
            new_key = ('ghost',) + row[1][1:]
            bad['rows'][0] = (row[0], new_key, row[2], row[3])
            bad['owners'][0] = new_key
        elif field == 'offset':
            bad['offsets'][1] += 1
        else:
            bad['owners'][10] = ('ghost', 0, 'b', 0)
        assert audit_view(bad, base, outputs, 16), field
    bad = deepcopy(view)
    target = bad['targets'][0]
    bad['targets'][0] = (10,) + target[1:]
    assert audit_view(bad, base, outputs, 16)
    print('negative controls detected: row/position, token, identity, coherent corruption, offsets, padding')

    twins = [Request('X', 1, 'b', 10, 2, tuple(range(12))),
             Request('Y', 1, 'b', 10, 2, tuple(range(12)))]
    twin_view = build_view(twins, sample_outputs(twins))
    assert not audit_view(twin_view, twins, sample_outputs(twins))
    reduced_keys = [row[1][1:] for row in twin_view['rows']]
    assert len(set(reduced_keys)) < len(reduced_keys)
    bad = deepcopy(twin_view)
    bad['rows'] = [(i, ('X',) + k[1:], p, token)
                   for i, k, p, token in bad['rows']]
    bad['owners'] = [row[1] for row in bad['rows']]
    assert audit_view(bad, twins, sample_outputs(twins))
    branches = [replace(twins[0], branch='left'), replace(twins[0], branch='right')]
    branch_outputs = sample_outputs(branches)
    assert not audit_view(build_view(branches, branch_outputs), branches, branch_outputs)

    prompt = Request('P', 3, 'b', 1, 3, tuple(range(8)))
    explicit = [(owner(prompt), 0, 'LOGPROB', 0),
                (owner(prompt), 2, 'LOGPROB', 0)]
    pv = build_view([prompt], explicit)
    assert not audit_view(pv, [prompt], explicit)
    assert [t[3] for t in pv['targets']] == [2, 4]
    bad = deepcopy(pv)
    bad['targets'] = bad['targets'][-1:]
    assert audit_view(bad, [prompt], explicit)
    assert not audit_view(build_view([prompt]), [prompt])
    rejects(lambda: build_view([prompt], [(owner(prompt), 2, 'SAMPLE', 0)]))
    rejects(lambda: build_view([prompt], explicit + explicit[:1]))
    print('observed: separate request/branch identities and explicit logprob source/target positions')

    over = [Request('U', 1, 'b', 0, 6, tuple(range(6))),
            Request('V', 1, 'b', 0, 5, tuple(range(5)))]
    assert fits(over, [1, 1], (2, 10, 6, 2, 1), omit_total=True)
    assert not reference_capacity(over, [1, 1], (2, 10, 6, 2, 1))
    assert not fits(over, [1, 1], (2, 10, 6, 2, 1))
    assert not fits(over, [2, 2], (2, 11, 6, 3, 2))
    assert not fits(over, [1, 1], (1, 11, 6, 2, 1))
    assert not fits(over, [1, 1], (2, 11, 5, 2, 1))
    print('negative control detected: per-request-only query capacity accepts 11>10')

    history = {(0, 1, 0), (1, 1, 0)}
    immutable = {(0, 2), (1, 2)}
    writes = [(ell, row[1], 100 + ell * 10 + i // 4, i % 4)
              for ell in range(2) for i, row in enumerate(view['rows'])]
    assert not audit_writes(view, 2, writes, history, immutable)
    for fault in ('alias', 'history', 'immutable', 'missing', 'duplicate'):
        bad = list(writes)
        ell, logical, page, offset = bad[1]
        if fault == 'alias':
            bad[1] = (ell, logical, bad[0][2], bad[0][3])
        elif fault == 'history':
            bad[1] = (ell, logical, 1, 0)
        elif fault == 'immutable':
            bad[1] = (ell, logical, 2, 3)
        elif fault == 'missing':
            bad.pop()
        else:
            bad.append(bad[0])
        assert audit_writes(view, 2, bad, history, immutable), fault
    print('negative controls detected: physical aliases, historical/shared writes, incomplete layer/query coverage')

    required = frozenset([('KV', ell, row[1]) for ell in range(2)
                          for row in view['rows']] +
                         [('LOGIT', t) for t in view['targets']])
    s = Slot()
    assert s.publish('view-7', 5, ('sample', 'copy'), required, ('auth', 3)) == 'OK'
    before = deepcopy(s.__dict__)
    assert s.launch(ready=False) == 'WAIT_READY'
    assert not audit_slot(s, expected=before)
    assert s.launch() == 'OK'
    assert s.ack('sample') == 'NOT_ACTIVE'
    before = deepcopy(s.__dict__)
    assert s.finish('view-7', 5, False, True, required) == 'DRAIN_WAIT'
    assert not audit_slot(s, expected=before)
    assert s.resolve(('auth', 3)) == 'NOT_READY'
    assert s.finish('view-7', 5, True, True, required) == 'OK'
    assert s.resolve(('auth', 3)) == 'COMMITTED'
    assert not audit_slot(s)
    assert s.ack('sample') == 'OK' and s.phase == 'DRAINING'
    before = deepcopy(s.__dict__)
    assert s.ack('sample') == 'DUPLICATE'
    assert not audit_slot(s, expected=before)
    assert s.consumers == {'copy'}
    assert s.ack('copy') == 'OK' and s.phase == 'FREE'
    assert s.high == 5 and not audit_slot(s)
    before = deepcopy(s.__dict__)
    assert s.publish('stale', 5, (), required, ('auth', 3)) == 'STALE'
    assert not audit_slot(s, expected=before)
    assert s.publish('view-8', 6, (), required, ('auth', 3)) == 'OK'
    assert s.launch() == 'OK'
    before = deepcopy(s.__dict__)
    assert s.finish('view-7', 5, True, True, required) == 'IGNORE_VIEW'
    assert not audit_slot(s, expected=before)
    assert s.finish('view-8', 5, True, True, required) == 'IGNORE_EPOCH'
    assert not audit_slot(s, expected=before)
    assert s.finish('view-8', 6, True, True, required) == 'OK'
    assert s.resolve(('auth', 3)) == 'COMMITTED' and s.phase == 'FREE'

    bad = make_running(required)
    before = deepcopy(bad.__dict__)
    assert bad.finish('view-7', 4, True, True, required, ignore_epoch=True) == 'OK'
    assert audit_slot(bad, expected=before)
    bad = make_running(required)
    assert bad.finish('view-7', 5, True, True, required) == 'OK'
    assert bad.resolve(('auth', 3)) == 'COMMITTED'
    assert bad.ack('sample', early=True) == 'OK'
    assert audit_slot(bad)
    bad = make_running(required)
    assert bad.finish('view-7', 5, True, True, required) == 'OK'
    assert bad.resolve(('auth', 3)) == 'COMMITTED'
    assert bad.ack('sample') == 'OK'
    before = deepcopy(bad.__dict__)
    # Actual faulty duplicate-ack behavior consumes somebody else's reference.
    bad.consumers.clear()
    bad.release()
    assert audit_slot(bad, expected=before)
    print('negative controls detected: wrong epoch, early release, duplicate consumer decrement')

    for changed in ('cancel', 'authorization', 'coverage', 'execution'):
        s = make_running(required)
        if changed == 'cancel':
            s.cancelled = True
            assert s.phase == 'RUNNING' and s.device
        covered = set(required)
        if changed == 'coverage':
            covered.remove(('KV', 0, view['rows'][0][1]))
        assert s.finish('view-7', 5, True, changed != 'execution', covered) == 'OK'
        current = ('auth', 4) if changed == 'authorization' else ('auth', 3)
        assert s.resolve(current) == 'REJECTED'
        assert not s.committed and s.phase == 'FREE'
        assert not audit_slot(s, permit_commit=False)
    for changed in ('cancel', 'authorization'):
        bad = make_running(required)
        bad.cancelled = changed == 'cancel'
        assert bad.finish('view-7', 5, True, True, required) == 'OK'
        current = ('auth', 4) if changed == 'authorization' else ('auth', 3)
        assert bad.resolve(current, ignore_guard=True) == 'COMMITTED'
        assert audit_slot(bad, permit_commit=False)
    busy = make_running(required)
    before = deepcopy(busy.__dict__)
    assert busy.publish('other', 6, (), required, ('auth', 3)) == 'BUSY'
    assert not audit_slot(busy, expected=before)
    print('observed: cancellation/stale snapshots/failure/incomplete coverage reject business commit after drain')
    print('scope: CPU symbolic mappings, bounded arrays, physical-address symbols and serial slot model')
    print('not covered: CUDA, numerical model forward, real COW/DMA, visibility, submission uncertainty or performance')


if __name__ == '__main__':
    main()
