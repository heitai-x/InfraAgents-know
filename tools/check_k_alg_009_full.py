#!/usr/bin/env python3
"""K-ALG-009: deterministic, offline, read-only stdlib CPU models.

Continuous metadata ranges, a four-layer scalar model, synthetic physical
leases/events, and baseline planner input-membership operation counts only.
No project imports, processes, files, network, or GPU use.
A terminal device failure/cancellation below means physical last use ended;
a failure notification or cancellation request alone does not mean that.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import permutations, product
from math import exp, gcd, isclose, tanh


def contains(span, lo, hi):
    return lo == hi or (span is not None and span[0] <= lo and hi <= span[1])


def direct(p, n, full, window, W, conv, K, checkpoints,
           initial=True, identity=True, context=True):
    if not identity or not context or not 0 <= p <= n:
        return False
    if p == 0:
        return initial
    if full is not None and not contains(full, 0, p):
        return False
    if window is not None and not contains(window, max(0, p-W+1), p):
        return False
    if conv is not None and not contains(conv, max(0, p-K+1), p):
        return False
    return checkpoints is None or p in checkpoints


def direct_reference(p, n, full, window, W, conv, K, checkpoints,
                     initial=True, identity=True, context=True):
    # Independent representation: enumerate legal key positions from p-t<W.
    if not identity or not context or p < 0 or p > n:
        return False
    if p == 0:
        return initial
    for span, width in ((full, None), (window, W), (conv, K)):
        if span is None:
            continue
        stored = set(range(span[0], span[1]))
        needed = {t for t in range(p) if width is None or p-t < width}
        if not needed.issubset(stored):
            return False
    return checkpoints is None or p in checkpoints


def plan(target, n, full, window, W, conv, K, checkpoints, known,
         initial=True):
    # Retain the round-3 baseline, including repeated interval scans.
    if not 0 <= target <= n:
        return None
    for c in range(target, -1, -1):
        if direct(c, n, full, window, W, conv, K, checkpoints, initial):
            if c == target:
                return ('direct', c, target, 0)
            if all(t in known for t in range(c, target)):
                return ('replay', c, target, target-c)
    return None


def check_metadata():
    spans = [None, (0, 0), (0, 3), (1, 4), (3, 5)]
    cases = 0
    for f, w, v, W, K, cps in product(spans, spans, spans, (1, 2, 4),
                                     (1, 3), (None, {1, 3, 5})):
        for p in range(6):
            assert direct(p, 5, f, w, W, v, K, cps) == direct_reference(
                p, 5, f, w, W, v, K, cps)
            cases += 1
    n = 1024
    args = (n, (0, n), (640, n), 128, (0, n), 4, {512, 768})
    positions = [p for p in range(n+1) if direct(p, *args)]
    assert positions == [0, 768]
    known = set(range(n))
    assert plan(1024, *args, known) == ('replay', 768, 1024, 256)
    assert plan(512, *args, known) == ('replay', 0, 512, 512)
    assert plan(768, *args, set()) == ('direct', 768, 768, 0)
    assert plan(1025, *args, known) is None
    assert plan(1024, *args, known-{900}) is None
    assert plan(512, *args, known, initial=False) is None
    assert not direct(0, *args, initial=False)
    assert not direct(768, *args, identity=False)
    assert not direct(768, *args, context=False)
    # Token endpoints are not consumption endpoints: stored 0..9 permits p=10.
    assert direct(10, 20, (0, 10), None, 1, None, 1, None)
    assert not direct(11, 20, (0, 10), None, 1, None, 1, None)
    assert direct(64, 100, (0, 100), (1, 64), 64, None, 1, {64})
    assert not direct(63, 100, (0, 100), (1, 64), 64, None, 1, {63})
    assert not direct(64, 100, (0, 100), (1, 64), 65, None, 1, {64})
    assert direct(4, 10, (0, 10), None, 1, (1, 4), 4, {4})
    assert not direct(3, 10, (0, 10), None, 1, (1, 4), 4, {3})
    assert direct(5, 10, None, (0, 0), 1, (0, 0), 1, {5})
    assert n != max(positions)  # faulty full-only direct-hit heuristic detected
    return cases, positions


class CountedKnown:
    """Count actual membership calls made by the unchanged plan function.

    Membership denotes an already aggregated per-position replay-readiness
    record. This does not model loading or validating external feature bytes.
    Only the counter changes; the represented input record is immutable.
    """
    def __init__(self, values):
        self.values = frozenset(values)
        self.queries = 0

    def __contains__(self, position):
        self.queries += 1
        return position in self.values


def check_planning_input_cost():
    # Independent closed-form budget and explicitly retained reviewer values.
    retained_counts = {64: 2080, 128: 8256, 256: 32896}
    observations = []
    rejected_linear_budgets = 0
    for n in (1, 2, 64, 128, 256):
        known = CountedKnown(range(n-1))
        original = known.values
        args = (n, (0, n-1), None, 1, None, 1, None)
        # Target is not direct; every c=0..n-1 is direct. All replay
        # intervals encounter the missing final input n-1 at their end.
        assert not direct(n, *args)
        assert all(direct(c, *args) for c in range(n))
        assert plan(n, *args, known) is None
        triangular = n*(n+1)//2
        assert known.queries == triangular, (n, known.queries, triangular)
        assert known.values == original
        if n in retained_counts:
            assert known.queries == retained_counts[n]
            observations.append((n, known.queries))
        # Intentionally faulty cost certificate: at most n input queries.
        # Detect it using observations from the real baseline plan path.
        if n > 1:
            try:
                assert known.queries <= n, 'faulty linear input-query budget'
            except AssertionError:
                rejected_linear_budgets += 1
            else:
                raise AssertionError('quadratic witness did not reject faulty budget')
        # Same state metadata, only the final input repaired: nearest start
        # succeeds after one membership query, rather than rescanning all c.
        complete = CountedKnown(range(n))
        assert plan(n, *args, complete) == ('replay', n-1, n, 1)
        assert complete.queries == 1
        # Immediate short-circuit at every candidate costs n, not triangular.
        empty = CountedKnown(())
        assert plan(n, *args, empty) is None
        assert empty.queries == n
        # A direct target never inspects replay input.
        direct_known = CountedKnown(())
        assert plan(n, n, (0, n), None, 1, None, 1, None,
                    direct_known) == ('direct', n, n, 0)
        assert direct_known.queries == 0
    assert rejected_linear_budgets == 4
    zero = CountedKnown(())
    assert plan(0, 0, (0, 0), None, 1, None, 1, None,
                zero) == ('direct', 0, 0, 0)
    assert plan(0, 0, (0, 0), None, 1, None, 1, None,
                zero, initial=False) is None
    assert plan(-1, 0, None, None, 1, None, 1, None, zero) is None
    assert plan(1, 0, None, None, 1, None, 1, None, zero) is None
    assert zero.queries == 0
    return observations, rejected_linear_budgets


KINDS = ('full', 'window', 'recurrent', 'full')
W = 3
K = 3


def dense_reference(tokens):
    """Layer-major, full-sequence reference; never calls the cached step."""
    n = len(tokens)
    values = [(x+1)*0.07+t*0.01 for t, x in enumerate(tokens)]
    layers = []
    for l, kind in enumerate(KINDS):
        row = {'kv': [], 'r': 0.1*(l+1), 'conv': []}
        output = []
        if kind != 'recurrent':
            keys = [0.3*h+0.01*(t+1) for t, h in enumerate(values)]
            vals = [0.6*h-0.02*l for h in values]
            for t, h in enumerate(values):
                js = [j for j in range(t+1) if kind == 'full' or t-j < W]
                scores = [(0.4*h+0.03*(l+1))*keys[j] for j in js]
                peak = max(scores)
                weights = [exp(s-peak) for s in scores]
                numerator = sum(a*vals[j] for a, j in zip(weights, js))
                output.append(tanh(h+numerator/sum(weights)))
            keep = range(n) if kind == 'full' else range(max(0, n-W+1), n)
            row['kv'] = [(t, keys[t], vals[t]) for t in keep]
        else:
            projected = [0.7*h+0.03*(l+1) for h in values]
            r = 0.1*(l+1)
            for t, h in enumerate(values):
                u = projected[t]
                previous = projected[t-1] if t >= 1 else 0.0
                older = projected[t-2] if t >= 2 else 0.0
                z = tanh(0.5*u+0.25*previous-0.125*older)
                r = (0.5+0.1*tanh(u))*r+0.3*z
                output.append(tanh(h+r+z))
            row['r'] = r
            row['conv'] = [(t, projected[t]) for t in range(max(0, n-K+1), n)]
        layers.append(row)
        values = output
    return {'p': n, 'layers': layers}, values


def cached_step(state, token, t):
    """Token-major candidate transaction; commit p after all layer candidates."""
    assert state['p'] == t
    candidate = deepcopy(state)
    hidden = (token+1)*0.07+t*0.01
    for l, kind in enumerate(KINDS):
        old = state['layers'][l]
        new = candidate['layers'][l]
        if kind in ('full', 'window'):
            q = hidden*0.4+(l+1)*0.03
            key = hidden*0.3+(t+1)*0.01
            val = hidden*0.6-l*0.02
            rows = list(old['kv'])+[(t, key, val)]
            assert all(j <= t for j, _, _ in rows)
            if kind == 'full':
                assert [j for j, _, _ in rows] == list(range(t+1))
            else:
                assert [j for j, _, _ in rows] == list(range(max(0, t-W+1), t+1))
            top = max(q*k for _, k, _ in rows)
            normalizer = 0.0
            weighted = 0.0
            for _, k, v in rows:
                a = exp(q*k-top)
                normalizer += a
                weighted += a*v
            new['kv'] = rows if kind == 'full' else (rows[-(W-1):] if W > 1 else [])
            hidden = tanh(hidden+weighted/normalizer)
        else:
            history = dict(old['conv'])
            assert set(history) == set(range(max(0, t-K+1), t))
            u = hidden*0.7+(l+1)*0.03
            convolution = u*0.5
            for distance, coefficient in ((1, 0.25), (2, -0.125)):
                if t-distance >= 0:
                    convolution += coefficient*history[t-distance]
            z = tanh(convolution)
            r = (0.5+0.1*tanh(u))*old['r']+0.3*z
            new['r'] = r
            new['conv'] = (old['conv']+[(t, u)])[-(K-1):] if K > 1 else []
            hidden = tanh(hidden+r+z)
    candidate['p'] = t+1
    return candidate, hidden


def replay(snapshot, tokens, stop):
    assert snapshot['p'] <= stop <= len(tokens)
    work = deepcopy(snapshot)
    outputs = []
    for t in range(work['p'], stop):
        work, output = cached_step(work, tokens[t], t)
        outputs.append(output)
    return work, outputs


def near(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(near(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(near(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return isclose(a, b, rel_tol=1e-11, abs_tol=1e-12)
    return a == b


def check_replay():
    tokens = [(i*5+3) % 17 for i in range(20)]
    final_ref, outputs_ref = dense_reference(tokens)
    comparisons = 0
    for p in range(len(tokens)+1):
        target_ref, target_outputs = dense_reference(tokens[:p])
        for c in range(p+1):
            snapshot, _ = dense_reference(tokens[:c])
            protected = deepcopy(snapshot)
            got, suffix = replay(snapshot, tokens, p)
            assert near(got, target_ref)
            assert near(suffix, target_outputs[c:])
            assert snapshot == protected  # replay must not mutate shared start
            continued, tail = replay(got, tokens, len(tokens))
            assert near(continued, final_ref)
            assert near(suffix+tail, outputs_ref[c:])
            comparisons += 1
    start, _ = dense_reference(tokens[:7])
    skipped = deepcopy(start)
    skipped['p'] = 20
    assert not near(skipped, final_ref)
    mixed = deepcopy(start)
    for t in range(7, 20):
        previous = deepcopy(mixed)
        mixed, _ = cached_step(mixed, tokens[t], t)
        mixed['layers'][2]['r'] = previous['layers'][2]['r']
    assert not near(mixed, final_ref)
    corrupted = deepcopy(start)
    j, key, val = corrupted['layers'][0]['kv'][0]
    corrupted['layers'][0]['kv'][0] = (j, key, val+1.0)
    _, corrupted_outputs = replay(corrupted, tokens, 20)
    assert not near(corrupted_outputs, outputs_ref[7:])
    return comparisons


def reserve_vector(free, demand):
    if len(free) != len(demand) or any(x < 0 for x in demand):
        raise ValueError('invalid resource vector')
    if any(a < b for a, b in zip(free, demand)):
        return None
    return tuple(a-b for a, b in zip(free, demand))


class Pool:
    def __init__(self, capacity=(2, 2)):
        self.capacity = capacity
        self.keys = {(g, i) for g, count in enumerate(capacity) for i in range(count)}
        self.free = set(self.keys)
        self.owner = {}
        self.epoch = {key: 0 for key in self.keys}
        self.data = {key: [None]*4 for key in self.keys}

    def reserve(self, owner, demand):
        if len(demand) != len(self.capacity) or any(x < 0 for x in demand):
            raise ValueError('invalid demand')
        assert owner not in self.owner.values()
        candidates = [[key for key in sorted(self.free) if key[0] == g]
                      for g in range(len(self.capacity))]
        if any(len(keys) < need for keys, need in zip(candidates, demand)):
            return None
        handles = []
        for keys, need in zip(candidates, demand):
            group = []
            for key in keys[:need]:
                self.free.remove(key)
                self.owner[key] = owner
                self.epoch[key] += 1
                self.data[key] = [None]*4
                group.append(key+(self.epoch[key],))
            handles.append(group)
        return handles

    def release_owned(self, owner):
        # Low-level operation. The transaction must establish last-use first.
        keys = [key for key, value in self.owner.items() if value == owner]
        for key in keys:
            del self.owner[key]
            self.free.add(key)
        return len(keys)


@dataclass(frozen=True)
class Operation:
    tx: str
    generation: int
    serial: int
    source: tuple
    destination: tuple
    lo: int
    hi: int
    direction: str = 'copy'

    @property
    def owner(self):
        return (self.tx, self.generation)


class PhysicalDevice:
    def __init__(self, pool):
        self.pool = pool
        self.active = {}
        self.terminal = {}
        self.cancel_requests = set()

    def submit(self, op):
        assert op not in self.active and op not in self.terminal
        for handle in (op.source, op.destination):
            assert self.pool.owner.get(handle[:2]) == op.owner
            assert self.pool.epoch[handle[:2]] == handle[2]
        assert 0 <= op.lo < op.hi <= 4
        self.active[op] = (op.tx, op.generation, op.serial)

    def request_cancel(self, op):
        if op in self.active:
            self.cancel_requests.add(op)
        # No terminal receipt and no removal from physical active here.

    def finish(self, op, status='done'):
        if op not in self.active:
            return None
        assert status in ('done', 'failed', 'cancelled')
        payload = self.active.pop(op)
        if status == 'done':
            # Raw physical address write: deliberately NO generation guard.
            # Early reuse must therefore be detected by the independent audit.
            key = op.destination[:2]
            for i in range(op.lo, op.hi):
                self.pool.data[key][i] = payload
        self.terminal[op] = status
        return (op, status)


class Transaction:
    def __init__(self, pool, device, tx, generation, demand=(2, 2)):
        self.pool = pool
        self.device = device
        self.owner = (tx, generation)
        self.demand = demand
        self.handles = pool.reserve(self.owner, demand)
        if self.handles is None:
            raise MemoryError('atomic reservation failed')
        self.state = 'ACTIVE'
        self.source_protected = True
        self.pending = set()
        self.issued = set()
        self.done = {}
        self.serial = 0

    def submit(self, which, lo=0, hi=1):
        if self.state != 'ACTIVE':
            raise RuntimeError('submission after abort or publish')
        self.serial += 1
        op = Operation(self.owner[0], self.owner[1], self.serial,
                       self.handles[1][which], self.handles[0][which], lo, hi)
        self.issued.add(op)
        self.pending.add(op)  # register before physical submission
        self.device.submit(op)
        return op

    def event(self, receipt):
        if receipt is None:
            return False
        op, status = receipt
        if op.owner != self.owner or op not in self.issued or op not in self.pending:
            return False
        if status not in ('done', 'failed', 'cancelled'):
            return False
        # In this model the separate device ledger represents the trusted
        # physical completion source. A fabricated known-ID receipt is invalid.
        if self.device.terminal.get(op) != status:
            return False
        self.pending.remove(op)
        self.done[op] = status
        if status == 'failed':
            self.abort()
        return True

    def abort(self):
        if self.state == 'ACTIVE':
            self.state = 'ABORTING'
        if self.state == 'ABORTING':
            for op in self.pending:
                self.device.request_cancel(op)

    def settle(self):
        if self.state != 'ABORTING' or self.pending:
            return False
        assert not any(op.owner == self.owner for op in self.device.active)
        assert self.pool.release_owned(self.owner) == sum(self.demand)
        self.source_protected = False
        self.state = 'SETTLED'
        return True

    def publish(self):
        if self.state != 'ACTIVE' or self.pending:
            return False
        if set(self.done) != self.issued or any(s != 'done' for s in self.done.values()):
            return False
        # Two mandatory destination families must each have range [0,1).
        required = set(self.handles[0])
        covered = {op.destination for op, status in self.done.items()
                   if status == 'done' and op.lo == 0 and op.hi >= 1}
        if not required.issubset(covered):
            return False
        self.state = 'PUBLISHED'
        self.source_protected = False
        return True

    def retire(self):
        if self.state != 'PUBLISHED' or self.pending:
            return False
        assert not any(op.owner == self.owner for op in self.device.active)
        assert self.pool.release_owned(self.owner) == sum(self.demand)
        self.state = 'RETIRED'
        return True


def lease_violations(pool, device):
    """Independent physical audit: does not use transaction pending/count."""
    errors = []
    owned = set(pool.owner)
    if pool.free & owned or pool.free | owned != pool.keys:
        errors.append('capacity partition')
    for g, cap in enumerate(pool.capacity):
        total = sum(key[0] == g for key in pool.free)+sum(key[0] == g for key in owned)
        if total != cap:
            errors.append('pool conservation')
    for op in device.active:
        for handle in (op.source, op.destination):
            key = handle[:2]
            if pool.owner.get(key) != op.owner or pool.epoch[key] != handle[2]:
                errors.append('active operation lost physical lease')
    for key, owner in pool.owner.items():
        if any(value is not None and value[:2] != owner for value in pool.data[key]):
            errors.append('payload belongs to another owner')
    return errors


def check_events():
    pool = Pool()
    device = PhysicalDevice(pool)
    baseline = set(pool.free)
    epochs = dict(pool.epoch)
    assert pool.reserve(('too-large', 1), (3, 1)) is None
    assert pool.free == baseline and pool.epoch == epochs and not pool.owner
    tx = Transaction(pool, device, 'retry', 1)
    a = tx.submit(0)
    b = tx.submit(1)
    tx.abort()
    assert tx.source_protected and not tx.publish() and not tx.settle()
    before = set(tx.pending)
    forged = [replace(a, serial=99), replace(a, serial=100),
              replace(a, tx='unknown'), replace(a, generation=0),
              replace(a, lo=1, hi=2), replace(a, direction='other'),
              replace(a, destination=a.destination[:2]+(99,))]
    for fake in forged:
        assert not tx.event((fake, 'done'))
        assert tx.pending == before
    assert not tx.event((a, 'done'))  # known ID, but no physical completion
    assert tx.pending == {a, b} and len(device.active) == 2
    ra = device.finish(a, 'failed')
    assert tx.event(ra) and not tx.event(ra)
    assert tx.pending == {b} and not tx.settle()
    assert not tx.event((b, 'cancelled'))  # cancel request is not confirmation
    assert b in device.active and tx.source_protected
    try:
        tx.submit(0)
    except RuntimeError:
        pass
    else:
        raise AssertionError('abort permitted new submission')
    rb = device.finish(b, 'done')
    assert not device.active and not tx.settle()  # receipt not consumed yet
    assert tx.source_protected and not pool.free
    assert tx.event(rb) and tx.settle()
    assert pool.free == baseline and not tx.source_protected
    assert not tx.settle() and not tx.event(rb)
    assert not lease_violations(pool, device)
    # Same transaction name, new attempt and reused physical addresses.
    newer = Transaction(pool, device, 'retry', 2)
    new_a = newer.submit(0)
    assert not newer.event(rb)
    assert newer.pending == {new_a}
    assert device.finish(b) is None  # duplicate receipt is not another raw write
    assert newer.event(device.finish(new_a))
    assert not newer.publish()  # one family completed; second family absent
    new_b = newer.submit(1)
    assert newer.event(device.finish(new_b)) and newer.publish()
    assert not lease_violations(pool, device)
    assert newer.retire() and pool.free == baseline
    # All terminal orders, including a real cancellation confirmation.
    orders = 0
    for order in permutations(range(3)):
        p = Pool()
        d = PhysicalDevice(p)
        t = Transaction(p, d, 'permutation', 1)
        ops = [t.submit(0), t.submit(1), t.submit(0, 1, 2)]
        t.abort()
        statuses = ('failed', 'done', 'cancelled')
        for count, i in enumerate(order, 1):
            assert not lease_violations(p, d)
            receipt = d.finish(ops[i], statuses[i])
            assert t.event(receipt)
            assert not t.event(receipt)
            assert t.settle() == (count == 3)
            assert not lease_violations(p, d)
        assert p.free == p.keys and not d.active and not t.source_protected
        orders += 1
    return orders


def check_event_negative_controls():
    pool = Pool()
    device = PhysicalDevice(pool)
    old = Transaction(pool, device, 'old', 1)
    a, b = old.submit(0), old.submit(1)
    old.abort()
    # Historical faulty reducer actually accepts two unknown event names.
    faulty_count = 2
    faulty_done = set()
    for unknown in ('unknown-1', 'unknown-2'):
        if unknown not in faulty_done:
            faulty_done.add(unknown)
            if faulty_count > 0:
                faulty_count -= 1
    assert faulty_count == 0 and len(device.active) == 2
    # Independent physical ledger refutes the faulty drain certificate.
    assert (faulty_count == 0) != (len(device.active) == 0)
    # Execute the unsafe transition, not merely a hypothetical free vector.
    assert pool.release_owned(old.owner) == 4
    assert 'active operation lost physical lease' in lease_violations(pool, device)
    fresh = Transaction(pool, device, 'reuse', 2)
    assert {h[:2] for group in fresh.handles for h in group} == pool.keys
    target = b.destination[:2]
    pool.data[target] = [('reuse', 2, 999)]*4
    device.finish(b, 'done')  # late DMA writes by raw address after reuse
    assert pool.data[target][0][:2] == ('old', 1)
    assert pool.data[target][0][:2] != fresh.owner
    errors = lease_violations(pool, device)
    assert 'payload belongs to another owner' in errors
    assert a in device.active
    return ('faulty unknown-event drain', 'early reuse lease violation', 'late raw-write pollution')


def check_costs_and_reservation():
    mib = 1024**2
    components = (8*4096*4096, 8*256*4096,
                  16*4096*16*4, 16*3*4096*2)
    assert components == (128*mib, 8*mib, 4*mib, 384*1024)
    total = sum(components)
    assert total/mib == 140.375
    snapshots = (components[2]+components[3])*16
    assert snapshots/mib == 70
    per_layer = (components[2]+components[3])//16
    page = 8*16*4096
    assert page == 512*1024 and per_layer == 280*1024
    padding = 16*(page-per_layer)
    assert padding/mib == 3.625
    assert 384//gcd(384, 512)*512 == 1536
    free = (1, 1)
    assert reserve_vector(free, (1, 1)) == (0, 0)
    assert reserve_vector(free, (0, 2)) is None and free == (1, 1)
    assert reserve_vector(free, (0, 0)) == free
    first = reserve_vector(free, (1, 0))
    trapped = reserve_vector(first, (0, 1))
    assert trapped == (0, 0)
    assert reserve_vector(trapped, (0, 1)) is None
    assert reserve_vector(trapped, (1, 0)) is None
    return total/mib, snapshots/mib, padding/mib


def main():
    metadata_cases, positions = check_metadata()
    input_counts, rejected_budgets = check_planning_input_cost()
    replay_cases = check_replay()
    event_orders = check_events()
    detected = check_event_negative_controls()
    costs = check_costs_and_reservation()
    print('K-ALG-009 scoped CPU models; no GPU or engine execution')
    print('metadata/reference comparisons:', metadata_cases)
    print('common direct positions:', positions, '; replay 768->1024: 256 steps')
    print('missing-final-input baseline counts (n, membership queries):', input_counts)
    print('count reference: n*(n+1)//2; fixed witnesses: 2080, 8256, 32896')
    print('negative control: faulty linear input-query budgets rejected:', rejected_budgets)
    print('input-cost edges: repaired final input=1 query; empty input=n; direct target=0')
    print('layer-major reference / token-major recovery cut pairs:', replay_cases)
    print('terminal event orders checked:', event_orders)
    print('unknown, forged, duplicate and old-attempt receipts rejected')
    print('negative controls detected:', ', '.join(detected))
    print('negative controls detected: full-only hit, skipped replay, mixed state, corrupted history')
    print('main / extra snapshots / padding MiB:', costs)
    print('all scoped assertions passed; operation counts are not timings; device visibility and performance unmeasured')


if __name__ == '__main__':
    main()
