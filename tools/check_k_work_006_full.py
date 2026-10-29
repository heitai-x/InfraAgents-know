from fractions import Fraction
from math import ceil


def expect_error(fn, kind=AssertionError):
    try:
        fn()
    except kind:
        return
    raise AssertionError('expected failure was not detected')


def unique_pages(sequences, block):
    if block <= 0:
        raise ValueError('invalid block')
    return {(tuple(seq[:start]), tuple(seq[start:start + block]))
            for seq in sequences for start in range(0, len(seq), block)}


def trie_reference(sequences, block):
    if block <= 0:
        raise ValueError('invalid block')
    edges, endpoints, next_node = {}, set(), 1
    for seq in sequences:
        node = 0
        for depth, token in enumerate(seq, 1):
            edge = (node, token)
            if edge not in edges:
                edges[edge] = next_node
                next_node += 1
            node = edges[edge]
            if depth % block == 0 or depth == len(seq):
                endpoints.add(node)
    return len(endpoints)


class Pool:
    # One serialized owner; counts are (L,H,R,D,F).
    def __init__(self, total):
        if not isinstance(total, int) or total < 0:
            raise ValueError('invalid total')
        self.total = total
        self.free = set(range(total))
        self.generation = [0] * total
        self.records, self.memory, self.eligible = {}, {}, {}
        self.operations, self.log = {}, []
        self.tx_number = self.op_number = 0

    def counts(self):
        return tuple(sum(r['state'] == state for r in self.records.values())
                     for state in ('L', 'H', 'R', 'D')) + (len(self.free),)

    def check(self):
        pages = [key[0] for key in self.records]
        assert len(pages) == len(set(pages))
        assert not set(pages) & self.free
        assert set(pages) | self.free == set(range(self.total))
        assert sum(self.counts()) == self.total
        assert all(r['state'] in ('L', 'H', 'R', 'D')
                   for r in self.records.values())

    def keys(self, tx, state):
        return sorted(k for k, r in self.records.items()
                      if r['owner'] == tx and r['state'] == state)

    def allocate(self, tx, number, state):
        result = []
        for _ in range(number):
            page = min(self.free)
            self.free.remove(page)
            self.generation[page] += 1
            key = (page, self.generation[page])
            self.records[key] = dict(owner=tx, state=state,
                                     active=set(), ready=False)
            self.memory[page] = (tx, key[1], 'allocation identity')
            self.log.append(('reserve', key, tx, state))
            result.append(key)
        return result

    def reserve(self, data, guard=0):
        if any(not isinstance(x, int) or x < 0 for x in (data, guard)):
            raise ValueError('invalid reservation')
        if data + guard > len(self.free):
            return None
        self.tx_number += 1
        tx = self.tx_number
        self.eligible[tx] = True
        self.log.append(('open', tx))
        self.allocate(tx, data, 'R')
        self.allocate(tx, guard, 'H')
        self.check()
        return tx

    def move(self, key, destination):
        old = self.records[key]['state']
        self.records[key]['state'] = destination
        self.log.append(('move', key, old, destination))

    def release(self, key):
        self.log.append(('release', key))
        del self.records[key]
        self.free.add(key[0])

    def accept(self, tx, keys):
        if not self.eligible.get(tx, False):
            raise ValueError('submission revoked')
        keys = tuple(sorted(set(keys)))
        if not keys:
            return None
        for key in keys:
            r = self.records[key]
            if r['owner'] != tx or r['state'] not in ('R', 'L'):
                raise ValueError('invalid submitted range')
        self.op_number += 1
        op = (tx, self.op_number)
        expected = tuple(self.memory[k[0]] for k in keys)
        self.operations[op] = (keys, expected)
        for key in keys:
            self.records[key]['active'].add(op)
        self.log.append(('accept', op, tx, keys))
        return op

    def submit(self, tx, state='R', limit=None):
        keys = self.keys(tx, state)
        return self.accept(tx, keys if limit is None else keys[:limit])

    def complete(self, op):
        if op not in self.operations:
            return []
        keys, expected = self.operations.pop(op)
        actual = tuple(self.memory[k[0]] for k in keys)
        self.log.append(('finish', op))
        for key in keys:
            r = self.records.get(key)
            if r is not None:
                r['active'].discard(op)
                r['ready'] = True
                if r['state'] == 'D' and not r['active']:
                    self.release(key)
        self.check()
        return list(zip(expected, actual))

    def publish(self, tx):
        if not self.eligible.get(tx, False):
            raise ValueError('publication revoked')
        keys = self.keys(tx, 'R')
        if any(not self.records[k]['ready'] or self.records[k]['active']
               for k in keys):
            raise ValueError('not ready')
        for key in keys:
            self.move(key, 'L')
        self.log.append(('publish', tx, tuple(keys)))
        self.check()

    def retire(self, keys, early=False):
        for key in sorted(set(keys)):
            r = self.records[key]
            if r['active'] and not early:
                if r['state'] != 'D':
                    self.move(key, 'D')
            else:
                self.release(key)
        self.check()

    def cancel(self, tx, early=False):
        self.eligible[tx] = False
        self.log.append(('cancel', tx))
        self.retire([k for k, r in self.records.items() if r['owner'] == tx], early)

    def take(self, tx, number):
        if not isinstance(number, int) or number < 0:
            raise ValueError('invalid growth')
        if not self.eligible.get(tx, False):
            raise ValueError('growth revoked')
        if self.keys(tx, 'R'):
            raise ValueError('unsettled preparation')
        guards = self.keys(tx, 'H')
        use = min(number, len(guards))
        if number - use > len(self.free):
            return None
        result = guards[:use]
        for key in result:
            self.move(key, 'R')
            self.records[key]['ready'] = False
        result += self.allocate(tx, number - use, 'R')
        self.check()
        return result

    def refill(self, tx, target):
        if not isinstance(target, int) or target < 0:
            raise ValueError('invalid target')
        if not self.eligible.get(tx, False):
            raise ValueError('refill revoked')
        missing = max(0, target - len(self.keys(tx, 'H')))
        if missing > len(self.free):
            return False
        self.allocate(tx, missing, 'H')
        self.check()
        return True


def audit_events(events, total):
    # Rebuild physical obligations without consulting Pool state.
    leases, occupied, active, eligible, generations = {}, {}, {}, {}, {}
    for event in events:
        tag = event[0]
        if tag == 'open':
            assert event[1] not in eligible
            eligible[event[1]] = True
        elif tag == 'reserve':
            _, key, tx, state = event
            page, generation = key
            assert 0 <= page < total and page not in occupied
            assert eligible[tx] and state in ('R', 'H')
            assert generation > generations.get(page, 0)
            generations[page] = generation
            occupied[page] = key
            leases[key] = [tx, state]
        elif tag == 'move':
            _, key, old, new = event
            assert leases[key][1] == old
            assert (old, new) in {('H', 'R'), ('R', 'L'), ('R', 'D'), ('L', 'D')}
            leases[key][1] = new
        elif tag == 'accept':
            _, op, tx, keys = event
            assert eligible[tx] and op not in active and keys
            for key in keys:
                assert occupied[key[0]] == key
                assert leases[key][0] == tx and leases[key][1] in ('R', 'L')
            active[op] = keys
        elif tag == 'finish':
            assert event[1] in active
            del active[event[1]]
        elif tag == 'cancel':
            eligible[event[1]] = False
        elif tag == 'publish':
            _, tx, keys = event
            assert eligible[tx]
            for key in keys:
                assert leases[key] == [tx, 'L']
                assert all(key not in used for used in active.values())
        elif tag == 'release':
            key = event[1]
            assert key in leases
            assert all(key not in used for used in active.values()), 'early physical release'
            del leases[key]
            del occupied[key[0]]
        else:
            raise AssertionError('unknown event')
        assert len(occupied) == len(leases) <= total
        assert all(k in leases for used in active.values() for k in used)
    return len(leases), len(active)


def commit_reserved(pool, tx):
    op = pool.submit(tx)
    assert all(a == b for a, b in pool.complete(op))
    pool.publish(tx)


def token_at(request, position, shared_tokens):
    return ('shared', position) if position < shared_tokens else (request, position)


def audit_private_destinations(published_views, destinations):
    protected = {key for view in published_views for key in view}
    assert len(destinations) == len(set(destinations))
    assert not protected.intersection(destinations), 'write aliases a published page'


def audit_frozen(data, frozen):
    assert all(data[key] == value for key, value in frozen.items()), 'published page mutated'


class Tails:
    # All requests share one serialized transaction owner in this CPU model.
    # Only complete shared prefix pages are shared; private tails are distinct.
    def __init__(self, pool, tx, requests, length, block, shared_tokens=0):
        if block <= 0 or length < 0 or shared_tokens < 0:
            raise ValueError('invalid view')
        if shared_tokens > length or shared_tokens % block:
            raise ValueError('shared prefix must contain complete pages')
        self.pool, self.tx, self.block = pool, tx, block
        self.shared_tokens = shared_tokens
        self.data, self.views, self.lengths = {}, [], [length] * requests
        live = pool.keys(tx, 'L')
        shared_count = shared_tokens // block
        private_count = ceil(length / block) - shared_count
        assert len(live) == shared_count + requests * private_count
        shared = live[:shared_count]
        for j, key in enumerate(shared):
            self.data[key] = tuple(token_at(0, pos, shared_tokens)
                                   for pos in range(j * block, (j + 1) * block))
        cursor = shared_count
        for request in range(requests):
            private = live[cursor:cursor + private_count]
            cursor += private_count
            self.views.append(list(shared) + private)
            for j, key in enumerate(private, shared_count):
                self.data[key] = tuple(token_at(request, pos, shared_tokens)
                                       for pos in range(j * block, min((j + 1) * block, length)))
        self.check_reference()

    def check_reference(self):
        # Independent reference constructs the whole sequence by position,
        # without using page replacement or allocation decisions.
        for request, (view, length) in enumerate(zip(self.views, self.lengths)):
            physical = tuple(value for key in view for value in self.data[key])
            expected = tuple(token_at(request, pos, self.shared_tokens)
                             for pos in range(length))
            assert physical == expected
            assert len(view) == ceil(length / self.block)
            assert all(self.pool.records[k]['state'] == 'L' for k in view)

    def start(self, selected):
        selected = list(selected)
        if len(selected) != len(set(selected)):
            raise ValueError('duplicate request')
        if any(i < 0 or i >= len(self.views) for i in selected):
            raise ValueError('invalid request')
        old_views = [tuple(view) for view in self.views]
        frozen = {key: self.data[key] for view in old_views for key in view}
        destinations = self.pool.take(self.tx, len(selected))
        if destinations is None:
            return None
        audit_private_destinations(old_views, destinations)
        entries = []
        sources = []
        for request, destination in zip(selected, destinations):
            n = self.lengths[request]
            old = self.views[request][-1] if n % self.block else None
            if old is not None:
                sources.append(old)
            entries.append((request, n, old, destination))
        op = self.pool.accept(self.tx, destinations + sources)
        return dict(entries=entries, destinations=destinations, frozen=frozen,
                    views=old_views, op=op, eligible=True)

    def write(self, pending, limit=None):
        if not pending['eligible']:
            raise ValueError('append publication revoked')
        entries = pending['entries']
        if limit is not None:
            entries = entries[:limit]
        for request, n, old, destination in entries:
            prefix = self.data[old] if old is not None else ()
            assert len(prefix) == n % self.block
            self.data[destination] = prefix + (token_at(request, n, self.shared_tokens),)
        audit_frozen(self.data, pending['frozen'])
        assert [tuple(v) for v in self.views] == pending['views']
        self.check_reference()

    def commit(self, pending):
        self.write(pending)
        assert all(a == b for a, b in self.pool.complete(pending['op']))
        self.pool.publish(self.tx)
        retired = []
        for request, n, old, destination in pending['entries']:
            assert self.lengths[request] == n
            if old is None:
                self.views[request].append(destination)
            else:
                self.views[request][-1] = destination
                retired.append(old)
            self.lengths[request] = n + 1
        self.pool.retire(retired)
        pending['eligible'] = False
        audit_frozen(self.data, pending['frozen'])
        self.check_reference()

    def abort(self, pending):
        if not pending['eligible']:
            raise ValueError('append already settled')
        pending['eligible'] = False
        self.pool.retire(pending['destinations'])
        audit_frozen(self.data, pending['frozen'])
        assert [tuple(v) for v in self.views] == pending['views']
        self.check_reference()


def make_tails(total, requests, length, block, guards=0, shared=0):
    live = shared // block + requests * (ceil(length / block) - shared // block)
    p = Pool(total)
    tx = p.reserve(live, guards)
    assert tx is not None
    commit_reserved(p, tx)
    return p, tx, Tails(p, tx, requests, length, block, shared)


def required_instances(rate, demand, capacity):
    if min(rate, demand, capacity) < 0:
        raise ValueError('negative resource input')
    work = Fraction(rate) * demand
    if work == 0:
        return 0
    if capacity == 0:
        raise ValueError('positive demand with zero capacity')
    return ceil(work / capacity)


def plan_instances(rate, resources):
    return max((required_instances(rate, d, c) for d, c in resources), default=0)


def enumerate_instances(rate, resources, bound=100):
    for number in range(bound + 1):
        if all(rate * d <= number * c for d, c in resources):
            return number
    return None


def phase_ratio(lp, ld, rp, rd):
    if min(lp, ld, rp, rd) < 0:
        raise ValueError('negative phase input')
    if (lp > 0 and rp == 0) or (ld > 0 and rd == 0):
        raise ValueError('unserviceable phase')
    if ld == 0:
        return None
    if lp == 0:
        return Fraction(0)
    return Fraction(lp, ld) * Fraction(rd, rp)


def fluid_backlog(segments, service):
    if service < 0:
        raise ValueError('negative service')
    q = Fraction(0)
    for arrival, duration in segments:
        if arrival < 0 or duration < 0:
            raise ValueError('negative segment')
        q = max(Fraction(0), q + (arrival - service) * duration)
    return q


def stepped_reference(segments, service, step=Fraction(1, 100)):
    q = arrived = served = Fraction(0)
    for arrival, duration in segments:
        remaining = Fraction(duration)
        while remaining:
            dt = min(step, remaining)
            new = arrival * dt
            done = min(q + new, service * dt)
            arrived += new
            served += done
            q += new - done
            remaining -= dt
            assert q == arrived - served and q >= 0
    return q


def test_pages():
    cases = [[], [tuple('ABCDEFG')], [tuple('ABCDEFG'), tuple('ABCDEFG')],
             [tuple('ABCDEFG'), tuple('ABCDEFGH')],
             [tuple('ABCDWXYZ'), tuple('EFGHWXYZ')]]
    for block in (1, 2, 4, 16):
        for sequences in cases:
            assert len(unique_pages(sequences, block)) == trie_reference(sequences, block)
    seqs = [tuple('ABCDEFG'), tuple('ABCDEFGH')]
    assert len(unique_pages(seqs, 4)) == 3
    assert sum(ceil(len(s) / 4) for s in seqs) == 4
    for number, expected in ((30, 728), (32, 776)):
        seqs = [tuple(token_at(r, pos, 128) for pos in range(512))
                for r in range(number)]
        assert len(unique_pages(seqs, 16)) == trie_reference(seqs, 16) == expected
    expect_error(lambda: unique_pages([], 0), ValueError)
    print('pages: independent trie agrees; owner sum 4 versus unique 3; shared totals 728/776')


def test_lifetimes():
    p = Pool(0)
    assert p.reserve(1) is None
    tx = p.reserve(0)
    p.publish(tx)
    p.cancel(tx)
    assert audit_events(p.log, 0) == (0, 0)

    p = Pool(2)
    tx = p.reserve(2)
    p.cancel(tx)
    assert p.counts() == (0, 0, 0, 0, 2)
    expect_error(lambda: p.publish(tx), ValueError)
    audit_events(p.log, 2)

    p = Pool(2)
    tx = p.reserve(2)
    first, last = p.submit(tx), p.submit(tx)
    expect_error(lambda: p.publish(tx), ValueError)
    p.cancel(tx)
    assert p.counts() == (0, 0, 0, 2, 0) and p.reserve(1) is None
    before = p.counts()
    assert p.complete((tx, 99999)) == [] and p.counts() == before
    p.complete(first)
    assert p.counts() == (0, 0, 0, 2, 0)
    p.complete(last)
    replacement = p.reserve(2)
    before = p.counts()
    assert p.complete(last) == [] and p.counts() == before
    expect_error(lambda: p.submit(tx), ValueError)
    p.cancel(replacement)
    assert audit_events(p.log, 2) == (0, 0)

    p = Pool(2)
    tx = p.reserve(2)
    commit_reserved(p, tx)
    read = p.submit(tx, 'L')
    p.cancel(tx)
    assert p.counts() == (0, 0, 0, 2, 0)
    p.complete(read)
    assert audit_events(p.log, 2) == (0, 0)

    p = Pool(3)
    tx = p.reserve(2, 1)
    partial = p.submit(tx, limit=1)
    protected = p.operations[partial][0][0][0]
    p.cancel(tx)
    assert p.counts() == (0, 0, 0, 1, 2)
    replacement = p.reserve(2)
    assert all(k[0] != protected for k in p.records
               if p.records[k]['owner'] == replacement)
    p.complete(partial)
    p.cancel(replacement)
    assert audit_events(p.log, 3) == (0, 0)

    p = Pool(1)
    tx = p.reserve(1)
    p.submit(tx)
    p.cancel(tx)
    assert p.counts() == (0, 0, 0, 1, 0) and p.reserve(1) is None
    assert audit_events(p.log, 1) == (1, 1)

    for published in (False, True):
        p = Pool(2)
        tx = p.reserve(2)
        if published:
            commit_reserved(p, tx)
        op = p.submit(tx, 'L' if published else 'R')
        old_pages = {k[0] for k in p.operations[op][0]}
        p.cancel(tx, early=True)
        replacement = p.reserve(2)
        assert old_pages == {k[0] for k, r in p.records.items()
                             if r['owner'] == replacement}
        before = p.counts()
        assert any(a != b for a, b in p.complete(op))
        assert p.counts() == before
        expect_error(lambda: audit_events(p.log, 2))
    print('lifetime: three cancellation paths, partial acceptance, last use and stale events checked; actual early-reuse mutants rejected')


def test_tail_edges_and_failure():
    for length in (0, 1, 3, 4, 5):
        p, tx, tails = make_tails(ceil(length / 4) + 1, 1, length, 4)
        initial_live = p.counts()[0]
        pending = tails.start([0])
        assert pending is not None and p.counts()[2] == 1
        assert tails.lengths == [length]
        tails.commit(pending)
        expected_live = initial_live + int(length % 4 == 0)
        assert p.counts()[0] == expected_live
        assert tails.lengths == [length + 1]
        audit_events(p.log, p.total)

    p, tx, tails = make_tails(1, 1, 1, 4)
    before = (p.counts(), tuple(p.log), tuple(tails.views[0]), tuple(tails.lengths))
    assert tails.start([0]) is None
    assert before == (p.counts(), tuple(p.log), tuple(tails.views[0]), tuple(tails.lengths))

    p, tx, tails = make_tails(3, 1, 2, 4)
    pending = tails.start([0])
    tails.write(pending, limit=1)
    tails.abort(pending)
    assert p.counts() == (1, 0, 0, 1, 1)
    assert tails.lengths == [2]
    expect_error(lambda: tails.commit(pending), ValueError)
    assert all(a == b for a, b in p.complete(pending['op']))
    assert p.counts() == (1, 0, 0, 0, 2)
    tails.check_reference()
    audit_events(p.log, 3)

    p, tx, tails = make_tails(3, 1, 3, 4)
    old = tails.views[0][-1]
    old_reader = p.accept(tx, [old])
    pending = tails.start([0])
    tails.commit(pending)
    assert tails.lengths == [4]
    assert p.counts() == (1, 0, 0, 1, 1)
    assert p.records[old]['state'] == 'D'
    assert all(a == b for a, b in p.complete(old_reader))
    assert p.counts() == (1, 0, 0, 0, 2)
    audit_events(p.log, 3)
    print('tail boundaries: empty/full/partial; zero-free replacement denied; failed private write retains old length; delayed old reader retains D')


def test_growth():
    p = Pool(800)
    assert p.reserve(776, 32) is None
    p, tx, tails = make_tails(800, 30, 512, 16, guards=30, shared=128)
    trace = [p.counts()]
    assert trace[-1] == (728, 30, 0, 0, 42)
    pending = tails.start(range(30))
    assert p.counts() == (728, 0, 30, 0, 42)
    trace.append(p.counts())
    tails.commit(pending)
    assert p.counts() == (758, 0, 0, 0, 42)
    trace.append(p.counts())
    assert p.refill(tx, 30)
    trace.append(p.counts())
    assert trace[-1] == (758, 30, 0, 0, 12)
    copied_tokens = 0
    for length in range(513, 528):
        old_tail_pages = {v[-1] for v in tails.views}
        pending = tails.start(range(30))
        assert p.counts() == (758, 0, 30, 0, 12)
        assert tails.lengths == [length] * 30
        copied_tokens += sum(len(tails.data[old])
                             for _, _, old, _ in pending['entries'])
        if length == 513:
            trace.append(p.counts())
        tails.commit(pending)
        assert p.counts() == (758, 0, 0, 0, 42)
        assert not old_tail_pages.intersection({v[-1] for v in tails.views})
        assert tails.lengths == [length + 1] * 30
        if length == 513:
            trace.append(p.counts())
        assert p.refill(tx, 30)
        assert p.counts() == (758, 30, 0, 0, 12)
    assert copied_tokens == 3600
    assert Fraction(copied_tokens * 48, 1024) == Fraction(675, 4)
    trace.append(p.counts())
    tails.commit(tails.start(range(30)))
    assert tails.lengths == [529] * 30
    assert p.counts() == (788, 0, 0, 0, 12)
    trace.append(p.counts())
    before = (p.counts(), len(p.log), tuple(tails.lengths))
    assert not p.refill(tx, 30)
    assert tails.start(range(30)) is None
    assert before == (p.counts(), len(p.log), tuple(tails.lengths))
    assert sum(ceil((n + 1) / 16) - ceil(n / 16) for n in tails.lengths) == 0
    for selected in (range(12), range(12, 24), range(24, 30)):
        tails.commit(tails.start(selected))
        assert p.counts() == (788, 0, 0, 0, 12)
    assert tails.lengths == [530] * 30
    assert audit_events(p.log, 800) == (788, 0)
    assert all(sum(row) == 800 for row in trace)
    assert sum((758, 30, 0, 0, 42)) != 800
    assert sum((758, 0, 0, 0, 12)) != 800
    print('growth: (L,H,R,D,F) trace=', trace)
    print('growth: 15 replacements copy 3600 valid states = 168.75 MiB each direction; all-30 denied at F=12, subsets succeed after old-page release')


def test_net_zero_mutant():
    p, tx, tails = make_tails(800, 30, 529, 16, shared=128)
    assert p.counts() == (788, 0, 0, 0, 12)
    old_views = [tuple(v) for v in tails.views]
    frozen = {k: tails.data[k] for v in old_views for k in v}
    net_growth = sum(ceil((n + 1) / 16) - ceil(n / 16) for n in tails.lengths)
    assert net_growth == 0
    # Deliberate wrong implementation: budget only net growth and use old
    # tails as write destinations. It accepts thirty writes with twelve free.
    wrong_reservation = p.take(tx, net_growth)
    assert wrong_reservation == []
    wrong_destinations = [view[-1] for view in tails.views]
    for request, destination in enumerate(wrong_destinations):
        tails.data[destination] += (token_at(request, 529, 128),)
    assert any(tails.data[k] != value for k, value in frozen.items())
    assert p.counts() == (788, 0, 0, 0, 12)
    expect_error(lambda: audit_private_destinations(old_views, wrong_destinations))
    expect_error(lambda: audit_frozen(tails.data, frozen))
    print('negative control: net-zero mutant accepts 30 with F=12 and mutates published tails; independent alias and frozen-content audits both reject it')


def test_service():
    checked = 0
    for rate in range(7):
        for demand in range(5):
            for capacity in range(1, 5):
                resources = [(demand, capacity), (1, 3)]
                assert plan_instances(rate, resources) == enumerate_instances(rate, resources)
                checked += 1
    resources = [(1000, 2500), (100, 250), (512, 800)]
    assert plan_instances(4, resources) == enumerate_instances(4, resources) == 3
    assert 4 * 512 > 2 * 800
    assert max((4 * d) // c for d, c in resources) == 2
    assert plan_instances(4, resources[:2]) == 2
    assert required_instances(0, 1, 0) == required_instances(4, 0, 0) == 0
    assert plan_instances(4, []) == 0
    expect_error(lambda: required_instances(1, 1, 0), ValueError)
    expect_error(lambda: required_instances(-1, 1, 1), ValueError)
    assert enumerate_instances(1, [(1, 0)]) is None
    assert phase_ratio(1, 4, 1, 1) == Fraction(1, 4)
    assert Fraction(enumerate_instances(1, [(1, 1)]),
                    enumerate_instances(4, [(1, 1)])) == Fraction(1, 4)
    assert phase_ratio(1, 4, 1, 1) != Fraction(1, 1)
    assert phase_ratio(0, 4, 0, 1) == 0
    assert phase_ratio(4, 0, 1, 0) is None
    assert phase_ratio(0, 0, 0, 0) is None
    expect_error(lambda: phase_ratio(1, 4, 0, 1), ValueError)
    for lp in range(1, 7):
        for ld in range(1, 7):
            for rp in range(1, 4):
                for rd in range(1, 4):
                    if lp % rp == 0 and ld % rd == 0:
                        np = enumerate_instances(lp, [(1, rp)])
                        nd = enumerate_instances(ld, [(1, rd)])
                        assert Fraction(np, nd) == phase_ratio(lp, ld, rp, rd)
    assert phase_ratio(1, 1, 2, 3) == Fraction(3, 2)
    assert required_instances(1, 1, 2) == required_instances(1, 1, 3) == 1
    print('service: integer enumeration cases=', checked,
          '; network lower bound 3 versus faulty 2; unequal-phase ratio 1/4; zero demand and rounding checked')


def test_backlog():
    segments = [(160, Fraction(2)), (60, Fraction(1)), (160, Fraction(2))]
    assert fluid_backlog(segments, 100) == stepped_reference(segments, 100) == 200
    reset_control = Fraction(0)
    for rate, duration in segments:
        reset_control = max(Fraction(0), (rate - 100) * duration)
    assert reset_control == 120
    irregular = [(130, Fraction(1, 37)), (90, Fraction(1, 17)), (180, Fraction(2, 31))]
    assert fluid_backlog(irregular, 100) == stepped_reference(irregular, 100)
    assert fluid_backlog([], 100) == 0
    assert fluid_backlog([(50, Fraction(10))], 100) == 0
    assert fluid_backlog([(3, Fraction(2))], 0) == 6
    expect_error(lambda: fluid_backlog([(-1, Fraction(1))], 100), ValueError)
    print('backlog: carried residual 200 versus reset control 120; fractional final steps checked')


def main():
    print('scope: deterministic CPU arithmetic and serialized event model; no GPU, NIC, visibility, bandwidth, quality or SLO measurements')
    test_pages()
    test_lifetimes()
    test_tail_edges_and_failure()
    test_growth()
    test_net_zero_mutant()
    test_service()
    test_backlog()
    print('all scoped checks passed')


if __name__ == '__main__':
    main()
