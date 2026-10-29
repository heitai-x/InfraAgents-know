from copy import deepcopy

CHECKS = 0
NEGATIVES = []
SETTLEMENT_REPORTS = []


def require(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def detected(label, action):
    try:
        action()
    except AssertionError:
        NEGATIVES.append(label)
    else:
        raise AssertionError('negative control escaped: ' + label)


def additional_pages(c, q, r, b, held):
    return max(0, (c + q + r + b - 1) // b - held)


def match(n, b, full, window, w, logits=False):
    cap = n if logits else max(0, n - 1)
    p = 0
    while p + b <= cap and p // b in full:
        p += b
    while p > 0:
        first = max(0, p - w + 1)
        if all(t // b in window for t in range(first, p)):
            return p
        p -= b
    return 0


def reference_match(n, b, full, window, w, logits=False):
    cap = n if logits else max(0, n - 1)
    legal = [0]
    for p in range(1, cap + 1):
        if p % b:
            continue
        full_tokens = {t for t in range(p) if t // b in full}
        needed_window = {t for t in range(p) if t >= p - w + 1}
        window_tokens = {t for t in range(p) if t // b in window}
        if full_tokens == set(range(p)) and needed_window <= window_tokens:
            legal.append(p)
    return max(legal)


class Pool:
    def __init__(self, count, block=4, fault=None):
        self.block = block
        self.pages = [dict(epoch=0, data=[None] * block, valid=set(),
                           owners=set(), holds=set(), uses=set(),
                           private=None, cached=False)
                      for _ in range(count)]
        self.ops = {}
        self.views = {}
        self.closed = set()
        self.transactions = {}
        self.settlement_events = []
        self.fault = fault

    def transaction(self, tx):
        if tx not in self.transactions:
            self.transactions[tx] = dict(state='ACTIVE', held=set(),
                                         targets=set(), receipt=None,
                                         settled=False)
        return self.transactions[tx]

    def handle(self, index):
        return (index, self.pages[index]['epoch'])

    def current(self, handle):
        i, epoch = handle
        return 0 <= i < len(self.pages) and self.pages[i]['epoch'] == epoch

    def cache(self, index, values):
        p = self.pages[index]
        p['data'] = list(values) + [None] * (self.block - len(values))
        p['valid'] = set(range(len(values)))
        p['cached'] = True
        return self.handle(index)

    def eligible(self, index):
        p = self.pages[index]
        return not (p['owners'] or p['holds'] or p['uses'] or p['private'] is not None)

    def pin(self, tx, handle):
        record = self.transaction(tx)
        require(record['state'] == 'ACTIVE', 'pin after transaction closed')
        require(self.current(handle), 'pin stale page')
        p = self.pages[handle[0]]
        require(p['cached'] and len(p['valid']) == self.block, 'pin incomplete cache')
        p['holds'].add(tx)
        record['held'].add(handle)

    def reserve(self, tx, count):
        require(count >= 0, 'negative demand')
        record = self.transactions.get(tx)
        if record is not None and record['state'] != 'ACTIVE':
            return None
        candidates = [i for i in range(len(self.pages)) if self.eligible(i)]
        if len(candidates) < count:
            return None
        record = self.transaction(tx)
        result = []
        for i in candidates[:count]:
            p = self.pages[i]
            p['cached'] = False
            p['epoch'] += 1
            p['data'] = [None] * self.block
            p['valid'].clear()
            p['private'] = tx
            h = self.handle(i)
            result.append(h)
            record['targets'].add(h)
        return result

    def begin(self, uid, reads=(), writes=(), tx=None):
        desc = (tuple(reads), tuple(writes), tx)
        if uid in self.ops:
            if self.ops[uid]['desc'] != desc:
                raise ValueError('operation identity conflict')
            return False
        if tx is not None:
            record = self.transactions.get(tx)
            if record is None or record['state'] != 'ACTIVE':
                return False
        refs = tuple(sorted(set(reads) | {item[0] for item in writes}))
        for handle in refs:
            require(self.current(handle), 'submit stale page')
        for handle, row, value in writes:
            p = self.pages[handle[0]]
            require(0 <= row < self.block, 'write outside page')
            require(p['private'] is not None and not p['uses'], 'write not private or ordered')
        self.ops[uid] = dict(desc=desc, refs=refs, writes=tuple(writes),
                             tx=tx, physical_done=False)
        for handle in refs:
            self.pages[handle[0]]['uses'].add(uid)
        return True

    def complete(self, uid, refs=None):
        op = self.ops.get(uid)
        if op is None or op['physical_done']:
            return False
        if refs is not None and tuple(refs) != op['refs']:
            return False
        # Physical writes target original addresses, regardless of a host epoch.
        for handle, row, value in op['writes']:
            p = self.pages[handle[0]]
            p['data'][row] = value
            p['valid'].add(row)
        op['physical_done'] = True
        if not all(self.current(h) for h in op['refs']):
            return False
        for handle in op['refs']:
            self.pages[handle[0]]['uses'].remove(uid)
        if op['tx'] is not None:
            self.settle_tx(op['tx'])
        return True

    def settle_tx(self, tx):
        record = self.transactions.get(tx)
        if record is None or record['state'] == 'ACTIVE':
            return
        live = {h for op in self.ops.values()
                if op['tx'] == tx and not op['physical_done']
                for h in op['refs']}
        selected = set(record['receipt'][1]) if record['receipt'] is not None else set()
        for h in sorted(tuple(record['held'])):
            require(self.current(h), 'held object lost original generation')
            if h in live:
                continue
            # Fault variants execute omitted cleanup; observers detect the result.
            if self.fault == 'omit_commit_h' and record['state'] == 'COMMITTED':
                continue
            if self.fault == 'target_handles_only' and h not in selected:
                continue
            if (self.fault == 'omit_abort_source_h' and
                    record['state'] == 'ABORTING' and h not in record['targets']):
                continue
            p = self.pages[h[0]]
            self.settlement_events.append((tx, h, tuple(sorted(p['owners'])),
                                           tuple(sorted(p['uses']))))
            p['holds'].discard(tx)
            record['held'].remove(h)
        for h in sorted(record['targets']):
            if not self.current(h):
                continue
            p = self.pages[h[0]]
            if p['private'] == tx and not p['uses']:
                p['private'] = None
        record['settled'] = not record['held'] and not any(
            self.current(h) and self.pages[h[0]]['private'] == tx
            for h in record['targets'])

    def publish(self, req, tx, handles, length):
        record = self.transactions.get(tx)
        receipt = (req, tuple(handles), length)
        if record is None:
            return False
        if record['state'] != 'ACTIVE':
            # A retry returns its receipt without reacquiring any ownership.
            return record['state'] == 'COMMITTED' and record['receipt'] == receipt
        if req in self.closed or not 0 <= length <= len(handles) * self.block:
            return False
        if not all(self.current(h) for h in handles):
            return False
        for t in range(length):
            p = self.pages[handles[t // self.block][0]]
            if t % self.block not in p['valid']:
                return False
        selected = set(handles)
        for op in self.ops.values():
            if not op['physical_done'] and any(h in selected for h, row, value in op['writes']):
                return False
        if any(self.pages[h[0]]['private'] not in (None, tx) for h in handles):
            return False
        old_handles = set(self.views.get(req, ((), 0))[0])
        # This single-thread model treats the following as one commit boundary.
        for h in handles:
            p = self.pages[h[0]]
            p['owners'].add(req)
            if p['private'] == tx:
                p['private'] = None
        self.views[req] = (tuple(handles), length)
        for h in old_handles - selected:
            self.pages[h[0]]['owners'].discard(req)
        record['receipt'] = receipt
        record['state'] = 'COMMITTED'
        # Enumerate the acquisition ledger, including non-target sources.
        self.settle_tx(tx)
        return True

    def retire(self, req):
        self.closed.add(req)
        self.views.pop(req, None)
        for p in self.pages:
            p['owners'].discard(req)

    def release_tx(self, tx):
        record = self.transactions.get(tx)
        if record is None:
            return
        if record['state'] == 'ACTIVE':
            record['state'] = 'ABORTING'
        self.settle_tx(tx)


def dense(pool, view):
    handles, length = view
    return [pool.pages[handles[t // pool.block][0]]['data'][t % pool.block]
            for t in range(length)]


def audit(pool):
    # Reconstruct obligations from the independent operation records.
    for uid, op in pool.ops.items():
        if not op['physical_done']:
            for h in op['refs']:
                require(pool.current(h), 'unfinished operation lost physical generation')
                require(uid in pool.pages[h[0]]['uses'], 'unfinished operation lost protection')
    for i, p in enumerate(pool.pages):
        require(not (p['owners'] & pool.closed), 'closed owner regained a page')
        for uid in p['uses']:
            op = pool.ops[uid]
            require(not op['physical_done'] and pool.handle(i) in op['refs'], 'spurious use')
    require(not (set(pool.views) & pool.closed), 'closed owner regained a view')


def settlement_observations(pool, tx, acquired, private_targets, other_holds, committed):
    # The fixture retains the original manifest independently of record['held'].
    # Expected responsibilities come from operation history and request views,
    # not eligible(), page holds, the settlement flag, or the cleanup loop.
    live_tx = {h for op in pool.ops.values()
               if op['tx'] == tx and not op['physical_done'] for h in op['refs']}
    all_live = {h for op in pool.ops.values() if not op['physical_done'] for h in op['refs']}
    owned = {h for handles, length in pool.views.values() for h in handles}
    expected_h = set(acquired) & live_tx
    expected_private = set() if committed else set(private_targets) & all_live
    actual_h = {pool.handle(i) for i, p in enumerate(pool.pages) if tx in p['holds']}
    actual_other = {pool.handle(i) for i, p in enumerate(pool.pages) if 'OTHER' in p['holds']}
    actual_private = {pool.handle(i) for i, p in enumerate(pool.pages) if p['private'] == tx}
    protected = owned | all_live | expected_h | expected_private | set(other_holds)
    expected_free = [pool.handle(i) for i in range(len(pool.pages))
                     if pool.handle(i) not in protected]
    actual_free = [pool.handle(i) for i in range(len(pool.pages)) if pool.eligible(i)]
    return dict(retained_h=sorted(actual_h - expected_h),
                missing_h=sorted(expected_h - actual_h),
                expected_free=expected_free, actual_free=actual_free,
                expected_private=sorted(expected_private), actual_private=sorted(actual_private),
                other_preserved=actual_other == set(other_holds),
                valid=(actual_h == expected_h and actual_free == expected_free and
                       actual_private == expected_private and actual_other == set(other_holds)))


def audit_settlement(pool, tx, acquired, private_targets, other_holds, committed):
    observed = settlement_observations(pool, tx, acquired, private_targets, other_holds, committed)
    require(observed['valid'], 'settlement responsibility/capacity mismatch: ' + repr(observed))
    return observed


def force_publish(pool, req, handles, length):
    # Deliberately wrong: bypass readiness and cancellation checks.
    pool.views[req] = (tuple(handles), length)
    for h in handles:
        pool.pages[h[0]]['owners'].add(req)


def check_geometry():
    cases = 0
    for b in range(1, 9):
        for c in range(17):
            for q in range(9):
                for r in range(6):
                    for extra in range(3):
                        held = (c + b - 1) // b + extra
                        required = {t // b for t in range(c + q + r)}
                        missing = required - set(range(held))
                        require(additional_pages(c, q, r, b, held) == len(missing), 'capacity mismatch')
                        cases += 1
    detected('floor instead of ceiling',
             lambda: require(max(0, 5 // 4 - 1) == len({t // 4 for t in range(5)} - {0}),
                             'floor underallocates crossing'))
    require(additional_pages(31, 2, 0, 16, 2) == 1, 'ordinary extension')
    require(additional_pages(31, 2, 16, 16, 2) == 2, 'lookahead extension')
    require(additional_pages(8, 2, 3, 4, 2) == 2, 'combined reservation')
    require(10 * 11 * 16 * 4096 == 7208960, 'grouped byte count')
    require(6 * 65536 + 15 * 4096 == 454656, 'full layer offset')
    require(10 * 65536 + 15 * 4096 == 716800, 'window layer offset')
    return cases


def check_recovery():
    cases = 0
    for b in range(1, 5):
        for n in range(13):
            for w in range(1, 6):
                for mask in range(16):
                    full = {j for j in range(4) if mask & (1 << j)}
                    other = (mask * 7 + 3) & 15
                    window = {j for j in range(4) if other & (1 << j)}
                    require(match(n, b, full, window, w) ==
                            reference_match(n, b, full, window, w), 'recovery mismatch')
                    cases += 1
    require(match(32, 16, {0, 1}, {0, 1}, 1) == 16, 'aligned prompt needs a block replay')
    require(match(33, 16, {0, 1}, {0, 1}, 1) == 32, 'one-token replay')
    require(match(32, 16, {0, 1}, {0, 1}, 1, True) == 32, 'saved-logits boundary')
    require(match(0, 4, set(), set(), 1) == 0, 'empty recovery')
    a, b = {0, 4, 8, 12}, {0, 6, 10}
    detected('minimum of maxima across sparse checkpoints',
             lambda: require(min(max(a), max(b)) in a & b, 'not a common checkpoint'))
    return cases


def check_lifecycle():
    empty = Pool(0)
    require(empty.reserve('empty', 0) == [], 'zero demand')
    require(empty.reserve('empty', 1) is None, 'empty pool')
    pool = Pool(6)
    h0 = pool.cache(0, range(4))
    h1 = pool.cache(1, range(100, 104))
    pool.pages[1]['cached'] = False
    require(pool.begin('old-reader', reads=(h1,)), 'old reader submission')
    pool.pin('T', h0)
    targets = pool.reserve('T', 3)
    require([h[0] for h in targets] == [2, 3, 4], 'protected cache must not be allocated')
    h2, h3, h4 = targets
    handles = (h0, h2, h3, h4)
    require(sum(not pool.eligible(i) for i in range(6)) * 128 == 640, 'protected peak')
    before = deepcopy(pool.__dict__)
    require(pool.reserve('other', 2) is None, 'joint capacity failure')
    require(pool.__dict__ == before, 'failed reservation mutated state')
    writes = tuple((h2, row, row + 4) for row in range(4))
    require(pool.begin('external', writes=writes, tx='T'), 'external submit')
    require(not pool.publish('A', 'T', handles, 8), 'must reject incomplete recovery')
    bad = deepcopy(pool)
    force_publish(bad, 'A', handles, 8)
    detected('directory allocation treated as ready',
             lambda: require(dense(bad, bad.views['A']) == list(range(8)), 'unarrived payload'))
    before = deepcopy(pool.__dict__)
    require(not pool.complete('unknown'), 'unknown event')
    require(not pool.complete('external', refs=(pool.handle(5),)), 'wrong event fields')
    require(pool.__dict__ == before, 'invalid event changed obligations')
    require(pool.complete('external'), 'external completion')
    require(not pool.begin('external', writes=writes, tx='T'), 'completed uid must not resubmit')
    work = ((h3, 0, 8), (h3, 1, 9))
    require(pool.begin('compute', reads=(h0, h2), writes=work, tx='T'), 'compute submit')
    require(not pool.publish('A', 'T', handles, 10), 'must reject pending compute')
    audit(pool)

    success = deepcopy(pool)
    require(success.complete('compute'), 'successful compute')
    require(success.publish('A', 'T', handles, 10), 'successful publication')
    snapshot = success.views['A']
    require(dense(success, snapshot) == list(range(10)), 'logical token reference')
    require(success.begin('consumer', reads=handles[:3]), 'snapshot reader')
    success.retire('A')
    require(all(not success.eligible(h[0]) for h in handles[:3]), 'reader protects retired pages')
    require(success.eligible(h4[0]), 'unused lookahead can return')
    require(dense(success, snapshot) == list(range(10)), 'old snapshot remains intact')
    audit(success)
    require(success.complete('consumer'), 'consumer finished')
    require(success.complete('old-reader'), 'old reader finished')
    require(all(success.eligible(i) for i in range(6)), 'successful transaction leaked capacity')
    audit(success)

    # Cancellation enters before the first request view was published.
    pool.retire('A')
    pool.release_tx('T')
    require(pool.eligible(h4[0]), 'unused private page not returned')
    require(all(not pool.eligible(h[0]) for h in (h0, h2, h3)), 'cancel released live access')
    require('T' in pool.pages[h0[0]]['holds'], 'conservative source H removed before last use')
    require(sum(not pool.eligible(i) for i in range(6)) * 128 == 512, 'cancel footprint')
    audit(pool)
    require(pool.complete('compute'), 'drain compute')
    require('T' not in pool.pages[h0[0]]['holds'], 'completion did not settle source H')
    require(pool.transactions['T']['settled'], 'completion did not finish transaction cleanup')
    require(not pool.publish('A', 'T', handles, 10), 'completion must not reopen cancelled owner')
    bad = deepcopy(pool)
    force_publish(bad, 'A', handles, 10)
    detected('late publish after prepublication cancellation', lambda: audit(bad))
    before = deepcopy(pool.__dict__)
    pool.release_tx('T')
    require(pool.__dict__ == before, 'repeated cleanup changed settled transaction')
    require(not pool.eligible(1), 'unrelated old reader still protects its page')
    require(pool.complete('old-reader'), 'final old reader')
    require(all(pool.eligible(i) for i in range(6)), 'drained pages unavailable')
    require(pool.reserve('next', 6) is not None, 'safe reuse')
    before = deepcopy(pool.__dict__)
    require(not pool.complete('compute'), 'duplicate completion after reuse')
    require(pool.__dict__ == before, 'old completion settled new generation')
    audit(pool)

    # Execute the erroneous release, reuse, and old physical write.
    bad = Pool(1)
    old = bad.reserve('old', 1)[0]
    require(bad.begin('delayed-write', writes=((old, 0, 7),), tx='old'), 'delayed write')
    bad.pages[0]['private'] = None
    bad.pages[0]['uses'].clear()
    new = bad.reserve('new', 1)[0]
    bad.pages[0]['data'][0] = 99
    bad.pages[0]['valid'].add(0)
    detected('reuse with outstanding physical lease', lambda: audit(bad))
    require(not bad.complete('delayed-write'), 'stale bookkeeping should reject old generation')
    require(new != old, 'negative control must actually reuse a new generation')
    detected('epoch cannot stop old physical write',
             lambda: require(bad.pages[new[0]]['data'][0] == 99, 'new content overwritten'))


def settlement_fixture(active, fault=None):
    pool = Pool(5, fault=fault)
    hit = pool.cache(0, range(4))
    source = pool.cache(1, range(100, 104))
    unused = pool.cache(2, range(200, 204))
    acquired = (hit, source, unused)
    for h in acquired:
        pool.pin('T', h)
    pool.pin('OTHER', source)
    targets = tuple(pool.reserve('T', 1))
    require(len(targets) == 1 and targets[0][0] == 3, 'fixture target allocation')
    target = targets[0]
    writes = tuple((target, row, row + 4) for row in range(4))
    require(pool.begin('init-target', writes=writes, tx='T'), 'fixture initialize target')
    require(pool.complete('init-target'), 'fixture target ready')
    if active:
        require(pool.begin('source-read', reads=(hit, source), tx='T'), 'fixture source read')
    return pool, acquired, targets, (hit, target), {source}


def check_h_settlement():
    cases = 0
    for outcome in ('commit', 'failure', 'cancel'):
        for active in (False, True):
            pool, acquired, targets, handles, other_holds = settlement_fixture(active)
            committed = outcome == 'commit'
            hit, source, unused = acquired
            if committed:
                require(pool.publish('A', 'T', handles, 8), 'settlement commit')
                # Prove takeover before retiring the request or starting readers.
                require(pool.views['A'] == (handles, 8), 'new request view missing')
                require(all('A' in pool.pages[h[0]]['owners'] for h in handles), 'O takeover missing')
                require(all(pool.pages[h[0]]['private'] is None for h in targets), 'T not transferred')
                if not active:
                    events = [e for e in pool.settlement_events if e[0] == 'T' and e[1] == hit]
                    require(len(events) == 1 and 'A' in events[0][2], 'H released before O takeover')
                require(pool.begin('consumer', reads=handles), 'request consumer')
                pool.retire('A')
            else:
                if outcome == 'cancel':
                    pool.retire('A')
                pool.release_tx('T')
                require('A' not in pool.views, 'abort installed request view')
            audit_settlement(pool, 'T', acquired, targets, other_holds, committed)
            require('T' not in pool.pages[unused[0]]['holds'], 'unused H not immediately settled')
            if active:
                require('T' in pool.pages[source[0]]['holds'], 'source H missing before last use')
                require(not pool.transactions['T']['settled'], 'premature settled state')
                require(pool.complete('source-read'), 'last transaction source use')
                # No release_tx or request event is inserted after this completion.
                require(all('T' not in p['holds'] for p in pool.pages), 'last use did not clear all T H')
            if committed:
                require(pool.complete('consumer'), 'request last reader')
            observed = audit_settlement(pool, 'T', acquired, targets, other_holds, committed)
            require(pool.transactions['T']['settled'], 'terminal transaction not settled')
            require([h[0] for h in observed['actual_free']] == [0, 2, 3, 4], 'capacity not restored')
            require(pool.pages[source[0]]['holds'] == {'OTHER'}, 'other transaction H changed')
            audit(pool)
            before = deepcopy(pool.__dict__)
            pool.release_tx('T')
            pool.settle_tx('T')
            if active:
                require(not pool.complete('source-read'), 'duplicate source completion')
            if committed:
                require(pool.publish('A', 'T', handles, 8), 'commit retry lost original receipt')
            else:
                require(not pool.publish('A', 'T', handles, 8), 'abort retry acquired publish right')
            require(pool.__dict__ == before, 'duplicate terminal settlement mutated state')
            require(not pool.begin('late-work', reads=(hit,), tx='T'), 'closed tx authorized new work')
            pool.release_tx('OTHER')
            require(all(pool.eligible(i) for i in range(5)), 'other H did not settle independently')
            SETTLEMENT_REPORTS.append('outcome=' + outcome + '; active_initial=' + str(active) +
                                      '; free_before_other_release=[0, 2, 3, 4]')
            cases += 1

    for fault in ('omit_commit_h', 'target_handles_only', 'omit_abort_source_h'):
        pool, acquired, targets, handles, other_holds = settlement_fixture(True, fault)
        committed = fault != 'omit_abort_source_h'
        if committed:
            require(pool.complete('source-read'), 'negative source last use')
            require(pool.publish('A', 'T', handles, 8), 'negative commit executes')
            require(pool.begin('consumer', reads=handles), 'negative consumer')
            pool.retire('A')
            require(pool.complete('consumer'), 'negative consumer finishes')
        else:
            pool.retire('A')
            pool.release_tx('T')
            require(pool.complete('source-read'), 'negative abort last use')
        # Construct observations outside the expected-failure capture.
        observed = settlement_observations(pool, 'T', acquired, targets, other_holds, committed)
        require(bool(observed['retained_h']), 'fault did not actually leave H')
        require(len(observed['actual_free']) < len(observed['expected_free']), 'fault did not retain capacity')
        require(observed['other_preserved'], 'fault accidentally changed OTHER H')
        detected(fault, lambda: audit_settlement(pool, 'T', acquired, targets, other_holds, committed))
        SETTLEMENT_REPORTS.append('fault=' + fault + '; retained_h=' + repr(observed['retained_h']) +
                                  '; actual_free=' + repr(observed['actual_free']) +
                                  '; expected_free=' + repr(observed['expected_free']))
    return cases


def check_cow():
    original = [4, 5, None, None]
    a, b = original[:], original[:]
    a[2], b[2] = 6, 60
    require(a[:3] == [4, 5, 6] and b[:3] == [4, 5, 60], 'isolated fork outputs')
    require(original == [4, 5, None, None], 'shared source changed')
    wrong_a = original[:]
    wrong_b = wrong_a
    wrong_a[2] = 6
    wrong_b[2] = 60
    detected('shared tail without isolation',
             lambda: require(wrong_a[:3] == [4, 5, 6], 'second fork overwrites first'))
    require(2 * 2 * 32 + 32 == 160, 'effective COW logical traffic')


def main():
    capacity = check_geometry()
    recovery = check_recovery()
    check_lifecycle()
    settlement = check_h_settlement()
    check_cow()
    print('scope=deterministic CPU symbolic pages and token identities; no GPU or real DMA')
    print('capacity_cases=' + str(capacity) + '; recovery_cases=' + str(recovery))
    print('h_settlement_positive_trajectories=' + str(settlement))
    for observed in SETTLEMENT_REPORTS:
        print('settlement_observation=' + observed)
    print('negative_controls_detected=' + str(len(NEGATIVES)))
    for label in NEGATIVES:
        print('detected=' + label)
    print('assertions_evaluated=' + str(CHECKS))
    print('limits=no upstream execution, real model numerics, hardware visibility, concurrency timing, or performance')


if __name__ == '__main__':
    main()
