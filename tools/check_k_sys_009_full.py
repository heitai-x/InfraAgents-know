'''K-SYS-009: deterministic offline CPU models, not GPU/NIC measurements.'''
import itertools
import json
from fractions import Fraction
from random import Random


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def rejected(action):
    try:
        action()
    except (AssertionError, ValueError):
        return
    raise AssertionError('faulty case escaped detection')


def merge_ranges(spans, limit):
    if not isinstance(limit, int) or limit < 0:
        raise ValueError('invalid limit')
    work = []
    for a, b in spans:
        if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a <= b <= limit:
            raise ValueError('invalid range')
        if a < b:
            work.append((a, b))
    result = []
    for a, b in sorted(work):
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], b))
        else:
            result.append((a, b))
    return result


def test_ranges():
    rng = Random(9009)
    count = 0
    for n in range(9):
        for unused in range(40):
            spans = [tuple(sorted((rng.randrange(n + 1), rng.randrange(n + 1))))
                     for j in range(rng.randrange(7))]
            got = merge_ranges(spans, n)
            reference = {i for i in range(n) if any(a <= i < b for a, b in spans)}
            actual = {i for a, b in got for i in range(a, b)}
            need(actual == reference, 'interval membership differs from position oracle')
            need(sum(b - a for a, b in got) == len(reference), 'union size mismatch')
            need(all(got[j][1] < got[j + 1][0] for j in range(len(got) - 1)), 'noncanonical union')
            count += 1
    for spans, n in [([(-1, 1)], 2), ([(2, 1)], 2), ([(0, 3)], 2)]:
        rejected(lambda spans=spans, n=n: merge_ranges(spans, n))
    spans = [(0, 2), (0, 2)]
    faulty_complete = sum(b - a for a, b in spans) >= 4
    reference_complete = all(any(a <= i < b for a, b in spans) for i in range(4))
    need(faulty_complete and not reference_complete, 'duplicate-count negative control')
    return count


def value(l, kind, t, h, c):
    return 1 + 10000 * l + 1000 * kind + 100 * t + 10 * h + c


def physical_reference(l, kind, heads, n, page_size, table, layout, width):
    # Independent oracle: enumerate physical slots and invert the page table.
    # No use of the transfer address formula, task planner or task keys.
    inverse = {physical: logical for logical, physical in enumerate(table)}
    result = []
    for physical in range(max(table, default=-1) + 1):
        if layout == 'HND':
            slots = ((j, u, c) for j in range(len(heads))
                     for u in range(page_size) for c in range(width))
        elif layout == 'NHD':
            slots = ((j, u, c) for u in range(page_size)
                     for j in range(len(heads)) for c in range(width))
        else:
            raise ValueError('unknown layout')
        for j, u, c in slots:
            logical = inverse.get(physical)
            t = -1 if logical is None else logical * page_size + u
            result.append(value(l, kind, t, heads[j], c) if 0 <= t < n else -1)
    return result


def address(table, page_size, local_heads, t, local_h, c, width, layout):
    physical = table[t // page_size]
    u = t % page_size
    if layout == 'HND':
        return ((physical * local_heads + local_h) * page_size + u) * width + c
    if layout == 'NHD':
        return ((physical * page_size + u) * local_heads + local_h) * width + c
    raise ValueError('unknown layout')


def plan_tasks(n, source_groups, target_groups, width, st, dt, ps=2, pd=3):
    tasks = []
    for l in range(4):
        for kind in range(2):
            for rank, heads in enumerate(target_groups):
                buffer = (0, rank, l, kind)
                # A frozen destination descriptor includes endpoint incarnation,
                # owner, replica slot, allocation identity/generation and mapping.
                destination = ('target', 11, 0, rank, ('replica', rank),
                               buffer, 7, tuple(dt), 'NHD', tuple(heads), width)
                for dh, h in enumerate(heads):
                    providers = [(r, hs.index(h)) for r, hs in enumerate(source_groups) if h in hs]
                    if not providers:
                        raise ValueError('target head has no source')
                    sr, sh = providers[0]
                    a = 0
                    while a < n:
                        b = min(n, (a // ps + 1) * ps, (a // pd + 1) * pd)
                        channels = ((0, 1),) if width == 1 else ((0, 1), (1, width))
                        for ca, cb in channels:
                            content = ('eta9', 'stable-snapshot', 'KV', l, kind,
                                       a, b, h, ca, cb, 'fp16-contract')
                            tasks.append({'content': content, 'destination': destination,
                                          'buffer': buffer, 'l': l, 'kind': kind,
                                          'a': a, 'b': b, 'h': h, 'ca': ca, 'cb': cb,
                                          'dh': dh, 'sr': sr, 'sh': sh,
                                          'source_binding': (l // 2, sr, l, kind,
                                                             tuple(st), 'HND', 4)})
                        a = b
    return tasks


def task_key(task, fault=''):
    content = task['content']
    if fault == 'omit_channels':
        content = content[:8] + content[10:]
    if fault == 'omit_target':
        return ('tx9', 'stage0', content)
    return ('tx9', 'stage0', content, task['destination'])


class TaskLedger:
    def __init__(self, fault=''):
        self.fault = fault
        self.entries = {}
        self.pending = set()
        self.visible_atoms = set()
        self.writes = 0
        self.duplicates = 0
        self.sealed = False

    def submit(self, task):
        key = task_key(task, self.fault)
        if key in self.entries:
            # Faulty controls deliberately reproduce the old collision behavior.
            if self.fault not in ('omit_target', 'omit_channels'):
                need(self.entries[key]['task'] == task, 'same key changed frozen descriptor')
            self.duplicates += 1
            return key
        need(not self.sealed, 'new write after seal')
        self.entries[key] = {'task': dict(task), 'state': 'ACCEPTED'}
        self.pending.add(key)
        return key

    def complete(self, key, source, target, source_groups, target_groups, st, dt, width):
        need(key in self.pending, 'unknown or repeated completion')
        task = self.entries[key]['task']
        sr, sh = task['sr'], task['sh']
        rank = task['destination'][3]
        for t in range(task['a'], task['b']):
            for c in range(task['ca'], task['cb']):
                si = address(st, 2, len(source_groups[sr]), t, sh, c, width,
                             'NHD' if self.fault == 'wrong_layout' else 'HND')
                di = address(dt, 3, len(target_groups[rank]), t, task['dh'], c, width, 'NHD')
                atom = (task['buffer'], t, task['h'], c)
                need(atom not in self.visible_atoms, 'overlapping target write')
                target[task['buffer']][di] = source[(task['l'] // 2, sr, task['l'], task['kind'])][si]
                self.visible_atoms.add(atom)
                self.writes += 1
        self.pending.remove(key)
        self.entries[key]['state'] = 'VISIBLE'

    def ready(self, required_atoms):
        need(not self.pending, 'writers remain')
        need(all(e['state'] == 'VISIBLE' for e in self.entries.values()), 'task incomplete')
        # The required set is built from the original target geometry, not keys.
        need(self.visible_atoms == required_atoms, 'original target demand incomplete')
        self.sealed = True
        return True

    def faulty_ready_from_own_table(self):
        return not self.pending and all(e['state'] == 'VISIBLE' for e in self.entries.values())


def transfer_case(n, source_groups, target_groups, width=2, fault=''):
    need(n >= 0 and width > 0, 'invalid geometry')
    ps, pd = 2, 3
    ns, nd = (n + ps - 1) // ps, (n + pd - 1) // pd
    st = [2, 0, 3] if ns == 3 else list(reversed(range(ns)))
    dt = [(i + 1) % nd for i in range(nd)]
    source, target, expected = {}, {}, {}
    # References are built before task planning, directly from original geometry.
    required_atoms = set()
    for l in range(4):
        for kind in range(2):
            for rank, heads in enumerate(source_groups):
                source[(l // 2, rank, l, kind)] = physical_reference(l, kind, heads, n, ps, st, 'HND', width)
            for rank, heads in enumerate(target_groups):
                buffer = (0, rank, l, kind)
                expected[buffer] = physical_reference(l, kind, heads, n, pd, dt, 'NHD', width)
                target[buffer] = [-1] * len(expected[buffer])
                for h in heads:
                    for t in range(n):
                        for c in range(width):
                            required_atoms.add((buffer, t, h, c))
    tasks = plan_tasks(n, source_groups, target_groups, width, st, dt)
    ledger = TaskLedger(fault)
    for task in tasks:
        ledger.submit(task)
        ledger.submit(dict(task))
    for key in list(ledger.entries):
        ledger.complete(key, source, target, source_groups, target_groups, st, dt, width)
    for task in reversed(tasks):
        ledger.submit(dict(task))
    return {'ledger': ledger, 'tasks': tasks, 'target': target,
            'expected': expected, 'required': required_atoms}


def audit_contents(case):
    need(case['target'] == case['expected'], 'independent physical-slot oracle: missing or wrong target data')


def validate_transfer(case):
    case['ledger'].ready(case['required'])
    audit_contents(case)
    need(case['ledger'].writes == len(case['required']), 'physical write count differs from demand')


def test_layout_and_identity():
    source = ((0, 1), (2, 3))
    target = ((0,), (1,), (2,), (3,))
    ordinary_writes = 0
    for n in (0, 1, 5, 7):
        case = transfer_case(n, source, target)
        validate_transfer(case)
        ordinary_writes += case['ledger'].writes
        need(case['ledger'].duplicates == 2 * len(case['tasks']), 'control replay not exercised')
    # Empty local owner, empty manifest, one channel, equivalent source replicas.
    for n, sg, tg, width in [
            (0, ((0,),), ((), (0,)), 1),
            (1, ((0,),), ((), (0,)), 1),
            (5, ((0,), (0,)), ((0,), (0,)), 1)]:
        validate_transfer(transfer_case(n, sg, tg, width))
    replicated = transfer_case(5, ((0, 1),), ((0,), (1,), (0,), (1,)), width=3)
    validate_transfer(replicated)
    need(replicated['ledger'].writes == 480, 'target head fanout omitted')
    need(len(replicated['ledger'].entries) == 256, 'page/channel segment task count')
    sample = replicated['tasks'][0]
    changed_source = dict(sample)
    changed_source['source_binding'] = ('different-source-snapshot',)
    rejected(lambda: replicated['ledger'].submit(changed_source))
    # Target incarnation, owner, replica and allocation changes must distinguish W.
    for index, replacement in [(1, 12), (3, 2), (4, ('replica', 99)),
                               (5, ('allocation', 99)), (6, 8)]:
        changed = dict(sample)
        dest = list(sample['destination'])
        dest[index] = replacement
        changed['destination'] = tuple(dest)
        need(task_key(changed) != task_key(sample), 'destination identity collapsed')
        rejected(lambda changed=changed: replicated['ledger'].submit(changed))
    new_tx_task = dict(sample)
    dest = list(sample['destination'])
    dest[6] = 8
    new_tx_task['destination'] = tuple(dest)
    need(task_key(new_tx_task) != task_key(sample), 'allocation generation lost')

    negatives = {}
    for fault in ('omit_target', 'omit_channels'):
        bad = transfer_case(5, ((0, 1),), ((0,), (1,), (0,), (1,)), width=3, fault=fault)
        ledger = bad['ledger']
        need(ledger.faulty_ready_from_own_table(), 'fault did not falsely declare task-table completion')
        need(ledger.writes < len(bad['required']), 'fault did not cause real missing writes')
        rejected(lambda bad=bad: bad['ledger'].ready(bad['required']))
        rejected(lambda bad=bad: audit_contents(bad))
        negatives[fault] = {'required_element_writes': len(bad['required']),
                            'actual_element_writes': ledger.writes,
                            'task_table_falsely_complete': True,
                            'demand_and_content_oracles_rejected': True}
    need(negatives['omit_target']['actual_element_writes'] == 240, 'replica-key counterexample changed')
    need(negatives['omit_channels']['actual_element_writes'] == 160, 'channel-key counterexample changed')
    for n in (5, 7):
        bad = transfer_case(n, source, target, fault='wrong_layout')
        need(bad['ledger'].ready(bad['required']), 'layout fault should cover all positions')
        rejected(lambda bad=bad: audit_contents(bad))
    rejected(lambda: transfer_case(1, ((0, 1),), ((2,),)))
    need(address([2, 0, 3], 2, 2, 4, 1, 1, 2, 'HND') == 29, 'source example offset')
    need(address([1, 0], 3, 1, 4, 0, 1, 2, 'NHD') == 3, 'target example offset')
    logical = 4 * 2 * 5 * 4 * 2 * 2
    source_capacity = 4 * 2 * 2 * 4 * 2 * 2 * 2 * 2
    target_capacity = 4 * 2 * 4 * 2 * 3 * 1 * 2 * 2
    need((logical, source_capacity, target_capacity) == (640, 1024, 768), 'content versus allocation bytes')
    return ordinary_writes, replicated['ledger'].writes, negatives


class Move:
    # Single-request final snapshot; operations and proofs are abstract CPU events.
    def __init__(self, required):
        self.required = frozenset(required)
        self.ops = {}
        self.pending = set()
        self.done = set()
        self.visible = set()
        self.frozen = False
        self.ready = False
        self.decision = None
        self.inc = 11
        self.current_inc = 11
        self.reserved = True
        self.published = False
        self.refs = 0
        self.log = []

    def submit(self, op, unit):
        if op in self.ops:
            need(self.ops[op] == unit, 'operation identity changed')
            return
        need(self.decision is None and not self.ready and self.reserved, 'write after seal or abort')
        need(unit in self.required, 'range outside manifest')
        need(unit not in self.ops.values(), 'overlapping reissue needs isolation')
        self.ops[op] = unit
        self.pending.add(op)
        self.log.append(('submit', op, unit))

    def complete(self, op, success=True):
        need(op in self.pending, 'unknown or duplicate completion')
        self.pending.remove(op)
        if success:
            self.done.add(self.ops[op])
        self.log.append(('complete', op, success))

    def make_visible(self, unit):
        need(self.decision is None and unit in self.done, 'visibility lacks successful completion')
        need(not any(self.ops[op] == unit for op in self.pending), 'writer still active')
        self.visible.add(unit)
        self.log.append(('visible', unit))

    def freeze(self):
        need(self.decision is None and not self.frozen, 'invalid freeze')
        self.frozen = True
        self.log.append(('freeze',))

    def prepare(self, faulty=False):
        required = {u for u in self.required if u.endswith(':data')} if faulty else self.required
        need(self.decision is None and self.frozen and not self.pending and self.reserved, 'unsealed attempt')
        need(required <= self.visible, 'manifest incomplete')
        self.ready = True
        self.log.append(('ready',))

    def decide(self, decision):
        need(decision in ('COMMIT', 'ABORT'), 'invalid decision')
        if self.decision is not None:
            need(self.decision == decision, 'authority decision reversal')
            return
        if decision == 'COMMIT':
            need(self.frozen and self.ready and self.reserved, 'commit before readiness')
            need(self.current_inc == self.inc, 'prepared endpoint restarted')
        self.decision = decision
        self.log.append(('decision', decision))

    def publish(self, incarnation=11):
        need(self.decision == 'COMMIT', 'no commit authorization')
        need(incarnation == self.inc == self.current_inc and self.reserved, 'stale incarnation')
        if not self.published:
            self.published = True
            self.refs += 1
        self.log.append(('publish', incarnation, self.current_inc))

    def consume(self):
        need(self.published and self.decision == 'COMMIT' and self.current_inc == self.inc, 'unready consumer')
        self.log.append(('consume',))

    def release_source(self):
        need(self.decision == 'COMMIT' and self.published and not self.pending, 'recovery or reader remains')
        self.log.append(('release_source',))

    def free_target(self, faulty=False):
        need(self.decision == 'ABORT' and self.refs == 0, 'target still owned')
        if not faulty:
            need(not self.pending, 'physical access still in flight')
        self.reserved = False
        self.log.append(('free',))

    def resume_source(self):
        need(self.decision == 'ABORT' and not self.pending, 'unknown decision or undrained source')
        self.log.append(('resume',))


def audit(log, required, incarnation=11):
    # Independent trace oracle: does not read Move state or call its methods.
    accepted, pending, done, visible = {}, set(), set(), set()
    frozen = ready = published = freed = False
    decision = None
    for event in log:
        tag = event[0]
        if tag == 'submit':
            op, unit = event[1:]
            need(decision is None and not ready and not freed, 'oracle: late submission')
            need(op not in accepted and unit in required, 'oracle: operation binding')
            need(unit not in accepted.values(), 'oracle: overlapping write')
            accepted[op] = unit
            pending.add(op)
        elif tag == 'complete':
            op, success = event[1:]
            need(op in pending and not freed, 'oracle: invalid completion')
            pending.remove(op)
            if success:
                done.add(accepted[op])
        elif tag == 'visible':
            unit = event[1]
            need(unit in done and not any(accepted[op] == unit for op in pending), 'oracle: premature visibility')
            visible.add(unit)
        elif tag == 'freeze':
            need(not frozen and decision is None, 'oracle: invalid freeze')
            frozen = True
        elif tag == 'ready':
            need(frozen and not pending and set(required) <= visible and not freed, 'oracle: partial manifest')
            ready = True
        elif tag == 'decision':
            new = event[1]
            need(new in ('COMMIT', 'ABORT') and decision in (None, new), 'oracle: split authority')
            if new == 'COMMIT':
                need(frozen and ready, 'oracle: commit before proofs')
            decision = new
        elif tag == 'publish':
            need(decision == 'COMMIT' and not freed and event[1] == incarnation == event[2], 'oracle: unauthorized publication')
            published = True
        elif tag == 'consume':
            need(published and decision == 'COMMIT' and not freed, 'oracle: premature consumption')
        elif tag == 'release_source':
            need(published and decision == 'COMMIT' and not pending, 'oracle: source responsibility remains')
        elif tag == 'free':
            need(decision == 'ABORT' and not published and not pending and not freed, 'oracle: reuse before drain')
            freed = True
        elif tag == 'resume':
            need(decision == 'ABORT' and not pending, 'oracle: source resumed without authorization')
        else:
            raise AssertionError('unknown trace event')


def ready_move(units):
    move = Move(units)
    move.freeze()
    for i, unit in enumerate(units):
        move.submit(i, unit)
        move.complete(i)
        move.make_visible(unit)
    move.prepare()
    return move


def test_protocol():
    units = ('L0:data', 'L0:scale', 'L1:data', 'L1:scale')
    count = 0
    for order in itertools.permutations(range(len(units))):
        move = Move(units)
        for i, unit in enumerate(units):
            move.submit(i, unit)
            move.submit(i, unit)
        need(len(move.pending) == len(units), 'duplicate submission')
        move.freeze()
        for i in order:
            move.complete(i)
            move.make_visible(units[i])
        move.submit(0, units[0])
        move.prepare()
        rejected(lambda: move.submit(99, units[0]))
        move.decide('COMMIT')
        move.decide('COMMIT')
        rejected(lambda: move.decide('ABORT'))
        rejected(move.release_source)
        move.publish()
        move.publish()
        need(move.refs == 1, 'duplicate installation changed references')
        move.consume()
        move.release_source()
        audit(move.log, units)
        count += 1
    empty = ready_move(())
    empty.decide('COMMIT')
    empty.publish()
    empty.consume()
    audit(empty.log, ())
    count += 1

    partial = Move(units)
    partial.submit(0, units[0])
    partial.submit(1, units[1])
    partial.freeze()
    rejected(partial.prepare)
    rejected(partial.consume)
    partial.decide('ABORT')
    rejected(partial.free_target)
    rejected(partial.resume_source)
    partial.complete(0, False)
    rejected(partial.free_target)
    partial.complete(1, True)
    partial.free_target()
    partial.resume_source()
    audit(partial.log, units)
    rejected(lambda: partial.submit(99, units[2]))

    unknown = Move(units)
    unknown.freeze()
    rejected(unknown.resume_source)
    rejected(lambda: unknown.decide('COMMIT'))

    failed = Move(units)
    failed.freeze()
    for i, unit in enumerate(units):
        failed.submit(i, unit)
        failed.complete(i, i != 3)
        if i != 3:
            failed.make_visible(unit)
    rejected(failed.prepare)

    restarted = ready_move(units)
    restarted.decide('COMMIT')
    restarted.current_inc = 12
    rejected(restarted.publish)
    rejected(lambda: restarted.publish(12))
    rejected(lambda: restarted.decide('ABORT'))
    need(restarted.decision == 'COMMIT', 'restart erased committed decision')

    missing_scale = Move(units)
    missing_scale.freeze()
    for i, unit in enumerate(units):
        if unit.endswith(':data'):
            missing_scale.submit(i, unit)
            missing_scale.complete(i)
            missing_scale.make_visible(unit)
    missing_scale.prepare(faulty=True)
    missing_scale.decide('COMMIT')
    missing_scale.publish()
    missing_scale.consume()
    rejected(lambda: audit(missing_scale.log, units))

    early_free = Move(units)
    early_free.submit(0, units[0])
    early_free.decide('ABORT')
    early_free.free_target(faulty=True)
    rejected(lambda: audit(early_free.log, units))
    return count


def test_cost():
    d, bandwidth, growth = Fraction(1000), Fraction(100), Fraction(10)
    total = elapsed = Fraction(0)
    tails = []
    for i in range(3):
        total += d
        elapsed += d / bandwidth
        d = growth * d / bandwidth
        tails.append(d)
    total += d
    elapsed += d / bandwidth
    need(tails == [100, 10, 1], 'precopy tails')
    need(total == 1111 and elapsed == Fraction(1111, 100), 'whole migration versus pause')
    need(d / bandwidth == Fraction(1, 100), 'final copy pause')
    h = Fraction(1, 10)
    fixed = growth * h / (1 - growth / bandwidth)
    need(fixed == Fraction(10, 9), 'fixed overhead residual')
    need(growth * (h + fixed / bandwidth) == fixed, 'fixed point')
    for g in (100, 120):
        need(Fraction(g) * 1000 / bandwidth >= 1000, 'nonconvergent regime')
    payload = 32 * 4096 * 8 * (128 + 128) * 2
    suffix = 32 * 1024 * 8 * (128 + 128) * 2
    mib = 2 ** 20
    need(payload == 512 * mib and suffix == 128 * mib, 'KV bytes')
    need(Fraction(suffix, 20 * 2 ** 30) == Fraction(1, 160), '6.25 ms lower bound')
    growth_reserve_mib = Fraction(32 * 16 * 8 * (128 + 128) * 2, mib)
    need(512 + 2 * 8 + growth_reserve_mib == 530, 'capacity example')
    unique_elements = sum(1 for l in range(4) for k in range(2)
                          for t in range(5) for h in (0, 1) for c in range(3))
    physical_elements = sum(1 for l in range(4) for k in range(2)
                            for t in range(5) for h in (0, 1, 0, 1) for c in range(3))
    need((unique_elements * 2, physical_elements * 2) == (480, 960), 'physical replica bytes')
    return {'total_MiB': int(total), 'total_seconds': str(elapsed),
            'tail_seconds': str(d / bandwidth), 'fixed_residual_MiB': str(fixed),
            'unique_replica_example_bytes': unique_elements * 2,
            'target_replica_example_bytes': physical_elements * 2}


def main():
    interval_cases = test_ranges()
    ordinary, replicated, key_negatives = test_layout_and_identity()
    orders = test_protocol()
    cost = test_cost()
    print(json.dumps({
        'scope': 'deterministic standalone CPU models; no GPU, NIC, network, file I/O, upstream implementation or performance measurements',
        'interval_cases': interval_cases,
        'ordinary_layout_element_writes': ordinary,
        'replicated_layout_element_writes': replicated,
        'replicated_planned_tasks': 256,
        'target_key_negative_controls': key_negatives,
        'valid_protocol_orders_including_empty_manifest': orders,
        'negative_controls': ['duplicate byte count hides a hole', 'HND read as NHD',
                              'target binding omitted before deduplication',
                              'channel range omitted before deduplication',
                              'scale omitted from readiness', 'aborted buffer freed before drain'],
        'cost_model': cost,
        'unverified': ['real device visibility', 'concurrent atomicity', 'transport cancellation',
                       'persistent authority recovery', 'actual quantization encoding',
                       'model numerical quality', 'hardware performance', 'agent knowledge-use benefit']
    }, ensure_ascii=True, sort_keys=True))


if __name__ == '__main__':
    main()
