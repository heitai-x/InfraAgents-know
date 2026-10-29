#!/usr/bin/env python3
'''K-HW-012: deterministic, offline, read-only CPU mechanism checks.
No project imports, file access, subprocesses, network, or devices.
'''

from fractions import Fraction
from heapq import heappush, heappop
from itertools import permutations


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def reject(call, exception, fragment):
    try:
        call()
    except exception as error:
        need(fragment in str(error), 'unexpected rejection: ' + str(error))
        return
    raise RuntimeError('negative control accepted: ' + fragment)


def merge_ranges(ranges):
    checked = []
    for item in ranges:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError('range requires exactly two endpoints')
        a, b = item
        if type(a) is not int or type(b) is not int or not 0 <= a <= b:
            raise ValueError('invalid nonnegative integer range')
        checked.append((a, b))
    out = []
    for a, b in sorted(checked):
        if a == b:
            continue
        if not out or a > out[-1][1]:
            out.append([a, b])
        else:
            out[-1][1] = max(out[-1][1], b)
    return [(a, b) for a, b in out]


def covers(required, completed):
    required = merge_ranges(required)
    completed = merge_ranges(completed)
    j = 0
    for a, b in required:
        while j < len(completed) and completed[j][1] <= a:
            j += 1
        if j == len(completed):
            return False
        if completed[j][0] > a or completed[j][1] < b:
            return False
    return True


def byte_reference(required, completed):
    wanted = {x for a, b in required for x in range(a, b)}
    arrived = {x for a, b in completed for x in range(a, b)}
    return wanted <= arrived


def wrong_consuming_covers(required, completed):
    done = merge_ranges(completed)
    j = 0
    for a, b in merge_ranges(required):
        while j < len(done) and done[j][1] <= a:
            j += 1
        if j == len(done) or done[j][0] > a or done[j][1] < b:
            return False
        j += 1
    return True


def project(events, key):
    return [(a, b) for identity, a, b in events if identity == key]


def settle_once(pending, terminal, operation):
    if operation in terminal or operation not in pending:
        return False
    pending.remove(operation)
    terminal.add(operation)
    return True


def capacity_reference(times, blocks, slots):
    '''Independent dependency recurrence, without events or a free list.
    FIFO stage edges plus final-use-to-next-lease edges for bundles.
    '''
    times = tuple(Fraction(t) for t in times)
    finished = []
    for i in range(blocks):
        row = []
        for k, duration in enumerate(times):
            predecessors = [Fraction(0)]
            if i:
                predecessors.append(finished[i - 1][k])
            if k:
                predecessors.append(row[k - 1])
            if k == 0 and i >= slots:
                predecessors.append(finished[i - slots][-1])
            row.append(max(predecessors) + duration)
        finished.append(row)
    return finished[-1][-1] if finished else Fraction(0)


def closed_flow(times, blocks):
    if blocks == 0 or not times:
        return Fraction(0)
    times = tuple(Fraction(t) for t in times)
    return sum(times, Fraction(0)) + (blocks - 1) * max(times)


def slot_collision(starts, lifetimes, slots):
    need(slots >= 1, 'positive slot count required')
    need(len(starts) == len(lifetimes), 'interval length mismatch')
    intervals = [(i % slots, start, start + lifetime)
                 for i, (start, lifetime) in enumerate(zip(starts, lifetimes))]
    return any(sa == sb and max(a, c) < min(b, d)
               for i, (sa, a, b) in enumerate(intervals)
               for sb, c, d in intervals[i + 1:])


def pump(blocks, slots, times, cancel_at=None, reject_at=(), partial_at=(),
         orphan=False, fault=None):
    '''Finite conservative-bundle model with fail-stop semantics per plan.
    Each read batch has two abstract byte cells. Accepted members share
    one stage service interval; later stages have one aggregate operation.
    Explicit synchronous rejection produces no completion event.
    partial_at accepts the first read member and rejects the second.
    orphan deliberately supplies an active state with no successor.
    fault selects an intentionally incorrect scheduling order.
    '''
    need(type(blocks) is int and blocks >= 0, 'invalid blocks')
    need(type(slots) is int and slots >= 1, 'invalid slots')
    times = tuple(Fraction(t) for t in times)
    need(bool(times) and all(t >= 0 for t in times), 'invalid stages')
    if cancel_at is not None:
        cancel_at = Fraction(cancel_at)
        need(cancel_at >= 0, 'invalid cancel time')
    reject_at = set(reject_at)
    partial_at = set(partial_at)
    need(all(k == 0 for _, k in partial_at), 'partial injection is read-only')
    need(fault in (None, 'late_cancel', 'wait_before_cleanup'), 'unknown fault')
    now = Fraction(0)
    next_job = 0
    lease = 0
    sequence = 0
    operation_id = 0
    cancelled = False
    stop_reason = None
    free = list(range(slots))
    active = {}
    pending = {}
    events = []
    trace = []
    completed = []
    accepted_read_cells = set()
    available = [Fraction(0) for _ in times]

    def emit(kind, job=-1, snapshot=None, **fields):
        state = snapshot if snapshot is not None else active.get(job)
        slot = state['slot'] if state is not None else -1
        generation = state['lease'] if state is not None else -1
        event = dict(kind=kind, time=now, job=job, slot=slot,
                     lease=generation, free=len(free))
        event.update(fields)
        trace.append(event)

    def enqueue(time, priority, kind, job=-1, stage=-1, operation=-1):
        nonlocal sequence
        heappush(events, (time, priority, sequence, kind, job, stage, operation))
        sequence += 1

    def close(job):
        state = active[job]
        if not state['closed']:
            state['closed'] = True
            state['next'] = None
            emit('close', job)

    def stop(reason):
        nonlocal stop_reason
        if stop_reason is None:
            stop_reason = reason
        for job in list(active):
            close(job)

    def admit():
        nonlocal next_job, lease
        job = next_job
        next_job += 1
        lease += 1
        active[job] = dict(slot=free.pop(0), lease=lease, next=0,
                           closed=False, operations=set())
        emit('reserve', job)
        return job

    def submit(job, stage):
        nonlocal operation_id
        state = active[job]
        state['next'] = None
        if orphan and job == 0 and stage == 0:
            emit('orphan', job)
            return
        members = 2 if stage == 0 else 1
        reject_all = (job, stage) in reject_at
        reject_tail = (job, stage) in partial_at
        finish = max(now, available[stage]) + times[stage]
        if not reject_all:
            available[stage] = finish
        for member in range(members):
            operation_id += 1
            operation = operation_id
            state['operations'].add(operation)
            pending[operation] = dict(job=job, stage=stage, phase='intent')
            emit('intent', job, operation=operation, stage=stage)
            if reject_all or (reject_tail and member == members - 1):
                del pending[operation]
                state['operations'].remove(operation)
                emit('reject', job, operation=operation, stage=stage)
            else:
                pending[operation] = dict(job=job, stage=stage,
                                          phase='accepted', finish=finish)
                if stage == 0:
                    accepted_read_cells.add((job, member))
                emit('submit', job, operation=operation,
                     stage=stage, finish=finish)
                enqueue(finish, 1, 'done', job, stage, operation)
        if reject_all or reject_tail:
            stop('FAILED')

    def retire(job):
        state = active.pop(job)
        free.append(state['slot'])
        emit('retire', job, snapshot=state)

    def receive_next():
        nonlocal now
        accepted = {op for op, info in pending.items()
                    if info['phase'] == 'accepted'}
        need(bool(accepted), 'wait without accepted operation')
        wake_operations = [event[6] for event in events
                           if event[3] == 'done' and event[6] in accepted]
        need(bool(wake_operations), 'accepted access has no completion route')
        emit('wait', wake_operations=tuple(wake_operations))
        now = events[0][0]

    def result(status):
        emit('return', status=status)
        return dict(status=status, finish=now, admitted=next_job,
                    completed=list(completed), free=len(free),
                    pending=len(pending), cancelled=cancelled,
                    unsubmitted=[(job, cell, cell + 1)
                                 for job in range(blocks) for cell in range(2)
                                 if (job, cell) not in accepted_read_cells],
                    trace=trace)

    if cancel_at is not None:
        enqueue(cancel_at, 0, 'cancel')
    if fault == 'late_cancel' and blocks:
        # Deliberate old ordering: work outside the event heap goes first.
        first = admit()
        submit(first, 0)

    for _ in range(10000):
        # Drain every already-arrived event before any new authorization.
        # At equal times, the heap places cancellation before completions.
        while events and events[0][0] <= now:
            _, _, _, kind, job, stage, operation = heappop(events)
            if kind == 'cancel':
                cancelled = True
                emit('cancel')
                stop('CANCELLED')
                continue
            state = active[job]
            need(operation in pending, 'unknown completion in producer')
            del pending[operation]
            state['operations'].remove(operation)
            emit('done', job, operation=operation, stage=stage)
            if not state['closed'] and not state['operations']:
                if stage == len(times) - 1:
                    completed.append(job)
                    close(job)
                else:
                    state['next'] = stage + 1

        if (fault == 'wait_before_cleanup' and stop_reason is not None
                and not pending and active):
            # This invokes a real waiting precondition, without hanging.
            receive_next()

        for job, state in list(active.items()):
            if state['closed'] and not state['operations']:
                retire(job)

        # Terminal checks precede receiving another event. In particular,
        # a future scheduled cancellation must not delay a finished plan.
        if not active and (stop_reason is not None or next_job == blocks):
            status = 'CANCELLED' if cancelled else (stop_reason or 'SUCCESS')
            return result(status)

        ready = next(((job, state['next']) for job, state in active.items()
                      if not state['closed'] and state['next'] is not None),
                     None)
        if ready is not None:
            submit(*ready)
            continue
        if stop_reason is None and next_job < blocks and free:
            admit()
            continue

        # Includes waiting empty + active nonempty + no successor.
        if not pending:
            stop('STALLED')
            continue
        receive_next()
    raise RuntimeError('finite-model step guard reached')


def audit_responsibilities(trace, slots, cancel_at=None):
    '''Independent observer: reconstruct leases and operation obligations.
    Input cancellation time is authoritative, even if its trace row is late.
    Producer active/pending/free data structures are never consulted.
    '''
    if cancel_at is not None:
        cancel_at = Fraction(cancel_at)
    live = {}
    operations = {}
    known_operations = set()
    closed = set()
    cancelled = False
    returned = False
    previous_time = Fraction(0)
    for event in trace:
        need(not returned, 'event after return')
        time = event['time']
        need(time >= previous_time, 'trace time moved backwards')
        previous_time = time
        kind = event['kind']
        key = (event['job'], event['lease'])
        slot = event['slot']
        if kind in ('reserve', 'intent', 'submit'):
            need(cancel_at is None or time < cancel_at,
                 'work at/after arrived cancellation')
        if kind == 'cancel':
            cancelled = True
        elif kind == 'reserve':
            need(not cancelled, 'admission after cancel')
            need(slot not in live, 'slot reused before retirement')
            need(0 <= slot < slots, 'invalid physical slot')
            live[slot] = key
        elif kind == 'wait':
            accepted = {op for op, info in operations.items()
                        if info['phase'] == 'accepted'}
            need(bool(accepted), 'observer: wait without accepted operation')
            need(any(op in accepted for op in event['wake_operations']),
                 'wait has no actual wake responsibility')
        elif kind == 'return':
            need(not live and not operations, 'return before drain')
            returned = True
        else:
            need(live.get(slot) == key, 'event does not own current lease')
            if kind == 'intent':
                operation = event['operation']
                need(not cancelled and key not in closed,
                     'new intent after authorization closed')
                need(operation not in known_operations, 'operation identity reused')
                known_operations.add(operation)
                operations[operation] = dict(key=key, phase='intent')
            elif kind == 'submit':
                operation = event['operation']
                info = operations.get(operation)
                need(info is not None and info['key'] == key
                     and info['phase'] == 'intent', 'submit without intent')
                need(not cancelled and key not in closed,
                     'accepted work after authorization closed')
                need(event['finish'] >= time, 'negative operation duration')
                operations[operation] = dict(key=key, phase='accepted',
                                              finish=event['finish'])
            elif kind == 'reject':
                operation = event['operation']
                info = operations.get(operation)
                need(info is not None and info['key'] == key
                     and info['phase'] == 'intent',
                     'rejection withdrew an accepted operation')
                del operations[operation]
            elif kind == 'done':
                operation = event['operation']
                info = operations.get(operation)
                need(info is not None and info['key'] == key
                     and info['phase'] == 'accepted',
                     'unknown or duplicate operation completion')
                need(info['finish'] == time, 'completion time mismatch')
                del operations[operation]
            elif kind == 'close':
                need(key not in closed, 'authorization closed twice')
                closed.add(key)
            elif kind == 'retire':
                need(key in closed, 'retirement before authorization closed')
                need(not any(info['key'] == key for info in operations.values()),
                     'retirement with pending access')
                del live[slot]
            elif kind != 'orphan':
                raise RuntimeError('unknown trace action: ' + kind)
        need(event['free'] >= 0 and event['free'] + len(live) == slots,
             'capacity conservation failed')
    need(returned, 'trace has no terminal return')


def reachable(edges, source, target):
    seen = {source}
    todo = [source]
    while todo:
        current = todo.pop()
        for a, b in edges:
            if a == current and b not in seen:
                seen.add(b)
                todo.append(b)
    return target in seen


def all_orders_before(nodes, edges, source, target):
    count = 0
    holds = True
    for order in permutations(nodes):
        position = {node: i for i, node in enumerate(order)}
        if all(position[a] < position[b] for a, b in edges):
            count += 1
            if position[source] >= position[target]:
                holds = False
    need(count > 0, 'ordering graph has no legal execution')
    return holds, count


def main():
    print('K-HW-012 CPU model checks')
    print('scope: exact integers/Fraction and finite events; no file/device I/O')

    intervals = [(a, b) for a in range(4) for b in range(a, 4)]
    lists = [[]] + [[x] for x in intervals]
    lists += [[x, y] for x in intervals for y in intervals]
    range_cases = 0
    for required in lists:
        for completed_ranges in lists:
            need(covers(required, completed_ranges) ==
                 byte_reference(required, completed_ranges), 'byte oracle mismatch')
            range_cases += 1
    need(range_cases == 12321, 'range corpus changed')
    spanning_required, spanning_done = [(0, 2), (4, 6)], [(0, 6)]
    need(covers(spanning_required, spanning_done), 'spanning completion lost')
    need(not wrong_consuming_covers(spanning_required, spanning_done),
         'consuming-completion fault not detected')
    reject(lambda: covers([(0, 12)], [(2, 0, 12), (2, 0, 12)]),
           ValueError, 'exactly two endpoints')
    reject(lambda: covers([(3, 2)], []), ValueError, 'invalid nonnegative')
    print('ranges: 12321 byte-reference cases; shape and consuming-range faults detected')

    key = ('artifact-a', 'tensor-t', 'file-A', 2, 7)
    required = [(0, 4), (4, 8), (8, 12)]
    duplicates = [(key, 0, 12), (key, 0, 12)]
    need(covers(required, project(duplicates, key)), 'projected duplicates rejected')
    partial = [(key, 8, 12), (key, 0, 4)]
    need(not covers(required, project(partial, key)), 'short-read hole lost')
    need(covers(required, project(partial + [(key, 4, 8)], key)),
         'out-of-order suffix rejected')
    aliens = [('artifact-a', 'tensor-t', 'file-A', 1, 7),
              ('artifact-a', 'tensor-t', 'file-A', 2, 6),
              ('artifact-a', 'tensor-t', 'file-B', 2, 7),
              ('artifact-b', 'tensor-t', 'file-A', 2, 7),
              ('artifact-a', 'tensor-other', 'file-A', 2, 7)]
    for alien in aliens:
        events = partial + [(alien, 0, 12)]
        need(not covers(required, project(events, key)), 'identity leaked')
        need(covers(required, [(a, b) for _, a, b in events]),
             'identity-dropping fault not reproduced')
    old_op, new_op = ('slot', 7, 'old'), ('slot', 8, 'new')
    pending, terminal = {old_op, new_op}, set()
    need(settle_once(pending, terminal, old_op), 'first settlement rejected')
    need(not settle_once(pending, terminal, old_op), 'duplicate settled twice')
    need(not settle_once(pending, terminal, ('slot', 6, 'unknown')),
         'unknown operation settled')
    need(pending == {new_op}, 'old completion changed new responsibility')
    print('identity: file/artifact/tensor/attempt/lease isolated; settlement idempotent')

    source = [[0, 1, 2], [3, 4, 5]]
    pulled = [source[q1][q0] for q0 in range(3) for q1 in range(2)]
    scattered = sorted((c1 * 2 + c0, value)
                       for c0, row in enumerate(source)
                       for c1, value in enumerate(row))
    need(pulled == [value for _, value in scattered] == [0, 3, 1, 4, 2, 5],
         'transpose mapping mismatch')
    need(pulled != [value for row in source for value in row],
         'unconverted-layout fault escaped')
    print('layout: 2x3 transpose pull agrees with independent source scatter')

    stage_sets = [(8, 3, 2, 5), (0, 0), (1, 2, 0),
                  (Fraction(5, 2), Fraction(3, 2), 3)]
    flow_cases = 0
    for durations in stage_sets:
        for blocks in (0, 1, 4, 7):
            for slots in (1, 2, 3, max(blocks, 1)):
                result = pump(blocks, slots, durations)
                audit_responsibilities(result['trace'], slots)
                expected = capacity_reference(durations, blocks, slots)
                need(result['finish'] == expected, 'capacity DAG/event mismatch')
                need(result['status'] == 'SUCCESS', 'normal flow failed')
                need(result['completed'] == list(range(blocks)), 'final block lost')
                need(result['free'] == slots and result['pending'] == 0,
                     'normal flow did not drain')
                if slots >= blocks:
                    need(expected == closed_flow(durations, blocks),
                         'ample-buffer closed-form mismatch')
                flow_cases += 1
    need(flow_cases == 64, 'flow corpus changed')
    need(closed_flow([], 4) == 0, 'empty stage formula')
    need(closed_flow((8, 3, 2, 5), 4) == 42, 'four-block result')
    need(4 * sum((8, 3, 2, 5)) != 42, 'serial fault escaped')
    need(4 * max((8, 3, 2, 5)) != 42, 'startup omission escaped')
    observed = []
    for slots, expected in ((1, 126), (2, 72), (3, 66)):
        result = pump(7, slots, (8, 3, 2, 5))
        audit_responsibilities(result['trace'], slots)
        need(result['finish'] == expected, 'finite-bundle table mismatch')
        observed.append(int(result['finish']))
    print('flow: 64 event/capacity-DAG cases including empty and zero duration')
    print('bounded bundles: seven blocks, slots=1/2/3 finish=%s ms' % observed)

    slot_cases = 0
    for lifetime in range(1, 10):
        for period in range(1, 7):
            slots = (lifetime + period - 1) // period
            count = 3 * slots + 2
            starts = [i * period for i in range(count)]
            lifetimes = [lifetime] * count
            need(not slot_collision(starts, lifetimes, slots), 'ceiling slots unsafe')
            if slots > 1:
                need(slot_collision(starts, lifetimes, slots - 1),
                     'insufficient-slot fault not detected')
            slot_cases += 1
    need(not slot_collision([0, 4], [4, 4], 1), 'half-open adjacency rejected')
    need(not slot_collision([], [], 1), 'empty slots rejected')
    print('slots: %d lifetime/period cases; insufficient slots detected' % slot_cases)

    # Concrete regressions: no completion events exist for these rejected reads.
    for blocks in (1, 2):
        result = pump(blocks, 1, (8, 3, 2, 5), reject_at=((0, 0),))
        audit_responsibilities(result['trace'], 1)
        need(result['status'] == 'FAILED' and result['finish'] == 0,
             'synchronous rejection did not return immediately')
        need(result['admitted'] == 1 and result['free'] == 1
             and result['pending'] == 0, 'rejection leaked capacity')
        need(not any(e['kind'] in ('submit', 'done', 'wait')
                     for e in result['trace']), 'rejection invented external work')
        expected = [(job, cell, cell + 1)
                    for job in range(blocks) for cell in range(2)]
        need(result['unsubmitted'] == expected, 'unaccepted ranges omitted')
    future_cancel = pump(2, 1, (8, 3, 2, 5), cancel_at=100,
                         reject_at=((0, 0),))
    audit_responsibilities(future_cancel['trace'], 1, 100)
    need(future_cancel['finish'] == 0 and future_cancel['status'] == 'FAILED',
         'future control event delayed failed terminal plan')
    reject(lambda: pump(2, 1, (8, 3, 2, 5), reject_at=((0, 0),),
                        fault='wait_before_cleanup'),
           RuntimeError, 'wait without accepted operation')
    print('sync rejection: N=1/2, one bundle, no external completion; return=0 ms')
    print('wait-before-cleanup fault detected; future cancellation does not delay return')

    # Rejection after an earlier stage and partial acceptance preserve actual access.
    for stage in range(4):
        result = pump(3, 2, (8, 3, 2, 5), reject_at=((0, stage),))
        audit_responsibilities(result['trace'], 2)
        need(result['status'] == 'FAILED' and result['free'] == 2
             and result['pending'] == 0, 'stage rejection did not drain')
    partially_accepted = pump(2, 1, (8, 3, 2, 5), partial_at=((0, 0),))
    audit_responsibilities(partially_accepted['trace'], 1)
    need(partially_accepted['status'] == 'FAILED'
         and partially_accepted['finish'] == 8
         and partially_accepted['free'] == 1, 'partial acceptance drain wrong')
    need(partially_accepted['unsubmitted'] == [(0, 1, 2), (1, 0, 1), (1, 1, 2)],
         'partial acceptance unsubmitted ranges wrong')
    need(all(e['stage'] == 0 for e in partially_accepted['trace']
             if e['kind'] == 'submit'), 'failed read authorized a successor')
    orphaned = pump(1, 1, (8,), orphan=True)
    audit_responsibilities(orphaned['trace'], 1)
    need(orphaned['status'] == 'STALLED' and orphaned['admitted'] == 1
         and orphaned['finish'] == 0 and orphaned['free'] == 1,
         'waiting-empty active-without-successor not handled')
    print('failure: stage rejection drains; partial read drains at 8 ms; orphan returns STALLED')

    for cancel_time in (0, 8, 9, 11, 18, 100):
        result = pump(5, 2, (8, 3, 2, 5), cancel_at=cancel_time)
        audit_responsibilities(result['trace'], 2, cancel_time)
        if cancel_time == 0:
            need(result['status'] == 'CANCELLED' and result['finish'] == 0,
                 'initial cancellation did not return')
            need(result['admitted'] == 0 and result['pending'] == 0
                 and result['free'] == 2, 'initial cancellation admitted work')
            need(not any(e['kind'] in ('reserve', 'intent', 'submit', 'wait')
                         for e in result['trace']), 'initial cancellation submitted')
            need(len(result['unsubmitted']) == 10, 'initial missing ranges lost')
        elif cancel_time == 8:
            need(result['finish'] == 16, 'same-time cancellation drain wrong')
            need(all(e['stage'] == 0 for e in result['trace']
                     if e['kind'] == 'submit'), 'read completion beat cancellation')
        elif cancel_time == 9:
            need(result['finish'] == 16 and result['admitted'] == 2
                 and not result['completed'], '9 ms cancellation trace wrong')
            need(result['unsubmitted'] == [(job, cell, cell + 1)
                                           for job in range(2, 5)
                                           for cell in range(2)],
                 'cancel unsubmitted range report wrong')
        elif cancel_time == 100:
            need(result['status'] == 'SUCCESS' and result['finish'] == 54,
                 'completed plan waited for future cancellation')
    faulty_cancel = pump(5, 2, (8, 3, 2, 5), cancel_at=0, fault='late_cancel')
    reject(lambda: audit_responsibilities(faulty_cancel['trace'], 2, 0),
           RuntimeError, 'work at/after arrived cancellation')
    cancelled = pump(5, 2, (8, 3, 2, 5), cancel_at=9)
    faulty = [dict(e) for e in cancelled['trace']]
    early = dict(next(e for e in faulty if e['kind'] == 'reserve'))
    early.update(kind='retire', time=Fraction(9), free=1)
    position = next(i for i, e in enumerate(faulty) if e['time'] > 9)
    faulty.insert(position, early)
    reject(lambda: audit_responsibilities(faulty, 2, 9),
           RuntimeError, 'retirement with pending access')
    print('cancel: at 0 ms zero admission/submission; at 9 ms drain=16 ms')
    print('same-time controls checked; late-cancel and premature-retirement faults detected')

    nodes = ('pin', 'dma_done', 'cpu_observe', 'cuda_submit',
             'kernel_start', 'unpin')
    good = [('pin', 'dma_done'), ('dma_done', 'cpu_observe'),
            ('cpu_observe', 'cuda_submit'), ('cuda_submit', 'kernel_start'),
            ('dma_done', 'unpin')]
    missing_submit = [edge for edge in good
                      if edge != ('cpu_observe', 'cuda_submit')]
    missing_unpin = [edge for edge in good if edge != ('dma_done', 'unpin')]
    counts = []
    for edges, kernel_ok, unpin_ok in ((good, True, True),
                                       (missing_submit, False, True),
                                       (missing_unpin, True, False)):
        reference_kernel, count = all_orders_before(nodes, edges,
                                                    'dma_done', 'kernel_start')
        reference_unpin, _ = all_orders_before(nodes, edges, 'dma_done', 'unpin')
        need(reference_kernel == reachable(edges, 'dma_done', 'kernel_start')
             == kernel_ok, 'kernel-order oracle mismatch')
        need(reference_unpin == reachable(edges, 'dma_done', 'unpin')
             == unpin_ok, 'unpin-order oracle mismatch')
        counts.append(count)
    need(counts == [4, 30, 6], 'ordering corpus changed')
    need(reachable(missing_submit, 'dma_done', 'cpu_observe') and
         not reachable(missing_submit, 'dma_done', 'kernel_start'),
         'host observation substituted for dependent CUDA submission')
    print('GPUDirect abstract ordering: legal total orders=4/30/6; missing edges detected')

    target = 4 * 1024 ** 3
    residence = Fraction(1, 1000)
    block = 1024 ** 2
    average = Fraction(target) * residence / block
    slots = (average.numerator + average.denominator - 1) // average.denominator
    need(average == Fraction(512, 125) and slots == 5, 'inflight calculation')
    need(Fraction(4 * block) / residence < target <= Fraction(5 * block) / residence,
         'integer capacity bracket failed')
    need(Fraction(0) * residence / block == 0, 'zero target edge')
    print('inflight: 4.096 average; four slots below target, five above, at assumed R')
    print('PASS: scoped CPU models only; no hardware or filesystem validation')


if __name__ == '__main__':
    main()
