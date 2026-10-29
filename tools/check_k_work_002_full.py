from collections import Counter, deque
from fractions import Fraction as F
import heapq


TERMINAL = {'COMPLETE', 'FAIL', 'REJECT', 'CANCEL', 'TIMEOUT'}
FIELDS = ('p', 'r', 's', 'g', 'a', 'b', 'd', 'x')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def spec(identifier, planned, **options):
    result = {'id': identifier, 'p': planned, 'tx': 1,
              'delay': F(1, 2), 'service': 1}
    result.update(options)
    for key in ('p', 'tx', 'delay', 'service', 'cancel', 'timeout'):
        if key in result:
            result[key] = F(str(result[key]))
    return result


def simulate(specs, horizon, stall_after_tx=False):
    """Finite, single-parent DAG; one transmit slot and one nonpreemptive server."""
    by_id = {s['id']: s for s in specs}
    require(len(by_id) == len(specs), 'duplicate input identity')
    for s in specs:
        require(all(s[k] >= 0 for k in ('tx', 'delay', 'service')),
                'negative model duration')
        for key in ('cancel', 'timeout'):
            require(key not in s or s[key] >= s['p'], 'termination before plan')
        seen = set()
        current = s['id']
        while current is not None:
            require(current in by_id, 'missing dependency')
            require(current not in seen, 'dependency cycle')
            seen.add(current)
            current = by_id[current].get('dep')
    ranks = {'FINISH': 0, 'TX_DONE': 1, 'PLAN': 2,
             'CANCEL': 3, 'TIMEOUT': 3, 'DECIDE': 4}
    events, log, rows = [], [], {}
    ready, queued = deque(), deque()
    serial = 0
    transmitter = None
    runner = None

    def schedule(t, kind, identifier):
        nonlocal serial
        heapq.heappush(events, (t, ranks[kind], serial, kind, identifier))
        serial += 1

    def emit(t, kind, identifier):
        log.append((t, kind, identifier))

    def terminate(t, kind, identifier):
        row = rows[identifier]
        if row['state'] in TERMINAL:
            return
        row['state'] = kind
        emit(t, kind, identifier)

    def wake(t):
        changed = True
        while changed:
            changed = False
            for s in specs:
                identifier = s['id']
                row = rows.get(identifier)
                if row is None or row['state'] != 'PLANNED':
                    continue
                parent = s.get('dep')
                if parent is not None:
                    upstream = rows.get(parent)
                    if upstream is None or upstream['state'] not in TERMINAL:
                        continue
                    if upstream['state'] != 'COMPLETE':
                        terminate(t, 'CANCEL', identifier)
                        changed = True
                        continue
                row['state'] = 'READY'
                ready.append(identifier)
                emit(t, 'READY', identifier)
                changed = True

    def pump(t):
        nonlocal transmitter, runner
        while transmitter is None and ready:
            identifier = ready.popleft()
            row = rows[identifier]
            if row['state'] != 'READY':
                continue
            transmitter = identifier
            row['state'] = 'SENT'
            emit(t, 'SEND', identifier)
            schedule(t + by_id[identifier]['tx'], 'TX_DONE', identifier)
            schedule(t + by_id[identifier]['delay'], 'DECIDE', identifier)
        while runner is None and queued:
            identifier = queued.popleft()
            row = rows[identifier]
            if row['state'] != 'QUEUED':
                continue
            runner = identifier
            row['state'] = 'RUNNING'
            emit(t, 'START', identifier)
            done = t + by_id[identifier]['service']
            schedule(done, 'FINISH', identifier)
            if by_id[identifier].get('duplicate'):
                schedule(done + F(1, 4), 'FINISH', identifier)

    for s in specs:
        schedule(s['p'], 'PLAN', s['id'])
        for key, kind in (('cancel', 'CANCEL'), ('timeout', 'TIMEOUT')):
            if key in s:
                schedule(s[key], kind, s['id'])
    while events:
        t, _, _, kind, identifier = heapq.heappop(events)
        if t > horizon:
            break
        if kind == 'PLAN':
            rows[identifier] = {'state': 'PLANNED', 'finished': False}
            emit(t, kind, identifier)
        elif kind == 'TX_DONE':
            require(transmitter == identifier, 'model transmit ownership')
            transmitter = None
            emit(t, kind, identifier)
        elif kind == 'DECIDE':
            row = rows[identifier]
            if by_id[identifier].get('reject'):
                emit(t, 'REJECT_DECISION', identifier)
                terminate(t, 'REJECT', identifier)
            else:
                emit(t, 'ADMIT', identifier)
                if row['state'] not in TERMINAL:
                    row['state'] = 'QUEUED'
                    queued.append(identifier)
        elif kind == 'FINISH':
            row = rows[identifier]
            emit(t, kind, identifier)
            if not row['finished']:
                require(runner == identifier, 'model execution ownership')
                row['finished'] = True
                runner = None
                if row['state'] not in TERMINAL:
                    result = 'FAIL' if by_id[identifier].get('fail') else 'COMPLETE'
                    terminate(t, result, identifier)
        else:
            terminate(t, kind, identifier)
        wake(t)
        if not (stall_after_tx and kind == 'TX_DONE'):
            pump(t)
    return log


def audit(log, lo, hi):
    """Replay emitted facts; integrate occupancy before each state change."""
    require(hi > lo, 'empty measurement duration')
    rows, work, tx, finished = {}, set(), set(), set()
    areas = {k: F(0) for k in ('logical', 'physical', 'draining', 'tx', 'resident')}
    previous = lo
    last_event = None
    duplicates = 0
    peak_resident = 0

    def integrate(t):
        nonlocal previous, peak_resident
        dt = max(F(0), min(t, hi) - max(previous, lo))
        values = {
            'logical': sum(r['decision'] == 'admit' and r['outcome'] is None
                           for r in rows.values()),
            'physical': len(work),
            'draining': sum(rows[i]['outcome'] is not None for i in work),
            'tx': len(tx),
            'resident': sum(r['decision'] == 'admit' and
                            (r['outcome'] is None or i in work)
                            for i, r in rows.items())}
        for key, value in values.items():
            areas[key] += dt * value
        if dt > 0:
            peak_resident = max(peak_resident, values['resident'])
        previous = t

    for t, kind, identifier in log:
        require(t <= hi, 'event beyond snapshot')
        require(last_event is None or t >= last_event, 'out of order event')
        last_event = t
        integrate(t)
        if kind == 'PLAN':
            require(identifier not in rows, 'duplicate plan identity')
            rows[identifier] = {'p': t, 'stage': 'PLANNED',
                                'decision': None, 'outcome': None}
            continue
        require(identifier in rows, 'event without request identity')
        row = rows[identifier]
        if kind == 'READY':
            require(row['stage'] == 'PLANNED' and row['outcome'] is None,
                    'illegal readiness')
            row['r'], row['stage'] = t, 'READY'
        elif kind == 'SEND':
            require(row['stage'] == 'READY' and row['outcome'] is None and not tx,
                    'illegal send or slot overcommit')
            row['s'], row['stage'] = t, 'SENT'
            tx.add(identifier)
        elif kind == 'TX_DONE':
            require(identifier in tx, 'transmit double release')
            tx.remove(identifier)
            row['g'] = t
        elif kind in ('ADMIT', 'REJECT_DECISION'):
            require('s' in row and row['decision'] is None,
                    'admission without send or duplicate decision')
            if kind == 'ADMIT':
                row['a'], row['decision'] = t, 'admit'
                if row['outcome'] is None:
                    row['stage'] = 'QUEUED'
            else:
                row['decision'] = 'reject'
        elif kind == 'START':
            require(row['stage'] == 'QUEUED' and row['outcome'] is None and
                    row['decision'] == 'admit' and not work and 'b' not in row,
                    'illegal start or server overcommit')
            row['b'], row['stage'] = t, 'RUNNING'
            work.add(identifier)
        elif kind == 'FINISH':
            if identifier in finished:
                duplicates += 1
                continue
            require(identifier in work, 'completion without physical ownership')
            work.remove(identifier)
            finished.add(identifier)
            row['d'] = t
        elif kind in TERMINAL:
            if row['outcome'] is not None:
                require(row['outcome'] == kind, 'terminal outcome overwritten')
                continue
            if kind in ('COMPLETE', 'FAIL'):
                require(row.get('d') == t, 'model result before physical completion')
            if kind == 'REJECT':
                require(row['decision'] == 'reject', 'business rejection without decision')
            row['outcome'], row['x'] = kind, t
        else:
            raise AssertionError('unknown event kind: ' + kind)
    integrate(hi)
    outcomes = Counter(r['outcome'] for r in rows.values())
    counts = {
        'planned': len(rows),
        'attempted': sum('s' in r for r in rows.values()),
        'admitted': sum(r['decision'] == 'admit' for r in rows.values()),
        'rejected_decisions': sum(r['decision'] == 'reject' for r in rows.values()),
        'pending_decisions': sum('s' in r and r['decision'] is None for r in rows.values()),
        'completed': outcomes['COMPLETE'], 'failed': outcomes['FAIL'],
        'rejected': outcomes['REJECT'], 'cancelled': outcomes['CANCEL'],
        'timed_out': outcomes['TIMEOUT'], 'remaining': outcomes[None],
        'physical_inflight': len(work),
        'draining': sum(rows[i]['outcome'] is not None for i in work),
        'duplicate_finishes': duplicates}
    require(counts['planned'] == sum(counts[k] for k in
            ('completed', 'failed', 'rejected', 'cancelled', 'timed_out', 'remaining')),
            'business population conservation')
    require(counts['attempted'] == counts['admitted'] +
            counts['rejected_decisions'] + counts['pending_decisions'],
            'attempt decision conservation')
    return rows, areas, counts, peak_resident


def clipped_sum(intervals, lo, hi):
    return sum((max(F(0), min(end, hi) - max(start, lo))
                for start, end in intervals), F(0))


def midpoint_sweep(intervals, lo, hi):
    points = {lo, hi}
    for start, end in intervals:
        points.add(max(lo, min(hi, start)))
        points.add(max(lo, min(hi, end)))
    points = sorted(points)
    area, peak = F(0), 0
    for left, right in zip(points, points[1:]):
        mid = (left + right) / 2
        live = sum(start <= mid < end for start, end in intervals)
        area += live * (right - left)
        peak = max(peak, live)
    return area, peak


def verify(log, expected, lo, hi):
    rows, areas, counts, peak = audit(log, lo, hi)
    require(set(rows) == set(expected), 'reference population differs')
    intervals = {k: [] for k in areas}
    for identifier, ref in expected.items():
        row = rows[identifier]
        require((row['outcome'] or row['stage']) == ref['state'],
                identifier + ': reference business state')
        for key in FIELDS:
            wanted = F(str(ref[key])) if key in ref else None
            require(row.get(key) == wanted, identifier + ': reference timestamp ' + key)
        if 'a' in ref:
            intervals['logical'].append((F(str(ref['a'])), F(str(ref.get('x', hi)))))
            if 'x' not in ref or ('b' in ref and 'd' not in ref):
                release = hi
            else:
                release = max(F(str(ref['x'])), F(str(ref.get('d', ref['x']))))
            intervals['resident'].append((F(str(ref['a'])), release))
        if 'b' in ref:
            end = F(str(ref.get('d', hi)))
            intervals['physical'].append((F(str(ref['b'])), end))
            if 'x' in ref:
                intervals['draining'].append((max(F(str(ref['b'])),
                                                    F(str(ref['x']))), end))
        if 's' in ref:
            intervals['tx'].append((F(str(ref['s'])), F(str(ref.get('g', hi)))))
    for key in areas:
        reference = clipped_sum(intervals[key], lo, hi)
        sweep, _ = midpoint_sweep(intervals[key], lo, hi)
        require(reference == sweep == areas[key], key + ': independent window area')
    _, reference_peak = midpoint_sweep(intervals['resident'], lo, hi)
    require(peak == reference_peak, 'independent resident peak')
    return areas, counts, peak


def must_reject(label, action):
    try:
        action()
    except AssertionError as error:
        print('negative control detected:', label, '|', str(error))
        return
    raise AssertionError('negative control accepted: ' + label)


def fluid_reference(segments, step=F(1, 4), initial=F(0)):
    backlog = initial
    endpoints = []
    for duration, arrival_rate, service_rate in segments:
        duration, arrival_rate, service_rate = map(F, (duration, arrival_rate, service_rate))
        ticks = duration / step
        require(ticks.denominator == 1, 'fluid step must divide segment')
        for _ in range(int(ticks)):
            before = backlog
            arrivals = arrival_rate * step
            served = min(before + arrivals, service_rate * step)
            backlog = before + arrivals - served
            require(backlog >= 0 and before + arrivals == served + backlog,
                    'fluid conservation')
        endpoints.append(backlog)
    return endpoints


def burst_formula(lb, mu, duration, baseline, initial=F(0)):
    backlog = max(F(0), initial + (lb - mu) * duration)
    drain = F(0) if backlog == 0 else (
        backlog / (mu - baseline) if baseline < mu else None)
    return backlog, drain


def verify_burst_candidate(backlog, drain):
    endpoints = fluid_reference([(2, 160, 100), (3, 60, 100)])
    require(backlog == endpoints[0], 'burst independent conservation reference')
    require(drain == 3 and endpoints[1] == 0,
            'drain independent conservation reference')
    before_empty = fluid_reference([(2, 160, 100), (F(11, 4), 60, 100)])[-1]
    require(before_empty > 0, 'drain must not be reported too early')


def check_little_and_fluid():
    complete = [(F(0), F(t)) for t in (1, 3, 4, 5)]
    area, _ = midpoint_sweep(complete, F(0), F(5))
    require(area == clipped_sum(complete, F(0), F(5)) == 13,
            'complete-cohort independent area')
    arrival_rate = F(4, 5)
    completed_rate = F(4, 5)
    mean_stay = F(13, 4)
    require(area / 5 == arrival_rate * mean_stay == completed_rate * mean_stay,
            'completed rate valid for conserved complete cohort')
    over = [(F(0), F(9)), (F(99, 10), F(109, 10))]
    under = [(F(0), F(1)), (F(1), F(20))]
    for intervals, expected_area, direction in (
            (over, F(91, 10), 'over'), (under, F(10), 'under')):
        exact = clipped_sum(intervals, F(0), F(10)) / 10
        sweep, _ = midpoint_sweep(intervals, F(0), F(10))
        completed_stays = [end - start for start, end in intervals if end <= 10]
        mixed = F(len(intervals), 10) * sum(completed_stays) / len(completed_stays)
        require(sweep == expected_area == exact * 10, 'censored area reference')
        require((mixed > exact) if direction == 'over' else (mixed < exact),
                'censoring bias direction')
        print('finite window:', direction, 'exact L=', exact, 'mixed estimate=', mixed)
    b, d = burst_formula(F(160), F(100), F(2), F(60))
    verify_burst_candidate(b, d)
    require(burst_formula(F(100), F(100), F(2), F(60)) == (0, 0),
            'zero backlog boundary')
    critical = burst_formula(F(160), F(100), F(2), F(100))
    require(critical == (120, None), 'critical baseline has no finite drain')
    require(fluid_reference([(2, 160, 100), (10, 100, 100)])[-1] == 120,
            'critical baseline independent reference')
    require(fluid_reference([(2, 160, 100), (1, 60, 100), (2, 160, 100)]) ==
            [120, 80, 200], 'residual backlog between bursts')
    must_reject('arrival-only burst ignores service',
                lambda: verify_burst_candidate(F(160 * 2), F(3)))
    print('fluid reference: burst=', b, 'drain=', d, 'repeated-burst backlog=200')


def main():
    print('K-WORK-002 deterministic event and conservation model')
    inputs = [
        spec('I', -2, service=3),
        spec('A', 0, delay=F(3, 2), service=2, cancel=2, duplicate=True),
        spec('B', F(1, 4), cancel=F(5, 2)),
        spec('R', F(1, 2), delay=F(1, 4), reject=True),
        spec('C', F(1, 10), dep='I'),
        spec('U', F(13, 4), service=10, timeout=F(11, 2)),
        spec('D', F(1, 5), dep='C')]
    expected = {
        'I': dict(p=-2, r=-2, s=-2, g=-1, a=F(-3, 2), b=F(-3, 2),
                  d=F(3, 2), x=F(3, 2), state='COMPLETE'),
        'A': dict(p=0, r=0, s=0, g=1, a=F(3, 2), b=F(3, 2),
                  d=F(7, 2), x=2, state='CANCEL'),
        'B': dict(p=F(1, 4), r=F(1, 4), s=1, g=2, a=F(3, 2),
                  x=F(5, 2), state='CANCEL'),
        'R': dict(p=F(1, 2), r=F(1, 2), s=2, g=3,
                  x=F(9, 4), state='REJECT'),
        'C': dict(p=F(1, 10), r=F(3, 2), s=3, g=4, a=F(7, 2),
                  b=F(7, 2), d=F(9, 2), x=F(9, 2), state='COMPLETE'),
        'U': dict(p=F(13, 4), r=F(13, 4), s=4, g=5, a=F(9, 2),
                  b=F(9, 2), x=F(11, 2), state='TIMEOUT'),
        'D': dict(p=F(1, 5), r=F(9, 2), s=5, g=6, a=F(11, 2), state='QUEUED')}
    lo, hi = F(0), F(6)
    log = simulate(inputs, hi)
    areas, counts, peak = verify(log, expected, lo, hi)
    require(areas == {'logical': F(11, 2), 'physical': F(6),
                     'draining': F(2), 'tx': F(6), 'resident': F(15, 2)},
            'composite trace reference areas')
    require(counts == {'planned': 7, 'attempted': 7, 'admitted': 6,
                      'rejected_decisions': 1, 'pending_decisions': 0,
                      'completed': 2, 'failed': 0, 'rejected': 1,
                      'cancelled': 2, 'timed_out': 1, 'remaining': 1,
                      'physical_inflight': 1, 'draining': 1,
                      'duplicate_finishes': 1}, 'composite cohort counters')
    require(peak == 2, 'conditional resident peak')
    print('window areas (request-seconds; resident assumes 1 MiB/admission):',
          {key: str(value) for key, value in areas.items()})
    print('cohort snapshot:', counts)
    print('conditional resident mean MiB=', areas['resident'] / (hi - lo),
          'peak MiB=', peak)

    cancelled_success = []
    for event in log:
        cancelled_success.append(event)
        if event == (F(7, 2), 'FINISH', 'A'):
            cancelled_success.append((F(7, 2), 'COMPLETE', 'A'))
    omitted_rejection = [e for e in log if not
                         (e[2] == 'R' and e[1] in ('REJECT_DECISION', 'REJECT'))]
    wrong_admission = sorted(
        [(F(3), k, i) if k == 'ADMIT' and i == 'C' else (t, k, i)
         for t, k, i in log], key=lambda e: e[0])
    early_release = sorted(
        [(F(2), k, i) if (t, k, i) == (F(7, 2), 'FINISH', 'A') else (t, k, i)
         for t, k, i in log], key=lambda e: e[0])
    controls = [
        ('cancelled request counted successful', cancelled_success),
        ('rejection silently omitted', omitted_rejection),
        ('send timestamp used as admission', wrong_admission),
        ('transmit release does not pump backlog', simulate(inputs, hi, True)),
        ('physical completion recorded at cancellation', early_release)]
    for label, faulty_log in controls:
        must_reject(label, lambda trace=faulty_log: verify(trace, expected, lo, hi))

    verify(simulate([], F(1)), {}, F(0), F(1))
    zero = [spec('F', 0, tx=0, delay=0, service=0, fail=True)]
    zero_expected = {'F': dict(p=0, r=0, s=0, g=0, a=0, b=0,
                               d=0, x=0, state='FAIL')}
    verify(simulate(zero, F(1)), zero_expected, F(0), F(1))
    simultaneous = [spec('S', 0, tx=0, delay=0), spec('T', 0, tx=0, delay=0)]
    simultaneous_expected = {
        'S': dict(p=0, r=0, s=0, g=0, a=0, b=0, d=1, x=1, state='COMPLETE'),
        'T': dict(p=0, r=0, s=0, g=0, a=0, b=1, d=2, x=2, state='COMPLETE')}
    verify(simulate(simultaneous, F(2)), simultaneous_expected, F(0), F(2))
    print('edge cases: empty cohort, zero-duration failure, simultaneous arrivals')
    check_little_and_fluid()
    print('negative controls detected: five event defects and one fluid defect')
    print('scope: offline exact-rational software model; no CPU/GPU service, network,')
    print('clock alignment, real thread synchronization, streaming output, or SLO measurement')
    print('all model checks passed')


if __name__ == '__main__':
    main()
