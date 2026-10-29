from collections import Counter
from fractions import Fraction
from itertools import product, permutations
import json

# Deterministic CPU models; integer event times are model inputs.
KEYS = ('offered', 'pending', 'rejected', 'pre_cancel', 'pre_timeout',
        'pre_failed', 'admitted', 'active', 'normal', 'post_cancel',
        'post_timeout', 'post_failed', 'qpass', 'qfail', 'qunknown',
        'late', 'good', 'tokens')
PRIORITY = {'d': 0, 't': 1, 'z': 2, 'q': 3}
TERMINALS = {'normal': 0, 'cancel': 1, 'timeout': 2, 'failed': 3}
NEGATIVES = []


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def audit(actual, expected):
    require(actual == expected, 'independent reference mismatch: %r != %r' %
            (actual, expected))


def detect(name, faulty, expected):
    actual = faulty()
    try:
        audit(actual, expected)
    except AssertionError:
        NEGATIVES.append(name)
    else:
        raise AssertionError('negative control escaped: ' + name)


def row(name, a=0, d=None, ts=(), z=None, q=None, extra=()):
    return dict(id=name, a=a, d=d, ts=tuple(ts), z=z, q=q,
                extra=tuple(extra))


def event_key(event):
    # Validate before deduplication, so True cannot alias integer position 1.
    if not isinstance(event, tuple) or len(event) != 3:
        raise ValueError('invalid event record')
    t, kind, value = event
    if type(t) is not int or kind not in PRIORITY:
        raise ValueError('invalid event time or type')
    if kind in ('d', 'q'):
        if type(value) is not bool:
            raise ValueError('decision and quality require boolean values')
        ordinal = int(value)
    elif kind == 't':
        if type(value) is not int or value < 0:
            raise ValueError('token position requires nonnegative integer')
        ordinal = value
    else:
        if type(value) is not str or value not in TERMINALS:
            raise ValueError('unknown terminal')
        ordinal = TERMINALS[value]
    return t, PRIORITY[kind], ordinal


def evaluate(records, T, H, first=3, gap=2, avg=2, bug=None):
    if T <= 0 or H < T:
        raise ValueError('invalid observation interval')
    out = Counter({key: 0 for key in KEYS})
    seen = set()
    for r in records:
        if r['id'] in seen:
            raise ValueError('duplicate request identity')
        seen.add(r['id'])
        if not 0 <= r['a'] < T:
            continue
        out['offered'] += 1
        events = []
        if r['d'] is not None:
            events.append((r['d'][0], 'd', r['d'][1]))
        events.extend((t, 't', k) for k, t in enumerate(r['ts']))
        if r['z'] is not None:
            events.append((r['z'][0], 'z', r['z'][1]))
        if r['q'] is not None:
            events.append((r['q'][0], 'q', r['q'][1]))
        events.extend(r['extra'])
        unique = set()
        for event in events:
            event_key(event)
            if event[0] <= H:
                unique.add(event)
        if bug == 'lexical_order':
            events = sorted(unique,
                            key=lambda e: (e[0], PRIORITY[e[1]], str(e[2])))
        else:
            events = sorted(unique, key=event_key)
        state, quality, decided, times = 'pending', None, False, []
        for t, kind, value in events:
            if t < r['a']:
                raise ValueError('event before offered arrival')
            if kind == 'd':
                if decided:
                    raise ValueError('conflicting decision')
                decided = True
                if state.startswith('pre_'):
                    out['late'] += 1
                elif state == 'pending':
                    state = 'active' if value else 'rejected'
                else:
                    raise ValueError('illegal decision transition')
            elif kind == 't':
                if state != 'active' or value != len(times):
                    raise ValueError('illegal visible token')
                times.append(t)
            elif kind == 'z':
                if state == 'pending' and value != 'normal':
                    state = 'pre_' + value
                elif state == 'active':
                    state = 'normal' if value == 'normal' else 'post_' + value
                else:
                    raise ValueError('illegal terminal transition')
            else:
                if state != 'normal' or quality is not None:
                    raise ValueError('unbound or conflicting quality')
                quality = value
        out[state] += 1
        if state in ('active', 'normal') or state.startswith('post_'):
            out['admitted'] += 1
        if state != 'normal':
            continue
        out['qunknown' if quality is None else
            ('qpass' if quality else 'qfail')] += 1
        gaps = [b - a for a, b in zip(times, times[1:])]
        tail_ok = (not gaps or
                   (max(gaps) <= gap and Fraction(sum(gaps), len(gaps)) <= avg))
        if bug == 'singleton' and len(times) == 1:
            tail_ok = False
        quality_ok = quality is True
        if bug == 'ignore_quality':
            quality_ok = True
        elif bug == 'unknown_quality':
            quality_ok = quality is not False
        if times and times[0] - r['a'] <= first and tail_ok and quality_ok:
            out['good'] += 1
            out['tokens'] += len(times)
    require(out['offered'] == out['pending'] + out['rejected'] +
            sum(out['pre_' + x] for x in ('cancel', 'timeout', 'failed')) +
            out['admitted'], 'offered conservation')
    require(out['admitted'] == out['active'] + out['normal'] +
            sum(out['post_' + x] for x in ('cancel', 'timeout', 'failed')),
            'admitted conservation')
    require(out['normal'] == out['qpass'] + out['qfail'] + out['qunknown'],
            'quality conservation')
    return dict(out)


def reference(records, T, H, first=3, gap=2, avg=2):
    # Direct endpoint classification of valid canonical records.
    # No event sorting, state-machine traversal or event_key reuse.
    out = Counter({key: 0 for key in KEYS})
    for r in records:
        if not 0 <= r['a'] < T:
            continue
        out['offered'] += 1
        d = r['d'] if r['d'] is not None and r['d'][0] <= H else None
        z = r['z'] if r['z'] is not None and r['z'][0] <= H else None
        q = r['q'] if r['q'] is not None and r['q'][0] <= H else None
        accepted = bool(d is not None and d[1] and
                        (z is None or d[0] <= z[0]))
        if z is not None and d is not None and d[0] > z[0]:
            out['late'] += 1
        if accepted:
            out['admitted'] += 1
            state = 'active' if z is None else (
                'normal' if z[1] == 'normal' else 'post_' + z[1])
        elif z is not None:
            state = 'pre_' + z[1]
        elif d is not None:
            state = 'rejected'
        else:
            state = 'pending'
        out[state] += 1
        if state != 'normal':
            continue
        out['qunknown' if q is None else ('qpass' if q[1] else 'qfail')] += 1
        ts = tuple(t for t in r['ts'] if t <= H)
        if not ts or q is None or not q[1] or ts[0] > r['a'] + first:
            continue
        edges_ok = all(ts[k] <= ts[k - 1] + gap for k in range(1, len(ts)))
        mean_ok = len(ts) == 1 or ts[-1] - ts[0] <= avg * (len(ts) - 1)
        if edges_ok and mean_ok:
            out['good'] += 1
            out['tokens'] += len(ts)
    return dict(out)


class Budget:
    def __init__(self, capacity, faulty=False):
        self.capacity, self.faulty = capacity, faulty
        self.held, self.used, self.done = {}, set(), set()

    def free(self):
        return self.capacity - sum(r['units'] for r in self.held.values())

    def reserve(self, identity, units):
        if identity in self.used or units < 0:
            raise ValueError('invalid reservation identity or amount')
        if units > self.free():
            return False
        self.used.add(identity)
        self.held[identity] = dict(units=units, live=True, readers=set())
        return True

    def launch(self, identity, ticket):
        r = self.held[identity]
        if not r['live'] or ticket in r['readers'] or (identity, ticket) in self.done:
            raise ValueError('invalid new access')
        r['readers'].add(ticket)

    def close(self, identity):
        r = self.held[identity]
        r['live'] = False
        if self.faulty:
            r['units'] = 0
        if not r['readers']:
            del self.held[identity]

    def finish(self, identity, ticket):
        key = (identity, ticket)
        if key in self.done:
            return False
        r = self.held[identity]
        if ticket not in r['readers']:
            raise ValueError('completion without registered access')
        r['readers'].remove(ticket)
        self.done.add(key)
        if not r['live'] and not r['readers']:
            del self.held[identity]
        return True


def lifetime_trace(faulty=False):
    b = Budget(5, faulty)
    require(b.reserve('A:g1', 4), 'initial reservation')
    b.launch('A:g1', 'dma')
    b.launch('A:g1', 'kernel')
    trace = [b.free()]
    b.close('A:g1')
    trace.extend((b.free(), b.reserve('B:g1', 2)))
    b.finish('A:g1', 'dma')
    trace.append(b.free())
    b.finish('A:g1', 'kernel')
    trace.append(b.free())
    old = b.free()
    require(not b.finish('A:g1', 'kernel') and b.free() == old,
            'duplicate completion must be idempotent')
    return trace


def dominates(a, b):
    return (a[1] >= b[1] and a[2] <= b[2] and
            (a[1] > b[1] or a[2] < b[2]))


def frontier(points, bypass_feasibility=False):
    kept = []
    for p in points:
        if not p[3] and not bypass_feasibility:
            continue
        if any(dominates(q, p) for q in kept):
            continue
        kept = [q for q in kept if not dominates(p, q)]
        kept.append(p)
    return sorted(p[0] for p in kept)


def expect_invalid(name, records):
    try:
        evaluate(records, 10, 10)
    except ValueError:
        return
    raise AssertionError('invalid trace accepted: ' + name)


def stage_model(offered_rate, alpha, visits):
    admitted_rate = offered_rate * alpha
    return tuple(admitted_rate * v for v in visits)


def within_resources(rates, capacities=(12, 10, 4), K=Fraction(1, 2)):
    return (rates[0] <= capacities[0] and rates[1] <= capacities[1]
            and rates[2] * K <= capacities[2])


def flow_checks():
    # Independent identity ledger, with fixed single-path admission decisions.
    duration = 10
    offered = tuple(range(200))
    admitted = tuple(i for i in offered if i % 5 < 2)
    rejected = tuple(i for i in offered if i % 5 >= 2)
    ledger = []
    for identity in admitted:
        ledger.extend(((identity, 'p'), (identity, 'd'), (identity, 'x')))
    audit((len(offered), len(admitted), len(rejected)), (200, 80, 120))
    counts = Counter(stage for _, stage in ledger)
    rates = tuple(Fraction(counts[s], duration) for s in ('p', 'd', 'x'))
    predicted = stage_model(Fraction(len(offered), duration),
                            Fraction(len(admitted), len(offered)), (1, 1, 1))
    audit(predicted, rates)
    audit(rates, (8, 8, 8))
    require(within_resources(rates), 'allowed-rejection average resource point')
    # The exact same resource predicate must reject the faulty population.
    detect('offered_rate_as_stage_rate',
           lambda: within_resources((20, 20, 20)), within_resources(rates))

    # Complete byte-time area of the 80 cohort objects, including the tail.
    # Dividing by the arrival duration is a cohort normalization; a periodic
    # stable repetition has the corresponding long-run average occupancy.
    intervals = tuple((Fraction(k, 8), Fraction(k, 8) + Fraction(1, 5),
                       Fraction(1, 2)) for k in range(len(admitted)))
    area_reference = sum(((end - start) * size for start, end, size in intervals),
                         Fraction()) / duration
    buffer_model = rates[2] * Fraction(1, 5) * Fraction(1, 2)
    audit(buffer_model, area_reference)
    audit(area_reference, Fraction(4, 5))
    detect('offered_rate_as_buffer_entry_rate',
           lambda: 20 * Fraction(1, 5) * Fraction(1, 2), area_reference)

    # Same admitted population: extra complete attempts, some decode bypasses,
    # plus one p-only cancellation. Counts do not depend on the rate formula.
    varied = list(ledger)
    for identity in admitted[:8]:
        varied.extend(((identity, 'p'), (identity, 'd'), (identity, 'x')))
    bypass = set(admitted[8:24])
    varied = [(i, s) for i, s in varied if not (i in bypass and s in ('d', 'x'))]
    cancelled = admitted[24]
    varied = [(i, s) for i, s in varied if not (i == cancelled and s in ('d', 'x'))]
    varied_counts = Counter(s for _, s in varied)
    audit(tuple(varied_counts[s] for s in ('p', 'd', 'x')), (88, 71, 71))
    varied_rates = tuple(Fraction(varied_counts[s], duration) for s in ('p', 'd', 'x'))
    audit(stage_model(20, Fraction(2, 5),
                      (Fraction(88, 80), Fraction(71, 80), Fraction(71, 80))),
          varied_rates)
    detect('ignore_retries_bypass_and_cancel', lambda: predicted, varied_rates)
    # A transfer-only retry changes x while leaving the two pool visits fixed.
    transfer_retry = varied + [(admitted[0], 'x')]
    tr = Counter(s for _, s in transfer_retry)
    audit(tuple(tr[s] for s in ('p', 'd', 'x')), (88, 71, 72))
    require(not within_resources((8, 8, 9)), 'migration overload must fail')
    require(within_resources((8, 8, 100), K=0), 'zero-byte path has no byte demand')
    return dict(offered=200, admitted=80, rejected=120,
                stage_rates_per_s=[str(r) for r in rates],
                buffer_entry_rate_per_s=str(rates[2]),
                source_buffer_GiB=str(area_reference),
                varied_visit_counts=[88, 71, 71],
                transfer_retry_counts=[88, 71, 72])


def main():
    records = []
    shapes = ((), (1,), (3,), (4,), (1, 3), (1, 4), (1, 2, 3, 9))
    for a, offsets, quality in product((0, 5, 9, 10), shapes, (True, False, None)):
        ts = tuple(a + x for x in offsets)
        end = (ts[-1] if ts else a) + 1
        records.append(row('g' + str(len(records)), a, (a, True), ts,
                           (end, 'normal'),
                           None if quality is None else (end + 1, quality)))
    records.extend((row('reject', 1, (1, False)),
                    row('late', 1, (5, True), z=(3, 'cancel')),
                    row('pending', 2),
                    row('failed', 0, (0, True), (1,), (2, 'failed'))))
    for r in records:
        audit(evaluate([r], 10, 10), reference([r], 10, 10))
    total = evaluate(records, 10, 10)
    audit(total, reference(records, 10, 10))
    audit(evaluate([], 10, 10), reference([], 10, 10))
    for status in ('cancel', 'timeout', 'failed'):
        for d in (None, (0, True), (5, True), (5, False)):
            r = row('terminal', 0, d, z=(3, status))
            audit(evaluate([r], 10, 10), reference([r], 10, 10))

    tie_sizes = (1, 2, 10, 11, 12, 101)
    for size in tie_sizes:
        same = row('same', 0, (0, True), (1,) * size,
                   (2, 'normal'), (3, True))
        expected = reference([same], 10, 10)
        audit(evaluate([same], 10, 10), expected)
        audit((expected['good'], expected['tokens']), (1, size))
        all_tied = row('all_tied', 0, (1, True), (1,) * size,
                       (1, 'normal'), (1, True))
        audit(evaluate([all_tied], 10, 10), reference([all_tied], 10, 10))
        # Shuffle the full content log using every cyclic shift, including
        # descending order, while canonical endpoint data stays independent.
        if size in (11, 12):
            events = tuple((1, 't', k) for k in range(size))
            for shift in range(size):
                shuffled = tuple(reversed(events[shift:] + events[:shift]))
                raw = dict(same, ts=(), extra=shuffled)
                audit(evaluate([raw], 10, 10), expected)

    regression = row('same', 0, (0, True), (1,) * 11,
                     (2, 'normal'), (3, True))
    regression_expected = reference([regression], 10, 10)
    audit((regression_expected['good'], regression_expected['tokens']), (1, 11))
    audit(evaluate([regression], 10, 10), regression_expected)
    # Rejecting this known-valid record is itself the negative-control failure.
    try:
        evaluate([regression], 10, 10, bug='lexical_order')
    except ValueError as exc:
        require(str(exc) == 'illegal visible token', 'unexpected negative failure')
        NEGATIVES.append('lexical_same_time_token_order')
    else:
        raise AssertionError('lexical sorting negative escaped')

    singleton = row('single', 0, (0, True), (1,), (2, 'normal'), (3, True))
    repeated = dict(singleton, extra=((2, 'z', 'normal'), (3, 'q', True)))
    audit(evaluate([repeated], 10, 10), reference([singleton], 10, 10))
    expect_invalid('normal before admission', [row('bad', z=(1, 'normal'))])
    expect_invalid('conflicting terminal', [dict(singleton, extra=((4, 'z', 'cancel'),))])
    expect_invalid('token after terminal', [row('bad', 0, (0, True), (4,), (2, 'cancel'))])
    expect_invalid('duplicate identity', [singleton, singleton])
    expect_invalid('conflicting quality', [dict(singleton, extra=((4, 'q', False),))])
    expect_invalid('same-time conflicting decision',
                   [dict(singleton, extra=((0, 'd', False),))])
    expect_invalid('same-time conflicting terminal',
                   [dict(singleton, extra=((2, 'z', 'cancel'),))])
    expect_invalid('same-time conflicting certificate',
                   [dict(singleton, extra=((3, 'q', False),))])
    expect_invalid('boolean token position',
                   [dict(singleton, extra=((1, 't', True),))])
    expect_invalid('integer decision', [row('bad', d=(0, 1))])
    expect_invalid('unknown event', [row('bad', extra=((1, 'unknown', 0),))])
    expect_invalid('position missing',
                   [row('bad', d=(0, True), extra=((1, 't', 1),))])
    expect_invalid('decreasing position time',
                   [row('bad', d=(0, True), ts=(2, 1))])

    marginal = []
    for i in range(20):
        ts = (4, 5) if i == 19 else ((1, 4) if i == 0 else (1, 3))
        marginal.append(row('m' + str(i), 0, (0, True), ts,
                            (6, 'normal'), (7, True)))
    joint = reference(marginal, 10, 10)
    require(joint['good'] == 18, 'disjoint failure fixture')
    first_pass = sum(r['ts'][0] <= 3 for r in marginal)
    gap_pass = sum(r['ts'][1] - r['ts'][0] <= 2 for r in marginal)
    wrong_join = dict(evaluate(marginal, 10, 10), good=min(first_pass, gap_pass))
    detect('marginal_minimum_as_joint', lambda: wrong_join, joint)
    detect('ignore_quality', lambda: evaluate(records, 10, 10, bug='ignore_quality'),
           reference(records, 10, 10))
    detect('unknown_quality_as_pass',
           lambda: evaluate(records, 10, 10, bug='unknown_quality'),
           reference(records, 10, 10))
    detect('drop_rejected_offered',
           lambda: evaluate([r for r in records if r['id'] != 'reject'], 10, 10),
           reference(records, 10, 10))
    detect('singleton_requires_nonexistent_interval',
           lambda: evaluate([singleton], 10, 10, bug='singleton'),
           reference([singleton], 10, 10))
    stall = row('stall', 0, (0, True), (1, 2, 3, 13),
                (14, 'normal'), (15, True))
    audit(evaluate([stall], 10, 20, gap=5, avg=4),
          reference([stall], 10, 20, gap=5, avg=4))
    detect('mean_hides_stall', lambda: evaluate([stall], 10, 20, gap=100, avg=4),
           reference([stall], 10, 20, gap=5, avg=4))
    drain = row('drain', 9, (9, True), (10,), (11, 'normal'), (12, True))
    audit(evaluate([drain], 10, 12), reference([drain], 10, 12))
    require(evaluate([drain], 10, 10)['good'] == 0, 'cutoff censoring')
    qualified = evaluate([drain], 10, 12)['good']
    detect('arrival_denominator_after_drain', lambda: Fraction(qualified * 1000, 10),
           Fraction(qualified * 1000, 12))
    require(Fraction(81, 12) == Fraction(27, 4), 'cohort example rate')
    audit(lifetime_trace(), [1, 1, False, 1, 5])
    detect('free_on_cancel_before_last_reader', lambda: lifetime_trace(True),
           [1, 1, False, 1, 5])

    points = (('a', 6, 1, True), ('b', 13, 2, True),
              ('c', 5, 2, True), ('d', 14, 4, True),
              ('e', 13, 3, True), ('invalid', 100, 1, False))
    feasible = [p for p in points if p[3]]
    # Direct pairwise reference spells out inequalities independently.
    oracle = sorted(p[0] for p in feasible if not any(
        q[1] >= p[1] and q[2] <= p[2] and
        (q[1] > p[1] or q[2] < p[2]) for q in feasible))
    permutation_count = 0
    for order in permutations(points):
        audit(frontier(order), oracle)
        permutation_count += 1
    audit(frontier(()), [])
    audit(frontier((('x', 1, 1, True), ('y', 1, 1, True))), ['x', 'y'])
    detect('infeasible_point_enters_frontier', lambda: frontier(points, True), oracle)
    rates = {1: False, 2: True, 3: False, 4: True}
    audit([r for r in sorted(rates) if rates[r]], [2, 4])
    audit((20 + 5 + 40 + 5, 20 + 5 + 40 + 25 + 10 + 8 + 2,
           20 + 5 + 40 + 10 + 10 + 8 + 2), (70, 110, 95))
    require((6 + 6 - 1) // 6 == 1 and 2 * ((6 + 13 - 1) // 13) == 2,
            'replica rounding reversal')
    flows = flow_checks()
    print(json.dumps(dict(
        scope='Deterministic stdlib CPU event, ownership, flow and frontier models; no device or quality measurement',
        base_records=len(records), counts=total,
        same_time_token_sizes=list(tie_sizes),
        frontier=oracle, frontier_orders=permutation_count,
        negative_controls_detected=NEGATIVES,
        lifetime_available_units=lifetime_trace(),
        flow_observations=flows,
        pd_client_times_ms=[70, 110, 95],
        python_script_executed=True), sort_keys=True))


if __name__ == '__main__':
    main()
