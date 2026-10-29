#!/usr/bin/env python3
'''Deterministic, read-only, offline stdlib model check for K-TOOLS-001.

No project imports, file access, network, credentials or subprocesses.
Integer endpoints and bounded common offsets are model inputs.
Manifest values are assumptions, not evidence about a collector.
'''

from fractions import Fraction
from itertools import product


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def interval(value):
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError('expected endpoint tuple')
    s, e = value
    if type(s) is not int or type(e) is not int or e < s:
        raise ValueError('invalid integer interval')
    return s, e


def merge(xs):
    checked = [interval(x) for x in xs]
    out = []
    for s, e in sorted(x for x in checked if x[0] < x[1]):
        if out and s <= out[-1][1]:
            out[-1][1] = max(e, out[-1][1])
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def union_length(xs):
    return sum(e - s for s, e in merge(xs))


def coverage(query, xs):
    q0, q1 = interval(query)
    if q0 == q1:
        return False
    cursor = q0
    for s, e in merge(xs):
        if e <= cursor:
            continue
        if s > cursor:
            return False
        cursor = max(cursor, e)
        if cursor >= q1:
            return True
    return False


def occupied_units(xs):
    result = set()
    for x in xs:
        s, e = interval(x)
        result.update(range(s, e))
    return result


def faulty_envelope(query, xs):
    q0, q1 = interval(query)
    return bool(xs) and min(s for s, e in xs) <= q0 and max(e for s, e in xs) >= q1


def shape(record):
    s, e = interval((record['start'], record['end']))
    form = record.get('form')
    if form == 'point':
        if s != e:
            raise ValueError('point needs one timestamp')
        return 'point', s, e
    if form != 'duration':
        raise ValueError('unknown geometry')
    if s == e:
        if record.get('zero_is_occurrence') is not True:
            raise ValueError('ambiguous zero-duration row')
        return 'point', s, e
    return 'span', s, e


def temporal(event, shifts, query):
    '''Common continuous offset interval; active-event semantics only.'''
    form, s, e = event
    lo, hi = interval(shifts)
    q0, q1 = interval(query)
    if q0 == q1:
        raise ValueError('empty temporal query')
    if form == 'point' and s == e:
        possible = s + lo < q1 and s + hi >= q0
        certain = s + lo >= q0 and s + hi < q1
    elif form == 'span' and s < e:
        possible = s + lo < q1 and e + hi > q0
        certain = s + hi < q1 and e + lo > q0
    else:
        raise ValueError('invalid normalized event')
    return 'ALL' if certain else 'MAY' if possible else 'NONE'


def reference_temporal(event, shifts, query):
    '''Independent offset and half-unit occupied-cell enumeration.

    All input boundaries are integers. The offset predicates change at
    integer boundaries; integer endpoints and half-unit interior witnesses
    therefore cover the continuous offset classifications in this family.
    '''
    form, s, e = event
    lo, hi = interval(shifts)
    q0, q1 = interval(query)
    window_cells = set(range(2 * q0, 2 * q1))
    answers = []
    for shift2 in range(2 * lo, 2 * hi + 1):
        if form == 'point':
            answers.append(2 * s + shift2 in window_cells)
        else:
            cells = set(range(2 * s + shift2, 2 * e + shift2))
            answers.append(bool(cells.intersection(window_cells)))
    return 'ALL' if all(answers) else 'MAY' if any(answers) else 'NONE'


GATES = (
    'supported', 'enabled', 'scope_complete', 'loss_free', 'table',
    'schema', 'query_complete', 'event_complete', 'finalized',
    'boundary_complete', 'clock_certified',
)


def manifest():
    result = {key: True for key in GATES}
    result.update(query=(0, 10), chunks=[(0, 5), (5, 10)])
    return result


def row(s, e, **changes):
    result = dict(kind='event', scope='dev0', physical_id='p0',
                  members=('R',), fields_ok=True, form='duration',
                  start=s, end=e, zero_is_occurrence=True, shifts=(0, 0))
    result.update(changes)
    return result


def decide_states(trusted, uncertain, unjoined, reasons, conditions):
    '''Chapter step 9, including its actual mutually exclusive structure.

    There is no early return on presence. Both branches join before
    returning to the duration/causal stages in audit().
    '''
    if trusted:
        presence = 'SUPPORTED'
        absence = 'REFUTED'
    else:
        presence = 'PRESENT_UNJOINED' if unjoined else 'UNKNOWN'
        if (not uncertain and not reasons and
                all(conditions.get(key) is True for key in GATES)):
            absence = 'SUPPORTED_UNDER_CERTIFICATE'
        else:
            absence = 'UNSUPPORTED'
    return presence, absence


def faulty_unconditional_step10(trusted, uncertain, unjoined, reasons, conditions):
    '''Reproduce the retained review's overwrite after the presence branch.'''
    presence, absence = decide_states(trusted, uncertain, unjoined, reasons, conditions)
    if (not trusted and not uncertain and not reasons and
            all(conditions.get(key) is True for key in GATES)):
        absence = 'SUPPORTED_UNDER_CERTIFICATE'
    else:
        absence = 'UNSUPPORTED'
    return presence, absence


def audit(m, records, state_resolver=decide_states):
    '''Conservative reduced audit for active-event queries.

    Bad potentially relevant rows remain in uncertain/reasons. Complete
    duplicate groups are resolved before a witness is trusted. The scope
    is one explicitly named domain, device and request, with exact-clock
    trustworthy coverage chunks. It does not parse real collector output.
    '''
    q0, q1 = interval(m['query'])
    if q0 == q1:
        return 'NO_QUERY', 'NOT_APPLICABLE', ('EMPTY_QUERY',), {}
    reasons = {key for key in GATES if m.get(key) is not True}
    uncertain = []
    unjoined = []
    trusted = []
    try:
        if not coverage(m['query'], m['chunks']):
            reasons.add('WINDOW_GAP')
    except (KeyError, TypeError, ValueError):
        reasons.add('COVERAGE_INVALID')

    normalized = {}
    conflicts = set()
    for r in records:
        if not isinstance(r, dict):
            reasons.add('INVALID_ROW')
            uncertain.append(r)
            continue
        kind, scope = r.get('kind'), r.get('scope')
        if isinstance(kind, str) and kind != 'event':
            continue
        if isinstance(scope, str) and scope != 'dev0':
            continue
        if kind != 'event' or scope != 'dev0':
            reasons.add('UNKNOWN_DOMAIN_OR_SCOPE')
            uncertain.append(r)
            continue
        pid = r.get('physical_id')
        if r.get('fields_ok') is not True or not isinstance(pid, str) or not pid:
            reasons.add('INVALID_ROW')
            uncertain.append(r)
            continue
        try:
            event = shape(r)
        except (KeyError, TypeError, ValueError):
            reasons.add('INVALID_GEOMETRY')
            uncertain.append(r)
            continue
        members = r.get('members')
        if members is not None and (not isinstance(members, tuple) or
                any(not isinstance(x, str) or not x for x in members)):
            reasons.add('INVALID_IDENTITY')
            uncertain.append(r)
            continue
        try:
            shifts = interval(r.get('shifts'))
        except (TypeError, ValueError):
            reasons.add('TIME_UNKNOWN')
            uncertain.append(r)
            continue
        if m.get('clock_certified') is not True:
            reasons.add('TIME_UNKNOWN')
            uncertain.append(r)
            continue
        value = (event, shifts, members)
        if pid in normalized:
            old_record, old_value = normalized[pid]
            if old_record != r or old_value != value:
                conflicts.add(pid)
                reasons.add('DUPLICATE_CONFLICT')
        else:
            normalized[pid] = (r, value)

    for pid, (raw, value) in normalized.items():
        if pid in conflicts:
            uncertain.append(raw)
            continue
        event, shifts, members = value
        membership = temporal(event, shifts, m['query'])
        if membership == 'NONE' or (members is not None and 'R' not in members):
            continue
        if members is None:
            reasons.add('JOIN_UNKNOWN')
            uncertain.append(raw)
            if membership == 'ALL':
                unjoined.append(raw)
        elif membership == 'MAY':
            reasons.add('UNKNOWN_AT_BOUNDARY')
            uncertain.append(raw)
        else:
            trusted.append((pid, event, shifts))

    # Steps 9-10: the branch joins here; no witness-specific return.
    presence, absence = state_resolver(trusted, uncertain, unjoined, reasons, m)

    # Step 11 is reached with and without witnesses. Point values are
    # busy-time contributions, not claims about real execution duration.
    contributions = {}
    for pid, event, shifts in trusted:
        form, s, e = event
        lo, hi = shifts
        if form == 'point':
            contributions[pid] = 0
            continue
        complete = s + lo >= q0 and e + hi <= q1
        try:
            complete = complete and coverage((s + lo, e + hi), m['chunks'])
        except (KeyError, TypeError, ValueError):
            complete = False
        contributions[pid] = e - s if complete else 'PARTIAL'

    # Step 12: no dependency evidence is supplied by this reduced model.
    # A successful join or time match must not invent a causal proof.
    post = dict(busy_contributions=contributions,
                causal_status='REQUIRES_DEPENDENCY_EVIDENCE',
                reached=('duration', 'causal'))
    return presence, absence, tuple(sorted(reasons)), post


def faulty_drop_invalid(m, records):
    return audit(m, [r for r in records if r.get('fields_ok') and r.get('physical_id')])


def faulty_nominal_clock(m, records):
    return audit(m, [dict(r, shifts=(0, 0)) for r in records])


def faulty_drop_points(m, records):
    return audit(m, [r for r in records if r['start'] < r['end']])


def test_intervals():
    intervals = [(s, e) for s in range(6) for e in range(s, 6)]
    count = 0
    for left, right in product(intervals, repeat=2):
        xs = [left, right]
        units = occupied_units(xs)
        need(union_length(xs) == len(units), 'union versus independent units')
        need(coverage((0, 5), xs) == set(range(5)).issubset(units),
             'coverage versus independent units')
        count += 1
    need(union_length([]) == 0, 'empty union')
    need(not coverage((0, 5), []), 'empty coverage')
    need(not coverage((5, 5), [(0, 10)]), 'empty query is not a certificate')
    need(coverage((0, 10), [(0, 5), (5, 10)]), 'adjacent coverage')
    gap = [(0, 5), (6, 10)]
    need(faulty_envelope((0, 10), gap) and not coverage((0, 10), gap),
         'coverage-envelope negative control not detected')
    overlap = [(0, 10), (5, 15)]
    need(union_length(overlap) == 15 and sum(e - s for s, e in overlap) == 20,
         'sum-as-union negative control not detected')
    try:
        union_length([(6, 5)])
    except ValueError:
        pass
    else:
        raise AssertionError('negative original interval accepted')
    return count


def test_temporal():
    count = 0
    for q0 in range(-1, 2):
        for q1 in range(q0 + 1, 5):
            for s in range(-2, 6):
                for e in range(s, 6):
                    event = ('point' if s == e else 'span', s, e)
                    for lo in range(-2, 2):
                        for hi in range(lo, 3):
                            need(temporal(event, (lo, hi), (q0, q1)) ==
                                 reference_temporal(event, (lo, hi), (q0, q1)),
                                 'time classification versus offset enumeration')
                            count += 1
    cases = [
        (('point', 0, 0), (0, 0), 'ALL'),
        (('point', 5, 5), (0, 0), 'ALL'),
        (('point', 10, 10), (0, 0), 'NONE'),
        (('point', 10, 10), (-1, 0), 'MAY'),
        (('span', 10, 11), (-2, 0), 'MAY'),
        (('span', -2, 0), (0, 0), 'NONE'),
    ]
    for event, shifts, expected in cases:
        need(temporal(event, shifts, (0, 10)) == expected, 'fixed temporal case')
    return count


def test_state_contract():
    # Independent declarative contract: a witness refutes absence even
    # when the hypothetical completeness flag is false. No control-flow
    # simulation is used to derive these four expected absence states.
    absence_table = {
        (True, True): 'REFUTED',
        (True, False): 'REFUTED',
        (False, True): 'SUPPORTED_UNDER_CERTIFICATE',
        (False, False): 'UNSUPPORTED',
    }
    presence_table = {
        (True, True): 'SUPPORTED',
        (True, False): 'SUPPORTED',
        (False, True): 'PRESENT_UNJOINED',
        (False, False): 'UNKNOWN',
    }
    count = killed = 0
    for witness, unknown, blocked, gate, joined_unknown in product(
            (False, True), (False, True), (False, True),
            (True, False, None), (False, True)):
        m = manifest()
        m['loss_free'] = gate
        trusted = ['witness'] if witness else []
        uncertain = ['candidate'] if unknown else []
        unjoined = ['domain-event'] if joined_unknown else []
        reasons = ['WINDOW_GAP'] if blocked else []
        certificate_ready = not unknown and not blocked and gate is True
        expected = (presence_table[witness, joined_unknown],
                    absence_table[witness, certificate_ready])
        args = (trusted, uncertain, unjoined, reasons, m)
        need(decide_states(*args) == expected, 'state contract mismatch')
        if witness:
            need(faulty_unconditional_step10(*args) != expected,
                 'unconditional step 10 overwrite was not detected')
            killed += 1
        count += 1
    need((count, killed) == (48, 24), 'state family was not fully visited')
    return count, killed


def test_audit():
    m = manifest()
    certificate = 'SUPPORTED_UNDER_CERTIFICATE'
    count = 0

    def expect(records, absence, data=None, presence=None):
        nonlocal count
        result = audit(m if data is None else data, records)
        need(result[1] == absence, 'audit absence mismatch: %r' % (result,))
        if presence is not None:
            need(result[0] == presence, 'audit presence mismatch')
        count += 1
        return result

    expect([], certificate)
    for gate in GATES:
        for missing in (False, True):
            broken = manifest()
            if missing:
                del broken[gate]
            else:
                broken[gate] = False
            expect([], 'UNSUPPORTED', broken)
        broken = manifest()
        broken[gate] = False
        # Explicit mutant repairs/ignores precisely the failing gate.
        repaired = dict(broken)
        repaired[gate] = True
        need(audit(repaired, [])[1] == certificate and
             audit(broken, [])[1] != certificate, 'omitted-gate mutant not detected')

    gap = manifest()
    gap['chunks'] = [(0, 5), (6, 10)]
    expect([], 'UNSUPPORTED', gap)
    empty = manifest()
    empty['query'] = (5, 5)
    expect([], 'NOT_APPLICABLE', empty, 'NO_QUERY')
    for r in [row(0, 0, form='point'), row(5, 5, form='point'),
              row(5, 5), row(4, 9)]:
        expect([r], 'REFUTED', presence='SUPPORTED')
    for r in [row(10, 10, form='point'), row(10, 11),
              row(4, 9, members=('OTHER',)), row(4, 9, members=()),
              row(4, 9, kind='other'), row(4, 9, scope='other')]:
        expect([r], certificate)

    invalid = row(4, 9, physical_id=None)
    bad_field = row(4, 9, fields_ok=False)
    shifted = row(10, 11, shifts=(-2, 0))
    unknowns = [invalid, bad_field, shifted, row(100, 101, shifts=None),
                row(5, 5, zero_is_occurrence=False), row(6, 5)]
    for r in unknowns:
        expect([r], 'UNSUPPORTED')
    result = expect([row(4, 9, members=None)], 'UNSUPPORTED', presence='PRESENT_UNJOINED')
    need('JOIN_UNKNOWN' in result[2], 'unjoined blocker lost')
    expect([row(10, 10, form='point', shifts=(-1, 0))], 'UNSUPPORTED')
    expect([row(10, 11, shifts=(-2, -2))], 'REFUTED', presence='SUPPORTED')
    expect([row(100, 101, physical_id=None)], 'UNSUPPORTED')

    duplicate = row(4, 9)
    expect([duplicate, dict(duplicate)], 'REFUTED', presence='SUPPORTED')
    result = expect([duplicate, row(20, 21)], 'UNSUPPORTED')
    need('DUPLICATE_CONFLICT' in result[2], 'conflict group not quarantined')
    expect([duplicate, row(6, 7, physical_id='p1')], 'REFUTED', presence='SUPPORTED')

    for bad in (invalid, bad_field):
        need(faulty_drop_invalid(m, [bad])[1] == certificate and
             audit(m, [bad])[1] != certificate, 'invalid-row deletion not detected')
    need(faulty_nominal_clock(m, [shifted])[1] == certificate and
         audit(m, [shifted])[1] != certificate, 'nominal clock mutant not detected')
    point = row(5, 5, form='point')
    need(faulty_drop_points(m, [point])[1] == certificate and
         audit(m, [point])[1] == 'REFUTED', 'point deletion not detected')
    need(union_length([(5, 5)]) == 0 and audit(m, [row(5, 5)])[0] == 'SUPPORTED',
         'zero busy length incorrectly implies absence')
    return count


def test_witness_with_blockers():
    count = 0
    for form in ('point', 'span'):
        witness = row(5, 5, form='point') if form == 'point' else row(4, 9)
        expected_duration = 0 if form == 'point' else 5
        for has_gap, has_unknown in product((False, True), repeat=2):
            m = manifest()
            if has_gap:
                # Gap is elsewhere; the witness remains completely covered.
                m['chunks'] = [(1, 10)]
            records = [witness]
            if has_unknown:
                records.append(row(7, 8, physical_id=None))
            result = audit(m, records)
            need(result[:2] == ('SUPPORTED', 'REFUTED'),
                 'valid witness lost its existence/refutation state')
            if has_gap:
                need('WINDOW_GAP' in result[2], 'gap reason was erased')
            if has_unknown:
                need('INVALID_ROW' in result[2], 'unknown reason was erased')
            post = result[3]
            need(post['reached'] == ('duration', 'causal'),
                 'presence branch bypassed later processing')
            need(post['busy_contributions']['p0'] == expected_duration,
                 'later duration processing failed')
            need(post['causal_status'] == 'REQUIRES_DEPENDENCY_EVIDENCE',
                 'presence manufactured a causal proof')
            faulty = audit(m, records, faulty_unconditional_step10)
            need(faulty[:2] == ('SUPPORTED', 'UNSUPPORTED') and
                 faulty[:2] != result[:2], 'actual-audit overwrite mutant not detected')
            need(faulty[3] == post, 'negative control changed unrelated postprocessing')
            count += 1
    # Complete record with a cropped query still supports active presence,
    # but its full duration cannot be charged as completely in that query.
    partial = audit(manifest(), [row(-1, 5)])
    need(partial[:2] == ('SUPPORTED', 'REFUTED') and
         partial[3]['busy_contributions']['p0'] == 'PARTIAL',
         'partial duration changed existence or became full duration')
    return count


def queue_scan(initial, schedule):
    if type(initial) is not int or initial < 0:
        raise ValueError('invalid initial occupancy')
    q = peak = initial
    arrivals_total = drain_total = 0
    for arrivals, capacity in schedule:
        if type(arrivals) is not int or type(capacity) is not int or min(arrivals, capacity) < 0:
            raise ValueError('invalid queue input')
        arrivals_total += arrivals
        q += arrivals
        peak = max(peak, q)
        drained = min(q, capacity)
        q -= drained
        drain_total += drained
        need(q == initial + arrivals_total - drain_total, 'byte conservation')
        need(0 <= drain_total <= initial + arrivals_total, 'drain of nonexistent bytes')
    return peak, q


def reference_queue(initial, schedule):
    queue = [None] * initial
    peak = len(queue)
    for arrivals, capacity in schedule:
        for _ in range(arrivals):
            queue.append(None)
            peak = max(peak, len(queue))
        for _ in range(capacity):
            if queue:
                queue.pop(0)
    return peak, len(queue)


def test_buffer():
    count = 0
    choices = list(product(range(3), repeat=2))
    for initial in range(3):
        for schedule in product(choices, repeat=3):
            need(queue_scan(initial, schedule) == reference_queue(initial, schedule),
                 'queue versus independent byte reference')
            count += 1
    peak, final = queue_scan(2, [(9, 0), (0, 10)])
    need((peak, final) == (11, 1), 'hand-computable peak')
    need(final <= 10 and not peak <= 10, 'endpoint-only mutant not detected')
    need(peak <= 11, 'capacity 11 should cover this trajectory')
    need(queue_scan(0, [(0, 100), (9, 0)]) == (9, 9), 'unused service was banked')
    need(queue_scan(2, [(9, 10)])[0] == 11, 'same-time peak hidden')

    fluid_count = 0
    for a, d, initial in product(range(4), repeat=3):
        for rates in product(range(a + 1), repeat=3):
            q = peak = Fraction(initial)
            for rate in rates:
                # On each constant-rate segment the fluid trajectory is
                # monotone until empty; its maximum is at an endpoint.
                q = max(Fraction(0), q + rate - d)
                peak = max(peak, q)
            bound = initial + max(0, a - d) * len(rates)
            need(peak <= bound, 'no-burst fluid rate bound')
            fluid_count += 1
    initial, a, d, horizon, capacity = 11, 0, 10, 2, 10
    need(initial + (a - d) * horizon <= capacity and
         not initial + max(0, a - d) * horizon <= capacity,
         'initial-occupancy mutant not detected')
    return count, fluid_count


def test_shared():
    physical = [(4, 9)]
    logical_waits = [(4, 9), (4, 9)]
    need(union_length(physical) == 5, 'physical work duration')
    need(sum(e - s for s, e in logical_waits) == 10 and
         union_length(physical) != 10, 'logical double-charge not detected')
    weights = (Fraction(1, 2), Fraction(1, 2))
    need(sum(weights) == 1 and sum(5 * w for w in weights) == 5,
         'resource charge conservation')
    return 1


def main():
    counts = {}
    counts['interval_reference_cases'] = test_intervals()
    counts['active_temporal_reference_cases'] = test_temporal()
    counts['state_contract_cases'], counts['state_overwrite_mutants_detected'] = test_state_contract()
    counts['certificate_cases'] = test_audit()
    counts['witness_with_blockers_cases'] = test_witness_with_blockers()
    counts['byte_queue_cases'], counts['fluid_rate_cases'] = test_buffer()
    counts['shared_work_cases'] = test_shared()
    for name in sorted(counts):
        print(name + '=' + str(counts[name]))
    print('negative_controls=unconditional-step10,coverage-envelope,sum-as-union,omitted-gate,point-deletion,invalid-row-deletion,nominal-clock,buffer-endpoint,initial-occupancy,logical-double-charge')
    print('PASS: finite observability states, active-event times, intervals and buffers agree with scoped references/invariants')
    print('scope=CPU arithmetic only; manifest assumptions are not collector evidence; no GPU, profiler, I/O, RDMA or service measurements')
    print('limits=no real trace parser, general clock drift, uncertain coverage classifier or causal dependency graph implementation')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
