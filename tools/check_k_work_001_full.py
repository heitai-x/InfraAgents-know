#!/usr/bin/env python3
"""K-WORK-001: deterministic, read-only, offline CPU reference models.
No project imports, subprocesses, network, credentials or file writes.
"""
from fractions import Fraction
from itertools import combinations_with_replacement, product
import struct


class MetricError(ValueError):
    pass


def ensure(condition, message):
    if not condition:
        raise MetricError(message)


def F(value):
    ensure(type(value) is int or isinstance(value, Fraction), 'expected exact rational')
    return Fraction(value)


def check(actual, expected):
    ensure(actual == expected, 'report mismatch: ' + repr((actual, expected)))


def expect_error(action, label):
    try:
        action()
    except MetricError:
        return
    except Exception as exc:
        raise MetricError(label + ': unexpected ' + type(exc).__name__) from exc
    raise MetricError(label + ': faulty candidate was accepted')


def event(eid, seq, kind, time, **extra):
    result = dict(id=eid, seq=seq, kind=kind, t=time, attempt=1, domain='client')
    result.update(extra)
    return result


def make_trace(times, groups=None, terminal='final', send=0):
    times = list(times)
    groups = [1] * len(times) if groups is None else list(groups)
    ensure(sum(groups) == len(times), 'group coverage')
    events = [event('send', 0, 'send', send)]
    cursor = 0
    for size in groups:
        ensure(type(size) is int and size > 0, 'group size')
        stop = cursor + size
        events.append(event('c' + str(cursor), len(events), 'content', times[stop - 1], start=cursor, stop=stop))
        cursor = stop
    if terminal is not None:
        end = times[-1] + 1 if times else send + 1
        events.append(event('end', len(events), terminal, end))
    return events


def request_metrics(events, attempt=1, mode='token'):
    ensure(mode in ('token', 'chunk'), 'unknown mode')
    seen = {}
    sequence = 0
    previous = None
    send = None
    end = None
    status = 'unsent'
    cursor = 0
    times = []
    sizes = []
    foreign = 0
    required = {'id', 'seq', 'kind', 't', 'attempt', 'domain'}
    for e in events:
        ensure(required <= e.keys(), 'missing event field')
        if e['attempt'] != attempt:
            foreign += 1
            continue
        ensure(isinstance(e['id'], str) and bool(e['id']), 'bad event id')
        if e['id'] in seen:
            ensure(seen[e['id']] == e, 'conflicting duplicate')
            continue
        seen[e['id']] = dict(e)
        ensure(e['domain'] == 'client', 'mixed boundary or clock domain')
        ensure(type(e['seq']) is int and e['seq'] == sequence, 'observer sequence')
        sequence += 1
        time = F(e['t'])
        ensure(previous is None or time >= previous, 'event time reversal')
        previous = time
        if status == 'unsent':
            ensure(e['kind'] == 'send', 'first selected event must be send')
            send = time
            status = 'open'
            continue
        ensure(status == 'open', 'event after terminal')
        if e['kind'] == 'meta':
            continue
        if e['kind'] == 'content':
            ensure({'start', 'stop'} <= e.keys(), 'missing content range')
            start, stop = e['start'], e['stop']
            ensure(type(start) is int and type(stop) is int, 'noninteger content range')
            ensure(start == cursor and stop > start, 'gap or overlapping content')
            size = stop - start
            ensure(mode != 'token' or size == 1, 'token event must represent one token')
            cursor = stop
            times.append(time)
            sizes.append(size)
        else:
            ensure(e['kind'] in ('final', 'cancel', 'timeout', 'error'), 'unknown event kind')
            status = e['kind']
            end = time
    ensure(send is not None, 'missing selected send')
    exact = mode == 'token' or all(size == 1 for size in sizes)
    gaps = [times[i] - times[i - 1] for i in range(1, len(times))] if exact and cursor > 1 else None
    tpot = (times[-1] - times[0]) / (cursor - 1) if exact and cursor > 1 else None
    return dict(tokens=cursor, ttft=times[0] - send if times else None,
                latency=end - send if status == 'final' else None,
                end=end, status=status, tpot=tpot, itl=gaps, foreign=foreign)


def reference_report(events, mode='token', attempt=1):
    """Declarative valid-trace reference; TPOT sums adjacent gaps."""
    unique = {}
    for e in events:
        if e['attempt'] == attempt and e['id'] not in unique:
            unique[e['id']] = e
    selected = list(unique.values())
    contents = [e for e in selected if e['kind'] == 'content']
    times = [F(e['t']) for e in contents]
    count = sum(e['stop'] - e['start'] for e in contents)
    start = F(next(e['t'] for e in selected if e['kind'] == 'send'))
    terminals = [e for e in selected if e['kind'] in ('final', 'cancel', 'timeout', 'error')]
    terminal = terminals[0] if terminals else None
    exact = mode == 'token' or all(e['stop'] - e['start'] == 1 for e in contents)
    gaps = [b - a for a, b in zip(times, times[1:])] if exact and count > 1 else None
    return dict(tokens=count, ttft=min(times) - start if times else None,
                latency=F(terminal['t']) - start if terminal and terminal['kind'] == 'final' else None,
                end=F(terminal['t']) if terminal else None,
                status=terminal['kind'] if terminal else 'open',
                tpot=sum(gaps, Fraction(0)) / len(gaps) if gaps is not None else None,
                itl=gaps, foreign=sum(e['attempt'] != attempt for e in events))


def from_token_envelopes(send, groups, final):
    """Exported token traces, explicitly NOT network chunks."""
    ensure(final is not None, 'explicit final event required')
    send = F(send)
    final = F(final)
    times = []
    previous = send
    for group in groups:
        ensure(group.get('kind') == 'token_trace', 'envelope is not a token trace')
        count = group['tokens']
        stamps = list(group['token_times'])
        ensure(type(count) is int and count > 0 and len(stamps) == count, 'envelope count')
        ensure(F(group['t']) == F(stamps[-1]), 'envelope end disagrees with last token')
        for stamp in stamps:
            stamp = F(stamp)
            ensure(stamp >= previous, 'token time reversal or token before send')
            times.append(stamp)
            previous = stamp
    ensure(final >= previous, 'final before last content')
    events = make_trace(times, send=send)
    events[-1]['t'] = final
    return request_metrics(events)


def partitions(count):
    if count == 0:
        yield []
        return
    for mask in range(1 << (count - 1)):
        sizes = []
        size = 1
        for i in range(count - 1):
            if mask & (1 << i):
                sizes.append(size)
                size = 1
            else:
                size += 1
        yield sizes + [size]


def nearest_rank(values, p):
    p = F(p)
    ensure(0 < p <= 1, 'quantile probability')
    values = sorted(F(v) for v in values)
    if not values:
        return None
    index = (p.numerator * len(values) + p.denominator - 1) // p.denominator
    return values[index - 1]


def validate_quantile(value, values, p):
    """Independent empirical-CDF criterion, without a rank calculation."""
    if not values:
        ensure(value is None, 'empty quantile must be undefined')
        return
    ensure(value in values, 'nearest-rank result must be an observed value')
    lower = Fraction(sum(x < value for x in values), len(values))
    upper = Fraction(sum(x <= value for x in values), len(values))
    ensure(lower < p <= upper, 'quantile violates empirical CDF')


def aggregate(reports):
    macros = [r['tpot'] for r in reports if r['tpot'] is not None]
    gaps = [gap for r in reports if r['itl'] is not None for gap in r['itl']]
    return dict(macro=sum(macros, Fraction(0)) / len(macros) if macros else None,
                pooled=sum(gaps, Fraction(0)) / len(gaps) if gaps else None,
                defined_requests=len(macros), intervals=len(gaps))


COUNT_KEYS = ('offered', 'pending', 'rejected', 'preterminated',
              'pre_cancel', 'pre_timeout', 'pre_error', 'admitted',
              'finished', 'incomplete', 'final', 'cancel', 'timeout',
              'error', 'good', 'late_decision')
TERMINALS = ('final', 'cancel', 'timeout', 'error')


def cohort_report(records, w0, w1, cutoff):
    """Complete ledger; direct time-condition classification.

    Decision wins equal timestamps. A later decision cannot revive a request
    already terminated before admission. Missing acquisition coverage is
    outside this complete-ledger input contract.
    """
    w0, w1, cutoff = F(w0), F(w1), F(cutoff)
    ensure(w0 < w1 and cutoff >= w1, 'invalid observation window')
    counts = dict.fromkeys(COUNT_KEYS, 0)
    seen = set()
    for rec in records:
        ensure(rec['id'] not in seen, 'duplicate request identity')
        seen.add(rec['id'])
        offered = F(rec['offered'])
        decision = rec['decision']
        terminal = rec['terminal']
        for flag in ('success', 'quality', 'slo'):
            ensure(type(rec[flag]) is bool, 'explicit qualification flags required')
        if decision is not None:
            decision_time, admitted = decision
            ensure(type(admitted) is bool, 'invalid admission result')
            ensure(F(decision_time) >= offered, 'decision before offer')
        if terminal is not None:
            terminal_time, reason = terminal
            ensure(F(terminal_time) >= offered, 'terminal before offer')
            ensure(reason in TERMINALS, 'unknown terminal')
            if reason == 'final':
                ensure(decision is not None and decision[1]
                       and F(decision[0]) <= F(terminal_time),
                       'normal final requires effective admission')
        if not w0 <= offered < w1:
            continue
        counts['offered'] += 1
        preterminal = (terminal is not None and terminal[1] != 'final'
                       and F(terminal[0]) <= cutoff
                       and (decision is None or F(terminal[0]) < F(decision[0])))
        if preterminal:
            counts['preterminated'] += 1
            counts['pre_' + terminal[1]] += 1
            if decision is not None and F(decision[0]) <= cutoff:
                counts['late_decision'] += 1
            continue
        if decision is None or F(decision[0]) > cutoff:
            counts['pending'] += 1
            continue
        if not decision[1]:
            counts['rejected'] += 1
            continue
        counts['admitted'] += 1
        if terminal is None or F(terminal[0]) > cutoff:
            counts['incomplete'] += 1
            continue
        counts['finished'] += 1
        counts[terminal[1]] += 1
        if terminal[1] == 'final' and rec['success'] and rec['quality'] and rec['slo']:
            counts['good'] += 1
    ensure(counts['offered'] == counts['pending'] + counts['rejected']
           + counts['preterminated'] + counts['admitted'], 'offered conservation')
    ensure(counts['preterminated'] == sum(counts[k] for k in ('pre_cancel', 'pre_timeout', 'pre_error')),
           'preadmission terminal conservation')
    ensure(counts['admitted'] == counts['finished'] + counts['incomplete'], 'admitted conservation')
    ensure(counts['finished'] == sum(counts[k] for k in TERMINALS), 'admitted terminal conservation')
    ensure(counts['late_decision'] <= counts['preterminated'], 'late decision subset')
    return dict(counts=counts,
                goodput_elapsed=Fraction(counts['good']) / (cutoff - w0),
                offered_rate=Fraction(counts['offered']) / (w1 - w0),
                qualified_fraction=Fraction(counts['good'], counts['offered']) if counts['offered'] else None)


def cohort_reference(records, w0, w1, cutoff):
    """Independent valid-input reference: replay events, then count states.

    This does not call cohort_report or use its preterminal inequality.
    """
    counts = dict.fromkeys(COUNT_KEYS, 0)
    for rec in records:
        if not F(w0) <= F(rec['offered']) < F(w1):
            continue
        counts['offered'] += 1
        state = 'pending'
        late = 0
        events = []
        if rec['decision'] is not None:
            time, accepted = rec['decision']
            events.append((F(time), 0, 'decision', accepted))
        if rec['terminal'] is not None:
            time, reason = rec['terminal']
            events.append((F(time), 1, 'terminal', reason))
        for time, priority, kind, value in sorted(events):
            if time > F(cutoff):
                break
            if kind == 'decision':
                if state.startswith('pre_'):
                    late += 1
                else:
                    ensure(state == 'pending', 'reference decision state')
                    state = 'incomplete' if value else 'rejected'
            elif state == 'pending':
                ensure(value != 'final', 'reference final without admission')
                state = 'pre_' + value
            elif state == 'incomplete':
                state = value
            else:
                ensure(state == 'rejected' and value != 'final', 'reference terminal state')
        counts[state] += 1
        if state.startswith('pre_'):
            counts['preterminated'] += 1
        if state in ('incomplete',) + TERMINALS:
            counts['admitted'] += 1
        if state in TERMINALS:
            counts['finished'] += 1
        if state == 'final' and rec['success'] and rec['quality'] and rec['slo']:
            counts['good'] += 1
        counts['late_decision'] += late
    return dict(counts=counts,
                goodput_elapsed=Fraction(counts['good']) / (F(cutoff) - F(w0)),
                offered_rate=Fraction(counts['offered']) / (F(w1) - F(w0)),
                qualified_fraction=Fraction(counts['good'], counts['offered']) if counts['offered'] else None)


def faulty_completed_denominator(records, w0, w1, cutoff):
    result = cohort_report(records, w0, w1, cutoff)
    counts = result['counts']
    result['goodput_elapsed'] = Fraction(counts['good'], counts['finished']) if counts['finished'] else None
    return result


def clock_interval(a, b, low, high):
    a, b, low, high = F(a), F(b), F(low), F(high)
    ensure(low <= high, 'reversed offset bounds')
    lower = max(Fraction(0), b - a - high)
    upper = b - a - low
    return None if lower > upper else (lower, upper)


def pcm_bytes(duration, rate, channels=1, bytes_per_sample=2):
    duration = F(duration)
    ensure(duration >= 0, 'negative duration')
    ensure(type(rate) is int and rate > 0, 'sample rate')
    ensure(type(channels) is int and channels > 0, 'channels')
    ensure(type(bytes_per_sample) is int and bytes_per_sample > 0, 'sample representation')
    samples = duration * rate
    ensure(samples.denominator == 1, 'fractional sample count')
    return int(samples) * channels * bytes_per_sample


def pcm_from_input(case):
    return pcm_bytes(case['duration'], case['rate'], case['channels'], case['bytes_per_sample'])


def faulty_codec_tokens_as_pcm_bytes(case):
    """Mechanism mutant: use codec token count as raw PCM byte count."""
    ensure(type(case['codec_tokens']) is int and case['codec_tokens'] >= 0, 'codec count')
    return case['codec_tokens']


def pcm_reference(case, frames, sample_format):
    """Construct actual silent PCM bytes from independently supplied frames."""
    ensure(type(frames) is int and frames >= 0, 'reference frame count')
    check(F(case['duration']), Fraction(frames, case['rate']))
    check(struct.calcsize('<' + sample_format), case['bytes_per_sample'])
    frame = struct.pack('<' + sample_format * case['channels'], *([0] * case['channels']))
    raw = frame * frames
    return len(raw)


def validate_pcm_output(actual, case, frames, sample_format):
    check(actual, pcm_reference(case, frames, sample_format))


def audio_buffer(arrivals, play_start, horizon):
    play_start, horizon = F(play_start), F(horizon)
    ensure(horizon >= play_start, 'invalid playback horizon')
    items = [(F(t), F(d)) for t, d in arrivals]
    for i, (time, duration) in enumerate(items):
        ensure(duration >= 0, 'negative playable duration')
        ensure(i == 0 or time >= items[i - 1][0], 'audio order')
    buffer = Fraction(0)
    underrun = Fraction(0)
    cursor = play_start
    for time, duration in items:
        if time > horizon:
            break
        if time <= play_start:
            buffer += duration
            continue
        elapsed = time - cursor
        underrun += max(Fraction(0), elapsed - buffer)
        buffer = max(Fraction(0), buffer - elapsed) + duration
        cursor = time
    elapsed = horizon - cursor
    underrun += max(Fraction(0), elapsed - buffer)
    buffer = max(Fraction(0), buffer - elapsed)
    return underrun, buffer


def record(rid, offered, decision, terminal, success=True, quality=True, slo=True):
    return dict(id=rid, offered=offered, decision=decision, terminal=terminal,
                success=success, quality=quality, slo=slo)


def main():
    print('K-WORK-001 CPU rational reference models; no service, network, audio device or GPU measurement')
    controls = []

    def negative(action, label):
        expect_error(action, label)
        controls.append(label)

    token_cases = 0
    partition_cases = 0
    for count in range(6):
        for stamps_tuple in combinations_with_replacement(range(1, 5), count):
            stamps = list(stamps_tuple)
            trace = make_trace(stamps)
            check(request_metrics(trace), reference_report(trace))
            token_cases += 1
            for sizes in partitions(count):
                chunk_trace = make_trace(stamps, sizes)
                check(request_metrics(chunk_trace, mode='chunk'), reference_report(chunk_trace, mode='chunk'))
                envelopes = []
                cursor = 0
                for size in sizes:
                    group_times = stamps[cursor:cursor + size]
                    cursor += size
                    envelopes.append(dict(kind='token_trace', t=group_times[-1], tokens=size, token_times=group_times))
                final = stamps[-1] + 1 if stamps else 1
                check(from_token_envelopes(0, envelopes, final), request_metrics(trace))
                partition_cases += 1

    old = [dict(kind='token_trace', t=10, tokens=2, token_times=[10, 20])]
    fixed = [dict(kind='token_trace', t=20, tokens=2, token_times=[10, 20])]
    negative(lambda: from_token_envelopes(0, old, 20), 'invalid input: envelope ends before its last token')
    negative(lambda: from_token_envelopes(0, fixed, 10), 'invalid input: final before last token')
    negative(lambda: from_token_envelopes(0, fixed, None), 'invalid input: missing final')
    repaired = from_token_envelopes(0, fixed, 20)
    check((repaired['ttft'], repaired['tpot'], repaired['latency']), (10, 10, 20))

    trace = make_trace([10])
    correct = reference_report(trace)
    check(request_metrics(trace), correct)
    negative(lambda: check(dict(correct, tpot=Fraction(0)), correct), 'report sensitivity: single-token TPOT zero')
    trace = make_trace([10, 20], [2])
    correct = reference_report(trace, 'chunk')
    check(request_metrics(trace, mode='chunk'), correct)
    negative(lambda: check(dict(correct, tpot=Fraction(0)), correct), 'report sensitivity: chunk inferred exact zero')
    trace = make_trace([10], terminal=None)
    correct = reference_report(trace)
    check(request_metrics(trace), correct)
    negative(lambda: check(dict(correct, end=Fraction(10), status='final', latency=Fraction(10)), correct),
             'report sensitivity: fabricated terminal')
    trace = make_trace([10, 20])
    trace.insert(1, event('old', 0, 'content', 2, attempt=0, start=0, stop=1))
    correct = reference_report(trace)
    check(request_metrics(trace), correct)
    negative(lambda: check(dict(correct, ttft=Fraction(2)), correct), 'report sensitivity: foreign attempt TTFT')

    duplicate = make_trace([10])
    duplicate.insert(2, dict(duplicate[1]))
    check(request_metrics(duplicate), reference_report(duplicate))
    shifted = make_trace([-90, -80], send=-100)
    check((request_metrics(shifted)['ttft'], request_metrics(shifted)['tpot']), (10, 10))
    metadata = make_trace([10, 20])
    metadata.insert(1, event('meta', 1, 'meta', 5))
    for i, e in enumerate(metadata):
        e['seq'] = i
    check(request_metrics(metadata), reference_report(metadata))
    check(request_metrics(make_trace([10, 10]))['tpot'], 0)
    for terminal in ('cancel', 'timeout', 'error'):
        trace = make_trace([10, 20], terminal=terminal)
        check(request_metrics(trace), reference_report(trace))
        check(request_metrics(trace)['latency'], None)

    invalid = []
    trace = make_trace([10])
    trace.insert(2, dict(trace[1], t=11))
    invalid.append(trace)
    trace = make_trace([10, 20])
    trace[2]['start'] = 0
    invalid.append(trace)
    trace = make_trace([10, 20])
    trace[2]['t'] = 9
    invalid.append(trace)
    trace = make_trace([10])
    trace.append(event('late', 3, 'content', 12, start=1, stop=2))
    invalid.append(trace)
    trace = make_trace([10])
    trace[1]['domain'] = 'device'
    invalid.append(trace)
    for trace in invalid:
        expect_error(lambda trace=trace: request_metrics(trace), 'invalid selected trace')

    quantile_cases = 0
    for count in range(6):
        for values in combinations_with_replacement(range(1, 5), count):
            for numerator in range(1, 11):
                p = Fraction(numerator, 10)
                validate_quantile(nearest_rank(values, p), values, p)
                quantile_cases += 1
    values = [1, 2, 3, 100]
    validate_quantile(nearest_rank(values, Fraction(99, 100)), values, Fraction(99, 100))
    negative(lambda: validate_quantile(Fraction(sum(values), len(values)), values, Fraction(99, 100)),
             'report sensitivity: mean as p99')
    expect_error(lambda: nearest_rank([1], 0), 'invalid quantile probability')

    reports = [request_metrics(make_trace(times)) for times in ([10, 20], list(range(10, 111, 10)), [10, 40])]
    expected_aggregate = dict(macro=Fraction(50, 3), pooled=Fraction(35, 3), defined_requests=3, intervals=12)
    check(aggregate(reports), expected_aggregate)
    negative(lambda: check(dict(expected_aggregate, macro=expected_aggregate['pooled']), expected_aggregate),
             'report sensitivity: macro/pooled swap')
    check(aggregate([]), dict(macro=None, pooled=None, defined_requests=0, intervals=0))
    print('aggregation: macro=50/3 ms, pooled=35/3 ms; distinct populations')

    records = [
        record('a', 0, (0, True), (5, 'final')),
        record('b', 1, (1, True), (11, 'final')),
        record('c', 2, (2, False), None),
        record('d', 3, (3, True), (9, 'final'), quality=False),
        record('e', 9, (9, True), (10, 'final')),
        record('outside', 10, (10, True), (10, 'final')),
        record('f', 4, (4, True), (8, 'cancel')),
        record('g', 5, (5, True), (9, 'error')),
        record('h', 6, (6, True), (10, 'timeout')),
        record('i', 7, (12, True), (13, 'final')),
        record('j', 8, (8, True), (9, 'final'), slo=False),
    ]
    expected_counts = dict(offered=10, pending=1, rejected=1, preterminated=0,
                           pre_cancel=0, pre_timeout=0, pre_error=0, admitted=8,
                           finished=7, incomplete=1, final=4, cancel=1,
                           timeout=1, error=1, good=2, late_decision=0)
    expected_window = dict(counts=expected_counts, goodput_elapsed=Fraction(1, 5),
                           offered_rate=Fraction(1), qualified_fraction=Fraction(1, 5))
    check(cohort_report(records, 0, 10, 10), expected_window)
    check(cohort_reference(records, 0, 10, 10), expected_window)
    negative(lambda: check(faulty_completed_denominator(records, 0, 10, 10), expected_window),
             'algorithm mutant: completed-count denominator')
    omit_slo = [dict(rec, slo=True) for rec in records]
    negative(lambda: check(cohort_report(omit_slo, 0, 10, 10), expected_window),
             'input-policy mutant: SLO omitted')
    drained = cohort_report(records, 0, 10, 13)
    check(drained, cohort_reference(records, 0, 10, 13))
    check(drained['counts']['good'], 4)
    check(drained['goodput_elapsed'], Fraction(4, 13))
    empty_window = cohort_report([], 0, 10, 10)
    check((empty_window['goodput_elapsed'], empty_window['qualified_fraction']), (0, None))
    expect_error(lambda: cohort_report(records + [records[0]], 0, 10, 10), 'duplicate request')
    expect_error(lambda: cohort_report(records, 0, 10, 9), 'cutoff before cohort closed')

    # Exact reviewer trajectory, with an independently written expected ledger.
    x = record('x', 0, None, (2, 'cancel'), success=False, quality=False, slo=False)
    x_counts = dict(offered=1, pending=0, rejected=0, preterminated=1,
                    pre_cancel=1, pre_timeout=0, pre_error=0, admitted=0,
                    finished=0, incomplete=0, final=0, cancel=0,
                    timeout=0, error=0, good=0, late_decision=0)
    x_expected = dict(counts=x_counts, goodput_elapsed=Fraction(0),
                      offered_rate=Fraction(1, 10), qualified_fraction=Fraction(0))
    check(cohort_report([x], 0, 10, 10), x_expected)
    check(cohort_reference([x], 0, 10, 10), x_expected)
    for reason in ('cancel', 'timeout', 'error'):
        rec = dict(x, terminal=(2, reason))
        expected = dict(x_counts, pre_cancel=0)
        expected['pre_' + reason] = 1
        check(cohort_report([rec], 0, 10, 10)['counts'], expected)
        trace = [event('send', 0, 'send', 0), event('end', 1, reason, 2)]
        boundary_report = request_metrics(trace)
        check((boundary_report['status'], boundary_report['end'], boundary_report['latency']), (reason, 2, None))
        check(cohort_report([rec], 0, 10, 10), cohort_reference([rec], 0, 10, 10))
    negative(lambda: check(cohort_report([dict(x, terminal=None)], 0, 10, 10), x_expected),
             'input mutant: discard preadmission terminal, producing pending')
    check(cohort_report([x], 0, 1, 1)['counts']['pending'], 1)

    for accepted in (True, False):
        late = dict(x, decision=(5, accepted))
        for cutoff, late_count in ((4, 0), (10, 1)):
            actual = cohort_report([late], 0, 2, cutoff)
            check(actual, cohort_reference([late], 0, 2, cutoff))
            check(actual['counts'], dict(x_counts, late_decision=late_count))
    late_accept = dict(x, decision=(5, True))
    late_expected = dict(x_expected, counts=dict(x_counts, late_decision=1))
    negative(lambda: check(cohort_report([dict(late_accept, terminal=None)], 0, 10, 10), late_expected),
             'input mutant: erase terminal and accept late admission')
    tie_accept = cohort_report([dict(x, decision=(2, True))], 0, 10, 10)['counts']
    check((tie_accept['admitted'], tie_accept['cancel'], tie_accept['preterminated']), (1, 1, 0))
    tie_reject = cohort_report([dict(x, decision=(2, False))], 0, 10, 10)['counts']
    check((tie_reject['rejected'], tie_reject['preterminated'], tie_reject['finished']), (1, 0, 0))
    reject_then_cancel = cohort_report([dict(x, decision=(1, False))], 0, 10, 10)['counts']
    check((reject_then_cancel['rejected'], reject_then_cancel['preterminated']), (1, 0))
    for bad in (
        record('bad', 0, None, (2, 'final')),
        record('bad', 0, (1, False), (2, 'final')),
        record('bad', 0, (3, True), (2, 'final')),
        record('bad', 0, None, (-1, 'cancel')),
    ):
        expect_error(lambda bad=bad: cohort_report([bad], 0, 10, 10), 'invalid complete business ledger')

    # Exhaustive finite boundary-order grid, independently replayed.
    decisions = [None] + [(time, accepted) for time in (0, 2, 4, 6) for accepted in (True, False)]
    terminals = [None] + [(time, reason) for time in range(7) for reason in TERMINALS]
    ledger_cases = 0
    invalid_final_cases = 0
    for decision in decisions:
        for terminal in terminals:
            rec = record('grid', 0, decision, terminal)
            legal_final = (terminal is None or terminal[1] != 'final'
                           or (decision is not None and decision[1] and decision[0] <= terminal[0]))
            if not legal_final:
                expect_error(lambda rec=rec: cohort_report([rec], 0, 2, 6), 'normal final needs effective admission')
                invalid_final_cases += 1
                continue
            for cutoff in (2, 4, 6):
                check(cohort_report([rec], 0, 2, cutoff), cohort_reference([rec], 0, 2, cutoff))
                ledger_cases += 1
    print('preadmission cancel hand ledger:', x_counts)
    print('ledger cutoff cases:', ledger_cases, '; invalid normal-final cases rejected:', invalid_final_cases)

    clock_cases = 0
    for a in range(4):
        for b in range(4):
            for low in range(-2, 3):
                for high in range(low, 3):
                    feasible = [b - a - delta for delta in range(low, high + 1) if b - a - delta >= 0]
                    reference = (min(feasible), max(feasible)) if feasible else None
                    check(clock_interval(a, b, low, high), reference)
                    clock_cases += 1
    check(clock_interval(10, 15, -2, 4), (1, 7))
    check(clock_interval(10, 15, 4, 8), (0, 1))
    negative(lambda: check((5, 5), clock_interval(10, 15, 4, 8)), 'report sensitivity: clock point outside bounds')
    check(clock_interval(10, 8, -1, 1), None)

    audio_cases = 0
    for durations in product(range(3), repeat=4):
        arrivals = [(Fraction(i, 10), Fraction(d, 10)) for i, d in enumerate(durations)]
        buffer_units = 0
        missing_units = 0
        for step in range(5):
            if step < len(durations):
                buffer_units += durations[step]
            if buffer_units > 0:
                buffer_units -= 1
            else:
                missing_units += 1
        reference = (Fraction(missing_units, 10), Fraction(buffer_units, 10))
        check(audio_buffer(arrivals, 0, Fraction(1, 2)), reference)
        audio_cases += 1
    check(audio_buffer([(0, Fraction(1, 5)), (Fraction(3, 10), Fraction(1, 5))], 0, Fraction(1, 2)), (Fraction(1, 10), 0))
    check(audio_buffer([(0, Fraction(2, 5)), (Fraction(3, 10), Fraction(1, 5))], 0, Fraction(1, 2)), (0, Fraction(1, 10)))
    check(audio_buffer([(-1, Fraction(1, 5))], 0, Fraction(1, 5)), (0, 0))
    check(audio_buffer([(1, Fraction(1))], 0, Fraction(1, 5)), (Fraction(1, 5), 0))

    # Each faulty output is produced from the SAME case as the correct output.
    # Both go through validate_pcm_output, backed by packed bytes, not constants.
    pcm_cases = [
        (dict(duration=Fraction(1, 5), rate=24000, channels=1, bytes_per_sample=2, codec_tokens=400), 4800, 'h'),
        (dict(duration=Fraction(2, 5), rate=24000, channels=1, bytes_per_sample=2, codec_tokens=800), 9600, 'h'),
        (dict(duration=Fraction(1, 100), rate=16000, channels=2, bytes_per_sample=4, codec_tokens=20), 160, 'i'),
    ]
    for index, (case, frames, sample_format) in enumerate(pcm_cases):
        proper = pcm_from_input(case)
        faulty = faulty_codec_tokens_as_pcm_bytes(case)
        reference = pcm_reference(case, frames, sample_format)
        validate_pcm_output(proper, case, frames, sample_format)
        negative(lambda faulty=faulty, case=case, frames=frames, sample_format=sample_format:
                 validate_pcm_output(faulty, case, frames, sample_format),
                 'algorithm mutant: codec tokens as PCM bytes, case ' + str(index))
        print('PCM case', index, ': correct=', proper, '; packed-byte reference=', reference, '; codec-as-byte mutant=', faulty)
    check(pcm_bytes(0, 24000), 0)
    expect_error(lambda: pcm_bytes(Fraction(1, 7), 24000), 'fractional PCM sample')
    expect_error(lambda: pcm_bytes(Fraction(1, 5), 24000, channels=0), 'invalid PCM channels')

    print('token cases:', token_cases, '; grouping cases:', partition_cases)
    print('quantile cases:', quantile_cases, '; clock cases:', clock_cases, '; audio cases:', audio_cases)
    print('invalid selected traces rejected:', len(invalid))
    print('detected probes (labels distinguish invalid inputs, report sensitivity and executed mutants):', len(controls))
    for label in controls:
        print('detected:', label)
    print('Finite CPU models verified; acquisition coverage, real service and hardware remain unmeasured')


if __name__ == '__main__':
    main()
