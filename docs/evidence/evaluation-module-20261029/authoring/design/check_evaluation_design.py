#!/usr/bin/env python3
"""Finite, read-only CPU models for the evaluation-design chapter.

No device, network, workload generator or statistical coverage is measured.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import permutations
from math import lcm
import json


class Unsupported(ValueError):
    pass


class Checks:
    def __init__(self):
        self.names = []
        self.negatives = []

    def equal(self, name, actual, expected):
        if actual != expected:
            raise AssertionError(f'{name}: {actual!r} != {expected!r}')
        self.names.append(name)

    def true(self, name, condition):
        self.equal(name, bool(condition), True)

    def raises(self, name, fn, error=ValueError):
        try:
            fn()
        except error:
            self.names.append(name)
        else:
            raise AssertionError(f'{name}: input was incorrectly accepted')

    def negative(self, name, actual, correct):
        if actual == correct:
            raise AssertionError(f'{name}: mutant escaped the reference')
        self.names.append(name)
        self.negatives.append(name)


def average(values):
    values = list(values)
    if not values:
        raise Unsupported('empty mean')
    return sum(values, F(0)) / len(values)


CELLS = {(0, 1), (1, 1), (0, 8), (1, 8)}


def factor_estimates(rows):
    """Each row is one run summary: (block, G, C, milliseconds)."""
    blocks = defaultdict(dict)
    for block, g, c, value in rows:
        if (g, c) not in CELLS or (g, c) in blocks[block]:
            raise ValueError('unknown or duplicate factor cell')
        value = F(value)
        if value < 0:
            raise ValueError('negative duration')
        blocks[block][g, c] = value
    if not blocks or any(set(cells) != CELLS for cells in blocks.values()):
        raise Unsupported('incomplete factor block; do not delete its failed arm')
    contrasts = []
    for block in sorted(blocks):
        cells = blocks[block]
        low = cells[1, 1] - cells[0, 1]
        high = cells[1, 8] - cells[0, 8]
        contrasts.append((low, high, high - low))
    return {
        'blocks': len(blocks),
        'cell_runs': len(rows),
        'contrasts': contrasts,
        'mean_contrasts': tuple(average(v[k] for v in contrasts) for k in range(3)),
    }


def direct_signed_contrast(rows):
    """Reference reduces raw records with fixed signed coefficients."""
    answer = []
    for block in sorted({row[0] for row in rows}):
        data = [r for r in rows if r[0] == block]
        low = sum(F(v) * (1 if g else -1) for _, g, c, v in data if c == 1)
        high = sum(F(v) * (1 if g else -1) for _, g, c, v in data if c == 8)
        interaction = sum(F(v) * (1 if (g == 1) == (c == 8) else -1)
                          for _, g, c, v in data)
        answer.append((low, high, interaction))
    return answer


def standardized_distribution(rows, target_weights):
    weights = {stratum: F(weight) for stratum, weight in target_weights.items()}
    if not weights or any(w < 0 for w in weights.values()) or sum(weights.values()) != 1:
        raise ValueError('target weights must be nonnegative and sum to one')
    strata = defaultdict(list)
    for stratum, value in rows:
        strata[stratum].append(F(value))
    mass = defaultdict(F)
    for stratum, weight in weights.items():
        if weight == 0:
            continue
        values = strata[stratum]
        if not values:
            raise Unsupported('a positive-weight target stratum has no observations')
        for value in values:
            mass[value] += weight / len(values)
    return dict(mass)


def weighted_mean(mass):
    return sum(value * weight for value, weight in mass.items())


def inverse_cdf(mass, probability):
    probability = F(probability)
    if not 0 < probability <= 1 or not mass or sum(mass.values()) != 1:
        raise ValueError('invalid CDF or probability')
    cumulative = F(0)
    for value, weight in sorted(mass.items()):
        cumulative += weight
        if cumulative >= probability:
            return value
    raise AssertionError('normalized CDF did not reach one')


def equal_weight_reference(rows, weights):
    """For these small rational fixtures, construct literal equiprobable copies.

    This reference does not consume the production weighted distribution.
    """
    labels = [s for s, w in weights.items() if F(w) > 0]
    values = {s: [F(v) for label, v in rows if label == s] for s in labels}
    if any(not values[s] for s in labels):
        raise Unsupported('empty positive-weight stratum')
    denominator = lcm(*(F(weights[s]).denominator * len(values[s]) for s in labels))
    expanded = []
    for s in labels:
        w = F(weights[s])
        copies = w.numerator * (denominator // (w.denominator * len(values[s])))
        for value in values[s]:
            expanded.extend([value] * copies)
    return expanded


def nearest_rank(values, probability):
    values = sorted(values)
    probability = F(probability)
    if not values or not 0 < probability <= 1:
        raise ValueError('invalid quantile')
    rank = (probability.numerator * len(values) + probability.denominator - 1) // probability.denominator
    return values[rank - 1]


def fcfs_responses(service):
    finish = F(0)
    result = []
    for cost in service:
        finish += F(cost)
        result.append(finish)  # all jobs in this finite reference arrive at zero
    return result


@dataclass(frozen=True)
class Request:
    rid: int
    send: int
    first: int
    gaps: tuple[int, int, int]
    final: int
    tokens: tuple[int, int, int, int] = (1, 2, 3, 4)


REFERENCE_TOKENS = (1, 2, 3, 4)


def service_fixture(arm):
    result = []
    for rid in range(10):
        send = 200 + 900 * rid
        if arm == 'A':
            request = Request(rid, send, 200, (40, 40, 40), 1900 if rid == 9 else 340)
        elif arm == 'B':
            if rid == 8:
                request = Request(rid, send, 100, (10, 130, 10), 280)
            elif rid == 9:
                request = Request(rid, send, 350, (40, 40, 40), 490)
            else:
                request = Request(rid, send, 100, (30, 30, 30), 220,
                                  (1, 2, 0, 4) if rid == 7 else REFERENCE_TOKENS)
        else:
            raise ValueError('unknown arm')
        result.append(request)
    return result


def endpoint_summary(requests, cutoff=10000, maximum_gap=None, ignore_quality=False):
    if cutoff <= 0 or len({r.rid for r in requests}) != len(requests):
        raise ValueError('invalid cutoff or duplicate request')
    selected = [r for r in requests if 0 <= r.send < cutoff]
    normal, timeout, good = [], [], []
    for r in selected:
        if r.first < 0 or min(r.gaps) < 0 or r.final < r.first + sum(r.gaps):
            raise ValueError('invalid token or final order')
        # The fixture explicitly gives normal completion priority at an exact deadline.
        if r.final <= 1000 and r.send + r.final <= cutoff:
            normal.append(r)
            tpot = F(sum(r.gaps), 3)
            quality_known = r.send + r.final + 1 <= cutoff
            if (quality_known and (ignore_quality or r.tokens == REFERENCE_TOKENS)
                    and r.first <= 300 and tpot <= 50
                    and (maximum_gap is None or max(r.gaps) <= maximum_gap)):
                good.append(r)
        elif r.send + 1000 <= cutoff:
            timeout.append(r)
    return {
        'offered': len(selected), 'normal': len(normal), 'timeout': len(timeout),
        'pending': len(selected) - len(normal) - len(timeout), 'good': len(good),
        'goodput': F(1000 * len(good), cutoff),
        'normal_ttft_mean': average(r.first for r in normal) if normal else None,
        'normal_tpot_mean': average(F(sum(r.gaps), 3) for r in normal) if normal else None,
    }


def event_reference(requests, cutoff=10000, maximum_gap=None):
    """Independent reference follows individual events and an absorbing business state."""
    events = []
    for r in requests:
        if not 0 <= r.send < cutoff:
            continue
        events.append((r.send + 1, 0, r.rid, 'admit', None))
        stamp = r.send + r.first
        for index, value in enumerate(r.tokens):
            if index:
                stamp += r.gaps[index - 1]
            events.append((stamp, 1, r.rid, 'content', (index, value)))
        events.append((r.send + r.final, 2, r.rid, 'final', None))
        events.append((r.send + 1000, 3, r.rid, 'timeout', None))
        events.append((r.send + r.final + 1, 4, r.rid, 'quality', r.tokens == REFERENCE_TOKENS))
    selected = {r.rid: r for r in requests if 0 <= r.send < cutoff}
    state = {rid: {'state': 'pending', 'content': [], 'quality': None, 'finish': None} for rid in selected}
    for stamp, _, rid, kind, payload in sorted(events):
        if stamp > cutoff:
            break
        record = state[rid]
        if kind == 'admit' and record['state'] == 'pending':
            record['state'] = 'running'
        elif kind == 'content' and record['state'] == 'running':
            index, token = payload
            if index != len(record['content']):
                raise AssertionError('event content lost its position')
            record['content'].append((stamp, token))
        elif kind == 'final' and record['state'] == 'running':
            record['state'] = 'final'
            record['finish'] = stamp
        elif kind == 'timeout' and record['state'] in {'pending', 'running'}:
            record['state'] = 'timeout'
        elif kind == 'quality' and record['state'] == 'final':
            record['quality'] = payload
    normals = [rid for rid, rec in state.items() if rec['state'] == 'final']
    ttfts, tpots = [], []
    good = 0
    for rid in normals:
        rec = state[rid]
        times = [t for t, _ in rec['content']]
        ttft = times[0] - selected[rid].send
        gaps = [b - a for a, b in zip(times, times[1:])]
        tpot = F(times[-1] - times[0], len(times) - 1)
        ttfts.append(ttft)
        tpots.append(tpot)
        if (rec['quality'] is True and tuple(t for _, t in rec['content']) == REFERENCE_TOKENS
                and ttft <= 300 and tpot <= 50
                and rec['finish'] - selected[rid].send <= 1000
                and (maximum_gap is None or all(g <= maximum_gap for g in gaps))):
            good += 1
    timeouts = sum(rec['state'] == 'timeout' for rec in state.values())
    return {
        'offered': len(selected), 'normal': len(normals), 'timeout': timeouts,
        'pending': len(selected) - len(normals) - timeouts, 'good': good,
        'goodput': F(1000 * good, cutoff),
        'normal_ttft_mean': average(ttfts) if ttfts else None,
        'normal_tpot_mean': average(tpots) if tpots else None,
    }


def require_target(measured, declared):
    """Finite identity guard; it is not a general causal-validity checker."""
    if measured != declared:
        raise Unsupported('measured population, boundary or estimand differs from the frozen target')
    return True


def main():
    checks = Checks()
    base = [(0, 0, 1, 10), (0, 1, 1, 8), (0, 0, 8, 12), (0, 1, 8, 13)]
    for index, order in enumerate(permutations(base)):
        result = factor_estimates(list(order))
        checks.equal(f'factor permutation {index}', result['contrasts'], direct_signed_contrast(order))
        checks.equal(f'factor signed reference {index}', result['mean_contrasts'], (F(-2), F(1), F(3)))
    three_blocks = [(b, g, c, value + shift) for b, shift in enumerate((0, 5, -3))
                    for _, g, c, value in base]
    result = factor_estimates(three_blocks)
    checks.equal('common block shifts preserve contrasts', result['contrasts'], [(F(-2), F(1), F(3))] * 3)
    checks.equal('true repeated unit count', result['blocks'], 3)
    checks.equal('record count is separate', result['cell_runs'], 12)
    checks.negative('mutant counts arms as independent interaction repeats', result['cell_runs'], result['blocks'])
    checks.raises('missing factor cell', lambda: factor_estimates(base[:-1]), Unsupported)
    checks.raises('duplicate cell cannot fill a missing cell', lambda: factor_estimates(base[:-1] + [base[0]]))
    checks.raises('two diagonal cells cannot identify effects', lambda: factor_estimates([base[0], base[3]]), Unsupported)
    checks.raises('empty factor plan', lambda: factor_estimates([]), Unsupported)
    checks.negative('mutant additive prediction omits interaction', F(10 + (8 - 10) + (12 - 10)), F(13))
    low, high, _ = result['mean_contrasts']
    checks.equal('equal-condition average effect', (low + high) / 2, F(-1, 2))
    checks.equal('predeclared skewed-condition effect', F(1, 4) * low + F(3, 4) * high, F(1, 4))

    a = [('S', 1)] + [('L', 9)] * 9
    b = [('S', 2)] * 9 + [('L', 10)]
    checks.equal('A pooled descriptive mean', average(v for _, v in a), F(41, 5))
    checks.equal('B pooled descriptive mean', average(v for _, v in b), F(14, 5))
    for weights in ({'S': F(1, 2), 'L': F(1, 2)}, {'S': F(1, 4), 'L': F(3, 4)}):
        for label, rows in [('A', a), ('B', b)]:
            mass = standardized_distribution(rows, weights)
            expanded = equal_weight_reference(rows, weights)
            checks.equal(f'{label} literal-copy mean {weights}', weighted_mean(mass), average(expanded))
            for probability in (F(1, 2), F(19, 20), F(1)):
                checks.equal(f'{label} mixture CDF {weights} {probability}', inverse_cdf(mass, probability), nearest_rank(expanded, probability))
    half = {'S': F(1, 2), 'L': F(1, 2)}
    checks.equal('common population means', (weighted_mean(standardized_distribution(a, half)), weighted_mean(standardized_distribution(b, half))), (F(5), F(6)))
    checks.negative('mutant selects using each arm own sample mix', average(v for _, v in a) < average(v for _, v in b), True)
    checks.negative('mutant averages stratum p95s', F(1, 2) * 1 + F(1, 2) * 9, inverse_cdf(standardized_distribution(a, half), F(19, 20)))
    checks.raises('unsupported positive-weight stratum', lambda: standardized_distribution([('S', 1)], half), Unsupported)
    checks.raises('invalid target weights', lambda: standardized_distribution(a, {'S': -1, 'L': 2}))
    checks.raises('weights do not normalize', lambda: standardized_distribution(a, {'S': F(1, 4), 'L': F(1, 4)}))
    checks.equal('empty zero-weight stratum is harmless', weighted_mean(standardized_distribution([('S', 2)], {'S': 1, 'L': 0})), F(2))
    checks.equal('short-first shared queue', fcfs_responses([1, 9]), [F(1), F(10)])
    checks.equal('long-first shared queue', fcfs_responses([9, 1]), [F(9), F(10)])
    checks.negative('mutant predicts short-first queue by reweighting isolated work', F(5), average(fcfs_responses([1, 9])))
    checks.negative('mutant predicts long-first queue by reweighting isolated work', F(5), average(fcfs_responses([9, 1])))

    outcomes = {}
    for arm in ('A', 'B'):
        requests = service_fixture(arm)
        for cap in (None, 80):
            actual = endpoint_summary(requests, maximum_gap=cap)
            reference = event_reference(requests, maximum_gap=cap)
            checks.equal(f'{arm} complete event-reference cap={cap}', actual, reference)
            checks.equal(f'{arm} offered partition cap={cap}', actual['offered'], actual['normal'] + actual['timeout'] + actual['pending'])
        outcomes[arm] = endpoint_summary(requests)
    checks.equal('service primary goodput', (outcomes['A']['goodput'], outcomes['B']['goodput']), (F(9, 10), F(4, 5)))
    checks.equal('service extended goodput', endpoint_summary(service_fixture('B'), maximum_gap=80)['goodput'], F(7, 10))
    checks.equal('conditional TTFT means', (outcomes['A']['normal_ttft_mean'], outcomes['B']['normal_ttft_mean']), (F(200), F(125)))
    checks.equal('conditional TPOT means', (outcomes['A']['normal_tpot_mean'], outcomes['B']['normal_tpot_mean']), (F(40), F(33)))
    checks.equal('finite-window offered rate', F(10, 10), F(1))
    checks.negative('mutant reciprocal-gap offered rate', F(1000, 900), F(1))
    checks.negative('mutant discards quality', endpoint_summary(service_fixture('B'), ignore_quality=True)['good'], outcomes['B']['good'])
    checks.negative('mutant substitutes TPOT for maximum gap', outcomes['B']['good'], event_reference(service_fixture('B'), maximum_gap=80)['good'])
    late_normal = sum(r.send + r.final <= 10500 for r in service_fixture('A'))
    checks.negative('mutant revives timeout during drain', late_normal, outcomes['A']['normal'])
    latency_winner = min(outcomes, key=lambda arm: outcomes[arm]['normal_ttft_mean'])
    goodput_winner = max(outcomes, key=lambda arm: outcomes[arm]['goodput'])
    checks.negative('mutant substitutes conditional latency for declared goodput', latency_winner, goodput_winner)
    checks.raises('duplicate request identity', lambda: endpoint_summary(service_fixture('A') + [service_fixture('A')[0]]))
    checks.equal('empty cohort has zero count rate', endpoint_summary([])['goodput'], F(0))
    checks.equal('empty conditional mean is undefined', endpoint_summary([])['normal_ttft_mean'], None)

    kernel_target = ('fixed_shape', 'device_complete', 'mean_duration_ms')
    client_target = ('offered_trace', 'client_first_content', 'mean_TTFT_ms')
    offline_target = ('fixed_ready_bag', 'all_required_completed', 'tasks_per_second')
    capacity_target = ('independent_arrivals', 'stable_joint_SLO', 'capacity_requests_per_second')
    checks.true('same frozen estimand identity', require_target(client_target, client_target))
    checks.raises('microbenchmark cannot replace client target', lambda: require_target(kernel_target, client_target), Unsupported)
    checks.raises('offline bag cannot replace stable service capacity', lambda: require_target(offline_target, capacity_target), Unsupported)
    worlds = [(0, 10, 0), (30, 10, 5)]  # host wait, device work, delivery delay
    device_durations = [world[1] for world in worlds]
    client_first = [sum(world) for world in worlds]
    checks.equal('device observation is identical in two worlds', device_durations, [10, 10])
    checks.equal('client observations differ in those worlds', client_first, [10, 45])
    checks.negative('mutant infers TTFT from device observation alone', device_durations, client_first)

    print(json.dumps({'result': 'PASS', 'checks': len(checks.names),
                      'negative_controls_detected': len(checks.negatives),
                      'negative_controls': checks.negatives,
                      'scope': 'Exact arithmetic, finite designs and synthetic event traces only; no device execution, service measurement or statistical coverage claim.'}, indent=2))


if __name__ == '__main__':
    main()
