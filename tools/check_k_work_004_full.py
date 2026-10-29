#!/usr/bin/env python3
'''K-WORK-004: deterministic, offline, read-only finite-model checks.

Only Python standard library. No files, processes, network, project imports,
or device measurements. A supplied t critical value is an arithmetic input,
not a certificate of normality or confidence-interval validity.
'''

from fractions import Fraction as F
from itertools import product
from math import comb, isfinite, sqrt


FIELDS = (
    'task_contract', 'environment', 'workload', 'state_policy',
    'event_boundary', 'aggregation', 'instrumentation', 'repetition',
)
MILESTONES = ('start', 'loaded', 'ready', 'send', 'first', 'final', 'physical')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_error(function, expected, label):
    try:
        function()
    except expected:
        return
    except Exception as error:
        raise AssertionError(label + ': wrong error ' + type(error).__name__) from error
    raise AssertionError(label + ': no error')


def comparability(a, b):
    problems = []
    for field in FIELDS:
        if field not in a or field not in b:
            problems.append('missing:' + field)
        elif a[field] != b[field]:
            problems.append('mismatch:' + field)
    return problems


def faulty_comparability(a, b):
    # Deliberately ignores the event boundary.
    return all(a.get(k) == b.get(k) for k in FIELDS if k != 'event_boundary')


def warmup_ready(required_counts, completed_counts, pending):
    if any(k < 1 for k in required_counts.values()):
        raise ValueError('required counts must be positive')
    if any(k < 0 for k in completed_counts.values()):
        raise ValueError('completed counts must be nonnegative')
    return not pending and all(
        completed_counts.get(p, 0) >= count
        for p, count in required_counts.items()
    )


def faulty_first_warm_only(required_counts, completed_counts, pending):
    # Deliberately ignores missing buckets and pending preparation.
    return any(count > 0 for count in completed_counts.values())


def cold_metrics(events):
    if set(events) != set(MILESTONES):
        raise ValueError('missing or extra milestone')
    values = [events[k] for k in MILESTONES]
    if not all(isfinite(x) for x in values):
        raise ValueError('nonfinite timestamp')
    if any(a > b for a, b in zip(values, values[1:])):
        raise ValueError('milestones out of order')
    start, loaded, ready, send, first, final, physical = values
    return {
        'full_cold': final - start,
        'loaded_cold': final - loaded,
        'startup_ready': ready - start,
        'request': final - send,
        'ttft': first - send,
        'cold_first': first - start,
        'post_final_drain': physical - final,
    }


def faulty_request_only_cold(events):
    return events['final'] - events['send']


def mean(xs):
    if not xs:
        raise ValueError('empty sample')
    return sum((F(x) for x in xs), F(0)) / len(xs)


def ratio_of_means(a, b):
    numerator, denominator = mean(a), mean(b)
    if denominator == 0:
        raise ZeroDivisionError('zero mean denominator')
    return numerator / denominator


def mean_of_ratios(a, b):
    if len(a) != len(b) or not a:
        raise ValueError('nonempty paired samples required')
    if any(y == 0 for y in b):
        raise ZeroDivisionError('zero paired denominator')
    return mean([F(x) / F(y) for x, y in zip(a, b)])


def variance(xs):
    if len(xs) < 2:
        raise ValueError('at least two observations required')
    center = mean(xs)
    return sum(((F(x) - center) ** 2 for x in xs), F(0)) / (len(xs) - 1)


def pairwise_variance_reference(xs):
    # Independent identity: sum of pairwise squared differences / n(n-1).
    if len(xs) < 2:
        raise ValueError('at least two observations required')
    pair_sum = sum(
        ((F(xs[i]) - F(xs[j])) ** 2
         for i in range(len(xs)) for j in range(i + 1, len(xs))),
        F(0),
    )
    return pair_sum / (len(xs) * (len(xs) - 1))


def hierarchical_variance_of_mean(run_means):
    return variance(run_means) / len(run_means)


def faulty_flat_variance_of_mean(run_means, per_run):
    if not isinstance(per_run, int) or per_run < 1:
        raise ValueError('positive integer per_run required')
    flat = [x for x in run_means for _ in range(per_run)]
    return variance(flat) / len(flat)


def interval_arithmetic(xs, critical):
    # Computes endpoints only. Does not infer distributional suitability.
    if len(xs) < 2:
        raise ValueError('at least two paired differences required')
    if not isfinite(critical) or critical <= 0:
        raise ValueError('positive finite critical value required')
    if not all(isfinite(x) for x in xs):
        raise ValueError('finite differences required')
    center = mean(xs)
    half_width = float(critical) * sqrt(float(variance(xs) / len(xs)))
    return float(center) - half_width, float(center) + half_width


def main():
    detected = []

    def kill(name, distinguished):
        require(distinguished, 'negative control survived: ' + name)
        require(name not in detected, 'duplicate negative control')
        detected.append(name)

    base = {field: field + '-contract' for field in FIELDS}
    a, b = dict(base, candidate='A'), dict(base, candidate='B')
    require(comparability(a, b) == [], 'candidate identity is the treatment')
    for field in FIELDS:
        changed = dict(b)
        changed[field] = 'different'
        require(comparability(a, changed) == ['mismatch:' + field], field)
        missing = dict(b)
        del missing[field]
        require(comparability(a, missing) == ['missing:' + field], 'missing ' + field)
    changed = dict(b, event_boundary='first-visible')
    kill('ignores_event_boundary', faulty_comparability(a, changed)
         and comparability(a, changed) == ['mismatch:event_boundary'])
    print('comparability: all contract fields and missing fields checked; candidate A/B allowed')

    required = {'prefill': 1, 'decode-1': 1, 'decode-32': 2}
    partial = {'prefill': 1, 'decode-1': 1}
    complete = {'prefill': 1, 'decode-1': 1, 'decode-32': 2}
    require(not warmup_ready(required, partial, set()), 'missing path accepted')
    require(not warmup_ready(required, dict(complete, **{'decode-32': 1}), set()),
            'insufficient completed count accepted')
    require(warmup_ready(required, complete, set()), 'complete warmup rejected')
    require(not warmup_ready(required, complete, {'capture-inflight'}),
            'pending preparation accepted')
    require(warmup_ready({}, {}, set()), 'empty no-work coverage identity')
    require(not warmup_ready({}, {}, {'pending'}), 'empty coverage ignored pending work')
    expect_error(lambda: warmup_ready({'p': 0}, {}, set()), ValueError, 'zero requirement')
    expect_error(lambda: warmup_ready({'p': 1}, {'p': -1}, set()), ValueError,
                 'negative completions')
    kill('first_warm_only', faulty_first_warm_only(required, partial, {'pending'})
         and not warmup_ready(required, partial, {'pending'}))
    print('warmup: path counts, missing bucket, pending preparation and no-work boundary checked')

    events = dict(zip(MILESTONES, (F(0), F(10), F(12), F(12), F(49, 4), F(13), F(14))))
    observed = cold_metrics(events)
    expected = {
        'full_cold': F(13), 'loaded_cold': F(3), 'startup_ready': F(12),
        'request': F(1), 'ttft': F(1, 4), 'cold_first': F(49, 4),
        'post_final_drain': F(1),
    }
    require(observed == expected, 'nonzero-load reference trajectory')
    kill('request_only_cold_timer', faulty_request_only_cold(events) != expected['full_cold'])
    cases = 0
    for load, prep, gap, service, drain in product((0, 1, 10), (0, 2), (0, 1), (1, 3), (0, 1)):
        # Reference comes from independent stage durations, not endpoint subtraction.
        loaded = F(7 + load)
        ready = loaded + prep
        send = ready + gap
        final = send + service
        trace = dict(zip(MILESTONES, (
            F(7), loaded, ready, send, send + F(service, 2), final, final + drain,
        )))
        got = cold_metrics(trace)
        require(got['full_cold'] == load + prep + gap + service, 'stage sum')
        require(got['loaded_cold'] == prep + gap + service, 'loaded-cold stage sum')
        require(got['request'] == service, 'request duration')
        require(got['post_final_drain'] == drain, 'drain excluded from response endpoint')
        shifted = {key: value + 101 for key, value in trace.items()}
        require(cold_metrics(shifted) == got, 'clock-origin translation invariant')
        cases += 1
    missing_final = dict(events)
    del missing_final['final']
    expect_error(lambda: cold_metrics(missing_final), ValueError, 'missing success endpoint')
    expect_error(lambda: cold_metrics(dict(events, loaded=F(15))), ValueError, 'bad chronology')
    expect_error(lambda: cold_metrics(dict(events, first=float('nan'))), ValueError, 'nan')
    print('cold_boundaries: full=13 loaded=3 request=1 startup_ready=12 drain=1; duration_cases=' + str(cases))

    rom, mor = ratio_of_means([1, 9], [2, 8]), mean_of_ratios([1, 9], [2, 8])
    require(rom == F(1) and mor == F(13, 16), 'independent rational references')
    kill('mean_of_ratios_for_ratio_of_means', mor != rom)
    expect_error(lambda: ratio_of_means([], [1]), ValueError, 'empty numerator')
    expect_error(lambda: ratio_of_means([1], []), ValueError, 'empty denominator')
    expect_error(lambda: ratio_of_means([1], [0]), ZeroDivisionError, 'zero mean denominator')
    expect_error(lambda: mean_of_ratios([1], [0]), ZeroDivisionError, 'zero paired denominator')
    expect_error(lambda: mean_of_ratios([1, 2], [1]), ValueError, 'unpaired samples')
    print('estimands: ratio_of_means=1; mean_of_ratios=13/16; empty and zero-denominator cases checked')

    for xs in ([100, 200], [1, 2, 5], [3, 3, 3], [F(1, 3), F(2, 7), F(5, 2)]):
        require(variance(xs) == pairwise_variance_reference(xs), 'pairwise variance identity')
    hierarchical = hierarchical_variance_of_mean([100, 200])
    require(hierarchical == F(2500), 'absolute hierarchical reference')
    require(faulty_flat_variance_of_mean([100, 200], 1) == hierarchical, 'n=1 boundary')
    flat = faulty_flat_variance_of_mean([100, 200], 1000)
    require(flat == F(2500, 1999), 'absolute flat reference')
    require(hierarchical_variance_of_mean([100, 100]) == 0, 'equal-run boundary')
    expect_error(lambda: hierarchical_variance_of_mean([100]), ValueError, 'single run')
    kill('flat_requests_as_independent_runs', flat < hierarchical)
    print('hierarchy: run_variance_of_mean=2500; flat_n1000=2500/1999; pairwise reference checked')

    critical = F(2131, 1000)  # Rounded 95%, df=15 table value; not a quantile implementation.
    # Eight -2+a and eight -2-a differences have exact sample variance 1 when a=sqrt(15)/4.
    # To avoid irrational inputs, validate the chapter's supplied-statistics arithmetic directly.
    half = float(critical) / sqrt(16)
    require(abs((-2 - half) - (-2.53275)) < 1e-12, 'given-statistics lower endpoint')
    require(abs((-2 + half) - (-1.46725)) < 1e-12, 'given-statistics upper endpoint')
    symmetric = [-3] * 8 + [-1] * 8
    lo, hi = interval_arithmetic(symmetric, critical)
    reference_half = float(critical) / sqrt(15)
    require(abs(lo - (-2 - reference_half)) < 1e-12, 'nonzero variance lower')
    require(abs(hi - (-2 + reference_half)) < 1e-12, 'nonzero variance upper')
    expect_error(lambda: interval_arithmetic([-2], critical), ValueError, 'single difference')
    expect_error(lambda: interval_arithmetic([-2, -1], 0), ValueError, 'zero critical')
    expect_error(lambda: interval_arithmetic([-2, float('inf')], critical), ValueError, 'infinite data')

    # Independent finite population: D=-2 with 99/100; D=98 with 1/100.
    p = F(1, 100)
    population_mean = (1 - p) * (-2) + p * 98
    population_variance = (1 - p) * 4 + p * (98 ** 2) - population_mean ** 2
    require(population_mean == -1 and population_variance == 99, 'population moments')
    n = 16
    total_mass, covered_mass = F(0), F(0)
    count_groups = 0
    for k in range(n + 1):
        mass = F(comb(n, k)) * p ** k * (1 - p) ** (n - k)
        total_mass += mass
        sample = [-2] * (n - k) + [98] * k
        lo, hi = interval_arithmetic(sample, critical)
        # Closed-form count moments, independent from the centered-data implementation.
        count_mean = F(-2) + F(100 * k, n)
        count_variance = F(10000 * k * (n - k), n * (n - 1))
        require(mean(sample) == count_mean, 'count mean reference')
        require(variance(sample) == count_variance, 'count variance reference')
        reference_covers = ((count_mean - population_mean) ** 2
                            <= critical ** 2 * count_variance / n)
        computed_covers = lo <= float(population_mean) <= hi
        require(computed_covers == reference_covers, 'count interval reference')
        if reference_covers:
            covered_mass += mass
        count_groups += 1
    require(total_mass == 1, 'binomial mass conservation')
    all_ordinary = (1 - p) ** n
    coverage_upper_bound = 1 - all_ordinary
    require(interval_arithmetic([-2] * n, critical) == (-2.0, -2.0), 'degenerate sample interval')
    require(all_ordinary > F(85, 100), 'rare-event absence probability')
    require(covered_mass <= coverage_upper_bound < F(15, 100), 'coverage bound')
    kill('iid_alone_guarantees_95pct_t_coverage', coverage_upper_bound < F(95, 100))
    print('t_arithmetic: supplied-statistics interval=[-2.53275,-1.46725]; nonzero-variance and invalid-input cases checked')
    print('rare_tail: count_groups=' + str(count_groups)
          + ' mass=1 all_ordinary_probability=' + format(float(all_ordinary), '.12f')
          + ' coverage_upper_bound=' + format(float(coverage_upper_bound), '.12f')
          + ' rounded_t_finite_model_coverage=' + format(float(covered_mass), '.12f'))
    print('negative_controls_detected=' + str(len(detected)) + ' names=' + ','.join(detected))
    print('scope=CPU-only exact algebra and finite state/coverage models; no real timing, GPU, profiler, service, distribution fitting or learner-effect measurement')


if __name__ == '__main__':
    main()
