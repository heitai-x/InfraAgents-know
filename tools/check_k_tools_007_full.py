#!/usr/bin/env python3
"""K-TOOLS-007: deterministic, offline, read-only finite-model checks."""
import itertools
import math
import random
from collections import Counter
from fractions import Fraction
from statistics import mean


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def detect(label, comparison):
    try:
        comparison()
    except AssertionError:
        print('NEGATIVE_DETECTED ' + label)
        return
    raise AssertionError('negative control escaped: ' + label)


def nearest_rank(values, p):
    p = Fraction(p)
    if not values or not 0 < p <= 1:
        raise ValueError('nonempty values and 0 < p <= 1 required')
    ordered = sorted(values)
    rank = (p.numerator * len(ordered) + p.denominator - 1) // p.denominator
    return ordered[rank - 1]


def t_summary(center, sd, n, critical):
    if n < 2 or not all(math.isfinite(x) for x in (center, sd, critical)):
        raise ValueError('invalid summary')
    if sd < 0 or critical <= 0:
        raise ValueError('invalid scale or critical value')
    half = critical * sd / math.sqrt(n)
    return center - half, center + half


def paired_difference(values, critical):
    if len(values) < 2 or not all(math.isfinite(x) for x in values):
        raise ValueError('need at least two finite differences')
    center = mean(values)
    sd = math.sqrt(sum((x - center) ** 2 for x in values) / (len(values) - 1))
    if sd == 0:
        return {'status': 'UNRESOLVED', 'point': center, 'interval': None, 'reason': 'zero_sample_sd'}
    return {'status': 'ANALYZED', 'point': center,
            'interval': t_summary(center, sd, len(values), critical), 'reason': ''}


def validate_data(data):
    if not data or any(not run for run in data):
        raise ValueError('nonempty runs required')
    for run in data:
        for pair in run:
            if len(pair) != 2 or any(not math.isfinite(x) or x < 0 for x in pair):
                raise ValueError('finite nonnegative paired observations required')


def aggregate(data, plan=None):
    if plan is None:
        plan = tuple((r, tuple(range(len(run)))) for r, run in enumerate(data))
    if len(plan) != len(data):
        raise ValueError('wrong outer draw count')
    a_runs, b_runs = [], []
    for r, indices in plan:
        if not isinstance(r, int) or not 0 <= r < len(data):
            raise ValueError('invalid run index')
        if len(indices) != len(data[r]):
            raise ValueError('wrong inner draw count')
        if any(not isinstance(j, int) or not 0 <= j < len(data[r]) for j in indices):
            raise ValueError('invalid leaf index')
        a_runs.append(mean([data[r][j][0] for j in indices]))
        b_runs.append(mean([data[r][j][1] for j in indices]))
    return mean(a_runs), mean(b_runs)


def all_plans(data):
    count = len(data)
    for runs in itertools.product(range(count), repeat=count):
        choices = [tuple(itertools.product(range(len(data[r])), repeat=len(data[r]))) for r in runs]
        for inner in itertools.product(*choices):
            yield tuple(zip(runs, inner))


def seeded_plans(data, count, seed):
    rng = random.Random(seed)
    for _ in range(count):
        plan = []
        for _ in data:
            r = rng.randrange(len(data))
            indices = tuple(rng.randrange(len(data[r])) for _ in data[r])
            plan.append((r, indices))
        yield tuple(plan)


def bootstrap_ratio(data, plans, alpha=Fraction(1, 20), floor=0.0):
    validate_data(data)
    alpha = Fraction(alpha)
    if not 0 < alpha < 1 or not math.isfinite(floor) or floor < 0:
        raise ValueError('invalid alpha or denominator floor')
    result = {'status': 'UNRESOLVED', 'point': None, 'interval': None,
              'reason': '', 'replicates': [], 'denominators': [], 'invalid': 0}
    ma, mb = aggregate(data)
    if mb <= floor:
        result['reason'] = 'point_denominator'
        return result
    point = ma / mb
    if not math.isfinite(point):
        result['reason'] = 'point_nonfinite_ratio'
        return result
    result['point'] = point
    if len(data) < 2:
        result['reason'] = 'too_few_outer_units'
        return result
    for plan in plans:
        ra, rb = aggregate(data, plan)
        result['denominators'].append(rb)
        ratio = None if rb <= floor else ra / rb
        if ratio is None or not math.isfinite(ratio):
            result['invalid'] += 1
            result['replicates'].append(None)
        else:
            result['replicates'].append(ratio)
    if result['invalid']:
        result['reason'] = 'replicate_denominator_or_nonfinite_ratio'
        return result
    samples = result['replicates']
    if len(samples) < 2:
        result['reason'] = 'too_few_replicates'
        return result
    if min(samples) == max(samples):
        result['reason'] = 'degenerate_resampling'
        return result
    result['interval'] = (nearest_rank(samples, alpha / 2), nearest_rank(samples, 1 - alpha / 2))
    result['status'] = 'ANALYZED'
    return result


def exact_distribution(data):
    # Independent reference: unordered occupancy vectors with multinomial weights.
    # It neither calls aggregate nor enumerates ordered leaf-index plans.
    options = []
    for run in data:
        n = len(run)
        local = []
        for counts in itertools.product(range(n + 1), repeat=n):
            if sum(counts) != n:
                continue
            weight = math.factorial(n)
            for k in counts:
                weight //= math.factorial(k)
            am = sum((Fraction(pair[0]) * k for pair, k in zip(run, counts)), Fraction(0)) / n
            bm = sum((Fraction(pair[1]) * k for pair, k in zip(run, counts)), Fraction(0)) / n
            local.append((am, bm, weight))
        options.append(local)
    distribution = Counter()
    R = len(data)
    for runs in itertools.product(range(R), repeat=R):
        for selected in itertools.product(*(options[r] for r in runs)):
            numerator = sum((entry[0] for entry in selected), Fraction(0))
            denominator = sum((entry[1] for entry in selected), Fraction(0))
            weight = math.prod(entry[2] for entry in selected)
            if denominator == 0:
                distribution[None] += weight
            else:
                distribution[numerator / denominator] += weight
    return distribution


def rounded_counter(values):
    return Counter(None if value is None else round(float(value), 12) for value in values)


def rounded_reference(reference):
    result = Counter()
    for value, weight in reference.items():
        result[None if value is None else round(float(value), 12)] += weight
    return result


def bad_average_ratios(data, plan):
    return mean([mean([data[r][j][0] / data[r][j][1] for j in indices])
                 for r, indices in plan])


def bad_unpaired_ratio(data, plan):
    numerator, _ = aggregate(data, plan)
    denominator = mean([mean([data[(r + 1) % len(data)][j][1] for j in indices])
                        for r, indices in plan])
    return numerator / denominator


def bad_zero_as_zero(numerator, denominator):
    return 0.0 if denominator == 0 else numerator / denominator


def check_ratio():
    data = (((2, 1), (6, 4)), ((8, 3), (12, 8)))
    plans = tuple(all_plans(data))
    result = bootstrap_ratio(data, plans)
    reference = exact_distribution(data)
    require(len(plans) == 64 and sum(reference.values()) == 64, 'enumeration count')
    require(result['status'] == 'ANALYZED' and close(result['point'], 1.75), 'ratio branch')
    require(rounded_counter(result['replicates']) == rounded_reference(reference), 'full distribution differs')
    require(close(min(result['replicates']), 1.5), 'minimum')
    require(close(max(result['replicates']), 8 / 3), 'maximum')
    for p, q in zip((Fraction(1, 40), Fraction(39, 40)), result['interval']):
        less = sum(weight for value, weight in reference.items() if float(value) < q - 1e-12)
        leq = sum(weight for value, weight in reference.items() if float(value) <= q + 1e-12)
        require(Fraction(less, 64) < p <= Fraction(leq, 64), 'CDF quantile invariant')
    wrong = [bad_average_ratios(data, plan) for plan in plans]
    detect('average_pair_ratios_instead_of_ratio_of_means',
           lambda: require(rounded_counter(wrong) == rounded_reference(reference), 'wrong estimand'))
    proportional = (((2, 1), (8, 4)), ((10, 5), (18, 9)))
    prop = bootstrap_ratio(proportional, all_plans(proportional))
    require(all(value == 2 for value in prop['replicates']), 'pair invariant')
    require(prop['status'] == 'UNRESOLVED' and prop['reason'] == 'degenerate_resampling', 'degenerate policy')
    broken = [bad_unpaired_ratio(proportional, plan) for plan in all_plans(proportional)]
    detect('destroy_pairing', lambda: require(all(close(value, 2) for value in broken), 'pair invariant broken'))
    seeded = bootstrap_ratio(data, seeded_plans(data, 128, 7007))
    repeat = bootstrap_ratio(data, seeded_plans(data, 128, 7007))
    require(seeded == repeat and len(seeded['replicates']) == 128, 'seeded determinism')
    require(set(rounded_counter(seeded['replicates'])) <= set(rounded_reference(reference)), 'draw outside support')
    unequal = (((1, 1),), ((9, 1), (9, 1), (9, 1)))
    require(aggregate(unequal) == (5, 1), 'run weighting changed to pooled weighting')
    print('OBS ratio: 64 ordered plans match exact multinomial reference; point=1.75; support=[1.5, 8/3]')
    print('OBS pairing invariant, degenerate state, unequal run lengths, and fixed-seed plans checked')


def check_denominators():
    original_zero = (((2, 0),), ((4, 0),))
    out = bootstrap_ratio(original_zero, all_plans(original_zero))
    require(out['reason'] == 'point_denominator' and out['point'] is None and out['interval'] is None, 'original zero')
    zero_in_draw = (((2, 0),), ((2, 2),))
    result = bootstrap_ratio(zero_in_draw, all_plans(zero_in_draw))
    require(result['point'] == 2 and result['invalid'] == 1, 'numeric zero draw not consumed')
    require(len(result['replicates']) == 4 and result['replicates'].count(None) == 1, 'invalid draw lost')
    require(result['status'] == 'UNRESOLVED' and result['interval'] is None, 'zero draw must block interval')
    require(exact_distribution(zero_in_draw)[None] == 1, 'independent zero count')
    small = (((2, 1e-12),), ((2, 2),))
    near = bootstrap_ratio(small, all_plans(small), floor=1e-6)
    require(near['invalid'] == 1 and near['interval'] is None, 'near-zero draw guard')
    point_small = (((2, 1e-12),), ((2, 1e-12),))
    require(bootstrap_ratio(point_small, all_plans(point_small), floor=1e-6)['reason'] == 'point_denominator', 'near-zero point')
    equal_floor = (((2, 1),), ((2, 1),))
    require(bootstrap_ratio(equal_floor, all_plans(equal_floor), floor=1)['reason'] == 'point_denominator', 'floor equality')
    one = (((2, 1), (6, 4)),)
    require(bootstrap_ratio(one, all_plans(one))['reason'] == 'too_few_outer_units', 'one outer unit')
    expect_value_error(lambda: bootstrap_ratio((), ()))
    expect_value_error(lambda: bootstrap_ratio(((),), ()))
    expect_value_error(lambda: bootstrap_ratio((((1, -1),), ((2, 1),)), ()))
    expect_value_error(lambda: bootstrap_ratio((((math.nan, 1),), ((2, 1),)), ()))
    expect_value_error(lambda: bootstrap_ratio(zero_in_draw, (), floor=-1))
    expect_value_error(lambda: bootstrap_ratio(zero_in_draw, (), alpha=0))
    expect_value_error(lambda: aggregate(zero_in_draw, ((0, (1,)), (1, (0,)))))
    detect('replace_numeric_zero_denominator_with_zero_ratio',
           lambda: require(bad_zero_as_zero(2, 0) is None, 'numeric value cannot represent unresolved ratio'))
    print('OBS denominators: original zero, resampled zero (1/4 plans), positive near-zero, and floor equality checked')
    print('OBS no invalid replicate is dropped, clipped, or redrawn; finite interval withheld')


def check_difference():
    lo, hi = t_summary(-2, 1, 16, 2.131)
    require(close(lo, -2.53275) and close(hi, -1.46725), 'summary arithmetic')
    require(close((lo + hi) / 2, -2) and close(hi - lo, 2 * 2.131 / 4), 'center or width')
    expect_value_error(lambda: t_summary(0, 1, 1, 2.131))
    expect_value_error(lambda: paired_difference([1], 2.131))
    require(paired_difference([-2] * 16, 2.131)['interval'] is None, 'zero sd inference')
    require(paired_difference([0, 2], 1)['interval'] == (0, 2), 'nondegenerate difference')
    population_mean = Fraction(99, 100) * (-2) + Fraction(1, 100) * 98
    population_var = Fraction(99, 100) * (-2 - population_mean) ** 2 + Fraction(1, 100) * (98 - population_mean) ** 2
    require(population_mean == -1 and population_var == 99, 'discrete reference')
    bound = 1 - Fraction(99, 100) ** 16
    require(float(bound) < 0.15, 'coverage bound')
    detect('degenerate_interval_claimed_to_cover_population_mean',
           lambda: require(-2 <= population_mean <= -2, 'true mean outside mechanical interval'))
    print('OBS difference summary checked; discrete-model coverage upper bound=%.9f' % float(bound))


def variance_of_mean(values):
    if len(values) < 2:
        raise ValueError('at least two independent units required')
    center = mean(values)
    return sum((x - center) ** 2 for x in values) / ((len(values) - 1) * len(values))


def check_run_variance():
    correct = variance_of_mean([100, 200])
    pooled = variance_of_mean([100] * 1000 + [200] * 1000)
    require(close(correct, 2500) and close(pooled, 2500 / 1999), 'variance reference')
    expect_value_error(lambda: variance_of_mean([100]))
    detect('pooled_requests_as_independent_runs', lambda: require(close(pooled, correct), 'wrong independent unit'))
    print('OBS run-level SE=%.6f; erroneous pooled SE=%.6f' % (math.sqrt(correct), math.sqrt(pooled)))


def zero_upper(n, alpha):
    if n < 1 or not 0 < alpha < 1:
        raise ValueError('invalid binomial inputs')
    return -math.expm1(math.log(alpha) / n)


def check_tail():
    upper = zero_upper(100, 0.05)
    require(0.0294 < upper < 0.0296, 'zero-exceedance bound')
    require(close((1 - upper) ** 100, 0.05), 'binomial inversion')
    expect_value_error(lambda: zero_upper(0, 0.05))
    require(Fraction(99, 100) ** 100 > Fraction(1, 20), 'one percent counterexample')
    detect('zero_failures_exclude_one_percent',
           lambda: require(Fraction(99, 100) ** 100 <= Fraction(1, 20), 'one percent not excluded'))
    values = list(range(1, 101))
    require(nearest_rank(values, Fraction(99, 100)) == 99, 'p99 rank')
    require(nearest_rank([42], Fraction(99, 100)) == 42, 'singleton rank')
    require(nearest_rank([1, 1, 2, 2], Fraction(1, 2)) == 1, 'ties')
    expect_value_error(lambda: nearest_rank([], Fraction(99, 100)))
    expect_value_error(lambda: nearest_rank(values, 0))
    wrong = sorted(values)[min(len(values) - 1, int(0.99 * len(values)))]
    detect('wrong_p99_index', lambda: require(wrong == 99, 'off-by-one rank'))
    print('OBS zero-exceedance upper=%.9f; nearest-rank edges checked' % upper)


def validate_contract(contract):
    required = ('version', 'model', 'hardware', 'input_dist', 'cache', 'quality', 'measure_event', 'arrival', 'completion')
    reasons = ['missing ' + key for key in required if not contract.get(key)]
    if not contract.get('failures_retained'):
        reasons.append('failures not retained')
    if contract.get('paired_unit') not in ('run', 'block'):
        reasons.append('invalid paired unit')
    if not contract.get('holdout_separate'):
        reasons.append('holdout not separate')
    return reasons


def check_cost_selection_contract():
    noise = list(itertools.product((-1, 1), repeat=4))
    expected = Fraction(sum(max(row) for row in noise), len(noise))
    require(expected == Fraction(7, 8), 'selection reference')
    detect('selected_max_claimed_unbiased', lambda: require(expected == 0, 'selection bias'))
    cost = 2 * (100 + 10 * 1 + 5 + 2) + 10 + 3
    independent_bill = sum((100, 10, 5, 2, 100, 10, 5, 2, 10, 3))
    require(cost == independent_bill == 247, 'full cost')
    wrong_cost = 2 * (10 + 5 + 2) + 10 + 3
    detect('omitted_build_cost', lambda: require(wrong_cost == independent_bill, 'missing preparation'))
    contract = {key: 'declared' for key in ('version', 'model', 'hardware', 'input_dist', 'cache', 'quality', 'measure_event', 'arrival', 'completion')}
    contract.update(failures_retained=True, paired_unit='run', holdout_separate=True)
    require(validate_contract(contract) == [], 'complete contract')
    missing_quality = dict(contract, quality='')
    dropped = dict(contract, failures_retained=False)
    require('missing quality' in validate_contract(missing_quality), 'missing quality accepted')
    require('failures not retained' in validate_contract(dropped), 'dropped failures accepted')
    print('OBS expected selection noise=7/8; full cost=247; contract declaration failures rejected')


def main():
    checks = (check_ratio, check_denominators, check_difference, check_run_variance,
              check_tail, check_cost_selection_contract)
    failed = []
    for check in checks:
        try:
            check()
        except Exception as exc:
            failed.append(check.__name__)
            print('FAIL %s: %r' % (check.__name__, exc))
    print('SCOPE host arithmetic and finite paired two-level resampling only; no GPU or service measurements; no coverage-rate experiment')
    if failed:
        print('RESULT FAIL ' + ','.join(failed))
        raise SystemExit(1)
    print('RESULT PASS finite-model assertions only')


if __name__ == '__main__':
    main()
