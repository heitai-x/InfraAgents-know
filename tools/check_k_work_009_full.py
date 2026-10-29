from fractions import Fraction as F
import json


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def counter_step(x, y, modulus, upper, identity0='package-0:g1',
                 identity1='package-0:g1', no_reset=True, dt=1):
    require(isinstance(modulus, int) and modulus > 0, 'modulus')
    require(isinstance(x, int) and isinstance(y, int), 'integer readings')
    require(0 <= x < modulus and 0 <= y < modulus, 'reading range')
    require(identity0 == identity1, 'identity or generation changed')
    require(no_reset is True, 'reset not excluded')
    require(dt > 0, 'time order')
    require(0 <= upper < modulus, 'increment not uniquely bounded')
    delta = (y - x) % modulus
    require(delta <= upper, 'incompatible bound')
    return delta


def domain_total(selected, coverage, energy, target):
    covered = set()
    total = F(0)
    for name in selected:
        require(name in coverage and name in energy, 'unknown domain')
        atoms = set(coverage[name])
        require(bool(atoms), 'empty coverage')
        require(not covered.intersection(atoms), 'overlapping meters')
        covered.update(atoms)
        total += energy[name]
    require(covered == set(target), 'incomplete or excess coverage')
    return total


def charge(rows, quantum, rate):
    require(isinstance(quantum, int) and quantum > 0, 'quantum')
    require(rate >= 0, 'rate')
    seen = {}
    total = F(0)
    for key, start, stop in rows:
        require(isinstance(start, int) and isinstance(stop, int), 'time ticks')
        require(stop >= start, 'session order')
        payload = (start, stop)
        if key in seen:
            require(seen[key] == payload, 'conflicting bill item')
            continue
        seen[key] = payload
        duration = stop - start
        blocks = (duration + quantum - 1) // quantum
        total += blocks * quantum * rate
    return total


def charge_reference(rows, quantum, rate):
    unique = {}
    for key, start, stop in rows:
        if key in unique and unique[key] != (start, stop):
            raise ValueError('conflicting reference item')
        unique[key] = (start, stop)
    total = F(0)
    for start, stop in unique.values():
        cursor = start
        while cursor < stop:
            total += quantum * rate
            cursor += quantum
    return total


def integrate_power(samples, start, stop, max_gap):
    require(len(samples) >= 2 and start <= stop, 'window')
    require(max_gap > 0, 'sampling contract')
    require(samples[0][0] <= start <= stop <= samples[-1][0], 'coverage')
    total = F(0)
    for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
        require(t1 > t0 and p0 >= 0 and p1 >= 0, 'power samples')
        left, right = max(t0, start), min(t1, stop)
        if left >= right:
            continue
        require(t1 - t0 <= max_gap, 'sampling gap')
        slope = F(p1 - p0) / (t1 - t0)
        pl = p0 + slope * (left - t0)
        pr = p0 + slope * (right - t0)
        total += (pl + pr) * (right - left) / 2
    return total


def cohort(records, cutoff):
    seen = set()
    good, unknown, pending = set(), set(), set()
    for row in records:
        rid = row['rid']
        require(rid not in seen, 'duplicate logical request')
        seen.add(rid)
        state, done = row['state'], row['done']
        require(state in {'normal', 'reject', 'timeout', 'cancel', 'error', 'pending'},
                'unknown terminal state')
        require((state == 'pending') == (done is None), 'terminal evidence')
        for field in ('quality', 'slo'):
            require(row[field] is None or type(row[field]) is bool, 'predicate type')
        if done is None or done > cutoff:
            pending.add(rid)
        elif state == 'normal':
            if row['quality'] is None or row['slo'] is None:
                unknown.add(rid)
            elif row['quality'] and row['slo']:
                good.add(rid)
    return good, unknown, pending


def unit(total, good_count):
    require(total is not None, 'incomplete numerator')
    require(good_count >= 0, 'negative count')
    return None if good_count == 0 else F(total) / good_count


def allocate(components, weights):
    require(len(components) == len(weights), 'component identity')
    result = {}
    for cost, shares in zip(components, weights):
        require(cost >= 0 and all(w >= 0 for w in shares.values()), 'negative share')
        require(sum(shares.values(), F(0)) == 1, 'allocation conservation')
        for tenant, weight in shares.items():
            result[tenant] = result.get(tenant, F(0)) + cost * weight
    return result


def main():
    observations = {}
    mutants = []
    rejected = []

    def equal(actual, expected, label):
        if actual != expected:
            raise AssertionError((label, actual, expected))

    def detect(label, actual, expected):
        if actual == expected:
            raise AssertionError(('negative control escaped', label))
        mutants.append({'name': label, 'faulty': str(actual), 'reference': str(expected)})

    def reject(label, call):
        try:
            call()
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError(('invalid input accepted', label))

    cases, abs_mismatches = 0, 0
    first_abs = None
    for modulus in range(2, 18):
        for x in range(modulus):
            for truth in range(modulus):
                y = (x + truth) % modulus
                actual = counter_step(x, y, modulus, modulus - 1)
                equal(actual, truth, 'known injected increment')
                faulty = abs(y - x)
                if faulty != truth:
                    abs_mismatches += 1
                    if first_abs is None:
                        first_abs = (faulty, truth)
                cases += 1
    require(first_abs is not None, 'negative control witness missing')
    detect('absolute counter difference', *first_abs)
    equal(counter_step(20, 20, 100, 99), 0, 'equal endpoints under strict bound')
    equal(counter_step(90, 20, 100, 40), 30, 'wrap')
    reject('ambiguous full cycle', lambda: counter_step(20, 20, 100, 100))
    reject('multiple cycle ambiguity', lambda: counter_step(90, 20, 100, 130))
    reject('bound inconsistent with readings', lambda: counter_step(90, 20, 100, 29))
    reject('reset not excluded', lambda: counter_step(90, 20, 100, 40, no_reset=False))
    reject('changed device', lambda: counter_step(90, 20, 100, 40, identity1='package-1:g1'))
    reject('changed generation', lambda: counter_step(90, 20, 100, 40, identity1='package-0:g2'))
    reject('out of range', lambda: counter_step(100, 0, 100, 99))
    reject('zero modulus', lambda: counter_step(0, 0, 0, 0))
    reject('nonpositive sample interval', lambda: counter_step(0, 0, 100, 1, dt=0))
    increments = [70, 60, 80]
    readings = [90]
    for delta in increments:
        readings.append((readings[-1] + delta) % 100)
    recovered = sum(counter_step(x, y, 100, 99) for x, y in zip(readings, readings[1:]))
    equal(recovered, sum(increments), 'sum of individually recoverable intervals')
    detect('discarding intermediate counter samples', (readings[-1] - readings[0]) % 100,
           sum(increments))
    observations['counter'] = {'cases': cases, 'abs_fault_mismatches': abs_mismatches,
                               'multi_interval_microjoules': recovered}

    coverage = {'package': {'core', 'uncore', 'other'}, 'core': {'core'},
                'uncore': {'uncore'}, 'board': {'gpu'}, 'wall': {'core', 'uncore', 'other', 'gpu'}}
    energies = {'package': F(50), 'core': F(20), 'uncore': F(15),
                'board': F(70), 'wall': F(120)}
    target = {'core', 'uncore', 'other', 'gpu'}
    equal(domain_total(['package', 'board'], coverage, energies, target), 120, 'disjoint meters')
    equal(domain_total(['wall'], coverage, energies, target), 120, 'whole boundary')
    reject('parent and child overlap', lambda: domain_total(['package', 'core', 'board'], coverage, energies, target))
    reject('separate meter alias overlap', lambda: domain_total(['wall', 'board'], coverage, energies, target))
    reject('missing package remainder', lambda: domain_total(['core', 'uncore', 'board'], coverage, energies, target))
    detect('sum parent and children', sum(energies[n] for n in ['package', 'core', 'uncore', 'board']), F(120))

    rate = F(1, 100)
    billing_cases = 0
    for quantum in range(1, 8):
        for duration in range(22):
            rows = [('one-real-session', 3, 3 + duration)]
            expected = charge_reference(rows, quantum, rate)
            equal(charge(rows, quantum, rate), expected, 'enumerated billing blocks')
            billing_cases += 1
    rows = [('main', 0, 10), ('standby', 0, 10)]
    equal(charge(rows, 1, rate), F(1, 5), 'complete reservation cost')
    equal(charge(rows + [rows[0]], 1, rate), F(1, 5), 'identical duplicate export')
    reject('conflicting bill item', lambda: charge(rows + [('main', 0, 11)], 1, rate))
    reject('zero billing quantum', lambda: charge(rows, 0, rate))
    reject('reversed session', lambda: charge([('bad', 10, 9)], 1, rate))
    equal(charge(rows, 6, rate), F(6, 25), 'six second quantum')
    detect('floor billing', 2 * (10 // 6) * 6 * rate, charge_reference(rows, 6, rate))
    actual_split = charge([('first-session', 0, 1), ('second-session', 1, 2)], 6, rate)
    equal(actual_split, F(3, 25), 'two actual sessions')
    detect('merging actual independent sessions', charge([('merged', 0, 2)], 6, rate), actual_split)
    observations['billing'] = {'cases': billing_cases, 'one_second_cost': str(charge(rows, 1, rate)),
                               'six_second_cost': str(charge(rows, 6, rate))}

    samples = [(F(0), F(100)), (F(1), F(200)), (F(4), F(500))]
    left, right = F(1, 2), F(7, 2)
    analytic = 100 * (right - left) + 50 * (right * right - left * left)
    equal(integrate_power(samples, left, right, 3), analytic, 'linear antiderivative with clipped endpoints')
    equal(integrate_power(samples, 0, 4, 3), 1200, 'irregular full interval')
    equal(integrate_power(samples, 1, 1, 3), 0, 'zero integration interval')
    detect('unweighted power sample average', sum(p for _, p in samples) / len(samples) * 4, F(1200))
    reject('sampling gap', lambda: integrate_power(samples, 0, 4, 2))
    reject('missing boundary coverage', lambda: integrate_power(samples, -1, 4, 3))
    reject('nonmonotone samples', lambda: integrate_power([(0, 100), (0, 100)], 0, 0, 1))
    observations['power'] = {'clipped_joules': str(analytic), 'full_joules': '1200'}

    records = [
        {'rid': 'r1', 'state': 'normal', 'done': 6, 'quality': True, 'slo': True, 'attempts': 2},
        {'rid': 'r2', 'state': 'normal', 'done': 7, 'quality': False, 'slo': True, 'attempts': 1},
        {'rid': 'r3', 'state': 'timeout', 'done': 7, 'quality': None, 'slo': False, 'attempts': 1},
        {'rid': 'r4', 'state': 'reject', 'done': 3, 'quality': None, 'slo': None, 'attempts': 0}
    ]
    equal(cohort(records, 8), ({'r1'}, set(), set()), 'manual business qualification')
    equal(cohort(records, 4), (set(), set(), {'r1', 'r2', 'r3'}), 'early frozen snapshot')
    equal(cohort([], 8), (set(), set(), set()), 'empty cohort')
    late = {'rid': 'r5', 'state': 'normal', 'done': 9, 'quality': True, 'slo': True, 'attempts': 1}
    equal(cohort(records + [late], 8), ({'r1'}, set(), {'r5'}), 'late completion excluded')
    equal(cohort(records + [late], 9), ({'r1', 'r5'}, set(), set()), 'separately named later snapshot')
    unknown = {'rid': 'r6', 'state': 'normal', 'done': 5, 'quality': None, 'slo': True, 'attempts': 1}
    equal(cohort([unknown], 8), (set(), {'r6'}, set()), 'unknown quality')
    for state in ['cancel', 'error', 'timeout', 'reject']:
        row = dict(records[0], rid=state, state=state)
        equal(cohort([row], 8)[0], set(), 'non-normal terminal')
    reject('duplicate logical request', lambda: cohort(records + [records[0]], 8))
    reject('missing terminal evidence', lambda: cohort([dict(records[0], done=None)], 8))
    good = cohort(records, 8)[0]
    faulty_good = sum(row['attempts'] for row in records if row['rid'] in good)
    detect('attempt count as qualified result count', faulty_good, 1)

    stages = [(2, 400), (6, 600), (2, 300)]
    total_energy = sum(duration * power for duration, power in stages) + 10 * 80
    equal(total_energy, 5800, 'manual complete trajectory energy')
    total_cost = charge(rows, 1, rate)
    equal(unit(total_cost, len(good)), F(1, 5), 'cost per qualified request')
    equal(unit(total_energy, len(good)), 5800, 'energy per qualified request')
    detect('stop resource cost at business freeze', charge([('main', 0, 8), ('standby', 0, 8)], 1, rate), total_cost)
    detect('stop energy at business freeze', 2 * 400 + 6 * 600 + 8 * 80, total_energy)
    equal(unit(100, 0), None, 'zero good undefined')
    reject('incomplete energy numerator', lambda: unit(None, 1))
    incremental = F(50) - F(100) * 1
    equal(incremental, -50, 'signed incremental energy')
    detect('clamp negative baseline residual', max(F(0), incremental), F(-50))

    shares = allocate([F(1, 10), F(1, 10)],
                      [{'tenant': F(1)}, {'platform': F(1)}])
    equal(sum(shares.values()), total_cost, 'physical cost conservation')
    split = allocate([total_cost], [{'a': F(3, 5), 'b': F(2, 5)}])
    equal(split, {'a': F(3, 25), 'b': F(2, 25)}, 'contract allocation')
    reject('allocation drops idle cost', lambda: allocate([total_cost], [{'tenant': F(1, 2)}]))
    equal(F(500 * 3600, 10000), 180, 'A total unit energy')
    equal(F(350 * 3600, 6000), 210, 'B total unit energy')
    equal(F(400 * 3600, 10000), 144, 'A incremental unit energy')
    equal(F(250 * 3600, 6000), 150, 'B incremental unit energy')
    equal(F(3) / F(4, 10000), 7500, 'cost break even qualified count')
    equal(F(350 * 3600, 180), 7000, 'energy break even qualified count')
    equal(F(400 - 10 * 4, 100 - 10), 4, 'shutdown interval threshold')
    equal(400 + 10 * (10 - 4), 460, 'shutdown and restart joules')
    observations['trajectory'] = {'joules': total_energy, 'cost': str(total_cost), 'good': len(good)}
    observations['scope'] = 'Deterministic arithmetic and accounting model; no hardware, service, sensor or GPU measurements.'
    observations['rejected_invalid_inputs'] = rejected
    observations['detected_negative_controls'] = mutants
    print(json.dumps(observations, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
