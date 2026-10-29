#!/usr/bin/env python3
'''K-TOOLS-005: deterministic, offline, read-only finite CPU logic checks.
No project imports, subprocesses, network, credentials, or data-file access.
'''
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product


class CheckError(AssertionError):
    pass


def require(condition, message):
    if not condition:
        raise CheckError(message)


def expect_rejected(action, message):
    try:
        action()
    except CheckError:
        return
    raise CheckError(message)


@dataclass(frozen=True)
class Event:
    eid: str
    time: int
    violation: bool
    coverage_gap: bool = False


def predecessors(events, edges):
    ids = {e.eid for e in events}
    require(len(ids) == len(events), 'duplicate event identity')
    pred = {eid: set() for eid in ids}
    succ = {eid: set() for eid in ids}
    for a, b in edges:
        require(a in ids and b in ids, 'missing edge endpoint')
        pred[b].add(a)
        succ[a].add(b)
    degree = {eid: len(pred[eid]) for eid in ids}
    ready = sorted(eid for eid in ids if degree[eid] == 0)
    visited = 0
    while ready:
        a = ready.pop()
        visited += 1
        for b in sorted(succ[a]):
            degree[b] -= 1
            if degree[b] == 0:
                ready.append(b)
    require(visited == len(ids), 'causal input must be a DAG')
    return pred


def ancestors(pred, eid):
    found = set()
    stack = list(pred[eid])
    while stack:
        a = stack.pop()
        if a not in found:
            found.add(a)
            stack.extend(pred[a])
    return found


def earliest_frontier(events, edges, ignore_self=False):
    '''Projection onto failure candidates; gaps are supplied model assumptions.
    The second output combines unknown local evidence and unknown earliestness.
    It is not the chapter's entire four-part local-certificate interface.
    '''
    byid = {e.eid: e for e in events}
    pred = predecessors(events, edges)
    bad = {e.eid for e in events if e.violation}
    eligible, unknown = set(), set()
    for eid in sorted(bad):
        past = ancestors(pred, eid)
        self_gap = byid[eid].coverage_gap and not ignore_self
        if self_gap or any(byid[a].coverage_gap for a in past):
            unknown.add(eid)
        else:
            eligible.add(eid)
    frontier = {eid for eid in eligible if not (ancestors(pred, eid) & bad)}
    return tuple(sorted(frontier)), tuple(sorted(unknown))


def matrix_reference(events, edges):
    '''Independent all-pairs reachability, including checkpoint self-coverage.'''
    ids = [e.eid for e in events]
    require(len(set(ids)) == len(ids), 'reference duplicate identity')
    index = {eid: i for i, eid in enumerate(ids)}
    n = len(ids)
    reach = [[False] * n for _ in range(n)]
    for a, b in edges:
        require(a in index and b in index, 'reference endpoint')
        reach[index[a]][index[b]] = True
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])
    require(not any(reach[i][i] for i in range(n)), 'reference cycle')
    frontier, unknown = [], []
    for j, v in enumerate(events):
        if not v.violation:
            continue
        incomplete = any(
            e.coverage_gap and (i == j or reach[i][j])
            for i, e in enumerate(events)
        )
        if incomplete:
            unknown.append(v.eid)
        elif not any(e.violation and reach[i][j] for i, e in enumerate(events)):
            frontier.append(v.eid)
    return tuple(sorted(frontier)), tuple(sorted(unknown))


def audit_frontier(events, edges, claim):
    require(claim == matrix_reference(events, edges), 'independent frontier mismatch')


def check_frontier():
    events = [Event('rank0_alloc', 1, True), Event('rank1_timeout', 10, True)]
    edges = [('rank0_alloc', 'rank1_timeout')]
    require(earliest_frontier(events, edges) == (('rank0_alloc',), ()), 'earliest cause')
    last = max(events, key=lambda e: e.time).eid
    expect_rejected(lambda: audit_frontier(events, edges, ((last,), ())),
                    'last-timestamp negative control escaped')
    multi = [Event('a', 2, True), Event('b', 1, True)]
    require(earliest_frontier(multi, []) == (('a', 'b'), ()), 'multiple frontiers')
    gap = [Event('gap', 0, False, True), Event('hidden', 5, True)]
    require(earliest_frontier(gap, [('gap', 'hidden')]) == ((), ('hidden',)), 'ancestor gap')
    self_gap = [Event('uncovered_checkpoint', 1, True, True)]
    fixed = ((), ('uncovered_checkpoint',))
    require(earliest_frontier(self_gap, []) == fixed, 'own coverage must be UNKNOWN')
    require(matrix_reference(self_gap, []) == fixed, 'fixed self-gap reference')
    expect_rejected(
        lambda: audit_frontier(self_gap, [], earliest_frontier(self_gap, [], True)),
        'ignore-own-coverage negative control escaped',
    )
    expect_rejected(lambda: earliest_frontier(multi, [('a', 'b'), ('b', 'a')]),
                    'cyclic causal input accepted')
    cases = detections = 0
    for n in range(4):
        pairs = list(combinations(range(n), 2))
        for edge_bits in product((False, True), repeat=len(pairs)):
            graph = [(str(a), str(b)) for (a, b), bit in zip(pairs, edge_bits) if bit]
            for bad_bits in product((False, True), repeat=n):
                for gap_bits in product((False, True), repeat=n):
                    sample = [Event(str(i), n-i, bad_bits[i], gap_bits[i]) for i in range(n)]
                    reference = matrix_reference(sample, graph)
                    require(earliest_frontier(sample, graph) == reference, 'enumerated frontier')
                    detections += earliest_frontier(sample, graph, True) != reference
                    cases += 1
    require(cases == 549 and detections == 271, 'enumeration changed unexpectedly')
    print('OBS frontier cases', cases, 'ignore_self_detected', detections, 'self_gap', fixed)


@dataclass(frozen=True)
class Obligation:
    oid: str
    expected: tuple
    observed: object
    covered: bool
    value: object


def summarize_obligations(rows):
    require(len({r.oid for r in rows}) == len(rows), 'duplicate obligation')
    true, bad, unknown, invalid = {}, {}, {}, {}
    fields = ('request', 'version', 'unit', 'generation')
    for row in rows:
        require(len(row.expected) == 4, 'expected identity shape')
        mismatch = []
        if row.observed is not None:
            require(len(row.observed) == 4, 'observed identity shape')
            mismatch = [
                '%s: expected=%r observed=%r' % (name, a, b)
                for name, a, b in zip(fields, row.expected, row.observed)
                if a is not None and b is not None and a != b
            ]
        if mismatch:
            invalid[row.oid] = tuple(mismatch)
        elif row.observed is None or any(v is None for v in row.expected + row.observed):
            unknown[row.oid] = 'identity unavailable'
        elif not row.covered:
            unknown[row.oid] = 'checkpoint coverage unavailable'
        elif row.value is None:
            unknown[row.oid] = 'predicate unavailable'
        else:
            require(type(row.value) is bool, 'predicate must be bool or None')
            (true if row.value else bad)[row.oid] = 'valid local comparison'
    total = len(rows)
    checked = len(true) + len(bad)
    status = ('NO_OBLIGATIONS' if not total else
              'FAILURE_FOUND' if bad else
              'NO_FAILURE_FOUND' if checked else 'NO_VALID_COMPARISON')
    return dict(TrueSet=true, Bad=bad, Unknown=unknown, OracleInvalid=invalid,
                total=total, checked=checked, status=status,
                complete=bool(total) and checked == total,
                all_good=bool(total) and len(true) == total)


def audit_partition(rows, result):
    groups = [set(result[k]) for k in ('TrueSet', 'Bad', 'Unknown', 'OracleInvalid')]
    require(set().union(*groups) == {r.oid for r in rows}, 'lost obligation')
    require(sum(map(len, groups)) == len(rows), 'overlapping classifications')
    require(result['checked'] == len(groups[0]) + len(groups[1]), 'invalid counted as checked')


def audit_all_good(rows, claim):
    # Direct contract, independent of summarize_obligations and its counters.
    valid = bool(rows) and all(
        row.observed is not None
        and len(row.expected) == 4
        and len(row.observed) == 4
        and all(x is not None for x in row.expected + row.observed)
        and row.expected == row.observed
        and row.covered
        and row.value is True
        for row in rows
    )
    require(claim == valid, 'all-valid-no-failure contract violated')


def check_obligations():
    identity = ('q', 'v1', 'ms', 4)
    t = Obligation('valid', identity, identity, True, True)
    b = Obligation('bad', identity, identity, True, False)
    i = Obligation('invalid', identity, ('q', 'v2', 'ms', 4), True, True)
    u = Obligation('unknown', identity, identity, False, None)
    fixtures = [
        ([], ('NO_OBLIGATIONS', 0, False, False)),
        ([i], ('NO_VALID_COMPARISON', 0, False, False)),
        ([t, i], ('NO_FAILURE_FOUND', 1, False, False)),
        ([b, i], ('FAILURE_FOUND', 1, False, False)),
        ([u], ('NO_VALID_COMPARISON', 0, False, False)),
        ([t], ('NO_FAILURE_FOUND', 1, True, True)),
        ([t, b], ('FAILURE_FOUND', 2, True, False)),
    ]
    for rows, expected in fixtures:
        result = summarize_obligations(rows)
        audit_partition(rows, result)
        audit_all_good(rows, result['all_good'])
        actual = tuple(result[k] for k in ('status', 'checked', 'complete', 'all_good'))
        require(actual == expected, 'fixed obligation summary')
    invalid_only = summarize_obligations([i])
    mixed = summarize_obligations([t, i])
    require('version' in invalid_only['OracleInvalid']['invalid'][0], 'invalid reason lost')
    for rows in ([i], [t, i]):
        result = summarize_obligations(rows)
        old_claim = not result['Bad'] and not result['Unknown']
        expect_rejected(lambda: audit_all_good(rows, old_claim),
                        'invalid-blind summary negative control escaped')
    for field, replacement in enumerate(('other_request', 'v2', 'bytes', 5)):
        actual = list(identity)
        actual[field] = replacement
        row = Obligation('mismatch', identity, tuple(actual), True, False)
        result = summarize_obligations([row])
        require(result['checked'] == 0 and bool(result['OracleInvalid']), 'identity dimension lost')
    missing = Obligation('missing', identity, None, True, True)
    require('missing' in summarize_obligations([missing])['Unknown'], 'missing identity passed')
    # A known mismatch dominates a separate unknown identity field.
    partly_known = Obligation('partial', identity, (None, 'v2', 'ms', 4), False, None)
    require('partial' in summarize_obligations([partly_known])['OracleInvalid'],
            'known mismatch disappeared into unknown')
    print('OBS obligations fixed_cases', len(fixtures), 'invalid_only', invalid_only['status'],
          'mixed_checked', mixed['checked'], 'mixed_complete', mixed['complete'])


REQUIRED_COVERAGE = frozenset(('checkpoint', 'attempt', 'causal_past'))
CORE = frozenset(('A', 'B', 'overlap'))


@dataclass(frozen=True)
class Observation:
    identity: object
    coverage: object
    preserves: object
    same_failure: object


def observation_flags(obs):
    coverage_ok = None if obs.coverage is None else REQUIRED_COVERAGE <= obs.coverage
    return obs.identity, coverage_ok, obs.preserves, obs.same_failure


def classify_flags(flags):
    labels = ('identity', 'coverage', 'mechanism', 'failure')
    require(len(flags) == len(labels), 'four required conditions')
    require(all(value is None or type(value) is bool for value in flags), 'three-valued input')
    for label, value in zip(labels, flags):
        if value is False:
            return 'REJECT', label
    for label, value in zip(labels, flags):
        if value is None:
            return 'UNKNOWN', label
    return 'ACCEPT', 'all obligations witnessed'


def completion_reference(flags):
    '''Independent possible-worlds reference for conjunction under unknowns.'''
    possibilities = [(False, True) if value is None else (value,) for value in flags]
    answers = [all(assignment) for assignment in product(*possibilities)]
    if all(answers):
        return 'ACCEPT'
    if not any(answers):
        return 'REJECT'
    return 'UNKNOWN'


def run_model(case, guards):
    '''Explicit Boolean fixture, not a scheduler or real overlapping write.'''
    case = frozenset(case)
    core_present = CORE <= case
    identity = True
    if 'identity_guard' in guards and 'identity_guard' not in case:
        identity = False
    if 'identity_unknown_guard' in guards and 'identity_unknown_guard' not in case:
        identity = None
    coverage = REQUIRED_COVERAGE
    if 'coverage_guard' in guards and 'coverage_guard' not in case:
        coverage = REQUIRED_COVERAGE - {'causal_past'}
    failure = core_present
    if 'observe_guard' in guards and 'observe_guard' not in case:
        failure = None
    return Observation(identity, coverage, core_present, failure)


@dataclass(frozen=True)
class Trial:
    version: int
    current: tuple
    removed: str
    candidate: tuple
    observation: Observation
    verdict: str
    reason: str


def minimize_case(case, guards=frozenset(), budget=100, mode='good'):
    require(type(budget) is int and budget >= 1, 'budget must permit initial observation')
    current = tuple(sorted(case))
    require(len(set(current)) == len(current), 'duplicate removable item')
    initial = run_model(current, guards)
    calls, version = 1, 0
    history = []
    if classify_flags(observation_flags(initial))[0] != 'ACCEPT':
        return dict(current=current, version=0, initial=initial, calls=calls,
                    history=(), active=(), pending=(), status='INITIAL_UNPROVEN')
    while True:
        active = []
        pending = ()
        changed = False
        for index, item in enumerate(current):
            if calls >= budget:
                pending = current[index:]
                break
            candidate = tuple(x for x in current if x != item)
            require(len(candidate) < len(current), 'no strict reduction')
            obs = run_model(candidate, guards)
            calls += 1
            verdict, reason = classify_flags(observation_flags(obs))
            if mode == 'weak' and obs.preserves is True and obs.same_failure is True:
                verdict, reason = 'ACCEPT', 'faulty F-and-P-only gate'
            if mode == 'accept_any':
                verdict, reason = 'ACCEPT', 'faulty missing-failure gate'
            trial = Trial(version, current, item, candidate, obs, verdict, reason)
            history.append(trial)
            if not (mode == 'drop_unknown' and verdict == 'UNKNOWN'):
                active.append(trial)
            if verdict == 'ACCEPT':
                current = candidate
                version += 1
                changed = True
                break
        if changed:
            continue
        status = ('INCOMPLETE' if pending or any(t.verdict == 'UNKNOWN' for t in active)
                  else 'ONE_MINIMAL_ADMISSIBLE')
        return dict(current=current, version=version, initial=initial, calls=calls,
                    history=tuple(history), active=tuple(active), pending=pending, status=status)


def audit_minimization(start, guards, budget, result):
    '''Audit full history and candidate universe without using the tested gate.'''
    require(completion_reference(observation_flags(result['initial'])) == 'ACCEPT',
            'initial failure unproven')
    require(result['initial'] == run_model(start, guards), 'initial identity mismatch')
    cursor = tuple(sorted(start))
    version = 0
    seen = set()
    for trial in result['history']:
        require((trial.version, trial.current) == (version, cursor), 'wrong current version')
        key = (version, trial.removed)
        require(key not in seen, 'same-version no-progress retry')
        seen.add(key)
        require(trial.removed in cursor, 'unknown deletion')
        expected_candidate = tuple(x for x in cursor if x != trial.removed)
        require(trial.candidate == expected_candidate, 'candidate identity mismatch')
        require(set(trial.candidate) < set(cursor), 'non-decreasing reduction')
        require(trial.observation == run_model(trial.candidate, guards), 'observation fixture mismatch')
        expected_verdict = completion_reference(observation_flags(trial.observation))
        require(trial.verdict == expected_verdict, 'acceptance predicate violated')
        if trial.verdict == 'ACCEPT':
            cursor = trial.candidate
            version += 1
    require((result['current'], result['version']) == (cursor, version), 'final state mismatch')
    expected_active = tuple(t for t in result['history'] if t.version == version)
    require(result['active'] == expected_active, 'current-version unknown or rejected record lost')
    require(not any(t.verdict == 'ACCEPT' for t in expected_active), 'unapplied final acceptance')
    tested = {t.removed for t in expected_active}
    untested = tuple(item for item in cursor if item not in tested)
    require(result['pending'] == untested, 'untried deletion omitted')
    require(result['calls'] == 1 + len(result['history']), 'call accounting mismatch')
    require(result['calls'] <= budget, 'budget exceeded')
    incomplete = bool(untested) or any(t.verdict == 'UNKNOWN' for t in expected_active)
    expected_status = 'INCOMPLETE' if incomplete else 'ONE_MINIMAL_ADMISSIBLE'
    require(result['status'] == expected_status, 'minimality status overstated')
    require(completion_reference(observation_flags(run_model(cursor, guards))) == 'ACCEPT',
            'final target failure lost')


def check_minimal():
    truth_cases = 0
    for flags in product((False, True, None), repeat=4):
        require(classify_flags(flags)[0] == completion_reference(flags), 'three-valued gate mismatch')
        truth_cases += 1
    require(truth_cases == 81, 'truth table incomplete')
    start = CORE | {'trace', 'seed'}
    valid = minimize_case(start)
    audit_minimization(start, frozenset(), 100, valid)
    require(set(valid['current']) == CORE, 'irrelevant items retained or core lost')
    require(valid['status'] == 'ONE_MINIMAL_ADMISSIBLE', 'complete repro proof expected')

    coverage_start = CORE | {'coverage_guard', 'trace'}
    coverage_guards = frozenset(('coverage_guard',))
    coverage = minimize_case(coverage_start, coverage_guards)
    audit_minimization(coverage_start, coverage_guards, 100, coverage)
    require('coverage_guard' in coverage['current'] and 'trace' not in coverage['current'],
            'coverage reduction')
    coverage_trial = next(t for t in coverage['active'] if t.removed == 'coverage_guard')
    require(coverage_trial.observation.same_failure is True
            and coverage_trial.observation.preserves is True,
            'controlled coverage candidate must retain F and P')
    require((coverage_trial.verdict, coverage_trial.reason) == ('REJECT', 'coverage'),
            'coverage loss accepted')

    unknown_start = CORE | {'observe_guard', 'trace'}
    unknown_guards = frozenset(('observe_guard',))
    unknown = minimize_case(unknown_start, unknown_guards)
    audit_minimization(unknown_start, unknown_guards, 100, unknown)
    require(unknown['status'] == 'INCOMPLETE', 'unobservable deletion certified')
    require(any(t.verdict == 'UNKNOWN' and t.version == unknown['version']
                for t in unknown['active']), 'active unknown missing')
    require(any(t.verdict == 'UNKNOWN' and t.version < unknown['version']
                for t in unknown['history']), 'historical unknown version lost')

    budget_start = CORE | {'trace'}
    budgeted = minimize_case(budget_start, budget=1)
    audit_minimization(budget_start, frozenset(), 1, budgeted)
    require(budgeted['status'] == 'INCOMPLETE' and len(budgeted['pending']) == 4,
            'budget boundary lost')
    for guard, expected in (('identity_guard', 'ONE_MINIMAL_ADMISSIBLE'),
                            ('identity_unknown_guard', 'INCOMPLETE')):
        sample = CORE | {guard}
        guards = frozenset((guard,))
        result = minimize_case(sample, guards)
        audit_minimization(sample, guards, 100, result)
        require(result['status'] == expected, 'identity decision mismatch')
    unproven = minimize_case({'A', 'overlap'})
    require(unproven['status'] == 'INITIAL_UNPROVEN', 'invalid initial repro minimized')

    weak = minimize_case(coverage_start, coverage_guards, mode='weak')
    expect_rejected(lambda: audit_minimization(coverage_start, coverage_guards, 100, weak),
                    'F-and-P-only minimizer escaped')
    dropped = minimize_case(unknown_start, unknown_guards, mode='drop_unknown')
    expect_rejected(lambda: audit_minimization(unknown_start, unknown_guards, 100, dropped),
                    'dropped-unknown minimizer escaped')
    missing_failure = minimize_case(start, mode='accept_any')
    expect_rejected(lambda: audit_minimization(start, frozenset(), 100, missing_failure),
                    'missing-failure minimizer escaped')
    print('OBS minimization truth_cases', truth_cases, 'coverage', coverage['status'],
          'unobservable', unknown['status'], 'final_version', unknown['version'],
          'budget_pending', len(budgeted['pending']), 'negative_controls_detected', 3)


def resident_rate(capacity, task_size, duration):
    require(type(capacity) is int and capacity >= 0, 'nonnegative integer capacity')
    require(type(task_size) is int and task_size > 0, 'positive integer task size')
    require(type(duration) is int and duration > 0, 'positive integer duration')
    return Fraction(capacity // task_size, duration)


def resident_reference(capacity, task_size, duration):
    '''Construct distinct occupied cells; do not use division to count slots.
    All cells have equal size. Each packed task repeats every duration seconds.
    '''
    cells = [False] * capacity
    tasks = 0
    while True:
        free = []
        for index, occupied in enumerate(cells):
            if not occupied:
                free.append(index)
                if len(free) == task_size:
                    break
        if len(free) != task_size:
            return Fraction(tasks, duration)
        for index in free:
            cells[index] = True
        tasks += 1


def audit_resident_rate(capacity, task_size, duration, claim):
    require(claim == resident_reference(capacity, task_size, duration),
            'resident rate differs from independent cell packing')


def service_ceiling(service, work):
    require(service >= 0 and work > 0, 'service and work domains')
    return Fraction(service) / Fraction(work)


def audit_upper_bound(witness_rate, claimed_bound):
    require(witness_rate <= claimed_bound, 'feasible witness exceeds claimed upper bound')


def check_cost():
    cases = 0
    for capacity in range(21):
        for task_size in range(1, 8):
            for duration in range(1, 5):
                audit_resident_rate(capacity, task_size, duration,
                                    resident_rate(capacity, task_size, duration))
                cases += 1
    require(cases == 588, 'resident enumeration incomplete')
    base = resident_rate(10, 3, 1)
    nine = resident_rate(9, 3, 1)
    eight = resident_rate(8, 3, 1)
    require((base, nine, eight) == (Fraction(3), Fraction(3), Fraction(2)),
            'fixed discrete-residency counterexample')
    require(resident_rate(0, 3, 1) == 0 and resident_rate(2, 3, 1) == 0,
            'zero and infeasible capacity boundaries')
    old_memory_bound = base * Fraction(9, 10)
    require(old_memory_bound == Fraction(27, 10), 'old bound fixture')
    expect_rejected(lambda: audit_upper_bound(nine, old_memory_bound),
                    'proportional memory-throughput upper bound escaped')
    expect_rejected(lambda: audit_resident_rate(9, 3, 1, old_memory_bound),
                    'proportional resident-rate predictor escaped')
    expect_rejected(lambda: resident_rate(10, 0, 1), 'zero task size accepted')
    expect_rejected(lambda: resident_rate(10, 3, 0), 'zero duration accepted')

    service_cases = 0
    for supply in range(1, 13):
        for work in range(1, 7):
            saturated_base = service_ceiling(supply, work)
            for retained_quarters in range(5):
                retained = Fraction(retained_quarters, 4)
                after_supply = retained * supply
                after_ceiling = service_ceiling(after_supply, work)
                require(after_ceiling == retained * saturated_base, 'linear service scaling')
                # Independent resource conservation: each completion consumes work.
                require(after_ceiling * work == after_supply, 'resource work conservation')
                service_cases += 1
    require(service_cases == 360, 'service enumeration incomplete')
    original_capacity = service_ceiling(10, 3)
    remaining_capacity = service_ceiling(9, 3)
    offered = Fraction(2)
    observed_before = min(offered, original_capacity)
    feasible_after = min(offered, remaining_capacity)
    require(observed_before == feasible_after == 2, 'underloaded fixed fixture')
    wrong_observed_bound = observed_before * Fraction(9, 10)
    expect_rejected(lambda: audit_upper_bound(feasible_after, wrong_observed_bound),
                    'observed baseline used as saturated capacity escaped')
    audit_upper_bound(feasible_after, Fraction(9, 10) * original_capacity)
    print('OBS cost resident_cases', cases, 'service_cases', service_cases,
          'resident_10_9_8', tuple(map(str, (base, nine, eight))),
          'obsolete_memory_bound', str(old_memory_bound),
          'underloaded_before_after', str(observed_before), str(feasible_after),
          'faulty_bound_families_detected', 2)


def process_facts(kind, rc):
    if kind == 'spawn_error':
        require(rc is None, 'spawn error has no return code in fixture')
        return ('spawn_error', False, False, None, None)
    if kind == 'run_timeout':
        require(rc is None, 'timeout is not an exit code')
        return ('timeout_after_kill', True, True, None, None)
    if kind == 'communicate_timeout':
        require(rc is None, 'timeout is not an exit code')
        return ('timeout_child_may_live', True, False, None, None)
    require(kind == 'completed' and type(rc) is int, 'invalid completed fixture')
    if rc < 0:
        return ('signaled', True, True, -rc, None)
    return ('exited', True, True, None, rc)


def check_process():
    # Fixed public-contract expectations; no real process is launched.
    fixtures = [
        ('spawn_error', None, ('spawn_error', False, False, None, None)),
        ('completed', 0, ('exited', True, True, None, 0)),
        ('completed', 7, ('exited', True, True, None, 7)),
        ('completed', -9, ('signaled', True, True, 9, None)),
        ('run_timeout', None, ('timeout_after_kill', True, True, None, None)),
        ('communicate_timeout', None, ('timeout_child_may_live', True, False, None, None)),
    ]
    for kind, rc, expected in fixtures:
        require(process_facts(kind, rc) == expected, 'public process contract mismatch')
    wrong_signal = ('exited', True, True, None, 9)
    wrong_timeout = ('timeout_child_may_live', True, True, None, None)
    require(wrong_signal != fixtures[3][2], 'signal-as-exit negative escaped')
    require(wrong_timeout != fixtures[5][2], 'timeout-as-kill negative escaped')
    print('OBS process contract_fixtures', len(fixtures), 'real_processes', 0)


def argmax(values):
    require(bool(values), 'empty logits')
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def first_semantic_divergence(a, b):
    require(len(a) == len(b), 'sequence shape')
    for index, (left, right) in enumerate(zip(a, b)):
        require(len(left) == len(right) and bool(left), 'logit shape')
        if argmax(left) != argmax(right):
            return index
    return None


def check_numeric():
    a = ((0.0, 1.0), (2.0, 3.0), (1.0, 2.0))
    b = ((1e-12, 1.0), (3.0, 2.0), (1.0, 2.0000001))
    numeric = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    semantic = first_semantic_divergence(a, b)
    require(numeric == 0 and semantic == 1, 'hand-computed divergence positions')
    tiny_only = ((1e-12, 1.0), (2.0, 3.0000001), (1.0, 2.0))
    require(first_semantic_divergence(a, tiny_only) is None,
            'tiny numeric difference called task failure')
    require(semantic != numeric, 'numeric-only negative control escaped')
    require(first_semantic_divergence((), ()) is None, 'empty finite sequence boundary')
    expect_rejected(lambda: first_semantic_divergence(((1.0,),), ((1.0, 2.0),)),
                    'shape mismatch escaped')
    print('OBS numeric first_numeric', numeric, 'first_argmax', semantic,
          'tiny_only_argmax_change', False)


def main():
    check_frontier()
    check_obligations()
    check_minimal()
    check_cost()
    check_process()
    check_numeric()
    print('RESULT scoped_cpu_logic_checks_ok; no GPU, OS, network, or service measurements')


if __name__ == '__main__':
    main()
