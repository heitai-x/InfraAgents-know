#!/usr/bin/env python3
'''K-SYS-006 deterministic, standalone evidence-promotion model check.

Python standard library only. No file access, writes, network, credentials,
subprocesses or project imports. chapter_gate transcribes P01-P17 including
P13 and P13b. The reference is a union of declarative state certificates.
The interval fixtures reach both the actual gate and those certificates.

Exhaustion covers each kind's relevant fields. Irrelevant fields use all-false
and all-true projections, not their entire Cartesian product. This does not
validate evidence extraction, statistical coverage, real dependencies, devices
or Agent benefits. The supplied t critical value is not computed here.
'''

from fractions import Fraction
from itertools import product
from math import isclose, isfinite, sqrt
import sys

COMMON = (
    'scope_declared', 'identity_match', 'terminal_misreported',
    'bad_control_accepted', 'same_scope_conflict',
)
RUN = (
    'terminal_complete', 'observation', 'same_condition',
    'base_reference', 'base_control_rejected',
)
EXTRA = {
    'MECH': ('mechanism', 'source_supported', 'source_scope_match'),
    'OBS': ('mechanism', 'invalidation_stated'),
    'TRANSFER': (
        'mechanism', 'heldout_changed', 'heldout_support', 'heldout_conflict',
        'changed_reference', 'changed_control_rejected',
    ),
    'BUILD_EFFECT': (
        'mechanism', 'paired_or_blocked', 'budget_matched',
        'evaluator_fixed', 'heldout_build', 'documented_use_path',
    ),
}
RELEVANT = {
    kind: COMMON + (() if kind == 'MECH' else RUN) + extra
    for kind, extra in EXTRA.items()
}
BOOL_KEYS = tuple(dict.fromkeys(
    key for relevant in RELEVANT.values() for key in relevant
))
ADVERSE = frozenset((
    'terminal_misreported', 'bad_control_accepted',
    'same_scope_conflict', 'heldout_conflict',
))
FIELD_RULES = {
    'scope_declared': 'P02: required locatable scope',
    'identity_match': 'P02: normalized required identities match',
    'terminal_misreported': 'P03: non-success reported as success',
    'bad_control_accepted': 'P04: used checker accepts known bad',
    'same_scope_conflict': 'P05: valid same-scope contradiction',
    'source_supported': 'P06: fixed source/derivation supports M',
    'source_scope_match': 'P06: scope satisfies source preconditions',
    'terminal_complete': 'P07: complete unique planned terminals',
    'observation': 'P08: valid complete observation exists',
    'same_condition': 'P08: observation matches comparison scope',
    'base_reference': 'P09: original dimensions independently checked',
    'base_control_rejected': 'P09: original controls executed/rejected',
    'mechanism': 'P06/P10/P13b/P16: expanded kind-specific explanation',
    'invalidation_stated': 'P10: explicit OBS invalidation conditions',
    'heldout_changed': 'P11: changed-condition evaluation actually executed',
    'heldout_support': 'P13: evaluated result meets frozen support rule for P',
    'heldout_conflict': 'P12: evaluated result refutes P under frozen rule',
    'changed_reference': 'P13b: independent changed-condition reference',
    'changed_control_rejected': 'P13b: changed-condition controls rejected',
    'paired_or_blocked': 'P15: complete construction runs paired/blocked',
    'budget_matched': 'P15: matching budget definition and limit',
    'evaluator_fixed': 'P15: frozen common evaluator and aggregation',
    'heldout_build': 'P16: held-out construction result supports P',
    'documented_use_path': 'P16: knowledge/read/decision/artifact/check trace',
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def chapter_gate(kind, f):
    # P01
    if kind not in EXTRA:
        return 'BLOCKED'
    # P02
    if not f['scope_declared'] or not f['identity_match']:
        return 'BLOCKED'
    # P03
    if f['terminal_misreported']:
        return 'BLOCKED'
    # P04
    if f['bad_control_accepted']:
        return 'BLOCKED'
    # P05
    if f['same_scope_conflict']:
        return 'CONTESTED'
    # P06
    if kind == 'MECH':
        if f['mechanism'] and f['source_supported'] and f['source_scope_match']:
            return 'SOURCE_SUPPORTED'
        return 'BLOCKED'
    # P07
    if not f['terminal_complete']:
        return 'BLOCKED'
    # P08
    if not f['observation'] or not f['same_condition']:
        return 'BLOCKED'
    # P09
    if not f['base_reference'] or not f['base_control_rejected']:
        return 'OBSERVED'
    # P10
    if kind == 'OBS':
        if f['mechanism'] and f['invalidation_stated']:
            return 'SUPPORTED'
        return 'CHECKED'
    if kind == 'TRANSFER':
        # P11: neither result field is consumed without an evaluation.
        if not f['heldout_changed']:
            return 'CHECKED'
        # P12: refutation wins even over a contradictory support flag.
        if f['heldout_conflict']:
            return 'CONTESTED'
        # P13: executed and not refuted does not imply supported.
        if not f['heldout_support']:
            return 'CHECKED'
        # P13b
        if (not f['changed_reference'] or not f['changed_control_rejected']
                or not f['mechanism']):
            return 'CHECKED'
        # P14
        return 'TRANSFERABLE'
    # P15: only BUILD_EFFECT can reach here.
    if (not f['paired_or_blocked'] or not f['budget_matched']
            or not f['evaluator_fixed']):
        return 'CHECKED'
    # P16
    if (not f['heldout_build'] or not f['documented_use_path']
            or not f['mechanism']):
        return 'CHECKED'
    # P17
    return 'SUPPORTED_EFFECT'


def state_certificates(kind):
    '''Declarative DNF contracts, independent of the sequential gate.

    Each tuple is (state, required_true, required_false). Different certificates
    may justify the same state; each profile must admit exactly one state.
    '''
    result = []

    def add(state, positive=(), negative=()):
        result.append((state, frozenset(positive), frozenset(negative)))

    for key in ('scope_declared', 'identity_match'):
        add('BLOCKED', negative=(key,))
    for key in ('terminal_misreported', 'bad_control_accepted'):
        add('BLOCKED', positive=(key,))

    global_positive = ('scope_declared', 'identity_match')
    global_negative = ('terminal_misreported', 'bad_control_accepted')
    add('CONTESTED', global_positive + ('same_scope_conflict',), global_negative)
    clean = global_negative + ('same_scope_conflict',)

    if kind == 'MECH':
        needed = ('mechanism', 'source_supported', 'source_scope_match')
        add('SOURCE_SUPPORTED', global_positive + needed, clean)
        for key in needed:
            add('BLOCKED', global_positive, clean + (key,))
    else:
        observed = ('terminal_complete', 'observation', 'same_condition')
        for key in observed:
            add('BLOCKED', global_positive, clean + (key,))
        runtime = global_positive + observed
        for key in ('base_reference', 'base_control_rejected'):
            add('OBSERVED', runtime, clean + (key,))
        checked = runtime + ('base_reference', 'base_control_rejected')

        if kind == 'OBS':
            needed = ('mechanism', 'invalidation_stated')
            add('SUPPORTED', checked + needed, clean)
            for key in needed:
                add('CHECKED', checked, clean + (key,))
        elif kind == 'TRANSFER':
            # Not evaluated: H and F are both irrelevant.
            add('CHECKED', checked, clean + ('heldout_changed',))
            # Evaluated refutation: H and all later gates are irrelevant.
            add('CONTESTED', checked + ('heldout_changed', 'heldout_conflict'), clean)
            transfer_positive = checked + ('heldout_changed',)
            transfer_negative = clean + ('heldout_conflict',)
            needed = (
                'heldout_support', 'changed_reference',
                'changed_control_rejected', 'mechanism',
            )
            add('TRANSFERABLE', transfer_positive + needed, transfer_negative)
            for key in needed:
                add('CHECKED', transfer_positive, transfer_negative + (key,))
        elif kind == 'BUILD_EFFECT':
            needed = (
                'mechanism', 'paired_or_blocked', 'budget_matched',
                'evaluator_fixed', 'heldout_build', 'documented_use_path',
            )
            add('SUPPORTED_EFFECT', checked + needed, clean)
            for key in needed:
                add('CHECKED', checked, clean + (key,))
    return tuple(result)


CERTIFICATES = {kind: state_certificates(kind) for kind in EXTRA}


def allowed_states(kind, f):
    if kind not in CERTIFICATES:
        return frozenset(('BLOCKED',))
    return frozenset(
        state for state, positive, negative in CERTIFICATES[kind]
        if all(f[key] for key in positive)
        and all(not f[key] for key in negative)
    )


def all_good():
    return {key: key not in ADVERSE for key in BOOL_KEYS}


def exhaustive_check():
    profiles = 0
    evaluations = 0
    by_kind = {}
    for kind, relevant in RELEVANT.items():
        by_kind[kind] = 0
        irrelevant = set(BOOL_KEYS).difference(relevant)
        for bits in product((False, True), repeat=len(relevant)):
            f = dict.fromkeys(BOOL_KEYS, False)
            f.update(zip(relevant, bits))
            allowed = allowed_states(kind, f)
            require(len(allowed) == 1, ('nonunique_or_missing_certificate', kind, f, allowed))
            actual = chapter_gate(kind, f)
            require(actual in allowed, ('state_mismatch', kind, f, actual, allowed))
            profiles += 1
            by_kind[kind] += 1
            evaluations += 1

            changed = dict(f)
            changed.update((key, True) for key in irrelevant)
            require(allowed_states(kind, changed) == allowed,
                    ('irrelevant_certificate_change', kind, f))
            require(chapter_gate(kind, changed) == actual,
                    ('irrelevant_state_change', kind, f))
            evaluations += 1
    require(profiles == 135424, ('profile_count', profiles))
    require(evaluations == 270848, ('evaluation_count', evaluations))
    return profiles, evaluations, by_kind


PROMOTION_REQUIREMENTS = {
    'MECH': ('scope_declared', 'identity_match') + EXTRA['MECH'],
    'OBS': ('scope_declared', 'identity_match') + RUN + EXTRA['OBS'],
    'TRANSFER': ('scope_declared', 'identity_match') + RUN + (
        'mechanism', 'heldout_changed', 'heldout_support',
        'changed_reference', 'changed_control_rejected',
    ),
    'BUILD_EFFECT': ('scope_declared', 'identity_match') + RUN + EXTRA['BUILD_EFFECT'],
}


def faulty_ignore_predicate(kind, f, key, forced_value):
    '''Intentional single-guard omission mutant; not an evidence oracle.'''
    altered = dict(f)
    altered[key] = forced_value
    return chapter_gate(kind, altered)


def mutation_check():
    missing_detected = 0
    adverse_detected = 0
    directed = []
    special = {
        ('OBS', 'invalidation_stated'),
        ('TRANSFER', 'changed_reference'),
        ('TRANSFER', 'heldout_support'),
        ('BUILD_EFFECT', 'mechanism'),
        ('BUILD_EFFECT', 'evaluator_fixed'),
        ('BUILD_EFFECT', 'documented_use_path'),
    }
    for kind in EXTRA:
        good = all_good()
        top = chapter_gate(kind, good)
        require(allowed_states(kind, good) == frozenset((top,)), ('positive_case', kind))
        for key in PROMOTION_REQUIREMENTS[kind]:
            bad = dict(good)
            bad[key] = False
            require(sum(bad[x] != good[x] for x in BOOL_KEYS) == 1, 'not_single_change')
            expected = allowed_states(kind, bad)
            actual = chapter_gate(kind, bad)
            require(len(expected) == 1 and actual in expected and top not in expected,
                    ('missing_predicate_not_rejected', kind, key, actual, expected))
            mutant = faulty_ignore_predicate(kind, bad, key, True)
            require(mutant == top and mutant not in expected,
                    ('undetected_single_guard_mutant', kind, key))
            missing_detected += 1
            if (kind, key) in special:
                require(actual == 'CHECKED', ('required_directed_case', kind, key, actual))
                directed.append((kind, key, actual, mutant))

        for key in ('terminal_misreported', 'bad_control_accepted', 'same_scope_conflict'):
            bad = dict(good)
            bad[key] = True
            expected = allowed_states(kind, bad)
            require(len(expected) == 1 and top not in expected, ('adverse_contract', kind, key))
            require(chapter_gate(kind, bad) in expected, ('adverse_gate', kind, key))
            mutant = faulty_ignore_predicate(kind, bad, key, False)
            require(mutant == top and mutant not in expected,
                    ('undetected_adverse_mutant', kind, key))
            adverse_detected += 1

    require(missing_detected == 39, ('missing_mutant_count', missing_detected))
    require(adverse_detected == 12, ('adverse_mutant_count', adverse_detected))
    require(len(directed) == 6, ('directed_case_count', directed))
    return missing_detected, adverse_detected, directed


def priority_edges():
    cases = (
        ('unknown_kind', 'UNKNOWN', {}, 'BLOCKED'),
        ('mech_no_observation', 'MECH', {
            'terminal_complete': False, 'observation': False, 'same_condition': False,
        }, 'SOURCE_SUPPORTED'),
        ('base_conflict_before_missing_terminal', 'OBS', {
            'same_scope_conflict': True, 'terminal_complete': False,
        }, 'CONTESTED'),
        ('misreport_before_conflict', 'OBS', {
            'same_scope_conflict': True, 'terminal_misreported': True,
        }, 'BLOCKED'),
        ('no_evaluation_ignores_both_result_fields', 'TRANSFER', {
            'heldout_changed': False, 'heldout_support': True, 'heldout_conflict': True,
        }, 'CHECKED'),
        ('refutation_before_missing_support_and_changed_reference', 'TRANSFER', {
            'heldout_conflict': True, 'heldout_support': False, 'changed_reference': False,
        }, 'CONTESTED'),
        ('missing_base_check_before_transfer_branch', 'TRANSFER', {
            'base_reference': False, 'heldout_conflict': True,
        }, 'OBSERVED'),
        ('absent_control_is_not_rejected_control', 'BUILD_EFFECT', {
            'base_control_rejected': False,
        }, 'OBSERVED'),
        ('contradictory_results_preserve_refutation', 'TRANSFER', {
            'heldout_support': True, 'heldout_conflict': True,
            'mechanism': False, 'changed_control_rejected': False,
        }, 'CONTESTED'),
        ('executed_but_unresolved', 'TRANSFER', {
            'heldout_support': False, 'heldout_conflict': False,
        }, 'CHECKED'),
    )
    for name, kind, patch, expected in cases:
        f = all_good()
        f.update(patch)
        require(chapter_gate(kind, f) == expected, ('priority_gate', name))
        require(allowed_states(kind, f) == frozenset((expected,)), ('priority_contract', name))
    return len(cases)


def paired_interval(pairs, critical):
    '''Arithmetic model only; caller must establish actual inference assumptions.

    Return None when this model cannot construct the declared positive-variance
    paired t interval. This is not a general statistical inference library.
    '''
    if len(pairs) < 2 or not isfinite(critical) or critical <= 0:
        return None
    differences = []
    for a, b in pairs:
        if not isfinite(a) or not isfinite(b):
            return None
        d = a - b
        if not isfinite(d):
            return None
        differences.append(d)
    n = len(differences)
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
    if not isfinite(mean) or not isfinite(variance) or variance <= 0:
        return None
    half = critical * sqrt(variance / n)
    low, high = mean - half, mean + half
    if not isfinite(low) or not isfinite(high):
        return None
    return mean, variance, (low, high)


def classify_latency_interval(interval):
    '''Frozen example rule for P: latency mean difference A-B < 0.

    Strict negative interval supports; strict positive interval refutes;
    touching/crossing zero or an unavailable interval is unresolved.
    This rule is specific to this P, not all transfer claims.
    '''
    if interval is None:
        return False, False
    low, high = interval
    if not isfinite(low) or not isfinite(high) or low > high:
        return False, False
    if high < 0:
        return True, False
    if low > 0:
        return False, True
    return False, False


def changed_evidence(evaluated, interval):
    if not evaluated:
        return {
            'heldout_changed': False,
            'heldout_support': False,
            'heldout_conflict': False,
        }
    support, conflict = classify_latency_interval(interval)
    return {
        'heldout_changed': True,
        'heldout_support': support,
        'heldout_conflict': conflict,
    }


def interval_transfer_cases():
    # Sixteen constructed differences: sum=0, sum of squares=240.
    # Sample variance is exactly 240/15=16, so s=4 and SE=1.
    differences = (-1, 1) * 5 + (-3, 3, -5, 5, -9, 9)
    exact = tuple(Fraction(x) for x in differences)
    exact_mean = sum(exact) / len(exact)
    exact_variance = sum((x - exact_mean) ** 2 for x in exact) / (len(exact) - 1)
    require(len(exact) == 16 and exact_mean == 0 and exact_variance == 16,
            'independent_fixture_moments')
    # Independent rational reference: supplied critical times known SE=1.
    exact_half = Fraction(2131, 1000)
    cases = (
        ('unresolved', 0, (False, False), 'CHECKED'),
        ('supports', -4, (True, False), 'TRANSFERABLE'),
        ('refutes', 4, (False, True), 'CONTESTED'),
    )
    rows = []
    for name, shift, expected_flags, expected_state in cases:
        pairs = tuple((100.0 + d + shift, 100.0) for d in differences)
        computed = paired_interval(pairs, 2.131)
        require(computed is not None, ('interval_missing', name))
        mean, variance, interval = computed
        exact_low = Fraction(shift) - exact_half
        exact_high = Fraction(shift) + exact_half
        require(isclose(mean, shift, rel_tol=0, abs_tol=1e-12), ('mean', name))
        require(isclose(variance, 16, rel_tol=0, abs_tol=1e-12), ('variance', name))
        require(all(isclose(a, float(b), rel_tol=0, abs_tol=1e-12)
                    for a, b in zip(interval, (exact_low, exact_high))),
                ('independent_interval_reference', name, interval))
        evidence = changed_evidence(True, interval)
        require((evidence['heldout_support'], evidence['heldout_conflict']) == expected_flags,
                ('numeric_decision', name, evidence))
        f = all_good()
        f.update(evidence)
        actual = chapter_gate('TRANSFER', f)
        reference = allowed_states('TRANSFER', f)
        require(actual == expected_state and reference == frozenset((expected_state,)),
                ('numeric_case_promotion', name, actual, reference))
        faulty = faulty_ignore_predicate('TRANSFER', f, 'heldout_support', True)
        if name == 'unresolved':
            require(faulty == 'TRANSFERABLE' and faulty not in reference,
                    'support_only_omission_not_detected')
        if name == 'refutes':
            require(faulty == 'CONTESTED', 'support_mutant_hid_refutation')
        rows.append((name, mean, sqrt(variance), interval, actual, faulty))

    # Independent expected states, including strict boundary equality.
    boundary_cases = (
        ((-2.0, 0.0), 'CHECKED'),
        ((0.0, 2.0), 'CHECKED'),
        ((0.0, 0.0), 'CHECKED'),
        ((-2.0, -0.001), 'TRANSFERABLE'),
        ((0.001, 2.0), 'CONTESTED'),
        (None, 'CHECKED'),
        ((2.0, -2.0), 'CHECKED'),
        ((float('nan'), 1.0), 'CHECKED'),
    )
    for interval, expected in boundary_cases:
        f = all_good()
        f.update(changed_evidence(True, interval))
        require(chapter_gate('TRANSFER', f) == expected, ('interval_boundary_gate', interval))
        require(allowed_states('TRANSFER', f) == frozenset((expected,)),
                ('interval_boundary_certificate', interval))

    # A favorable stale interval cannot provide an unevaluated claim's result.
    not_evaluated = all_good()
    not_evaluated.update(changed_evidence(False, (-6.131, -1.869)))
    require(chapter_gate('TRANSFER', not_evaluated) == 'CHECKED', 'unevaluated_promoted')
    require(allowed_states('TRANSFER', not_evaluated) == frozenset(('CHECKED',)),
            'unevaluated_certificate')

    invalid_inputs = (
        (),
        ((100.0, 100.0),),
        ((100.0, 100.0), (100.0, 100.0)),
        ((float('nan'), 100.0), (101.0, 100.0)),
    )
    for pairs in invalid_inputs:
        result = paired_interval(pairs, 2.131)
        require(result is None, ('invalid_interval_constructed', pairs))
        f = all_good()
        f.update(changed_evidence(True, None))
        require(chapter_gate('TRANSFER', f) == 'CHECKED', 'invalid_interval_promoted')
        require(allowed_states('TRANSFER', f) == frozenset(('CHECKED',)),
                'invalid_interval_certificate')
    return rows, len(boundary_cases), len(invalid_inputs)


def strict_identity(expected, observed):
    '''This small contract requires one successful result per ID.'''
    seen = set()
    for request_id, value, terminal in observed:
        if request_id in seen or request_id not in expected:
            return False
        if terminal != 'success' or value != expected[request_id]:
            return False
        seen.add(request_id)
    return seen == set(expected)


def permissive_identity(expected, observed):
    '''Faulty control: loses identity, cardinality and terminal meaning.'''
    return all(value in expected.values() for _, value, _ in observed)


def identity_cases():
    expected = {'A': 17, 'B': 99}
    cases = (
        [('A', 17, 'success'), ('B', 99, 'success')],
        [('A', 17, 'success'), ('B', 17, 'success')],
        [],
        [('A', 17, 'success'), ('A', 17, 'success'), ('B', 99, 'success')],
        [('A', 17, 'success'), ('B', 99, 'cancelled')],
        [('A', 17, 'success'), ('B', 99, 'success'), ('C', 5, 'success')],
    )
    for index, observed in enumerate(cases):
        require(strict_identity(expected, observed) == (index == 0), ('identity_case', index))
    for index in (1, 2):
        require(permissive_identity(expected, cases[index]), ('identity_mutant_not_activated', index))
        require(not strict_identity(expected, cases[index]), ('identity_mutant_not_detected', index))
    return len(cases)


def reference_independent(graph, root, forbidden):
    '''Reachability on declared synthetic dependencies; no code inspection.'''
    pending = [root]
    visited = set()
    while pending:
        node = pending.pop()
        if node in forbidden:
            return False
        if node in visited:
            continue
        visited.add(node)
        pending.extend(graph.get(node, ()))
    return True


def dependency_cases():
    shared = {'ref': ('wrapper',), 'wrapper': ('tested_walk',), 'tested_walk': ('tokens',)}
    separate = {'ref': ('rebuild',), 'rebuild': ('tokens',), 'tokens': ()}
    cyclic_shared = {'ref': ('a',), 'a': ('ref', 'tested_walk')}
    forbidden = frozenset(('tested_walk',))
    require(not reference_independent(shared, 'ref', forbidden), 'shared_dependency_accepted')
    require(reference_independent(separate, 'ref', forbidden), 'separate_dependency_rejected')
    require(not reference_independent(cyclic_shared, 'ref', forbidden), 'cyclic_shared_dependency_accepted')
    name_only_mutant = 'ref' not in forbidden
    require(name_only_mutant and not reference_independent(shared, 'ref', forbidden),
            'dependency_negative_control_not_detected')
    return 3


def cost_example():
    # Independent event ledger versus repetition expansion.
    builds = [Fraction(120), Fraction(120)]
    run_setup = [Fraction(10)] * 4
    work = [Fraction(2)] * 12
    observe = [Fraction(1, 2)] * 12
    transfers = [Fraction(1)] * 4
    other = [Fraction(60), Fraction(30), Fraction(6), Fraction(180)]
    ledger = sum(builds + run_setup + work + observe + transfers + other)
    n_b, runs_per_build, samples_per_run = 2, 2, 3
    n_run = n_b * runs_per_build
    n_measure = n_run * samples_per_run
    expanded = (60 + n_b * 120 + n_run * 10
                + n_measure * (2 + Fraction(1, 2)) + 4 * 1 + 30 + 6 + 180)
    require(ledger == expanded == 590, ('serial_cost', ledger, expanded))
    storage_gib_seconds = 2 * 86400
    budget = Fraction(410, 100) + Fraction(180 * 2, 100) + Fraction(2, 1000)
    loss = 1000 * Fraction(1, 100) + 100 * Fraction(2, 100) + budget
    require(budget == Fraction(7702, 1000), ('money_cost', budget))
    require(loss == Fraction(19702, 1000), ('expected_loss', loss))
    faulty_double_count = expanded + n_measure * 2
    require(faulty_double_count != ledger, 'double_count_negative_control_not_detected')
    return ledger, storage_gib_seconds, budget, loss


def main():
    require(len(BOOL_KEYS) == 24, ('predicate_count', len(BOOL_KEYS)))
    require(set(FIELD_RULES) == set(BOOL_KEYS), 'field_mapping_incomplete')
    profiles, evaluations, by_kind = exhaustive_check()
    missing, adverse, directed = mutation_check()
    edges = priority_edges()
    intervals, boundaries, invalid_intervals = interval_transfer_cases()
    identities = identity_cases()
    dependencies = dependency_cases()
    seconds, residence, money, loss = cost_example()

    print('SCOPED K-SYS-006 Python stdlib model check')
    print('predicates={0}'.format(len(BOOL_KEYS)))
    print('profiles_by_kind={0}'.format(by_kind))
    print('profiles_checked={0}; state_evaluations={1}'.format(profiles, evaluations))
    print('certificate_nonunique_or_missing=0; state_mismatches=0; irrelevant_field_failures=0')
    print('single_missing_predicate_mutants_detected={0}'.format(missing))
    print('adverse_predicate_mutants_detected={0}'.format(adverse))
    print('priority_edges={0}'.format(edges))
    for kind, field, expected, faulty in directed:
        print('directed={0}/{1}: expected={2}, faulty={3}'.format(kind, field, expected, faulty))
    for name, mean, sd, interval, state, faulty in intervals:
        print('transfer_interval={0}: n=16, mean={1}, sd={2}, CI={3}, state={4}, omit_support={5}'.format(
            name, mean, sd, interval, state, faulty))
    print('interval_boundary_cases={0}; invalid_interval_inputs={1}; unevaluated_case=1'.format(
        boundaries, invalid_intervals))
    print('support_only_omission_negative_control_detected=True')
    print('identity_cases={0}; identity_faulty_acceptances_detected=2'.format(identities))
    print('synthetic_dependency_graphs={0}'.format(dependencies))
    print('serial_seconds={0}; retained_GiB_seconds={1}'.format(seconds, residence))
    print('money_cost={0}; expected_money_loss={1}'.format(money, loss))
    print('PASS scoped: certificates, interval decisions, support-guard mutant, priorities, identity and cost')
    print('LIMIT: no statistical coverage, real evidence extraction/dependencies, device behavior or Agent benefits')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('FAIL scoped: {0}: {1}'.format(type(exc).__name__, exc), file=sys.stderr)
        sys.exit(1)
