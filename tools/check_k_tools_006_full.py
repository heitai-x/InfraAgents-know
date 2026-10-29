from fractions import Fraction as F
from copy import deepcopy

PARTS = ('Free', 'Active', 'Inflight', 'Retained', 'Quarantined')


class Suite:
    def __init__(self):
        self.checks = 0
        self.negatives = []

    def check(self, name, condition, detail=None):
        self.checks += 1
        if not condition:
            raise AssertionError((name, detail))

    def clean(self, name, issues):
        self.check(name, not issues, issues)

    def reject(self, name, issues, tag):
        self.check(name, any(x[0] == tag for x in issues), issues)
        self.negatives.append(name)

    def invalid(self, name, action):
        try:
            action()
        except ValueError:
            self.check(name, True)
            self.negatives.append(name)
            return
        self.check(name, False, 'invalid input was accepted')


def validate_identity(expected, results, terminals):
    issues = []
    if not expected:
        issues.append(('no_obligations',))
    for kind, records, payload in (
        ('result', results, 'value'),
        ('terminal', terminals, 'status'),
    ):
        seen = set()
        for rec in records:
            rid = rec.get('rid')
            if rid in seen:
                issues.append(('duplicate_' + kind, rid))
            seen.add(rid)
            if rid not in expected:
                issues.append(('extra_' + kind, rid))
                continue
            want = expected[rid]
            for field in ('attempt', 'slot', 'epoch'):
                if rec.get(field) != want[field]:
                    issues.append((kind + '_' + field, rid,
                                   want[field], rec.get(field)))
            if rec.get(payload) != want[payload]:
                issues.append((kind + '_' + payload, rid,
                               want[payload], rec.get(payload)))
        for rid in sorted(set(expected) - seen):
            issues.append(('missing_' + kind, rid))
    return issues


def validate_distribution(p, allowed):
    issues = []
    if not p:
        issues.append(('empty_distribution',))
    for i, probability in enumerate(p):
        if probability < 0:
            issues.append(('negative_probability', i))
    if sum(p, F(0)) != 1:
        issues.append(('sum_not_one',))
    if any(probability > 0 and i not in allowed
           for i, probability in enumerate(p)):
        issues.append(('support_outside_allowed',))
    return issues


def cdf_sample(p, u):
    if validate_distribution(p, set(range(len(p)))) or not 0 <= u < 1:
        raise ValueError('CDF domain')
    cumulative = F(0)
    for i, probability in enumerate(p):
        cumulative += probability
        if u < cumulative:
            return i
    raise ValueError('CDF coverage gap')


def faulty_nonstrict_cdf(p, u):
    if validate_distribution(p, set(range(len(p)))) or not 0 <= u < 1:
        raise ValueError('CDF domain')
    cumulative = F(0)
    for i, probability in enumerate(p):
        cumulative += probability
        if u <= cumulative:
            return i
    raise ValueError('CDF coverage gap')


def compare_sampler(sampler, cases):
    issues = []
    for name, p, u, expected in cases:
        actual = sampler(p, u)
        if actual != expected:
            issues.append(('cdf_choice', name, expected, actual))
        if not 0 <= actual < len(p) or p[actual] <= 0:
            issues.append(('cdf_support', name, actual))
    return issues


def weighted_reference(tokens, weights):
    if len(tokens) != len(weights):
        raise ValueError('reference length')
    return sum((token * weight for token, weight in zip(tokens, weights)), F(0))


def chunked_model(tokens, weights, chunks, reset=False):
    if len(tokens) != len(weights):
        raise ValueError('model length')
    cursor = 0
    total = F(0)
    for start, end in chunks:
        if not (start == cursor and 0 <= start <= end <= len(tokens)):
            raise ValueError('chunk partition')
        for i in range(start, end):
            position = i - start if reset else i
            total += tokens[i] * weights[position]
        cursor = end
    if cursor != len(tokens):
        raise ValueError('incomplete chunks')
    return total


def validate_trace(events, expected_body):
    state = 'init'
    starts = 0
    terminals = 0
    next_seq = 0
    chunks = []
    issues = []
    for event in events:
        kind = event['type']
        if kind == 'start':
            starts += 1
            if state != 'init':
                issues.append(('bad_start_state', state))
            state = 'started'
        elif kind == 'data':
            if state != 'started':
                issues.append(('data_outside_started', state))
            if event['seq'] != next_seq:
                issues.append(('bad_sequence', next_seq, event['seq']))
            next_seq += 1
            chunks.append(event['body'])
        elif kind in ('end', 'error'):
            if state != 'started':
                issues.append(('bad_terminal_state', state))
            terminals += 1
            state = 'terminal'
        else:
            issues.append(('unknown_event', kind))
    if starts != 1:
        issues.append(('start_count', starts))
    if terminals != 1:
        issues.append(('terminal_count', terminals))
    if b''.join(chunks) != expected_body:
        issues.append(('body_mismatch',))
    return issues


def resource_state(epoch, owners=(), uses=(), outputs=(), bindings=(),
                   holds=(), retained=(), quarantine=(), closed=True):
    return dict(epoch=epoch, owners=owners, uses=uses, outputs=outputs,
                bindings=bindings, holds=holds, retained=retained,
                quarantine=quarantine, closed=closed)


def classify_resources(total, raw):
    parts = {name: [] for name in PARTS}
    for slot in total:
        item = raw[slot]
        if item['quarantine']:
            category = 'Quarantined'
        elif item['uses'] or item['outputs']:
            category = 'Inflight'
        elif (item['owners'] or item['bindings'] or item['holds']
              or not item['closed']):
            category = 'Active'
        elif item['retained']:
            category = 'Retained'
        else:
            category = 'Free'
        parts[category].append(slot)
    return parts


def validate_resources(total, raw, parts, access_ledger, quiescent=False):
    issues = []
    domain = set(total)
    if set(raw) != domain:
        issues.append(('raw_domain',))
    if set(access_ledger) != domain:
        issues.append(('ledger_domain',))
    seen = {}
    for category in PARTS:
        for slot in parts[category]:
            if slot in seen:
                issues.append(('duplicate_physical', slot))
            seen[slot] = category
    if set(seen) != domain:
        issues.append(('union_mismatch',))
    capacity = sum(total[s] for category in PARTS
                   for s in parts[category] if s in total)
    if capacity != sum(total.values()):
        issues.append(('capacity_mismatch', capacity))
    for slot in sorted(domain & set(raw) & set(access_ledger)):
        item = raw[slot]
        # The ledger is supplied independently of classification and raw snapshots.
        for field in ('uses', 'outputs'):
            actual = tuple(item[field])
            expected = tuple(access_ledger[slot][field])
            if len(actual) != len(set(actual)):
                issues.append(('duplicate_access', slot, field))
            if set(actual) != set(expected):
                issues.append(('access_ledger', slot, field))
            for ticket in actual:
                # Ticket identity: (rid, attempt, physical slot, epoch, use id).
                if ticket[2] != slot or ticket[3] != item['epoch']:
                    issues.append(('ticket_identity', slot, ticket))
        isolated = bool(item['quarantine'])
        accessing = bool(item['uses'] or item['outputs'])
        owned = bool(item['owners'] or item['bindings'] or item['holds']
                     or not item['closed'])
        kept = bool(item['retained'])
        # Predicates verify the meaning of each emitted category, not just its union.
        valid_category = {
            'Quarantined': isolated,
            'Inflight': not isolated and accessing,
            'Active': not isolated and not accessing and owned,
            'Retained': not isolated and not accessing and not owned and kept,
            'Free': not isolated and not accessing and not owned and not kept,
        }
        if slot in seen and not valid_category[seen[slot]]:
            issues.append(('category_mismatch', slot, seen[slot]))
        if quiescent and (isolated or accessing or owned):
            issues.append(('not_quiescent', slot))
    return issues


def access(uses=(), outputs=()):
    return dict(uses=uses, outputs=outputs)


def stable_top1(scores, epsilon):
    if len(scores) < 2 or epsilon < 0:
        raise ValueError('margin domain')
    ordered = sorted(scores, reverse=True)
    margin = ordered[0] - ordered[1]
    return margin > 2 * epsilon


def main():
    suite = Suite()

    # Frozen expected authorization; neither result nor terminal creates this table.
    expected = {
        'A': dict(attempt=2, slot='a', epoch=8, value=F(17), status='ok'),
        'B': dict(attempt=4, slot='b', epoch=3, value=F(99), status='ok'),
    }
    results = [
        dict(rid='A', attempt=2, slot='a', epoch=8, value=F(17)),
        dict(rid='B', attempt=4, slot='b', epoch=3, value=F(99)),
    ]
    terminals = [
        dict(rid='A', attempt=2, slot='a', epoch=8, status='ok'),
        dict(rid='B', attempt=4, slot='b', epoch=3, status='ok'),
    ]
    suite.clean('identity_baseline', validate_identity(expected, results, terminals))
    suite.clean('identity_reordered', validate_identity(
        expected, list(reversed(results)), list(reversed(terminals))))
    for kind in ('result', 'terminal'):
        for field, old in (('attempt', 1), ('slot', 'old-slot'), ('epoch', 7)):
            changed_results = deepcopy(results)
            changed_terminals = deepcopy(terminals)
            target = changed_results if kind == 'result' else changed_terminals
            target[0][field] = old
            suite.reject(kind + '_only_' + field,
                         validate_identity(expected, changed_results, changed_terminals),
                         kind + '_' + field)
    wrong_content = deepcopy(results)
    wrong_content[1]['value'] = F(17)
    suite.reject('content_alias', validate_identity(expected, wrong_content, terminals),
                 'result_value')
    suite.reject('missing_result', validate_identity(expected, results[:1], terminals),
                 'missing_result')
    suite.reject('duplicate_result', validate_identity(expected, results + [results[0]], terminals),
                 'duplicate_result')
    extra = dict(rid='C', attempt=1, slot='c', epoch=1, value=F(17))
    suite.reject('extra_result', validate_identity(expected, results + [extra], terminals),
                 'extra_result')
    suite.reject('duplicate_terminal', validate_identity(expected, results, terminals + [terminals[0]]),
                 'duplicate_terminal')
    suite.reject('missing_terminal', validate_identity(expected, results, terminals[:1]),
                 'missing_terminal')
    wrong_status = deepcopy(terminals)
    wrong_status[0]['status'] = 'cancelled'
    suite.reject('wrong_but_enumerated_status', validate_identity(expected, results, wrong_status),
                 'terminal_status')
    suite.reject('empty_obligations', validate_identity({}, [], []), 'no_obligations')

    tokens = list(map(F, (1, 2, 3, 4)))
    weights = list(map(F, (1, 3, 5, 7)))
    reference = weighted_reference(tokens, weights)
    suite.check('independent_position_value', reference == 50, reference)
    # All eight partitions of four positions, with legal empty endpoint chunks.
    for mask in range(8):
        cuts = [0] + [i for i in range(1, 4) if mask & (1 << (i - 1))] + [4]
        chunks = [(0, 0)] + list(zip(cuts, cuts[1:])) + [(4, 4)]
        suite.check('chunk_partition_' + str(mask),
                    chunked_model(tokens, weights, chunks) == reference)
    wrong = chunked_model(tokens, weights, [(0, 2), (2, 4)], reset=True)
    suite.reject('position_reset', [] if wrong == reference else [('position_mismatch', wrong)],
                 'position_mismatch')
    suite.check('empty_arithmetic', weighted_reference([], []) == 0
                and chunked_model([], [], [(0, 0)]) == 0)
    suite.invalid('chunk_gap', lambda: chunked_model(tokens, weights, [(0, 1), (2, 4)]))
    suite.invalid('chunk_incomplete', lambda: chunked_model(tokens, weights, [(0, 3)]))

    p = (F(1, 5), F(3, 10), F(1, 2))
    # Expected choices are literal interval examples, not calls to a second CDF loop.
    cases = [
        ('zero', p, F(0), 0),
        ('before_first_boundary', p, F(199, 1000), 0),
        ('first_boundary', p, F(1, 5), 1),
        ('before_second_boundary', p, F(499, 1000), 1),
        ('second_boundary', p, F(1, 2), 2),
        ('tail', p, F(999, 1000), 2),
        ('leading_zero', (F(0), F(1)), F(0), 1),
        ('internal_zero', (F(0), F(1, 5), F(0), F(4, 5), F(0)), F(1, 5), 3),
        ('trailing_zero', (F(1), F(0)), F(999, 1000), 0),
        ('singleton', (F(1),), F(0), 0),
    ]
    suite.clean('cdf_literal_boundaries', compare_sampler(cdf_sample, cases))
    suite.reject('cdf_nonstrict_boundary_mutant',
                 compare_sampler(faulty_nonstrict_cdf, cases), 'cdf_choice')
    suite.reject('cdf_nonstrict_support_mutant',
                 compare_sampler(faulty_nonstrict_cdf, cases), 'cdf_support')
    # Independent finite RNG grid: ten equally weighted outcomes with explicit bins.
    bins = (0, 0, 1, 1, 1, 2, 2, 2, 2, 2)
    suite.check('cdf_tenths_grid',
                tuple(cdf_sample(p, F(k, 10)) for k in range(10)) == bins)
    suite.clean('probability_valid', validate_distribution(p, {0, 1, 2}))
    suite.reject('negative_probability', validate_distribution((F(-1, 10), F(11, 10)), {0, 1}),
                 'negative_probability')
    suite.reject('wrong_probability_sum', validate_distribution((F(1, 3), F(1, 3)), {0, 1}),
                 'sum_not_one')
    suite.reject('forbidden_support', validate_distribution((F(0), F(1)), {0}),
                 'support_outside_allowed')
    suite.invalid('cdf_u_one', lambda: cdf_sample(p, F(1)))
    suite.invalid('cdf_u_negative', lambda: cdf_sample(p, F(-1, 10)))
    suite.invalid('cdf_empty', lambda: cdf_sample((), F(0)))
    suite.invalid('cdf_zero_mass', lambda: cdf_sample((F(0), F(0)), F(0)))

    events = [dict(type='start'), dict(type='data', seq=0, body=b'ab'),
              dict(type='data', seq=1, body=b'cd'), dict(type='end')]
    suite.clean('protocol_baseline', validate_trace(events, b'abcd'))
    suite.clean('protocol_empty_body', validate_trace([dict(type='start'), dict(type='end')], b''))
    suite.reject('protocol_duplicate_end', validate_trace(events + [dict(type='end')], b'abcd'),
                 'terminal_count')
    suite.reject('protocol_missing_end', validate_trace(events[:-1], b'abcd'), 'terminal_count')
    suite.reject('protocol_after_end', validate_trace(events + [dict(type='data', seq=2, body=b'')], b'abcd'),
                 'data_outside_started')
    wrong_sequence = deepcopy(events)
    wrong_sequence[2]['seq'] = 2
    suite.reject('protocol_sequence', validate_trace(wrong_sequence, b'abcd'), 'bad_sequence')
    suite.reject('protocol_body', validate_trace(events, b'abce'), 'body_mismatch')

    # Physical unit sizes in bytes. Q remains an unresolved responsibility, not free.
    total = {'P': 4096, 'a': 4096, 'F': 4096, 'R': 8192, 'Q': 4096}
    tb = ('B', 4, 'P', 3, 'kernel-B')
    ta = ('A', 1, 'a', 7, 'dma-A')
    tq = ('Z', 5, 'Q', 9, 'external-reader')
    raw = {
        'P': resource_state(3, owners=('A', 'B'), uses=(tb,), closed=False),
        'a': resource_state(7, owners=('A',), uses=(ta,), closed=False),
        'F': resource_state(0),
        'R': resource_state(2, retained=('prefix-policy',)),
        'Q': resource_state(9, uses=(tq,), quarantine=('missing-last-access',)),
    }
    # Literal expected access ledger, independent of raw/classifier construction.
    ledger = {'P': access((tb,)), 'a': access((ta,)), 'F': access(),
              'R': access(), 'Q': access((tq,))}
    expected_parts = {'Free': ['F'], 'Active': [], 'Inflight': ['P', 'a'],
                      'Retained': ['R'], 'Quarantined': ['Q']}
    parts = classify_resources(total, raw)
    suite.check('resource_manual_partition', parts == expected_parts, parts)
    suite.clean('resource_shared_reader', validate_resources(total, raw, parts, ledger))
    suite.check('resource_orthogonal_fields', raw['P']['owners'] == ('A', 'B')
                and raw['P']['uses'] == (tb,))
    byte_counts = {name: sum(total[s] for s in parts[name]) for name in PARTS}
    suite.check('resource_capacity_reference', byte_counts == {
        'Free': 4096, 'Active': 0, 'Inflight': 8192,
        'Retained': 8192, 'Quarantined': 4096}, byte_counts)
    wrong_category = deepcopy(parts)
    wrong_category['Inflight'].remove('P')
    wrong_category['Active'].append('P')
    suite.check('wrong_category_still_unique_complete',
                sorted(s for name in PARTS for s in wrong_category[name]) == sorted(total))
    suite.reject('resource_active_only_mutant',
                 validate_resources(total, raw, wrong_category, ledger), 'category_mismatch')
    lost_use = deepcopy(raw)
    lost_use['P']['uses'] = ()
    suite.reject('resource_erased_use_mutant', validate_resources(
        total, lost_use, classify_resources(total, lost_use), ledger), 'access_ledger')
    duplicate = deepcopy(parts)
    duplicate['Active'].append('P')
    suite.reject('resource_double_shared', validate_resources(total, raw, duplicate, ledger),
                 'duplicate_physical')
    missing = deepcopy(parts)
    missing['Inflight'].remove('a')
    suite.reject('resource_missing_inflight', validate_resources(total, raw, missing, ledger),
                 'union_mismatch')
    wrong_epoch = deepcopy(raw)
    wrong_epoch['P']['epoch'] = 2
    suite.reject('resource_ticket_epoch', validate_resources(total, wrong_epoch, parts, ledger),
                 'ticket_identity')
    suite.reject('resource_false_quiescence', validate_resources(total, raw, parts, ledger, True),
                 'not_quiescent')

    cancelled = deepcopy(raw)
    cancelled['P']['owners'] = ('B',)
    cancelled['a']['owners'] = ()
    cancelled['a']['closed'] = True
    suite.check('cancel_preserves_inflight_partition',
                classify_resources(total, cancelled) == expected_parts)
    suite.clean('cancel_preserves_tickets', validate_resources(
        total, cancelled, expected_parts, ledger))
    completed_b = deepcopy(cancelled)
    completed_b['P']['uses'] = ()
    completed_ledger = deepcopy(ledger)
    completed_ledger['P'] = access()
    after_b = {'Free': ['F'], 'Active': ['P'], 'Inflight': ['a'],
               'Retained': ['R'], 'Quarantined': ['Q']}
    suite.check('reader_done_owner_remains', classify_resources(total, completed_b) == after_b)
    suite.clean('reader_done_snapshot', validate_resources(total, completed_b, after_b, completed_ledger))
    drained = deepcopy(completed_b)
    drained['P']['owners'] = ()
    drained['P']['closed'] = True
    drained['P']['retained'] = ('prefix-policy-P',)
    drained['a']['uses'] = ()
    drained['Q']['uses'] = ()
    drained['Q']['quarantine'] = ()
    drained_ledger = {'P': access(), 'a': access(), 'F': access(), 'R': access(), 'Q': access()}
    final_parts = {'Free': ['a', 'F', 'Q'], 'Active': [], 'Inflight': [],
                   'Retained': ['P', 'R'], 'Quarantined': []}
    suite.check('drained_manual_partition', classify_resources(total, drained) == final_parts)
    suite.clean('drained_quiescent_with_retention', validate_resources(
        total, drained, final_parts, drained_ledger, True))

    suite.check('margin_certified', stable_top1((F(5), F(4)), F(2, 5)))
    suite.check('margin_equality_unproven', not stable_top1((F(5), F(4)), F(1, 2)))
    suite.check('margin_below_threshold', not stable_top1((F(5), F(4)), F(3, 5)))

    print('PASS: finite host-side logical oracle checks')
    print('checks=', suite.checks)
    print('negative_controls_detected=', len(suite.negatives))
    print('identity=rid,attempt,slot,epoch,value,terminal-status; one authorized result per request')
    print('cdf=exact Fraction intervals and explicit finite grid; no empirical frequency test')
    print('resources=manual snapshots, byte partition, independent access ledger; no device completion adapter')
    print('scope=no project imports, engine, GPU, kernel, performance, task-quality, or Agent-effect measurement')


if __name__ == '__main__':
    main()
