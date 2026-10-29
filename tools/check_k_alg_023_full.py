from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import permutations, product


def q(value):
    return Fraction(str(value))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


@dataclass(frozen=True)
class Candidate:
    id: str
    eta: str
    parts: tuple = (0, 0, 0, 0, 0, 0)
    alive: bool = True
    accepting: bool = True
    tenant: bool = True
    group_ready: bool = True
    epoch_valid: bool = True
    input_ready: bool = True
    output_ready: bool = True
    prefix_valid: bool = True
    prefix: int = 0
    source_ready: bool = True
    source_protected: bool = True
    available: int = 8
    need: int = 4


def score(c):
    return sum((q(x) for x in c.parts), Fraction(0))


def eligible(c, req):
    if req.get('terminal', False) or c.eta != req['eta']:
        return False
    if not (c.alive and c.accepting and c.tenant):
        return False
    if not (c.group_ready and c.epoch_valid):
        return False
    if not (c.input_ready and c.output_ready and c.prefix_valid):
        return False
    if not 0 <= c.prefix <= req['n']:
        return False
    if c.prefix > 0 and not (c.source_ready and c.source_protected):
        return False
    if c.need < 0 or c.available < c.need:
        return False
    if len(c.parts) != 6 or any(q(x) < 0 for x in c.parts):
        return False
    return True


def route(candidates, req, now=0, deadline=1000,
          allow_wait=False, wait_until=0):
    best = None
    for c in candidates:
        if not eligible(c, req):
            continue
        cost = score(c)
        if q(now) + cost > q(deadline):
            continue
        key = (cost, c.id)
        if best is None or key < best:
            best = key
    if best is not None:
        return ('ADMIT', best[1], best[0], None)
    end = min(q(deadline), q(wait_until))
    if not req.get('terminal', False) and allow_wait and q(now) < end:
        return ('WAIT', None, None, end)
    return ('REJECT', None, None, None)


def route_reference(candidates, req, now=0, deadline=1000,
                    allow_wait=False, wait_until=0):
    # Independent set intersection, without eligible(), score() or route().
    indices = set(range(len(candidates)))
    if req.get('terminal', False):
        indices.clear()
    indices &= {i for i, c in enumerate(candidates) if c.eta == req['eta']}
    for field in ('alive', 'accepting', 'tenant', 'group_ready',
                  'epoch_valid', 'input_ready', 'output_ready', 'prefix_valid'):
        indices &= {i for i, c in enumerate(candidates) if getattr(c, field)}
    indices &= {i for i, c in enumerate(candidates)
                if 0 <= c.prefix <= req['n']}
    indices &= {i for i, c in enumerate(candidates)
                if c.prefix == 0 or (c.source_ready and c.source_protected)}
    indices &= {i for i, c in enumerate(candidates)
                if 0 <= c.need <= c.available}
    options = []
    for i in indices:
        values = tuple(Fraction(str(x)) for x in candidates[i].parts)
        if len(values) != 6 or any(x < 0 for x in values):
            continue
        finish = Fraction(str(now))
        for duration in values:
            finish += duration
        if finish <= Fraction(str(deadline)):
            options.append((finish - Fraction(str(now)), candidates[i].id))
    if options:
        winner = min(options)
        return ('ADMIT', winner[1], winner[0], None)
    end = min(Fraction(str(deadline)), Fraction(str(wait_until)))
    if allow_wait and not req.get('terminal', False) and Fraction(str(now)) < end:
        return ('WAIT', None, None, end)
    return ('REJECT', None, None, None)


def route_cost_only_bug(candidates):
    cost, cid = min((score(c), c.id) for c in candidates)
    return ('ADMIT', cid, cost, None)


def migration(D0, g, B, alpha=0, eps=1, max_stages=100,
              alpha_final=0, handoff=0, late_guard=False):
    D0, g, B, alpha, eps, alpha_final, handoff = map(
        q, (D0, g, B, alpha, eps, alpha_final, handoff))
    if (D0 < 0 or g < 0 or B <= 0 or alpha < 0 or eps < 0
            or alpha_final < 0 or handoff < 0
            or not isinstance(max_stages, int) or max_stages < 0):
        raise ValueError('invalid migration parameters')
    delta = D0
    stages, events = [], []
    elapsed, moved = Fraction(0), Fraction(0)
    while True:
        count = len(stages)
        events.append(('ENTRY', count, delta))
        if not late_guard:
            events.append(('CHECK_THRESHOLD', count, delta))
            if delta <= eps:
                break
            events.append(('CHECK_BUDGET', count, max_stages))
            if count >= max_stages:
                break
        # Deliberate mutation late_guard=True performs real modeled work
        # before either admission guard, reproducing the reviewed defect.
        events.append(('RESERVE', count, delta))
        duration = alpha + delta / B
        events.append(('COPY', count, delta, duration))
        stages.append((delta, duration))
        moved += delta
        elapsed += duration
        delta = g * duration
        if late_guard:
            events.append(('CHECK_THRESHOLD', len(stages), delta))
            if delta <= eps:
                break
            events.append(('CHECK_BUDGET', len(stages), max_stages))
            if len(stages) >= max_stages:
                break
    reached = delta <= eps
    if reached:
        events.append(('FREEZE', len(stages), delta))
        events.append(('FINAL_RESERVE', delta))
        events.append(('FINAL_COPY', delta))
        downtime = alpha_final + delta / B + handoff
        moved += delta
        status = 'STOP_AND_HANDOFF'
    else:
        events.append(('ABORT', len(stages), delta))
        downtime = Fraction(0)
        status = 'ABORT'
    return {
        'status': status, 'reached': reached, 'stages': tuple(stages),
        'events': tuple(events), 'delta': delta, 'precopy': elapsed,
        'downtime': downtime, 'wall': elapsed + downtime, 'bytes': moved
    }


def closed_reference(D0, g, B, alpha=0, eps=1, max_stages=100,
                     alpha_final=0, handoff=0):
    # Solve the affine recurrence directly; never invoke migration().
    D0, g, B, alpha, eps, alpha_final, handoff = (
        Fraction(str(x)) for x in
        (D0, g, B, alpha, eps, alpha_final, handoff))
    ratio, extra = g / B, g * alpha

    def at(k):
        if ratio == 1:
            return D0 + k * extra
        return ratio ** k * D0 + extra * (1 - ratio ** k) / (1 - ratio)

    feasible = [k for k in range(max_stages + 1) if at(k) <= eps]
    rounds = min(feasible) if feasible else max_stages
    reached = bool(feasible)
    before = sum((at(k) for k in range(rounds)), Fraction(0))
    precopy = rounds * alpha + before / B
    stop = alpha_final + at(rounds) / B + handoff if reached else Fraction(0)
    expected_events = []
    # Event admission is derived from the closed-form earliest stopping
    # index, independent of the implementation's loop exit predicates.
    for k in range(rounds):
        require(at(k) > eps and k < max_stages, 'reference admitted invalid stage')
        expected_events.extend((
            ('ENTRY', k, at(k)),
            ('CHECK_THRESHOLD', k, at(k)),
            ('CHECK_BUDGET', k, max_stages),
            ('RESERVE', k, at(k)),
            ('COPY', k, at(k), alpha + at(k) / B)))
    expected_events.extend((('ENTRY', rounds, at(rounds)),
                            ('CHECK_THRESHOLD', rounds, at(rounds))))
    if reached:
        expected_events.extend((('FREEZE', rounds, at(rounds)),
                                ('FINAL_RESERVE', at(rounds)),
                                ('FINAL_COPY', at(rounds))))
    else:
        expected_events.extend((('CHECK_BUDGET', rounds, max_stages),
                                ('ABORT', rounds, at(rounds))))
    return {
        'rounds': rounds, 'delta': at(rounds), 'reached': reached,
        'precopy': precopy, 'downtime': stop, 'wall': precopy + stop,
        'bytes': before + (at(rounds) if reached else 0),
        'status': 'STOP_AND_HANDOFF' if reached else 'ABORT',
        'stages': tuple((at(k), alpha + at(k) / B) for k in range(rounds)),
        'events': tuple(expected_events)
    }


def audit_migration(actual, args):
    expected = closed_reference(**args)
    require(len(actual['stages']) == expected['rounds'], 'stage count mismatch')
    for field in ('delta', 'reached', 'precopy', 'downtime', 'wall',
                  'bytes', 'status', 'stages', 'events'):
        require(actual[field] == expected[field], 'migration mismatch: ' + field)
    if actual['reached']:
        require(actual['bytes'] == q(args['D0']) + q(args['g']) * actual['precopy'],
                'successful-copy byte conservation failed')
    else:
        require(actual['downtime'] == 0, 'abort performed final stop')
        require(actual['bytes'] == sum((d for d, _ in actual['stages']), Fraction(0)),
                'abort charged uncopied final delta')


def check_migration(args):
    actual = migration(**args)
    audit_migration(actual, args)
    return actual


def expect_rejected(actual, args, label):
    try:
        audit_migration(actual, args)
    except AssertionError:
        return
    raise AssertionError('negative control escaped audit: ' + label)


def range_copy(initial_end, growths, omit_initial=False):
    # Integer logical positions; independent of continuous byte arithmetic.
    end = initial_end
    frontier = initial_end if omit_initial else 0
    copied, ranges = set(), []
    for growth in growths:
        upper = end
        ranges.append((frontier, upper))
        copied.update(range(frontier, upper))
        frontier = upper
        end += growth
    ranges.append((frontier, end))
    copied.update(range(frontier, end))
    return copied, end, ranges


def summary(rec):
    values = {key: (str(value) if isinstance(value, Fraction) else value)
              for key, value in rec.items() if key not in ('stages', 'events')}
    values['rounds'] = len(rec['stages'])
    values['event_names'] = tuple(event[0] for event in rec['events'])
    return values


def run():
    req = {'eta': 'm1', 'n': 16}
    try:
        Candidate('missing-eta')
    except TypeError:
        pass
    else:
        raise AssertionError('candidate eta must be explicit')
    A = Candidate('A', req['eta'], (100, 0, 0, 0, 10, 0), prefix=8)
    B = Candidate('B', req['eta'], (0, 0, 80, 0, 0, 0))
    C = Candidate('C', req['eta'], (10, 0, 0, 40, 10, 0), prefix=8)
    base = (A, B, C)
    require(route(base, req) == ('ADMIT', 'C', Fraction(60), None), 'base C at 60 ms')
    print('OBSERVATION base_route', route(base, req))
    wrong_fixture = tuple(replace(c, eta='m') for c in base)
    require(route(wrong_fixture, req)[0] == 'REJECT', 'wrong eta must reject')
    require(route(wrong_fixture, req) != route(base, req), 'identity fixture defect undetected')
    print('OBSERVATION prior_fixture_negative_control', route(wrong_fixture, req))

    fields = ('alive', 'accepting', 'tenant', 'group_ready', 'epoch_valid',
              'input_ready', 'output_ready', 'prefix_valid',
              'source_ready', 'source_protected')
    route_cases = 0
    for bits in product((False, True), repeat=len(fields)):
        changed = replace(C, **dict(zip(fields, bits)))
        candidates = (A, B, changed)
        actual = route(candidates, req)
        require(actual == route_reference(candidates, req), 'route gate reference')
        require(actual[1] == ('C' if all(bits) else 'B'), 'gate combination choice')
        route_cases += 1
    for available in (3, 4, 5):
        candidates = (A, B, replace(C, available=available))
        actual = route(candidates, req)
        require(actual == route_reference(candidates, req), 'capacity reference')
        require(actual[1] == ('B' if available < C.need else 'C'), 'capacity boundary')
    for field in ('source_ready', 'source_protected'):
        cold = replace(B, **{field: False})
        require(route((cold,), req)[1] == 'B', 'cold plan requires nonexistent cache')
    for prefix in (-1, 17):
        require(route((replace(C, prefix=prefix), B), req)[1] == 'B', 'prefix range')
    require(route((replace(C, eta='m2'), B), req)[1] == 'B', 'eta guard')
    require(route((replace(C, parts=(-1, 0, 0, 0, 0, 0)), B), req)[1] == 'B',
            'negative estimate guard')
    require(route((), req)[0] == 'REJECT', 'empty candidates')
    for order in permutations(base):
        require(route(order, req) == route_reference(order, req), 'permutation reference')
        require(route(order, req)[1] == 'C', 'permutation changes minimum')
    ties = (replace(A, parts=(10, 0, 0, 0, 0, 0)),
            replace(B, parts=(10, 0, 0, 0, 0, 0)))
    for order in permutations(ties):
        require(route(order, req)[1] == 'A', 'stable tie')
    for now, deadline, expected_action in ((0, 60, 'ADMIT'), (1, 60, 'REJECT'),
                                           (20, 80, 'ADMIT'), (20, 79, 'REJECT')):
        kwargs = {'now': now, 'deadline': deadline}
        require(route(base, req, **kwargs) == route_reference(base, req, **kwargs),
                'absolute deadline reference')
        require(route(base, req, **kwargs)[0] == expected_action, 'absolute deadline')
    for now in (0, 4, 5, 9):
        kwargs = {'now': now, 'deadline': 100, 'allow_wait': True, 'wait_until': 5}
        actual = route((), req, **kwargs)
        require(actual == route_reference((), req, **kwargs), 'wait reference')
        require(actual[0] == ('WAIT' if now < 5 else 'REJECT'), 'wait budget reset')
    require(route((), req, now=3, deadline=4, allow_wait=True, wait_until=9)[3] == 4,
            'wait exceeds hard deadline')
    require(route(base, dict(req, terminal=True), allow_wait=True, wait_until=10)[0] == 'REJECT',
            'terminal request waits or admits')
    for broken in (replace(C, available=3), replace(C, source_ready=False)):
        candidates = (A, B, broken)
        correct, buggy = route_reference(candidates, req), route_cost_only_bug(candidates)
        require(correct[1] == 'B' and buggy[1] == 'C' and correct != buggy,
                'illegal cheapest candidate escaped')
    print('OBSERVATION route_guards', {
        'combinations': route_cases,
        'capacity_guard': route((A, B, replace(C, available=3)), req),
        'source_guard': route((A, B, replace(C, source_ready=False)), req),
        'cost_only_negative_controls_detected': True})

    # Exact boundary records correspond to the chapter's entry-path table.
    boundary_inputs = (
        ('zero_budget_abort', dict(D0=10, g=1, B=2, eps=1, max_stages=0),
         ('ABORT', 0, 0, 0, 0, 10)),
        ('initial_threshold_direct_freeze', dict(D0=1, g=1, B=2, eps=1, max_stages=0),
         ('STOP_AND_HANDOFF', 0, 0, Fraction(1, 2), 1, 1)),
        ('next_entry_budget_abort', dict(D0=10, g=1, B=2, eps=1, max_stages=1),
         ('ABORT', 1, 5, 0, 10, 5)),
        ('last_stage_threshold_first', dict(D0=2, g=1, B=2, eps=1, max_stages=1),
         ('STOP_AND_HANDOFF', 1, 1, Fraction(1, 2), 3, 1)))
    for name, args, expected in boundary_inputs:
        rec = check_migration(args)
        observed = (rec['status'], len(rec['stages']), rec['precopy'],
                    rec['downtime'], rec['bytes'], rec['delta'])
        require(observed == expected, 'chapter entry-path accounting: ' + name)
        print('OBSERVATION ' + name, summary(rec))

    zero_args = boundary_inputs[0][1]
    early_args = boundary_inputs[1][1]
    late_zero = migration(**zero_args, late_guard=True)
    late_early = migration(**early_args, late_guard=True)
    # The mutant actually executes the forbidden work; it does not merely
    # return a hand-written incorrect result or proactively raise an error.
    require(late_zero['precopy'] == 5 and late_zero['bytes'] == 10
            and len(late_zero['stages']) == 1, 'reviewed fault did not trigger')
    require(late_early['precopy'] == Fraction(1, 2)
            and late_early['bytes'] == Fraction(3, 2), 'threshold fault did not trigger')
    expect_rejected(late_zero, zero_args, 'copy before zero-budget guard')
    expect_rejected(late_early, early_args, 'copy before initial-threshold guard')
    print('OBSERVATION entry_guard_negative_controls', {
        'zero_budget_mutant': summary(late_zero),
        'initial_threshold_mutant': summary(late_early),
        'both_rejected_by_closed_reference': True})

    direct = check_migration(dict(D0=Fraction(1, 2), g=3, B=1, eps=1,
                                  max_stages=0, alpha=7,
                                  alpha_final=Fraction(1, 10), handoff=Fraction(1, 5)))
    require(direct['precopy'] == 0 and direct['wall'] == Fraction(4, 5),
            'direct freeze paid precopy overhead or required B>g')
    require(all(e[0] != 'CHECK_BUDGET' for e in direct['events']),
            'precopy budget wrongly evaluated before satisfied threshold')
    print('OBSERVATION direct_freeze_final_overhead', summary(direct))

    geo_args = dict(D0=1000, g=10, B=100, eps=1, max_stages=3)
    geometric = check_migration(geo_args)
    require(geometric['reached'] and len(geometric['stages']) == 3, 'last-stage equality')
    require(geometric['wall'] == Fraction(1111, 100), 'geometric wall')
    require(geometric['downtime'] == Fraction(1, 100), 'geometric downtime')
    require(geometric['bytes'] == 1111, 'geometric bytes')
    print('OBSERVATION geometric_last_allowed_stage', summary(geometric))

    fail_args = dict(D0=1000, g=10, B=8, eps=1, max_stages=5)
    failed = check_migration(fail_args)
    require(failed['status'] == 'ABORT', 'low bandwidth not aborted')
    require(failed['delta'] == Fraction(390625, 128), 'nonconvergent delta')
    print('OBSERVATION nonconvergent_abort', summary(failed))
    floor = check_migration(dict(D0=100, g=10, B=100, alpha=Fraction(1, 20),
                                eps=Fraction(1, 10), max_stages=64))
    require(not floor['reached'] and floor['delta'] > Fraction(5, 9), 'fixed floor')
    print('OBSERVATION fixed_overhead_floor', summary(floor))
    empty = check_migration(dict(D0=0, g=10, B=100, max_stages=0,
                                alpha_final=Fraction(1, 1000), handoff=Fraction(3, 1000)))
    require(empty['bytes'] == 0 and empty['downtime'] == Fraction(1, 250), 'empty control cost')
    print('OBSERVATION empty_payload_control_only', summary(empty))

    migration_cases = 0
    for D0, growth, bandwidth, overhead, threshold, budget in product(
            (0, 1, 10), (0, 1, 3), (1, 2, 4),
            (0, Fraction(1, 2)), (0, 1, 2), (0, 1, 3)):
        check_migration(dict(D0=D0, g=growth, B=bandwidth, alpha=overhead,
                             eps=threshold, max_stages=budget,
                             alpha_final=Fraction(1, 10), handoff=Fraction(1, 5)))
        migration_cases += 1
    equal_floor = check_migration(dict(D0=10, g=1, B=2, alpha=1, eps=2, max_stages=8))
    require(not equal_floor['reached'], 'asymptotic equality treated as finite success')
    zero_growth = check_migration(dict(D0=10, g=0, B=2, alpha=1, eps=0, max_stages=1))
    require(zero_growth['reached'] and zero_growth['delta'] == 0, 'zero growth')

    for invalid in ({'D0': -1}, {'g': -1}, {'B': 0}, {'alpha': -1},
                    {'eps': -1}, {'max_stages': -1}, {'alpha_final': -1}, {'handoff': -1}):
        args = dict(D0=10, g=1, B=2)
        args.update(invalid)
        try:
            migration(**args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid migration accepted: ' + str(invalid))

    copied, end, ranges = range_copy(10, (2, 1))
    required_positions = set(range(end))
    require(copied == required_positions and ranges[0] == (0, 10), 'initial history coverage')
    bad_copied, bad_end, bad_ranges = range_copy(10, (2, 1), omit_initial=True)
    require(bad_end == end and bad_copied != required_positions,
            'missing initial history escaped')
    require(required_positions - bad_copied == set(range(10)), 'missing-history diagnosis')

    wrong = dict(geometric, downtime=geometric['wall'])
    expect_rejected(wrong, geo_args, 'wall as downtime')
    wrong = dict(failed, reached=True, status='STOP_AND_HANDOFF')
    expect_rejected(wrong, fail_args, 'false convergence')
    wrong = dict(geometric, reached=geometric['stages'][-1][0] <= 1)
    expect_rejected(wrong, geo_args, 'last-stage off-by-one')
    wrong = dict(failed, bytes=failed['bytes'] + failed['delta'])
    expect_rejected(wrong, fail_args, 'uncopied abort bytes')
    print('OBSERVATION migration_reference_grid', {
        'cases': migration_cases,
        'scope': 'exact rational serial model, phase admission events, no devices or network'})
    print('OBSERVATION accounting_and_range_negative_controls', {
        'initial_history_missing': True, 'wall_as_downtime': True,
        'false_success': True, 'last_stage_off_by_one': True,
        'uncopied_abort_bytes': True, 'buggy_ranges': bad_ranges})
    print('CHECK_OK: finite CPU routing, serial migration admission/accounting and symbolic ranges only')


if __name__ == '__main__':
    run()
