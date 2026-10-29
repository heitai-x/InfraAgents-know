from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product

G = 1 << 30
OPEN = frozenset(('LOWER_BOUND_SURVIVOR', 'EVIDENCE_STALE'))
TIME_EXCLUDED = frozenset(('EXCLUDED_LOWER_BOUND', 'EXCLUDED_MEASURED'))


@dataclass(frozen=True)
class Page:
    identity: str
    tokens: int

    def __post_init__(self):
        if not isinstance(self.identity, str) or not self.identity:
            raise ValueError('nonempty physical page identity required')
        if type(self.tokens) is not int or self.tokens <= 0:
            raise ValueError('positive integer token allocation required')


def kv_bytes_per_token(layers, kv_heads, head_dim, elem_bytes):
    if any(type(x) is not int or x <= 0 for x in
           (layers, kv_heads, head_dim, elem_bytes)):
        raise ValueError('positive integer model dimensions required')
    return 2 * layers * kv_heads * head_dim * elem_bytes


def unique_bytes(groups, bytes_per_token):
    sizes = {}
    for pages in groups:
        for page in pages:
            if page.identity in sizes and sizes[page.identity] != page.tokens:
                raise ValueError('conflicting allocation for one physical page')
            sizes[page.identity] = page.tokens
    return sum(sizes.values()) * bytes_per_token


def logical_bytes(groups, bytes_per_token):
    return sum(page.tokens for pages in groups for page in pages) * bytes_per_token


def replica_lb(weight_bytes, traffic_per_replica, bandwidth, independent=False):
    if not traffic_per_replica or bandwidth <= 0 or weight_bytes < 0:
        raise ValueError('replicas, positive bandwidth and nonnegative weights required')
    if any(x < 0 for requests in traffic_per_replica for x in requests):
        raise ValueError('negative traffic')
    share = F(bandwidth) if independent else F(bandwidth, len(traffic_per_replica))
    return [F(0) if not requests else (F(weight_bytes) + sum(requests)) / share
            for requests in traffic_per_replica]


@dataclass(frozen=True)
class Evidence:
    kind: str
    low: object
    high: object
    scope: bool = True

    def __post_init__(self):
        if self.kind not in ('upper', 'observed', 'interval'):
            raise ValueError('unknown evidence kind')
        if type(self.scope) is not bool:
            raise ValueError('scope must be an externally validated boolean')
        high = F(self.high)
        low = None if self.low is None else F(self.low)
        if high < 0:
            raise ValueError('negative upper endpoint')
        if self.kind == 'upper':
            if low is not None:
                raise ValueError('a conservative upper bound supplies no lower endpoint')
        elif low is None or low < 0 or low > high:
            raise ValueError('invalid observation interval')
        if self.kind == 'observed' and low != high:
            raise ValueError('exact finite observation requires equal endpoints')
        object.__setattr__(self, 'low', low)
        object.__setattr__(self, 'high', high)


def classify(legal, capacity, lb, target, evidence=(), quality='unknown', quality_scope=True):
    """One scalar latency target. Capacity is already resolved, not an uncertain bound.

    Evidence validity, scope matching and statistical admissibility are caller premises.
    An observed point describes only its declared finite execution/replay estimand.
    The function consumes quality conclusions; it does not evaluate a real model.
    """
    if type(legal) is not bool or type(capacity) is not bool:
        raise ValueError('resolved legality and capacity booleans required')
    if quality not in ('pass', 'fail', 'unknown') or type(quality_scope) is not bool:
        raise ValueError('invalid quality state or scope')
    lb, target = F(lb), F(target)
    if lb < 0 or target < 0:
        raise ValueError('negative timing input')
    if not legal:
        return 'ILLEGAL_LEGALITY'
    if not capacity:
        return 'ILLEGAL_CAPACITY'

    active = [e for e in evidence if e.scope]
    stale = any(not e.scope for e in evidence)
    stale = stale or (quality != 'unknown' and not quality_scope)
    q = quality if quality_scope else 'unknown'
    lower = max([lb] + [e.low for e in active if e.low is not None])
    upper = min((e.high for e in active), default=None)

    # A contradictory bound cannot first be used to prune this candidate.
    if upper is not None and lower > upper:
        return 'INCONSISTENT'
    if q == 'fail':
        return 'EXCLUDED_QUALITY'
    if lb > target:
        return 'EXCLUDED_LOWER_BOUND'
    if any(e.kind != 'upper' and e.low > target for e in active):
        return 'EXCLUDED_MEASURED'
    if upper is not None and upper <= target:
        return 'VERIFIED' if q == 'pass' else 'QUALITY_PENDING'
    return 'EVIDENCE_STALE' if stale else 'LOWER_BOUND_SURVIVOR'


def finite_world_reference(case):
    """Independent set semantics, not the classifier's ordered decision branches.

    Exhaustive cases have integer endpoints in [0, 80], target 60 and lb <= 70.
    Enumerating integer durations 0..100 supplies every relevant threshold witness:
    unbounded-above cases always retain a witness >60. This is a finite model,
    not evidence that a real continuous latency distribution has been measured.
    """
    if not case['legal']:
        return frozenset(('ILLEGAL_LEGALITY',))
    if not case['capacity']:
        return frozenset(('ILLEGAL_CAPACITY',))
    worlds = set(range(101))
    worlds = {t for t in worlds if t >= case['lb']}
    for e in case['evidence']:
        if e.scope:
            admissible = {t for t in range(101)
                          if t <= e.high and (e.low is None or t >= e.low)}
            worlds.intersection_update(admissible)
    if not worlds:
        return frozenset(('INCONSISTENT',))
    q = case['quality'] if case['quality_scope'] else 'unknown'
    if q == 'fail':
        return frozenset(('EXCLUDED_QUALITY',))
    passing = {t for t in worlds if t <= case['target']}
    if not passing:
        return TIME_EXCLUDED
    if passing == worlds:
        return frozenset(('VERIFIED' if q == 'pass' else 'QUALITY_PENDING',))
    return OPEN


def make_case(lb=40, evidence=(), quality='pass', quality_scope=True,
              legal=True, capacity=True, target=60):
    return dict(legal=legal, capacity=capacity, lb=lb, target=target,
                evidence=tuple(evidence), quality=quality, quality_scope=quality_scope)


def verify_decision(fn, case):
    result = fn(**case)
    allowed = finite_world_reference(case)
    assert result in allowed, (fn.__name__, result, sorted(allowed), case)
    return result


def faulty_upper_as_failure(**case):
    # Actually execute the incorrect inference from a loose upper bound.
    if case['legal'] and case['capacity'] and case['lb'] <= case['target']:
        for e in case['evidence']:
            if e.scope and e.kind == 'upper' and e.high > case['target']:
                return 'EXCLUDED_MEASURED'
    return classify(**case)


def faulty_ignore_quality(**case):
    # The wrong implementation promotes a passing time without a quality gate.
    active = [e for e in case['evidence'] if e.scope]
    if case['legal'] and case['capacity'] and active:
        upper = min(e.high for e in active)
        if case['lb'] <= upper <= case['target']:
            return 'VERIFIED'
    return classify(**case)


def faulty_lower_bound_first(**case):
    if case['legal'] and case['capacity'] and case['lb'] > case['target']:
        return 'EXCLUDED_LOWER_BOUND'
    return classify(**case)


def faulty_lower_bound_as_verified(**case):
    if case['legal'] and case['capacity'] and not case['evidence']:
        if case['lb'] <= case['target']:
            return 'VERIFIED'
    return classify(**case)


def expect_rejected(fn, case):
    try:
        verify_decision(fn, case)
    except AssertionError:
        return
    raise AssertionError('faulty classifier escaped independent reference: ' + fn.__name__)


def check_resources():
    b = kv_bytes_per_token(24, 4, 128, 2)
    assert b == 49152
    groups = [[Page('prefix', 4), Page('a', 2)],
              [Page('prefix', 4), Page('c', 3)]]
    # Independent references materialize individual allocated slots and reads.
    allocated = {(p.identity, offset)
                 for pages in groups for p in pages for offset in range(p.tokens)}
    reads = [(request, p.identity, offset)
             for request, pages in enumerate(groups)
             for p in pages for offset in range(p.tokens)]
    u_ref, l_ref = len(allocated) * b, len(reads) * b
    assert len(allocated) == 9 and len(reads) == 13
    assert unique_bytes(groups, b) == u_ref
    assert logical_bytes(groups, b) == l_ref
    assert unique_bytes([], b) == logical_bytes([], b) == 0
    try:
        unique_bytes([[Page('same', 1)], [Page('same', 2)]], b)
    except ValueError:
        pass
    else:
        raise AssertionError('conflicting physical page allocation accepted')

    # Wrong algorithms really run, and fail the slot/read references.
    faulty_traffic = unique_bytes(groups, b)
    faulty_capacity = logical_bytes(groups, b)
    assert faulty_traffic != l_ref
    assert faulty_capacity != u_ref

    hist = 2048 * b
    weight, bandwidth = 3 * G, 96 * G
    a1 = replica_lb(weight, [[hist]], bandwidth)[0]
    a8 = 8 * a1
    batch = replica_lb(weight, [[hist] * 8], bandwidth)[0]
    shared = replica_lb(weight, [[hist] * 4, [hist] * 4], bandwidth)[0]
    independent = replica_lb(weight, [[hist] * 4, [hist] * 4], bandwidth, True)[0]
    large = replica_lb(weight, [[hist] * 64], bandwidth)[0]
    # Independently hand-expanded rational results in seconds.
    assert (a1, a8, batch, shared, independent, large) == (
        F(33, 1024), F(33, 128), F(5, 128), F(9, 128), F(9, 256), F(3, 32))
    cap8 = F(5) + F(8 * hist, G)
    cap64 = F(5) + F(64 * hist, G)
    cap_two = F(10) + F(8 * hist, G)
    assert (cap8, cap64, cap_two) == (F(23, 4), F(11), F(43, 4))
    assert replica_lb(weight, [[]], bandwidth) == [0]
    print('resource model: KV_B_per_token=%d unique_B=%d logical_read_B=%d' %
          (b, u_ref, l_ref))
    print('conditional seconds: A1=%s A8=%s B=%s C_shared=%s C_independent=%s B64=%s' %
          (a1, a8, batch, shared, independent, large))
    print('residency main terms GiB: batch8=%s batch64=%s two_replicas=%s' %
          (cap8, cap64, cap_two))
    return batch


def check_classification(batch_seconds):
    # Classification examples below use milliseconds, unlike resource seconds above.
    lower = batch_seconds * 1000
    actual, loose_upper, target = F(50), F(80), F(60)
    assert lower == F(625, 16)
    assert lower <= actual <= loose_upper and actual <= target
    loose = make_case(lb=lower, evidence=(Evidence('upper', None, loose_upper),))
    assert verify_decision(classify, loose) == 'LOWER_BOUND_SURVIVOR'
    exact = make_case(lb=lower, evidence=(Evidence('upper', None, 80),
                                         Evidence('observed', actual, actual)))
    assert verify_decision(classify, exact) == 'VERIFIED'

    passed_time = (Evidence('upper', None, 55),)
    assert verify_decision(classify, make_case(evidence=passed_time, quality='fail')) == 'EXCLUDED_QUALITY'
    assert verify_decision(classify, make_case(evidence=passed_time, quality='unknown')) == 'QUALITY_PENDING'
    assert verify_decision(classify, make_case(evidence=passed_time, quality='pass',
                                             quality_scope=False)) == 'QUALITY_PENDING'
    conflict = make_case(lb=70, evidence=passed_time)
    assert verify_decision(classify, conflict) == 'INCONSISTENT'
    assert verify_decision(classify, make_case(lb=70, evidence=(
        Evidence('upper', None, 55, False),))) == 'EXCLUDED_LOWER_BOUND'
    assert verify_decision(classify, make_case(evidence=(
        Evidence('interval', 50, 70),))) == 'LOWER_BOUND_SURVIVOR'
    assert verify_decision(classify, make_case(evidence=(
        Evidence('interval', 65, 75),))) == 'EXCLUDED_MEASURED'
    assert verify_decision(classify, make_case(evidence=(
        Evidence('observed', 60, 60),))) == 'VERIFIED'
    assert verify_decision(classify, make_case(lb=0, target=0, evidence=(
        Evidence('upper', None, 0),))) == 'VERIFIED'
    assert verify_decision(classify, make_case(lb=60)) == 'LOWER_BOUND_SURVIVOR'

    count = 0
    observation_specs = (None, ('observed', 50, 50), ('observed', 70, 70),
                         ('interval', 50, 70), ('interval', 65, 75))
    combinations = product((0, 40, 60, 70), (None, 55, 60, 80), observation_specs,
                           (False, True), (False, True), ('pass', 'fail', 'unknown'),
                           (False, True))
    for lb, ub, obs, upper_scope, obs_scope, q, q_scope in combinations:
        evidence = []
        if ub is not None:
            evidence.append(Evidence('upper', None, ub, upper_scope))
        if obs is not None:
            evidence.append(Evidence(obs[0], obs[1], obs[2], obs_scope))
        verify_decision(classify, make_case(lb=lb, evidence=evidence,
                                           quality=q, quality_scope=q_scope))
        count += 1
    assert count == 1920
    for legal, capacity in product((False, True), repeat=2):
        verify_decision(classify, make_case(legal=legal, capacity=capacity))

    expect_rejected(faulty_upper_as_failure, loose)
    expect_rejected(faulty_ignore_quality, make_case(evidence=passed_time, quality='fail'))
    expect_rejected(faulty_ignore_quality, make_case(evidence=passed_time, quality='unknown'))
    expect_rejected(faulty_lower_bound_first, conflict)
    expect_rejected(faulty_lower_bound_as_verified, make_case())

    # An open or inconsistent target cell prevents deletion across all cells.
    terminal = TIME_EXCLUDED | frozenset(('ILLEGAL_LEGALITY', 'ILLEGAL_CAPACITY',
                                         'EXCLUDED_QUALITY'))
    surviving_states = [classify(**loose), classify(**conflict), 'EXCLUDED_LOWER_BOUND']
    assert not all(status in terminal for status in surviving_states)
    assert all(status in terminal for status in ('EXCLUDED_MEASURED', 'EXCLUDED_QUALITY'))
    print('evidence model: %d exhaustive finite-world cases plus boundary cases' % count)
    print('loose upper counterexample ms: lower=%s actual_witness=%s upper=%s target=%s status=%s' %
          (lower, actual, loose_upper, target, classify(**loose)))
    print('conflict ms: lower=70 upper=55 target=60 status=' + classify(**conflict))
    print('quality gates: fail=EXCLUDED_QUALITY unknown=QUALITY_PENDING')
    print('negative controls rejected: traffic dedup, capacity double count, loose-upper exclusion,')
    print('quality-fail bypass, quality-unknown bypass, lower-bound-first, lower-bound-as-verified')


def main():
    print('K-SYS-010 scope: offline CPU arithmetic and abstract evidence decisions only.')
    print('No CPU/GPU timing, real forward, quality evaluation or statistical coverage validation.')
    batch = check_resources()
    check_classification(batch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
