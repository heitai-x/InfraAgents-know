from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations, product
import hashlib
import json


RESPONSIBILITIES = ('refs', 'readers', 'writers', 'outputs', 'bindings', 'quarantine')
POOLS = ('own:A', 'own:A', 'own:A', 'shared', 'shared', 'shared', 'own:B', 'own:B')
FIXTURES = {
    'pools': POOLS,
    'initial': {'A_exclusive': 3, 'A_active_borrowed': 2, 'A_directory_only': 1, 'B_free': 2},
    'responsibilities': RESPONSIBILITIES,
    'demand_range': [0, 1, 2, 3, 4, 5],
    'process_counts': {'A': 3, 'B': 1},
    'scope': 'finite CPU symbolic model; no device or service measurements',
}


@dataclass(frozen=True)
class Page:
    slot: int
    generation: int
    pool: str
    tenant: str
    directory: bool = False
    refs: tuple = ()
    readers: tuple = ()
    writers: tuple = ()
    outputs: tuple = ()
    bindings: tuple = ()
    quarantine: tuple = ()
    phase: str = 'PUBLISHED'
    initialized: bool = True

    @property
    def cache_only(self):
        # Directory is excluded from refs; execution and other blockers are separate.
        return self.directory and not self.refs


@dataclass(frozen=True)
class Plan:
    tenant: str
    demand: int
    slots: tuple
    observed: tuple


@dataclass(frozen=True)
class Grant:
    request: str
    tenant: str
    demand: int
    handles: tuple


class Bank:
    def __init__(self):
        self.pools = POOLS
        self.limits = {'A': 6, 'B': 4}
        self.pages = {}
        for s in range(3):
            self.pages[s] = Page(s, 1, self.pools[s], 'A', refs=('A:request',))
        for s in (3, 4):
            self.pages[s] = Page(s, 1, 'shared', 'A', refs=('A:request',), readers=('read:' + str(s),))
        self.pages[5] = Page(5, 1, 'shared', 'A', directory=True)
        self.generations = {s: (1 if s < 6 else 0) for s in range(8)}
        self.grants = {}
        self.events = []


def snapshot(bank):
    return (dict(bank.pages), dict(bank.generations), dict(bank.grants), tuple(bank.events))


def charged(bank, tenant):
    return sum(p.tenant == tenant for p in bank.pages.values())


def pool_allowed(pool, tenant):
    return pool in ('own:' + tenant, 'shared')


def immediately_reclaimable(page, omit=''):
    return (
        page.cache_only
        and page.phase == 'PUBLISHED'
        and all(not getattr(page, k) for k in RESPONSIBILITIES if k != omit)
    ) if omit != 'refs' else (
        page.directory
        and page.phase == 'PUBLISHED'
        and all(not getattr(page, k) for k in RESPONSIBILITIES if k != 'refs')
    )


def make_plan(bank, tenant, demand, omit=''):
    if not isinstance(demand, int) or demand < 0 or tenant not in bank.limits:
        return None
    if charged(bank, tenant) + demand > bank.limits[tenant]:
        return None
    free = [s for s, pool in enumerate(bank.pools) if pool_allowed(pool, tenant) and s not in bank.pages]
    victims = [
        s for s, p in sorted(bank.pages.items())
        if pool_allowed(p.pool, tenant) and immediately_reclaimable(p, omit)
    ]
    # All pages have equal size; using free slots before victims is complete here.
    chosen = tuple((free + victims)[:demand])
    if len(chosen) != demand:
        return None
    observed = tuple((s, bank.generations[s], s in bank.pages) for s in chosen)
    return Plan(tenant, demand, chosen, observed)


def execute_plan(bank, request, plan, omit='', omit_new_writer=False, fail_at=''):
    if plan is None:
        return 'WAIT', None
    if request in bank.grants:
        g = bank.grants[request]
        if (g.tenant, g.demand) == (plan.tenant, plan.demand):
            return 'READY', g
        return 'DENIED', None
    if charged(bank, plan.tenant) + plan.demand > bank.limits[plan.tenant]:
        return 'WAIT', None
    if len(set(plan.slots)) != plan.demand:
        return 'DENIED', None
    for s, generation, was_occupied in plan.observed:
        if bank.generations[s] != generation or not pool_allowed(bank.pools[s], plan.tenant):
            return 'WAIT', None
        p = bank.pages.get(s)
        if was_occupied:
            if p is None or not immediately_reclaimable(p, omit):
                return 'WAIT', None
        elif p is not None:
            return 'WAIT', None
    if plan.demand == 0:
        return 'EMPTY', None
    if fail_at == 'before_eviction':
        return 'WAIT', None

    # The symbolic fixed-pool allocator has no external asynchronous operations.
    # Real implementations require last-access and initialization evidence.
    for s, generation, was_occupied in plan.observed:
        if was_occupied:
            p = bank.pages[s]
            bank.pages[s] = replace(p, directory=False, phase='RETIRING')
            bank.events.append(('detach_all_directories', s, generation))
            retired = bank.pages[s]
            if any(getattr(retired, k) for k in RESPONSIBILITIES if k != omit):
                raise AssertionError('reclaim gate violated')
            del bank.pages[s]
            bank.events.append(('free_old_lifetime', s, generation))
    if fail_at == 'after_eviction':
        return 'WAIT', None

    created = []

    def rollback_unstarted():
        for s in created:
            # These are reservations only: no device/network work has started.
            del bank.pages[s]
            bank.events.append(('release_unstarted_private', s))
        # Safely evicted cache contents and consumed generations are not restored.

    for s in plan.slots:
        if s in bank.pages:
            raise AssertionError('new backing was not actually free')
        bank.generations[s] += 1
        generation = bank.generations[s]
        w = () if omit_new_writer else ('W:' + request + ':' + str(s),)
        bank.pages[s] = Page(
            s, generation, bank.pools[s], plan.tenant,
            refs=('grant:' + request,), writers=w, phase='STAGED', initialized=False,
        )
        created.append(s)
        bank.events.append(('allocate_new_lifetime', s, generation))
        if w:
            bank.events.append(('register_private_writer', s, generation))
        # Atomic initialization is a model assumption, not a GPU observation.
        bank.pages[s] = replace(bank.pages[s], initialized=True)
        bank.events.append(('initialize_exposed_range', s, generation))
        if fail_at == 'after_one_allocation':
            rollback_unstarted()
            return 'WAIT', None
    if fail_at == 'before_publish':
        rollback_unstarted()
        return 'WAIT', None
    handles = tuple((s, bank.pages[s].generation) for s in plan.slots)
    grant = Grant(request, plan.tenant, plan.demand, handles)
    bank.grants[request] = grant
    bank.events.append(('publish', request))
    return 'READY', grant


def reference_possible(bank, tenant, demand):
    # Independent finite-slot oracle: flatten original ownership/access relations,
    # enumerate slot combinations, and evaluate the post-state capacity contract.
    if not isinstance(demand, int) or demand < 0 or tenant not in bank.limits:
        return False
    tenant_slots = {s for s, p in bank.pages.items() if p.tenant == tenant}
    if len(tenant_slots) + demand > bank.limits[tenant]:
        return False
    relations = {
        (s, kind, holder)
        for s, p in bank.pages.items()
        for kind in RESPONSIBILITIES
        for holder in getattr(p, kind)
    }
    blocked = {s for s, kind, holder in relations}
    for selected in combinations(range(len(bank.pools)), demand):
        legal = True
        for s in selected:
            if bank.pools[s] not in ('own:' + tenant, 'shared'):
                legal = False
                break
            p = bank.pages.get(s)
            if p is not None and (s in blocked or not p.directory or p.phase != 'PUBLISHED'):
                legal = False
                break
        if legal:
            return True
    return False


def audit_transition(before, after, grant):
    assert len(after.pages) <= len(after.pools)
    for s, p in before.pages.items():
        if any(getattr(p, k) for k in RESPONSIBILITIES):
            assert after.pages.get(s) == p, ('old responsibility overwritten', s)
    for s, p in after.pages.items():
        assert p.slot == s and p.pool == after.pools[s]
        assert p.generation == after.generations[s]
    for tenant, limit in after.limits.items():
        assert charged(after, tenant) <= limit
    if grant is None:
        return
    assert len(grant.handles) == grant.demand
    assert len(set(grant.handles)) == grant.demand
    assert after.grants[grant.request] == grant
    for s, generation in grant.handles:
        p = after.pages[s]
        assert p.generation == generation and p.tenant == grant.tenant
        assert p.refs == ('grant:' + grant.request,)
        assert p.writers == ('W:' + grant.request + ':' + str(s),)
        assert p.initialized and not p.directory and p.phase == 'STAGED'
        assert pool_allowed(p.pool, grant.tenant)
        events = after.events
        allocation = events.index(('allocate_new_lifetime', s, generation))
        writer = events.index(('register_private_writer', s, generation))
        initialization = events.index(('initialize_exposed_range', s, generation))
        publication = events.index(('publish', grant.request))
        assert allocation < writer < initialization < publication
        if s in before.pages:
            old = before.pages[s]
            detach = events.index(('detach_all_directories', s, old.generation))
            release = events.index(('free_old_lifetime', s, old.generation))
            assert detach < release < allocation
            assert generation > old.generation


def unique_backings(records):
    by_slot = {}
    for p in records:
        if p.slot in by_slot and by_slot[p.slot] != p:
            raise ValueError('conflicting descriptions of one physical slot')
        by_slot[p.slot] = p
    return tuple(by_slot.values())


def must_detect(action):
    try:
        action()
    except AssertionError:
        return
    raise AssertionError('intentionally faulty control escaped its oracle')


def check_allocator():
    baseline = Bank()
    assert sum(pool == 'shared' for pool in baseline.pools) == 3
    assert sum(p.pool == 'shared' for p in baseline.pages.values()) == 3
    assert charged(baseline, 'A') == 6 and charged(baseline, 'B') == 0
    assert len(baseline.pools) - len(baseline.pages) == 2
    assert baseline.pages[5].cache_only

    b = deepcopy(baseline)
    before = deepcopy(b)
    status, grant = execute_plan(b, 'B3', make_plan(b, 'B', 3))
    assert status == 'READY' and len(b.pages) == 8
    assert charged(b, 'A') == 5 and charged(b, 'B') == 3
    assert sum(p.tenant == 'B' and p.pool == 'shared' for p in b.pages.values()) == 1
    audit_transition(before, b, grant)
    stable = snapshot(b)
    assert execute_plan(b, 'B3', Plan('B', 3, (), ()))[1] == grant
    assert snapshot(b) == stable
    # Alternative branch starts from baseline, not the already-full B3 state.
    assert make_plan(baseline, 'B', 4) is None
    assert not reference_possible(baseline, 'B', 4)

    cases = 0
    for bits in product((0, 1), repeat=len(RESPONSIBILITIES)):
        for demand in range(6):
            b = Bank()
            updates = {kind: ((kind + ':holder',) if bit else ()) for kind, bit in zip(RESPONSIBILITIES, bits)}
            b.pages[5] = replace(b.pages[5], **updates)
            before = deepcopy(b)
            expected = reference_possible(before, 'B', demand)
            status, grant = execute_plan(b, 'probe', make_plan(b, 'B', demand))
            assert (status in ('READY', 'EMPTY')) == expected
            audit_transition(before, b, grant)
            if not expected or demand == 0:
                assert snapshot(before) == snapshot(b)
            cases += 1
    assert cases == 384

    for kind in RESPONSIBILITIES:
        b = Bank()
        b.pages[5] = replace(b.pages[5], **{kind: (kind + ':holder',)})
        before = deepcopy(b)
        assert not reference_possible(before, 'B', 3)
        assert make_plan(b, 'B', 3) is None
        status, bad_grant = execute_plan(b, 'bad', make_plan(b, 'B', 3, omit=kind), omit=kind)
        assert status == 'READY'
        must_detect(lambda: audit_transition(before, b, bad_grant))

    paused = Bank()
    paused.pages[5] = replace(paused.pages[5], refs=('A:paused_mapping',))
    assert not paused.pages[5].cache_only
    assert not paused.pages[5].readers and not paused.pages[5].writers
    assert make_plan(paused, 'B', 3) is None
    paused.pages[5] = replace(paused.pages[5], refs=())
    assert make_plan(paused, 'B', 3) is not None

    b = Bank()
    plan = make_plan(b, 'B', 3)
    b.pages[5] = replace(b.pages[5], refs=('A:new_mapping_between_plan_and_commit',))
    stable = snapshot(b)
    assert execute_plan(b, 'race', plan) == ('WAIT', None)
    assert snapshot(b) == stable

    b = Bank()
    plan = make_plan(b, 'B', 3)
    b.generations[5] += 1
    b.pages[5] = replace(b.pages[5], generation=b.generations[5])
    stable = snapshot(b)
    assert execute_plan(b, 'stale_plan', plan) == ('WAIT', None)
    assert snapshot(b) == stable

    for failure in ('before_eviction', 'after_eviction', 'after_one_allocation', 'before_publish'):
        b = Bank()
        before = deepcopy(b)
        status, grant = execute_plan(b, 'failed', make_plan(b, 'B', 3), fail_at=failure)
        assert status == 'WAIT' and grant is None and not b.grants
        assert charged(b, 'B') == 0
        audit_transition(before, b, None)
        if failure == 'before_eviction':
            assert snapshot(b) == snapshot(before)
        else:
            assert 5 not in b.pages and len(b.pages) == 5
        retry_before = deepcopy(b)
        status, grant = execute_plan(b, 'retry', make_plan(b, 'B', 3))
        assert status == 'READY'
        audit_transition(retry_before, b, grant)

    b = Bank()
    before = deepcopy(b)
    status, bad_grant = execute_plan(b, 'missing_W', make_plan(b, 'B', 3), omit_new_writer=True)
    assert status == 'READY'
    must_detect(lambda: audit_transition(before, b, bad_grant))

    b = Bank()
    b.limits['B'] = 2
    assert make_plan(b, 'B', 3) is None and not reference_possible(b, 'B', 3)
    assert make_plan(b, 'B', -1) is None
    b = Bank()
    # Even when A's three exclusive slots become physically free, B cannot use them.
    for s in (0, 1, 2):
        del b.pages[s]
    assert len(b.pools) - len(b.pages) == 5
    assert make_plan(b, 'B', 4) is None and not reference_possible(b, 'B', 4)

    p = Bank().pages[5]
    assert len(unique_backings((p, p, p))) == 1
    try:
        unique_backings((p, replace(p, generation=2)))
    except ValueError:
        pass
    else:
        raise AssertionError('conflicting physical identity silently deduplicated')
    print('[allocator] 384 responsibility/demand cases agree with slot enumeration')
    print('[allocator] B3/B4 alternative branches, paused mapping, revalidation, rollback and retry checked')
    print('[negative-controls] ignoring each of refs/R/W/O/B/Q and missing private W detected')


@dataclass(frozen=True)
class CacheEntry:
    bucket: str = 'forced-collision'
    parent: str = 'exact-parent'
    tokens: tuple = (10, 20)
    acl: frozenset = frozenset(('A',))
    namespace: str = 'nsA'
    model: str = 'm1'
    adapter: str = 'a1'
    generation: int = 7
    ready: bool = True
    full: bool = True


def cache_hit(entry, query, omit=''):
    checks = {
        'bucket': entry.bucket == query['bucket'],
        'parent': entry.parent == query['parent'],
        'tokens': entry.tokens == query['tokens'],
        'acl': query['principal'] in entry.acl,
        'namespace': entry.namespace == query['namespace'],
        'model': entry.model == query['model'],
        'adapter': entry.adapter == query['adapter'],
        'generation': entry.generation == query['generation'],
        'ready': entry.ready,
        'full': entry.full,
    }
    return all(value for key, value in checks.items() if key != omit)


def longest_contiguous(hits):
    length = 0
    for hit in hits:
        if not hit:
            break
        length += 1
    return length


def check_cache():
    query = {
        'principal': 'A', 'bucket': 'forced-collision', 'parent': 'exact-parent',
        'tokens': (10, 20), 'namespace': 'nsA', 'model': 'm1', 'adapter': 'a1', 'generation': 7,
    }
    good = CacheEntry()
    assert cache_hit(good, query)
    isolated_bad_values = {
        'bucket': 'different-bucket', 'parent': 'different-exact-parent', 'tokens': (10, 99),
        'acl': frozenset(('B',)), 'namespace': 'nsB', 'model': 'm2', 'adapter': 'a2',
        'generation': 6, 'ready': False, 'full': False,
    }
    for gate, value in isolated_bad_values.items():
        bad = replace(good, **{gate: value})
        # Fixed contract: exactly one incompatible condition must reject this entry.
        assert not cache_hit(bad, query), gate
        assert cache_hit(bad, query, omit=gate), ('negative control was masked', gate)
    public = replace(good, acl=frozenset(('A', 'B')), namespace='public')
    public_query = dict(query, principal='B', namespace='public')
    assert cache_hit(public, public_query)
    assert longest_contiguous(()) == 0
    assert longest_contiguous((True, False, True)) == 1
    assert sum((True, False, True)) == 2  # Faulty skip-over-miss control.
    assert longest_contiguous((True, True, True)) == 3
    print('[cache] isolated ACL/namespace/identity/generation/readiness controls and contiguous-prefix boundary checked')


@dataclass(frozen=True)
class Replica:
    name: str
    rate: int
    domains: frozenset


def surviving_rates(replicas):
    domains = sorted(set().union(*(r.domains for r in replicas))) if replicas else []
    return {d: sum(r.rate for r in replicas if d not in r.domains) for d in domains}


def check_faults():
    spread = (
        Replica('r1', 40, frozenset(('host1', 'power1'))),
        Replica('r2', 40, frozenset(('host2', 'power2'))),
        Replica('r3', 40, frozenset(('host3', 'power3'))),
    )
    shared = (spread[0], replace(spread[1], domains=frozenset(('host2', 'power1'))), spread[2])
    assert min(surviving_rates(spread).values()) == 80
    assert min(surviving_rates(shared).values()) == 40
    assert surviving_rates(shared)['power1'] == 40
    assert min(surviving_rates(spread).values()) >= 70
    assert min(surviving_rates(shared).values()) < 70
    faulty_ignore_domains = sum(r.rate for r in shared)
    assert faulty_ignore_domains >= 70
    assert faulty_ignore_domains != min(surviving_rates(shared).values())
    assert surviving_rates(()) == {}
    one = (Replica('only', 40, frozenset(('host',))),)
    assert surviving_rates(one) == {'host': 0}
    print('[fault] single-domain minima: independent=80, shared-power=40 requests/s; sum-only control detected')


def ideal_process_shares(counts):
    if any(not isinstance(n, int) or n < 0 for n in counts.values()):
        raise ValueError('invalid active process count')
    total = sum(counts.values())
    if total == 0:
        return {}
    return {tenant: Fraction(n, total) for tenant, n in counts.items()}


def round_robin_reference(counts):
    processes = [(tenant, i) for tenant in sorted(counts) for i in range(counts[tenant])]
    if not processes:
        return {}
    slots = [processes[i % len(processes)][0] for i in range(17 * len(processes))]
    return {tenant: Fraction(slots.count(tenant), len(slots)) for tenant in counts}


def check_shares():
    for a, b in product(range(4), repeat=2):
        counts = {'A': a, 'B': b}
        assert ideal_process_shares(counts) == round_robin_reference(counts)
    actual_model = ideal_process_shares({'A': 3, 'B': 1})
    assert actual_model == {'A': Fraction(3, 4), 'B': Fraction(1, 4)}
    faulty_client_equal = {'A': Fraction(1, 2), 'B': Fraction(1, 2)}
    assert actual_model != faulty_client_equal
    assert ideal_process_shares({'A': 1, 'B': 1}) == faulty_client_equal
    assert ideal_process_shares({}) == {}
    try:
        ideal_process_shares({'A': -1})
    except ValueError:
        pass
    else:
        raise AssertionError('negative process count accepted')
    weights = {'A': 1, 'B': 3}
    weighted = {t: Fraction(4 * w, sum(weights.values())) for t, w in weights.items()}
    assert weighted == {'A': Fraction(1), 'B': Fraction(3)}
    assert weighted != {'A': Fraction(2), 'B': Fraction(2)}
    print('[shares] process-level 3:1 gives 3/4:1/4; client-equal control detected; CPU weighted arithmetic checked')


def main():
    encoded = json.dumps(FIXTURES, sort_keys=True, separators=(',', ':')).encode('utf-8')
    print('fixture_sha256=' + hashlib.sha256(encoded).hexdigest())
    check_allocator()
    check_cache()
    check_faults()
    check_shares()
    print('scope: deterministic CPU symbolic allocation, identity, fault-domain and ideal-share checks only')
    print('unmeasured: GPU/cgroup/MPS/MIG behavior, memory visibility, asynchronous drain, performance and tenant security')


if __name__ == '__main__':
    main()
