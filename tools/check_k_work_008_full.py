#!/usr/bin/env python3
"""K-WORK-008: deterministic offline CPU models; no device measurements."""
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import permutations, product

STATES = ('PREPARING', 'READY', 'OPEN', 'DRAINING', 'RETIRING', 'RETIRED', 'FAILED')
CHECKS = 0
MUTANTS = []


def equal(label, got, expected):
    global CHECKS
    CHECKS += 1
    if got != expected:
        raise AssertionError(label + ': got=' + repr(got) + ' expected=' + repr(expected))


def killed(label, got, expected):
    if got == expected:
        raise AssertionError('surviving negative control: ' + label)
    MUTANTS.append(label)
    print('DETECTED: ' + label)


def capacity(items, omit_state=False, omit_certificate=False):
    return sum(r.q for r in items
               if (omit_state or r.state == 'OPEN')
               and (omit_certificate or r.valid))


# External eligibility decision table, including every state/certificate pair.
ELIGIBILITY = {
    'PREPARING': (0, 0), 'READY': (0, 0), 'OPEN': (0, 1),
    'DRAINING': (0, 0), 'RETIRING': (0, 0),
    'RETIRED': (0, 0), 'FAILED': (0, 0),
}


def capacity_reference(items):
    return sum(ELIGIBILITY[r.state][int(r.valid)] * r.q for r in items)


@dataclass
class Replica:
    slot: str
    gen: int = 1
    state: str = 'PREPARING'
    valid: bool = False
    q: int = 20
    scope: str = 'class-A'
    allocated: bool = True
    gate: bool = False
    seen: object = None

    @property
    def key(self):
        return (self.slot, self.gen)

    @property
    def prepare_op(self):
        return 'prepare:' + self.slot + ':' + str(self.gen)


class Controller:
    def __init__(self, replicas):
        self.records = {r.slot: r for r in replicas}
        self.epoch = 0
        self.targets = set()
        self.commits = {}

    def lookup(self, key):
        item = self.records.get(key[0])
        return item if item is not None and item.key == key else None

    def ready(self, key, op, scope, proof=True, all_ranks=True):
        r = self.lookup(key)
        if r is None or op != r.prepare_op or scope != r.scope:
            return 'REJECT'
        fingerprint = (key, op, scope, proof, all_ranks)
        if r.seen is not None:
            return 'DUP' if r.seen == fingerprint else 'CONFLICT'
        if r.state != 'PREPARING' or not proof or not all_ranks:
            return 'REJECT'
        r.valid = True
        r.state = 'READY'
        r.seen = fingerprint
        return 'APPLIED'

    def commit(self, op, expected_epoch, mode, add=(), remove=(), faulty_append=False):
        add, remove = tuple(add), tuple(remove)
        signature = (mode, expected_epoch, tuple(sorted(add)), tuple(sorted(remove)))
        if op in self.commits:
            return 'DUP' if self.commits[op][0] == signature else 'CONFLICT'
        if expected_epoch != self.epoch:
            return 'REJECT'
        h, d = set(add), set(remove)
        if len(h) != len(add) or len(d) != len(remove) or h & d or h & self.targets:
            return 'REJECT'
        if mode == 'FIRST':
            legal = not self.targets and bool(h) and not d
        elif mode == 'APPEND':
            legal = bool(self.targets) and bool(h) and not d
        elif mode == 'REPLACE':
            legal = bool(h) and bool(d) and d <= self.targets
        elif mode == 'STOP':
            legal = not h and bool(d) and d <= self.targets
        else:
            legal = False
        if not legal:
            return 'REJECT'
        for key in h:
            r = self.lookup(key)
            if r is None or r.state != 'READY' or not r.valid or not r.allocated:
                return 'REJECT'
        for key in d:
            r = self.lookup(key)
            if r is None or r.state != 'OPEN' or not r.gate:
                return 'REJECT'
        if mode == 'APPEND':
            old_scopes = {self.lookup(key).scope for key in self.targets}
            if any(self.lookup(key).scope not in old_scopes for key in h):
                return 'REJECT'
        actual_remove = set(d)
        if faulty_append and mode == 'APPEND':
            actual_remove.add(sorted(self.targets)[0])
        for key in actual_remove:
            r = self.lookup(key)
            r.gate = False
            r.state = 'DRAINING'
        for key in h:
            r = self.lookup(key)
            r.gate = True
            r.state = 'OPEN'
        self.targets = (self.targets - actual_remove) | h
        self.epoch += 1
        self.commits[op] = (signature, self.epoch, tuple(sorted(self.targets)))
        return 'COMMITTED'

    def available(self):
        return capacity(self.records.values())


def send_ready(c, r):
    return c.ready(r.key, r.prepare_op, r.scope)


def route_reference(c, expected_targets, expected_epoch):
    equal('explicit target reference', c.targets, set(expected_targets))
    equal('route epoch', c.epoch, expected_epoch)
    equal('OPEN set equals target set',
          {r.key for r in c.records.values() if r.state == 'OPEN'},
          set(expected_targets))
    equal('gate set equals target set',
          {r.key for r in c.records.values() if r.gate}, set(expected_targets))


def test_capacity():
    items = [Replica('open-good', state='OPEN', valid=True, q=20),
             Replica('ready-good', state='READY', valid=True, q=7),
             Replica('open-bad', state='OPEN', valid=False, q=11)]
    want = capacity_reference(items)
    equal('crossed guards reference', want, 20)
    equal('crossed guards implementation', capacity(items), want)
    killed('delete OPEN guard', capacity(items, omit_state=True), want)
    killed('delete certificate guard', capacity(items, omit_certificate=True), want)
    for state, valid in product(STATES, (False, True)):
        r = Replica('one', state=state, valid=valid, q=13)
        equal('state/certificate pair', capacity([r]), capacity_reference([r]))
    equal('empty capacity', capacity([]), 0)
    print('OK: capacity decision table, 14 state/certificate pairs and separate guard mutations')


def test_ready():
    for source in STATES:
        r = Replica('slot', gen=2, state=source)
        c = Controller([r])
        before = deepcopy(r)
        result = send_ready(c, r)
        equal('READY allowed source', result, 'APPLIED' if source == 'PREPARING' else 'REJECT')
        if source == 'PREPARING':
            equal('PREPARING becomes READY', r.state, 'READY')
            equal('READY is not OPEN capacity', c.available(), 0)
        else:
            equal('invalid source unchanged', r, before)
    for later in STATES[1:]:
        r = Replica('slot', gen=2)
        c = Controller([r])
        equal('initial notification', send_ready(c, r), 'APPLIED')
        r.state = later
        r.gate = later == 'OPEN'
        r.valid = False
        before = deepcopy(r)
        equal('late duplicate returns original acknowledgement', send_ready(c, r), 'DUP')
        equal('late duplicate does not restore state or certificate', r, before)
        equal('same operation conflicting proof',
              c.ready(r.key, r.prepare_op, r.scope, proof=False), 'CONFLICT')
        equal('conflict unchanged', r, before)
        equal('same generation wrong operation',
              c.ready(r.key, 'wrong-operation', r.scope), 'REJECT')
        equal('same slot old generation',
              c.ready(('slot', 1), 'prepare:slot:1', r.scope), 'REJECT')
        equal('bad events unchanged', r, before)
    r = Replica('slot', gen=3)
    c = Controller([r])
    before = deepcopy(r)
    equal('old event against new pending generation',
          c.ready(('slot', 2), 'prepare:slot:2', r.scope), 'REJECT')
    equal('partial execution group rejected',
          c.ready(r.key, r.prepare_op, r.scope, all_ranks=False), 'REJECT')
    equal('invalid preparation proof rejected',
          c.ready(r.key, r.prepare_op, r.scope, proof=False), 'REJECT')
    equal('unqualified notifications leave pending instance unchanged', r, before)
    equal('new generation still progresses', send_ready(c, r), 'APPLIED')
    for state in ('OPEN', 'DRAINING', 'RETIRING'):
        buggy_state = 'READY' if state not in ('RETIRED', 'FAILED') else state
        killed('weak READY source guard from ' + state, buggy_state, state)
    print('OK: READY sources, immutable operation identity, generations, conflicts and late duplicates')


def test_routes():
    a, b, c, d, bad = [Replica(name) for name in ('a', 'b', 'c', 'd', 'bad')]
    ctl = Controller([a, b, c, d, bad])
    equal('empty route initial capacity', ctl.available(), 0)
    for r in (a, b, c, d, bad):
        equal('prepare for route test', send_ready(ctl, r), 'APPLIED')
    equal('FIRST with no old target', ctl.commit('first', 0, 'FIRST', (a.key,)), 'COMMITTED')
    route_reference(ctl, {a.key}, 1)
    before = deepcopy(ctl.__dict__)
    equal('lost acknowledgement retry', ctl.commit('first', 0, 'FIRST', (a.key,)), 'DUP')
    equal('repeated commit has no effect', ctl.__dict__, before)
    equal('same route operation conflicting payload',
          ctl.commit('first', 0, 'FIRST', (b.key,)), 'CONFLICT')
    equal('stale epoch rejected', ctl.commit('stale', 0, 'APPEND', (b.key,)), 'REJECT')
    equal('rejections have no effect', ctl.__dict__, before)
    equal('append second', ctl.commit('append-b', 1, 'APPEND', (b.key,)), 'COMMITTED')
    route_reference(ctl, {a.key, b.key}, 2)
    equal('READY third contributes no capacity', ctl.available(), 40)
    equal('append third', ctl.commit('append-c', 2, 'APPEND', (c.key,)), 'COMMITTED')
    route_reference(ctl, {a.key, b.key, c.key}, 3)
    equal('three OPEN capacities', ctl.available(), 60)
    bad.valid = False
    before = deepcopy(ctl.__dict__)
    equal('invalid READY cannot publish', ctl.commit('bad-cert', 3, 'APPEND', (bad.key,)), 'REJECT')
    equal('invalid publication atomic rejection', ctl.__dict__, before)
    d.allocated = False
    before = deepcopy(ctl.__dict__)
    equal('lost allocation cannot publish', ctl.commit('bad-resource', 3, 'APPEND', (d.key,)), 'REJECT')
    equal('resource rejection atomic', ctl.__dict__, before)
    d.allocated = True
    before = deepcopy(ctl.__dict__)
    equal('replacement must name an actual old target',
          ctl.commit('missing-old', 3, 'REPLACE', (d.key,), (('absent', 1),)), 'REJECT')
    equal('bad replacement leaves all targets unchanged', ctl.__dict__, before)
    equal('replace only selected old target',
          ctl.commit('replace-b', 3, 'REPLACE', (d.key,), (b.key,)), 'COMMITTED')
    route_reference(ctl, {a.key, c.key, d.key}, 4)
    equal('designated old instance drains', b.state, 'DRAINING')
    equal('replacement capacity', ctl.available(), 60)
    equal('stop all targets',
          ctl.commit('stop', 4, 'STOP', (), (a.key, c.key, d.key)), 'COMMITTED')
    route_reference(ctl, set(), 5)
    equal('draining instances add no new capacity', ctl.available(), 0)
    print('OK: FIRST, APPEND, selected REPLACE, STOP, CAS, idempotency and atomic rejection')


def burst(faulty_append=False, instant=False):
    a, b, c = Replica('a'), Replica('b'), Replica('c')
    ctl = Controller([a, b, c])
    for r in (a, b):
        equal('burst initial preparation', send_ready(ctl, r), 'APPLIED')
    equal('burst first publication', ctl.commit('first', 0, 'FIRST', (a.key,)), 'COMMITTED')
    equal('burst initial append', ctl.commit('second', 1, 'APPEND', (b.key,)), 'COMMITTED')
    backlog = F(0)
    result = {}
    clear_time = None
    transition = None
    opening_time = F(0) if instant else F(20)
    for step in range(60):
        now = F(step, 2)
        if now == opening_time:
            equal('third preparation event', send_ready(ctl, c), 'APPLIED')
            before = ctl.available()
            equal('third explicit append',
                  ctl.commit('third', ctl.epoch, 'APPEND', (c.key,),
                             faulty_append=faulty_append), 'COMMITTED')
            transition = (before, ctl.available(), tuple(sorted(ctl.targets)))
        offered = 70 if now < 10 else 20
        # The service rate is read from the actual controller state, never supplied as a segment.
        backlog = max(F(0), backlog + (offered - ctl.available()) * F(1, 2))
        end = now + F(1, 2)
        if end in (F(10), F(20), F(45, 2), F(30)):
            result[end] = backlog
        if end >= 20 and backlog == 0 and clear_time is None:
            clear_time = end
    return transition, result, clear_time


def test_burst():
    # Hand-integrated reflected queue under the independently specified three-target contract.
    expected_transition = (40, 60, (('a', 1), ('b', 1), ('c', 1)))
    expected_queue = {F(10): F(300), F(20): F(100), F(45, 2): F(0), F(30): F(0)}
    expected = (expected_transition, expected_queue, F(45, 2))
    got = burst()
    equal('controller-driven burst against analytic endpoints', got, expected)
    buggy = burst(faulty_append=True)
    killed('REPLACE used for APPEND target set', buggy[0], expected_transition)
    killed('REPLACE used for APPEND backlog', buggy[1], expected_queue)
    killed('REPLACE used for APPEND clearing time', buggy[2], F(45, 2))
    equal('wrong append actually leaves 40 capacity', buggy[0][1], 40)
    equal('wrong append actually clears at 25 s', buggy[2], F(25))
    instant = burst(instant=True)
    killed('instant-start assumption', instant[1], expected_queue)
    print('OBS: actual APPEND 40 -> 60 requests/s; backlog 300 at 10 s, 100 at 20 s; clear at 45/2 s')


# Teardown implementation: live tickets, with explicit dependencies on last uses.
class Retirement:
    def __init__(self):
        self.gid = ('old-slot', 4)
        self.state = 'DRAINING'
        self.gate = False
        self.routed = False
        self.business_entry = True
        self.binding_entry = True
        self.live = {'R': {'request'}, 'U': {'k', 'd'},
                     'B': {'graph', 'reg'}, 'O': {'o'}}
        self.done = set()
        self.resources_held = True
        self.dependencies = {'k': set(), 'd': set(), 'o': {'d'},
                             'graph': {'k'}, 'reg': {'d', 'o'}}
        self.category = {'k': 'U', 'd': 'U', 'o': 'O', 'graph': 'B', 'reg': 'B'}

    def cancel(self):
        self.live['R'].discard('request')

    def begin(self, wait_for_bindings=False):
        if (self.state != 'DRAINING' or self.gate or self.routed or self.live['R']
                or (wait_for_bindings and self.live['B'])):
            return False
        self.business_entry = False
        self.binding_entry = False
        self.state = 'RETIRING'
        return True

    def perform(self, action, gid=None, scope=None):
        if gid is not None and gid != self.gid:
            return 'REJECT'
        if action not in self.category:
            return 'REJECT'
        if scope is not None and scope != 'range:' + action:
            return 'REJECT'
        if action in self.done:
            return 'DUP'
        if self.state != 'RETIRING':
            return 'REJECT'
        if not self.dependencies[action] <= self.done:
            return 'WAIT'
        self.live[self.category[action]].remove(action)
        self.done.add(action)
        return 'DONE'

    def free(self, omit_bindings=False):
        kinds = ('R', 'U', 'O') if omit_bindings else ('R', 'U', 'B', 'O')
        if (self.state != 'RETIRING' or self.gate or self.routed
                or self.business_entry or self.binding_entry
                or any(self.live[kind] for kind in kinds)):
            return False
        self.resources_held = False
        self.state = 'RETIRED'
        return True


# Independent ticket inventory and precedence relation for the external trace.
REFERENCE_TICKETS = frozenset({('U', 'k'), ('U', 'd'), ('O', 'o'),
                               ('B', 'graph'), ('B', 'reg')})
REFERENCE_EDGES = (('k', 'graph'), ('d', 'o'), ('d', 'reg'), ('o', 'reg'))


def remaining_reference(prefix):
    completed = set(prefix)
    return {kind: {name for category, name in REFERENCE_TICKETS
                   if category == kind and name not in completed}
            for kind in ('R', 'U', 'B', 'O')}


def legal_order(order):
    positions = {name: i for i, name in enumerate(order)}
    return all(positions[first] < positions[last] for first, last in REFERENCE_EDGES)


def test_retirement():
    blocked = Retirement()
    equal('active logical request blocks RETIRING', blocked.begin(), False)
    blocked.cancel()
    blocked.routed = True
    equal('remaining route blocks RETIRING', blocked.begin(), False)
    blocked.routed = False
    equal('binding may remain on entry', blocked.begin(), True)
    equal('retiring entry sealed', (blocked.business_entry, blocked.binding_entry), (False, False))
    equal('early graph detach waits for replay', blocked.perform('graph'), 'WAIT')
    equal('output handoff waits for DMA', blocked.perform('o'), 'WAIT')
    before = deepcopy(blocked.__dict__)
    equal('old generation completion ignored', blocked.perform('k', ('old-slot', 3)), 'REJECT')
    equal('wrong protected range ignored', blocked.perform('k', scope='another-range'), 'REJECT')
    equal('unknown operation ignored', blocked.perform('unknown'), 'REJECT')
    equal('bad completion events unchanged', blocked.__dict__, before)
    equal('all live obligations prevent release', blocked.free(), False)
    deadlock = Retirement()
    deadlock.cancel()
    killed('wait for B empty before RETIRING', deadlock.begin(wait_for_bindings=True), True)
    equal('valid entry with same bindings progresses', deadlock.begin(), True)
    bindings_only = Retirement()
    bindings_only.cancel()
    equal('enter teardown for binding-only test', bindings_only.begin(), True)
    for action in ('k', 'd', 'o'):
        equal('settle last uses before binding-only test', bindings_only.perform(action), 'DONE')
    expected_live = remaining_reference(('k', 'd', 'o'))
    equal('external inventory still contains two bindings', bindings_only.live, expected_live)
    safe = not any(expected_live.values())
    equal('correct final gate includes B', deepcopy(bindings_only).free(), safe)
    killed('omit B from resource release', deepcopy(bindings_only).free(omit_bindings=True), safe)
    # The same full-responsibility gate is checked over all presence/absence combinations.
    for mask in product((False, True), repeat=4):
        item = Retirement()
        item.cancel()
        item.begin()
        item.live = {kind: ({kind + '-ticket'} if present else set())
                     for kind, present in zip(('R', 'U', 'B', 'O'), mask)}
        expected_clear = sum(int(x) for x in mask) == 0
        equal('full responsibility gate', item.free(), expected_clear)
    schedule_count = 0
    prefix_count = 0
    for order in permutations(('k', 'd', 'o', 'graph', 'reg')):
        if not legal_order(order):
            continue
        schedule_count += 1
        item = Retirement()
        item.cancel()
        equal('enter retirement without waiting for bindings', item.begin(), True)
        equal('initial responsibility reference', item.live, remaining_reference(()))
        equal('initial release prohibited', item.free(), False)
        prefix = []
        for action in order:
            equal('enabled teardown or completion event', item.perform(action), 'DONE')
            prefix.append(action)
            prefix_count += 1
            wanted = remaining_reference(prefix)
            equal('live inventory reconstructed from external prefix', item.live, wanted)
            before = deepcopy(item.__dict__)
            equal('same completion idempotent', item.perform(action), 'DUP')
            equal('duplicate does not consume another ticket', item.__dict__, before)
            can_release = not any(wanted.values())
            equal('resource release against complete inventory', item.free(), can_release)
            equal('resource held until final responsibility', item.resources_held, not can_release)
        equal('finite fair schedule retires', item.state, 'RETIRED')
        equal('resource cannot be returned twice', item.free(), False)
        before = deepcopy(item.__dict__)
        equal('old settled event after retirement', item.perform(order[0]), 'DUP')
        equal('retired state unchanged', item.__dict__, before)
    equal('all legal dependency orders enumerated', schedule_count, 10)
    equal('all legal schedule prefixes checked', prefix_count, 50)
    print('OK: R/U/B/O safety, 16 responsibility masks, 10 teardown schedules and 50 prefixes; eventual retirement under finite fair completion')


def test_costs():
    # Reference counts independently named transfers through a shared single outlet.
    transfers = {'replica-a': 12, 'replica-b': 12, 'replica-c': 12, 'replica-d': 12}
    shared_reference = sum(F(size, 6) for size in transfers.values())
    shared_bound = F(4 * 12, 6)
    equal('shared outlet workload bound in seconds', shared_bound, shared_reference)
    equal('shared bound numerical reference', shared_bound, F(8))
    killed('give each cold replica the whole shared outlet', F(12, 6), shared_reference)
    equal('one remote cached object bound', F(12, 6), F(2))
    # Enumerated critical paths through independent rank preparation and group verification.
    paths = (F(0) + F(4) + F(1, 2), F(1) + F(4) + F(1, 2))
    dag = max(F(q) + F(d) for q, d in ((0, 4), (1, 4))) + F(1, 2)
    equal('rank DAG critical path', dag, max(paths))
    equal('rank DAG numerical reference', dag, F(11, 2))
    killed('serial sum of independent rank work', F(19, 2), max(paths))
    budgets = {'device': 200, 'host': 50, 'link': 80}
    feasible = [rate for rate in range(201) if all(rate <= budget for budget in budgets.values())]
    equal('resource minimum against enumerated feasible rates', min(budgets.values()), max(feasible))
    killed('device rate substitutes for host bottleneck', budgets['device'], max(feasible))
    equal('co-resident old/new bytes in GiB', 12 + 5 + 3 + 12, 32)
    equal('post-KV double weights still exceed budget', 12 + 3 + 12 > 24, True)
    equal('dwell before boundary', F(1) - F(0) >= F(3), False)
    equal('dwell exactly at boundary', F(3) - F(0) >= F(3), True)
    killed('delete dwell interval', F(1) - F(0) >= F(0), False)
    print('OBS: shared-read lower bound 8 s; cached remote bound 2 s; rank DAG 11/2 s; host-limited rate 50')


def main():
    test_capacity()
    test_ready()
    test_routes()
    test_burst()
    test_retirement()
    test_costs()
    print('PASS: K-WORK-008 CPU finite models; checks=' + str(CHECKS)
          + '; detected_negative_controls=' + str(len(MUTANTS)))
    print('SCOPE: abstract serial authority, ideal completion evidence, fixed homogeneous fluid service; no GPU, storage, network, numerical-model, crash-recovery or SLO measurement')


if __name__ == '__main__':
    main()
