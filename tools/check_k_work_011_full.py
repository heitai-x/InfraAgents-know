"""K-WORK-011: deterministic offline CPU logic models.
Node: (name, ((device_type, memory_units), ...), cpu_units).
Rank: (device_type, memory_units, cpu_units).
Events model command issuance only, not delivery or worker execution.
No project imports, files, subprocesses, network, credentials or devices.
"""
from itertools import product, permutations, combinations
from copy import deepcopy
from contextlib import contextmanager


def must(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_rejected(action, exception=AssertionError):
    try:
        action()
    except exception:
        return
    raise AssertionError('Expected rejection did not occur')


def validate(nodes, ranks, limit, budget):
    if not ranks or type(limit) is not int or limit < 1:
        raise ValueError('Nonempty ranks and positive candidate limit required')
    if type(budget) is not int or budget < 0:
        raise ValueError('Nonnegative integer budget required')
    if len({n[0] for n in nodes}) != len(nodes):
        raise ValueError('Duplicate physical node identity')
    for name, devices, cpu in nodes:
        if not isinstance(name, str) or type(cpu) is not int or cpu < 0:
            raise ValueError('Invalid node')
        for kind, memory in devices:
            if not isinstance(kind, str) or type(memory) is not int or memory < 0:
                raise ValueError('Invalid device')
    for kind, memory, cpu in ranks:
        if (not isinstance(kind, str) or type(memory) is not int or memory < 0
                or type(cpu) is not int or cpu < 0):
            raise ValueError('Invalid rank')


def generate(nodes, ranks, same_node=True, limit=1000, budget=10000):
    validate(nodes, ranks, limit, budget)
    slots = [(name, index, kind, memory)
             for name, devices, _ in nodes
             for index, (kind, memory) in enumerate(devices)]
    remaining = {name: cpu for name, _, cpu in nodes}
    used, path, plans = set(), [], []
    expanded = 0
    stop = 'COMPLETE'

    def search(i):
        nonlocal expanded, stop
        if i == len(ranks):
            plans.append(tuple(path))
            if len(plans) == limit:
                stop = 'CANDIDATE_LIMIT'
            return
        kind, memory, cpu = ranks[i]
        for name, index, available_kind, available_memory in slots:
            if stop != 'COMPLETE':
                return
            if expanded == budget:
                stop = 'BUDGET_EXHAUSTED'
                return
            expanded += 1
            sid = (name, index)
            if sid in used:
                continue
            if same_node and path and name != path[0][0]:
                continue
            if kind != available_kind or memory > available_memory:
                continue
            if cpu > remaining[name]:
                continue
            used.add(sid)
            path.append(sid)
            remaining[name] -= cpu
            search(i + 1)
            remaining[name] += cpu
            path.pop()
            used.remove(sid)

    search(0)
    must(not path and not used, 'Backtracking leaked slot state')
    must(remaining == {name: cpu for name, _, cpu in nodes},
         'Backtracking leaked aggregate resources')
    plans.sort()
    return {'plans': plans, 'stop': stop, 'expanded': expanded}


def reference_placements(nodes, ranks, same_node):
    # Construct complete injective physical assignments without pruning.
    physical = {}
    capacities = {}
    for name, devices, cpu in nodes:
        capacities[name] = cpu
        for index in range(len(devices)):
            physical[(name, index)] = devices[index]
    legal = set()
    for assignment in permutations(tuple(physical), len(ranks)):
        locations = [sid[0] for sid in assignment]
        if same_node and len(set(locations)) != 1:
            continue
        valid = True
        for i, sid in enumerate(assignment):
            actual_type, actual_memory = physical[sid]
            if actual_type != ranks[i][0] or actual_memory < ranks[i][1]:
                valid = False
        totals = {name: sum(ranks[i][2] for i in range(len(ranks))
                            if locations[i] == name)
                  for name in capacities}
        if any(totals[name] > capacities[name] for name in capacities):
            valid = False
        if valid:
            legal.add(assignment)
    return legal


def audit_plan(nodes, ranks, same_node, plan):
    must(len(plan) == len(ranks), 'Incomplete rank coverage')
    must(len(set(plan)) == len(plan), 'Repeated physical slot')
    physical = {(name, index): devices[index]
                for name, devices, _ in nodes
                for index in range(len(devices))}
    must(all(sid in physical for sid in plan), 'Unknown physical slot')
    must(not same_node or len({sid[0] for sid in plan}) == 1,
         'Same-node constraint violated')
    for i, sid in enumerate(plan):
        must(physical[sid][0] == ranks[i][0], 'Device type mismatch')
        must(physical[sid][1] >= ranks[i][1], 'Per-device memory exceeded')
    for name, _, capacity in nodes:
        total = sum(ranks[i][2] for i, sid in enumerate(plan) if sid[0] == name)
        must(total <= capacity, 'Node aggregate CPU exceeded')


def audit_result(nodes, ranks, same_node, result):
    reference = reference_placements(nodes, ranks, same_node)
    plans = result['plans']
    must(result['stop'] in {'COMPLETE', 'CANDIDATE_LIMIT', 'BUDGET_EXHAUSTED'},
         'Unknown stop state')
    must(len(plans) == len(set(plans)), 'Duplicate candidate')
    for plan in plans:
        audit_plan(nodes, ranks, same_node, plan)
        must(plan in reference, 'Returned assignment absent from reference')
    if result['stop'] == 'COMPLETE':
        must(set(plans) == reference, 'Exhaustive result differs from reference')
    if result['stop'] == 'CANDIDATE_LIMIT':
        must(bool(plans), 'Candidate limit cannot explain an empty result')
    return reference


def faulty_total_result(nodes, ranks):
    slots = [(name, index) for name, devices, _ in nodes
             for index in range(len(devices))]
    total_memory = sum(memory for _, devices, _ in nodes for _, memory in devices)
    accepts = (len(slots) >= len(ranks)
               and total_memory >= sum(rank[1] for rank in ranks))
    return {'plans': [tuple(slots[:len(ranks)])] if accepts else [],
            'stop': 'CANDIDATE_LIMIT' if accepts else 'COMPLETE', 'expanded': 0}


class ReserveFailure(Exception):
    pass


class SerialAdmission:
    """Only _decide_locked mutates admission ownership.
    PREPARE events are issued after that critical section and before return.
    Guards are instrumentation for a serial model, not production locks.
    """
    skip_prepare = False

    def __init__(self, nodes, current):
        self.nodes = nodes
        self.current = dict(current)
        self.owner = {(name, index): None for name, devices, _ in nodes
                      for index in range(len(devices))}
        self.free_cpu = {name: cpu for name, _, cpu in nodes}
        self.decisions = {}
        self.allocations = {}
        self.state = {}
        self.serial = 0
        self.call_count = 0
        self.locked = False
        self.events = []

    def emit(self, call, kind, key, data=()):
        # (call identity, kind, request key, immutable payload, lock observation)
        self.events.append((call, kind, key, data, self.locked))

    @contextmanager
    def critical(self, call, key):
        must(not self.locked, 'Nested critical section')
        self.locked = True
        self.emit(call, 'ENTER', key)
        try:
            yield
        finally:
            self.locked = False
            self.emit(call, 'EXIT', key)

    def admit(self, group, epoch, payload, ranks, plan, same_node=True,
              fail_after=None, fail_reserved=False):
        self.call_count += 1
        call = self.call_count
        key = (group, epoch)
        try:
            with self.critical(call, key):
                decision, commands = self._decide_locked(
                    call, group, epoch, payload, ranks, plan, same_node,
                    fail_after, fail_reserved)
        except Exception:
            self.emit(call, 'ERROR', key)
            raise
        # The early-return mutant deliberately omits this successor block.
        if not self.skip_prepare:
            for command in commands:
                self.dispatch(call, key, command)
        self.emit(call, 'RETURN', key, decision)
        return decision

    def dispatch(self, call, key, command):
        must(not self.locked, 'Node preparation dispatched in critical section')
        # An ideal event sink: no RPC, worker, device work or completion claim.
        self.emit(call, 'PREPARE', key, command)

    def _decide_locked(self, call, group, epoch, payload, ranks, plan,
                       same_node, fail_after, fail_reserved):
        must(self.locked, 'Authoritative decision outside critical section')
        validate(self.nodes, ranks, 1, 0)
        key = (group, epoch)
        fingerprint = (payload, tuple(ranks), same_node)
        if key in self.decisions:
            old_fingerprint, decision = self.decisions[key]
            if old_fingerprint != fingerprint:
                raise ValueError('Same identity with conflicting payload')
            self.emit(call, 'HISTORY', key, decision)
            return decision, ()
        if self.current.get(group) != epoch:
            raise ValueError('Stale or unauthorized epoch')
        if any(other[0] == group for other in self.allocations):
            raise ValueError('Older generation still owns resources')
        self.state[key] = 'PENDING'
        self.serial += 1
        token = (group, epoch, self.serial)
        journal = []
        commands = ()
        try:
            try:
                audit_plan(self.nodes, ranks, same_node, plan)
            except AssertionError as error:
                raise ReserveFailure('illegal-plan: ' + str(error))
            for i, sid in enumerate(plan):
                if fail_after == len(journal):
                    raise ReserveFailure('injected-partial-failure')
                cpu = ranks[i][2]
                if self.owner[sid] is not None:
                    raise ReserveFailure('slot-owned')
                if self.free_cpu[sid[0]] < cpu:
                    raise ReserveFailure('aggregate-exhausted')
                self.owner[sid] = token
                self.free_cpu[sid[0]] -= cpu
                journal.append((sid, cpu))
            self.state[key] = 'RESERVED'
            if fail_reserved:
                raise ReserveFailure('injected-after-reserved')
            must(len(journal) == len(ranks), 'Reservation incomplete')
            must(all(self.owner[sid] == token for sid, _ in journal),
                 'Reservation ownership lost')
            plan = tuple(plan)
            decision = ('COMMITTED', plan)
            self.allocations[key] = (token, plan, tuple(ranks))
            self.state[key] = 'COMMITTED'
            commands = tuple((token, payload, plan, i, sid)
                             for i, sid in enumerate(plan))
        except ReserveFailure as error:
            for sid, cpu in reversed(journal):
                must(self.owner[sid] == token, 'Rollback touched another owner')
                self.owner[sid] = None
                self.free_cpu[sid[0]] += cpu
            self.state[key] = 'PENDING'
            decision = ('REJECTED', str(error))
            self.state[key] = 'REJECTED'
        self.decisions[key] = (fingerprint, decision)
        if decision[0] == 'COMMITTED':
            self.emit(call, 'COMMIT', key, (token, payload, tuple(plan)))
        else:
            self.emit(call, 'REJECT', key, decision)
        return decision, commands

    def retire_quiescent(self, key, ideal_no_users):
        # The caller supplies ideal completion evidence in this CPU model.
        # This method does not establish real worker quiescence.
        must(ideal_no_users, 'Retirement requires completion evidence')
        with self.critical(0, key):
            if self.state.get(key) == 'RETIRED':
                return
            must(self.state.get(key) == 'COMMITTED', 'No live model allocation')
            token, plan, ranks = self.allocations.pop(key)
            for i, sid in enumerate(plan):
                must(self.owner[sid] == token, 'Retirement owner mismatch')
                self.owner[sid] = None
                self.free_cpu[sid[0]] += ranks[i][2]
            self.state[key] = 'RETIRED'
            self.emit(0, 'RETIRED', key)


class EarlyReturnAdmission(SerialAdmission):
    """Negative control: commit and return without issuing preparation."""
    skip_prepare = True


def audit_ledger(controller):
    # Rebuild live physical ownership independently from allocation records.
    expected_owner = {sid: None for sid in controller.owner}
    expected_cpu = {name: cpu for name, _, cpu in controller.nodes}
    for key, (token, plan, ranks) in controller.allocations.items():
        must(controller.state[key] == 'COMMITTED', 'Live allocation has wrong state')
        must(controller.decisions[key][1] == ('COMMITTED', plan),
             'Allocation has no matching unique decision')
        for i, sid in enumerate(plan):
            must(sid in expected_owner, 'Unknown slot in allocation ledger')
            must(expected_owner[sid] is None, 'Two decisions claim one slot')
            expected_owner[sid] = token
            expected_cpu[sid[0]] -= ranks[i][2]
    for key, (_, decision) in controller.decisions.items():
        if decision[0] == 'COMMITTED':
            must(key in controller.allocations or controller.state[key] == 'RETIRED',
                 'Committed decision lost its allocation without retirement')
            if controller.state[key] == 'RETIRED':
                must(key not in controller.allocations, 'Retired instance owns resources')
        else:
            must(controller.state[key] == 'REJECTED'
                 and key not in controller.allocations, 'Rejected identity owns resources')
    must(controller.owner == expected_owner, 'Orphan or missing slot ownership')
    must(controller.free_cpu == expected_cpu, 'Orphan or missing CPU charge')
    must(all(value >= 0 for value in expected_cpu.values()), 'CPU overcommitted')
    must(all(state in {'COMMITTED', 'REJECTED', 'RETIRED'}
             for state in controller.state.values()), 'Transient state escaped')
    must(not controller.locked, 'Critical section escaped')


def audit_prepares(controller):
    # Event specification is separate from command construction and ledger audit.
    # A COMMIT supplies the immutable placement; derive exact rank obligations.
    calls = {}
    for event in controller.events:
        call, kind, key, data, locked = event
        if call == 0:
            must(kind != 'PREPARE', 'Lifecycle operation started preparation')
            continue
        calls.setdefault(call, []).append(event)
    must(set(calls) == set(range(1, controller.call_count + 1)),
         'Missing invocation trace')
    committed_keys = set()
    for call in sorted(calls):
        events = calls[call]
        must(len({e[2] for e in events}) == 1, 'Invocation mixed request keys')
        key = events[0][2]
        enters = [i for i, e in enumerate(events) if e[1] == 'ENTER']
        exits = [i for i, e in enumerate(events) if e[1] == 'EXIT']
        terminals = [i for i, e in enumerate(events) if e[1] in {'RETURN', 'ERROR'}]
        must(enters == [0] and len(exits) == 1, 'Broken critical-section trace')
        exit_index = exits[0]
        must(terminals == [len(events) - 1] and exit_index < terminals[0],
             'Invocation returned before leaving critical section')
        must(events[0][4] and not events[exit_index][4], 'Incorrect lock observations')
        for i, e in enumerate(events):
            if e[1] == 'PREPARE':
                must(exit_index < i < terminals[0] and not e[4],
                     'Preparation must be outside lock and before return')
        commits = [e for e in events if e[1] == 'COMMIT']
        rejected = [e for e in events if e[1] == 'REJECT']
        history = [e for e in events if e[1] == 'HISTORY']
        prepares = [e[3] for e in events if e[1] == 'PREPARE']
        must(len(commits) <= 1, 'Multiple commits in one invocation')
        must(sum(bool(x) for x in (commits, rejected, history)) <= 1,
             'Invocation mixed commit, rejection and history')
        expected = []
        if commits:
            commit = commits[0]
            must(commit[4] and events.index(commit) < exit_index,
                 'Commit outside authoritative critical section')
            must(key not in committed_keys, 'Historical identity committed again')
            committed_keys.add(key)
            token, version, plan = commit[3]
            expected = [(token, version, plan, i, plan[i]) for i in range(len(plan))]
            must(events[-1][1] == 'RETURN'
                 and events[-1][3] == ('COMMITTED', plan),
                 'Fresh commit did not return its decision')
        must(prepares == expected,
             'Missing, duplicate, conflicting or unauthorized preparation commands')
        if events[-1][1] == 'RETURN':
            must(bool(commits or rejected or history), 'Return has no decision source')
    expected_commits = {key for key, (_, decision) in controller.decisions.items()
                        if decision[0] == 'COMMITTED'}
    must(committed_keys == expected_commits, 'Decision table and commit events differ')


def capacity_by_dependencies(resources, groups, failed):
    capacity = 0
    for rate, required in groups:
        dependencies = set()
        for resource in required:
            dependencies.update(resources[resource])
        if dependencies.isdisjoint(failed):
            capacity += rate
    return capacity


def reference_capacity(resources, groups, failed):
    # Remove actual resources first, then check group completeness.
    working = set(resources)
    for domain in failed:
        for name, domains in resources.items():
            if domain in domains:
                working.discard(name)
    return sum(rate for rate, required in groups
               if all(name in working for name in required))


def audit_capacity(resources, groups, failed, reported):
    must(reported == reference_capacity(resources, groups, failed),
         'Capacity includes an incomplete execution group')


def check_placement():
    variants = [
        ((), 0),
        ((('A', 1),), 0),
        ((('A', 1),), 1),
        ((('A', 2),), 1),
        ((('A', 1), ('A', 2)), 2),
        ((('B', 2),), 2),
    ]
    rank_options = [('A', 1, 1), ('A', 2, 1), ('B', 1, 0)]
    checked = 0
    for left, right in product(variants, repeat=2):
        nodes = (('n0', left[0], left[1]), ('n1', right[0], right[1]))
        for count in range(1, 4):
            for ranks in product(rank_options, repeat=count):
                for same_node in (False, True):
                    before = deepcopy((nodes, ranks))
                    result = generate(nodes, ranks, same_node)
                    must(result['stop'] == 'COMPLETE', 'Finite domain not exhausted')
                    must(result['expanded'] <= 10000, 'Expansion budget exceeded')
                    audit_result(nodes, ranks, same_node, result)
                    must((nodes, ranks) == before, 'Input was mutated')
                    checked += 1
    must(checked == 2808, 'Finite placement input domain changed unexpectedly')
    empty_ranks = (('A', 1, 1),)
    must(not audit_result((), empty_ranks, True, generate((), empty_ranks)),
         'Empty supply accepted')
    exact_nodes = (('n0', (('A', 8), ('A', 8)), 2),)
    exact_ranks = (('A', 8, 1), ('A', 8, 1))
    exact = generate(exact_nodes, exact_ranks)
    must(bool(audit_result(exact_nodes, exact_ranks, True, exact)),
         'Exact capacity boundary rejected')
    duplicate = dict(exact, plans=[(('n0', 0), ('n0', 0))])
    expect_rejected(lambda: audit_result(exact_nodes, exact_ranks, True, duplicate))
    cpu_nodes = (('n0', (('A', 8), ('A', 8)), 1),)
    must(not audit_result(cpu_nodes, exact_ranks, True,
                          generate(cpu_nodes, exact_ranks)), 'CPU limit ignored')
    memory_case = ((('n0', (('A', 8), ('A', 8)), 4),), (('A', 16, 1),))
    shape_nodes = (('n0', (('A', 16), ('A', 16)), 4),
                   ('n1', (('A', 16), ('A', 16)), 4))
    shape_case = (shape_nodes, (('A', 8, 1),) * 4)
    for nodes, ranks in (memory_case, shape_case):
        reference = audit_result(nodes, ranks, True, generate(nodes, ranks))
        faulty = faulty_total_result(nodes, ranks)
        must(bool(faulty['plans']) != bool(reference),
             'Total-resource negative control did not diverge')
        expect_rejected(lambda n=nodes, r=ranks, f=faulty: audit_result(n, r, True, f))
    for budget in range(7):
        result = generate(exact_nodes, exact_ranks, budget=budget)
        must(result['expanded'] <= budget, 'Failed branches escaped budget')
        audit_result(exact_nodes, exact_ranks, True, result)
    zero = generate(exact_nodes, exact_ranks, budget=0)
    must(zero['stop'] == 'BUDGET_EXHAUSTED' and not zero['plans'],
         'Zero budget incorrectly established infeasibility')
    limited = generate(exact_nodes, exact_ranks, limit=1)
    must(limited['stop'] == 'CANDIDATE_LIMIT' and len(limited['plans']) == 1,
         'Candidate output limit incorrect')
    audit_result(exact_nodes, exact_ranks, True, limited)
    impossible = generate(memory_case[0], memory_case[1], limit=1, budget=100)
    must(impossible['expanded'] > 1 and not impossible['plans'],
         'Output limit confused with search-work limit')
    expect_rejected(lambda: generate(exact_nodes, ()), ValueError)
    expect_rejected(lambda: generate(exact_nodes, exact_ranks, limit=0), ValueError)
    expect_rejected(lambda: generate(exact_nodes, exact_ranks, budget=-1), ValueError)
    print('placement finite cases =', checked,
          '; exact boundaries, backtracking and stop states audited')
    print('placement negative controls rejected: total memory, total slots, duplicate slot')


def audit_controller(controller):
    audit_ledger(controller)
    audit_prepares(controller)


def check_admission():
    nodes = (('n0', (('A', 8), ('A', 8), ('A', 8)), 3),)
    ranks = (('A', 8, 1),)
    plans = [(('n0', 0),), (('n0', 1),)]
    for order in permutations((0, 1)):
        controller = SerialAdmission(nodes, {'G': 1, 'H': 1})
        first = controller.admit('G', 1, 'v1', ranks, plans[order[0]])
        snapshot = deepcopy((controller.owner, controller.free_cpu))
        starts = sum(e[1] == 'PREPARE' for e in controller.events)
        second = controller.admit('G', 1, 'v1', ranks, plans[order[1]])
        must(first == second and first[0] == 'COMMITTED', 'Retry changed decision')
        must(snapshot == (controller.owner, controller.free_cpu), 'Retry charged twice')
        must(sum(e[1] == 'PREPARE' for e in controller.events) == starts == 1,
             'Retry started preparation again')
        expect_rejected(lambda: controller.admit('G', 1, 'v2', ranks, plans[0]),
                        ValueError)
        audit_controller(controller)
    for order in permutations(('G', 'H')):
        controller = SerialAdmission(nodes, {'G': 1, 'H': 1})
        outcomes = [controller.admit(g, 1, 'v1', ranks, plans[0])[0] for g in order]
        must(outcomes == ['COMMITTED', 'REJECTED'], 'Competing stale plan committed')
        audit_controller(controller)
    for fail_after, fail_reserved in ((1, False), (None, True)):
        controller = SerialAdmission(nodes, {'G': 1, 'H': 1})
        controller.admit('G', 1, 'v1', ranks, plans[0])
        before = deepcopy((controller.owner, controller.free_cpu))
        two_ranks = ranks * 2
        two_slots = (('n0', 1), ('n0', 2))
        rejected = controller.admit('H', 1, 'v1', two_ranks, two_slots,
                                    fail_after=fail_after, fail_reserved=fail_reserved)
        must(rejected[0] == 'REJECTED', 'Injected failure committed')
        must(before == (controller.owner, controller.free_cpu),
             'Rollback leaked or removed another allocation')
        must(controller.admit('H', 1, 'v1', two_ranks, two_slots) == rejected,
             'Rejected identity changed its original decision')
        audit_controller(controller)
    stale = SerialAdmission(nodes, {'G': 2})
    expect_rejected(lambda: stale.admit('G', 1, 'v1', ranks, plans[0]), ValueError)
    audit_controller(stale)
    controller = SerialAdmission(nodes, {'G': 1, 'H': 1})
    controller.admit('G', 1, 'v1', ranks, plans[0])
    controller.current['G'] = 2
    expect_rejected(lambda: controller.admit('G', 2, 'v1', ranks, plans[1]), ValueError)
    old_resources = deepcopy((controller.owner, controller.free_cpu))
    must(controller.admit('G', 1, 'v1', ranks, plans[1])[0] == 'COMMITTED',
         'Historical retry lost its decision')
    must(old_resources == (controller.owner, controller.free_cpu),
         'Historical retry acquired resources')
    audit_controller(controller)
    orphan = deepcopy(controller)
    orphan.owner[('n0', 1)] = ('G', 1, 'duplicate-retry')
    orphan.free_cpu['n0'] -= 1
    expect_rejected(lambda: audit_ledger(orphan))
    collision = deepcopy(controller)
    key, token = ('H', 1), ('H', 1, 99)
    collision.allocations[key] = (token, plans[0], ranks)
    collision.decisions[key] = (('v1', ranks, True), ('COMMITTED', plans[0]))
    collision.state[key] = 'COMMITTED'
    expect_rejected(lambda: audit_ledger(collision))
    print('admission: both serial orders, retries, conflicts, epochs and owned rollback audited')
    print('ledger negative controls rejected: orphan retry allocation and duplicate committed slot')


def check_start_events():
    nodes = (('n0', (('A', 8), ('A', 8), ('A', 8)), 3),)
    ranks = (('A', 8, 1), ('A', 8, 1))
    plan = (('n0', 0), ('n0', 1))
    alternate = (('n0', 1), ('n0', 2))
    key = ('G', 1)
    controller = SerialAdmission(nodes, {'G': 1, 'H': 1})
    first = controller.admit('G', 1, 'v1', ranks, plan)
    must(first == ('COMMITTED', plan), 'First admission did not commit')
    audit_controller(controller)
    starts = [e for e in controller.events if e[1] == 'PREPARE']
    must(len(starts) == 2 and {e[3][3] for e in starts} == {0, 1},
         'First admission did not issue every necessary rank command')
    must(controller.admit('G', 1, 'v1', ranks, alternate) == first,
         'Retry changed placement')
    retry_call = controller.call_count
    rejected = controller.admit('H', 1, 'v1', ranks, plan)
    rejection_call = controller.call_count
    must(rejected[0] == 'REJECTED', 'Competing group was not rejected')
    must(len([e for e in controller.events if e[1] == 'PREPARE']) == 2,
         'Retry or rejection issued a new preparation')
    audit_controller(controller)
    expect_rejected(lambda: controller.retire_quiescent(key, False))
    controller.retire_quiescent(key, True)
    released = deepcopy((controller.owner, controller.free_cpu))
    controller.retire_quiescent(key, True)
    must(released == (controller.owner, controller.free_cpu), 'Resources returned twice')
    must(controller.admit('G', 1, 'v1', ranks, alternate) == first,
         'Retired retry lost historical decision')
    retired_retry_call = controller.call_count
    must(controller.state[key] == 'RETIRED' and key not in controller.allocations,
         'Historical retry reactivated retired instance')
    must(released == (controller.owner, controller.free_cpu),
         'Retired retry reacquired resources')
    audit_controller(controller)

    # Exact regression: ledger is sound, but first commit has no prepare event.
    broken = EarlyReturnAdmission(nodes, {'G': 1})
    must(broken.admit('G', 1, 'v1', ranks, plan) == first,
         'Early-return mutant did not reproduce committed state')
    audit_ledger(broken)
    must(not any(e[1] == 'PREPARE' for e in broken.events),
         'Early-return mutant unexpectedly dispatched')
    expect_rejected(lambda: audit_prepares(broken))

    # Feed unauthorized event traces to the same event auditor.
    template = starts[0]
    for call in (retry_call, rejection_call, retired_retry_call):
        faulty = deepcopy(controller)
        return_index = next(i for i, e in enumerate(faulty.events)
                            if e[0] == call and e[1] == 'RETURN')
        call_key = faulty.events[return_index][2]
        faulty.events.insert(return_index,
                             (call, 'PREPARE', call_key, template[3], False))
        audit_ledger(faulty)
        expect_rejected(lambda f=faulty: audit_prepares(f))
    inside = deepcopy(controller)
    prepare_index = next(i for i, e in enumerate(inside.events) if e[1] == 'PREPARE')
    event = inside.events.pop(prepare_index)
    exit_index = next(i for i, e in enumerate(inside.events)
                      if e[0] == event[0] and e[1] == 'EXIT')
    inside.events.insert(exit_index, event[:4] + (True,))
    audit_ledger(inside)
    expect_rejected(lambda: audit_prepares(inside))
    missing_rank = deepcopy(controller)
    missing_rank.events.remove(starts[1])
    expect_rejected(lambda: audit_prepares(missing_rank))
    print('start events: first two-rank commit issued 2 commands outside lock before return')
    print('start events: retry, rejection and retired retry issued 0 additional commands')
    print('event negative controls rejected: early return, missing rank, inside-lock dispatch,')
    print('  dispatch after rejection, live retry and retired retry; ledger alone missed early return')


def check_faults():
    resources = {
        'gpuA': frozenset(('nodeA', 'rackA', 'power0')),
        'gpuA2': frozenset(('nodeA2', 'rackA', 'power0')),
        'gpuB': frozenset(('nodeB', 'rackB', 'power1')),
        'gpuC': frozenset(('nodeC', 'rackC', 'power2')),
        'exit': frozenset(('storageX',)),
    }
    independent = ((40, ('gpuA',)), (40, ('gpuB',)), (40, ('gpuC',)))
    correlated = ((40, ('gpuA',)), (40, ('gpuA2',)), (40, ('gpuC',)))
    cross = ((40, ('gpuA', 'gpuB', 'exit')), (40, ('gpuC',)))
    for domain in ('rackA', 'rackB', 'rackC'):
        reported = capacity_by_dependencies(resources, independent, {domain})
        audit_capacity(resources, independent, {domain}, reported)
        must(reported == 80 and reported >= 70, 'Independent-replica example changed')
    bad_capacity = capacity_by_dependencies(resources, correlated, {'rackA'})
    must(bad_capacity == 40 and bad_capacity < 70, 'Correlated fault was missed')
    domains = sorted(set().union(*resources.values()))
    checked = 0
    for count in range(3):
        for selected in combinations(domains, count):
            failed = frozenset(selected)
            for groups in (independent, correlated, cross):
                audit_capacity(resources, groups, failed,
                               capacity_by_dependencies(resources, groups, failed))
                checked += 1
    must(checked == 201, 'Finite fault input domain changed unexpectedly')
    observations = []
    for domain in ('rackA', 'rackB', 'storageX', 'rackC'):
        value = capacity_by_dependencies(resources, cross, {domain})
        must(value == 40, 'Incomplete cross-domain group retained capacity')
        observations.append((domain, value))
    must(capacity_by_dependencies(resources, cross, {'rackA', 'rackC'}) == 0,
         'Combined fault retained a dead group')
    representatives = ((40, 'rackB'), (40, 'rackC'))
    faulty = sum(rate for rate, representative in representatives
                 if representative != 'rackA')
    must(faulty != reference_capacity(resources, cross, {'rackA'}),
         'Representative-domain negative control did not diverge')
    expect_rejected(lambda: audit_capacity(resources, cross, {'rackA'}, faulty))
    audit_capacity(resources, correlated, {'rackA'}, bad_capacity)
    print('fault finite cases =', checked, '; independent rack capacities = [80, 80, 80]')
    print('correlated rackA capacity =', bad_capacity,
          '; cross-domain observations =', observations)
    print('fault negative control rejected: representative-domain capacity')


def main():
    print('K-WORK-011: offline CPU models; no GPU measurements or real concurrency')
    check_placement()
    check_admission()
    check_start_events()
    check_faults()
    print('All modeled checks completed without assertion failure.')
    print('Preparation events mean issuance only; no network delivery or worker execution.')
    print('Excluded: real locks, crashes, asynchronous revocation, devices, topology timing and SLO.')


if __name__ == '__main__':
    main()
