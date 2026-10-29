from decimal import Decimal, localcontext
from fractions import Fraction as F
from itertools import permutations
import math


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def detected(label, action):
    try:
        action()
    except AssertionError:
        print('DETECTED:', label)
        return
    raise AssertionError('negative control escaped: ' + label)


def rejected_value(label, action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError('invalid partition accepted: ' + label)


def dense(x, w, n):
    return tuple(tuple(sum((row[k] * w[k][j] for k in range(len(row))), F(0))
                       for j in range(n)) for row in x)


def shard(x, w, indices):
    rows = []
    for row in x:
        values = []
        for j in indices:
            acc = F(0)
            for k in reversed(range(len(row))):
                acc += row[k] * w[k][j]
            values.append(acc)
        rows.append(tuple(values))
    return tuple(rows)


def scatter(groups, m, n):
    out = [[None] * n for _ in range(m)]
    for indices, values in groups:
        if len(values) != m or any(len(row) != len(indices) for row in values):
            raise ValueError('shape')
        for t, j in enumerate(indices):
            if not 0 <= j < n:
                raise ValueError('range')
            for i in range(m):
                if out[i][j] is not None:
                    raise ValueError('duplicate')
                out[i][j] = values[i][t]
    if any(value is None for row in out for value in row):
        raise ValueError('missing')
    return tuple(tuple(row) for row in out)


def check_algebra():
    x = [[F(1, 2), F(-3), F(5, 3)], [F(2), F(0), F(-1)]]
    w = [[F(1), F(-2), F(3), F(0), F(5)],
         [F(2), F(1), F(-1), F(4), F(0)],
         [F(-3), F(2), F(1), F(0), F(2)]]
    ref = dense(x, w, 5)
    for mask in range(32):
        c = [j for j in range(5) if mask & (1 << j)]
        g = list(reversed([j for j in range(5) if j not in c]))
        got = scatter([(c, shard(x, w, c)), (g, shard(x, w, g))], 2, 5)
        require(got == ref, 'dense rational reference mismatch')
    require(scatter([([], ((), ()))], 2, 0) == ((), ()), 'empty output')
    zero = dense([[], []], [], 3)
    require(scatter([([0, 2], shard([[], []], [], [0, 2])),
                     ([1], shard([[], []], [], [1]))], 2, 3) == zero,
            'zero reduction dimension')
    rejected_value('missing', lambda: scatter([([0], ((F(1),),))], 1, 2))
    rejected_value('duplicate', lambda: scatter([([0, 0], ((F(1), F(2)),))], 1, 2))
    rejected_value('range', lambda: scatter([([2], ((F(1),),))], 1, 2))
    require(max(F(0), F(-2) + F(3)) != max(F(0), F(-2)) + max(F(0), F(3)),
            'premature nonlinear negative control did not differ')
    print('PASS: 32 rational output partitions, empty output, K=0; invalid coverage and premature ReLU detected')
    return ref[0]


def stats(indices, scores, values, dim):
    if not indices:
        return None
    m = max(scores[i] for i in indices)
    weights = [math.exp(scores[i] - m) for i in indices]
    l = math.fsum(weights)
    u = [math.fsum(weights[t] * values[i][d] for t, i in enumerate(indices))
         for d in range(dim)]
    return m, l, u


def merge_stats(parts, dim):
    nonempty = [part for part in parts if part is not None]
    if not nonempty:
        return tuple(0.0 for _ in range(dim)), True
    m = max(part[0] for part in nonempty)
    factors = [math.exp(part[0] - m) for part in nonempty]
    l = math.fsum(f * part[1] for f, part in zip(factors, nonempty))
    out = tuple(math.fsum(f * part[2][d] for f, part in zip(factors, nonempty)) / l
                for d in range(dim))
    return out, False


def decimal_attention(indices, scores, values, dim):
    if not indices:
        return tuple(0.0 for _ in range(dim)), True
    with localcontext() as ctx:
        ctx.prec = 70
        weights = [Decimal(str(scores[i])).exp() for i in indices]
        total = sum(weights, Decimal(0))
        out = tuple(float(sum((weights[t] * Decimal(str(values[i][d]))
                               for t, i in enumerate(indices)), Decimal(0)) / total)
                    for d in range(dim))
    return out, False


def check_attention():
    values = [[0, 1], [2, 1], [1, 3], [4, 2], [-3, 5]]
    cases = [([0.0, 1.0, 2.0, 3.0, -2.0], [[], [0, 2], [1, 3, 4]]),
             ([1000.0, -1000.0, 1001.0, 999.0, -999.0], [[0, 1], [], [2, 3, 4]]),
             ([-1000.0, -1001.0, -999.0, -998.0, -1002.0], [[0, 4], [1, 2, 3]]),
             ([0.0] * 5, [[], [], []]),
             ([0.0] * 5, [[4], []])]
    for scores, groups in cases:
        indices = [i for group in groups for i in group]
        require(len(indices) == len(set(indices)), 'test contains duplicate keys')
        got, empty = merge_stats([stats(group, scores, values, 2) for group in groups], 2)
        ref, ref_empty = decimal_attention(indices, scores, values, 2)
        require(empty == ref_empty, 'attention empty flag')
        require(max(abs(a - b) for a, b in zip(got, ref)) < 2e-12,
                'attention differs from independent Decimal direct reference')
    scores = [0.0, 1.0, 2.0, 3.0, -2.0]
    groups = [[0, 1], [2, 3, 4]]
    normalized = [merge_stats([stats(group, scores, values, 2)], 2)[0] for group in groups]
    wrong = tuple(sum(part[d] for part in normalized) / 2 for d in range(2))
    ref = decimal_attention(list(range(5)), scores, values, 2)[0]
    require(max(abs(a - b) for a, b in zip(wrong, ref)) > 1e-3,
            'equal-average faulty control escaped')
    print('PASS: attention versus 70-digit Decimal direct reference, empty/all-empty and extreme scores; wrong averaging detected')


IDENTITY = ('request-A', 4, 7, 11, 'plan-2', 3, 9)
UNITS = ('cpu', 'gpu')


class Hybrid:
    def __init__(self, groups, expected, mutation=None, omit=None):
        self.groups = {u: tuple(groups[u]) for u in UNITS}
        self.expected = tuple(expected)
        flat = [j for u in UNITS for j in self.groups[u]]
        if sorted(flat) != list(range(len(expected))):
            raise ValueError('incomplete or duplicate output partition')
        self.identity = IDENTITY
        self.active = IDENTITY
        self.mutation = mutation
        self.omit = omit
        self.pending = {u for u in UNITS if self.groups[u]}
        self.status = {u: 'SUBMITTED' if self.groups[u] else 'DONE_EMPTY' for u in UNITS}
        self.parts = {u: () for u in UNITS if not self.groups[u]}
        self.phase = 'WAITING'
        self.cancelled = False
        self.failed = False
        self.merge_pending = False
        self.consumer = False
        self.candidate = None
        self.visible = None
        self.history = []
        self.reads = []
        self.commits = []
        self.maybe_merge()

    def matches(self, identity):
        return len(identity) == len(self.identity) and all(
            a == b for i, (a, b) in enumerate(zip(identity, self.identity)) if i != self.omit)

    def maybe_merge(self):
        if self.phase != 'WAITING' or self.cancelled or self.failed or self.active != self.identity:
            return
        ready = all(self.status[u] in ('DONE', 'DONE_EMPTY') for u in UNITS)
        if self.mutation == 'early_merge':
            ready = any(self.status[u] == 'DONE' for u in UNITS)
        if not ready:
            return
        self.reads.append((len(self.history), frozenset(self.parts)))
        output = [F(0)] * len(self.expected)
        for u in UNITS:
            part = self.parts.get(u, (F(0),) * len(self.groups[u]))
            for j, value in zip(self.groups[u], part):
                output[j] = value
        self.candidate = tuple(output)
        self.phase = 'MERGING'
        self.merge_pending = True

    def done(self, unit, identity, values, ok=True, ended=True):
        values = tuple(values)
        self.history.append(('done', unit, tuple(identity), values, ok, ended))
        if unit not in self.pending or not self.matches(identity):
            return False
        if not ended:
            self.failed = True
            return False
        self.pending.remove(unit)
        if not ok or len(values) != len(self.groups[unit]):
            self.failed = True
            self.status[unit] = 'FAILED'
            return True
        self.parts[unit] = values
        self.status[unit] = 'DONE'
        self.maybe_merge()
        return True

    def merge_done(self, identity=IDENTITY, ok=True, ended=True):
        self.history.append(('merge', tuple(identity), ok, ended))
        if self.phase != 'MERGING' or identity != self.identity:
            return False
        if not ended:
            self.failed = True
            return False
        self.merge_pending = False
        self.phase = 'MERGED'
        if not ok:
            self.failed = True
        return True

    def cancel(self):
        self.history.append(('cancel',))
        self.cancelled = True

    def invalidate(self):
        new_identity = self.identity[:4] + ('plan-new',) + self.identity[5:]
        self.history.append(('invalidate', new_identity))
        self.active = new_identity
        self.failed = True

    def commit(self):
        self.history.append(('commit',))
        stopped = self.cancelled or self.failed
        if self.mutation == 'ignore_stop':
            stopped = False
        if self.phase != 'MERGED' or stopped or self.active != self.identity:
            return False
        self.visible = self.candidate
        self.phase = 'COMMITTED'
        self.consumer = True
        self.commits.append((len(self.history), self.visible))
        return True

    def consume(self):
        self.history.append(('consume',))
        if not self.consumer:
            return False
        self.consumer = False
        return True

    def can_reclaim(self):
        return (not self.pending and not self.merge_pending and not self.consumer
                and (self.cancelled or self.failed or self.phase == 'COMMITTED'))

    def snapshot(self):
        return (self.phase, frozenset(self.pending), tuple(sorted(self.status.items())),
                tuple(sorted(self.parts.items())), self.visible, self.candidate,
                self.cancelled, self.failed, self.merge_pending, self.consumer,
                tuple(self.reads), tuple(self.commits), self.active)


def reference_prefix(model, length):
    # Replay raw events using the full identity; do not call the tested matcher,
    # transition methods or cached accepted-completion set.
    pending = {u for u in UNITS if model.groups[u]}
    success = {u for u in UNITS if not model.groups[u]}
    failed = False
    cancelled = False
    active = IDENTITY
    merge_certificates = []
    for position, event in enumerate(model.history[:length], 1):
        if event[0] == 'done':
            _, unit, identity, values, ok, ended = event
            if unit in pending and identity == IDENTITY:
                if not ended:
                    failed = True
                else:
                    pending.remove(unit)
                    if ok and len(values) == len(model.groups[unit]):
                        success.add(unit)
                    else:
                        failed = True
        elif event[0] == 'cancel':
            cancelled = True
        elif event[0] == 'invalidate':
            active = event[1]
            failed = True
        elif event[0] == 'merge' and event[1] == IDENTITY:
            launched = any(read_position < position for read_position, _ in model.reads)
            if launched:
                if not event[2] or not event[3]:
                    failed = True
                if event[2] and event[3]:
                    merge_certificates.append(position)
    return pending, success, failed, cancelled, active, merge_certificates


def audit(model):
    pending, _, _, _, _, _ = reference_prefix(model, len(model.history))
    require(model.pending == pending, 'physical completion ledger differs from raw event reference')
    require(len(model.reads) <= 1 and len(model.commits) <= 1, 'duplicate merge or commit')
    for position, actually_read in model.reads:
        _, success, failed, cancelled, active, _ = reference_prefix(model, position)
        require(success == set(UNITS), 'merge read before all required successful completions')
        require(actually_read == frozenset(UNITS), 'read absent result handle')
        require(not failed and not cancelled and active == IDENTITY, 'merge after loss of eligibility')
    for position, output in model.commits:
        _, success, failed, cancelled, active, certificates = reference_prefix(model, position)
        require(success == set(UNITS), 'incomplete commit')
        require(not failed and not cancelled and active == IDENTITY, 'commit after loss of eligibility')
        require(bool(model.reads) and bool(certificates), 'commit without finished merge')
        require(model.reads[0][0] < certificates[-1] < position, 'merge/commit ordering')
        require(output == model.expected, 'committed output differs from independent dense reference')
    require(model.visible == (model.commits[0][1] if model.commits else None), 'visible bypasses commit')


def checked(model, method, *args, **kwargs):
    result = getattr(model, method)(*args, **kwargs)
    audit(model)
    return result


def complete_values(model, unit):
    return tuple(model.expected[j] for j in model.groups[unit])


def finish_computations(model, order):
    for unit in order:
        require(checked(model, 'done', unit, IDENTITY, complete_values(model, unit)), 'valid completion rejected')


def check_protocol(expected):
    groups_list = [({'cpu': [0, 2], 'gpu': [1, 3, 4]}, expected),
                   ({'cpu': [], 'gpu': list(range(5))}, expected),
                   ({'cpu': list(range(5)), 'gpu': []}, expected),
                   ({'cpu': [], 'gpu': []}, ())]
    normal_count = 0
    for groups, target in groups_list:
        required = [u for u in UNITS if groups[u]]
        for order in permutations(required):
            model = Hybrid(groups, target)
            audit(model)
            require(not checked(model, 'commit'), 'commit before merge')
            for unit in order:
                require(checked(model, 'done', unit, IDENTITY, complete_values(model, unit)), 'completion')
                require(model.visible is None, 'completion published output')
                require(model.phase == ('WAITING' if model.pending else 'MERGING'), 'success did not advance')
                require(not checked(model, 'commit'), 'commit before merge certificate')
            require(model.phase == 'MERGING', 'normal or all-empty path stalled')
            require(checked(model, 'merge_done'), 'merge certificate rejected')
            require(model.visible is None and model.phase == 'MERGED', 'merge bypassed final transaction')
            require(checked(model, 'commit') and model.visible == tuple(target), 'normal joint commit failed')
            require(not model.can_reclaim(), 'reclaimed with live consumer')
            require(not checked(model, 'commit'), 'duplicate commit')
            for unit in UNITS:
                before = model.snapshot()
                require(not checked(model, 'done', unit, IDENTITY, complete_values(model, unit)), 'duplicate accepted')
                require(before == model.snapshot(), 'rejected completion changed state')
            require(checked(model, 'consume') and model.can_reclaim(), 'consumer protection not released')
            normal_count += 1
    require(normal_count == 5, 'normal trace enumeration changed')
    print('PASS: normal joint commits =', normal_count, '(both completion orders, each empty side, all-empty output)')

    groups = groups_list[0][0]
    for field in range(len(IDENTITY)):
        wrong = list(IDENTITY)
        wrong[field] = wrong[field] + 1 if isinstance(wrong[field], int) else wrong[field] + '-old'
        wrong = tuple(wrong)
        require(sum(a != b for a, b in zip(wrong, IDENTITY)) == 1, 'identity mutation not isolated')
        good = Hybrid(groups, expected)
        before = good.snapshot()
        require(not checked(good, 'done', 'cpu', wrong, complete_values(good, 'cpu')), 'wrong identity accepted')
        require(before == good.snapshot(), 'wrong identity polluted state')
        finish_computations(good, ('gpu', 'cpu'))
        checked(good, 'merge_done')
        require(checked(good, 'commit'), 'stale rejection prevented current success')
        def faulty_identity(field=field, wrong=wrong):
            bad = Hybrid(groups, expected, omit=field)
            bad.done('cpu', wrong, complete_values(bad, 'cpu'))
            audit(bad)
        detected('omit exactly identity field ' + str(field), faulty_identity)

    for order in permutations(UNITS):
        def faulty_gate(order=order):
            bad = Hybrid(groups, expected, mutation='early_merge')
            bad.done(order[0], IDENTITY, complete_values(bad, order[0]))
            audit(bad)
        detected('merge without all completions, first=' + order[0], faulty_gate)

    for order in permutations(('cpu', 'gpu', 'cancel')):
        model = Hybrid(groups, expected)
        for action in order:
            if action == 'cancel':
                checked(model, 'cancel')
            else:
                checked(model, 'done', action, IDENTITY, complete_values(model, action))
        if model.merge_pending:
            checked(model, 'merge_done')
        require(not checked(model, 'commit') and model.visible is None, 'cancelled result committed')
        require(model.can_reclaim(), 'drained cancellation retained model responsibilities')
    for order in permutations(UNITS):
        model = Hybrid(groups, expected)
        for unit in order:
            checked(model, 'done', unit, IDENTITY, complete_values(model, unit), ok=(unit != 'cpu'))
        require(not checked(model, 'commit') and model.can_reclaim(), 'failure did not stop publication or drain')

    model = Hybrid(groups, expected)
    checked(model, 'done', 'cpu', IDENTITY, complete_values(model, 'cpu'), ended=False)
    checked(model, 'cancel')
    require('cpu' in model.pending and not model.can_reclaim(), 'incomplete certificate erased physical use')
    finish_computations(model, UNITS)
    require(not checked(model, 'commit') and model.can_reclaim(), 'failed generation resumed publication')

    model = Hybrid(groups, expected)
    finish_computations(model, UNITS)
    checked(model, 'merge_done', ended=False)
    require(model.merge_pending and not model.can_reclaim(), 'incomplete merge certificate erased physical use')
    checked(model, 'merge_done')
    require(not checked(model, 'commit') and model.can_reclaim(), 'merge failure eligibility not retained')

    for stop in ('cancel', 'invalidate'):
        model = Hybrid(groups, expected)
        finish_computations(model, UNITS)
        checked(model, 'merge_done')
        checked(model, stop)
        require(not checked(model, 'commit') and model.can_reclaim(), 'stop between merge and commit ignored')
    model = Hybrid(groups, expected)
    finish_computations(model, UNITS)
    checked(model, 'merge_done')
    checked(model, 'commit')
    saved = model.visible
    checked(model, 'cancel')
    require(model.visible == saved and not model.can_reclaim(), 'post-commit cancel revoked result or protection')
    checked(model, 'consume')
    require(model.can_reclaim(), 'post-commit cancellation failed to drain')

    def faulty_commit():
        bad = Hybrid(groups, expected, mutation='ignore_stop')
        finish_computations(bad, UNITS)
        checked(bad, 'merge_done')
        checked(bad, 'cancel')
        bad.commit()
        audit(bad)
    detected('commit ignores cancellation after merge', faulty_commit)
    print('PASS: raw-history audit, seven isolated identity fields, duplicates, cancellation orders, failures and retained responsibilities')


def resource_bound(flows, bandwidth):
    traffic = {}
    for resource, byte_count in flows:
        traffic[resource] = traffic.get(resource, 0) + byte_count
    return max((F(count, bandwidth[resource]) for resource, count in traffic.items()), default=F(0))


def service_schedule(flows, bandwidth, release_period, count):
    # Construct service intervals with one transfer at a time per resource.
    available = {resource: F(0) for resource in bandwidth}
    completions = []
    intervals = []
    for unit in range(count):
        release = unit * release_period
        ends = []
        for resource, byte_count in flows:
            start = max(release, available[resource])
            end = start + F(byte_count, bandwidth[resource])
            intervals.append((resource, start, end))
            available[resource] = end
            ends.append(end)
        completions.append(max(ends, default=release))
    for resource in bandwidth:
        spans = sorted((start, end) for r, start, end in intervals if r == resource)
        require(all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1)),
                'reference schedule overbooks resource')
    return completions


def check_cost():
    mib, gib = 1024 ** 2, 1024 ** 3
    bandwidth = 16 * gib
    weight_ms = F(1000 * 512 * mib, bandwidth)
    activation_ms = F(1000 * 2 * mib, bandwidth)
    move = max(weight_ms, F(6)) + F(1) + F(1, 2)
    cpu = max(F(6), F(8) + activation_ms + F(1)) + F(1, 2)
    resident = F(6) + F(1)
    high_cpu = max(F(6), F(80) + activation_ms + F(1)) + F(1, 2)
    require(weight_ms == F(125, 4) and activation_ms == F(125, 1024), 'unit conversion')
    require(move == F(131, 4) and cpu == F(9853, 1024), 'candidate arithmetic')
    require(resident < cpu < move < high_cpu, 'conditional ordering did not reverse')
    boundary = move - F(1, 2) - F(1) - activation_ms
    path = lambda tc: max(F(6), tc + activation_ms + F(1)) + F(1, 2)
    require(path(boundary - F(1, 10)) < move < path(boundary + F(1, 10)), 'break-even')

    q, b = 64 * mib, 8 * gib
    shared_flows = [('dram', q), ('dram', q)]
    independent_flows = [('r0', q), ('r1', q)]
    shared_bw, independent_bw = {'dram': b}, {'r0': b, 'r1': b}
    shared_bound = resource_bound(shared_flows, shared_bw)
    independent_bound = resource_bound(independent_flows, independent_bw)
    # Fixed periods go into a constructive reference, not back into the tested formula.
    shared_ends = service_schedule(shared_flows, shared_bw, F(1, 64), 5)
    independent_ends = service_schedule(independent_flows, independent_bw, F(1, 128), 5)
    observed_shared = shared_ends[1] - shared_ends[0]
    observed_independent = independent_ends[1] - independent_ends[0]
    require(all(shared_ends[i + 1] - shared_ends[i] == observed_shared for i in range(4)), 'shared period')
    require(all(independent_ends[i + 1] - independent_ends[i] == observed_independent for i in range(4)), 'independent period')
    require(shared_bound == observed_shared == F(1, 64), 'shared physical aggregation')
    require(independent_bound == observed_independent == F(1, 128), 'independent max bound')
    wrong_isolated_max = max(F(count, shared_bw[r]) for r, count in shared_flows)
    wrong_cross_resource_sum = sum((F(q, independent_bw[r]) for r in independent_bw), F(0))
    def assert_shared_period(value):
        require(value == observed_shared, 'candidate differs from constructed shared service')
    def assert_feasible_lower_bound(value):
        require(value <= observed_independent, 'lower bound excludes a feasible independent schedule')
    detected('isolated max undercounts shared DRAM', lambda: assert_shared_period(wrong_isolated_max))
    detected('cross-resource sum excludes feasible overlap', lambda: assert_feasible_lower_bound(wrong_cross_resource_sum))
    detected('resident path retains weight transfer', lambda: require(move == resident, 'unnecessary transfer'))
    serial_single_latency = F(q, b) + F(q, b)
    require(serial_single_latency == F(1, 64) and serial_single_latency > independent_bound,
            'single-unit dependency latency confused with period')
    require(resource_bound([], {}) == 0, 'empty resource demand')
    print('PASS: exact cost arithmetic, resident baseline and break-even reversal')
    print('PASS: constructive independent period =', float(observed_independent * 1000),
          'ms; shared DRAM period =', float(observed_shared * 1000), 'ms')


def main():
    expected = check_algebra()
    check_attention()
    check_protocol(expected)
    check_cost()
    print('SCOPE: deterministic CPU rational/Decimal arithmetic, finite event protocol and analytic service schedules only.')
    print('LIMITS: no GPU, DMA, PCIe, NUMA, kernel, memory-visibility, real KV-update, predictor-quality, latency or Agent-benefit measurement.')
    print('LIMITS: one serial state owner; completion certificates are model inputs, with abstract shard/merge responsibilities and one consumer.')


if __name__ == '__main__':
    main()
