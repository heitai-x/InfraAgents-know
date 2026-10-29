#!/usr/bin/env python3
"""K-HW-007: deterministic, standalone, read-only, offline CPU models.

No project imports, file access, subprocesses, network, MLX, Metal or GPU.
Integer causal arithmetic checks protocol structure, not transformer numerics.
"""
from dataclasses import dataclass, replace
from fractions import Fraction

L = 2
GIB = 1024 ** 3
DETECTED = []
UNSET = object()
LOGITS = ('logits',)
RHO = ('rng_input',)
SAMPLE = ('sample',)
TOKEN = ('sample_token',)
NEXT_RHO = ('sample_rng',)


@dataclass(frozen=True)
class Cell:
    k: int
    v: int


@dataclass(frozen=True)
class State:
    x: tuple
    k: int
    kv: tuple
    rho: int = 3
    version: int = 0
    ell: int = -1
    logits: object = None
    logits_id: object = None


@dataclass(frozen=True)
class Prepared:
    base: tuple
    delta: tuple
    b: int
    e: int
    k_after: int
    logits: object
    logits_id: object
    install: bool
    roots: frozenset


def identity(x):
    return ('toy-model-v1', tuple(x))


def dense_reference(x):
    # Independent layer-major evaluation of the entire prefix.
    hidden = [5 * token + t + 1 for t, token in enumerate(x)]
    all_kv = []
    for layer in range(L):
        keys = [(layer + 2) * value + t for t, value in enumerate(hidden)]
        values = [(layer + 3) * value - t for t, value in enumerate(hidden)]
        next_hidden = []
        for t, value in enumerate(hidden):
            causal = 0
            for j in range(t + 1):
                causal += (j + 1) * keys[j] + values[j]
            next_hidden.append(2 * value + causal + layer + 1)
        all_kv.append(tuple(Cell(a, b) for a, b in zip(keys, values)))
        hidden = next_hidden
    logits = None
    if x:
        logits = tuple(((i + 1) * hidden[-1] + i * i) % 97 for i in range(7))
    return tuple(all_kv), logits


def make_state(x, k, cached=False):
    x = tuple(x)
    kv, _ = dense_reference(x[:k])
    if cached:
        assert k == len(x) and x
        _, logits = dense_reference(x)
        return State(x, k, kv, ell=len(x) - 1,
                     logits=logits, logits_id=identity(x))
    return State(x, k, kv)


def validate(s, q):
    if type(q) is not int or q < 1:
        raise ValueError('q must be a positive integer')
    if not s.x or not 0 <= s.k <= len(s.x):
        raise ValueError('invalid known/KV boundary')
    if len(s.kv) != L or any(len(row) != s.k for row in s.kv):
        raise ValueError('incomplete layer state')
    if s.logits is None:
        if s.ell != -1 or s.logits_id is not None:
            raise ValueError('invalid absent-logits metadata')
    elif not (s.k == len(s.x) and s.ell == len(s.x) - 1
              and s.logits_id == identity(s.x)):
        raise ValueError('incompatible cached logits')


def prepare(s, q, fault=None):
    validate(s, q)
    n = len(s.x)
    base = s.kv
    if fault == 'copy_prefix':
        base = tuple(tuple(Cell(c.k, c.v) for c in row) for row in base)
    empty = tuple(() for _ in range(L))
    if s.k == n and s.logits is not None:
        return Prepared(base, empty, n, n, n, s.logits,
                        s.logits_id, False, frozenset())
    if s.k == n and fault == 'full_stall':
        return Prepared(base, empty, n, n, n, None, None, False, frozenset())
    repair = s.k == n
    # b/e describe actual toy-model work, including private last-row repair.
    b = n - 1 if repair else s.k
    e = n if repair else min(n, s.k + q)
    delta = [[] for _ in range(L)]
    for t in range(b, e):
        h = 5 * s.x[t] + t + 1
        for layer in range(L):
            cell = Cell((layer + 2) * h + t, (layer + 3) * h - t)
            delta[layer].append(cell)
            local = base[layer][:b] + tuple(delta[layer])
            causal = 0
            for j, prior in enumerate(local):
                causal += (j + 1) * prior.k + prior.v
            h = 2 * h + causal + layer + 1
    logits = None
    lid = None
    if e == n or fault == 'middle_sample':
        logits = tuple(((i + 1) * h + i * i) % 97 for i in range(7))
        lid = identity(s.x[:e])
    roots = {('kv', layer, t) for layer in range(L) for t in range(b, e)}
    if logits is not None:
        roots.add(LOGITS)
    return Prepared(base, tuple(tuple(row) for row in delta), b, e,
                    n if repair else e, logits, lid, not repair, frozenset(roots))


def installed_kv(p):
    if not p.install:
        return p.base
    return tuple(p.base[layer][:p.b] + p.delta[layer] for layer in range(L))


def sample(logits, rho):
    assert logits is not None, 'sample accessed absent logits'
    rho_next = (17 * rho + 11) % 101
    token = (logits.index(max(logits)) + rho_next) % 7
    return token, rho_next


def assert_prepared(s, p):
    reference_kv, reference_logits = dense_reference(s.x[:p.k_after])
    got = installed_kv(p)
    assert got == reference_kv, 'incremental KV differs from full-prefix reference'
    for layer in range(L):
        for t in range(s.k):
            assert got[layer][t] is s.kv[layer][t], 'old physical prefix was copied'
    if p.k_after < len(s.x):
        assert p.logits is None, 'sampling from an intermediate prefix'
        assert p.logits_id is None
    else:
        assert p.logits == reference_logits, 'missing/wrong complete-prefix logits'
        assert p.logits_id == identity(s.x)
    expected_roots = {('kv', layer, t)
                      for layer in range(L) for t in range(p.b, p.e)}
    if p.logits is not None and p.b != p.e:
        expected_roots.add(LOGITS)
    assert p.roots == frozenset(expected_roots)
    if s.k == len(s.x) and s.logits is None:
        assert p.b == len(s.x) - 1 and p.e == len(s.x)
        assert not p.install and all(len(row) == 1 for row in p.delta)
        assert got is s.kv, 'repair installed recomputed KV'


class SamplingExecution:
    """Execute host visibility or construct/evaluate a private sampling DAG.

    Graph completion and HOST_VISIBLE are CPU model events. Neither represents
    real hardware synchronization. Unknown roots and absent accesses fail.
    """
    def __init__(self, s, p, mode):
        assert mode in ('host', 'device')
        self.s, self.p, self.mode = s, p, mode
        self.deps = {node: () for node in p.roots}
        self.values = {}
        self.completed = set()
        if p.logits is not None:
            parents = tuple(sorted(node for node in p.roots if node[0] == 'kv'))
            self.deps[LOGITS] = parents
            self.values[LOGITS] = p.logits
            if not p.roots:
                # Compatible cache carries prior result-completion evidence.
                assert s.logits is p.logits and s.logits_id == identity(s.x)
                self.completed.add(LOGITS)
        self.roots = set(p.roots)
        self.events = []
        self.eval_snapshots = []
        self.token, self.rho_next = UNSET, UNSET
        self.host_visible = False

    def visible_logits(self):
        self.events.append('HOST_VISIBLE')
        assert self.p.logits is not None, 'HOST_VISIBLE accessed absent logits'
        assert LOGITS in self.deps, 'HOST_VISIBLE referenced nonexistent node'
        assert LOGITS in self.completed, 'HOST_VISIBLE before result completion'
        self.host_visible = True
        return self.values[LOGITS]

    def evaluate(self):
        self.events.append('EVAL')
        self.eval_snapshots.append((frozenset(self.roots), self.token, self.rho_next))
        visiting = set()

        def visit(node):
            assert node in self.deps, 'EVAL referenced nonexistent root/dependency'
            if node in self.completed:
                return self.values.get(node)
            assert node not in visiting, 'cycle in graph'
            visiting.add(node)
            args = [visit(parent) for parent in self.deps[node]]
            if node == SAMPLE:
                self.events.append('DEVICE_SAMPLE')
                self.values[node] = sample(args[0], args[1])
            elif node == TOKEN:
                self.values[node] = args[0][0]
            elif node == NEXT_RHO:
                self.values[node] = args[0][1]
            elif node == LOGITS:
                assert self.p.logits is not None
                self.values[node] = self.p.logits
            elif node == RHO:
                self.values[node] = self.s.rho
            else:
                assert node[0] == 'kv'
                self.values[node] = None
            visiting.remove(node)
            self.completed.add(node)
            return self.values[node]

        for root in sorted(self.roots):
            visit(root)

    def run(self, fault=None):
        n = len(self.s.x)
        if self.p.e < n:
            assert self.p.logits is None and self.p.k_after == self.p.e
            has_full_logits = False
        else:
            assert self.p.e == n and self.p.k_after == n
            assert self.p.logits is not None, 'complete branch missing logits'
            assert self.p.logits_id == identity(self.s.x)
            has_full_logits = True
        if self.mode == 'host':
            self.evaluate()
            if fault == 'host_before_guard':
                self.visible_logits()
            if has_full_logits:
                z = self.visible_logits()
                assert self.host_visible
                self.events.append('HOST_SAMPLE')
                self.token, self.rho_next = sample(z, self.s.rho)
            else:
                self.events.append('NO_SAMPLE')
                self.token, self.rho_next = None, self.s.rho
        else:
            if has_full_logits:
                self.deps[RHO] = ()
                self.deps[SAMPLE] = (LOGITS, RHO)
                self.deps[TOKEN] = (SAMPLE,)
                self.deps[NEXT_RHO] = (SAMPLE,)
                self.roots.update((TOKEN, NEXT_RHO))
            else:
                self.events.append('NO_SAMPLE')
                self.token, self.rho_next = None, self.s.rho
            if fault == 'device_unconditional_roots':
                # Reproduce the reviewed defect: outputs have no producer.
                self.roots.update((TOKEN, NEXT_RHO))
            self.evaluate()
            if has_full_logits:
                self.events.append('RESULT_READ')
                assert TOKEN in self.completed and NEXT_RHO in self.completed
                self.token, self.rho_next = self.values[TOKEN], self.values[NEXT_RHO]
        assert self.token is not UNSET and self.rho_next is not UNSET
        return self.token, self.rho_next


def assert_sampling_execution(s, p, execution):
    # Oracle uses full-prefix completion, not the driver's has-logits branch.
    full = p.k_after == len(s.x)
    expected_roots = set(p.roots)
    if full:
        _, z = dense_reference(s.x)
        expected_rho = (s.rho * 17 + 11) % 101
        best = max(range(7), key=lambda i: z[i])
        expected = ((best + expected_rho) % 7, expected_rho)
        if execution.mode == 'device':
            expected_roots.update((TOKEN, NEXT_RHO))
            expected_events = ['EVAL', 'DEVICE_SAMPLE', 'RESULT_READ']
        else:
            expected_events = ['EVAL', 'HOST_VISIBLE', 'HOST_SAMPLE']
    else:
        expected = (None, s.rho)
        assert p.logits is None
        assert LOGITS not in execution.deps
        assert all(node not in execution.deps for node in (SAMPLE, TOKEN, NEXT_RHO))
        if execution.mode == 'device':
            expected_events = ['NO_SAMPLE', 'EVAL']
            # Inspect values at the EVAL boundary, not just after run().
            _, token_at_eval, rho_at_eval = execution.eval_snapshots[0]
            assert token_at_eval is None and rho_at_eval == s.rho
        else:
            expected_events = ['EVAL', 'NO_SAMPLE']
    assert (execution.token, execution.rho_next) == expected
    assert execution.roots == expected_roots, 'wrong result-root set'
    assert execution.events == expected_events, 'wrong visibility/sampling order'
    assert execution.roots <= execution.completed
    assert p.roots <= execution.completed


class Owner:
    def __init__(self, state):
        self.s = state
        self.live = True
        self.eligible = True
        self.canceled = False
        self.failed = False
        self.phase = 'READY'
        self.active = None
        self.serial = 0
        self.pending = set()
        self.records = ()
        self.delivered = ()
        self.snap = None
        self.p = None
        self.results = frozenset()

    def start(self, q, fault=None):
        if (not self.live or not self.eligible or self.canceled or self.failed
                or self.active is not None or self.pending
                or self.phase not in ('READY', 'COMMITTED')):
            raise ValueError('request not eligible for execution')
        self.snap = self.s
        self.p = prepare(self.snap, q)
        self.serial += 1
        self.active = ('request-instance', self.serial)
        self.pending.add((self.active, 'eval'))
        self.phase = 'EVALUATING'
        self.results = frozenset()
        if fault == 'early_publish':
            self.records += ((len(self.records), 99),)
        if fault == 'overwrite_committed':
            rows = list(self.s.kv)
            row = list(rows[0])
            row[0] = Cell(row[0].k, row[0].v + 1)
            rows[0] = tuple(row)
            self.s = replace(self.s, kv=tuple(rows))
        return self.active

    def use_done(self, execution, complete=True):
        key = (execution, 'eval')
        if execution != self.active or key not in self.pending or not complete:
            return
        self.pending.remove(key)
        if not self.eligible:
            self.phase = 'TERMINAL' if not self.pending else 'DRAINING'

    def fail(self, ended=False):
        self.failed = True
        self.live = False
        self.eligible = False
        if ended:
            self.use_done(self.active)
        self.phase = 'DRAINING' if self.pending else 'TERMINAL'

    def cancel(self):
        self.canceled = True
        self.live = False
        self.eligible = False
        self.phase = 'DRAINING' if self.pending else 'TERMINAL'

    def finish(self, roots=None, ended=True):
        self.results = self.p.roots if roots is None else frozenset(roots)
        if ended:
            self.use_done(self.active)
        if not self.p.roots <= self.results:
            self.fail()
            return False
        if not self.live or not self.eligible or self.canceled or self.failed:
            self.phase = 'DRAINING' if self.pending else 'TERMINAL'
            return False
        if self.pending:
            return False
        self.phase = 'READY_TO_COMMIT'
        return True

    def commit(self, token, rho_next, fault=None):
        if self.active is None or self.s.version != self.snap.version or self.pending:
            return False
        if fault != 'allow_cancel':
            if (not self.live or not self.eligible or self.canceled or self.failed
                    or self.phase != 'READY_TO_COMMIT'
                    or not self.p.roots <= self.results):
                return False
        if self.p.logits is None:
            if token is not None or rho_next != self.snap.rho:
                return False
        elif (token is None or self.p.k_after != len(self.snap.x)
              or self.p.logits_id != identity(self.snap.x)):
            return False
        x = self.snap.x if token is None else self.snap.x + (token,)
        k = self.p.k_after
        if fault == 'advance_k' and token is not None:
            k = len(x)
        rho = self.snap.rho if token is None else rho_next
        if fault == 'wrong_rng' and token is not None:
            rho = self.snap.rho
        records = self.records
        if token is not None:
            records += ((len(records), token),)
        # Serialized business linearization point; no external callbacks.
        self.s = State(x, k, installed_kv(self.p), rho, self.snap.version + 1)
        self.records = records
        self.phase = 'COMMITTED'
        self.active = None
        return True

    def deliver(self):
        self.delivered = self.records
        return self.delivered


def expect_detected(name, operation):
    try:
        operation()
    except AssertionError:
        DETECTED.append(name)
    else:
        raise AssertionError('negative control escaped: ' + name)


def expect_value_error(operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError('invalid input accepted')


def check_preparation():
    count = 0
    for n in range(1, 7):
        x = tuple((3 * t + n) % 7 for t in range(n))
        for k in range(n + 1):
            for q in range(1, 5):
                s = make_state(x, k)
                before = repr(s)
                assert_prepared(s, prepare(s, q))
                assert repr(s) == before
                count += 1
        s = make_state(x, n, cached=True)
        p = prepare(s, 1)
        assert_prepared(s, p)
        assert p.b == p.e == n and not p.roots
        assert p.logits is s.logits and installed_kv(p) is s.kv
    s = make_state((2, 1, 3, 0), 2)
    for q in (0, -1, True):
        expect_value_error(lambda q=q: prepare(s, q))
    expect_value_error(lambda: prepare(make_state((), 0), 1))
    expect_value_error(lambda: prepare(replace(s, k=5), 1))
    expect_value_error(lambda: prepare(replace(s, kv=(s.kv[0], ())), 1))
    cached = make_state((2, 1, 3), 3, cached=True)
    expect_value_error(lambda: prepare(replace(cached, logits_id=identity((3, 1, 2))), 1))
    expect_detected('middle-prefix-sampling',
                    lambda: assert_prepared(s, prepare(s, 1, 'middle_sample')))
    full = make_state((2, 1, 3), 3)
    expect_detected('full-KV-stall',
                    lambda: assert_prepared(full, prepare(full, 1, 'full_stall')))
    expect_detected('physical-prefix-copy',
                    lambda: assert_prepared(s, prepare(s, 1, 'copy_prefix')))
    return count


def check_sampling_paths():
    count = 0
    for n in range(1, 7):
        x = tuple((3 * t + n) % 7 for t in range(n))
        for k in range(n + 1):
            for q in range(1, 5):
                s = make_state(x, k)
                p = prepare(s, q)
                for mode in ('host', 'device'):
                    execution = SamplingExecution(s, p, mode)
                    execution.run()
                    assert_sampling_execution(s, p, execution)
                    count += 1
        s = make_state(x, n, cached=True)
        p = prepare(s, 1)
        for mode in ('host', 'device'):
            execution = SamplingExecution(s, p, mode)
            execution.run()
            assert_sampling_execution(s, p, execution)
            count += 1
    # Exact legal witness from R2-A01; execute both full control-flow paths.
    s = make_state((2, 1, 3, 0), 2)
    p = prepare(s, 1)
    assert p.e == 3 < len(s.x) and p.logits is None
    observations = {}
    for mode in ('host', 'device'):
        execution = SamplingExecution(s, p, mode)
        assert execution.run() == (None, s.rho)
        assert_sampling_execution(s, p, execution)
        observations[mode] = {'events': tuple(execution.events),
                              'roots': tuple(sorted(execution.roots)),
                              'token': execution.token,
                              'rho_next': execution.rho_next}
        count += 1
    expect_detected('host-empty-logits-before-guard',
                    lambda: SamplingExecution(s, p, 'host').run('host_before_guard'))
    expect_detected('device-nonexistent-sampling-roots',
                    lambda: SamplingExecution(s, p, 'device').run('device_unconditional_roots'))
    # A complete branch with a missing result is failure, not a silent prefill.
    full = make_state((2, 1, 3), 2)
    full_p = prepare(full, 1)
    broken = replace(full_p, logits=None, logits_id=None)
    expect_detected('complete-branch-missing-result',
                    lambda: SamplingExecution(full, broken, 'host').run())
    return count, observations


def check_multistep(mode):
    owner = Owner(make_state((2, 1, 3, 0), 0))
    outputs = []
    for _ in range(7):
        old = owner.s
        old_records = owner.records
        delivered = owner.deliver()
        owner.start(2)
        assert_prepared(old, owner.p)
        assert owner.s is old and owner.records == old_records
        execution = SamplingExecution(old, owner.p, mode)
        token, rho_next = execution.run()
        assert_sampling_execution(old, owner.p, execution)
        assert owner.s is old and owner.records == old_records
        assert owner.deliver() == delivered
        assert owner.finish(roots=execution.completed)
        assert owner.commit(token, rho_next)
        assert owner.s.version == old.version + 1
        assert owner.s.ell == -1 and owner.s.logits is None
        if token is None:
            assert owner.s.x == old.x and owner.s.rho == old.rho
            assert owner.s.k == min(len(old.x), old.k + 2)
            assert owner.records == old_records
        else:
            assert owner.s.x == old.x + (token,)
            assert owner.s.k == len(old.x) == len(owner.s.x) - 1
            assert owner.s.kv == dense_reference(old.x)[0]
            assert owner.s.rho == rho_next
            assert owner.records == old_records + ((len(old_records), token),)
            outputs.append(token)
        assert owner.deliver() == owner.records
    assert outputs == [1, 6, 0, 5, 4, 0]
    return outputs


def check_commit_fault(fault):
    owner = Owner(make_state((2, 1, 3), 2))
    old = owner.s
    owner.start(1)
    execution = SamplingExecution(old, owner.p, 'host')
    token, rho_next = execution.run()
    assert owner.finish(roots=execution.completed)
    assert owner.commit(token, rho_next, fault)
    assert owner.s.k == len(old.x) == len(owner.s.x) - 1, 'KV advanced without work'
    assert owner.s.rho == rho_next, 'sampled RNG not installed'
    assert owner.s.kv == dense_reference(old.x)[0]


def check_private_fault(fault):
    owner = Owner(make_state((2, 1, 3), 2))
    old = owner.s
    owner.start(1, fault)
    assert owner.s is old, 'device preparation modified committed state'
    assert owner.records == (), 'publication before transaction'
    assert owner.deliver() == ()


def cancel_case(mode, fault=None):
    owner = Owner(make_state((2, 1, 3), 2))
    old = owner.s
    owner.start(1)
    execution = SamplingExecution(old, owner.p, mode)
    token, rho_next = execution.run()
    assert_sampling_execution(old, owner.p, execution)
    assert owner.finish(roots=execution.completed)
    owner.cancel()
    owner.commit(token, rho_next, fault)
    assert owner.s is old and owner.records == (), 'candidate installed after cancel'
    assert owner.deliver() == ()


def check_failure_and_commit():
    for mode in ('host', 'device'):
        cancel_case(mode)
        expect_detected(mode + '-cancel-late-commit',
                        lambda mode=mode: cancel_case(mode, 'allow_cancel'))
    for fault in ('early_publish', 'overwrite_committed'):
        expect_detected(fault, lambda fault=fault: check_private_fault(fault))
    for fault in ('advance_k', 'wrong_rng'):
        expect_detected(fault, lambda fault=fault: check_commit_fault(fault))
    owner = Owner(make_state((2, 1, 3), 2))
    old = owner.s
    execution_id = owner.start(1)
    # Result completeness and physical last-use evidence are separate.
    assert not owner.finish(ended=False)
    assert owner.pending
    token, rho_next = sample(owner.p.logits, old.rho)
    assert not owner.commit(token, rho_next)
    owner.fail()
    expect_value_error(lambda: owner.start(1))
    assert not owner.commit(token, rho_next)
    before = set(owner.pending)
    owner.use_done(('wrong-instance', 99))
    owner.use_done(execution_id, complete=False)
    assert owner.pending == before and owner.phase == 'DRAINING'
    owner.use_done(execution_id)
    assert not owner.pending and owner.phase == 'TERMINAL'
    owner.use_done(execution_id)
    assert not owner.pending and not owner.live and not owner.eligible
    expect_value_error(lambda: owner.start(1))
    assert owner.s is old and not owner.records
    missing = Owner(make_state((2, 1, 3), 2))
    snapshot = missing.s
    missing.start(1)
    assert not missing.finish(roots={LOGITS})
    assert not missing.commit(*sample(missing.p.logits, snapshot.rho))
    assert missing.s is snapshot and not missing.records
    return 'cancel, incomplete roots, failure re-entry, late and duplicate last-use events'


def closure(roots, deps):
    found = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node not in deps:
            raise ValueError('unknown graph node')
        if node not in found:
            found.add(node)
            stack.extend(deps[node])
    return found


def check_graph():
    deps = {
        'x': (), 'k0': ('x',), 'h0': ('k0',), 'write0': ('k0',),
        'k1': ('h0',), 'h1': ('k1',), 'write1': ('k1',), 'logits': ('h1',)
    }
    forward = closure({'logits'}, deps)
    assert forward == {'x', 'k0', 'h0', 'k1', 'h1', 'logits'}
    assert closure({'logits', 'write0', 'write1'}, deps) == set(deps)
    assert closure(set(), deps) == set()
    expect_value_error(lambda: closure({'unknown'}, deps))

    def faulty():
        assert {'write0', 'write1'} <= forward, 'independent write roots omitted'

    expect_detected('logits-only-writeback', faulty)


@dataclass(frozen=True)
class Hold:
    key: str
    birth: int
    retire: int
    size: int


def physical_peak(holds):
    sizes, events = {}, {}
    for h in holds:
        if h.size < 0 or h.birth < 0 or h.retire < h.birth:
            raise ValueError('invalid physical hold')
        if h.key in sizes and sizes[h.key] != h.size:
            raise ValueError('inconsistent physical allocation size')
        sizes[h.key] = h.size
        if h.birth == h.retire:
            continue
        events.setdefault(h.birth, []).append((h.key, 1))
        events.setdefault(h.retire, []).append((h.key, -1))
    active, total, peak = {}, 0, 0
    for t in sorted(events):
        for key, delta in sorted(events[t], key=lambda item: item[1]):
            prior = active.get(key, 0)
            after = prior + delta
            assert after >= 0
            if prior == 0 and after > 0:
                total += sizes[key]
            if prior > 0 and after == 0:
                total -= sizes[key]
            active[key] = after
        peak = max(peak, total)
    assert not any(active.values())
    return peak


def physical_reference(holds, aliases=False):
    # Independent pointwise union at endpoints and interval midpoints.
    times = sorted({h.birth for h in holds} | {h.retire for h in holds})
    probes = set(times)
    probes.update(Fraction(a + b, 2) for a, b in zip(times, times[1:]))
    peak = 0
    for t in probes:
        alive = [h for h in holds if h.birth <= t < h.retire]
        if aliases:
            value = sum(h.size for h in alive)
        else:
            value = sum({h.key: h.size for h in alive}.values())
        peak = max(peak, value)
    return peak


def check_costs():
    holds = [Hold('prefix', 0, 10, 100), Hold('prefix', 5, 15, 100),
             Hold('tailA', 2, 8, 40), Hold('tailB', 8, 12, 40),
             Hold('empty', 3, 3, 999)]
    assert physical_peak(holds) == physical_reference(holds) == 140
    assert physical_peak([]) == physical_reference([]) == 0
    expect_value_error(lambda: physical_peak([Hold('p', 0, 1, 1), Hold('p', 0, 1, 2)]))
    expect_value_error(lambda: physical_peak([Hold('p', 2, 1, 1)]))

    def aliases():
        assert physical_reference(holds, aliases=True) == physical_reference(holds)

    expect_detected('alias-capacity-sum', aliases)
    b_kv = 2 * 32 * 8 * 128 * 2
    enumerated = sum(2 for layer in range(32) for head in range(8)
                     for channel in range(128) for kind in ('K', 'V'))
    assert b_kv == enumerated == 131072

    def omit_v():
        assert 32 * 8 * 128 * 2 == enumerated

    expect_detected('omit-V-payload', omit_v)
    read_cases = 0
    for b in range(9):
        for e in range(b, 11):
            formula = (e - b) * (b + e + 1) // 2
            accesses = [(t, j) for t in range(b, e) for j in range(t + 1)]
            assert formula == len(accesses)
            assert len(list(range(b, e))) == e - b
            read_cases += 1
    assert sum(t + 1 for t in range(2, 4)) == 7

    def old_history_only():
        assert 2 == len([(t, j) for t in range(2, 4) for j in range(t + 1)])

    expect_detected('prefill-old-history-only', old_history_only)
    weights = 12 * GIB
    workspace = 3 * GIB // 4
    graph_queue = GIB // 2
    reserve = GIB
    budget = 16 * GIB
    peak4 = weights + 2 * GIB + workspace + graph_queue
    peak3 = weights + 3 * GIB // 2 + workspace + graph_queue
    need4, need3 = peak4 + reserve, peak3 + reserve
    assert Fraction(peak3, GIB) == Fraction(59, 4)
    assert Fraction(need3, GIB) == Fraction(63, 4)
    assert need3 <= budget < need4

    def double_reserve():
        assert peak3 + reserve + reserve == need3

    def omit_reserve():
        assert (peak4 <= budget) == (need4 <= budget)

    expect_detected('double-reserve', double_reserve)
    expect_detected('omit-reserve', omit_reserve)
    shared = Fraction(14 * GIB + GIB // 2, 200 * GIB)
    assert shared == Fraction(29, 400)

    def independent_bandwidth():
        assert Fraction(max(14 * GIB, GIB // 2), 200 * GIB) == shared

    expect_detected('independent-full-bandwidth', independent_bandwidth)
    return {'read_cases': read_cases, 'KV_bytes_per_token': b_kv,
            'physical_peak': 140,
            'faulty_alias_peak': physical_reference(holds, True),
            'three_request_peak_GiB': str(Fraction(peak3, GIB)),
            'three_request_admission_GiB': str(Fraction(need3, GIB)),
            'shared_resource_seconds': str(shared)}


def main():
    cases = check_preparation()
    branch_cases, partial = check_sampling_paths()
    host_outputs = check_multistep('host')
    device_outputs = check_multistep('device')
    assert host_outputs == device_outputs
    protocol = check_failure_and_commit()
    check_graph()
    costs = check_costs()
    print('OBS full-prefix versus private incremental preparation cases:', cases)
    print('OBS host/device sampling control-flow cases:', branch_cases)
    print('OBS n=4 k=2 q=1 no-logits paths:', partial)
    print('OBS host multistep sampled tokens:', host_outputs)
    print('OBS device multistep sampled tokens:', device_outputs)
    print('OBS protocol:', protocol)
    print('OBS cost models:', costs)
    print('OBS detected negative controls:', ', '.join(DETECTED))
    print('PASS K-HW-007 scoped CPU integer, graph, visibility-event and protocol models')
    print('SCOPE no MLX/Metal/GPU, transformer numerics, device visibility, bandwidth or service measurements')


if __name__ == '__main__':
    main()
