#!/usr/bin/env python3
# Standalone deterministic CPU model. No I/O except scoped stdout observations.
from dataclasses import dataclass, field
from types import MappingProxyType

class CheckFailure(AssertionError):
    pass

def require(condition, message):
    if not condition:
        raise CheckFailure(message)

def expect_failure(fn, label):
    try:
        fn()
    except CheckFailure:
        return
    raise CheckFailure('negative control escaped: ' + label)

def observe(message):
    print('OBS: ' + message)

class LazyGraph:
    def __init__(self):
        self.nodes = []

    def input(self, value):
        self.nodes.append(('input', value))
        return len(self.nodes) - 1

    def op(self, kind, a, b):
        self.nodes.append((kind, a, b))
        return len(self.nodes) - 1

    def evaluate(self, roots):
        values = {}
        def visit(index):
            if index in values:
                return values[index]
            node = self.nodes[index]
            if node[0] == 'input':
                value = node[1]
            else:
                left, right = visit(node[1]), visit(node[2])
                value = left + right if node[0] == 'add' else left * right
            values[index] = value
            return value
        for root in roots:
            visit(root)
        return values

def roots_complete(values, required):
    require(set(required).issubset(values), 'required state root absent')

def test_lazy_closure():
    graph = LazyGraph()
    a, b, one = graph.input(2), graph.input(3), graph.input(1)
    logits = graph.op('add', a, b)
    kv = graph.op('mul', a, b)
    rng = graph.op('add', kv, one)
    partial = graph.evaluate((logits,))
    require(partial[logits] == 5, 'logits arithmetic')
    require(kv not in partial and rng not in partial, 'independent branch executed')
    expect_failure(lambda: roots_complete(partial, (logits, kv, rng)), 'partial roots')
    complete = graph.evaluate((logits, kv, rng))
    roots_complete(complete, (logits, kv, rng))
    require((complete[logits], complete[kv], complete[rng]) == (5, 6, 7), 'root results')
    observe('lazy closure: independent KV/RNG absent for logits-only roots; complete roots accepted')

@dataclass(frozen=True)
class Box:
    value: int

class Symbol:
    def __init__(self, builder, index):
        self.builder, self.index = builder, index

    def __add__(self, other):
        return self.builder.binary('add', self, other)

    def __rmul__(self, other):
        return self.builder.binary('mul', other, self)

class Builder:
    def __init__(self):
        self.nodes = []

    def emit(self, *instruction):
        self.nodes.append(tuple(instruction))
        return Symbol(self, len(self.nodes) - 1)

    def lift(self, value):
        if isinstance(value, Symbol):
            require(value.builder is self, 'foreign symbol')
            return value
        scalar = value.value if isinstance(value, Box) else value
        return self.emit('const', scalar)

    def binary(self, op, left, right):
        a, b = self.lift(left), self.lift(right)
        return self.emit(op, a.index, b.index)

@dataclass(frozen=True)
class Program:
    nodes: tuple
    output: int

def execute_ir(program, frame):
    # Generic interpreter: no capture-kind branch and no reference/oracle call.
    registers = []
    for node in program.nodes:
        op = node[0]
        if op == 'input':
            value = frame[node[1]]
        elif op == 'const':
            value = node[1]
        elif op == 'live':
            # Deliberately available only to encode a faulty trace representation.
            value = node[1][0].value
        elif op == 'add':
            value = registers[node[1]] + registers[node[2]]
        elif op == 'mul':
            value = registers[node[1]] * registers[node[2]]
        else:
            raise CheckFailure('unknown IR instruction')
        registers.append(value)
    return registers[program.output]

class CompiledProbe:
    def __init__(self, explicit, external, fault=None):
        self.explicit, self.external, self.fault = explicit, external, fault
        self.trace_count = 0
        self.program = self.trace()

    def trace(self):
        self.trace_count += 1
        builder = Builder()
        x = builder.emit('input', 0)
        if self.explicit and self.fault != 'freeze_explicit':
            state = [builder.emit('input', 1)]
        elif not self.explicit and self.fault == 'reread_implicit':
            state = [builder.emit('live', self.external)]
        else:
            state = [self.external[0]]
        output = 2 * x + state[0]
        return Program(tuple(builder.nodes), output.index)

    def __call__(self, x, current):
        if self.fault == 'retrace':
            self.program = self.trace()
        frame = (current, x) if self.fault == 'swap_inputs' else (x, current)
        return execute_ir(self.program, frame)

def capture_judge(factory, explicit):
    external = [Box(1)]
    trace_snapshot = external[0].value  # Oracle storage is outside trace and executor.
    compiled = factory(explicit, external)
    saved_program = compiled.program
    cases = ((1, 1, 1), (4, 5, 9), (-3, 0, -7), (0, 12, 20))
    for x, current, external_now in cases:
        external[0] = Box(external_now)
        expected = 2 * x + (current if explicit else trace_snapshot)
        actual = compiled(x, current)
        require(actual == expected, 'trace snapshot/input-slot oracle mismatch')
        require(compiled.trace_count == 1, 'execution retraced the source body')
        require(compiled.program is saved_program, 'saved representation replaced')
    return compiled

def test_capture():
    for explicit in (False, True):
        compiled = capture_judge(lambda e, s: CompiledProbe(e, s), explicit)
        tags = tuple(node[0] for node in compiled.program.nodes)
        require('live' not in tags, 'valid trace retained external read')
        slots = [node[1] for node in compiled.program.nodes if node[0] == 'input']
        require(slots == ([0, 1] if explicit else [0]), 'trace slot layout')
    faults = ((True, 'freeze_explicit'), (False, 'reread_implicit'),
              (True, 'swap_inputs'), (False, 'retrace'))
    for explicit, fault in faults:
        expect_failure(lambda e=explicit, f=fault: capture_judge(
            lambda kind, state: CompiledProbe(kind, state, f), e), fault)
    observe('capture: trace-once IR executed against external snapshot and per-call inputs; four faulty variants rejected')

@dataclass(frozen=True)
class Allocation:
    oid: str
    size_mib: int

class Ledger:
    def __init__(self):
        self.refs = []

    def physical_mib(self):
        unique = {}
        for allocation in self.refs:
            if allocation.oid in unique:
                require(unique[allocation.oid] == allocation.size_mib, 'conflicting allocation identity')
            unique[allocation.oid] = allocation.size_mib
        return sum(unique.values())

class FaultyLedger(Ledger):
    def physical_mib(self):
        return sum(allocation.size_mib for allocation in self.refs)

def capacity_judge(ledger_type):
    ledger = ledger_type()
    weights = Allocation('weights', 14336)
    ledger.refs = [weights, Allocation('kv', 2048), Allocation('work', 1024),
                   Allocation('graph', 512), weights]
    require(ledger.physical_mib() == 17920, 'shared physical allocation counted incorrectly')

def test_capacity_and_cost():
    capacity_judge(Ledger)
    expect_failure(lambda: capacity_judge(FaultyLedger), 'physical identity ledger')
    per_token = 2 * 32 * 8 * 128 * 2
    require(per_token == 131072, 'KV byte derivation')
    require(4 * 4096 * per_token == 2 * 1024 ** 3, 'active KV derivation')
    require(4 * per_token == 512 * 1024, 'private decode increment')
    require(400 * 2 == 200 + 400 * 1.5, 'compile break-even')
    require(200 + 1000 * 1.5 < 1000 * 2, 'compile amortization')
    require(200 + 1000 * (1.5 + .01 * 200) > 1000 * 2, 'recompilation cost')
    require((2 - 1.5) / 200 == .0025, 'recompilation threshold')
    first, second = (2, 3, 4), (5, 5, 3)
    require(first[0] * first[1] == 6, 'initial shape factor')
    require((second[0] * second[1] * second[2]) % 6 != 0, 'static reshape counterexample')
    flattened = (second[0] * second[1], second[2])
    require(flattened == (25, 3), 'flatten axes 0 and 1')
    observe('capacity/cost: identity ledger=17.5 GiB; 128 KiB/token; four private increments=512 KiB; arithmetic is synthetic')

LAYERS, VOCAB = 2, 5

@dataclass(frozen=True)
class KVCell:
    value: int

class KVStore:
    def __init__(self, cells=None):
        # Copy dictionary references, never the immutable payload objects.
        self.cells = MappingProxyType(dict(cells or {}))

    def complete(self, k):
        return all((layer, pos) in self.cells for pos in range(k) for layer in range(LAYERS))

class DeltaView:
    def __init__(self, base, b, e):
        self.base, self.b, self.e = base, b, e
        self.delta = {}

    def cell(self, layer, pos):
        require(0 <= layer < LAYERS and 0 <= pos < self.e, 'read outside view')
        return self.base.cells[(layer, pos)] if pos < self.b else self.delta[(layer, pos)]

    def write(self, layer, pos, value):
        require(0 <= layer < LAYERS and self.b <= pos < self.e, 'write into readonly prefix/outside private range')
        self.delta[(layer, pos)] = KVCell(value)

def block(layer, hidden, past, pos):
    hidden = (3 * hidden + 17 * layer + 5 * pos + sum(past)) % 101
    return hidden, (7 * hidden + layer + pos) % 97

def row(hidden, n):
    return tuple((11 * hidden + 7 * index + 3 * n) % 13 for index in range(VOCAB))

def serial_reference(tokens):
    # Reconstruct a complete prefix from scratch, with no request/delta state.
    table = [[] for _ in range(LAYERS)]
    last = None
    for pos, token in enumerate(tokens):
        hidden = token + 1
        for layer in range(LAYERS):
            total = sum(table[layer])
            hidden = (hidden * 3 + layer * 17 + pos * 5 + total) % 101
            table[layer].append((hidden * 7 + layer + pos) % 97)
        last = hidden
    cells = {(layer, pos): KVCell(table[layer][pos])
             for layer in range(LAYERS) for pos in range(len(tokens))}
    logits = None if not tokens else tuple(
        (last * 11 + index * 7 + len(tokens) * 3) % 13 for index in range(VOCAB))
    return KVStore(cells), logits

def sample(logits, rng):
    require(logits is not None, 'sample without final row')
    best, score = 0, -1
    for index, logit in enumerate(logits):
        candidate = (5 * logit + 3 * rng + 7 * index) % 29
        if candidate > score:
            best, score = index, candidate
    return best, (17 * rng + 3) % 101

def reference_sample(logits, rng):
    scores = [(value * 5 + rng * 3 + index * 7) % 29
              for index, value in enumerate(logits)]
    best = max(range(len(scores)), key=lambda index: (scores[index], -index))
    return best, (rng * 17 + 3) % 101

@dataclass(frozen=True)
class Snapshot:
    tokens: tuple
    kv: KVStore
    k: int
    ell: int
    logits: object
    rng: int
    version: int

@dataclass
class Prepared:
    base: KVStore
    delta: dict
    b: int
    e: int
    k_after: int
    logits: object
    scratch: object = None

@dataclass
class Execution:
    eid: int
    snap: Snapshot
    prepared: Prepared

@dataclass
class Req:
    tokens: list
    kv: KVStore = field(default_factory=KVStore)
    k: int = 0
    ell: int = -1
    logits: object = None
    rng: int = 0
    version: int = 0
    live: bool = True
    canceled: bool = False
    failed: bool = False
    commit_eligible: bool = True
    phase: str = 'READY'
    active: object = None
    next_eid: int = 0
    pending: set = field(default_factory=set)
    retained: dict = field(default_factory=dict)
    outbox: list = field(default_factory=list)
    published: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.tokens)

def snapshot(req):
    return Snapshot(tuple(req.tokens), req.kv, req.k, req.ell,
                    req.logits, req.rng, req.version)

def compute_view(tokens, view):
    last = None
    for pos in range(view.b, view.e):
        hidden = tokens[pos] + 1
        for layer in range(LAYERS):
            past = [view.cell(layer, previous).value for previous in range(pos)]
            hidden, value = block(layer, hidden, past, pos)
            view.write(layer, pos, value)
        last = hidden
    return last

def prepare(snap, q):
    n = len(snap.tokens)
    if snap.k < n:
        b, e = snap.k, min(n, snap.k + q)
        view = DeltaView(snap.kv, b, e)
        last = compute_view(snap.tokens, view)
        logits = row(last, n) if e == n else None
        return Prepared(snap.kv, view.delta, b, e, e, logits)
    if snap.ell == n - 1 and snap.logits is not None:
        return Prepared(snap.kv, {}, n, n, n, snap.logits)
    view = DeltaView(snap.kv, n - 1, n)
    last = compute_view(snap.tokens, view)
    return Prepared(snap.kv, {}, n, n, n, row(last, n), view)

def install(base, prepared, copy_prefix=False):
    require(prepared.base is base, 'base identity changed')
    expected = {(layer, pos) for pos in range(prepared.b, prepared.e)
                for layer in range(LAYERS)}
    require(set(prepared.delta) == expected, 'incomplete layer/range certificate')
    if not prepared.delta:
        return base
    cells = {key: KVCell(cell.value) if copy_prefix else cell
             for key, cell in base.cells.items()}
    cells.update(prepared.delta)
    return KVStore(cells)

def finish_terminal(req):
    if not req.commit_eligible and not req.pending:
        req.phase = 'COMPLETED'
        req.active = None
        req.retained.clear()

def on_failure(req, execution, accesses_ended):
    req.failed, req.live, req.commit_eligible = True, False, False
    if accesses_ended:
        req.pending.discard((execution.eid, 'eval'))
    req.phase = 'FAILED_DRAINING' if req.pending else 'COMPLETED'
    finish_terminal(req)

def use_done(req, eid, uid, ended):
    identity = (eid, uid)
    if eid not in req.retained or identity not in req.pending:
        return False
    if not ended:
        req.phase = 'QUARANTINED'
        return False
    req.pending.remove(identity)
    finish_terminal(req)
    return True

def cancel(req):
    req.canceled, req.live, req.commit_eligible = True, False, False
    req.phase = 'CANCELED_DRAINING' if req.pending else 'CANCELED'

def commit(req, execution, token, rng_next, faulty_guard=False,
           faulty_k=False, faulty_rng=False, after_commit=None):
    if req.version != execution.snap.version or req.active != execution.eid:
        return 'STALE'
    if not faulty_guard:
        if req.canceled:
            return 'CANCELED'
        if (not req.live or not req.commit_eligible or req.failed or req.pending
                or req.phase != 'READY_TO_COMMIT'):
            return 'BLOCKED'
    prepared = execution.prepared
    require(req.kv is execution.snap.kv, 'snapshot storage replaced without version')
    new_kv = install(req.kv, prepared)
    # One serial-owner transaction; no hook can interleave these assignments.
    req.kv, req.k = new_kv, prepared.k_after
    req.ell, req.logits = -1, None
    if token is not None:
        req.tokens.append(token)
        if not faulty_rng:
            req.rng = rng_next
        req.outbox.append(token)
    if faulty_k:
        req.k = req.n
    req.version += 1
    req.phase = 'COMMITTED'
    req.active = None
    req.retained.pop(execution.eid, None)
    # This hook is explicitly AFTER transaction and BEFORE delivery.
    if after_commit is not None:
        after_commit()
    if token is not None:
        req.published.append(token)
    return 'COMMITTED' if token is not None else 'PREFILLED'

def serve_step(req, q=2, eval_failure=False, fail_pending=False,
               after_sample=None, after_commit=None, faulty_entry=False,
               faulty_commit=False, faulty_k=False, faulty_rng=False):
    if req.canceled and not faulty_entry:
        return 'CANCELED'
    if not faulty_entry:
        if (not req.live or not req.commit_eligible or req.failed or req.pending
                or req.active is not None or req.phase not in ('READY', 'COMMITTED')):
            return 'BLOCKED'
    if req.n == 0:
        return 'REJECT_NO_INPUT'
    if q < 1:
        return 'REJECT_Q'
    if not 0 <= req.k <= req.n or not req.kv.complete(req.k):
        return 'REJECT_STATE'
    if req.ell != -1 and req.ell != req.n - 1:
        return 'REJECT_STATE'
    if (req.logits is None) != (req.ell == -1):
        return 'REJECT_STATE'
    if req.logits is not None and req.k != req.n:
        return 'REJECT_STATE'
    snap = snapshot(req)
    prepared = prepare(snap, q)
    req.next_eid += 1
    execution = Execution(req.next_eid, snap, prepared)
    req.active = execution.eid
    req.retained[execution.eid] = execution
    req.pending.add((execution.eid, 'eval'))
    req.phase = 'EVALUATING'
    # Arithmetic above is an eager toy of the result graph. This event boundary
    # models its completion certificate, not MLX execution or device visibility.
    if eval_failure:
        on_failure(req, execution, accesses_ended=not fail_pending)
        return 'FAILED'
    req.pending.remove((execution.eid, 'eval'))
    req.phase = 'READY_TO_COMMIT'
    token, rng_next = None, snap.rng
    if prepared.logits is not None:
        token, rng_next = sample(prepared.logits, snap.rng)
        if after_sample is not None:
            after_sample()
    outcome = commit(req, execution, token, rng_next, faulty_guard=faulty_commit,
                     faulty_k=faulty_k, faulty_rng=faulty_rng, after_commit=after_commit)
    if outcome not in ('COMMITTED', 'PREFILLED'):
        finish_terminal(req)
    return outcome

def with_prefix(tokens, k, rng=0, cached=False):
    full, logits = serial_reference(tokens)
    prefix = KVStore({key: cell for key, cell in full.cells.items() if key[1] < k})
    req = Req(list(tokens), kv=prefix, k=k, rng=rng)
    if cached:
        require(k == len(tokens), 'cached row without complete prefix')
        req.ell, req.logits = len(tokens) - 1, logits
    return req

def state_invariant(req):
    full, logits = serial_reference(req.tokens)
    require(0 <= req.k <= req.n, 'k outside confirmed length')
    expected_keys = {(layer, pos) for pos in range(req.k) for layer in range(LAYERS)}
    require(set(req.kv.cells) == expected_keys, 'missing KV layer or premature boundary')
    for key in expected_keys:
        require(req.kv.cells[key].value == full.cells[key].value, 'incremental KV differs from full prefix')
    if req.ell != -1:
        require(req.k == req.n and req.ell == req.n - 1, 'invalid final row identity')
        require(req.logits == logits, 'cached row differs from full prefix')
    else:
        require(req.logits is None, 'unidentified cached logits')

def business(req):
    # Include old KV object identity as well as values and all business fields.
    return (tuple(req.tokens), req.kv, tuple(sorted(
        (key, cell.value) for key, cell in req.kv.cells.items())), req.k,
        req.ell, req.logits, req.rng, req.version, tuple(req.outbox), tuple(req.published))

def terminal_judge(req, frozen):
    require(business(req) == frozen, 'failed request changed committed state/version/RNG/output')
    require(req.failed and not req.live and not req.commit_eligible, 'failure restored eligibility')
    if req.pending:
        require(req.phase in ('FAILED_DRAINING', 'QUARANTINED'), 'pending use hidden by success/completion')
    if req.phase == 'COMPLETED':
        require(not req.pending, 'completed before last use')
    state_invariant(req)

def test_serving_reference():
    for q in (1, 2, 9):
        prompt, rng = [1, 2, 4], 0
        req = Req(list(prompt))
        expected_tokens, expected_rng = list(prompt), rng
        expected_outputs = []
        for _ in range(4):
            _, logits = serial_reference(expected_tokens)
            token, expected_rng = reference_sample(logits, expected_rng)
            expected_tokens.append(token)
            expected_outputs.append(token)
        iterations = 0
        while len(req.published) < 4:
            old_n, old_k, old_rng, old_outputs = req.n, req.k, req.rng, len(req.published)
            outcome = serve_step(req, q)
            state_invariant(req)
            if outcome == 'PREFILLED':
                require(req.k > old_k and req.n == old_n, 'partial prefill progress')
                require(req.rng == old_rng and len(req.published) == old_outputs, 'prefill sampled')
            else:
                require(outcome == 'COMMITTED', 'unexpected normal outcome')
                require(req.n == old_n + 1 and req.k == old_n, 'new token KV advanced early')
            iterations += 1
            require(iterations < 30, 'normal serving stalled')
        require(req.tokens == expected_tokens and req.published == expected_outputs, 'serial generation reference')
        require(req.rng == expected_rng, 'serial RNG reference')
    require(serve_step(Req([])) == 'REJECT_NO_INPUT', 'empty input boundary')
    require(serve_step(Req([1]), q=0) == 'REJECT_Q', 'zero chunk boundary')
    require(serve_step(Req([1], k=2)) == 'REJECT_STATE', 'invalid KV boundary')
    observe('serving: chunked prefill and four-token generation match independent full-prefix traversal and sampling')

def storage_judge(base, prepared, merged):
    require(prepared.base is base, 'prefix snapshot was copied')
    for key, cell in base.cells.items():
        require(merged.cells[key] is cell, 'prefix payload physically copied or replaced')
    for key, cell in prepared.delta.items():
        require(merged.cells[key] is cell, 'private payload copied at install')
    live_ids = {id(cell) for store in (base, merged) for cell in store.cells.values()}
    require(len(live_ids) == len(base.cells) + len(prepared.delta), 'physical payload union mismatch')

def test_storage_and_repair():
    req = with_prefix([1, 2, 3, 4], 2)
    base, before = req.kv, business(req)
    prepared = prepare(snapshot(req), 2)
    require(business(req) == before, 'preparation changed public state')
    require(set(prepared.delta) == {(l, p) for p in (2, 3) for l in range(LAYERS)}, 'private delta range')
    view = DeltaView(base, 2, 4)
    require(view.cell(0, 0) is base.cells[(0, 0)], 'prefix read did not share object')
    expect_failure(lambda: view.write(0, 0, 999), 'readonly prefix write')
    merged = install(base, prepared)
    storage_judge(base, prepared, merged)
    expect_failure(lambda: storage_judge(base, prepared, install(base, prepared, True)), 'physical prefix clone')
    old_cells = dict(base.cells)
    require(serve_step(req) == 'COMMITTED', 'incremental commit')
    for key, cell in old_cells.items():
        require(req.kv.cells[key] is cell, 'actual service commit replaced old payload')
    for cached in (False, True):
        req = with_prefix([1, 3, 2, 4], 4, rng=5, cached=cached)
        old_kv = req.kv
        _, logits = serial_reference(req.tokens)
        expected, rng_next = reference_sample(logits, req.rng)
        require(serve_step(req) == 'COMMITTED', 'full KV row handling')
        require(req.kv is old_kv and req.k == 4 and req.n == 5, 'repair/sample-only changed shared KV')
        require(req.published == [expected] and req.rng == rng_next, 'repair/sample-only reference')
        state_invariant(req)
    observe('storage: immutable prefix payload shared, writes confined to delta, install transfers objects; cloned-prefix control rejected; last-row repair isolated')

def cancel_before_judge(fault=False):
    req = with_prefix([1, 2, 3], 3, rng=7, cached=True)
    frozen = business(req)
    outcome = serve_step(req, after_sample=lambda: cancel(req), faulty_commit=fault)
    require(business(req) == frozen, 'pre-transaction cancellation changed business state')
    require(outcome == 'CANCELED', 'pre-transaction cancellation outcome')

def test_cancel():
    cancel_before_judge()
    expect_failure(lambda: cancel_before_judge(True), 'omitted cancellation/eligibility commit guard')
    req = with_prefix([1, 2, 3], 3, rng=7, cached=True)
    old_n = req.n
    def between_transaction_and_delivery():
        require(req.n == old_n + 1 and len(req.outbox) == 1, 'transaction not yet installed')
        require(req.published == [], 'delivery occurred before hook')
        cancel(req)
    require(serve_step(req, after_commit=between_transaction_and_delivery) == 'COMMITTED', 'post-transaction cancel rolled back')
    require(len(req.published) == 1 and req.published == req.outbox, 'committed delivery record lost')
    state_invariant(req)
    observe('cancel: after-sample cancellation blocks commit; transaction-before-delivery cancellation preserves committed record')

def failed_request():
    req = with_prefix([1, 2, 3, 4], 2, rng=8)
    frozen = business(req)
    require(serve_step(req, eval_failure=True, fail_pending=True) == 'FAILED', 'failure injection')
    execution = req.retained[req.active]
    terminal_judge(req, frozen)
    require(req.pending == {(execution.eid, 'eval')}, 'lost failed execution responsibility')
    return req, frozen, execution

def failure_reentry_judge(fault=False):
    req, frozen, execution = failed_request()
    outcome = serve_step(req, faulty_entry=fault, faulty_commit=fault)
    terminal_judge(req, frozen)
    require(outcome == 'BLOCKED', 'failed request admitted another step')
    require(not use_done(req, execution.eid + 1, 'eval', True), 'foreign execution accepted')
    require(not use_done(req, execution.eid, 'unknown', True), 'unknown use accepted')
    require(not use_done(req, execution.eid, 'eval', False), 'incomplete certificate cleared use')
    require(req.phase == 'QUARANTINED', 'incomplete certificate did not isolate')
    terminal_judge(req, frozen)
    require(serve_step(req) == 'BLOCKED', 'quarantined request admitted')
    require(use_done(req, execution.eid, 'eval', True), 'real last-use event ignored')
    require(req.phase == 'COMPLETED' and not req.pending and not req.retained, 'drain cleanup')
    require(not use_done(req, execution.eid, 'eval', True), 'duplicate completion acted twice')
    require(serve_step(req) == 'BLOCKED', 'drain restored eligibility')
    token, next_rng = sample(execution.prepared.logits, execution.snap.rng)
    require(commit(req, execution, token, next_rng) == 'STALE', 'post-drain stale candidate installed')
    terminal_judge(req, frozen)

def late_candidate_judge(fault=False):
    req, frozen, execution = failed_request()
    token, next_rng = sample(execution.prepared.logits, execution.snap.rng)
    outcome = commit(req, execution, token, next_rng, faulty_guard=fault)
    terminal_judge(req, frozen)
    require(outcome == 'BLOCKED', 'late result committed after failure')

def test_failure():
    failure_reentry_judge()
    late_candidate_judge()
    expect_failure(lambda: failure_reentry_judge(True), 'omitted entry and commit eligibility')
    expect_failure(lambda: late_candidate_judge(True), 'omitted late-commit eligibility')
    # Entry omission alone still cannot bypass the independent COMMIT guard.
    req, frozen, _ = failed_request()
    require(serve_step(req, faulty_entry=True) == 'BLOCKED', 'commit defense did not reject bad admission')
    require(business(req) == frozen, 'bad admission changed business state despite commit guard')
    # Exception with complete physical-end evidence remains terminal too.
    req = with_prefix([1, 2, 3, 4], 2)
    frozen = business(req)
    require(serve_step(req, eval_failure=True) == 'FAILED', 'ended failure injection')
    terminal_judge(req, frozen)
    require(req.phase == 'COMPLETED' and serve_step(req) == 'BLOCKED', 'ended failure resumed')
    observe('failure: reentry and late candidate rejected before/after drain; foreign, incomplete and duplicate completions handled; omitted guards detected by same terminal invariant')

def test_kv_rng_negatives():
    req = Req([1, 2, 3])
    serve_step(req, q=2, faulty_k=True)
    expect_failure(lambda: state_invariant(req), 'premature all-layer KV boundary')
    def rng_judge(fault):
        req = with_prefix([1, 2], 2, rng=7, cached=True)
        token, expected_rng = reference_sample(req.logits, req.rng)
        require(serve_step(req, faulty_rng=fault) == 'COMMITTED', 'RNG test commit')
        require(req.published == [token] and req.rng == expected_rng, 'token/RNG pair not atomically installed')
    rng_judge(False)
    expect_failure(lambda: rng_judge(True), 'wrong RNG installation')
    observe('negative controls: premature KV and wrong RNG rejected by prefix/reference invariants')

def test_loss_and_stream_contracts():
    graph = LazyGraph()
    old = graph.input(100)
    two, ten = graph.input(2), graph.input(10)
    loss = graph.op('mul', old, two)
    new = graph.op('add', old, ten)
    partial = graph.evaluate((loss,))
    require(partial[loss] == 200 and new not in partial, 'loss-only updated independent parameter')
    current = graph.evaluate((new,))
    require(current[new] == 110 and graph.evaluate((old,))[old] == 100, 'old/current object distinction')
    expect_failure(lambda: require(current[new] == partial[old], 'stale parameter publication'), 'stale parameter view')
    def stream_judge(edges):
        ready = {'cpu_produce': False, 'gpu_consume': False}
        order = []
        while len(order) < 2:
            candidates = sorted(name for name in ready if not ready[name]
                                and all(ready[left] for left, right in edges if right == name))
            require(candidates, 'dependency cycle')
            name = candidates[0]
            ready[name] = True
            order.append(name)
        return order
    # Choose names so a missing edge visibly permits the wrong order.
    def named_order(edges):
        ops = ['a_gpu', 'z_cpu']
        order = []
        while len(order) < 2:
            candidates = [op for op in ops if op not in order
                          and all(left in order for left, right in edges if right == op)]
            require(candidates, 'cycle')
            order.append(candidates[0])
        return order
    def dependency_judge(edges):
        order = named_order(edges)
        require(order.index('z_cpu') < order.index('a_gpu'), 'consumer before producer')
    dependency_judge([('z_cpu', 'a_gpu')])
    expect_failure(lambda: dependency_judge([]), 'missing tracked stream edge')
    observe('loss/streams: current parameter root evaluates update, old object remains old; dependency model rejects missing producer edge')

def main():
    test_lazy_closure()
    test_capture()
    test_capacity_and_cost()
    test_serving_reference()
    test_storage_and_repair()
    test_cancel()
    test_failure()
    test_kv_rng_negatives()
    test_loss_and_stream_contracts()
    print('SCOPE: deterministic standalone CPU protocol, IR and immutable-payload model; arithmetic is synthetic.')
    print('UNMEASURED: real MLX/compiler, GPU execution/numerics/visibility, memory pressure, device timing, network delivery and Agent outcomes.')
    print('ALL MODEL CHECKS PASSED')

if __name__ == '__main__':
    main()
