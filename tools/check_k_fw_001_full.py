#!/usr/bin/env python3
'''K-FW-001: deterministic CPU abstract protocol check, Python stdlib only.
No PyTorch, device execution, real threads, subprocesses, files or network.
The executor advances explicit queues; events are pending until executed.
Independent oracles inspect dependencies, initialized reads and storage pins.
An auxiliary producer writes y before a separate numerical gather on f.
'''
import sys


class Violation(Exception):
    pass


def require(condition, message):
    if not condition:
        raise Violation(message)


class Context:
    def __init__(self, grad=True, depth=0):
        self.grad = grad
        self.depth = depth


def push(ctx, kind):
    require(kind in ('record', 'no_grad', 'inference'), 'invalid mode')
    require(not (kind == 'record' and ctx.depth), 'outer inference forbids record')
    saved = (ctx.grad, ctx.depth)
    if kind == 'inference':
        ctx.depth += 1
        ctx.grad = False
    else:
        ctx.grad = kind == 'record'
    return saved


def pop(ctx, saved):
    ctx.grad, ctx.depth = saved


class Buffer:
    def __init__(self, engine, number, data=None):
        self.engine = engine
        self.number = number
        self.data = None if data is None else tuple(data)
        self.pins = 1
        self.freed = False
        self.epoch = 0
        self.writer = None


class Event:
    def __init__(self, engine, command):
        self.engine = engine
        self.command = command

    @property
    def completed(self):
        return self.engine.ops[self.command]['done']


class Engine:
    def __init__(self):
        # All streams, including the auxiliary producer, are registered
        # before the first allocation or submission in the transaction.
        self.queues = {s: [] for s in ('f', 'c', 'k', 'a')}
        self.tail = {}
        self.ops = []
        self.buffers = []
        self.trace = []
        self.accepting = True
        self.quarantined = False

    def allocate(self, data=None):
        require(self.accepting, 'domain closed to allocations')
        b = Buffer(self, len(self.buffers), data)
        self.buffers.append(b)
        return b

    def submit(self, stream, label, reads=(), writes=(), action=None, deps=()):
        require(self.accepting and not self.quarantined, 'domain closed to submissions')
        require(stream in self.queues, 'unregistered stream')
        for b in tuple(reads) + tuple(writes):
            require(b.pins > 0 and not b.freed, 'submission without prior protection')
        parents = set(deps)
        if stream in self.tail:
            parents.add(self.tail[stream])
        number = len(self.ops)
        self.ops.append({'stream': stream, 'label': label, 'deps': parents,
                         'reads': tuple(reads), 'writes': tuple(writes),
                         'action': action, 'done': False})
        self.queues[stream].append(number)
        self.tail[stream] = number
        for b in writes:
            b.epoch += 1
            b.writer = number
        return number

    def fence(self, stream, label, fail=False):
        if fail:
            raise Violation('injected event registration failure: ' + label)
        return Event(self, self.submit(stream, label))

    def step(self, stream):
        if not self.queues[stream]:
            return False
        number = self.queues[stream][0]
        op = self.ops[number]
        if not all(self.ops[p]['done'] for p in op['deps']):
            return False
        for b in op['reads'] + op['writes']:
            require(b.pins > 0 and not b.freed, 'device accessed unprotected storage')
        for b in op['reads']:
            require(b.data is not None, 'device read before producer initialized storage')
        if op['action'] is not None:
            op['action']()
        op['done'] = True
        self.queues[stream].pop(0)
        self.trace.append(op['label'])
        return True

    def advance(self):
        progressed = False
        # Try consumers before producers. In particular, f is tried before a,
        # so missing selection dependencies cannot hide behind this schedule.
        for stream in ('c', 'k', 'f', 'a'):
            progressed = self.step(stream) or progressed
        require(progressed, 'pending work has no legal successor')

    def wait(self, event):
        require(event.engine is self, 'foreign event')
        while not event.completed:
            self.advance()

    def drain(self):
        while any(self.queues.values()):
            self.advance()

    def covers(self, event, writer):
        if writer is None or event.engine is not self:
            return False
        stack = [event.command]
        seen = set()
        while stack:
            number = stack.pop()
            if number == writer:
                return True
            if number not in seen:
                seen.add(number)
                stack.extend(self.ops[number]['deps'])
        return False

    def pending_accesses(self, b):
        return [op['label'] for op in self.ops
                if not op['done'] and b in op['reads'] + op['writes']]


class Tensor:
    def __init__(self, b, inference=False, graph=False, recorded=False,
                 origin='ordinary', source=None):
        self.b = b
        self.inference = inference
        self.graph = graph
        self.recorded = recorded
        self.origin = origin
        self.source = source
        self.ready = None


def bind(t, event):
    require(t.b.engine.covers(event, t.b.writer), 'event misses producing write')
    t.ready = (t.b.number, t.b.epoch, event)


def consumer_oracle(t, scope, ctx):
    require(scope in ('none', 'postprocess_only', 'through_model'), 'bad scope')
    b = t.b
    require(b.pins > 0 and not b.freed, 'consumer storage invalid')
    require(t.ready is not None, 'missing ready credential')
    number, epoch, event = t.ready
    require(number == b.number and epoch == b.epoch, 'wrong storage or stale version')
    require(event.engine is b.engine, 'wrong execution domain')
    require(b.engine.covers(event, b.writer), 'credential does not cover last write')
    require(event.completed, 'producer event is pending')
    require(b.data is not None, 'completed output is uninitialized')
    if t.source is not None:
        require(b is not t.source, 'copy aliases source')
    if scope != 'none':
        require(ctx.depth == 0 and not t.inference, 'illegal autograd tensor/context')
    if scope == 'through_model':
        require(t.graph and t.recorded, 'requested model graph was omitted')
    return True


def lifetime_oracle(engine):
    # This oracle does not consult request terminal state or ready credentials.
    for op in engine.ops:
        if not op['done']:
            for b in op['reads'] + op['writes']:
                require(b.pins > 0 and not b.freed,
                        'physical pending access outlived protection: ' + op['label'])
    return True


QUARANTINE = []


class Transaction:
    def __init__(self, ctx, scope, fault=None):
        self.engine = Engine()
        self.ctx = ctx
        self.scope = scope
        self.fault = fault
        self.state = 'Prepared'
        self.cleanup = 'Owned'
        self.visible = False
        self.error = None
        self.y = None
        self.raw = None
        self.copy = None
        self.z = None
        self.result = None
        # Represents a persistent supervisor owner, preceding all submissions.
        self.owner = 'supervisor'

    def tensor(self, data=None):
        return Tensor(self.engine.allocate(data), inference=self.ctx.depth > 0)

    def fail(self, error):
        require(not self.visible, 'cannot revoke an already visible result')
        self.state = 'Failed'
        self.cleanup = 'DrainPending'
        self.error = str(error)
        self.engine.accepting = False

    def release(self):
        for b in self.engine.buffers:
            require(not self.engine.pending_accesses(b), 'release before physical completion')
        for b in self.engine.buffers:
            b.pins = 0
            b.freed = True
        self.owner = None

    def cleanup_failure(self, proof_available=True):
        require(self.state == 'Failed', 'failure cleanup on nonfailed request')
        try:
            require(proof_available, 'injected inability to prove drain')
            self.engine.drain()
            lifetime_oracle(self.engine)
            self.release()
            self.cleanup = 'Reclaimed'
        except Violation:
            self.engine.quarantined = True
            self.engine.accepting = False
            self.cleanup = 'Quarantined'
            self.owner = 'quarantine supervisor'
            QUARANTINE.append(self)


def materialize(tx, raw, faulty_inside=False, omit_wait=False):
    ctx, engine = tx.ctx, tx.engine
    if not faulty_inside:
        require(ctx.depth == 0, 'outer inference prevents ordinary materialization')
    require(raw.inference, 'materializer requires inference source')
    saved = push(ctx, 'no_grad')
    try:
        out = tx.tensor()
        tx.copy = out
        out.origin = 'materialized_after_inference'
        out.source = raw.b
        require(out.b is not raw.b, 'allocator returned source alias')
        if not omit_wait:
            engine.submit('c', 'wait-forward', deps=(raw.ready[2].command,))
        engine.submit('c', 'copy', reads=(raw.b,), writes=(out.b,),
                      action=lambda: setattr(out.b, 'data', tuple(raw.b.data)))
        event = engine.fence('c', 'copy-ready', fail=tx.fault == 'copy_event')
        bind(out, event)
        return out
    finally:
        pop(ctx, saved)


def begin(data, kind='inference', scope='postprocess_only', fault=None,
          ctx=None, omit_copy_wait=False, auxiliary=False, positions=None,
          omit_select_wait=False):
    ctx = Context() if ctx is None else ctx
    tx = Transaction(ctx, scope, fault)
    try:
        require(scope in ('none', 'postprocess_only', 'through_model'), 'bad scope')
        if scope == 'through_model':
            require(kind == 'record' and ctx.depth == 0, 'model gradients need record forward')
        if scope == 'postprocess_only':
            require(kind in ('inference', 'no_grad'), 'postprocess forward contract')
        tx.state = 'Running'
        inp = tx.tensor(data)
        selected = (tuple(range(len(inp.b.data))) if positions is None
                    else tuple(positions))
        require(all(type(p) is int and 0 <= p < len(inp.b.data) for p in selected),
                'invalid output position')
        saved = push(ctx, kind)
        try:
            y = tx.tensor()
            tx.y = y
            raw = tx.tensor()
            tx.raw = raw
            raw.origin = 'inference' if raw.inference else 'ordinary'
            raw.recorded = ctx.grad and ctx.depth == 0
            raw.graph = raw.recorded
            producer_stream = 'a' if auxiliary else 'f'
            if auxiliary:
                tx.engine.submit('a', 'aux-delay')
            tx.engine.submit(producer_stream, 'forward', reads=(inp.b,), writes=(y.b,),
                             action=lambda: setattr(y.b, 'data',
                                                    tuple(2 * v + 1 for v in inp.b.data)))
            if auxiliary:
                # The toy forward has one auxiliary output producer. Its
                # submissions for this y version are now closed.
                e_y = tx.engine.fence('a', 'aux-ready')
                bind(y, e_y)
                if not omit_select_wait:
                    tx.engine.submit('f', 'wait-y-producers', deps=(e_y.command,))
            # This is a numerical gather into distinct protected storage,
            # rather than an alias or an assertion about output readiness.
            tx.engine.submit('f', 'select', reads=(y.b,), writes=(raw.b,),
                             action=lambda: setattr(raw.b, 'data',
                                                    tuple(y.b.data[p] for p in selected)))
            if auxiliary:
                # Retained even in the negative control: a terminal join
                # cannot repair a selection that has already read bad input.
                tx.engine.submit('f', 'join-internal-producers', deps=(e_y.command,))
            event = tx.engine.fence('f', 'forward-ready', fail=fault == 'forward_event')
            bind(raw, event)
        finally:
            pop(ctx, saved)
        if scope == 'postprocess_only' and raw.inference:
            tx.z = materialize(tx, raw, omit_wait=omit_copy_wait)
        else:
            tx.z = raw
        tx.state = 'Pending'
    except Exception as error:
        tx.fail(error)
    return tx


def resume(tx):
    require(tx.state == 'Pending', 'resume requires pending request')
    try:
        tx.engine.wait(tx.z.ready[2])
        consumer_oracle(tx.z, tx.scope, tx.ctx)
        if tx.fault == 'validation':
            raise Violation('injected consumer contract failure')
        if tx.scope == 'postprocess_only':
            leaf = Tensor(tx.z.b, inference=False, origin='ordinary')
            leaf.ready = tx.z.ready
            tx.z = leaf
        saved = push(tx.ctx, 'record' if tx.scope != 'none' else 'no_grad')
        try:
            result = tx.tensor()
            tx.result = result
            result.recorded = tx.ctx.grad and tx.ctx.depth == 0
            result.graph = result.recorded
            tx.engine.submit('k', 'consumer', reads=(tx.z.b,), writes=(result.b,),
                             action=lambda: setattr(result.b, 'data',
                                                    tuple(v * v for v in tx.z.b.data)))
            event = tx.engine.fence('k', 'consumer-ready',
                                    fail=tx.fault == 'consumer_event')
            bind(result, event)
        finally:
            pop(tx.ctx, saved)
        tx.engine.wait(result.ready[2])
        consumer_oracle(result, 'none', tx.ctx)
        if tx.fault == 'prepublish':
            raise Violation('injected publication preparation failure')
        # Atomic visible commit; no failure injection follows this point.
        tx.owner = 'result handle including graph roots'
        tx.visible = True
        tx.state = 'Committed'
        tx.cleanup = 'HandleOwned'
    except Exception as error:
        tx.fail(error)
    return tx


def validate_manifest(actual, expected):
    require(set(actual) == set(expected), 'manifest coverage changed')
    for name in expected:
        require(actual[name] == expected[name], 'module mode/identity changed: ' + name)
    return True


def retained_capacity(views):
    sizes = {}
    for storage, elements, width, capacity in views:
        require(storage not in sizes or sizes[storage] == capacity, 'inconsistent storage identity')
        sizes[storage] = capacity
    return sum(sizes.values())


def logical_payload(views):
    return sum(elements * width for _, elements, width, _ in views) + 16 * len(views)


def main():
    observations = 0

    def check(condition, message):
        nonlocal observations
        if not condition:
            raise AssertionError(message)
        observations += 1
        print('OK ' + message)

    def rejects(call, message):
        caught = False
        try:
            call()
        except Violation:
            caught = True
        check(caught, message)

    expected = {'root': (1, 'Root', False), 'drop': (2, 'Dropout', False),
                'norm': (3, 'BatchNorm', False)}
    actual = dict(expected)
    check(validate_manifest(actual, expected), 'full uniform mode manifest accepted')
    actual['drop'] = (2, 'Dropout', True)
    rejects(lambda: validate_manifest(actual, expected), 'child train mismatch rejected')
    check(actual['root'] == expected['root'], 'negative control: root-only validator misses child error')
    mixed = dict(expected)
    mixed['drop'] = (2, 'Dropout', True)
    check(validate_manifest(actual, mixed), 'declared mixed mode accepted')

    ctx = Context()
    first = push(ctx, 'inference')
    second = push(ctx, 'no_grad')
    pop(ctx, second)
    check((ctx.grad, ctx.depth) == (False, 1), 'nested no_grad preserves outer inference')
    rejects(lambda: push(ctx, 'record'), 'service record rule rejects active inference')
    pop(ctx, first)
    check((ctx.grad, ctx.depth) == (True, 0), 'contexts restore exactly')

    modes = (('inference', 'postprocess_only'),
             ('no_grad', 'postprocess_only'),
             ('record', 'through_model'), ('inference', 'none'))
    for kind, scope in modes:
        tx = begin((3, -2, 0), kind, scope)
        check(tx.state == 'Pending' and not tx.visible, kind + '/' + scope + ' starts pending')
        check(not tx.z.ready[2].completed, 'event is not fabricated complete at submission')
        rejects(lambda: consumer_oracle(tx.z, scope, tx.ctx), 'same oracle rejects pending consumption')
        check(lifetime_oracle(tx.engine), 'all submitted accesses have prior storage protection')
        if kind == 'no_grad':
            check(tx.copy is None and tx.z.ready[2] is tx.raw.ready[2],
                  'no_grad uses actual forward completion credential without copying')
        resume(tx)
        # Independent, hand-computed reference: (2x+1)^2 for x=(3,-2,0).
        check(tx.visible and tx.state == 'Committed' and tx.result.b.data == (49, 9, 1),
              'completed result matches independent numeric reference')
        check((tx.ctx.grad, tx.ctx.depth) == (True, 0), 'consumer restores worker context')
        if kind == 'inference' and scope == 'postprocess_only':
            check(not tx.copy.inference and tx.copy.b is not tx.raw.b,
                  'legal conversion creates distinct ordinary storage')
            rejects(lambda: consumer_oracle(tx.copy, 'through_model', tx.ctx),
                    'copied output cannot recover omitted model graph')
            t = tx.copy
            saved_ready = t.ready
            t.ready = None
            rejects(lambda: consumer_oracle(t, 'postprocess_only', tx.ctx),
                    'ordinary completed tensor without credential rejected')
            t.ready = (t.b.number, t.b.epoch, tx.raw.ready[2])
            rejects(lambda: consumer_oracle(t, 'postprocess_only', tx.ctx),
                    'completed forward event cannot certify later copy write')
            t.ready = (t.b.number, t.b.epoch - 1, saved_ready[2])
            rejects(lambda: consumer_oracle(t, 'postprocess_only', tx.ctx),
                    'stale write-version credential rejected')
            t.ready = saved_ready
            check(consumer_oracle(t, 'postprocess_only', tx.ctx),
                  'restored authentic completion credential accepted')
            print('OBS normal device trace=' + repr(tx.engine.trace))
        tx.engine.drain()
        tx.release()
        check(all(b.freed for b in tx.engine.buffers), 'completed handle retirement permits reclamation')

    # Auxiliary output production must precede the first real read by select.
    # The reference constants are supplied independently of queue actions.
    for kind, scope in modes:
        tx = begin((3, -2, 0), kind, scope,
                   auxiliary=True, positions=(2, 0, 2))
        check(tx.state == 'Pending' and not tx.visible,
              'auxiliary ' + kind + '/' + scope + ' starts pending')
        check(tx.y.b.data is None and tx.raw.b.data is None,
              'auxiliary y and selected raw are genuinely uninitialized')
        check(tx.engine.step('f') is False,
              'selection stream cannot pass its pre-read producer wait')
        check(tx.engine.step('a') and tx.engine.trace == ['aux-delay'],
              'auxiliary delay advances without producing y')
        check(tx.engine.step('f') is False and tx.raw.b.data is None,
              'delay alone cannot authorize selection')
        rejects(lambda: consumer_oracle(tx.z, scope, tx.ctx),
                'auxiliary output remains pending before production')
        resume(tx)
        check(tx.state == 'Committed' and tx.visible
              and tx.y.b.data == (7, -3, 1)
              and tx.raw.b.data == (1, 7, 1)
              and tx.result.b.data == (1, 49, 1),
              'delayed producer and actual reordered duplicate gather match hand reference')
        trace = tx.engine.trace
        check(trace.index('forward') < trace.index('aux-ready')
              < trace.index('wait-y-producers') < trace.index('select')
              < trace.index('forward-ready') < trace.index('consumer'),
              'executed trace orders producer before selection and raw event after selection')
        check(tx.engine.covers(tx.raw.ready[2], tx.y.b.writer)
              and tx.engine.covers(tx.raw.ready[2], tx.raw.b.writer),
              'raw completion credential covers producer and selection writes')
        check((tx.ctx.grad, tx.ctx.depth) == (True, 0)
              and lifetime_oracle(tx.engine),
              'auxiliary path preserves worker context and protected storage')
        if kind == 'inference' and scope == 'postprocess_only':
            print('OBS auxiliary device trace=' + repr(trace))
        tx.engine.drain()
        tx.release()

    for data, positions, raw_reference, result_reference in (
            ((), (), (), ()),
            ((3, -2, 0), (), (), ()),
            ((4,), (0,), (9,), (81,))):
        tx = begin(data, auxiliary=True, positions=positions)
        resume(tx)
        check(tx.state == 'Committed' and tx.visible
              and tx.raw.b.data == raw_reference
              and tx.result.b.data == result_reference,
              'auxiliary empty-input, empty-selection or singleton boundary matches reference')
        tx.engine.drain()
        tx.release()

    for positions in ((-1,), (3,)):
        invalid = begin((3, -2, 0), auxiliary=True, positions=positions)
        check(invalid.state == 'Failed' and not invalid.visible
              and invalid.engine.ops == []
              and invalid.error == 'invalid output position',
              'invalid logical output position fails before device submission')
        invalid.cleanup_failure()
        check(invalid.cleanup == 'Reclaimed',
              'invalid-position failure still reclaims protected input')

    missing_select_dep = begin((3, -2, 0), auxiliary=True,
                               positions=(2, 0, 2), omit_select_wait=True)
    check(any(op['label'] == 'join-internal-producers'
              for op in missing_select_dep.engine.ops),
          'negative selection control retains the terminal auxiliary join')
    reason = None
    try:
        missing_select_dep.engine.step('f')
    except Violation as error:
        reason = str(error)
    check(reason == 'device read before producer initialized storage'
          and missing_select_dep.y.b.data is None
          and missing_select_dep.raw.b.data is None
          and not missing_select_dep.visible,
          'negative control: missing pre-selection wait is rejected at actual uninitialized read')
    missing_select_dep.fail(Violation(reason))
    missing_select_dep.cleanup_failure(proof_available=False)
    check(missing_select_dep.cleanup == 'Quarantined'
          and lifetime_oracle(missing_select_dep.engine),
          'known faulty selection trace is retained without claiming a successful drain')
    print('OBS selection negative control: terminal join present; actual select read rejected: '
          + str(reason))

    aux_failed = begin((3, -2, 0), auxiliary=True,
                       positions=(2, 0, 2), fault='copy_event')
    check(aux_failed.state == 'Failed' and not aux_failed.visible
          and aux_failed.cleanup == 'DrainPending',
          'copy-event failure also terminates an auxiliary-producer request invisibly')
    check(all(t.b.pins == 1 and not t.b.freed
              for t in (aux_failed.y, aux_failed.raw, aux_failed.copy))
          and lifetime_oracle(aux_failed.engine),
          'auxiliary y, selected raw and copy target remain protected after failure')
    rejects(aux_failed.release,
            'auxiliary producer failure cannot reclaim before production and copy finish')
    aux_failed.cleanup_failure()
    check(aux_failed.cleanup == 'Reclaimed'
          and not aux_failed.engine.queues['a']
          and 'select' in aux_failed.engine.trace
          and 'copy' in aux_failed.engine.trace
          and all(b.freed for b in aux_failed.engine.buffers),
          'failure drain includes auxiliary production, selection and copy before reclamation')

    empty = begin(())
    rejects(lambda: consumer_oracle(empty.z, empty.scope, empty.ctx),
            'empty output still requires an executed completion point')
    resume(empty)
    check(empty.result.b.data == () and empty.visible, 'empty numerical boundary completes correctly')
    empty.release()

    outer_ctx = Context(grad=False, depth=1)
    outer = begin((1,), ctx=outer_ctx)
    check(outer.state == 'Failed' and not outer.visible and outer.copy is None,
          'outer inference is rejected before ordinary target allocation')
    check((outer_ctx.grad, outer_ctx.depth) == (False, 1), 'failed conversion preserves outer context')
    outer.cleanup_failure()
    check(outer.cleanup == 'Reclaimed', 'outer-context rejection still drains submitted forward')

    # Negative converter really allocates under inference, rather than changing a label.
    fake_tx = begin((1,), scope='none')
    saved = push(fake_tx.ctx, 'inference')
    try:
        fake = materialize(fake_tx, fake_tx.raw, faulty_inside=True)
    finally:
        pop(fake_tx.ctx, saved)
    fake_tx.engine.wait(fake.ready[2])
    check(fake.inference and fake.b is not fake_tx.raw.b, 'faulty inside-inference copy has new but restricted storage')
    rejects(lambda: consumer_oracle(fake, 'postprocess_only', fake_tx.ctx),
            'same oracle detects faulty inside-inference materialization')
    fake_tx.engine.drain()
    fake_tx.release()

    missing_dep = begin((1,), omit_copy_wait=True)
    rejects(lambda: missing_dep.engine.step('c'),
            'negative control: copy without producer dependency reads uninitialized storage')
    missing_dep.fail(Violation('known faulty producer dependency'))
    missing_dep.cleanup_failure(proof_available=False)

    failed = begin((1, 2), fault='copy_event')
    check(failed.state == 'Failed' and not failed.visible and failed.cleanup == 'DrainPending',
          'event registration failure terminates publication before physical cleanup')
    check(failed.copy.ready is None and failed.engine.pending_accesses(failed.copy.b) == ['copy'],
          'copy is genuinely queued although its completion event is absent')
    check(failed.raw.b.pins == 1 and failed.copy.b.pins == 1 and lifetime_oracle(failed.engine),
          'source and destination were protected before asynchronous submission')
    rejects(failed.release, 'physical reclaim oracle rejects release of a failed but in-flight request')
    print('OBS event-registration failure: terminal=Failed, visible=False, cleanup=DrainPending, pending-copy=True')
    failed.cleanup_failure()
    check(failed.cleanup == 'Reclaimed' and 'copy' in failed.engine.trace,
          'registered-stream drain completes copy without needing the missing event')
    check(all(b.freed for b in failed.engine.buffers), 'release follows successful physical drain')

    faulty_release = begin((1,), fault='copy_event')
    faulty_release.copy.b.freed = True
    faulty_release.copy.b.pins = 0
    rejects(lambda: lifetime_oracle(faulty_release.engine),
            'negative control: immediate exception release violates independent pending-access invariant')

    quarantined = begin((1,), fault='copy_event')
    quarantined.cleanup_failure(proof_available=False)
    check(quarantined.cleanup == 'Quarantined' and quarantined.owner == 'quarantine supervisor',
          'unproven drain transfers persistent ownership to quarantine')
    check(lifetime_oracle(quarantined.engine)
          and all(not b.freed and b.pins == 1 for b in quarantined.engine.buffers),
          'quarantine retains both source and target instead of claiming reclamation')
    rejects(lambda: quarantined.engine.submit('f', 'illegal-new-request'),
            'quarantined execution domain rejects reuse')

    for fault in ('forward_event', 'validation', 'consumer_event', 'prepublish'):
        tx = begin((1,), fault=fault)
        if tx.state == 'Pending':
            resume(tx)
        check(tx.state == 'Failed' and not tx.visible, fault + ' has explicit invisible failure terminal')
        check((tx.ctx.grad, tx.ctx.depth) == (True, 0), fault + ' restores context')
        check(lifetime_oracle(tx.engine), fault + ' retains physical lifetimes')
        tx.cleanup_failure()
        check(tx.cleanup == 'Reclaimed', fault + ' reaches physical cleanup through registered streams')

    mib = 1024 * 1024
    elements = 8 * 1024 * 4096
    independent = [(i, elements, 2, 64 * mib) for i in range(12)]
    check(logical_payload(independent) == 805306560,
          'independent compact activation logical bytes match hand-computed reference')
    check(retained_capacity(independent) == 768 * mib, 'independent physical storages total 768 MiB')
    small = [('shared', 1, 2, 64 * mib)] * 12
    full = [('shared', elements, 2, 64 * mib)] * 12
    check(logical_payload(small) == 216 and retained_capacity(small) == 64 * mib,
          'small views retain a much larger physical allocation')
    check(logical_payload(full) == 805306560 and retained_capacity(full) == 64 * mib,
          'full shared views count one physical storage')
    bad_capacity = sum(v[3] for v in full)
    check(bad_capacity == 768 * mib and bad_capacity != retained_capacity(full),
          'negative control: per-view capacity accounting is detected as duplicate billing')
    check(logical_payload([]) == 0 and retained_capacity([]) == 0, 'empty storage accounting')

    print('OBS scope: CPU abstract queues, delayed auxiliary producer, actual numerical gather, context and storage identities')
    print('OBS input references: full=(49,9,1); positions=(2,0,2) raw=(1,7,1), squared=(1,49,1)')
    print('OBS not measured: PyTorch autograd, real CUDA visibility/streams, allocator behavior, performance or model quality')
    print('OBS auxiliary read control detects uninitialized storage; it does not model all stale-version races')
    print('RESULT: PASS (' + str(observations) + ' scoped observations; targeted negative controls detected)')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (AssertionError, Violation) as error:
        print('RESULT: FAIL: ' + str(error))
        sys.exit(1)
