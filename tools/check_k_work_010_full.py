from fractions import Fraction
from heapq import heappop, heappush


TERMINAL = {'COMMITTED', 'FAILED', 'CANCELLED'}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def rejected(name, action):
    try:
        action()
    except AssertionError as exc:
        print('negative rejected: ' + name + ': ' + str(exc))
        return
    raise AssertionError('negative control escaped: ' + name)


def key(v, event):
    return (v, 0, event, 1)


def node(a=0, deps=(), events=(('FULL', 1),), lag=0, bad=(), extra=()):
    return dict(a=a, deps=tuple(deps), events=tuple(events), lag=lag,
                bad=frozenset(bad), extra=tuple(extra))


def operations(n):
    result = {event: (offset, event, event not in n['bad'])
              for event, offset in n['events']}
    result.update({uid: (offset, None, True) for uid, offset in n['extra']})
    return result


def validate(spec, cancellations):
    visiting, visited = set(), set()

    def walk(v):
        check(v not in visiting, 'cycle')
        if v in visited:
            return
        check(v.isascii() and v.isalpha() and v.isupper(), 'fixture node identity')
        visiting.add(v)
        n = spec[v]
        kinds = [e for e, _ in n['events']]
        check(n['a'] >= 0 and n['lag'] >= 0, 'negative time')
        check(kinds and len(kinds) == len(set(kinds)), 'event identity')
        check(kinds[-1] in ('FULL', 'TOOL_OK', 'QUALITY'), 'final event')
        offsets = [t for _, t in n['events']]
        check(offsets == sorted(offsets) and min(offsets) >= 0, 'event offsets')
        check(n['bad'] <= set(kinds), 'undeclared failed operation')
        extra_ids = [uid for uid, _ in n['extra']]
        check(len(extra_ids) == len(set(extra_ids)) and not set(extra_ids) & set(kinds),
              'operation identity')
        check(all(t >= 0 for _, t in n['extra']), 'extra offset')
        # The final event identifies the common publication, not the last
        # completing operation. Required auxiliary operations may finish later.
        for u, event, wait in n['deps']:
            check(u in spec and wait >= 0, 'edge contract')
            check(event in dict(spec[u]['events']), 'producer interface')
            walk(u)
        visiting.remove(v)
        visited.add(v)

    for v in spec:
        walk(v)
    check(all(v in spec and t >= 0 for v, t in cancellations.items()), 'cancel plan')


def reference(spec, cancellations, fault='', witness=None):
    # Independent DAG constraints: no heap, delivery counters or replay state.
    # A terminal retains its originating ordered cause, including propagation.
    # bad declares required event-operation failures; all extra operations succeed.
    # A final commit candidate exists only when all required outcomes succeed.
    memo = {}

    def visit(v):
        if v in memo:
            return memo[v]
        n = spec[v]
        bounds = [n['a']]
        impossible = []
        own_cancel = None
        if v in cancellations:
            own_cancel = ((cancellations[v], 0, v, ''), 'CANCELLED')
        for u, event, wait in n['deps']:
            upstream = visit(u)
            if event not in upstream['pub']:
                impossible.append((upstream['terminal'][0], 'FAILED'))
            else:
                pt = upstream['pub'][event]
                if fault == 'serial_wait':
                    bounds.append(pt + spec[u]['lag'] + wait)
                else:
                    bounds.extend((pt + wait, pt + spec[u]['lag']))
        t = max(bounds)
        if impossible:
            choices = impossible + ([own_cancel] if own_cancel else [])
            terminal = min(choices, key=lambda x: x[0])
            result = dict(start=None, pub={}, terminal=terminal)
        elif own_cancel and own_cancel[0][0] <= t:
            # The selected contract completes all external cancels at this time
            # before admitting new work, regardless of their identity order.
            result = dict(start=None, pub={}, terminal=own_cancel)
        else:
            final, offset = n['events'][-1]
            completion_keys = {uid: (t + data[0], 1, v, uid)
                               for uid, data in operations(n).items()}
            commit_key = max(completion_keys.values())
            if fault == 'final_completion_only':
                commit_key = completion_keys[final]
            final_eligible = not n['bad']
            choices = []
            if final_eligible or fault == 'old_final_eligibility':
                choices.append((commit_key, 'COMMITTED'))
            choices.extend(((t + off, 1, v, event), 'FAILED')
                           for event, off in n['events'] if event in n['bad'])
            if own_cancel:
                choices.append(own_cancel)
            terminal = min(choices, key=lambda x: x[0])
            publications, publication_keys = {}, {}
            for event, off in n['events']:
                ek = (t + off, 1, v, event)
                if event == final and terminal[1] == 'COMMITTED':
                    ek = commit_key
                final_gate = event != final or terminal[1] == 'COMMITTED'
                if fault in ('old_final_eligibility', 'final_as_nonfinal'):
                    final_gate = True
                if event not in n['bad'] and final_gate and (ek < terminal[0] or
                        (ek == terminal[0] and terminal[1] == 'COMMITTED')):
                    publications[event] = ek[0]
                    publication_keys[event] = ek + (int(event == final), event)
            result = dict(start=t, pub=publications, terminal=terminal,
                          publication_keys=publication_keys, commit_key=commit_key)
        memo[v] = result
        return result

    for v in spec:
        visit(v)
    starts, pub, states, outputs, terminals = {}, {}, {}, {}, {}
    for v, n in spec.items():
        r = memo[v]
        if witness is not None:
            witness[v] = {'terminal_key': r['terminal'][0],
                          'publication_keys': r.get('publication_keys', {}),
                          'commit_candidate_key': r.get('commit_key')}
        if r['start'] is not None:
            starts[v] = r['start']
        pub.update({key(v, e): t for e, t in r['pub'].items()})
        states[v] = r['terminal'][1]
        terminals[v] = (r['terminal'][0][0], states[v])
        if states[v] == 'COMMITTED':
            outputs[v] = (v, n['events'][-1][0], 1)
        if fault == 'cancel_plan_wins' and v in cancellations and states[v] != 'COMMITTED':
            states[v] = 'CANCELLED'
            terminals[v] = (cancellations[v], 'CANCELLED')
    return starts, pub, states, outputs, terminals


def replay(spec, cancellations, fault=''):
    validate(spec, cancellations)
    states = {v: 'BLOCKED' for v in spec}
    starts, outputs, published, terminals = {}, {}, {}, {}
    received = {v: {} for v in spec}
    credits = {v: 0 for v in spec}
    pending, done, held = {}, set(), {}
    candidates = {v: {} for v in spec}
    closed_intent, timers = set(), set()
    queue, trace = [], []
    serial = 0

    def push(t, priority, stable, action, *args):
        nonlocal serial
        serial += 1
        heappush(queue, (t, priority, stable, serial, action, args))

    def emit(t, action, *args):
        trace.append((t, action) + args)

    def terminate(t, v, state):
        if states[v] not in TERMINAL:
            states[v] = state
            terminals[v] = (t, state)
            candidates[v].clear()

    def publish_ready(t, v):
        if states[v] != 'RUNNING':
            return
        final = spec[v]['events'][-1][0]
        # Local successful events become visible before common final commit.
        # A saved final must be revisited even when the triggering uid has no event.
        ordered = sorted(candidates[v], key=lambda event: (event == final, event))
        for event in ordered:
            if event == final and not all((v, u) in done and data[2]
                                          for u, data in operations(spec[v]).items()):
                continue
            k = key(v, event)
            check(k not in published, 'repeated publication')
            payload = candidates[v].pop(event)
            published[k] = t
            emit(t, 'PUBLISH', k, payload)
            if event == final:
                outputs[v] = payload
                terminate(t, v, 'COMMITTED')
            for target, n in sorted(spec.items()):
                for i, (u, required, _) in enumerate(n['deps']):
                    if (u, required) == (v, event):
                        for copy in range(2):
                            push(t + spec[v]['lag'], 2,
                                 (target, '%06d:%d' % (i, copy)),
                                 'notify', target, i, k)

    def cancel(t, v, origin):
        emit(t, 'CANCEL', v, origin)
        closed_intent.add(v)
        terminate(t, v, 'CANCELLED')

    def fail_operation(t, v, uid):
        if states[v] not in TERMINAL:
            emit(t, 'FAIL_OP', v, uid)
            terminate(t, v, 'FAILED')
            if fault == 'failure_frees_inflight':
                held[v] = False

    def propagate(t):
        changed = True
        while changed:
            changed = False
            for v, n in sorted(spec.items()):
                if states[v] not in ('BLOCKED', 'READY'):
                    continue
                for u, event, _ in n['deps']:
                    if states[u] in ('FAILED', 'CANCELLED') and key(u, event) not in published:
                        emit(t, 'FAIL_DEP', v, u, event)
                        terminate(t, v, 'FAILED')
                        changed = True
                        break

    def reconsider(t):
        for v, n in sorted(spec.items()):
            if states[v] not in ('BLOCKED', 'READY') or v in starts:
                continue
            complete = len(received[v]) == len(n['deps'])
            if fault == 'duplicate_credit':
                complete = credits[v] >= len(n['deps'])
            if fault == 'ignore_dependencies':
                complete = True
            if not complete:
                continue
            bounds = [0 if fault == 'ignore_external' else n['a']]
            for i, k in received[v].items():
                wait = 0 if fault == 'ignore_wait' else n['deps'][i][2]
                bounds.append(published[k] + wait)
            release = max(bounds)
            if t < release:
                timer = (v, release)
                if timer not in timers:
                    timers.add(timer)
                    if fault != 'no_wait_wake':
                        push(release, 3, (v, ''), 'wake', v)
                continue
            starts[v] = t
            states[v] = 'RUNNING'
            pending[v] = set(operations(n))
            held[v] = True
            emit(t, 'START', v)
            for uid, (offset, _event, _ok) in sorted(operations(n).items()):
                for _copy in range(2):
                    push(t + offset, 1, (v, uid), 'complete', v, uid)

    for v, n in spec.items():
        push(n['a'], 3, (v, ''), 'wake', v)
    for v, t in cancellations.items():
        push(t, 0, (v, ''), 'cancel', v)
    steps = 0
    while queue:
        steps += 1
        check(steps < 10000, 'queue did not terminate')
        t, _priority, _stable, _serial, action, args = heappop(queue)
        if action == 'cancel':
            v = args[0]
            cancel(t, v, v)
            if fault == 'cancel_sibling' and v == 'B' and 'D' in spec:
                cancel(t, 'D', v)
        elif action == 'complete':
            v, uid = args
            opkey = (v, uid)
            if opkey in done:
                emit(t, 'COMPLETE_DUP', v, uid)
            else:
                _offset, event, ok = operations(spec[v])[uid]
                done.add(opkey)
                pending[v].remove(uid)
                emit(t, 'COMPLETE', v, uid, ok)
                if not ok and states[v] == 'RUNNING':
                    if fault == 'defer_failure':
                        if not pending[v]:
                            fail_operation(t, v, uid)
                    elif fault != 'no_failure_terminal':
                        fail_operation(t, v, uid)
                elif ok and fault == 'defer_failure' and states[v] == 'RUNNING' and not pending[v]:
                    failed = sorted(u for u in operations(spec[v])
                                    if not operations(spec[v])[u][2] and (v, u) in done)
                    if failed:
                        fail_operation(t, v, failed[0])
                if event is not None and ok:
                    k = key(v, event)
                    if states[v] != 'RUNNING':
                        emit(t, 'REJECT_CANDIDATE', k)
                    else:
                        payload = (v, event, 1)
                        candidates[v][event] = payload
                        emit(t, 'CANDIDATE', k, payload)
                final = spec[v]['events'][-1][0]
                if fault == 'final_event_only_recheck' and event != final:
                    # Publish this local event but omit the saved final recheck.
                    retained = candidates[v].pop(final, None)
                    publish_ready(t, v)
                    if retained is not None and states[v] == 'RUNNING':
                        candidates[v][final] = retained
                elif fault == 'no_auxiliary_recheck' and event is None:
                    pass
                else:
                    publish_ready(t, v)
                if fault == 'discard_unready_final' and states[v] == 'RUNNING':
                    candidates[v].pop(final, None)
        elif action == 'notify':
            v, i, k = args
            if fault == 'drop_old_publication' and k[0] in closed_intent:
                emit(t, 'DROP_NOTIFY', v, i, k)
            else:
                emit(t, 'NOTIFY', v, i, k)
                credits[v] += 1
                received[v].setdefault(i, k)
                if fault == 'revive_cancelled' and states[v] == 'CANCELLED':
                    states[v] = 'BLOCKED'
        elif action != 'wake':
            raise AssertionError('unknown queue action')
        # Each cancellation still propagates immediately and retains its order.
        propagate(t)
        # Only new starts wait until all external cancels at this time finish.
        if fault == 'dispatch_between_cancels' or not (
                queue and queue[0][0] == t and queue[0][1] == 0):
            reconsider(t)
        for v in held:
            if states[v] in TERMINAL and not pending[v]:
                held[v] = False
        snapshot = tuple((v, states[v] == 'RUNNING', tuple(sorted(pending[v])), held[v])
                         for v in sorted(held))
        emit(t, 'RESOURCE', snapshot)
    return dict(trace=trace, starts=starts, pub=published, states=states,
                outputs=outputs, terminals=terminals,
                pending={v: tuple(sorted(p)) for v, p in pending.items()},
                candidates={v: dict(candidates[v]) for v in starts})


def audit_events(spec, cancellations, result, reference_fault=''):
    # Ordered evidence reconstruction; no access to replay's dependency counters.
    states = {v: 'BLOCKED' for v in spec}
    starts, publications, delivered, outputs, terminals = {}, {}, {}, {}, {}
    finished, obligations, seen_cancels = {}, {}, set()
    completion_evidence = {}
    staged = {v: {} for v in spec}
    pending_failure = None
    previous_time = -1

    def terminal(v, t, state):
        if states[v] not in TERMINAL:
            states[v] = state
            terminals[v] = (t, state)
            staged[v].clear()

    for sequence, row in enumerate(result['trace']):
        t, action, *args = row
        check(t >= previous_time, 'trace time moved backwards')
        previous_time = t
        if pending_failure is not None:
            check(action == 'FAIL_OP' and tuple(args) == pending_failure,
                  'definite failed result must terminate in the same transaction')
        if action == 'START':
            v = args[0]
            check(all(cv in seen_cancels for cv, ct in cancellations.items() if ct == t),
                  'new start before same-time external cancellation phase ends')
            check(states[v] == 'BLOCKED' and v not in starts, 'terminal or repeated start: ' + v)
            check(t >= spec[v]['a'], 'external time gate: ' + v)
            for i, (u, event, wait) in enumerate(spec[v]['deps']):
                k = key(u, event)
                check(delivered.get((v, i)) == k, 'missing delivered edge: ' + v)
                check(k in publications, 'missing publication evidence: ' + v)
                pt, ps = publications[k]
                check(ps < sequence and t >= pt + wait, 'event order or wait gate: ' + v)
            starts[v] = t
            states[v] = 'RUNNING'
            obligations[v] = set(operations(spec[v]))
        elif action == 'COMPLETE':
            v, uid, ok = args
            check(v in starts and uid in operations(spec[v]), 'unknown completion')
            offset, _event, expected_ok = operations(spec[v])[uid]
            check(t == starts[v] + offset and ok == expected_ok, 'completion contract')
            check((v, uid) not in finished and uid in obligations[v], 'duplicate settlement')
            finished[(v, uid)] = ok
            completion_evidence[(v, uid)] = (t, sequence)
            obligations[v].remove(uid)
            if not ok and states[v] == 'RUNNING':
                pending_failure = (v, uid)
        elif action == 'COMPLETE_DUP':
            v, uid = args
            check((v, uid) in finished, 'duplicate without completed ticket')
        elif action == 'FAIL_OP':
            v, uid = args
            check(pending_failure == (v, uid) and states[v] == 'RUNNING', 'execution failure cause')
            check(finished.get((v, uid)) is False, 'execution failure without failed result')
            terminal(v, t, 'FAILED')
            pending_failure = None
        elif action == 'CANDIDATE':
            k, payload = args
            v, attempt, event, version = k
            check(states[v] == 'RUNNING' and attempt == 0 and version == 1,
                  'candidate identity or eligibility')
            check(finished.get((v, event)) is True and completion_evidence[(v, event)][1] < sequence,
                  'candidate without prior successful completion')
            check(k not in publications and event not in staged[v], 'duplicate candidate')
            check(payload == (v, event, 1), 'candidate payload identity')
            staged[v][event] = payload
        elif action == 'PUBLISH':
            k, payload = args
            v, attempt, event, version = k
            check(attempt == 0 and version == 1, 'publication identity')
            check(states[v] == 'RUNNING' and k not in publications, 'publication eligibility')
            check(event in dict(spec[v]['events']), 'undeclared event')
            check(finished.get((v, event)) is True, 'publication lacks successful operation')
            check(payload == (v, event, 1), 'payload identity')
            check(staged[v].get(event) == payload, 'publication lost the retained candidate')
            del staged[v][event]
            publications[k] = (t, sequence)
            if event == spec[v]['events'][-1][0]:
                check(not obligations[v] and all(finished[(v, u)] for u in operations(spec[v])),
                      'final publication before required operation success')
                last = max(completion_evidence[(v, u)] for u in operations(spec[v]))
                check(t == last[0] and last[1] < sequence, 'final publication before last required evidence')
                outputs[v] = payload
                terminal(v, t, 'COMMITTED')
            else:
                check(t == starts[v] + dict(spec[v]['events'])[event], 'local producer offset')
        elif action == 'NOTIFY':
            v, i, k = args
            check(k in publications and publications[k][1] < sequence, 'unpublished notification')
            u, event, _wait = spec[v]['deps'][i]
            check(k == key(u, event), 'notification edge identity')
            check(t >= publications[k][0] + spec[u]['lag'], 'notification transport')
            check((v, i) not in delivered or delivered[(v, i)] == k, 'edge conflict')
            delivered[(v, i)] = k
        elif action == 'CANCEL':
            v, origin = args
            check(v == origin and cancellations.get(v) == t, 'unauthorized sibling cancellation')
            check(v not in seen_cancels, 'duplicate cancellation command')
            seen_cancels.add(v)
            terminal(v, t, 'CANCELLED')
        elif action == 'FAIL_DEP':
            v, u, event = args
            check(states[v] == 'BLOCKED', 'dependency failure overwrote active or terminal node')
            check(states[u] in ('FAILED', 'CANCELLED') and key(u, event) not in publications,
                  'dependency failure without impossible evidence')
            check(any((a, b) == (u, event) for a, b, _ in spec[v]['deps']), 'failure edge')
            terminal(v, t, 'FAILED')
        elif action == 'REJECT_CANDIDATE':
            check(states[args[0][0]] in TERMINAL, 'unexplained candidate rejection')
        elif action == 'RESOURCE':
            expected = tuple((v, states[v] == 'RUNNING', tuple(sorted(obligations[v])),
                              states[v] == 'RUNNING' or bool(obligations[v]))
                             for v in sorted(starts))
            check(args[0] == expected, 'operation-buffer lifetime mismatch')
        elif action != 'DROP_NOTIFY':
            raise AssertionError('unknown audit record')
    check(pending_failure is None, 'unprocessed definite failure')
    check(all(not uses for uses in obligations.values()), 'model failed to drain registered operations')
    actual_pub = {k: t for k, (t, _sequence) in publications.items()}
    witness = {}
    expected = reference(spec, cancellations, reference_fault, witness)
    observed = (starts, actual_pub, states, outputs, terminals)
    check(observed == expected, 'independent start/publication/first-terminal reference mismatch')
    for v in starts:
        if states[v] == 'COMMITTED':
            last_uid = max(operations(spec[v]), key=lambda uid: completion_evidence[(v, uid)])
            last = completion_evidence[(v, last_uid)]
            cause = (last[0], 1, v, last_uid)
            final = spec[v]['events'][-1][0]
            check(witness[v]['terminal_key'] == cause and
                  witness[v]['publication_keys'][final] == cause + (1, final),
                  'reference final order differs from last required success evidence')
    check(seen_cancels == set(cancellations), 'missing external cancellation')
    check((result['starts'], result['pub'], result['states'], result['outputs'], result['terminals'])
          == observed, 'reported installation mismatch')
    check(result['pending'] == {v: () for v in starts}, 'reported obligation mismatch')
    check(result['candidates'] == {v: staged[v] for v in starts}, 'reported candidate retention mismatch')


class Memory:
    def __init__(self, fault=''):
        self.fault = fault
        self.free = {0, 1}
        self.epochs = {0: -1, 1: -1}
        self.current, self.objects, self.operations = {}, {}, {}
        self.finished = set()

    def allocate(self, owners):
        if not self.free:
            return None
        slot = min(self.free)
        self.free.remove(slot)
        self.epochs[slot] += 1
        x = (slot, self.epochs[slot])
        self.current[slot] = x
        self.objects[x] = dict(L=set(owners), U=set(), retiring=False)
        return x

    def reclaim(self, x):
        o = self.objects[x]
        if o['retiring'] and not o['L'] and not o['U'] and self.current.get(x[0]) == x:
            del self.current[x[0]]
            self.free.add(x[0])

    def issue(self, uid, objects):
        check(uid not in self.operations, 'operation reused')
        self.operations[uid] = tuple(objects)
        for x in objects:
            check(self.current.get(x[0]) == x, 'issue to stale allocation')
            self.objects[x]['U'].add(uid)

    def drop(self, owner):
        for x in list(self.current.values()):
            o = self.objects[x]
            o['L'].discard(owner)
            if not o['L']:
                o['retiring'] = True
                if self.fault == 'cancel_free' and owner == 'A':
                    o['U'].clear()
            self.reclaim(x)

    def complete(self, uid, proof=None):
        if uid not in self.operations:
            return
        if proof is not None and tuple(proof) != self.operations[uid]:
            return
        if uid in self.finished:
            if self.fault == 'duplicate_settles_other':
                for x in self.operations.get('kernel', ()):
                    self.objects[x]['U'].discard('kernel')
                    self.reclaim(x)
            return
        self.finished.add(uid)
        for x in self.operations[uid]:
            self.objects[x]['U'].discard(uid)
            self.reclaim(x)

    def snapshot(self):
        rows = []
        for x in self.current.values():
            o = self.objects[x]
            rows.append((x[0], x[1], tuple(sorted(o['L'])), tuple(sorted(o['U'])), o['retiring']))
        return tuple(sorted(rows)), tuple(sorted(self.free))


def memory_trace(fault=''):
    m = Memory(fault)
    s, p = m.allocate({'cache', 'A', 'B'}), m.allocate({'A'})
    m.issue('copy', (s, p))
    m.issue('kernel', (s, p))
    m.issue('brother', (s,))
    observations = {'initial': m.snapshot()}
    m.drop('A')
    observations['cancel'] = m.snapshot()
    early = m.allocate({'C'})
    observations['early_request'] = m.snapshot()
    m.complete('copy')
    observations['copy_done'] = m.snapshot()
    m.complete('copy')
    observations['copy_duplicate'] = m.snapshot()
    m.complete('kernel', ((0, 99), p))
    observations['wrong_epoch'] = m.snapshot()
    m.complete('kernel')
    observations['kernel_done'] = m.snapshot()
    later = m.allocate({'C'})
    observations['C_allocated'] = m.snapshot()
    m.complete('kernel')
    observations['old_duplicate'] = m.snapshot()
    m.drop('B')
    m.drop('cache')
    observations['shared_retired'] = m.snapshot()
    blocked = m.allocate({'D'})
    m.complete('brother')
    observations['brother_done'] = m.snapshot()
    last = m.allocate({'D'})
    observations['D_allocated'] = m.snapshot()
    return observations, (early, later, blocked, last)


def audit_memory(result):
    # Independent hand-derived obligations: copy/kernel use S/P; brother uses S.
    # O/B/Q are empty, and these are physical allocation identities, not contents.
    def row(slot, epoch, owners, uses, retiring=False):
        return slot, epoch, tuple(sorted(owners)), tuple(sorted(uses)), retiring

    def snap(rows, free=()):
        return tuple(sorted(rows)), tuple(sorted(free))

    s0 = row(0, 0, {'cache', 'A', 'B'}, {'copy', 'kernel', 'brother'})
    p0 = row(1, 0, {'A'}, {'copy', 'kernel'})
    s1 = row(0, 0, {'cache', 'B'}, {'copy', 'kernel', 'brother'})
    p1 = row(1, 0, set(), {'copy', 'kernel'}, True)
    s2 = row(0, 0, {'cache', 'B'}, {'kernel', 'brother'})
    p2 = row(1, 0, set(), {'kernel'}, True)
    s3 = row(0, 0, {'cache', 'B'}, {'brother'})
    c = row(1, 1, {'C'}, set())
    sr = row(0, 0, set(), {'brother'}, True)
    d = row(0, 1, {'D'}, set())
    expected = {
        'initial': snap([s0, p0]), 'cancel': snap([s1, p1]),
        'early_request': snap([s1, p1]), 'copy_done': snap([s2, p2]),
        'copy_duplicate': snap([s2, p2]), 'wrong_epoch': snap([s2, p2]),
        'kernel_done': snap([s3], {1}), 'C_allocated': snap([s3, c]),
        'old_duplicate': snap([s3, c]), 'shared_retired': snap([sr, c]),
        'brother_done': snap([c], {0}), 'D_allocated': snap([d, c]),
    }
    observed, allocations = result
    check(set(observed) == set(expected), 'resource checkpoint coverage')
    for label, wanted in expected.items():
        check(observed[label] == wanted, 'physical obligation mismatch at ' + label)
    check(allocations == (None, (1, 1), None, (0, 1)), 'allocation lifetime outcomes')


def turn_model(system, users, answers, faulty=False):
    computed, history, result = 0, system, []
    for user, answer in zip(users, answers):
        n = history + user
        result.append(n - computed)
        computed = n + (answer if faulty else max(answer - 1, 0))
        history = n + answer
    return result, computed


def turn_reference(system, users, answers):
    history = [('S', i) for i in range(system)]
    consumed, result = set(), []
    for turn, (user, answer) in enumerate(zip(users, answers)):
        history.extend(('U', turn, i) for i in range(user))
        result.append(sum(token not in consumed for token in history))
        consumed.update(history)
        generated = [('A', turn, i) for i in range(answer)]
        history.extend(generated)
        consumed.update(generated[:-1])
    return result, len(consumed)


def branch_layout(page, length, tails):
    source = [('P', i) for i in range(length)]
    physical = {('source', i // page): tuple(source[i:i + page])
                for i in range(0, length, page)}
    remainder = length % page
    for branch, tail in enumerate(tails):
        if tail == 0:
            continue
        local = source[-remainder:] if remainder else []
        local = local + [('T', branch, i) for i in range(tail)]
        for i in range(0, len(local), page):
            physical[('branch', branch, i // page)] = tuple(local[i:i + page])
    peak = len(physical)
    if remainder and all(t > 0 for t in tails):
        del physical[('source', length // page)]
    return peak, len(physical)


def ceildiv(n, d):
    return (n + d - 1) // d


def run_checks():
    pipeline = {
        'A': node(events=(('FIRST', 1), ('FULL', 4))),
        'B': node(deps=(('A', 'FIRST', 2),)),
        'C': node(a=10, deps=(('A', 'FULL', 2),)),
        'D': node(deps=(('B', 'FULL', 0),), events=(('TOOL_OK', 2),)),
        'Q': node(deps=(('D', 'TOOL_OK', 0),), events=(('QUALITY', 1),)),
        'J': node(deps=(('C', 'FULL', 0), ('Q', 'QUALITY', 1)), events=(('FULL', 0),)),
    }
    duplicate = {
        'A': node(), 'B': node(events=(('FULL', 5),)),
        'V': node(deps=(('A', 'FULL', 0), ('B', 'FULL', 0))),
    }
    branches = {
        'R': node(),
        'B': node(deps=(('R', 'FULL', 0),), events=(('FIRST', 1), ('FULL', 5)), lag=2),
        'D': node(deps=(('R', 'FULL', 0),), events=(('FULL', 4),)),
        'F': node(deps=(('B', 'FIRST', 0),)),
        'X': node(deps=(('D', 'FULL', 0),)),
        'J': node(deps=(('B', 'FULL', 0), ('D', 'FULL', 0))),
    }
    committed_then_cancelled = {'U': node(lag=3), 'V': node(deps=(('U', 'FULL', 0),))}
    before_first = {
        'T': node(events=(('FIRST', 2), ('FULL', 3))),
        'W': node(deps=(('T', 'FIRST', 0),)),
    }
    overlap = {'U': node(lag=3), 'V': node(deps=(('U', 'FULL', 2),))}
    wait_dominates = {'U': node(lag=1), 'V': node(deps=(('U', 'FULL', 5),))}
    dep_first = {'U': node(events=(('FULL', 3),)), 'V': node(deps=(('U', 'FULL', 0),))}
    reverse_tie = {'Z': node(events=(('FULL', 3),)), 'A': node(deps=(('Z', 'FULL', 0),))}
    last_failure = {'U': node(bad=('FULL',))}
    draining_failure = {
        'U': node(events=(('FULL', 2),), bad=('FULL',), extra=(('AUX', 5),)),
        'V': node(deps=(('U', 'FULL', 0),)),
        'W': node(deps=(('V', 'FULL', 0),)),
    }
    first_then_failure = {
        'U': node(events=(('FIRST', 1), ('FULL', 2)), bad=('FULL',), lag=3),
        'F': node(deps=(('U', 'FIRST', 2),)),
        'V': node(deps=(('U', 'FULL', 0),)),
    }
    cases = [
        ('typed events and time gates', pipeline, {}),
        ('duplicate notifications with two dependencies', duplicate, {}),
        ('running cancellation and preserved FIRST', branches, {'B': 3, 'X': 2}),
        ('FULL committed before cancellation', committed_then_cancelled, {'U': 2}),
        ('cancellation before FIRST', before_first, {'T': 1}),
        ('positive transport overlaps positive wait', overlap, {}),
        ('wait exceeds transport', wait_dominates, {}),
        ('dependency failure precedes local cancellation', dep_first, {'U': 1, 'V': 2}),
        ('local cancellation precedes dependency failure', dep_first, {'V': 0, 'U': 1}),
        ('same-time upstream cancellation first', dep_first, {'U': 1, 'V': 1}),
        ('same-time consumer cancellation first', reverse_tie, {'Z': 1, 'A': 1}),
        ('last operation fails with complete last-use evidence', last_failure, {}),
        ('failure while auxiliary operation drains; required consumers wait', draining_failure, {}),
        ('FIRST survives later execution failure', first_then_failure, {}),
        ('late cancel preserves execution failure', draining_failure, {'U': 3, 'V': 4}),
        ('empty DAG', {}, {}),
    ]
    for name, spec, cancels in cases:
        result = replay(spec, cancels)
        audit_events(spec, cancels, result)
        print('event model accepted: ' + name + '; starts=' + str(sorted(result['starts'].items()))
              + '; terminals=' + str(sorted(result['terminals'].items())))

    check(reference(overlap, {})[0]['V'] == 4, 'transport/wait hand reference must be 4, not 6')
    check(reference(wait_dominates, {})[0]['V'] == 6, 'wait-dominated hand reference')
    check(reference(dep_first, {'U': 1, 'V': 2})[4]['V'] == (1, 'FAILED'),
          'first irreversible dependency terminal')
    check(reference(reverse_tie, {'Z': 1, 'A': 1})[4]['A'] == (1, 'CANCELLED'),
          'stable same-time consumer-first terminal')
    drained = replay(draining_failure, {})
    snapshots = [(t, data[0]) for t, action, *data in drained['trace'] if action == 'RESOURCE']
    check(any(t == 2 and ('U', False, ('AUX',), True) in rows for t, rows in snapshots),
          'failed producer retains its other physical ticket at time 2')
    check(any(t == 5 and ('U', False, (), False) in rows for t, rows in snapshots),
          'auxiliary completion releases buffer without reviving producer')
    check(drained['terminals'] == {'U': (2, 'FAILED'), 'V': (2, 'FAILED'), 'W': (2, 'FAILED')},
          'execution failure must propagate before physical drain')

    controls = [
        ('ignore dependencies', pipeline, {}, 'ignore_dependencies'),
        ('duplicate notification counts twice', duplicate, {}, 'duplicate_credit'),
        ('cancel sibling', branches, {'B': 3, 'X': 2}, 'cancel_sibling'),
        ('missing wait wake', pipeline, {}, 'no_wait_wake'),
        ('ignore nonroot external gate', pipeline, {}, 'ignore_external'),
        ('ignore dependency wait', pipeline, {}, 'ignore_wait'),
        ('revive cancelled successor', branches, {'B': 3, 'X': 2}, 'revive_cancelled'),
        ('discard pre-cancel FIRST', branches, {'B': 3, 'X': 2}, 'drop_old_publication'),
        ('discard committed FULL after cancel', committed_then_cancelled, {'U': 2}, 'drop_old_publication'),
        ('failed last operation leaves RUNNING', last_failure, {}, 'no_failure_terminal'),
        ('failure notification deferred until physical drain', draining_failure, {}, 'defer_failure'),
        ('failure releases another in-flight operation buffer', draining_failure, {}, 'failure_frees_inflight'),
    ]
    for name, spec, cancels, fault in controls:
        rejected(name, lambda s=spec, c=cancels, f=fault: audit_events(s, c, replay(s, c, f)))
    rejected('old oracle serially adds transport and wait',
             lambda: audit_events(overlap, {}, replay(overlap, {}), 'serial_wait'))
    rejected('old oracle lets later cancellation replace dependency failure',
             lambda: audit_events(dep_first, {'U': 1, 'V': 2},
                                  replay(dep_first, {'U': 1, 'V': 2}), 'cancel_plan_wins'))

    # Systematic combinations test both relative orders and both nonzero gates.
    combinations = 0
    for lag in range(4):
        for wait in range(4):
            for external in (0, 2, 7):
                for uc in (None, 0, 1, 2, 5):
                    for vc in (None, 0, 1, 2, 4, 7):
                        spec = {'U': node(lag=lag),
                                'V': node(a=external, deps=(('U', 'FULL', wait),))}
                        cancels = {}
                        if uc is not None:
                            cancels['U'] = uc
                        if vc is not None:
                            cancels['V'] = vc
                        audit_events(spec, cancels, replay(spec, cancels))
                        combinations += 1
    check(combinations == 1440, 'retained event combination coverage')
    print('event constraint combinations accepted: ' + str(combinations))

    audit_memory(memory_trace())
    print('resource model accepted: private reuse after kernel end; shared reuse after brother end')
    rejected('cancel frees in-flight private page', lambda: audit_memory(memory_trace('cancel_free')))
    rejected('duplicate completion settles kernel', lambda: audit_memory(memory_trace('duplicate_settles_other')))

    turn_cases = [(100, [20, 20, 20], [30, 30, 30]), (1, [1, 2], [0, 0]),
                  (2, [1, 3], [1, 2]), (0, [2, 1, 4], [3, 0, 1])]
    for system, users, answers in turn_cases:
        check(turn_model(system, users, answers) == turn_reference(system, users, answers),
              'turn token-identity reference')
    check(turn_model(100, [20] * 3, [30] * 3) == ([120, 21, 21], 249), 'declared turn fixture')
    rejected('assume final sampled token consumed',
             lambda: check(turn_model(100, [20] * 3, [30] * 3, True) ==
                           turn_reference(100, [20] * 3, [30] * 3), 'last-token boundary'))
    print('turn model accepted: nonuniform turns and zero/one-token answers; prompt-completed scope')

    count = 0
    for page in range(1, 6):
        for length in range(11):
            for tails in ([0, 0], [1, 1], [page + 1, page + 1], [0, page], [page, 2 * page + 1]):
                q, u = divmod(length, page)
                expected_peak = q + int(u > 0) + sum(ceildiv(u + t, page) for t in tails if t > 0)
                peak, steady = branch_layout(page, length, tails)
                check(peak == expected_peak, 'layout peak reference')
                check(steady == expected_peak - int(u > 0 and all(t > 0 for t in tails)),
                      'source tail last-use release')
                count += 1
    check(branch_layout(4, 3, [4, 4]) == (5, 4), 'unaligned fixture')
    rejected('apply aligned formula to unaligned COW',
             lambda: check(ceildiv(3, 4) + 2 * ceildiv(4, 4) == branch_layout(4, 3, [4, 4])[0],
                           'COW peak differs from physical slot objects'))
    print('layout model accepted: ' + str(count) + ' cases; unaligned peak=5,steady=4 pages')

    segments = [(-1, 2, 4), (2, 3, 6), (5, 9, 100)]
    area = sum(Fraction(max(0, min(end, 5) - max(start, 0)) * size)
               for start, end, size in segments)
    discrete = sum(sum(size for start, end, size in segments if start <= t < end) for t in range(5))
    check(area == discrete == 14 and area / 5 == Fraction(14, 5), 'window-clipped residency integral')
    rejected('unweighted mean size times waiting duration',
             lambda: check(Fraction(5 * 3, 5) == area / 5, 'residency arithmetic'))
    check(Fraction(8) / (Fraction(1, 5) + Fraction(1, 20)) == 32, 'closed-loop population relation')
    print('residency model accepted: clipped area=14 MB*s, mean=14/5 MB')
    print('scope: deterministic CPU symbolic models; no model forward, GPU, DMA, network or quality measurement')


def run_closeout_checks():
    # Hand-given outcomes independently constrain both the replay and oracle.
    # The original 16 cases, 1440 combinations and other models remain above.
    eligibility_cases = [
        ('failed final only', {'U': node(bad=('FULL',))}, {},
         {'U': 0}, {}, {'U': (1, 'FAILED')}),
        ('same-time failed required operation before final key',
         {'U': node(events=(('FIRST', 1), ('FULL', 1)), bad=('FIRST',))}, {},
         {'U': 0}, {}, {'U': (1, 'FAILED')}),
        ('same-time failed required operation after final key',
         {'U': node(events=(('TAIL', 1), ('FULL', 1)), bad=('TAIL',))}, {},
         {'U': 0}, {}, {'U': (1, 'FAILED')}),
        ('cancel before failed final', {'U': node(bad=('FULL',))}, {'U': 0},
         {}, {}, {'U': (0, 'CANCELLED')}),
        ('same-time cancel precedes failed final', {'U': node(bad=('FULL',))}, {'U': 1},
         {'U': 0}, {}, {'U': (1, 'CANCELLED')}),
        ('legal FIRST remains before same-time failed FULL',
         {'U': node(events=(('FIRST', 1), ('FULL', 1)), bad=('FULL',))}, {},
         {'U': 0}, {key('U', 'FIRST'): 1}, {'U': (1, 'FAILED')}),
    ]
    for name, spec, cancels, starts, pub, terminals in eligibility_cases:
        result = replay(spec, cancels)
        audit_events(spec, cancels, result)
        states = {v: state for v, (_time, state) in terminals.items()}
        wanted = (starts, pub, states, {}, terminals)
        observed = (result['starts'], result['pub'], result['states'],
                    result['outputs'], result['terminals'])
        check(observed == wanted, 'hand eligibility outcome: ' + name)
        check(reference(spec, cancels) == wanted, 'hand eligibility oracle: ' + name)
        check(result['pending'] == {v: () for v in starts}, 'eligibility drain: ' + name)
        print('closeout accepted: ' + name + '; starts=' + str(starts)
              + '; pub=' + str(pub) + '; terminals=' + str(terminals))

    failed_final = {'U': node(bad=('FULL',))}
    rejected('old oracle unconditionally offers COMMITTED for a failed final',
             lambda: audit_events(failed_final, {}, replay(failed_final, {}),
                                  'old_final_eligibility'))
    failed_tail = {'U': node(events=(('TAIL', 1), ('FULL', 1)), bad=('TAIL',))}
    rejected('old oracle publishes final through the nonfinal path',
             lambda: audit_events(failed_tail, {}, replay(failed_tail, {}),
                                  'final_as_nonfinal'))

    spec = {'U': node(lag=0), 'V': node(a=2, deps=(('U', 'FULL', 0),))}
    cancels = {'U': 2, 'V': 2}
    wanted = ({'U': 0}, {key('U', 'FULL'): 1},
              {'U': 'COMMITTED', 'V': 'CANCELLED'}, {'U': ('U', 'FULL', 1)},
              {'U': (1, 'COMMITTED'), 'V': (2, 'CANCELLED')})
    for label, selected_spec, selected_cancels in [
            ('stable input order', spec, cancels),
            ('reversed insertion order', dict(reversed(list(spec.items()))),
             dict(reversed(list(cancels.items()))))]:
        result = replay(selected_spec, selected_cancels)
        audit_events(selected_spec, selected_cancels, result)
        check((result['starts'], result['pub'], result['states'], result['outputs'],
               result['terminals']) == wanted, 'cancel admission hand outcome: ' + label)
        check(reference(selected_spec, selected_cancels) == wanted,
              'cancel admission hand oracle: ' + label)
        check(result['pending'] == {'U': ()}, 'cancelled unstarted V acquired obligations')
        cancel_rows = [(t, args[0]) for t, action, *args in result['trace'] if action == 'CANCEL']
        check(cancel_rows == [(2, 'U'), (2, 'V')], 'same-time stable cancellation identity order')
        print('closeout accepted: same-time cancellation admission; ' + label
              + '; starts=' + str(result['starts']) + '; terminals=' + str(result['terminals']))
    old_dispatch = replay(spec, cancels, 'dispatch_between_cancels')
    check(old_dispatch['starts'] == {'U': 0, 'V': 2},
          'dispatch fault did not exercise the old extra start')
    check(old_dispatch['states'] == wanted[2] and old_dispatch['terminals'] == wanted[4],
          'dispatch fixture must distinguish equal terminals with different starts')
    check(any(t == 2 and ('V', False, ('FULL',), True) in args[0]
              for t, action, *args in old_dispatch['trace'] if action == 'RESOURCE'),
          'dispatch fault did not create cancelled V physical obligations')
    rejected('dispatch between same-time cancellations violates the selected contract',
             lambda: audit_events(spec, cancels, old_dispatch))
    print('closeout scope: 6 eligibility and 2 cancellation-order regressions;'
          ' 3 targeted negative controls; CPU symbolic evidence only')


def run_final_candidate_checks():
    success_cases = [
        ('TAIL/FULL at the same time', node(events=(('TAIL', 1), ('FULL', 1))),
         {key('U', 'TAIL'): 1, key('U', 'FULL'): 1}, 1),
        ('FIRST/FULL at the same time', node(events=(('FIRST', 1), ('FULL', 1))),
         {key('U', 'FIRST'): 1, key('U', 'FULL'): 1}, 1),
        ('auxiliary identity before FULL', node(extra=(('AUX', 1),)),
         {key('U', 'FULL'): 1}, 1),
        ('auxiliary identity after FULL', node(extra=(('ZCOPY', 1),)),
         {key('U', 'FULL'): 1}, 1),
        ('auxiliary finishes after final candidate', node(extra=(('ZCOPY', 3),)),
         {key('U', 'FULL'): 3}, 3),
        ('FIRST remains local while final waits for auxiliary',
         node(events=(('FIRST', 1), ('FULL', 2)), extra=(('AUX', 0), ('ZCOPY', 4))),
         {key('U', 'FIRST'): 1, key('U', 'FULL'): 4}, 4),
    ]
    for label, producer, pubs, end in success_cases:
        spec = {'U': producer}
        result = replay(spec, {})
        audit_events(spec, {}, result)
        wanted = ({'U': 0}, pubs, {'U': 'COMMITTED'}, {'U': ('U', 'FULL', 1)},
                  {'U': (end, 'COMMITTED')})
        check((result['starts'], result['pub'], result['states'], result['outputs'],
               result['terminals']) == wanted, 'hand final-candidate result: ' + label)
        check(reference(spec, {}) == wanted, 'hand final-candidate reference: ' + label)
        check(result['pending'] == {'U': ()} and result['candidates'] == {'U': {}},
              'completed producer retained work or candidate: ' + label)
        final_rows = [i for i, row in enumerate(result['trace'])
                      if row[1] == 'PUBLISH' and row[2] == key('U', 'FULL')]
        completion_rows = [i for i, row in enumerate(result['trace'])
                           if row[1] == 'COMPLETE' and row[2] == 'U']
        check(len(final_rows) == 1 and final_rows[0] > max(completion_rows),
              'final publication did not follow every unique required completion: ' + label)
        print('final-candidate accepted: ' + label + '; pub=' + str(pubs) + '; commit=' + str(end))

    # Full success is checked at the exact identities of the previous failed TAIL case.
    same_time = {'U': node(events=(('TAIL', 1), ('FULL', 1)))}
    late_aux = {'U': node(extra=(('ZCOPY', 3),))}
    rejected('forget saved final when another event completes',
             lambda: audit_events(same_time, {}, replay(same_time, {}, 'final_event_only_recheck')))
    rejected('discard a final candidate whose other required result is pending',
             lambda: audit_events(same_time, {}, replay(same_time, {}, 'discard_unready_final')))
    rejected('auxiliary completion fails to revisit the saved final',
             lambda: audit_events(late_aux, {}, replay(late_aux, {}, 'no_auxiliary_recheck')))
    rejected('reference commits at final operation rather than last required evidence',
             lambda: audit_events(late_aux, {}, replay(late_aux, {}), 'final_completion_only'))
    rejected('same-time reference uses FULL key before the last required TAIL key',
             lambda: audit_events(same_time, {}, replay(same_time, {}), 'final_completion_only'))

    spec = {'U': node(events=(('FIRST', 1), ('FULL', 2)), extra=(('ZCOPY', 4),)),
            'F': node(deps=(('U', 'FIRST', 0),)),
            'V': node(deps=(('U', 'FULL', 0),))}
    for cancelled_at in (3, 4):
        cancels = {'U': cancelled_at}
        result = replay(spec, cancels)
        audit_events(spec, cancels, result)
        wanted = ({'U': 0, 'F': 1}, {key('U', 'FIRST'): 1, key('F', 'FULL'): 2},
                  {'U': 'CANCELLED', 'F': 'COMMITTED', 'V': 'FAILED'},
                  {'F': ('F', 'FULL', 1)},
                  {'U': (cancelled_at, 'CANCELLED'), 'F': (2, 'COMMITTED'),
                   'V': (cancelled_at, 'FAILED')})
        check((result['starts'], result['pub'], result['states'], result['outputs'],
               result['terminals']) == wanted, 'cancel while saved final awaits auxiliary')
        check(reference(spec, cancels) == wanted, 'reference must preserve FIRST and reject pending FULL')
        check(result['candidates'] == {'U': {}, 'F': {}} and result['pending'] == {'U': (), 'F': ()},
              'cancelled final must be abandoned while physical work drains')
        print('final-candidate accepted: cancellation at ' + str(cancelled_at)
              + ' preserves FIRST, rejects final and drains auxiliary')

    # The reference keeps the last required identity, even when wall times tie.
    # A mutant using FULL's own key must be visible in the ordered publication witness.
    result = replay(same_time, {})
    complete_uids = [row[3] for row in result['trace'] if row[1] == 'COMPLETE']
    pub_uids = [row[2][2] for row in result['trace'] if row[1] == 'PUBLISH']
    check(complete_uids == ['FULL', 'TAIL'] and pub_uids == ['TAIL', 'FULL'],
          'the exact TAIL/FULL counterexample order was bypassed')
    print('final-candidate scope: 6 successful common-commit paths, 2 cancellation paths,'
          ' 5 actual negative controls; original event, memory and cost checks retained')


if __name__ == '__main__':
    run_checks()
    run_closeout_checks()
    run_final_candidate_checks()
