#!/usr/bin/env python3
"""K-SYS-005: deterministic CPU contract model; no files, network or GPU."""
from dataclasses import dataclass, replace
from itertools import product
from math import isclose


class ContractError(ValueError):
    pass


def need(condition, reason):
    if not condition:
        raise ContractError(reason)


def rejected(fn):
    try:
        fn()
    except ContractError:
        return True
    return False


@dataclass(frozen=True)
class Spec:
    kind: str = 'allreduce'
    count: int = 2
    dtype: str = 'int32'
    root: object = None
    op: object = 'sum'
    identity: tuple = ('C', 3, 7, 0, 'request-A', 'model-A', 'layout-A')


def role_ranges(s, p, r):
    n = s.count
    if s.kind == 'allreduce':
        return n, n
    if s.kind == 'allgather':
        return n, p * n
    if s.kind == 'reducescatter':
        return p * n, n
    if s.kind == 'reduce':
        return n, n if r == s.root else 0
    if s.kind == 'broadcast':
        return (n if r == s.root else 0), n
    raise ContractError('unknown operation')


def prepare(specs, payloads, capacities, absent_sum=False, zero_supported=False):
    p = len(specs)
    need(p > 0 and len(payloads) == p and len(capacities) == p, 'participants')
    s = specs[0]
    need(all(x == s for x in specs), 'common descriptor or identity mismatch')
    need(isinstance(s.count, int) and s.count >= 0, 'count range')
    need(s.dtype == 'int32', 'model supports exact int32 examples only')
    if s.kind in ('reduce', 'broadcast'):
        need(isinstance(s.root, int) and 0 <= s.root < p, 'root is rank')
    else:
        need(s.root is None, 'unexpected root')
    if s.kind in ('allreduce', 'reduce', 'reducescatter'):
        need(s.op == 'sum', 'only sum is modeled')
    else:
        need(s.op is None, 'unexpected reduction')
    need(s.count != 0 or zero_supported, 'zero API capability not established')
    inputs = []
    for r in range(p):
        ni, no = role_ranges(s, p, r)
        need(capacities[r] >= no, 'output range missing')
        row = payloads[r]
        if row is None:
            if ni == 0:
                row = []
            else:
                need(absent_sum and s.kind in ('allreduce', 'reduce', 'reducescatter'),
                     'missing input has no declared neutral interpretation')
                row = [0] * ni
        need(len(row) == ni, 'input range/count mismatch')
        need(all(isinstance(x, int) and -(2**31) <= x < 2**31 for x in row),
             'input model range')
        inputs.append(list(row))
    return s, inputs


def evaluate(s, inputs):
    p = len(inputs)
    n = s.count
    if s.kind == 'broadcast':
        return [list(inputs[s.root]) for _ in range(p)]
    if s.kind == 'allgather':
        joined = [v for row in inputs for v in row]
        return [list(joined) for _ in range(p)]
    width = p * n if s.kind == 'reducescatter' else n
    reduced = []
    for j in range(width):
        value = 0
        for r in range(p):
            value += inputs[r][j]
        need(-(2**31) <= value < 2**31, 'reference example would overflow')
        reduced.append(value)
    if s.kind == 'allreduce':
        return [list(reduced) for _ in range(p)]
    if s.kind == 'reduce':
        return [list(reduced) if r == s.root else [] for r in range(p)]
    return [reduced[r*n:(r+1)*n] for r in range(p)]


def check_payloads():
    cases = 0
    kinds = ('allreduce', 'allgather', 'reducescatter', 'reduce', 'broadcast')
    for p, n, kind in product((1, 2, 4), (0, 1, 8), kinds):
        root = min(1, p - 1) if kind in ('reduce', 'broadcast') else None
        op = 'sum' if kind in ('allreduce', 'reduce', 'reducescatter') else None
        s = Spec(kind=kind, count=n, root=root, op=op)
        rows, caps = [], []
        for r in range(p):
            ni, no = role_ranges(s, p, r)
            rows.append([100*r+j for j in range(ni)])
            caps.append(no)
        actual = evaluate(*prepare([s]*p, rows, caps, zero_supported=True))
        base = 100*p*(p-1)//2
        if kind == 'allreduce':
            expected = [[base+p*j for j in range(n)] for _ in range(p)]
        elif kind == 'reduce':
            expected = [[base+p*j for j in range(n)] if r == root else []
                        for r in range(p)]
        elif kind == 'reducescatter':
            expected = [[base+p*(r*n+j) for j in range(n)] for r in range(p)]
        elif kind == 'broadcast':
            expected = [[100*root+j for j in range(n)] for _ in range(p)]
        else:
            expected = [[100*r+j for r in range(p) for j in range(n)]
                        for _ in range(p)]
        need(actual == expected, 'closed-form scalar reference disagreement')
        cases += 1
    s = Spec()
    prepared = prepare([s]*3, [[1, 2], None, [7, 8]], [2]*3, absent_sum=True)
    need(evaluate(*prepared) == [[8, 10]]*3, 'neutral rank changes result')
    bad = [s, replace(s, count=0), s]
    need(rejected(lambda: prepare(bad, [[1, 2], [], [7, 8]], [2]*3)),
         'mixed zero/nonzero count accepted')
    need(rejected(lambda: prepare([s]*2, [[1, 2], []], [2]*2)),
         'positive count with no input accepted')
    need(rejected(lambda: prepare([s]*2, [[1, 2], None], [2]*2)),
         'undeclared identity accepted')
    ag = Spec(kind='allgather', op=None)
    need(rejected(lambda: prepare([ag]*2, [[1, 2], None], [4]*2, absent_sum=True)),
         'gather missing block treated as neutral')
    bc = Spec(kind='broadcast', root=0, op=None)
    need(evaluate(*prepare([bc]*2, [[4, 5], None], [2]*2)) == [[4, 5]]*2,
         'broadcast nonroot role')
    need(rejected(lambda: prepare([bc]*2, [None, None], [2]*2, absent_sum=True)),
         'broadcast missing root accepted')
    red = Spec(kind='reduce', root=0)
    need(evaluate(*prepare([red]*2, [[1, 2], [3, 4]], [2, 0])) == [[4, 6], []],
         'reduce role result')
    need(rejected(lambda: prepare([red]*2, [[1, 2], None], [2, 0])),
         'reduce nonroot input omitted')
    need(rejected(lambda: prepare([bc]*2, [[4, 5], None], [2, 0])),
         'broadcast output omitted')
    zero = replace(s, count=0)
    need(rejected(lambda: prepare([zero]*2, [[], []], [0, 0])),
         'mathematical empty implies API support')
    wrong_id = replace(s, identity=('C', 3, 8, 0, 'request-A', 'model-A', 'layout-A'))
    need(rejected(lambda: prepare([s, wrong_id], [[1, 2]]*2, [2]*2)),
         'same-shape wrong sequence accepted')
    need(rejected(lambda: prepare([s, replace(s, dtype='float32')], [[1, 2]]*2, [2]*2)),
         'datatype mismatch accepted')
    print('CPU payload model:', cases, 'closed-form cases; role ranges, neutral input, mixed counts and logical identity checked')


def slot_key(i):
    return ('C', 3, 7, i, 11)


def make_ticket(i, covered, complete=False):
    return dict(key=slot_key(i), stream='comm-stream', recorded=True,
                covered=frozenset(slot_key(j) for j in covered), complete=complete)


def validate_ticket(ticket, i, completion=False):
    need(ticket['recorded'], 'unrecorded event')
    need(ticket['key'] == slot_key(i), 'stale or wrong event identity')
    need(ticket['stream'] == 'comm-stream', 'unrelated stream')
    need(slot_key(i) in ticket['covered'], 'event does not cover communication')
    need(not completion or ticket['complete'], 'device not complete')


def issue_model(group, returns, polls, nslots=2, mutant=False):
    trace = []
    units = [tuple(range(nslots))] if group else [(i,) for i in range(nslots)]
    confirmed = set()
    for uidx, unit in enumerate(units):
        for i in unit:
            trace.append(('call', (i,), 'STAGED' if group else 'DIRECT'))
        ret = returns[uidx]
        trace.append(('return', unit, ret))
        if ret == 'PENDING' and not (mutant and not group):
            resolved = False
            for state in polls.get(uidx, ()):
                trace.append(('poll', unit, state))
                if state == 'PENDING':
                    continue
                if state != 'SUCCESS':
                    return 'RECOVERY_PENDING', trace
                resolved = True
                break
            if not resolved:
                trace.append(('deadline', unit, 'UNKNOWN'))
                return 'RECOVERY_PENDING', trace
        elif ret not in ('SUCCESS', 'PENDING'):
            return 'RECOVERY_PENDING', trace
        confirmed.update(unit)
        for i in unit:
            trace.append(('record', (i,), make_ticket(i, confirmed)))
    return 'ISSUED', trace


def issue_reference(returns, polls):
    for idx, ret in enumerate(returns):
        if ret == 'SUCCESS':
            continue
        if ret != 'PENDING':
            return 'RECOVERY_PENDING'
        first_resolution = next((x for x in polls.get(idx, ()) if x != 'PENDING'), None)
        if first_resolution != 'SUCCESS':
            return 'RECOVERY_PENDING'
    return 'ISSUED'


def audit_issue(result, trace, nslots=2):
    pending = None
    failed = False
    issued, recorded = set(), set()
    for event, unit, value in trace:
        if event == 'call':
            need(pending is None and not failed, 'next call while issue unresolved')
        elif event == 'return':
            need(pending is None and not failed, 'overlapping issue units')
            if value == 'SUCCESS':
                issued.update(unit)
            elif value == 'PENDING':
                pending = unit
            else:
                failed = True
        elif event == 'poll':
            need(pending == unit, 'poll wrong unit')
            if value == 'SUCCESS':
                issued.update(unit)
                pending = None
            elif value != 'PENDING':
                pending = None
                failed = True
        elif event == 'deadline':
            need(pending == unit, 'deadline without pending issue')
            pending = None
            failed = True
        elif event == 'record':
            i = unit[0]
            need(pending is None and not failed and i in issued,
                 'event recorded before issue confirmation')
            validate_ticket(value, i)
            need(value['covered'] <= frozenset(slot_key(j) for j in issued),
                 'coverage claims unissued work')
            need(i not in recorded, 'duplicate record')
            recorded.add(i)
        else:
            raise ContractError('unknown issue event')
    if result == 'ISSUED':
        need(not failed and pending is None, 'false issue success')
        need(issued == recorded == set(range(nslots)), 'missing issue or event')
    else:
        need(result == 'RECOVERY_PENDING' and failed, 'unexplained recovery result')


def check_issue_and_tickets():
    poll_paths = ((), ('PENDING', 'SUCCESS'), ('ERROR',),
                  ('PENDING', 'PENDING'), ('TIMEOUT',))
    cases = 0
    for group in (False, True):
        n_units = 1 if group else 2
        for returns in product(('SUCCESS', 'PENDING', 'ERROR'), repeat=n_units):
            for path in poll_paths:
                polls = {i: path for i in range(n_units)}
                result, trace = issue_model(group, returns, polls)
                need(result == issue_reference(returns, polls), 'issue outcome/reference mismatch')
                audit_issue(result, trace)
                cases += 1
    bad_result, bad_trace = issue_model(False, ('PENDING', 'SUCCESS'), {}, mutant=True)
    need(rejected(lambda: audit_issue(bad_result, bad_trace)),
         'negative control dropped non-group IN_PROGRESS was not detected')
    good_result, good_trace = issue_model(True, ('SUCCESS',), {})
    no_records = [x for x in good_trace if x[0] != 'record']
    need(rejected(lambda: audit_issue(good_result, no_records)), 'missing event production accepted')
    good = make_ticket(0, (0, 1), complete=True)
    validate_ticket(good, 0, completion=True)
    mutations = (
        dict(recorded=False),
        dict(key=('C', 3, 7, 0, 10)),
        dict(key=('C', 2, 7, 0, 11)),
        dict(key=('C', 3, 6, 0, 11)),
        dict(stream='other-stream'),
        dict(covered=frozenset((slot_key(1),))),
        dict(complete=False),
    )
    for mutation in mutations:
        bad = dict(good)
        bad.update(mutation)
        need(rejected(lambda bad=bad: validate_ticket(bad, 0, completion=True)),
             'bad completion ticket accepted')
    print('CPU issue model:', cases, 'group/non-group success, pending, error and timeout paths audited')
    print('Negative controls detected: dropped non-group pending, omitted records, unrecorded/stale/uncovered/wrong-stream/incomplete events')


def validate_closure(members, events):
    members = set(members)
    need(bool(members), 'empty participants')
    live = {r: 0 for r in members}
    readers = {r: 0 for r in members}
    stopped, released, done, ready, published, submitted = (set() for _ in range(6))
    terminal = {}
    decision = 'RUN'
    for r, event in events:
        if event == 'commit':
            need(r is None and decision == 'RUN' and ready == members, 'invalid group commit')
            need(all(live[x] == 0 for x in members), 'commit with in-flight work')
            decision = 'COMMIT'
            continue
        need(r in members, 'event from nonmember')
        need(r not in terminal, 'event after member terminal')
        if event == 'submit':
            need(decision == 'RUN' and r not in stopped and r not in ready,
                 'submission after stop or seal')
            need(r not in submitted, 'duplicate modeled slot')
            submitted.add(r)
            live[r] += 1
        elif event == 'device_done':
            need(live[r] > 0, 'completion without attempted work')
            live[r] = 0
            done.add(r)
        elif event == 'ready':
            need(decision == 'RUN' and r in done and live[r] == 0, 'unsealed READY')
            ready.add(r)
        elif event in ('local_fail', 'timeout'):
            if decision != 'COMMIT':
                decision = 'ABORT'
                ready.clear()
                stopped.add(r)
            # A failure ordered after COMMIT affects recovery of the committed
            # prefix/suffix; it cannot retroactively change this decision.
        elif event == 'stop':
            need(decision == 'ABORT', 'stop acknowledgment without abort')
            stopped.add(r)
        elif event == 'drain':
            need(decision == 'ABORT' and r in stopped and live[r] > 0,
                 'drain lacks stopped worker or outstanding access')
            live[r] = 0
            done.discard(r)
        elif event == 'publish':
            need(decision == 'COMMIT' and r in done and r not in released,
                 'publication without committed completed storage')
            need(r not in published, 'duplicate publication')
            published.add(r)
        elif event == 'consume':
            need(r in published and r not in released, 'consumer lacks live published handle')
            readers[r] += 1
        elif event == 'consumer_done':
            need(readers[r] > 0, 'unmatched consumer completion')
            readers[r] -= 1
        elif event == 'release':
            need(decision in ('ABORT', 'COMMIT'), 'release before decision')
            need(live[r] == 0 and readers[r] == 0, 'release before physical last use')
            need(r not in released, 'duplicate release')
            need((decision == 'ABORT' and r in stopped) or
                 (decision == 'COMMIT' and r in published), 'release lacks ownership settlement')
            released.add(r)
        elif event == 'abort':
            need(decision == 'ABORT' and r in stopped and r in released,
                 'logical abort is not resource closure')
            need(live[r] == 0 and readers[r] == 0, 'unsettled abort')
            terminal[r] = 'ABORTED'
        elif event == 'finish':
            need(decision == 'COMMIT' and r in published and r in released,
                 'unfinished committed delivery')
            terminal[r] = 'FINISHED'
        else:
            raise ContractError('unknown closure event')
    expected = 'ABORTED' if decision == 'ABORT' else 'FINISHED'
    need(decision != 'RUN', 'undecided execution')
    need(set(terminal) == members and all(x == expected for x in terminal.values()),
         'not every participant reached the common terminal')
    need(all(live[r] == readers[r] == 0 for r in members), 'unsettled physical use')
    return decision


def closure_error(members, events):
    try:
        validate_closure(members, events)
    except ContractError as exc:
        return str(exc)
    return None


def old_faulty_closure(events):
    failed, aborted = set(), set()
    for rank, event in events:
        if event == 'local_fail':
            failed.add(rank)
        elif event == 'abort':
            aborted.add(rank)
        elif event == 'publish' and failed:
            return 'publish after failure'
        # Intentionally faulty: timeout and absent participants are ignored.
    return None if failed <= aborted else 'failed rank not closed'


def normal_trace(p):
    events = []
    for r in range(p):
        events.extend(((r, 'submit'), (r, 'device_done'), (r, 'ready')))
    events.append((None, 'commit'))
    for r in range(p):
        events.extend(((r, 'publish'), (r, 'consume'), (r, 'consumer_done'),
                       (r, 'release'), (r, 'finish')))
    return events


def check_closure():
    for p in (1, 2, 4):
        need(validate_closure(range(p), normal_trace(p)) == 'COMMIT', 'normal closure')
    good_abort = [(0, 'submit'), (1, 'submit'), (0, 'timeout'), (1, 'stop'),
                  (0, 'drain'), (1, 'drain'), (0, 'release'), (0, 'abort'),
                  (1, 'release'), (1, 'abort')]
    need(validate_closure((0, 1), good_abort) == 'ABORT', 'drained abort rejected')
    reviewer_traces = (
        [(0, 'local_fail'), (0, 'abort'), (1, 'timeout')],
        [(0, 'timeout'), (1, 'publish')],
    )
    for events in reviewer_traces:
        need(old_faulty_closure(events) is None, 'negative control no longer reproduces old omission')
        need(closure_error((0, 1), events) is not None, 'reviewer leak trace accepted')
    missing_member = [(0, 'local_fail'), (0, 'release'), (0, 'abort'), (1, 'timeout')]
    need(closure_error((0, 1), missing_member) is not None, 'missing participant terminal accepted')
    live_abort = [(0, 'submit'), (0, 'timeout'), (0, 'release'), (0, 'abort')]
    need(closure_error((0,), live_abort) is not None, 'abort falsely drains device')
    early_release = normal_trace(1)
    early_release.remove((0, 'consumer_done'))
    need(closure_error((0,), early_release) is not None, 'consumer last use ignored')
    after_fail_commit = [(0, 'submit'), (0, 'device_done'), (0, 'ready'),
                         (0, 'timeout'), (None, 'commit')]
    need(closure_error((0,), after_fail_commit) is not None, 'UNKNOWN followed by commit')
    after_commit_fail = normal_trace(1)
    after_commit_fail.insert(4, (0, 'local_fail'))
    need(validate_closure((0,), after_commit_fail) == 'COMMIT', 'committed prefix retroactively aborted')
    survivor_submit = [(0, 'timeout'), (1, 'submit')]
    need(closure_error((0, 1), survivor_submit) is not None, 'submission after abort decision')
    print('CPU closure model: explicit participants, common decision, all terminals, in-flight access and consumer last use checked')
    print('Both retained reviewer leak traces rejected; old timeout-ignoring negative control accepts both')
    print('Commit boundary checked: precommit UNKNOWN prevents commit; a later failure preserves the committed prefix')


def ring_closed(p, size, alpha, beta):
    need(p >= 1 and size >= 0 and alpha >= 0 and beta > 0, 'invalid cost parameters')
    return 2*(p-1)*alpha + 2*(p-1)*size/(p*beta)


def ring_round_reference(p, size, alpha, beta):
    total = 0.0
    for phase in ('reduce-scatter', 'all-gather'):
        for step in range(p - 1):
            total += alpha + (size / p) / beta
    return total


def check_costs():
    alpha, beta = 5e-6, 40 * 1024**3
    for p, size in product((1, 2, 4, 8), (0, 8192, 16*1024**2)):
        need(isclose(ring_closed(p, size, alpha, beta),
                     ring_round_reference(p, size, alpha, beta), rel_tol=1e-12, abs_tol=1e-18),
             'ring cost disagrees with round enumeration')
    example = ring_closed(4, 16*1024**2, alpha, beta)
    need(isclose(example*1e6, 615.9375, rel_tol=1e-12), 'unit conversion')
    faulty_one_phase = 3*alpha + 3*(16*1024**2)/(4*beta)
    need(not isclose(faulty_one_phase, example, rel_tol=1e-12), 'one-phase negative control undetected')
    ag = Spec(kind='allgather', count=8, op=None)
    bc = Spec(kind='broadcast', count=8, root=1, op=None)
    need([sum(role_ranges(ag, 4, r))*4 for r in range(4)] == [160]*4,
         'gather input plus output capacity')
    need([sum(role_ranges(bc, 4, r))*4 for r in range(4)] == [32, 64, 32, 32],
         'separate broadcast root source and output capacity')
    need(4*8*4 != sum(role_ranges(ag, 4, 0))*4, 'missing input capacity negative control')
    need(sum((3, 5, 7)) == 15 and max((3, 5, 7)) == 7, 'concurrent versus sequential live ranges')
    print('Ideal cost model: enumerated ring rounds agree; 4 ranks, 16 MiB = %.4f us' % (example*1e6))
    print('Cost negative controls detected: omitted ring phase and missing AllGather input capacity')


def main():
    check_payloads()
    check_issue_and_tickets()
    check_closure()
    check_costs()
    print('Scope: finite CPU protocol and integer/cost models only; drain/device_done are supplied proof events')
    print('Not measured: GPU, NCCL, CUDA visibility, physical fault cleanup, floating point, network, overlap, performance or agent benefit')


if __name__ == '__main__':
    main()
