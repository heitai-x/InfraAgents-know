#!/usr/bin/env python3
"""Deterministic offline scalar models for K-TOOLS-003; no device measurements."""
from dataclasses import dataclass, replace
from fractions import Fraction as F
from itertools import product


class ContractError(ValueError):
    pass


NEGATIVE_CONTROLS = []


def require(condition, message):
    if not condition:
        raise ContractError(message)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def rejected(label, action):
    try:
        action()
    except ContractError:
        NEGATIVE_CONTROLS.append(label)
    else:
        raise AssertionError('fault was accepted: ' + label)


def differs(label, faulty, reference):
    check(faulty != reference, 'fault was not detected: ' + label)
    NEGATIVE_CONTROLS.append(label)


def perf_count(value, wall, enabled, running, representative=True, already_scaled=False):
    value, wall, enabled, running = map(F, (value, wall, enabled, running))
    require(value >= 0, 'negative count')
    # This fixture represents one event, not a sum across CPUs or inherited events.
    require(0 < running <= enabled <= wall, 'invalid single-event time metadata')
    require(representative, 'running fragments are not representative')
    return value if already_scaled else value * enabled / running


def test_perf():
    # Independent timeline oracle: each tuple is (duration, rate, enabled, running).
    stationary = [(100, 1, True, True), (300, 1, True, False),
                  (600, 0, False, False)]
    wall = sum(dt for dt, rate, en, run in stationary)
    enabled = sum(dt for dt, rate, en, run in stationary if en)
    running = sum(dt for dt, rate, en, run in stationary if run)
    raw = sum(dt * rate for dt, rate, en, run in stationary if run)
    enabled_truth = sum(dt * rate for dt, rate, en, run in stationary if en)
    check(perf_count(raw, wall, enabled, running) == enabled_truth == 400,
          'scaling target is enabled interval')
    check(perf_count(250, 1000, 1000, 250) == 1000, 'running differs from enabled')
    check(perf_count(1000, 1000, 1000, 250, already_scaled=True) == 1000,
          'scaled output must not be scaled twice')
    differs('using enabled/wall instead of running/enabled',
            F(250) / F(1000, 1000), F(1000))
    differs('wall extrapolation changes target', F(raw) * wall / running, F(enabled_truth))
    differs('double scaling', F(1000) * 1000 / 250, F(1000))
    rejected('zero running', lambda: perf_count(0, 1000, 1000, 0))
    rejected('running exceeds enabled', lambda: perf_count(10, 1000, 100, 200))
    rejected('zero enabled', lambda: perf_count(0, 1000, 0, 0))
    # A burst is fully counted, but the rest of the enabled window has rate zero.
    burst = [(250, 1, True, True), (750, 0, True, False)]
    burst_truth = sum(dt * rate for dt, rate, en, run in burst if en)
    blind_scaled = perf_count(250, 1000, 1000, 250)
    differs('blind scaling of a nonstationary burst', blind_scaled, F(burst_truth))
    rejected('known nonrepresentative fragments',
             lambda: perf_count(250, 1000, 1000, 250, representative=False))
    print('perf: enabled truth=400; raw 250 scaled to 1000; burst truth=250, blind estimate=1000')


X = ('app-role', 'gpu0', 'ctx0', 'stream0', 7, 'kernel', (8, 1, 1), (128, 1, 1), 'shape-A', 'decode')
STATE = ('input-A', 'cold-cache', 'fixed-clock', 'isolated')
SCOPE = ('kernel', 'gpu0', 'read-sectors')


@dataclass(frozen=True)
class Sample:
    value: int
    physical_pass: int
    event: str
    instance: tuple = X
    scope: tuple = SCOPE
    state: tuple = STATE
    unit: str = 'sector'


@dataclass(frozen=True)
class Anchor:
    duration: int
    work: int = 100
    input_id: str = 'input-A'
    output_id: str = 'output-A'


def spread(values):
    values = tuple(map(F, values))
    require(bool(values) and min(values) >= 0, 'invalid stability values')
    return F(0) if max(values) == 0 else (max(values) - min(values)) / max(values)


def hit_ratio(hits, queries, anchors=None, controlled=False, epsilon=F(1, 20), target=X):
    require(bool(hits) and bool(queries), 'empty event observations')
    require(0 <= epsilon < 1, 'invalid diagnostic tolerance')
    samples = tuple(hits) + tuple(queries)
    first = samples[0]
    for sample in samples:
        require(sample.instance == target, 'wrong logical launch')
        require(sample.scope == first.scope, 'scope mismatch')
        require(sample.state == first.state, 'state mismatch')
        require(sample.unit == first.unit == 'sector', 'unit mismatch')
        require(sample.value >= 0, 'negative count')
    require(all(s.event == 'hits' for s in hits), 'numerator event mismatch')
    require(all(s.event == 'queries' and s.value > 0 for s in queries), 'invalid denominator')
    if len(hits) == len(queries) == 1 and hits[0].physical_pass == queries[0].physical_pass:
        point = F(hits[0].value, queries[0].value)
        require(point <= 1, 'direct hit rate out of range; retain raw samples')
        return 'direct', point, (point, point)
    # This scalar checker chooses repeated observations to form an empirical envelope.
    # Its 5% default is a fixture policy, not a hardware or chapter acceptance threshold.
    require(controlled and anchors is not None, 'missing replay evidence')
    require(len(hits) >= 2 and len(queries) >= 2, 'no repeated event observations')
    require(len({s.physical_pass for s in hits}) == len(hits), 'duplicate hit observation')
    require(len({s.physical_pass for s in queries}) == len(queries), 'duplicate query observation')
    pass_ids = {s.physical_pass for s in samples}
    require(pass_ids <= set(anchors), 'missing per-pass anchor')
    selected = [anchors[p] for p in sorted(pass_ids)]
    require(all(a.duration > 0 and a.work > 0 for a in selected), 'invalid anchor')
    require(len({(a.work, a.input_id, a.output_id) for a in selected}) == 1,
            'work or input/output identity changed')
    require(spread(a.duration for a in selected) <= epsilon, 'nonstationary duration')
    require(spread(s.value for s in hits) <= epsilon, 'nonstationary hits')
    require(spread(s.value for s in queries) <= epsilon, 'nonstationary queries')
    mean_h = sum(F(s.value) for s in hits) / len(hits)
    mean_q = sum(F(s.value) for s in queries) / len(queries)
    point = mean_h / mean_q
    low = F(min(s.value for s in hits), max(s.value for s in queries))
    high = F(max(s.value for s in hits), min(s.value for s in queries))
    require(0 <= low <= point <= high <= 1, 'incompatible hit/query envelope')
    return 'cross-pass-estimate', point, (low, high)


def test_ratios():
    direct = hit_ratio([Sample(80, 10, 'hits')], [Sample(100, 10, 'queries')])
    check(direct == ('direct', F(4, 5), (F(4, 5), F(4, 5))), 'direct ratio')
    hits = [Sample(80, 0, 'hits'), Sample(82, 2, 'hits')]
    queries = [Sample(100, 1, 'queries'), Sample(102, 3, 'queries')]
    anchors = {p: Anchor(d) for p, d in enumerate((1000, 1001, 1002, 1000))}
    result = hit_ratio(hits, queries, anchors, controlled=True)
    # Independent Cartesian enumeration is the reference for the envelope.
    reference = [F(h.value, q.value) for h, q in product(hits, queries)]
    check(result == ('cross-pass-estimate', F(81, 101), (min(reference), max(reference))),
          'stable matched cross-pass estimates must be accepted')
    check(result[2] == (F(40, 51), F(41, 50)), 'explicit empirical envelope')
    bad_x = X[:4] + (8,) + X[5:]
    wrong_launch = [replace(queries[0], instance=bad_x), queries[1]]
    # A blind merger still produces a plausible number; metadata validation rejects it.
    check(F(hits[0].value, wrong_launch[0].value) == F(4, 5), 'plausible bad merge fixture')
    rejected('wrong launch with same name/grid',
             lambda: hit_ratio(hits, wrong_launch, anchors, True))
    rejected('wrong scope', lambda: hit_ratio(hits, [replace(q, scope=('system',)) for q in queries], anchors, True))
    rejected('wrong state', lambda: hit_ratio(hits, [replace(q, state=('hot-cache',)) for q in queries], anchors, True))
    rejected('wrong units', lambda: hit_ratio(hits, [replace(q, unit='request') for q in queries], anchors, True))
    rejected('no stability evidence', lambda: hit_ratio(hits, queries))
    rejected('missing pass anchor', lambda: hit_ratio(hits, queries, {0: Anchor(1000)}, True))
    rejected('nonstationary event counts',
             lambda: hit_ratio([hits[0], replace(hits[1], value=120)], queries, anchors, True))
    bad_anchors = dict(anchors)
    bad_anchors[2] = Anchor(1500)
    rejected('nonstationary duration', lambda: hit_ratio(hits, queries, bad_anchors, True))
    bad_anchors = dict(anchors)
    bad_anchors[2] = Anchor(1000, work=101)
    rejected('changed completed work', lambda: hit_ratio(hits, queries, bad_anchors, True))
    rejected('zero queries', lambda: hit_ratio([Sample(0, 0, 'hits')], [Sample(0, 0, 'queries')]))
    rejected('out-of-range direct hits', lambda: hit_ratio([Sample(101, 0, 'hits')], [Sample(100, 0, 'queries')]))
    zero = hit_ratio([replace(h, value=0) for h in hits], queries, anchors, True)
    check(zero[1:] == (F(0), (F(0), F(0))), 'zero hits are legal')
    print('ratios: stable cross-pass estimate=81/101, empirical envelope=[40/51,41/50]; invalid joins rejected')


MODES = ('kernel', 'application', 'range', 'application-range')
BASE = {'x': 3, 'y': 10, 'z': 0, 'h': -1}


def replay(mode, passes, fault=None, synchronizable=True, supported_api=True,
           host_addresses_valid=True, current_context=True):
    require(mode in MODES and isinstance(passes, int) and passes >= 0, 'invalid replay plan')
    if passes == 0:
        return (), ()
    if mode == 'range':
        require(synchronizable, 'cannot synchronize range contexts')
        require(supported_api, 'unsupported captured API')
        require(host_addresses_valid, 'invalid restored host address')
    if mode == 'application-range':
        require(current_context, 'range context is not current/active')
    mem = dict(BASE)
    original = None
    previous = None
    writes = ()
    history, trace = [], []
    for p in range(passes):
        if mode in ('application', 'application-range'):
            if p == 0 or fault != 'app-no-restart':
                trace.extend([('application-start', p), ('host-setup', p)])
                mem = dict(BASE)
        elif mode == 'kernel':
            if p == 0 and passes > 1:
                original = dict(mem)
                trace.append(('save-accessible', p))
            elif p > 0 and fault != 'no-restore':
                source = previous if fault == 'rolling-backup' else original
                for key in writes:
                    mem[key] = source[key]
                trace.append(('restore-original-write-set', p))
        else:
            if p == 0:
                trace.extend([('synchronize-contexts', p), ('capture-range', p)])
                original = dict(mem)
            else:
                keys = ('y',) if fault == 'kernel-only' else ('y', 'z', 'h')
                for key in keys:
                    mem[key] = original[key]
                trace.append(('restore-range', p))
                trace.append(('replay-captured-range', p))
        # Abstract state effects only; this is not real concurrent kernel scheduling.
        mem['y'] += mem['x']
        trace.append(('execute-kernel', p))
        if mode in ('range', 'application-range'):
            mem['z'] += 2 * mem['y']
            mem['h'] += mem['z']
            trace.extend([('execute-second-kernel', p), ('copy-to-live-host-object', p)])
        if mode == 'kernel' and p == 0 and passes > 1:
            writes = ('y',)
            trace.append(('discover-write-set', p))
        if mode in ('application', 'application-range'):
            trace.append(('application-suffix-complete', p))
        history.append(tuple(mem[k] for k in ('x', 'y', 'z', 'h')))
        previous = dict(mem)
    return tuple(history), tuple(trace)


def test_replay():
    # Pure one-shot references do not use replay(), its snapshot or its execution loop.
    kernel_ref = (BASE['x'], BASE['y'] + BASE['x'], BASE['z'], BASE['h'])
    range_ref = (BASE['x'], BASE['y'] + BASE['x'],
                 BASE['z'] + 2 * (BASE['y'] + BASE['x']),
                 BASE['h'] + BASE['z'] + 2 * (BASE['y'] + BASE['x']))
    for mode in MODES:
        ref = range_ref if mode in ('range', 'application-range') else kernel_ref
        good, trace = replay(mode, 4)
        check(good == (ref,) * 4, 'one-shot reference for ' + mode)
        check(replay(mode, 0) == ((), ()), 'empty plan for ' + mode)
        if mode.startswith('application'):
            check(sum(name == 'application-start' for name, p in trace) == 4,
                  'each application pass restarts the complete application')
            check(not any('restore' in name or 'capture' in name for name, p in trace),
                  'application modes must not perform tool state restoration')
            differs(mode + ' only repeats target without restarting application',
                    replay(mode, 4, fault='app-no-restart')[0], good)
    good, trace = replay('kernel', 4)
    check(sum(name == 'save-accessible' for name, p in trace) == 1, 'one original backup')
    check(sum(name == 'restore-original-write-set' for name, p in trace) == 3, 'restore before later passes')
    check(not any('save' in name or 'restore' in name for name, p in replay('kernel', 1)[1]),
          'single-pass kernel saves/restores zero bytes')
    differs('kernel omitted restore', replay('kernel', 4, fault='no-restore')[0], good)
    differs('kernel restored previous output instead of original',
            replay('kernel', 4, fault='rolling-backup')[0], good)
    range_good = replay('range', 4)[0]
    differs('range restored only one kernel write set',
            replay('range', 4, fault='kernel-only')[0], range_good)
    rejected('range unsynchronizable context', lambda: replay('range', 2, synchronizable=False))
    rejected('range unsupported API', lambda: replay('range', 2, supported_api=False))
    rejected('range invalid host address', lambda: replay('range', 2, host_addresses_valid=False))
    rejected('application-range invalid context', lambda: replay('application-range', 2, current_context=False))
    check(replay('application-range', 2, supported_api=False)[0] == (range_ref,) * 2,
          'application-range does not require explicit API capture support')
    rejected('negative passes', lambda: replay('kernel', -1))
    print('replay: four mode state traces match independent one-shot references; range output=', range_ref)


def kernel_serial(executions, setup, auxiliaries, save_bytes, write_bytes, save_rate, restore_rate):
    require(len(executions) == len(auxiliaries), 'missing per-pass auxiliary cost')
    require(all(v >= 0 for v in tuple(executions) + tuple(auxiliaries)), 'negative duration')
    require(setup >= 0 and save_bytes >= 0 and write_bytes >= 0, 'negative setup/bytes')
    require(save_rate > 0 and restore_rate > 0, 'invalid path rate')
    p = len(executions)
    if p == 0:
        return F(0)
    state = F(0) if p == 1 else F(save_bytes, save_rate) + (p - 1) * F(write_bytes, restore_rate)
    return F(setup) + sum(map(F, executions)) + sum(map(F, auxiliaries)) + state


def dag_finish(nodes):
    finish = {}
    for name, duration, predecessors in nodes:
        require(duration >= 0 and all(p in finish for p in predecessors), 'invalid topological DAG')
        finish[name] = F(duration) + max((finish[p] for p in predecessors), default=F(0))
    return max(finish.values(), default=F(0))


def test_cost():
    budget = kernel_serial((3, 4, 5), 2, (1, 1, 1), 100, 20, 10, 10)
    # Independent chronological ledger: setup, snapshot, then auxiliary/execution and restores.
    ledger = (2, 10, 1, 3, 2, 1, 4, 2, 1, 5)
    check(budget == sum(ledger) == 31, 'conditional serial budget')
    single = kernel_serial((3,), 2, (1,), 100, 20, 10, 10)
    check(single == 6, 'single pass has zero snapshot/restore contribution')
    differs('unnecessary single-pass snapshot cost', single + F(100, 10), single)
    check(kernel_serial((), 2, (), 100, 20, 10, 10) == 0, 'no acquisition has no budget')
    nodes = [('host-prepare', 4, ()), ('independent-copy', 6, ()),
             ('execute', 2, ('host-prepare', 'independent-copy'))]
    bound = dag_finish(nodes)
    # Independent feasible schedule on distinct resources.
    schedule = {'host-prepare': (0, 4), 'independent-copy': (0, 6), 'execute': (6, 8)}
    for name, duration, predecessors in nodes:
        begin, end = schedule[name]
        check(end - begin == duration, 'scheduled duration')
        check(all(schedule[p][1] <= begin for p in predecessors), 'scheduled dependency')
    makespan = max(end for begin, end in schedule.values())
    check(bound == makespan == 8, 'DAG and independent feasible schedule')
    serial_sum = sum(duration for name, duration, predecessors in nodes)
    differs('serial sum asserted as a necessary lower bound', serial_sum, makespan)
    check(makespan < serial_sum == 12, 'legal overlap disproves the claimed sum lower bound')
    rejected('invalid restore rate', lambda: kernel_serial((1, 1), 0, (0, 0), 1, 1, 1, 0))
    print('cost: serial fixture=31; single pass=6; legal overlap makespan=8 versus serial sum=12')


def segments(offset, stride, lanes):
    require(offset >= 0 and stride > 0 and lanes >= 0, 'invalid address fixture')
    result = set()
    for lane in range(lanes):
        address = offset + 4 * stride * lane
        result.add(address // 32)
        result.add((address + 3) // 32)
    return result


def byte_reference(offset, stride, lanes):
    touched = set()
    for lane in range(lanes):
        for byte in range(4):
            touched.add(offset + 4 * stride * lane + byte)
    return {byte // 32 for byte in touched}


def test_traffic():
    for offset, stride, lanes in product(range(64), (1, 2, 3, 8, 17), (0, 1, 31, 32)):
        check(segments(offset, stride, lanes) == byte_reference(offset, stride, lanes),
              'segment model differs from byte enumeration')
    aligned, shifted = segments(0, 1, 32), segments(4, 1, 32)
    next_warp = segments(132, 1, 32)
    check(len(aligned) == 4 and len(shifted) == 5, 'aligned and offset segment counts')
    check(len(segments(0, 2, 32)) == 8, 'stride-two segment count')
    amplification, efficiency = F(160, 128), F(128, 160)
    check(amplification == F(5, 4) and efficiency == F(4, 5), 'byte ratios')
    differs('efficiency mislabeled as amplification', efficiency, amplification)
    request_bytes = 32 * (len(shifted) + len(next_warp))
    cold_fill_bytes = 32 * len(shifted | next_warp)
    check(request_bytes == 320 and cold_fill_bytes == 288, 'specified cache union boundary')
    differs('request segments treated as actual cold fills', request_bytes, cold_fill_bytes)
    # 9/10 is only a supplied relative-throughput observation, not this byte model's efficiency.
    differs('relative throughput substituted for byte efficiency', F(9, 10), efficiency)
    print('traffic: aligned=4 segments, shifted=5, stride2=8; A=5/4; two-warp request=320 B, cold union=288 B')


def test_occupancy():
    warp, quantum, threads, registers_per_thread = 32, 256, 128, 37
    warps = (threads + warp - 1) // warp
    allocated_per_warp = ((registers_per_thread * warp + quantum - 1) // quantum) * quantum
    registers_per_block = warps * allocated_per_warp
    formula = min(65536 // registers_per_block, 102400 // 16384,
                  2048 // threads, 64 // warps, 16)
    # Independent allocation oracle counts fixed-size chunks and accumulates each resource.
    chunks_per_warp = len(range(0, registers_per_thread * warp, quantum))
    feasible = []
    for blocks in range(17):
        register_use = sum(chunks_per_warp * quantum for _ in range(blocks * warps))
        shared_use = sum(16384 for _ in range(blocks))
        thread_use = sum(threads for _ in range(blocks))
        warp_use = sum(warps for _ in range(blocks))
        if register_use <= 65536 and shared_use <= 102400 and thread_use <= 2048 and warp_use <= 64:
            feasible.append(blocks)
    check(registers_per_block == 5120 and max(feasible) == formula == 6, 'resource occupancy upper bound')
    ignoring_shared = min(65536 // registers_per_block, 2048 // threads, 64 // warps, 16)
    differs('occupancy ignored shared allocation', ignoring_shared, formula)
    print('occupancy: modeled registers/block=5120; feasible bound=6; faulty ignoring shared=12')


def main():
    print('scope: deterministic CPU scalar arithmetic/state models; no GPU, PMU, actual cache or service measurements')
    test_perf()
    test_ratios()
    test_replay()
    test_cost()
    test_traffic()
    test_occupancy()
    print('negative controls detected:', len(NEGATIVE_CONTROLS))
    for label in NEGATIVE_CONTROLS:
        print('detected:', label)
    print('completed scoped model checks; hardware behavior and performance remain unmeasured')


if __name__ == '__main__':
    main()
