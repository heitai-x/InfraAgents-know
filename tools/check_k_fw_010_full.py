#!/usr/bin/env python3
'''Standalone deterministic CPU checks. Stdlib only; no files, processes or network.'''

import math
import sys
from decimal import Decimal, localcontext

TOL = 2e-12


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def near(actual, expected, label):
    if actual == expected:
        return
    require(math.isfinite(actual) and math.isfinite(expected), label + ': nonfinite mismatch')
    require(abs(actual - expected) <= TOL * max(1.0, abs(expected)),
            label + ': got {!r}, expected {!r}'.format(actual, expected))


def empty_state(dv):
    return (-math.inf, 0.0, [0.0] * dv)


def copy_state(state):
    return (state[0], state[1], state[2][:])


def merge(old, local, fault=None):
    m1, l1, u1 = old
    m2, l2, u2 = local
    if l1 == 0.0:
        return copy_state(local)
    if l2 == 0.0:
        return copy_state(old)
    mn = max(m1, m2)
    lc = uc = math.exp(m1 - mn)
    if fault == 'omit_rescale':
        lc = uc = 1.0
    elif fault == 'reverse_u_only':
        uc = math.exp(mn - m1)
    elif fault == 'reverse_both':
        lc = uc = math.exp(mn - m1)
    r2 = math.exp(m2 - mn)
    return (mn, lc * l1 + r2 * l2,
            [uc * x + r2 * y for x, y in zip(u1, u2)])


def tiled_attention(Q, K, V, visible, bias, br, bc, scale, dv, fault=None):
    require(br > 0 and bc > 0 and dv > 0, 'positive tile sizes required')
    L, S = len(Q), len(K)
    outputs = [[0.0] * dv for _ in Q]
    flags = [True] * L
    lses = [-math.inf] * L
    traces = [[] for _ in Q]
    for q0 in range(0, L, br):
        for r in range(q0, min(q0 + br, L)):
            state = empty_state(dv)
            previous = None
            for k0 in range(0, S, bc):
                stop = min(k0 + bc, S)
                ids = [j for j in range(k0, stop) if visible[r][j]]
                local = empty_state(dv)
                if ids:
                    scores = [scale * math.fsum(x * y for x, y in zip(Q[r], K[j]))
                              + bias[r][j] for j in ids]
                    mb = max(scores)
                    weights = [math.exp(z - mb) for z in scores]
                    local = (mb, math.fsum(weights),
                             [math.fsum(weights[t] * V[j][c]
                                        for t, j in enumerate(ids)) for c in range(dv)])
                    previous = copy_state(local)
                elif fault == 'empty_reuse' and previous is not None:
                    local = copy_state(previous)
                state = merge(state, local, fault)
                traces[r].append((stop, copy_state(state)))
            m, mass, numerator = state
            if mass != 0.0:
                outputs[r] = [x / mass for x in numerator]
                flags[r] = False
                lses[r] = m + math.log(mass)
    return outputs, flags, lses, traces


def row_reference(Q, K, V, visible, bias, scale, dv, r, stop):
    # Whole-prefix evaluation; does not call merge or the tiled block computation.
    ids = [j for j in range(stop) if visible[r][j]]
    if not ids:
        return (-math.inf, 0.0, [0.0] * dv, [0.0] * dv, True, -math.inf)
    z = []
    for j in ids:
        products = [Q[r][c] * K[j][c] for c in range(len(Q[r]))]
        z.append(scale * math.fsum(products) + bias[r][j])
    maximum = max(z)
    weights = [math.exp(value - maximum) for value in z]
    mass = math.fsum(weights)
    numerator = [math.fsum(weights[t] * V[j][c] for t, j in enumerate(ids))
                 for c in range(dv)]
    output = [math.fsum((weights[t] / mass) * V[j][c]
                        for t, j in enumerate(ids)) for c in range(dv)]
    return maximum, mass, numerator, output, False, maximum + math.log(mass)


def check_attention():
    L, S, d, dv = 5, 7, 3, 2
    Q = [[((r + 1) * (c + 2) % 7 - 3) / 3.0 for c in range(d)] for r in range(L)]
    K = [[((j + 2) * (c + 1) % 5 - 2) / 2.0 for c in range(d)] for j in range(S)]
    V = [[(j % 3 - 1) / 2.0, (j % 5 - 2) / 3.0] for j in range(S)]
    visible = [[j % 2 == 0 for j in range(S)], [False] * S, [True] * S,
               [j == 6 for j in range(S)], [j == 0 for j in range(S)]]
    bias = [[(r - 2 * j) / 11.0 for j in range(S)] for r in range(L)]
    max_error = 0.0
    prefix_checks = 0
    for br in (1, 2, 8):
        for bc in (1, 3, 10):
            out, flags, lses, traces = tiled_attention(Q, K, V, visible, bias, br, bc, 0.7, dv)
            for r in range(L):
                reference = row_reference(Q, K, V, visible, bias, 0.7, dv, r, S)
                require(flags[r] == reference[4], 'empty flag mismatch')
                require(len(out[r]) == dv, 'output width mismatch')
                for c in range(dv):
                    near(out[r][c], reference[3][c], 'attention output')
                    max_error = max(max_error, abs(out[r][c] - reference[3][c]))
                near(lses[r], reference[5], 'LSE')
                for stop, state in traces[r]:
                    ref = row_reference(Q, K, V, visible, bias, 0.7, dv, r, stop)
                    near(state[0], ref[0], 'prefix maximum')
                    near(state[1], ref[1], 'prefix mass')
                    for c in range(dv):
                        near(state[2][c], ref[2][c], 'prefix numerator')
                    prefix_checks += 1
            if bc == 1:
                require(all(item[1] == traces[4][0][1] for item in traces[4]),
                        'trailing empty blocks changed state')

    out0, flags0, lse0, trace0 = tiled_attention([[1.0], [2.0]], [], [],
                                               [[], []], [[], []], 1, 2, 1.0, 2)
    require(out0 == [[0.0, 0.0], [0.0, 0.0]] and flags0 == [True, True], 'S=0 output')
    require(lse0 == [-math.inf, -math.inf] and trace0 == [[], []], 'S=0 created work')
    require(tiled_attention([], [], [], [], [], 1, 1, 1.0, 1) == ([], [], [], []),
            'L=0 output')
    extreme = tiled_attention([[1.0]], [[-1000.0], [1000.0]], [[0.25], [1.0]],
                              [[True, True]], [[0.0, 0.0]], 1, 1, 1.0, 1)
    near(extreme[0][0][0], 1.0, 'extreme finite scores')

    args = ([[0.0]], [[0.0], [0.0], [0.0]], [[1.0], [9.0], [2.0]],
            [[True, False, True]], [[0.0, 0.0, 0.0]], 1, 1, 1.0, 1)
    good = tiled_attention(*args)
    bad = tiled_attention(*args, fault='empty_reuse')
    near(good[0][0][0], 1.5, 'empty-block independent result')
    near(bad[0][0][0], 4.0 / 3.0, 'empty-reuse negative result')
    require(bad[3][0][1][1] != good[3][0][1][1], 'empty-reuse state negative missed')
    return 'max output error={:.3g}; prefix checks={}; bias, empty rows/blocks, L=0, S=0, tails, d!=dv and finite extremes; empty-reuse detected'.format(max_error, prefix_checks)


def check_rescale_closed_forms():
    with localcontext() as ctx:
        ctx.prec = 70
        e = Decimal(2).exp()
        one = Decimal(1)
        expected = {
            None: float(one / (one + e)),
            'reverse_u_only': float(e / (one + one / e)),
            'reverse_both': float(e / (one + e)),
            'omit_rescale': 0.5,
        }
    observed = {}
    for fault, target in expected.items():
        result = tiled_attention([[1.0]], [[0.0], [2.0]], [[1.0], [0.0]],
                                 [[True, True]], [[0.0, 0.0]], 1, 1, 1.0, 1, fault)
        value = result[0][0][0]
        near(value, target, 'independent closed form ' + str(fault))
        observed[fault] = value
        if fault is not None:
            require(abs(value - expected[None]) > 0.1, 'rescale negative not detected')
    require(observed['reverse_u_only'] > 1.0, 'U-only error should violate convex hull')
    require(0.0 <= observed['reverse_both'] <= 1.0, 'reverse-both convex hull conclusion')
    require(0.0 <= observed['omit_rescale'] <= 1.0, 'omission convex hull conclusion')
    return 'correct={:.9f}; reverse_U={:.9f}; reverse_both={:.9f}; omitted={:.9f}; all match Decimal closed forms'.format(
        observed[None], observed['reverse_u_only'], observed['reverse_both'], observed['omit_rescale'])


def validate_double_fp8(p, qc, m, n):
    if qc != 2 or not (1 <= p <= 128 and 1 <= m <= 128 and 1 <= n <= 512):
        raise ValueError('invalid abstract double-FP8 tile')


def coord_k(p, q, extent):
    if not (0 <= p < extent and q in (0, 1)):
        raise ValueError('invalid coordinate')
    return p + extent * q


def build_layout(A, B, p):
    m, raw_k, n = len(A), len(A[0]), len(B[0])
    validate_double_fp8(p, 2, m, n)
    require(raw_k == len(B) and 0 < raw_k <= 2 * p, 'raw contraction range')
    stationary = [[[0.0] * m for _ in range(2)] for _ in range(p)]
    moving = [[[0.0] * n for _ in range(2)] for _ in range(p)]
    for pp in range(p):
        for q in range(2):
            k = coord_k(pp, q, p)
            if k < raw_k:
                stationary[pp][q] = [A[r][k] for r in range(m)]
                moving[pp][q] = B[k][:]
    return stationary, moving


def layout_product(stationary, moving, swap=False):
    p, m, n = len(stationary), len(stationary[0][0]), len(moving[0][0])
    return [[math.fsum(stationary[pp][q][r] * moving[pp][1 - q if swap else q][c]
                       for pp in range(p) for q in range(2))
             for c in range(n)] for r in range(m)]


def check_double_fp8_layout():
    validate_double_fp8(128, 2, 128, 512)
    invalid = [(0, 2, 1, 1), (129, 2, 8, 8), (8, 3, 8, 8),
               (8, 2, 129, 8), (8, 2, 8, 513)]
    rejected = 0
    for shape in invalid:
        try:
            validate_double_fp8(*shape)
        except ValueError:
            rejected += 1
    require(rejected == len(invalid), 'illegal shape accepted')
    coordinates = [coord_k(p, q, 128) for q in range(2) for p in range(128)]
    require(coordinates == list(range(256)), 'coordinate coverage/order')
    require(coord_k(2, 1, 128) == 130, 'independent coordinate example')
    negative_difference = 0.0
    for raw_k in (15, 16):
        p, m, n = 8, 3, 5
        A = [[float((3 * r + 5 * k) % 7 - 3) for k in range(raw_k)] for r in range(m)]
        B = [[float((7 * k + 2 * c) % 11 - 5) for c in range(n)] for k in range(raw_k)]
        stationary, moving = build_layout(A, B, p)
        result = layout_product(stationary, moving)
        wrong = layout_product(stationary, moving, swap=True)
        reference = [[math.fsum(A[r][k] * B[k][c] for k in range(raw_k))
                      for c in range(n)] for r in range(m)]
        permuted = [[math.fsum(A[r][2 * pp + q] * B[2 * pp + q][c]
                              for pp in range(p) for q in range(2) if 2 * pp + q < raw_k)
                     for c in range(n)] for r in range(m)]
        for r in range(m):
            for c in range(n):
                near(result[r][c], reference[r][c], 'layout dense reference')
                near(permuted[r][c], reference[r][c], 'common permutation invariant')
                negative_difference = max(negative_difference, abs(wrong[r][c] - reference[r][c]))
        if raw_k == 15:
            require(stationary[7][1] == [0.0] * m and moving[7][1] == [0.0] * n,
                    'odd contraction padding not zero')
    require(negative_difference > 1.0, 'half-swap negative not detected')
    return 'coordinate bijection and p=2,q=1 -> k=130; odd padding and common permutation verified; half-swap max difference={}; abstract values only, no FP8 rounding'.format(negative_difference)


def attention_bytes(bq, bk, d, dv, depth, separate_ub=False):
    total = bq * d * 2 + depth * bk * (d + dv) * 2
    total += bq * bk * 4 + bq * dv * 4 + bq * 8
    if separate_ub:
        total += bq * dv * 4
    return total


def check_resource_ledger():
    require(128 * 224 * 1024 == 28 * 1024 * 1024, 'SBUF total')
    a, b, psum = 128 * 128 * 2, 128 * 512 * 2, 128 * 512 * 4
    require((a, b, psum, 2 * (a + b)) == (32768, 131072, 262144, 327680),
            'BF16 matrix ledger')
    require(128 * 2 * 128 == a and 128 * 2 * 512 == b, 'FP8 byte relation')
    require(2 * 128 * 512 * 256 == 2 * (2 * 128 * 512 * 128), 'work relation')
    require(128 * 256 * 2 + 256 * 512 * 2 != a + b, 'wrong FP8 dtype negative missed')
    small = attention_bytes(64, 64, 64, 64, 1)
    double = attention_bytes(64, 64, 64, 64, 2)
    # Independent explicit component sums, in bytes, from the chapter table.
    require(small == sum([8192, 8192, 8192, 16384, 16384, 512]) == 57856,
            'small attention single-slot bytes')
    require(double == sum([8192, 16384, 16384, 16384, 16384, 512]) == 74240,
            'small attention double-buffer bytes')
    require(small / 1024 == 56.5 and double / 1024 == 72.5, 'binary units')
    require(attention_bytes(64, 64, 64, 64, 1, True) == 74240, 'single local Ub')
    require(attention_bytes(64, 64, 64, 64, 2, True) == 90624, 'double local Ub')
    require(attention_bytes(128, 512, 128, 128, 1) == 623616 == 609 * 1024,
            'large single-slot attention')
    require(attention_bytes(128, 512, 128, 128, 2) == 885760 == 865 * 1024,
            'large double-buffer attention')
    require(abs(57.9 * 1024 - small) > 1000 and abs(74.2 * 1024 - double) > 1000,
            'historical kB/KiB negative not detected')
    require(attention_bytes(64, 64, 64, 64, 1, True) != small,
            'omitting live Ub negative not detected')
    return 'matrix A={} B={} PSUM={} double-input={}; attention fused={} B / 56.5 KiB, double={} B / 72.5 KiB, separate-Ub double=90624 B; wrong units/dtype/omitted-Ub detected; totals do not prove residency'.format(a, b, psum, 2 * (a + b), small, double)


class SlotError(RuntimeError):
    pass


class Slot:
    def __init__(self, sid):
        self.sid = sid
        self.gen = -1
        self.block = None
        self.phase = 'free'
        self.pending = set()

    def token(self):
        return (self.sid, self.gen, self.block)

    def snapshot(self):
        return (self.token(), self.phase, frozenset(self.pending))

    def submit(self, block, expected):
        if expected != self.token() or self.phase != 'free':
            raise SlotError('identity mismatch or previous readers/writers remain')
        self.gen += 1
        self.block = block
        self.phase = 'loading'
        self.pending = {'qk', 'pv'}
        return self.token()

    def publish(self, event):
        if event != self.token():
            return False
        if self.phase != 'loading':
            raise SlotError('publish outside loading phase')
        self.phase = 'ready'
        return True

    def release(self, event, reader):
        # Identity is checked before every state mutation.
        if event != self.token():
            return False
        if reader not in ('qk', 'pv'):
            raise SlotError('unregistered reader')
        if reader not in self.pending:
            return True
        if self.phase != 'ready':
            raise SlotError('completion before published input')
        self.pending.remove(reader)
        if not self.pending:
            self.phase = 'free'
        return True


def expect_slot_error(slot, operation):
    before = slot.snapshot()
    try:
        operation()
    except SlotError:
        require(slot.snapshot() == before, 'rejected operation changed slot')
        return
    raise AssertionError('illegal slot operation accepted')


def unsafe_release(slot, event, reader):
    # Intentionally faulty: event identity is ignored.
    slot.pending.discard(reader)
    if not slot.pending:
        slot.phase = 'free'


def reused_slot(order=('qk', 'pv')):
    slot = Slot(0)
    old = slot.submit(('work-A', 0), slot.token())
    expect_slot_error(slot, lambda: slot.release(old, 'qk'))
    require(slot.publish(old), 'initial publish')
    require(slot.release(old, order[0]), 'first genuine reader')
    duplicate_snapshot = slot.snapshot()
    require(slot.release(old, order[0]), 'duplicate completion not idempotent')
    require(slot.snapshot() == duplicate_snapshot, 'duplicate removed another reader')
    expect_slot_error(slot, lambda: slot.submit(('work-A', 2), old))
    require(slot.release(old, order[1]) and slot.phase == 'free', 'last reader did not free')
    new = slot.submit(('work-A', 2), old)
    return slot, old, new


def check_slot_protocol():
    stale_rejected = 0
    for order in (('qk', 'pv'), ('pv', 'qk')):
        slot, old, new = reused_slot(order)
        before = slot.snapshot()
        # Old completions must not touch new readers even while new input loads.
        for reader in ('qk', 'pv'):
            require(slot.release(old, reader) is False, 'stale loading completion accepted')
            require(slot.snapshot() == before, 'stale loading completion mutated new state')
            stale_rejected += 1
        require(slot.publish(old) is False and slot.snapshot() == before,
                'stale publication mutated new state')
        require(slot.publish(new), 'new publication failed')
        before = slot.snapshot()
        bad_tokens = [old, (99, new[1], new[2]),
                      (new[0], new[1], ('work-A', 0)),
                      (new[0], new[1] + 1, new[2])]
        for event in bad_tokens:
            for reader in ('qk', 'pv'):
                require(slot.release(event, reader) is False, 'foreign completion accepted')
                require(slot.snapshot() == before, 'foreign completion mutated new readers/free state')
                stale_rejected += 1
        require(slot.release(new, order[0]), 'new first reader failed')
        before_last = slot.snapshot()
        require(slot.phase == 'ready' and len(slot.pending) == 1, 'first reader freed early')
        for reader in ('qk', 'pv'):
            require(slot.release(old, reader) is False, 'stale event accepted before final reader')
            require(slot.snapshot() == before_last, 'stale event removed final real reader')
            stale_rejected += 1
        expect_slot_error(slot, lambda: slot.submit(('work-A', 4), new))
        require(slot.release(new, order[1]), 'new last reader rejected')
        require(slot.phase == 'free' and not slot.pending, 'new last reader did not release')
        require(slot.submit(('work-A', 4), new)[1] == 2, 'next genuine generation blocked')

    bad, old, new = reused_slot()
    bad.publish(new)
    snapshot = bad.snapshot()
    unsafe_release(bad, old, 'qk')
    unsafe_release(bad, old, 'pv')
    require(bad.snapshot() != snapshot and bad.phase == 'free',
            'missing identity guard negative did not reproduce premature release')
    overwritten = bad.submit(('work-A', 4), new)
    require(overwritten[1] == 2, 'negative did not permit premature overwrite')

    trajectories = 0
    for depth in (1, 2, 4):
        for count in (0, 1, 2, 3, 5):
            slots = [Slot(s) for s in range(depth)]
            for j in range(min(depth, count)):
                slots[j].submit(('pipeline', j), slots[j].token())
            consumed = []
            for j in range(count):
                slot = slots[j % depth]
                event = slot.token()
                require(event[1] == j // depth and event[2] == ('pipeline', j),
                        'logical stage or generation mismatch')
                require(slot.publish(event), 'pipeline publish failed')
                require(slot.release(event, 'qk'), 'pipeline qk failed')
                require(slot.phase != 'free', 'pipeline early release')
                consumed.append(j)
                require(slot.release(event, 'pv') and slot.phase == 'free', 'pipeline pv failed')
                if j + depth < count:
                    slot.submit(('pipeline', j + depth), event)
                    before = slot.snapshot()
                    for reader in ('qk', 'pv'):
                        require(slot.release(event, reader) is False, 'old pipeline event accepted')
                        require(slot.snapshot() == before, 'old pipeline event changed next stage')
            require(consumed == list(range(count)), 'missing or duplicate consumed stage')
            require(all(slot.phase == 'free' for slot in slots), 'pipeline did not drain')
            trajectories += 1
    return 'foreign/stale completions rejected without mutation={}; real final readers release; duplicate events idempotent; missing-identity negative permits early overwrite; {} finite multi-slot trajectories drain exactly once'.format(stale_rejected, trajectories)


def main():
    checks = [('attention', check_attention), ('rescale_closed_forms', check_rescale_closed_forms),
              ('double_fp8_layout', check_double_fp8_layout), ('resource_ledger', check_resource_ledger),
              ('slot_protocol', check_slot_protocol)]
    failures = []
    for name, check in checks:
        try:
            print(name + ': ' + check())
        except Exception as exc:
            failures.append(name)
            print(name + ': ' + type(exc).__name__ + ': ' + str(exc))
    print('scope: CPU binary64/Decimal and sequential event model; no NKI, Neuron/GPU execution, FP8 rounding, concurrency or performance measurement')
    if failures:
        print('scoped failures: ' + ', '.join(failures))
        return 1
    print('all scoped assertions and intentional negative controls satisfied')
    return 0


if __name__ == '__main__':
    sys.exit(main())
