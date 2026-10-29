#!/usr/bin/env python3
"""Independent CPU references for the scoped Attention reasoning additions.

The explicitly selected E4M3FN finite codebook is a software model. Python
arithmetic is not a hardware FP32/WGMMA emulator. Buffer exploration is finite.
"""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def code_value(code):
    exponent, mantissa = divmod(code, 8)
    return mantissa * 2**-9 if not exponent else (1 + mantissa/8) * 2**(exponent-7)


CODEBOOK = [code_value(c) for c in range(127)]  # 0..448; final bit pattern is NaN.


def round_fp8(value):
    if not math.isfinite(value):
        raise ValueError('This reference requires finite inputs')
    x = min(abs(value), CODEBOOK[-1])
    code = min(range(len(CODEBOOK)), key=lambda c: (abs(CODEBOOK[c]-x), c % 2))
    return math.copysign(CODEBOOK[code], value)


def quantize(values):
    if any(not math.isfinite(x) for x in values):
        raise ValueError('This reference requires finite blocks')
    peak = max(map(abs, values), default=0)
    scale = peak / CODEBOOK[-1] if peak else 1.0
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Scale is outside the explicitly modeled domain')
    return [round_fp8(x/scale) for x in values], scale


def blocks(matrix, size):
    codes, decoded, scales = [], [], []
    width = len(matrix[0])
    for start in range(0, len(matrix), size):
        rows = matrix[start:start+size]
        flat, scale = quantize([x for row in rows for x in row])
        for row in range(len(rows)):
            values = flat[row*width:(row+1)*width]
            codes.append(values); decoded.append([scale*x for x in values]); scales.append(scale)
    return codes, decoded, scales


def dense_weights(q, k, mask, bias, scale):
    scores = [scale*sum(a*b for a, b in zip(q, key))+bias[j] for j, key in enumerate(k)]
    maximum = max((s for s, keep in zip(scores, mask) if keep), default=-math.inf)
    numerators = [math.exp(s-maximum) if keep else 0.0 for s, keep in zip(scores, mask)]
    mass = sum(numerators)
    return [x/mass if mass else 0.0 for x in numerators], scores


def weighted(weights, values):
    return [sum(p*v[d] for p, v in zip(weights, values)) for d in range(len(values[0]))]


def fp8_row(qcodes, sq, kcodes, sk, vcodes, sv, mask, bias, scale, tile, quantize_p):
    """Online codes/scales path; does not call the dense probability reference."""
    maximum, mass = -math.inf, 0.0
    out = [0.0]*len(vcodes[0])
    for start in range(0, len(kcodes), tile):
        indices = [j for j in range(start, min(start+tile, len(kcodes))) if mask[j]]
        if not indices:
            continue
        scores = [scale*sq*sk[j]*sum(a*b for a, b in zip(qcodes, kcodes[j]))+bias[j] for j in indices]
        new = max(maximum, max(scores)); ratio = math.exp(maximum-new) if mass else 0.0
        probabilities = [math.exp(x-new) for x in scores]
        mass = ratio*mass+sum(probabilities)
        if quantize_p:
            pcodes, sp = quantize(probabilities)
        else:
            pcodes, sp = probabilities, 1.0
        out = [ratio*u + sp*sum(p*sv[j]*vcodes[j][d] for p, j in zip(pcodes, indices)) for d, u in enumerate(out)]
        maximum = new
    return [u/mass if mass else 0.0 for u in out]


def independent_p_weights(scores, mask, tile):
    """Global normalization + local P quantization, with no online m/ell/U state.

    Per-row local amax quantization commutes with a positive scalar rescaling.
    Normalize each tile locally, quantize, then restore global units.
    """
    maximum = max((s for s, keep in zip(scores, mask) if keep), default=-math.inf)
    mass = sum(math.exp(s-maximum) for s, keep in zip(scores, mask) if keep)
    weights = [0.0]*len(scores)
    if not mass:
        return weights
    for start in range(0, len(scores), tile):
        indices = [j for j in range(start, min(start+tile, len(scores))) if mask[j]]
        if not indices:
            continue
        local = max(scores[j] for j in indices)
        p = [math.exp(scores[j]-local) for j in indices]
        codes, scale = quantize(p)
        for j, value in zip(indices, codes):
            weights[j] = value*scale*math.exp(local-maximum)/mass
    return weights


def actions(state, count, slots, consumers):
    next_block, buffers = state; full = (1 << consumers)-1
    choices = []
    if next_block < count and buffers[next_block % slots][0] == -1:
        choices.append(('acquire', next_block, -1))
    for block, phase, started, ended in buffers:
        if block < 0:
            continue
        if phase == 1: choices.append(('load_complete', block, -1))
        if phase == 2: choices.append(('publish', block, -1))
        if phase == 3:
            for consumer in range(consumers):
                bit = 1 << consumer
                if not started & bit: choices.append(('read_issue', block, consumer))
                elif not ended & bit: choices.append(('read_complete', block, consumer))
            if ended == full: choices.append(('release', block, -1))
    return choices


def transition(state, action, count, slots, consumers):
    if action not in actions(state, count, slots, consumers):
        raise ValueError('Illegal, duplicate, or stale-generation event')
    next_block, old = state; buffers = list(old)
    operation, block, consumer = action; slot = block % slots
    _, phase, started, ended = buffers[slot]
    if operation == 'acquire':
        next_block += 1; buffers[slot] = (block, 1, 0, 0)
    elif operation == 'load_complete': buffers[slot] = (block, 2, started, ended)
    elif operation == 'publish': buffers[slot] = (block, 3, started, ended)
    elif operation == 'read_issue': buffers[slot] = (block, phase, started | 1 << consumer, ended)
    elif operation == 'read_complete': buffers[slot] = (block, phase, started, ended | 1 << consumer)
    elif operation == 'release': buffers[slot] = (-1, 0, 0, 0)
    return next_block, tuple(buffers)


def explore(count, slots, consumers):
    initial = (0, tuple([(-1, 0, 0, 0)]*slots))
    visited = {initial}; pending = deque([initial]); terminal = 0
    while pending:
        state = pending.popleft(); choices = actions(state, count, slots, consumers)
        assert all(block < 0 or block % slots == slot for slot, (block, _, _, _) in enumerate(state[1]))
        assert all(not ended & ~started for _, _, started, ended in state[1])
        if not choices:
            assert state[0] == count and all(x[0] == -1 for x in state[1]), 'Reachable deadlock'
            terminal += 1
        for action in choices:
            candidate = transition(state, action, count, slots, consumers)
            if candidate not in visited: visited.add(candidate); pending.append(candidate)
    return len(visited), terminal


def decode_cost(c, utilization, reuse_heads=1):
    requests = 32/reuse_heads*8192*256*2
    extra = 32*c*129*4 if c > 1 else 0
    main = requests/(2**40*utilization)*1e6
    return {'ctas': 32//reuse_heads*c, 'requests': requests, 'workspace': extra,
            'main_us': main, 'total_us': 8+3+main+(3+2*extra/2**40*1e6 if c > 1 else 0)}


def run():
    checks = []
    def record(name, passed, **detail):
        checks.append({'name': name, 'passed': bool(passed), **detail})
    def close(a, b, tolerance=2e-11):
        return max((abs(x-y) for x, y in zip(a, b)), default=0) <= tolerance
    def rejects(state, event, **kwargs):
        try: transition(state, event, **kwargs)
        except ValueError: return True
        return False

    record('format/codebook', len(CODEBOOK) == 127 and CODEBOOK[-1] == 448 and CODEBOOK[1] == 2**-9)
    for value, expected in [(0, 0), (1.0625, 1), (1.1875, 1.25), (2**-10, 0), (3*2**-10, 2**-8), (1000, 448), (-1000, -448)]:
        record(f'format/round/{value}', round_fp8(value) == expected)
    zero, scale = quantize([0, 0]); record('format/zero_block', zero == [0, 0] and scale == 1)
    try: quantize([math.ulp(0.0)])
    except ValueError: underflow_rejected = True
    else: underflow_rejected = False
    record('format/unrepresentable_scale_rejected', underflow_rejected)
    rng = random.Random(81027)
    for rows in [1, 3]:
        for length in [1, 3, 5]:
            q = [[rng.uniform(-2, 2) for _ in range(4)] for _ in range(rows)]
            k = [[rng.uniform(-2, 2) for _ in range(4)] for _ in range(length)]
            v = [[rng.uniform(-3, 3) for _ in range(3)] for _ in range(length)]
            qc, qd, qs = blocks(q, 2)
            for tile in [1, 2, 4]:
                kc, kd, ks = blocks(k, tile); vc, vd, vs = blocks(v, tile)
                for mode in ['dense', 'empty', 'first_empty', 'causal_tail']:
                    for i in range(rows):
                        mask = [mode == 'dense' or (mode == 'first_empty' and j >= tile) or (mode == 'causal_tail' and j <= length-rows+i) for j in range(length)]
                        bias = [rng.uniform(-0.3, 0.3) for _ in k]
                        p, scores = dense_weights(qd[i], kd, mask, bias, 0.5)
                        reference = weighted(p, vd)
                        low_weights = independent_p_weights(scores, mask, tile)
                        low_reference = weighted(low_weights, vd)
                        exact = fp8_row(qc[i], qs[i], kc, ks, vc, vs, mask, bias, 0.5, tile, False)
                        low = fp8_row(qc[i], qs[i], kc, ks, vc, vs, mask, bias, 0.5, tile, True)
                        original_p, _ = dense_weights(q[i], k, mask, bias, 0.5)
                        original = weighted(original_p, v)
                        bound = [sum(abs(pl-po)*abs(v[j][d])+abs(pl)*abs(vd[j][d]-v[j][d]) for j, (pl, po) in enumerate(zip(low_weights, original_p))) for d in range(3)]
                        record(f'quantized/rows={rows}/keys={length}/tile={tile}/{mode}/row={i}',
                               close(exact, reference) and close(low, low_reference) and all(abs(a-b) <= e+1e-11 for a, b, e in zip(low, original, bound)))

    # Failures should be visible even when tensor shapes remain unchanged.
    value_options = ([0.0], 1.0, [[0.0]], [1.0], [[448.0]])
    restored = fp8_row(*value_options, [2/448], [True], [0.0], 1.0, 1, True)
    omitted = fp8_row(*value_options, [1.0], [True], [0.0], 1.0, 1, True)
    record('negative/missing_value_scale', close(restored, [2.0]) and not close(omitted, [2.0]))
    restored = fp8_row([2], .5, [[3], [1]], [.25, .25], [[1], [0]], [1, 1], [True, True], [0, 0], 1, 2, False)
    omitted = fp8_row([2], 1, [[3], [1]], [1, 1], [[1], [0]], [1, 1], [True, True], [0, 0], 1, 2, False)
    expected = 1/(1+math.exp(-.5))
    record('negative/missing_QK_scales', close(restored, [expected]) and not close(omitted, [expected]))
    old_codes, old_scale = quantize([1.0]); _, new_scale = quantize([1.0, 2.0])
    record('negative/change_scale_without_reencoding', abs(old_codes[0]*old_scale-1)<1e-12 and old_codes[0]*new_scale == 2)
    p = [0.1, 0.2, 0.7]; values = [[1, -1], [5, 2], [-2, 4]]; order = [2, 0, 1]
    record('layout/matched_key_permutation', close(weighted([p[i] for i in order], [values[i] for i in order]), weighted(p, values)))
    record('negative/one_sided_permutation', not close(weighted([p[i] for i in order], values), weighted(p, values)))
    q, k = [1, 0], [0, 1]; h = 1/math.sqrt(2)
    transform = lambda x: [(x[0]+x[1])*h, (x[0]-x[1])*h]
    record('orthogonal/common_transform', math.isclose(sum(a*b for a, b in zip(transform(q), transform(k))), 0, abs_tol=1e-12))
    record('negative/one_sided_transform', not math.isclose(sum(a*b for a, b in zip(transform(q), k)), 0, abs_tol=1e-12))
    rotate = lambda x, t: [x[0]*math.cos(t)-x[1]*math.sin(t), x[0]*math.sin(t)+x[1]*math.cos(t)]
    after = sum(a*b for a, b in zip(transform(rotate(q, .2)), transform(rotate(k, .8))))
    before = sum(a*b for a, b in zip(rotate(transform(q), .2), rotate(transform(k), .8)))
    record('negative/moving_transform_before_position', not math.isclose(after, before))

    state_counts = {}
    for count in [1, 2, 3, 4]:
        for slots in [1, 2]:
            for consumers in [1, 2]:
                states, terminals = explore(count, slots, consumers)
                state_counts[f'{count}/{slots}/{consumers}'] = states
                record(f'buffer_exploration/blocks={count}/slots={slots}/readers={consumers}', terminals == 1, states=states)
    options = dict(count=3, slots=1, consumers=2)
    state = (0, ((-1, 0, 0, 0),))
    state = transition(state, ('acquire', 0, -1), **options)
    record('negative/publish_before_load_complete', rejects(state, ('publish', 0, -1), **options))
    record('negative/read_before_publish', rejects(state, ('read_issue', 0, 0), **options))
    for event in [('load_complete', 0, -1), ('publish', 0, -1), ('read_issue', 0, 0), ('read_complete', 0, 0)]:
        state = transition(state, event, **options)
    record('negative/release_after_one_reader', rejects(state, ('release', 0, -1), **options))
    record('negative/overwrite_live_slot', rejects(state, ('acquire', 1, -1), **options))
    record('negative/duplicate_completion', rejects(state, ('read_complete', 0, 0), **options))
    for event in [('read_issue', 0, 1), ('read_complete', 0, 1), ('release', 0, -1), ('acquire', 1, -1)]:
        state = transition(state, event, **options)
    record('negative/stale_generation_release', rejects(state, ('release', 0, -1), **options))

    cases = [(1, .25, 499.28125, 0), (4, .75, 176.881, 66048), (8, 1, 136.311, 132096), (16, 1, 136.551, 264192)]
    estimates = []
    for c, use, expected, extra in cases:
        cost = decode_cost(c, use); estimates.append(cost)
        record(f'cost/decode/C={c}', abs(cost['total_us']-expected)<.0006 and cost['workspace']==extra and cost['ctas']==32*c, **cost)
    record('cost/saturation_reverses_choice', decode_cost(1, 1)['total_us']<decode_cost(8, 1)['total_us'])
    record('cost/more_partitions_not_always_better', estimates[-1]['total_us']>estimates[-2]['total_us'])
    grouped = decode_cost(1, 1, 4)
    record('cost/GQA_shared_candidate', grouped['requests']==32*2**20 and grouped['ctas']==8)
    record('cost/prefill_parallelism', 32*math.ceil(8192/128)==2048 and 8*math.ceil(8192/128)==512)
    record('cost/main_flops', 2*32*8192*(128+128)==134217728)
    two = lambda n: 10+2*n; three = lambda n: 18+1.7*n; degraded = lambda n: 18+2.4*n
    record('diagnosis/fill_drain_crossover', two(8)<three(8) and two(64)>three(64) and math.isclose((18-10)/(2-1.7), 26.6666666666667))
    record('diagnosis/steady_state_counterexample', degraded(64)>two(64) and degraded(64)-two(64)>degraded(8)-two(8))
    documents = ['knowledge/frameworks/attention-backends.md', 'knowledge/sources/S-FLASHATTENTION3.md',
                 'knowledge/sources/S-FLASHINFERPAPER.md', 'knowledge/sources/S-FI.md', 'knowledge/sources/S-TRITON.md']
    return {'result': 'PASS' if all(c['passed'] for c in checks) else 'FAIL', 'count': len(checks), 'checks': checks,
            'finite_buffer_state_counts': state_counts,
            'documents': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in documents},
            'scope': 'CPU finite E4M3FN codebook and staged scalar attention references, mask/tail/zero cases, error inequality, negative scale/layout/position cases, finite buffer protocol exploration and invalid events, declared cost and diagnosis arithmetic. No hardware FP32/FP8 emulator, GPU synchronization proof, physical overlap/performance or Agent construction evaluation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--report', type=Path); args = parser.parse_args()
    result = run()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'result': result['result'], 'count': result['count'],
                      'failed': [c for c in result['checks'] if not c['passed']], 'scope': result['scope']}, ensure_ascii=False))
    raise SystemExit(result['result'] != 'PASS')
