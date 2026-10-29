import itertools
import math


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rejected(label, action):
    try:
        action()
    except AssertionError as exc:
        print('negative control rejected: ' + label + '; ' + str(exc))
        return
    raise AssertionError('negative control escaped validator: ' + label)


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected), label + ': nonfinite value')
    require(abs(actual - expected) <= 1e-10 * max(1.0, abs(expected)), label + ': mismatch')


def rounded(value, quantum):
    return ((value + quantum - 1) // quantum) * quantum


def occupancy_model(T, R, S, p, defect=None):
    thread_cap = p['threadsPerBlock']
    if defect == 'omit_block_cap':
        thread_cap = p['threadsPerSM']
    if T <= 0 or T > thread_cap or R < 0 or R > p['regsPerThread'] or S < 0:
        return 0
    W = (T + 31) // 32
    rb = W * rounded(32 * R, p['regQuantum'])
    sb = rounded(S, p['sharedQuantum'])
    if defect == 'raw_regs':
        rb = T * R
    if defect == 'raw_shared':
        sb = S
    if defect == 'zero_shared_illegal' and S == 0:
        return 0
    if defect == 'zero_regs_illegal' and R == 0:
        return 0
    if rb > p['regsPerBlock'] or sb > p['sharedPerBlock']:
        return 0
    limits = [p['blocksPerSM'], p['threadsPerSM'] // T,
              p['warpsPerSM'] // W, p['barrierBlocks']]
    if rb:
        limits.append(p['regsPerSM'] // rb)
    if sb:
        limits.append(p['sharedPerSM'] // sb)
    return min(limits)


def occupancy_reference(T, R, S, p):
    # Independent legality, quantum accumulation and feasible-count enumeration.
    if not 1 <= T <= p['threadsPerBlock']:
        return 0
    if not 0 <= R <= p['regsPerThread'] or S < 0:
        return 0
    warp_starts = list(range(0, T, 32))
    one_warp = 0
    while one_warp < 32 * R:
        one_warp += p['regQuantum']
    register_allocation = sum(one_warp for _ in warp_starts)
    shared_allocation = 0
    while shared_allocation < S:
        shared_allocation += p['sharedQuantum']
    if register_allocation > p['regsPerBlock']:
        return 0
    if shared_allocation > p['sharedPerBlock']:
        return 0
    feasible = [0]
    for count in range(1, p['blocksPerSM'] + 1):
        demands = (
            (count * T, p['threadsPerSM']),
            (count * len(warp_starts), p['warpsPerSM']),
            (count * register_allocation, p['regsPerSM']),
            (count * shared_allocation, p['sharedPerSM']),
            (count, p['barrierBlocks']),
        )
        if all(demand <= capacity for demand, capacity in demands):
            feasible.append(count)
    return max(feasible)


def validate_occupancy(engine, T, R, S, p):
    actual = engine(T, R, S, p)
    expected = occupancy_reference(T, R, S, p)
    require(actual == expected,
            'occupancy T/R/S=' + str((T, R, S)) + ': got ' + str(actual) + ', expected ' + str(expected))
    return actual


def tile_report(c, budget, defect=None):
    kv_stages = c['nkv'] - (1 if defect == 'single_kv' else 0)
    p_stages = 0 if defect == 'omit_p' else c['np']
    parts = {
        'Q': c['br'] * c['d'] * c['qbytes'],
        'KV': kv_stages * c['bc'] * (c['d'] * c['kbytes'] + c['dv'] * c['vbytes']),
        'P': p_stages * c['br'] * c['bc'] * c['pbytes'],
        'U': c['br'] * c['dv'] * c['ubytes'],
        'stats': 2 * c['br'] * c['statbytes'],
        'other': c['other'],
    }
    total = sum(parts.values())
    return {'parts': parts, 'total': total, 'fits': total <= budget}


def tile_manifest(c):
    # Reference expands each required physical object, including every stage.
    objects = [('Q', 'Q', c['br'], c['d'], c['qbytes']),
               ('U', 'U', c['br'], c['dv'], c['ubytes']),
               ('m', 'stats', c['br'], 1, c['statbytes']),
               ('ell', 'stats', c['br'], 1, c['statbytes'])]
    for stage in range(c['nkv']):
        objects.append(('K' + str(stage), 'KV', c['bc'], c['d'], c['kbytes']))
        objects.append(('V' + str(stage), 'KV', c['bc'], c['dv'], c['vbytes']))
    for stage in range(c['np']):
        objects.append(('P' + str(stage), 'P', c['br'], c['bc'], c['pbytes']))
    parts = dict.fromkeys(('Q', 'KV', 'P', 'U', 'stats', 'other'), 0)
    details = []
    for name, family, rows, columns, size in objects:
        payload = sum(columns * size for _ in range(rows))
        parts[family] += payload
        details.append((name, payload))
    parts['other'] = c['other']
    details.append(('other', c['other']))
    return parts, details


def validate_tile(engine, c, budget):
    actual = engine(c, budget)
    expected, details = tile_manifest(c)
    require(actual['parts'] == expected, 'tile object/stage mismatch against manifest ' + str(details))
    total = sum(payload for _, payload in details)
    require(actual['total'] == total, 'tile total differs from object manifest')
    require(actual['fits'] == (total <= budget), 'tile admission differs from object manifest')
    return actual


def segments_model(addresses, unit=32):
    # Scope: nonnegative, naturally aligned 4-byte scalar accesses.
    require(all(a >= 0 and a % 4 == 0 for a in addresses), 'unaligned scalar outside check scope')
    return len({a // unit for a in addresses})


def segments_reference(addresses):
    if not addresses:
        return 0
    count = 0
    for segment_start in range((min(addresses) // 32) * 32, max(addresses) + 4, 32):
        if any(a < segment_start + 32 and a + 4 > segment_start for a in addresses):
            count += 1
    return count


def validate_segments(engine, addresses):
    actual = engine(addresses)
    expected = segments_reference(addresses)
    require(actual == expected, 'segment coverage mismatch: ' + str((actual, expected)))
    return actual


def banks_model(words, count_lanes=False):
    banks = [[] for _ in range(32)]
    for word in words:
        banks[word % 32].append(word)
    if count_lanes:
        return max(map(len, banks), default=0)
    return max((len(set(group)) for group in banks), default=0)


def banks_reference(words):
    busiest = 0
    for bank in range(32):
        distinct = []
        for word in words:
            if word % 32 == bank and word not in distinct:
                distinct.append(word)
        busiest = max(busiest, len(distinct))
    return busiest


def validate_banks(engine, words):
    actual = engine(words)
    expected = banks_reference(words)
    require(actual == expected, 'bank distinct-address mismatch: ' + str((actual, expected)))
    return actual


def dense_state(scores, values, visible, end, width):
    indices = [j for j in range(end) if visible[j]]
    if not indices:
        return (-math.inf, 0.0, tuple(0.0 for _ in range(width)))
    m = max(scores[j] for j in indices)
    weights = [math.exp(scores[j] - m) for j in indices]
    ell = math.fsum(weights)
    u = tuple(math.fsum(weight * values[j][axis] for j, weight in zip(indices, weights))
              for axis in range(width))
    return m, ell, u


def attention_stream(scores, values, visible, block, width, wrong_rescale=False):
    require(block > 0, 'block must be positive')
    m, ell, u = -math.inf, 0.0, [0.0] * width
    trace = []
    for start in range(0, len(scores), block):
        end = min(start + block, len(scores))
        indices = [j for j in range(start, end) if visible[j]]
        if indices:
            new_m = max(m, max(scores[j] for j in indices))
            if ell == 0.0:
                factor = 0.0
            else:
                factor = math.exp(new_m - m if wrong_rescale else m - new_m)
            weights = [math.exp(scores[j] - new_m) for j in indices]
            u = [factor * u[axis] + sum(weight * values[j][axis]
                 for j, weight in zip(indices, weights)) for axis in range(width)]
            ell = factor * ell + sum(weights)
            m = new_m
        trace.append((end, m, ell, tuple(u)))
    return trace


def validate_attention(engine, scores, values, visible, block, width=2):
    trace = engine(scores, values, visible, block, width)
    expected_ends = list(range(block, len(scores), block))
    if scores:
        expected_ends.append(len(scores))
    require([state[0] for state in trace] == expected_ends, 'attention missing/duplicated block')
    for end, m, ell, u in trace:
        rm, rl, ru = dense_state(scores, values, visible, end, width)
        require(m == rm, 'attention prefix maximum mismatch')
        close(ell, rl, 'attention prefix mass')
        require(len(u) == len(ru), 'attention output width mismatch')
        for actual, expected in zip(u, ru):
            close(actual, expected, 'attention prefix U')
    final = trace[-1][1:] if trace else (-math.inf, 0.0, tuple(0.0 for _ in range(width)))
    _, ref_ell, ref_u = dense_state(scores, values, visible, len(scores), width)
    actual_output = [v / final[1] if final[1] else 0.0 for v in final[2]]
    expected_output = [v / ref_ell if ref_ell else 0.0 for v in ref_u]
    for actual, expected in zip(actual_output, expected_output):
        close(actual, expected, 'attention output')


def step_model(device, host, extra_device=0.0):
    return max(device + extra_device, host)


def step_reference(device, host, extra_device):
    # Ideal independent host/device pipeline, with startup retained in events.
    host_end = 0.0
    device_end = 0.0
    completions = []
    for _ in range(6):
        host_end += host
        device_end = max(host_end, device_end) + device + extra_device
        completions.append(device_end)
    return (completions[-1] - completions[0]) / 5


def validate_step(engine, device, host, extra_device=0.0):
    close(engine(device, host, extra_device), step_reference(device, host, extra_device), 'ideal steady step')


def wave_model(grid, resident, sms):
    return (grid + resident * sms - 1) // (resident * sms)


def validate_waves(grid, resident, sms):
    remaining, count = grid, 0
    while remaining:
        remaining -= min(remaining, resident * sms)
        count += 1
    require(wave_model(grid, resident, sms) == count, 'wave packing mismatch')


def persistent_trace(grid, capacity, layers, omit_consumer=False):
    # Synthetic finite model: CTAs retain slots through all phases.
    # The last CTA produces each flag; resident consumers wait before a barrier.
    resident = list(range(min(grid, capacity)))
    trace = []
    for layer in range(layers):
        if grid - 1 not in resident:
            return trace
        trace.append(('produce', layer, grid - 1))
        for worker in resident:
            if omit_consumer and worker == 0:
                continue
            trace.append(('consume', layer, worker))
        if len(resident) != grid:
            return trace
        trace.append(('barrier', layer))
    return trace


def validate_persistent(engine, grid, capacity, layers):
    actual = engine(grid, capacity, layers)
    expected = []
    for layer in range(layers):
        expected.append(('produce', layer, grid - 1))
        expected.extend(('consume', layer, worker) for worker in range(grid))
        expected.append(('barrier', layer))
    require(actual == expected, 'persistent phase trace incomplete; queued producer or missing participant')


def main():
    print('K-HW-005 deterministic CPU models; no CUDA, filesystem, subprocesses or network')
    p = {'threadsPerBlock': 1024, 'threadsPerSM': 2048, 'warpsPerSM': 64,
         'blocksPerSM': 32, 'regsPerThread': 255, 'regsPerBlock': 65536,
         'regsPerSM': 65536, 'sharedPerBlock': 98304, 'sharedPerSM': 98304,
         'regQuantum': 256, 'sharedQuantum': 128, 'barrierBlocks': 32}
    cases = 0
    for T, R, S, barriers in itertools.product(
            (0, 1, 31, 32, 33, 128, 320, 1024, 1025, 2048),
            (0, 1, 31, 32, 37, 255, 256),
            (0, 1, 127, 128, 98304, 98305), (0, 2, 32)):
        config = dict(p, barrierBlocks=barriers)
        validate_occupancy(occupancy_model, T, R, S, config)
        cases += 1
    require(validate_occupancy(occupancy_model, 128, 37, 0, p) == 12, '75 percent example')
    require(12 * 4 / 64 == 0.75, 'occupancy conversion')
    require(validate_occupancy(occupancy_model, 128, 0, 0, p) == 16, 'zero resource example')
    require(validate_occupancy(occupancy_model, 1024, 1, 0, p) == 2, '1024 boundary')
    require(validate_occupancy(occupancy_model, 1025, 1, 0, p) == 0, '1025 boundary')
    require(validate_occupancy(occupancy_model, 33, 32, 0, p) == 32, 'partial warp allocation')
    for defect, args in (
            ('omit_block_cap', (1025, 1, 0)),
            ('zero_shared_illegal', (128, 37, 0)),
            ('zero_regs_illegal', (128, 0, 0)),
            ('raw_regs', (128, 37, 0))):
        faulty = lambda T, R, S, config, d=defect: occupancy_model(T, R, S, config, d)
        rejected(defect, lambda f=faulty, a=args: validate_occupancy(f, *a, p))
    p2 = dict(p, regsPerSM=10**9, sharedPerBlock=1000, sharedPerSM=1000)
    require(validate_occupancy(occupancy_model, 128, 1, 1, p2) == 7, 'shared rounding boundary')
    rejected('raw_shared', lambda: validate_occupancy(
        lambda T, R, S, config: occupancy_model(T, R, S, config, 'raw_shared'), 128, 1, 1, p2))
    p3 = dict(p, regsPerBlock=4096)
    require(validate_occupancy(occupancy_model, 128, 37, 0, p3) == 0, 'independent block register cap')
    print('occupancy: ' + str(cases) + ' formula/enumeration cases; zero use, block/SM limits and quantum boundaries')

    tile = {'br': 64, 'bc': 64, 'd': 128, 'dv': 128,
            'qbytes': 2, 'kbytes': 2, 'vbytes': 2, 'pbytes': 2,
            'ubytes': 4, 'statbytes': 4, 'nkv': 2, 'np': 2, 'other': 0}
    report = validate_tile(tile_report, tile, 99 * 1024)
    require(report['parts'] == {'Q': 16384, 'KV': 65536, 'P': 16384,
                               'U': 32768, 'stats': 512, 'other': 0}, 'analytic tile object example')
    require(report['total'] == 131584 and not report['fits'], '99 KiB shared-only example')
    require(validate_tile(tile_report, tile, 227 * 1024)['fits'], '227 KiB payload example')
    require(validate_tile(tile_report, tile, 131584)['fits'], 'exact payload boundary')
    require(not validate_tile(tile_report, tile, 131583)['fits'], 'one-byte payload boundary')
    require(tile_report(tile, 120 * 1024, 'omit_p')['fits'], 'omit-P defect should falsely fit 120 KiB')
    require(tile_report(tile, 99 * 1024, 'single_kv')['fits'], 'missing-stage defect should falsely fit 99 KiB')
    rejected('tile omit P at 120 KiB', lambda: validate_tile(
        lambda c, b: tile_report(c, b, 'omit_p'), tile, 120 * 1024))
    rejected('tile missing K/V stage at 99 KiB', lambda: validate_tile(
        lambda c, b: tile_report(c, b, 'single_kv'), tile, 99 * 1024))
    small = dict(tile, br=1, bc=3, d=5, dv=2, nkv=1, np=0, other=17)
    validate_tile(tile_report, small, 100000)
    print('tile payload: independent object manifest totals 131584 bytes; shared-only admission checked')

    aligned = [4 * lane for lane in range(32)]
    offset = [4 * lane + 4 for lane in range(32)]
    require(validate_segments(segments_model, aligned) == 4, 'aligned segment example')
    require(validate_segments(segments_model, offset) == 5, 'offset segment example')
    for addresses in ([], [28], [4 * lane for lane in range(8)],
                      [8 * lane for lane in range(32)], [0] * 32):
        validate_segments(segments_model, addresses)
    rejected('128-byte segment unit', lambda: validate_segments(lambda a: segments_model(a, 128), offset))
    for words, expected in (([], 0), (list(range(32)), 1),
                            ([8 * lane for lane in range(32)], 8), ([7] * 32, 1)):
        require(validate_banks(banks_model, words) == expected, 'analytic bank case')
    rejected('broadcast counted by lanes', lambda: validate_banks(lambda w: banks_model(w, True), [7] * 32))
    print('scalar address models: 32-byte coverage and distinct-word banks; no HBM transaction measurements')

    scores = [0.0, 2.0, -1.0, 7.0, 3.0]
    values = [(1.0, 2.0), (0.0, -1.0), (3.0, 0.0), (-2.0, 4.0), (1.0, 1.0)]
    for mask, block in itertools.product(
            ([True] * 5, [False, True, False, False, True], [True, False, False, False, False], [False] * 5),
            (1, 2, 3, 8)):
        validate_attention(attention_stream, scores, values, mask, block)
    validate_attention(attention_stream, [], [], [], 2)
    validate_attention(attention_stream, [1000.0, -1000.0], [(1.0, 0.0), (0.0, 1.0)], [True, True], 1)
    rejected('inverse online rescaling', lambda: validate_attention(
        lambda s, v, mask, block, width: attention_stream(s, v, mask, block, width, True),
        [0.0, 2.0], [(1.0, 0.0), (0.0, 1.0)], [True, True], 1))
    print('online attention: dense prefix statistics and outputs; empty blocks/rows, zero keys and tails')

    for device, host, extra in ((12.0, 1.2, 0.0), (0.8, 1.2, 0.0),
                                (0.8, 0.2, 0.0), (12.0, 1.2, 0.5), (0.0, 0.0, 0.0)):
        validate_step(step_model, device, host, extra)
    rejected('always add submission', lambda: validate_step(lambda d, h, c: d + h + c, 12.0, 1.2))
    rejected('device synchronization charged to host', lambda: validate_step(
        lambda d, h, c: max(d, h + c), 12.0, 1.2, 0.5))
    for grid in (0, 1, 1000, 1584, 1585, 2000):
        validate_waves(grid, 12, 132)
    print('ideal timing and waves: event-schedule/packing references; no observed timings')

    validate_persistent(persistent_trace, 2, 2, 32)
    validate_persistent(persistent_trace, 1, 2, 1)
    require(persistent_trace(3, 2, 32) == [], 'queued producer should have no executable production event')
    rejected('ordinary oversubscribed persistent wait', lambda: validate_persistent(persistent_trace, 3, 2, 32))
    rejected('missing phase consumer', lambda: validate_persistent(
        lambda g, cap, layers: persistent_trace(g, cap, layers, True), 2, 2, 32))
    print('finite persistent model: 2 resident slots/3 waiting CTAs has no producer progress; not a CUDA proof')
    print('all scoped CPU model validators completed; injected defects were rejected by the same validators')


if __name__ == '__main__':
    main()
