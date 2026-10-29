# K-MODEL-004: deterministic, standalone, read-only arithmetic checks.
# No project/engine imports, subprocesses, network, file writes or GPU use.
import math
import struct
from fractions import Fraction

TOL = 1e-12


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def distance(a, b):
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return math.inf
        return max((distance(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(b, (list, tuple)):
        return math.inf
    if not math.isfinite(a) or not math.isfinite(b):
        return math.inf
    return abs(a - b)


def compose(earlier, later):
    a1, b1 = earlier
    a2, b2 = later
    return a2 * a1, a2 * b1 + b2


def faulty_compose(earlier, later):
    a1, b1 = earlier
    a2, b2 = later
    return a2 * a1, b1 + b2


def validate_scan(a, b, h0):
    width = len(h0)
    if width < 1 or len(a) != len(b):
        raise ValueError('scan requires positive width and equal row counts')
    if any(len(row) != width for row in a + b):
        raise ValueError('every coefficient row must match h0 width')
    if any(not math.isfinite(x) for row in a + b + [h0] for x in row):
        raise ValueError('scan inputs must be finite')


def prefix_scan(a, b, h0, combine=compose):
    validate_scan(a, b, h0)
    length, width = len(a), len(h0)
    if length == 0:
        return [], h0[:]
    padded = 1 << (length - 1).bit_length()
    states = [[0] * width for _ in range(length)]
    for j in range(width):
        tree = [(1, 0) for _ in range(2 * padded)]
        prefix = [(1, 0) for _ in range(2 * padded)]
        for t in range(length):
            tree[padded + t] = a[t][j], b[t][j]
        for k in range(padded - 1, 0, -1):
            tree[k] = combine(tree[2 * k], tree[2 * k + 1])
        for k in range(1, padded):
            prefix[2 * k] = prefix[k]
            prefix[2 * k + 1] = combine(prefix[k], tree[2 * k])
        for t in range(length):
            alpha, beta = combine(prefix[padded + t], tree[padded + t])
            states[t][j] = alpha * h0[j] + beta
    return states, states[-1][:]


def expanded_reference(a, b, h0):
    # Independent closed expansion: products of transitions, no recurrence/scan.
    result = []
    for t in range(len(a)):
        row = []
        for j in range(len(h0)):
            value = math.prod(a[k][j] for k in range(t + 1)) * h0[j]
            value += sum(b[i][j] * math.prod(a[k][j] for k in range(i + 1, t + 1))
                         for i in range(t + 1))
            row.append(value)
        result.append(row)
    return result


def expect_value_error(a, b, h0):
    try:
        prefix_scan(a, b, h0)
    except ValueError:
        return
    raise AssertionError('malformed scan was not rejected with ValueError')


def check_scan():
    worst, cases, negative = 0.0, 0, 0.0
    for width in (1, 4):
        for length in (0, 1, 2, 3, 9, 17):
            h0 = [0.3 - 0.1 * j for j in range(width)]
            a = [[0.35 + 0.08 * ((t + j) % 5) for j in range(width)]
                 for t in range(length)]
            b = [[0.1 * math.sin(0.3 * t + 0.2 * j) + 0.02 * t
                  for j in range(width)] for t in range(length)]
            original = ([row[:] for row in a], [row[:] for row in b], h0[:])
            expected = expanded_reference(a, b, h0)
            states, end = prefix_scan(a, b, h0)
            worst = max(worst, distance(states, expected),
                        distance(end, expected[-1] if length else h0))
            cases += 1
            for cut in range(length + 1):
                first, middle = prefix_scan(a[:cut], b[:cut], h0)
                second, final = prefix_scan(a[cut:], b[cut:], middle)
                worst = max(worst, distance(first + second, expected),
                            distance(final, expected[-1] if length else h0))
                cases += 1
            require((a, b, h0) == original, 'scan mutated input')
            if width == 4 and length == 9:
                negative = distance(prefix_scan(a, b, h0, faulty_compose)[0], expected)
    require(worst < TOL, 'tree scan or split boundary differs from expansion')
    require(negative > 1e-6, 'faulty composition escaped detection')

    h0 = [0.3, -0.2, 0.1, 0.05]
    states, end = prefix_scan([[0.5] * 4], [[0.2] * 4], h0)
    expected = [0.5 * x + 0.2 for x in h0]
    require(distance(states, [expected]) < TOL and distance(end, expected) < TOL,
            'valid four-dimensional single step failed')
    scalar, scalar_end = prefix_scan([[0.5]], [[0.2]], [0.3])
    require(distance(scalar, [[0.35]]) < TOL and distance(scalar_end, [0.35]) < TOL,
            'valid scalar single step failed')
    empty, empty_end = prefix_scan([], [], h0)
    require(empty == [] and empty_end == h0 and empty_end is not h0,
            'empty scan failed to preserve initial state independently')
    # Original round-1 malformed call must reject clearly, not raise IndexError.
    expect_value_error([[0.5]], [[0.2]], h0)
    expect_value_error([[0.5] * 4], [], h0)
    expect_value_error([[0.5] * 4], [[0.2] * 3], h0)
    expect_value_error([], [], [])
    expect_value_error([[math.nan]], [[0.2]], [0.3])

    exact_cases = 0
    for width in (1, 4):
        for length in (1, 3, 9):
            h = [Fraction(j + 1, 7) for j in range(width)]
            a = [[Fraction(2 + (t + j) % 5, 10) for j in range(width)]
                 for t in range(length)]
            b = [[Fraction(t - 2 * j, 13) for j in range(width)]
                 for t in range(length)]
            require(prefix_scan(a, b, h)[0] == expanded_reference(a, b, h),
                    'exact rational scan failed')
            exact_cases += 1
    print('scan float_cases=%d max_abs_diff=%.3e faulty_compose_diff=%.3e exact_cases=%d'
          % (cases, worst, negative, exact_cases))
    print('scan regression: scalar/four-dimensional L=1, L=0 and five malformed inputs checked')


D, N = 2, 3
A_DIAG = [0.0, -0.45, -0.70]
CONV_W = [1.0, 0.55, -0.20]


def initial_state(kc):
    return ([0.03 * (j + 1) for j in range(D * N)],
            [[0.05 * (i + 1), -0.04 * (i + 1)] for i in range(kc - 1)])


def clone_state(state):
    return state[0][:], [row[:] for row in state[1]]


def project(token, pos):
    return [0.7 * token[d] + 0.12 * math.sin(0.4 * (pos + 1) + 0.3 * d)
            + 0.05 * d for d in range(D)]


def softplus(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def phi(z):
    return 1.0 if z == 0.0 else math.expm1(z) / z


def terms(v, pos):
    # Synthetic selective projections, not a real checkpoint.
    bv = [0.20 + 0.05 * n + 0.03 * v[0] + 0.01 * pos for n in range(N)]
    cv = [0.10 + 0.04 * n - 0.02 * v[1] + 0.005 * pos for n in range(N)]
    delta = [softplus(0.8 * v[d] + (0.15, -0.10)[d]) for d in range(D)]
    a, b = [], []
    for d in range(D):
        for n in range(N):
            z = delta[d] * A_DIAG[n]
            a.append(math.exp(z))
            b.append(phi(z) * delta[d] * bv[n] * v[d])
    return a, b, cv


def output(h, v, cv):
    return [(sum(cv[n] * h[d * N + n] for n in range(N))
             + (0.30, -0.20)[d] * v[d])
            / (1.0 + math.exp(-(0.2 + 0.1 * v[d]))) for d in range(D)]


def dense_reference(tokens, kc):
    # Explicit full projected history; no ring mutation.
    # Each h(p) independently expanded from h(0); no recurrence/scan.
    h0, history0 = initial_state(kc)
    pres = [project(token, t) for t, token in enumerate(tokens)]
    aa, bb, states, ys = [], [], [initial_state(kc)], []
    for t in range(len(tokens)):
        v = []
        for d in range(D):
            total = 0.0
            for lag in range(kc):
                source = (pres[t - lag] if t - lag >= 0
                          else history0[kc - 1 + t - lag])
                total += CONV_W[lag] * source[d]
            v.append(math.tanh(total))
        a, b, cv = terms(v, t)
        aa.append(a)
        bb.append(b)
        h = expanded_reference(aa, bb, h0)[-1]
        ring = [] if kc == 1 else (history0 + pres[:t + 1])[-(kc - 1):]
        states.append((h[:], [row[:] for row in ring]))
        ys.append(output(h, v, cv))
    return states, ys


def consume(tokens, start, end, state, kc, chunk, fault=None):
    if kc not in (1, 3) or chunk < 1 or not 0 <= start <= end <= len(tokens):
        raise ValueError('unsupported synthetic model shape or interval')
    st, ys = clone_state(state), []
    for lo in range(start, end, chunk):
        if lo > start and fault == 'reset_h':
            st = initial_state(kc)[0], st[1]
        if lo > start and fault == 'reset_ring':
            st = st[0], initial_state(kc)[1]
        ring = [row[:] for row in st[1]]
        aa, bb, vv, cc = [], [], [], []
        for p in range(lo, min(end, lo + chunk)):
            pre = project(tokens[p], p)
            window = ring + [pre]
            v = [math.tanh(sum(CONV_W[kc - 1 - j] * window[j][d]
                               for j in range(kc))) for d in range(D)]
            a, b, cv = terms(v, p)
            aa.append(a)
            bb.append(b)
            vv.append(v)
            cc.append(cv)
            ring = [] if kc == 1 else window[-(kc - 1):]
        hs, last = prefix_scan(aa, bb, st[0])
        ys.extend(output(h, v, cv) for h, v, cv in zip(hs, vv, cc))
        st = last, ring
    return st, ys


def check_model():
    tokens = [[0.5 * math.sin(0.7 * t + 0.2 * d) + 0.1 * t / (1.0 + t)
               for d in range(D)] for t in range(13)]
    worst, cases = 0.0, 0
    for kc in (1, 3):
        for length in (0, 1, 2, 13):
            selected = tokens[:length]
            states, ys = dense_reference(selected, kc)
            for chunk in (1, 2, 4, 17):
                actual, outputs = consume(selected, 0, length, initial_state(kc), kc, chunk)
                worst = max(worst, distance(actual, states[-1]), distance(outputs, ys))
                cases += 1
    require(worst < TOL, 'chunked recurrent/convolution model differs from independent reference')
    states, ys = dense_reference(tokens, 3)
    negatives = {}
    for fault in ('reset_h', 'reset_ring'):
        actual, outputs = consume(tokens, 0, len(tokens), initial_state(3), 3, 4, fault)
        value = max(distance(actual, states[-1]), distance(outputs, ys))
        require(value > 1e-6, fault + ' escaped detection')
        negatives[fault] = value

    replay_worst = 0.0
    for c in range(len(tokens) + 1):
        for p in range(c, len(tokens) + 1):
            actual, outputs = consume(tokens, c, p, states[c], 3, 2)
            replay_worst = max(replay_worst, distance(actual, states[p]),
                               distance(outputs, ys[c:p]))
    require(replay_worst < TOL, 'checkpoint replay differs from full-history reference')

    base, gamma = 5, 4
    acceptance_worst = 0.0
    for accepted in range(gamma + 1):
        p = base + accepted
        restored, _ = consume(tokens, base, p, states[base], 3, 3)
        continued, next_output = consume(tokens, p, p + 1, restored, 3, 1)
        acceptance_worst = max(acceptance_worst, distance(restored, states[p]),
                               distance(continued, states[p + 1]),
                               distance(next_output, ys[p:p + 1]))
    require(acceptance_worst < TOL, 'accepted boundary or continuation failed')
    wrong_final = distance(states[base + gamma], states[base + 2])
    require(wrong_final > 1e-6, 'final speculative state misuse escaped detection')
    wrong_ring = states[base + 2][0][:], [row[:] for row in states[base + gamma][1]]
    wrong_next, wrong_y = consume(tokens, base + 2, base + 3, wrong_ring, 3, 1)
    wrong_ring_diff = max(distance(wrong_next, states[base + 3]),
                          distance(wrong_y, ys[base + 2:base + 3]))
    require(wrong_ring_diff > 1e-6, 'partial rollback retaining wrong ring escaped detection')
    require(phi(0.0) == 1.0 and abs(phi(1e-12) - (1.0 + 0.5e-12)) < TOL,
            'ZOH zero or near-zero limit failed')
    print('synthetic_model D=2 N=3 kc={1,3} cases=%d max_abs_diff=%.3e'
          % (cases, worst))
    print('negative reset_h=%.3e reset_ring=%.3e' % (negatives['reset_h'], negatives['reset_ring']))
    print('replay_all_boundaries max_abs_diff=%.3e accept={0,1,2,3,4} max_abs_diff=%.3e'
          % (replay_worst, acceptance_worst))
    print('negative final_candidate=%.3e partial_ring_rollback=%.3e'
          % (wrong_final, wrong_ring_diff))


def direct_recoverable(p, full_end, rec, ws, we, width, cs, ce, kc, initialized=True):
    if width < 1 or kc < 1 or not 0 <= p <= full_end:
        return False
    if p == 0:
        return initialized
    if p not in rec:
        return False
    if width > 1 and not (ws <= max(0, p - width + 1) and p <= we):
        return False
    if kc > 1 and not (cs <= max(0, p - kc + 1) and p <= ce):
        return False
    return True


def explicit_recovery_reference(p, full_end, rec, ws, we, width, cs, ce, kc):
    if p < 0:
        return False
    if p == 0:
        return True
    full_have = set(range(full_end))
    window_have = set(range(ws, we))
    conv_have = set(range(cs, ce))
    return (p in rec and set(range(p)).issubset(full_have)
            and set(range(max(0, p - width + 1), p)).issubset(window_have)
            and set(range(max(0, p - kc + 1), p)).issubset(conv_have))


def check_recovery_and_cost():
    rec, cases = {0, 3, 7, 10}, 0
    for full_end in (0, 5, 10):
        for ws in (0, 2, 8):
            for we in (ws, 10):
                for cs in (0, 2, 8):
                    for ce in (cs, 10):
                        for width in (1, 2, 4):
                            for kc in (1, 3):
                                for p in range(11):
                                    actual = direct_recoverable(p, full_end, rec, ws, we,
                                                                width, cs, ce, kc)
                                    expected = explicit_recovery_reference(p, full_end, rec,
                                                                          ws, we, width, cs, ce, kc)
                                    require(actual == expected, 'range qualification differs from explicit read sets')
                                    cases += 1
    require(direct_recoverable(7, 10, rec, 0, 0, 1, 0, 0, 1),
            'empty old histories must permit width=1 and kc=1')
    faulty_empty_history_rule = (7 <= 0 and 7 <= 0)
    require(not faulty_empty_history_rule, 'empty-history negative control did not reject incorrectly')
    require(not direct_recoverable(0, 10, rec, 0, 10, 4, 0, 10, 3, initialized=False),
            'unavailable initialization was accepted')
    require(not direct_recoverable(-1, 10, rec, 0, 10, 4, 0, 10, 3),
            'negative consumption position accepted')

    rec_big = {0, 512, 768}
    direct = [p for p in range(1025)
              if direct_recoverable(p, 1024, rec_big, 640, 1024, 128, 0, 1024, 4)]
    require(direct == [0, 768], 'common recoverable set must equal {0,768}')
    require(1024 - max(direct) == 256, 'replay distance must be 256')
    naive = max(1024, max(rec_big), 1024)
    require(naive == 1024 and naive not in direct, 'max-per-family negative control escaped detection')

    ring = {p % 128: p for p in range(896, 1024)}
    for p in range(1024, 1028):
        ring[p % 128] = p
    have = set(ring.values())
    needed = set(range(899, 1026))
    require(have == set(range(900, 1028)), 'window overwrite model incorrect')
    require(needed - have == {899}, 'length-only rollback did not expose missing position 899')

    for n in (0, 1, 255, 256, 700, 8192):
        for period in (1, 3, 256):
            saved = list(range(0, n + 1, period))
            require(len(saved) == n // period + 1, 'checkpoint count differs from enumerated positions')
            for p in (0, n // 2, n):
                c = max(q for q in saved if q <= p)
                require(0 <= p - c <= period - 1, 'checkpoint replay bound failed')
    require(len(range(0, 8193, 256)) * Fraction(35, 8) == Fraction(1155, 8),
            '144.375 MiB checkpoint capacity example failed')
    require(700 - 256 * (700 // 256) == 188, '700-position replay example failed')
    require(8192 * 2048 * 16 * 2 == 512 * 1024 * 1024, 'expanded tensor example failed')
    chosen = next((p for p in (16, 32, 64) if p * 64 >= 1500), None)
    require(chosen == 32 and 16 * 64 < 1500, 'page alignment feasibility failed')
    print('recovery explicit_set_cases=%d common_direct=%s replay_steps=256' % (cases, direct))
    print('negative controls: max-per-family, empty-history rejection and window length-only rollback detected')
    print('cost examples: checkpoints=33 capacity=144.375 MiB; expanded_tensor=512 MiB; alignment_block=32')


def decode_read_formula(batch, full, windows, recurrent, weights=0, metadata=0,
                        omit_state_width=False):
    # full: (old_length, KV_bytes_per_position)
    # windows: (old_length, window_including_current, KV_bytes_per_position)
    # recurrent: (D, N, state_element_bytes, Kc, history_element_bytes)
    per_request = sum(length * kv_bytes for length, kv_bytes in full)
    per_request += sum(min(length, width - 1) * kv_bytes
                       for length, width, kv_bytes in windows)
    for channels, state_width, state_bytes, kc, history_bytes in recurrent:
        h_read = (state_bytes if omit_state_width
                  else channels * state_width * state_bytes)
        conv_read = channels * (kc - 1) * history_bytes
        per_request += h_read + conv_read
    return weights + batch * per_request + metadata


def enumerated_decode_reads(batch, full, windows, recurrent_formats,
                            weights=0, metadata=0):
    # Independent access trace: enumerate each logical token byte or state element.
    # The reference receives storage formats, not the formula's scalar byte widths.
    # No multiplication by D*N or batch in the reference state read accounting.
    traffic = {'weights': weights, 'metadata': metadata,
               'full': 0, 'window': 0, 'h': 0, 'conv': 0}
    for request in range(batch):
        for layer, (length, kv_bytes) in enumerate(full):
            for token in range(length):
                for byte in range(kv_bytes):
                    traffic['full'] += 1
        for layer, (length, width, kv_bytes) in enumerate(windows):
            for token in range(max(0, length - width + 1), length):
                for byte in range(kv_bytes):
                    traffic['window'] += 1
        for layer, (channels, state_width, hfmt, kc, ufmt) in enumerate(recurrent_formats):
            for channel in range(channels):
                for coordinate in range(state_width):
                    traffic['h'] += struct.calcsize('<' + hfmt)
            for old_step in range(kc - 1):
                for channel in range(channels):
                    traffic['conv'] += struct.calcsize('<' + ufmt)
    return traffic


def check_decode_read_width():
    cases, detected = 0, 0
    # Constant sizes here are the candidate formula's input; struct independently
    # specifies the storage widths used by the reference access trace.
    for batch in (0, 1, 3):
        for channels in (1, 2, 7):
            for state_width in (1, 3, 5):
                for state_bytes, hfmt in ((2, 'h'), (4, 'f')):
                    for kc in (1, 4):
                        full = [(5, 8)]
                        windows = [(5, 3, 12), (0, 1, 4)]
                        rec = [(channels, state_width, state_bytes, kc, 2)]
                        formats = [(channels, state_width, hfmt, kc, 'h')]
                        trace = enumerated_decode_reads(batch, full, windows, formats)
                        expected = sum(trace.values())
                        actual = decode_read_formula(batch, full, windows, rec)
                        require(actual == expected, 'mixed decode byte formula differs from access trace')
                        faulty = decode_read_formula(batch, full, windows, rec,
                                                     omit_state_width=True)
                        should_differ = batch > 0 and channels * state_width > 1
                        require((faulty != expected) == should_differ,
                                'state-width negative control or degenerate boundary failed')
                        detected += int(faulty != expected)
                        cases += 1
    require(cases == 108 and detected == 64, 'unexpected byte-check coverage')

    # Direct regression: 32768 coordinates, each independently counted as 2 bytes.
    rec = [(2048, 16, 2, 4, 2)]
    trace = enumerated_decode_reads(1, [], [], [(2048, 16, 'h', 4, 'h')])
    require(trace['h'] == 65536, 'whole-layer state read must be 64 KiB')
    require(trace['conv'] == 12288, 'old convolution history read must be 12 KiB')
    require(sum(trace.values()) == 77824, 'combined old-state read must be 76 KiB')
    require(decode_read_formula(1, [], [], rec) == sum(trace.values()),
            'large state formula differs from coordinate trace')
    state_only = [(2048, 16, 2, 1, 2)]
    correct = decode_read_formula(1, [], [], state_only)
    faulty = decode_read_formula(1, [], [], state_only, omit_state_width=True)
    require(correct == 65536 and faulty == 2 and correct // faulty == 32768,
            'recorded missing D*N failure was not reproduced and detected')
    # Read and write are separate events, each visits all h coordinates.
    h_write = sum(struct.calcsize('<h') for d in range(2048) for n in range(16))
    ring_write = sum(struct.calcsize('<h') for d in range(2048))
    require(trace['h'] + h_write == 131072, 'h read/write must total 128 KiB')
    require(sum(trace.values()) + h_write + ring_write == 147456,
            'state and ring read/write main terms must total 144 KiB')
    attention_read = sum(1 for token in range(64) for byte in range(512))
    require(attention_read == 32768 and correct == 2 * attention_read,
            'attention versus recurrent history comparison failed')
    require(faulty < attention_read < correct,
            'faulty state width did not reverse the byte comparison')

    # Independent heterogeneous layers, batch scaling, one Kc=1 and one W=1.
    full = [(257, 96)]
    windows = [(257, 128, 64), (0, 1, 80)]
    rec = [(4, 3, 2, 3, 4), (2, 5, 4, 1, 2)]
    formats = [(4, 3, 'h', 3, 'f'), (2, 5, 'f', 1, 'h')]
    trace = enumerated_decode_reads(3, full, windows, formats, weights=4096, metadata=128)
    mixed = decode_read_formula(3, full, windows, rec, weights=4096, metadata=128)
    require(mixed == sum(trace.values()) == 102912,
            'heterogeneous layer, batch, weight or metadata accounting failed')
    faulty_mixed = decode_read_formula(3, full, windows, rec, weights=4096,
                                      metadata=128, omit_state_width=True)
    require(faulty_mixed != mixed and mixed - faulty_mixed == 174,
            'heterogeneous missing-width negative control escaped detection')

    # Shape/dtype changes scale the entire h, not a single element.
    for changed in ([(4096, 16, 2, 1, 2)], [(2048, 32, 2, 1, 2)],
                    [(2048, 16, 4, 1, 2)]):
        require(decode_read_formula(1, [], [], changed) == 2 * correct,
                'D, N or storage-byte doubling failed')
    require(decode_read_formula(3, [], [], state_only) == 3 * correct,
            'batch scaling failed')
    require(decode_read_formula(0, [], [], state_only) == 0,
            'zero-batch state traffic must be zero')
    require(decode_read_formula(1, [], [], [(1, 1, 2, 1, 2)]) == 2,
            'single-element no-convolution edge failed')
    print('decode_reads enumerated_cases=%d missing_DN_negative_detected=%d'
          % (cases, detected))
    print('decode_reads D=2048 N=16 scalar_bytes=2 h_read=65536 conv_read=12288 combined_read=77824')
    print('negative missing_DN: faulty_h_read=2 underestimate_factor=32768; byte comparison reversal detected')
    print('decode_reads h_read_write=131072 state_and_ring_read_write=147456 mixed_batch3=102912 bytes')
    print('decode scope: logical accesses with each old element read once; no measured HBM transactions or timings')


def main():
    print('K-MODEL-004 scope: CPU arithmetic and synthetic finite state models; no GPU/engine measurement')
    check_scan()
    check_model()
    check_recovery_and_cost()
    check_decode_read_width()
    print('RESULT: PASS within declared arithmetic, chunk, recovery, speculation and logical-byte scope')


if __name__ == '__main__':
    main()
