import math
from fractions import Fraction


class DomainError(ValueError):
    pass


class Mismatch(AssertionError):
    pass


class HarnessError(RuntimeError):
    pass


def close(actual, expected):
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise Mismatch('sequence shape differs')
        for a, e in zip(actual, expected):
            close(a, e)
    elif not (math.isfinite(actual) and math.isfinite(expected)
              and math.isclose(actual, expected, rel_tol=2e-11, abs_tol=2e-12)):
        raise Mismatch('value differs: %r versus %r' % (actual, expected))


def cycle(name, verifier, good, bad):
    verifier(good)
    try:
        verifier(bad)
    except Mismatch:
        pass
    else:
        raise HarnessError(name + ': faulty implementation accepted')
    verifier(good)
    print(name + ': correct accepted; faulty rejected; restored accepted')


def expect_domain(action):
    try:
        action()
    except DomainError:
        return
    raise HarnessError('illegal input accepted')


def geometry(f, height, width):
    if any(type(x) is not int or x <= 0 for x in (f, height, width)):
        raise DomainError('positive integer geometry required')
    if (f - 1) % 4 or height % 16 or width % 16:
        raise DomainError('reference first-frame and patch contract')
    tl = 1 + (f - 1) // 4
    return tl, height // 8, width // 8, tl * (height // 16) * (width // 16)


def bad_geometry(f, height, width):
    tl, hl, wl, _ = geometry(f, height, width)
    wrong_tl = f // 4
    return wrong_tl, hl, wl, wrong_tl * (height // 16) * (width // 16)


def verify_geometry(fn):
    for f in (1, 5, 9, 13, 81):
        for height, width in ((16, 32), (32, 16), (720, 1280)):
            groups = [[0]] + [list(range(i, i + 4)) for i in range(1, f, 4)]
            temporal = len(groups)
            spatial = sum(1 for _i in range(0, height, 16)
                          for _j in range(0, width, 16))
            close(fn(f, height, width),
                  (temporal, height // 8, width // 8, temporal * spatial))


def validate_partition(n, parts):
    flat = [j for part in parts for j in part]
    if any(type(j) is not int or j < 0 or j >= n for j in flat):
        raise DomainError('invalid key index')
    if sorted(flat) != list(range(n)):
        raise DomainError('missing or repeated key')


def dense_attention(q, keys, values, allowed):
    indices = [j for j in range(len(keys)) if j in allowed]
    if not indices:
        raise DomainError('empty legal row')
    scores = [math.fsum(q[c] * keys[j][c] for c in range(len(q)))
              / math.sqrt(len(q)) for j in indices]
    peak = max(scores)
    weights = [math.exp(s - peak) for s in scores]
    total = math.fsum(weights)
    return [math.fsum(w * values[j][c] for j, w in zip(indices, weights)) / total
            for c in range(len(values[0]))]


def attention(q, keys, values, parts, allowed, wrong_average=False):
    validate_partition(len(keys), parts)
    m, mass = -math.inf, 0.0
    numerator = [0.0] * len(values[0])
    local_outputs = []
    for part in parts:
        pm, pl = -math.inf, 0.0
        pa = [0.0] * len(numerator)
        for j in part:
            if j not in allowed:
                continue
            score = sum(a * b for a, b in zip(q, keys[j])) / math.sqrt(len(q))
            if not math.isfinite(score):
                raise DomainError('nonfinite score')
            nm = max(pm, score)
            scale = math.exp(pm - nm) if pl else 0.0
            weight = math.exp(score - nm)
            pa = [scale * a + weight * v for a, v in zip(pa, values[j])]
            pl = scale * pl + weight
            pm = nm
        if not pl:
            continue
        local_outputs.append([a / pl for a in pa])
        nm = max(m, pm)
        left = math.exp(m - nm) if mass else 0.0
        right = math.exp(pm - nm)
        numerator = [left * a + right * b for a, b in zip(numerator, pa)]
        mass = left * mass + right * pl
        m = nm
    if not mass:
        raise DomainError('empty legal row')
    if wrong_average:
        return [sum(o[c] for o in local_outputs) / len(local_outputs)
                for c in range(len(numerator))]
    return [a / mass for a in numerator]


KEYS = [[1.0, 0.0], [-1.0, 2.0], [2.0, -1.0], [0.0, 1.0], [1.0, 3.0]]
VALUES = [[1.0, 2.0], [4.0, -1.0], [-2.0, 3.0], [7.0, 0.0], [0.0, -5.0]]
PARTS = [[0, 2], [], [1], [3, 4]]


def verify_attention(fn):
    layouts = [PARTS, [[4, 3], [2, 1, 0]], [[], list(range(5)), []]]
    for q in ([3.0, -2.0], [1000.0, -700.0], [0.0, 0.0]):
        for parts in layouts:
            for allowed in (set(range(5)), {0, 3}, {2}):
                close(fn(q, KEYS, VALUES, parts, allowed),
                      dense_attention(q, KEYS, VALUES, allowed))


def positioned(parts, reset=False):
    keys = [None] * 6
    values = [[j * j - 3.0 * j, float((-1) ** j)] for j in range(6)]
    for part in parts:
        for local, global_j in enumerate(part):
            position = local if reset else global_j
            time, column = divmod(position, 3)
            keys[global_j] = [time + 0.1 * column, 0.2 * time - 0.3 * column]
    return attention([0.3, -0.7], keys, values, parts, set(range(6)))


def verify_positions(fn):
    keys = [[j // 3 + 0.1 * (j % 3), 0.2 * (j // 3) - 0.3 * (j % 3)]
            for j in range(6)]
    values = [[j * j - 3.0 * j, float((-1) ** j)] for j in range(6)]
    expected = dense_attention([0.3, -0.7], keys, values, set(range(6)))
    for parts in ([[0, 1, 2], [3, 4, 5]], [[0, 2, 4], [], [1, 3, 5]]):
        close(fn(parts), expected)


def dense_conv(x, weights, stride=1):
    return [math.fsum(w * (x[t - j] if t >= j else 0.0)
                      for j, w in enumerate(weights))
            for t in range(0, len(x), stride)]


class StreamConv:
    def __init__(self, weights, stride=1, reset_phase=False):
        if not weights or type(stride) is not int or stride <= 0:
            raise DomainError('invalid convolution')
        self.weights = list(weights)
        self.stride = stride
        self.reset_phase = reset_phase
        self.seen = 0
        self.next_position = 0
        self.history = {}

    def push(self, chunk):
        if self.reset_phase:
            self.next_position = self.seen
        outputs = []
        for value in chunk:
            self.history[self.seen] = value
            self.seen += 1
            while self.next_position < self.seen:
                value_out = 0.0
                for j, weight in enumerate(self.weights):
                    index = self.next_position - j
                    if index >= 0:
                        value_out += weight * self.history[index]
                outputs.append(value_out)
                self.next_position += self.stride
            cutoff = self.seen - (len(self.weights) - 1)
            self.history = {i: v for i, v in self.history.items() if i >= cutoff}
        return outputs


def compositions(n):
    if n == 0:
        yield []
        return
    for mask in range(1 << (n - 1)):
        result, start = [], 0
        for i in range(1, n):
            if mask & (1 << (i - 1)):
                result.append(i - start)
                start = i
        yield result + [n - start]


def stream_conv(x, groups, stride, reset_phase=False):
    conv = StreamConv([2.0, -1.0, 3.0], stride, reset_phase)
    offset, outputs = 0, []
    for size in groups:
        outputs.extend(conv.push(x[offset:offset + size]))
        offset += size
    if offset != len(x):
        raise DomainError('chunk coverage')
    return outputs


def verify_conv(fn):
    for n in range(7):
        x = [((i * 7) % 11) - 5.0 + 0.1 * i for i in range(n)]
        for middle in compositions(n):
            groups = [0] + middle + [0]
            for stride in (1, 2, 3):
                close(fn(x, groups, stride), dense_conv(x, [2.0, -1.0, 3.0], stride))


def dense_decode(z, r):
    expanded = []
    for i, value in enumerate(z):
        expanded.extend([value] * (1 if i == 0 else r))
    hidden = [math.tanh(v) for v in dense_conv(expanded, [0.5, -0.25, 0.125])]
    return dense_conv(hidden, [0.7, 0.2, -0.1])


def decode(z, groups, r, bug=None):
    if type(r) is not int or r < 1 or any(type(n) is not int or n < 0 for n in groups):
        raise DomainError('invalid decode plan')
    if sum(groups) != len(z):
        raise DomainError('latent coverage')
    first = StreamConv([0.5, -0.25, 0.125])
    second = StreamConv([0.7, 0.2, -0.1])
    index, outputs = 0, []
    for size in groups:
        if bug == 'reset':
            first = StreamConv([0.5, -0.25, 0.125])
            second = StreamConv([0.7, 0.2, -0.1])
        expanded = []
        for _ in range(size):
            copies = 1 if index == 0 and bug != 'first' else r
            expanded.extend([z[index]] * copies)
            index += 1
        hidden = [math.tanh(v) for v in first.push(expanded)]
        outputs.extend(second.push(hidden))
    return outputs


def verify_decode(fn):
    close(fn([], [0], 4), [])
    close(fn([2.0], [0, 1, 0], 4), dense_decode([2.0], 4))
    z = [1.0, 2.0, -1.0, 0.5, 3.0]
    for r in (1, 2, 4):
        for groups in ([1, 0, 2, 2], [2, 1, 2], [5], [0, 1, 1, 1, 1, 1, 0]):
            close(fn(z, groups, r), dense_decode(z, r))
            close(len(fn(z, groups, r)), 1 + r * (len(z) - 1))


def flow_step(z, velocity, t, target, wrong_sign=False):
    if not (math.isfinite(t) and math.isfinite(target) and 0 <= t < target <= 1):
        raise DomainError('invalid flow boundaries')
    if len(z) != len(velocity) or any(not math.isfinite(v) for v in z + velocity):
        raise DomainError('invalid flow state')
    delta = (t - target) if wrong_sign else (target - t)
    return [x + delta * v for x, v in zip(z, velocity)]


def verify_flow(fn):
    noise, data = [2.0, -1.0, 0.25], [-3.0, 4.0, 1.5]
    velocity = [y - x for x, y in zip(noise, data)]
    for times in ([0.0, 1.0], [0.0, 0.1, 0.45, 1.0], [0.0, 0.25, 0.75, 1.0]):
        state = list(noise)
        for t, target in zip(times, times[1:]):
            state = fn(state, velocity, t, target)
            analytic = [(1 - target) * x + target * y for x, y in zip(noise, data)]
            close(state, analytic)
        close(state, data)


def ring_bytes(n, d, p, element_bytes, double_receive=False):
    if any(type(v) is not int or v <= 0 for v in (n, d, p, element_bytes)) or n % p:
        raise DomainError('uniform positive ring geometry required')
    chunk = 2 * (n // p) * d * element_bytes
    receive = (p - 1) * chunk
    send = (p - 1) * chunk
    temporaries = 2 * chunk if p > 1 else 0
    return (2 * receive if double_receive else receive), send, temporaries


def verify_ring(fn):
    for p in (1, 2, 4, 8):
        n, d, b = 8, 3, 2
        rows = [list(range(r * (n // p), (r + 1) * (n // p))) for r in range(p)]
        rank = 0
        received = sum(2 * len(rows[owner]) * d * b for owner in range(p) if owner != rank)
        sent = sum(2 * len(rows[(rank - step) % p]) * d * b for step in range(p - 1))
        temporary = sum(2 * len(rows[0]) * d * b for _ in range(2)) if p > 1 else 0
        close(fn(n, d, p, b), (received, sent, temporary))
    receive, send, temporary = fn(75600, 1536, 4, 2)
    close(receive, 348364800)
    close(send, 348364800)
    close(temporary, 232243200)


def main():
    print('Scope: deterministic stdlib scalar models, offline and read-only.')
    cycle('first-frame geometry', verify_geometry, geometry, bad_geometry)
    cycle('global softmax merge', verify_attention, attention,
          lambda q, k, v, p, a: attention(q, k, v, p, a, True))
    cycle('global position identity', verify_positions, positioned,
          lambda p: positioned(p, True))
    cycle('convolution history and stride phase', verify_conv, stream_conv,
          lambda x, g, s: stream_conv(x, g, s, True))
    cycle('per-layer decode history', verify_decode, decode,
          lambda z, g, r: decode(z, g, r, 'reset'))
    cycle('decode first-frame emission', verify_decode, decode,
          lambda z, g, r: decode(z, g, r, 'first'))
    cycle('flow direction and nonuniform boundaries', verify_flow, flow_step,
          lambda z, v, t, s: flow_step(z, v, t, s, True))
    cycle('ring directional payload and temporary capacity', verify_ring, ring_bytes,
          lambda n, d, p, b: ring_bytes(n, d, p, b, True))
    for args in ((0, 16, 16), (80, 720, 1280), (5, 17, 16), (5, 16, 0)):
        expect_domain(lambda args=args: geometry(*args))
    for parts in ([[0, 1], [2, 3], []], [[0, 2, 4], [1, 4], [3]], [[-1], [0, 1, 2, 3, 4]]):
        expect_domain(lambda parts=parts: attention([3.0, -2.0], KEYS, VALUES, parts, set(range(5))))
    expect_domain(lambda: attention([3.0, -2.0], KEYS, VALUES, PARTS, set()))
    expect_domain(lambda: StreamConv([1.0], 0))
    expect_domain(lambda: decode([1.0], [0], 4))
    expect_domain(lambda: flow_step([1.0], [2.0], 0.7, 0.2))
    expect_domain(lambda: flow_step([1.0], [2.0], float('nan'), 1.0))
    expect_domain(lambda: ring_bytes(7, 3, 4, 2))
    receive = ring_bytes(75600, 1536, 4, 2)[0]
    observations = {
        'video_geometry': geometry(81, 720, 1280),
        'hidden_MiB': Fraction(75600 * 1536 * 2, 2 ** 20),
        'single_head_scores_GiB': Fraction(75600 ** 2 * 2, 2 ** 30),
        'ring_receive_MiB_per_layer_branch_rank': Fraction(receive, 2 ** 20),
        'assumed_25_GiB_s_receive_ms': Fraction(receive * 1000, 25 * 2 ** 30),
        'self_pair_TFLOPs_per_layer_branch': Fraction(4 * 75600 ** 2 * 1536, 10 ** 12),
        'pts_span_seconds': Fraction(80, 16),
        'full_frame_duration_seconds': Fraction(81, 16),
        'reference_decode_frames': len(dense_decode([1.0, 2.0, -1.0, 0.5, 3.0], 4))
    }
    print('All specified scalar checks and negative controls completed.')
    for name, value in observations.items():
        print(name + ': ' + str(value))
    print('No checkpoint execution, GPU measurements, codec validation, performance or quality claim.')


if __name__ == '__main__':
    main()
