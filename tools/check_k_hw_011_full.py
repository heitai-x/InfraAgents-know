import math
from decimal import Decimal, localcontext


class CheckError(AssertionError):
    pass


def req(condition, message):
    if not condition:
        raise CheckError(message)


def close(a, b, label):
    req(math.isfinite(a) and math.isfinite(b), label + ': nonfinite')
    req(abs(a - b) <= 1e-10 * max(1.0, abs(a), abs(b)), label)


def rejected(check, label):
    try:
        check()
    except CheckError:
        return
    raise CheckError(label + ': faulty variant escaped observer')


def matrix(rows, cols, seed):
    return [[float((17 * r + 13 * c + 7 * seed) % 19 - 9)
             for c in range(cols)] for r in range(rows)]


def dense_gemm(A, B, M, N, K):
    return [[math.fsum(A[r][k] * B[k][c] for k in range(K))
             for c in range(N)] for r in range(M)]


def simt_gemm(A, B, M, N, K, omit_last=False):
    out = [[0.0] * N for _ in range(M)]
    writers = [[0] * N for _ in range(M)]
    ra = rb = products = 0
    gm, gn, stages = (M + 15) // 16, (N + 31) // 32, (K + 7) // 8
    for bm in range(gm):
        for bn in range(gn):
            m0, n0 = 16 * bm, 32 * bn
            acc = [[0.0] * 8 for _ in range(64)]
            for j in range(max(0, stages - int(omit_last))):
                sa, sb = [0.0] * 128, [0.0] * 256
                for tau in range(64):
                    for v in range(2):
                        a = tau + 64 * v
                        r, k = divmod(a, 8)
                        if m0 + r < M and 8 * j + k < K:
                            sa[a] = A[m0 + r][8 * j + k]
                            ra += 1
                    for v in range(4):
                        b = tau + 64 * v
                        k, c = divmod(b, 32)
                        if 8 * j + k < K and n0 + c < N:
                            sb[b] = B[8 * j + k][n0 + c]
                            rb += 1
                for tau in range(64):
                    w, lane = divmod(tau, 32)
                    for u in range(8):
                        r = 2 * u + lane // 16
                        c = 16 * w + lane % 16
                        for k in range(8):
                            acc[tau][u] += sa[8 * r + k] * sb[32 * k + c]
                            products += 1
            for tau in range(64):
                w, lane = divmod(tau, 32)
                for u in range(8):
                    r, c = 2 * u + lane // 16, 16 * w + lane % 16
                    if m0 + r < M and n0 + c < N:
                        out[m0 + r][n0 + c] = acc[tau][u]
                        writers[m0 + r][n0 + c] += 1
    return out, writers, products, ra, rb


def observe_gemm(M, N, K, omit_last=False):
    A, B = matrix(M, K, 1), matrix(K, N, 2)
    got, writers, products, ra, rb = simt_gemm(A, B, M, N, K, omit_last)
    ref = dense_gemm(A, B, M, N, K)
    for r in range(M):
        for c in range(N):
            close(got[r][c], ref[r][c], 'SIMT output')
            req(writers[r][c] == 1, 'unique output writer')
    gm, gn, stages = (M + 15) // 16, (N + 31) // 32, (K + 7) // 8
    req(products == 16 * 32 * 8 * gm * gn * stages, 'executed products')
    req(ra == M * K * gn, 'A requests')
    req(rb == K * N * gm, 'B requests')


def bank_degree(words):
    banks = {}
    for word in words:
        banks.setdefault(word % 32, set()).add(word)
    return max((len(addresses) for addresses in banks.values()), default=0)


def shared_a_words(v, stride):
    return [(a // 8) * stride + a % 8 for a in
            (lane + 64 * v for lane in range(32))]


def observe_banks(stride=8):
    for v in range(2):
        req(bank_degree(shared_a_words(v, stride)) == 1, 'A write bank conflict')
    for v in range(4):
        req(bank_degree([lane + 64 * v for lane in range(32)]) == 1,
            'B write banks')
    for u in range(8):
        for k in range(8):
            req(bank_degree([(2 * u + lane // 16) * 8 + k
                             for lane in range(32)]) == 1, 'A broadcasts')
            for w in range(2):
                req(bank_degree([32 * k + 16 * w + lane % 16
                                 for lane in range(32)]) == 1, 'B broadcasts')


class Slots:
    def __init__(self, depth, guard_copy=True, guard_done=True,
                 release_one=False, publish_early=False):
        self.depth = depth
        self.guard_copy = guard_copy
        self.guard_done = guard_done
        self.release_one = release_one
        self.publish_early = publish_early
        self.slots = [dict(identity=None, status='EMPTY', parts=set(),
                           copied=False, readers=set(), active=set())
                      for _ in range(depth)]

    def submit(self, stage):
        s, g = stage % self.depth, stage // self.depth
        st = self.slots[s]
        if st['status'] != 'EMPTY':
            raise ValueError('submission to occupied slot')
        st.update(identity=(s, g, stage), status='FILLING', parts=set(),
                  copied=False, readers=set(), active=set())
        return st['identity']

    def copy_done(self, ticket, part):
        st = self.slots[ticket[0]]
        if self.guard_copy and ticket != st['identity']:
            return
        if st['status'] != 'FILLING':
            return
        if part not in ('A', 'B'):
            raise ValueError('unknown completion component')
        st['parts'].add(part)
        st['copied'] = st['parts'] == {'A', 'B'}
        if st['copied'] or self.publish_early:
            st['status'] = 'READY'
            st['readers'] = {0, 1}

    def read_start(self, ticket, reader):
        st = self.slots[ticket[0]]
        if ticket != st['identity']:
            return
        if st['status'] not in ('READY', 'READING') or not st['copied']:
            raise ValueError('read before complete publication')
        if reader not in st['readers']:
            raise ValueError('reader has no responsibility')
        st['active'].add(reader)
        st['status'] = 'READING'

    def read_done(self, ticket, reader):
        st = self.slots[ticket[0]]
        if self.guard_done and ticket != st['identity']:
            return
        if reader not in st['active']:
            return
        st['active'].remove(reader)
        st['readers'].remove(reader)
        if not st['readers'] or self.release_one:
            st['status'] = 'EMPTY'
            if self.release_one:
                st['readers'].clear()
                st['active'].clear()

    def snapshot(self, index):
        st = self.slots[index]
        return (st['identity'], st['status'], frozenset(st['parts']),
                st['copied'], frozenset(st['readers']), frozenset(st['active']))


def expected_snapshot(identity, status, parts, copied, readers=(), active=()):
    return (identity, status, frozenset(parts), copied,
            frozenset(readers), frozenset(active))


def observe_slots(**fault):
    # Observations depend on public state, never on mutant-generated labels.
    for depth in (1, 2, 3):
        for order in ((0, 1), (1, 0)):
            engine = Slots(depth, **fault)
            old = []
            for s in range(depth):
                ticket = engine.submit(s)
                old.append(ticket)
                engine.copy_done(ticket, 'A')
                req(engine.snapshot(s) == expected_snapshot(
                    ticket, 'FILLING', ('A',), False), 'partial copy published')
                before = engine.snapshot(s)
                engine.copy_done(ticket, 'A')
                req(engine.snapshot(s) == before, 'duplicate component changed state')
                engine.copy_done(ticket, 'B')
                req(engine.snapshot(s) == expected_snapshot(
                    ticket, 'READY', ('A', 'B'), True, (0, 1)), 'initial ready')
            for s, ticket in enumerate(old):
                engine.read_start(ticket, 0)
                engine.read_start(ticket, 1)
                first, last = order
                engine.read_done(ticket, first)
                req(engine.snapshot(s) == expected_snapshot(
                    ticket, 'READING', ('A', 'B'), True, (last,), (last,)),
                    'released before last reader')
                before = engine.snapshot(s)
                engine.read_done(ticket, first)
                req(engine.snapshot(s) == before, 'duplicate reader completion')
                engine.read_done(ticket, last)
                req(engine.snapshot(s) == expected_snapshot(
                    ticket, 'EMPTY', ('A', 'B'), True), 'old generation not drained')
                current = engine.submit(s + depth)
                engine.copy_done(current, 'A')
                req(engine.snapshot(s) == expected_snapshot(
                    current, 'FILLING', ('A',), False), 'new generation initialization')
                before = engine.snapshot(s)
                engine.copy_done(ticket, 'B')
                req(engine.snapshot(s) == before,
                    'stale copy changed new identity/parts/copied/status/readers')
                engine.copy_done(current, 'B')
                req(engine.snapshot(s) == expected_snapshot(
                    current, 'READY', ('A', 'B'), True, (0, 1)),
                    'genuine current ticket must still publish')
                engine.read_start(current, 0)
                engine.read_start(current, 1)
                before = engine.snapshot(s)
                engine.read_done(ticket, first)
                req(engine.snapshot(s) == before, 'stale read completion changed new readers')
                engine.read_done(current, first)
                req(engine.snapshot(s) == expected_snapshot(
                    current, 'READING', ('A', 'B'), True, (last,), (last,)),
                    'new generation released before last reader')
                engine.read_done(current, last)
                req(engine.snapshot(s) == expected_snapshot(
                    current, 'EMPTY', ('A', 'B'), True), 'current generation progress')


def wmma_contract(shape=(16, 16, 16), dtypes=('fp16', 'fp16', 'fp32'),
                  bases=(0, 512, 1024), lds=(16, 16, 16), lanes=32,
                  uniform=True, skip_ld=False):
    # Contract for this chapter's row-major WMMA16 candidate only.
    if shape != (16, 16, 16) or dtypes != ('fp16', 'fp16', 'fp32'):
        return False
    if lanes != 32 or uniform is not True:
        return False
    if len(bases) != 3 or len(lds) != 3:
        return False
    if any(type(p) is not int or p < 0 or p % 32 for p in bases):
        return False
    if not skip_ld:
        for ld, multiple in zip(lds, (8, 8, 4)):
            if type(ld) is not int or ld < 16 or ld % multiple:
                return False
    return True


def observe_wmma(skip_ld=False):
    examples = [({}, True), ({'lds': (24, 32, 20)}, True),
                ({'lds': (15, 16, 16)}, False),
                ({'lds': (20, 16, 16)}, False),
                ({'lds': (16, 20, 16)}, False),
                ({'lds': (16, 16, 18)}, False),
                ({'lds': (8, 16, 16)}, False),
                ({'bases': (2, 512, 1024)}, False),
                ({'lanes': 31}, False), ({'uniform': False}, False),
                ({'shape': (16, 16, 8)}, False),
                ({'dtypes': ('fp32', 'fp16', 'fp32')}, False)]
    for parameters, wanted in examples:
        req(wmma_contract(skip_ld=skip_ld, **parameters) == wanted,
            'parameterized WMMA16 contract: ' + repr(parameters))


def enumerate_wmma_bytes(depth):
    objects = []
    for stage in range(depth):
        objects.extend([('A' + str(stage), 16, 16, 2),
                        ('B' + str(stage), 16, 16, 2)])
    objects.append(('C', 16, 16, 4))
    counts = {}
    offset = 0
    occupied = set()
    for name, rows, cols, width in objects:
        offset = ((offset + 31) // 32) * 32
        start = offset
        for r in range(rows):
            for c in range(cols):
                for byte in range(width):
                    address = start + (r * cols + c) * width + byte
                    req(address not in occupied, 'independent shared objects overlap')
                    occupied.add(address)
                    counts[name] = counts.get(name, 0) + 1
        offset += rows * cols * width
    return counts, len(occupied)


def observe_capacity(depth=1, faulty_input_count=False):
    counts, enumerated = enumerate_wmma_bytes(depth)
    inputs = sum(size for name, size in counts.items() if name != 'C')
    declared_input = (2048 if faulty_input_count else 1024) * depth
    req(inputs == declared_input, 'WMMA input bytes disagree with object enumeration')
    req(counts['C'] == 1024, 'WMMA output scratch bytes')
    req(enumerated == declared_input + 1024, 'WMMA total shared bytes')


def latency_recurrence(J, depth, L, I, C, ignore_reuse=False):
    out = []
    xprev, yprev = -I, 0
    for j in range(J):
        x = xprev + I
        if j >= depth and not ignore_reuse:
            x = max(x, out[j - depth])
        y = max(x + L, yprev) + C
        out.append(y)
        xprev, yprev = x, y
    return out


def tick_reference(J, depth, L, I, C):
    # Independent integer-time events: retire, issue, then start consumption.
    slots = [None] * depth
    issued = next_consume = 0
    next_issue = 0
    busy = None
    finish = None
    completed = []
    t = 0
    while len(completed) < J:
        req(t < (J + 1) * (L + I + C + 1), 'tick model progress bound')
        if busy is not None and finish == t:
            slots[busy % depth] = None
            completed.append(t)
            busy = None
        while issued < J and t >= next_issue and slots[issued % depth] is None:
            slots[issued % depth] = (issued, t + L)
            issued += 1
            next_issue = t + I
        if busy is None and next_consume < issued:
            slot = slots[next_consume % depth]
            if slot is not None and slot[0] == next_consume and slot[1] <= t:
                busy = next_consume
                next_consume += 1
                finish = t + C
        t += 1
    return completed


def observe_timing(ignore_reuse=False):
    combinations = ((6, 2, 3), (2, 1, 4), (5, 5, 1),
                    (1, 4, 2), (0, 0, 1), (9, 2, 1))
    for J in range(13):
        for depth in range(1, 6):
            for L, I, C in combinations:
                req(latency_recurrence(J, depth, L, I, C, ignore_reuse) ==
                    tick_reference(J, depth, L, I, C), 'finite timing observation')


def decimal_statistics(items):
    visible = [(Decimal(str(z)), Decimal(str(v)))
               for z, v in items if z is not None]
    if not visible:
        return -math.inf, 0.0, 0.0
    with localcontext() as context:
        context.prec = 60
        m = max(z for z, _ in visible)
        weights = [(z - m).exp() for z, _ in visible]
        mass = sum(weights, Decimal(0))
        weighted = sum((p * v for p, (_, v) in zip(weights, visible)), Decimal(0))
        return float(m), float(mass), float(weighted)


def online_statistics(blocks, fault=None):
    m, mass, weighted = -math.inf, 0.0, 0.0
    trace = []
    for block in blocks:
        visible = [(z, v) for z, v in block if z is not None]
        if visible:
            new_m = max(m, max(z for z, _ in visible))
            if mass == 0.0:
                r_l = r_u = 0.0
            else:
                r_l = r_u = math.exp(m - new_m)
                if fault == 'reverse_both':
                    r_l = r_u = math.exp(new_m - m)
                elif fault == 'reverse_u_only':
                    r_u = math.exp(new_m - m)
            ps = [math.exp(z - new_m) for z, _ in visible]
            mass = r_l * mass + math.fsum(ps)
            weighted = r_u * weighted + math.fsum(
                p * v for p, (_, v) in zip(ps, visible))
            m = new_m
        trace.append((m, mass, weighted))
    return trace


def output_of(statistics):
    _, mass, weighted = statistics
    return weighted / mass if mass else 0.0


def observe_attention(blocks, fault=None):
    prefix = []
    trace = online_statistics(blocks, fault)
    for block, got in zip(blocks, trace):
        prefix.extend(block)
        reference = decimal_statistics(prefix)
        req(got[0] == reference[0], 'prefix maximum')
        close(got[1], reference[1], 'prefix mass')
        close(got[2], reference[2], 'prefix weighted sum')
        close(output_of(got), output_of(reference), 'prefix output')
        values = [v for z, v in prefix if z is not None]
        if values:
            req(min(values) - 1e-10 <= output_of(got) <= max(values) + 1e-10,
                'necessary convex hull condition')
        else:
            req(got == (-math.inf, 0.0, 0.0), 'empty identity')
    return trace[-1] if trace else (-math.inf, 0.0, 0.0)


def main():
    observations = []
    cases = ((0, 5, 3), (3, 0, 4), (3, 5, 0), (1, 1, 1),
             (3, 5, 7), (16, 32, 8), (1, 1, 9), (7, 9, 17),
             (19, 35, 17), (33, 17, 9), (5, 37, 15))
    for dimensions in cases:
        observe_gemm(*dimensions)
    rejected(lambda: observe_gemm(1, 1, 9, omit_last=True), 'omit final stage')
    observations.append('SIMT: 11 shapes match independent dense GEMM; output ownership and request/product counts checked; omitted final stage rejected')
    observe_banks()
    rejected(lambda: observe_banks(stride=9), 'changed shared row stride')
    observations.append('32-bank scalar model: tight writes and same-word broadcasts checked; stride-9 A writes rejected')
    observe_slots()
    for fault in ({'guard_copy': False}, {'guard_done': False},
                  {'release_one': True}, {'publish_early': True}):
        rejected(lambda fault=fault: observe_slots(**fault), 'slot mutation ' + repr(fault))
    observations.append('Slot snapshots: depths 1/2/3, both reader orders, partial/duplicate/stale events and genuine progress checked; four faulty variants rejected')
    observe_wmma()
    rejected(lambda: observe_wmma(skip_ld=True), 'missing leading-dimension guard')
    for depth in (1, 2, 3):
        observe_capacity(depth)
    rejected(lambda: observe_capacity(faulty_input_count=True), 'double-counted WMMA input capacity')
    observations.append('WMMA16 parameter contract checked through one entry; object enumeration gives 512+512+1024=2048 bytes, double input buffering gives 3072 bytes')
    observe_timing()
    rejected(lambda: observe_timing(ignore_reuse=True), 'ignored slot reuse')
    wanted = {1: [9, 18, 27, 36, 45, 54, 63, 72],
              2: [9, 12, 18, 21, 27, 30, 36, 39],
              3: [9, 12, 15, 18, 21, 24, 27, 30]}
    for depth, timeline in wanted.items():
        req(tick_reference(8, depth, 6, 2, 3) == timeline, 'chapter timeline')
    def waves(service, residents, total):
        return ((total + residents - 1) // residents) * service
    req(waves(39, 6144 // (2 * 1536), 4) == 78, 'two-level waves')
    req(waves(30, 6144 // (3 * 1536), 4) == 120, 'three-level waves')
    req(waves(30, 9216 // (3 * 1536), 4) == 60, 'larger budget waves')
    observations.append('390 finite timing comparisons against integer-event reference; reuse omission rejected; synthetic wave times 78/120/60 tau reproduced')
    attention_cases = [[], [[]], [[(None, 99.0)], []],
                       [[], [(0.0, 1.0)], [], [(2.0, 0.0)], []],
                       [[(4.0, -2.0), (None, 100.0)], [(1.0, 3.0)]],
                       [[(-1000.0, 2.0)], [(1000.0, -1.0)]],
                       [[(2.0, 3.0), (2.0, -2.0)], [(2.0, 7.0)]]]
    for blocks in attention_cases:
        observe_attention(blocks)
    counterexample = [[(0.0, 1.0)], [(2.0, 0.0)]]
    correct = output_of(observe_attention(counterexample))
    wrong = output_of(online_statistics(counterexample, 'reverse_both')[-1])
    req(0.0 <= correct <= 1.0 and 0.0 <= wrong <= 1.0,
        'convex hull must miss the double reverse counterexample')
    close(correct, 1.0 / (1.0 + math.exp(2.0)), 'correct closed form')
    close(wrong, math.exp(2.0) / (1.0 + math.exp(2.0)), 'faulty double reverse closed form')
    rejected(lambda: observe_attention(counterexample, 'reverse_both'),
             'double reverse independent statistics')
    wrong_u = output_of(online_statistics(counterexample, 'reverse_u_only')[-1])
    req(wrong_u > 1.0, 'U-only reversal should violate convex hull in this example')
    rejected(lambda: observe_attention(counterexample, 'reverse_u_only'),
             'U-only reverse independent statistics')
    observations.append('Attention: Decimal whole-prefix reference checks m/mass/U, empty/masked blocks and finite extremes; double reversal rejected despite convex-hull acceptance; U-only reversal separately rejected')
    print('K-HW-011 scoped CPU observations; no GPU execution or measurement:')
    for observation in observations:
        print(' - ' + observation)
    print('softmax counterexample: correct={:.12f}, reverse_both={:.12f}, reverse_U_only={:.12f}'.format(correct, wrong, wrong_u))
    print('All declared scalar and finite-event assertions completed.')


if __name__ == '__main__':
    main()
