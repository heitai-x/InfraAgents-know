#!/usr/bin/env python3
'''K-HW-001: deterministic, read-only, offline standard-library models.
No files, processes, network, project imports or device access.
This script tests arithmetic, addresses, lifetimes and small schedules only.
'''
from fractions import Fraction
from itertools import product


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(observed, expected, message):
    check(observed == expected, message + ': observed=' + repr(observed) + ', expected=' + repr(expected))


def rejects_valueerror(label, operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError('invalid input accepted: ' + label)


def negative(label, observed, independent_expected):
    # The faulty operation is evaluated before this comparator. Crashes are
    # not caught and cannot masquerade as detection of a wrong numeric result.
    check(observed != independent_expected, 'negative control escaped: ' + label)
    print('negative_control_detected=' + label)


def roofline(F, D, P, B):
    F, D, P, B = map(Fraction, (F, D, P, B))
    require(F >= 0 and D >= 0, 'F and D must be nonnegative')
    require(P > 0 and B > 0, 'P and B must be positive')
    tc, tm = F / P, D / B
    intensity = F / D if D else None
    roof = min(P, B * intensity) if D else (P if F else Fraction(0))
    return max(tc, tm), tc, tm, intensity, roof


def test_roofline():
    count = 0
    for F, D, P, B in product((0, 1, 7, 20), (0, 1, 9), (2, 5), (3, 8)):
        total, tc, tm, intensity, roof = roofline(F, D, P, B)
        check(total >= tc and total >= tm, 'resource feasibility invariant')
        check(total == tc or total == tm, 'one resource must attain the local bound')
        if F > 0:
            # Dual throughput expression, separate from the max-time form.
            equal(Fraction(F) / roof, total, 'time/throughput duality')
        if D == 0:
            equal(intensity, None, 'zero-byte intensity is undefined')
        if F == 0 and D > 0:
            equal(total, Fraction(D, B), 'pure movement')
            equal(roof, 0, 'zero useful arithmetic throughput')
        count += 1
    equal(roofline(0, 0, 2, 3)[0], 0, 'empty work')
    equal(roofline(100, 0, 50, 16)[0], 2, 'compute only')
    invalid = ((1, 1, 0, 1), (1, 1, 1, 0), (-1, 0, 1, 1),
               (0, -1, 1, 1), (0, 0, 0, 1), (1, 1, -2, 1))
    for args in invalid:
        rejects_valueerror(repr(args), lambda args=args: roofline(*args))
    # Catching AssertionError here would recreate the recorded exception mismatch.
    total, tc, tm, _, _ = roofline(1000, 1000, 50, 16)
    negative('min_instead_of_resource_max', min(tc, tm), total)
    useful_time = Fraction(128, 80)
    equal(useful_time, Fraction(160, 100), 'matched byte/rate conventions')
    negative('transaction_amplification_charged_twice', Fraction(160, 80), useful_time)
    print('roofline_valid_cases=' + str(count) + ' invalid_inputs_rejected=' + str(len(invalid)))


def validate_refs(refs):
    rows, sizes = [], {}
    for key, start, end, size in refs:
        start, end = Fraction(start), Fraction(end)
        require(start <= end, 'reversed lifetime')
        require(isinstance(size, int) and size >= 0, 'size must be nonnegative integer bytes')
        require(key not in sizes or sizes[key] == size, 'physical key size mismatch')
        sizes[key] = size
        rows.append((key, start, end, size))
    return rows, sizes


def capacity_events(refs):
    rows, sizes = validate_refs(refs)
    events = {}
    for key, start, end, size in rows:
        if start == end:
            continue
        events.setdefault(start, [[], []])[1].append(key)
        events.setdefault(end, [[], []])[0].append(key)
    counts = {}
    total, peak = 0, 0
    area = Fraction(0)
    previous = None
    for time in sorted(events):
        if previous is not None:
            area += total * (time - previous)
        retire, birth = events[time]
        for key in retire:
            require(counts.get(key, 0) > 0, 'retire without live hold')
            counts[key] -= 1
            if counts[key] == 0:
                del counts[key]
                total -= sizes[key]
        for key in birth:
            if key not in counts:
                counts[key] = 0
                total += sizes[key]
            counts[key] += 1
        # Observe after all simultaneous changes, not after individual births.
        peak = max(peak, total)
        previous = time
    check(total == 0 and not counts, 'capacity events did not drain')
    return peak, area


def capacity_reference(refs):
    # Independent interval partition: no event counts or release operations.
    rows, _ = validate_refs(refs)
    points = sorted({p for _, start, end, _ in rows for p in (start, end)})
    peak, area = 0, Fraction(0)
    for left, right in zip(points, points[1:]):
        midpoint = (left + right) / 2
        active = {key: size for key, start, end, size in rows if start <= midpoint < end}
        occupied = sum(active.values())
        peak = max(peak, occupied)
        area += occupied * (right - left)
    return peak, area


def test_capacity():
    refs = [('p', 0, 10, 4), ('p', 0, 10, 4), ('q', 2, 8, 2),
            ('workspace', 4, 6, 1), ('zero', 3, 3, 100)]
    equal(capacity_events(refs), (7, Fraction(54)), 'shared physical fixture')
    for mask in range(1 << len(refs)):
        subset = [row for i, row in enumerate(refs) if mask & (1 << i)]
        equal(capacity_events(subset), capacity_reference(subset), 'interval subset oracle')
    fixtures = [[], [('z', 3, 3, 100)],
                [('a', 0, 2, 2), ('b', 2, 4, 3)],
                [('p', 0, 2, 4), ('p', 2, 4, 4)],
                [('p', 0, 2, 4), ('p', 0, 8, 4)],
                [('p', 0, 8, 4), ('new', 2, 10, 4), ('copy-work', 2, 3, 1)]]
    for rows in fixtures:
        equal(capacity_events(rows), capacity_reference(rows), 'capacity boundary oracle')
    equal(capacity_events(fixtures[2]), (3, Fraction(10)), 'touching distinct allocations')
    equal(capacity_events(fixtures[4]), (4, Fraction(32)), 'last reader protects full lifetime')
    equal(capacity_events(fixtures[5]), (9, Fraction(65)), 'COW transition overlap')
    owner_rows = [(str(i), start, end, size) for i, (_, start, end, size) in enumerate(refs)]
    equal(capacity_reference(owner_rows)[0], 11, 'owner mutant fixture')
    negative('owners_counted_as_physical_copies', capacity_reference(owner_rows), capacity_reference(refs))
    prematurely_retired = [('p', 0, 2, 4)]
    # Both peaks are 4: the integral, rather than the peak, exposes the fault.
    equal(capacity_events(prematurely_retired)[0], capacity_reference(fixtures[4])[0], 'equal peaks hide early retirement')
    negative('first_reader_retires_shared_page', capacity_events(prematurely_retired), capacity_reference(fixtures[4]))
    zero_as_nonempty = [row if row[1] != row[2] else (row[0], row[1], row[2] + 1, row[3]) for row in refs]
    negative('zero_lifetime_counted_as_nonempty', capacity_events(zero_as_nonempty), capacity_reference(refs))
    for bad in ([('x', 2, 1, 1)], [('x', 0, 1, -1)],
                [('x', 0, 1, 1), ('x', 2, 3, 2)]):
        rejects_valueerror('events invalid lifetime/size', lambda bad=bad: capacity_events(bad))
        rejects_valueerror('reference invalid lifetime/size', lambda bad=bad: capacity_reference(bad))
    print('capacity_peak_bytes=7 residency_byte_time=54 subset_oracles=32 zero_and_touching_intervals_checked')


def dag_bound(nodes):
    # Each node exclusively uses one named resource for its stated duration.
    ids = [node[0] for node in nodes]
    require(len(ids) == len(set(ids)), 'duplicate stage identity')
    known = set(ids)
    normalized = []
    for name, resource, duration, parents in nodes:
        duration = Fraction(duration)
        require(duration >= 0, 'negative stage duration')
        require(set(parents) <= known, 'unknown predecessor')
        normalized.append((name, resource, duration, tuple(parents)))
    pending, finished, work = list(normalized), {}, {}
    for name, resource, duration, parents in normalized:
        work[resource] = work.get(resource, Fraction(0)) + duration
    while pending:
        ready = [row for row in pending if all(p in finished for p in row[3])]
        require(bool(ready), 'cyclic dependency graph')
        for row in ready:
            name, resource, duration, parents = row
            finished[name] = duration + max((finished[p] for p in parents), default=Fraction(0))
            pending.remove(row)
    return max([Fraction(0)] + list(finished.values()) + list(work.values()))


def exhaustive_two_chunk_schedule():
    # Four integer start times, duration 2. The horizon contains a feasible
    # finish at 6 and every schedule that could beat it. No DAG-bound call.
    best = None
    feasible = 0
    for a, b, c, d in product(range(7), repeat=4):
        if b < a + 2 or d < c + 2:
            continue
        if not (a + 2 <= c or c + 2 <= a):
            continue
        if not (b + 2 <= d or d + 2 <= b):
            continue
        finish = max(a, b, c, d) + 2
        best = finish if best is None else min(best, finish)
        feasible += 1
    check(best is not None, 'enumeration has no feasible schedule')
    return best, feasible


def test_dag():
    independent = [('a', 'X', 20, ()), ('b', 'Y', 10, ())]
    shared = [('a', 'X', 20, ()), ('b', 'X', 10, ())]
    serial = [('a', 'X', 20, ()), ('b', 'Y', 10, ('a',))]
    equal(dag_bound(independent), 20, 'independent resource bound')
    equal(dag_bound(shared), 30, 'same-resource work bound')
    equal(dag_bound(serial), 30, 'dependency path bound')
    equal(dag_bound([]), 0, 'empty graph')
    negative('max_used_for_same_resource', max(20, 10), dag_bound(shared))
    negative('sum_claimed_necessary_for_independent_resources', 20 + 10, dag_bound(independent))
    pipeline = [('ax', 'X', 2, ()), ('ay', 'Y', 2, ('ax',)),
                ('bx', 'X', 2, ()), ('by', 'Y', 2, ('bx',))]
    lower = dag_bound(pipeline)
    optimum, feasible = exhaustive_two_chunk_schedule()
    equal(lower, 4, 'pipeline necessary lower bound')
    equal(optimum, 6, 'exhaustive optimum')
    check(lower < optimum, 'fixture must separate lower bound and attainable time')
    equal(2 + 2 + (2 - 1) * max(2, 2), optimum, 'two-stage fill/drain formula')
    negative('lower_bound_reported_as_completion_time', lower, optimum)
    rejects_valueerror('cycle', lambda: dag_bound([('a', 'X', 1, ('b',)), ('b', 'Y', 1, ('a',))]))
    rejects_valueerror('unknown predecessor', lambda: dag_bound([('a', 'X', 1, ('missing',))]))
    print('two_chunk_schedule lower_bound=4 optimum=6 feasible_enumerated=' + str(feasible))


def segments(offset, nbytes, segment=32):
    require(all(isinstance(x, int) for x in (offset, nbytes, segment)), 'segment inputs must be integers')
    require(offset >= 0 and nbytes >= 0 and segment > 0, 'invalid segment input')
    if nbytes == 0:
        return 0
    return (offset + nbytes + segment - 1) // segment - offset // segment


def segment_reference(offset, nbytes, segment=32):
    return len({byte // segment for byte in range(offset, offset + nbytes)})


def bank_depth(words):
    banks = {}
    for word in words:
        require(isinstance(word, int) and word >= 0, 'invalid shared word')
        banks.setdefault(word % 32, set()).add(word)
    return max((len(addresses) for addresses in banks.values()), default=0)


def test_addresses():
    count = 0
    for offset in range(65):
        for length in (0, 1, 31, 32, 33, 64, 128):
            equal(segments(offset, length), segment_reference(offset, length), 'byte-set segment oracle')
            count += 1
    equal(segments(0, 128), 4, 'aligned warp')
    equal(segments(4, 128), 5, 'offset warp')
    negative('segment_offset_ignored', (128 + 31) // 32, segment_reference(4, 128))
    equal(bank_depth(list(range(32))), 1, 'distinct banks')
    equal(bank_depth([8 * lane + 3 for lane in range(32)]), 8, 'eight-way scalar conflict')
    equal(bank_depth([9] * 32), 1, 'same-word broadcast')
    negative('broadcast_treated_as_distinct_addresses', 32, bank_depth([9] * 32))
    rejects_valueerror('negative byte offset', lambda: segments(-1, 128))
    print('segment_byte_set_oracles=' + str(count) + ' aligned=4 offset4=5 bank_model=32bit_scalar_only')


def matrices(M, N, K):
    A = [[(i * 3 + k * 2) % 11 - 5 for k in range(K)] for i in range(M)]
    W = [[(k * 5 + j * 7) % 13 - 6 for j in range(N)] for k in range(K)]
    return A, W


def dense_reference(M, N, K):
    A, W = matrices(M, N, K)
    return [[sum(A[i][k] * W[k][j] for k in range(K)) for j in range(N)] for i in range(M)]


def tiled_model(M, N, K, wrong_stride=False):
    require(all(isinstance(x, int) and x >= 0 for x in (M, N, K)), 'invalid matrix dimensions')
    A, W = matrices(M, N, K)
    Y = [[0] * N for _ in range(M)]
    loads, stores = 0, 0
    for m0 in range(0, M, 8):
        for n0 in range(0, N, 32):
            acc = [[0] * 32 for _ in range(8)]
            for k0 in range(0, K, 32):
                shared_a, shared_w = [0] * 256, [0] * 1024
                for u in range(8):
                    for lane in range(32):
                        if m0 + u < M and k0 + lane < K:
                            shared_a[32 * u + lane] = A[m0 + u][k0 + lane]
                            loads += 1
                        for v in range(4):
                            t = u + 8 * v
                            if k0 + t < K and n0 + lane < N:
                                shared_w[32 * t + lane] = W[k0 + t][n0 + lane]
                                loads += 1
                # Sequential model only: every shared write precedes all reads.
                for t in range(32):
                    for u in range(8):
                        for lane in range(32):
                            stride = 8 if wrong_stride else 32
                            acc[u][lane] += shared_a[32 * u + t] * shared_w[stride * t + lane]
                # All readers finish before these arrays are replaced.
            for u in range(8):
                for lane in range(32):
                    if m0 + u < M and n0 + lane < N:
                        Y[m0 + u][n0 + lane] = acc[u][lane]
                        stores += 1
    return Y, 4 * (loads + stores)


def test_tile():
    shapes = ((0, 5, 3), (3, 0, 4), (2, 3, 0), (1, 1, 1),
              (3, 5, 7), (8, 32, 32), (9, 35, 33))
    for M, N, K in shapes:
        output, byte_events = tiled_model(M, N, K)
        equal(output, dense_reference(M, N, K), 'tiled output versus direct dot products')
        expected = 0 if M * N == 0 else 4 * (M * K * ((N + 31) // 32) + K * N * ((M + 7) // 8) + M * N)
        equal(byte_events, expected, 'masked load/store events versus traffic formula')
    wrong, _ = tiled_model(3, 5, 7, wrong_stride=True)
    negative('global_or_wrong_stride_used_for_shared_W', wrong, dense_reference(3, 5, 7))
    equal(4 * (8 * 32 + 32 * 32), 5120, 'single-buffer shared allocation')
    rejects_valueerror('negative matrix dimension', lambda: tiled_model(-1, 5, 7))
    print('tile_dense_and_traffic_oracles=' + str(len(shapes)) + ' shared_bytes=5120 exact_integer_model_with_M_N_K_tails')


def round_time(weight_mib, kv_mib, batch, bandwidth_mib_s):
    require(weight_mib >= 0 and kv_mib >= 0, 'negative traffic')
    require(isinstance(batch, int) and batch > 0 and bandwidth_mib_s > 0, 'invalid batch/rate')
    return (Fraction(weight_mib) + batch * Fraction(kv_mib)) / Fraction(bandwidth_mib_s)


def test_service_numbers():
    r1 = round_time(8192, 128, 1, 1048576)
    r16 = round_time(8192, 128, 16, 1048576)
    equal(r16, Fraction(5, 512), '10 GiB over 1 TiB/s')
    equal(r16 / 16, Fraction(5, 8192), 'per-output-token amortization')
    check(r16 > r1 and r16 / 16 < r1, 'round latency and amortization must move differently here')
    negative('per_output_token_time_reported_as_request_wait', r16 / 16, r16)
    equal(Fraction(8192, 128), 64, 'weight/KV crossover batch')
    equal(sum(map(Fraction, (8, 2, '0.5', '1.5', '0.25', '0.5'))), Fraction(51, 4), 'disjoint capacity categories')
    prefix, readers = 128 * 1024, 4
    read_events = [prefix for _ in range(readers)]
    equal(sum(read_events), 512 * 1024, 'independent consumer reads')
    negative('unique_capacity_used_as_consumer_traffic', prefix, sum(read_events))
    h2d = Fraction(1, 16)
    hbm = Fraction(10, 1024)
    equal((h2d + hbm) * 1000, Fraction(9250, 128), 'serial host/device example in ms')
    equal(max(h2d, hbm) * 1000, Fraction(125, 2), 'independent overlap bound in ms')
    page = 16 * 128 * 1024
    equal(page, 2 * 1024 * 1024, 'full-layer page bytes')
    equal(2 * 15 * 128 * 1024, 3932160, 'valid-row COW read plus write')
    print('service_example round16_ms=' + str(float(r16 * 1000)) + ' amortized_ms=' + str(float(r16 * 1000 / 16)) + ' crossover_batch=64 peak_GiB=12.75')


def amdahl(f, s):
    f, s = Fraction(f), Fraction(s)
    require(0 <= f <= 1 and s > 0, 'invalid Amdahl parameters')
    return 1 / ((1 - f) + f / s)


def test_amdahl():
    actual = amdahl(Fraction(3, 10), 2)
    # Independent old/new duration construction: 1000 -> 700 + 300/2.
    expected = Fraction(1000, 700 + 150)
    equal(actual, expected, 'Amdahl versus duration decomposition')
    equal(actual, Fraction(20, 17), 'Amdahl exact rational')
    equal(amdahl(0, 2), 1, 'nothing accelerated')
    equal(amdahl(1, 2), 2, 'all work accelerated')
    negative('local_speedup_reported_as_total', Fraction(2), expected)
    rejects_valueerror('invalid fraction', lambda: amdahl(2, 2))
    print('amdahl_total_speedup=' + str(float(actual)))


def main():
    test_roofline()
    test_capacity()
    test_dag()
    test_addresses()
    test_tile()
    test_service_numbers()
    test_amdahl()
    print('scope: CPU rational/integer arithmetic, finite interval events, scalar address model and small exhaustive schedules only.')
    print('not measured: CPU/GPU kernels, floating-point kernel error, hardware visibility/barriers, DRAM transactions, bandwidth, overlap, clocks, energy, SLO or Agent benefits.')


if __name__ == '__main__':
    main()
