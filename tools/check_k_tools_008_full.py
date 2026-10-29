#!/usr/bin/env python3
"""K-TOOLS-008: fixed-input CPU arithmetic and discrete models only.
Deterministic, standalone, offline and read-only. No CUDA or device probing.
"""
from fractions import Fraction
import math

GIB = 1024 ** 3
TIB = 1024 ** 4
NEGATIVES = []


def equal(actual, expected):
    assert actual == expected, (actual, expected)


def rejected(label, check):
    try:
        check()
    except AssertionError:
        NEGATIVES.append(label)
        print('negative detected: ' + label)
        return
    raise AssertionError('negative escaped: ' + label)


def raises(kind, check):
    try:
        check()
    except kind:
        return
    raise AssertionError('expected ' + kind.__name__)


def natural(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError('expected a nonnegative integer')


def rate(byte_count, elapsed):
    natural(byte_count)
    if elapsed <= 0:
        raise ValueError('elapsed must be positive')
    return Fraction(byte_count) / elapsed


def matched_time(useful, transactions, b_use, b_tx):
    natural(useful)
    natural(transactions)
    if b_use <= 0 or b_tx <= 0:
        raise ValueError('rates must be positive')
    t_use = Fraction(useful) / b_use
    t_tx = Fraction(transactions) / b_tx
    if t_use != t_tx:
        raise ValueError('inconsistent byte/rate identities')
    return t_use


def segment_ids(offset, stride, lanes=32, width=4, granule=32):
    for value in (offset, stride, lanes):
        natural(value)
    if width <= 0 or granule <= 0:
        raise ValueError('sizes must be positive')
    result = set()
    for lane in range(lanes):
        start = offset + lane * stride
        for sector in range(start // granule, (start + width - 1) // granule + 1):
            result.add(sector)
    return frozenset(result)


def segment_reference(offset, stride, lanes=32, width=4, granule=32):
    # Independent enumeration of every requested byte, including crossings.
    return frozenset(
        byte // granule
        for lane in range(lanes)
        for byte in range(offset + lane * stride, offset + lane * stride + width)
    )


def read_ledger(reads, initial=(), granule=32):
    if not isinstance(granule, int) or granule <= 0:
        raise ValueError('granule must be a positive integer')
    valid = set(initial)
    for sector in valid:
        natural(sector)
    useful = hit = miss = transactions = 0
    fills = []
    for address, size in reads:
        natural(address)
        natural(size)
        if size == 0:
            continue
        end = address + size
        for sector in range(address // granule, (end - 1) // granule + 1):
            lo = sector * granule
            count = min(end, lo + granule) - max(address, lo)
            useful += count
            if sector in valid:
                hit += count
            else:
                miss += count
                transactions += granule
                fills.append(sector)
                valid.add(sector)
    return {'use': useful, 'hit': hit, 'miss': miss,
            'tx': transactions, 'fills': tuple(fills)}


def byte_fill_reference(reads, initial=(), granule=32):
    # Materialize valid bytes. Classify each request BEFORE filling it.
    # Count actual full payloads inserted, rather than subtracting hits.
    resident = set()
    for sector in initial:
        resident.update(range(sector * granule, (sector + 1) * granule))
    useful = hits = misses = transferred = 0
    events = []
    for address, size in reads:
        wanted = list(range(address, address + size))
        missing = [byte for byte in wanted if byte not in resident]
        useful += len(wanted)
        hits += len(wanted) - len(missing)
        misses += len(missing)
        for byte in missing:
            if byte in resident:
                continue
            base = byte - byte % granule
            payload = list(range(base, base + granule))
            transferred += len(payload)
            resident.update(payload)
            events.append(base // granule)
    return {'use': useful, 'hit': hits, 'miss': misses,
            'tx': transferred, 'fills': tuple(events)}


def verify_ledger(candidate, reads, initial=()):
    reference = byte_fill_reference(reads, initial)
    equal(candidate, reference)
    equal(candidate['use'], candidate['hit'] + candidate['miss'])


def roofline(work, byte_count, p_eff, b_eff, overlap=True):
    if work < 0 or byte_count < 0 or p_eff <= 0 or b_eff <= 0:
        raise ValueError('invalid resource input')
    compute = Fraction(work) / p_eff
    memory = Fraction(byte_count) / b_eff
    return max(compute, memory) if overlap else compute + memory


def round_cycle(device_us, host_us):
    natural(device_us)
    natural(host_us)
    return max(device_us, host_us)


def graph_total(rounds, device_us, eager_host_us, graph_host_us, capture_us):
    natural(rounds)
    for value in (device_us, eager_host_us, graph_host_us, capture_us):
        natural(value)
    return (rounds * round_cycle(device_us, eager_host_us),
            capture_us + rounds * round_cycle(device_us, graph_host_us))


def graph_break_even(device_us, eager_host_us, graph_host_us, capture_us):
    natural(capture_us)
    delta = round_cycle(device_us, eager_host_us) - round_cycle(device_us, graph_host_us)
    return Fraction(capture_us, delta) if delta > 0 else math.inf


def budget_reference(rounds, device_us, host_us, preparation_us):
    # Ideal independent-budget model, NOT a CUDA submission simulator.
    # One tick is one microsecond; each round waits for both obligations.
    elapsed = preparation_us
    for _ in range(rounds):
        host_left, device_left = host_us, device_us
        while host_left > 0 or device_left > 0:
            if host_left > 0:
                host_left -= 1
            if device_left > 0:
                device_left -= 1
            elapsed += 1
    return elapsed


def bad_graph_without_capture(rounds, device_us, graph_host_us):
    return rounds * round_cycle(device_us, graph_host_us)


def duplex_time(h2d, d2h, b_h2d, b_d2h, serial=False):
    if min(h2d, d2h) < 0 or min(b_h2d, b_d2h) <= 0:
        raise ValueError('invalid link input')
    h_time = Fraction(h2d) / b_h2d
    d_time = Fraction(d2h) / b_d2h
    return h_time + d_time if serial else max(h_time, d_time)


def duplex_reference(h2d, d2h, b_h2d, b_d2h):
    # Fixed fixtures finish on 0.1-second boundaries.
    quantum = Fraction(1, 10)
    remaining_h, remaining_d = Fraction(h2d), Fraction(d2h)
    elapsed = Fraction(0)
    while remaining_h > 0 or remaining_d > 0:
        remaining_h = max(Fraction(0), remaining_h - b_h2d * quantum)
        remaining_d = max(Fraction(0), remaining_d - b_d2h * quantum)
        elapsed += quantum
    return elapsed


def bad_peak_sum_time(h2d, d2h, one_way_h, one_way_d):
    return Fraction(h2d + d2h, one_way_h + one_way_d)


def consumer_share(total, consumers):
    if total < 0 or consumers <= 0:
        raise ValueError('invalid consumer input')
    return Fraction(total) / consumers


def select_entry(entries, mode, working_set, concurrency):
    found = [entry for entry in entries
             if entry['mode'] == mode
             and entry['consumers'] == concurrency
             and entry['lo'] <= working_set < entry['hi']]
    if len(found) != 1:
        raise LookupError('no unique matching parameter')
    return found[0]


def run_checks():
    elapsed = Fraction(1)
    b_use, b_tx = rate(128, elapsed), rate(160, elapsed)
    equal(matched_time(128, 160, b_use, b_tx), elapsed)
    rejected('transaction bytes divided by effective bandwidth',
             lambda: equal(Fraction(160) / b_use, elapsed))
    raises(ValueError, lambda: matched_time(128, 160, b_use, 128))

    equal(segment_ids(0, 4), frozenset(range(4)))
    equal(segment_ids(4, 4), frozenset(range(5)))
    equal(segment_ids(0, 8), frozenset(range(8)))
    for offset in (0, 4, 31, 32):
        for stride in (0, 4, 8, 33):
            for lanes in (0, 1, 7, 32):
                equal(segment_ids(offset, stride, lanes),
                      segment_reference(offset, stride, lanes))
    rejected('offset access always uses four segments',
             lambda: equal(4, len(segment_reference(4, 4))))
    print('observation: aligned/offset4/stride8 request segments = 4/5/8')

    fixtures = [
        ([], ()),
        ([(0, 0)], ()),
        ([(0, 4)], ()),
        ([(31, 4)], ()),
        ([(0, 32)] * 4, ()),
        ([(0, 4), (4, 4), (32, 4), (0, 4)], ()),
        ([(0, 32)], (0,)),
        ([(0, 64)], (0,)),
    ]
    for reads, initial in fixtures:
        verify_ledger(read_ledger(reads, initial), reads, initial)

    sparse_reads = [(0, 4)]
    sparse = read_ledger(sparse_reads)
    equal((sparse['use'], sparse['hit'], sparse['miss'], sparse['tx']),
          (4, 0, 4, 32))
    wrong_ledger = dict(sparse)
    wrong_ledger['tx'] = sparse['miss']
    rejected('effective miss demand renamed as HBM transactions',
             lambda: verify_ledger(wrong_ledger, sparse_reads))

    # Specified abstract service: every recorded 32-byte fill takes 1 second.
    # Hit service, instruction issue, latency overlap and other traffic are absent.
    sparse_reference = byte_fill_reference(sparse_reads)
    fill_time = sum(Fraction(1) for _ in sparse_reference['fills'])
    transaction_capacity = Fraction(32)
    equal(Fraction(sparse['tx']) / transaction_capacity, fill_time)
    miss_effective_rate = rate(sparse['miss'], fill_time)
    equal(Fraction(sparse['miss']) / miss_effective_rate, fill_time)
    bad_sparse_time = Fraction(sparse['miss']) / transaction_capacity
    rejected('effective miss bytes divided by transaction bandwidth',
             lambda: equal(bad_sparse_time, fill_time))
    print('observation: cold 4-byte read: miss=4, fill=32; reference=1 s, faulty=%s s'
          % bad_sparse_time)

    hot_reads = [(0, 32)] * 4
    hot = read_ledger(hot_reads)
    equal((hot['use'], hot['hit'], hot['miss'], hot['tx']), (128, 96, 32, 32))
    hot_reference = byte_fill_reference(hot_reads)
    hot_time = sum(Fraction(1) for _ in hot_reference['fills'])
    apparent_rate = rate(hot['use'], hot_time)
    actual_transaction_rate = rate(hot['tx'], hot_time)
    equal(actual_transaction_rate, transaction_capacity)
    bad_hot_time = Fraction(hot['tx']) / apparent_rate
    rejected('apparent L2 reuse rate substituted for HBM transaction bandwidth',
             lambda: equal(bad_hot_time, hot_time))
    print('observation: repeated full-sector reads: use/hit/miss/fill=128/96/32/32; '
          'apparent=128 B/s, transaction=32 B/s, faulty time=%s s' % bad_hot_time)
    equal(read_ledger([(0, 32)], (0,))['tx'], 0)
    equal(read_ledger([(31, 4)])['tx'], 64)

    independent_reference = len(set(range(2)) | set(range(3)))
    serial_reference = sum(1 for duration in (2, 3) for _ in range(duration))
    equal(roofline(2, 3, 1, 1), independent_reference)
    equal(roofline(2, 3, 1, 1, False), serial_reference)
    rejected('max reported as dependent serial completion',
             lambda: equal(roofline(2, 3, 1, 1), serial_reference))
    equal(roofline(0, 0, 1, 1), 0)
    equal(roofline(0, 5, 1, 1), 5)

    device, eager_host, graph_host, capture = 800, 1200, 200, 10000
    equal(round_cycle(device, eager_host), 1200)
    equal(round_cycle(device, graph_host), 800)
    equal(graph_break_even(device, eager_host, graph_host, capture), 25)
    expected = {24: (28800, 29200), 25: (30000, 30000), 26: (31200, 30800)}
    for rounds in (0, 1, 24, 25, 26):
        eager_ref = budget_reference(rounds, device, eager_host, 0)
        graph_ref = budget_reference(rounds, device, graph_host, capture)
        equal(graph_total(rounds, device, eager_host, graph_host, capture),
              (eager_ref, graph_ref))
        if rounds in expected:
            equal((eager_ref, graph_ref), expected[rounds])
            equal(graph_ref < eager_ref, rounds > 25)
            print('observation: N=%d eager=%d us graph-with-capture=%d us'
                  % (rounds, eager_ref, graph_ref))
        faulty_total = bad_graph_without_capture(rounds, device, graph_host)
        rejected('capture omitted from total at N=%d' % rounds,
                 lambda faulty=faulty_total, ref=graph_ref: equal(faulty, ref))
    # Capture is already paid in the zero-use epoch above. If not prepared,
    # zero work has zero capture cost and can return immediately.
    equal(graph_total(0, device, eager_host, graph_host, 0), (0, 0))
    for rounds in (1, 26):
        long_ref = (budget_reference(rounds, 12000, eager_host, 0),
                    budget_reference(rounds, 12000, graph_host, capture))
        equal(graph_total(rounds, 12000, eager_host, graph_host, capture), long_ref)
        equal(long_ref[1] - long_ref[0], capture)
    assert math.isinf(graph_break_even(12000, eager_host, graph_host, capture))
    print('observation: common device work=12000 us leaves no steady cycle gain')

    volume = 28 * GIB
    one_way = 28 * GIB
    duplex_each = 20 * GIB
    link_ref = duplex_reference(volume, volume, duplex_each, duplex_each)
    equal(link_ref, Fraction(7, 5))
    equal(duplex_time(volume, volume, duplex_each, duplex_each), link_ref)
    equal(duplex_time(volume, volume, one_way, one_way, True), 2)
    faulty_link = bad_peak_sum_time(volume, volume, one_way, one_way)
    equal(faulty_link, 1)
    rejected('sum of two one-way capacities used for duplex completion',
             lambda: equal(faulty_link, link_ref))
    equal(duplex_time(0, 0, one_way, one_way), 0)
    pure_ref = duplex_reference(volume, 0, one_way, one_way)
    equal(pure_ref, 1)
    equal(duplex_time(volume, 0, one_way, one_way), pure_ref)
    rejected('unused direction counted as available pure-transfer capacity',
             lambda: equal(bad_peak_sum_time(volume, 0, one_way, one_way), pure_ref))
    print('observation: symmetric duplex reference=1.4 s, peak-sum fault=1.0 s; '
          'separate serial path=2.0 s')

    total_two, single = Fraction(11, 10), Fraction(1)
    equal(consumer_share(total_two, 2), Fraction(11, 20))
    rejected('single-consumer rate multiplied by replicas',
             lambda: equal(2 * single, total_two))

    entries = [
        {'mode': 'seq', 'consumers': 1, 'lo': 0, 'hi': 16 * 1024 ** 2,
         'rate': Fraction(5, 2) * TIB},
        {'mode': 'seq', 'consumers': 1, 'lo': 16 * 1024 ** 2, 'hi': 16 * GIB,
         'rate': Fraction(TIB)},
        {'mode': 'random128', 'consumers': 1, 'lo': 0, 'hi': 16 * GIB,
         'rate': Fraction(7, 20) * TIB},
    ]
    chosen = select_entry(entries, 'random128', 8 * GIB, 1)
    equal(chosen['rate'], Fraction(7, 20) * TIB)
    workload = 350 * GIB
    random_time = Fraction(workload) / chosen['rate']
    wrong_seq_time = Fraction(workload, TIB)
    equal(random_time / wrong_seq_time, Fraction(20, 7))
    rejected('sequential bandwidth substituted for random-access entry',
             lambda: equal(wrong_seq_time, random_time))
    equal(select_entry(entries, 'seq', 16 * 1024 ** 2, 1)['rate'], TIB)
    raises(LookupError, lambda: select_entry(entries, 'seq', 8 * GIB, 2))
    raises(LookupError, lambda: select_entry(entries + [chosen], 'random128', 8 * GIB, 1))

    # Complete the chapter's sequential combination trace with exact units.
    weight = Fraction(8 * GIB, TIB)
    kv = Fraction(2 * GIB) / (Fraction(7, 20) * TIB)
    output_copy = Fraction(64 * 1024 ** 2, 28 * GIB)
    equal(weight, Fraction(1, 128))
    equal(output_copy * 1000, Fraction(125, 56))
    equal((weight + kv + output_copy) * 1000, Fraction(125, 8))

    raises(ValueError, lambda: rate(-1, 1))
    raises(ValueError, lambda: rate(1, 0))
    raises(ValueError, lambda: read_ledger([(-1, 4)]))
    raises(ValueError, lambda: read_ledger([(0, -1)]))
    raises(ValueError, lambda: read_ledger([], granule=0))
    raises(ValueError, lambda: segment_ids(-1, 4))
    raises(ValueError, lambda: roofline(-1, 0, 1, 1))
    raises(ValueError, lambda: graph_total(-1, device, eager_host, graph_host, capture))
    raises(ValueError, lambda: duplex_time(1, 1, 0, 1))
    print('observation: fixed-input empty, warm, crossing, invalid-input and '
          'parameter-selection boundaries checked')
    print('scoped observations complete: %d faulty predictions detected; '
          'CPU models only; no GPU/runtime/service measurements' % len(NEGATIVES))


if __name__ == '__main__':
    run_checks()
