#!/usr/bin/env python3
"""K-FW-005 CPU discrete model. Stdlib, deterministic, offline, no file I/O.
This is not a CUDA API, device visibility, numerical kernel or performance test.
"""

from dataclasses import dataclass
from itertools import permutations, product
from math import exp, isclose, log


@dataclass(frozen=True)
class Request:
    rid: int
    q: int
    p: int
    sig: str = 'A'


@dataclass(frozen=True)
class Bucket:
    ncap: int
    qcap: int
    pcap: int
    qmax: int
    pmax: int
    sig: str = 'A'


def layout(requests, bucket):
    """Chapter compact layout with active ranges and repeated tail pointers."""
    qptr, pptr = [0], [0]
    for req in requests:
        if (req.sig != bucket.sig or req.q < 1 or req.p < 0
                or req.q > bucket.qmax or req.p > bucket.pmax):
            return None
        qptr.append(qptr[-1] + req.q)
        pptr.append(pptr[-1] + req.p)
    n, q, p = len(requests), qptr[-1], pptr[-1]
    if n > bucket.ncap or q > bucket.qcap or p > bucket.pcap:
        return None
    while len(qptr) < bucket.ncap + 1:
        qptr.append(q)
        pptr.append(p)
    rows = [(r.rid, j) for r in requests for j in range(r.q)]
    pages = [(r.rid, j) for r in requests for j in range(r.p)]
    rows += [None] * (bucket.qcap - q)
    pages += [None] * (bucket.pcap - p)
    return {'n': n, 'q': q, 'p': p, 'qptr': qptr, 'pptr': pptr,
            'rows': rows, 'pages': pages}


def bounded_reference(requests, bucket):
    """Independent feasibility oracle: place items one by one in bounded arrays.
    It does not use prefix sums or aggregate-capacity comparisons.
    Repeated page identities here represent index records, not physical bytes.
    """
    lanes = [None] * bucket.ncap
    rows = [None] * bucket.qcap
    pages = [None] * bucket.pcap
    row_cursor = page_cursor = 0
    try:
        for lane, req in enumerate(requests):
            if (req.sig != bucket.sig or req.q < 1 or req.p < 0
                    or req.q > bucket.qmax or req.p > bucket.pmax):
                return None
            lanes[lane] = req.rid
            for j in range(req.q):
                rows[row_cursor] = (req.rid, j)
                row_cursor += 1
            for j in range(req.p):
                pages[page_cursor] = (req.rid, j)
                page_cursor += 1
    except IndexError:
        return None
    return rows, pages


def old_per_request_only(requests, bucket):
    """Intentionally faulty: no request-count or batch aggregate checks."""
    return all(r.sig == bucket.sig and 1 <= r.q <= bucket.qmax
               and r.q <= bucket.qcap and 0 <= r.p <= bucket.pmax
               and r.p <= bucket.pcap for r in requests)


def verify_layout(requests, bucket):
    actual = layout(requests, bucket)
    reference = bounded_reference(requests, bucket)
    assert (actual is None) == (reference is None), (requests, bucket)
    if actual is None:
        return
    rows, pages = reference
    assert actual['rows'] == rows
    assert actual['pages'] == pages
    assert len(actual['qptr']) == len(actual['pptr']) == bucket.ncap + 1
    assert actual['qptr'][0] == actual['pptr'][0] == 0
    assert actual['qptr'][-1] == actual['q']
    assert actual['pptr'][-1] == actual['p']
    for ptr in (actual['qptr'], actual['pptr']):
        assert all(a <= b for a, b in zip(ptr, ptr[1:]))
    for i, req in enumerate(requests):
        qa, qb = actual['qptr'][i:i + 2]
        pa, pb = actual['pptr'][i:i + 2]
        assert actual['rows'][qa:qb] == [(req.rid, j) for j in range(req.q)]
        assert actual['pages'][pa:pb] == [(req.rid, j) for j in range(req.p)]
    for i in range(len(requests), bucket.ncap):
        assert actual['qptr'][i] == actual['qptr'][i + 1]
        assert actual['pptr'][i] == actual['pptr'][i + 1]


def check_capacity():
    buckets = [Bucket(1, 4, 4, 4, 4), Bucket(2, 4, 8, 4, 4),
               Bucket(3, 8, 4, 4, 4), Bucket(3, 8, 8, 2, 2)]
    choices = list(product((1, 2, 4), repeat=2))
    comparisons = 0
    batches = 0
    for n in range(4):
        for values in product(choices, repeat=n):
            requests = tuple(Request(i, q, p) for i, (q, p) in enumerate(values))
            batches += 1
            for bucket in buckets:
                verify_layout(requests, bucket)
                comparisons += 1
    assert batches == 820 and comparisons == 3280
    failures = [
        ('query_total', Bucket(2, 64, 16, 64, 8),
         (Request(0, 40, 4), Request(1, 40, 4))),
        ('page_total', Bucket(2, 8, 8, 8, 8),
         (Request(0, 1, 5), Request(1, 1, 5))),
        ('request_total', Bucket(2, 8, 8, 8, 8),
         tuple(Request(i, 1, 1) for i in range(3))),
    ]
    for name, bucket, requests in failures:
        assert old_per_request_only(requests, bucket), name
        assert layout(requests, bucket) is None, name
        assert bounded_reference(requests, bucket) is None, name
    exact = (Request(0, 32, 8), Request(1, 32, 8))
    bucket = Bucket(2, 64, 16, 64, 8)
    verify_layout(exact, bucket)
    assert layout(exact, bucket)['qptr'] == [0, 32, 64]
    assert layout(exact, bucket)['pptr'] == [0, 8, 16]
    assert layout((Request(0, 65, 1),), bucket) is None
    assert layout((Request(0, 1, 9),), bucket) is None
    assert layout((Request(0, 1, 1, 'B'),), bucket) is None
    assert layout((Request(0, 0, 1),), bucket) is None
    assert layout((Request(0, 1, -1),), bucket) is None
    verify_layout((Request(0, 1, 0),), bucket)
    verify_layout((), bucket)
    # Empty representation is valid; the submission algorithm skips its replay.
    split = [(r,) for r in failures[0][2]]
    assert all(layout(part, failures[0][1]) is not None for part in split)
    assert tuple(r for part in split for r in part) == failures[0][2]
    print('OBS layout_batches=%d comparisons=%d' % (batches, comparisons))
    print('OBS aggregate_negative_controls=query_total,page_total,request_total detected')
    print('OBS exact_capacity_offsets=query:[0,32,64] page:[0,8,16]; split preserves work')


@dataclass(frozen=True)
class CapturedGraph:
    addresses: tuple

    def replay(self, memory):
        # Every access follows captured addresses, never a requested slot number.
        inp, meta, plan, out, workspace = self.addresses
        gi, value = memory[inp]
        gm, bias = memory[meta]
        gp, factor = memory[plan]
        assert gi == gm == gp, 'mixed-generation payload in captured binding'
        memory[workspace] = (gi, value + bias)
        gw, intermediate = memory[workspace]
        memory[out] = (gw, intermediate * factor)
        return memory[out]


def check_binding():
    memory = {}
    bindings = []
    for slot in range(2):
        addresses = tuple('b0.s%d.%s' % (slot, part)
                          for part in ('input', 'metadata', 'plan', 'output', 'workspace'))
        assert not set(addresses).intersection(memory)
        memory.update(dict.fromkeys(addresses))
        bindings.append(addresses)
    e0, e1 = [CapturedGraph(a) for a in bindings]
    for addresses, g, value, bias, factor in [
            (bindings[0], 2, 10, 1, 2), (bindings[1], 3, 20, 3, 4)]:
        memory[addresses[0]] = (g, value)
        memory[addresses[1]] = (g, bias)
        memory[addresses[2]] = (g, factor)
    expected = (3, (20 + 3) * 4)
    assert e1.replay(memory) == expected
    assert memory[bindings[0][3]] is None
    memory[bindings[1][3]] = None
    wrong = e0.replay(memory)  # Intentional original binding error.
    assert wrong != expected
    assert memory[bindings[1][3]] is None
    assert memory[bindings[0][3]] == (2, 22)
    print('OBS binding_correct=(3,92) wrong_graph=(2,22); wrong output address detected')


class Slot:
    """Single-owner serialized state model; abstract completion, no CUDA events."""
    def __init__(self, maintain_watermark=True):
        self.phase = 'FREE'
        self.gen = -1
        self.last = -1
        self.payload = None
        self.pending = set()
        self.maintain_watermark = maintain_watermark

    def snapshot(self):
        return (self.phase, self.gen, self.last, self.payload,
                tuple(sorted(self.pending)))

    def publish(self, generation, payload):
        if self.phase != 'FREE' or generation <= self.last:
            return False
        self.phase = 'WRITING'
        self.payload = tuple(payload)
        self.gen = generation
        if self.maintain_watermark:
            self.last = generation
        self.pending = {'graph', 'output'}
        self.phase = 'READY'
        return True

    def launch(self):
        assert self.phase == 'READY'
        self.phase = 'RUNNING'

    def complete(self, generation, reader):
        if generation != self.gen or reader not in self.pending:
            return False
        if reader == 'graph' and self.phase != 'RUNNING':
            return False
        if reader == 'output' and (self.phase != 'DRAINING' or 'graph' in self.pending):
            return False
        self.pending.remove(reader)
        self.phase = 'DRAINING' if self.pending else 'FREE'
        return True


def drain(slot):
    slot.launch()
    assert slot.complete(slot.gen, 'graph')
    assert slot.complete(slot.gen, 'output')
    assert slot.phase == 'FREE'


def check_generations():
    accepted_count = 0
    for arrival_order in permutations(range(4)):
        slot = Slot()
        expected_high = -1  # Reference: record maxima of the arrival sequence.
        for g in arrival_order:
            before = slot.snapshot()
            expected_accept = g > expected_high
            accepted = slot.publish(g, (g,))
            assert accepted == expected_accept, arrival_order
            if accepted:
                expected_high = g
                accepted_count += 1
                assert slot.last == expected_high and slot.payload == (g,)
                drain(slot)
            else:
                assert slot.snapshot() == before
            assert slot.last == expected_high
    slot = Slot()
    assert slot.publish(2, ('newer',))
    drain(slot)
    before = slot.snapshot()
    assert not slot.publish(1, ('older',))
    assert not slot.publish(2, ('duplicate',))
    assert slot.snapshot() == before
    # Faulty negative control omits exactly the watermark update under review.
    bad = Slot(maintain_watermark=False)
    assert bad.publish(2, ('newer',))
    drain(bad)
    assert bad.publish(1, ('older',))
    assert bad.payload == ('older',) and bad.last == -1
    # Graph completion alone does not release an output source.
    assert slot.publish(3, ('next',))
    before = slot.snapshot()
    assert not slot.publish(4, ('busy',))
    assert slot.snapshot() == before
    slot.launch()
    before = slot.snapshot()
    assert not slot.complete(3, 'output')  # Output depends on graph completion.
    assert slot.snapshot() == before
    assert slot.complete(3, 'graph')
    before = slot.snapshot()
    assert not slot.publish(4, ('output-still-reading',))
    assert slot.snapshot() == before
    assert slot.complete(3, 'output')
    assert slot.publish(4, ('current',))
    slot.launch()
    before = slot.snapshot()
    assert not slot.complete(3, 'output')
    assert not slot.complete(3, 'graph')
    assert slot.snapshot() == before
    assert slot.complete(4, 'graph')
    before = slot.snapshot()
    assert not slot.complete(4, 'graph')
    assert slot.snapshot() == before
    assert slot.complete(4, 'output')
    print('OBS generation_orders=24 accepted_record_maxima=%d' % accepted_count)
    print('OBS newer_then_older=rejected_unchanged; missing_watermark_negative=overwrites')
    print('OBS busy,output_hold,old_completion,duplicate_completion gates checked')


def formula_slots(lifetime, period):
    if lifetime < 0 or period <= 0:
        raise ValueError('invalid lifetime or period')
    return max(1, (lifetime + period - 1) // period)


def cyclic_conflict(lifetime, period, count, jobs=64):
    for a in range(jobs):
        for b in range(a):
            if a % count == b % count:
                if max(a * period, b * period) < min(a * period + lifetime,
                                                       b * period + lifetime):
                    return b, a
    return None


def check_slots_and_padding():
    cases = [(0, 8), (8, 8), (9, 8), (10, 3), (10, 6), (1, 1000)]
    observations = []
    for lifetime, period in cases:
        reference = next(n for n in range(1, 65)
                         if cyclic_conflict(lifetime, period, n) is None)
        assert formula_slots(lifetime, period) == reference
        observations.append((lifetime, period, reference))
    conflict = cyclic_conflict(10, 3, formula_slots(10, 3) - 1)
    assert conflict is not None
    actual = layout((Request(0, 2, 1), Request(1, 1, 1)), Bucket(3, 8, 4, 4, 4))
    expected = {(0, 0): 1, (0, 1): 1, (1, 0): 1}

    def write_rows(guard_tail):
        kv = {}
        for owner in actual['rows']:
            if owner is None:
                if guard_tail:
                    continue
                owner = (0, 0)  # Fault: default padded destination aliases real KV.
            kv[owner] = kv.get(owner, 0) + 1
        return kv

    assert write_rows(True) == expected
    assert write_rows(False) != expected
    assert write_rows(False)[(0, 0)] == 6
    print('OBS slot_formula_reference=%r' % observations)
    print('OBS undersized_ring_negative_conflict=%r' % (conflict,))
    print('OBS guarded_tail_writes=3; unguarded_negative_real_destination_writes=6')


def dense_attention(items):
    if not items:
        return None
    m = max(score for score, value in items)
    weights = [exp(score - m) for score, value in items]
    denominator = sum(weights)
    output = sum(w * item[1] for w, item in zip(weights, items)) / denominator
    return output, m + log(denominator)


def merge_attention(a, b):
    if a is None:
        return b
    if b is None:
        return a
    oa, la = a
    ob, lb = b
    m = max(la, lb)
    wa, wb = exp(la - m), exp(lb - m)
    return (wa * oa + wb * ob) / (wa + wb), m + log(wa + wb)


def check_attention_and_capacity_arithmetic():
    datasets = [[], [(0.0, 0.0), (0.0, 2.0)],
                [(1000.0, 3.0), (999.0, -2.0), (-1000.0, 7.0)]]
    comparisons = 0
    for items in datasets:
        for cut in range(len(items) + 1):
            reference = dense_attention(items)
            actual = merge_attention(dense_attention(items[:cut]), dense_attention(items[cut:]))
            if reference is None:
                assert actual is None
            else:
                assert all(isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                           for a, b in zip(actual, reference))
            comparisons += 1
    items = [(0.0, 0.0), (0.0, 2.0)]
    wrong = merge_attention(dense_attention(items[:1]), dense_attention(items))
    assert isclose(wrong[0], 2.0 / 3.0)
    assert not isclose(wrong[0], dense_attention(items)[0])
    pool_bytes = 4 * 64 * 1024 * 1024
    extra_disjoint_workspace = 2 * 8 * 1024 * 1024
    kv_per_token = 128 * 1024
    assert pool_bytes // kv_per_token == 2048
    assert extra_disjoint_workspace // kv_per_token == 128
    assert (pool_bytes + extra_disjoint_workspace) // kv_per_token == 2176
    print('OBS attention_partition_reference_comparisons=%d; overlap_negative=2/3 versus 1' % comparisons)
    print('OBS disjoint_pool_and_workspace_token_equivalent=2176')


def main():
    check_capacity()
    check_binding()
    check_generations()
    check_slots_and_padding()
    check_attention_and_capacity_arithmetic()
    print('OBS scope=CPU discrete layout/state/math model; no CUDA API, GPU measurements or performance claims')


if __name__ == '__main__':
    main()
