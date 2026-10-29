#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Read-only, offline, deterministic CPU models for K-ALG-014.

Only Python standard library is used. No project imports, filesystem access,
network access, subprocesses, or device measurements. Arithmetic references
use exact rational numbers; this is not a GPU floating-point validation.
'''
from collections import namedtuple
from fractions import Fraction as F

Adapter = namedtuple('Adapter', 'A B scale')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def reject_unless(condition, message):
    if not condition:
        raise ValueError(message)


def expect_rejection(operation, message):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError(message)


def validate(W, x, adapter):
    di, do = len(x), len(W)
    reject_unless(di > 0 and do > 0, 'base dimensions must be positive')
    reject_unless(all(len(row) == di for row in W), 'W must be do by di')
    r = len(adapter.A)
    reject_unless(all(len(row) == di for row in adapter.A), 'A must be r by di')
    reject_unless(len(adapter.B) == do, 'B must have do output rows')
    reject_unless(all(len(row) == r for row in adapter.B), 'B must be do by r')
    return di, do, r


def branch(W, x, adapter):
    di, do, r = validate(W, x, adapter)
    u = [sum((adapter.A[k][t] * x[t] for t in range(di)), F(0))
         for k in range(r)]
    return [sum((W[o][t] * x[t] for t in range(di)), F(0))
            + adapter.scale * sum((adapter.B[o][k] * u[k]
                                   for k in range(r)), F(0))
            for o in range(do)]


def dense_reference(W, x, adapter):
    # Independent association: explicitly materialize the dense updated matrix.
    di, do, r = validate(W, x, adapter)
    merged = []
    for o in range(do):
        row = []
        for t in range(di):
            coefficient = F(W[o][t])
            for k in range(r):
                coefficient += adapter.scale * adapter.B[o][k] * adapter.A[k][t]
            row.append(coefficient)
        merged.append(row)
    result = []
    for row in merged:
        value = F(0)
        for coefficient, input_value in zip(row, x):
            value += coefficient * input_value
        result.append(value)
    return result


def fixtures():
    W = [[1, 0, 2], [0, 1, -1]]
    a = Adapter([[1, 0, 0], [0, 1, 0]], [[1, 0], [0, 1]], F(7, 10))
    # Regression repair: for do=2 and r=1, B is 2 by 1, not 1 by 2.
    b = Adapter([[1, 1, 0]], [[1], [-1]], F(1, 2))
    c = Adapter([[0, 1, 1]], [[2], [-1]], F(1, 3))
    base = Adapter([], [[], []], F(0))
    return W, a, b, c, base


def padded(adapter, di, width):
    r = len(adapter.A)
    reject_unless(width >= r, 'padding width smaller than rank')
    return Adapter([list(row) for row in adapter.A]
                   + [[0] * di for _ in range(width - r)],
                   [list(row) + [0] * (width - r) for row in adapter.B],
                   adapter.scale)


def check_arithmetic_and_shapes():
    W, a, b, c, base = fixtures()
    count = 0
    for x in ([1, 2, 1], [-1, 3, 2], [0, 0, 0], [2, -1, 1]):
        for adapter in (a, b, c, base):
            require(branch(W, x, adapter) == dense_reference(W, x, adapter),
                    'two-stage arithmetic differs from dense reference')
            count += 1
    x = [1, 2, 1]
    base_result = dense_reference(W, x, base)
    zero_a = Adapter([[0, 0, 0], [0, 0, 0]], a.B, a.scale)
    zero_b = Adapter(a.A, [[0, 0], [0, 0]], a.scale)
    zero_scale = Adapter(a.A, a.B, F(0))
    for adapter in (zero_a, zero_b, zero_scale, base):
        require(branch(W, x, adapter) == base_result, 'zero increment changed output')
    pad = padded(a, 3, 4)
    reference = dense_reference(W, x, a)
    require(branch(W, x, pad) == reference, 'zero padding changed output')
    dirty_a = [list(row) for row in pad.A]
    dirty_b = [list(row) for row in pad.B]
    dirty_a[2][0] = 1
    dirty_b[0][2] = 1
    dirty = Adapter(dirty_a, dirty_b, pad.scale)
    require(branch(W, x, dirty) != reference, 'nonzero padding fault was not detected')
    wrong_scale = Adapter(pad.A, pad.B, pad.scale * F(2, 4))
    require(branch(W, x, wrong_scale) != reference,
            'recomputing alpha/R instead of retaining alpha/r was not detected')
    malformed = Adapter(b.A, [[1, -1]], b.scale)
    expect_rejection(lambda: branch(W, x, malformed),
                     'original 1-by-2 B regression was accepted')
    expect_rejection(lambda: dense_reference(W, x, malformed),
                     'reference accepted malformed B')
    expect_rejection(lambda: branch(W, x, Adapter([[1, 2]], b.B, b.scale)),
                     'wrong A input width was accepted')
    expect_rejection(lambda: branch(W, x, Adapter(b.A, [[1, 2], [-1, 0]], b.scale)),
                     'wrong B inner width was accepted')
    return ('exact_reference_cases=%d rank_zero_and_zero_increment=1 '
            'original_B_shape_rejected=1 padding_and_scale_faults_detected=1' % count)


def grouped(W, rows, identities, objects, fault=None):
    reject_unless(len(rows) == len(identities), 'row identity count mismatch')
    if not rows:
        return []
    groups = {}
    for i, identity in enumerate(identities):
        validate(W, rows[i], objects[identity])
        groups.setdefault(identity, []).append(i)
    output = [[sum((W[o][t] * x[t] for t in range(len(x))), F(0))
               for o in range(len(W))] for x in rows]
    first_by_rank = {}
    for identity in groups:
        adapter = objects[identity]
        first_by_rank.setdefault(len(adapter.A), adapter)
    append_only = []
    # Deliberately use a different execution order from input order.
    for identity in reversed(list(groups)):
        adapter = objects[identity]
        if fault == 'rank_only':
            adapter = first_by_rank[len(adapter.A)]
        r = len(adapter.A)
        indices = groups[identity]
        intermediate = [[sum((adapter.A[k][t] * rows[i][t]
                              for t in range(len(rows[i]))), F(0))
                         for k in range(r)] for i in indices]
        for j, i in enumerate(indices):
            value = list(output[i])
            for o in range(len(W)):
                value[o] += adapter.scale * sum((adapter.B[o][k] * intermediate[j][k]
                                                for k in range(r)), F(0))
            output[i] = value
            append_only.append(value)
    return append_only if fault == 'no_scatter' else output


def check_grouped_rows():
    W, a, b, c, base = fixtures()
    require(len(b.A) == len(c.A) == 1 and b != c,
            'same-rank negative control fixture is invalid')
    objects = {('a', 1): a, ('b', 1): b, ('c', 1): c, ('base', 0): base}
    rows = [[1, 2, 1], [-1, 3, 2], [2, -1, 1], [0, 1, -2], [3, 0, 1], [1, 0, 0]]
    ids = [('a', 1), ('b', 1), ('c', 1), ('b', 1), ('base', 0), ('a', 1)]
    reference = [dense_reference(W, x, objects[identity]) for x, identity in zip(rows, ids)]
    require(grouped(W, rows, ids, objects) == reference, 'grouped inverse mapping failed')
    require(grouped(W, rows, ids, objects, 'rank_only') != reference,
            'same-rank parameter substitution was not detected')
    require(grouped(W, rows, ids, objects, 'no_scatter') != reference,
            'missing inverse scatter was not detected')
    require(grouped(W, [], [], objects) == [], 'empty batch must be a no-op')
    return 'rows=6 mixed_rank=1 same_rank_wrong_adapter_detected=1 missing_scatter_detected=1 empty_batch=1'


def check_paged_weights():
    W, a, b, c, base = fixtures()
    page_elements = 3
    memory = {p: [F(999)] * page_elements for p in range(20)}

    def pack(adapter, pages):
        flat = [v for row in adapter.A for v in row]
        flat += [v for row in adapter.B for v in row]
        needed = (len(flat) + page_elements - 1) // page_elements
        require(len(pages) == needed, 'page fixture length mismatch')
        for index, value in enumerate(flat):
            logical, offset = divmod(index, page_elements)
            memory[pages[logical]][offset] = F(value)
        return (adapter, tuple(pages), len(flat))

    da = pack(a, [7, 1, 9, 4])
    db = pack(b, [12, 2])
    dz = pack(base, [])
    snapshot = {p: tuple(values) for p, values in memory.items()}

    def execute(x, descriptor, wrong_page=False):
        adapter, pages, length = descriptor
        di, do, r = validate(W, x, adapter)

        def load(index):
            reject_unless(0 <= index < length, 'logical weight offset out of bounds')
            logical, offset = divmod(index, page_elements)
            physical = logical if wrong_page else pages[logical]
            return memory[physical][offset]

        u = [sum((load(k * di + t) * x[t] for t in range(di)), F(0))
             for k in range(r)]
        b_offset = r * di
        return [sum((W[o][t] * x[t] for t in range(di)), F(0))
                + adapter.scale * sum((load(b_offset + o * r + k) * u[k]
                                       for k in range(r)), F(0))
                for o in range(do)]

    x = [1, 2, 1]
    for descriptor in (da, db, dz):
        require(execute(x, descriptor) == dense_reference(W, x, descriptor[0]),
                'non-contiguous gather differs from dense reference')
    require(execute(x, da, True) != dense_reference(W, x, a),
            'ignoring the physical page map was not detected')
    require(snapshot == {p: tuple(values) for p, values in memory.items()},
            'weight reads changed memory')
    return 'noncontiguous_and_cross_page_gather=1 rank_zero=1 ignored_page_map_detected=1 reads_preserve_storage=1'


def state_key(identity, tokens, parent=(), position='rope', dtype='fp16', tp=0,
              salt='tenant-a', omit_version=False):
    object_part = identity[0] if omit_version else identity
    return ('base-v1', object_part, parent, tuple(tokens), position, dtype, tp, salt)


def check_cache_identity():
    W, a, b, c, base = fixtures()
    x, tokens = [1, 2, 1], [10, 11, 12, 13]
    old_value = tuple(dense_reference(W, x, b))
    new_value = tuple(dense_reference(W, x, c))
    require(old_value != new_value, 'cache payloads must expose wrong reuse')
    k1 = state_key(('a', 1), tokens)
    k2 = state_key(('a', 2), tokens)
    directory = {k1: (True, True, old_value)}

    def lookup(key):
        item = directory.get(key)
        if item is None or not item[0] or not item[1]:
            return None
        return item[2]

    require(lookup(k1) == old_value and lookup(k2) is None,
            'versioned cache lookup failed')
    variants = [state_key(('b', 1), tokens),
                state_key(('a', 1), tokens, parent=('different-prefix',)),
                state_key(('a', 1), tokens, position='different-position'),
                state_key(('a', 1), tokens, dtype='different-format'),
                state_key(('a', 1), tokens, tp=1),
                state_key(('a', 1), tokens, salt='tenant-b')]
    require(all(key != k1 and lookup(key) is None for key in variants),
            'state dependency was omitted')
    directory[k2] = (True, False, new_value)
    require(lookup(k2) is None, 'unfinished state was published')
    directory[k2] = (False, True, new_value)
    require(lookup(k2) is None, 'partial block entered full-block cache')
    directory[k2] = (True, True, new_value)
    require(lookup(k2) == new_value, 'completed new state could not be used')
    bad_directory = {state_key(('a', 1), tokens, omit_version=True): old_value}
    faulty_hit = bad_directory[state_key(('a', 2), tokens, omit_version=True)]
    require(faulty_hit != new_value, 'omitted-version wrong payload was not detected')
    return 'version_and_dependencies_isolated=1 completion_and_full_block_gated=1 bad_cache_hit_returns_wrong_state=1'


def ceil_pages(byte_count, page):
    reject_unless(byte_count >= 0 and page > 0, 'invalid capacity input')
    return (byte_count + page - 1) // page


def check_capacity():
    page, kv_token, capacity = 1 << 20, 128 * 1024, 64

    def occupancy(adapters, tokens):
        return adapters * 16 + ceil_pages(tokens * kv_token, page)

    require(occupancy(1, 384) == 64, 'exact fit changed')
    require(occupancy(1, 385) == 65, 'one-token page overflow missed')
    require(occupancy(3, 200) == 73, 'shared-pool overflow changed')
    bad_double_budget = 3 * 16 <= capacity and ceil_pages(200 * kv_token, page) <= capacity
    require(bad_double_budget and occupancy(3, 200) > capacity,
            'duplicated-capacity negative control did not expose false acceptance')
    require(ceil_pages(400 * kv_token, page) == 50 <= capacity,
            'common pool should accept 400 tokens without adapters')
    require(ceil_pages(400 * kv_token, page) > 48,
            'fixed 48-page KV partition should show stranded adapter capacity')
    refs = [set(range(16)), set(range(16)), set(range(16, 64))]
    unique = set().union(*refs)
    require(len(unique) == capacity and sum(map(len, refs)) == 80,
            'shared physical pages were not distinguished from references')
    require(sum(ceil_pages(n, page) for n in (page // 2, page // 2)) == 2,
            'per-object rounding failed')
    require(ceil_pages(page, page) == 1, 'packed-object rounding failed')
    # The chapter update trajectory: 16+24, then 16+16+24, then another 16.
    require(16 + 24 == 40 and 16 + 16 + 24 == 56,
            'trajectory occupancy changed')
    require(16 + 16 + 16 + 24 == 72 > capacity,
            'new version was admitted without peak capacity')
    return 'exact_fit=64 one_token_overflow=65 three_adapter_demand=73 duplicate_budget_fault_detected=1 shared_pages_once=1 rounding_and_update_peak=1'


class Slot:
    def __init__(self, index, generation, version, payload):
        self.index = index
        self.generation = generation
        self.version = version
        self.payload = payload
        self.writes = 1
        self.ready = False
        self.retired = False
        self.owners = 0
        self.readers = set()
        self.next_reader = 0

    def complete_load(self, success):
        reject_unless(self.writes == 1, 'load already completed')
        self.writes = 0
        self.ready = bool(success)
        self.retired = not success

    def pin(self):
        reject_unless(self.ready and not self.retired, 'object not published for new use')
        self.owners += 1

    def release(self):
        reject_unless(self.owners > 0, 'logical reference underflow')
        self.owners -= 1

    def start_read(self):
        reject_unless(self.ready and not self.retired and self.owners > 0,
                      'read lacks a live protected binding')
        token = (self.index, self.generation, self.next_reader)
        self.next_reader += 1
        self.readers.add(token)
        return token

    def finish_read(self, token):
        if token[:2] != (self.index, self.generation) or token not in self.readers:
            return False
        self.readers.remove(token)
        return True

    def can_reclaim(self):
        return self.retired and self.owners == 0 and not self.readers and self.writes == 0


def check_version_lifecycle():
    W, a, b, c, base = fixtures()
    x = [1, 2, 1]
    old_reference = dense_reference(W, x, b)
    old = Slot(0, 1, 1, b)
    expect_rejection(old.pin, 'unfinished adapter became readable')
    old.complete_load(True)
    old.pin()  # Request binding.
    old.pin()  # Graph binding is another logical owner in this model.
    read = old.start_read()
    new = Slot(1, 1, 2, c)
    new.complete_load(True)
    new.pin()
    require(dense_reference(W, x, old.payload) == old_reference,
            'publishing v2 changed v1')
    require(dense_reference(W, x, new.payload) != old_reference,
            'version fixture does not distinguish results')
    old.release()  # Cancellation removes the request owner, not existing reads.
    old.retired = True
    require(not old.can_reclaim(), 'cancelled old object was reclaimed while in use')
    require(old.finish_read(read), 'real reader completion was lost')
    require(not old.can_reclaim(), 'graph reference was ignored')
    old.release()
    require(old.can_reclaim(), 'fully drained retired object did not become reclaimable')
    require(not old.finish_read(read), 'duplicate completion was accepted')
    reused = Slot(0, 2, 3, a)
    reused.complete_load(True)
    reused.pin()
    new_read = reused.start_read()
    require(not reused.finish_read(read) and new_read in reused.readers,
            'old completion released a new-generation reader')
    failed = Slot(2, 1, 4, c)
    failed.complete_load(False)
    expect_rejection(failed.pin, 'failed load was published')
    require(failed.can_reclaim(), 'failed, completed load leaked its model reservation')
    bad = Slot(3, 1, 1, b)
    bad.complete_load(True)
    bad.pin()
    bad.start_read()
    bad.release()
    bad.retired = True
    require(not bad.can_reclaim(), 'negative-control setup lacks an active reader')
    bad.payload = c  # Deliberate bypass of the reclamation gate.
    require(dense_reference(W, x, bad.payload) != old_reference,
            'in-place overwrite did not expose old-reader corruption')
    mixed = Adapter(b.A, c.B, b.scale)
    require(dense_reference(W, x, mixed) != old_reference,
            'separate A/B publication fault was not detected')
    return 'loading_and_failure_gated=1 cancel_and_graph_refs_protected=1 stale_completion_rejected=1 overwrite_and_mixed_AB_faults_detected=1'


def tp_output_partition(W, x, adapter, devices):
    di, do, r = validate(W, x, adapter)
    reject_unless(devices > 0 and do % devices == 0 and r % devices == 0,
                  'output/rank partition requires divisible dimensions')
    rank_chunk, output_chunk = r // devices, do // devices
    gathered = []
    for p in range(devices):
        for k in range(p * rank_chunk, (p + 1) * rank_chunk):
            gathered.append(sum((adapter.A[k][t] * x[t] for t in range(di)), F(0)))
    output = []
    for p in range(devices):
        for o in range(p * output_chunk, (p + 1) * output_chunk):
            output.append(sum((W[o][t] * x[t] for t in range(di)), F(0))
                          + adapter.scale * sum((adapter.B[o][k] * gathered[k]
                                                 for k in range(r)), F(0)))
    return output


def tp_input_partition(W, x, adapter, devices, duplicate_delta=False):
    di, do, r = validate(W, x, adapter)
    reject_unless(devices > 0 and di % devices == 0 and do % devices == 0,
                  'input/output partition requires divisible dimensions')
    input_chunk, output_chunk = di // devices, do // devices
    partial_u = []
    for p in range(devices):
        partial_u.append([sum((adapter.A[k][t] * x[t]
                               for t in range(p * input_chunk, (p + 1) * input_chunk)), F(0))
                          for k in range(r)])
    u = [sum((partial_u[p][k] for p in range(devices)), F(0)) for k in range(r)]
    partial_outputs = []
    for p in range(devices):
        local = [sum((W[o][t] * x[t]
                      for t in range(p * input_chunk, (p + 1) * input_chunk)), F(0))
                 for o in range(do)]
        owned = range(do) if duplicate_delta else range(p * output_chunk, (p + 1) * output_chunk)
        for o in owned:
            local[o] += adapter.scale * sum((adapter.B[o][k] * u[k]
                                            for k in range(r)), F(0))
        partial_outputs.append(local)
    return [sum((partial_outputs[p][o] for p in range(devices)), F(0)) for o in range(do)]


def check_tp_dataflow():
    W = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, -1]]
    first = Adapter([[1, 0, -1, 0], [0, 1, 0, 1]],
                    [[1, 0], [0, 1], [1, -1], [2, 1]], F(1, 2))
    second = Adapter([[1, 1, 0, 0], [0, 0, 1, 1]],
                     [[1, 0], [0, 1], [1, 1], [-1, 1]], F(1, 3))
    x = [1, -2, 3, 1]
    first_reference = dense_reference(W, x, first)
    activated = [max(F(0), value) for value in first_reference]
    final_reference = dense_reference(W, activated, second)
    for devices in (1, 2):
        local_first = tp_output_partition(W, x, first, devices)
        require(local_first == first_reference, 'output-partition gather is incorrect')
        local_activation = [max(F(0), value) for value in local_first]
        require(tp_input_partition(W, local_activation, second, devices) == final_reference,
                'two-layer partitioned dataflow differs from dense reference')
    bad = tp_input_partition(W, activated, second, 2, True)
    require(bad != final_reference, 'duplicating the complete delta on each rank was not detected')
    base_only = Adapter([], [[], [], [], []], F(0))
    base_value = dense_reference(W, activated, base_only)
    require([bad[i] - base_value[i] for i in range(4)]
            == [2 * (final_reference[i] - base_value[i]) for i in range(4)],
            'negative-control error is not the predicted N-fold delta')
    return 'two_layer_with_nonlinearity_devices_1_and_2=1 dense_reference_equal=1 N_fold_delta_fault_detected=1'


def ring_ledger(devices, rows, width, phases):
    # Explicit per-rank sends for a balanced ring: each phase has N-1 steps.
    ledger = [[] for _ in range(devices)]
    for _ in range(phases):
        for _ in range(devices - 1):
            for p in range(devices):
                ledger[p].append(F(rows * width, devices))
    return [sum(messages, F(0)) for messages in ledger]


def check_communication_ledger():
    rows, rank, hidden = 3, 4, 8
    for devices in (1, 2, 4):
        base = ring_ledger(devices, rows, hidden, 2)
        gathers = [ring_ledger(devices, rows, rank, 1) for _ in range(3)]
        reduction = ring_ledger(devices, rows, rank, 2)
        extra = [sum((g[p] for g in gathers), F(0)) + reduction[p]
                 for p in range(devices)]
        expected_base = F(2 * (devices - 1) * rows * hidden, devices)
        expected_extra = F(5 * (devices - 1) * rows * rank, devices)
        require(all(value == expected_base for value in base), 'base ring ledger mismatch')
        require(all(value == expected_extra for value in extra), 'LoRA ring ledger mismatch')
        if devices == 1:
            require(base == extra == [0], 'single-rank communication must be zero')
        else:
            require(extra[0] / base[0] == F(5 * rank, 2 * hidden),
                    'conditional communication ratio changed')
            faulty = sum((g[0] for g in gathers), F(0))
            require(faulty != extra[0], 'omitted low-rank all-reduce was not detected')
    return 'balanced_ring_element_ledger_devices_1_2_4=1 single_rank_zero=1 missing_collective_detected=1 no_latency_claim=1'


def main():
    checks = [
        ('arithmetic_and_shapes', check_arithmetic_and_shapes),
        ('grouped_rows', check_grouped_rows),
        ('paged_weights', check_paged_weights),
        ('cache_identity', check_cache_identity),
        ('capacity', check_capacity),
        ('version_lifecycle', check_version_lifecycle),
        ('tp_dataflow', check_tp_dataflow),
        ('communication_ledger', check_communication_ledger),
    ]
    for name, operation in checks:
        print('%s: %s' % (name, operation()))
    print('scope: CPU exact rational arithmetic and finite state models only; no GPU, runtime allocator, graph, network, or performance validation')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
