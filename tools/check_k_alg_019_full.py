from fractions import Fraction as Q
from itertools import combinations, product


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rejected(action):
    try:
        action()
    except ValueError:
        return
    raise RuntimeError('Negative control escaped detection')


def weights(version, expert):
    return (((expert + 1, -1, version), (1, version + expert, 2)),
            ((1, expert + 1), (version, -1), (2, 1 - expert)))


def direct_expert(x, version, expert):
    up, down = weights(version, expert)
    h = tuple(max(Q(0), x[0] * up[0][j] + x[1] * up[1][j])
              for j in range(3))
    return tuple(h[0] * down[0][j] + h[1] * down[1][j]
                 + h[2] * down[2][j] for j in range(2))


def matmul(a, b):
    result = [[Q(0) for _ in range(len(b[0]))] for _ in a]
    for i in range(len(a)):
        for k in range(len(b)):
            for j in range(len(b[0])):
                result[i][j] += a[i][k] * b[k][j]
    return result


def grouped_expert(rows, loaded_weights):
    up, down = loaded_weights
    hidden = matmul(rows, up)
    for row in hidden:
        for j in range(len(row)):
            row[j] = max(Q(0), row[j])
    return matmul(hidden, down)


class Slot:
    def __init__(self, capacity=1):
        self.capacity = capacity
        self.lease = 0
        self.serial = 0
        self.state = 'FREE'
        self.key = None
        self.data = None
        self.active = {}

    def ticket(self, kind):
        self.serial += 1
        ticket = (self.lease, kind, self.serial)
        self.active[ticket] = kind
        return ticket

    def begin_copy(self, key, size=1):
        require(0 < size <= self.capacity, 'Object does not fit')
        require(self.state != 'COPYING' and not self.active, 'Slot busy')
        self.lease += 1
        self.key = key
        self.data = None
        self.state = 'COPYING'
        return self.ticket('copy')

    def copied(self, ticket, data, success=True):
        if self.active.get(ticket) != 'copy':
            return False
        del self.active[ticket]
        self.state = 'READY' if success else 'FAILED'
        self.data = data if success else None
        return True

    def begin_read(self, key):
        require(self.state == 'READY' and self.key == key,
                'Not ready or wrong key')
        return self.ticket('read')

    def finished(self, ticket):
        if self.active.get(ticket) != 'read':
            return False
        del self.active[ticket]
        return True


def validate_routes(x, routes):
    require(set(x) == set(routes), 'Token domain mismatch')
    for token, items in routes.items():
        ids = [expert for expert, gate in items]
        require(len(ids) == len(set(ids)), 'Repeated expert in route')
        require(all(expert >= 0 for expert in ids), 'Invalid expert')
        require(all(gate > 0 for expert, gate in items), 'Invalid gate')
        require(not items or sum(gate for expert, gate in items) == 1,
                'Synthetic test requires normalized gates')


def direct_layer(version, x, routes):
    validate_routes(x, routes)
    result = {}
    for token in x:
        result[token] = tuple(sum((gate * direct_expert(x[token], version, expert)[j]
                                  for expert, gate in routes[token]), Q(0))
                              for j in range(2))
    return result


def offloaded_records(generation, version, x, routes, slot):
    validate_routes(x, routes)
    buckets = {}
    for token, items in routes.items():
        for expert, gate in items:
            buckets.setdefault(expert, []).append((token, gate))
    pending = set(buckets)
    records = []
    loads = []
    while pending:
        hits = [e for e in pending
                if slot.state == 'READY' and slot.key == (version, e)]
        expert = min(hits) if hits else min(pending)
        key = (version, expert)
        if not hits:
            transfer = slot.begin_copy(key)
            require(slot.copied(transfer, weights(version, expert)),
                    'Copy completion lost')
            loads.append(key)
        read = slot.begin_read(key)
        items = buckets[expert]
        values = grouped_expert([x[token] for token, gate in items], slot.data)
        for (token, gate), value in zip(items, values):
            records.append((generation, token, expert, version,
                            tuple(gate * v for v in value)))
        require(slot.finished(read), 'Read completion lost')
        pending.remove(expert)
    return records, loads


def combine(generation, version, x, routes, records, eligible=True):
    expected = {(token, expert) for token, items in routes.items()
                for expert, gate in items}
    seen = set()
    result = {token: [Q(0), Q(0)] for token in x}
    for record_generation, token, expert, record_version, value in records:
        require(record_generation == generation and record_version == version,
                'Wrong invocation or version')
        item = (token, expert)
        require(item in expected and item not in seen,
                'Unknown or duplicate item')
        require(len(value) == 2, 'Wrong output width')
        seen.add(item)
        for j in range(2):
            result[token][j] += value[j]
    require(seen == expected, 'Missing contribution')
    require(eligible, 'Cancelled invocation cannot publish')
    return {token: tuple(value) for token, value in result.items()}


def check_experts():
    x = {(0, 0): (Q(2), Q(-1)), (1, 0): (Q(-1), Q(3)),
         (1, 2): (Q(1), Q(2))}
    routes = {(0, 0): ((2, Q(1, 4)), (0, Q(3, 4))),
              (1, 0): ((1, Q(1, 2)), (2, Q(1, 2))),
              (1, 2): ((0, Q(1, 4)), (1, Q(3, 4)))}
    slot = Slot()
    cases = 0
    saved = None
    for version in (1, 2):
        records, loads = offloaded_records(7, version, x, routes, slot)
        require(len(loads) == 3 and len(set(loads)) == 3,
                'Not once per cold expert')
        reference = direct_layer(version, x, routes)
        for shift in range(len(records)):
            order = list(reversed(records[shift:] + records[:shift]))
            require(combine(7, version, x, routes, order) == reference,
                    'Grouped execution differs from direct formula')
            cases += 1
        if version == 1:
            saved = records
    reference = direct_layer(1, x, routes)
    rejected(lambda: combine(7, 1, x, routes, saved[1:]))
    rejected(lambda: combine(7, 1, x, routes, saved + [saved[0]]))
    wrong = list(saved)
    g, token, expert, version, value = wrong[0]
    wrong[0] = (g - 1, token, expert, version, value)
    rejected(lambda: combine(7, 1, x, routes, wrong))
    wrong = list(saved)
    wrong[0] = (g, token, expert, version + 1, value)
    rejected(lambda: combine(7, 1, x, routes, wrong))
    swapped = list(saved)
    indices = [i for i, record in enumerate(swapped) if record[2] == 2]
    a, b = indices
    swapped[a] = swapped[a][:4] + (saved[b][4],)
    swapped[b] = swapped[b][:4] + (saved[a][4],)
    rejected(lambda: require(combine(7, 1, x, routes, swapped) == reference,
                             'Same expert, wrong token value'))
    rejected(lambda: combine(7, 1, x, routes, saved, eligible=False))
    empty, loads = offloaded_records(8, 1, {}, {}, Slot())
    require(empty == [] and loads == [] and combine(8, 1, {}, {}, empty) == {},
            'Empty batch')
    zero_routes = {token: () for token in x}
    records, loads = offloaded_records(9, 1, x, zero_routes, Slot())
    require(not loads and combine(9, 1, x, zero_routes, records)
            == direct_layer(1, x, zero_routes), 'Empty routed branch')
    warm = Slot()
    copy = warm.begin_copy((1, 2))
    warm.copied(copy, weights(1, 2))
    records, loads = offloaded_records(7, 1, x, routes, warm)
    require(len(loads) == 2 and combine(7, 1, x, routes, records) == reference,
            'Initial cache hit evicted before use')
    print('Synthetic rational ReLU FFN: %d output-order cases; six combine faults detected; empty and warm-cache cases passed' % cases)


def check_slots():
    slot = Slot()
    copy = slot.begin_copy((1, 0))
    rejected(lambda: slot.begin_read((1, 0)))
    rejected(lambda: slot.begin_copy((2, 0)))
    slot.copied(copy, weights(1, 0))
    first = slot.begin_read((1, 0))
    second = slot.begin_read((1, 0))
    slot.finished(first)
    rejected(lambda: slot.begin_copy((2, 0)))
    slot.finished(second)
    newer = slot.begin_copy((2, 0))
    require(not slot.copied(copy, weights(1, 0)) and newer in slot.active,
            'Stale completion changed current responsibility')
    slot.copied(newer, weights(2, 0))
    rejected(lambda: slot.begin_read((1, 0)))
    snapshot = (slot.lease, slot.key, slot.state)
    rejected(lambda: slot.begin_copy((3, 0), size=2))
    require(snapshot == (slot.lease, slot.key, slot.state),
            'Rejected allocation mutated slot')
    bad_copy = slot.begin_copy((3, 0))
    slot.copied(bad_copy, None, success=False)
    rejected(lambda: slot.begin_read((3, 0)))
    print('One-slot abstract lifecycle: six unsafe operations rejected; multiple readers, stale completion and failed copy covered')


def check_unions():
    cases = 0
    for experts in range(1, 5):
        for k in range(experts + 1):
            choices = tuple(combinations(range(experts), k))
            for n in range(4):
                total = 0
                count = 0
                for route_tuple in product(choices, repeat=n):
                    touched = set()
                    for route in route_tuple:
                        touched.update(route)
                    total += len(touched)
                    count += 1
                enumerated = Q(total, count)
                formula = experts * (1 - (1 - Q(k, experts)) ** n)
                require(enumerated == formula, 'Union expectation differs')
                cases += 1
    require(3 * (1 - Q(2, 3) ** 3) != 1,
            'Correlated all-same routing should refute independent formula')
    print('Independent uniform routes: %d exact enumeration cases; correlated-route counterexample distinguished' % cases)


def dense(n, layers, microbatch, rounds, layer_major, reverse=False):
    values = [i - 2 for i in range(n)]
    visits = []
    order = list(range(layers))
    if reverse:
        order.reverse()
    for generation in range(rounds):
        if layer_major:
            pairs = [(layer, start) for layer in order
                     for start in range(0, n, microbatch)]
        else:
            pairs = [(layer, start) for start in range(0, n, microbatch)
                     for layer in order]
        for layer, start in pairs:
            visits.append((generation, start, layer))
            for row in range(start, min(n, start + microbatch)):
                values[row] = max(0, values[row] * (layer + 2)
                                  + (3 if layer == 0 else -1))
    return values, visits


def simulate_loads(sequence, sizes, slot_count, reset_before=(), future=False):
    require(slot_count > 0, 'No executable slot')
    require(all(size > 0 for size in sizes), 'Nonpositive object size')
    sequence = list(sequence)
    resets = set(reset_before)
    capacity = max(sizes, default=1)
    slots = [Slot(capacity) for _ in range(slot_count)]
    installed = [0] * slot_count
    serial = 0
    events = []

    def evict(position, index):
        slot = slots[index]
        if slot.key is None:
            return
        require(not slot.active and slot.state == 'READY',
                'Eviction before last reader')
        events.append(('evict', position, index, slot.key[1], 0))
        slot.state = 'FREE'
        slot.key = None
        slot.data = None

    for position, layer in enumerate(sequence):
        require(0 <= layer < len(sizes), 'Unknown weight')
        if position in resets:
            for index in range(slot_count):
                evict(position, index)
        key = (1, layer)
        hits = [j for j, slot in enumerate(slots)
                if slot.state == 'READY' and slot.key == key]
        if hits:
            index = hits[0]
            events.append(('hit', position, index, layer, 0))
        else:
            free = [j for j, slot in enumerate(slots) if slot.key is None]
            if free:
                index = free[0]
            elif future:
                def next_use(j):
                    cached = slots[j].key[1]
                    try:
                        return sequence.index(cached, position + 1)
                    except ValueError:
                        return len(sequence)
                index = max(range(slot_count), key=next_use)
            else:
                index = min(range(slot_count), key=lambda j: installed[j])
            evict(position, index)
            transfer = slots[index].begin_copy(key, size=sizes[layer])
            require(slots[index].copied(transfer, key), 'Dense copy failed')
            serial += 1
            installed[index] = serial
            events.append(('copy', position, index, layer, sizes[layer]))
        read = slots[index].begin_read(key)
        require(slots[index].data == key, 'Wrong physical slot contents')
        events.append(('read', position, index, layer, 0))
        require(slots[index].finished(read), 'Dense reader not released')
    return events


def audit_loads(sequence, sizes, slot_count, events):
    resident = [None] * slot_count
    arrivals = [0] * len(sequence)
    reads = [0] * len(sequence)
    byte_count = 0
    last_position = -1
    for kind, position, index, layer, size in events:
        require(0 <= position < len(sequence) and 0 <= index < slot_count,
                'Event outside trace or capacity')
        require(position >= last_position, 'Events went backwards')
        last_position = position
        if kind == 'evict':
            require(resident[index] == layer, 'Wrong eviction identity')
            resident[index] = None
        elif kind == 'copy':
            require(resident[index] is None and layer not in resident,
                    'Copy overwrote occupied slot or duplicated a valid copy')
            require(layer == sequence[position] and size == sizes[layer],
                    'Copy range or byte count differs')
            resident[index] = layer
            byte_count += size
            arrivals[position] += 1
        elif kind == 'hit':
            require(resident[index] == layer == sequence[position] and size == 0,
                    'False cache hit')
            arrivals[position] += 1
        else:
            require(kind == 'read' and size == 0,
                    'Unknown event or nonzero read payload')
            require(resident[index] == layer == sequence[position],
                    'Reader consumed wrong weight')
            reads[position] += 1
        require(len([x for x in resident if x is not None]) <= slot_count,
                'Physical capacity exceeded')
    require(all(x == 1 for x in arrivals) and all(x == 1 for x in reads),
            'Every visit requires one hit-or-copy and one read')
    return byte_count


def fifo_reference(sequence, sizes, slot_count, reset_before=()):
    queue = []
    resets = set(reset_before)
    total = 0
    for position, layer in enumerate(sequence):
        if position in resets:
            queue = []
        if layer not in queue:
            if len(queue) == slot_count:
                queue.pop(0)
            queue.append(layer)
            total += sizes[layer]
    return total


def minimum_copies_reference(sequence, slot_count):
    states = {frozenset(): 0}
    for layer in sequence:
        next_states = {}
        for resident, count in states.items():
            if layer in resident:
                choices = [(resident, count)]
            elif len(resident) < slot_count:
                choices = [(resident | {layer}, count + 1)]
            else:
                choices = [((resident - {old}) | {layer}, count + 1)
                           for old in resident]
            for new_resident, new_count in choices:
                next_states[new_resident] = min(
                    next_states.get(new_resident, new_count), new_count)
        states = next_states
    return min(states.values())


def resets_for(visits, force_microbatch=False, keep_rounds=False):
    resets = []
    for position, visit in enumerate(visits):
        generation, start, layer = visit
        if position == 0:
            resets.append(position)
        elif generation != visits[position - 1][0]:
            if not keep_rounds:
                resets.append(position)
        elif force_microbatch and start != visits[position - 1][1]:
            resets.append(position)
    return resets


def checked_traffic(visits, sizes, slot_count=2, force_microbatch=False,
                    keep_rounds=False):
    sequence = [visit[2] for visit in visits]
    resets = resets_for(visits, force_microbatch, keep_rounds)
    events = simulate_loads(sequence, sizes, slot_count, resets)
    actual = audit_loads(sequence, sizes, slot_count, events)
    require(actual == sum(event[4] for event in events if event[0] == 'copy'),
            'Event payload tally differs from replay')
    require(actual == fifo_reference(sequence, sizes, slot_count, resets),
            'Slot execution differs from independent FIFO queue')
    return actual


def visits_as_misses_fault(sequence, sizes):
    return sum(sizes[layer] for layer in sequence)


def check_cache_events():
    row = [0, 1, 2, 0, 1, 2]
    column = [0, 0, 1, 1, 2, 2]
    sizes = (7, 7, 7)
    retained = simulate_loads(row, sizes, 2, future=True)
    grouped = simulate_loads(column, sizes, 2, future=True)
    forced = simulate_loads(row, sizes, 2, reset_before=(0, 3), future=True)
    row_bytes = audit_loads(row, sizes, 2, retained)
    column_bytes = audit_loads(column, sizes, 2, grouped)
    forced_bytes = audit_loads(row, sizes, 2, forced)
    require((row_bytes, column_bytes, forced_bytes) == (28, 21, 42),
            'Three-layer two-slot regression')
    require([event[3] for event in retained if event[0] == 'copy']
            == [0, 1, 2, 1], 'Regression copy identities')
    require([event[3] for event in retained if event[0] == 'hit']
            == [0, 2], 'Cross-microbatch retention absent')
    require(audit_loads(row, sizes, 2, simulate_loads(row, sizes, 2)) == 42,
            'FIFO comparison should have six misses')
    rejected(lambda: require(visits_as_misses_fault(row, sizes) == row_bytes,
                             'Fault counted all visits as misses'))
    rejected(lambda: require(visits_as_misses_fault(column, sizes) == column_bytes,
                             'Fault counted within-layer hits as misses'))
    missing_copy = list(retained)
    del missing_copy[0]
    rejected(lambda: audit_loads(row, sizes, 2, missing_copy))
    wrong_bytes = list(retained)
    first = wrong_bytes[0]
    wrong_bytes[0] = first[:4] + (first[4] + 1,)
    rejected(lambda: audit_loads(row, sizes, 2, wrong_bytes))
    require(audit_loads(row, sizes, 3, simulate_loads(row, sizes, 3)) == 21,
            'All weights fit: later visits should hit')
    require(audit_loads(row, sizes, 1, simulate_loads(row, sizes, 1)) == 42,
            'One slot should force six copies')
    unequal = (2, 5, 9)
    require(audit_loads(row, unequal, 2,
                        simulate_loads(row, unequal, 2, future=True)) == 21,
            'Traffic must use actual object bytes')
    token_order = [0, 1, 0, 1]
    bucket_order = [0, 0, 1, 1]
    require(audit_loads(token_order, (1, 1), 1,
                        simulate_loads(token_order, (1, 1), 1)) == 4,
            'Token-order single-slot thrashing not executed')
    require(audit_loads(bucket_order, (1, 1), 1,
                        simulate_loads(bucket_order, (1, 1), 1)) == 2,
            'Bucket-order single-slot reuse not executed')
    cases = 0
    for length in range(7):
        for sequence in product(range(3), repeat=length):
            for slot_count in range(1, 4):
                events = simulate_loads(sequence, (1, 1, 1), slot_count,
                                        future=True)
                actual = audit_loads(sequence, (1, 1, 1), slot_count, events)
                require(actual == minimum_copies_reference(sequence, slot_count),
                        'Future-use policy differs from exhaustive cache states')
                cases += 1
    print('Cache event execution: %d equal-size oracle cases; retained/layer-major/forced copies = 4/3/6; visit-as-miss and corrupt-log faults detected; actual expert thrashing = 4 versus 2' % cases)


def check_dense():
    cases = 0
    for n, layers, batch, rounds in product(range(8), range(5),
                                           range(1, 5), range(4)):
        reference, row_visits = dense(n, layers, batch, rounds, False)
        result, layer_visits = dense(n, layers, batch, rounds, True)
        require(result == reference, 'Loop interchange changed function')
        sizes = tuple(layer + 1 for layer in range(layers))
        expected = rounds * sum(sizes) if n else 0
        layer_bytes = checked_traffic(layer_visits, sizes)
        forced_bytes = checked_traffic(row_visits, sizes, force_microbatch=True)
        checked_traffic(row_visits, sizes)
        require(layer_bytes == expected, 'Layer-major once-per-cold-layer premise')
        require(forced_bytes == expected * len(range(0, n, batch)),
                'Forced-reload baseline did not reload each microbatch')
        cases += 1
    rejected(lambda: require(dense(5, 3, 2, 1, True, True)[0]
                             == dense(5, 3, 2, 1, False)[0], 'Wrong layer order'))
    _, visits = dense(2, 2, 1, 3, False)
    require(checked_traffic(visits, (2, 5), keep_rounds=True) == 7,
            'Cross-round retained weights were counted again')
    require(checked_traffic(visits, (2, 5)) == 21,
            'Per-round cold reset was not modeled')
    print('Dense loop interchange: %d output cases; retained, forced and layer-major event traffic checked against FIFO reference; empty and short microbatches, cross-round retention and reversed-layer fault covered' % cases)


def recurrence(transfers, computes):
    d, c = [], []
    for i, (transfer, compute) in enumerate(zip(transfers, computes)):
        d.append(max(d[i - 1] if i else 0,
                     c[i - 2] if i >= 2 else 0) + transfer)
        c.append(max(c[i - 1] if i else 0, d[i]) + compute)
    return c[-1] if c else 0


def tick_reference(transfers, computes):
    now = 0
    next_copy = 0
    next_compute = 0
    dma = None
    alu = None
    slots = [None, None]
    ready = set()
    n = len(transfers)
    while next_compute < n or alu is not None:
        if dma is not None and dma[1] == now:
            ready.add(dma[0])
            dma = None
        if alu is not None and alu[1] == now:
            slots[alu[0] % 2] = None
            alu = None
        if alu is None and next_compute in ready:
            ready.remove(next_compute)
            alu = (next_compute, now + computes[next_compute])
            next_compute += 1
        if dma is None and next_copy < n and slots[next_copy % 2] is None:
            slots[next_copy % 2] = next_copy
            dma = (next_copy, now + transfers[next_copy])
            next_copy += 1
        if next_compute == n and alu is None:
            return now
        now += 1
    return now


def check_pipeline():
    cases = 0
    for n in range(4):
        for durations in product(range(1, 4), repeat=2 * n):
            transfers, computes = durations[:n], durations[n:]
            require(recurrence(transfers, computes)
                    == tick_reference(transfers, computes),
                    'Recurrence disagrees with resource simulator')
            cases += 1
    transfers, computes = (4, 4, 4), (6, 6, 6)
    require(recurrence(transfers, computes) == 22, 'Worked pipeline example')
    rejected(lambda: require(max(sum(transfers), sum(computes))
                             == tick_reference(transfers, computes),
                             'Resource lower bound mistaken for completion time'))
    print('Two-slot pipeline: %d integer timing cases; 22 versus serial 30; lower-bound-as-latency fault detected' % cases)


def main():
    check_experts()
    check_slots()
    check_unions()
    check_cache_events()
    check_dense()
    check_pipeline()
    print('Scope: deterministic CPU arithmetic and abstract slot/event models only; logical payload bytes, no real router, SwiGLU, quantization, CUDA visibility, DMA, HBM transactions or performance measurement')


if __name__ == '__main__':
    main()
