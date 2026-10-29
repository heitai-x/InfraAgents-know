"""K-FW-003: bounded CPU dependency/lifetime model; no CUDA or measurements."""
import itertools
import json


class Failure(Exception):
    pass


def require(condition, message):
    if not condition:
        raise Failure(message)


def graph(ops, mode='ok', reference=False):
    nodes = ['A', 'P']
    for i in range(len(ops)):
        nodes.extend(('s%d' % i, 'e%d' % i))
    nodes.extend(('F', 'N'))
    edges = set()
    if mode != 'missing_allocation' or reference:
        edges.add(('A', 'P'))
    edges.add(('F', 'N'))
    writer = 'P'
    readers = {}
    tails = {}
    for i, (stream, kind) in enumerate(ops):
        start, end = 's%d' % i, 'e%d' % i
        edges.add((start, end))
        if stream in tails:
            edges.add((tails[stream], start))
        tails[stream] = end
        if reference:
            edges.add(('P', start))
            for j in range(i):
                if kind == 'W' or ops[j][1] == 'W':
                    edges.add(('e%d' % j, start))
        else:
            if mode != 'missing_ready':
                edges.add((writer, start))
            ticket = start if mode == 'early_event' else end
            if kind == 'W':
                if mode != 'ignore_readers':
                    for reader in readers.values():
                        edges.add((reader, start))
                readers.clear()
                writer = ticket
            else:
                key = 0 if mode == 'last_reader' else stream
                readers[key] = ticket
    if reference:
        edges.add(('P', 'F'))
        for i in range(len(ops)):
            edges.add(('e%d' % i, 'F'))
    else:
        edges.add((writer, 'F'))
        if mode != 'free_writer_only':
            for reader in readers.values():
                edges.add((reader, 'F'))
    return nodes, edges


def closure(g):
    nodes, edges = g
    index = {node: i for i, node in enumerate(nodes)}
    rows = [0] * len(nodes)
    for left, right in edges:
        rows[index[left]] |= 1 << index[right]
    for k in range(len(nodes)):
        bit = 1 << k
        for i in range(len(nodes)):
            if rows[i] & bit:
                rows[i] |= rows[k]
    require(all(not (row & (1 << i)) for i, row in enumerate(rows)), 'cycle')
    return tuple(rows)


def expected_versions(ops):
    # Serial specification, independent of stream/frontier bookkeeping.
    version = 0
    result = []
    for unused_stream, kind in ops:
        if kind == 'W':
            version += 1
        result.append(version)
    return result


def validate_prefix(ops, trace):
    allocated = False
    initialized = False
    freed = False
    tag = None
    active = {}
    ended = set()
    versions = expected_versions(ops)
    for node in trace:
        if node == 'A':
            require(not allocated and not freed, 'duplicate allocation')
            allocated = True
        elif node == 'P':
            require(allocated and not freed, 'initial producer before allocation')
            initialized = True
            tag = 0
        elif node == 'F':
            require(allocated and initialized and not freed, 'invalid free')
            require(not active, 'free during active access')
            require(len(ended) == len(ops), 'free before all accepted accesses ended')
            allocated = False
            freed = True
        elif node == 'N':
            require(freed and not active, 'reuse before old lifetime ended')
            allocated = True
            tag = 'new_epoch'
        else:
            i = int(node[1:])
            kind = ops[i][1]
            if node[0] == 's':
                require(allocated and initialized and not freed, 'access outside old lifetime')
                require(i not in active and i not in ended, 'duplicate access start')
                if kind == 'W':
                    require(not active, 'writer overlaps a prior access')
                else:
                    require(all(ops[j][1] == 'R' for j in active), 'reader overlaps writer')
                    require(tag == versions[i], 'reader sees wrong scheduled version')
                active[i] = kind
            else:
                require(i in active, 'access ended without start')
                if kind == 'R':
                    require(tag == versions[i], 'read content changed before last use')
                else:
                    tag = versions[i]
                del active[i]
                ended.add(i)


def enumerate_schedules(ops, g):
    nodes, edges = g
    predecessors = {node: set() for node in nodes}
    for left, right in edges:
        predecessors[right].add(left)
    count = 0

    def visit(trace, done):
        nonlocal count
        try:
            validate_prefix(ops, trace)
        except Failure as exc:
            raise Failure(str(exc) + '; trace=' + repr(trace)) from None
        if len(done) == len(nodes):
            count += 1
            return
        choices = [node for node in nodes
                   if node not in done and predecessors[node] <= done]
        require(bool(choices), 'dependency deadlock')
        for node in choices:
            visit(trace + (node,), done | {node})

    visit((), set())
    return count


class Slot:
    def __init__(self):
        self.epoch = 0
        self.free = True
        self.closing = False
        self.owners = set()
        self.pending = set()
        self.done = set()

    def allocate(self, owners):
        require(self.free, 'reuse while slot protected')
        require(bool(owners), 'allocation needs an owner')
        self.epoch += 1
        self.free = False
        self.closing = False
        self.owners = set(owners)
        self.pending = set()
        self.done = set()
        return self.epoch

    def submit(self, name):
        require(not self.free and not self.closing and bool(self.owners), 'late access')
        token = (self.epoch, name)
        require(token not in self.pending and token not in self.done, 'duplicate use identity')
        self.pending.add(token)
        return token

    def collect(self):
        if self.closing and not self.owners and not self.pending:
            self.free = True

    def drop(self, epoch, owner):
        if epoch != self.epoch or self.free or owner not in self.owners:
            return False
        self.owners.remove(owner)
        if not self.owners:
            self.closing = True
        self.collect()
        return True

    def complete(self, token):
        if self.free or token[0] != self.epoch or token not in self.pending:
            return False
        self.pending.remove(token)
        self.done.add(token)
        self.collect()
        return True

    def snapshot(self):
        return (self.epoch, self.free, self.closing, frozenset(self.owners),
                frozenset(self.pending), frozenset(self.done))


class BrokenEpoch(Slot):
    def complete(self, token):
        return super().complete((self.epoch, token[1]))


def check_stale(slot_type):
    slot = slot_type()
    old_epoch = slot.allocate({'request'})
    old = slot.submit('B')
    slot.drop(old_epoch, 'request')
    require(slot.complete(old) and slot.free, 'old lifetime did not retire')
    slot.allocate({'request'})
    slot.submit('B')
    before = slot.snapshot()
    accepted = slot.complete(old)
    require(not accepted and slot.snapshot() == before, 'old completion changed new lifetime')
    require(not slot.drop(old_epoch, 'request'), 'old owner release accepted')
    require(slot.snapshot() == before, 'old owner release changed new lifetime')


def check_ledger():
    cases = 0
    for n in range(4):
        names = tuple('use%d' % i for i in range(n))
        for order in itertools.permutations(names):
            for drop_early in (False, True):
                slot = Slot()
                epoch = slot.allocate({'request', 'snapshot'})
                tokens = {name: slot.submit(name) for name in names}
                slot.drop(epoch, 'request')
                before = slot.snapshot()
                require(not slot.drop(epoch, 'request'), 'duplicate owner release accepted')
                require(slot.snapshot() == before, 'duplicate release changed state')
                if drop_early:
                    slot.drop(epoch, 'snapshot')
                completed = set()
                require(slot.free == (drop_early and n == 0), 'empty-use boundary')
                if drop_early and n:
                    try:
                        slot.submit('late')
                    except Failure:
                        pass
                    else:
                        raise Failure('closing slot admitted new access')
                for name in order:
                    require(slot.complete(tokens[name]), 'first completion rejected')
                    completed.add(name)
                    before = slot.snapshot()
                    require(not slot.complete(tokens[name]), 'duplicate completion accepted')
                    require(slot.snapshot() == before, 'duplicate completion changed state')
                    # Oracle derives remaining responsibility from the immutable grant list.
                    remaining = set(names) - completed
                    require(slot.free == (drop_early and not remaining), 'responsibility lost')
                if not drop_early:
                    require(not slot.free, 'future owner ignored')
                    slot.drop(epoch, 'snapshot')
                require(slot.free and not slot.pending and not slot.owners, 'not drained')
                cases += 1
    check_stale(Slot)
    return cases


def peak_bytes(intervals):
    events = []
    for start, end, size in intervals:
        require(start < end and size >= 0, 'invalid occupancy interval')
        events.extend(((start, size), (end, -size)))
    current = 0
    peak = 0
    # Half-open intervals: ends precede starts at an equal timestamp.
    for unused_time, delta in sorted(events):
        current += delta
        require(current >= 0, 'capacity accounting underflow')
        peak = max(peak, current)
    require(current == 0, 'capacity accounting did not drain')
    return peak


def check_capacity():
    for life in range(1, 10):
        for period in range(1, 6):
            intervals = [(i * period, i * period + life, 16) for i in range(20)]
            slots = (life + period - 1) // period
            require(peak_bytes(intervals) == slots * 16, 'slot lower bound mismatch')
    x = [(4 * i, 4 * i + 6, 64) for i in range(8)]
    workspace = [(4 * i + 2, 4 * i + 5, 16) for i in range(8)]
    host = [(4 * i + 2, 4 * i + 10, 64) for i in range(8)]
    gpu_peak = peak_bytes(x + workspace)
    host_peak = peak_bytes(host)
    require((gpu_peak, host_peak) == (144, 128), 'composite capacity example')
    return {'gpu_buffer_peak_MiB': gpu_peak, 'pinned_buffer_peak_MiB': host_peak}


def main():
    alphabet = tuple(itertools.product(range(3), ('R', 'W')))
    graph_cases = 0
    linearizations = 0
    for n in range(5):
        for ops in itertools.product(alphabet, repeat=n):
            candidate = graph(ops)
            reference = graph(ops, reference=True)
            require(closure(candidate) == closure(reference), 'frontier/reference mismatch')
            if n <= 3:
                linearizations += enumerate_schedules(ops, candidate)
            graph_cases += 1
    negatives = {}
    faulty_graphs = (
        ('missing_allocation', ()),
        ('missing_ready', ((1, 'R'),)),
        ('ignore_readers', ((1, 'R'), (2, 'W'))),
        ('last_reader', ((1, 'R'), (2, 'R'))),
        ('free_writer_only', ((1, 'R'),)),
        ('early_event', ((1, 'W'), (2, 'R'))),
    )
    for mode, ops in faulty_graphs:
        faulty = graph(ops, mode=mode)
        require(closure(faulty) != closure(graph(ops, reference=True)),
                'negative graph unexpectedly equals reference: ' + mode)
        try:
            enumerate_schedules(ops, faulty)
        except Failure as exc:
            negatives[mode] = str(exc)
        else:
            raise Failure('unsafe negative escaped execution oracle: ' + mode)
    ledger_cases = check_ledger()
    try:
        check_stale(BrokenEpoch)
    except Failure as exc:
        negatives['accept_old_epoch'] = str(exc)
    else:
        raise Failure('old-epoch negative escaped')
    observations = {
        'scope': 'Deterministic bounded CPU event and ownership model only; no CUDA, GPU timing, allocator implementation or network.',
        'graph_cases': graph_cases,
        'complete_linearizations_n_at_most_3': linearizations,
        'ownership_cases': ledger_cases,
        'capacity': check_capacity(),
        'detected_negative_controls': negatives,
        'unmeasured': ['GPU visibility and API mapping', 'real DMA and graph replay',
                       'driver pool reuse', 'hardware overlap and performance',
                       'device failure recovery', 'Agent effectiveness'],
    }
    print(json.dumps(observations, ensure_ascii=True, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
