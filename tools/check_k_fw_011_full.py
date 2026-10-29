import copy
import heapq
import itertools
from collections import defaultdict


class ContractError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def product(shape):
    result = 1
    for size in shape:
        result *= size
    return result


def validate(records, files, required):
    names = {}
    spans = defaultdict(list)
    for record in records:
        name = record['name']
        if name in names:
            raise ContractError('duplicate name')
        names[name] = record
        shape = tuple(record['shape'])
        if any(type(size) is not int or size < 0 for size in shape):
            raise ContractError('invalid dimension')
        order = tuple(record['order'])
        if any(type(axis) is not int for axis in order):
            raise ContractError('invalid axis')
        if sorted(order) != list(range(len(shape))):
            raise ContractError('invalid permutation')
        if 'alias_of' in record:
            continue
        file_id = record['file']
        if file_id not in files:
            raise ContractError('missing fixed-version file')
        offset = record['offset']
        if type(offset) is not int or offset < 0 or offset % 8:
            raise ContractError('invalid offset')
        size = product(shape) * 4
        if offset + size > len(files[file_id]):
            raise ContractError('range outside corresponding file')
        if size:
            spans[file_id].append((offset, offset + size, name))
    if not set(required).issubset(names):
        raise ContractError('missing required tensor')
    for file_spans in spans.values():
        file_spans.sort()
        end = -1
        for start, stop, name in file_spans:
            if start < end:
                raise ContractError('undeclared overlap in one file')
            end = stop
    roots = {}
    visiting = set()

    def resolve(name):
        if name in roots:
            return roots[name]
        if name not in names:
            raise ContractError('missing alias root')
        if name in visiting:
            raise ContractError('alias cycle')
        visiting.add(name)
        record = names[name]
        if 'alias_of' in record:
            root_name = resolve(record['alias_of'])
            root = names[root_name]
            if (tuple(record['shape']), tuple(record['order'])) != (tuple(root['shape']), tuple(root['order'])):
                raise ContractError('incompatible alias representation')
        else:
            root_name = name
        visiting.remove(name)
        roots[name] = root_name
        return root_name

    for name in names:
        resolve(name)
    return names, roots


def make_file(values, offset=8):
    return bytes(offset) + b''.join(value.to_bytes(4, 'little', signed=True) for value in values)


def materialize(records, files, required, ignore_order=False):
    names, roots = validate(records, files, required)
    outputs = {}
    for name, record in names.items():
        if roots[name] != name:
            continue
        shape = tuple(record['shape'])
        target_shape = tuple(shape[axis] for axis in record['order'])
        source = files[record['file']]
        result = []
        if ignore_order:
            for index in range(product(shape)):
                address = record['offset'] + 4 * index
                result.append(int.from_bytes(source[address:address + 4], 'little', signed=True))
        else:
            for target in itertools.product(*(range(size) for size in target_shape)):
                coords = [0] * len(shape)
                for target_axis, source_axis in enumerate(record['order']):
                    coords[source_axis] = target[target_axis]
                index = 0
                for coord, size in zip(coords, shape):
                    index = index * size + coord
                address = record['offset'] + 4 * index
                result.append(int.from_bytes(source[address:address + 4], 'little', signed=True))
        outputs[name] = result
    for name in names:
        outputs[name] = outputs[roots[name]]
    return outputs


def source_order_reference(values, shape, order):
    # Enumerate source coordinates and sort by target coordinates.
    # This does not use the tested target-to-source linear address calculation.
    pairs = []
    coordinates = itertools.product(*(range(size) for size in shape))
    for value, coords in zip(values, coordinates):
        pairs.append((tuple(coords[axis] for axis in order), value))
    return [value for coords, value in sorted(pairs)]


def expect_contract_error(records, files, required):
    try:
        validate(records, files, required)
    except ContractError:
        return
    raise AssertionError('invalid manifest accepted')


def check_manifest_and_layout():
    f0, f1 = ('artifact-v1', 'shard-0'), ('artifact-v1', 'shard-1')
    files = {f0: make_file(list(range(6))), f1: make_file([100, 101, 102, 103])}
    records = [
        {'name': 'A', 'file': f0, 'offset': 8, 'shape': (2, 3), 'order': (1, 0)},
        {'name': 'B', 'alias_of': 'A', 'shape': (2, 3), 'order': (1, 0)},
        {'name': 'C', 'file': f1, 'offset': 8, 'shape': (4,), 'order': (0,)}
    ]
    original = copy.deepcopy(records)
    output = materialize(records, files, {'A', 'B', 'C'})
    require(records == original, 'validation modified input descriptors')
    require(output['A'] == [0, 3, 1, 4, 2, 5], 'literal transpose oracle')
    require(output['C'] == [100, 101, 102, 103], 'wrong shard selected')
    require(output['B'] is output['A'], 'alias must share target object in this model')
    names, roots = validate(records, files, {'A', 'B', 'C'})
    unique_bytes = sum(len(output[name]) * 4 for name in set(roots.values()))
    logical_bytes = sum(len(value) * 4 for value in output.values())
    require((unique_bytes, logical_bytes) == (40, 64), 'alias capacity model')
    bad = copy.deepcopy(records)
    bad[2]['file'] = f0
    expect_contract_error(bad, files, {'A', 'C'})
    bad = copy.deepcopy(records)
    bad[0]['offset'] = -8
    expect_contract_error(bad, files, {'A'})
    bad = copy.deepcopy(records)
    bad[0]['offset'] = 16
    expect_contract_error(bad, files, {'A'})
    bad = copy.deepcopy(records)
    bad[0]['file'] = ('artifact-v2', 'shard-0')
    expect_contract_error(bad, files, {'A'})
    bad = copy.deepcopy(records)
    bad[0]['order'] = (0, 0)
    expect_contract_error(bad, files, {'A'})
    bad = copy.deepcopy(records)
    bad[1]['alias_of'] = 'B'
    expect_contract_error(bad, files, {'A'})
    expect_contract_error(records + [copy.deepcopy(records[0])], files, {'A'})
    expect_contract_error(records, files, {'missing'})
    expect_contract_error([{'name': 'X', 'alias_of': 'Y', 'shape': (1,), 'order': (0,)}], {}, {'X'})
    tested = 0
    for shape in [(), (0,), (1,), (2, 3), (2, 1, 3)]:
        values = list(range(product(shape)))
        for order in itertools.permutations(range(len(shape))):
            record = {'name': 'T', 'file': f0, 'offset': 8, 'shape': shape, 'order': order}
            actual = materialize([record], {f0: make_file(values)}, {'T'})['T']
            expected = source_order_reference(values, shape, order)
            require(actual == expected, 'source-enumeration layout oracle')
            tested += 1
    buggy = materialize(records, files, {'A', 'C'}, ignore_order=True)
    require(buggy['A'] != output['A'], 'ignored transpose negative control escaped')
    print('CPU manifest/layout: cross-shard equal offsets accepted; same-file overlap rejected; alias identity bytes=', unique_bytes, logical_bytes, '; permutation cases=', tested)
    print('CPU layout negative control: actual linear copy rejected by transpose oracle')


def union_capacity(intervals):
    by_backing = defaultdict(list)
    for backing, start, end in intervals:
        require(0 <= start <= end, 'invalid physical range')
        by_backing[backing].append((start, end))
    total = 0
    for spans in by_backing.values():
        right = None
        for start, end in sorted(spans):
            if right is None or start > right:
                total += end - start
            else:
                total += max(0, end - right)
            right = end if right is None else max(right, end)
    return total


def exclusive_capacity(categories):
    # Independent pairwise ownership invariant across accounting categories.
    items = [(category, backing, start, end) for category, spans in categories.items() for backing, start, end in spans]
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if a[1] == b[1] and max(a[2], b[2]) < min(a[3], b[3]):
                raise ContractError('capacity counted more than once')
    return sum(end - start for category, backing, start, end in items)


def flow_end(count, times):
    finishes = [0] * len(times)
    for chunk in range(count):
        upstream = 0
        for stage, duration in enumerate(times):
            finishes[stage] = max(finishes[stage], upstream) + duration
            upstream = finishes[stage]
    return finishes[-1] if count else 0


def tick_end(count, times):
    # Independent integer-tick stage servers, with unlimited intermediate queues.
    waiting = [[] for duration in times]
    waiting[0] = list(range(count))
    running = [None] * len(times)
    finished = 0
    clock = 0
    while finished < count:
        for stage in range(len(times)):
            if running[stage] is None and waiting[stage]:
                running[stage] = [waiting[stage].pop(0), times[stage]]
        completions = []
        for stage, task in enumerate(running):
            if task is not None:
                task[1] -= 1
                if task[1] == 0:
                    completions.append((stage, task[0]))
        clock += 1
        for stage, chunk in completions:
            running[stage] = None
            if stage + 1 == len(times):
                finished += 1
            else:
                waiting[stage + 1].append(chunk)
        require(clock <= 1000, 'tick model failed to terminate')
    return clock


def check_costs():
    # Range lengths here are abstract MiB units, not allocated memory measurements.
    pool = [('host-pool', 0, 32)]
    active = [('host-pool', 0, 16), ('host-pool', 16, 32)]
    require(union_capacity(pool + active) == 32, 'unique backing oracle')
    require(exclusive_capacity({'pinned_pool': pool}) == 32, 'pool counted incorrectly')
    try:
        exclusive_capacity({'pinned_pool': pool, 'active_blocks': active})
    except ContractError:
        pass
    else:
        raise AssertionError('double-counted pool negative control escaped')
    graph = [('graph-pool', 0, 64)]
    graph_active = [('graph-pool', 0, 48)]
    require(union_capacity(graph + graph_active) == 64, 'graph backing counted twice')
    cases = 0
    for count in range(5):
        for times in itertools.product(range(1, 4), repeat=3):
            actual = flow_end(count, times)
            expected = tick_end(count, times)
            formula = sum(times) + (count - 1) * max(times) if count else 0
            require(actual == expected == formula, 'pipeline model disagreement')
            cases += 1
    require(flow_end(2, (1, 1, 1)) == 4, 'finite pipeline drain')
    serial_work = 2 * sum((1, 1, 1))
    require(serial_work == 6 and serial_work != tick_end(2, (1, 1, 1)), 'serial-time negative control escaped')
    ready = max(flow_end(2, (1, 1, 1)), 1 + flow_end(2, (1, 1, 1))) + 0.5
    require(ready == 5.5, 'independent rank critical path')
    print('CPU capacity model: two 16 MiB slots in 32 MiB pool count as 32; graph pool counts as 64; overlapping accounting rejected')
    print('CPU integer-time pipeline: cases=', cases, '; two chunks finish=4, serial work=6; independent rank example=5.5')


CONSUMERS = frozenset(('dma', 'audit'))
GATES = frozenset(('rank-0', 'rank-1', 'reference', 'warmup', 'budget'))


def token(event):
    return (event['g'], event['cid'], event['lease'], event['uid'])


def ticket(event):
    return (event['g'], event['cid'], event['lease'])


class Loader:
    # A narrow executable model: synchronous preparation, two asynchronous host readers.
    def __init__(self, count, slots=2, early_reuse=False, early_publish=False,
                 stop_at=None, stop_kind='failure', missing_completion=False,
                 unsafe_cleanup=False, gates=GATES, dependencies=None):
        self.count = count
        self.g = 7
        self.now = 0
        self.sequence = 0
        self.pending = set(range(count))
        self.states = {cid: 'PENDING' for cid in range(count)}
        self.owners = {slot: None for slot in range(slots)}
        self.leases = {slot: 0 for slot in range(slots)}
        self.free = set(range(slots))
        self.memory = {}
        self.live = {}
        self.events = []
        self.trace = []
        self.aborted = None
        self.published = False
        self.early_reuse = early_reuse
        self.early_publish = early_publish
        self.stop_at = stop_at
        self.stop_kind = stop_kind
        self.missing_completion = missing_completion
        self.unsafe_cleanup = unsafe_cleanup
        self.gates = frozenset(gates)
        self.dependencies = dependencies or {cid: set() for cid in range(count)}
        if stop_at is not None and stop_at > 0:
            self.enqueue(stop_at, {'kind': 'signal', 'reason': stop_kind})

    def emit(self, kind, **fields):
        self.trace.append(dict(kind=kind, time=self.now, **fields))

    def enqueue(self, when, event):
        self.sequence += 1
        heapq.heappush(self.events, (when, self.sequence, event))

    def submit(self, cid, slot):
        require(cid in self.pending and slot in self.free, 'invalid scheduler selection')
        self.pending.remove(cid)
        self.states[cid] = 'INFLIGHT'
        self.free.remove(slot)
        self.leases[slot] += 1
        lease = self.leases[slot]
        fields = dict(g=self.g, cid=cid, lease=lease, slot=slot)
        self.owners[slot] = (self.g, cid, lease)
        self.emit('acquire', **fields)
        self.memory[slot] = 10 + cid
        self.emit('write', value=10 + cid, **fields)
        for uid, delay in [('dma', 3), ('audit', 5)]:
            event = dict(kind='completion', uid=uid, expected=10 + cid, **fields)
            self.live[token(event)] = event
            self.emit('submit', uid=uid, expected=10 + cid, **fields)
            if not (self.missing_completion and cid == 0 and uid == 'audit'):
                self.enqueue(self.now + delay + cid % 2, event)
        if self.early_publish and cid == 0:
            self.published = True
            self.emit('publish')
        if self.early_reuse and cid == 0:
            # Actual faulty allocator action. A later submit acquires this same slot
            # and overwrites memory while the old physical accesses remain live.
            self.owners[slot] = None
            self.free.add(slot)
            self.emit('release', **fields)

    def complete(self, event):
        key = token(event)
        if key not in self.live:
            self.emit('ignored', **{name: event[name] for name in ('g', 'cid', 'lease', 'uid', 'slot')})
            return
        del self.live[key]
        fields = {name: event[name] for name in ('g', 'cid', 'lease', 'uid', 'slot')}
        self.emit('physical_done', observed=self.memory[event['slot']], **fields)
        cid = event['cid']
        if not any(key[1] == cid for key in self.live):
            if self.aborted:
                self.states[cid] = 'CANCELLED' if self.aborted == 'cancel' else 'FAILED'
            else:
                self.states[cid] = 'DONE'
                self.emit('chunk_done', cid=cid)
            if self.owners[event['slot']] == ticket(event):
                slot = event['slot']
                self.owners[slot] = None
                self.free.add(slot)
                self.emit('release', g=event['g'], cid=cid, lease=event['lease'], slot=slot)

    def abort(self, reason):
        if self.aborted:
            return
        self.aborted = reason
        for cid in sorted(self.pending):
            self.states[cid] = 'SKIPPED'
        self.pending = set()
        for cid, state in list(self.states.items()):
            if state == 'INFLIGHT':
                self.states[cid] = 'FAILED_DRAINING'
        self.emit('abort', reason=reason)

    def snapshot(self):
        return copy.deepcopy((self.live, self.states, self.owners, self.free))

    def run(self):
        if self.stop_at == 0:
            self.abort(self.stop_kind)
        while self.pending or self.live:
            if not self.aborted:
                while self.free:
                    eligible = [cid for cid in sorted(self.pending)
                                if all(self.states.get(dep) == 'DONE'
                                       for dep in self.dependencies.get(cid, set()))]
                    if not eligible:
                        break
                    self.submit(eligible[0], min(self.free))
            if self.aborted and self.unsafe_cleanup:
                for slot, owner in list(self.owners.items()):
                    if owner is not None:
                        g, cid, lease = owner
                        self.owners[slot] = None
                        self.free.add(slot)
                        self.emit('release', g=g, cid=cid, lease=lease, slot=slot)
                self.emit('cleanup')
                return 'FAILED'
            if not self.pending and not self.live:
                break
            if self.events:
                self.now, sequence, event = heapq.heappop(self.events)
                if event['kind'] == 'signal':
                    self.abort(event['reason'])
                else:
                    self.complete(event)
            elif self.live:
                self.emit('quarantine')
                return 'QUARANTINED'
            else:
                self.abort('no_progress')
        if self.aborted:
            self.emit('cleanup')
            return 'CANCELLED' if self.aborted == 'cancel' else 'FAILED'
        if self.gates != GATES:
            self.abort('validation')
            self.emit('cleanup')
            return 'FAILED'
        self.emit('gates', values=sorted(self.gates))
        if not self.published:
            self.published = True
            self.emit('publish')
        return 'READY'


class Observer:
    # Reconstruct physical responsibility from the public event trace.
    # Never consult Loader.free, Loader.live, or its fault switches.
    def __init__(self, count):
        self.required = set(range(count))
        self.owners = {}
        self.uses = {}
        self.submitted = defaultdict(set)
        self.finished = defaultdict(set)
        self.done = set()
        self.aborted = False
        self.gates = set()
        self.errors = []
        self.max_slots = 0
        self.live_at_abort = 0
        self.publish_count = 0

    def fail(self, code):
        self.errors.append(code)

    def consume(self, events):
        for event in events:
            kind = event['kind']
            if kind == 'acquire':
                slot = event['slot']
                if slot in self.owners or any(use['slot'] == slot for use in self.uses.values()):
                    self.fail('acquire_overlaps_unfinished_use')
                self.owners[slot] = ticket(event)
            elif kind == 'write':
                if any(use['slot'] == event['slot'] for use in self.uses.values()):
                    self.fail('overwrite_during_unfinished_use')
            elif kind == 'submit':
                key = token(event)
                if self.owners.get(event['slot']) != ticket(event):
                    self.fail('submit_without_lease')
                if key in self.uses or event['uid'] in self.submitted[event['cid']]:
                    self.fail('duplicate_submission')
                self.uses[key] = event
                self.submitted[event['cid']].add(event['uid'])
            elif kind == 'physical_done':
                key = token(event)
                use = self.uses.pop(key, None)
                if use is None:
                    self.fail('completion_without_matching_use')
                else:
                    if event['observed'] != use['expected']:
                        self.fail('consumer_read_overwritten_payload')
                    self.finished[event['cid']].add(event['uid'])
            elif kind == 'release':
                if any(ticket(use) == ticket(event) and use['slot'] == event['slot']
                       for use in self.uses.values()):
                    self.fail('release_before_last_use')
                if self.owners.get(event['slot']) != ticket(event):
                    self.fail('release_wrong_lease')
                else:
                    del self.owners[event['slot']]
            elif kind == 'chunk_done':
                cid = event['cid']
                if (cid in self.done or self.aborted or self.submitted[cid] != CONSUMERS
                        or self.finished[cid] != CONSUMERS
                        or any(use['cid'] == cid for use in self.uses.values())):
                    self.fail('chunk_done_without_completion')
                self.done.add(cid)
            elif kind == 'abort':
                self.aborted = True
                self.live_at_abort = len(self.uses)
            elif kind == 'gates':
                self.gates = set(event['values'])
            elif kind == 'publish':
                self.publish_count += 1
                if self.aborted or self.done != self.required or self.uses:
                    self.fail('publish_before_complete_or_after_abort')
                if self.gates != GATES:
                    self.fail('publish_without_gates')
            elif kind == 'cleanup':
                if self.uses or self.owners:
                    self.fail('cleanup_with_unfinished_responsibility')
            elif kind == 'quarantine':
                if not self.uses:
                    self.fail('quarantine_without_outstanding_use')
                for use in self.uses.values():
                    if self.owners.get(use['slot']) != ticket(use):
                        self.fail('quarantine_lost_storage_protection')
            elif kind != 'ignored':
                self.fail('unknown_event')
            self.max_slots = max(self.max_slots, len({use['slot'] for use in self.uses.values()}))
        return self


def observe(loader):
    return Observer(loader.count).consume(loader.trace)


def check_lifetimes():
    good = Loader(3)
    require(good.run() == 'READY', 'normal load did not terminate')
    good_observer = observe(good)
    require(not good_observer.errors, str(good_observer.errors))
    require(good_observer.max_slots == 2, 'distinct-slot concurrency was not exercised')
    require(all(state == 'DONE' for state in good.states.values()), 'chunks did not leave pending/inflight')
    acquired = [event for event in good.trace if event['kind'] == 'acquire']
    require(len(acquired) == 3 and len({event['slot'] for event in acquired}) == 2, 'no actual slot reuse')
    require(not good.live and not good_observer.uses and not good_observer.owners, 'last batch not drained')
    require(good_observer.publish_count == 1, 'publication not unique')
    for count in (0, 1):
        edge = Loader(count, slots=1)
        require(edge.run() == 'READY' and not observe(edge).errors, 'empty/single-work boundary')
    for reason in ('failure', 'cancel'):
        failed = Loader(3, stop_at=2, stop_kind=reason)
        expected_status = 'CANCELLED' if reason == 'cancel' else 'FAILED'
        require(failed.run() == expected_status, 'failure terminal status')
        checked = observe(failed)
        require(not checked.errors and checked.live_at_abort == 4, 'failure drain invariant')
        require(not failed.published and not failed.live, 'failure published or leaked responsibility')
        require(failed.states[2] == 'SKIPPED', 'failure submitted pending work')
        require(failed.trace[-1]['kind'] == 'cleanup', 'failure did not return after cleanup')
    cancelled = Loader(2, stop_at=0, stop_kind='cancel')
    require(cancelled.run() == 'CANCELLED' and not observe(cancelled).errors, 'cancel-before-submit')
    require(not any(event['kind'] == 'submit' for event in cancelled.trace), 'cancel still submitted work')
    blocked = Loader(2, dependencies={0: {1}, 1: {0}})
    require(blocked.run() == 'FAILED' and blocked.aborted == 'no_progress', 'no-progress was not distinguished from waiting')
    require(not observe(blocked).errors, 'no-progress cleanup failed')
    missing_rank = Loader(1, gates=GATES - {'rank-1'})
    require(missing_rank.run() == 'FAILED' and not missing_rank.published, 'missing rank published')
    require(not observe(missing_rank).errors, 'validation failure cleanup')
    quarantine = Loader(2, stop_at=2, missing_completion=True)
    require(quarantine.run() == 'QUARANTINED', 'missing last-use proof was treated as completion')
    quarantine_observer = observe(quarantine)
    require(not quarantine_observer.errors and len(quarantine_observer.uses) == 1, 'quarantine responsibility lost')
    require(quarantine.live and not any(event['kind'] == 'cleanup' for event in quarantine.trace), 'unproved access reclaimed')

    # Probe stale and duplicate events before and after actual reuse.
    stale = Loader(2, slots=1)
    stale.submit(0, 0)
    old_event = copy.deepcopy(next(iter(stale.live.values())))
    wrong_generation = dict(old_event, g=stale.g - 1)
    before = stale.snapshot()
    stale.complete(wrong_generation)
    require(stale.snapshot() == before, 'stale generation changed actual responsibility')
    while stale.live:
        stale.now, sequence, event = heapq.heappop(stale.events)
        stale.complete(event)
    stale.submit(1, 0)
    before = stale.snapshot()
    stale.complete(old_event)
    require(stale.snapshot() == before, 'old lease completion cleared new use')
    require(stale.run() == 'READY' and not observe(stale).errors, 'stale-event continuation')

    reuse = Loader(3, early_reuse=True)
    require(reuse.run() == 'READY', 'faulty producer did not expose its candidate')
    reuse_errors = observe(reuse).errors
    for expected in ('release_before_last_use', 'acquire_overlaps_unfinished_use',
                     'overwrite_during_unfinished_use', 'consumer_read_overwritten_payload'):
        require(expected in reuse_errors, 'actual reuse negative control escaped: ' + expected)
    early = Loader(2, early_publish=True)
    early.run()
    early_errors = observe(early).errors
    require('publish_before_complete_or_after_abort' in early_errors, 'early publication escaped')
    unsafe = Loader(3, stop_at=2, unsafe_cleanup=True)
    unsafe.run()
    unsafe_errors = observe(unsafe).errors
    require(unsafe.live and 'cleanup_with_unfinished_responsibility' in unsafe_errors, 'unsafe failure cleanup escaped')
    print('CPU lifetime positives: distinct-slot concurrency, 3 chunks/2 slots, final drain, failure/cancel drain, stale-event isolation, empty/single work')
    print('CPU no-progress: rejected; missing rank: rejected; missing completion: quarantined with one protected use')
    print('CPU actual same-slot reuse negative control:', sorted(set(reuse_errors)))
    print('CPU early publish negative control:', sorted(set(early_errors)))
    print('CPU premature failure cleanup negative control:', sorted(set(unsafe_errors)))


def main():
    check_manifest_and_layout()
    check_costs()
    check_lifetimes()
    print('PASS: deterministic offline CPU models only; no checkpoint, GPU, I/O, pinning, graph or performance measurement')


if __name__ == '__main__':
    main()
