from collections import Counter
from fractions import Fraction as Q
from itertools import product

MIB = 1024 ** 2
GIB = 1024 ** 3


def require(condition, message='invariant failed'):
    if not condition:
        raise AssertionError(message)


def must_fail(action, exception=AssertionError):
    try:
        action()
    except exception:
        return
    raise AssertionError('negative control was not detected')


def must_fail_with(action, message):
    try:
        action()
    except AssertionError as error:
        require(str(error) == message, 'unexpected negative-control failure: ' + str(error))
        return
    raise AssertionError('negative control was not detected: ' + message)


def pages(start, end, granule):
    if start < 0 or end < start or granule <= 0:
        raise ValueError('invalid range or granule')
    if start == end:
        return set()
    return set(range(start // granule, (end - 1) // granule + 1))


def byte_pages(start, end, granule):
    return {byte // granule for byte in range(start, end)}


def test_pages():
    cases = 0
    faulty_witnesses = 0
    for granule in (1, 2, 3, 8):
        for start in range(25):
            for end in range(start, 25):
                reference = byte_pages(start, end, granule)
                require(pages(start, end, granule) == reference)
                bad = set() if start == end else set(range(start // granule, end // granule + 1))
                faulty_witnesses += bad != reference
                cases += 1
    for args in ((-1, 1, 4), (2, 1, 4), (0, 1, 0)):
        must_fail(lambda args=args: pages(*args), ValueError)
    require(faulty_witnesses > 0)
    print('pages: %d ranges; inclusive-end fault detected in %d cases' % (cases, faulty_witnesses))


def copy_byte_reference(size, reads):
    local = []
    for start, end in reads:
        local.extend(range(start, end))
    transferred = list(range(size)) if local else []
    return len(transferred), len(local)


def mapped_byte_reference(reads):
    remote = 0
    for start, end in reads:
        for unused_byte in range(start, end):
            remote += 1
    return remote


def threshold(repeats, link_bw, local_bw, alpha=Q(1)):
    denominator = repeats * (alpha - Q(link_bw, local_bw))
    if repeats <= 0 or denominator <= 0:
        return None
    return 1 / denominator


def formula_cost(size, repeats, fraction, link_bw, local_bw, alpha=Q(1)):
    useful = repeats * fraction * size
    if useful == 0:
        return Q(0), Q(0)
    return Q(size, link_bw) + useful / local_bw, alpha * useful / link_bw


def test_costs():
    link_bw, local_bw = 16 * GIB, 512 * GIB
    cases = 0
    for size in (8, 31, 64):
        for repeats in (0, 1, 4):
            for accessed in (0, 1, size // 2, size):
                reads = [(0, accessed)] * repeats
                copied, local = copy_byte_reference(size, reads)
                remote = mapped_byte_reference(reads)
                expected = (Q(copied, link_bw) + Q(local, local_bw), Q(remote, link_bw))
                actual = formula_cost(size, repeats, Q(accessed, size), link_bw, local_bw)
                require(actual == expected, 'same-trajectory byte ledger')
                require(local == remote)
                cases += 1
    star = threshold(4, link_bw, local_bw)
    require(star == Q(8, 31))
    for fraction, copy_wins in ((star - Q(1, 100), False), (star + Q(1, 100), True)):
        copied, mapped = formula_cost(64 * MIB, 4, fraction, link_bw, local_bw)
        require((copied < mapped) == copy_wins)
    equal_copy, equal_map = formula_cost(64 * MIB, 4, star, link_bw, local_bw)
    require(equal_copy == equal_map)
    require(threshold(0, link_bw, local_bw) is None)
    require(threshold(4, link_bw, local_bw, Q(1, 64)) is None)
    require(threshold(1, link_bw, local_bw) > 1)
    size = 10000
    reads = [(0, 2700)] * 4
    copied_bytes, local_bytes = copy_byte_reference(size, reads)
    remote_bytes = mapped_byte_reference(reads)
    require((copied_bytes, local_bytes, remote_bytes) == (10000, 10800, 10800))
    reference_copy = Q(copied_bytes, link_bw) + Q(local_bytes, local_bw)
    reference_map = Q(remote_bytes, link_bw)
    old_copy = Q(size, link_bw) + Q(4 * size, local_bw)
    require(reference_copy < reference_map < old_copy, 'old workload mismatch must reverse selection')
    require(Q(1, 4) + Q(link_bw, local_bw) != star, 'old threshold detected')
    scale = Q(64 * MIB, size)
    print('costs: %d enumerated workloads; x*=8/31; old formula/threshold detected' % cases)
    print('costs: scaled 64 MiB, R=4, x=.27: copy=%.10f ms, mapped=%.10f ms' %
          (float(reference_copy * scale * 1000), float(reference_map * scale * 1000)))


def migration_reference(events, page_sizes):
    residence = {page: 'H' for page in page_sizes}
    moved = 0
    for page, source, destination in events:
        require(source == residence[page] and source != destination, 'invalid page event')
        moved += page_sizes[page]
        residence[page] = destination
    return moved


def migration_by_counts(events, page_sizes):
    counts = Counter(page for page, source, destination in events)
    return sum(count * page_sizes[page] for page, count in counts.items())


def test_migration_and_mapping():
    granule = 64 * 1024
    sizes = {page: granule for page in range(16)}
    events = []
    for source, destination in (('H', 'D'), ('D', 'H'), ('H', 'D')):
        events.extend((page, source, destination) for page in range(16))
    moved = migration_reference(events, sizes)
    require(moved == migration_by_counts(events, sizes) == 3 * MIB)
    require(len(events) == 48)
    bad_double_count = len(events) * len(sizes) * granule
    require(bad_double_count == 48 * MIB and bad_double_count != moved)
    reads = [(0, MIB), (0, MIB)]
    require(sum(end - start for start, end in reads) == 2 * MIB)
    unequal = [(0, 'H', 'D'), (1, 'H', 'D'), (0, 'D', 'H'),
               (2, 'H', 'D'), (0, 'H', 'D'), (2, 'D', 'H')]
    unequal_sizes = {0: granule, 1: granule, 2: granule}
    require(migration_reference(unequal, unequal_sizes) == 6 * granule)
    require(migration_by_counts(unequal, unequal_sizes) == 6 * granule)
    require(Counter(page for page, source, destination in unequal) == {0: 3, 1: 1, 2: 2})
    require(migration_reference([(0, 'H', 'D')], {0: granule}) != 1,
            'one useful byte is not one migrated byte')
    accesses = [(page * granule, page * granule + 4) for page in range(16)]
    transaction = 32
    candidate_remote = sum(((end - 1) // transaction - start // transaction + 1) * transaction
                           for start, end in accesses)
    reference_remote = sum(len({byte // transaction for byte in range(start, end)}) * transaction
                           for start, end in accesses)
    require(candidate_remote == reference_remote == 512)
    page_coverage = len(set().union(*(pages(start, end, granule) for start, end in accesses))) * granule
    smaller_page_coverage = len(set().union(*(pages(start, end, granule // 2)
                                            for start, end in accesses))) * (granule // 2)
    require(page_coverage != reference_remote)
    require(smaller_page_coverage != page_coverage)
    print('migration: 48 page events=3 MiB; unequal counts=6 pages; double-count/payload faults detected')
    print('mapping: synthetic uncached 32 B granule gives 512 B for 64 useful B; page-transaction fault detected')


class CopyPath:
    def __init__(self, size, fault=''):
        self.size = size
        self.values = {'H': [None] * size, 'D': [None] * size}
        self.tags = {'H': [None] * size, 'D': [None] * size}
        self.current = [None] * size
        self.version = 0
        self.pending = {}
        self.next_ticket = 0
        self.first_local = False
        self.moves = []
        self.fault = fault

    def bounds(self, start, end):
        if not 0 <= start <= end <= self.size:
            raise ValueError('invalid range')

    def write(self, processor, start, values):
        end = start + len(values)
        self.bounds(start, end)
        if start == end:
            return
        if self.pending:
            raise ValueError('copy source/target protected by logical lease')
        self.version += 1
        other = 'D' if processor == 'H' else 'H'
        for index, value in enumerate(values, start):
            self.values[processor][index] = value
            self.tags[processor][index] = self.version
            self.current[index] = self.version
            self.tags[other][index] = None

    def submit_copy(self, target, start, end):
        self.bounds(start, end)
        require(start < end and not self.pending)
        source = 'D' if target == 'H' else 'H'
        items = []
        for index in range(start, end):
            version = self.current[index]
            require(version is not None and self.tags[source][index] == version)
            require(self.tags[target][index] != version)
            items.append((index, version, self.values[source][index]))
        self.next_ticket += 1
        ticket = self.next_ticket
        self.pending[ticket] = (source, target, tuple(items))
        if self.fault == 'early_publish':
            for index, version, value in items:
                self.tags[target][index] = version
                self.values[target][index] = value
        return ticket

    def complete_copy(self, ticket):
        source, target, items = self.pending.pop(ticket)
        for index, version, value in items:
            require(self.current[index] == version and self.tags[source][index] == version)
            self.tags[target][index] = version
            self.values[target][index] = value
        self.moves.append((source, target, len(items)))
        if target == 'D':
            self.first_local = True

    def ensure(self, processor, start, end):
        missing = [index for index in range(start, end)
                   if self.current[index] is None or self.tags[processor][index] != self.current[index]]
        if not missing:
            return
        if self.fault == 'skip_refill' and self.first_local:
            return
        if processor == 'D' and not self.first_local:
            require(all(version is not None and self.tags['H'][index] == version
                        for index, version in enumerate(self.current)))
            missing = list(range(self.size))
        offset = 0
        while offset < len(missing):
            first = missing[offset]
            end_offset = offset + 1
            while end_offset < len(missing) and missing[end_offset] == missing[end_offset - 1] + 1:
                end_offset += 1
            ticket = self.submit_copy(processor, first, missing[end_offset - 1] + 1)
            self.complete_copy(ticket)
            offset = end_offset

    def read(self, processor, start, end):
        self.bounds(start, end)
        self.ensure(processor, start, end)
        for index in range(start, end):
            require(self.current[index] is not None and
                    self.tags[processor][index] == self.current[index], 'missing version blocks consumer')
        return self.values[processor][start:end]


def replay_copy(size, operations, fault=''):
    candidate = CopyPath(size, fault)
    reference = [None] * size
    for mode, processor, start, payload in operations:
        if mode == 'W':
            for index, value in enumerate(payload, start):
                reference[index] = value
            candidate.write(processor, start, payload)
        else:
            expected = reference[start:payload]
            require(all(value is not None for value in expected))
            require(candidate.read(processor, start, payload) == expected,
                    'copy path differs from single authoritative byte array')
    return candidate


def unpublished(candidate, ticket):
    source, target, items = candidate.pending[ticket]
    require(all(candidate.tags[target][index] != version for index, version, value in items),
            'pending copy must not publish valid data')


def test_copy_progression():
    combination = [('W', 'H', 0, list(range(64))), ('R', 'D', 0, 1),
                   ('W', 'H', 0, [99]), ('R', 'D', 0, 1)]
    candidate = replay_copy(64, combination)
    require(candidate.moves == [('H', 'D', 64), ('H', 'D', 1)])
    require(candidate.version == 2)
    require(candidate.current[0] == 2 and candidate.current[1:] == [1] * 63)
    must_fail(lambda: replay_copy(64, combination, 'skip_refill'))
    reverse = [('W', 'H', 0, list(range(8))), ('R', 'D', 1, 5),
               ('W', 'D', 2, [90, 91, 92]), ('R', 'H', 0, 8),
               ('W', 'H', 0, [-1]), ('R', 'D', 0, 8)]
    candidate = replay_copy(8, reverse)
    require(candidate.moves == [('H', 'D', 8), ('D', 'H', 3), ('H', 'D', 1)])
    empty = CopyPath(0)
    require(empty.read('D', 0, 0) == [])
    empty.write('H', 0, [])
    require(empty.version == 0 and empty.moves == [])
    pending = CopyPath(4)
    pending.write('H', 0, [1, 2, 3, 4])
    ticket = pending.submit_copy('D', 0, 4)
    unpublished(pending, ticket)
    must_fail(lambda: pending.write('H', 0, [9]), ValueError)
    pending.complete_copy(ticket)
    require(pending.read('D', 0, 4) == [1, 2, 3, 4])
    faulty = CopyPath(4, 'early_publish')
    faulty.write('H', 0, [1, 2, 3, 4])
    faulty_ticket = faulty.submit_copy('D', 0, 4)
    must_fail(lambda: unpublished(faulty, faulty_ticket))
    print('copy progression: tiny-byte [64,1] refill and reverse [8,3,1]; missing-refill/early-publication faults detected')


class Ownership:
    def __init__(self, fault=''):
        self.phase = 'OPEN'
        self.epoch = 1
        self.readers = set()
        self.writer = None
        self.used = set()
        self.fault = fault

    def step(self, event):
        operation, epoch, ident = event
        if operation == 'done':
            if epoch != self.epoch and self.fault != 'ignore_epoch':
                return 'IGNORE'
            if ident in self.readers:
                self.readers.remove(ident)
                return 'DONE'
            if ident == self.writer and self.writer is not None:
                self.writer = None
                return 'DONE'
            return 'IGNORE'
        if epoch != self.epoch:
            return 'BLOCK'
        if operation in ('read', 'write'):
            if self.phase != 'OPEN' or ident in self.used:
                return 'BLOCK'
            if operation == 'read' and self.writer is not None and self.fault != 'read_with_writer':
                return 'BLOCK'
            if operation == 'write' and (self.writer is not None or
                    (self.readers and self.fault != 'write_with_reader')):
                return 'BLOCK'
            self.used.add(ident)
            if operation == 'read':
                self.readers.add(ident)
            else:
                self.writer = ident
            return 'ACCEPT'
        if operation == 'close':
            if self.phase != 'OPEN' or ((self.readers or self.writer is not None)
                                         and self.fault != 'close_busy'):
                return 'BLOCK'
            self.phase = 'CLOSED'
            return 'ACCEPT'
        if operation == 'retire':
            if self.phase != 'CLOSED':
                return 'BLOCK'
            self.phase = 'RETIRED'
            return 'ACCEPT'
        if operation == 'reuse':
            if self.phase != 'RETIRED':
                return 'BLOCK'
            self.phase = 'OPEN'
            self.epoch += 1
            self.used.clear()
            return 'ACCEPT'
        raise ValueError('unknown operation')

    def snapshot(self):
        active = [(self.epoch, ident, 'R') for ident in self.readers]
        if self.writer is not None:
            active.append((self.epoch, self.writer, 'W'))
        return self.phase, self.epoch, tuple(sorted(active))


class HistoryReference:
    CONFLICTS = {('R', 'W'), ('W', 'R'), ('W', 'W')}

    def __init__(self):
        self.lifecycle = ['OPEN']
        self.epoch = 1
        self.started = []
        self.finished = set()

    def active(self):
        return [record for record in self.started if record[:2] not in self.finished]

    def step(self, event):
        operation, epoch, ident = event
        active = self.active()
        phase = self.lifecycle[-1]
        if operation == 'done':
            token = (epoch, ident)
            if not any(record[:2] == token for record in active):
                return 'IGNORE'
            self.finished.add(token)
            return 'DONE'
        if epoch != self.epoch:
            return 'BLOCK'
        if operation in ('read', 'write'):
            mode = 'R' if operation == 'read' else 'W'
            already_used = any(record[:2] == (epoch, ident) for record in self.started)
            conflicting = any((mode, record[2]) in self.CONFLICTS for record in active)
            if phase != 'OPEN' or already_used or conflicting:
                return 'BLOCK'
            self.started.append((epoch, ident, mode))
            return 'ACCEPT'
        transitions = {'close': ('OPEN', 'CLOSED'), 'retire': ('CLOSED', 'RETIRED'),
                       'reuse': ('RETIRED', 'OPEN')}
        previous, following = transitions[operation]
        if phase != previous or active:
            return 'BLOCK'
        self.lifecycle.append(following)
        if operation == 'reuse':
            self.epoch += 1
        return 'ACCEPT'

    def snapshot(self):
        return self.lifecycle[-1], self.epoch, tuple(sorted(self.active()))


def judge_ownership(trace, fault=''):
    candidate = Ownership(fault)
    reference = HistoryReference()
    for event in trace:
        expected = reference.step(event)
        actual = candidate.step(event)
        snapshot = candidate.snapshot()
        active = snapshot[2]
        if any(mode == 'W' for epoch, ident, mode in active) and len(active) != 1:
            return False
        if snapshot[0] != 'OPEN' and active:
            return False
        if actual != expected or snapshot != reference.snapshot():
            return False
    return True


def test_ownership():
    alphabet = [('read', 1, 'r'), ('read', 1, 's'), ('write', 1, 'w'),
                ('done', 1, 'r'), ('done', 1, 's'), ('done', 1, 'w'),
                ('close', 1, None), ('retire', 1, None)]
    checked = 0
    for length in range(5):
        for trace in product(alphabet, repeat=length):
            require(judge_ownership(trace), 'ownership candidate/reference mismatch')
            checked += 1
    collision = [('read', 1, 'r'), ('done', 1, 'r'), ('close', 1, None),
                 ('retire', 1, None), ('reuse', 1, None), ('read', 2, 'r'),
                 ('done', 1, 'r'), ('close', 2, None), ('done', 2, 'r'),
                 ('close', 2, None)]
    require(judge_ownership(collision))
    probe = Ownership()
    outcomes = [probe.step(event) for event in collision]
    require(outcomes[6] == 'IGNORE' and outcomes[7] == 'BLOCK')
    require(outcomes[8:] == ['DONE', 'ACCEPT'])
    additional = [
        [('read', 1, 'r'), ('done', 1, 'r'), ('read', 1, 'r')],
        [('read', 1, 'r'), ('read', 1, 's'), ('done', 1, 's'), ('done', 1, 'r'), ('close', 1, None)],
        [('read', 1, 'r'), ('done', 2, 'r'), ('close', 1, None)],
        [('write', 1, 'w'), ('done', 1, 'w'), ('done', 1, 'w'), ('read', 1, 'r')],
    ]
    for trace in additional:
        require(judge_ownership(trace))
    controls = {
        'read_with_writer': [('write', 1, 'w'), ('read', 1, 'r')],
        'write_with_reader': [('read', 1, 'r'), ('write', 1, 'w')],
        'close_busy': [('read', 1, 'r'), ('close', 1, None)],
        'ignore_epoch': collision,
    }
    for fault, trace in controls.items():
        require(not judge_ownership(trace, fault), 'undetected ownership fault: ' + fault)
    require(checked == 4681)
    print('ownership: %d short traces plus reuse/duplicate traces; four mutations rejected by same reference/invariants' % checked)


def mapped_access_trace(mode, start, end, initialized, fault=''):
    if mode not in ('READ', 'OVERWRITE', 'UPDATE') or not 0 <= start <= end <= 4:
        raise ValueError('invalid modeled access')
    if start == end:
        return []
    needs_old = mode != 'OVERWRITE'
    if needs_old and not initialized:
        raise ValueError('old content is uninitialized')
    token, uid, preparation = ('H', 1), 'u1', 'p1'
    rights = 'r' if mode == 'READ' else 'w' if mode == 'OVERWRITE' else 'rw'
    events = [('alloc', token, 4), ('hold', token, uid)]
    if fault != 'old_content_guard' or needs_old:
        mapped_end = end - 1 if fault == 'short_mapping' else end
        mapped_rights = 'r' if fault == 'read_only' else rights
        events.extend([('register', token, start, end),
                       ('map', token, preparation, start, mapped_end, mapped_rights)])
        if fault != 'missing_ready':
            events.append(('ready', token, preparation))
    if needs_old:
        events.append(('restore', token, uid, start, end))
    events.extend([('submit', token, uid, mode, start, end),
                   ('done', token, uid), ('drop', token, uid)])
    return events


def audit_address(events):
    live = {}
    registered = {}
    mapping = {}
    preparations = set()
    ready = set()
    holds = set()
    active = {}
    submitted = []
    restored = 0
    required_rights = {'READ': {'r'}, 'OVERWRITE': {'w'}, 'UPDATE': {'r', 'w'}}
    for event in events:
        operation, token, *args = event
        if operation == 'alloc':
            size, = args
            require(token not in live and size > 0, 'backing identity')
            live[token] = size
            registered[token] = set()
        elif operation == 'hold':
            uid, = args
            require(token in live and (token, uid) not in holds, 'backing hold')
            holds.add((token, uid))
        elif operation == 'register':
            start, end = args
            require(token in live and 0 <= start < end <= live[token], 'registration bounds')
            require(any(owner_token == token for owner_token, uid in holds), 'registration lifetime')
            registered[token].update(range(start, end))
        elif operation == 'map':
            preparation, start, end, rights = args
            require(token in live and 0 <= start < end <= live[token], 'mapping bounds')
            require(set(rights) <= {'r', 'w'} and bool(rights), 'permission domain')
            require((token, preparation) not in preparations, 'preparation identity')
            require(any(owner_token == token for owner_token, uid in holds), 'mapping lifetime')
            for busy_token, busy_mode, busy_start, busy_end in active.values():
                require(not (start < busy_end and busy_start < end), 'mapping changed while in flight')
            for byte in range(start, end):
                require(byte in registered[token], 'unregistered byte')
                mapping[byte] = (token, preparation, frozenset(rights))
            preparations.add((token, preparation))
        elif operation == 'ready':
            preparation, = args
            require((token, preparation) in preparations, 'unknown preparation completion')
            ready.add((token, preparation))
        elif operation == 'restore':
            uid, start, end = args
            require((token, uid) in holds and 0 <= start < end <= live[token], 'restore lifetime/range')
            restored += end - start
        elif operation == 'submit':
            uid, mode, start, end = args
            require(mode in required_rights, 'unknown access mode')
            require(token in live and 0 <= start < end <= live[token], 'access bounds/backing')
            require((token, uid) in holds and (token, uid) not in active, 'access lifetime')
            for byte in range(start, end):
                entry = mapping.get(byte)
                require(entry is not None, 'address qualification: unmapped byte')
                mapped_token, preparation, permissions = entry
                require(mapped_token == token, 'address qualification: backing or epoch mismatch')
                require((token, preparation) in ready, 'address qualification: preparation pending')
                require(required_rights[mode] <= permissions, 'address qualification: permission')
            active[(token, uid)] = (token, mode, start, end)
            submitted.append((mode, start, end))
        elif operation == 'done':
            uid, = args
            require((token, uid) in active, 'unknown last use')
            del active[(token, uid)]
        elif operation == 'drop':
            uid, = args
            require((token, uid) in holds and (token, uid) not in active,
                    'lifetime: release while in flight')
            holds.remove((token, uid))
        elif operation == 'unmap':
            require(not any(record[0] == token for record in active.values()),
                    'lifetime: unmap while in flight')
            for byte in list(mapping):
                if mapping[byte][0] == token:
                    del mapping[byte]
        else:
            raise AssertionError('unknown address event: ' + operation)
    return {'submitted': tuple(submitted), 'restored': restored,
            'active': len(active), 'mapped': len(mapping), 'holds': len(holds)}


def test_address_preparation():
    checked = 0
    for mode in ('READ', 'OVERWRITE', 'UPDATE'):
        for start in range(5):
            for end in range(start, 5):
                events = mapped_access_trace(mode, start, end, mode != 'OVERWRITE')
                observation = audit_address(events)
                expected_submits = () if start == end else ((mode, start, end),)
                require(observation['submitted'] == expected_submits)
                require(observation['restored'] == (0 if mode == 'OVERWRITE' else end - start))
                require(observation['active'] == observation['holds'] == 0)
                if start == end:
                    require(events == [] and observation['mapped'] == 0)
                checked += 1
    first = mapped_access_trace('OVERWRITE', 0, 4, False)
    require(audit_address(first[:1])['mapped'] == 0, 'initial backing must not imply mapping')
    require(audit_address(first)['restored'] == 0, 'overwrite must not recover old bytes')
    require([event[0] for event in first] ==
            ['alloc', 'hold', 'register', 'map', 'ready', 'submit', 'done', 'drop'])
    for fault in ('old_content_guard', 'missing_ready', 'read_only', 'short_mapping'):
        must_fail(lambda fault=fault: audit_address(
            mapped_access_trace('OVERWRITE', 0, 4, False, fault)))
    submit_index = next(index for index, event in enumerate(first) if event[0] == 'submit')
    ready_index = next(index for index, event in enumerate(first) if event[0] == 'ready')
    pending = first[:ready_index] + [first[submit_index]]
    must_fail(lambda: audit_address(pending))
    require(audit_address(first[:submit_index])['submitted'] == ())
    done_index = next(index for index, event in enumerate(first) if event[0] == 'done')
    in_flight = first[:done_index]
    token = ('H', 1)
    must_fail(lambda: audit_address(in_flight + [('unmap', token)]))
    must_fail(lambda: audit_address(in_flight + [('drop', token, 'u1')]))
    reused = first + [('hold', token, 'u2'), ('map', token, 'p2', 0, 4, 'r'),
                      ('ready', token, 'p2'), ('submit', token, 'u2', 'READ', 0, 4),
                      ('done', token, 'u2'), ('drop', token, 'u2'), ('unmap', token)]
    observation = audit_address(reused)
    require(observation['submitted'] == (('OVERWRITE', 0, 4), ('READ', 0, 4)))
    require(observation['mapped'] == observation['holds'] == observation['active'] == 0)
    must_fail(lambda: mapped_access_trace('READ', 0, 1, False), ValueError)
    must_fail(lambda: mapped_access_trace('UPDATE', 0, 1, False), ValueError)
    require(checked == 45)
    print('address: %d range/mode traces from no device mapping; first OVERWRITE restores 0 old bytes' % checked)
    print('address: skipped/pending preparation, read-only permission, short mapping and two early-release controls rejected')
    print('address: completed permission change/reuse accepted; empty ranges submit nothing')


def audit_physical(events):
    # Independent reference: enumerate physical bytes, not candidate intervals.
    active = {}
    seen = set()
    for event in events:
        if event[0] == 'begin':
            _, ident, kind, queue, accesses = event
            require(ident not in seen, 'duplicate physical identity')
            seen.add(ident)
            cells = {}
            for token, mode, start, end in accesses:
                require(mode in ('R', 'W') and start < end)
                for byte in range(start, end):
                    key = (token, byte)
                    cells[key] = 'W' if mode == 'W' or cells.get(key) == 'W' else 'R'
            for old_ident, old_cells in active.items():
                for key in cells.keys() & old_cells.keys():
                    require(cells[key] == old_cells[key] == 'R',
                            'physical conflict: %s/%s on %r' % (old_ident, ident, key))
            active[ident] = cells
        elif event[0] == 'end':
            _, ident = event
            require(ident in active, 'unmatched physical completion')
            del active[ident]
        else:
            raise AssertionError('unknown physical event')
    return active


class RestoreScheduler:
    # One state owner; advance is the public normal event/progress entry.
    # This schedules established restores and consumers, not an entire phase/DAG.
    def __init__(self, size=4, fault=''):
        self.size = size
        self.epoch = 1
        self.h = ('host_backing', self.epoch)
        self.d = ('device_backing', self.epoch)
        self.reference = tuple(17 + 3 * byte for byte in range(size))
        self.nu = tuple(1 + byte // 2 for byte in range(size))
        self.source_values = list(self.reference)
        self.source_tags = list(self.nu)
        self.values = [None] * size
        self.tags = [None] * size
        self.parents = {}
        self.tasks = {}
        self.physical = {}
        self.events = []
        self.counter = 0
        self.joins = 0
        self.fault = fault
        self.addr = {token: {'ready': True, 'start': 0, 'end': size,
                             'rights': frozenset(('R', 'W'))}
                     for token in (self.h, self.d)}
        self.holds = {}
        self.external_done = set()

    def reserve(self, uid, start, end, queue):
        require(uid not in self.parents and uid not in self.tasks and
                0 <= start < end <= self.size)
        self.parents[uid] = {'start': start, 'end': end, 'queue': queue,
                             'state': 'WAIT', 'tasks': set()}

    def overlaps(self, first, second):
        token, mode, start, end = first
        old_token, old_mode, old_start, old_end = second
        return (token == old_token and start < old_end and old_start < end and
                (mode == 'W' or old_mode == 'W'))

    def begin(self, ident, kind, queue, accesses):
        if self.fault == 'future_parent_blocker' and kind == 'copy' and self.parents:
            return False
        bypass = ((self.fault in ('duplicate_unordered', 'move_ignores_reader') and kind == 'copy')
                  or (self.fault == 'consumer_ignores_moves' and kind == 'read'))
        if not bypass:
            for old_accesses in self.physical.values():
                if any(self.overlaps(new, old) for new in accesses for old in old_accesses):
                    return False
        require(ident not in self.physical)
        self.physical[ident] = tuple(accesses)
        self.events.append(('begin', ident, kind, queue, tuple(accesses)))
        return True

    def restore(self, uid, start, end, queue):
        require(uid in self.parents and 0 <= start < end <= self.size)
        parent = self.parents[uid]
        require(parent['state'] == 'WAIT')
        frozen = self.nu[start:end]
        key = (self.d, start, end, frozen)
        if self.fault != 'duplicate_unordered':
            for ident, task in self.tasks.items():
                if task['state'] in ('WAIT', 'ACTIVE') and task['key'] == key:
                    task['users'].add(uid)
                    parent['tasks'].add(ident)
                    self.joins += 1
                    return ident
            for task in self.tasks.values():
                if (task['state'] != 'DONE' and
                        start < task['end'] and task['start'] < end):
                    return None
        if all(self.tags[byte] == self.nu[byte] for byte in range(start, end)):
            return None
        self.counter += 1
        ident = 'c%d' % self.counter
        require(ident not in self.parents)
        accesses = ((self.h, 'R', start, end), (self.d, 'W', start, end))
        self.tasks[ident] = {'key': key, 'start': start, 'end': end, 'queue': queue,
                             'source': self.h, 'target': self.d, 'frozen': frozen,
                             'accesses': accesses, 'state': 'WAIT', 'users': {uid},
                             'attempted': False, 'submits': 0}
        self.holds[ident] = frozenset((self.h, self.d))
        parent['tasks'].add(ident)
        return ident

    def start_copy(self, ident):
        # TRY_START_RESTORE primitive. Normal regressions invoke it only via advance.
        task = self.tasks[ident]
        if task['state'] != 'WAIT':
            return False
        task['attempted'] = True
        start, end = task['start'], task['end']
        expected_accesses = ((self.h, 'R', start, end), (self.d, 'W', start, end))
        identity_ok = (task['source'] == self.h and task['target'] == self.d and
                       self.h[1] == self.d[1] == self.epoch and
                       task['key'] == (self.d, start, end, task['frozen']) and
                       task['accesses'] == expected_accesses)
        snapshot_ok = (self.nu[start:end] == task['frozen'] and
                       tuple(self.source_tags[start:end]) == task['frozen'])
        parents_ok = bool(task['users']) and all(
            uid in self.parents and self.parents[uid]['state'] == 'WAIT' and
            ident in self.parents[uid]['tasks'] for uid in task['users'])
        holds_ok = self.holds.get(ident) == frozenset((self.h, self.d))
        if not (identity_ok and snapshot_ok and parents_ok and holds_ok):
            task['state'] = 'QUARANTINED'
            return False
        for token, mode, first, last in task['accesses']:
            qualification = self.addr.get(token)
            if (qualification is None or not qualification['ready'] or
                    mode not in qualification['rights'] or
                    qualification['start'] > first or qualification['end'] < last):
                return False
        if not self.begin(ident, 'copy', task['queue'], task['accesses']):
            return False
        # Atomic in this single-owner model: record all ranges, change state,
        # and consume exactly one submit right before handing work to executor.
        task['state'] = 'ACTIVE'
        task['submits'] += 1
        require(task['submits'] == 1, 'one submit per restore identity')
        return True

    def retry_wait(self):
        started = 0
        # Finite creation-order snapshot; ACTIVE and DONE are never resubmitted.
        for ident in tuple(self.tasks):
            task = self.tasks[ident]
            if task['state'] != 'WAIT':
                continue
            # Fault permits the original attempt but disables existing-WAIT retries.
            if self.fault == 'skip_wait_retry' and task['attempted']:
                continue
            started += self.start_copy(ident)
        return started

    def finish_copy(self, ident):
        task = self.tasks[ident]
        require(task['state'] == 'ACTIVE')
        start, end = task['start'], task['end']
        require(self.nu[start:end] == task['frozen'] and
                tuple(self.source_tags[start:end]) == task['frozen'], 'frozen copy completion')
        for byte in range(start, end):
            self.values[byte] = self.source_values[byte]
            self.tags[byte] = task['frozen'][byte - start]
        task['state'] = 'DONE'
        del self.physical[ident]
        del self.holds[ident]
        self.events.append(('end', ident))

    def start_read(self, uid):
        parent = self.parents[uid]
        require(parent['state'] == 'WAIT')
        start, end = parent['start'], parent['end']
        for ident, task in self.tasks.items():
            if task['state'] != 'DONE' and start < task['end'] and task['start'] < end:
                if self.fault != 'consumer_ignores_moves' or ident in parent['tasks']:
                    return False
        if any(self.tags[byte] != self.nu[byte] for byte in range(start, end)):
            return False
        if not self.begin(uid, 'read', parent['queue'], ((self.d, 'R', start, end),)):
            return False
        require(tuple(self.values[start:end]) == self.reference[start:end], 'content reference')
        require(tuple(self.tags[start:end]) == self.nu[start:end], 'read version reference')
        parent['state'] = 'ACTIVE'
        return True

    def finish_read(self, uid):
        parent = self.parents[uid]
        require(parent['state'] == 'ACTIVE')
        start, end = parent['start'], parent['end']
        require(tuple(self.values[start:end]) == self.reference[start:end], 'last-read content')
        require(tuple(self.tags[start:end]) == self.nu[start:end], 'version reference')
        parent['state'] = 'DONE'
        del self.physical[uid]
        self.events.append(('end', uid))

    def advance(self, completions=()):
        # The caller supplies only real completions of ACTIVE work.
        # A repeated matching DONE notification is idempotent in this fixed-epoch model.
        progress = False
        for kind, ident in completions:
            if kind == 'copy':
                if self.tasks[ident]['state'] == 'DONE':
                    continue
                self.finish_copy(ident)
            elif kind == 'read':
                if self.parents[ident]['state'] == 'DONE':
                    continue
                self.finish_read(ident)
            elif kind == 'external':
                if ident in self.external_done:
                    continue
                require(ident not in self.tasks and ident not in self.parents and
                        ident in self.physical, 'registered external last use')
                del self.physical[ident]
                self.events.append(('end', ident))
                self.external_done.add(ident)
            else:
                raise AssertionError('unknown completion kind')
            progress = True
        # Required transition: completion cleanup -> retry existing WAIT -> consumers.
        progress = bool(self.retry_wait()) or progress
        for uid in tuple(self.parents):
            if self.parents[uid]['state'] == 'WAIT':
                progress = self.start_read(uid) or progress
        self.audit()
        if any(task['state'] == 'QUARANTINED' for task in self.tasks.values()):
            return 'QUARANTINED'
        if self.physical:
            return 'IN_FLIGHT'
        if (all(task['state'] == 'DONE' for task in self.tasks.values()) and
                all(parent['state'] == 'DONE' for parent in self.parents.values())):
            return 'DRAINED'
        return 'PROGRESSED' if progress else 'STALLED'

    def audit(self):
        active = audit_physical(self.events)
        require(set(active) == set(self.physical), 'physical responsibility ledger')


def begin_count(model, ident):
    return sum(event[0] == 'begin' and event[1] == ident for event in model.events)


def require_drained(model):
    model.audit()
    require(not model.physical and not audit_physical(model.events), 'no in-flight ranges at drain')
    require(all(parent['state'] == 'DONE' for parent in model.parents.values()),
            'all consumers must complete')
    require(all(task['state'] == 'DONE' for task in model.tasks.values()),
            'all restores must complete')
    require(not model.holds, 'restore holds end after real last uses')
    require(all(begin_count(model, ident) == task['submits'] == 1
                for ident, task in model.tasks.items()), 'single physical begin per restore')


def expanded_wait(fault=''):
    model = RestoreScheduler(fault=fault)
    model.reserve('old', 0, 2, 'gpu_old')
    c0 = model.restore('old', 0, 2, 'copy0')
    model.advance()
    require(model.tasks[c0]['state'] == 'ACTIVE')
    model.advance((('copy', c0),))
    require(model.parents['old']['state'] == 'ACTIVE')
    model.reserve('new', 2, 4, 'gpu_new')
    move = model.restore('new', 0, 4, 'copy_full')
    model.advance()
    require(model.tasks[move]['state'] == 'WAIT', 'expanded target waits for old physical R')
    require(begin_count(model, move) == 0 and model.tasks[move]['attempted'])
    model.audit()
    return model, move


def run_expanded_regression(fault=''):
    model, move = expanded_wait(fault)
    saved_task = model.tasks[move]
    saved_parent = model.parents['new']
    saved_key = saved_task['key']
    saved_users = frozenset(saved_task['users'])
    saved_counter = model.counter
    # Re-entering restore only joins; it cannot replace the established WAIT identity.
    require(model.restore('new', 0, 4, 'ignored_retry_queue') == move)
    require(model.counter == saved_counter)
    # No direct start_copy or manufactured copy completion after the reader finishes.
    model.advance((('read', 'old'),))
    require(model.tasks[move] is saved_task and model.parents['new'] is saved_parent,
            'WAIT retry preserves task and parent objects')
    require(saved_task['key'] == saved_key and frozenset(saved_task['users']) == saved_users and
            model.counter == saved_counter and saved_parent['tasks'] == {move},
            'WAIT retry preserves key, uid and subscriptions')
    if fault == 'skip_wait_retry':
        require(not model.physical and saved_task['state'] == 'WAIT' and
                saved_parent['state'] == 'WAIT', 'fault leaves no in-flight work to release WAIT')
        require(model.advance() == 'STALLED' and begin_count(model, move) == 0,
                'normal advance exposes missing-retry stall')
    require(saved_task['state'] == 'ACTIVE', 'A04: existing WAIT was not retried')
    require(begin_count(model, move) == saved_task['submits'] == 1)
    require(saved_parent['state'] == 'WAIT', 'consumer still waits for actual copy completion')
    model.advance()
    require(begin_count(model, move) == 1, 'ACTIVE cannot be resubmitted')
    model.advance((('copy', move),))
    require(saved_parent['state'] == 'ACTIVE', 'copy completion automatically admits consumer')
    require(tuple(model.values) == model.reference and tuple(model.tags) == model.nu,
            'expanded byte and range-version reference')
    # A repeated DONE notification must not create another physical begin/end.
    ends_before = sum(event == ('end', move) for event in model.events)
    model.advance((('copy', move),))
    require(begin_count(model, move) == 1 and
            sum(event == ('end', move) for event in model.events) == ends_before == 1)
    require(model.advance((('read', 'new'),)) == 'DRAINED')
    require(model.advance() == 'DRAINED')
    require_drained(model)
    return model


def test_parallel_restore():
    checked = 0
    for request_order in (('r1', 'r2'), ('r2', 'r1')):
        for completion_order in ((0, 1), (1, 0)):
            for reader_order in (('r1', 'r2'), ('r2', 'r1')):
                st = RestoreScheduler()
                for uid in request_order:
                    st.reserve(uid, 0, 4, 'gpu_' + uid)
                first, second = request_order
                tasks = [st.restore(first, 0, 2, 'copy0'),
                         st.restore(first, 2, 4, 'copy1')]
                require(st.restore(second, 0, 2, 'other_copy0') == tasks[0])
                require(st.restore(second, 2, 4, 'other_copy1') == tasks[1])
                require(len(st.tasks) == 2 and st.joins == 2)
                st.advance()
                require(all(st.tasks[ident]['state'] == 'ACTIVE' for ident in tasks),
                        'logical leases must not block own recovery')
                require(all(parent['state'] == 'WAIT' for parent in st.parents.values()))
                st.advance((('copy', tasks[completion_order[0]]),))
                require(all(parent['state'] == 'WAIT' for parent in st.parents.values()),
                        'one fragment must not authorize a full-range read')
                st.advance((('copy', tasks[completion_order[1]]),))
                require(all(parent['state'] == 'ACTIVE' for parent in st.parents.values()),
                        'normal progress automatically admits both readers')
                st.advance()
                for uid in reader_order:
                    st.advance((('read', uid),))
                require_drained(st)
                require(tuple(st.values) == st.reference and tuple(st.tags) == st.nu)
                require(sum(task['end'] - task['start'] for task in st.tasks.values()) == 4)
                require(all(task['users'] == {'r1', 'r2'} for task in st.tasks.values()))
                checked += 1

    # Partial-overlap replanning is explicitly supplied; full DAG planning is out of scope.
    st = RestoreScheduler()
    st.reserve('a', 0, 3, 'gpu_a')
    st.reserve('b', 1, 4, 'gpu_b')
    c1 = st.restore('a', 0, 3, 'copy_a')
    st.advance()
    require(st.restore('b', 1, 4, 'copy_b') is None and len(st.tasks) == 1)
    st.advance((('copy', c1),))
    require(st.parents['a']['state'] == 'ACTIVE' and st.parents['b']['state'] == 'WAIT')
    c2 = st.restore('b', 3, 4, 'copy_b')
    st.advance()
    require(st.tasks[c2]['state'] == 'ACTIVE', 'disjoint half-open boundary may progress')
    require(st.parents['b']['state'] == 'WAIT')
    st.advance((('copy', c2),))
    require(st.parents['b']['state'] == 'ACTIVE')
    st.advance((('read', 'b'), ('read', 'a')))
    require_drained(st)

    run_expanded_regression()
    must_fail_with(lambda: run_expanded_regression('skip_wait_retry'),
                   'A04: existing WAIT was not retried')

    # Safety mutations deliberately invoke primitives to generate illegal begin events.
    bad, move = expanded_wait()
    bad.fault = 'move_ignores_reader'
    require(bad.start_copy(move))
    must_fail(lambda: bad.audit())

    bad = RestoreScheduler(fault='duplicate_unordered')
    bad.reserve('a', 0, 4, 'gpu_a')
    bad.reserve('b', 0, 4, 'gpu_b')
    ca = bad.restore('a', 0, 4, 'copy_a')
    cb = bad.restore('b', 0, 4, 'copy_b')
    require(ca != cb and bad.start_copy(ca) and bad.start_copy(cb))
    must_fail(lambda: bad.audit())
    bad.finish_copy(cb)
    # Isolate RAW from the WAW already detected in complete history.
    remaining_snapshot = [('begin', ca, 'copy', 'copy_a', bad.physical[ca])]
    bad.fault = 'consumer_ignores_moves'
    require(bad.start_read('b'))
    must_fail(lambda: audit_physical(remaining_snapshot + [bad.events[-1]]))

    st, move = expanded_wait()
    st.reserve('probe', 0, 2, 'gpu_probe')
    require(not st.start_read('probe'), 'foreign WAIT destination write must be covered')
    st.advance((('read', 'old'),))
    require(st.tasks[move]['state'] == 'ACTIVE' and st.parents['probe']['state'] == 'WAIT')
    st.fault = 'consumer_ignores_moves'
    require(st.start_read('probe'))
    must_fail(lambda: st.audit())

    dead = RestoreScheduler(fault='future_parent_blocker')
    dead.reserve('a', 0, 4, 'gpu_a')
    task = dead.restore('a', 0, 4, 'copy_a')
    require(dead.advance() == 'STALLED')
    must_fail_with(lambda: require(dead.tasks[task]['state'] == 'ACTIVE',
                                  'future parent self-dependency blocks progress'),
                   'future parent self-dependency blocks progress')

    # Primitive source W probe; no logical content update is claimed here.
    source = RestoreScheduler()
    require(source.begin('source_writer', 'write', 'host_old', ((source.h, 'W', 0, 4),)))
    source.reserve('a', 0, 4, 'gpu_a')
    task = source.restore('a', 0, 4, 'copy_a')
    source.advance()
    require(source.tasks[task]['state'] == 'WAIT' and begin_count(source, task) == 0,
            'copy source R waits for existing source W')
    source.advance((('external', 'source_writer'),))
    require(source.tasks[task]['state'] == 'ACTIVE' and begin_count(source, task) == 1,
            'source completion retries the same WAIT through normal advance')
    source.advance((('copy', task),))
    require(source.parents['a']['state'] == 'ACTIVE')
    require(source.advance((('read', 'a'),)) == 'DRAINED')
    require_drained(source)
    require(checked == 8)
    print('physical restore: 8 normal event-loop request/copy/read orders; shared subscriptions and byte/version reference')
    print('physical restore: partial overlap and half-open boundary retained; WAW, RAW/WAR and foreign writer controls detected')
    print('WAIT retry: expanded target and source W release start the original task once through normal advance and drain')
    print('WAIT retry: disabled-existing-WAIT scan leaves no in-flight work and is detected as A04; future-parent control retained')


def test_retry_revalidation():
    st, move = expanded_wait()
    task = st.tasks[move]
    key = task['key']
    st.addr[st.h]['ready'] = False
    st.advance((('read', 'old'),))
    require(task['state'] == 'WAIT' and begin_count(st, move) == 0,
            'retry must recheck preparation readiness')
    st.addr[st.h]['ready'] = True
    st.advance()
    require(task['state'] == 'ACTIVE' and task['key'] == key and begin_count(st, move) == 1)
    st.advance((('copy', move),))
    st.advance((('read', 'new'),))
    require_drained(st)

    for missing in ('source_version', 'content_version', 'hold'):
        model, ident = expanded_wait()
        saved = model.tasks[ident]
        original_key = saved['key']
        if missing == 'source_version':
            model.source_tags[2] = 99
        elif missing == 'content_version':
            model.nu = model.nu[:2] + (99,) + model.nu[3:]
        else:
            del model.holds[ident]
        require(model.advance((('read', 'old'),)) == 'QUARANTINED')
        require(saved['state'] == 'QUARANTINED' and saved['key'] == original_key and
                begin_count(model, ident) == 0 and model.parents['new']['state'] == 'WAIT',
                'failed revalidation must not silently replace identity or consume')
    print('WAIT retry revalidation: pending address remains WAIT; readiness resumes; source/content-version and hold mismatch quarantine')


def test_capacity():
    need, staging, available = 64, 1, 64
    require(available < need + staging and available >= need, 'ignored staging changes admission')
    aliases = [('p1', 10), ('p1', 10), ('p2', 5)]
    unique = {}
    for identity, size in aliases:
        if identity in unique:
            require(unique[identity] == size)
        unique[identity] = size
    reference_union = {('p1', byte) for byte in range(10)} | {('p2', byte) for byte in range(5)}
    require(sum(unique.values()) == len(reference_union) == 15)
    require(sum(size for identity, size in aliases) != len(reference_union))
    print('capacity: staging omission and physical-alias double-count detected')


def main():
    test_pages()
    test_costs()
    test_migration_and_mapping()
    test_copy_progression()
    test_ownership()
    test_address_preparation()
    test_parallel_restore()
    test_retry_revalidation()
    test_capacity()
    print('scope: deterministic offline CPU byte/event models only; no GPU, driver, allocator, graph, or upstream execution')


if __name__ == '__main__':
    main()
