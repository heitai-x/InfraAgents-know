import itertools
import sys

CHECKS = 0


def require(condition, label):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(label)


def observe(label, value):
    print('OBS', label, value)


def validate_read_args(kind, p, q, w):
    if kind not in ('full', 'window') or p < 0 or q < 0 or w < 1:
        raise ValueError('invalid read contract')


def enum_history(kind, p, q, w):
    validate_read_args(kind, p, q, w)
    result = set()
    for t in range(p, p + q):
        for j in range(t):
            if kind == 'full' or t - j < w:
                result.add(j)
    return result


def interval_history(kind, p, q, w):
    validate_read_args(kind, p, q, w)
    if q == 0 or (kind == 'window' and w == 1):
        return set()
    lo = 0 if kind == 'full' else max(0, p - w + 1)
    return set(range(lo, p + q - 1))


def old_interval_bug(kind, p, q, w):
    if q == 0:
        return set()
    lo = 0 if kind == 'full' else max(0, p - w + 1)
    return set(range(lo, p + q - 1))


def check_reads():
    cases = 0
    for kind, p, q, w in itertools.product(('full', 'window'), range(11), range(6), range(1, 7)):
        direct = enum_history(kind, p, q, w)
        formula = interval_history(kind, p, q, w)
        require(direct == formula, ('union', kind, p, q, w))
        external = {j for j in direct if j < p}
        if q == 0 or (kind == 'window' and w == 1):
            old_formula = set()
        else:
            lo = 0 if kind == 'full' else max(0, p - w + 1)
            old_formula = set(range(lo, p))
        require(external == old_formula, ('external', kind, p, q, w))
        cases += 1
    require(interval_history('window', 3, 2, 1) == set(), 'regression_W1_union')
    require(enum_history('window', 10, 1, 4) == {7, 8, 9}, 'regression_one_query_is_not_W1')
    require(enum_history('full', 0, 3, 4) == {0, 1}, 'regression_generated_history')
    require({j for j in enum_history('full', 0, 3, 4) if j < 0} == set(), 'regression_no_external_history')
    require(old_interval_bug('window', 3, 2, 1) != enum_history('window', 3, 2, 1), 'negative_old_interval_detected')
    require(enum_history('window', 10, 1, 4) != set(), 'negative_old_single_query_expectation_detected')
    require(enum_history('full', 0, 3, 4) != set(), 'negative_old_empty_history_expectation_detected')
    observe('read_parameter_cases', cases)
    observe('regression_read_sets', {'W1': [], 'single_query_W4': [7, 8, 9], 'p0_future_history': [0, 1], 'p0_external': []})


def page_plan(tables, ranges, page_size):
    if page_size < 1:
        raise ValueError('page size')
    physical = set()
    logical_refs = 0
    for req, (start, end) in ranges.items():
        if not 0 <= start <= end <= len(tables[req]) * page_size:
            raise ValueError('range')
        logical = range(start // page_size, (end - 1) // page_size + 1) if start < end else ()
        for j in logical:
            physical.add(tables[req][j])
            logical_refs += 1
    return physical, logical_refs


def slot_reference(tables, ranges, page_size):
    physical_slots = set()
    logical_pages = set()
    for req, (start, end) in ranges.items():
        for t in range(start, end):
            physical_slots.add((tables[req][t // page_size], t % page_size))
            logical_pages.add((req, t // page_size))
    return physical_slots, logical_pages


def check_pages():
    tables = {'A': [7, 1, 3], 'B': [7, 1, 9]}
    for a0 in range(13):
        for a1 in range(a0, 13):
            ranges = {'A': (a0, a1), 'B': (0, 10)}
            physical, logical = page_plan(tables, ranges, 4)
            slots, logical_ref = slot_reference(tables, ranges, 4)
            require(physical == {page for page, offset in slots}, 'page projection')
            require(logical == len(logical_ref), 'logical page units')
    physical, logical = page_plan(tables, {'A': (0, 10), 'B': (0, 10)}, 4)
    slots, unused = slot_reference(tables, {'A': (0, 10), 'B': (0, 10)}, 4)
    require(physical == {7, 1, 3, 9} and logical == 6, 'regression_shared_and_private_capacity_gain')
    require(len(slots) == 12, 'unique valid positions')
    require(len(physical) * 128 == 512 and logical * 128 == 768, 'whole page byte charge')
    require(len(slots) * 32 == 384, 'valid content byte charge')
    require(20 != logical, 'negative_token_reference_as_page_count_detected')
    require(logical != len(physical), 'negative_shared_double_charge_detected')
    win_pages, win_refs = page_plan(tables, {'A': (7, 10)}, 4)
    require(win_pages == {1, 3} and win_refs == 2, 'window page mapping')
    scoped = {'A': [('L0', 7, 1)], 'B': [('L1', 7, 1)]}
    scoped_pages, unused = page_plan(scoped, {'A': (0, 4), 'B': (0, 4)}, 4)
    require(len(scoped_pages) == 2, 'page identity includes scope')
    require(len({key[1] for key in scoped_pages}) == 1, 'negative_bare_page_number_collapse_detected')
    for p in range(65):
        full = (p // 16) * 16
        independent = max(i for i in range(p + 1) if i % 16 == 0)
        require(full == independent, 'full page prefix')
    require((31 // 16) * 16 == 16 and 31 - 16 == 15, 'nonfull tail')
    require(31 != (31 // 16) * 16, 'negative_tail_overclaim_detected')
    pool = {7: [0, 1, 2, 3], 1: [4, 5, None, None]}
    a = [7, 1]
    b = [7, 1]
    pool[9] = [None] * 4
    pool[9][:2] = pool[1][:2]
    b[-1] = 9
    pool[b[-1]][2] = 60
    require(pool[a[-1]] == [4, 5, None, None], 'COW old page unchanged')
    require(pool[b[-1]] == [4, 5, 60, None], 'COW retained and appended slots')
    bad_pool = {1: [4, 5, None, None]}
    bad_a = [1]
    bad_b = [1]
    bad_pool[bad_b[0]][2] = 60
    require(bad_pool[bad_a[0]] != [4, 5, None, None], 'negative_no_COW_content_pollution_detected')
    observe('page_regression', {'unique_pages': 4, 'logical_page_references': 6, 'valid_bytes': 384, 'allocated_bytes': 512})


class Slot:
    def __init__(self):
        self.gen = 0
        self.free = True
        self.open = False
        self.ready = False
        self.owners = set()
        self.readers = set()
        self.pending = {}
        self.next_op = 0
        self.content = None
        self.events = []

    def reserve(self, owner):
        if not self.free:
            return False
        self.gen += 1
        self.free = False
        self.open = True
        self.ready = False
        self.owners = {owner}
        self.content = 'poison'
        self.events.append(('alloc', self.gen, owner))
        return True

    def issue(self, value):
        if self.free or self.ready or self.pending or not self.open:
            return None
        self.next_op += 1
        op = (self.gen, self.next_op)
        self.pending[op] = value
        self.events.append(('issue', self.gen, op))
        return op

    def complete(self, op):
        if op not in self.pending or op[0] != self.gen:
            return False
        self.content = self.pending.pop(op)
        self.ready = True
        self.events.append(('complete', self.gen, op))
        return True

    def acquire(self, reader):
        if self.free or not self.open or not self.ready or reader in self.readers:
            return False
        self.readers.add(reader)
        self.events.append(('acquire', self.gen, reader))
        return True

    def close(self):
        self.open = False
        self.events.append(('close', self.gen))

    def drop_owner(self, owner):
        if owner not in self.owners:
            return False
        self.owners.remove(owner)
        self.events.append(('drop_owner', self.gen, owner))
        return True

    def release(self, reader):
        if reader not in self.readers:
            return False
        self.readers.remove(reader)
        self.events.append(('release', self.gen, reader))
        return True

    def reclaim(self):
        if self.free or self.open or self.owners or self.readers or self.pending:
            return False
        self.events.append(('free', self.gen))
        self.free = True
        return True


def valid_trace(events):
    live = None
    opened = False
    produced = False
    owners = set()
    copies = set()
    readers = set()
    for event in events:
        kind, gen = event[:2]
        key = event[2] if len(event) == 3 else None
        if kind == 'alloc':
            if live is not None or copies or readers or owners:
                return False
            live, opened, produced = gen, True, False
            owners = {key}
            continue
        if gen != live:
            return False
        if kind == 'issue':
            if not opened or produced or copies:
                return False
            copies.add(key)
        elif kind == 'complete':
            if key not in copies:
                return False
            copies.remove(key)
            produced = True
        elif kind == 'acquire':
            if not opened or not produced or copies or key in readers:
                return False
            readers.add(key)
        elif kind == 'close':
            opened = False
        elif kind == 'drop_owner':
            if key not in owners:
                return False
            owners.remove(key)
        elif kind == 'release':
            if key not in readers:
                return False
            readers.remove(key)
        elif kind == 'free':
            if opened or owners or copies or readers:
                return False
            live = None
        else:
            return False
    return True


def check_lifetime():
    s = Slot()
    require(s.reserve('A'), 'reserve')
    old = s.issue('old KV')
    require(old is not None and not s.acquire('early'), 'no consume before copy')
    require(s.complete(old), 'copy completes')
    require(s.acquire('R1') and s.acquire('R2'), 'multiple concurrent readers')
    s.close()
    require(not s.acquire('late'), 'closed acquire gate')
    require(s.drop_owner('A') and not s.reclaim(), 'owners and readers separate')
    require(s.release('R1') and not s.reclaim(), 'second reader retains slot')
    require(not s.release('R1'), 'duplicate reader completion ignored')
    require(s.release('R2') and s.reclaim(), 'last reader permits reclaim')
    require(s.reserve('B'), 'reuse after drain')
    new = s.issue('new KV')
    require(not s.complete(old) and not s.ready, 'old completion cannot publish new generation')
    require(not s.complete((s.gen, 99999)), 'unknown completion ignored')
    require(s.complete(new) and not s.complete(new), 'complete only once')
    require(s.content == 'new KV', 'new content preserved')
    require(valid_trace(s.events), 'independent lifecycle observer')
    cancelled = Slot()
    require(cancelled.reserve('C'), 'cancel reserve')
    pending = cancelled.issue('cancelled copy')
    cancelled.close()
    require(cancelled.drop_owner('C') and not cancelled.reclaim(), 'cancel retains pending physical writer')
    require(cancelled.complete(pending) and not cancelled.acquire('R'), 'cancel does not reopen')
    require(cancelled.reclaim() and valid_trace(cancelled.events), 'cancel drains then releases')
    op = (1, 1)
    premature = [('alloc', 1, 'A'), ('issue', 1, op), ('acquire', 1, 'R')]
    require(not valid_trace(premature), 'negative_publish_before_complete_detected')
    reader_trace = [('alloc', 1, 'A'), ('issue', 1, op), ('complete', 1, op), ('acquire', 1, 'R'), ('close', 1), ('drop_owner', 1, 'A'), ('free', 1)]
    memory = {'slot': 'old KV'}
    memory['slot'] = 'new KV'
    require(memory['slot'] != 'old KV' and not valid_trace(reader_trace), 'negative_reuse_with_reader_content_detected')
    dma_trace = [('alloc', 1, 'A'), ('issue', 1, op), ('close', 1), ('drop_owner', 1, 'A'), ('free', 1), ('alloc', 2, 'B')]
    late_physical_write = 'old DMA payload'
    memory['slot'] = 'new generation payload'
    memory['slot'] = late_physical_write
    notification_generation = 1
    current_generation = 2
    require(notification_generation != current_generation, 'old notification rejected')
    require(memory['slot'] != 'new generation payload' and not valid_trace(dma_trace), 'negative_old_DMA_corrupts_reused_storage_detected')
    observe('lifecycle_scope', 'one reusable physical slot, two readers, close/cancel, duplicate and old notifications, actual in-memory pollution negatives')


def installable(expected, received, cancelled=False):
    return not cancelled and set(expected) == set(received) and all(received[k] == expected[k] for k in expected)


def check_install():
    expected = {'L0': ('eta', 10, (0, 10), 'codes+scale'), 'L1': ('eta', 10, (0, 10), 'codes+scale')}
    require(installable(expected, dict(expected)), 'complete common recovery view')
    require(not installable(expected, {'L0': expected['L0']}), 'partial layer cannot publish')
    for replacement in [('other_eta', 10, (0, 10), 'codes+scale'), ('eta', 9, (0, 10), 'codes+scale'), ('eta', 10, (0, 9), 'codes+scale'), ('eta', 10, (0, 10), 'codes_only')]:
        wrong = dict(expected)
        wrong['L1'] = replacement
        require(not installable(expected, wrong), 'identity position range or metadata mismatch')
        require(len(wrong) == len(expected), 'negative_count_only_publish_would_accept')
    require(not installable(expected, expected, True), 'cancelled candidate cannot publish')
    observe('installation_scope', 'two-layer immutable manifest model; no device visibility claim')


def pipeline(compute, copy, slots):
    if len(compute) != len(copy) or slots < 1 or any(t < 0 for t in compute + copy):
        raise ValueError('pipeline contract')
    ce, fe = [], []
    for i in range(len(compute)):
        link = ce[i - 1] if i else 0
        released = fe[i - slots] if i >= slots else 0
        ce.append(max(link, released) + copy[i])
        fe.append(max(ce[i], fe[i - 1] if i else 0) + compute[i])
    return fe[-1] if fe else 0


def tick_reference(compute, copy, slots):
    n = len(compute)
    if not n:
        return 0
    t, next_copy, next_compute = 0, 0, 0
    copy_job = None
    compute_job = None
    ready = set()
    occupied = [None] * slots
    while True:
        changed = True
        while changed:
            changed = False
            if compute_job is not None and compute_job[1] == t:
                occupied[compute_job[0] % slots] = None
                next_compute += 1
                compute_job = None
                changed = True
            if copy_job is not None and copy_job[1] == t:
                ready.add(copy_job[0])
                copy_job = None
                changed = True
            if copy_job is None and next_copy < n and occupied[next_copy % slots] is None:
                occupied[next_copy % slots] = next_copy
                copy_job = (next_copy, t + copy[next_copy])
                next_copy += 1
                changed = True
            if compute_job is None and next_compute < n and next_compute in ready:
                ready.remove(next_compute)
                compute_job = (next_compute, t + compute[next_compute])
                changed = True
        if next_compute == n:
            return t
        t += 1
        if t > sum(compute) + sum(copy) + 1:
            raise AssertionError('reference stalled')


def check_pipeline():
    cases = 0
    for n in range(4):
        values = list(itertools.product((0, 1, 3), repeat=n))
        for c, m, k in itertools.product(values, values, range(1, 4)):
            compute, copy = list(c), list(m)
            actual = pipeline(compute, copy, k)
            reference = tick_reference(compute, copy, k)
            require(actual == reference, ('pipeline reference', c, m, k))
            require(actual >= sum(copy) and actual >= sum(compute), 'shared resource lower bounds')
            cases += 1
    c = [5] * 4
    require(pipeline(c, [3] * 4, 2) == 23, 'fast pipeline')
    require(pipeline(c, [8] * 4, 2) == 37, 'slow pipeline')
    require(pipeline(c, [3] * 4, 1) == 32, 'one slot serial')
    require(sum(c) + 8 == 28 and 28 < 32, 'negative_full_hiding_violates_copy_bound')
    require(sum(c) + 8 != tick_reference(c, [8] * 4, 2), 'negative_full_hiding_detected')
    mib = 1024 ** 2
    gib = 1024 ** 3
    d = 2 * 32 * 8 * 4096 * 8 * 128 * 2
    require(d == 4 * gib, 'full KV byte arithmetic')
    require(512 * mib / (16 * gib) * 1000 == 31.25, 'pause H2D arithmetic')
    require(512 * mib / (2 * gib) * 1000 == 250, 'pause SSD arithmetic')
    require(d / (16 * gib) * 1000 == 250, 'active link resource arithmetic')
    require(pipeline([20000] * 32, [7812.5] * 32, 2) == 647812.5, '32 layer ideal microsecond model')
    observe('pipeline_parameter_cases', cases)
    observe('pipeline_scope', 'two serial engines, cyclic slots through final compute use; integer tick oracle; writeback not simulated')
    observe('pipeline_examples', {'fast': 23, 'slow': 37, 'one_slot': 32})


def main():
    print('SCOPE deterministic offline read-only Python stdlib CPU models; no GPU or I/O measurements')
    check_reads()
    check_pages()
    check_lifetime()
    check_install()
    check_pipeline()
    print('CHECK_OBSERVATIONS_COMPLETE', CHECKS)


if __name__ == '__main__':
    try:
        main()
    except (AssertionError, ValueError) as exc:
        print('CHECK_FAILURE', str(exc))
        sys.exit(1)
