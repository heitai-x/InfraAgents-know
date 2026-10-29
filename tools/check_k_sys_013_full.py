import hashlib
import json
from dataclasses import dataclass
from itertools import product

UNIT = 1024 ** 2
ETA = ('model-v1', 'adapter-a', 'tenant-a', 'numeric-contract-a')
GEOMETRY = ('all-required-layers', 'kv-heads', 'absolute-positions')
REPRESENTATION = ('immutable', 'format-v1')
TOKENS = tuple(range(13))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(value):
    encoded = json.dumps(value, ensure_ascii=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def block_chain(tokens, page, skip_first=False):
    if page <= 0:
        raise ValueError('nonpositive page')
    parent = digest(['root', ETA])
    out = []
    for j in range(len(tokens) // page):
        block = tuple(tokens[j * page:(j + 1) * page])
        if skip_first and j == 0:
            block = ()
        parent = digest(['block', parent, block, j * page, (j + 1) * page,
                         GEOMETRY, REPRESENTATION])
        out.append(parent)
    return out


@dataclass(frozen=True)
class Need:
    key: tuple
    identity: tuple
    cells: frozenset
    target_bytes: int


def need(family, index, tokens=TOKENS, page=2):
    if family == 'kv':
        end = (index + 1) * page
        cells = frozenset(range(index * page, end))
        size = 4 * UNIT
    else:
        end = index
        cells = (frozenset(range(max(0, end - 4), end))
                 if family == 'win' else frozenset((end,)))
        size = {'win': 2, 'rec': 1, 'ext': 1}[family] * UNIT
    identity = (ETA, family, index, tuple(tokens[:end]), REPRESENTATION)
    return Need((family, index), identity, cells, size)


def manifest(position, tokens=TOKENS, page=2):
    if position == 0:
        return ()
    return tuple(need('kv', j, tokens, page) for j in range(position // page)) + tuple(
        need(f, position, tokens, page) for f in ('rec', 'win', 'ext'))


@dataclass
class Replica:
    rid: str
    key: tuple
    identity: tuple
    cells: frozenset
    slot: int
    generation: int = 1
    state: str = 'READY'
    visible: bool = True
    readers: int = 0


@dataclass
class Acquisition:
    position: int
    holds: tuple
    requirements: tuple
    reserved_bytes: int
    active: bool = True


class Store:
    def __init__(self):
        self.directory = {}
        self.replicas = {}
        self.slot_generation = {}
        self.used = 0
        self.rollback_pins = 0

    def add(self, item, state='READY', visible=True, advertise=True):
        slot = len(self.replicas)
        rid = 'r' + str(slot)
        r = Replica(rid, item.key, item.identity, item.cells, slot,
                    state=state, visible=visible)
        self.replicas[rid] = r
        self.slot_generation[slot] = 1
        if advertise:
            self.directory.setdefault(item.key, []).append(rid)
        return rid

    def usable(self, r, item, fault):
        if fault == 'directory_presence':
            return True
        if r.state != 'READY' or not r.visible:
            return False
        if fault != 'ignore_generation' and self.slot_generation.get(r.slot) != r.generation:
            return False
        if r.key != item.key:
            return False
        if fault != 'ignore_identity' and r.identity != item.identity:
            return False
        if fault != 'ignore_coverage' and not item.cells.issubset(r.cells):
            return False
        return True

    def release(self, rid):
        r = self.replicas[rid]
        require(r.readers > 0, 'unowned or duplicate source release')
        r.readers -= 1

    def acquire(self, tokens, page, task_positions, recovery, quota, fault=''):
        if page <= 0:
            raise ValueError('nonpositive page')
        last = (len(tokens) // page) * page
        for p in range(last, -1, -page):
            if p not in task_positions:
                continue
            if fault != 'ignore_sets' and any(p not in s for s in recovery.values()):
                continue
            items = manifest(p, tokens, page)
            holds = []
            complete = True
            for item in items:
                chosen = None
                for rid in self.directory.get(item.key, ()):
                    r = self.replicas[rid]
                    if self.usable(r, item, fault):
                        chosen = rid
                        break
                if chosen is None:
                    complete = False
                    break
                self.replicas[chosen].readers += 1
                holds.append(chosen)
            unique = {u.key: u for u in items}
            target_bytes = 2 * UNIT + sum(u.target_bytes for u in unique.values())
            if complete and self.used + target_bytes <= quota:
                self.used += target_bytes
                return Acquisition(p, tuple(holds), items, target_bytes)
            if fault != 'leak_rollback':
                for rid in holds:
                    self.release(rid)
                    self.rollback_pins += 1
        return None

    def close(self, result):
        require(result.active, 'duplicate result settlement')
        for rid in result.holds:
            self.release(rid)
        self.used -= result.reserved_bytes
        require(self.used >= 0, 'negative target accounting')
        result.active = False

    def evict(self, rid):
        r = self.replicas[rid]
        if r.readers:
            raise RuntimeError('active source lease')
        self.directory[r.key].remove(rid)
        r.state = 'RETIRED'
        self.slot_generation[r.slot] += 1


def recovery_sets():
    return {'kv': set(range(0, 13, 2)), 'rec': {0, 4, 8, 12},
            'win': {0, 8, 12}, 'ext': {0, 8, 12}}


def fixture(missing=-1, building=False, stale=False):
    s = Store()
    for j in range(6):
        if j != missing:
            s.add(need('kv', j))
    for p in (4, 8, 12):
        rid = s.add(need('rec', p), state='BUILDING' if building and p == 12 else 'READY')
        if stale and p == 12:
            s.slot_generation[s.replicas[rid].slot] = 2
    for p in (8, 12):
        s.add(need('win', p))
        s.add(need('ext', p))
    return s


def reference(s, tokens, page, task_positions, recovery, quota):
    feasible = []
    for p in range(len(tokens) + 1):
        if p % page or p not in task_positions:
            continue
        if not all(p in positions for positions in recovery.values()):
            continue
        demands = []
        if p:
            for t in range(p):
                j = t // page
                endpoint = (j + 1) * page
                identity = (ETA, 'kv', j, tuple(tokens[:endpoint]), REPRESENTATION)
                demands.append((('kv', j), identity, t))
            for t in range(max(0, p - 4), p):
                demands.append((('win', p),
                                (ETA, 'win', p, tuple(tokens[:p]), REPRESENTATION), t))
            for family in ('rec', 'ext'):
                demands.append(((family, p),
                                (ETA, family, p, tuple(tokens[:p]), REPRESENTATION), p))
        valid = True
        for key, identity, cell in demands:
            found = False
            for r in s.replicas.values():
                if (r.rid in s.directory.get(key, ()) and r.key == key
                        and r.identity == identity and cell in r.cells
                        and r.state == 'READY' and r.visible
                        and s.slot_generation.get(r.slot) == r.generation):
                    found = True
                    break
            if not found:
                valid = False
                break
        # Independent backing ledger: four units per full KV page;
        # two for the window, one each for the recurrent/external snapshots;
        # two for disjoint workspace and growth reservation. Empty target only.
        backing = [UNIT, UNIT]
        if p:
            backing += [UNIT for _ in range(4 * (p // page))]
            backing += [UNIT, UNIT, UNIT, UNIT]
        if valid and s.used + sum(backing) <= quota:
            feasible.append(p)
    return max(feasible) if feasible else None


def audit(s, result):
    for rid, r in s.replicas.items():
        expected = result.holds.count(rid) if result else 0
        require(r.readers == expected, 'lease leakage or missing protection')
    require(s.used == (result.reserved_bytes if result else 0), 'target reservation mismatch')
    if result:
        require(len(result.holds) == len(result.requirements), 'incomplete handoff')
        for rid, item in zip(result.holds, result.requirements):
            require(s.usable(s.replicas[rid], item, ''), 'invalid delivered object')


def settle(s, result):
    if result:
        s.close(result)
    audit(s, None)


def rejected_by_audit(s, result, expected):
    try:
        require((result.position if result else None) == expected, 'wrong recovery position')
        audit(s, result)
    except AssertionError:
        return True
    return False


def main():
    task = set(range(14))
    sets = recovery_sets()
    capacities = (0, 2, 14, 21, 22, 26, 30, 100)
    matrix_count = 0
    for cap, missing, building, stale in product(capacities, (-1, 0, 3, 5),
                                                 (False, True), (False, True)):
        s = fixture(missing, building, stale)
        expected = reference(s, TOKENS, 2, task, sets, cap * UNIT)
        result = s.acquire(TOKENS, 2, task, sets, cap * UNIT)
        require((result.position if result else None) == expected, 'reference mismatch')
        audit(s, result)
        settle(s, result)
        matrix_count += 1
    require(matrix_count == 128, 'matrix size')

    s = fixture()
    result = s.acquire(TOKENS, 2, task, sets, 26 * UNIT)
    require(result.position == 8 and result.reserved_bytes == 22 * UNIT, 'capacity fallback')
    require(s.rollback_pins == 9 and len(result.holds) == 7, 'fallback pin settlement')
    audit(s, result)
    held = result.holds[0]
    blocked = False
    try:
        s.evict(held)
    except RuntimeError:
        blocked = True
    require(blocked, 'source eviction must reject active lease')
    settle(s, result)
    s.evict(held)
    require(s.replicas[held].state == 'RETIRED', 'retirement after final reader')
    duplicate_blocked = False
    try:
        s.close(result)
    except AssertionError:
        duplicate_blocked = True
    require(duplicate_blocked, 'duplicate settlement')

    zero_sets = {f: {0} for f in sets}
    empty = Store()
    result = empty.acquire((), 2, {0}, zero_sets, 2 * UNIT)
    require(result is not None and result.position == 0 and not result.holds, 'empty state')
    settle(empty, result)
    absent_initial = dict(zero_sets)
    absent_initial['ext'] = set()
    require(empty.acquire((), 2, {0}, absent_initial, 100 * UNIT) is None,
            'missing initial external state cannot silently become zero')
    result = empty.acquire((9,), 2, {0}, zero_sets, 2 * UNIT)
    require(result.position == 0, 'partial page cannot be a cached full block')
    settle(empty, result)
    require(fixture().acquire(TOKENS, 2, set(), sets, 100 * UNIT) is None, 'empty task set')

    s = fixture()
    s.add(need('kv', 0), state='BUILDING')
    s.add(need('rec', 12), advertise=False)
    invisible = next(r for r in s.replicas.values() if r.key == ('rec', 12) and
                     r.rid in s.directory[('rec', 12)])
    invisible.visible = False
    expected = reference(s, TOKENS, 2, task, sets, 100 * UNIT)
    result = s.acquire(TOKENS, 2, task, sets, 100 * UNIT)
    require(expected == 8 and result.position == 8, 'visibility and unadvertised replica')
    audit(s, result)
    settle(s, result)

    negatives = []
    for fault in ('ignore_sets', 'directory_presence', 'ignore_generation',
                  'ignore_identity', 'ignore_coverage', 'leak_rollback'):
        s = fixture(building=fault == 'directory_presence', stale=fault == 'ignore_generation')
        allowed = recovery_sets()
        cap = 26 if fault == 'leak_rollback' else 100
        if fault == 'ignore_sets':
            allowed['ext'] = {0, 6, 10}
        if fault == 'ignore_identity':
            r = next(r for r in s.replicas.values() if r.key == ('rec', 12))
            r.identity = ('wrong-parent-or-namespace',)
        if fault == 'ignore_coverage':
            r = next(r for r in s.replicas.values() if r.key == ('win', 12))
            r.cells = r.cells - {11}
        expected = reference(s, TOKENS, 2, task, allowed, cap * UNIT)
        bad = s.acquire(TOKENS, 2, task, allowed, cap * UNIT, fault)
        require(rejected_by_audit(s, bad, expected), 'undetected negative control: ' + fault)
        negatives.append(fault)

    original = block_chain(TOKENS, 2)
    changed_first = (99,) + TOKENS[1:]
    changed_middle = TOKENS[:4] + (99,) + TOKENS[5:]
    require(len(original) == 6, 'chain/full-block count')
    require(original[0] == digest(['block', digest(['root', ETA]), (0, 1), 0, 2,
                                   GEOMETRY, REPRESENTATION]), 'first-block/root correspondence')
    require(all(a != b for a, b in zip(original, block_chain(changed_first, 2))),
            'first block must affect every descendant identity')
    modified = block_chain(changed_middle, 2)
    require(original[:2] == modified[:2] and all(a != b for a, b in zip(original[2:], modified[2:])),
            'causal identity propagation')
    require(block_chain((), 2) == [] and block_chain((1,), 2) == [], 'empty/partial identity')
    bad_a = block_chain(TOKENS, 2, skip_first=True)
    bad_b = block_chain(changed_first, 2, skip_first=True)
    require(bad_a == bad_b, 'first-block omission negative control must be exposed')
    require(original != bad_a, 'faulty chain escaped identity reference')
    negatives.append('first_block_omission')

    input_segments = (3, 4, 5, 6, 5, 7)
    generation = 10
    restored_segments = 3
    suffix = sum(input_segments[restored_segments:]) + generation
    formula_recompute = sum(input_segments) + generation
    formula_remote = 20 + suffix
    # Independent event trace includes each required input event and the common
    # generation event once; no call to the candidate formulas defines this oracle.
    direct_events = [('input-' + str(i), t) for i, t in enumerate(input_segments)]
    direct_events.append(('generation', generation))
    remote_events = [('acquire-and-publish', 20), ('input-3', 6),
                     ('input-4', 5), ('input-5', 7), ('generation', 10)]
    require(formula_recompute == sum(t for _, t in direct_events) == 40, 'recompute event ledger')
    require(formula_remote == sum(t for _, t in remote_events) == 48, 'nonempty suffix ledger')
    require(formula_recompute < formula_remote, 'correct candidate selection')
    faulty_recompute = sum(input_segments) + formula_recompute
    require(faulty_recompute == 70 and formula_remote < faulty_recompute,
            'duplicate input charge must reverse this decision')
    require(faulty_recompute != sum(t for _, t in direct_events), 'undetected double charge')
    negatives.append('duplicate_input_charge')

    transfer_ms = (512 * 1024 ** 2) / (20 * 1024 ** 3) * 1000
    require(transfer_ms == 25, 'binary transfer units')
    demand_bytes_per_second = 4 * 0.3 * (512 * 1024 ** 2)
    decimal_gbps = demand_bytes_per_second * 8 / 10 ** 9
    binary_gibits = demand_bytes_per_second * 8 / 2 ** 30
    require(abs(decimal_gbps - 5.1539607552) < 1e-12, 'decimal network conversion')
    require(abs(binary_gibits - 4.8) < 1e-12, 'binary bit conversion')
    require(abs(decimal_gbps - 4.8) > 0.35, 'unit-confusion negative control')
    negatives.append('GiB_to_Gbit_confusion')

    print('CPU acquisition matrix: 128 deterministic cases matched independent per-position/cell reference')
    print('capacity fallback: p=8 tokens, 4 KV blocks, 7 held objects, 22 MiB reserved; 9 discarded pins released')
    print('edges: empty/partial input, missing initialization, empty task set, visibility, duplicate settlement, leased eviction')
    print('negative controls detected: ' + ', '.join(negatives))
    print('cost event ledger: recompute=40 ms, remote=48 ms; faulty duplicate charge=70 ms')
    print('units: 512 MiB / 20 GiB/s = 25 ms; 0.6 GiB/s = 5.1539607552 Gbit/s = 4.8 Gibit/s')
    print('scope: standalone offline read-only Python stdlib CPU model; disjoint target allocations, serialized control')
    print('not modeled: transfer, conversion, publication transaction, real concurrency, DMA drain, GPU/NIC/RDMA performance or SLO')


if __name__ == '__main__':
    main()
