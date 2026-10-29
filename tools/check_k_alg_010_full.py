#!/usr/bin/env python3
'''K-ALG-010: deterministic offline stdlib CPU model; no file access.
No project imports, processes, credentials, network or GPU calls.
Physical pages use lossless affine encoding without rounding. This checks
transport of position/scale/zero-point metadata, not quantization accuracy.
The event order is synchronous; no concurrent allocator or recycler is modeled.
'''
import math
from dataclasses import dataclass, replace


class DomainError(ValueError):
    pass


class StaleCache(DomainError):
    pass


class Mismatch(AssertionError):
    pass


class HarnessError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Mismatch(message)


def shape(a):
    if not a:
        return (0, 0)
    width = len(a[0])
    if any(len(row) != width for row in a):
        raise DomainError('ragged matrix')
    return (len(a), width)


def mm(a, w):
    nr, nc = shape(w)
    if nr == 0 or nc == 0:
        raise DomainError('empty weights')
    if not a:
        return ()
    if shape(a)[1] != nr:
        raise DomainError('projection input width')
    return tuple(tuple(sum(row[k] * w[k][j] for k in range(nr))
                       for j in range(nc)) for row in a)


def reference_projection(e, w):
    return tuple(tuple(math.fsum(row[a] * w[a][b] for a in range(len(w)))
                       for b in range(len(w[0]))) for row in e)


def compare(a, b, label):
    require(shape(a) == shape(b), label + ': shape mismatch')
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            require(math.isfinite(x) and math.isfinite(y), label + ': nonfinite')
            require(abs(x - y) <= 1e-11 * max(1.0, abs(x), abs(y)),
                    label + ': numerical mismatch')


def negative(name, action, expected_message=None):
    try:
        action()
    except Mismatch as exc:
        if expected_message is not None and expected_message not in str(exc):
            raise HarnessError('unexpected rejection for ' + name + ': ' + str(exc))
        print('DETECTED negative control: ' + name + ' [' + str(exc) + ']')
        return
    raise HarnessError('negative control escaped: ' + name)


def expect_domain(action, label):
    try:
        action()
    except DomainError:
        return
    raise Mismatch(label + ': invalid input accepted')


def weights(rows, cols, seed):
    return tuple(tuple(((r * 7 + c * 3 + seed) % 17 - 8) / 8.0
                       for c in range(cols)) for r in range(rows))


WQ = weights(5, 8, 3)
WO = weights(12, 7, 11)
X = ((0.25, -0.5, 0.75, 1.0, -0.25),
     (-0.75, 0.5, 0.25, -0.5, 1.0))


@dataclass(frozen=True)
class Source:
    lineage: str
    content: str
    version: int
    e: tuple
    positions: tuple
    valid: tuple
    parent: object = None
    processor: str = 'processor-v1'
    encoder: str = 'encoder-v1'
    representation: str = 'float64-model'

    @property
    def key(self):
        # A trusted immutable content registration is an input assumption.
        return (self.content, self.version, self.processor, self.encoder,
                self.representation, len(self.e), self.positions, self.valid)


@dataclass(frozen=True)
class Contract:
    layer: str = 'cross-layer-2'
    wk: tuple = weights(3, 4, 5)
    wv: tuple = weights(3, 6, 9)
    hq: int = 4
    hkv: int = 2
    dk: int = 2
    dv: int = 3
    rho: tuple = (0, 0, 1, 1)
    layout: str = 'NHD'
    page_size: int = 2
    dtype: str = 'float64-affine-exact-model'
    scales: str = 'fixed-per-row-affine-v1'
    position_rule: str = 'unrotated-source-v1'


def validate_source(s, c):
    if len(s.positions) != len(s.e) or len(s.valid) != len(s.e):
        raise DomainError('source metadata length')
    if not s.e or not any(s.valid):
        raise DomainError('EMPTY_SOURCE')
    if len(set(s.positions)) != len(s.positions):
        raise DomainError('duplicate positions')
    if shape(s.e)[1] != len(c.wk) or len(c.wv) != len(c.wk):
        raise DomainError('source width')
    if c.page_size <= 0 or c.layout not in ('NHD', 'HND'):
        raise DomainError('layout')
    if shape(c.wk)[1] != c.hkv * c.dk or shape(c.wv)[1] != c.hkv * c.dv:
        raise DomainError('KV geometry')
    if len(c.rho) != c.hq or any(h < 0 or h >= c.hkv for h in c.rho):
        raise DomainError('head mapping')
    if any(not math.isfinite(v) for row in s.e for v in row):
        raise DomainError('nonfinite source')


def offset(c, u, h, d, width):
    if c.layout == 'NHD':
        return (u * c.hkv + h) * width + d
    return (h * c.page_size + u) * width + d


class Page:
    def __init__(self, c):
        self.k = [math.nan] * (c.page_size * c.hkv * c.dk)
        self.v = [math.nan] * (c.page_size * c.hkv * c.dv)
        self.positions = [None] * c.page_size
        self.metadata = [None] * c.page_size
        self.sealed = False
        self.borrow = 0


class Pool:
    def __init__(self):
        self.pages = {}
        self.refs = {}
        self.done = set()
        self.events = []

    def allocate(self, c):
        pid = 17 + 7 * len(self.pages)
        page = Page(c)
        require(all(math.isnan(x) for x in page.k + page.v), 'allocation not poisoned')
        require(all(x is None for x in page.positions + page.metadata),
                'allocation metadata not poisoned')
        self.pages[pid] = page
        self.refs[pid] = 0
        return pid

    def write(self, pid, u, j, position, k, v, c):
        page = self.pages[pid]
        require(not page.sealed, 'write to published page')
        # Powers of two and binary-rational fixtures make this reversible.
        # There is no rounding, clipping or integer quantization here.
        sk, zk = 2.0 ** (-(j % 3 + 1)), j % 3 - 1
        sv, zv = 2.0 ** (-((j + 1) % 3 + 1)), j % 5 - 2
        for h in range(c.hkv):
            for d in range(c.dk):
                page.k[offset(c, u, h, d, c.dk)] = k[h * c.dk + d] / sk + zk
            for d in range(c.dv):
                page.v[offset(c, u, h, d, c.dv)] = v[h * c.dv + d] / sv + zv
        page.positions[u] = position
        page.metadata[u] = (sk, zk, sv, zv)

    def copy_slot(self, src, dst, u, c, copy_metadata=True):
        old, new = self.pages[src], self.pages[dst]
        require(old.sealed and old.borrow > 0 and self.refs[src] > 0,
                'copy source not protected')
        require(not new.sealed, 'copy destination published')
        for h in range(c.hkv):
            for d in range(c.dk):
                a = offset(c, u, h, d, c.dk)
                new.k[a] = old.k[a]
            for d in range(c.dv):
                a = offset(c, u, h, d, c.dv)
                new.v[a] = old.v[a]
        if copy_metadata:
            new.positions[u] = old.positions[u]
            new.metadata[u] = old.metadata[u]

    def fingerprint(self, table):
        def frozen(values):
            return tuple('poison' if math.isnan(x) else x for x in values)
        return tuple((pid, frozen(self.pages[pid].k), frozen(self.pages[pid].v),
                      tuple(self.pages[pid].positions), tuple(self.pages[pid].metadata))
                     for pid in table)


@dataclass(frozen=True)
class Cache:
    source: Source
    contract: Contract
    pool: object
    table: tuple
    mode: str
    generation: int
    projected: tuple
    copied: tuple
    ready: tuple
    state: str = 'READY'

    @property
    def identity(self):
        return (self.source.key, self.contract.layer, self.contract)


class Registry:
    def __init__(self):
        self.index = {}
        self.lookups = []
        self.generation = 0
        self.pool = Pool()

    def predecessor(self, source, c, wrong_full_key=False):
        if source.parent is None:
            return None
        wanted = source.key if wrong_full_key else source.parent
        index_key = (source.lineage, wanted[1], c.layer)
        self.lookups.append((index_key, wanted))
        for old in self.index.get(index_key, ()):
            if old.source.key == wanted and old.state == 'READY':
                return old
        return None

    def build(self, source, c, fault=None, permutation=None):
        validate_source(source, c)
        old = self.predecessor(source, c, fault == 'full-key-lookup')
        stable = False
        if old is not None:
            nold = len(old.source.e)
            base = (source.version > old.source.version and len(source.e) >= nold
                    and c == old.contract
                    and source.processor == old.source.processor
                    and source.encoder == old.source.encoder
                    and source.representation == old.source.representation)
            prefix = (source.e[:nold] == old.source.e
                      and source.positions[:nold] == old.source.positions
                      and source.valid[:nold] == old.source.valid)
            stable = base and (prefix or fault == 'ignore-prefix')
        self.generation += 1
        g = self.generation
        p, n = c.page_size, len(source.e)
        count = (n + p - 1) // p
        deps = set(old.ready if stable else ())
        copied = []
        if stable:
            nold = len(old.source.e)
            shared = len(old.table) if n == nold else nold // p
            table = list(old.table[:shared])
            mode = 'APPEND'
            start = nold
        else:
            table = []
            mode = 'REBUILD'
            start = 0
        while len(table) < count:
            table.append(self.pool.allocate(c))
        if not stable:
            order = tuple(reversed(range(count))) if permutation is None else permutation
            if sorted(order) != list(range(count)):
                raise DomainError('physical permutation')
            table = [table[i] for i in order]
        elif permutation is not None:
            raise DomainError('append cannot silently rearrange existing pages')

        if stable and n > nold and nold % p:
            logical_page = nold // p
            src, dst = old.table[logical_page], table[logical_page]
            require(src != dst, 'non-full tail not isolated')
            require(set(old.ready) <= self.pool.done, 'copy before predecessor ready')
            self.pool.pages[src].borrow += 1
            self.pool.events.append(('hold', g, src))
            for j in range(logical_page * p, nold):
                if source.valid[j] and fault != 'omit-tail-copy':
                    self.pool.copy_slot(src, dst, j % p, c,
                                        copy_metadata=fault != 'omit-tail-metadata')
                    copied.append(j)
            # Faulty variants deliberately claim completion despite missing data.
            # The independent physical/reference validator must expose that hole.
            self.pool.done.add(('copy', g))
            deps.add(('copy', g))
            self.pool.events.append(('copy_done', g, src))
            self.pool.pages[src].borrow -= 1
            self.pool.events.append(('unhold', g, src))
            require(('copy', g) in self.pool.done, 'new writes before copy completion')

        projected = []
        for j in range(start, n):
            if source.valid[j]:
                k = mm((source.e[j],), c.wk)[0]
                v = mm((source.e[j],), c.wv)[0]
                self.pool.write(table[j // p], j % p, j, source.positions[j], k, v, c)
                projected.append(j)
        self.pool.done.add(('project', g))
        self.pool.events.append(('project_done', g))
        deps.add(('project', g))
        require(deps <= self.pool.done, 'publish before declared dependency completion')
        for pid in table:
            self.pool.pages[pid].sealed = True
            self.pool.refs[pid] += 1
        cache = Cache(source, c, self.pool, tuple(table), mode, g,
                      tuple(projected), tuple(copied), tuple(sorted(deps)))
        self.pool.events.append(('publish', g))
        self.index.setdefault((source.lineage, source.version, c.layer), []).append(cache)
        return cache


def physical_token(cache, j, h, reads):
    reads.append((j, h))
    c = cache.contract
    page = cache.pool.pages[cache.table[j // c.page_size]]
    u = j % c.page_size
    require(page.sealed, 'read unsealed page')
    require(page.positions[u] is not None and page.metadata[u] is not None,
            'row ' + str(j) + ': missing position/scale/zero-point')
    sk, zk, sv, zv = page.metadata[u]
    k = tuple((page.k[offset(c, u, h, d, c.dk)] - zk) * sk for d in range(c.dk))
    v = tuple((page.v[offset(c, u, h, d, c.dv)] - zv) * sv for d in range(c.dv))
    require(all(math.isfinite(x) for x in k + v), 'row ' + str(j) + ': poison data')
    return k, v, page.positions[u]


def validate_cache(cache, source, c):
    # Expectations never come from the page table, copied rows or cache tensors.
    kr = reference_projection(source.e, c.wk)
    vr = reference_projection(source.e, c.wv)
    require(cache.identity == (source.key, c.layer, c), 'published complete identity')
    for j in range(len(source.e)):
        if source.valid[j]:
            krow, vrow = [], []
            for h in range(c.hkv):
                k, v, position = physical_token(cache, j, h, [])
                require(position == source.positions[j], 'row ' + str(j) + ': position mismatch')
                krow.extend(k)
                vrow.extend(v)
            compare((tuple(krow),), (kr[j],), 'row ' + str(j) + ' full K reference')
            compare((tuple(vrow),), (vr[j],), 'row ' + str(j) + ' full V reference')


@dataclass(frozen=True)
class Request:
    target: str
    expected: tuple


def bind(source, contract, target):
    return Request(target, (source.key, contract.layer, contract))


def guard(cache, request, weak=False):
    if cache.state != 'READY':
        raise StaleCache('not published')
    if weak:
        if (cache.source.version != request.expected[0][1]
                or cache.contract != request.expected[2]):
            raise StaleCache('weak version/representation mismatch')
    elif cache.identity != request.expected:
        raise StaleCache('complete request identity mismatch')


def read(cache, request, x, allow, fault=None, reads=None, weak=False):
    if reads is None:
        reads = []
    guard(cache, request, weak)
    c = cache.contract
    require(set(cache.ready) <= cache.pool.done, 'consumer before ready')
    if shape(WQ) != (5, c.hq * c.dk) or shape(WO) != (c.hq * c.dv, 7):
        raise DomainError('Q/O weight shape')
    if not x:
        return ()
    if shape(x)[1] != 5:
        raise DomainError('expected unprojected hidden width 5')
    if any(not math.isfinite(v) for row in x for v in row):
        raise DomainError('nonfinite hidden state')
    qflat = mm(x, WQ)
    rows = []
    n = len(cache.source.e)
    for i in range(len(x)):
        joined = []
        for h in range(c.hq):
            m, ell, u = -math.inf, 0.0, [0.0] * c.dv
            kvh = h % c.hkv if fault == 'rho-modulo' else c.rho[h]
            for a in range(len(cache.table)):
                for slot in range(c.page_size):
                    j = a * c.page_size + slot
                    if j >= n and fault != 'ignore-tail':
                        continue
                    if j < n and not cache.source.valid[j]:
                        continue
                    if fault != 'ignore-mask' and not allow(i, h, j):
                        continue
                    k, v, position = physical_token(cache, j, kvh, reads)
                    score = sum(qflat[i][h * c.dk + d] * k[d]
                                for d in range(c.dk)) / math.sqrt(c.dk)
                    score += (h + 1) * (i - position) / 32.0
                    mn = max(m, score)
                    alpha = 0.0 if ell == 0.0 else math.exp(m - mn)
                    w = math.exp(score - mn)
                    u = [alpha * prev + w * value for prev, value in zip(u, v)]
                    ell = alpha * ell + w
                    m = mn
            if ell == 0.0:
                raise DomainError('EMPTY_VISIBILITY')
            joined.extend(value / ell for value in u)
        rows.append(tuple(joined))
    if fault == 'first-head-output':
        return tuple(row[:c.dv] for row in rows)
    return mm(tuple(rows), WO)


def dense_reference(source, c, x, allow):
    # Independent full-source scalar reference: no mm, offset, page pool,
    # token reader, registry, guard or online recurrence is used.
    result = []
    for i, xr in enumerate(x):
        joined = []
        for h in range(c.hq):
            kh = c.rho[h]
            q = [math.fsum(xr[a] * WQ[a][h * c.dk + d] for a in range(5))
                 for d in range(c.dk)]
            js = [j for j in range(len(source.e))
                  if source.valid[j] and allow(i, h, j)]
            if not js:
                raise DomainError('reference empty visibility')
            scores = []
            for j in js:
                key = [math.fsum(source.e[j][a] * c.wk[a][kh * c.dk + d]
                                 for a in range(len(source.e[j])))
                       for d in range(c.dk)]
                scores.append(math.fsum(q[d] * key[d] for d in range(c.dk))
                              / math.sqrt(c.dk)
                              + (h + 1) * (i - source.positions[j]) / 32.0)
            maximum = max(scores)
            masses = [math.exp(score - maximum) for score in scores]
            total = math.fsum(masses)
            for d in range(c.dv):
                joined.append(math.fsum(
                    mass / total * math.fsum(
                        source.e[j][a] * c.wv[a][kh * c.dv + d]
                        for a in range(len(source.e[j])))
                    for j, mass in zip(js, masses)))
        result.append(tuple(math.fsum(joined[a] * WO[a][b]
                                      for a in range(c.hq * c.dv))
                            for b in range(7)))
    return tuple(result)


def expect_stale(cache, request, weak=False):
    accesses = []
    try:
        read(cache, request, X, lambda i, h, j: True, reads=accesses, weak=weak)
    except StaleCache:
        require(not accesses, 'wrong cache read before identity rejection')
        return
    raise Mismatch('wrong source cache accepted')


def make_source(n, content='A-v1', version=1, parent=None, lineage='A'):
    e = tuple(((j + 1) / 4.0, ((j * 3) % 7 - 2) / 4.0,
               (-1.0 if j % 2 else 1.0) / 2.0) for j in range(n))
    return Source(lineage, content, version, e,
                  tuple(10 + 3 * j for j in range(n)), (True,) * n, parent)


def check_cow_fixture(fault=None):
    c = Contract()
    r = Registry()
    s0 = make_source(3)
    old = r.build(s0, c)
    before = r.pool.fingerprint(old.table)
    s1 = make_source(5, 'A-v2', 2, s0.key)
    new = r.build(s1, c, fault=fault)
    require(new.mode == 'APPEND' and new.projected == (3, 4),
            '3->5 must project only new rows')
    require(new.table[0] == old.table[0], 'complete prefix page not shared')
    require(new.table[1] != old.table[1], 'tail page not isolated')
    require(r.pool.fingerprint(old.table) == before, 'old physical bytes changed')
    validate_cache(old, s0, c)
    validate_cache(new, s1, c)
    all_visible = lambda i, h, j: True
    compare(read(old, bind(s0, c, 'old-target'), X, all_visible),
            dense_reference(s0, c, X, all_visible), 'old snapshot after append')
    compare(read(new, bind(s1, c, 'new-target'), X, all_visible),
            dense_reference(s1, c, X, all_visible), 'new snapshot after append')
    require(new.copied == (2,), 'copy must retain exactly old valid tail row 2')
    old_tail, new_tail = r.pool.pages[old.table[1]], r.pool.pages[new.table[1]]
    require(new_tail.positions[0] == old_tail.positions[0]
            and new_tail.metadata[0] == old_tail.metadata[0], 'tail metadata not copied')
    g, pid = new.generation, old.table[1]
    events = r.pool.events
    require(events.index(('hold', g, pid)) < events.index(('copy_done', g, pid))
            < events.index(('unhold', g, pid)) < events.index(('project_done', g))
            < events.index(('publish', g)), 'copy lifetime/publication event order')
    require(('copy', g) in new.ready and ('project', g) in new.ready
            and set(old.ready) <= set(new.ready), 'incomplete new ready dependencies')
    require(old_tail.borrow == 0 and r.pool.refs[pid] == 1
            and r.pool.refs[old.table[0]] == 2, 'copy/source snapshot references')
    last = r.pool.pages[new.table[-1]]
    require(last.positions[1] is None and last.metadata[1] is None,
            'unused new final slot was populated')
    require(r.pool.pages[old.table[1]].positions[1] is None,
            'old published empty tail slot was written')
    return r, old, new


def main():
    check_cow_fixture()
    print('OBSERVED P=2 3->5: shared full page; copied physical row 2; projected only 3,4')
    print('OBSERVED both snapshots match dense reference; old bytes unchanged; copy is a ready dependency')
    negative('omit old valid tail slot copy',
             lambda: check_cow_fixture('omit-tail-copy'), 'row 2: missing')
    negative('copy tail encoding but omit position/scale/zero-point',
             lambda: check_cow_fixture('omit-tail-metadata'), 'row 2: missing')
    check_cow_fixture()
    print('OBSERVED correct physical copy restored after both tail negative controls')

    c = Contract()
    registry = Registry()
    old_source = make_source(3)
    old = registry.build(old_source, c)
    new_source = make_source(5, 'A-v2', 2, old_source.key)
    appended = registry.build(new_source, c)
    require(appended.mode == 'APPEND' and appended.projected == (3, 4), 'stable append path')
    require(registry.lookups[-1] == (('A', 1, c.layer), old_source.key),
            'predecessor lookup did not use explicit old identity')
    require(old_source.key != new_source.key, 'new complete source key unchanged')
    validate_cache(appended, new_source, c)

    def wrong_lookup():
        r = Registry()
        r.build(old_source, c)
        bad = r.build(new_source, c, fault='full-key-lookup')
        require(bad.mode == 'APPEND', 'new full key cannot select old predecessor')
    negative('search predecessor with new complete identity', wrong_lookup)

    changed_e = list(make_source(6).e)
    changed_e[0] = (2.0, -1.0, 0.75)
    changed = replace(make_source(6, 'A-v3-changed', 3, new_source.key), e=tuple(changed_e))
    rebuilt = registry.build(changed, c)
    require(rebuilt.mode == 'REBUILD', 'changed actual prefix appended')
    validate_cache(rebuilt, changed, c)
    bad = registry.build(changed, c, fault='ignore-prefix')
    negative('ignore actual changed prefix', lambda: validate_cache(bad, changed, c))
    validate_cache(registry.build(changed, c), changed, c)
    for src in (
            replace(new_source, content='no-parent', version=4, parent=None),
            replace(new_source, content='missing-parent', version=4,
                    parent=replace(old_source, content='absent').key),
            make_source(2, 'shrunk', 3, new_source.key),
            replace(new_source, content='position-change', version=3,
                    positions=tuple(range(20, 25)), parent=new_source.key),
            replace(new_source, content='validity-change', version=3,
                    valid=(True, False, True, True, True), parent=new_source.key)):
        candidate = registry.build(src, c)
        require(candidate.mode == 'REBUILD', 'invalid append condition accepted')
        validate_cache(candidate, src, c)
    require(registry.build(new_source, replace(c, scales='different-scale-contract')).mode
            == 'REBUILD', 'changed representation appended')
    print('OBSERVED actual predecessor lookup, changed prefix and invalid append boundaries covered')

    request_a = bind(new_source, c, 'target-A')
    request_b = bind(new_source, c, 'different-target-B')
    allow = lambda i, h, j: j == 0 or (j + i + h) % 3 != 1
    ref = dense_reference(new_source, c, X, allow)
    compare(read(appended, request_a, X, allow), ref, 'hidden-to-output interface')
    require(shape(ref) == (2, 7) and shape(mm(X, WQ)) == (2, 8), 'non-equal interface widths')
    compare(read(appended, request_b, X[:1], allow),
            dense_reference(new_source, c, X[:1], allow), 'same source different target')
    expect_domain(lambda: read(appended, request_a, mm(X, WQ), allow), 'double query projection')
    for fault in ('first-head-output', 'rho-modulo', 'ignore-mask'):
        negative(fault, lambda fault=fault: compare(
            read(appended, request_a, X, allow, fault=fault), ref, fault))
    compare(read(appended, request_a, X, allow), ref, 'correct multihead restored')
    print('OBSERVED Hq=4 Hkv=2: hidden 5 -> Q 8 -> concatenated 12 -> output 7')

    source_b = replace(new_source, lineage='B', content='different-source-B-v2', parent=None,
                       e=tuple(tuple(value + 0.75 for value in row) for row in new_source.e))
    wrong = registry.build(source_b, c)
    require(wrong.source.version == appended.source.version
            and len(wrong.source.e) == len(appended.source.e)
            and wrong.contract == appended.contract, 'wrong-source fixture geometry/version')
    expect_stale(wrong, request_a)
    negative('weak version/geometry guard accepts another source',
             lambda: expect_stale(wrong, request_a, weak=True))
    negative('wrong source changes numerical output', lambda: compare(
        read(wrong, request_a, X, allow, weak=True), ref, 'wrong source'))
    for src in (
            replace(new_source, content='different-content'),
            replace(new_source, version=9),
            replace(new_source, processor='processor-v2'),
            replace(new_source, encoder='encoder-v2'),
            replace(new_source, positions=tuple(range(30, 35)))):
        expect_stale(appended, bind(src, c, 'target'))
    for cc in (
            replace(c, layer='other-layer'), replace(c, wk=weights(3, 4, 6)),
            replace(c, dtype='other-dtype-label'), replace(c, scales='other-scales'),
            replace(c, rho=(0, 1, 0, 1)), replace(c, layout='HND')):
        expect_stale(appended, bind(new_source, cc, 'target'))
    expect_stale(replace(appended, state='BUILDING'), request_a)
    compare(read(appended, request_b, X, allow), ref, 'sharing restored')
    print('OBSERVED complete mismatches rejected before token reads; distinct targets share same source')

    cases = 0
    for n in (1, 3, 5, 8):
        for p in (1, 2, 4):
            for layout in ('NHD', 'HND'):
                cc = replace(c, page_size=p, layout=layout)
                src = replace(make_source(n, 'layout-' + str(n)),
                              valid=tuple(j % 3 != 1 for j in range(n)))
                r = Registry()
                cache = r.build(src, cc)
                request = bind(src, cc, 'layout-target')
                expected = dense_reference(src, cc, X, allow)
                validate_cache(cache, src, cc)
                compare(read(cache, request, X, allow), expected, 'paged reference')
                count = (n + p - 1) // p
                alternate = r.build(src, cc, permutation=tuple(range(count)))
                compare(read(alternate, request, X, allow), expected, 'page permutation')
                cases += 1
    print('OBSERVED ' + str(cases) + ' length/page/layout cases with physical permutations agree')

    cow_cases = 0
    for nold in (1, 2, 3, 4):
        for p in (1, 2, 4):
            for delta in (0, 1, 2, 5):
                for layout in ('NHD', 'HND'):
                    cc = replace(c, page_size=p, layout=layout)
                    r = Registry()
                    s0 = replace(make_source(nold),
                                 valid=tuple(j % 3 != 1 for j in range(nold)))
                    s1 = replace(make_source(nold + delta, 'growth-v2', 2, s0.key),
                                 valid=tuple(j % 3 != 1 for j in range(nold + delta)))
                    a = r.build(s0, cc)
                    before = r.pool.fingerprint(a.table)
                    b = r.build(s1, cc)
                    require(b.mode == 'APPEND', 'stable physical append rebuilt')
                    require(b.projected == tuple(j for j in range(nold, nold + delta)
                                                  if s1.valid[j]), 'old row reprojected')
                    expected_copy = ()
                    if delta and nold % p:
                        expected_copy = tuple(j for j in range(nold // p * p, nold)
                                              if s0.valid[j])
                        require(a.table[nold // p] != b.table[nold // p], 'shared tail modified')
                    require(b.copied == expected_copy, 'incorrect old tail copy coverage')
                    if delta == 0:
                        require(a.table == b.table, 'no-growth version copied pages')
                    require(r.pool.fingerprint(a.table) == before, 'old snapshot bytes changed')
                    for cache, src in ((a, s0), (b, s1)):
                        validate_cache(cache, src, cc)
                        compare(read(cache, bind(src, cc, 'cow-target'), X, allow),
                                dense_reference(src, cc, X, allow), 'COW snapshot output')
                    cow_cases += 1
    print('OBSERVED ' + str(cow_cases) +
          ' COW cases: no growth, exact boundaries, within-page/multi-page growth and validity holes')

    all_visible = lambda i, h, j: True
    tail_ref = dense_reference(new_source, c, X, all_visible)
    compare(read(appended, request_a, X, all_visible), tail_ref, 'correct final length')
    require(all_visible(0, 0, 5), 'tail mask must permit poison to be exposed')
    negative('ignore final length despite permissive mask', lambda: compare(
        read(appended, request_a, X, all_visible, fault='ignore-tail'), tail_ref, 'tail poison'),
        'row 5: missing')
    compare(read(appended, request_a, X, all_visible), tail_ref, 'correct tail restored')
    expect_domain(lambda: read(appended, request_a, X, lambda i, h, j: False), 'empty visibility')
    require(read(appended, request_a, (), all_visible) == (), 'empty query')
    expect_domain(lambda: Registry().build(make_source(0), c), 'empty source')
    expect_domain(lambda: Registry().build(replace(new_source, valid=(False,) * 5), c),
                  'all invalid source')
    large_x = tuple(tuple(value * 1000.0 for value in row) for row in X)
    compare(read(appended, request_a, large_x, allow),
            dense_reference(new_source, c, large_x, allow), 'large finite scores')
    print('OBSERVED poisoned unused tail, empty source/query/visibility and large finite scores covered')

    ebytes = 1000 * 1024 * 2
    cross = sum(1000 * (1024 + 1024) * 2 for _ in range(12))
    mac = sum(1000 * 1024 * (1024 + 1024) for _ in range(12))
    require(ebytes == 2048000 and cross == 49152000 and mac == 25165824000,
            'capacity or projection MAC arithmetic')
    require(100 * cross / 1024 ** 3 == 4.57763671875, 'conditional repeated traffic')
    require(len(check_cow_fixture()[0].pool.pages) == 4, '3->5 unique physical page capacity')
    print('OBSERVED arithmetic: E=1.953125 MiB; cross=46.875 MiB; build=25165824000 MAC')
    print('ALL CHECKS PASSED: synchronous stdlib float64/affine CPU model; no GPU measurements')


if __name__ == '__main__':
    main()
