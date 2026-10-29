"""Independent byte/dense references and a finite CPU page-lifetime model.

This is author-side validation of declared algorithms, not a GPU allocator,
memory-model proof, benchmark, or assessment of an Agent's construction ability.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
CHECKS: list[dict] = []


def check(name, condition, **details):
    if not condition:
        raise AssertionError((name, details))
    CHECKS.append({'name': name, 'status': 'PASS', **details})


def rejects(name, action, error=AssertionError):
    try:
        action()
    except error:
        check(name, True)
    else:
        raise AssertionError(f'Negative control was not detected: {name}')


def addr(page, u, h, c, page_tokens, heads, dim, stride, layout):
    offset = ((u*heads+h)*dim+c if layout == 'NHD'
              else (h*page_tokens+u)*dim+c)
    return page*stride+2*offset


def dense_values(length, heads, dim, kind):
    # Exactly representable FP16 values; expectation never consults page tables.
    return [[[((t+1)/8 + h/32 + c/128) * (1 if kind == 'K' else -1)
              for c in range(dim)] for h in range(heads)] for t in range(length)]


def lay_out(dense, table, page_tokens, heads, dim, stride, layout):
    """Writer enumerates the physical axis order, independent of addr()."""
    buffer = bytearray(b'\xff' * ((max(table, default=-1)+1)*stride))
    for logical_page, physical in enumerate(table):
        axes = ([(u, h, c) for u in range(page_tokens) for h in range(heads)
                 for c in range(dim)] if layout == 'NHD' else
                [(u, h, c) for h in range(heads) for u in range(page_tokens)
                 for c in range(dim)])
        for linear, (u, h, c) in enumerate(axes):
            t = logical_page*page_tokens+u
            if t < len(dense):
                struct.pack_into('<e', buffer, physical*stride+2*linear, dense[t][h][c])
    return buffer


def gather(buffer, table, length, page_tokens, heads, dim, stride, layout):
    return [[[struct.unpack_from('<e', buffer, addr(table[t//page_tokens],
              t % page_tokens, h, c, page_tokens, heads, dim, stride, layout))[0]
              for c in range(dim)] for h in range(heads)] for t in range(length)]


def attention(keys, values, query, kv_head):
    if not keys:
        return [0.0]*len(values[0][0]) if values else []
    scores = [math.fsum(q*k for q, k in zip(query, token[kv_head])) for token in keys]
    m = max(scores)
    weights = [math.exp(s-m) for s in scores]
    den = math.fsum(weights)
    return [math.fsum(w*token[kv_head][c] for w, token in zip(weights, values))/den
            for c in range(len(values[0][0]))]


def byte_checks():
    cases = 0
    for p in (1, 3, 4, 8):
        for heads, dk, dv in ((1, 1, 3), (2, 4, 4), (3, 5, 2)):
            for length in sorted({0, 1, max(0, p-1), p, p+1, 2*p+1}):
                table = [5, 1, 4][:math.ceil(length/p)]
                for layout in ('NHD', 'HND'):
                    k = dense_values(length, heads, dk, 'K')
                    v = dense_values(length, heads, dv, 'V')
                    sk = math.ceil(p*heads*dk*2/32)*32
                    sv = math.ceil(p*heads*dv*2/32)*32
                    kb = lay_out(k, table, p, heads, dk, sk, layout)
                    vb = lay_out(v, table, p, heads, dv, sv, layout)
                    gotk = gather(kb, table, length, p, heads, dk, sk, layout)
                    gotv = gather(vb, table, length, p, heads, dv, sv, layout)
                    assert gotk == k and gotv == v
                    locations = [addr(page, u, h, c, p, heads, dk, sk, layout)
                                 for page in table for u in range(p)
                                 for h in range(heads) for c in range(dk)]
                    assert len(locations) == len(set(locations))
                    if length:
                        for qh in range(2*heads):
                            query = [(c+1)/dk for c in range(dk)]
                            expected = attention(k, v, query, qh//2)
                            observed = attention(gotk, gotv, query, qh//2)
                            assert observed == expected
                    cases += 1
    check('byte layouts reconstruct independent logical tensors and attention', True,
          cases=cases, lengths='empty/full/tail/multiple pages', d_k_may_differ_from_d_v=True)
    check('published NHD K/V example addresses',
          4096+addr(1, 1, 1, 2, 4, 2, 4, 64, 'NHD') == 4188 and
          8192+addr(1, 1, 1, 2, 4, 2, 4, 64, 'NHD') == 8284)
    check('published HND K/V example addresses',
          4096+addr(1, 1, 1, 2, 4, 2, 4, 64, 'HND') == 4204 and
          8192+addr(1, 1, 1, 2, 4, 2, 4, 64, 'HND') == 8300)
    k = dense_values(6, 2, 4, 'K')
    kb = lay_out(k, [3, 1], 4, 2, 4, 64, 'NHD')
    check('changing descriptor alone is a detected wrong layout',
          gather(kb, [3, 1], 6, 4, 2, 4, 64, 'HND') != k)
    poison = gather(kb, [3, 1], 8, 4, 2, 4, 64, 'NHD')
    check('reading page capacity instead of valid length reaches poison',
          any(math.isnan(x) for token in poison[6:] for head in token for x in head))
    check('application pages are not byte-granularity virtual pages',
          4*128*1024 != 64*1024)
    return cases


@dataclass
class Page:
    generation: int = 0
    data: tuple[int, ...] | None = None
    owners: set[str] = field(default_factory=set)
    readers: set[str] = field(default_factory=set)
    writer: str | None = None

    def free(self):
        return not (self.owners or self.readers or self.writer)


@dataclass(frozen=True)
class Snapshot:
    request: str
    request_generation: int
    sequence: int
    length: int
    pages: tuple[tuple[int, int], ...]


@dataclass
class Request:
    identity: str
    generation: int
    current: Snapshot
    logical: tuple[int, ...]
    active: bool = True
    pending: str | None = None


@dataclass
class Reader:
    ticket: str
    packet: str
    snapshot: Snapshot
    expected: tuple[int, ...]
    done: bool = False


@dataclass
class Transaction:
    identity: str
    request: Request
    base: Snapshot
    appended: tuple[int, ...]
    new_pages: tuple[tuple[int, int], ...]
    source: tuple[int, int] | None
    ready: bool = False
    abandoned: bool = False
    done: bool = False


class Pool:
    """Serialized host transitions; async copies/completions are explicit events."""
    def __init__(self, count=6, page_tokens=4, max_length=64):
        self.pages = [Page() for _ in range(count)]
        self.p = page_tokens
        self.max_length = max_length
        self.requests: dict[str, Request] = {}
        self.readers: dict[str, Reader] = {}
        self.transactions: dict[str, Transaction] = {}
        self.packets: dict[str, str] = {}
        self.counter = 0

    def handle(self, item):
        p, g = item
        assert self.pages[p].generation == g, 'stale physical page handle'
        return self.pages[p]

    def collect(self):
        for page in self.pages:
            if page.free():
                page.data = None

    def live(self):
        return sum(not p.free() for p in self.pages)

    def new_request(self, identity, tokens=(), physical=None):
        assert identity not in self.requests or not self.requests[identity].active
        generation = self.requests[identity].generation+1 if identity in self.requests else 1
        tokens = tuple(tokens)
        n = math.ceil(len(tokens)/self.p)
        ids = physical if physical is not None else [i for i, x in enumerate(self.pages) if x.free()][:n]
        assert len(ids) == n and all(self.pages[i].free() for i in ids)
        handles = []
        for j, i in enumerate(ids):
            page = self.pages[i]
            page.generation += 1
            page.data = tokens[j*self.p:(j+1)*self.p]
            page.owners.add(identity)
            handles.append((i, page.generation))
        snap = Snapshot(identity, generation, 0, len(tokens), tuple(handles))
        req = Request(identity, generation, snap, tokens)
        self.requests[identity] = req
        return req

    def share(self, identity, source):
        assert identity not in self.requests
        snap = replace(source.current, request=identity, request_generation=1)
        req = Request(identity, 1, snap, source.logical)
        self.requests[identity] = req
        for item in snap.pages:
            self.handle(item).owners.add(identity)
        return req

    def materialize(self, snap):
        assert len(snap.pages) == math.ceil(snap.length/self.p)
        tokens = []
        for j, item in enumerate(snap.pages):
            page = self.handle(item)
            assert page.data is not None
            take = min(self.p, snap.length-j*self.p)
            assert len(page.data) >= take, 'published length exceeds initialized data'
            tokens.extend(page.data[:take])
        return tuple(tokens)

    def read(self, req, packet):
        assert req.active and self.requests[req.identity] is req
        assert packet not in self.packets, 'overwriting an active graph descriptor'
        self.counter += 1
        ticket = f'reader{self.counter}'
        for item in req.current.pages:
            page = self.handle(item)
            assert page.writer is None and page.data is not None
        reader = Reader(ticket, packet, req.current, req.logical)
        for item in req.current.pages:
            self.handle(item).readers.add(ticket)
        self.readers[ticket] = reader
        self.packets[packet] = ticket
        return reader

    def finish(self, reader):
        if reader.done:
            return False
        assert self.packets[reader.packet] == reader.ticket
        assert self.materialize(reader.snapshot) == reader.expected
        for item in reader.snapshot.pages:
            self.handle(item).readers.remove(reader.ticket)
        del self.packets[reader.packet]
        reader.done = True
        self.collect()
        return True

    def prepare(self, req, appended):
        appended = tuple(appended)
        assert req.active and self.requests[req.identity] is req and req.pending is None
        assert appended and req.current.length+len(appended) <= self.max_length
        touched = sorted({t//self.p for t in range(req.current.length,
                                                  req.current.length+len(appended))})
        free = [i for i, page in enumerate(self.pages) if page.free()]
        if len(free) < len(touched):
            raise MemoryError('no complete reservation; nothing was written')
        self.counter += 1
        identity = f'tx{self.counter}'
        new = []
        for i in free[:len(touched)]:
            page = self.pages[i]
            page.generation += 1
            page.writer = identity
            page.data = None
            new.append((i, page.generation))
        source = req.current.pages[-1] if req.current.length % self.p else None
        if source:
            self.handle(source).readers.add(identity)
        tx = Transaction(identity, req, req.current, appended, tuple(new), source)
        req.pending = identity
        self.transactions[identity] = tx
        return tx

    def complete_copy(self, tx):
        assert not tx.done and not tx.ready
        payload = (self.handle(tx.source).data[:tx.base.length % self.p] if tx.source else ())
        payload += tx.appended
        for j, item in enumerate(tx.new_pages):
            page = self.handle(item)
            assert page.writer == tx.identity
            page.data = payload[j*self.p:(j+1)*self.p]
        tx.ready = True
        if tx.abandoned:
            self._release(tx)

    def _release(self, tx):
        assert tx.ready and not tx.done
        for item in tx.new_pages:
            self.handle(item).writer = None
        if tx.source:
            self.handle(tx.source).readers.remove(tx.identity)
        if tx.request.pending == tx.identity:
            tx.request.pending = None
        tx.done = True
        self.collect()

    def commit(self, tx):
        req = tx.request
        assert tx.ready and not tx.done and not tx.abandoned, 'publish before data ready'
        assert req.active and self.requests[req.identity] is req and req.current == tx.base, 'stale request/snapshot'
        cut = tx.base.length//self.p
        pages = tx.base.pages[:cut]+tx.new_pages
        new_snap = Snapshot(req.identity, req.generation, tx.base.sequence+1,
                            tx.base.length+len(tx.appended), pages)
        for item in tx.base.pages[cut:]:
            self.handle(item).owners.remove(req.identity)
        for item in tx.new_pages:
            self.handle(item).owners.add(req.identity)
        req.current = new_snap
        req.logical = req.logical+tx.appended
        self._release(tx)

    def abort(self, tx):
        assert not tx.done
        tx.abandoned = True
        if tx.ready:
            self._release(tx)

    def cancel(self, req):
        assert req.active
        req.active = False
        for item in req.current.pages:
            self.handle(item).owners.remove(req.identity)
        self.collect()

    def invariants(self):
        owners = [set() for _ in self.pages]
        readers = [set() for _ in self.pages]
        writers = [None for _ in self.pages]
        for req in self.requests.values():
            if req.active:
                assert self.materialize(req.current) == req.logical
                for i, g in req.current.pages:
                    assert self.pages[i].generation == g
                    owners[i].add(req.identity)
        for reader in self.readers.values():
            if not reader.done:
                assert self.materialize(reader.snapshot) == reader.expected
                for i, g in reader.snapshot.pages:
                    assert self.pages[i].generation == g
                    readers[i].add(reader.ticket)
                assert self.packets[reader.packet] == reader.ticket
        for tx in self.transactions.values():
            if not tx.done:
                if tx.source:
                    i, g = tx.source
                    assert self.pages[i].generation == g
                    readers[i].add(tx.identity)
                for i, g in tx.new_pages:
                    assert self.pages[i].generation == g
                    writers[i] = tx.identity
        for i, page in enumerate(self.pages):
            assert page.owners == owners[i] and page.readers == readers[i] and page.writer == writers[i]
            assert not page.free() or page.data is None


def base_pool():
    pool = Pool()
    a = pool.new_request('A', range(6), [3, 1])
    b = pool.share('B', a)
    return pool, a, b


def lifetime_checks():
    for order in permutations(('cancel_a', 'cancel_b', 'finish_old', 'finish_new')):
        pool, a, b = base_pool()
        old = pool.read(a, 'G0')
        tx = pool.prepare(a, [6])
        pool.invariants()
        rejects('unready publication rejected '+str(order), lambda: pool.commit(tx))
        pool.complete_copy(tx)
        assert pool.live() == 3
        pool.commit(tx)
        assert pool.materialize(old.snapshot) == tuple(range(6))
        new = pool.read(a, 'G1')
        assert pool.materialize(new.snapshot) == tuple(range(7))
        actions = {'cancel_a': lambda: pool.cancel(a), 'cancel_b': lambda: pool.cancel(b),
                   'finish_old': lambda: pool.finish(old), 'finish_new': lambda: pool.finish(new)}
        for event in order:
            actions[event]()
            pool.invariants()
        assert pool.live() == 0 and not pool.packets
    check('all cancellation/completion orders after two legal snapshot launches', True, permutations=24)

    pool, a, b = base_pool()
    old = pool.read(a, 'G0')
    tx = pool.prepare(a, [6])
    pool.complete_copy(tx)
    check('ready but unpublished pages retain writer reservation',
          all(pool.handle(item).writer == tx.identity for item in tx.new_pages))
    pool.commit(tx)
    new = pool.read(a, 'G1')
    rejects('busy descriptor cannot be overwritten', lambda: pool.read(a, 'G0'))
    pool.cancel(b)
    check('cancel B leaves old tail pinned', len(pool.pages[1].owners) == 0 and bool(pool.pages[1].readers))
    pool.cancel(a)
    check('all mappings removed yet three pages remain in flight', pool.live() == 3)
    pool.finish(new)
    check('new reader may end first without freeing shared prefix', pool.live() == 2)
    c = pool.new_request('C', [99], [new.snapshot.pages[-1][0]])
    c_reader = pool.read(c, 'G1')
    before = (c.current, set(pool.handle(c.current.pages[0]).readers))
    check('duplicate completion is rejected and does not affect new resource generation',
          pool.finish(new) is False and before == (c.current, set(pool.handle(c.current.pages[0]).readers)))
    rejects('old physical handle detects page reuse', lambda: pool.handle(new.snapshot.pages[-1]))
    pool.finish(old)
    pool.finish(c_reader)
    pool.cancel(c)
    pool.invariants()
    check('full trajectory reclaims all resources exactly once', pool.live() == 0)

    small = Pool(count=2)
    r = small.new_request('R', range(6), [0, 1])
    before = (r.current, tuple((x.generation, x.data) for x in small.pages))
    rejects('insufficient complete reservation is atomic', lambda: small.prepare(r, [6]), MemoryError)
    check('capacity failure preserves bytes and snapshot', before == (r.current, tuple((x.generation, x.data) for x in small.pages)))

    for cancel_first in (False, True):
        pool, a, b = base_pool()
        tx = pool.prepare(a, [6, 7, 8, 9, 10])
        base = a.current
        if cancel_first:
            pool.cancel(a)
        pool.abort(tx)
        pool.invariants()
        check('aborted pending copy retains writer reservation '+str(cancel_first),
              all(pool.handle(item).writer == tx.identity for item in tx.new_pages))
        pool.complete_copy(tx)
        pool.invariants()
        check('abort releases only after copy completion '+str(cancel_first), tx.done and a.current == base)
        if not cancel_first:
            assert pool.materialize(a.current) == tuple(range(6))

    pool, a, b = base_pool()
    tx = pool.prepare(a, [6])
    pool.complete_copy(tx)
    pool.cancel(a)
    replacement = pool.new_request('A')
    rejects('request-slot reuse prevents old transaction publication', lambda: pool.commit(tx))
    pool.abort(tx)
    pool.invariants()
    check('old transaction cannot change replacement request', replacement.current.length == 0 and replacement.pending is None)

    for p in (1, 3, 4, 8):
        for length in range(0, 2*p+1):
            for q in (1, 2, p+1):
                pool = Pool(count=20, page_tokens=p)
                a = pool.new_request('A', range(length))
                old = pool.read(a, 'old')
                tx = pool.prepare(a, range(length, length+q))
                enumerated = len({t//p for t in range(length, length+q)})
                assert len(tx.new_pages) == enumerated == math.ceil((length+q)/p)-length//p
                pool.complete_copy(tx)
                pool.commit(tx)
                pool.invariants()
                assert pool.materialize(a.current) == tuple(range(length+q))
                assert pool.materialize(old.snapshot) == tuple(range(length))
    check('empty/full/tail and multi-page append preserve old and new dense states', True)

    bad, a, b = base_pool()
    a.current = replace(a.current, length=7)
    rejects('length-only publication is detected by independent materialization', bad.invariants)
    bad, a, b = base_pool()
    old = bad.read(a, 'G0')
    bad.cancel(a)
    bad.cancel(b)
    bad.pages[1].data = None  # Deliberate broken early reuse, while reader is pending.
    rejects('owner-zero early reclamation breaks a still-live snapshot', bad.invariants)
    return 24


def cost_checks():
    tau = 32*8*(128+128)*2
    check('all-layer bytes per token and original capacity example',
          tau == 128*1024 and math.ceil(18/16)*16*tau == 4*1024**2 and
          18*tau == 2.25*1024**2)
    lengths = (5, 6, 9)
    terms = {}
    for p in (4, 8):
        page_count = sum(math.ceil(length/p) for length in lengths)
        old_tail_tokens = sum(length % p for length in lengths)
        occupied = page_count*p*tau
        temporary = 3*p*tau
        traffic = (2*old_tail_tokens+3)*tau
        terms[p] = {'pages': page_count, 'occupied': occupied, 'peak': occupied+temporary,
                    'traffic': traffic, 'metadata': page_count*8+4*4+3*8}
    check('page size capacity/metadata comparison', terms[4] == {
        'pages': 7, 'occupied': 3670016, 'peak': 5242880, 'traffic': 1441792, 'metadata': 96}
        and terms[8] == {
        'pages': 4, 'occupied': 4194304, 'peak': 7340032, 'traffic': 3538944, 'metadata': 72})
    beta = 1024**4
    times = {alpha: {p: t['pages']*alpha+t['traffic']/beta*1e6 for p, t in terms.items()}
             for alpha in (.8, .1)}
    check('conditional cost ranking reverses', times[.8][8] < times[.8][4] and times[.1][4] < times[.1][8],
          microseconds=times)
    check('capacity legality precedes latency ranking', terms[4]['peak'] <= 6*1024**2 < terms[8]['peak'])
    # Separately rounded regions are generally larger than rounding one aggregate.
    regions = (33, 33, 33, 33)
    granularity = 64
    check('virtual backing rounds each region independently',
          sum(math.ceil(x/granularity)*granularity for x in regions) == 256 and
          math.ceil(sum(regions)/granularity)*granularity == 192)
    return terms, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    layouts = byte_checks()
    orders = lifetime_checks()
    terms, times = cost_checks()
    inputs = ['knowledge/algorithms/paged-kv-capacity.md', 'knowledge/sources/S-PAGED.md',
              'knowledge/sources/S-VATTENTION.md']
    report = {'status': 'PASS', 'checked_at': '2026-10-28', 'scope': 'CPU byte addresses, independent dense tensors, finite serialized-host lifecycle events, and synthetic cost arithmetic',
              'checks': CHECKS, 'check_count': len(CHECKS), 'byte_layout_cases': layouts,
              'declared_event_permutations': orders,
              'capacity_terms': terms, 'synthetic_times_microseconds': times,
              'inputs': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in inputs},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limits': ['No GPU kernel, allocator, graph or transport was executed.',
                         'Finite event permutations assume atomic serialized host metadata transitions; they are not a proof of all concurrent executions or device memory visibility.',
                         'Scalar attention only checks representation equivalence with fixed arithmetic, not hardware matrix rounding.',
                         'Cost inputs are declared synthetic assumptions, not measured device timings or a general optimum.',
                         'No controlled Agent construction, transfer, diagnosis or optimization effect was assessed.']}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'checks': len(CHECKS), 'byte_layout_cases': layouts,
                      'event_permutations': orders}, ensure_ascii=False))


if __name__ == '__main__':
    main()
