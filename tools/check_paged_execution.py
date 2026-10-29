"""CPU references for paged-to-shared loading and mixed in-place/COW append.

These are declared scalar-address/serialized-host models, not GPU kernels,
allocator concurrency proofs, benchmarks or controlled Agent-effect evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
CHECKS = []


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
        raise AssertionError(name)


def logical_token(t, c, value=False):
    return (t+1+c/16) * (-1 if value else 1)


def physical_pool(length, table, value=False):
    """Independent writer enumerates each physical page's token/channel axes."""
    pool = bytearray(b'\xff' * ((max(table, default=-1)+1)*96))
    for logical_page, physical in enumerate(table):
        page_values = []
        for slot in range(3):
            t = logical_page*3+slot
            for c in range(16):
                page_values.append(logical_token(t, c, value) if t < length else float('nan'))
        page_bytes = struct.pack('<48e', *page_values)
        pool[physical*96:(physical+1)*96] = page_bytes
    return pool


def paged_tile(pool, table, length, start):
    shared = bytearray(512)
    requests = []
    metadata = []
    banks = []
    for v in range(4):
        row_handles = {}
        for leader in (0, 8, 16, 24):
            a = (leader+32*v)//8
            t = start+a
            if t < length:
                j = t//3
                row_handles[a] = table[j]
                metadata.append(j)
            else:
                row_handles[a] = None
        iteration_banks = []
        for lane in range(32):
            w = lane+32*v
            a, c = w//8, 2*(w % 8)
            t = start+a
            if t < length:
                source = 96*row_handles[a]+32*(t % 3)+2*c
                assert source % 4 == 0
                word = struct.unpack_from('<I', pool, source)[0]
                requests.append((v, lane, source))
            else:
                word = 0
            struct.pack_into('<I', shared, 4*w, word)
            iteration_banks.append(w % 32)
        banks.append(iteration_banks)
    got = struct.unpack('<256e', shared)
    return [list(got[a*16:(a+1)*16]) for a in range(16)], requests, metadata, banks


def mapping_checks():
    cases = 0
    for length in (0, 1, 2, 3, 4, 15, 16, 17, 19, 33):
        count = (length+2)//3
        table = [(5*j+7) % 97 for j in range(count)]
        for start in (0, 1, 2, 3, 15, 16, 31):
            for value in (False, True):
                pool = physical_pool(length, table, value)
                got, requests, metadata, banks = paged_tile(pool, table, length, start)
                reference = [[logical_token(start+a, c, value) if start+a < length else 0.0
                              for c in range(16)] for a in range(16)]
                assert got == reference
                valid = max(0, min(16, length-start))
                assert len(requests) == valid*8 and len(metadata) == valid
                assert sum(len({source//32 for v0, _, source in requests if v0 == v})
                           for v in range(4)) == valid
                assert all(sorted(group) == list(range(32)) for group in banks)
                assert len({(v*32+lane) for v, lane, _ in requests}) == len(requests)
                cases += 1
    check('paged warp copies reconstruct independent logical K/V tiles', True, cases=cases)
    check('empty and tile-tail rows do not query page table or global data', True,
          cases_include='L=0 and tile starts beyond L')
    table = [7, 1, 5, 0, 6, 3]
    pool = physical_pool(17, table)
    got, requests, metadata, banks = paged_tile(pool, table, 17, 2)
    row_starts = {((lane+32*v)//8): source for v, lane, source in requests
                  if lane % 8 == 0}
    expected_offsets = [736, 96, 128, 160, 480, 512, 544, 0, 32, 64,
                        576, 608, 640, 288, 320]
    check('published non-contiguous row addresses', list(row_starts.values()) == expected_offsets)
    check('published byte and metadata request accounting',
          len(requests)*4*2 == 960 and len(metadata)*8 == 120 and len(set(metadata)) == 6)
    check('four issue rounds each write one word per shared bank',
          all(group == list(range(32)) for group in banks))
    rejects('contiguous-address assumption can leave the physical pool',
            lambda: struct.unpack_from('<16e', pool, expected_offsets[0]+32), struct.error)
    next_tile, _, _, _ = paged_tile(pool, table, 17, 3)
    bad = [list(struct.unpack_from('<16e', pool, 96+32*a)) for a in range(4)]
    check('contiguous-address assumption can also read different content', bad != next_tile[:4])
    check('tight odd dimensions invalidate the declared 32-bit row loads',
          (30 % 4 != 0) and ((90+30) % 4 == 0) and ((90+60) % 4 != 0))
    return cases


@dataclass
class Page:
    generation: int = 0
    data: list | None = None
    owners: set = field(default_factory=set)
    readers: dict = field(default_factory=dict)
    writer: str | None = None

    def free(self):
        return not (self.owners or self.readers or self.writer)


@dataclass(frozen=True)
class Snapshot:
    request: str
    generation: int
    length: int
    pages: tuple


@dataclass
class Request:
    name: str
    generation: int
    snapshot: Snapshot
    logical: tuple
    active: bool = True
    pending: str | None = None


@dataclass
class Reader:
    name: str
    snapshot: Snapshot
    expected: tuple
    actual_bytes: dict
    done: bool = False


@dataclass
class Transaction:
    name: str
    request: Request
    base: Snapshot
    appended: tuple
    targets: tuple
    allocated: tuple
    inplace: bool
    source: tuple | None
    writer_pages: tuple
    traffic: int
    ready: bool = False
    abandoned: bool = False
    done: bool = False


class Pool:
    """Each method is a serialized host transition; async writes are explicit."""
    def __init__(self, page_tokens=4, count=64):
        self.p = page_tokens
        self.pages = [Page() for _ in range(count)]
        self.requests = {}
        self.readers = []
        self.transactions = []
        self.sequence = 0

    def handle(self, item):
        index, generation = item
        assert self.pages[index].generation == generation
        return self.pages[index]

    def collect(self):
        for page in self.pages:
            if page.free():
                page.data = None

    def live(self):
        return sum(not page.free() for page in self.pages)

    def reserve(self, count, writer):
        free = [i for i, page in enumerate(self.pages) if page.free()]
        if len(free) < count:
            raise MemoryError('complete reservation unavailable')
        handles = []
        for index in free[:count]:
            page = self.pages[index]
            page.generation += 1
            page.writer = writer
            page.data = [None]*self.p
            handles.append((index, page.generation))
        return tuple(handles)

    def create(self, name, tokens):
        assert name not in self.requests or not self.requests[name].active
        generation = self.requests[name].generation+1 if name in self.requests else 1
        tokens = tuple(tokens)
        pages = self.reserve((len(tokens)+self.p-1)//self.p, 'initial')
        for j, item in enumerate(pages):
            page = self.handle(item)
            for u, value in enumerate(tokens[j*self.p:(j+1)*self.p]):
                page.data[u] = value
            page.owners.add((name, generation))
            page.writer = None
        snap = Snapshot(name, generation, len(tokens), pages)
        req = Request(name, generation, snap, tokens)
        self.requests[name] = req
        return req

    def share(self, name, req):
        assert req.active and req.pending is None and name not in self.requests
        snap = Snapshot(name, 1, req.snapshot.length, req.snapshot.pages)
        other = Request(name, 1, snap, req.logical)
        for item in snap.pages:
            page = self.handle(item)
            assert page.writer is None
            page.owners.add((name, 1))
        self.requests[name] = other
        return other

    def materialize(self, snap):
        assert len(snap.pages) == (snap.length+self.p-1)//self.p
        values = []
        for j, item in enumerate(snap.pages):
            page = self.handle(item)
            assert page.data is not None
            values.extend(page.data[:min(self.p, snap.length-j*self.p)])
        return tuple(values)

    def read(self, req, wide=False):
        assert req.active and req.pending is None
        name = f'R{len(self.readers)}'
        actual = {}
        for j, item in enumerate(req.snapshot.pages):
            page = self.handle(item)
            extent = self.p if wide else min(self.p, req.snapshot.length-j*self.p)
            indices = set(range(extent))
            page.readers[name] = indices
            for index in indices:
                actual[(item, index)] = page.data[index]
        reader = Reader(name, req.snapshot, req.logical, actual)
        self.readers.append(reader)
        return reader

    def finish(self, reader):
        if reader.done:
            return False
        self.check_reader(reader)
        for item in reader.snapshot.pages:
            del self.handle(item).readers[reader.name]
        reader.done = True
        self.collect()
        return True

    def check_reader(self, reader):
        assert self.materialize(reader.snapshot) == reader.expected
        for (item, index), value in reader.actual_bytes.items():
            assert self.handle(item).data[index] == value, 'actual read/write regions overlap'

    def prepare(self, req, appended, mode='mixed', force_inplace=False):
        assert req.active and req.pending is None and self.requests[req.name] is req
        appended = tuple(appended)
        assert appended
        base = req.snapshot
        r = base.length % self.p
        old = self.handle(base.pages[-1]) if r else None
        write_range = set(range(r, min(self.p, r+len(appended))))
        eligible = bool(old and len(old.owners) == 1 and old.writer is None and
                        all(not indices & write_range for indices in old.readers.values()))
        inplace = bool(r and mode == 'mixed' and (eligible or force_inplace))
        logical_new = (base.length+len(appended)+self.p-1)//self.p-len(base.pages)
        needed = logical_new+(1 if r and not inplace else 0)
        self.sequence += 1
        name = f'W{self.sequence}'
        allocated = self.reserve(needed, name)
        cut = base.length//self.p
        targets = base.pages+allocated if inplace else base.pages[:cut]+allocated
        writer_pages = ((base.pages[-1],) if inplace else ())+allocated
        if inplace:
            assert old.writer is None
            old.writer = name
        source = base.pages[-1] if r and not inplace else None
        if source:
            self.handle(source).readers[name] = set(range(r))
        traffic = len(appended)+(2*r if source else 0)
        tx = Transaction(name, req, base, appended, targets, allocated, inplace,
                         source, writer_pages, traffic)
        req.pending = name
        self.transactions.append(tx)
        return tx

    def complete_writes(self, tx):
        assert not tx.done and not tx.ready
        if tx.source:
            r = tx.base.length % self.p
            target = self.handle(tx.allocated[0])
            target.data[:r] = self.handle(tx.source).data[:r]
        for offset, value in enumerate(tx.appended):
            t = tx.base.length+offset
            self.handle(tx.targets[t//self.p]).data[t % self.p] = value
        tx.ready = True
        if tx.abandoned:
            self.release(tx)

    def release(self, tx):
        assert tx.ready and not tx.done
        for item in tx.writer_pages:
            page = self.handle(item)
            assert page.writer == tx.name
            page.writer = None
        if tx.source:
            del self.handle(tx.source).readers[tx.name]
        if tx.request.pending == tx.name:
            tx.request.pending = None
        tx.done = True
        self.collect()

    def commit(self, tx):
        req = tx.request
        assert tx.ready and not tx.done and not tx.abandoned
        assert req.active and self.requests[req.name] is req and req.snapshot == tx.base
        owner = (req.name, req.generation)
        old_set, new_set = set(tx.base.pages), set(tx.targets)
        for item in old_set-new_set:
            self.handle(item).owners.remove(owner)
        for item in new_set-old_set:
            self.handle(item).owners.add(owner)
        req.logical += tx.appended
        req.snapshot = Snapshot(req.name, req.generation, len(req.logical), tx.targets)
        self.release(tx)

    def abort(self, tx):
        assert not tx.done
        tx.abandoned = True
        if tx.ready:
            self.release(tx)

    def cancel(self, req):
        assert req.active
        req.active = False
        for item in req.snapshot.pages:
            self.handle(item).owners.remove((req.name, req.generation))
        self.collect()

    def invariants(self):
        for req in self.requests.values():
            if req.active:
                assert self.materialize(req.snapshot) == req.logical
                for item in req.snapshot.pages:
                    assert (req.name, req.generation) in self.handle(item).owners
        for reader in self.readers:
            if not reader.done:
                self.check_reader(reader)
        for tx in self.transactions:
            if not tx.done:
                assert all(self.handle(item).writer == tx.name for item in tx.writer_pages)
        for page in self.pages:
            assert not page.free() or page.data is None


def append(pool, req, tokens, mode):
    tx = pool.prepare(req, tokens, mode)
    pool.invariants()
    peak = pool.live()
    pool.complete_writes(tx)
    pool.invariants()
    pool.commit(tx)
    pool.invariants()
    return tx.traffic, peak


def transition_checks():
    cases = 0
    for p in (1, 3, 4, 8):
        for length in range(2*p+1):
            for q in (1, 2, p+1):
                for wide in (False, True):
                    for shared in (False, True):
                        for mode in ('immutable', 'mixed'):
                            pool = Pool(p)
                            req = pool.create('A', range(length))
                            reader = pool.read(req, wide)
                            other = pool.share('B', req) if shared else None
                            tx = pool.prepare(req, range(length, length+q), mode)
                            r = length % p
                            expected_inplace = bool(mode == 'mixed' and r and not wide and not shared)
                            assert tx.inplace == expected_inplace
                            pool.complete_writes(tx)
                            pool.invariants()
                            pool.commit(tx)
                            assert pool.materialize(req.snapshot) == tuple(range(length+q))
                            assert pool.materialize(reader.snapshot) == tuple(range(length))
                            if other:
                                assert pool.materialize(other.snapshot) == tuple(range(length))
                            pool.finish(reader)
                            pool.cancel(req)
                            if other:
                                pool.cancel(other)
                            pool.invariants()
                            assert pool.live() == 0
                            cases += 1
    check('both append schemes preserve independent old and new logical states', True, cases=cases)

    pool = Pool()
    req = pool.create('A', [0])
    old = pool.read(req)
    tx = pool.prepare(req, [1, 2])
    rejects('new consumer waits while an append transaction owns the request', lambda: pool.read(req))
    rejects('sharing cannot race an acquired in-place write lease', lambda: pool.share('B', req))
    pool.complete_writes(tx)
    pool.abort(tx)
    pool.invariants()
    check('abort leaves published prefix intact despite uncommitted suffix bytes',
          req.snapshot.length == 1 and pool.materialize(old.snapshot) == (0,))
    append(pool, req, [9], 'mixed')
    check('retry overwrites every newly published position', pool.materialize(req.snapshot) == (0, 9))

    for ready in (False, True):
        pool = Pool()
        req = pool.create('A', [0])
        tx = pool.prepare(req, [1])
        if ready:
            pool.complete_writes(tx)
        pool.cancel(req)
        check('cancelled in-place target retains W until abort '+str(ready), pool.live() == 1)
        rejects('cancelled writer-owned page cannot be assigned as a free page '+str(ready),
                lambda: assert_free(pool.pages[0]))
        pool.abort(tx)
        if not ready:
            assert pool.live() == 1
            pool.complete_writes(tx)
        pool.invariants()
        check('aborted target is reclaimable only after final writes '+str(ready), pool.live() == 0)

    pool = Pool(count=1)
    req = pool.create('A', [0])
    old = pool.read(req, wide=True)
    before = (req.snapshot, list(pool.pages[0].data), pool.pages[0].writer)
    rejects('COW capacity shortage changes no live state', lambda: pool.prepare(req, [1]), MemoryError)
    check('capacity failure preserves bytes, descriptor and writer ownership',
          before == (req.snapshot, list(pool.pages[0].data), pool.pages[0].writer))
    pool.finish(old)
    tx = pool.prepare(req, [1])
    check('draining a conflicting reader makes in-place candidate feasible', tx.inplace and not tx.allocated)
    pool.complete_writes(tx)
    pool.commit(tx)

    bad = Pool()
    req = bad.create('A', [0])
    bad.read(req, wide=True)
    tx = bad.prepare(req, [1], force_inplace=True)
    bad.complete_writes(tx)
    rejects('ignoring the actual wide read range exposes a write/read conflict', bad.invariants)
    check('shared tail selects the declared COW branch', not shared_tail_inplace_selected())
    pool = Pool()
    req = pool.create('A', [0])
    tx = pool.prepare(req, [1])
    pool.cancel(req)
    replacement = pool.create('A', [])
    pool.complete_writes(tx)
    rejects('old in-place transaction cannot publish to a reused request name', lambda: pool.commit(tx))
    pool.abort(tx)
    pool.invariants()
    check('old abort leaves the replacement request and its pending state intact',
          replacement.snapshot.length == 0 and replacement.pending is None)
    return cases


def assert_free(page):
    assert page.free()


def shared_tail_inplace_selected():
    pool = Pool()
    req = pool.create('A', [0, 1])
    pool.share('B', req)
    return pool.prepare(req, [2]).inplace


def cost_checks():
    comparison = {}
    for mode in ('immutable', 'mixed'):
        pool = Pool()
        req = pool.create('A', [0])
        pool.read(req)
        traffic1, peak1 = append(pool, req, [1], mode)
        pool.read(req)
        traffic2, peak2 = append(pool, req, [2], mode)
        pool.share('B', req)
        traffic3, peak3 = append(pool, req, [3], mode)
        comparison[mode] = {'step_traffic_token_bytes': [traffic1, traffic2, traffic3],
                            'total_bytes': (traffic1+traffic2+traffic3)*32,
                            'peak_pages': max(peak1, peak2, peak3),
                            'peak_bytes': max(peak1, peak2, peak3)*4*32}
    check('published same-trajectory traffic and capacity comparison', comparison == {
        'immutable': {'step_traffic_token_bytes': [3, 5, 7], 'total_bytes': 480, 'peak_pages': 4, 'peak_bytes': 512},
        'mixed': {'step_traffic_token_bytes': [1, 1, 7], 'total_bytes': 288, 'peak_pages': 2, 'peak_bytes': 256}})
    for p in (1, 3, 4, 16):
        traffic = {}
        for mode in ('immutable', 'mixed'):
            pool = Pool(p)
            req = pool.create('A', [])
            traffic[mode] = sum(append(pool, req, [i], mode)[0] for i in range(p))
        assert traffic == {'immutable': p*p, 'mixed': p}
    check('sequential fill has P squared versus P declared token-byte traffic', True,
          page_lengths=[1, 3, 4, 16])
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    mapping_cases = mapping_checks()
    transitions = transition_checks()
    comparison = cost_checks()
    inputs = ['knowledge/algorithms/paged-kv-capacity.md', 'knowledge/frameworks/attention-backends.md']
    report = {'status': 'PASS', 'checked_at': '2026-10-28', 'checks': CHECKS,
              'check_count': len(CHECKS), 'mapping_cases': mapping_cases,
              'append_cases': transitions, 'same_trajectory': comparison,
              'inputs': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in inputs},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Independent CPU byte-layout writer versus declared warp loads; serialized-host range/ownership transitions and logical cost arithmetic.',
              'limits': ['No GPU kernel, graph, physical matrix fragment or generated machine instruction was executed.',
                         'Read ranges are declared in the CPU model; actual instructions and device visibility must be established in an implementation.',
                         'Host transitions are serialized, so this is not an exhaustive proof of concurrent allocators.',
                         'Copy bytes and request segments are logical models, not DRAM measurements or performance evidence.',
                         'Integer token payloads model fixed-format K/V state identity, not full model generation or quantized scale updates.',
                         'No controlled Agent knowledge-benefit experiment was performed.']}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checks': len(CHECKS),
                      'mapping_cases': mapping_cases, 'append_cases': transitions}))


if __name__ == '__main__':
    main()
