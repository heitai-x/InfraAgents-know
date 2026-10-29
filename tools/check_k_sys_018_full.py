#!/usr/bin/env python3
"""Deterministic CPU protocol models; no backend or device measurement."""
import hashlib
import itertools
import json


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')


def rejected(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError('invalid input was accepted')


NEGATIVES = []


def detected(name, action):
    try:
        action()
    except AssertionError:
        NEGATIVES.append(name)
        return
    raise AssertionError('negative control escaped: ' + name)


def covers(n, ranges):
    ranges = list(ranges)
    if type(n) is not int or n < 0:
        raise ValueError('invalid extent')
    for item in ranges:
        if len(item) != 2:
            raise ValueError('interval arity')
        lo, hi = item
        if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi <= n:
            raise ValueError('invalid interval')
    end = 0
    for lo, hi in sorted(ranges):
        if lo > end:
            return False
        end = max(end, hi)
    return end >= n


class Store:
    """One namespace, identity transform, contiguous logical byte space."""
    def __init__(self):
        self.objects = {}
        self.manifests = {}
        self.persistent = {}
        self.published = set()
        self.pins = {}
        self.root = (0, None)

    def prepare(self, version, representation, n, rows, durable=True):
        if version in self.manifests or not version or not representation:
            return False
        if type(n) is not int or n < 0:
            return False
        names = [row[0] for row in rows]
        if len(names) != len(set(names)):
            return False
        records = []
        for name, lo, data, expected in rows:
            if type(lo) is not int or lo < 0 or lo + len(data) > n:
                return False
            if sha(data) != expected:
                return False
            records.append((name, lo, len(data), expected))
        cursor = 0
        for name, lo, size, expected in sorted(records, key=lambda r: (r[1], r[0])):
            if lo != cursor:
                return False
            cursor += size
        if cursor != n:
            return False
        body = (version, representation, n, tuple(sorted(records, key=lambda r: (r[1], r[0]))))
        self.manifests[version] = (body, sha(encode(body)))
        self.persistent[version] = durable
        self.pins[version] = 0
        for name, lo, data, expected in rows:
            self.objects[(version, name)] = bytes(data)
        return True

    def publish(self, version, expected_epoch):
        if expected_epoch != self.root[0] or not self.persistent.get(version, False):
            return False
        self.published.add(version)
        self.root = (expected_epoch + 1, version)
        return True

    def open(self, representation, version=None):
        version = self.root[1] if version is None else version
        if version not in self.manifests or version not in self.published:
            raise ValueError('missing or unpublished manifest')
        body, expected = self.manifests[version]
        if sha(encode(body)) != expected or body[1] != representation:
            raise ValueError('manifest identity or digest')
        self.pins[version] += 1
        try:
            pieces = []
            for name, lo, size, checksum in body[3]:
                data = self.objects.get((version, name))
                if data is None or len(data) != size or sha(data) != checksum:
                    raise ValueError('missing, short or corrupt shard')
                pieces.append(data)
            return version, b''.join(pieces)
        except ValueError:
            self.release(version)
            raise

    def release(self, version):
        require(self.pins[version] > 0, 'unbalanced pin release')
        self.pins[version] -= 1

    def collect(self, version):
        if version not in self.manifests or version == self.root[1] or self.pins[version]:
            return False
        del self.manifests[version]
        self.published.discard(version)
        del self.persistent[version]
        del self.pins[version]
        for key in list(self.objects):
            if key[0] == version:
                del self.objects[key]
        return True


class Cache:
    def __init__(self, contracts, bad_cancel=False):
        self.contracts = contracts
        self.bad_cancel = bad_cancel
        self.entries = {}
        self.serial = 0
        self.starts = {}
        self.results = {}

    def get(self, key, owner):
        entry = self.entries.get(key)
        if entry is None or entry['state'] == 'FAILED':
            self.serial += 1
            entry = dict(g=self.serial, state='LOADING', waiters={owner}, readers=set())
            self.entries[key] = entry
            self.starts[key] = self.starts.get(key, 0) + 1
            return 'LOAD', entry['g']
        if entry['state'] == 'LOADING':
            entry['waiters'].add(owner)
            return 'WAIT', entry['g']
        entry['readers'].add(owner)
        return 'READY', entry['g']

    def cancel_wait(self, key, owner):
        entry = self.entries.get(key)
        if entry is None:
            return
        if self.bad_cancel:
            del self.entries[key]
        else:
            entry['waiters'].discard(owner)

    def complete(self, key, generation, data):
        entry = self.entries.get(key)
        if entry is None or entry['g'] != generation or entry['state'] != 'LOADING':
            return False
        size, checksum = self.contracts[key]
        ok = len(data) == size and sha(data) == checksum
        entry['state'] = 'READY' if ok else 'FAILED'
        for owner in entry['waiters']:
            self.results[(key, generation, owner)] = ('READY', data) if ok else ('FAILED', None)
        if ok:
            entry['readers'].update(entry['waiters'])
        entry['waiters'].clear()
        return ok

    def release(self, key, owner):
        self.entries[key]['readers'].remove(owner)

    def evict(self, key):
        entry = self.entries.get(key)
        if entry is None or entry['state'] != 'READY' or entry['readers'] or entry['waiters']:
            return False
        del self.entries[key]
        return True


class Log:
    """Complete records on modeled persistent storage; recovery requires a prefix."""
    def __init__(self, early_ack=False):
        self.records = []
        self.by_cid = {}
        self.disk = {}
        self.acks = []
        self.early_ack = early_ack

    def append(self, cid, params, payload):
        content = (cid, params, payload.hex())
        if cid in self.by_cid:
            old = self.records[self.by_cid[cid]]
            return self.query(cid) if old[:3] == content else 'CONFLICT'
        self.by_cid[cid] = len(self.records)
        self.records.append(content + (sha(encode(content)),))
        return 'PENDING'

    def sync(self, index, torn=False):
        row = self.records[index]
        self.disk[index] = row[:-1] + ('bad-checksum',) if torn else row

    def recover(self):
        out = []
        for index in range(len(self.records)):
            row = self.disk.get(index)
            if row is None or sha(encode(row[:3])) != row[3]:
                break
            out.append(row[0])
        return out

    def query(self, cid):
        return 'DURABLE' if self.by_cid[cid] < len(self.recover()) else 'PENDING'

    def ack(self, cid):
        if self.query(cid) != 'DURABLE' and not self.early_ack:
            return False
        self.acks.append(cid)
        return True


def audit_ack(log):
    require(set(log.acks) <= set(log.recover()), 'durable ack absent after crash')


def fixture(version):
    chunks = [version.encode('ascii') + b'A', version.encode('ascii') + b'B']
    rows = []
    lo = 0
    for name, data in zip(('a', 'b'), chunks):
        rows.append((name, lo, data, sha(data)))
        lo += len(data)
    return lo, rows, b''.join(chunks)


def test_ranges():
    count = 0
    for n in range(6):
        spans = [(lo, hi) for lo in range(n + 1) for hi in range(lo, n + 1)]
        lists = [()] + [(x,) for x in spans] + list(itertools.product(spans, repeat=2))
        for ranges in lists:
            reference = set(range(n)) <= {i for lo, hi in ranges for i in range(lo, hi)}
            require(covers(n, ranges) == reference, 'coverage versus byte-set reference')
            count += 1
    rejected(lambda: covers(3, [(0, 4)]))
    rejected(lambda: covers(3, [(2, 1)]))
    rejected(lambda: covers(3, [(7, 0, 3)]))
    detected('byte-count-hides-hole', lambda: require(
        (sum(hi - lo for lo, hi in [(0, 1), (0, 1)]) >= 2) == covers(2, [(0, 1), (0, 1)]),
        'total length is not coverage'))
    return count


def test_store():
    store = Store()
    n1, rows1, bytes1 = fixture('v1')
    n2, rows2, bytes2 = fixture('v2')
    require(store.prepare('v1', 'q', n1, rows1), 'prepare v1')
    require(store.publish('v1', 0), 'publish v1')
    snap = store.open('q')
    require(store.prepare('v2', 'q', n2, rows2), 'prepare v2')
    require(not store.publish('v2', 0), 'stale CAS accepted')
    require(store.root == (1, 'v1'), 'failed CAS changed root')
    require(store.publish('v2', 1), 'publish v2')
    require(snap == ('v1', bytes1), 'snapshot changed')
    second = store.open('q', 'v1')
    require(second == snap, 'old pinned version content')
    require(not store.collect('v1'), 'GC ignored readers')
    store.release('v1')
    require(not store.collect('v1'), 'GC ignored second reader')
    store.release('v1')
    require(store.collect('v1'), 'unreferenced old version not collectable')
    rejected(lambda: store.open('q', 'v1'))
    rejected(lambda: store.open('other-q'))
    detected('latest-following-reader', lambda: require(
        bytes1[:len(rows1[0][2])] + bytes2[len(rows2[0][2]):] == bytes1, 'mixed versions'))
    damaged = store.objects.pop(('v2', 'b'))
    rejected(lambda: store.open('q'))
    require(store.pins['v2'] == 0, 'failed read leaked pin')
    store.objects[('v2', 'b')] = damaged[:-1]
    rejected(lambda: store.open('q'))
    store.objects[('v2', 'b')] = b'Z' * len(damaged)
    rejected(lambda: store.open('q'))
    store.objects[('v2', 'b')] = damaged
    body, checksum = store.manifests['v2']
    store.manifests['v2'] = ((body[0], body[1], body[2] + 1, body[3]), checksum)
    rejected(lambda: store.open('q'))
    store.manifests['v2'] = (body, checksum)
    before = store.root
    require(not store.prepare('bad-gap', 'q', n1, rows1[:1]), 'missing required range')
    require(not store.prepare('bad-duplicate', 'q', n1, [rows1[0], rows1[0]]), 'duplicate shard')
    overlap = [('a', 0, b'ab', sha(b'ab')), ('b', 1, b'cd', sha(b'cd'))]
    require(not store.prepare('bad-overlap', 'q', 3, overlap), 'overlapping logical writes')
    require(not store.prepare('bad-hash', 'q', 1, [('a', 0, b'x', sha(b'y'))]), 'digest mismatch')
    require(not store.prepare('bad-offset', 'q', 1, [('a', -1, b'x', sha(b'x'))]), 'negative offset')
    require(not store.prepare('bad-empty', 'q', 1, []), 'empty nonempty artifact')
    require(not store.prepare('v2', 'q', n2, rows2), 'immutable version overwritten')
    require(store.root == before, 'invalid preparation changed root')
    require(store.prepare('empty', 'q', 0, []), 'explicit empty schema')
    require(store.prepare('volatile', 'q', n1, rows1, durable=False), 'volatile staging')
    require(not store.publish('volatile', 2), 'publication ignored required durability')
    rejected(lambda: store.open('q', 'volatile'))
    require(store.root == before, 'volatile staging changed root')
    require(not store.collect('v2'), 'GC collected current root')


def cache_contract(bad_cancel=False):
    data = b'ABCD'
    key = ('tenant', 'v1', 'q', 'transform-1', 'part-0')
    other = ('tenant', 'v2', 'q', 'transform-1', 'part-0')
    cache = Cache({key: (4, sha(data)), other: (4, sha(data))}, bad_cancel)
    mode, generation = cache.get(key, 'A')
    require(mode == 'LOAD', 'first miss')
    require(cache.get(key, 'B') == ('WAIT', generation), 'single-flight wait')
    cache.cancel_wait(key, 'A')
    cache.complete(key, generation, data)
    require(cache.results.get((key, generation, 'B')) == ('READY', data), 'surviving waiter lost')
    require(cache.starts[key] == 1, 'duplicate loader')
    require(cache.get(other, 'D')[0] == 'LOAD', 'different versions merged')
    require(not cache.evict(key), 'reader not protected')
    cache.cancel_wait(key, 'B')
    require(not cache.evict(key), 'cancellation released acquired reader')
    cache.release(key, 'B')
    require(cache.evict(key), 'reader release not collectable')
    mode, newer = cache.get(key, 'C')
    require(newer != generation, 'attempt identity reused')
    require(not cache.complete(key, generation, data), 'stale completion accepted')
    require(not cache.complete(key, newer, b'ZZZZ'), 'corruption accepted')
    mode, retry = cache.get(key, 'E')
    require(mode == 'LOAD' and retry != newer, 'failed attempt not retried')
    require(not cache.complete(key, newer, data), 'failed old attempt installed retry')
    cache.cancel_wait(key, 'E')
    require(not cache.evict(key), 'in-flight loader buffer reclaimed')
    require(cache.complete(key, retry, data), 'retry failed')
    require(cache.evict(key), 'no-waiter terminal load not evictable')


def test_log():
    log = Log()
    require(log.append('c0', 'p0', b'out0') == 'PENDING', 'new visible record')
    require(log.append('c0', 'p0', b'out0') == 'PENDING', 'pending retry became durable')
    require(log.append('c0', 'changed', b'out0') == 'CONFLICT', 'params conflict')
    require(log.append('c0', 'p0', b'changed') == 'CONFLICT', 'payload conflict')
    require(len(log.records) == 1, 'retry advanced sequence')
    log.append('c1', 'p1', b'out1')
    log.sync(1)
    require(log.recover() == [] and not log.ack('c1'), 'hole allowed durable acknowledgement')
    detected('recover-skips-hole', lambda: require(
        [log.disk[i][0] for i in sorted(log.disk)] == [], 'non-prefix replay'))
    log.sync(0, torn=True)
    require(log.recover() == [], 'torn record recovered')
    log.sync(0)
    require(log.recover() == ['c0', 'c1'], 'complete prefix missing')
    require(log.append('c0', 'p0', b'out0') == 'DURABLE', 'durable retry lost decision')
    require(log.ack('c0') and log.ack('c1'), 'durable ack rejected')
    log.append('c2', 'p2', b'out2')
    require(not log.ack('c2') and log.recover() == ['c0', 'c1'], 'visible survived as committed')
    audit_ack(log)
    bad = Log(early_ack=True)
    bad.append('c', 'p', b'x')
    bad.ack('c')
    detected('visible-is-durable', lambda: audit_ack(bad))
    count = 0
    for order in itertools.permutations(range(3)):
        model = Log()
        for i in range(3):
            model.append('c' + str(i), 'p', bytes([i]))
        done = set()
        for step in range(4):
            if step:
                model.sync(order[step - 1])
                done.add(order[step - 1])
            frontier = min(set(range(3)) - done, default=3)
            require(model.recover() == ['c' + str(i) for i in range(frontier)], 'disk subset reference')
            count += 1
    return count


def test_pipeline():
    count = 0
    for n in range(6):
        for times in itertools.product((0, 1, 3), repeat=3):
            previous = [0, 0, 0]
            for unused in range(n):
                row = []
                for k, duration in enumerate(times):
                    row.append(max(previous[k], row[k - 1] if k else 0) + duration)
                previous = row
            reference = sum(times) + (n - 1) * max(times) if n else 0
            require(previous[-1] == reference, 'pipeline recurrence versus closed form')
            count += 1
    detected('serial-work-is-makespan', lambda: require(2 * (1 + 1 + 1) == 4, 'serial sum'))
    require(16 * 32 // 8 == 64, 'shared egress units')
    return count


def main():
    ranges = test_ranges()
    test_store()
    cache_contract()
    detected('cancel-deletes-shared-fill', lambda: cache_contract(bad_cancel=True))
    logs = test_log()
    pipeline = test_pipeline()
    require(len(NEGATIVES) == 6, 'negative-control coverage incomplete')
    print(json.dumps({
        'scope': 'deterministic in-memory CPU protocol and algebra models',
        'interval_reference_cases': ranges,
        'log_disk_subset_observations': logs,
        'pipeline_reference_cases': pipeline,
        'detected_negative_controls': NEGATIVES,
        'shared_egress_seconds_from_assumptions': 64,
        'real_storage_network_gpu_measured': False
    }, sort_keys=True))


if __name__ == '__main__':
    main()
