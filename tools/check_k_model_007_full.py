import math
from collections import Counter


class Rules:
    def __init__(self, w, sinks=(), globals_=(), full=False):
        if w < 1:
            raise ValueError('window must include the current token')
        self.w = w
        self.sinks = frozenset(sinks)
        self.globals = tuple(globals_)
        self.protected = self.sinks | frozenset(j for j, _, _ in self.globals)
        self.full = full

    def visible(self, q):
        if q < 0:
            return set()
        if self.full:
            return set(range(q + 1))
        result = set(range(max(0, q - self.w + 1), q + 1))
        result.update(j for j in self.sinks if 0 <= j <= q)
        result.update(j for j, start, stop in self.globals
                      if 0 <= j <= q and start <= q < stop)
        return result

    def required(self, p):
        if self.full:
            return set(range(p))
        return (set(range(max(0, p - self.w + 1), p)) |
                {j for j in self.protected if 0 <= j < p})


def reference_positions(rules, q):
    # Independent scan; does not call visible() or reuse its endpoints.
    result = []
    for j in range(q + 1):
        local = q - j < rules.w
        sink = j in rules.sinks
        global_now = any(j == pos and start <= q < stop
                         for pos, start, stop in rules.globals)
        if rules.full or local or sink or global_now:
            result.append(j)
    return result


def reference_required(rules, p):
    assert p >= 0
    # Retention includes globals that will activate later, not just visible(p).
    return {j for j in range(p)
            if rules.full or p - j < rules.w or j in rules.protected}


def payload(j):
    return (j, ((7 * j) % 13 - 6) / 9.0, float(j))


def score(q, record, equal=False):
    j, key, _ = record
    if equal:
        return 0.0
    return ((q % 7) - 3) / 4.0 * key + 0.03 * (q - j)


def dense_reference(rules, q, equal=False):
    records = [payload(j) for j in reference_positions(rules, q)]
    if not records:
        return 0.0
    logits = [score(q, record, equal) for record in records]
    maximum = max(logits)
    weights = [math.exp(z - maximum) for z in logits]
    return math.fsum(w * record[2] for w, record in zip(weights, records)) / math.fsum(weights)


def online_attention(q, records, equal=False):
    maximum, mass, numerator = -math.inf, 0.0, 0.0
    for record in records:
        z = score(q, record, equal)
        value = record[2]
        if not math.isfinite(z) or not math.isfinite(value):
            raise ValueError('nonfinite visible input')
        if mass == 0.0:
            maximum, mass, numerator = z, 1.0, value
        else:
            new_max = max(maximum, z)
            old_scale = math.exp(maximum - new_max)
            weight = math.exp(z - new_max)
            mass = old_scale * mass + weight
            numerator = old_scale * numerator + weight * value
            maximum = new_max
    return numerator / mass if mass else 0.0


class Slot:
    # Each method is an indivisible abstract transition, not a device lock.
    def __init__(self):
        self.pos = None
        self.record = None
        self.generation = 0
        self.open = False
        self.serial = 0
        self.live = {}

    def acquire(self, wanted):
        if not self.open or self.pos != wanted:
            return None
        self.serial += 1
        ticket = (self.generation, self.serial, wanted)
        self.live[ticket] = wanted
        return ticket

    def read(self, ticket):
        assert ticket in self.live, 'unprotected physical read'
        assert ticket[0] == self.generation, 'generation changed during read'
        assert self.pos == ticket[2], 'logical identity changed during read'
        assert self.record[0] == ticket[2], 'payload identity mismatch'
        return self.record

    def release(self, ticket):
        if ticket not in self.live:
            return False
        del self.live[ticket]
        return True

    def close(self):
        self.open = False

    def publish(self, pos, record):
        if self.open or self.live:
            return False
        assert record[0] == pos
        self.generation += 1
        self.pos, self.record, self.open = pos, record, True
        return True


class Cache:
    def __init__(self, rules):
        self.rules = rules
        # Unused full-layer ring entries are empty descriptors, not KV payloads.
        self.ring = [Slot() for _ in range(rules.w)]
        self.protected = {}
        self.full = {}
        self.p = 0
        self.restore_collisions = []

    def locate(self, j, create=False):
        if self.rules.full:
            table = self.full
        elif j in self.rules.protected:
            table = self.protected
        else:
            return self.ring[j % self.rules.w]
        if create and j not in table:
            table[j] = Slot()
        return table.get(j)

    def put(self, j, record):
        slot = self.locate(j, create=True)
        slot.close()
        assert slot.publish(j, record), 'write attempted before reader drain'

    def load(self, j):
        slot = self.locate(j)
        if slot is None:
            raise KeyError(j)
        ticket = slot.acquire(j)
        if ticket is None:
            raise KeyError(j)
        try:
            return slot.read(ticket)
        finally:
            assert slot.release(ticket)

    def slots(self):
        return self.ring + list(self.protected.values()) + list(self.full.values())

    def acquire_many(self, positions):
        # No interleaving inside this abstract directory transaction.
        held = {}
        try:
            for j in sorted(positions):
                slot = self.locate(j)
                if slot is None:
                    raise KeyError(j)
                ticket = slot.acquire(j)
                if ticket is None:
                    raise KeyError(j)
                held[j] = (slot, ticket)
            return held
        except BaseException:
            for slot, ticket in held.values():
                assert slot.release(ticket)
            raise


def warm(rules, n):
    cache = Cache(rules)
    for j in range(n):
        cache.put(j, payload(j))
    cache.p = n
    return cache


def restore_target(source, faulty_full_to_ring=False):
    # Direct translation of RestoreLayer's destination branches.
    # Deliberately does NOT call Cache.put or use locate() to choose destinations.
    # The faulty option builds a private, invalid candidate for negative controls.
    p, rules = source.p, source.rules
    expected = rules.required(p)
    assert expected == reference_required(rules, p)
    held = source.acquire_many(expected)
    target = Cache(rules)
    try:
        for j in sorted(expected):
            source_slot, ticket = held[j]
            record = source_slot.read(ticket)
            assert record == payload(j), 'source identity or K/V content mismatch'
            if rules.full and not faulty_full_to_ring:
                assert j not in target.full
                slot = Slot()
                target.full[j] = slot
            elif j in rules.protected:
                assert j not in target.protected
                slot = Slot()
                target.protected[j] = slot
            else:
                slot = target.ring[j % rules.w]
            if slot.pos is not None:
                target.restore_collisions.append((slot.pos, j))
            assert slot is not source_slot, 'copy target aliases mutable source'
            slot.close()
            assert slot.publish(j, record), 'private target publication failed'
        # Tentative local boundary only; caller must audit before acceptance.
        target.p = p
    finally:
        for source_slot, ticket in held.values():
            assert source_slot.release(ticket)
    return target


def audit_restore(target):
    rules, p = target.rules, target.p
    expected = reference_required(rules, p)
    assert not target.restore_collisions, ('target collision', target.restore_collisions)
    if rules.full:
        assert set(target.full) == set(range(p)), 'full destination missing history'
        assert not target.protected, 'full destination incorrectly split by P'
        assert all(slot.pos is None for slot in target.ring), 'full destination used ring'
    else:
        assert not target.full
        assert set(target.protected) == {j for j in rules.protected if 0 <= j < p}
    addresses = []
    for j in sorted(expected):
        slot = target.locate(j)
        assert slot is not None and slot.pos == j, ('wrong destination identity', j)
        assert slot.open and slot.record == payload(j), ('wrong destination K/V', j)
        assert target.load(j) == payload(j)
        addresses.append(id(slot))
    assert len(addresses) == len(set(addresses)), 'required positions alias'
    assert all(not slot.live for slot in target.slots())


def restore(source):
    target = restore_target(source)
    audit_restore(target)
    return target


def prefill(cache, b, e, block, equal=False, faulty_old=False):
    assert block >= 1 and cache.p == b and 0 <= b <= e
    if b == e:
        return {}, {}
    rules = cache.rules
    staging = {j: payload(j) for j in range(b, e)}
    masks = {q: rules.visible(q) for q in range(b, e)}
    old_union = set()
    for positions in masks.values():
        old_union.update(j for j in positions if j < b)
    held = cache.acquire_many(old_union)
    outputs, seen = {}, {}
    try:
        blocks = sorted({j // block for j in old_union})
        for q in range(b, e):
            records = []
            for key_block in blocks:
                # Faulty control omits the old-range test and consumes current
                # keys from a candidate block a second time.
                for j in range(key_block * block, min((key_block + 1) * block, e)):
                    if j not in masks[q]:
                        continue
                    if not faulty_old and (j >= b or j not in old_union):
                        continue
                    if j < b:
                        slot, ticket = held[j]
                        record = slot.read(ticket)
                    else:
                        record = staging[j]
                    assert record == payload(j)
                    records.append(record)
            for key_block in range(b // block, (e - 1) // block + 1):
                lo = max(b, key_block * block)
                hi = min(e, (key_block + 1) * block)
                for j in range(lo, hi):
                    if j in masks[q]:
                        records.append(staging[j])
            seen[q] = Counter(record[0] for record in records)
            outputs[q] = online_attention(q, records, equal)
    finally:
        for slot, ticket in held.values():
            assert slot.release(ticket)
    return outputs, seen


def commit(cache, b, e):
    assert cache.p == b
    if cache.rules.full:
        keep = set(range(b, e))
    else:
        keep = ({j for j in range(b, e) if j in cache.rules.protected} |
                set(range(max(b, e - cache.rules.w), e)))
    for j in sorted(keep):
        cache.put(j, payload(j))
    cache.p = e


def assert_close(actual, expected):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-12), (actual, expected)


def full_restore_check():
    rows = 0
    for rules in (Rules(4, full=True),
                  Rules(4, sinks=(0,), globals_=((2, 9, 100),), full=True),
                  Rules(1, full=True)):
        for p in (0, 1, 3, 5, 12):
            source = warm(rules, p)
            target = restore(source)
            assert set(target.full) == set(range(p))
            for j in range(p):
                assert target.locate(j) is target.full[j]
                assert target.full[j] is not source.full[j]
                assert target.load(j) == source.load(j) == payload(j)
            assert len({id(slot) for slot in target.full.values()}) == p
            assert not target.protected
            assert all(slot.pos is None for slot in target.ring)
            for equal in (True, False):
                outputs, counts = prefill(target, p, p + 3, 4, equal=equal)
                for q in range(p, p + 3):
                    assert counts[q] == Counter(range(q + 1))
                    assert_close(outputs[q], dense_reference(rules, q, equal))
                    rows += 1
                if rules.w == 4 and p == 5 and equal:
                    assert_close(outputs[5], 2.5)
            commit(target, p, p + 3)
            target = restore(target)
            outputs, counts = prefill(target, p + 3, p + 4, 6)
            assert counts[p + 3] == Counter(range(p + 4))
            assert_close(outputs[p + 3], dense_reference(rules, p + 3))
            rows += 1
            # Continuing the copied target did not modify the source history.
            assert set(source.full) == set(range(p))
            for j in range(p):
                assert source.load(j) == payload(j)
    print('full restore: explicit full[j] destinations, all identity/K/V checks, {} continued rows against independent dense reference'.format(rows))

    for p in (3, 5, 12):
        source = warm(Rules(4, full=True), p)
        bad = restore_target(source, faulty_full_to_ring=True)
        assert not bad.full, 'negative control failed to exercise wrong address domain'
        if p == 3:
            assert not bad.restore_collisions
        if p == 5:
            assert bad.restore_collisions == [(0, 4)]
            assert bad.ring[0].record == payload(4)
        if p == 12:
            assert len(bad.restore_collisions) == 8
        try:
            audit_restore(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('full-to-ring restoration accepted')
        # Deliberately bypass candidate validation: normal full Locate still
        # rejects the missing object when a subsequent attention is attempted.
        try:
            prefill(bad, p, p + 1, 4)
        except KeyError as error:
            assert error.args == (0,)
        else:
            raise AssertionError('attention consumed incomplete full restore')
        assert all(not slot.live for slot in bad.slots())
        assert all(not slot.live for slot in source.slots())
        print('negative control detected: full p={} restored to ring; collisions={}; full[0] missing; continuation rejected'.format(p, bad.restore_collisions))


def interleavings(sequences, prefix=()):
    if all(not sequence for sequence in sequences):
        yield prefix
        return
    for index, sequence in enumerate(sequences):
        if sequence:
            remaining = list(sequences)
            remaining[index] = sequence[1:]
            yield from interleavings(tuple(remaining), prefix + (sequence[0],))


def event_check():
    count = successful = blocked = 0
    sequences = (('a0', 'u0', 'r0'), ('a1', 'u1', 'r1'), ('close', 'publish'))
    for schedule in interleavings(sequences):
        count += 1
        slot = Slot()
        assert slot.publish(0, payload(0))
        tickets = {}
        pending = published = False
        for event in schedule:
            if event.startswith('a'):
                ticket = slot.acquire(0)
                tickets[event[1]] = ticket
                successful += ticket is not None
            elif event.startswith('u'):
                ticket = tickets[event[1]]
                if ticket is not None:
                    assert slot.read(ticket) == payload(0)
            elif event.startswith('r'):
                ticket = tickets[event[1]]
                if ticket is not None:
                    assert slot.release(ticket)
                    assert not slot.release(ticket), 'duplicate release accepted'
            elif event == 'close':
                slot.close()
                assert slot.acquire(0) is None
            else:
                pending = True
                if slot.live:
                    blocked += 1
                    assert not slot.publish(4, payload(4))
            if pending and not slot.live:
                assert slot.publish(4, payload(4))
                pending, published = False, True
        assert published and not slot.live and slot.record == payload(4)
        assert slot.acquire(0) is None
    assert (count, successful, blocked) == (560, 672, 262)
    print('event model: 560 schedules; 672 successful acquires; 262 blocked publish attempts')

    slot = Slot()
    assert slot.publish(0, payload(0))
    checked_old = slot.open and slot.pos == 0
    assert checked_old
    slot.close()
    assert slot.publish(4, payload(4))
    legacy_ticket = slot.acquire(slot.pos)
    wrong = slot.read(legacy_ticket)
    assert wrong == payload(4) and wrong != payload(0)
    assert slot.release(legacy_ticket)
    print('negative control detected: split check/register physically read position 4 for request 0')

    cache = warm(Rules(4), 5)
    missing = cache.locate(3)
    missing.close()
    missing.pos = None
    try:
        prefill(cache, 5, 7, 4)
    except KeyError as error:
        assert error.args == (3,)
    else:
        raise AssertionError('missing old position accepted')
    assert cache.p == 5 and all(not slot.live for slot in cache.slots())
    print('failure path: partial unsubmitted acquisition released; no new boundary committed')


def main():
    full_restore_check()
    rows = chunks = 0
    for w in (1, 2, 4, 7):
        variants = (
            Rules(w),
            Rules(w, sinks=(0,)),
            Rules(w, sinks=(0,), globals_=((2, 7, 1000000),)),
            Rules(w, full=True),
        )
        for rules in variants:
            for n in (0, 1, 3, 12):
                for chunk_size in (1, 2, 5):
                    for block in (1, 4, 6):
                        cache = restore(Cache(rules))
                        assert prefill(cache, 0, 0, block) == ({}, {})
                        b = 0
                        while b < n:
                            e = min(n, b + chunk_size)
                            output, seen = prefill(cache, b, e, block)
                            for q in range(b, e):
                                expected_positions = reference_positions(rules, q)
                                assert rules.visible(q) == set(expected_positions)
                                assert seen[q] == Counter(expected_positions), (w, b, e, q, seen[q])
                                assert_close(output[q], dense_reference(rules, q))
                                rows += 1
                            assert all(not slot.live for slot in cache.slots())
                            commit(cache, b, e)
                            cache = restore(cache)
                            for j in reference_required(rules, e):
                                assert cache.load(j) == payload(j)
                            b = e
                            chunks += 1
    print('cache/chunk model: {} rows and {} chunks matched independent scan + dense softmax'.format(rows, chunks))

    rules = Rules(4, sinks=(0,))
    cache = warm(rules, 5)
    actual = {j: cache.load(j) for j in rules.visible(4)}
    assert set(actual) == {0, 1, 2, 3, 4}
    assert actual[0] == payload(0) and actual[4] == payload(4)
    assert cache.locate(0) is not cache.locate(4)
    print('sink collision: positions 0 and 4 survive in distinct objects with verified K/V identity')

    bad_ring = [None] * 4
    for j in range(5):
        bad_ring[j % 4] = payload(j)
    assert bad_ring[0] == payload(4) and bad_ring[0] != payload(0)
    print('negative control detected: ring-only sink mapping returns position 4 instead of 0')

    dynamic = Rules(4, sinks=(0,), globals_=((2, 7, 1000000),))
    cache = warm(dynamic, 7)
    assert 2 not in dynamic.visible(6) and 2 in dynamic.required(7)
    resumed = restore(cache)
    output, seen = prefill(resumed, 7, 10, 4)
    for q in range(7, 10):
        assert 2 in seen[q] and resumed.load(2) == payload(2)
        assert_close(output[q], dense_reference(dynamic, q))
    del cache.protected[2]
    try:
        restore(cache)
    except KeyError as error:
        assert error.args == (2,)
    else:
        raise AssertionError('missing future global accepted')
    assert all(not slot.live for slot in cache.slots())
    print('global restore: inactive protected position survives then is consumed; missing object rejected')

    cache = warm(Rules(4), 5)
    correct, counts = prefill(cache, 5, 7, 4, equal=True)
    wrong, wrong_counts = prefill(cache, 5, 7, 4, equal=True, faulty_old=True)
    assert counts[5] == Counter({2: 1, 3: 1, 4: 1, 5: 1})
    assert_close(correct[5], 3.5)
    assert wrong_counts[5][5] == 2
    assert_close(wrong[5], 3.8)
    assert not math.isclose(wrong[5], dense_reference(cache.rules, 5, True), abs_tol=1e-12)
    print('negative control detected: unaligned old/current duplicate changes 3.5 to 3.8')

    assert reference_positions(Rules(4), 5) == [2, 3, 4, 5]
    last_row = Rules(4).visible(6)
    assert set(reference_positions(Rules(4), 5)) - last_row == {2}
    assert {j // 64 for j in Rules(256).visible(255)} == {0, 1, 2, 3}
    assert {j // 64 for j in Rules(256).visible(256)} == {0, 1, 2, 3, 4}
    assert not all(j // 64 in {0, 1, 2, 3} for j in reference_positions(Rules(256), 256))
    print('geometry: endpoint, last-row omission, and unaligned fifth-block negative controls detected')

    assert Rules(4).required(10) == {7, 8, 9}
    assert Rules(1).required(10) == set()
    assert restore(warm(Rules(1), 10)).p == 10
    assert Rules(1, full=True).required(10) == set(range(10))
    missing = warm(Rules(4), 10)
    missing.locate(7).close()
    try:
        restore(missing)
    except KeyError:
        pass
    else:
        raise AssertionError('incomplete W-1 history accepted')
    for invalid in (0, -1):
        try:
            Rules(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid W accepted')
    assert online_attention(0, []) == 0.0
    assert Rules(4).visible(-1) == set()

    w, d, dv, hq, hkv = 4096, 128, 128, 32, 8
    one_head = 2 * w * (d + dv)
    whole_layer = sum(2 * w * (d + dv) for _ in range(hq))
    assert one_head == 2097152 and whole_layer == 67108864
    assert whole_layer != 2 * hkv * w * (d + dv)
    kv_bytes = hkv * (d * 2 + dv * 2)
    assert kv_bytes == 4096
    assert w * kv_bytes == 16 * 1024 * 1024
    assert 8 * 8192 * kv_bytes + 24 * w * kv_bytes == 640 * 1024 * 1024
    varied = [reference_positions(Rules(4, sinks=(0,)), q) for q in range(3, 8)]
    counted = sum(2 * (d + dv) for _ in range(hq) for positions in varied for _ in positions)
    assert counted == 2 * hq * sum(map(len, varied)) * (d + dv)
    print('cost axes: single-head 2097152 FLOPs; 32-query-head layer 67108864 FLOPs; KV uses 8 heads')

    event_check()
    print('scope: deterministic CPU scalar/cache/finite-event model only; no project imports, file writes, subprocesses, network, GPU execution, model quality, or device performance measurements')


if __name__ == '__main__':
    main()
