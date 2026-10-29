import math
from fractions import Fraction


def require(value):
    assert value


def rejected(fn):
    try:
        fn()
    except (AssertionError, ValueError):
        return True
    return False


class Aligner:
    def __init__(self, width=4):
        self.width = width
        self.cells = {}
        self.records = set()
        self.next_b = 0
        self.closed = False
        self.final_status = None

    def feed(self, chunk=None, flush=False, allow_partial=True):
        if chunk is not None:
            a, b, data = chunk
            if a < 0 or b <= a or len(data) != b - a:
                raise ValueError('bad interval')
            record = (a, b, tuple(data))
            if record not in self.records:
                if self.closed:
                    raise ValueError('new source after flush')
                if any(t in self.cells for t in range(a, b)):
                    raise ValueError('non-identical overlap')
                self.records.add(record)
                self.cells.update(zip(range(a, b), data))
        if self.closed:
            return [], self.final_status
        out = []
        while True:
            a = self.next_b * self.width
            b = a + self.width
            if not all(t in self.cells for t in range(a, b)):
                break
            out.append((self.next_b, a, b, False))
            self.next_b += 1
        start = self.next_b * self.width
        tail = [t for t in self.cells if t >= start]
        if flush:
            self.closed = True
            if not tail:
                status = 'FINAL'
            elif any(t not in self.cells for t in range(start, max(tail) + 1)):
                status = 'FAILED_GAP'
            elif not allow_partial:
                status = 'FAILED_PARTIAL_CONTRACT'
            else:
                end = max(tail) + 1
                assert start < end < start + self.width
                out.append((self.next_b, start, end, True))
                self.next_b += 1
                status = 'FINAL'
            self.final_status = status
        elif not tail:
            status = 'EMPTY'
        elif any(t not in self.cells for t in range(start, max(tail) + 1)):
            status = 'WAIT_GAP'
        else:
            status = 'WAIT_MORE'
        return out, status


def partition_reference(blocks, end, width=4):
    covered = [t for _, a, b, _ in blocks for t in range(a, b)]
    assert covered == list(range(end))
    assert len(covered) == len(set(covered))
    for i, a, b, partial in blocks:
        assert a == i * width and a < b <= a + width
        assert partial == (b - a < width)


def check_alignment():
    a = Aligner()
    chunk = (0, 4, tuple(range(4)))
    assert a.feed(chunk) == ([(0, 0, 4, False)], 'EMPTY')
    assert a.feed(chunk) == ([], 'EMPTY')
    assert a.feed(flush=True) == ([], 'FINAL')
    assert a.feed(chunk) == ([], 'FINAL')
    assert rejected(lambda: a.feed((4, 5, (4,))))
    a = Aligner()
    out, status = a.feed((0, 6, tuple(range(6))))
    assert out == [(0, 0, 4, False)] and status == 'WAIT_MORE'
    assert a.feed((0, 6, tuple(range(6)))) == ([], 'WAIT_MORE')
    last, status = a.feed(flush=True)
    assert last == [(1, 4, 6, True)] and status == 'FINAL'
    partition_reference(out + last, 6)
    assert rejected(lambda: require(([], 'WAIT_MORE')[0] == out))
    assert rejected(lambda: partition_reference(out + [(1, 4, 4, True)], 4))
    a = Aligner()
    assert a.feed((4, 8, tuple(range(4, 8)))) == ([], 'WAIT_GAP')
    out, status = a.feed((0, 4, tuple(range(4))))
    assert status == 'EMPTY'
    partition_reference(out, 8)
    assert rejected(lambda: a.feed((2, 5, (2, 3, 4))))
    a = Aligner()
    a.feed((4, 5, (4,)))
    assert a.feed(flush=True)[1] == 'FAILED_GAP'
    a = Aligner()
    assert a.feed(flush=True) == ([], 'FINAL')
    a = Aligner()
    a.feed((0, 1, (0,)))
    assert a.feed(flush=True, allow_partial=False)[1] == 'FAILED_PARTIAL_CONTRACT'
    cases = 0
    for end in range(1, 13):
        for split in range(1, end + 1):
            a = Aligner()
            blocks = []
            for lo, hi in ((0, split), (split, end)):
                if lo < hi:
                    new, _ = a.feed((lo, hi, tuple(range(lo, hi))))
                    blocks.extend(new)
            new, status = a.feed(flush=True)
            blocks.extend(new)
            assert status == 'FINAL'
            partition_reference(blocks, end)
            cases += 1
    print('alignment:', cases, 'partitions; duplicate/gap/flush edges; faulty return and empty block detected')


class Credits:
    def __init__(self, t=0, k=0, limit=5, byte_cap=20):
        self.t, self.k = t, k
        self.limit, self.byte_cap = limit, byte_cap
        self.queue = [2] * (t - k)
        self.reserved = []
        self.retired = []
        self.log = []
        self.invariant()

    def used(self):
        return sum(self.queue + self.reserved + self.retired)

    def invariant(self):
        assert 0 <= self.k <= self.t
        assert len(self.queue) == self.t - self.k
        assert self.t - self.k + len(self.reserved) <= self.limit
        assert self.used() <= self.byte_cap

    def reserve(self, count):
        assert count >= 0
        if self.t - self.k + len(self.reserved) + count > self.limit:
            return False
        if self.used() + 2 * count > self.byte_cap:
            return False
        self.reserved.extend([2] * count)
        self.log.append(('reserve', count))
        self.invariant()
        return True

    def commit_one(self):
        assert self.reserved
        self.queue.append(self.reserved.pop(0))
        self.t += 1
        self.log.append(('publish', self.t))
        self.invariant()

    def produce(self, count):
        if not self.reserve(count):
            return False
        for _ in range(count):
            self.commit_one()
        return True

    def consume(self, defer=False):
        assert self.queue
        size = self.queue.pop(0)
        self.k += 1
        if defer:
            self.retired.append(size)
        self.log.append(('consume', self.k))
        self.invariant()

    def faulty_publish(self):
        self.queue.append(2)
        self.t += 1
        self.invariant()


def check_credits_and_frontiers():
    c = Credits(8, 3)
    assert not c.produce(1) and not c.log
    c.consume()
    assert c.produce(1) and (c.t, c.k) == (9, 4)
    assert [x[0] for x in c.log] == ['consume', 'reserve', 'publish']
    c = Credits(7, 3)
    assert not c.produce(2) and c.produce(1)
    assert rejected(lambda: Credits(8, 3).faulty_publish())
    c = Credits(2, 0, byte_cap=5)
    assert not c.produce(1)
    c.consume(defer=True)
    assert not c.produce(1)
    c.retired.clear()
    assert c.produce(1)
    cases = 0
    for t in range(9):
        for k in range(t + 1):
            for limit in range(t - k, 9):
                for count in range(4):
                    c = Credits(t, k, limit, 16)
                    expected = len(range(k, t)) + count <= min(limit, 8)
                    assert c.produce(count) == expected
                    c.invariant()
                    cases += 1
    for dt, dk, expected in ((0, 0, 3), (1, 0, 4), (0, 1, 2), (2, 1, 4)):
        assert 3 + dt - dk == len(range(3 + dk, 6 + dt)) == expected
    assert rejected(lambda: require(3 + (6 - 3) == len(range(3, 6))))
    assert 1999 // 2000 == 0 and 2000 // 2000 == 1
    assert 1960 // 40 == 49 and 2000 // 40 == 50
    previous = {0, 1, 49}
    assert not previous.intersection({50, 51})
    assert rejected(lambda: require(not previous.intersection({49})))
    assert rejected(lambda: require(1 == 2))
    print('credits/frontiers:', cases, 'capacity cases; delayed release; idle/producer/consumer/both references; negatives detected')


def weighted(q, keys, values):
    scores = [q * k for k in keys]
    peak = max(scores)
    weights = [math.exp(s - peak) for s in scores]
    return sum(w * v for w, v in zip(weights, values)) / sum(weights)


def full_prefix_reference(inputs, layers=2):
    h = list(inputs)
    for _ in range(layers):
        keys, values = [0.7 * x for x in h], [0.5 * x for x in h]
        out = []
        for i, x in enumerate(h):
            a = x + weighted(0.3 * x, keys[:i + 1], values[:i + 1])
            out.append(a + 0.1 * math.tanh(a))
        h = out
    return h


class ScalarDecoder:
    def __init__(self, inputs, layers=2):
        assert inputs
        self.known = list(inputs)
        self.n = 0
        self.kv = [[] for _ in range(layers)]
        self.hidden = []
        self.logits = None

    def invariant(self):
        assert 0 <= self.n <= len(self.known)
        assert all(len(layer) == self.n for layer in self.kv)
        assert len(self.hidden) == self.n
        if self.logits is not None:
            position, hidden = self.logits
            assert position == self.n == len(self.known)
            assert hidden == self.hidden[-1]

    def consume(self, generation=True):
        assert self.n < len(self.known)
        h = self.known[self.n]
        pending = []
        for layer in self.kv:
            k, v = 0.7 * h, 0.5 * h
            keys = [x[0] for x in layer] + [k]
            values = [x[1] for x in layer] + [v]
            a = h + weighted(0.3 * h, keys, values)
            h = a + 0.1 * math.tanh(a)
            pending.append((k, v))
        for layer, pair in zip(self.kv, pending):
            layer.append(pair)
        self.hidden.append(h)
        self.n += 1
        self.logits = (self.n, h) if generation and self.n == len(self.known) else None
        self.invariant()
        reference = full_prefix_reference(self.known[:self.n], len(self.kv))
        assert all(abs(a - b) < 1e-12 for a, b in zip(reference, self.hidden))

    def append_known(self, value):
        assert self.n == len(self.known)
        self.known.append(value)
        self.logits = None
        self.invariant()


def check_decoder_and_tracks():
    d = ScalarDecoder([0.2, -0.4])
    d.consume()
    assert d.logits is None
    d.consume()
    old_n = d.n
    d.append_known(1.0)
    assert d.n == old_n and len(d.known) == old_n + 1
    for _ in range(5):
        d.consume()
        d.append_known(1.0)
    d.consume()
    assert rejected(d.consume)
    bad = ScalarDecoder([0.2])
    bad.consume()
    bad.append_known(1.0)
    for layer in bad.kv:
        layer.append((0.0, 0.0))
    assert rejected(bad.invariant)
    d = ScalarDecoder([0.25])
    d.consume()
    consumed, blocks, tail, text_track = 0, [], [], []
    for j, (needed, code) in enumerate(zip((1, 1, 2, 2, 2), range(10, 15))):
        while consumed < needed:
            d.append_known(0.1 * (consumed + 1))
            d.consume()
            consumed += 1
        assert d.logits is not None
        text_track.append('EOS_TEXT' if j == 0 else 'PAD_TEXT')
        old_n = d.n
        d.append_known(code * 0.01)
        assert d.n == old_n
        tail.append(code)
        if len(tail) == 2:
            blocks.append(tuple(tail))
            tail = []
        d.consume()
    if tail:
        blocks.append(tuple(tail))
    assert consumed == 2 and blocks == [(10, 11), (12, 13), (14,)]
    assert text_track == ['EOS_TEXT'] + ['PAD_TEXT'] * 4
    assert rejected(lambda: require([10, 11] == [x for b in blocks for x in b]))
    print('decoder/tracks: incremental KV versus layer-major full-prefix reference; 2 conditions/5 codes; selected-KV and length-coupling faults detected')


def window_ready(b, codes, final=False, left=True, right=True):
    assert b >= 0
    if b not in codes:
        return False
    if final:
        assert codes == set(range(len(codes)))
    for j in range(b - 2, b + 2):
        if j < 0:
            if not left:
                raise ValueError('missing left contract')
        elif j in codes:
            continue
        elif final and j >= len(codes):
            if not right:
                raise ValueError('missing right contract')
        else:
            return False
    return True


def check_windows():
    assert window_ready(0, {0, 1})
    assert not window_ready(0, {0})
    assert window_ready(0, {0}, final=True)
    assert window_ready(1, {0, 1, 2})
    assert window_ready(2, {0, 1, 2, 3})
    assert not window_ready(2, {1, 2, 3})
    assert rejected(lambda: window_ready(0, {0, 1}, left=False))
    assert rejected(lambda: window_ready(0, {0}, final=True, right=False))
    assert rejected(lambda: require(all(j in {0, 1} for j in (-2, -1, 0, 1))))
    print('windows: non-final first block, real lookahead, final right boundary; missing-contract/history negatives detected')


class Thinker:
    # Forced token labels are a deterministic fixture, not a sampling-quality test.
    def __init__(self, labels, limit):
        assert limit >= 0
        self.labels = tuple(labels)
        self.limit = limit
        self.d = ScalarDecoder([0.2])
        self.c = Credits()
        self.done = False
        self.closed = False
        self.final = None
        self.reason = None
        self.epoch = 1
        self.tx = None
        self.conditions = []
        self.events = []
        self.selections = 0
        if limit == 0:
            self.finish('LENGTH')
        else:
            self.d.consume()

    def finish(self, reason):
        if self.done:
            return
        self.done = self.closed = True
        self.reason = reason
        self.final = self.c.t
        self.tx = None
        self.c.reserved.clear()
        self.d.logits = None
        self.events.append(('TEXT_END', reason, self.final))

    def prepare(self, label, faulty=False):
        if not faulty and (self.done or self.closed):
            return False
        if self.tx is not None:
            return True
        assert self.d.logits is not None
        if not self.c.reserve(1):
            return False
        position, h = self.d.logits
        value = 0.0 if label == 'EOS' else 1.0
        self.tx = (label, value, position - 1, h, self.epoch)
        self.selections += 1
        return True

    def commit(self, faulty=False):
        assert self.tx is not None
        label, value, i, h, epoch = self.tx
        if not faulty and (self.done or self.closed or epoch != self.epoch):
            self.tx = None
            self.c.reserved.clear()
            self.d.logits = None
            return False
        assert self.d.logits is not None
        assert (i + 1, h) == self.d.logits
        self.d.append_known(value)
        self.c.commit_one()
        self.conditions.append((i, h, label))
        if label != 'EOS':
            self.events.append(('TEXT', label))
        self.tx = None
        if label == 'EOS' or self.c.t >= self.limit:
            self.finish('EOS' if label == 'EOS' else 'LENGTH')
        return True

    def step(self, faulty=False):
        if self.done and not faulty:
            return False
        if self.tx is not None:
            return self.commit(faulty)
        if self.d.logits is not None:
            label = self.labels[self.c.t] if self.c.t < len(self.labels) else 'after'
            return self.prepare(label, faulty)
        if self.d.n < len(self.d.known):
            self.d.consume()
            return True
        return False

    def snapshot(self):
        return (self.c.t, tuple(self.conditions), tuple(self.events),
                len(self.d.known), self.selections, self.reason, self.final)

    def invariant(self):
        self.c.invariant()
        self.d.invariant()
        assert len(self.conditions) == self.c.t
        ends = [e for e in self.events if e[0] == 'TEXT_END']
        if self.done:
            assert self.closed and self.final == self.c.t
            assert len(ends) == 1 and self.events[-1] == ends[0]
            assert self.tx is None and self.d.logits is None
            assert not self.c.reserved
        else:
            assert not self.closed and self.final is None and not ends


def termination_reference(labels, limit):
    selected, events = [], []
    reason = 'LENGTH'
    for label in labels[:limit]:
        selected.append(label)
        if label != 'EOS':
            events.append(('TEXT', label))
        if label == 'EOS':
            reason = 'EOS'
            break
    assert limit == 0 or selected
    events.append(('TEXT_END', reason, len(selected)))
    return selected, events, reason


def disposition(complete, runnable, enough_budget, wakes, faulty=False):
    if complete:
        return 'COMPLETE'
    if runnable:
        return 'RUN' if enough_budget else 'PAUSED_BUDGET'
    if faulty:
        return 'WAIT'
    return 'WAIT' if wakes else 'FAILED_CONTRACT_DEADLOCK'


class Drain:
    # One logical code per frame, two frames per block; synthetic acoustic work.
    def __init__(self, thinker, demand):
        self.t = thinker
        self.demand = tuple(demand)
        self.frames = []
        self.tail = []
        self.codes = set()
        self.audio_done = False
        self.next_decode = 0
        self.pcm = []
        self.sent_pcm = []
        self.text_cursor = 0
        self.source_ready = True
        self.trace = []

    def complete(self):
        return (self.t.done and self.t.closed and self.audio_done
                and self.t.c.k == self.t.c.t
                and self.next_decode == len(self.codes)
                and not self.pcm and self.text_cursor == len(self.t.events))

    def step(self, writable=True, faulty=None):
        if faulty == 'source_early_return' and not self.source_ready:
            return 'WAIT'
        if (faulty == 'window_early_return'
                and not window_ready(self.next_decode, self.codes, self.audio_done)):
            return 'WAIT'
        events = []
        if writable and self.text_cursor < len(self.t.events):
            self.text_cursor += 1
            events.append('send_text')
        if writable and self.pcm:
            self.sent_pcm.append(self.pcm.pop(0))
            events.append('send_pcm')
        if self.complete():
            self.trace.append(tuple(events))
            return 'COMPLETE'
        if window_ready(self.next_decode, self.codes, self.audio_done):
            self.pcm.append(self.next_decode)
            self.next_decode += 1
            events.append('decode')
        if not self.audio_done:
            j = len(self.frames)
            needed = self.demand[j] if j < len(self.demand) else self.t.c.t
            if self.t.c.k < needed and self.t.c.queue:
                self.t.c.consume()
                events.append('condition')
            elif j < len(self.demand) and self.t.c.k >= needed:
                self.frames.append(j)
                self.tail.append(j)
                if len(self.tail) == 2:
                    self.codes.add(len(self.codes))
                    self.tail = []
                events.append('code')
            elif j == len(self.demand) and self.t.done and self.t.c.k == self.t.c.t:
                if self.tail:
                    self.codes.add(len(self.codes))
                    self.tail = []
                self.audio_done = True
                events.append('audio_eos')
        if self.source_ready and self.t.step():
            events.append('thinker')
        self.t.invariant()
        self.trace.append(tuple(events))
        if self.complete():
            return disposition(True, False, False, set())
        if events:
            return 'PROGRESSED'
        wakes = set()
        if not writable and (self.pcm or self.text_cursor < len(self.t.events)):
            wakes.add('SOCKET_WRITABLE')
        if not self.source_ready and not self.t.done:
            wakes.add('SOURCE_READY')
        return disposition(False, False, True, wakes)


def check_termination_and_drain():
    observations = []
    for labels, limit, demand in (
        (('word', 'EOS', 'after'), 10, (1, 1, 2, 2, 2)),
        (('last', 'after'), 1, (1, 1, 1, 1, 1)),
    ):
        t = Thinker(labels, limit)
        s = Drain(t, demand)
        expected_labels, expected_events, reason = termination_reference(labels, limit)
        frozen = None
        stop_round = None
        n_at_stop = None
        for round_no in range(80):
            status = s.step(writable=(round_no % 3 == 0))
            if t.done and frozen is None:
                frozen, stop_round, n_at_stop = t.snapshot(), round_no, t.d.n
                s.source_ready = False
            if frozen is not None:
                assert t.snapshot() == frozen and t.d.n == n_at_stop
                assert t.c.t == len(expected_labels)
                assert t.events == expected_events
                assert [c[2] for c in t.conditions] == expected_labels
            if status == 'COMPLETE':
                assert round_no - stop_round >= 3
                break
            assert status != 'FAILED_CONTRACT_DEADLOCK'
        else:
            raise AssertionError('drain did not complete')
        assert t.reason == reason and s.sent_pcm == [0, 1, 2]
        assert len(s.frames) == 5 and t.c.k == t.c.t
        for _ in range(8):
            assert s.step() == 'COMPLETE'
            assert t.snapshot() == frozen and t.d.n == n_at_stop
        # Optional cache completion may add KV, but cannot reopen generation.
        assert t.d.n < len(t.d.known)
        t.d.consume(generation=False)
        assert t.snapshot() == frozen and t.d.logits is None
        assert not t.prepare('after') and not t.step()
        t.invariant()
        # A stale restored transaction must also fail at the commit gate.
        t.tx = ('after', 1.0, 0, 0.0, t.epoch)
        t.c.reserved = [2]
        assert t.commit() is False
        assert t.snapshot() == frozen
        t.invariant()
        observations.append((reason, t.c.t, len(s.frames), len(s.sent_pcm)))

    zero = Thinker(('after',), 0)
    zero.invariant()
    assert zero.c.t == 0 and zero.d.n == 0
    assert zero.events == [('TEXT_END', 'LENGTH', 0)]
    assert not zero.step()
    for label, limit in (('EOS', 10), ('last', 1)):
        bad = Thinker((label, 'after'), limit)
        assert bad.prepare(label) and bad.commit()
        frozen = bad.snapshot()
        # Reproduce the reviewed bad order: consume final token, select, publish.
        for _ in range(3):
            bad.step(faulty=True)
        assert rejected(lambda: require(bad.snapshot() == frozen))
        assert rejected(bad.invariant)
    t = Thinker(('word',), 3)
    assert t.prepare('word')
    selected = t.selections
    assert t.prepare('word') and t.selections == selected
    t.epoch += 1
    assert not t.commit() and t.c.t == 0 and not t.events
    t.invariant()
    print('termination:', observations, '; EOS/LENGTH frozen during multi-round drain; zero limit, STATE_ONLY, selected resume and commit gates; missing-stop negatives detected')


def check_scheduler():
    assert disposition(False, False, True, set()) == 'FAILED_CONTRACT_DEADLOCK'
    assert disposition(False, False, True, {'DEVICE_DONE'}) == 'WAIT'
    assert disposition(False, True, False, set()) == 'PAUSED_BUDGET'
    assert disposition(True, True, False, set()) == 'COMPLETE'
    assert disposition(False, False, False, set()) == 'FAILED_CONTRACT_DEADLOCK'
    cases = 0
    for complete in (False, True):
        for runnable in (False, True):
            for enough in (False, True):
                for wakes in (set(), {'DEVICE_DONE'}):
                    status = disposition(complete, runnable, enough, wakes)
                    assert status != 'WAIT' or wakes
                    if complete:
                        assert status == 'COMPLETE'
                    elif runnable:
                        assert status == ('RUN' if enough else 'PAUSED_BUDGET')
                    else:
                        assert status == ('WAIT' if wakes else 'FAILED_CONTRACT_DEADLOCK')
                    cases += 1
    assert rejected(lambda: require(
        disposition(False, False, True, set(), faulty=True) == 'FAILED_CONTRACT_DEADLOCK'))
    # Existing downstream work must survive source and window waits.
    t = Thinker(('word', 'EOS'), 10)
    assert t.prepare('word') and t.commit()
    s = Drain(t, (1, 1))
    s.source_ready = False
    assert s.step() == 'PROGRESSED' and t.c.k == 1
    s.pcm = [99]
    s.codes = {0}
    assert s.step() == 'PROGRESSED' and s.sent_pcm == [99]
    t = Thinker(('word',), 10)
    t.prepare('word')
    t.commit()
    bad = Drain(t, (1, 1))
    bad.source_ready = False
    bad.step(faulty='source_early_return')
    assert rejected(lambda: require(t.c.k == 1))
    bad = Drain(Thinker(('word',), 10), (1,))
    bad.pcm = [99]
    bad.codes = {0}
    bad.step(faulty='window_early_return')
    assert rejected(lambda: require(bad.sent_pcm == [99]))
    print('scheduler:', cases, 'decision combinations; COMPLETE/budget/event wait/deadlock priorities; bad WAIT order and local early returns detected')


def play_events(initial, arrivals, end):
    if initial < 0 or end < 0:
        raise ValueError('negative buffer or horizon')
    b, u, last = initial, 0, 0
    trace = []
    for at, amount in list(arrivals) + [(end, 0)]:
        if not last <= at <= end or amount < 0:
            raise ValueError('invalid arrival')
        dt = at - last
        played = min(b, dt)
        b -= played
        u += dt - played
        before = b
        b += amount
        trace.append((at, before, b, u))
        last = at
    return b, u, trace


def play_tick_reference(initial, arrivals, end):
    # Independent reference: add events at each integer clock boundary, then
    # consume one unit during the following unit interval, or count one outage.
    arrivals = tuple(arrivals)
    assert initial >= 0 and end >= 0
    assert all(0 <= at <= end and amount >= 0 for at, amount in arrivals)
    q, outage = initial, 0
    for clock in range(end + 1):
        q += sum(amount for at, amount in arrivals if at == clock)
        if clock < end:
            if q:
                q -= 1
            else:
                outage += 1
    return q, outage


def faulty_aggregate(initial, arrivals, end):
    total = initial + sum(amount for _, amount in arrivals)
    return max(0, total - end), max(0, end - total)


def check_playout():
    early = [(0, 10)]
    late = [(9, 10)]
    assert play_events(0, early, 10)[:2] == (0, 0)
    assert play_events(0, late, 10)[:2] == (9, 9)
    assert sum(a for _, a in early) == sum(a for _, a in late)
    assert rejected(lambda: require(faulty_aggregate(0, late, 10) == play_tick_reference(0, late, 10)))
    cases = 0
    for initial in range(4):
        for at in range(11):
            for amount in range(13):
                events = [(at, amount)]
                b, u, _ = play_events(initial, events, 10)
                assert (b, u) == play_tick_reference(initial, events, 10)
                assert b == initial + amount - 10 + u
                cases += 1
    multi_cases = 0
    for first in range(7):
        for second in range(first, 7):
            for a in range(4):
                for b in range(4):
                    events = [(first, a), (second, b)]
                    out = play_events(2, events, 6)
                    assert out[:2] == play_tick_reference(2, events, 6)
                    assert out[0] == 2 + a + b - 6 + out[1]
                    multi_cases += 1
    assert play_events(3, [], 5)[:2] == (0, 2)
    assert play_events(0, [(10, 10)], 10)[:2] == (10, 10)
    assert play_events(0, [(0, 3), (0, 2)], 0)[:2] == (5, 0)
    assert rejected(lambda: play_events(0, [(3, 1), (2, 1)], 5))
    assert rejected(lambda: play_events(0, [(6, 1)], 5))
    assert rejected(lambda: play_events(0, [(1, -1)], 5))
    # Aggregation is valid when all audio is available at the start.
    for initial in range(4):
        for amount in range(5):
            assert play_events(initial, [(0, amount)], 6)[:2] == faulty_aggregate(initial, [(0, amount)], 6)
    # It is also valid without clipping after a no-underrun event proof.
    full = play_events(10, [(3, 2), (8, 1)], 10)
    assert full[:2] == (3, 0)
    seconds = Fraction(9, 10)
    assert seconds * 24000 * 2 == 43200
    print('playout:', cases, 'single-arrival and', multi_cases,
          'two-arrival integer-clock references; early=(0,0), late=(0.9,0.9) seconds; unconditional aggregate fault detected')


def main():
    check_alignment()
    check_credits_and_frontiers()
    check_decoder_and_tracks()
    check_windows()
    check_termination_and_drain()
    check_scheduler()
    check_playout()
    print('scope: deterministic CPU synthetic state/arithmetic models; representative decoder and boundary contracts; no GPU, real Omni, network, audio-quality or learner-effect measurement')


if __name__ == '__main__':
    main()
