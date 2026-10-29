import codecs
import itertools


class Stream:
    """CPU logical model; output is an audit log, not a bounded send queue."""
    def __init__(self, stops=(), include=False):
        for p in stops:
            if not isinstance(p, str) or not p:
                raise ValueError('stops must be nonempty Unicode strings')
            p.encode('utf-8', 'strict')
        self.stops = tuple(dict.fromkeys(stops))
        self.include = include
        self.width = max((len(p) for p in self.stops), default=0)
        self.decoder = codecs.getincrementaldecoder('utf-8')('strict')
        self.pending = ''
        self.output = ''
        self.phase = 'running'
        self.reason = None
        self.matched = None

    def _character(self, ch):
        self.pending += ch
        hits = [p for p in self.stops if self.pending.endswith(p)]
        if hits:
            p = min(hits, key=lambda x: (-len(x), self.stops.index(x)))
            self.output += self.pending[:-len(p)]
            if self.include:
                self.output += p
            self.pending = ''
            self.phase = 'stopped'
            self.reason = 'stop'
            self.matched = p
            return
        keep = 0
        for p in self.stops:
            for n in range(1, len(p)):
                if self.pending.endswith(p[:n]):
                    keep = max(keep, n)
        cut = len(self.pending) - keep
        self.output += self.pending[:cut]
        self.pending = self.pending[cut:]

    def feed(self, piece):
        if not isinstance(piece, bytes):
            raise TypeError('bytes required')
        if self.phase == 'failed':
            raise RuntimeError('failed stream')
        if self.phase != 'running':
            return ''
        before = len(self.output)
        try:
            for byte in piece:
                chars = self.decoder.decode(bytes((byte,)), final=False)
                for ch in chars:
                    self._character(ch)
                    if self.phase != 'running':
                        break
                if self.phase != 'running':
                    break
        except UnicodeError:
            self.phase = 'failed'
            raise
        return self.output[before:]

    def finish(self, reason='eos'):
        if reason not in ('eos', 'length'):
            raise ValueError('unsupported normal finish')
        if self.phase == 'failed':
            raise RuntimeError('failed stream')
        if self.phase != 'running':
            return ''
        before = len(self.output)
        try:
            for ch in self.decoder.decode(b'', final=True):
                self._character(ch)
                if self.phase != 'running':
                    break
        except UnicodeError:
            self.phase = 'failed'
            raise
        if self.phase == 'running':
            self.output += self.pending
            self.pending = ''
            self.phase = 'finished'
            self.reason = reason
        return self.output[before:]

    def cancel(self):
        if self.phase == 'running':
            self.pending = ''
            self.decoder.reset()
            self.phase = 'cancelled'
            self.reason = 'cancelled'

    def result(self):
        return self.output, self.reason, self.matched


def reference(data, stops, include=False, reason='eos'):
    # Independent whole-string oracle: order complete occurrences by their
    # completion position, then start position, then configured pattern order.
    text = data.decode('utf-8', 'strict')
    hits = []
    for order, pattern in enumerate(stops):
        start = text.find(pattern)
        if start >= 0:
            hits.append((start + len(pattern), start, order, pattern))
    if not hits:
        return text, reason, None
    end, start, _, pattern = min(hits)
    return text[:end if include else start], 'stop', pattern


def partitions(data):
    if not data:
        yield ()
        return
    for bits in range(1 << (len(data) - 1)):
        chunks = []
        previous = 0
        for cut in range(1, len(data)):
            if bits & (1 << (cut - 1)):
                chunks.append(data[previous:cut])
                previous = cut
        chunks.append(data[previous:])
        yield tuple(chunks)


def equal(actual, expected):
    if actual != expected:
        raise AssertionError((actual, expected))


def expect_exception(kind, operation):
    try:
        operation()
    except kind:
        return
    raise AssertionError('expected ' + kind.__name__)


def exercise(data, stops, include=False, reason='eos'):
    expected = reference(data, stops, include, reason)
    count = 0
    for chunks in partitions(data):
        stream = Stream(stops, include)
        for chunk in chunks:
            stream.feed(chunk)
            if not expected[0].startswith(stream.output):
                raise AssertionError(('intermediate leak', chunks, stream.result(), expected))
            if len(stream.pending) > max(stream.width - 1, 0):
                raise AssertionError('unbounded stop suffix')
            if len(stream.decoder.getstate()[0]) > 3:
                raise AssertionError('unbounded UTF-8 suffix')
        stream.finish(reason)
        equal(stream.result(), expected)
        equal(stream.finish(reason), '')
        stream.feed(b'ignored after terminal')
        equal(stream.result(), expected)
        count += 1
    return count


def last_valid(mask):
    for index in range(len(mask) - 1, -1, -1):
        if mask[index]:
            return index
    raise ValueError('no valid row')


def main():
    comparisons = 0
    pool = ('a', 'b', 'ab', 'ba', 'aba')
    for length in range(6):
        for letters in itertools.product('ab', repeat=length):
            data = ''.join(letters).encode('ascii')
            for bits in range(1 << len(pool)):
                stops = tuple(p for j, p in enumerate(pool) if bits & (1 << j))
                for include in (False, True):
                    comparisons += exercise(data, stops, include)

    cases = (
        ('abcd', ('abcd', 'bc')),
        ('xabc', ('ab', 'abc')),
        ('abc', ('bc', 'abc')),
        ('ababa', ('aba', 'ba')),
        ('helloST', ('STOP',)),
        ('STOPtail', ('STOP',)),
        ('你🙂ab尾', ('🙂ab', 'ab')),
        ('A你!', ('你!',)),
        ('é你', ()),
        ('', ()),
        ('ab', ('ab', 'ab')),
    )
    for text, stops in cases:
        for include in (False, True):
            comparisons += exercise(text.encode('utf-8'), stops, include)
    comparisons += exercise('A你'.encode('utf-8'), ('你!',), reason='length')

    expect_exception(ValueError, lambda: Stream(('',)))
    invalid = Stream()
    expect_exception(UnicodeDecodeError, lambda: invalid.feed(b'\xff'))
    equal(invalid.phase, 'failed')
    truncated = Stream()
    truncated.feed(b'\xe4')
    expect_exception(UnicodeDecodeError, truncated.finish)
    equal(truncated.phase, 'failed')

    # Once stop is confirmed, later bytes are outside the accepted text stream.
    trailing = Stream(('END',))
    trailing.feed(b'END\xff')
    equal(trailing.result(), ('', 'stop', 'END'))

    cancelled = Stream(('你!',))
    cancelled.feed('A你'.encode('utf-8') + b'\xe4')
    equal(cancelled.output, 'A')
    cancelled.cancel()
    cancelled.feed(b'\xff')
    equal(cancelled.finish(), '')
    equal(cancelled.result(), ('A', 'cancelled', None))
    equal(cancelled.pending, '')

    # Negative control 1: a whole-chunk leftmost-start matcher chooses abcd.
    data = b'abcd'
    stops = ('abcd', 'bc')
    expected = reference(data, stops)
    text = data.decode('ascii')
    starts = [(text.find(p), j, p) for j, p in enumerate(stops) if p in text]
    start, _, wrong_pattern = min(starts)
    faulty = (text[:start], 'stop', wrong_pattern)
    expect_exception(AssertionError, lambda: equal(faulty, expected))

    # Negative control 2: independently decoding byte fragments drops a codepoint.
    pieces = (b'\xe4', b'\xbd', b'\xa0', b'!')
    faulty_text = ''.join(p.decode('utf-8', errors='ignore') for p in pieces)
    expected_text = reference(b''.join(pieces), ())[0]
    expect_exception(AssertionError, lambda: equal(faulty_text, expected_text))

    # Negative control 3: final truncation can conceal an earlier visible leak.
    expected_text = reference(b'helloSTOPtail', ('STOP',))[0]
    leaked_intermediate = 'helloST'
    expect_exception(AssertionError,
                     lambda: equal(expected_text.startswith(leaked_intermediate), True))

    masks = ((1, 0, 0), (1, 1, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))
    for mask in masks:
        independent_rows = [i for i, value in enumerate(mask) if value == 1]
        equal(last_valid(mask), max(independent_rows))
    expect_exception(ValueError, lambda: last_valid(()))
    expect_exception(ValueError, lambda: last_valid((0, 0)))
    expect_exception(AssertionError,
                     lambda: equal(len(masks[0]) - 1, last_valid(masks[0])))

    print('scope: deterministic offline CPU logical models; Python standard library only')
    print('stream partition comparisons:', comparisons)
    print('checked: first-completion stops, intermediate prefixes, UTF-8, length flush, cancel')
    print('checked: empty/duplicate patterns, terminal idempotence, invalid/truncated UTF-8')
    print('detected negative controls: chunk-order mismatch, dropped Unicode, stop leak, padding row')
    print('limits: no real tokenizer, model forward, GPU, network, timing or learner-effect measurements')


if __name__ == '__main__':
    main()
