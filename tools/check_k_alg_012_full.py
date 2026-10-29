#!/usr/bin/env python3
'''K-ALG-012: deterministic, offline, read-only CPU logic checks.

No project/upstream imports, files, subprocesses, network or GPU measurements.
The parser supports the finite-closure grammars used here, not arbitrary CFGs.
'''
import itertools
import math
from fractions import Fraction

EOS = '<EOS>'
PROMPT = '<PROMPT>'

class Failure(ValueError):
    pass

class Parser:
    def __init__(self, productions, start='S'):
        productive = set()
        while True:
            added = {
                nt for nt, choices in productions.items()
                if any(all(s not in productions or s in productive for s in rhs)
                       for rhs in choices)
            }
            if added <= productive:
                break
            productive.update(added)
        self.rules = {
            nt: tuple(rhs for rhs in choices
                      if all(s not in productions or s in productive for s in rhs))
            for nt, choices in productions.items() if nt in productive
        }
        self.initial = self.closure({(start,)}) if start in productive else frozenset()

    def closure(self, stacks):
        work = list(stacks)
        seen = set()
        while work:
            st = work.pop()
            if st in seen:
                continue
            seen.add(st)
            if st and st[-1] in self.rules:
                for rhs in self.rules[st[-1]]:
                    work.append(st[:-1] + tuple(reversed(rhs)))
        return frozenset(seen)

    def advance(self, states, text):
        cur = frozenset(states)
        for ch in text:
            cur = self.closure({st[:-1] for st in cur if st and st[-1] == ch})
            if not cur:
                break
        return cur

    @staticmethod
    def accepting(states):
        return () in states

    def mask(self, states, tokens):
        result = {t for t, text in tokens.items()
                  if text and self.advance(states, text)}
        if self.accepting(states):
            result.add(EOS)
        return result

PRODUCTIONS = {
    'S': (('[', 'E', ']'),),
    'E': ((), ('V',), ('V', ',', 'V')),
    'V': (('a',), ('b',), ('[', 'F', ']')),
    'F': ((), ('W',), ('W', ',', 'W')),
    'W': (('a',), ('b',)),
}

# Independent language construction: no parser transitions or grammar expansion.
def finite_language():
    atoms = ('a', 'b')
    inner_contents = ('',) + atoms + tuple(a + ',' + b for a in atoms for b in atoms)
    values = atoms + tuple('[' + x + ']' for x in inner_contents)
    contents = ('',) + values + tuple(a + ',' + b for a in values for b in values)
    return frozenset('[' + x + ']' for x in contents)

TOKENS = ('[', ']', 'a', 'b', ',', '[]', '[a', 'a]', ']]', 'a,]',
          'a,', '[[a', 'a,b]', ']]]', 'b]]', 'x', '[a,b]')

class ToyModel:
    '''A prefix-indexed score table; computed tracks the model-input frontier.'''
    def __init__(self, tokens, table, default=-20.0):
        self.tokens = tokens
        self.table = table
        self.default = default
        self.computed = ()
        self.calls = []

    def forward(self, history):
        history = tuple(history)
        if history[:len(self.computed)] != self.computed:
            raise AssertionError('model frontier no longer matches its history')
        if len(history) <= len(self.computed):
            raise AssertionError('next logits require a new model production event')
        self.computed = history
        self.calls.append(history)
        scores = self.table.get(history[1:], {})
        return {t: float(scores.get(t, self.default))
                for t in tuple(self.tokens) + (EOS,)}

def generate(parser, tokens, model, budget, prefill=(),
             stale_logits=False, accepting_only=False, late_eos=False):
    if budget < 0 or EOS in tokens or any(not x for x in tokens.values()):
        raise Failure('invalid invocation')
    states = parser.initial
    text = ''
    for t in prefill:
        text += tokens[t]
        states = parser.advance(states, tokens[t])
    if not states:
        raise Failure('invalid prefix')
    history = (PROMPT,) + tuple(prefill)
    sampled = []
    frontiers = []
    first_logits = None
    order = tuple(tokens) + (EOS,)
    for _ in range(budget):
        legal = parser.mask(states, tokens)
        if accepting_only and parser.accepting(states):
            legal = {EOS}  # Intentionally faulty negative control.
        if not legal:
            raise Failure('empty mask')
        scores = model.forward(history)
        if first_logits is None:
            first_logits = dict(scores)
        if stale_logits:
            scores = first_logits  # Intentionally faulty negative control.
        if any(math.isnan(scores[t]) or scores[t] == math.inf for t in legal):
            raise Failure('invalid numeric scores')
        available = [t for t in order if t in legal and math.isfinite(scores[t])]
        if not available:
            raise Failure('zero legal mass')
        # Greedy specialization; first in declared token order wins a tie.
        chosen = max(available, key=lambda t: scores[t])
        sampled.append(chosen)
        if late_eos:
            parser.advance(states, tokens[chosen])  # Deliberate EOS-dispatch bug.
        if chosen == EOS:
            if not parser.accepting(states):
                raise AssertionError('nonaccepting EOS')
            frontiers.append((len(history), len(model.computed)))
            return {'status': 'complete', 'text': text, 'accepting': True,
                    'sampled': sampled, 'frontiers': frontiers}
        states = parser.advance(states, tokens[chosen])
        if not states:
            raise AssertionError('mask accepted an invalid ordinary token')
        text += tokens[chosen]
        history += (chosen,)
        frontiers.append((len(history), len(model.computed)))
    return {'status': 'length_limit', 'text': text,
            'accepting': parser.accepting(states),
            'sampled': sampled, 'frontiers': frontiers}

def restrict(weights, legal):
    if not legal:
        raise Failure('empty mask')
    total = sum((weights[t] for t in legal), Fraction(0))
    if total <= 0:
        raise Failure('zero legal mass')
    return {t: weights[t] / total if t in legal else Fraction(0) for t in weights}

def expect_failure(fn, message):
    try:
        fn()
    except Failure as exc:
        if message not in str(exc):
            raise AssertionError('wrong failure reason: ' + str(exc))
    else:
        raise AssertionError('expected failure: ' + message)

def expect_detected(fn, label, exception=AssertionError):
    try:
        fn()
    except exception:
        print('negative control detected:', label)
    else:
        raise AssertionError('negative control escaped: ' + label)

def merge_masks(vocab, represented):
    partial_acc = set()
    partial_rej = set(vocab)
    for kind, stored in represented:
        if kind == 'rej':
            partial_rej.intersection_update(stored)
        else:
            partial_acc.update(stored)
    return set(vocab) - (partial_rej - partial_acc)

def check_masks():
    parser = Parser(PRODUCTIONS)
    accepted = finite_language()
    prefixes = {s[:i] for s in accepted for i in range(len(s) + 1)}
    assert len(accepted) == 91 and len(prefixes) == 282
    tested = prefixes | {'x', ']', '[a,]', '[]x'}
    count = 0
    faulty_count = 0
    for prefix in sorted(tested):
        states = parser.advance(parser.initial, prefix)
        assert parser.accepting(states) == (prefix in accepted), prefix
        for token in TOKENS:
            expected = any(s.startswith(prefix + token) for s in accepted)
            actual = bool(parser.advance(states, token))
            assert actual == expected, (prefix, token, expected, actual)
            first_only = bool(parser.advance(states, token[:1]))
            faulty_count += first_only != expected
            count += 1
    assert faulty_count > 0
    assert not parser.advance(parser.advance(parser.initial, '[a'), ']]')
    assert parser.advance(parser.advance(parser.initial, '[[a'), ']]')
    assert not any(s.startswith('a,b]') for s in accepted)
    assert any(s.startswith('[' + 'a,b]') for s in accepted)
    mapping = {t: t for t in TOKENS}
    assert parser.mask(parser.advance(parser.initial, '[]'), mapping) == {EOS}
    print('finite grammar: accepted=%d prefixes=%d unique_tokens=%d comparisons=%d'
          % (len(accepted), len(prefixes), len(TOKENS), count))
    print('state-aware first-character negative-control disagreements:', faulty_count)
    return parser

def check_merge():
    vocab = frozenset(range(3))
    subsets = [frozenset(c) for n in range(4)
               for c in itertools.combinations(sorted(vocab), n)]
    count = 0
    for a, b in itertools.product(subsets, repeat=2):
        for ka, kb in itertools.product(('acc', 'rej'), repeat=2):
            stored_a = a if ka == 'acc' else vocab - a
            stored_b = b if kb == 'acc' else vocab - b
            observed = merge_masks(vocab, ((ka, stored_a), (kb, stored_b)))
            assert observed == a | b
            count += 1
    print('mixed accepted/rejected representation merges:', count)

def check_generation(parser):
    tokens = {'open': '[', 'a': 'a', 'close': ']', 'empty': '[]', 'bad': 'a,]'}
    table = {
        (): {'open': 9, 'close': 8, 'a': 0, 'empty': 1},
        ('open',): {'a': 9, 'close': 8},
        ('open', 'a'): {'close': 9},
        ('open', 'a', 'close'): {EOS: 9},
        ('empty',): {EOS: 9},
    }
    def run(**kwargs):
        model = ToyModel(tokens, table)
        return generate(parser, tokens, model, 8, **kwargs), model
    def verify_main(pair):
        result, model = pair
        assert result['status'] == 'complete' and result['text'] == '[a]'
        assert result['sampled'] == ['open', 'a', 'close', EOS]
        assert model.calls == [(PROMPT,), (PROMPT, 'open'),
                               (PROMPT, 'open', 'a'),
                               (PROMPT, 'open', 'a', 'close')]
        assert result['frontiers'] == [(2, 1), (3, 2), (4, 3), (4, 4)]
    verify_main(run())
    expect_detected(lambda: verify_main(run(stale_logits=True)), 'stale first-prefix logits')
    expect_detected(lambda: run(late_eos=True), 'ordinary expansion before EOS dispatch', KeyError)
    result, model = run(prefill=('empty',))
    assert result['text'] == '[]' and result['sampled'] == [EOS]
    assert model.calls == [(PROMPT, 'empty')]
    assert result['frontiers'] == [(2, 2)]
    model = ToyModel(tokens, table)
    r = generate(parser, tokens, model, 0)
    assert r['status'] == 'length_limit' and not model.calls and not r['sampled']
    r = generate(parser, tokens, ToyModel(tokens, table), 1)
    assert r['status'] == 'length_limit' and r['text'] == '[' and not r['accepting']
    complete_text_table = {(): {'empty': 9, 'open': 1}}
    r = generate(parser, tokens, ToyModel(tokens, complete_text_table), 1)
    assert r['status'] == 'length_limit' and r['text'] == '[]' and r['accepting']
    expect_failure(lambda: generate(parser, tokens, ToyModel(tokens, table), 1,
                                   prefill=('bad',)), 'invalid prefix')
    expect_failure(lambda: generate(parser, tokens,
                                   ToyModel(tokens, {}, default=-math.inf), 1),
                   'zero legal mass')
    print('generation: prefix-specific logits, lagging KV frontier, EOS dispatch, prefill and length limits')

def check_accepting_extension():
    parser = Parser({'S': (('a',), ('a', 'b'))})
    tokens = {'a': 'a', 'b': 'b', 'x': 'x'}
    state = parser.advance(parser.initial, 'a')
    assert parser.mask(state, tokens) == {EOS, 'b'}
    table = {('a',): {'b': 9, EOS: 8}, ('a', 'b'): {EOS: 9}}
    def verify_result(result):
        assert result['status'] == 'complete' and result['text'] == 'ab'
        assert result['sampled'] == ['b', EOS]
    verify_result(generate(parser, tokens, ToyModel(tokens, table), 3, prefill=('a',)))
    expect_detected(lambda: verify_result(generate(
        parser, tokens, ToyModel(tokens, table), 3, prefill=('a',), accepting_only=True)),
        'accepting state incorrectly forced to EOS')
    unavailable = Parser({'S': (('c',),)})
    expect_failure(lambda: generate(unavailable, tokens, ToyModel(tokens, {}), 1), 'empty mask')
    print('accepting extension: a permits EOS and b; unrepresentable completion fails explicitly')

def check_productivity_and_bytes():
    parser = Parser({'S': (('a',), ('x', 'U')), 'U': (('z', 'U'),)})
    assert parser.mask(parser.initial, {'a': 'a', 'x': 'x', 'z': 'z'}) == {'a'}
    assert not parser.advance(parser.initial, 'x')
    impossible = Parser({'S': (('S',),)})
    assert not impossible.initial
    utf = Parser({'S': ((195, 169),)})
    tokens = {'lead': bytes((195,)), 'tail': bytes((169,)),
              'whole': bytes((195, 169)), 'wrong': bytes((195, 65))}
    assert utf.mask(utf.initial, tokens) == {'lead', 'whole'}
    partial = utf.advance(utf.initial, tokens['lead'])
    assert partial and not utf.accepting(partial)
    assert utf.mask(partial, tokens) == {'tail'}
    final = utf.advance(partial, tokens['tail'])
    assert utf.mask(final, tokens) == {EOS}
    print('productivity and bytes: dead recursive branch pruned; split UTF-8 prefix completes')

def check_probability():
    weights = {'a': Fraction(4, 10), 'b': Fraction(3, 10),
               'c': Fraction(2, 10), 'd': Fraction(1, 10)}
    observed = restrict(weights, {'a', 'c'})
    assert observed == {'a': Fraction(2, 3), 'b': 0, 'c': Fraction(1, 3), 'd': 0}
    assert sum(observed.values()) == 1
    assert observed['a'] / observed['c'] == weights['a'] / weights['c']
    expect_failure(lambda: restrict(weights, set()), 'empty mask')
    expect_failure(lambda: restrict({'a': Fraction(0), 'b': Fraction(1)}, {'a'}),
                   'zero legal mass')
    print('exact rational distribution: support, normalization, ratios and zero-mass failure')

def main():
    parser = check_masks()
    check_merge()
    check_generation(parser)
    check_accepting_extension()
    check_productivity_and_bytes()
    check_probability()
    print('PASS: scoped CPU logic checks; no production engine or GPU behavior measured')

if __name__ == '__main__':
    main()
