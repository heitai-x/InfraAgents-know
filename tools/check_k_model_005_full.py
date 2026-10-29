#!/usr/bin/env python3
"""K-MODEL-005: deterministic, read-only CPU arithmetic and state checks.
No files, network, subprocesses, project imports, GPU, or real model inference.
The two-layer decoder deliberately omits normalization and uses a toy FFN;
it checks cache/input/selection/resume relations, not a named model forward.
"""
import copy
import math


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def flat(x):
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(flat(item))
        return out
    return [x]


def difference(a, b):
    a, b = flat(a), flat(b)
    require(len(a) == len(b), 'comparison shape mismatch')
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def close(a, b):
    require(difference(a, b) < 1e-11, 'numerical reference mismatch')


def rejected(call):
    try:
        call()
    except (ValueError, AssertionError):
        return True
    return False


def mv(x, w):
    require(len(x) == len(w), 'projection input mismatch')
    return [sum(x[i] * w[i][j] for i in range(len(x)))
            for j in range(len(w[0]))]


def project(rows, wk, wv, calls=None):
    if calls is not None:
        calls.append(len(rows))
    return [mv(x, wk) for x in rows], [mv(x, wv) for x in rows]


def attention(q, keys, values, visible):
    require(len(keys) == len(values) == len(visible), 'attention shape mismatch')
    allowed = [i for i, yes in enumerate(visible) if yes]
    if not allowed:
        raise ValueError('empty visible set: wait or fail before cross decoding')
    scores = [sum(a * b for a, b in zip(q, k)) / math.sqrt(len(q)) for k in keys]
    peak = max(scores[i] for i in allowed)
    weights = [(i, math.exp(scores[i] - peak)) for i in allowed]
    total = sum(w for _, w in weights)
    return [sum(w * values[i][j] for i, w in weights) / total
            for j in range(len(values[0]))]


def linear_reference(x, w):
    # Separate explicit two-dimensional formula, including non-symmetric weights.
    return [x[0] * w[0][0] + x[1] * w[1][0],
            x[0] * w[0][1] + x[1] * w[1][1]]


def attention_reference(q, keys, values, last):
    # Independent small-input formula: no stable-softmax helper or mask helper.
    exps = [math.exp((q[0] * k[0] + q[1] * k[1]) / math.sqrt(2))
            for k in keys[:last + 1]]
    denom = math.fsum(exps)
    return [math.fsum(exps[i] * values[i][j] for i in range(last + 1)) / denom
            for j in range(2)]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def embedding(token, pos):
    return [0.05 * (token + 1) + 0.01 * pos,
            0.2 - 0.03 * token + 0.02 * pos]


def parameters():
    layers = []
    source = [[0.3, -0.2], [-0.1, 0.4], [0.2, 0.1]]
    for j in range(2):
        t = 0.02 * j
        p = {
            'q': [[0.2 + t, -0.1], [0.05, 0.3]],
            'k': [[0.1, 0.2 + t], [-0.15, 0.1]],
            'v': [[0.3, -0.1], [0.2 + t, 0.25]],
            'o': [[0.4, 0.1], [-0.1, 0.3]],
            'cq': [[0.2, -0.1], [0.1, 0.3 + t]],
            'co': [[0.3, 0.05], [0.1, 0.2]],
            'f1': [[0.2, -0.1], [0.1, 0.3]],
            'f2': [[0.25, 0.05], [-0.1, 0.2]],
        }
        p['ck'] = [linear_reference(x, [[0.4, 0.1], [0.2, 0.3]]) for x in source]
        p['cv'] = [linear_reference(x, [[0.2 + t, -0.1], [0.1, 0.4]]) for x in source]
        layers.append(p)
    return layers


HEAD = [[0.3, -0.2], [-0.1, 0.4]]
EOS = 4


def incremental(token, pos, caches, layers):
    require(all(len(k) == len(v) == pos for k, v in caches), 'duplicate or skipped input')
    h = embedding(token, pos)
    for cache, p in zip(caches, layers):
        q, k, v = mv(h, p['q']), mv(h, p['k']), mv(h, p['v'])
        cache[0].append(k)
        cache[1].append(v)
        h = add(h, mv(attention(q, cache[0], cache[1], [True] * (pos + 1)), p['o']))
        h = add(h, mv(attention(mv(h, p['cq']), p['ck'], p['cv'], [True] * 3), p['co']))
        f = [max(0.0, x) for x in mv(h, p['f1'])]
        h = add(h, mv(f, p['f2']))
    return mv(h, HEAD)


def full_prefix_reference(tokens, layers):
    # Layer-major full recomputation; it never reads an incremental cache.
    rows = [embedding(token, i) for i, token in enumerate(tokens)]
    expected_caches = []
    for p in layers:
        qs = [linear_reference(x, p['q']) for x in rows]
        ks = [linear_reference(x, p['k']) for x in rows]
        vs = [linear_reference(x, p['v']) for x in rows]
        expected_caches.append((ks, vs))
        output = []
        for i, h in enumerate(rows):
            a = attention_reference(qs[i], ks, vs, i)
            x = add(h, linear_reference(a, p['o']))
            c = attention_reference(linear_reference(x, p['cq']), p['ck'], p['cv'], 2)
            r = add(x, linear_reference(c, p['co']))
            f = [max(0.0, value) for value in linear_reference(r, p['f1'])]
            output.append(add(r, linear_reference(f, p['f2'])))
        rows = output
    return [linear_reference(h, HEAD) for h in rows], expected_caches


def initialize(prompt, limit=8):
    if not prompt:
        raise ValueError('model requires a nonempty BOS/task prefix')
    return {'prompt': list(prompt), 'known': list(prompt), 'out': [], 'n': 0,
            'cache': [([], []), ([], [])], 'logits': None, 'stop': None,
            'consumed': [], 'limit': limit}


def validate_progress(s):
    require(s['known'] == s['prompt'] + s['out'], 'known prefix ownership')
    require(0 <= s['n'] <= len(s['known']), 'computed length')
    require(all(len(k) == len(v) == s['n'] for k, v in s['cache']), 'all layer lengths')
    require(s['consumed'] == list(range(s['n'])), 'each input consumed exactly once')
    if s['logits'] is not None:
        require(s['n'] == len(s['known']) and s['n'] > 0, 'logits prediction position')


def validate_event(s, event, flushed, tail_ok):
    validate_progress(s)
    if event == 'END_OF_REQUEST':
        require(flushed and tail_ok and s['stop'] in ('EOS', 'LENGTH'), 'illegal terminal')


def run_slice(s, layers, budget, ready=True, flushed=True, tail_ok=True):
    spent = 0
    while True:
        validate_progress(s)
        if flushed and not tail_ok:
            return 'FAILED_TAIL'
        if len(s['out']) >= s['limit'] and s['stop'] is None:
            s['stop'] = 'LENGTH'
        if s['stop'] is not None:
            event = 'END_OF_REQUEST' if flushed else 'WAIT_SOURCE_FINAL'
            validate_event(s, event, flushed, tail_ok)
            return event
        if not ready:
            return 'WAIT_MORE_SOURCE'
        if spent >= budget:
            return 'PAUSED_BUDGET'
        if s['n'] < len(s['known']):
            pos = s['n']
            logits = incremental(s['known'][pos], pos, s['cache'], layers)
            s['n'] += 1
            s['consumed'].append(pos)
            s['logits'] = logits if s['n'] == len(s['known']) else None
        else:
            require(s['logits'] is not None, 'missing first/next token logits')
            # A deterministic toy grammar permits two content tokens followed by EOS.
            token = EOS if len(s['out']) == 2 else 2 + max(range(2), key=lambda j: s['logits'][j])
            s['known'].append(token)
            s['out'].append(token)
            s['logits'] = None
            if token == EOS:
                s['stop'] = 'EOS'
            elif len(s['out']) == s['limit']:
                s['stop'] = 'LENGTH'
        spent += 1


def validate_ledger(old, new):
    if new['version'] < old['version']:
        raise ValueError('stale source version')
    if new['committed'] < old['committed']:
        raise ValueError('global committed sample offset regressed')
    if not new['stable'].startswith(old['stable']):
        raise ValueError('stable text regressed')
    return new


def faulty_advance(old, new):
    # Reproduces the old missing cross-version offset guard.
    if new['version'] < old['version']:
        raise ValueError('stale source version')
    if new['version'] == old['version'] and new['committed'] < old['committed']:
        raise ValueError('same-version rollback')
    if not new['stable'].startswith(old['stable']):
        raise ValueError('text rollback')
    return dict(new)


def build_source(raw, process, encode, trace, faulty=False):
    if not raw and not faulty:
        return 'EMPTY_RAW'
    trace.append('processor')
    x, valid = process(raw)
    if not any(valid) and not faulty:
        return 'EMPTY_FEATURES'
    trace.append('encoder')
    e, valid_e = encode(x)
    if not any(valid_e) and not faulty:
        return 'EMPTY_ENCODER'
    trace.append('project')
    kept = [row for row, yes in zip(e, valid_e) if yes]
    project(kept, [[1, 0], [0, 1]], [[1, 0], [0, 1]])
    if not raw:
        return 'EMPTY_RAW'
    if not any(valid):
        return 'EMPTY_FEATURES'
    if not any(valid_e):
        return 'EMPTY_ENCODER'
    return 'READY'


def validate_source(status, trace):
    expected = {'EMPTY_RAW': [], 'EMPTY_FEATURES': ['processor'],
                'EMPTY_ENCODER': ['processor', 'encoder'],
                'READY': ['processor', 'encoder', 'project']}
    require(trace == expected[status], 'empty source reached a forbidden consumer')


def capacity(length, modules):
    # Per module: key width, value width, key bytes, value bytes.
    return length * sum(dk * bk + dv * bv for dk, dv, bk, bv in modules)


def main():
    rows = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    wk = [[1.0, 0.0], [0.5, 1.0]]
    wv = [[1.0, 0.5], [0.0, 1.0]]
    keys, values = project(rows, wk, wv)
    close(keys, [[1.0, 0.0], [0.5, 1.0], [1.5, 1.0]])
    close(values, [[1.0, 0.5], [0.0, 1.0], [1.0, 1.5]])
    # With q=(sqrt(2),0), scores are (1,1/2,3/2).
    # Relative weights are (a,1,a^2), a=exp(1/2).
    a = math.exp(0.5)
    denom = a + 1 + a * a
    expected = [(a + a * a) / denom, (0.5 * a + 1 + 1.5 * a * a) / denom]
    q = [math.sqrt(2), 0.0]
    actual = attention(q, keys, values, [True] * 3)
    close(actual, expected)
    require(rejected(lambda: close(keys[0], [1.0, 0.5])), 'old wrong key oracle undetected')
    causal_diff = difference(actual, attention(q, keys, values, [True, False, False]))
    require(causal_diff > 1e-5, 'causal-cross mutation not detected')
    close(attention(q, keys, values, [False, True, False]), [0.0, 1.0])
    require(rejected(lambda: attention(q, [], [], [])), 'empty attention accepted')
    require(rejected(lambda: attention(q, keys, values, [False] * 3)), 'all-masked accepted')
    self_ok = attention(rows[1], rows, rows, [True, True, False])
    self_bad = attention(rows[1], rows, rows, [True, True, True])
    require(difference(self_ok, self_bad) > 1e-5, 'future-self mutation not detected')

    calls = []
    old_k, old_v = project(rows, wk, wv, calls)
    suffix = [[0.5, 0.25]]
    new_k, new_v = project(suffix, wk, wv, calls)
    joined_k, joined_v = old_k + new_k, old_v + new_v
    require(calls == [3, 1], 'append reprojected old source rows')
    require(joined_k[0] is old_k[0] and joined_v[0] is old_v[0], 'old cache not preserved')
    full_k, full_v = project(rows + suffix, wk, wv)
    close(joined_k, full_k)
    close(joined_v, full_v)
    close(attention(q, joined_k, joined_v, [True] * 4),
          attention(q, full_k, full_v, [True] * 4))
    edited = [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]]
    edit_k, edit_v = project(edited, wk, wv)
    stale_diff = difference(actual, attention(q, edit_k, edit_v, [True] * 3))
    require(stale_diff > 1e-5, 'edited-source stale cache not detected')

    layers = parameters()
    tokens = [0, 1, 2, 3, 2]
    caches = [([], []), ([], [])]
    decoder_error = 0.0
    for pos, token in enumerate(tokens):
        got = incremental(token, pos, caches, layers)
        ref, expected_caches = full_prefix_reference(tokens[:pos + 1], layers)
        decoder_error = max(decoder_error, difference(got, ref[-1]))
        close(got, ref[-1])
        close(caches, expected_caches)
    require(rejected(lambda: incremental(2, 0, caches, layers)), 'duplicate prefix accepted')

    for prompt in ([0], [0, 1, 1]):
        continuous = initialize(prompt)
        require(run_slice(continuous, layers, 100) == 'END_OF_REQUEST', 'continuous decode')
        resumed = initialize(prompt)
        require(run_slice(resumed, layers, len(prompt)) == 'PAUSED_BUDGET', 'prefill pause')
        require(resumed['n'] == len(prompt) and resumed['logits'] is not None,
                'last prompt logits missing')
        require(resumed['out'] == [], 'selected token improperly counted during prefill')
        snapshot = copy.deepcopy(resumed)
        require(run_slice(resumed, layers, 0) == 'PAUSED_BUDGET', 'zero budget')
        require(resumed == snapshot, 'zero budget changed state')
        require(run_slice(resumed, layers, 1) == 'PAUSED_BUDGET', 'selection pause')
        require(len(resumed['out']) == 1 and resumed['n'] == len(prompt), 'selected token owns KV too early')
        bad = copy.deepcopy(resumed)
        bad['n'] += 1
        require(rejected(lambda: validate_progress(bad)), 'premature KV count undetected')
        event = None
        for _ in range(30):
            event = run_slice(resumed, layers, 1)
            if event == 'END_OF_REQUEST':
                break
        require(event == 'END_OF_REQUEST', 'resume has no terminating successor')
        require(resumed['out'] == continuous['out'], 'slice-dependent output')
        require(resumed['consumed'] == continuous['consumed'], 'slice-dependent consumption')
        close(resumed['cache'], continuous['cache'])
        require(resumed['n'] == len(resumed['known']) - 1, 'EOS need not be consumed')

    waiting = initialize([0, 1])
    require(run_slice(waiting, layers, 1, ready=False, flushed=False) == 'WAIT_MORE_SOURCE', 'source wait')
    pause = initialize([0, 1])
    require(run_slice(pause, layers, 1, flushed=True) == 'PAUSED_BUDGET', 'final-source budget pause')
    require(rejected(lambda: validate_event(pause, 'END_OF_REQUEST', True, True)),
            'faulty budget-as-terminal event accepted')
    early = initialize([0], limit=1)
    require(run_slice(early, layers, 100, flushed=False) == 'WAIT_SOURCE_FINAL', 'early target stop')
    require(run_slice(early, layers, 0, flushed=True) == 'END_OF_REQUEST', 'finalization after flush')
    require(early['stop'] == 'LENGTH', 'length reason lost')
    require(run_slice(initialize([0]), layers, 100, tail_ok=False) == 'FAILED_TAIL', 'tail contract')
    require(run_slice(initialize([0], limit=0), layers, 0) == 'END_OF_REQUEST', 'zero generation limit')
    require(rejected(lambda: initialize([])), 'empty BOS/task prefix accepted')

    old = {'version': 2, 'committed': 800, 'stable': 'alpha beta'}
    invalid = [
        {'version': 1, 'committed': 800, 'stable': 'alpha beta'},
        {'version': 2, 'committed': 799, 'stable': 'alpha beta'},
        {'version': 3, 'committed': 799, 'stable': 'alpha beta'},
        {'version': 2, 'committed': 800, 'stable': 'alpha'},
    ]
    for candidate in invalid:
        require(rejected(lambda c=candidate: validate_ledger(old, c)), 'ledger violation accepted')
    validate_ledger(old, dict(old))
    validate_ledger(old, {'version': 3, 'committed': 900, 'stable': 'alpha beta gamma'})
    faulty = faulty_advance(old, invalid[2])
    require(rejected(lambda: validate_ledger(old, faulty)), 'faulty ledger escaped common validator')

    def process(raw):
        return [[1.0, 0.0]], [True]

    def process_empty(raw):
        return [[1.0, 0.0]], [False]

    def encode(x):
        return x, [True] * len(x)

    def encode_empty(x):
        return x, [False] * len(x)

    for raw, proc, enc in [([], process, encode), ([1], process_empty, encode),
                           ([1], process, encode_empty), ([1], process, encode)]:
        trace = []
        status = build_source(raw, proc, enc, trace)
        validate_source(status, trace)
    for raw, proc, enc in [([], process, encode), ([1], process_empty, encode),
                           ([1], process, encode_empty)]:
        trace = []
        status = build_source(raw, proc, enc, trace, faulty=True)
        require(rejected(lambda: validate_source(status, trace)), 'late empty guard escaped validator')

    cross_bytes = capacity(1000, [(1024, 1024, 2, 2)] * 12)
    self_bytes = capacity(100, [(1024, 1024, 2, 2)] * 12)
    require(cross_bytes == 49152000 and self_bytes == 4915200, 'capacity example')
    require(capacity(1000, [(4, 4, 2, 2)] * 3) == 48000, 'small cross capacity')
    require(capacity(7, [(4, 6, 2, 4), (2, 3, 1, 2)]) == 280, 'heterogeneous geometry')

    print('scope: Python stdlib CPU toy arithmetic/state checks; no GPU, named-model or WER measurements')
    print('analytic projected attention: %.12f %.12f; literal non-symmetric projection checked' % tuple(actual))
    print('negative controls: causal-cross difference=%.6g; edited-source stale difference=%.6g' % (causal_diff, stale_diff))
    print('append flow: projected old rows once and suffix only; projection calls=%r' % calls)
    print('two-layer toy incremental vs independent full-prefix reference max error=%.3e' % decoder_error)
    print('prefill/first-token/pending-input/EOS and one-action resume traces checked')
    print('budget pause, source wait, source-final wait, length stop and tail failure distinguished')
    print('ledger violations isolated; faulty new-version rollback rejected by common validator')
    print('raw/feature/encoder empty branches and late-guard negative controls checked')
    print('cross bytes=%d; self bytes=%d; heterogeneous capacity checked' % (cross_bytes, self_bytes))
    print('all scoped checks completed; intentionally faulty controls were detected')


if __name__ == '__main__':
    main()
