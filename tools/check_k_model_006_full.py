"""K-MODEL-006: deterministic offline CPU mathematics, no GPU measurements.
Run after coordinator materializes this file: python tools/check_k_model_006_full.py
Only Python standard library; no file, network, subprocess or project access.
"""
import math
from decimal import Decimal, localcontext
from fractions import Fraction

H, DH, DC, DR, DO = 2, 2, 3, 2, 3
UK = [[[1, 0, 2], [0, 1, -1]],
      [[0, 2, 1], [1, -1, 0]]]
UV = [[[1, 0, 1], [0, 1, 2]],
      [[0, 1, 0], [2, 0, 1]]]
UQ = [[[1, 2], [-1, 1]], [[2, -1], [1, 1]]]
QR = [[[1, 0], [0, 1]], [[1, 1], [1, -1]]]
WO = [[1, 0, 0, 1], [0, 1, 1, 0], [1, -1, 2, 1]]
RAW = [[1, 0, 2], [0, 2, 1], [2, -1, 0],
       [-1, 1, 1], [0.5, -2, 3], [2, 0.25, -1], [-3, 1, 2]]
KBASE = [[0.25, -0.5], [0.75, 0.25], [-0.5, 1],
         [1, -0.25], [-1, 0.5], [0.5, 1.25], [1.5, -0.75]]
POS = [2, 5, 8, 12, 17, 23, 30]
QRAW = [0.75, -1.25]
KG, QG = [1, 1.25, 0.75], [0.875, 1.125]
EPS, OMEGA = 0.125, 0.31
invariant_steps = 0


def dec(x):
    return Decimal(str(x))


def dot(a, b):
    assert len(a) == len(b)
    return sum(x * y for x, y in zip(a, b))


def mv(matrix, vec):
    return [dot(row, vec) for row in matrix]


def norm(vec, gain):
    den = math.sqrt(sum(x*x for x in vec)/len(vec) + EPS)
    return [g*x/den for g, x in zip(gain, vec)]


def dnorm(vec, gain):
    # Independent Decimal normalization, from raw inputs rather than cached s.
    dv = [dec(x) for x in vec]
    den = (sum(x*x for x in dv)/Decimal(len(dv)) + dec(EPS)).sqrt()
    return [dec(g)*x/den for g, x in zip(gain, dv)]


def rotate(vec, pos):
    angle = OMEGA*pos
    c, s = math.cos(angle), math.sin(angle)
    return [c*vec[0]-s*vec[1], s*vec[0]+c*vec[1]]


def require_finite(values):
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError("non-finite comparison or visible arithmetic")


def distance(a, b):
    assert len(a) == len(b)
    require_finite(a)
    require_finite(b)
    return max([abs(float(x)-float(y)) for x, y in zip(a, b)] or [0.0])


def expanded_reference(n, qpos, visible):
    # No absorbed projection, online merge, or candidate normalization is used.
    with localcontext() as ctx:
        ctx.prec = 60
        cache = [dnorm(RAW[j], KG) for j in range(n)]
        cq = dnorm(QRAW, QG)
        alpha = Decimal(DH + DR).sqrt()
        heads = []
        for i in range(H):
            qc = [sum(dec(w)*x for w, x in zip(row, cq)) for row in UQ[i]]
            qb = [sum(dec(w)*x for w, x in zip(row, cq)) for row in QR[i]]
            scores, values = [], []
            for j in visible:
                kc = [sum(dec(w)*x for w, x in zip(row, cache[j]))
                      for row in UK[i]]
                vj = [sum(dec(w)*x for w, x in zip(row, cache[j]))
                      for row in UV[i]]
                # Relative-angle reference; trig values still use binary64 math.
                angle = OMEGA*(POS[j]-qpos)
                c, s = dec(math.cos(angle)), dec(math.sin(angle))
                x, y = map(dec, KBASE[j])
                kr = [c*x-s*y, s*x+c*y]
                scores.append((sum(x*y for x, y in zip(qc, kc)) +
                               sum(x*y for x, y in zip(qb, kr)))/alpha)
                values.append(vj)
            if not visible:
                heads.extend([Decimal(0)]*DH)
                continue
            top = max(scores)
            weights = [(x-top).exp() for x in scores]
            mass = sum(weights)
            heads.extend(sum(weights[j]*values[j][c] for j in range(len(visible)))/mass
                         for c in range(DH))
        out = [sum(dec(w)*x for w, x in zip(row, heads)) for row in WO]
        return heads + out


def empty():
    return (-math.inf, 0.0, [0.0]*DC)


def block_stats(scores, cache, indices):
    if not indices:
        return empty()
    require_finite([scores[j] for j in indices])
    top = max(scores[j] for j in indices)
    weights = [math.exp(scores[j]-top) for j in indices]
    return (top, sum(weights),
            [sum(w*cache[j][c] for w, j in zip(weights, indices)) for c in range(DC)])


def merge(a, b, faulty=False):
    if a[1] == 0:
        return (b[0], b[1], b[2][:])
    if b[1] == 0:
        return (a[0], a[1], a[2][:])
    top = max(a[0], b[0])
    x = 1.0 if faulty else math.exp(a[0]-top)
    y = 1.0 if faulty else math.exp(b[0]-top)
    return (top, x*a[1]+y*b[1], [x*u+y*v for u, v in zip(a[2], b[2])])


def check_prefix_state(state, scores, cache, processed):
    # Whole-set Decimal statistics independently check every intermediate merge.
    global invariant_steps
    invariant_steps += 1
    if not processed:
        assert state == empty()
        return
    with localcontext() as ctx:
        ctx.prec = 60
        top = max(dec(scores[j]) for j in processed)
        weights = [(dec(scores[j])-top).exp() for j in processed]
        mass = sum(weights)
        totals = [sum(w*dec(cache[j][c]) for w, j in zip(weights, processed))
                  for c in range(DC)]
        assert abs(state[0]-float(top)) < 1e-11
        assert abs(state[1]-float(mass)) < 1e-10
        assert distance(state[2], totals) < 1e-10


def absorbed(n, qpos, visible, block=2, reverse=False, fault=""):
    assert 0 <= n <= len(RAW) and block > 0
    assert len(visible) == len(set(visible))
    assert all(0 <= j < n for j in visible)
    cache = [list(RAW[j]) if fault == "kv_norm" else norm(RAW[j], KG)
             for j in range(n)]
    cq = list(QRAW) if fault == "query_norm" else norm(QRAW, QG)
    if fault == "mask" and visible:
        visible = [0]
    selected = set(visible)
    groups = [list(range(k, min(k+block, n))) for k in range(0, n, block)]
    groups = [[]] + groups + [[], []]
    if reverse:
        groups.reverse()
    heads = []
    for i in range(H):
        qc, qb = mv(UQ[i], cq), mv(QR[i], cq)
        qr = rotate(qb, qpos)
        wk = UK[(i+1) % H] if fault == "head" else UK[i]
        qt = [sum(wk[r][c]*qc[r] for r in range(DH)) for c in range(DC)]
        scale = math.sqrt(DC+DR if fault == "scale" else DH+DR)
        scores = []
        for j in range(n):
            kr = rotate(KBASE[j], 0 if fault == "position" else POS[j])
            rope = 0.0 if fault == "omit_rope" else dot(qr, kr)
            content = dot(qt[:-1], cache[j][:-1]) if fault == "feature" else dot(qt, cache[j])
            scores.append((content+rope)/scale)
        x = empty()
        processed = []
        for group in groups:
            indices = [j for j in group if j in selected]
            y = block_stats(scores, cache, indices)
            x = merge(x, y, faulty=(fault == "merge"))
            processed.extend(indices)
            if not fault:
                check_prefix_state(x, scores, cache, processed)
        z = [v/x[1] for v in x[2]] if x[1] else [0.0]*DC
        heads.extend(mv(UV[i], z))
    out = []
    for row in WO:
        if fault == "output_head":
            row = row[DH:] + row[:DH]
        out.append(dot(row, heads))
    require_finite(heads + out)
    return heads + out


def exact_absorption():
    # Rational identities also cover each head with distinct attention weights.
    for i in range(H):
        q = [Fraction(i+1, 3), Fraction(2-i, 5)]
        qt = [sum(UK[i][r][c]*q[r] for r in range(DH)) for c in range(DC)]
        for raw in RAW[:4]:
            s = list(map(Fraction, raw))
            assert dot(q, mv(UK[i], s)) == dot(qt, s)
        weights = [Fraction(j+i+1, 10+4*i) for j in range(4)]
        assert sum(weights) == 1
        vref = [sum(weights[j]*mv(UV[i], RAW[j])[c] for j in range(4))
                for c in range(DH)]
        latent = [sum(weights[j]*Fraction(RAW[j][c]) for j in range(4)) for c in range(DC)]
        assert vref == mv(UV[i], latent)


def scalar_mac_counts(m, n, heads, dc, dh, dr, pairs):
    # Independent operational reference: count scalar MAC events in each path.
    # Same newly produced s, queries, and W_O are omitted from both candidates.
    expanded = absorbed_count = 0
    for j in range(n):
        for i in range(heads):
            for r in range(dh):
                for c in range(dc):
                    expanded += 1  # W_UK[r,c] * s[j,c]
                    expanded += 1  # W_UV[r,c] * s[j,c]
    for t in range(m):
        for i in range(heads):
            for c in range(dc):
                for r in range(dh):
                    absorbed_count += 1  # W_UK[r,c] * qC[t,i,r]
            for r in range(dh):
                for c in range(dc):
                    absorbed_count += 1  # W_UV[r,c] * zbar[t,i,c]
    for t, j in pairs:
        assert 0 <= t < m and 0 <= j < n
        for i in range(heads):
            for c in range(dh):
                expanded += 1  # Content score.
                expanded += 1  # Value accumulation.
            for c in range(dc):
                absorbed_count += 1  # Absorbed content score.
                absorbed_count += 1  # Latent accumulation.
            for a in range(dr):
                expanded += 1
                absorbed_count += 1  # Same positional score.
    return expanded, absorbed_count


def dense_absorbed_cheaper(m, n, dc, dh):
    assert min(m, n, dc, dh) > 0
    denominator = dc*dh + n*(dc-dh)
    if denominator > 0:
        return Fraction(m) < Fraction(n*dc*dh, denominator)
    # n*dc*dh > 0, while m*denominator <= 0.
    return True


def sign(x):
    return (x > 0) - (x < 0)


def cost_crossovers():
    cases = old_rule_disagreements = 0
    denominator_signs = set()
    prefill_width_signs = set()
    for m in range(1, 7):
        for n in range(1, 6):
            dense = [(t, j) for t in range(m) for j in range(n)]
            masks = [dense, [(t, j) for t, j in dense if (t+j) % 3 == 0], []]
            for heads in (1, 2):
                for dc in range(1, 5):
                    for dh in range(1, 5):
                        denominator = dc*dh + n*(dc-dh)
                        denominator_signs.add(sign(denominator))
                        for mask_index, pairs in enumerate(masks):
                            expanded, got = scalar_mac_counts(m, n, heads, dc, dh, 2, pairs)
                            p = len(pairs)
                            delta = 2*heads*dc*dh*(m-n) + 2*p*heads*(dc-dh)
                            assert got-expanded == delta
                            if mask_index == 0:
                                assert dense_absorbed_cheaper(m, n, dc, dh) == (got < expanded)
                                assert (m*denominator < n*dc*dh) == (got < expanded)
                                if denominator:
                                    # Intentionally faulty old rule: division without sign handling.
                                    bad = Fraction(m) < Fraction(n*dc*dh, denominator)
                                    old_rule_disagreements += bad != (got < expanded)
                            if m == n and p > 0:
                                assert sign(got-expanded) == sign(dc-dh)
                                prefill_width_signs.add(sign(dc-dh))
                            if m == n and p == 0:
                                assert got == expanded
                            cases += 1
    assert cases == 2880
    assert denominator_signs == {-1, 0, 1}
    assert prefill_width_signs == {-1, 0, 1}
    assert old_rule_disagreements > 0

    # An exact positive-denominator crossing: equality is not strict improvement.
    expanded, got = scalar_mac_counts(1, 2, 1, 2, 1, 2, [(0, 0), (0, 1)])
    assert (expanded, got) == (16, 16)
    assert not dense_absorbed_cheaper(1, 2, 2, 1)
    # Zero denominator must not trigger division by zero or rejection.
    expanded, got = scalar_mac_counts(1, 2, 1, 1, 2, 2, [(0, 0), (0, 1)])
    assert 1*2 + 2*(1-2) == 0
    assert got < expanded and dense_absorbed_cheaper(1, 2, 1, 2)

    # Fixed reviewer counterexamples, independently recomputed by table stages.
    for m, expected in [(1, (4260352, 49664, -4210688)),
                        (256, (21102592, 12713984, -8388608))]:
        n, heads, dc, dh, dr = 256, 1, 64, 128, 2
        p = m*n
        expanded = 2*n*heads*dc*dh + p*heads*(2*dh+dr)
        got = 2*m*heads*dc*dh + p*heads*(2*dc+dr)
        assert (expanded, got, got-expanded) == expected
        assert dense_absorbed_cheaper(m, n, dc, dh)
        denominator = dc*dh + n*(dc-dh)
        assert denominator == -8192
        bad = Fraction(m) < Fraction(n*dc*dh, denominator)
        assert bad != (got < expanded)
        if m == n:
            assert got-expanded < 0  # Detects the faulty unconditional positive claim.
        print("modeled MAC counterexample: M=%d expanded=%d absorbed=%d delta=%d D=%d"
              % (m, expanded, got, got-expanded, denominator))
    print("CPU MAC/event checks: cases=%d denominator_signs=%s old_rule_disagreements=%d"
          % (cases, sorted(denominator_signs), old_rule_disagreements))
    print("cost boundaries: exact tie, D=0, D<0, sparse/empty visibility and prefill signs checked")


def main():
    cost_crossovers()
    exact_absorption()
    cases = [(0, 0, []), (1, 2, [0]), (5, 17, []),
             (5, 17, [0, 1, 2, 3, 4]), (7, 30, [0, 1, 3, 4, 6]),
             (7, 23, [0, 1, 2, 3, 4, 5]), (7, 30, [6])]
    max_gap, comparisons = 0.0, 0
    for n, pos, vis in cases:
        ref = expanded_reference(n, pos, vis)
        for block in (1, 2, 3, 16):
            for reverse in (False, True):
                got = absorbed(n, pos, vis, block, reverse)
                gap = distance(got, ref)
                assert gap < 1e-10, (n, pos, vis, block, reverse, gap)
                max_gap = max(max_gap, gap)
                comparisons += 1
    print("CPU expanded Decimal vs absorbed binary64: comparisons=%d max_gap=%.3g"
          % (comparisons, max_gap))
    print("CPU online-prefix invariants checked:", invariant_steps)

    n, pos, vis = 7, 30, [0, 1, 3, 4, 6]
    ref = expanded_reference(n, pos, vis)
    negatives = ("head", "output_head", "kv_norm", "query_norm", "scale",
                 "position", "omit_rope", "feature", "merge", "mask")
    for fault in negatives:
        gap = distance(absorbed(n, pos, vis, 2, False, fault), ref)
        assert gap > 1e-7, ("undetected negative control", fault, gap)
        print("negative control detected: %s gap=%.6g" % (fault, gap))

    # Large finite scores, unequal maxima, a tail partition, empty identities.
    scores = [1000.0, -1000.0, 999.0]
    values = [[1.0, 2.0, -1.0], [9.0, -4.0, 0.0], [-2.0, 0.5, 3.0]]
    x = merge(merge(empty(), block_stats(scores, values, [0, 1])),
              block_stats(scores, values, [2]))
    x = merge(x, empty())
    with localcontext() as ctx:
        ctx.prec = 60
        weights = [dec(0).exp(), dec(-2000).exp(), dec(-1).exp()]
        ref_large = [sum(weights[j]*dec(values[j][c]) for j in range(3))/sum(weights)
                     for c in range(DC)]
    assert distance([v/x[1] for v in x[2]], ref_large) < 1e-12
    assert merge(empty(), empty()) == empty()
    assert merge(empty(), x) == x and merge(x, empty()) == x
    try:
        distance([float("nan")], [0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("comparison accepted NaN")
    try:
        block_stats([math.inf], [[1.0]*DC], [0])
    except ValueError:
        pass
    else:
        raise AssertionError("visible positive infinity accepted")
    print("CPU finite/empty/large-score boundary checks: satisfied")

    # Independently fixed expected resource quantities from the chapter's scenario.
    layers, tokens, heads, dh, dc, dr, width_bytes, ranks = 32, 8192, 32, 128, 512, 64, 2, 4
    mla = layers*tokens*(dc+dr)*width_bytes
    mha = layers*tokens*2*heads*dh*width_bytes
    group = sum(mla for _ in range(ranks))
    score_tensor = heads*tokens*4
    # Ring sends one A/p chunk in each reduce-scatter and all-gather round.
    send = sum(score_tensor//ranks for _ in range(2*(ranks-1)))
    assert (mla, mha, group, send) == (301989888, 4294967296, 1207959552, 1572864)
    assert send*layers == 50331648
    assert mla//ranks == 75497472
    assert (2+4)*mla == 1811939328
    assert layers*tokens*dc*width_bytes != mla  # Omitted RoPE capacity fault.
    assert mla//ranks != mla  # Replicated cache must not be divided by TP.
    print("modeled bytes: MLA=%d MHA=%d TP4_group=%d score_send_per_rank_layer=%d"
          % (mla, mha, group, send))
    print("scope: projected-input CPU algebra and resource arithmetic only; no GPU,"
          " BF16/FP8, actual paging/quantization, networking, migration faults or model-quality evidence")


if __name__ == "__main__":
    main()
