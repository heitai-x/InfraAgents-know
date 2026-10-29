"""K-MODEL-013: deterministic CPU logical checks, Python standard library only."""
import math
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import permutations

EPS = 1e-12
ATOL = 1e-10
RTOL = 1e-10
checks = 0


def require(ok, name, detail=""):
    global checks
    checks += 1
    if not ok:
        raise AssertionError(name + ": " + str(detail))
    print("OK", name, detail)


def close(a, b):
    a, b = float(a), float(b)
    return (math.isfinite(a) and math.isfinite(b)
            and abs(a - b) <= ATOL + RTOL * abs(b))


def vector_close(a, b):
    return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))


def rejected(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def mean_masked(rows, mask):
    if not rows or len(rows) != len(mask):
        raise ValueError("empty rows or mask length mismatch")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("invalid row width")
    if any(type(flag) is not bool for flag in mask):
        raise ValueError("mask must be boolean")
    count = 0
    acc = [Fraction(0)] * width
    for row, valid in zip(rows, mask):
        if valid:
            count += 1
            for j, value in enumerate(row):
                acc[j] += Fraction(value)
    if count == 0:
        raise ValueError("empty pooling domain")
    return tuple(value / count for value in acc)


def mean_reference(valid_rows):
    # Input is the independent unpadded fixture, not selected through the tested mask.
    return tuple(sum(Fraction(row[j]) for row in valid_rows) / len(valid_rows)
                 for j in range(len(valid_rows[0])))


def normalize(vec, eps=EPS):
    values = tuple(float(x) for x in vec)
    if (not values or not math.isfinite(eps) or eps < 0
            or not all(math.isfinite(x) for x in values)):
        raise ValueError("invalid norm input")
    scale = max(abs(x) for x in values)
    if scale == 0:
        return tuple(0.0 for _ in values), True
    radius = math.sqrt(math.fsum((x / scale) ** 2 for x in values))
    # Compare scale*radius to eps without overflowing the product.
    if scale <= eps / radius:
        return tuple(0.0 for _ in values), True
    return tuple((x / scale) / radius for x in values), False


def norm_reference(vec):
    # Exact rational threshold avoids Decimal rounding at norm == EPS.
    floats = tuple(float(x) for x in vec)
    squared = sum(Fraction.from_float(x) ** 2 for x in floats)
    if squared <= Fraction.from_float(EPS) ** 2:
        return tuple(0.0 for _ in floats), True
    with localcontext() as ctx:
        ctx.prec = 70
        values = [Decimal.from_float(x) for x in floats]
        radius = sum(x * x for x in values).sqrt()
        return tuple(float(x / radius) for x in values), False


def padded(seq, width, side, fill):
    if width < len(seq) or side not in ("left", "right"):
        raise ValueError("invalid padding")
    padding = [fill] * (width - len(seq))
    if side == "left":
        return padding + list(seq), [False] * len(padding) + [True] * len(seq)
    return list(seq) + padding, [True] * len(seq) + [False] * len(padding)


def block_visibility(lengths):
    owners = [owner for owner, length in enumerate(lengths) for _ in range(length)]
    return [[a == b for b in owners] for a in owners]


def visibility_errors(lengths, mask):
    # Independent interval specification rather than the constructor's owner vector.
    total = sum(lengths)
    if len(mask) != total or any(len(row) != total for row in mask):
        return ["shape"]
    errors = []
    start = 0
    for length in lengths:
        stop = start + length
        for i in range(start, stop):
            for j in range(total):
                if mask[i][j] != (start <= j < stop):
                    errors.append((i, j))
        start = stop
    return errors


def attention(q, k, v, mask):
    width = len(q[0])
    out = []
    for i, query in enumerate(q):
        visible = [j for j, valid in enumerate(mask[i]) if valid]
        if not visible:
            raise ValueError("empty visible domain")
        scores = {j: math.fsum(a * b for a, b in zip(query, k[j]))
                  / math.sqrt(width) for j in visible}
        maximum = max(scores.values())
        mass = {j: math.exp(scores[j] - maximum) for j in visible}
        denominator = math.fsum(mass.values())
        out.append(tuple(math.fsum(mass[j] * v[j][c] for j in visible)
                         / denominator for c in range(len(v[0]))))
    return out


def attention_reference(q, k, v):
    # Unpacked per-request Decimal reference; no tested mask or softmax helper.
    with localcontext() as ctx:
        ctx.prec = 70
        dq = [[Decimal.from_float(float(x)) for x in row] for row in q]
        dk = [[Decimal.from_float(float(x)) for x in row] for row in k]
        dv = [[Decimal.from_float(float(x)) for x in row] for row in v]
        scale = Decimal(len(q[0])).sqrt()
        output = []
        for query in dq:
            weights = [(sum(a * b for a, b in zip(query, key)) / scale).exp()
                       for key in dk]
            total = sum(weights)
            output.append(tuple(float(sum(weights[j] * dv[j][c]
                                          for j in range(len(k))) / total)
                                for c in range(len(v[0]))))
        return output


def rerank(records):
    # key=(doc_id, occurrence_id); repeated documents remain separate obligations.
    if len({key for key, score in records}) != len(records):
        raise ValueError("duplicate occurrence key")
    if any(not math.isfinite(float(score)) for key, score in records):
        raise ValueError("non-finite score")
    return sorted([(key, float(score)) for key, score in records],
                  key=lambda item: (-item[1], item[0]))


def pairing_errors(expected, observed):
    # Freeze obligations before observing output. Malformed keys yield a witness.
    errors = []
    seen = Counter()
    for record in observed:
        if not isinstance(record, (tuple, list)) or len(record) != 2:
            errors.append("malformed record")
            continue
        key, score = record
        if (type(key) is not tuple or len(key) != 2
                or not all(type(part) is str for part in key)):
            errors.append("malformed key")
            continue
        seen[key] += 1
        if key not in expected:
            errors.append("unknown key")
        elif not close(score, expected[key]):
            errors.append("score mismatch")
    for key in expected:
        if seen[key] != 1:
            errors.append("missing or duplicate key")
    return errors


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def cosine(a, b):
    na, za = normalize(a)
    nb, zb = normalize(b)
    if za or zb:
        raise ValueError("undefined cosine")
    return dot(na, nb)


def softmax(scores):
    maximum = max(scores)
    values = [math.exp(x - maximum) for x in scores]
    total = math.fsum(values)
    return [x / total for x in values]


def main():
    seq = [(1, 0), (0, 1)]
    reference = mean_reference(seq)
    require(reference == (Fraction(1, 2), Fraction(1, 2)), "exact pooling fixture")
    # Historical fixture explains why normalized-only checking was blind.
    collinear = (100, 100)
    require(vector_close(normalize(collinear)[0], normalize(reference)[0]),
            "collinear last-row fixture is a normalized-output blind spot")
    require(not vector_close(collinear, reference),
            "same comparator rejects historical wrong pooled value")
    fill = (100, -20)
    for side in ("left", "right"):
        rows, mask = padded(seq, 5, side, fill)
        pooled = mean_masked(rows, mask)
        require(pooled == reference, "masked mean independent reference " + side)
        require(vector_close(normalize(pooled)[0], norm_reference(reference)[0]),
                "normalized pooling " + side)
        last = max(i for i, valid in enumerate(mask) if valid)
        require(rows[last] == seq[-1], "last-valid-row mapping " + side)
    rows, mask = padded(seq, 3, "right", fill)
    expected_vector = normalize(reference)[0]
    require(not vector_close(normalize(rows[-1])[0], expected_vector),
            "negative fixed-last-column detected after normalization")
    require(not vector_close(mean_masked(rows, [True] * 3), reference),
            "negative padding included in pooling detected")
    wrong_denominator = tuple(x * Fraction(2, 3) for x in reference)
    require(not vector_close(wrong_denominator, reference),
            "negative denominator-only error detected before normalization")
    require(vector_close(normalize(wrong_denominator)[0], expected_vector),
            "denominator-only error may be erased by L2 normalization")
    for fixture, name in [
            (lambda: mean_masked([], []), "empty rows"),
            (lambda: mean_masked([(1, 2)], [False]), "empty pooling domain"),
            (lambda: mean_masked([(1, 2)], []), "mask length mismatch"),
            (lambda: mean_masked([(1, 2), (3,)], [True, True]), "ragged rows"),
            (lambda: mean_masked([(1, 2)], [1]), "non-boolean mask")]:
        require(rejected(fixture), name + " rejected")
    require(mean_masked([(2, 3), (float("nan"), float("inf"))],
                        [True, False]) == (2, 3), "invalid padding not read into sum")
    for vec in [(0, 0), (EPS / 2, 0), (EPS, 0), (2 * EPS, 0),
                (3, 4), (1e308, 1e308)]:
        value, zero = normalize(vec)
        ref, ref_zero = norm_reference(vec)
        require(zero == ref_zero and vector_close(value, ref),
                "scaled norm vs Decimal", str(vec))
    require(rejected(lambda: normalize((float("nan"), 0))), "NaN norm rejected")
    require(rejected(lambda: normalize((float("inf"), 0))), "infinite norm rejected")

    lengths = [2, 1, 3]
    mask = block_visibility(lengths)
    require(not visibility_errors(lengths, mask), "block visibility interval reference")
    total = sum(lengths)
    global_mask = [[True] * total for _ in range(total)]
    require(bool(visibility_errors(lengths, global_mask)), "negative global mask detected")
    q = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (-1, 1)]
    k = [(1, 1), (2, -1), (0, 1), (1, 0), (-1, 2), (2, 2)]
    v = [(1, 3), (4, -2), (20, 10), (-5, 8), (7, 1), (3, 9)]
    expected = []
    start = 0
    for length in lengths:
        stop = start + length
        expected.extend(attention_reference(q[start:stop], k[start:stop], v[start:stop]))
        start = stop
    good = attention(q, k, v, mask)
    require(all(vector_close(a, b) for a, b in zip(good, expected)),
            "packed attention vs independent per-request Decimal reference")
    bad = attention(q, k, v, global_mask)
    require(any(not vector_close(a, b) for a, b in zip(bad, expected)),
            "negative cross-request attention changes actual values")

    expected_scores = {("a", "0"): 0.5, ("b", "0"): 0.9,
                       ("c", "0"): 0.5, ("d", "0"): 0.1}
    expected_order = [(("b", "0"), 0.9), (("a", "0"), 0.5),
                      (("c", "0"), 0.5), (("d", "0"), 0.1)]
    records = [(key, expected_scores[key]) for key in
               [("d", "0"), ("c", "0"), ("b", "0"), ("a", "0")]]
    ordered = rerank(records)
    require(ordered == expected_order and not pairing_errors(expected_scores, ordered),
            "positive score-ID pairing and tie order")
    require(all(rerank(list(p)) == expected_order for p in permutations(records)),
            "all candidate permutations preserve score-ID pairs")
    stable_bad = sorted(records, key=lambda item: -item[1])
    require(stable_bad != expected_order and not pairing_errors(expected_scores, stable_bad),
            "negative input-order ties isolated from score pairing")
    # Repair: extract IDs from sorted (ID,score) records, then deliberately mis-pair.
    misaligned = [(record[0], original[1]) for record, original in zip(ordered, records)]
    require(bool(pairing_errors(expected_scores, misaligned)),
            "negative score-ID misalignment detected without KeyError", str(misaligned))
    malformed_old_bug = [(record, original[1]) for record, original in zip(ordered, records)]
    require(bool(pairing_errors(expected_scores, malformed_old_bug)),
            "historical nested-key defect yields witness, not exception")
    for observed, name in [
            (ordered[:-1], "missing result"),
            (ordered + [ordered[0]], "duplicate result"),
            (ordered + [(("z", "0"), 1.0)], "extra result")]:
        require(bool(pairing_errors(expected_scores, observed)), name + " detected")
    repeated = [(("a", "first"), 0.3), (("a", "second"), 0.3)]
    require(len(rerank(repeated)) == 2, "duplicate documents keep occurrence identity")
    require(rejected(lambda: rerank(repeated + repeated[:1])), "duplicate occurrence rejected")
    require(rejected(lambda: rerank([(("a", "0"), float("nan"))])), "NaN score rejected")
    require(rerank([]) == [], "empty candidate set")

    query, a, b = (1, 1), (10, 0), (1, 1)
    require(dot(query, a) > dot(query, b) and cosine(query, a) < cosine(query, b),
            "negative unnormalized-dot-as-cosine ranking detected")
    nq = normalize(query)[0]
    require(close(dot(nq, normalize(a)[0]), cosine(query, a)),
            "normalized dot equals cosine")
    one = softmax([0, 0])
    expanded = softmax([0, 0, 0])
    require(vector_close(one, (0.5, 0.5)) and not close(one[0], expanded[0]),
            "candidate-set softmax changes previous probabilities")
    # Effective identity includes positions/mask/pooling, not merely output dimension.
    key = ("A1", (11, 12), (0, 1), "bidirectional", "mean-valid", "l2-eps12", "fp32")
    changed = ("A1", (11,), (0,), "bidirectional", "mean-valid", "l2-eps12", "fp32")
    require(key != changed and (key[0], 768) == (changed[0], 768),
            "negative dimension-only key collides for different effective inputs")

    # Count coordinate pairs independently of the squared-length formula.
    small = [1, 3, 2]
    pairs = sum(1 for n in small for i in range(n) for j in range(n))
    require(pairs == sum(n * n for n in small), "attention pair formula by enumeration")
    padded_pairs = 8 * 512 * 512
    bucket_pairs = 4 * 64 * 64 + 4 * 512 * 512
    require((padded_pairs, bucket_pairs, 8 * 768 * 4)
            == (2097152, 1064960, 24576), "C12-A logical counts and bytes")
    require(20 * (320 * 320 - 288 * 288) == 389120,
            "rerank padding adds cross terms")
    require((320 * 320 - 288 * 288) != (320 - 288) ** 2,
            "negative padding-square-only cost detected")
    print("SUMMARY checks=" + str(checks)
          + "; scope=CPU Fraction/Decimal/binary64 logical model;"
          + " no Transformer forward, GPU, service timing, or retrieval quality measurement")


if __name__ == "__main__":
    main()
