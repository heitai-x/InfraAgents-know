#!/usr/bin/env python3
"""K-SYS-025: deterministic, read-only, offline CPU models; stdlib only."""
from fractions import Fraction as Q
from itertools import product
import hashlib
import json
from pathlib import Path

KiB, MiB, GiB, TiB = (1024 ** i for i in range(1, 5))
FIXTURES = {
    "bandwidth": {"weight_GiB": 8, "rate_TiB_s": [4, 5],
                  "kv_KiB_token": 128, "history": 1024, "other_ms": 1},
    "chunks": {"prefix": 1000, "lengths": [512] * 8,
               "fixed_ms": 2, "linear_ms": [1, 100],
               "pair_ms": [1, 1000000]},
    "spec": {"alpha": [4, 5], "gamma": 4, "round_cost": 24},
    "kv": {"layers": 40, "heads": 8, "dim": 128, "group": 32,
           "residual_limit": 128, "bits": 2, "metadata_bytes": 4},
}
OBS = []
NEG = []


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def integer(n, low=0):
    if type(n) is not int or n < low:
        raise ValueError("integer outside domain")


def reject(label, validator, faulty):
    try:
        validator(faulty)
    except AssertionError:
        NEG.append(label)
    else:
        raise AssertionError("negative control escaped: " + label)


def invalid(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid input accepted")


def max_batch(deadline_ms, other_ms, w, rate, per_request):
    if deadline_ms < 0 or other_ms < 0 or w < 0 or rate <= 0 or per_request <= 0:
        raise ValueError("invalid bandwidth model")
    available = (deadline_ms - other_ms) * rate / 1000 - w
    return max(0, int(available // per_request))


def causal_counts(h, chunks, frozen_history=False):
    integer(h)
    total = 0
    for q in chunks:
        integer(q, 1)
        total += q * h + q * (q + 1) // 2
        if not frozen_history:
            h += q
    return total


def causal_sets(h, chunks):
    integer(h)
    pairs = set()
    for q in chunks:
        integer(q, 1)
        for p in range(h, h + q):
            for k in range(p + 1):
                pairs.add((p, k))
        h += q
    return pairs


def expected_tail(alpha, gamma):
    if not 0 <= alpha <= 1:
        raise ValueError("acceptance")
    integer(gamma)
    return sum((alpha ** i for i in range(gamma + 1)), Q(0))


def expected_paths(alpha, gamma):
    total, mass = Q(0), Q(0)
    for bits in product((0, 1), repeat=gamma):
        probability = Q(1)
        for bit in bits:
            probability *= alpha if bit else 1 - alpha
        accepted = 0
        for bit in bits:
            if not bit:
                break
            accepted += 1
        total += probability * (accepted + 1)
        mass += probability
    require(mass == 1, "path probability mass")
    return total


def kv_formula(n):
    integer(n)
    f = FIXTURES["kv"]
    e = f["layers"] * f["heads"] * f["dim"]
    r_k = n % f["residual_limit"]
    r_v = min(n, f["residual_limit"])
    q_k, q_v = n - r_k, n - r_v
    groups_k = (q_k // f["group"]) * e
    groups_v = q_v * f["layers"] * f["heads"] * (f["dim"] // f["group"])
    codes = (q_k + q_v) * e * f["bits"] // 8
    return codes + (r_k + r_v) * e * 2 + (groups_k + groups_v) * f["metadata_bytes"]


def kv_group_reference(n):
    # Explicitly build one channel's sealed token groups, then one token's
    # channel groups. Different loops from the aggregate capacity formula.
    f = FIXTURES["kv"]
    e = f["layers"] * f["heads"] * f["dim"]
    sealed_k = []
    residual_k = 0
    value_groups = 0
    residual_v = 0
    for _ in range(n):
        residual_k += 1
        if residual_k == f["residual_limit"]:
            for start in range(0, residual_k, f["group"]):
                sealed_k.append(min(f["group"], residual_k - start))
            residual_k = 0
        residual_v += 1
        if residual_v > f["residual_limit"]:
            residual_v -= 1
            for start in range(0, f["dim"], f["group"]):
                value_groups += 1
    key_bytes = sum((length * f["bits"] + 7) // 8 + f["metadata_bytes"]
                    for length in sealed_k) * e
    value_bytes = value_groups * (f["group"] * f["bits"] // 8 + f["metadata_bytes"])
    value_bytes *= f["layers"] * f["heads"]
    return key_bytes + value_bytes + (residual_k + residual_v) * e * 2


def padded(m, bucket):
    integer(m)
    integer(bucket, 1)
    return (m + bucket - 1) // bucket * bucket


def union_scan(records):
    sizes, events, counts = {}, {}, {}
    for key, size, start, end in records:
        if size < 0 or start > end or (key in sizes and sizes[key] != size):
            raise ValueError("invalid physical record")
        sizes[key] = size
        if start == end:
            continue
        events.setdefault(start, []).append((key, 1))
        events.setdefault(end, []).append((key, -1))
    total, peak, area, previous = 0, 0, 0, None
    for t in sorted(events):
        if previous is not None:
            area += total * (t - previous)
        # Group equal timestamps; half-open intervals cause no transient peak.
        for key, delta in sorted(events[t], key=lambda event: event[1]):
            old = counts.get(key, 0)
            new = old + delta
            require(new >= 0, "unmatched release")
            if old == 0 and new:
                total += sizes[key]
            if old and new == 0:
                total -= sizes[key]
            counts[key] = new
        peak = max(peak, total)
        previous = t
    require(total == 0, "leaked record")
    return peak, area


def union_reference(records):
    times = sorted({t for _, _, a, b in records for t in (a, b)})
    peak, area = 0, 0
    for a, b in zip(times, times[1:]):
        midpoint = Q(a + b, 2)
        live = {key: size for key, size, start, end in records
                if start <= midpoint < end}
        amount = sum(live.values())
        peak = max(peak, amount)
        area += amount * (b - a)
    return peak, area


def gaps(deliveries):
    return [b - a for a, b in zip(deliveries, deliveries[1:])]


def main():
    f = FIXTURES["bandwidth"]
    w, rate = f["weight_GiB"] * GiB, Q(*f["rate_TiB_s"]) * TiB
    per = f["history"] * f["kv_KiB_token"] * KiB
    def duration(b):
        return Q(1000) * (w + b * per) / rate + f["other_ms"]
    for deadline in (0, 5, Q(625, 64) + 1, 20, duration(64)):
        closed = max_batch(deadline, 1, w, rate, per)
        feasible = [b for b in range(1, 201) if duration(b) <= deadline]
        require(closed == max(feasible, default=0), "batch enumeration reference")
    bmax = max_batch(Q(20), 1, w, rate, per)
    require(bmax == 60 and duration(60) <= 20 < duration(61), "20 ms boundary")
    require(duration(64) == Q(657, 32), "64 requests cost 20.53125 ms")
    def valid_batch(b):
        require(b > 0 and duration(b) <= 20, "candidate violates complete budget")
    reject("rounded budget admits b=64", valid_batch, 64)
    reject("amortized round used as request ITL",
           lambda t: require(t == Q(1000) * w / rate, "round time mismatch"),
           Q(1000) * w / rate / 2)
    invalid(lambda: max_batch(20, 1, w, 0, per))
    OBS.append({"batch_max": bmax, "t60_ms": str(duration(60)),
                "t61_ms": str(duration(61)), "t64_ms": str(duration(64))})

    for h, chunks in ((0, []), (0, [1]), (3, [2, 1, 3]), (5, [1, 1, 1])):
        p = h + sum(chunks)
        reference = {(row, k) for row in range(h, p) for k in range(row + 1)}
        require(causal_sets(h, chunks) == reference, "causal relation")
        require(causal_counts(h, chunks) == len(reference), "pair count")
    reference = causal_counts(3, [2, 1, 3])
    reject("history not advanced", lambda n: require(n == reference, "lost history"),
           causal_counts(3, [2, 1, 3], True))
    invalid(lambda: causal_counts(0, [0]))
    f = FIXTURES["chunks"]
    h, chunks = f["prefix"], f["lengths"]
    n = sum(chunks)
    pairs = causal_counts(h, chunks)
    full = Q(2) + Q(1, 100) * n + Q(1, 1000000) * pairs
    sliced = sum((Q(2) + Q(1, 100) * q +
                  Q(1, 1000000) * sum(range(h + sum(chunks[:i]) + 1,
                                          h + sum(chunks[:i + 1]) + 1))
                  for i, q in enumerate(chunks)), Q(0))
    require(sliced - full == 2 * (len(chunks) - 1), "extra fixed costs")
    require(sliced - 2 * (len(chunks) - 1) == full, "retain one fixed charge")
    reject("remove all eight fixed charges",
           lambda t: require(t == full, "missing original fixed charge"),
           sliced - 2 * len(chunks))
    OBS.append({"prefill_full_ms": str(full), "sliced_ms": str(sliced)})

    for gamma in range(6):
        for alpha in (Q(0), Q(1, 3), Q(4, 5), Q(1)):
            require(expected_tail(alpha, gamma) == expected_paths(alpha, gamma),
                    "expectation from independent path enumeration")
    expected = expected_tail(Q(4, 5), 4)
    require(expected == Q(2101, 625), "spec length")
    unit = 24 / expected
    require(6 < unit < 10, "conditional sign reversal")
    reject("all speculative candidates accepted",
           lambda y: require(y == expected_paths(Q(4, 5), 4), "wrong production"),
           Q(5))
    # Burst delivery [0,10,10,10,20,20,20] has mean below 5, max above 5.
    burst_gaps = gaps([0, 10, 10, 10, 20, 20, 20])
    require(max(burst_gaps) == 10 and Q(sum(burst_gaps), len(burst_gaps)) < 5,
            "delivery mean does not bound largest token gap")
    reject("average speculative gap certifies maximum ITL",
           lambda estimate: require(estimate == max(burst_gaps), "hidden burst gap"),
           Q(sum(burst_gaps), len(burst_gaps)))
    OBS.append({"expected_tokens": str(expected), "spec_unit_time": str(unit)})

    for n in (0, 1, 31, 32, 127, 128, 129, 255, 256, 1024):
        require(kv_formula(n) == kv_group_reference(n), "KV grouping byte reference")
    require(kv_formula(1024) == Q(305, 8) * MiB, "38.125 MiB including metadata")
    require(kv_formula(128) < kv_formula(127), "key sealing sawtooth")
    reject("ideal bitwidth omits metadata and residual",
           lambda b: require(b == kv_group_reference(1024), "capacity undercount"),
           1024 * 20 * KiB)
    # At seal: full K/V for 128 tokens plus new key codes/metadata coexist.
    seal_peak = 128 * 160 * KiB + 128 * 10 * KiB + (128 // 32) * 40960 * 4
    require(seal_peak == Q(175, 8) * MiB and seal_peak > kv_formula(128),
            "old/new representation transition peak")
    invalid(lambda: kv_formula(-1))
    OBS.append({"kv1024_bytes": kv_formula(1024), "seal_peak_bytes": seal_peak})

    for m in range(65):
        for bucket in (1, 4, 16, 64):
            bins = []
            for row in range(m):
                if row % bucket == 0:
                    bins.append([None] * bucket)
                bins[-1][row % bucket] = row
            require(padded(m, bucket) == sum(map(len, bins)), "bucket placement")
    invalid(lambda: padded(1, 0))
    records = [("pool", 256, 0, 10), ("workspace", 16, 0, 10),
               ("shared", 128, 1, 4), ("shared", 128, 2, 8),
               ("new", 128, 8, 10), ("empty", 999, 3, 3)]
    reference = union_reference(records)
    require(union_scan(records) == reference == (400, 3872), "physical union")
    reject("shared owner capacity counted twice",
           lambda ledger: require(ledger == reference, "alias double count"),
           (528, 4128))
    early = [r if r[0] != "shared" or r[2] == 1 else
             (r[0], r[1], r[2], 4) for r in records]
    reject("reclaim shared page at first reader",
           lambda ledger: require(ledger == reference, "missing reader lifetime"),
           union_scan(early))
    # Standalone additional backing only: 256 MiB pools + 16 MiB workspace.
    capacity = (24 * GiB - 8 * GiB - 256 * MiB - 16 * MiB) // (128 * KiB)
    require(capacity == 128896, "KV capacity after independent backing reservations")
    reject("ignore graph/workspace",
           lambda count: require(count == capacity, "over-admission"),
           (24 * GiB - 8 * GiB) // (128 * KiB))
    OBS.append({"physical_peak_units": reference[0], "residency_integral": reference[1],
                "kv_token_capacity": capacity})

    require(Q(1) / (Q(7, 10) + Q(3, 10) / 2) == Q(20, 17), "Amdahl")
    # Explicit task intervals, two independent engines versus one serial chain.
    parallel = [("host", 0, 20), ("device", 0, 10)]
    serial = [("host", 0, 20), ("device", 20, 30)]
    makespan = lambda tasks: max(end for _, _, end in tasks)
    require(makespan(parallel) == 20 and makespan(serial) == 30, "dependency schedule")
    reject("max replaces serial dependency",
           lambda t: require(t == makespan(serial), "dependency lost"),
           max(20, 10))
    # Full compound trace: A prefill plus B ready decode.
    require(causal_counts(5, [2]) + causal_counts(8, [1]) == 22, "first view")
    require(causal_counts(7, [4]) + causal_counts(9, [1]) == 48, "second view")
    require(padded(3, 16) == padded(5, 16) == 16, "compound padding")
    require(Q(1000) * Q(17, 2) * GiB / TiB == Q(2125, 256),
            "binary units: 8.5 GiB at 1 TiB/s")
    OBS.append({"scope": "CPU exact algebra, finite grouping, lifetime and delivery models",
                "no_measurements": ["GPU", "CUDA", "DMA", "bandwidth", "model quality"]})


if __name__ == "__main__":
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fixture_hash = hashlib.sha256(json.dumps(FIXTURES, sort_keys=True,
                                separators=(",", ":")).encode()).hexdigest()
    print(json.dumps({"script_sha256": script_hash, "fixture_sha256": fixture_hash}))
    main()
    print(json.dumps({"observations": OBS, "detected_negative_controls": NEG},
                     ensure_ascii=False, sort_keys=True))
