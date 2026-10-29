#!/usr/bin/env python3
"""K-SYS-001: deterministic, read-only, offline CPU model checks.

Only Python stdlib; no project imports, file access, subprocesses or devices.
Interval tests use integer endpoints; DAG tests use nonnegative integer
durations, releases and edge lags in one normalized time domain.
"""
from collections import deque
import random


def require(ok, detail):
    if not ok:
        raise AssertionError(detail)


def expect_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("invalid input accepted")


def interval_stats(records, window):
    lo, hi = window
    if lo > hi:
        raise ValueError("invalid window")
    work = []
    for start, end in records:
        if start > end:
            raise ValueError("invalid interval")
        a, b = max(start, lo), min(end, hi)
        if a < b:
            work.append((a, b))
    work.sort()
    merged = []
    for a, b in work:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    active = peak = 0
    events = sorted((t, delta) for a, b in work
                    for t, delta in ((a, 1), (b, -1)))
    for _, delta in events:
        active += delta
        peak = max(peak, active)
        require(active >= 0, "negative active count")
    require(active == 0, "unbalanced interval endpoints")
    total = sum(b - a for a, b in work)
    union = sum(b - a for a, b in merged)
    return tuple(map(tuple, merged)), total, union, peak, hi - lo - union


def reference_cells(records, window):
    # Independent raster reference: no clipping, sorting or endpoint sweep.
    lo, hi = window
    if lo > hi or any(a > b for a, b in records):
        raise ValueError("invalid raster input")
    counts = [sum(a <= t < b for a, b in records) for t in range(lo, hi)]
    covered = [lo + i for i, count in enumerate(counts) if count]
    segments = []
    for t in covered:
        if segments and segments[-1][1] == t:
            segments[-1][1] = t + 1
        else:
            segments.append([t, t + 1])
    return (tuple(map(tuple, segments)), sum(counts), len(covered),
            max(counts, default=0), len(counts) - len(covered))


def bad_start_first(records):
    events = sorted(((t, delta) for a, b in records if a < b
                     for t, delta in ((a, 1), (b, -1))),
                    key=lambda item: (item[0], -item[1]))
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    return peak


def earliest(durations, releases, edges, ignore_release=False,
             choose_min=False):
    n = len(durations)
    if len(releases) != n or any(x < 0 for x in durations + releases):
        raise ValueError("invalid node values")
    pred = [[] for _ in range(n)]
    succ = [[] for _ in range(n)]
    degree = [0] * n
    seen = set()
    for u, v, lag in edges:
        if not (0 <= u < n and 0 <= v < n) or lag < 0:
            raise ValueError("invalid edge")
        if (u, v) in seen:
            raise ValueError("duplicate edge")
        seen.add((u, v))
        pred[v].append((u, lag))
        succ[u].append(v)
        degree[v] += 1
    ready = deque(v for v in range(n) if degree[v] == 0)
    order = []
    while ready:
        u = ready.popleft()
        order.append(u)
        for v in succ[u]:
            degree[v] -= 1
            if degree[v] == 0:
                ready.append(v)
    if len(order) != n:
        raise ValueError("cycle")
    es, ef, parent = [0] * n, [0] * n, [None] * n
    for v in order:
        release = 0 if ignore_release else releases[v]
        es[v] = release
        options = [(ef[u] + lag, u) for u, lag in pred[v]]
        if options:
            value, u = min(options) if choose_min else max(options)
            if value > es[v]:
                es[v], parent[v] = value, u
        ef[v] = es[v] + durations[v]
    paths = []
    for end in range(n):
        path = []
        v = end
        while v is not None:
            path.append(v)
            v = parent[v]
        paths.append(tuple(reversed(path)))
    return es, ef, paths


def enumerate_paths(durations, releases, edges):
    # Independent oracle: every vertex can start a release-constrained path.
    n = len(durations)
    succ = [[] for _ in range(n)]
    for u, v, lag in edges:
        succ[u].append((v, lag))
    scores = [None] * n
    def visit(v, score, path):
        if v in path:
            raise ValueError("cycle in reference")
        scores[v] = score if scores[v] is None else max(scores[v], score)
        for w, lag in succ[v]:
            visit(w, score + lag + durations[w], path + (v,))
    for start in range(n):
        visit(start, releases[start] + durations[start], ())
    return scores


def verify_dag(durations, releases, edges):
    es, ef, paths = earliest(durations, releases, edges)
    require(ef == enumerate_paths(durations, releases, edges),
            ("path oracle mismatch", durations, releases, edges, ef))
    lags = {(u, v): lag for u, v, lag in edges}
    for v, path in enumerate(paths):
        require(es[v] >= releases[v], ("release violated", v))
        require(path and path[-1] == v, ("bad witness end", v, path))
        score = releases[path[0]] + sum(durations[u] for u in path)
        for u, w in zip(path, path[1:]):
            require((u, w) in lags, ("witness missing edge", path))
            score += lags[u, w]
        require(score == ef[v], ("wrong witness score", path, score, ef[v]))
    for u, v, lag in edges:
        require(es[v] >= ef[u] + lag, ("precedence violated", u, v))
    return ef


def clock_bounds(sent, received, offset_lo, offset_hi):
    if offset_lo > offset_hi:
        raise ValueError("invalid offset bounds")
    lower = received - sent - offset_hi
    upper = received - sent - offset_lo
    if upper < 0:
        raise ValueError("clock bounds contradict send-before-receive")
    return max(0, lower), upper


def main():
    rng = random.Random(20261028)
    fixed = [
        ([], (0, 20)),
        ([(3, 3), (20, 20)], (0, 20)),
        ([(0, 10), (5, 15)], (0, 20)),
        ([(0, 10), (10, 20)], (0, 20)),
        ([(0, 10), (0, 10), (2, 3)], (3, 7)),
        ([(-20, -10), (30, 40)], (0, 20)),
        ([(0, 10)], (5, 5)),
        ([(0, 10), (2, 3), (4, 8)], (0, 20)),
    ]
    cases = list(fixed)
    for _ in range(300):
        records = []
        for _ in range(rng.randrange(8)):
            a = rng.randrange(-6, 13)
            records.append((a, a + rng.randrange(9)))
        lo = rng.randrange(-8, 9)
        cases.append((records, (lo, lo + rng.randrange(11))))
    for records, window in cases:
        got = interval_stats(records, window)
        require(got == reference_cells(records, window), (records, window, got))
        _, total, union, peak, idle = got
        require(0 <= union <= total <= peak * union, "integral bounds")
        require(union + idle == window[1] - window[0], "window conservation")
    expect_value_error(interval_stats, [(5, 4)], (0, 1))
    expect_value_error(interval_stats, [], (2, 1))
    overlap = interval_stats([(0, 10), (5, 15)], (0, 20))
    require(overlap[1] == 20 and overlap[2] == 15, "overlap example")
    require(overlap[1] != overlap[2], "sum-as-union mutant survived")
    require(bad_start_first([(0, 10), (10, 20)]) !=
            reference_cells([(0, 10), (10, 20)], (0, 20))[3],
            "closed-boundary mutant survived")

    d, r = [40, 8, 12, 15, 35], [0] * 5
    edges = [(0, 1, 0), (1, 2, 0), (2, 3, 0)]
    require(verify_dag(d, r, edges)[3] == 75, "base completion")
    require(verify_dag([40, 8, 12, 15, 15], r, edges)[3] == 75,
            "off-path shortening")
    require(verify_dag([20, 8, 12, 15, 35], r, edges)[3] == 55,
            "on-path shortening")
    require(sum(d) != 75, "sum-as-path mutant survived")
    require(earliest(d, r, edges[1:])[1][3] !=
            enumerate_paths(d, r, edges)[3], "missing-edge mutant survived")
    require(verify_dag([2, 3], [0, 10], [(0, 1, 4)]) == [2, 13],
            "internal release must dominate")
    require(earliest([2, 3], [0, 10], [(0, 1, 4)],
                     ignore_release=True)[1] != [2, 13],
            "ignored-release mutant survived")
    require(verify_dag([2, 3, 1], [0, 0, 0],
                       [(0, 2, 1), (1, 2, 4)]) == [2, 3, 8],
            "join waits for all predecessors")
    require(earliest([2, 3, 1], [0, 0, 0],
                     [(0, 2, 1), (1, 2, 4)], choose_min=True)[1] != [2, 3, 8],
            "min-predecessor mutant survived")
    require(verify_dag([], [], []) == [], "empty DAG")
    require(verify_dag([0, 0], [0, 0], [(0, 1, 0)]) == [0, 0],
            "zero duration")
    expect_value_error(earliest, [1, 1], [0, 0], [(0, 1, 0), (1, 0, 0)])
    expect_value_error(earliest, [-1], [0], [])
    expect_value_error(earliest, [1], [], [])
    expect_value_error(earliest, [1], [0], [(0, 1, 0)])
    expect_value_error(earliest, [1, 1], [0, 0], [(0, 1, -1)])
    expect_value_error(earliest, [1, 1], [0, 0], [(0, 1, 0), (0, 1, 1)])
    for _ in range(300):
        n = rng.randrange(1, 8)
        d = [rng.randrange(7) for _ in range(n)]
        r = [rng.randrange(13) for _ in range(n)]
        labels = list(range(n))
        rng.shuffle(labels)
        edges = [(labels[i], labels[j], rng.randrange(5))
                 for i in range(n) for j in range(i + 1, n)
                 if rng.random() < 0.25]
        verify_dag(d, r, edges)

    require(clock_bounds(10, 15, -2, 4) == (1, 7), "clock example")
    require(clock_bounds(10, 15, 4, 8) == (0, 1), "causal intersection")
    expect_value_error(clock_bounds, 10, 15, 6, 8)
    expect_value_error(clock_bounds, 10, 15, 4, -2)
    require(clock_bounds(10, 15, -2, 4) != (5, 5),
            "uncalibrated-point mutant survived")
    for offset in range(-3, 4):
        for elapsed in range(8):
            received = 10 + offset + elapsed
            lo, hi = clock_bounds(10, received, offset - 2, offset + 3)
            feasible = [received - 10 - delta
                        for delta in range(offset - 2, offset + 4)
                        if received - 10 - delta >= 0]
            require((lo, hi) == (min(feasible), max(feasible)),
                    "offset enumeration mismatch")
            require(lo <= elapsed <= hi, "true elapsed excluded")
    outbound, processing, inbound = 2, 1, 5
    require((outbound + processing + inbound) / 2 != outbound,
            "half-RTT mutant survived")
    print("CPU integer interval model: 308 cases match independent unit cells")
    print("CPU DAG model: 300 random DAGs plus release/lag/join/empty cases match all-path enumeration")
    print("Clock model: causal bounds checked against 56 enumerated cases")
    print("Detected mutants: sum-as-union, start-first, sum-as-path, missing-edge, ignored-release, min-predecessor, point-offset, half-RTT")
    print("Scope: finite CPU models only; no CUDA, DMA, profiler, GPU or service measurements")


if __name__ == "__main__":
    main()
