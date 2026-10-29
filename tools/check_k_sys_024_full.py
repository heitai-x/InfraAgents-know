#!/usr/bin/env python3
"""K-SYS-024: deterministic CPU discrete models; no device or timing results."""
from dataclasses import dataclass, replace
from fractions import Fraction as F
from itertools import product, permutations


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def detected(thunk, label):
    try:
        thunk()
    except AssertionError:
        print("negative control detected:", label)
    else:
        raise AssertionError("undetected negative control: " + label)


def check_cost():
    def formula(epochs, build, fast, slow):
        return len(epochs) * build + sum(a * fast + b * slow for a, b in epochs)

    def ledger(epochs, build, fast, slow):
        entries = []
        for covered, missed in epochs:
            require(covered >= 0 and missed >= 0, "nonnegative call counts")
            entries.append(build)
            entries.extend([fast] * covered)
            entries.extend([slow] * missed)
        return sum(entries, F(0))

    for epochs in ([], [(0, 0)], [(3, 2)], [(1, 0), (0, 2)], [(2, 1)] * 4):
        for build, fast, slow in product((F(0), F(7)), (F(2), F(5)), (F(3), F(9))):
            require(formula(epochs, build, fast, slow) ==
                    ledger(epochs, build, fast, slow), "cost formula vs per-call ledger")
    build, fast, base, slow = F(40000), F(5), F(10), F(12)
    one = formula([(100000, 0)], build, fast, slow)
    repeated = formula([(1000, 0)] * 100, build, fast, slow)
    require((one, repeated) == (F(540000), F(4500000)), "rebuild totals")
    detected(lambda: require(one == repeated, "once-only build omits epochs"), "ignored rebuilds")
    for covered, expected in ((80000, 680000), (20000, 1100000)):
        total = formula([(covered, 100000 - covered)], build, fast, slow)
        require(total == expected, "fallback accounting")
        print("coverage and total_us:", F(covered, 100000), total)
    for f, tf, threshold in ((F(1), base, F(8000)), (F(4, 5), slow, F(100000, 9))):
        gain = base - (f * fast + (1 - f) * tf)
        require(build / gain == threshold, "break-even threshold")
        require(build + threshold * (f * fast + (1 - f) * tf) == threshold * base,
                "equality is break-even, not strict gain")
        require(build + (threshold + 1) * (f * fast + (1 - f) * tf)
                < (threshold + 1) * base, "strict gain after threshold")
    require(base - (F(1, 5) * fast + F(4, 5) * slow) < 0, "no finite break-even")
    require(256 * 1024**2 // (128 * 1024) == 2048, "pool KV opportunity cost")
    require((256 + 16) * 1024**2 // (128 * 1024) == 2176, "independent workspaces")
    print("epoch totals_us:", one, repeated)


@dataclass(frozen=True)
class Bucket:
    n_max: int
    q_max: int
    p_max: int
    q_item_max: int
    p_item_max: int


def feasible_batch(b, rows):
    rows = tuple(rows)
    return (len(rows) <= b.n_max
            and all(0 <= q <= b.q_item_max and 0 <= p <= b.p_item_max for q, p in rows)
            and sum(q for q, _ in rows) <= b.q_max
            and sum(p for _, p in rows) <= b.p_max)


def placement_reference(b, rows):
    # Independent bounded arrays: actually place every query and index entry.
    requests = [None] * b.n_max
    query = [None] * b.q_max
    pages = [None] * b.p_max
    qi = pi = 0
    try:
        for i, (q, p) in enumerate(rows):
            if q < 0 or p < 0:
                return False
            requests[i] = i
            local_q = [None] * b.q_item_max
            local_p = [None] * b.p_item_max
            for j in range(q):
                local_q[j] = j
                query[qi] = (i, j)
                qi += 1
            for j in range(p):
                local_p[j] = j
                pages[pi] = (i, j)
                pi += 1
    except IndexError:
        return False
    return True


def per_request_only(b, rows):
    return all(0 <= q <= b.q_item_max and 0 <= p <= b.p_item_max for q, p in rows)


def check_bucket():
    original = Bucket(2, 64, 16, 64, 4)
    require(not feasible_batch(original, [(64, 16)]),
            "recorded failure: aggregate boundary does not override item limit 4")
    require(not placement_reference(original, [(64, 16)]), "reference agrees on original fixture")
    require(feasible_batch(original, [(64, 4)]), "query boundary with legal item pages")
    exact = Bucket(2, 64, 16, 64, 8)
    require(feasible_batch(exact, [(32, 8), (32, 8)]), "all aggregate bounds at equality")
    require(placement_reference(exact, [(32, 8), (32, 8)]), "exact bounded placement")
    require(not feasible_batch(exact, [(32, 9)]), "item pages fail below aggregate limit")
    fixtures = (
        (original, [(40, 4), (40, 4)], "aggregate Q"),
        (Bucket(2, 20, 16, 10, 9), [(10, 9), (10, 8)], "aggregate P"),
        (Bucket(2, 100, 100, 100, 100), [(1, 1)] * 3, "request count"),
    )
    for b, rows, label in fixtures:
        require(per_request_only(b, rows), "mutant must actually accept " + label)
        require(not placement_reference(b, rows), "bounded reference rejects " + label)
        detected(lambda b=b, rows=rows: require(per_request_only(b, rows) ==
                 placement_reference(b, rows), "mutant differs from placement"), label)
    checked = 0
    for b in (Bucket(2, 3, 3, 2, 2), Bucket(1, 2, 1, 3, 2), Bucket(0, 0, 0, 0, 0)):
        choices = tuple(product(range(-1, 5), repeat=2))
        for n in range(4):
            for rows in product(choices, repeat=n):
                require(feasible_batch(b, rows) == placement_reference(b, rows),
                        "capacity predicate differs from actual placement")
                checked += 1
    print("bucket cases:", checked, "; original (64,16) rejected; exact batch accepted")


@dataclass(frozen=True)
class Signature:
    model: str = "m1"
    tokenizer: str = "t1"
    adapter: str = "a1"
    dtype_layout: str = "bf16/packed1"
    backend: str = "b1"
    runtime: str = "r1"
    device: str = "uuid1"
    tenant: str = "tenant1"
    prefix: str = "prefix1"


def guard(plan, request, length, budget):
    # This illustrative plan freezes every Signature field.
    return plan == request and 1 <= length <= 1024 and 0 <= budget <= 64


def check_guard():
    plan = Signature()
    require(guard(plan, plan, 1024, 64), "closed upper bounds")
    for length, budget in ((0, 1), (1025, 1), (1, -1), (1, 65)):
        require(not guard(plan, plan, length, budget), "range violation")
    for field in plan.__dataclass_fields__:
        other = replace(plan, **{field: "different"})
        require(not guard(plan, other, 512, 32), "identity field: " + field)
        detected(lambda other=other: require(True == guard(plan, other, 512, 32),
                 "shape-only mutant accepted identity mismatch"), "shape-only " + field)
    print("identity fields checked:", len(plan.__dataclass_fields__))


class Plan:
    def __init__(self, generation):
        self.generation = generation
        self.phase = "READY"
        self.pending = {}
        self.next_call = 0
        self.cancelled = set()
        self.committed = set()

    def admit(self):
        if self.phase != "READY":
            return None
        ticket = (self.generation, self.next_call)
        self.next_call += 1
        self.pending[ticket] = {"device", "copy", "consumer"}
        return ticket

    def invalidate(self):
        if self.phase != "READY":
            return False
        self.phase = "DRAINING"
        self.retire_if_empty()
        return True

    def retire_if_empty(self):
        if self.phase == "DRAINING" and not self.pending:
            self.phase = "RETIRED"

    def cancel(self, ticket):
        if ticket not in self.pending:
            return False
        self.cancelled.add(ticket)
        return True

    def complete(self, ticket, role):
        owed = self.pending.get(ticket)
        if owed is None or role not in owed:
            return False
        if role == "copy" and "device" in owed:
            return False
        if role == "consumer" and ("device" in owed or "copy" in owed):
            return False
        owed.remove(role)
        if not owed:
            if ticket not in self.cancelled:
                self.committed.add(ticket)
            del self.pending[ticket]
        self.retire_if_empty()
        return True


def no_early_reclaim(plan, obligations):
    require(not (plan.phase == "RETIRED" and obligations),
            "retired while explicit access obligations remain")


def check_state():
    # Independent reference is the explicit event responsibility set.
    for cancelled in (False, True):
        for order in permutations(("device", "copy", "consumer")):
            p = Plan(1)
            ticket = p.admit()
            owed = {"device", "copy", "consumer"}
            if cancelled:
                p.cancel(ticket)
            p.invalidate()
            require(p.admit() is None, "draining cannot admit new work")
            for role in order:
                permitted = (role == "device" or
                             role == "copy" and "device" not in owed or
                             role == "consumer" and not ({"device", "copy"} & owed))
                before = {k: set(v) for k, v in p.pending.items()}
                result = p.complete(ticket, role)
                require(result == permitted, "dependency order")
                if permitted:
                    owed.remove(role)
                else:
                    require(before == p.pending, "rejected event cannot mutate responsibilities")
                actual = set().union(*p.pending.values()) if p.pending else set()
                require(actual == owed, "explicit responsibility reference")
                no_early_reclaim(p, owed)
            # Retry rejected early notifications after prerequisites have completed.
            for role in ("device", "copy", "consumer"):
                if role in owed:
                    require(p.complete(ticket, role), "ordered completion retry")
                    owed.remove(role)
            require(p.phase == "RETIRED", "last consumer retires")
            require((ticket in p.committed) == (not cancelled), "cancel suppresses commit")
            require(not p.complete(ticket, "consumer"), "duplicate completion rejected")
    empty = Plan(5)
    empty.invalidate()
    require(empty.phase == "RETIRED", "zero readers retire without waiting for absent event")
    old, new = Plan(8), Plan(9)
    old_ticket, new_ticket = old.admit(), new.admit()
    old.invalidate()
    snapshot = {k: set(v) for k, v in new.pending.items()}
    require(not new.complete(old_ticket, "device"), "late old notification rejected")
    require(snapshot == new.pending and new_ticket in new.pending, "new generation unchanged")
    faulty = Plan(11)
    faulty.admit()
    faulty.invalidate()
    faulty.phase = "RETIRED"  # Actual faulty transition.
    detected(lambda: no_early_reclaim(faulty, {"device", "copy", "consumer"}),
             "release before last reader")
    wanted_epoch = 2
    def can_publish(build_epoch, phase):
        return build_epoch == wanted_epoch and phase == "VERIFIED"
    require(not can_publish(1, "VERIFIED"), "old build cannot publish")
    require(not can_publish(2, "BUILDING"), "partial build cannot publish")
    require(not can_publish(2, "DRAINING"), "draining cannot republish")
    require(can_publish(2, "VERIFIED"), "matching verified build can publish")
    detected(lambda: require((1 == 1) == can_publish(1, "VERIFIED"),
             "mutant ignores wanted epoch"), "stale build publication")
    print("state: completion permutations, cancellation, zero readers, duplicate and late events")


def check_packing():
    def pack(w, n, k, bn, bk, wrong=False):
        nt, kt = (n + bn - 1) // bn, (k + bk - 1) // bk
        buf = [0] * (nt * kt * bn * bk)
        for row in range(n):
            for col in range(k):
                no, ni = divmod(row, bn)
                ko, ki = divmod(col, bk)
                local = ni * bk + ki if wrong else ki * bn + ni
                index = (no * kt + ko) * bk * bn + local
                buf[index] = w[row][col]
        return buf

    def consume(buf, x, n, k, bn, bk):
        kt = (k + bk - 1) // bk
        out = []
        for row in range(n):
            no, ni = divmod(row, bn)
            value = 0
            for ko in range(kt):
                for ki in range(bk):
                    col = ko * bk + ki
                    weight = buf[(no * kt + ko) * bk * bn + ki * bn + ni]
                    value += (x[col] if col < k else 0) * weight
            out.append(value)
        return out

    cases = 0
    for n, k, bn, bk in product(range(5), range(5), range(1, 4), range(1, 4)):
        w = [[(i + 1) * 7 - j * 3 for j in range(k)] for i in range(n)]
        x = [j - 2 for j in range(k)]
        direct = [sum(x[j] * w[i][j] for j in range(k)) for i in range(n)]
        buf = pack(w, n, k, bn, bk)
        require(consume(buf, x, n, k, bn, bk) == direct, "packed vs direct scalar matvec")
        cases += 1
    w, x = [[1, 2], [3, 4]], [1, 2]
    bad = consume(pack(w, 2, 2, 2, 2, True), x, 2, 2, 2, 2)
    detected(lambda: require(bad == [5, 11], "wrong local axis changes model"), "packing axes")
    print("packing cases:", cases, "; exact integer arithmetic, no low-precision result")


def check_segments():
    # Closed interval formula vs byte-by-byte segment enumeration.
    for offset in range(32):
        addresses = [offset + 4 * lane for lane in range(32)]
        reference = {byte // 32 for a in addresses for byte in range(a, a + 4)}
        lo, hi = min(addresses), max(addresses) + 3
        count = hi // 32 - lo // 32 + 1
        require(count == len(reference), "contiguous warp segment coverage")
    aligned = {byte // 32 for byte in range(128)}
    shifted = {byte // 32 for byte in range(4, 132)}
    require((len(aligned), len(shifted)) == (4, 5), "four vs five segments")
    detected(lambda: require(128 // 32 == len(shifted), "logical-byte mutant ignores alignment"),
             "logical bytes treated as transaction bytes")
    print("32-byte segment model: aligned=4 shifted=5; no DRAM observation")


def main():
    print("scope: CPU discrete model only; no CUDA, model forward, graph, sampler, DMA or timing")
    check_cost()
    check_bucket()
    check_guard()
    check_state()
    check_packing()
    check_segments()
    print("observations complete: deterministic algebra, bounded placement and event models only")


if __name__ == "__main__":
    main()
