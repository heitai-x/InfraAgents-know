#!/usr/bin/env python3
"""Deterministic finite-model checks for K-TOOLS-007; no service measurements.

Only standard library, fixed in-file model inputs, no network or writes.
Actual execution evidence is recorded separately from these algorithms.
"""
from fractions import Fraction as F
from itertools import product
from math import ceil, comb, floor, log, sqrt
from statistics import NormalDist
import json


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def precision_n(sigma, h, z):
    if sigma <= 0 or h <= 0 or z <= 0:
        raise ValueError("positive known population sigma, halfwidth and z required")
    return ceil((z * sigma / h) ** 2)


def detection_n(sigma, margin, alpha, beta):
    if sigma <= 0 or margin <= 0 or not 0 < alpha < .5 or not 0 < beta < .5:
        raise ValueError("known-normal model requires positive sigma/margin, alpha,beta in (0,.5)")
    z_sum = NormalDist().inv_cdf(1 - alpha) + NormalDist().inv_cdf(1 - beta)
    return ceil((z_sum * sigma / margin) ** 2)


def required_zero(alpha, qmax, strict=True):
    """Logarithmic proposal, repaired/verified by exact rational probabilities."""
    alpha, qmax = F(alpha), F(qmax)
    if not 0 < alpha < 1 or not 0 < qmax < 1:
        raise ValueError("alpha and qmax must be strictly inside (0,1)")
    ratio = log(float(alpha)) / log(1 - float(qmax))
    guess = max(1, floor(ratio) + 1 if strict else ceil(ratio))

    def meets(n):
        zero_probability = (1 - qmax) ** n
        return zero_probability < alpha if strict else zero_probability <= alpha

    while not meets(guess):
        guess += 1
    while guess > 1 and meets(guess - 1):
        guess -= 1
    return guess


def zero_reference(alpha, qmax, strict):
    """Independent sequential rational search; never uses logarithms/rounding."""
    probability = F(1)
    for n in range(1, 10000):
        probability *= 1 - qmax
        if probability < alpha or (not strict and probability == alpha):
            return n
    raise AssertionError("reference limit exceeded")


def upper_tail(n, s):
    if not 0 <= s <= n:
        raise ValueError("binomial success count out of range")
    return F(sum(comb(n, k) for k in range(s, n + 1)), 2 ** n)


def lower_tail(n, s, q0):
    q0 = F(q0)
    if not 0 <= s <= n or not 0 < q0 < 1:
        raise ValueError("invalid lower-tail inputs")
    return sum((F(comb(n, k)) * q0 ** k * (1 - q0) ** (n - k)
                for k in range(s + 1)), F(0))


def validate_plan(looks, amounts):
    if not looks or len(looks) != len(amounts):
        raise ValueError("one positive amount for each planned look")
    if any(type(n) is not int or n <= 0 for n in looks):
        raise ValueError("look sizes must be positive integers")
    if any(a >= b for a, b in zip(looks, looks[1:])):
        raise ValueError("look sizes must increase")
    if any(not 0 < amount < 1 for amount in amounts):
        raise ValueError("invalid error amounts")


def first_rejection_enumeration(looks, amounts):
    """Independent bit-sequence enumeration, including all early-stop paths."""
    validate_plan(looks, amounts)
    limit = looks[-1]
    first_counts = [0] * len(looks)
    survivors = 0
    for path in product((0, 1), repeat=limit):
        for j, (n, amount) in enumerate(zip(looks, amounts)):
            # Direct count over the frozen prefix, no DP state reused.
            s = sum(path[:n])
            if upper_tail(n, s) < amount:
                first_counts[j] += 1
                break
        else:
            survivors += 1
    require(sum(first_counts) + survivors == 2 ** limit, "enumeration probability conservation")
    return [F(count, 2 ** limit) for count in first_counts], F(survivors, 2 ** limit)


def first_rejection_dp(looks, amounts):
    """Count prefixes not yet rejected; lift removed prefixes to final length."""
    validate_plan(looks, amounts)
    limit = looks[-1]
    look_map = dict(zip(looks, range(len(looks))))
    alive = {0: 1}
    first_counts = [0] * len(looks)
    for n in range(1, limit + 1):
        advanced = {}
        for successes, count in alive.items():
            advanced[successes] = advanced.get(successes, 0) + count
            advanced[successes + 1] = advanced.get(successes + 1, 0) + count
        alive = advanced
        if n in look_map:
            j = look_map[n]
            rejected = [s for s in alive if upper_tail(n, s) < amounts[j]]
            for s in rejected:
                first_counts[j] += alive.pop(s) * 2 ** (limit - n)
    survivor_count = sum(alive.values())
    require(sum(first_counts) + survivor_count == 2 ** limit, "DP probability conservation")
    return [F(count, 2 ** limit) for count in first_counts], F(survivor_count, 2 ** limit)


def fixed_size(n, amount):
    return sum((F(comb(n, s), 2 ** n) for s in range(n + 1)
                if upper_tail(n, s) < amount), F(0))


def planned_trial(streams, looks, amounts, confirmation, final_amounts, hard_failure=False,
                  total_alpha=F(1, 20)):
    """One candidate, multiple necessary coin endpoints, fresh final sample.

    This is the chapter's stopping protocol instantiated in the finite coin model.
    It consumes already supplied labeled streams, never generates device events.
    Confirmation identity/disjointness must be established by the real caller;
    a Python object being distinct cannot prove statistical independence.
    """
    validate_plan(looks, amounts)
    if not streams or len(streams) != len(confirmation) or len(streams) != len(final_amounts):
        raise ValueError("endpoint shape mismatch")
    if any(not 0 < a < 1 for a in final_amounts):
        raise ValueError("invalid confirmation amount")
    if len(streams) * sum(amounts) + sum(final_amounts) > total_alpha:
        raise ValueError("planned endpoint/look/confirmation error budget exceeded")
    for path in list(streams) + list(confirmation):
        if any(type(x) is not int or x not in (0, 1) for x in path):
            raise ValueError("coin paths must contain integer 0/1")
    log_rows = []
    if hard_failure:
        return "REJECTED", log_rows
    selected = False
    for n, amount in zip(looks, amounts):
        if any(len(path) < n for path in streams):
            return "INVALID", log_rows
        values = [upper_tail(n, sum(path[:n])) for path in streams]
        log_rows.append((n, values))
        if all(p < amount for p in values):
            selected = True
            break
    if not selected:
        return "NOT_CONFIRMED", log_rows
    if any(len(path) == 0 for path in confirmation):
        return "INVALID", log_rows
    final_values = [upper_tail(len(path), sum(path)) for path in confirmation]
    log_rows.append(("confirm", final_values))
    return ("CONFIRMED" if all(p < a for p, a in zip(final_values, final_amounts))
            else "NOT_CONFIRMED"), log_rows


def main():
    negative_controls = []

    # Rational inequality reference is independent of the sqrt/ceil implementation.
    for sigma, h, expected in [(4, F(1), 62), (4, F(1, 2), 246)]:
        z = F(49, 25)
        n = precision_n(F(sigma), h, z)
        ref = next(r for r in range(1, 1000) if h * h * r >= z * z * sigma * sigma)
        require(n == ref == expected, "known-normal precision inversion")
        require(z * sigma / sqrt(n) <= h, "precision target met")
        require(z * sigma / sqrt(n - 1) > h, "precision minimality")
    require(detection_n(4, 2, .05, .2) == 25, "model detection planning at sigma=4")
    require(detection_n(6, 2, .05, .2) == 56, "variance sensitivity")
    for sigma in (4, 6):
        n = detection_n(sigma, 2, .05, .2)
        z_alpha = NormalDist().inv_cdf(.95)
        power = NormalDist().cdf(2 * sqrt(n) / sigma - z_alpha)
        prior = NormalDist().cdf(2 * sqrt(n - 1) / sigma - z_alpha)
        require(power >= .8 and prior < .8, "model power target and minimality")
    feasible = [r for r in range(1, 301) if r * 30 <= 900]
    require(max(feasible) == 30 and 25 in feasible and 56 not in feasible and 246 not in feasible,
            "complete paired-block resource budget")
    require(246 * 30 == 7380, "precision resource cost")

    for alpha in (F(1, 20), F(1, 100), F(1, 4)):
        for qmax in (F(1, 100), F(1, 2)):
            for strict in (False, True):
                require(required_zero(alpha, qmax, strict) == zero_reference(alpha, qmax, strict),
                        "zero-exceedance rational reference")
    require(required_zero(F(1, 20), F(1, 100)) == 299, "strict 95% below 1%")
    require(required_zero(F(1, 100), F(1, 100)) == 459, "allocated 99% below 1%")
    require(required_zero(F(1, 4), F(1, 2), False) == 2 and
            required_zero(F(1, 4), F(1, 2), True) == 3, "strict boundary equality")
    require(F(99, 100) ** 298 > F(1, 20) > F(99, 100) ** 299, "299 minimal exact probabilities")
    wrong_strict = ceil(log(.25) / log(.5))
    require(wrong_strict == 2 and F(1, 2) ** wrong_strict == F(1, 4), "strict ceil mutant exposed")
    negative_controls.append("strict-boundary-ceil")
    require(lower_tail(299, 0, F(1, 100)) == F(99, 100) ** 299,
            "zero branch agrees with exact lower-tail test")
    require(upper_tail(8, 8) == F(1, 256) and lower_tail(8, 8, F(1, 2)) == 1,
            "tail direction mutant exposed")
    negative_controls.append("wrong-tail-direction")

    plans = [((5, 8), (F(1, 20), F(1, 20))),
             ((8, 16), (F(1, 40), F(1, 40))),
             ((5, 8), (F(1, 40), F(1, 40)))]
    cumulative = []
    for looks, amounts in plans:
        enum_rejections, enum_survival = first_rejection_enumeration(looks, amounts)
        dp_rejections, dp_survival = first_rejection_dp(looks, amounts)
        require(enum_rejections == dp_rejections and enum_survival == dp_survival,
                "DP and independent complete sequence enumeration")
        sizes = [fixed_size(n, a) for n, a in zip(looks, amounts)]
        require(all(size <= amount for size, amount in zip(sizes, amounts)), "fixed-test sizes")
        total = sum(dp_rejections, F(0))
        require(total <= sum(sizes) <= sum(amounts), "finite-look union bound")
        cumulative.append(total)
    naive = F(8 + 9 - 4, 256)
    spent = F(256 + 697 - 93, 65536)
    require(cumulative[0] == naive == F(13, 256) > F(1, 20), "naive peeking exceeds nominal 5%")
    require(cumulative[1] == spent == F(215, 16384) <= F(1, 20), "planned-spending model bound")
    require(cumulative[2] == F(1, 256), "coarse small-n spending")
    require(fixed_size(8, F(1, 20)) == F(9, 256) != naive, "last-look-only mutant exposed")
    negative_controls.append("last-look-only-misses-early-stops")
    require(1 - F(31, 32) ** 4 > F(1, 20), "four independent unadjusted claim opportunities")

    # All required endpoints at one look, then an independent fixed final sample.
    looks, amounts = (8, 16), (F(1, 100), F(1, 100))
    heads, tails = [1] * 16, [0] * 16
    final_amounts = (F(1, 200), F(1, 200))
    require(2 * sum(amounts) + sum(final_amounts) == F(1, 20), "all-endpoint/look/final budget ledger")
    require(8 * F(1, 400) + 2 * F(3, 200) == F(1, 20), "two-candidate chapter budget example")
    state, _ = planned_trial((heads, heads), looks, amounts, (heads, heads), final_amounts)
    require(state == "CONFIRMED", "all gates and final confirmation")
    state, rows = planned_trial((heads, heads), looks, amounts, (heads, tails), final_amounts)
    require(rows[0][0] == 8 and state == "NOT_CONFIRMED", "screen pass does not skip confirmation")
    negative_controls.append("skip-independent-confirmation")
    state, rows = planned_trial((heads, tails), looks, amounts, (heads, heads), final_amounts)
    require(state == "NOT_CONFIRMED" and len(rows) == 2, "all gates required, bounded final look")
    require(any(p < amounts[0] for p in rows[0][1]), "any-gate mutant would accept at first look")
    negative_controls.append("any-gate-instead-of-all")
    state, _ = planned_trial((heads[:7], heads), looks, amounts, (heads, heads), final_amounts)
    require(state == "INVALID", "incomplete planned prefix invalid")
    state, _ = planned_trial((heads, heads), looks, amounts, (heads, heads), final_amounts, True)
    require(state == "REJECTED", "hard correctness failure")
    state, _ = planned_trial((heads, heads), looks, amounts, ([], heads), final_amounts)
    require(state == "INVALID", "missing confirmation invalid")

    print(json.dumps({
        "scope": "known-normal planning arithmetic; exact iid Bernoulli finite-look models; protocol branches",
        "precision_R": [62, 246], "design_detection_R": [25, 56],
        "zero_exceedance_n": {"strict_95_below_1pct": 299, "strict_99_below_1pct": 459},
        "naive_first_rejection_probability": str(naive),
        "spent_first_rejection_probability": str(spent),
        "negative_controls_detected": negative_controls,
        "limits": ["not service-tail coverage", "no GPU or CPU performance measurement",
                   "no general bootstrap/sequential theorem", "no causal knowledge-benefit experiment"]
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
