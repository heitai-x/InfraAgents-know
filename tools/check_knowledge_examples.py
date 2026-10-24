#!/usr/bin/env python3
"""Recalculate the knowledge edition's explicit synthetic examples.

Only the Python standard library is used. This script does not import a target
engine, simulate its implementation, invoke a GPU, or measure model/service
performance. Inputs are the declared example fixtures, not observed hardware
facts. Chapter hashes identify the accompanying prose; passing the arithmetic
does not establish that every statement in those chapters is correct.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


KIB = 2**10
MIB = 2**20
GIB = 2**30
ROOT = Path(__file__).resolve().parents[1]


def equal(actual, expected):
    """Use exact comparison for discrete facts, tolerance only for real values."""
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual == expected
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(equal(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, (list, tuple)):
        return (isinstance(actual, (list, tuple)) and len(actual) == len(expected)
                and all(equal(a, e) for a, e in zip(actual, expected)))
    if isinstance(actual, float) or isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


def example(groups, family, name, path, section, inputs, assumptions):
    item = {"family": family, "example": name, "document": path,
            "section": section, "inputs": inputs, "assumptions": assumptions,
            "evidence_class": "synthetic_arithmetic_or_counterexample", "checks": []}
    groups.append(item)
    return item


def check(item, name, actual, expected, unit, relation):
    item["checks"].append({"check": name, "actual": actual, "expected": expected,
                           "unit": unit, "relation": relation,
                           "passed": equal(actual, expected)})


def peak_live(intervals):
    # Half-open lifetimes: a slot finishing at t can be reused starting at t.
    events = sorted([(start, 1) for start, end in intervals]
                    + [(end, -1) for start, end in intervals])
    live = peak = 0
    for _, change in events:
        live += change
        peak = max(peak, live)
    return peak


def communication_examples(groups):
    item = example(groups, "communication", "four_stage_pipeline",
                   "knowledge/hardware/host-device-transfer.md", "设定算例：阶段、槽位与共享带宽",
                   {"stage_ms": [3, 2, 8, 1], "batches": 8, "period_ms": 8,
                    "input_MiB": 16, "output_MiB": 1},
                   ["Independent deterministic stage resources, no extra dependencies.",
                    "Each input lives from H2D start to GPU completion; each output from GPU start to D2H completion.",
                    "Half-open lifetimes permit immediate reuse at a completion boundary."])
    stages_ms = item["inputs"]["stage_ms"]
    batches = []
    for batch in range(item["inputs"]["batches"]):
        t = batch * item["inputs"]["period_ms"]
        intervals = []
        for duration in stages_ms:
            intervals.append((t, t + duration))
            t += duration
        batches.append(intervals)
    check(item, "eight_batch_pipeline_completion_ms", batches[-1][-1][1], 70,
          "ms", "sum(stage_ms) + (8 - 1) * max(stage_ms)")
    check(item, "eight_batch_serial_completion_ms", sum(stages_ms) * len(batches), 112,
          "ms", "8 * sum(stage_ms)")
    check(item, "first_batch_completion_ms", batches[0][-1][1], 14, "ms", "sum(stage_ms)")
    check(item, "stage_resource_non_overlap",
          all(batches[b][s][1] <= batches[b + 1][s][0]
              for s in range(4) for b in range(7)), True,
          "predicate", "No two batches occupy the same stage resource simultaneously.")
    input_slots = peak_live([(b[1][0], b[2][1]) for b in batches])
    output_slots = peak_live([(b[2][0], b[3][1]) for b in batches])
    check(item, "input_slots_from_timeline", input_slots, 2, "slots", "ceil(10 ms / 8 ms)")
    check(item, "output_slots_from_timeline", output_slots, 2, "slots", "ceil(9 ms / 8 ms)")
    check(item, "device_buffers_mib", input_slots * 16 + output_slots * 1, 34,
          "MiB", "input_slots * 16 MiB + output_slots * 1 MiB; other allocations excluded")

    item = example(groups, "communication", "shared_dram_bound",
                   "knowledge/hardware/host-device-transfer.md", "设定算例：阶段、槽位与共享带宽",
                   {"traffic_MiB_per_batch": 128, "shared_bandwidth_GiB_per_s": 8},
                   ["Traffic is aggregated for the same shared DRAM resource.",
                    "Bandwidth is an assumed sustainable concurrent value, not a platform specification."])
    check(item, "shared_dram_period_ms", 128 * MIB / (8 * GIB) * 1000, 15.625,
          "ms/batch", "128 MiB / (8 GiB/s), converted from seconds to milliseconds")

    item = example(groups, "communication", "identifiable_collective_values",
                   "knowledge/system-semantics/collective-ordering.md",
                   "用可辨识反例检查所选通信边界",
                   {"ranks": 4, "elements_per_rank": 8, "input_value": "100 * rank + index"},
                   ["Exact small integers; this is not floating-point validation or an NCCL execution.",
                    "ReduceScatter uses 32 input elements per rank and 8 output elements per rank."])
    summed = [sum(100 * r + i for r in range(4)) for i in range(8)]
    check(item, "identifiable_allreduce", summed, [600 + 4 * i for i in range(8)],
          "integer values", "sum over ranks of (100 * rank + index)")
    gathered = [[100 * r + i for i in range(8)] for r in range(4)]
    check(item, "allgather_block_origin", [b[0] for b in gathered], [0, 100, 200, 300],
          "integer values", "rank-ordered blocks retain their distinct first element")
    sums32 = [sum(100 * r + i for r in range(4)) for i in range(32)]
    check(item, "reduce_scatter_blocks", [sums32[8*r:8*r+8] for r in range(4)],
          [[600 + 4 * i for i in range(8*r, 8*r+8)] for r in range(4)],
          "integer values", "rank r receives the reduced global indices [8*r, 8*r+8)")

    item = example(groups, "communication", "ideal_ring_cost",
                   "knowledge/hardware/topology-parallelism.md",
                   "集合操作怎样使用链路",
                   {"ranks": 4, "startup_us_per_round": 5, "one_way_bandwidth_GiB_per_s": 40,
                    "large_MiB": 16, "small_KiB": 8},
                   ["A selected ideal ring; no claim about the algorithm chosen by NCCL.",
                    "Full-duplex send/receive overlap, uniform chunks, stable one-way bandwidth, no contention."])
    p, alpha, bandwidth = 4, Fraction(5, 10**6), 40 * GIB
    def ring_seconds(size):
        return 2 * (p - 1) * alpha + Fraction(2 * (p - 1) * size, p * bandwidth)
    check(item, "ring_large_message_ms", float(ring_seconds(16 * MIB) * 1000), 0.6159375,
          "ms", "2*(p-1)*alpha + 2*(p-1)*A/(p*B)")
    check(item, "ring_small_message_us", float(ring_seconds(8 * KIB) * 10**6), 30.286102294921875,
          "us", "same model; seconds converted to microseconds")


def systems_examples(groups):
    item = example(groups, "systems", "slow_client_queue",
                   "knowledge/system-semantics/service-protocol.md", "一个慢客户端的容量算例",
                   {"queue_enqueue_B_per_s": 8000, "queue_dequeue_B_per_s": 2000,
                    "capacity_KiB": 64, "initial_B": 0, "requests": 128, "private_state_MiB_each": 6},
                   ["Rates refer to the same bounded queue, remain constant before saturation, and exclude packet overhead.",
                    "Remote consumption alone does not determine local dequeue rate; downstream buffers are excluded."])
    capacity, growth = 64 * KIB, 8000 - 2000
    check(item, "queue_growth_and_residency",
          {"capacity_B": capacity, "net_B_per_s": growth, "full_after_s": capacity / growth,
           "output_buffers_MiB": 128 * capacity / MIB, "private_state_MiB": 128 * 6},
          {"capacity_B": 65536, "net_B_per_s": 6000, "full_after_s": 10.922666666666666,
           "output_buffers_MiB": 8, "private_state_MiB": 768},
          "field suffixes", "(capacity - initial)/(enqueue - dequeue); separate output and model-state allocations")

    item = example(groups, "systems", "service_accounting",
                   "knowledge/system-semantics/service-protocol.md", "验收必须包含未完成和被拒绝的请求",
                   {"window_s": 10, "offered": 100, "rejected": 10, "completed": 80, "qualified": 72},
                   ["Qualified means the task's quality and latency gates; not all completed requests qualify.",
                    "Pending outcomes must be followed after the window; they are not silently discarded."])
    check(item, "population_and_qualified_rate",
          {"accepted": 100 - 10, "pending": 100 - 10 - 80, "qualified_per_s": 72 / 10,
           "qualified_fraction_offered": 72 / 100, "qualified_fraction_completed": 72 / 80},
          {"accepted": 90, "pending": 10, "qualified_per_s": 7.2,
           "qualified_fraction_offered": 0.72, "qualified_fraction_completed": 0.9},
          "counts, requests/s, dimensionless fractions", "offered=rejected+accepted; accepted=completed+pending; denominators stay distinct")

    item = example(groups, "systems", "host_cpu_budget",
                   "knowledge/system-semantics/host-runtime.md", "有效核数与线程组织的候选比较",
                   {"allowed_CPUs": 8, "quota_cpu_ms": 400, "period_wall_ms": 100,
                    "request_rate_per_s": 200, "request_cpu_ms": [10, 2, 3], "background_cpu_s_per_s": 1.2},
                   ["No stricter ancestor limit; quota is a time ceiling, not guaranteed immediate execution."])
    per_request = sum([10, 2, 3])
    foreground = Fraction(200 * per_request, 1000)
    total = foreground + Fraction(6, 5)
    budget = min(8, Fraction(400, 100))
    check(item, "cpu_time_units",
          {"core_equivalent_limit": float(budget), "cpu_ms_per_request": per_request,
           "foreground_cpu_s_per_s": float(foreground), "total_cpu_s_per_s": float(total),
           "demand_exceeds_limit": total > budget},
          {"core_equivalent_limit": 4, "cpu_ms_per_request": 15,
           "foreground_cpu_s_per_s": 3, "total_cpu_s_per_s": 4.2, "demand_exceeds_limit": True},
          "CPU milliseconds/request and CPU seconds/wall-second", "request rate * CPU ms / 1000 plus background CPU time")

    item = example(groups, "systems", "host_buffer_capacity",
                   "knowledge/system-semantics/host-runtime.md", "队列容量必须连接对象大小与同时存活范围",
                   {"input_requests": 512, "input_KiB_each": 64, "token_arrays": 512,
                    "tokens_each": 4096, "token_id_B": 4, "output_queues": 128,
                    "output_KiB_each": 64, "tasks_ahead": 20, "serial_service_ms": 5},
                   ["The three allocations may peak simultaneously and do not share physical storage.",
                    "Metadata, media, IPC, pinned buffers and model state are excluded; queue wait assumes serial fixed work."])
    capacity_parts = [512 * 64 * KIB, 512 * 4096 * 4, 128 * 64 * KIB]
    check(item, "independent_buffers_and_wait",
          {"input_MiB": capacity_parts[0] / MIB, "token_arrays_MiB": capacity_parts[1] / MIB,
           "output_MiB": capacity_parts[2] / MIB, "total_MiB": sum(capacity_parts) / MIB,
           "serial_wait_ms": 20 * 5},
          {"input_MiB": 32, "token_arrays_MiB": 8, "output_MiB": 8, "total_MiB": 48, "serial_wait_ms": 100},
          "MiB and ms", "sum distinct simultaneous allocations; tasks ahead * service time")


def contiguous_strides(shape):
    strides = []
    product = 1
    for size in reversed(shape):
        strides.append(product)
        product *= size
    return list(reversed(strides))


def operator_examples(groups):
    ops = "knowledge/algorithms/operator-primitives.md"
    attn = "knowledge/frameworks/attention-backends.md"
    triton = "knowledge/frameworks/triton-kernels.md"
    item = example(groups, "operators", "layout_offsets", ops, "线性、索引与布局",
                   {"BLHD_shape": [2, 3, 2, 4], "axis_order": [0, 2, 1, 3], "index_BHLD": [0, 1, 0, 0]},
                   ["Offsets and strides count elements, not bytes; view and contiguous layout are separate representations."])
    shape, order, index = [2, 3, 2, 4], [0, 2, 1, 3], [0, 1, 0, 0]
    source_strides = contiguous_strides(shape)
    target_shape = [shape[i] for i in order]
    view_strides = [source_strides[i] for i in order]
    packed_strides = contiguous_strides(target_shape)
    check(item, "stride_and_offset_relations",
          {"source_strides": source_strides, "target_shape": target_shape, "view_strides": view_strides,
           "contiguous_strides": packed_strides,
           "offsets": [sum(i*s for i, s in zip(index, strides)) for strides in [view_strides, packed_strides]]},
          {"source_strides": [24, 8, 4, 1], "target_shape": [2, 2, 3, 4],
           "view_strides": [24, 4, 8, 1], "contiguous_strides": [24, 12, 4, 1], "offsets": [4, 12]},
          "elements", "contiguous stride is the product of later dimensions; offset is the index/stride dot product")

    item = example(groups, "operators", "qkv_geometry", ops, "把线性投影接到真实权重与后端",
                   {"hidden": 4096, "query_heads": 32, "kv_heads": 8, "head_dim": 128, "rows": 32, "element_B": 2},
                   ["One packed QKV projection in the stated geometry; no padding, extra copies or workspace."])
    q, kv = 32 * 128, 8 * 128
    width = q + 2 * kv
    check(item, "packed_qkv_sizes",
          {"Q_width": q, "K_or_V_width": kv, "packed_width": width,
           "output_KiB": 32 * width * 2 / KIB, "Q_KiB": 32 * q * 2 / KIB,
           "K_or_V_KiB": 32 * kv * 2 / KIB, "weight_MiB": 4096 * width * 2 / MIB},
          {"Q_width": 4096, "K_or_V_width": 1024, "packed_width": 6144,
           "output_KiB": 384, "Q_KiB": 256, "K_or_V_KiB": 64, "weight_MiB": 48},
          "elements, KiB, MiB", "Q=Hq*d; K=V=Hkv*d; bytes=element_count*element_B")

    item = example(groups, "operators", "rms_padding_limit", ops, "归一化融合需要保留统计域和活跃 residual",
                   {"true_hidden": 3072, "padded_slots": 4096},
                   ["Nonzero output; the true mean square dominates epsilon.",
                    "This is the limiting ratio, not an exact result for arbitrary finite epsilon."])
    check(item, "padding_scale_limit", math.sqrt(4096 / 3072), 1.1547005383792515,
          "dimensionless output ratio", "sqrt(P/H); finite-epsilon ratio is sqrt((q+epsilon)/(q*H/P+epsilon))")

    item = example(groups, "operators", "residual_traffic", ops, "归一化融合需要保留统计域和活跃 residual",
                   {"rows": 32, "hidden": 4096, "element_B": 2},
                   ["Common gamma reads excluded; independent norm reads u once.",
                    "4N preserves externally live u; 3N is legal only if u has no external consumer.",
                    "Effective read/write bytes, not measured HBM traffic; cache and spill may change physical traffic."])
    size = 32 * 4096 * 2
    check(item, "residual_effective_bytes",
          {"tensor_KiB": size / KIB, "separate_KiB": 5 * size / KIB,
           "fused_live_u_KiB": 4 * size / KIB, "fused_no_external_u_KiB": 3 * size / KIB,
           "saved_live_u_KiB": size / KIB, "saved_no_external_u_KiB": 2 * size / KIB},
          {"tensor_KiB": 256, "separate_KiB": 1280, "fused_live_u_KiB": 1024,
           "fused_no_external_u_KiB": 768, "saved_live_u_KiB": 256, "saved_no_external_u_KiB": 512},
          "KiB", "5N versus 4N or 3N element operations, times bytes per element")

    item = example(groups, "operators", "kv_materialization", attn, "两个可复算的接入反例",
                   {"length": 8192, "kv_heads": 8, "query_heads": 32, "head_dim": 128, "element_B": 2},
                   ["One layer's K and V; 128 MiB occurs only if all heads are physically materialized.",
                    "A mathematical repeat expression is not evidence of actual backend allocation."])
    kv_bytes = 2 * 8192 * 8 * 128 * 2
    check(item, "kv_copy_and_gqa_sizes",
          {"logical_kv_MiB": kv_bytes / MIB, "gather_read_write_MiB": 2 * kv_bytes / MIB,
           "materialized_gqa_MiB": kv_bytes * (32 // 8) / MIB},
          {"logical_kv_MiB": 32, "gather_read_write_MiB": 64, "materialized_gqa_MiB": 128},
          "MiB", "2*L*Hkv*d*b; a layout copy reads and writes; physical GQA expansion multiplies by Hq/Hkv")

    item = example(groups, "operators", "partition_log_base", triton, "在线归约的底数、近似与输出精度",
                   {"partition_mass": [1, 4], "normalized_output": [0, 2]},
                   ["Disjoint partitions; masses and outputs are exact synthetic values.",
                    "Natural log and base-2 log statistics require explicit conversion."])
    masses, outputs = [1, 4], [0, 2]
    correct = sum(m*y for m, y in zip(masses, outputs)) / sum(masses)
    log2_mass = [math.log2(m) for m in masses]
    wrong_mass = [math.exp(x) for x in log2_mass]
    converted_mass = [math.exp(math.log(2) * x) for x in log2_mass]
    check(item, "partition_merge_and_wrong_base",
          {"correct": correct, "wrong_base": sum(m*y for m, y in zip(wrong_mass, outputs)) / sum(wrong_mass),
           "converted": sum(m*y for m, y in zip(converted_mass, outputs)) / sum(converted_mass)},
          {"correct": 1.6, "wrong_base": 1.7615941559557649, "converted": 1.6},
          "normalized output values", "mass-weighted output; exp(log2(mass)) is wrong, exp(ln(2)*log2(mass)) restores mass")

    item = example(groups, "operators", "plan_frequency", attn, "两个可复算的接入反例",
                   {"A_kernel_us": 20, "B_kernel_us": 12, "one_time_plan_us": 160,
                    "reuse_calls": 32, "extra_metadata_us": 6},
                   ["One legal plan serves 32 calls of compatible geometry.",
                    "23 us assumes 6 us per invocation. If 6 us occurs once per plan, the result is 17.1875 us.",
                    "The author corrected the prose from per-plan to per-invocation after review; arithmetic does not substitute for that wording correction."])
    check(item, "amortized_plan_and_frequency_counterexample",
          {"once_us": 160 + 12, "reused32_us_per_call": 160 / 32 + 12,
           "metadata_each_call_us": 160 / 32 + 12 + 6, "metadata_once_per_plan_us": (160 + 6) / 32 + 12},
          {"once_us": 172, "reused32_us_per_call": 17, "metadata_each_call_us": 23, "metadata_once_per_plan_us": 17.1875},
          "us/call", "plan/reuse + kernel + per-call work; once-per-plan work is divided by reuse")

    item = example(groups, "operators", "amdahl_serial_fraction", ops, "把局部内核收益连接到完整 forward",
                   {"fraction": 0.1, "local_speedup": 2}, ["No overlap; other work unchanged; no added adaptation cost."])
    new_cost = Fraction(9, 10) + Fraction(1, 10) / 2
    check(item, "serial_fraction_speedup", {"new_normalized_cost": float(new_cost), "speedup": float(1 / new_cost)},
          {"new_normalized_cost": 0.95, "speedup": 1.0526315789473684},
          "dimensionless", "(1-f)+f/s; speedup is its reciprocal")

    item = example(groups, "operators", "tuning_break_even", triton, "调优回报与引擎验收",
                   {"tuning_s": 1.2, "saved_us_per_call": 2, "calls_per_step": 32},
                   ["Every counted call is in the valid domain and has the same saving.",
                    "Maintenance, residency, misses and overlap are excluded."])
    cost_us = Fraction(6, 5) * 10**6
    calls = cost_us / 2
    steps = calls / 32
    check(item, "strict_tuning_threshold",
          {"cost_us": float(cost_us), "strict_threshold_calls": float(calls),
           "minimum_integer_calls": math.floor(calls) + 1, "strict_threshold_steps": float(steps),
           "minimum_integer_steps": math.floor(steps) + 1,
           "calls_at_minimum_steps": (math.floor(steps) + 1) * 32},
          {"cost_us": 1200000, "strict_threshold_calls": 600000, "minimum_integer_calls": 600001,
           "strict_threshold_steps": 18750, "minimum_integer_steps": 18751, "calls_at_minimum_steps": 600032},
          "us, calls, steps", "N*2 us > 1.2 s; strict inequality also applies after dividing by 32 calls/step")

    item = example(groups, "operators", "index_units", triton, "指针、对齐与整数索引也是内核输入",
                   {"element_offset": 2**31, "element_B": 2},
                   ["This is an integer representability boundary, not a universal library/device indexing limit."])
    check(item, "signed_index_and_byte_units",
          {"offset_bytes_GiB": 2**31 * 2 / GIB, "positive_offset_fits_signed_int32": 2**31 <= 2**31 - 1},
          {"offset_bytes_GiB": 4, "positive_offset_fits_signed_int32": False},
          "GiB and predicate", "element offset * element size gives byte position; max signed int32 is 2**31 - 1")


def identity_example_accepts(expected, observed, terminal_ids):
    """A tiny complete-population predicate for the A/B example, not an engine oracle."""
    population = Counter({key: 1 for key in expected})
    return (Counter(key for key, value in observed) == population
            and Counter(terminal_ids) == population
            and all(expected[key] == value for key, value in observed))


def decision_and_oracle_examples(groups):
    item = example(groups, "decisions", "layout_conversion_break_even",
                   "knowledge/system-semantics/design-synthesis.md", "布局改变后的完整成本与反转条件",
                   {"baseline_ms": 1.0, "kernel_ms": 0.6, "conversion_ms": 0.5},
                   ["Dependent non-overlapped conversion and computation.",
                    "Reuse count is integer; the same immutable converted data remains valid.",
                    "Additional capacity, invalidation, contention and synchronization costs are excluded."])
    old, new, conversion = Fraction(1), Fraction(3, 5), Fraction(1, 2)
    threshold = conversion / (old - new)
    check(item, "conversion_cost_and_strict_threshold",
          {"once_ms": float(conversion + new), "strict_threshold_reuses": float(threshold),
           "minimum_integer_reuses": math.floor(threshold) + 1,
           "at_two_old_ms": float(2 * old), "at_two_new_ms": float(conversion + 2 * new),
           "one_reuse_improves": conversion + new < old, "two_reuses_improve": conversion + 2 * new < 2 * old},
          {"once_ms": 1.1, "strict_threshold_reuses": 1.25, "minimum_integer_reuses": 2,
           "at_two_old_ms": 2, "at_two_new_ms": 1.7, "one_reuse_improves": False, "two_reuses_improve": True},
          "ms, counts, predicates", "0.5+0.6*N < 1.0*N iff N>1.25")

    expected = {"A": 17, "B": 99}
    item = example(groups, "oracles", "identity_population_counterexamples",
                   "knowledge/tools/correctness-oracles.md", "验证器也要能拒绝已知错误",
                   {"expected": expected},
                   ["This example requires exactly one complete result and terminal event per expected identity.",
                    "A different task's cancellation/partial-output rules require a different expected population.",
                    "List multiplicities are retained; converting observed results to a dict would hide duplicates."])
    cases = {
        "complete": ([("A", 17), ("B", 99)], ["A", "B"]),
        "wrong_value": ([("A", 17), ("B", 17)], ["A", "B"]),
        "missing_result": ([("A", 17)], ["A"]),
        "duplicate_result": ([("A", 17), ("B", 99), ("B", 99)], ["A", "B"]),
        "missing_terminal": ([("A", 17), ("B", 99)], ["A"]),
        "duplicate_terminal": ([("A", 17), ("B", 99)], ["A", "B", "B"]),
        "extra_result": ([("A", 17), ("B", 99), ("C", 0)], ["A", "B", "C"]),
        "empty": ([], []),
    }
    item["inputs"]["cases"] = cases
    check(item, "complete_identity_and_value_predicate",
          {name: identity_example_accepts(expected, obs, term) for name, (obs, term) in cases.items()},
          {name: name == "complete" for name in cases},
          "acceptance predicates", "expected multiplicities = observed multiplicities = terminal multiplicities, and values match")
    check(item, "subset_and_empty_vacuous_truth",
          {"missing_subset_naively_passes": all(expected[k] == v for k, v in cases["missing_result"][0]),
           "empty_naively_passes": all(expected[k] == v for k, v in cases["empty"][0])},
          {"missing_subset_naively_passes": True, "empty_naively_passes": True},
          "predicates", "all() on retained correct values or an empty set does not establish population completeness")

    item = example(groups, "oracles", "experiment_count_cost_counterexample",
                   "knowledge/tools/statistical-experiments.md", "针对知识库作用安排受控构建实验",
                   {"old_count": 10, "old_cost_each": 1, "new_count": 5, "new_cost_each": 3},
                   ["Arbitrary consistent cost units; experiment count is a proxy, not measured total cost."])
    check(item, "fewer_experiments_can_cost_more",
          {"old_total": 10 * 1, "new_total": 5 * 3, "fewer_experiments": 5 < 10, "lower_total_cost": 5 * 3 < 10 * 1},
          {"old_total": 10, "new_total": 15, "fewer_experiments": True, "lower_total_cost": False},
          "cost units and predicates", "total cost = count * cost each; 5<10 but 15>10")

    item = example(groups, "statistics", "previously_reviewed_interval_arithmetic",
                   "knowledge/tools/statistical-experiments.md", "配对、区组与区间",
                   {"n": 16, "mean_difference_ms": 2, "sample_sd_ms": 1, "given_t_critical": 2.131,
                    "zero_exceedance_n": 100, "one_sided_alpha": 0.05},
                   ["The t critical value is supplied, not estimated by this script.",
                    "Arithmetic does not establish independent sampling, t-model assumptions, or statistical power.",
                    "The zero-exceedance bound assumes independent Bernoulli observations with a common exceedance probability."])
    item["additional_sections"] = ["尾延迟与失败不能被均值掩盖"]
    half_width = 2.131 * 1 / math.sqrt(16)
    check(item, "paired_t_interval_using_stated_critical", [2 - half_width, 2 + half_width],
          [1.46725, 2.53275], "ms", "mean +/- t_critical * sample_sd / sqrt(n)")
    check(item, "zero_exceedance_upper_bound", 1 - 0.05**(1 / 100), 0.029513049607039932,
          "probability", "solve (1-q)^100=alpha for the one-sided upper bound")


def run_examples():
    groups = []
    communication_examples(groups)
    systems_examples(groups)
    operator_examples(groups)
    decision_and_oracle_examples(groups)
    documents = sorted({g["document"] for g in groups}
                       | {p for g in groups for p in g.get("additional_documents", [])})
    identities = []
    for relative in documents:
        path = ROOT / relative
        raw = path.read_bytes() if path.is_file() else None
        identities.append({"path": relative, "exists": raw is not None,
                           "sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None})
    results = [c for g in groups for c in g["checks"]]
    counts = Counter(g["family"] for g in groups for _ in g["checks"])
    return {
        "schema_version": 1,
        "evidence_class": "synthetic_arithmetic_and_counterexamples",
        "scope": "Recalculate explicit example fixtures and units. No GPU, NCCL, engine, model, Agent, or service experiment was executed.",
        "source_cutoff": "2026-04-30",
        "units": {"KiB": KIB, "MiB": MIB, "GiB": GIB, "byte_unit": "B", "seconds_to_ms": 1000,
                  "seconds_to_us": 1000000, "rates": "Numerator and denominator boundaries are specified per example."},
        "comparison": {"integer_and_discrete": "exact", "real_rel_tol": 1e-12, "real_abs_tol": 1e-12},
        "script": {"path": "tools/check_knowledge_examples.py", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "documents": identities,
        "examples": groups,
        "summary": {"example_groups": len(groups), "checks": len(results), "passed_checks": sum(c["passed"] for c in results),
                    "failed_checks": sum(not c["passed"] for c in results), "checks_by_family": dict(sorted(counts.items()))},
        "limits": ["All timings, capacities and workloads are synthetic declared inputs, not measurements.",
                   "Passing arithmetic or synthetic predicates does not establish freedom of exploration, appropriate topic ownership or coverage of unfamiliar architectures.",
                   "Chapter hashes bind the accompanying text; fixtures are explicit and are not automatically extracted from prose.",
                   "A later chapter edit needs semantic review and, where its example changed, a fixture update.",
                   "These calculations do not establish backend legality, concurrency correctness, model quality, engine completion, knowledge usefulness or scientific novelty."],
        "passed": all(c["passed"] for c in results) and all(d["exists"] for d in identities),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Write the full JSON report here; otherwise print it to stdout.")
    args = parser.parse_args()
    result = run_examples()
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
        print(json.dumps({"report": str(args.report), "passed": result["passed"],
                          "evidence_class": result["evidence_class"], **result["summary"]}, ensure_ascii=False))
    else:
        print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
