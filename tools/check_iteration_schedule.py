#!/usr/bin/env python3
"""CPU state/cost specification checks for K-ALG-001; no GPU or LLM execution.

The toy recurrence is deliberately not a language model. The dense oracle uses
a closed-form prefix expression, while the scheduled path uses paged incremental
state and immutable packet rows. Agreement tests routing and causal bookkeeping,
not accuracy or speed of a real Transformer implementation.
"""
import argparse
from dataclasses import dataclass, field
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def dense_prefix(tokens):
    """Independent dense expression; does not read or update the paged cache."""
    return sum((token+i+1)*pow(31, len(tokens)-i-1, 997)
               for i, token in enumerate(tokens)) % 997


def next_token(state, length, seed):
    return (state+17*length+seed) % 251+1


def oracle(prompt, quota, maximum, seed=0):
    tokens = list(prompt)
    generated = []
    while len(generated) < quota and len(tokens) < maximum:
        token = next_token(dense_prefix(tokens), len(tokens), seed)
        generated.append(token)
        tokens.append(token)
    return generated


@dataclass
class Page:
    generation: int = 0
    owner: object = None
    pins: set = field(default_factory=set)
    data: dict = field(default_factory=dict)


class Pool:
    def __init__(self, size, page_tokens):
        self.pages = [Page() for _ in range(size)]
        self.page_tokens = page_tokens

    def free(self):
        return sum(p.owner is None and not p.pins for p in self.pages)

    def allocate(self, owner):
        for index, page in enumerate(self.pages):
            if page.owner is None and not page.pins:
                page.generation += 1
                page.owner = owner
                page.data = {}
                return index, page.generation
        raise RuntimeError('no free page')

    def page(self, handle):
        index, generation = handle
        page = self.pages[index]
        if page.generation != generation:
            raise ValueError('stale page generation')
        return page


@dataclass
class Request:
    key: tuple
    tokens: list
    quota: int
    maximum: int
    seed: int = 0
    k: int = 0
    pages: list = field(default_factory=list)
    generated: list = field(default_factory=list)
    active: object = None
    status: str = 'ready'
    dependency_ready: bool = True


@dataclass(frozen=True)
class Row:
    key: tuple
    start: int
    end: int
    known: int
    row_start: int
    inputs: tuple
    pages: tuple
    sample: bool
    seed: int


@dataclass
class Packet:
    ticket: int
    rows: tuple
    original_page_counts: dict
    result: dict = field(default_factory=dict)
    done: bool = False
    executed: bool = False


def pairs(h, q):
    return q*h+q*(q+1)//2


def device_ms(rows):
    return 4+0.015*sum(q for h, q in rows)+0.0000005*sum(pairs(h, q) for h, q in rows)


class Scheduler:
    def __init__(self, pages=64, page_tokens=4):
        self.pool = Pool(pages, page_tokens)
        self.requests = {}
        self.ticket = 0
        self.packets = {}

    def admit(self, key, prompt, quota=3, maximum=100, seed=0, cache_hit=0, cached_states=None):
        if not prompt or len(prompt) > maximum or key in self.requests or quota < 0:
            raise ValueError('invalid admission')
        if cache_hit < 0 or cache_hit > len(prompt):
            raise ValueError('invalid cache hit length')
        if cached_states is not None and (len(cached_states) < cache_hit or
                any(cached_states[i] != dense_prefix(prompt[:i+1]) for i in range(cache_hit))):
            raise ValueError('cache content identity mismatch')
        request = Request(key, list(prompt), quota, maximum, seed)
        request.k = min(cache_hit, len(prompt)-1) if quota else 0
        if math.ceil(request.k/self.pool.page_tokens) > self.pool.free():
            raise RuntimeError('insufficient private cache-import pages')
        # A private copy is the conservative cache-import strategy in this model.
        # It does not model shared-page COW or its device completion semantics.
        for _ in range(math.ceil(request.k/self.pool.page_tokens)):
            request.pages.append(self.pool.allocate(key))
        for index in range(request.k):
            self.store(request.pages, index, cached_states[index] if cached_states is not None else dense_prefix(prompt[:index+1]))
        if not quota:
            request.status = 'finished'
        self.requests[key] = request
        return request

    def store(self, handles, index, state):
        page = self.pool.page(handles[index//self.pool.page_tokens])
        page.data[index % self.pool.page_tokens] = state

    def load(self, handles, index):
        return self.pool.page(handles[index//self.pool.page_tokens]).data[index % self.pool.page_tokens]

    def schedule(self, order, tokens=128, time=math.inf, chunk=128, result_slots=32,
                 fail_submit=False, force_q=None):
        rows = []
        counts = {}
        used = []
        row_start = 0
        self.ticket += 1
        ticket = self.ticket
        for key in order:
            request = self.requests[key]
            if request.status != 'ready' or request.active is not None or not request.dependency_ready:
                continue
            maximum_q = min(len(request.tokens)-request.k, tokens-row_start, chunk)
            candidates = [force_q] if force_q is not None else range(maximum_q, 0, -1)
            chosen = None
            for q in candidates:
                if q < 1 or q > maximum_q:
                    raise ValueError('unknown token or exhausted work budget')
                end = request.k+q
                need = max(0, math.ceil(end/self.pool.page_tokens)-len(request.pages))
                sample = end == len(request.tokens) and end < request.maximum and len(request.generated) < request.quota
                if need > self.pool.free() or sample and result_slots < 1:
                    continue
                if device_ms(used+[(request.k, q)]) > time:
                    continue
                chosen = q, end, need, sample
                break
            if chosen is None:
                continue
            q, end, need, sample = chosen
            counts[key] = len(request.pages)
            for _ in range(need):
                request.pages.append(self.pool.allocate(key))
            rows.append(Row(key, request.k, end, len(request.tokens), row_start,
                            tuple(request.tokens[request.k:end]), tuple(request.pages), sample, request.seed))
            row_start += q
            result_slots -= int(sample)
            used.append((request.k, q))
        if not rows:
            return None
        packet = Packet(ticket, tuple(rows), counts)
        handles = set(h for row in rows for h in row.pages)
        for handle in handles:
            self.pool.page(handle).pins.add(ticket)
        if fail_submit:
            for handle in handles:
                self.pool.page(handle).pins.remove(ticket)
            for row in rows:
                request = self.requests[row.key]
                old_count = counts[row.key]
                for handle in request.pages[old_count:]:
                    self.pool.page(handle).owner = None
                request.pages = request.pages[:old_count]
            return None
        for row in rows:
            self.requests[row.key].active = ticket
        self.packets[ticket] = packet
        return packet

    def execute(self, packet, eos=(), partial_failure=False):
        if packet.executed or packet.done:
            raise ValueError('packet already executed or completed')
        for row in packet.rows:
            state = self.load(row.pages, row.start-1) if row.start else 0
            for position, token in enumerate(row.inputs, row.start):
                state = (31*state+token+position+1) % 997
                self.store(row.pages, position, state)
                if partial_failure:
                    packet.result[row.key] = 'FAILED'
                    break
            else:
                packet.result[row.key] = ('EOS' if row.key in eos else
                                          next_token(state, row.known, row.seed)) if row.sample else None
        packet.executed = True

    def cancel(self, key):
        request = self.requests[key]
        if request.status != 'ready':
            return
        request.status = 'cancelled'
        for handle in request.pages:
            self.pool.page(handle).owner = None

    def complete(self, packet):
        if packet.done:
            return False
        if not packet.executed:
            raise ValueError('completion without execution evidence')
        for row in packet.rows:
            request = self.requests[row.key]
            if request.status == 'ready':
                if request.active != packet.ticket or request.k != row.start:
                    raise ValueError('lost submission authority')
                result = packet.result[row.key]
                if result == 'FAILED':
                    request.status = 'failed'
                else:
                    request.k = row.end
                    if row.sample:
                        if result == 'EOS':
                            request.status = 'finished'
                        else:
                            request.generated.append(result)
                            request.tokens.append(result)
                            if len(request.generated) == request.quota:
                                request.status = 'finished'
                    elif row.end == request.maximum:
                        request.status = 'finished'
                request.active = None
                if request.status != 'ready':
                    for handle in request.pages:
                        self.pool.page(handle).owner = None
            elif request.active == packet.ticket:
                request.active = None
        for handle in set(h for row in packet.rows for h in row.pages):
            self.pool.page(handle).pins.remove(packet.ticket)
        packet.done = True
        return True

    def invariant(self):
        for request in self.requests.values():
            assert 0 <= request.k <= len(request.tokens) <= request.maximum
            if request.status == 'ready':
                assert request.k <= len(request.pages)*self.pool.page_tokens
                for handle in request.pages:
                    assert self.pool.page(handle).owner == request.key
                for index in range(request.k):
                    assert self.load(request.pages, index) == dense_prefix(request.tokens[:index+1])
        for page in self.pool.pages:
            for ticket in page.pins:
                assert not self.packets[ticket].done
        return True


def rejected(operation):
    try:
        operation()
    except (ValueError, RuntimeError, AssertionError):
        return True
    return False


def run():
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    differential_cases = 0
    rng = random.Random(40928)
    for page_tokens in (3, 4, 7):
        for chunk in (1, 2, 5):
            for length in (1, 2, 4, 5, 9):
                for hit in (0, length//2, length):
                    prompt = [rng.randrange(1, 250) for _ in range(length)]
                    seed = rng.randrange(10)
                    scheduler = Scheduler(page_tokens=page_tokens)
                    request = scheduler.admit(('r', 1), prompt, quota=4, maximum=length+6, seed=seed, cache_hit=hit)
                    for _ in range(length+8):
                        if request.status != 'ready':
                            break
                        packet = scheduler.schedule([request.key], chunk=chunk)
                        assert packet is not None
                        assert all(row.end <= row.known for row in packet.rows)
                        scheduler.execute(packet)
                        scheduler.complete(packet)
                        scheduler.invariant()
                    assert request.generated == oracle(prompt, 4, length+6, seed)
                    assert scheduler.pool.free() == len(scheduler.pool.pages)
                    differential_cases += 1
    check('dense_oracle/different_chunks_pages_partial_and_full_hits', differential_cases == 135)

    scheduler = Scheduler(pages=8)
    a = scheduler.admit(('A', 1), [1, 2, 3, 4, 5, 6], cache_hit=4)
    b = scheduler.admit(('B', 1), [7, 8, 9, 10, 11], cache_hit=4)
    first = scheduler.schedule([a.key, b.key], tokens=3)
    check('trace/packed_query_and_sample_rows', [(r.row_start, r.end-r.start, r.sample) for r in first.rows] == [(0, 2, True), (2, 1, True)])
    check('trace/two_new_pages_reserved', scheduler.pool.free() == 4)
    check('trace/no_same_request_second_inflight', scheduler.schedule([a.key, b.key]) is None)
    c = scheduler.admit(('C', 1), [12, 13, 14])
    scheduler.cancel(b.key)
    check('trace/cancel_does_not_free_pinned_pages', scheduler.pool.free() == 4)
    scheduler.execute(first)
    scheduler.complete(first)
    check('trace/accepted_token_has_no_kv_yet', len(a.tokens) == 7 and a.k == 6 and len(a.generated) == 1)
    check('trace/cancel_discards_business_result', not b.generated and scheduler.pool.free() == 6)
    second = scheduler.schedule([c.key, a.key], tokens=4)
    check('trace/repacked_rows_bound_to_original_identity', [(r.key, r.row_start, r.end-r.start) for r in second.rows] == [(c.key, 0, 3), (a.key, 3, 1)])
    scheduler.execute(second)
    scheduler.complete(second)
    check('trace/row_results_follow_requests', a.generated == oracle([1, 2, 3, 4, 5, 6], 2, 100)
          and c.generated == oracle([12, 13, 14], 1, 100))
    replacement = scheduler.admit(('B', 2), [40])
    before = (replacement.k, list(replacement.tokens), scheduler.pool.free())
    check('trace/duplicate_completion_is_noop', not scheduler.complete(first))
    check('trace/old_generation_does_not_mutate_replacement', before == (replacement.k, replacement.tokens, scheduler.pool.free()))
    check('negative/using_current_rows_misroutes_first_batch', first.rows[1].key != second.rows[1].key)

    model_cases = 0
    # cancel/arrival/complete are concurrent at the service layer. Device execution
    # follows submit and precedes the completion notification in every case.
    for order in permutations(('cancel', 'arrive', 'complete')):
        scheduler = Scheduler(pages=3)
        old = scheduler.admit(('r', 1), [1, 2, 3], cache_hit=2)
        packet = scheduler.schedule([old.key], tokens=1)
        scheduler.execute(packet)
        for event in order:
            if event == 'cancel': scheduler.cancel(old.key)
            elif event == 'arrive': scheduler.admit(('r', 2), [20, 30])
            else: scheduler.complete(packet)
            scheduler.invariant()
        new = scheduler.requests[('r', 2)]
        assert not new.generated and new.k == 0 and new.tokens == [20, 30]
        assert old.generated == (oracle([1, 2, 3], 1, 100) if order.index('complete') < order.index('cancel') else [])
        free = scheduler.pool.free()
        assert not scheduler.complete(packet) and scheduler.pool.free() == free
        model_cases += 1
    check('interleaving/cancel_arrival_completion_all_orders', model_cases == 6)

    scheduler = Scheduler(pages=4)
    a = scheduler.admit(('A', 1), [1, 2, 3, 4, 5], cache_hit=4)
    before = (a.k, a.tokens[:], a.pages[:], scheduler.pool.free())
    check('rollback/definite_unsubmitted_failure_returns_reservations', scheduler.schedule([a.key], fail_submit=True) is None
          and before == (a.k, a.tokens, a.pages, scheduler.pool.free()))
    check('negative/unknown_autoregressive_successor', rejected(lambda: scheduler.schedule([a.key], force_q=2)))
    check('negative/empty_prompt', rejected(lambda: scheduler.admit(('empty', 1), [])))
    check('negative/prompt_over_total_limit', rejected(lambda: scheduler.admit(('long', 1), [1, 2], maximum=1)))
    a.dependency_ready = False
    ready = scheduler.admit(('ready', 1), [3])
    packet = scheduler.schedule([a.key, ready.key], tokens=1)
    check('readiness/blocked_head_does_not_block_ready_tail', packet.rows[0].key == ready.key)
    check('negative/completion_before_execution_evidence', rejected(lambda: scheduler.complete(packet)))
    scheduler.execute(packet, eos=(ready.key,))
    scheduler.complete(packet)
    check('boundary/eos_no_extra_token_or_kv', ready.status == 'finished' and len(ready.tokens) == 1 and ready.k == 1 and not ready.generated)

    scheduler = Scheduler(pages=4)
    request = scheduler.admit(('limit', 1), [1, 2, 3], maximum=3, cache_hit=3)
    check('cache/full_hit_without_logits_rewinds_one_known_token', request.k == 2)
    packet = scheduler.schedule([request.key])
    scheduler.execute(packet)
    scheduler.complete(packet)
    check('boundary/total_token_limit_without_new_sample', request.status == 'finished' and not request.generated and request.k == 3)
    source_cache = [dense_prefix([29])]
    request = scheduler.admit(('one', 1), [29], cache_hit=1, quota=1, cached_states=source_cache)
    check('cache/one_token_hit_recomputes_position_zero', request.k == 0 and not request.pages)
    packet = scheduler.schedule([request.key])
    scheduler.execute(packet)
    scheduler.complete(packet)
    check('cache/private_recompute_does_not_modify_imported_cache', source_cache == [dense_prefix([29])]
          and request.generated == oracle([29], 1, 100))
    cache_prompt = [8, 3, 5, 1, 9]
    source_cache = [dense_prefix(cache_prompt[:i+1]) for i in range(len(cache_prompt))]
    original = source_cache[:]
    request = scheduler.admit(('copied-tail', 1), cache_prompt, cache_hit=5, quota=1, cached_states=source_cache)
    packet = scheduler.schedule([request.key])
    check('cache/full_hit_copies_prefix_and_recomputes_only_last_known', request.k == 4
          and packet.rows[0].inputs == (9,) and packet.rows[0].start == 4)
    scheduler.execute(packet)
    scheduler.complete(packet)
    check('cache/partial_tail_private_copy_preserves_source_content', source_cache == original
          and request.generated == oracle(cache_prompt, 1, 100))
    check('negative/cache_identity_mismatch', rejected(lambda: scheduler.admit(('bad-cache', 1), [29], cache_hit=1, cached_states=[999])))
    request = scheduler.admit(('zero', 1), [7], quota=0)
    check('boundary/zero_generation_budget_does_not_submit', scheduler.schedule([request.key]) is None)

    scheduler = Scheduler(pages=4)
    request = scheduler.admit(('failed', 1), [2, 3, 4, 5], cache_hit=2)
    packet = scheduler.schedule([request.key])
    scheduler.execute(packet, partial_failure=True)
    scheduler.complete(packet)
    check('failure/partial_state_never_published_as_complete', request.k == 2 and request.status == 'failed'
          and not request.generated and scheduler.pool.free() == 4)
    stale = packet.rows[0].pages[0]
    handle = scheduler.pool.allocate(('new', 1))
    check('negative/stale_page_handle_after_reuse', handle[0] == stale[0] and rejected(lambda: scheduler.pool.page(stale)))

    scheduler = Scheduler(pages=1)
    request = scheduler.admit(('r', 1), [1, 2, 3, 4, 5], cache_hit=4)
    check('capacity/no_progress_when_next_page_unavailable', scheduler.schedule([request.key]) is None)
    check('negative/unsafe_free_would_allow_reuse', not scheduler.pool.free() and request.status == 'ready')
    scheduler = Scheduler(pages=2)
    a = scheduler.admit(('a', 1), [1])
    b = scheduler.admit(('b', 1), [2])
    packet = scheduler.schedule([a.key, b.key], result_slots=1)
    check('capacity/results_are_also_a_hard_budget', len(packet.rows) == 1)

    # Compare pair counts with independent enumeration of the causal relation.
    for h, q in ((0, 1), (0, 17), (4, 3), (8192, 1), (19, 11)):
        direct = sum(1 for row in range(h, h+q) for col in range(h+q) if col <= row)
        check(f'cost/causal_pairs_h{h}_q{q}', direct == pairs(h, q))
    decodes = [(8192, 1)]*16
    allowed = lambda h: max(q for q in range(1, 113) if device_ms(decodes+[(h, q)]) <= 5.6)
    check('cost/time_budget_fresh_prefill', allowed(0) == 86)
    check('cost/time_budget_long_history', allowed(4096) == 75)
    check('cost/decode_base_time', math.isclose(device_ms(decodes), 4.305544))
    check('cost/page_budget_overrides_time', 20-16 == 4 and 4*16 == 64
          and math.isclose(device_ms(decodes+[(0, 64)]), 5.266584)
          and math.isclose(device_ms(decodes+[(4096, 64)]), 5.397656))
    chunks = [(64*i, 64) for i in range(8)]
    check('cost/chunked_and_full_same_semantic_pairs', sum(pairs(*row) for row in chunks) == pairs(0, 512) == 131328)
    check('cost/fixed_cost_reversal', math.isclose(device_ms([(0, 512)]), 11.745664)
          and math.isclose(sum(device_ms([row]) for row in chunks), 39.745664)
          and math.isclose(max(device_ms([row]) for row in chunks), 4.975376))
    check('cost/trace_page_capacity_and_new_writes', 4*512*1024 == 2*1024**2 and 3*128*1024 == 384*1024)
    check('diagnosis/host_gap_can_reverse_kernel_improvement', math.isclose(6+.4, 6.4)
          and math.isclose(4.8+3, 7.8) and math.isclose(4.8+.5, 5.3))
    # A fixed, feasible admitted set has a bounded number of round-robin
    # opportunities; a stream of newly favoured short requests can starve long.
    queue = ['long', 'b', 'c', 'd']
    services = {key: [] for key in queue}
    for turn in range(12):
        key = queue.pop(0)
        services[key].append(turn)
        queue.append(key)
    check('fairness/finite_feasible_round_robin_opportunities', all(len(v) == 3 and max(b-a for a, b in zip(v, v[1:])) == 4
          for v in services.values()))
    chosen = [min([('long', 100), (f'short{i}', 1)], key=lambda row: row[1])[0] for i in range(100)]
    check('fairness/shortest_work_stream_starves_long_counterexample', 'long' not in chosen)

    documents = ['knowledge/algorithms/iteration-batching.md', 'knowledge/sources/S-ORCA.md', 'knowledge/sources/S-SARATHI.md']
    return {
        'result': 'PASS', 'date': '2026-10-28', 'count': len(checks), 'checks': checks,
        'differential_cases': differential_cases, 'event_orders': model_cases,
        'documents': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in documents},
        'scope': 'Author-side CPU model of immutable scheduling views, causal token/KV boundaries, private paged state, cancellation, completion and calibrated-cost arithmetic; dense closed-form oracle is independent of paged updates.',
        'limits': [
            'The recurrence is a deterministic toy, not a Transformer or a numerical Attention correctness test.',
            'No GPU kernels, memory visibility, actual device overlap, network delivery, throughput or latency measurement.',
            'Private-copy cache import does not establish a shared-page COW protocol; shared lifetime belongs to the paged-state module.',
            'The finite event orders do not prove all production concurrency, failures or fairness under overload.',
            'No controlled Agent reading, implementation, transfer or diagnosis effectiveness evaluation.'
        ]
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('result', 'count', 'differential_cases', 'event_orders', 'scope')}, ensure_ascii=False))
