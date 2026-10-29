"""CPU checks for declared scalar mappings, finite protocols and synthetic costs.

This does not compile a GPU kernel or model device memory visibility.  Integer
inputs make the scalar products exact in Python's binary64 range; no FP32 MMA,
compiler, hardware overlap, or independent Agent benefit is claimed.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS: list[dict] = []


def check(name, condition, **evidence):
    if not condition:
        raise AssertionError(name)
    CHECKS.append({'name': name, 'status': 'PASS', **evidence})


def ownership():
    return [(thread, u, 2*u+(thread % 32)//16,
             16*(thread//32)+(thread % 16))
            for thread in range(64) for u in range(8)]


def bank_words(words):
    """Same-word readers broadcast; distinct words per bank imply conflicts."""
    return max((len({word for word in words if word % 32 == bank})
                for bank in range(32)), default=0)


def mapping_checks():
    assigned = ownership()
    coords = [(r, c) for _, _, r, c in assigned]
    check('output ownership is a 512-coordinate bijection',
          len(coords) == len(set(coords)) == 512
          and set(coords) == set(itertools.product(range(16), range(32))))
    for count, width, name in [(128, 8, 'A'), (256, 32, 'B')]:
        copies = [thread+64*v for thread in range(64)
                  for v in range(count//64)]
        check(f'{name} cooperative copy covers every shared word once',
              sorted(copies) == list(range(count)))
        check(f'{name} scalar producer stores have distinct banks',
              all(bank_words([thread+64*v for thread in range(32*w, 32*w+32)]) == 1
                  for w in range(2) for v in range(count//64)))
        check(f'{name} row-major shared coordinates are in range',
              all(0 <= a//width < count//width and 0 <= a % width < width
                  for a in copies))
    for warp in range(2):
        check(f'warp {warp} scalar shared reads use broadcasts without bank conflicts',
              all(bank_words([(2*u+lane//16)*8+k for lane in range(32)]) == 1
                  and bank_words([k*32+warp*16+lane % 16 for lane in range(32)]) == 1
                  for u in range(8) for k in range(8)))
    check('worked global byte addresses',
          (4*(18*17+11), 4*(11*35+34), 4*(18*35+34)) == (1268, 1676, 2656))
    check('odd strides do not imply 16-byte row alignment',
          4*17 % 16 != 0 and 4*35 % 16 != 0)


def dense(a, b, m, n, k):
    return [[math.fsum(a[r*k+t]*b[t*n+c] for t in range(k))
             for c in range(n)] for r in range(m)]


def mapped(a, b, m, n, k, *, omit_last=False, wrong_b_stride=False):
    result = [[None for _ in range(n)] for _ in range(m)]
    reads_a, reads_b, stores = [], [], []
    for m0 in range(0, m, 16):
        for n0 in range(0, n, 32):
            accum = {(thread, u): 0.0 for thread, u, _, _ in ownership()}
            stages = (k+7)//8
            for j in range(max(0, stages-int(omit_last))):
                sa, sb = [0.0]*128, [0.0]*256
                for thread in range(64):
                    for v in range(2):
                        word = thread+64*v
                        r, t = divmod(word, 8)
                        if m0+r < m and 8*j+t < k:
                            address = (m0+r)*k+8*j+t
                            reads_a.append(address)
                            sa[word] = a[address]
                    for v in range(4):
                        word = thread+64*v
                        t, c = divmod(word, 32)
                        if 8*j+t < k and n0+c < n:
                            address = (8*j+t)*(k if wrong_b_stride else n)+n0+c
                            if not 0 <= address < len(b):
                                raise IndexError('incorrect B stride read')
                            reads_b.append(address)
                            sb[word] = b[address]
                for thread, u, r, c in ownership():
                    for t in range(8):
                        accum[thread, u] += sa[r*8+t]*sb[t*32+c]
            for thread, u, r, c in ownership():
                if m0+r < m and n0+c < n:
                    result[m0+r][n0+c] = accum[thread, u]
                    stores.append((m0+r)*n+n0+c)
    return result, reads_a, reads_b, stores


def numerical_checks():
    shapes = [(0, 7, 3), (5, 0, 3), (1, 1, 0), (1, 1, 1),
              (16, 32, 7), (16, 32, 8), (16, 32, 9), (19, 35, 17),
              (17, 3, 2), (3, 33, 16), (32, 64, 24)]
    for m, n, k in shapes:
        a = [float(((r*13+t*7) % 19)-9) for r in range(m) for t in range(k)]
        b = [float(((t*11+c*3) % 17)-8) for t in range(k) for c in range(n)]
        got, ra, rb, stores = mapped(a, b, m, n, k)
        reference = dense(a, b, m, n, k)
        check(f'mapped scalar GEMM {m}x{n}x{k}', got == reference,
              outputs=m*n)
        check(f'valid addresses and unique output {m}x{n}x{k}',
              all(0 <= x < m*k for x in ra)
              and all(0 <= x < k*n for x in rb)
              and sorted(stores) == list(range(m*n)))
        gm, gn = (m+15)//16, (n+31)//32
        check(f'logical request count {m}x{n}x{k}',
              len(ra) == m*k*gn and len(rb) == k*n*gm)
    m, n, k = 19, 35, 17
    a = [float(1+(i % 7)) for i in range(m*k)]
    b = [float(1+(i % 11)) for i in range(k*n)]
    expected = dense(a, b, m, n, k)
    check('negative: omitting final K stage changes valid outputs',
          mapped(a, b, m, n, k, omit_last=True)[0] != expected)
    check('negative: using K instead of N as B row stride changes function',
          mapped(a, b, m, n, k, wrong_b_stride=True)[0] != expected)


@dataclass(frozen=True)
class Slot:
    generation: int = -1
    stage: int = -1
    copied: int = 0  # A/zero-fill=1, B/zero-fill=2
    visible: bool = False
    finished: int = 0  # warp0=1, warp1=2


@dataclass(frozen=True)
class State:
    issued: int
    next0: int
    next1: int
    slots: tuple[Slot, ...]


def updated(state, index, slot, **fields):
    slots = list(state.slots)
    slots[index] = slot
    return replace(state, slots=tuple(slots), **fields)


def validate(state, total):
    assert 0 <= state.next0 <= state.issued <= total
    assert 0 <= state.next1 <= state.issued <= total
    active = [x.stage for x in state.slots if x.stage >= 0]
    assert len(set(active)) == len(active)
    for index, slot in enumerate(state.slots):
        if slot.stage < 0:
            assert slot.copied == slot.finished == 0 and not slot.visible
            continue
        assert slot.stage % len(state.slots) == index
        assert slot.generation == slot.stage//len(state.slots)
        assert 0 <= slot.copied <= 3 and 0 <= slot.finished <= 3
        assert not slot.visible or slot.copied == 3
        assert slot.finished == 0 or slot.visible
        for warp, nxt in enumerate((state.next0, state.next1)):
            assert bool(slot.finished & (1 << warp)) == (nxt > slot.stage)


def event(state, name, index, generation, *, total, warp=None, part=None):
    """Transition implementation is checked against independent occupancy rules."""
    slot = state.slots[index]
    if name == 'issue':
        j = state.issued
        assert j < total and index == j % len(state.slots)
        assert generation == j//len(state.slots) and slot.stage < 0
        return updated(state, index, Slot(generation, j), issued=j+1)
    assert slot.stage >= 0 and generation == slot.generation
    if name == 'copy':
        assert part in (1, 2) and not slot.copied & part and not slot.visible
        return updated(state, index, replace(slot, copied=slot.copied | part))
    if name == 'publish':
        assert slot.copied == 3 and not slot.visible
        return updated(state, index, replace(slot, visible=True))
    if name == 'consume':
        assert warp in (0, 1) and slot.visible and not slot.finished & (1 << warp)
        assert (state.next0, state.next1)[warp] == slot.stage
        field = 'next0' if warp == 0 else 'next1'
        return updated(state, index, replace(slot, finished=slot.finished | (1 << warp)),
                       **{field: slot.stage+1})
    if name == 'release':
        assert slot.copied == 3 and slot.visible and slot.finished == 3
        return updated(state, index, Slot(generation=generation))
    raise AssertionError(name)


def successors(state, total):
    d = len(state.slots)
    if state.issued < total:
        j = state.issued
        if state.slots[j % d].stage < 0:
            yield event(state, 'issue', j % d, j//d, total=total)
    for index, slot in enumerate(state.slots):
        if slot.stage < 0:
            continue
        for part in (1, 2):
            if not slot.copied & part:
                yield event(state, 'copy', index, slot.generation, total=total, part=part)
        if slot.copied == 3 and not slot.visible:
            yield event(state, 'publish', index, slot.generation, total=total)
        if slot.visible:
            for warp, nxt in enumerate((state.next0, state.next1)):
                if nxt == slot.stage and not slot.finished & (1 << warp):
                    yield event(state, 'consume', index, slot.generation,
                                total=total, warp=warp)
        if slot.finished == 3:
            yield event(state, 'release', index, slot.generation, total=total)


def rejected(operation):
    try:
        operation()
    except AssertionError:
        return True
    return False


def protocol_checks():
    explored = []
    for total, d in [(0, 1), (1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (4, 2), (3, 3)]:
        start = State(0, 0, 0, tuple(Slot() for _ in range(d)))
        queue, seen, terminals = deque([start]), {start}, []
        while queue:
            state = queue.popleft()
            validate(state, total)
            children = list(successors(state, total))
            if not children:
                terminals.append(state)
                assert state.issued == state.next0 == state.next1 == total
                assert all(slot.stage < 0 for slot in state.slots)
            for child in children:
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
        check(f'all finite interleavings J={total}, slots={d}',
              bool(terminals), states=len(seen), terminal_states=len(terminals))
        explored.append({'stages': total, 'slots': d, 'states': len(seen)})
    start = State(0, 0, 0, (Slot(),))
    issued = event(start, 'issue', 0, 0, total=2)
    only_a = event(issued, 'copy', 0, 0, total=2, part=1)
    copied = event(only_a, 'copy', 0, 0, total=2, part=2)
    visible = event(copied, 'publish', 0, 0, total=2)
    consumed0 = event(visible, 'consume', 0, 0, total=2, warp=0)
    consumed1 = event(consumed0, 'consume', 0, 0, total=2, warp=1)
    freed = event(consumed1, 'release', 0, 0, total=2)
    new = event(freed, 'issue', 0, 1, total=2)
    cases = [
        ('partial input publication', lambda: event(only_a, 'publish', 0, 0, total=2)),
        ('consume before scope visibility', lambda: event(copied, 'consume', 0, 0, total=2, warp=0)),
        ('release after one reader', lambda: event(consumed0, 'release', 0, 0, total=2)),
        ('overwrite busy slot', lambda: event(consumed0, 'issue', 0, 1, total=2)),
        ('stale copy completion', lambda: event(new, 'copy', 0, 0, total=2, part=1)),
        ('duplicate reader completion', lambda: event(consumed0, 'consume', 0, 0, total=2, warp=0)),
        ('stale release', lambda: event(new, 'release', 0, 0, total=2)),
    ]
    for name, operation in cases:
        check('negative: '+name, rejected(operation))
    return explored


def recurrence(total, d, latency, interval, compute):
    issues, ends = [], []
    for j in range(total):
        issue = 0 if not j else issues[-1]+interval
        if j >= d:
            issue = max(issue, ends[j-d])
        end = max(issue+latency, ends[-1] if ends else 0)+compute
        issues.append(issue)
        ends.append(end)
    return issues, ends


def clock_simulation(total, d, latency, interval, compute):
    """Independent unit-clock as-if copy/compute engines with explicit slot states."""
    now, issued, next_compute, previous_issue = 0, 0, 0, -interval
    slots = [None]*d
    running = None
    ends, issues = [], []
    while next_compute < total or running is not None:
        if running is not None and running[1] == now:
            j = running[0]
            slots[j % d] = None
            ends.append(now)
            running = None
        if issued < total and now >= previous_issue+interval and slots[issued % d] is None:
            slots[issued % d] = (issued, now+latency)
            issues.append(now)
            previous_issue = now
            issued += 1
        if running is None and next_compute < total:
            slot = slots[next_compute % d]
            if slot is not None and slot[0] == next_compute and slot[1] <= now:
                running = (next_compute, now+compute)
                next_compute += 1
        now += 1
        assert now < 100000
    return issues, ends


def cost_checks():
    check('full tile byte and operation counts',
          2*16*32*8 == 8192 and 4*(16*8+8*32) == 1536
          and 64*8*4 == 2048 and 2*4*16*32*8 == 32768)
    combinations = 0
    for total, d, latency, interval, compute in itertools.product(
            (0, 1, 2, 3, 8, 17), (1, 2, 3, 4), (1, 6), (1, 2), (1, 3)):
        if interval > latency:
            continue
        combinations += 1
        expected = recurrence(total, d, latency, interval, compute)
        actual = clock_simulation(total, d, latency, interval, compute)
        if actual != expected:
            raise AssertionError((total, d, latency, interval, compute, actual, expected))
    check('delay recurrence matches independent unit-clock engine model', True,
          parameter_combinations=combinations)
    times = {d: recurrence(8, d, 6, 2, 3)[1] for d in (1, 2, 3, 4)}
    check('worked d1/d2/d3/d4 finish sequences',
          times[1] == list(range(9, 73, 9))
          and times[2] == [9, 12, 18, 21, 27, 30, 36, 39]
          and times[3] == times[4] == list(range(9, 31, 3)))
    check('finite warm-up: one and two stages',
          all(recurrence(1, d, 6, 2, 3)[1] == [9] for d in (1, 2, 3, 4))
          and recurrence(2, 2, 6, 2, 3)[1] == recurrence(2, 3, 6, 2, 3)[1] == [9, 12])
    bytes2, bytes3 = 2*1536, 3*1536
    resident2, resident3 = (6*1024)//bytes2, (6*1024)//bytes3
    grid2, grid3 = math.ceil(4/resident2)*times[2][-1], math.ceil(4/resident3)*times[3][-1]
    check('single CTA vs grid ranking reversal',
          times[3][-1] < times[2][-1] and (resident2, resident3) == (2, 1)
          and (grid2, grid3) == (78, 120))
    check('capacity change reverses preferred stage depth',
          math.ceil(4/((9*1024)//bytes3))*times[3][-1] == 60 < grid2)
    check('negative: assuming every 8.x has 1536 resident threads',
          {'8.0': 2048, '8.6': 1536, '8.7': 1536, '8.9': 1536, '9.0': 2048}['8.0'] != 1536)
    check('synthetic diagnosis fixed-cost crossover and contradicted slope',
          (10+2*8) < (18+1.7*8)
          and (10+2*64) > (18+1.7*64)
          and (18+2.4*64) > (10+2*64)
          and math.isclose(8/.3, 26.666666666666668))
    return {'unit': 'synthetic tau; not device measurement', 'single_cta_end_times': times,
            'six_kib_grid': {'two_stages': grid2, 'three_stages': grid3},
            'nine_kib_three_stage_grid': 60}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    mapping_checks()
    numerical_checks()
    states = protocol_checks()
    costs = cost_checks()
    paths = ['knowledge/hardware/gpu-kernel-pipelines.md', 'knowledge/sources/S-CUDA-ASYNC.md']
    report = {'schema_version': 1, 'status': 'PASS', 'scope': 'Declared CPU scalar/finite-event/synthetic-cost models only',
              'checked_at': '2026-10-27', 'checks_passed': len(CHECKS), 'checks': CHECKS,
              'document_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_exploration': states, 'synthetic_costs': costs,
              'limits': ['Python binary64 integer-valued examples do not model hardware FP32 rounding or compiler code generation.',
                         'Abstract scope visibility is an explicit event, not a proof of CUDA memory-model correctness.',
                         'Finite interleavings cover the declared stage/slot counts, not arbitrary GPU programs.',
                         'The ideal delay recurrence assumes timely independent issue and calibrated resource service; the conservative pseudocode need not attain it.',
                         'No real GPU synchronization, traffic, overlap, performance, or controlled Agent construction/transfer effects were measured.']}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'checks_passed': len(CHECKS),
                      'protocol_states': sum(x['states'] for x in states), 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
