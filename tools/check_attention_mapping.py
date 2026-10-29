#!/usr/bin/env python3
"""Independent address/numeric/state checks for the two Attention case studies.

This is a CPU specification model, not CUDA code, a Tensor Core emulator or an
Agent effectiveness experiment. No target GPU is required or claimed.
"""
import argparse
from dataclasses import dataclass, field
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import random
import struct

from check_attention_reasoning import quantize

ROOT = Path(__file__).resolve().parents[1]


def half(x):
    return struct.unpack('e', struct.pack('e', x))[0]


def product(a, b):
    # The scalar summation is deliberately NOT a model of a hardware MMA tree.
    return [[sum(a[i][k]*b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def mapped(q, k, v, visible, transpose_correct=True):
    """One-warp logical memory/ownership algorithm, including FP16 P conversion."""
    output = [[0.0]*16 for _ in q]
    for q0 in range(0, len(q), 16):
        shared_q = [q[q0+r][:] if q0+r < len(q) else [0.0]*16 for r in range(16)]
        maximum = [[-math.inf]*8 for _ in range(32)]
        mass = [[0.0]*8 for _ in range(32)]
        accum = [[0.0]*8 for _ in range(32)]
        for k0 in range(0, len(k), 16):
            shared_k = [k[k0+r][:] if k0+r < len(k) else [0.0]*16 for r in range(16)]
            shared_v = [v[k0+r][:] if k0+r < len(k) else [0.0]*16 for r in range(16)]
            # B[k,j] addresses K[j,k] under column-major interpretation.
            b = [[shared_k[j][d] if transpose_correct else shared_k[d][j]
                  for j in range(16)] for d in range(16)]
            z = product(shared_q, b)
            p = [[0.0]*16 for _ in range(16)]
            for t in range(8):
                for group in range(2):
                    row = 2*t+group
                    lanes = list(range(group*16, (group+1)*16))
                    scores = [z[row][lane % 16]/4 if q0+row < len(q) and k0+lane % 16 < len(k)
                              and visible[q0+row][k0+lane % 16] else -math.inf for lane in lanes]
                    # Explicit butterfly lane exchanges; dense reference below has none.
                    maxima = scores[:]
                    for distance in (8, 4, 2, 1):
                        maxima = [max(x, maxima[index ^ distance]) for index, x in enumerate(maxima)]
                    if maxima[0] == -math.inf:
                        continue
                    new = max(maximum[lanes[0]][t], maxima[0])
                    probs = [math.exp(s-new) if s != -math.inf else 0.0 for s in scores]
                    sums = probs[:]
                    for distance in (8, 4, 2, 1):
                        sums = [x+sums[index ^ distance] for index, x in enumerate(sums)]
                    for index, lane in enumerate(lanes):
                        ratio = math.exp(maximum[lane][t]-new) if mass[lane][t] else 0.0
                        accum[lane][t] *= ratio
                        mass[lane][t] = ratio*mass[lane][t]+sums[index]
                        maximum[lane][t] = new
                        p[row][lane % 16] = half(probs[index])
            value = product(p, shared_v)
            for lane in range(32):
                for t in range(8):
                    row, col = divmod(lane+32*t, 16)
                    accum[lane][t] += value[row][col]
        for lane in range(32):
            for t in range(8):
                row, col = divmod(lane+32*t, 16)
                if q0+row < len(q):
                    output[q0+row][col] = accum[lane][t]/mass[lane][t] if mass[lane][t] else 0.0
    return output


def dense_with_probability_rounding(q, k, v, visible):
    """Global dense scores + independently weighted per-block rounded P.

    For block j, use its prefix maximum to reproduce the declared probability
    conversion, then restore each contribution directly to the final maximum.
    There are no lane states, online accumulators or warp reductions here.
    """
    result = []
    for query, keep in zip(q, visible):
        scores = [math.fsum(a*b for a, b in zip(query, key))/4 if allowed else -math.inf
                  for key, allowed in zip(k, keep)]
        maximum = max(scores, default=-math.inf)
        if maximum == -math.inf:
            result.append([0.0]*16)
            continue
        denominator = math.fsum(math.exp(s-maximum) for s in scores if s != -math.inf)
        weights = []
        for j, score in enumerate(scores):
            prefix_maximum = max(scores[:min(((j//16)+1)*16, len(scores))])
            weights.append(half(math.exp(score-prefix_maximum))*math.exp(prefix_maximum-maximum)/denominator
                           if score != -math.inf else 0.0)
        result.append([math.fsum(w*row[c] for w, row in zip(weights, v)) for c in range(16)])
    return result


def close(a, b):
    return len(a) == len(b) and all(len(ra) == len(rb) and
               all(math.isclose(x, y, rel_tol=3e-11, abs_tol=3e-11) for x, y in zip(ra, rb))
               for ra, rb in zip(a, b))


def sectors(base):
    # Count request sectors per scalar load instruction, rather than unique
    # sectors of the combined four instructions or downstream DRAM traffic.
    return [len({(base+4*(lane+32*u)+byte)//32 for lane in range(32) for byte in range(4)})
            for u in range(4)]


class InvalidTransition(ValueError):
    pass


@dataclass
class Resource:
    generation: int
    data: object
    owners: set = field(default_factory=set)
    pins: int = 0
    writers: int = 0
    ready: bool = False


class Pool:
    def __init__(self):
        self.objects = {}
        self.completed = set()
        self.serial = 0

    def allocate(self, identity, data, owners=(), ready=True):
        old = self.objects.get(identity)
        if old and not self.reusable(identity):
            raise InvalidTransition('Resource still has an owner, reader or writer')
        generation = old.generation+1 if old else 1
        self.objects[identity] = Resource(generation, data, set(owners), ready=ready,
                                          writers=0 if ready else 1)
        return identity, generation

    def resource(self, handle):
        identity, generation = handle
        obj = self.objects[identity]
        if obj.generation != generation:
            raise InvalidTransition('Stale resource generation')
        return obj

    def reusable(self, identity):
        obj = self.objects[identity]
        return not obj.owners and not obj.pins and not obj.writers

    def remove_owner(self, handle, owner):
        self.resource(handle).owners.remove(owner)

    def writer_done(self, handle):
        obj = self.resource(handle)
        if obj.writers != 1 or obj.ready:
            raise InvalidTransition('No matching writer')
        obj.writers = 0
        obj.ready = True

    def pin(self, handle):
        obj = self.resource(handle)
        if not obj.ready or obj.writers:
            raise InvalidTransition('Reader published before data/scale ready')
        self.serial += 1
        ticket = (self.serial, handle)
        obj.pins += 1
        return ticket

    def complete(self, ticket):
        if ticket in self.completed:
            raise InvalidTransition('Completion already consumed')
        obj = self.resource(ticket[1])
        if obj.pins <= 0:
            raise InvalidTransition('No outstanding reader')
        obj.pins -= 1
        self.completed.add(ticket)


@dataclass(frozen=True)
class Snapshot:
    request: tuple
    generation: int
    length: int
    pages: tuple
    tail: object


class Packet:
    def __init__(self, pool):
        self.pool, self.generation, self.busy, self.tickets = pool, 0, False, []

    def begin(self, snapshots, metadata_ready=True):
        if self.busy or not metadata_ready:
            raise InvalidTransition('Packet busy or metadata not ordered')
        handles = []
        for snapshot in snapshots:
            if snapshot.length > 16 or len(snapshot.pages)+(snapshot.tail is not None) > 4:
                raise InvalidTransition('Graph descriptor capacity exceeded')
            handles += list(snapshot.pages)+([snapshot.tail] if snapshot.tail else [])
        # Validate all publication dependencies before acquiring any reference.
        if any(not self.pool.resource(h).ready or self.pool.resource(h).writers for h in handles):
            raise InvalidTransition('Snapshot has unpublished resources')
        self.generation += 1
        self.busy = True
        self.tickets = [self.pool.pin(h) for h in handles]
        return self.generation

    def end(self, generation):
        if not self.busy or generation != self.generation:
            raise InvalidTransition('Stale/duplicate packet completion')
        for ticket in self.tickets:
            self.pool.complete(ticket)
        self.tickets = []
        self.busy = False


def seal(rows):
    # The two planes have separate immutable scales, as in the case study.
    keys, values = zip(*rows)
    kc, sk = quantize([x for row in keys for x in row])
    vc, sv = quantize([x for row in values for x in row])
    return (tuple(kc), sk, tuple(vc), sv, len(rows))


def decode(page):
    keys, sk, values, sv, count = page
    return [(tuple(half(sk*x) for x in keys[16*j:16*(j+1)]),
             tuple(half(sv*x) for x in values[16*j:16*(j+1)])) for j in range(count)]


def materialize(pool, snapshot):
    rows = []
    for handle in snapshot.pages:
        obj = pool.resource(handle)
        if not obj.ready:
            raise InvalidTransition('Unpublished page')
        rows += decode(obj.data)
    if snapshot.tail:
        rows += list(pool.resource(snapshot.tail).data)
    if len(rows) != snapshot.length:
        raise InvalidTransition('Length/content disagree')
    return rows


def case_fixture():
    rng = random.Random(7108)
    def row():
        return (tuple(half(rng.uniform(-5, 5)) for _ in range(16)),
                tuple(half(rng.uniform(-3, 3)) for _ in range(16)))
    pool = Pool()
    prefix = tuple(pool.allocate('prefix'+str(i), seal([row() for _ in range(4)]), {'A','B'}) for i in range(2))
    tail_a = pool.allocate('tailA0', tuple(row() for _ in range(3)), {'A'})
    tail_b = pool.allocate('tailB0', tuple(row() for _ in range(3)), {'B'})
    a = Snapshot(('A',1), 1, 11, prefix, tail_a)
    b = Snapshot(('B',1), 1, 11, prefix, tail_b)
    return pool, prefix, a, b, row()


def reject(action):
    try:
        action()
    except InvalidTransition:
        return True
    return False


def run():
    checks = []
    def check(name, condition):
        checks.append({'name':name, 'passed':bool(condition)})
        if not condition:
            raise AssertionError(name)

    ownership = [(lane+32*t, 2*t+lane//16, lane % 16) for lane in range(32) for t in range(8)]
    check('mapping/bijection', sorted(i for i, _, _ in ownership) == list(range(256)))
    check('mapping/row_column_identity', all(i == row*16+col for i, row, col in ownership))
    check('mapping/subgroups_do_not_cross_rows', all(len({2*t+lane//16 for lane in range(g*16,(g+1)*16)}) == 1
                                                   for t in range(8) for g in range(2)))
    words = [lane+32*u for u in range(4) for lane in range(32)]
    check('mapping/packed_copy_coverage', sorted(words) == list(range(128)))
    check('mapping/coalesced_sectors', sectors(0) == [4]*4)
    check('mapping/offset_sectors_are_per_instruction', sectors(4) == [5]*4)
    check('mapping/scalar_copy_banks', all(len({(lane+32*u) % 32 for lane in range(32)}) == 32 for u in range(4)))
    check('mapping/striped_scalar_banks', all(len({(lane+32*t) % 32 for lane in range(32)}) == 32 for t in range(8)))
    check('mapping/alternative_has_eight_way_conflicts', all(len({(8*lane+t) % 32 for lane in range(32)}) == 4 for t in range(8)))
    check('mapping/packed_P_writers', all(len({((lane+32*t)//2) % 32 for lane in range(0,32,2)}) == 16 for t in range(8)))
    packed = [None]*128
    for t in range(8):
        values = [half((lane+32*t)/257) for lane in range(32)]
        for lane in range(0,32,2):
            packed[(lane+32*t)//2] = struct.pack('ee',values[lane],values[lane ^ 1])
    unpacked = [value for word in packed for value in struct.unpack('ee',word)]
    check('mapping/paired_probability_store_preserves_order', unpacked == [half(i/257) for i in range(256)])
    check('mapping/odd_tail_protects_both_half_elements',
          [i for word in range(8) for i in (2*word,2*word+1) if i < 15] == list(range(15)))
    check('mapping/aligned_shared_objects', all(x % 32 == 0 for x in [0,512,1024,1536,2048,3072]))
    check('mapping/shared_capacity_and_state', 4*16*16*2+2*16*16*4 == 4096 and 32*24*4 == 3072)
    check('mapping/work_and_logical_shared_bytes', 2*2*16**3 == 16384 and 7*1024+512 == 7680)

    rng = random.Random(8107)
    for length in (1,7,16,19):
        for history in (0,1,15,16,17,33):
            q = [[half(rng.uniform(-2,2)) for _ in range(16)] for _ in range(length)]
            k = [[half(rng.uniform(-2,2)) for _ in range(16)] for _ in range(history)]
            v = [[half(rng.uniform(-2,2)) for _ in range(16)] for _ in range(history)]
            for mask in ('dense','empty','causal_tail','first_block_empty'):
                keep = [[mask=='dense' or (mask=='causal_tail' and j <= history-length+i)
                         or (mask=='first_block_empty' and j >= 16) for j in range(history)] for i in range(length)]
                check(f'numeric/L={length}/S={history}/{mask}', close(mapped(q,k,v,keep), dense_with_probability_rounding(q,k,v,keep)))
    keep = [[True]*len(k) for _ in q]
    check('negative/wrong_K_memory_interpretation', not close(mapped(q,k,v,keep,False), dense_with_probability_rounding(q,k,v,keep)))

    pool, prefix, a, b, new_token = case_fixture()
    old_b = tuple(materialize(pool,b))
    old_a = tuple(materialize(pool,a))
    check('negative/tail_from_another_request_changes_state', old_b[8:] != old_a[8:])
    first = Packet(pool); generation = first.begin([a,b])
    check('negative/reuse_live_packet', reject(lambda: first.begin([a,b])))
    check('negative/metadata_not_ordered', reject(lambda: Packet(pool).begin([b],False)))
    for handle in prefix+(a.tail,):
        pool.remove_owner(handle,'A')
    check('ownership/cancel_preserves_inflight_tail', not pool.reusable('tailA0'))
    check('ownership/shared_prefix_has_B_owner', all(pool.resource(h).owners == {'B'} for h in prefix))
    new_rows = tuple(pool.resource(b.tail).data)+(new_token,)
    new_page = pool.allocate('pageB2', seal(new_rows), {'B'}, ready=False)
    new_b = Snapshot(('B',1), 2, 12, prefix+(new_page,), None)
    check('negative/publish_unfinished_page', reject(lambda: Packet(pool).begin([new_b])))
    check('publication/rejection_acquires_no_partial_pins', all(pool.resource(h).pins == 2 for h in prefix)
          and pool.resource(new_page).pins == 0)
    pool.writer_done(new_page)
    pool.remove_owner(b.tail,'B')
    second = Packet(pool); second_generation = second.begin([new_b])
    check('snapshot/old_content_unchanged', tuple(materialize(pool,b)) == old_b)
    check('snapshot/new_has_complete_sealed_tail', len(materialize(pool,new_b)) == 12
          and materialize(pool,new_b)[8:] == decode(pool.resource(new_page).data))
    query = [[half((i-8)/5) for i in range(16)]]
    for label, snapshot in [('old',b),('new',new_b)]:
        rows = materialize(pool,snapshot); keys, values = zip(*rows)
        check('snapshot/'+label+'/attention_reference', close(mapped(query,keys,values,[[True]*len(rows)]),
                                                             dense_with_probability_rounding(query,keys,values,[[True]*len(rows)])))
    old_ticket = first.tickets[-1]
    second.end(second_generation)
    check('ownership/new_finishes_before_old', not pool.reusable('tailB0') and all(pool.resource(h).pins == 2 for h in prefix))
    first.end(generation)
    check('ownership/old_tails_retire_after_readers', pool.reusable('tailA0') and pool.reusable('tailB0'))
    check('negative/duplicate_reader_completion', reject(lambda: pool.complete(old_ticket)))
    replacement = pool.allocate('tailA0', (new_token,), {'C'})
    new_c = Snapshot(('C',2),1,1,(),replacement)
    fresh_generation = first.begin([new_c])
    check('negative/stale_packet_generation', reject(lambda: first.end(generation)) and pool.resource(replacement).pins == 1)
    check('negative/stale_resource_handle', reject(lambda: pool.pin(a.tail)))
    check('routing/old_identity_does_not_match_reused_row', a.request != new_c.request)
    first.end(fresh_generation)
    check('negative/descriptor_capacity', reject(lambda: Packet(pool).begin([Snapshot(('B',1),3,17,prefix,None)])))
    check('negative/length_without_content', reject(lambda: materialize(pool,Snapshot(('B',1),3,13,new_b.pages,None))))
    copied_page = list(pool.resource(new_page).data); copied_page[1] *= 2
    check('negative/scale_only_change_corrupts_old_values', decode(tuple(copied_page)) != decode(pool.resource(new_page).data))
    scratch = pool.allocate('writer_scratch', (), (), ready=False)
    check('ownership/unowned_writer_still_not_reusable', not pool.reusable(scratch[0]))
    pool.writer_done(scratch)
    check('ownership/completed_unowned_writer_reusable', pool.reusable(scratch[0]))

    aborted_pool, _, _, committed, next_row = case_fixture()
    committed_bytes = tuple(materialize(aborted_pool,committed))
    staging = aborted_pool.allocate('aborted_seal',seal(tuple(aborted_pool.resource(committed.tail).data)+(next_row,)),{'B'},ready=False)
    aborted_pool.remove_owner(staging,'B')
    check('publication/abort_still_waits_for_writer', not aborted_pool.reusable(staging[0]))
    aborted_pool.writer_done(staging)
    check('publication/abort_discards_only_staging', aborted_pool.reusable(staging[0])
          and tuple(materialize(aborted_pool,committed)) == committed_bytes
          and aborted_pool.resource(committed.tail).owners == {'B'})

    interleavings = 0
    for order in permutations(('cancel','build','publish','begin1','end0','end1')):
        position = {name:i for i,name in enumerate(order)}
        if not position['build'] < position['publish'] < position['begin1'] < position['end1']:
            continue
        pool, prefix, a, b, new_token = case_fixture()
        original = tuple(materialize(pool,b))
        zero, one = Packet(pool), Packet(pool)
        g0 = zero.begin([a,b]); page = newer = g1 = None
        for event in order:
            if event == 'cancel':
                for h in prefix+(a.tail,): pool.remove_owner(h,'A')
            elif event == 'build':
                page = pool.allocate('pageB2',seal(tuple(pool.resource(b.tail).data)+(new_token,)),{'B'},ready=False)
            elif event == 'publish':
                pool.writer_done(page); newer = Snapshot(('B',1),2,12,prefix+(page,),None)
                pool.remove_owner(b.tail,'B')
            elif event == 'begin1': g1 = one.begin([newer])
            elif event == 'end0': zero.end(g0)
            elif event == 'end1': one.end(g1)
            assert tuple(materialize(pool,b)) == original
            assert all(pool.resource(h).pins >= 0 and pool.resource(h).owners == ({'B'} if position['cancel'] <= position[event] else {'A','B'}) for h in prefix)
            assert not pool.reusable(a.tail[0]) or (not zero.busy and 'A' not in pool.resource(a.tail).owners)
        assert pool.reusable(a.tail[0]) and pool.reusable(b.tail[0])
        assert all(pool.resource(h).pins == 0 and pool.resource(h).owners == {'B'} for h in prefix)
        interleavings += 1
    check('composition/all_declared_interleavings', interleavings == 30)
    check('cost/page_payload_stride', 4*(16+16)+8 == 136 and math.ceil(136/32)*32 == 160)
    check('cost/prefix_sharing_capacity', 2*160 == 320 and 2*2*160 == 640)
    check('cost/tail_pool_and_live_overlap', 2*2*256 == 1024 and 256+256+160 == 672)
    check('cost/copy_and_seal_bytes', 3*(16+16)*2 == 192 and 4*(16+16)*2 == 256)
    documents = ['knowledge/frameworks/attention-backends.md', 'knowledge/sources/S-CUDA-CAPABILITY.md',
                 'knowledge/sources/S-CUDA-BEST.md']
    return {'result':'PASS','count':len(checks),'checks':checks,'composition_interleavings':interleavings,
            'documents':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in documents},
            'scope':'CPU address/ownership and scalar shared-memory model; global dense reference with explicit FP16 P conversion; snapshot/pool/packet model and negative cases. No hardware MMA rounding, compiled instructions, device visibility/barrier proof, GPU performance or controlled Agent effectiveness evaluation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--report',type=Path,required=True)
    args = parser.parse_args(); result = run()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['result','count','composition_interleavings','scope']},ensure_ascii=False))
