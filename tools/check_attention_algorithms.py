#!/usr/bin/env python3
"""Independent scalar checks of the attention exemplar; no device benchmark."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]


def dense(q, k, v, visible, scale, bias):
    output = []
    for i, query in enumerate(q):
        scores = [(j, scale * sum(x*y for x, y in zip(query, key)) + bias[i][j])
                  for j, key in enumerate(k) if visible[i][j]]
        if not scores:
            output.append([0.0] * len(v[0]))
            continue
        largest = max(z for _, z in scores)
        weights = [(j, math.exp(z-largest)) for j, z in scores]
        mass = sum(w for _, w in weights)
        output.append([sum(w * v[j][d] for j, w in weights) / mass for d in range(len(v[0]))])
    return output


def tiled_state(query, k, v, allowed, scale, bias, block):
    m, mass, weighted = -math.inf, 0.0, [0.0]*len(v[0])
    for start in range(0, len(k), block):
        scores = [(j, scale * sum(x*y for x, y in zip(query, k[j])) + bias[j])
                  for j in range(start, min(start+block, len(k))) if allowed[j]]
        if not scores:
            continue
        new_m = max(m, max(z for _, z in scores))
        rescale = math.exp(m-new_m) if mass else 0.0
        weights = [(j, math.exp(z-new_m)) for j, z in scores]
        weighted = [rescale*u + sum(w*v[j][d] for j, w in weights)
                    for d, u in enumerate(weighted)]
        mass = rescale*mass + sum(w for _, w in weights)
        m = new_m
    return ([u/mass for u in weighted], m+math.log(mass)) if mass else (weighted, -math.inf)


def merge(states, width):
    states = [(o, log_mass) for o, log_mass in states if log_mass != -math.inf]
    if not states:
        return [0.0]*width
    m = max(log_mass for _, log_mass in states)
    mass = sum(math.exp(log_mass-m) for _, log_mass in states)
    return [sum(math.exp(log_mass-m)*o[d] for o, log_mass in states)/mass for d in range(width)]


def run():
    rng = random.Random(417)
    checks = []
    def record(name, actual, expected):
        error = max((abs(a-b) for row, reference in zip(actual, expected)
                     for a, b in zip(row, reference)), default=0)
        checks.append({'name': name, 'max_absolute_error': error, 'passed': error < 1e-11})

    for length, history, dk, dv in [(1,7,3,2),(2,5,2,4),(7,9,4,3),(3,3,1,1)]:
        q = [[rng.uniform(-5,5) for _ in range(dk)] for _ in range(length)]
        k = [[rng.uniform(-5,5) for _ in range(dk)] for _ in range(history)]
        v = [[rng.uniform(-3,3) for _ in range(dv)] for _ in range(history)]
        bias = [[rng.uniform(-1,1) for _ in range(history)] for _ in range(length)]
        masks = {'dense': [[True]*history for _ in q],
                 'tail_causal': [[j <= history-length+i for j in range(history)] for i in range(length)],
                 'empty_first': [[i != 0 and j%2 == 0 for j in range(history)] for i in range(length)]}
        for kind, mask in masks.items():
            reference = dense(q,k,v,mask,1/math.sqrt(dk),bias)
            for tile in [1,2,4,history+1]:
                output = [tiled_state(query,k,v,mask[i],1/math.sqrt(dk),bias[i],tile)[0]
                          for i,query in enumerate(q)]
                record(f'tiled/{length}/{history}/{kind}/{tile}',output,reference)
            for partitions in [1,2,history+2]:
                output = []
                for i, query in enumerate(q):
                    states = [tiled_state(query,k,v,[allowed and j%partitions==c for j,allowed in enumerate(mask[i])],
                                          1/math.sqrt(dk),bias[i],2) for c in range(partitions)]
                    output.append(merge(states,dv))
                record(f'partitioned/{length}/{history}/{kind}/{partitions}',output,reference)
    # Overflow-safe scores, an empty tile preceding a visible tile, and empty merge.
    record('extreme_scores',[tiled_state([1000],[[1],[2]],[[0],[2]],[True,True],1,[0,0],1)[0]],[[2]])
    record('skip_empty_first_tile',[tiled_state([1],[[1],[2]],[[7],[3]],[False,True],1,[0,0],1)[0]],[[3]])
    record('empty_partitions',[merge([([0],-math.inf)],1)],[[0]])
    documents = ['knowledge/frameworks/attention-backends.md','knowledge/sources/S-SDPA.md','knowledge/sources/S-FLASH.md']
    return {'result': 'PASS' if all(c['passed'] for c in checks) else 'FAIL',
            'scope': 'Scalar finite-input algorithm checks against an independent materialized reference, including masks, non-square inputs, unequal QK/VO dimensions, partial/empty tiles, partitions and extreme scores. No kernel, model, GPU or performance validation.',
            'checks':checks,'count':len(checks),
            'documents':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in documents}}


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path)
    args=parser.parse_args();result=run()
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','documents')}))
    raise SystemExit(result['result']!='PASS')
