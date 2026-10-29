#!/usr/bin/env python3
"""Independent arithmetic and identity examples from the construction pass."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def run():
    checks=[]
    def check(name,actual,expected):
        same=math.isclose(actual,expected,rel_tol=1e-12) if isinstance(actual,float) else actual==expected
        checks.append({'name':name,'actual':actual,'expected':expected,'passed':same})
    check('streaming_memory_lower_bound_ms',(8*2**30)/(0.8*2**40)*1000,9.765625)
    check('specialization_break_even_calls',40_000/(10-5),8000)
    check('specialization_short_total_ms',(40_000+1000*5)/1000,45)
    check('specialization_long_total_ms',(40_000+100_000*5)/1000,540)
    check('page_crossing_increment',math.ceil((31+2)/16)-2,1)
    check('lookahead_page_increment',math.ceil((31+2+16)/16)-2,2)
    check('full_block_hit_with_last_logits_recompute',math.floor((32-1)/16)*16,16)
    offsets=[0]
    for count in [2,5,3]:offsets.append(offsets[-1]+count)
    check('ragged_offsets',offsets,[0,2,7,10]);check('last_logits_rows',[x-1 for x in offsets[1:]],[1,6,9])
    check('unused_slots_fraction',(16-10)/16,0.375)
    check('ideal_overlap_cycle_ms',max(2,5),5);check('ideal_overlap_segment_speedup',(2+5)/max(2,5),1.4)
    docs=['knowledge/system-semantics/known-tensions.md','knowledge/system-semantics/specialization-levers.md',
          'knowledge/engine-index/vllm-kv-manager.md','knowledge/engine-index/vllm-model-runner.md','knowledge/engine-index/tensorrt-llm.md']
    return {'result':'PASS' if all(x['passed'] for x in checks) else 'FAIL','checks':checks,'count':len(checks),
            'scope':'Synthetic declared arithmetic, slot identities and ideal dependency models only; no hardware or service measurements.',
            'documents':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in docs}}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args();result=run()
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','documents')}))
    raise SystemExit(result['result']!='PASS')
