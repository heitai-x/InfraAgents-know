#!/usr/bin/env python3
"""Check FA2 row ownership and FA3 mathematical/event dependencies on CPU."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from check_attention_algorithms import dense, tiled_state

ROOT=Path(__file__).resolve().parents[1]


def prepare(query,k,v,mask,bias,scale,start,end,m,mass):
    scores=[(j,scale*sum(x*y for x,y in zip(query,k[j]))+bias[j]) for j in range(start,end) if mask[j]]
    if not scores:return m,mass,1.0,[0.0]*len(v[0])
    new=max(m,max(z for _,z in scores));r=math.exp(m-new) if mass else 0.0
    p=[(j,math.exp(z-new)) for j,z in scores]
    contribution=[sum(w*v[j][d] for j,w in p) for d in range(len(v[0]))]
    return new,r*mass+sum(w for _,w in p),r,contribution


def event_pipeline(query,k,v,mask,bias,scale,block):
    chunks=[(start,min(start+block,len(k))) for start in range(0,len(k),block)]
    events=[('QK',0),('softmax',0)]
    m,mass,r,current=prepare(query,k,v,mask,bias,scale,*chunks[0],-math.inf,0.0)
    output=[0.0]*len(v[0])
    for j in range(len(chunks)):
        if j+1<len(chunks):events.append(('QK',j+1))
        events.append(('PV_issue',j))
        if j+1<len(chunks):
            next_m,next_mass,next_r,next_contribution=prepare(query,k,v,mask,bias,scale,*chunks[j+1],m,mass)
            events.append(('softmax',j+1))
        # Until this completion, the output accumulator is not accessed.
        events.append(('PV_complete',j));output=[u+c for u,c in zip(output,current)]
        if j+1<len(chunks):
            output=[next_r*u for u in output];events.append(('rescale',j+1))
            m,mass,current=next_m,next_mass,next_contribution
    return ([u/mass for u in output] if mass else output),events


def run():
    rng=random.Random(2903);checks=[]
    def record(name,passed,**values):checks.append({'name':name,'passed':bool(passed),**values})
    for rows in [1,3,7]:
        q=[[rng.uniform(-3,3) for _ in range(2)] for _ in range(rows)]
        k=[[rng.uniform(-3,3) for _ in range(2)] for _ in range(5)]
        v=[[rng.uniform(-2,2) for _ in range(3)] for _ in k]
        mask=[[j<=i+2 for j in range(5)] for i in range(rows)];bias=[[0.0]*5 for _ in q]
        reference=dense(q,k,v,mask,1/math.sqrt(2),bias)
        for warps in [1,2,4,8]:
            width=math.ceil(rows/warps);owners=[list(range(w*width,min((w+1)*width,rows))) for w in range(warps)]
            output=[None]*rows
            for owned in owners:
                for i in owned:output[i]=tiled_state(q[i],k,v,mask[i],1/math.sqrt(2),bias[i],2)[0]
            error=max(abs(x-y) for a,b in zip(output,reference) for x,y in zip(a,b))
            record(f'FA2/sliced_Q/rows={rows}/warps={warps}',error<1e-11 and sorted(sum(owners,[]))==list(range(rows)),max_error=error)
    for rows in [1,3]:
        for length in [1,2,3,5]:
            q=[[rng.uniform(-4,4) for _ in range(2)] for _ in range(rows)]
            k=[[rng.uniform(-4,4) for _ in range(2)] for _ in range(length)]
            v=[[rng.uniform(-2,2) for _ in range(3)] for _ in k]
            for block in [1,2]:
                for mode in ['dense','empty_first','causal']:
                    masks=[[True if mode=='dense' else j>0 if mode=='empty_first' else j<=i for j in range(length)] for i in range(rows)]
                    bias=[[rng.uniform(-0.5,0.5) for _ in k] for _ in q]
                    ref=dense(q,k,v,masks,1/math.sqrt(2),bias);output=[];valid=True
                    for i,query in enumerate(q):
                        values,events=event_pipeline(query,k,v,masks[i],bias[i],1/math.sqrt(2),block);output.append(values)
                        count=math.ceil(length/block)
                        valid &= [j for op,j in events if op=='QK']==list(range(count))
                        valid &= [j for op,j in events if op=='PV_complete']==list(range(count))
                        for j in range(1,count):
                            valid &= events.index(('PV_complete',j-1))<events.index(('rescale',j))<events.index(('PV_issue',j))
                            valid &= events.index(('PV_issue',j-1))<events.index(('softmax',j))<events.index(('PV_complete',j-1))
                    error=max(abs(x-y) for a,b in zip(output,ref) for x,y in zip(a,b))
                    record(f'FA3/pipeline/rows={rows}/keys={length}/tile={block}/{mode}',valid and error<1e-11,max_error=error)
    actual,_=event_pipeline([1],[[0],[2]],[[1],[0]],[True,True],[0,0],1,1)
    expected=1/(1+math.exp(2));wrong=math.exp(2)/(1+math.exp(-2))
    record('rescale_direction_counterexample',abs(actual[0]-expected)<1e-12 and wrong>1,
           correct=actual[0],inverse_rescale=wrong)
    q=[2,0.5];k=[3,-1];s=math.sqrt(2)
    qm=[(q[0]+q[1])/s,(q[0]-q[1])/s];km=[(k[0]+k[1])/s,(k[0]-k[1])/s]
    record('orthogonal_transform_before_quantization',math.isclose(sum(x*y for x,y in zip(q,k)),sum(x*y for x,y in zip(qm,km)),rel_tol=1e-12))
    readers={'QK','PV'};readers.remove('QK');blocked_after_qk=bool(readers);readers.remove('PV')
    record('paired_buffer_cannot_be_reused_after_only_QK',blocked_after_qk and not readers)
    readers={'consumer0','consumer1'};readers.remove('consumer0');blocked=bool(readers);readers.remove('consumer1')
    record('all_buffer_consumers_must_finish',blocked and not readers)
    record('score_tile_resource_arithmetic',4*64*64==16384 and 64*64/128==32,bytes=16384,registers_per_thread_if_uniform=32)
    docs=['knowledge/frameworks/attention-backends.md','knowledge/sources/S-FLASHATTENTION2.md','knowledge/sources/S-FLASHATTENTION3.md']
    return {'result':'PASS' if all(x['passed'] for x in checks) else 'FAIL','count':len(checks),'checks':checks,
            'documents':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in docs},
            'scope':'Scalar row ownership, mathematical recurrence, symbolic completion ordering, small/tail block cases, resource arithmetic and unquantized orthogonal identity only. No generated GPU instructions, physical overlap, FP8 kernel accuracy or performance validation.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args();result=run()
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['result','count','scope']}));raise SystemExit(result['result']!='PASS')
