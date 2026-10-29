"""Apply eight evidence-specific closeout actions without changing prior artifacts."""
from pathlib import Path
import hashlib
import json
import sys
import time

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[2]
WORKSPACE=HERE.parents[3]
sys.path.insert(0,str(PACKAGE/'src'))
from knowledge_workflow import build
from knowledge_workflow.cli import current_run, repository_writer
from knowledge_workflow.config import load_config

config=load_config(PACKAGE/'workflow.local.toml')
kb=config.repository
read=build.read
notes={name:read(HERE/name) for name in ['root-notes.json','protocol-notes.json','binding-notes.json','work010-notes.json']}

def check_refs(value):
    count=0
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
            p=Path(value['path'])
            if not p.is_absolute():
                p=(kb if value['path'].startswith(('docs/','knowledge/','tools/')) else WORKSPACE)/p
            assert p.is_file(), p
            assert build.digest(p)==value['sha256'], p
            count+=1
        for v in value.values(): count+=check_refs(v)
    elif isinstance(value,list):
        for v in value:count+=check_refs(v)
    return count

bindings=sum(check_refs(v) for v in notes.values())
entries=[]
for row in notes['protocol-notes.json']['entries']+notes['root-notes.json']['entries']:
    r=read(kb/row['review']['path'])
    manager=row['manager_notes'] if isinstance(row['manager_notes'],list) else [row['manager_notes']]
    if row.get('required_regressions'):
        manager=manager+['本次需要真实触发的回归与负对照：'+json.dumps(row['required_regressions'],ensure_ascii=False)]
    entries.append({'id':row['id'],'action':'targeted_repair','candidate':row['candidate'],
                    'additional_rounds':1,'findings':r['findings'],'manager_notes':manager,
                    'reason':'已复核最后候选，按明确阻塞 '+','.join(f['id'] for f in r['findings'])+' 进行一次定点修订；保留已成立内容和此前已关闭问题。'})
event=notes['work010-notes.json']
candidate=dict(event['candidate'])
candidate['path']=(WORKSPACE/candidate['path']).relative_to(kb).as_posix()
event_findings=[]
for f in event['findings']:
    event_findings.append({'id':f['id'],'origin':'coordinator_diagnosis','criterion':'algorithm',
        'blocking':True,'evidence':f['minimal_counterexample']+'；'+f['observed'],
        'impact':f.get('mechanism',f.get('classification')),
        'required_revision':f.get('required_repair',f.get('preferred_repair'))})
patch=(HERE/'work010-diagnostic.patch').read_text(encoding='utf-8')
entries.append({'id':'K-WORK-010','action':'targeted_repair','candidate':candidate,
    'additional_rounds':1,'findings':event_findings,
    'manager_notes':event['manager_notes']+[
        '诊断环境实际已运行通过的差异如下，供正式修订核对；正文的同刻时序说明也必须同步修改。这不是新候选已经执行的声明。\n'+patch],
    'reason':'真实隔离复现并定位最终提交/发布资格错误与同刻取消调度合同失配；完整诊断修正版通过，按已验证反例追加一次定点修订并重新独审。'})
binding=notes['binding-notes.json']
action=binding['recommended_action']
entries.append({'id':'K-WORK-011','action':'fresh_review_only','candidate':binding['candidate'],
                'additional_rounds':0,'manager_notes':action['manager_notes'],'reason':action['reason']})
assert len(entries)==len({e['id'] for e in entries})==8
with repository_writer(config):
    run=current_run(config)
    assert run.name=='knowledge-workflow-20261028-batch21'
    state=read(run/'workflow.json')
    assert set(state['jobs'])=={e['id'] for e in entries}
    assert all(j['state']=='needs_attention' for j in state['jobs'].values())
    evidence=run/'closeout'
    assert not evidence.exists(), 'Closeout already enqueued; inspect retained state instead of repeating'
    evidence.mkdir()
    (evidence/'workflow-before.json').write_bytes((run/'workflow.json').read_bytes())
    for name in [*notes,'work010-diagnostic.patch','enqueue.py']:
        (evidence/name).write_bytes((HERE/name).read_bytes())
    plan={'prepared_by':'/root','created_at':build.now(),'artifact_hashes_checked':bindings,'entries':entries}
    build.write(evidence/'plan.json',plan)
    for e in entries:
        if e['action']=='fresh_review_only':
            build.schedule_recheck(kb,run,e['id'],e['candidate']['path'],e['reason'],e['candidate']['original_author'])
        else:
            build.reopen(run,e['id'],e['reason'],additional_rounds=1)
        time.sleep(.15)
        current=read(run/'workflow.json')
        job=current['jobs'][e['id']]
        job['manager_notes']=e['manager_notes']
        job['issues']=e.get('findings',[])
        job['closeout_plan']=(evidence/'plan.json').relative_to(kb).as_posix()
        job['repair_baseline']=e['candidate']
        build.write(run/'workflow.json',current)
        time.sleep(.15)
    (evidence/'workflow-after.json').write_bytes((run/'workflow.json').read_bytes())
    receipt={'run':run.relative_to(kb).as_posix(),'actions':{e['id']:e['action'] for e in entries},
             'hash_bindings_verified':bindings,'completed_entries_created':0,'models_invoked':0,
             'plan':(evidence/'plan.json').relative_to(kb).as_posix()}
    build.write(evidence/'receipt.json',receipt)
    build.write(HERE/'enqueue-receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
