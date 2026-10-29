"""Hand real sub-agent candidates to the existing isolated-check/fresh-review pipeline."""
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
from knowledge_workflow.campaign import validate_candidate

config=load_config(PACKAGE/'workflow.local.toml')
kb=config.repository
specs=[
 ('K-HW-003','hw003-final','/root/closeout_protocols',5,'K-HW-003-R5-POST-END-ACCEPTANCE-RESOLUTION'),
 ('K-WORK-010','work010-final','/root/closeout_event_failure',4,'K-WORK-010-FINAL-CANDIDATE-RECONSIDER'),
 ('K-WORK-011','cluster-final','/root/closeout_binding',3,'K-WORK-011-R3-ADMIT-COST-01'),
]

def resolved_reference(name):
    if name.startswith('workspace:'):
        path=(WORKSPACE/name.split(':',1)[1]).resolve()
        assert path.is_relative_to(WORKSPACE)
        return path
    p=Path(name)
    if p.is_absolute(): return p
    if name.startswith(('docs/','knowledge/','tools/')) or name=='AGENTS.md': return kb/p
    return WORKSPACE/p

with repository_writer(config):
    run=current_run(config)
    state=build.read(run/'workflow.json')
    assert not state.get('registered_version') and not state.get('successor_run')
    assert all(j['state'] in ('needs_attention','revise','queued') for j in state['jobs'].values())
    assert build.digest(kb/'VERSION.json')==state['baseline_sha256']
    staged=[]
    for key,folder,author,prior_round,finding in specs:
        job=state['jobs'][key]
        assert job['state']=='needs_attention' and job['round']==prior_round
        assert any(x.get('id')==finding for x in job['issues'])
        source=HERE/folder
        candidate=build.read(source/'candidate.json')
        validate_candidate(kb,job,candidate)
        ledger=build.read(source/'author-reading.json')
        assert ledger['author']==author and ledger['files']
        for item in ledger['files']:
            path=resolved_reference(item['path'])
            assert path.is_file() and build.digest(path)==item['sha256'], (key,item['path'])
            assert item['reading_scope'] in ('full','excerpt')
        original_review=kb/f'docs/evidence/knowledge-workflow-20261028-batch21/full_build/{key}/round-{prior_round}/review-submitted.json'
        assert build.read(original_review)['decision']=='revise'
        staged.append((key,source,author,prior_round,finding,candidate,ledger,original_review))
    evidence=run/'closeout-handoff'
    assert not evidence.exists(), 'Handoff already exists; inspect before any recovery'
    evidence.mkdir()
    (evidence/'workflow-before.json').write_bytes((run/'workflow.json').read_bytes())
    (evidence/'handoff.py').write_bytes(Path(__file__).read_bytes())
    receipt=[]
    for key,source,author,prior_round,finding,candidate,ledger,original_review in staged:
        reason=f'新独审明确指出{finding}；实际定点作者已修复最终候选，保留原失败记录，追加一轮用于该候选的真实检查与新独审。'
        build.reopen(run,key,reason,additional_rounds=1)
        time.sleep(.15)
        build.claim(run,author,entry_id=key)
        time.sleep(.15)
        job=build.read(run/'workflow.json')['jobs'][key]
        directory=run/'full_build'/key/f"round-{job['round']}"
        directory.mkdir(parents=True)
        (directory/'candidate.json').write_bytes((source/'candidate.json').read_bytes())
        (directory/'author-materials.json').write_bytes((source/'author-reading.json').read_bytes())
        proofs=[]
        for name in ['delivery.json','execution-report.json','execution-result-002.json']:
            path=source/name
            if path.is_file():
                destination=directory/('subagent-'+name)
                destination.write_bytes(path.read_bytes())
                proofs.append({'path':destination.relative_to(kb).as_posix(),'sha256':build.digest(destination)})
        origin={'author':author,'kind':'actual_collaboration_subagent',
                'source_candidate':str(source/'candidate.json'),'candidate_sha256':build.digest(directory/'candidate.json'),
                'chapter_sha256':hashlib.sha256(candidate['chapter'].encode()).hexdigest(),
                'script_sha256':hashlib.sha256(candidate['test_script'].encode()).hexdigest(),
                'reading_ledger':{'path':(directory/'author-materials.json').relative_to(kb).as_posix(),'sha256':build.digest(directory/'author-materials.json')},
                'prior_semantic_review':{'path':original_review.relative_to(kb).as_posix(),'sha256':build.digest(original_review)},
                'author_execution_evidence':proofs,
                'limits':'Actual sub-agent authoring; inherited backend identity not independently exposed. No native CLI author invocation is claimed. Candidate still requires pipeline checks, a fresh non-author review and release verification.'}
        build.write(directory/'subagent-author-provenance.json',origin)
        receipt.append({'entry_id':key,'round':job['round'],'author':author,'candidate_sha256':origin['candidate_sha256']})
    (evidence/'workflow-after.json').write_bytes((run/'workflow.json').read_bytes())
    result={'run':run.relative_to(kb).as_posix(),'entries':receipt,'semantic_acceptances_created':0,'at':build.now()}
    build.write(evidence/'receipt.json',result)
    build.write(HERE/'handoff-receipt.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
