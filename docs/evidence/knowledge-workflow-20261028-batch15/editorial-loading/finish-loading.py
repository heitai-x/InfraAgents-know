from pathlib import Path
import json
import re
import sys
import time

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[2]
sys.path.insert(0,str(PACKAGE/'src'))
from knowledge_workflow import build
from knowledge_workflow.config import load_config
from knowledge_workflow.cli import current_run,repository_writer
from knowledge_workflow.campaign import next_version

config=load_config(PACKAGE/'workflow.local.toml')
root=config.repository
plan=build.read(HERE/'loading/plan.json')
review=build.read(HERE/'loading/review.json')
with repository_writer(config):
    assert (HERE/'completion.json').is_file(), 'Finish the first forty registered edits first'
    run=current_run(config)
    state=build.read(run/'workflow.json')
    assert not state.get('registered_version')
    assert all(j['state'] in ('queued','revise','needs_attention') for j in state['jobs'].values())
    assert build.digest(root/'VERSION.json')==state['baseline_sha256']
    assert plan['id'] not in state['jobs']
    assert build.digest(root/plan['path'])==plan['baseline_sha256']
    assert build.digest(HERE/'loading/checkpoint-loading.md')==plan['body_sha256']==review['body_sha256']
    assert review['decision']=='accept' and review['review_scope']=='editorial_only' and not review['checks']
    catalog={e['id']:e for e in build.read(root/'knowledge/catalog.json')['entries']}
    c=catalog[plan['id']]
    evidence=run/'editorial-loading'
    evidence.mkdir()
    for name in ['plan.json','review.json']:
        (evidence/name).write_bytes((HERE/'loading'/name).read_bytes())
    (evidence/'finish-loading.py').write_bytes(Path(__file__).read_bytes())
    state['jobs'][plan['id']]={'id':plan['id'],'path':plan['path'],
        'baseline_revision':c['revision'],'baseline_sha256':c['sha256'],
        'prior_review':'workflow_content_reviewed_with_limits','source_ids':c['source_ids'],
        'related':c.get('related',[]),'state':'queued','round':0,'author':None,
        'reviews':[],'checks':[],'agent_use':'unassessed','issues':[],
        'revision_scope':'editorial_only','prior_semantic_review':plan['prior_review'],
        'prior_algorithm_checks':plan['prior_checks']}
    build.write(run/'workflow.json',state)
    time.sleep(.2)
    build.claim(run,'/root',entry_id=plan['id'])
    (root/plan['path']).write_bytes((HERE/'loading/checkpoint-loading.md').read_bytes())
    time.sleep(.2)
    build.record_review(root,run,plan['id'],(evidence/'review.json').relative_to(root).as_posix())
    time.sleep(.2)
    build.execute_checks(root,run,plan['id'])
    result=build.register(root,run,next_version(build.read(root/'VERSION.json')['version_id']),config.runtime_source)
    build.write(HERE/'loading/publication.json',{'run':run.relative_to(root).as_posix(),**result})
    print(json.dumps(result,ensure_ascii=False),flush=True)
    verified=build.verify_release(root,run,config.runtime_source,verification_timeout=900)
    following=re.sub(r'batch(\d+)$',lambda m:'batch'+str(int(m[1])+1),run.name)
    continuation=build.continue_run(root,run,following)
    build.write(HERE/'loading/completion.json',{'verification':verified,'continuation':continuation,
        'revision_scope':'editorial_only','algorithm_checks_rerun':False})
    print(json.dumps({'verification':verified,'continuation':continuation},ensure_ascii=False),flush=True)
