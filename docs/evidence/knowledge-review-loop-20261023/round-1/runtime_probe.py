import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tomllib

sys.dont_write_bytecode = True
REPORT = pathlib.Path(__file__).resolve().parent
RUNTIME = pathlib.Path('/mnt/infra_agents/worktrees/gpu-knowledge-v2-runtime-20261023')
CANONICAL = pathlib.Path('/mnt/infra_agents/knowledge-worktrees/gpt-6-astra-2026-04-30-v2')
SCRATCH = REPORT / 'scratch-bank'
sys.path.insert(0, str(RUNTIME/'src'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
before = {name:sha(CANONICAL/name) for name in ['knowledge/catalog.json','knowledge/sources.json']}
if SCRATCH.exists():
    assert all(sha(SCRATCH/name)==value for name,value in before.items()), 'Existing scratch identity mismatch'
else:
    SCRATCH.mkdir()
    for name in ['knowledge','references','docs']:
        shutil.copytree(CANONICAL/name,SCRATCH/name)
task = tomllib.loads((REPORT/'task.toml').read_text(encoding='utf-8-sig'))
from infra_agent.knowledge import RepositoryKnowledge
from infra_agent.records import Records
from infra_agent.knowledge_store import audit, read_revision
records = Records(REPORT/'runtime-records-v2')
roles = ['planner','executor','experimenter','diagnostician','evolution']
models = {r:task['agent']['model'] for r in roles}
bank = RepositoryKnowledge(SCRATCH,records,selection=task['knowledge'],models=models)
snapshot = bank.prepare_work('review')
bank.begin_invocation('review','planner','probe',conversation_id='review-only',generation='one')
catalog=snapshot['catalog']['entries']
reads=[]
for e in catalog:
    relative='knowledge/'+e['path']
    chunks=[]; offset=0
    while True:
        result=bank.read(relative,offset)
        chunks.append(result['content'])
        if result['next_offset']>=result['total_characters']: break
        if result['next_offset']<=offset: raise AssertionError('Reader did not advance')
        offset=result['next_offset']
    direct=(SCRATCH/relative).read_bytes().decode('utf-8')
    joined=''.join(chunks)
    assert joined==direct, e['id']
    assert result['sha256']==e['sha256'], e['id']
    reads.append({'id':e['id'],'revision':e['revision'],'sha256':result['sha256'],
                  'characters':len(joined),'pages':len(chunks),'complete':True})
nav={}
for relative in ['knowledge/README.md','knowledge/DESIGN.md','knowledge/DIAGNOSIS.md','knowledge/INDEX.md','knowledge/SOURCES.md','knowledge/COVERAGE.md']:
    chunks=[]; offset=0
    while True:
        r=bank.read(relative,offset); chunks.append(r['content'])
        if r['next_offset']>=r['total_characters']: break
        offset=r['next_offset']
    body=''.join(chunks)
    linked=set(re.findall(r'K-[A-Z]+-\d+',body))
    nav[relative]={'characters':len(body),'entry_ids':sorted(linked),'entry_count':len(linked)}
restrictions=[]
sources={s['id']:s for s in snapshot['sources']['sources']}
for sid in task['knowledge']['restricted_originals']:
    card=bank.read(sources[sid]['local_path'])
    blocked=[]
    for name in ['fulltext.txt','reader.html','manifest.json']:
        path='references/'+sid+'/'+name
        try: bank.read(path)
        except ValueError as exc: blocked.append({'path':path,'error':str(exc)})
        else: raise AssertionError('Restricted original readable: '+path)
    restrictions.append({'id':sid,'card_readable':bool(card['content']),'blocked':blocked})
allowed=bank.read('references/S-HF-CACHE/fulltext.txt')
assert allowed['content']==(SCRATCH/'references/S-HF-CACHE/fulltext.txt').read_bytes().decode('utf-8')
archive_index=bank.read('references/index.html')['content']
for sid in task['knowledge']['restricted_originals']:
    assert '<td>'+sid+'</td>' not in archive_index, sid
queries=['online multi-turn session cancellation','多轮 会话 历史 增长','cancel KV in-flight ownership','精确 greedy BF16 numerical','CUDA Graph dynamic batch','prefill decode batching BS','CPU quota GPU memory','p95 latency offered workload']
retrieval={q:bank.search(q,limit=6) for q in queries}
uses=bank.validate_uses('planner',[{'entry_id':e['id'],'revision':e['revision'],'purpose':'Independent read-path validation; no model adoption or execution claim'} for e in catalog])
bank.end_invocation()
history_count=0
for e in catalog:
    for revision in range(1,e['revision']+1):
        meta,body=read_revision(SCRATCH,e['id'],revision)
        assert hashlib.sha256(body).hexdigest()==meta['sha256']
        history_count+=1
after = {name:sha(CANONICAL/name) for name in before}
assert before==after
data={'runtime_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME,text=True).strip(),
      'remote_kb_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=CANONICAL,text=True).strip(),
      'task_sha256':sha(REPORT/'task.toml'),'canonical_hashes_before':before,'canonical_hashes_after':after,
      'policy':task['knowledge'],'models':models,'index':bank.index(),'navigation':nav,
      'chapter_reads':reads,'restricted_originals_checks':restrictions,'retrieval':retrieval,
      'validated_use_count':len(uses),'history_versions_read':history_count,
      'allowed_original_read':{'id':'S-HF-CACHE','complete':True},
      'audit':audit(SCRATCH),'result':'PASS',
      'scope':'Actual RepositoryKnowledge API; no Agent process, model inference, GPU work, publication or canonical mutation. Temporary bank and Records are confined to report directory.'}
(REPORT/'runtime-probe.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'result':data['result'],'runtime_head':data['runtime_head'],'remote_kb_head':data['remote_kb_head'],
                  'entry_count':len(reads),'validated_use_count':len(uses),'history_versions_read':history_count,
                  'restricted_sources':len(restrictions),'nav_counts':{k:v['entry_count'] for k,v in nav.items()},
                  'hashes':before,'audit':data['audit']},ensure_ascii=False))
