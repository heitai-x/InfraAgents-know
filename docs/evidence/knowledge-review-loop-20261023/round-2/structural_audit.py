import collections
import hashlib
import json
import pathlib
import re
import subprocess
from urllib.parse import unquote

REPORT = pathlib.Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
KB = ROOT / 'InfraAgents-know-astra'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
catalog, sources = read(KB/'knowledge/catalog.json'), read(KB/'knowledge/sources.json')
entries = catalog['entries']
by_id = {e['id']: e for e in entries}
by_path = {(KB/'knowledge'/e['path']).resolve():e['id'] for e in entries}
issues=[]
links=[]
graph={e['id']:set(e['related']) for e in entries}
history=[]
for e in entries:
    p=KB/'knowledge'/e['path']
    text=p.read_text(encoding='utf-8-sig')
    if sha(p)!=e['sha256']: issues.append(['body_sha',e['id']])
    if not text.startswith('# '+e['id']): issues.append(['heading_id',e['id']])
    if '```' in text or '~~~' in text or re.search(r'https?://',text): issues.append(['noncode_local_rule',e['id']])
    if set(e['related'])-by_id.keys(): issues.append(['unknown_related',e['id']])
    for m in re.finditer(r'\[[^\]]*\]\(([^)]+)\)',text):
        dest=m.group(1).split('#',1)[0]
        if not dest or '://' in dest: continue
        target=(p.parent/unquote(dest)).resolve()
        row={'from':e['id'],'target':dest,'exists':target.is_file()}
        if target in by_path:
            row['to']=by_path[target]
            graph[e['id']].add(by_path[target])
        if not row['exists']: issues.append(['missing_link',e['id'],dest])
        links.append(row)
    hdir=KB/'knowledge/history'/e['id']
    revisions=[]
    for meta in sorted(hdir.glob('*.json')):
        value=read(meta); body=meta.with_suffix('.md')
        valid=body.is_file() and sha(body)==value.get('sha256')
        revisions.append({'revision':value.get('revision'),'metadata':meta.relative_to(KB).as_posix(),'sha_valid':valid})
        if not valid: issues.append(['history_sha',str(meta)])
    archived={r['revision'] for r in revisions}
    expected=set(range(1,e['revision']))
    continuous=expected.issubset(archived)
    if not continuous: issues.append(['missing_archived_history',e['id'],sorted(expected-archived)])
    history.append({'id':e['id'],'current_revision_on_live_path':e['revision'],
                    'all_prior_revisions_archived':continuous,'revisions':revisions})
source_ids={s['id'] for s in sources['sources']}
policy=read(KB/'knowledge/temporal-policy.json')
source_checks=[]
for s in sources['sources']:
    sid=s['id']; meta=KB/s['archive_manifest']; m=read(meta)
    files=[]
    for f in m['files']:
        p=meta.parent/f['path']
        ok=p.is_file() and sha(p)==f['sha256']
        if not ok: issues.append(['archive_file_sha',sid,f['path']])
        files.append(ok)
    p=policy['sources'][sid]
    date_ok=p['available_at']<='2026-04-30' and s['available_at']<='2026-04-30'
    row={'id':sid,'source_card_sha_valid':sha(KB/s['local_path'])==s['local_sha256'],
         'archive_manifest_sha_valid':sha(meta)==s['archive_sha256'],
         'archive_files':len(files),'all_archive_member_sha_valid':all(files),
         'available_at':s['available_at'],'cutoff_valid':date_ok}
    source_checks.append(row)
    if not all([row['source_card_sha_valid'],row['archive_manifest_sha_valid'],date_ok]): issues.append(['source_metadata',sid])
incoming=collections.Counter(x for ys in graph.values() for x in ys)
unseen=set(graph); components=[]
while unseen:
    todo=[next(iter(unseen))]; component=set()
    while todo:
        x=todo.pop()
        if x in component: continue
        component.add(x)
        todo.extend(graph[x]-component)
        todo.extend(k for k,v in graph.items() if x in v and k not in component)
    unseen-=component; components.append(sorted(component))
nav={}
for rel in ['knowledge/INDEX.md','knowledge/DESIGN.md','knowledge/DIAGNOSIS.md']+[f'knowledge/{c}/README.md' for c in sorted({e['category'] for e in entries})]:
    p=KB/rel; text=p.read_text(encoding='utf-8-sig'); targets=[]; bad=[]
    for m in re.finditer(r'\[[^\]]*\]\(([^)]+)\)',text):
        dest=m.group(1).split('#',1)[0]
        if not dest or '://' in dest: continue
        target=(p.parent/unquote(dest)).resolve()
        if not target.is_file(): bad.append(dest)
        if target in by_path: targets.append(by_path[target])
    nav[rel]={'linked_entry_count':len(set(targets)),'entry_ids':sorted(set(targets)),'missing_files':bad}
    for b in bad: issues.append(['navigation_missing',rel,b])
plan=read(KB/'docs/expansion-plan.json')
data={'kb_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=KB,text=True).strip(),
      'catalog_schema':catalog['schema_version'],'sources_schema':sources['schema_version'],
      'entry_count':len(entries),'source_count':len(source_ids),
      'categories':dict(collections.Counter(e['category'] for e in entries)),
      'statuses':dict(collections.Counter(e['status'] for e in entries)),
      'mechanism_body_links_checked':len(links),'connected_components':components,
      'zero_incoming_entries':sorted(set(graph)-incoming.keys()),'navigation':nav,
      'history':history,'sources':source_checks,'plan_status':plan['status'],
      'plan_existing_count':len(plan['existing_chapters']),'plan_new_count':len(plan['new_topics']),
      'issues':issues,'scope_note':'Mechanical checks supplement the separately recorded complete semantic reading; source hashes and dates are not semantic validation.'}
(REPORT/'structural-audit.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in data.items() if k not in ['history','sources','navigation','connected_components']},ensure_ascii=False,indent=2))
print('component_sizes',list(map(len,components)))
print('navigation_counts',{k:v['linked_entry_count'] for k,v in nav.items()})
