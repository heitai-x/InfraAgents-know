import hashlib
import json
import pathlib
import subprocess
import sys

REPORT=pathlib.Path(__file__).resolve().parent
ROOT=REPORT.parents[2]
KB=ROOT/'InfraAgents-know-astra'
PREVIOUS=REPORT.parent/'round-1'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
catalog=json.loads((KB/'knowledge/catalog.json').read_text(encoding='utf-8'))['entries']
old=json.loads((PREVIOUS/'review.json').read_text(encoding='utf-8'))
old_entries={e['id']:e for e in old['entries']}
changed=[e for e in catalog if sha(KB/'knowledge'/e['path'])!=old_entries[e['id']]['sha256']]
if sys.argv[1]=='inventory':
    rows=[]
    for e in catalog:
        p=KB/'knowledge'/e['path']; prior=old_entries[e['id']]
        current_sha=sha(p); same=current_sha==prior['sha256']
        rows.append({'id':e['id'],'path':'knowledge/'+e['path'],'title':e['title'],
                     'revision':e['revision'],'sha256':current_sha,'catalog_sha256':e['sha256'],
                     'line_count':len(p.read_text(encoding='utf-8').splitlines()),
                     'changed':not same,'read_ranges_this_round':[],
                     'prior_complete_read':{'round':1,'head':old['baseline']['head'],'sha256':prior['sha256'],'ranges':prior['read_ranges']},
                     'review_basis':'unchanged bytes carry forward complete round-1 reading' if same else 'requires complete reread',
                     'source_ids':e['source_ids']})
    (REPORT/'entry-inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CURRENT_HEAD',subprocess.check_output(['git','rev-parse','HEAD'],cwd=KB,text=True).strip())
    print('CHANGED',[(i,e['id'],e['path']) for i,e in enumerate(changed)])
elif sys.argv[1]=='read':
    for e in changed[int(sys.argv[2]):int(sys.argv[3])]:
        p=KB/'knowledge'/e['path']
        print('\n===== '+e['id']+' knowledge/'+e['path']+' =====')
        for i,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1): print(str(i)+': '+line)
elif sys.argv[1]=='ack':
    p=REPORT/'entry-inventory.json'; rows=json.loads(p.read_text(encoding='utf-8'))
    ids={e['id'] for e in changed[int(sys.argv[2]):int(sys.argv[3])]}
    for row in rows:
        if row['id'] in ids:
            row['read_ranges_this_round']=[[1,row['line_count']]]
            row['review_basis']='complete changed chapter reread at current SHA'
    p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ACK',sorted(ids))
