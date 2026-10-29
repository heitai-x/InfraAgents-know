#!/usr/bin/env python3
"""Check just the delivered attention module and its reading chain."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from check_attention_algorithms import run as algorithm_checks

ROOT=Path(__file__).resolve().parents[1]


def run():
    catalog=json.loads((ROOT/'knowledge/catalog.json').read_bytes())
    registry=json.loads((ROOT/'knowledge/sources.json').read_bytes())
    reading=json.loads((ROOT/'knowledge/source-reading.json').read_bytes())['sources']
    entry=next(item for item in catalog['entries'] if item['id']=='K-FW-006')
    by_source={item['id']:item for item in registry['sources']}
    files=[('knowledge/'+entry['path'],entry['sha256'])]
    files += [(by_source[sid]['local_path'],by_source[sid]['local_sha256']) for sid in entry['source_ids']]
    identities=[];links=[];issues=[]
    for name,expected in files:
        path=ROOT/name;raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest()
        identities.append({'path':name,'sha256':actual,'matches_registered':actual==expected})
        if actual!=expected:issues.append('Identity changed: '+name)
        for target in re.findall(r'\]\(([^)]+)\)',raw.decode('utf-8')):
            if '://' in target:continue
            resolved=(path.parent/target.split('#')[0]).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.is_file():issues.append('Missing reading path: '+target)
            if resolved.is_relative_to(ROOT/'references'):
                rel=resolved.relative_to(ROOT).as_posix();sid=resolved.relative_to(ROOT/'references').parts[0]
                if rel not in reading[sid]['paths']:issues.append('Implementation linked as reading material: '+rel)
            links.append({'from':name,'target':target,'exists':resolved.is_file()})
    algorithms=algorithm_checks()
    if algorithms['result']!='PASS':issues.append('Independent scalar algorithm checks failed')
    return {'result':'PASS' if not issues else 'FAIL','entry_id':entry['id'],'revision':entry['revision'],
            'chapter':'knowledge/'+entry['path'],'source_ids':entry['source_ids'],'identities':identities,
            'reading_paths_checked':len(links),'algorithm_checks':algorithms['count'],'issues':issues,
            'scope':'Current module identity, local reading chain, published reading members and independent finite-input scalar algorithms. No GPU/kernel/service measurement or automatic semantic-quality certification.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args();result=run()
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['result','entry_id','revision','reading_paths_checked','algorithm_checks','issues']},ensure_ascii=False))
    raise SystemExit(result['result']!='PASS')
