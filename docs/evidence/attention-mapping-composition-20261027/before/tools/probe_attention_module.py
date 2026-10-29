#!/usr/bin/env python3
"""Read only the current attention module through the real frozen reader."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def run(runtime):
    sys.path.insert(0,str(runtime.resolve()/'src'))
    from infra_agent.knowledge import RepositoryKnowledge
    from infra_agent.records import Records
    inputs={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['knowledge/catalog.json','knowledge/sources.json']}
    catalog=json.loads((ROOT/'knowledge/catalog.json').read_bytes())
    registry={x['id']:x for x in json.loads((ROOT/'knowledge/sources.json').read_bytes())['sources']}
    entry=next(x for x in catalog['entries'] if x['id']=='K-FW-006')
    def complete(bank,path):
        offset=0;parts=[];last=None
        while True:
            result=bank.read(path,offset);parts.append(result['content']);last=result
            if result['next_offset'] is None:break
            offset=result['next_offset']
        return ''.join(parts),last
    with tempfile.TemporaryDirectory(prefix='attention-module-') as folder:
        models={r:'reader-fixture' for r in ['planner','executor','experimenter','diagnostician','evolution']}
        bank=RepositoryKnowledge(ROOT,Records(Path(folder)/'run'),selection={'view':'full','condition':'seed','freeze_initial':True},models=models)
        bank.begin_invocation('w001','planner','attention-reader',conversation_id='attention',generation='1')
        try:
            structure=bank.read(entry['id'],view='structure')
            headings=[x['title'] for x in structure['sections']]
            assert any(x.startswith('FlashAttention-2：') for x in headings)
            assert any(x.startswith('FlashAttention-3：') for x in headings)
            checked_sections=[]
            for title in ['异步缓冲：发布、读取、释放与代次',
                          'FP8：从量化输入到输出的完整尺度路径',
                          '从形状推到工作、流量、驻留和时间',
                          '一个完整的 GQA decode 取舍算例',
                          '诊断案例：三级流水为何比两级更慢']:
                assert title in headings
                section=bank.read(entry['id'],section=title)
                assert title in section['content'] and section['next_offset'] is None
                checked_sections.append({'title':title,'characters':len(section['content'])})
            body,last=complete(bank,entry['id'])
            assert hashlib.sha256(body.encode()).hexdigest()==entry['sha256']
            cards=[]
            for sid in entry['source_ids']:
                text,result=complete(bank,registry[sid]['local_path'])
                assert result['sha256']==registry[sid]['local_sha256']
                cards.append({'id':sid,'sha256':result['sha256'],'characters':len(text)})
            paper,result=complete(bank,registry['S-FLASHATTENTION2']['archive_fulltext'])
            assert 'FlashAttention' in paper and 'Parallelism' in paper
            try:bank.read('references/S-SDPA/fulltext.txt')
            except ValueError:source_rejected=True
            else:source_rejected=False
            assert source_rejected
            use=bank.validate_uses('planner',[{'entry_id':entry['id'],'revision':entry['revision'],'purpose':'Module reader fixture, no Agent/model execution.'}])[0]
            assert use['complete_read'] and use['sha256']==entry['sha256']
        finally:bank.end_invocation()
    unchanged=all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest for path,digest in inputs.items())
    assert unchanged
    return {'result':'PASS','entry_id':entry['id'],'revision':entry['revision'],'sha256':entry['sha256'],
            'source_count':len(cards),'sources':cards,'headings':headings,'checked_sections':checked_sections,'canonical_unchanged':unchanged,
            'new_paper_text_characters':len(paper),'source_archive_rejected':source_rejected,'complete_read':use['complete_read'],
            'scope':'Actual current module/source reading, FA2 fulltext, five reasoning sections, source-member boundary and adoption binding. A reader fixture, not Agent/model construction or GPU/performance validation.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();result=run(args.runtime);args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['result','entry_id','revision','source_count','complete_read','canonical_unchanged']}))
