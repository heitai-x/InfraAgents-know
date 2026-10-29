#!/usr/bin/env python3
"""Exercise the real frozen reader for the three expanded modules and Attention.

This is an interface fixture. It invokes no language model or device kernel.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
IDS = ['K-ALG-001', 'K-ALG-002', 'K-HW-011', 'K-FW-006']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(runtime, identifiers=None):
    sys.path.insert(0, str(runtime.resolve()/'src'))
    from infra_agent.knowledge import RepositoryKnowledge
    from infra_agent.records import Records

    inputs = {p: digest(ROOT/p) for p in ['knowledge/catalog.json', 'knowledge/sources.json']}
    entries = {x['id']: x for x in json.loads((ROOT/'knowledge/catalog.json').read_bytes())['entries']}
    sources = {x['id']: x for x in json.loads((ROOT/'knowledge/sources.json').read_bytes())['sources']}

    def complete(bank, path, **kwargs):
        offset, parts = 0, []
        while True:
            result = bank.read(path, offset, **kwargs)
            parts.append(result['content'])
            if result['next_offset'] is None:
                return ''.join(parts), result
            assert result['next_offset'] > offset
            offset = result['next_offset']

    cards, modules, originals = {}, [], []
    with tempfile.TemporaryDirectory(prefix='mechanism-reader-') as folder:
        models = {r: 'reader-fixture' for r in ['planner', 'executor', 'experimenter', 'diagnostician', 'evolution']}
        bank = RepositoryKnowledge(ROOT, Records(Path(folder)/'run'),
                                   selection={'view': 'full', 'condition': 'seed', 'freeze_initial': True}, models=models)
        bank.begin_invocation('w001', 'planner', 'mechanism-reader', conversation_id='mechanisms', generation='1')
        try:
            for identifier in identifiers if identifiers is not None else IDS:
                entry = entries[identifier]
                structure = bank.read(identifier, view='structure')
                sections = []
                body, _ = complete(bank, identifier)
                assert hashlib.sha256(body.encode()).hexdigest() == entry['sha256']
                expected_titles = []
                in_fence = False
                for line in body.splitlines():
                    if line.startswith('```'):
                        in_fence = not in_fence
                    if not in_fence and line.startswith('## '):
                        expected_titles.append(line[3:])
                assert [x['title'] for x in structure['sections']] == expected_titles
                for section in structure['sections']:
                    title = section['title']
                    text, _ = complete(bank, identifier, section=title)
                    assert text.startswith('## '+title+'\n')
                    assert text in body
                    sections.append({'title': title, 'characters': len(text)})
                for sid in entry['source_ids']:
                    if sid in cards:
                        continue
                    source = sources[sid]
                    text, result = complete(bank, source['local_path'])
                    assert result['sha256'] == source['local_sha256']
                    assert hashlib.sha256(text.encode()).hexdigest() == source['local_sha256']
                    cards[sid] = {'sha256': result['sha256'], 'characters': len(text)}
                use = bank.validate_uses('planner', [{'entry_id': identifier, 'revision': entry['revision'],
                                                     'purpose': 'Frozen module reader fixture; no model invocation.'}])[0]
                assert use['complete_read'] and use['sha256'] == entry['sha256']
                modules.append({'id': identifier, 'revision': entry['revision'], 'sha256': entry['sha256'],
                                'characters': len(body), 'sections': sections, 'complete_read': True})
            for sid in ['S-PAGED', 'S-ORCA', 'S-CUDA-ASYNC']:
                if sid not in cards:
                    continue
                text, result = complete(bank, sources[sid]['archive_fulltext'])
                assert text.strip() and result['sha256']
                originals.append({'id': sid, 'characters': len(text), 'sha256': result['sha256']})
            rejected = []
            for path in ['references/S-VLLM-KV/fulltext.txt', 'references/S-VLLM-SCHED/fulltext.txt',
                         'references/S-SDPA/fulltext.txt', 'references/S-TRITON/fulltext.txt']:
                try:
                    bank.read(path)
                except ValueError:
                    rejected.append(path)
                else:
                    raise AssertionError(f'Implementation archive exposed: {path}')
        finally:
            bank.end_invocation()
    assert all(digest(ROOT/path) == sha for path, sha in inputs.items())
    return {'result': 'PASS', 'modules': modules, 'source_count': len(cards), 'sources': cards,
            'readable_originals': originals, 'implementation_paths_rejected': rejected,
            'canonical_unchanged': True, 'documents': inputs,
            'scope': 'Actual frozen body/H2/source reading and adoption identity for selected modules; source boundary enforcement.',
            'limits': ['No language model invocation, target-device execution, performance or controlled Agent effectiveness assessment.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--entries', help='Comma-separated catalog IDs; defaults to the original four-module fixture')
    args = parser.parse_args()
    result = run(args.runtime, args.entries.split(',') if args.entries else None)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'result': result['result'], 'module_count': len(result['modules']),
                      'source_count': result['source_count'], 'sections': sum(len(x['sections']) for x in result['modules'])}))
