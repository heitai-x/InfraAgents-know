import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
KB = ROOT / 'InfraAgents-know-astra'
REPORT = pathlib.Path(__file__).resolve().parent
CATALOG = json.loads((KB / 'knowledge/catalog.json').read_text(encoding='utf-8-sig'))

def inventory():
    rows = []
    for entry in CATALOG['entries']:
        path = KB / 'knowledge' / entry['path']
        raw = path.read_bytes()
        text = raw.decode('utf-8-sig')
        rows.append(dict(id=entry['id'], path='knowledge/' + entry['path'],
                         sha256=hashlib.sha256(raw).hexdigest(),
                         catalog_sha256=entry['sha256'],
                         line_count=len(text.splitlines()), chars=len(text),
                         source_ids=entry['source_ids'], related=entry['related'],
                         read_ranges=[], findings=[]))
    (REPORT / 'entry-inventory.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'entries': len(rows), 'characters': sum(r['chars'] for r in rows),
                      'sha_mismatches': [r['id'] for r in rows if r['sha256'] != r['catalog_sha256']]}, ensure_ascii=False))

if sys.argv[1] == 'inventory':
    inventory()
elif sys.argv[1] == 'read':
    for index in range(int(sys.argv[2]), int(sys.argv[3])):
        e = CATALOG['entries'][index]
        p = KB / 'knowledge' / e['path']
        print(f'\n===== {index}: {e["id"]} knowledge/{e["path"]} =====')
        print('Sources: ' + ', '.join(e['source_ids']))
        for i, line in enumerate(p.read_text(encoding='utf-8-sig').splitlines(), 1):
            print(f'{i}: {line}')
elif sys.argv[1] == 'ack':
    p = REPORT / 'entry-inventory.json'
    rows = json.loads(p.read_text(encoding='utf-8'))
    for index in range(int(sys.argv[2]), int(sys.argv[3])):
        rows[index]['read_ranges'] = [[1, rows[index]['line_count']]]
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Acknowledged complete display/read of entries {sys.argv[2]} through {int(sys.argv[3])-1}')
