import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

root = Path(__file__).resolve().parents[3]
evidence = root / 'docs/evidence/knowledge-cleanup-20261027'
spec = importlib.util.spec_from_file_location('review_validator', root / 'tools/validate_knowledge.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)

def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')

checks = []
with tempfile.TemporaryDirectory(prefix='astra-validator-negative-') as temporary:
    copied = Path(temporary) / 'bank'
    shutil.copytree(root, copied)
    initial = validator.validate(copied)
    assert initial['result'] == 'PASS', initial['issues']
    checks.append({'check': 'current_revision', 'passed': True})
    audit_path = copied / 'knowledge/temporal-audit.json'
    audit_bytes = audit_path.read_bytes()
    for name, edit, expected in [
        ('stale_profile', lambda x: x.update(profile_id='base-gpt-6-astra-2026-04-30-v2.1'), 'temporal_audit_profile'),
        ('stale_revision', lambda x: x['entries'][0].update(revision=x['entries'][0]['revision'] + 1), 'temporal_audit_entry_binding'),
        ('missing_audit_entry', lambda x: x['entries'].pop(), 'temporal_audit_entry_set'),
    ]:
        value = json.loads(audit_bytes)
        edit(value)
        audit_path.write_bytes(encode(value))
        result = validator.validate(copied)
        kinds = {issue['kind'] for issue in result['issues']}
        assert result['result'] == 'FAIL' and expected in kinds, (name, kinds)
        checks.append({'check': name, 'passed': True, 'rejected_by': sorted(kinds)})
        audit_path.write_bytes(audit_bytes)
    registry_path = copied / 'knowledge/sources.json'
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    source = next(s for s in registry['sources'] if s['id'] == 'S-TRTLLM-CODE')
    note_path = copied / source['local_path']
    note_bytes = note_path.read_bytes()
    note_path.write_bytes(note_bytes.replace(b'astra-source-locations.json', b'absent-source-locations.json'))
    source['local_sha256'] = hashlib.sha256(note_path.read_bytes()).hexdigest()
    registry_path.write_bytes(encode(registry))
    result = validator.validate(copied)
    assert any(issue['kind'] == 'missing_local_link' and Path(issue.get('file', '')).as_posix() == source['local_path'] for issue in result['issues']), result['issues']
    checks.append({'check': 'broken_source_locator_with_correct_body_hash', 'passed': True, 'rejected_by': 'missing_local_link'})
    note_path.write_bytes(note_bytes)
    registry_path.write_bytes(registry_bytes)
    assert validator.validate(copied)['result'] == 'PASS'
    checks.append({'check': 'restored_fixture', 'passed': True})

old_manifest = json.loads((evidence / 'delivery-v2.2-r1/MANIFEST.json').read_bytes())
old_catalog = json.loads((evidence / 'before/knowledge/catalog.json').read_bytes())
catalog = json.loads((root / 'knowledge/catalog.json').read_bytes())
old_by_id = {entry['id']: entry for entry in old_catalog['entries']}
changed = []
for entry in catalog['entries']:
    old = old_by_id[entry['id']]
    raw = (root / 'knowledge' / entry['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() == old['sha256']:
        assert entry == old
    else:
        changed.append(entry['id'])
        archived = root / f'knowledge/history/{entry["id"]}/{old["revision"]:04d}'
        assert hashlib.sha256(archived.with_suffix('.md').read_bytes()).hexdigest() == old['sha256']
        assert json.loads(archived.with_suffix('.json').read_bytes()) == old
        assert entry['revision'] == old['revision'] + 1
assert set(changed) == {'K-MODEL-001', 'K-FW-001', 'K-TOOLS-001', 'K-WORK-006'}
preserved = [item for item in old_manifest['files'] if item['path'].startswith((
    'InfraAgents-know/references/', 'InfraAgents-know/knowledge/history/', 'InfraAgents-know/docs/evidence/'))]
for item in preserved:
    assert hashlib.sha256((root.parent / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
report = {'result': 'PASS', 'scope': 'Isolated structural-negative fixtures, chapter continuity and byte preservation; no independent semantic reviewer or model execution.',
    'validator_sha256': hashlib.sha256((root / 'tools/validate_knowledge.py').read_bytes()).hexdigest(),
    'checks': checks, 'changed_chapters': changed, 'unchanged_chapters': 104,
    'prior_archive_history_and_evidence_files_unchanged': len(preserved)}
(evidence / 'regression.json').write_bytes(encode(report))
examples = json.loads((evidence / 'examples.json').read_bytes())
print(json.dumps({'regression': report, 'arithmetic': examples['summary'], 'arithmetic_passed': examples['passed']}, ensure_ascii=False))
