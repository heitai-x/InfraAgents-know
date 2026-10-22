import collections
import hashlib
import json
import math
import pathlib
import re
import subprocess

REPORT=pathlib.Path(__file__).resolve().parent
ROOT=REPORT.parents[2]
KB=ROOT/'InfraAgents-know-astra'
OLD='0a5a0c0636101d0fbce7c3734636fd50be70e93a'
NEW='69408e3fc62c8d34e576c2f1ce6d247488d568b9'
def sha(b): return hashlib.sha256(b).hexdigest()
def git(*args): return subprocess.check_output(['git',*args],cwd=KB)
def oldfile(path): return git('show',OLD+':'+path)
def read(path): return json.loads((KB/path).read_text(encoding='utf-8'))
assert git('rev-parse','HEAD').decode().strip()==NEW
before=json.loads(oldfile('knowledge/catalog.json'))['entries']
after=read('knowledge/catalog.json')['entries']
old_by_id={e['id']:e for e in before}
new_by_id={e['id']:e for e in after}
assert old_by_id.keys()==new_by_id.keys()
changed=[e for e in after if e['sha256']!=old_by_id[e['id']]['sha256']]
history=[]
for e in changed:
    prior=old_by_id[e['id']]
    body=KB/f'knowledge/history/{e["id"]}/{prior["revision"]:04d}.md'
    meta=body.with_suffix('.json')
    history.append({'entry_id':e['id'],'prior_revision':prior['revision'],'current_revision':e['revision'],
       'revision_increment_exactly_one':e['revision']==prior['revision']+1,
       'old_body_preserved_exactly':body.read_bytes()==oldfile('knowledge/'+e['path']),
       'old_metadata_preserved':json.loads(meta.read_text(encoding='utf-8'))==prior,
       'prior_sha256':sha(body.read_bytes()),'current_sha256':sha((KB/'knowledge'/e['path']).read_bytes())})
old_history=git('ls-tree','-r','--name-only',OLD,'--','knowledge/history').decode().splitlines()
old_history_modified=[p for p in old_history if git('rev-parse',OLD+':'+p)!=git('rev-parse',NEW+':'+p)]
original_diff=git('diff','--name-only',OLD,NEW,'--','references').decode().splitlines()
old_sources={s['id']:s for s in json.loads(oldfile('knowledge/sources.json'))['sources']}
new_sources={s['id']:s for s in read('knowledge/sources.json')['sources']}
assert old_sources.keys()==new_sources.keys()
fixed_fields=['url','version','archive_manifest','archive_sha256','archive_reader','archive_fulltext','archive_original','archive_format','available_at','temporal_basis']
source_identity_changes=[{'id':sid,'field':field} for sid in new_sources for field in fixed_fields if old_sources[sid].get(field)!=new_sources[sid].get(field)]
policy_before=json.loads(oldfile('knowledge/temporal-policy.json'))
policy_after=read('knowledge/temporal-policy.json')
policy_changed_keys=[k for k in policy_after if policy_after[k]!=policy_before.get(k)]

old_claim='Transformers 的缓存说明区分已存历史 K/V、新输入、attention mask 和 cache position。增量前向需要让新 token 的位置与已有缓存长度一致。'
coverage=new_sources['S-HF-CACHE']['coverage']
new_claims=[]
for e in after:
    p=KB/'knowledge'/e['path']; text=p.read_text(encoding='utf-8')
    if 'S-HF-CACHE' in e['source_ids']:
        lines=[i for i,line in enumerate(text.splitlines(),1) if 'S-HF-CACHE' in line and coverage in line]
        new_claims.append({'entry_id':e['id'],'path':'knowledge/'+e['path'],'provenance_lines':lines,
                           'new_coverage_present_once':len(lines)==1,'old_claim_removed':old_claim not in text})
source_card=(KB/'knowledge/sources/S-HF-CACHE.md').read_text(encoding='utf-8')
hf_original=(KB/'references/S-HF-CACHE/fulltext.txt').read_text(encoding='utf-8')
source_card_checks={'coverage_matches_registry':coverage in source_card,
                    'local_summary_matches_coverage':new_sources['S-HF-CACHE']['local_summary']==coverage,
                    'states_field_contract_not_defined':'未定义 cache_position 或 position_ids' in source_card,
                    'states_multiply_mask_limitation':'不能保证 softmax 后概率为零' in source_card,
                    'original_still_omits_field_contracts':not re.search(r'cache_position|cache position|position_ids',hf_original,re.I)}
moe=(KB/'knowledge/models/moe-routing.md').read_text(encoding='utf-8')
match=re.search(r'B＝(\d+).*?k＝(\d+).*?分配是 \[([\d,]+)\]',moe)
assert match is not None
batch,k=int(match[1]),int(match[2]); allocation=[int(x) for x in match[3].split(',')]
secondary=[e for e,n in enumerate(allocation[1:],1) for _ in range(n)]
witness=[{'token':i,'experts':[0,e]} for i,e in enumerate(secondary)]
observed=collections.Counter(e for r in witness for e in r['experts'])
moe_checks={'source_B':batch,'source_k':k,'source_allocation':allocation,
            'total_matches_B_k':sum(allocation)==batch*k,
            'expert_bound':all(0<=n<=batch for n in allocation),
            'witness_has_B_tokens':len(witness)==batch,
            'witness_top_k_distinct':all(len(r['experts'])==k and len(set(r['experts']))==k for r in witness),
            'witness_matches_allocation':[observed[i] for i in range(len(allocation))]==allocation,
            'witness':witness}
mask={'scores':[2,5],'boolean_mask':[1,0],'multiplied_scores':[2,0],
      'disallowed_probability_after_numeric_multiply':1/(math.exp(2)+1),
      'disallowed_probability_under_legal_set':0,
      'supports_source_note':1/(math.exp(2)+1)>0}
previous_calc=json.loads((REPORT.parent/'round-1/calculation-checks.json').read_text(encoding='utf-8'))
affected_calc=[r for r in previous_calc['checks'] if r['entry_id'] in {e['id'] for e in changed} and r.get('finding_id')!='KR1-001']
calc_check={'new_numeric_example':'MoE allocation independently constructed above',
            'unchanged_numeric_examples_in_reread_chapters':len(affected_calc),
            'prior_results_still_applicable':all(r['pass'] for r in affected_calc),
            'basis':'Those numbers and formulas are unchanged by the narrow provenance/API edits; complete affected chapters reread. No claim of new hardware measurements.'}
version=read('VERSION.json'); plan=read('docs/expansion-plan.json')
v2check=read('docs/evidence/final-linux-validation.json')
inherited={
 'VERSION.completed_at':{'value':version['completed_at'],'interpretation':'Inherited v2 build completion date; current status is awaiting_independent_rereview, not current approval.'},
 'VERSION.validation':{'value':version['validation'],'interpretation':'Input manifest binds v2 catalog/sources, not 69408e3.'},
 'VERSION.runtime_integration':{'value':version['runtime_integration'],'interpretation':'Older navigation publication at aedff22, not the current 0802c3f GPU runtime verification.'},
 'plan.completed_at':{'value':plan['completed_at'],'interpretation':'Inherited v2 build completion date, not round-2 approval.'},
 'plan.final_validation':{'value':plan['final_validation'],'interpretation':'Same v2-only validation file; current release should point to current review/probe.'},
 'old_validation_manifest_matches_current':{p:sha((KB/p).read_bytes())==h for p,h in v2check['input_manifest'].items()},
 'release_action':'On final publication retain the old v2 evidence under a clearly historical reference and bind current validation/review to the round-2 report and 69408e3 content identities. Current awaiting-review status accurately describes the candidate.'}
version_arrays=[(name,value) for name,value in version.items() if isinstance(value,list) and value and isinstance(value[0],dict) and 'entry_id' in value[0]]
version_entries_ok=[]
for name,items in version_arrays:
    for item in items:
        e=new_by_id[item['entry_id']]
        version_entries_ok.append(item['revision']==e['revision'] and item['sha256']==e['sha256'] and item['path']==e['path'])
data={'old_head':OLD,'new_head':NEW,'changed_entries':len(changed),'unchanged_entries':len(after)-len(changed),
      'history_preservation':history,'existing_history_files':len(old_history),'existing_history_modified':old_history_modified,
      'original_archive_changed_paths':original_diff,'source_fixed_identity_changes':source_identity_changes,
      'temporal_policy_changed_keys':policy_changed_keys,'source_dates_and_eligibility_unchanged':policy_after['sources']==policy_before['sources'],
      'hf_provenance':new_claims,'hf_source_card':source_card_checks,'moe_feasibility':moe_checks,
      'mask_warning_example':mask,'calculation_scope':calc_check,'inherited_evidence':inherited,
      'VERSION_entry_count':len(version_entries_ok),'VERSION_entries_match_catalog':all(version_entries_ok)}
passed=all(all(h[k] for k in ['revision_increment_exactly_one','old_body_preserved_exactly','old_metadata_preserved']) for h in history)
passed=passed and not old_history_modified and not original_diff and not source_identity_changes and policy_changed_keys==['profile_id']
passed=passed and len(new_claims)==12 and all(x['new_coverage_present_once'] and x['old_claim_removed'] for x in new_claims) and all(source_card_checks.values())
passed=passed and all(moe_checks[k] for k in ['total_matches_B_k','expert_bound','witness_has_B_tokens','witness_top_k_distinct','witness_matches_allocation'])
passed=passed and len(version_entries_ok)==108 and all(version_entries_ok)
data['result']='PASS' if passed else 'FAIL'
(REPORT/'repair-checks.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'result':data['result'],'changed':len(changed),'unchanged':len(after)-len(changed),
                  'history_preserved':len(history),'old_history_files_unchanged':len(old_history),
                  'original_archive_changed_paths':original_diff,'source_fixed_identity_changes':source_identity_changes,
                  'source_policy_changed_keys':policy_changed_keys,'HF_provenance_count':len(new_claims),
                  'MoE_allocation':allocation,'MoE_witness_valid':moe_checks['witness_matches_allocation'],
                  'version_entries_ok':len(version_entries_ok),'inherited':inherited},ensure_ascii=False,indent=2))
