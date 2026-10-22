import datetime
import hashlib
import json
from pathlib import Path
import subprocess

REPORT=Path(__file__).resolve().parent
ROOT=REPORT.parents[2]
KB=ROOT/'InfraAgents-know-astra'
OLD='0a5a0c0636101d0fbce7c3734636fd50be70e93a'
NEW='69408e3fc62c8d34e576c2f1ce6d247488d568b9'
def load(name):return json.loads((REPORT/name).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
entries=load('entry-inventory.json')
repairs=load('repair-checks.json')
struct=load('structural-audit.json')
runtime=load('runtime-probe.json')
changed=[e for e in entries if e['changed']]
assert len(changed)==13
assert all(e['read_ranges_this_round']==[[1,e['line_count']]] for e in changed)
assert repairs['result']=='PASS' and runtime['result']=='PASS' and not struct['issues']
assert runtime['remote_kb_head']==NEW
assert all(sha(KB/p)==h for p,h in runtime['canonical_hashes_before'].items())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=KB,text=True).strip()==NEW

resolutions=[
 {'finding_id':'KR1-001','previous_severity':'P2','status':'RESOLVED',
  'path':'knowledge/models/moe-routing.md','lines':[13,15],
  'evidence':'第15行明确B为本轮token数、每token选两个不同专家、总分派32、单专家最多16；新分配[16,4,3,3,2,2,1,1]满足上界并有显式16-token distinct top-2构造。热点可能成为慢端的结论同时保留放置/读取/调度条件。',
  'checks':'repair-checks.json:moe_feasibility',
  'new_issue_found':False},
 {'finding_id':'KR1-002','previous_severity':'P2','status':'RESOLVED',
  'path':'knowledge/sources/S-HF-CACHE.md','lines':[9,21],
  'affected_current_provenance':repairs['hf_provenance'],
  'evidence':'来源卡、registry coverage/local_summary及全部12处provenance均限缩为逐层KV复用、past+new mask和存储形态。SYS003第9行、SYS007第13行保留机制关系并将API维度/padding/推进约定交给固定Task实现核对。原文未定义字段的边界被显式说明。',
  'mask_warning':'源文score×mask示意的限制说明正确；独立例中scores=[2,5]乘[1,0]后，被屏蔽项仍获约0.1192概率，而合法集合定义应为0。没有改动固定原文或错误删除机制推导。',
  'checks':'repair-checks.json:hf_provenance,hf_source_card,mask_warning_example',
  'new_issue_found':False},
 {'finding_id':'KR1-003','previous_severity':'P3','status':'RESOLVED',
  'path':'knowledge/README.md','lines':[1,3],
  'evidence':'独立入口已写v2.1复审候选、73+35=108篇、114来源；根README也明确v2已交付、v2.1待独立复审。计数与catalog/plan一致。',
  'new_issue_found':False}]
source_reads=[
 {'source_id':'S-HF-CACHE','path':'references/S-HF-CACHE/fulltext.txt','read_ranges_this_round':[[1,130]],'complete_archived_text_read_this_round':True,
  'sha256':sha(KB/'references/S-HF-CACHE/fulltext.txt'),'purpose':'复核新的来源摘要、字段合同不在本文件及mask示意限制'},
 {'source_id':'S-MIXTRAL','path':'references/S-MIXTRAL/fulltext.txt','read_ranges_this_round':[[93,144]],'complete_archived_text_read_this_round':False,
  'sha256':sha(KB/'references/S-MIXTRAL/fulltext.txt'),'purpose':'复核TopK路由定义及两个不同权重专家，支持单专家上界'}]
for e in entries:
    e['assessment']='PASS_WITHIN_REVIEW_SCOPE'
    e['resolved_findings']=[r['finding_id'] for r in resolutions if r.get('path')==e['path'] or any(x['entry_id']==e['id'] for x in r.get('affected_current_provenance',[]))]
    e['scope_limit']='完整正文阅读结论与实际来源复核范围分别记录，不声称所有114来源已全读或目标引擎已通过。'
data={
 'schema_version':1,'round':2,'reviewer':'independent sub-agent /root/knowledge_release_review',
 'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'kb_verdict':'PASS','knowledge_gate':'PASS','new_blocking_findings':[],
 'decision':'KR1-001、KR1-002、KR1-003均已解决；本次13篇修订未发现新的机制、资源或来源范围问题。对69408e3内容与0802c3f知识读取接口签发知识库复审通过。',
 'baseline':{'repository':str(KB),'branch':'gpt-6-astra-2026-04-30','head':NEW,'previous_review_head':OLD,
             'catalog_sha256':sha(KB/'knowledge/catalog.json'),'sources_sha256':sha(KB/'knowledge/sources.json'),
             'source_cutoff_inclusive':'2026-04-30','entry_count':108,'source_count':114},
 'reading_scope':{'changed_chapters_fully_reread_this_round':13,'unchanged_chapters_with_round1_full_read_and_identical_bytes':95,
                  'complete_body_coverage_across_rounds':108,'all_108_semantically_reread_this_round':False,
                  'source_texts_fully_reread_this_round':1,'source_texts_partially_reread_this_round':1,
                  'cumulative_independent_source_reading':{'distinct_sources':19,'complete_texts':9,'partial_texts':10,'all_114_sources_fully_reread':False},
                  'note':'Round 1全文阅读108篇；Round 2全文复读13篇修改内容，95篇按精确SHA继承前轮结论。来源核查不是全114篇复读；原文图片/PDF版面未全部视觉复审。'},
 'resolutions':resolutions,'entries':entries,'source_reads_this_round':source_reads,
 'validation':{'structural_result':'PASS','body_links_checked':1921,'all_categories_preserved':struct['categories'],
               'history_old_current_bodies_and_metadata_preserved':13,'preexisting_history_files_unchanged':166,
               'original_archive_changed_paths':[],'fixed_source_identity_changes':[],
               'source_dates_and_eligibility_unchanged':True,'temporal_policy_changed_keys':['profile_id'],
               'temporal_result':runtime['audit']['temporal'],'VERSION_entry_identities_match_catalog':108,
               'moe_resource_example':'PASS with constructive legal routing witness',
               'mask_source_warning':'PASS with explicit nonzero-probability counterexample to numeric zero multiplication',
               'other_numeric_scope':repairs['calculation_scope'],
               'evidence':['repair-checks.json','structural-audit.json','runtime-probe.json']},
 'actual_runtime':{'head':runtime['runtime_head'],'bank_head':runtime['remote_kb_head'],
                   'canonical_hashes_match_local':True,'selection':runtime['index']['selection'],
                   'chapter_reads_complete':108,'knowledge_uses_validated':108,'current_and_historical_versions_read':204,
                   'navigation_entries':{k:v['entry_count'] for k,v in runtime['navigation'].items()},
                   'restricted_source_count':9,'restricted_text_reader_manifest_denials':27,
                   'source_cards_allowed':9,'allowed_original_full_read':'S-HF-CACHE',
                   'probe_scope':'真实RepositoryKnowledge与冻结调用/使用记录API；绑定五角色Luna，实际无模型读取调用为planner验证作用域。未运行Luna或GPU，不证明Agent已理解/采用知识，也未验证新的learned发布结果。'},
 'inherited_evidence':repairs['inherited_evidence'],
 'publication_followup':{'blocks_kb_content_pass':False,
                         'action':'发布记录提交应将本轮review/runtime-probe与69408e3内容身份绑定，旧v2 completed_at/validation/runtime_integration作为继承历史明确标注。当前awaiting_independent_rereview准确描述提交时状态，不应提前伪造当前通过。',
                         'planned_final_check':'可核对后续发布提交仅加入报告/状态、正文/catalog/sources/原始归档内容身份保持不变；若改机制正文则必须重新审查对应差异。'},
 'task_readiness':{'excluded_from_kb_verdict':True,'sampling_contract':'PENDING_USER_CLARIFICATION',
                  'note':'workload多种采样与quality每轮精确greedy的适用范围由用户确认。知识库通过不等于该Task契约已解决、已启动或引擎质量/性能达成。',
                  'new_task_started_by_reviewer':False},
 'scope_judgment':'适合作为固定Qwen3-8B/4090在线多轮、取消隔离、资源受限与独立验证的机制seed；BS<=16、无固定性能阈值及不允许近似仍由Task决定。',
 'non_actions':['未修改canonical KB','未修改Task或旧实验','未修改shared runtime','未启动Agent/模型/GPU','未推送','未引入截止日后外部技术事实'],
 'artifacts':['review.md','review.json','entry-inventory.json','repair-checks.json','structural-audit.json','runtime-probe.json']}
(REPORT/'review.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

md=['# 知识库独立复审：round 2','',
'**知识库结论：PASS。** 对实际提交 `69408e3fc62c8d34e576c2f1ce6d247488d568b9` 的本轮复审通过。KR1-001、KR1-002、KR1-003 均已解决，修订没有发现新的机制、资源算例或来源范围问题。该结论承接 round 1 的完整正文审查，并以本轮实际新内容、历史/原文身份及运行时检查为依据。','',
'**Task 采样契约仍独立 pending，不改变上述 KB PASS。** 用户尚需明确“多种采样方式”和“每轮精确 greedy token”的适用范围；知识库通过不等于 Task 契约已解决，也不等于真实 GPU 引擎已正确或性能达成。本审查没有启动 Agent、模型或 GPU。','',
'## 修订核查','',
'| 前轮问题 | 结论 | 本轮实际证据 |','|---|---|---|',
'| KR1-001 · MoE 偏斜算例 | 已解决 | K-MODEL-003 第15行改为 `[16,4,3,3,2,2,1,1]`，总量32、最大16，并明确B为本轮token数及distinct top-2。 |',
'| KR1-002 · HF 来源归因 | 已解决 | 来源卡、registry摘要及全部12处provenance只归因于固定文档实际说明的KV、mask与存储形态；SYS003/007保留机制推导并限定API合同。 |',
'| KR1-003 · README过时 | 已解决 | 根与knowledge README均明确v2.1复审候选；73+35=108篇、114来源与实际目录一致。 |','',
'MoE 复算不止检查 `16+4+3+3+2+2+1+1=32`：已构造16条token路由，每条均选择专家0及另一个非零专家；其他专家分别被选4、3、3、2、2、1、1次。每条路由有两个不同专家，所得计数与正文完全一致。正文也不再将热点项数直接等同全部关键路径，而保留专家放置、权重读取和调度条件。构造见 [repair-checks.json](repair-checks.json)。','',
'HF 固定归档 `cache_explanation.md` 再次全文读取（fulltext第1–130行）。新的来源摘要与第46–58、78–95行相符；来源卡明确该文件未定义 `cache_position`/`position_ids` 的参数合同。K-SYS-003第9行、K-SYS-007第13行仍保留位置、可见性、padding与缓存推进的独立关系，没有因原文缺少字段定义而删掉必要机制。','',
'关于原文 `score × mask` 示意的新增澄清也成立：scores `[2,5]` 数值乘mask `[1,0]`后为 `[2,0]`，被屏蔽项仍有 `1/(exp(2)+1)≈0.1192` 的softmax概率；合法集合或等价加性遮罩应使它为0。此说明纠正来源的读法，未修改原文，也未将当前正确的正文误判成缺陷。','',
'## 阅读范围与证据边界','',
'本轮完整复读了修改后的 **13篇 current 正文**，包括机制、公式、设定场景、provenance和延伸阅读；其余95篇与round 1逐篇SHA完全相同，继承前轮完整阅读结论。两轮合计覆盖全部108篇，不声称本轮又语义复读了未变的全部108篇。末尾逐篇表和 [review.json](review.json) 区分本轮重读与未变继承。','',
'本轮原文阅读为 S-HF-CACHE全文1–130行，以及S-MIXTRAL的TopK/专家定义93–144行。累计原文阅读范围仍是round 1所列的19项（9项完整归档文本、10项相关范围），**没有宣称完整复读全部114项来源**。原文图片和PDF版面未全部重新视觉检查。来源计数、归档哈希和既有通过报告均未代替机制审阅。','',
'与修订直接相关的MoE可行性和mask反例已重新计算。13篇中其他数值/公式未被本次provenance与API边界修订改变，完整复读后沿用前轮相应复算结果；不把这些数值当作目标硬件实测。','',
'## 历史、固定来源与时间边界','',
'本轮独立检查确认：','',
'- 13篇revision均只增加1；旧current正文按精确字节保存在对应history，旧metadata与此前catalog记录一致。',
'- 原有166个history文件未改动；其余95篇current正文未改动。',
'- `references/` 相对round 1没有任何变更；114项来源的URL、固定版本、归档身份、日期与时间资格均未改变。',
'- temporal policy只将profile ID从v2改为v2.1；来源上限仍为2026-04-30，全部114项仍是eligible。编辑/复核日期没有伪装成来源日期。',
'- VERSION的108个章节版本与SHA均匹配catalog；八类组织、schema、1,921个正文链接和完整索引通过检查。','',
'详见 [repair-checks.json](repair-checks.json) 和 [structural-audit.json](structural-audit.json)。这些机械结果限定在身份、引用和资格层；语义结论来自实际阅读与推导核查。','',
'## 实际运行时复验','',
'运行时为 `0802c3f140e0b46e26bfdd686ae55662db39c042`，远端候选知识库与本地均为 `69408e3fc62c8d34e576c2f1ce6d247488d568b9`；catalog/sources SHA精确一致。探针在本轮报告目录的独立bank副本和Records内使用真实RepositoryKnowledge API，避免恢复/锁文件触及canonical。','',
'| 检查 | 本轮结果 |','|---|---|',
'| Task知识条件 | `full` + `seed_and_learned`；五角色模型绑定 `gpt-6-luna` |',
'| 完整正文读取 | 108/108，所有返回字节与当前catalog SHA一致 |',
'| 知识使用版本记录 | 108/108精确revision声明通过 |',
'| 当前与历史读取 | 204个版本通过 |',
'| DESIGN / DIAGNOSIS / INDEX | 各覆盖108篇，分页完整读取 |',
'| 九项受限原文 | 说明卡9项均可读；fulltext/reader/manifest共27次读取全部拒绝 |',
'| 允许原文 | S-HF-CACHE完整可读 |',
'| bank audit / temporal | 通过；114来源、1,020原文成员、313,942,499字节，来源上限不变 |','',
'实际无模型读取调用在planner验证作用域进行；没有运行Luna角色进程或GPU。此验证证明接口可读、版本绑定和访问限制生效，不证明模型已理解或采用知识，也没有验证新的learned发布结果。完整输出见 [runtime-probe.json](runtime-probe.json)。','',
'## 继承的v2证据与发布记录','',
'当前VERSION/plan的待复审状态是准确的，不要求在复审前将其伪造为完成。以下字段仍属于继承的v2记录，**不能视为69408e3已经通过的当前证据**：','',
'| 当前字段 | 实际范围 | 发布时处理 |','|---|---|---|',
'| `VERSION.completed_at`、`plan.completed_at` | 继承的v2建设完成日期 | 明确标为前版建设日期，或在最终发布记录中区分当前完成事件。 |',
'| `VERSION.validation`、`plan.final_validation` → `docs/evidence/final-linux-validation.json` | 该报告输入manifest绑定v2；catalog、sources、policy、VERSION、plan五项均与当前身份不同 | 保留历史文件，并将当前验证入口绑定本轮review/probe及受审内容身份。 |',
'| `VERSION.runtime_integration` → `runtime-navigation-published.json` | 旧导航发布 `aedff22` 的证据 | 保留历史定位；当前GPU runtime知识读取由本轮 `0802c3f` probe支持。 |','',
'这属于已明确的候选到最终发布记录收口，不构成新的正文阻断。root可按既定安排单独提交round-2报告、状态及证据入口；后续校验应确认发布提交没有改变正文、catalog、sources、原始归档及其受审内容身份。若改变了机制内容，则应复审对应差异。','',
'## 对新Task的范围判断','',
'在已声明审查范围内，当前知识体系适合作为Qwen3-8B BF16、4090在线多轮、完整历史、取消隔离和独立验证的机制seed。它区分精确greedy、浮点数值关系、随机分布和近似质量；BS≤16保留为负载假设，章节设定的ITL/SLO数字不替换新Task的无固定性能阈值。固定模型制品与目标执行、实际资源峰值和最终服务表现仍需后续真实Task自己验证。','',
'Task采样质量澄清、最后发布记录及实际启动准备分别处理；本轮KB PASS不宣称这些工作已完成。本轮未修改canonical、Task、shared runtime或旧实验，未启动Agent/模型/GPU，未推送，也未从外网引入新技术事实。','',
'## 逐篇身份与阅读记录','',
'“R2完整”表示本轮在当前SHA下完整复读；“R1完整，字节未变”表示此前完整阅读且本轮验证SHA完全相同。两者的实际行范围均记录，不能把机械接口全读混称语义复读。','',
'| ID | 当前revision | SHA-256 | 阅读依据及行范围 |','|---|---|---|---|']
for e in entries:
    basis=f'R2完整：1–{e["line_count"]}' if e['changed'] else f'R1完整，字节未变：1–{e["line_count"]}'
    md.append(f'| {e["id"]} | {e["revision"]} | `{e["sha256"]}` | {basis} |')
(REPORT/'review.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print(json.dumps({'kb_verdict':data['kb_verdict'],'resolved':[r['finding_id'] for r in resolutions],
                  'reread':len(changed),'carried_forward':len(entries)-len(changed),'runtime':runtime['result'],
                  'history_versions':runtime['history_versions_read'],'reports':['review.md','review.json']},ensure_ascii=False))
