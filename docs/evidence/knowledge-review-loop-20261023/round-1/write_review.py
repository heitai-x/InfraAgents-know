import collections
import datetime
import hashlib
import json
import pathlib
import subprocess

REPORT=pathlib.Path(__file__).resolve().parent
ROOT=REPORT.parents[2]
KB=ROOT/'InfraAgents-know-astra'
def load(name): return json.loads((REPORT/name).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
entries=load('entry-inventory.json')
catalog=json.loads((KB/'knowledge/catalog.json').read_text(encoding='utf-8'))['entries']
by_id={e['id']:e for e in entries}
metadata={e['id']:e for e in catalog}
struct=load('structural-audit.json')
calc=load('calculation-checks.json')
runtime=load('runtime-probe.json')
assert all(e['read_ranges']==[[1,e['line_count']]] for e in entries)
assert not struct['issues']
assert runtime['result']=='PASS'

source_reads=[]
def source(sid,path,ranges,complete,claims,result='supported'):
    p=KB/path
    source_reads.append({'source_id':sid,'path':path,'sha256':sha(p),
       'total_lines':len(p.read_text(encoding='utf-8').splitlines()),'read_ranges':ranges,
       'complete_archived_text_read':complete,'images_or_pdf_visuals_reread':False,
       'checked_claims':claims,'assessment':result})
source('S-HF-CACHE','references/S-HF-CACHE/fulltext.txt',[[1,130]],True,
       ['逐层KV、历史与新token拼接、attention mask覆盖past+new、动态/静态/窗口缓存',
        '全文未定义cache_position；来源卡和12处provenance的具体归因超范围'], 'KR1-002')
source('S-TORCH-NUMERIC','references/S-TORCH-NUMERIC/fulltext.txt',[[1,167]],True,
       ['浮点重排、batched与slice计算不保证位级相同','BF16归约与SDPA中间精度的版本边界'])
source('S-TORCH-REPRO','references/S-TORCH-REPRO/fulltext.txt',[[1,181]],True,
       ['固定seed不保证跨平台/版本一致','确定性算法与算法选择分别约束'])
source('S-CHAT','references/S-CHAT/fulltext.txt',[[1,229]],True,
       ['对话角色/控制token属于输入函数','避免特殊token重复添加','生成提示与继续末消息分别定义'])
source('S-ASGI','references/S-ASGI/fulltext.txt',[[1,584]],True,
       ['响应开始与完成不同','send缓冲完成不是客户端消费','断开通知与发送错误存在竞态'])
source('S-NKI','references/S-NKI/fulltext.txt',[[1,369]],True,
       ['SBUF 28MiB/128 partitions、PSUM 2MiB','double FP8布局和不能组合的模式',
        'DMA与PSUM搬运不同、DGE不支持indirect gather/scatter','TODO注释不是已交付API'])
source('S-MLX-EXEC','references/S-MLX-EXEC/fulltext.txt',[[1,539]],True,
       ['compile的tracing/输入输出状态捕获','shape/type/数量变更与重编译','shapeless静态shape陷阱','operation的stream/device选择'])
source('S-MLX','references/S-MLX/fulltext.txt',[[1,146]],True,
       ['惰性求值与隐式eval','旧文档不编译措辞与专门Compilation文档的范围差异'])
source('S-MLX-MEM','references/S-MLX-MEM/fulltext.txt',[[1,80]],True,
       ['Apple Silicon共同内存池','MLX跟踪的跨stream数组依赖自动建立顺序'])
source('S-TORCH-CUDA','references/S-TORCH-CUDA/fulltext.txt',[[279,383],[1336,1479],[1651,1688],[1720,1792]],False,
       ['异步提交、显式跨stream依赖和record_stream生命周期','图地址稳定与内容更新',
        '多stream capture的依赖闭合','共享图池禁止并发重放及输出覆盖风险'])
source('S-SDPA','references/S-SDPA/upstream/torch/nn/functional.py',[[6138,6300]],False,
       ['bool true参与attention','非方形causal左上对齐','显式dropout参数','GQA整除与数值精度边界'])
source('S-FA-IMPL','references/S-FA-IMPL/upstream/README.md',[[210,270],[408,451]],False,
       ['v2.1非等长causal右下对齐','GQA映射与窗口对齐','deterministic参数针对backward','decode split与合并'])
source('S-FI','references/S-FI/upstream/flashinfer/prefill.py',[[1741,1892]],False,
       ['plan/run分页metadata和QK/VO维度','custom mask与causal优先关系',
        'plan不可capture/compile','fixed_split_size不保证CUDA Graph兼容'])
source('S-MIXTRAL','references/S-MIXTRAL/fulltext.txt',[[1,150]],False,
       ['每token选两个不同专家','TopK定义、SwiGLU和dispatch/combine','K-MODEL-003热点20项不满足16token上界'],'supports KR1-001')
source('S-ORCA','references/S-ORCA/fulltext.txt',[[650,810]],False,
       ['非attention算子按有效token聚批，attention按请求分开','请求自己的KV生命周期'])
source('S-SARATHI','references/S-SARATHI/fulltext.txt',[[750,835]],False,
       ['chunked prefill的历史mask与数学等价','更小chunk的强度/重复读取代价','decode-maximal batching属于具体策略'])
source('S-AWQ','references/S-AWQ/fulltext.txt',[[312,386]],False,
       ['v1平均激活与归一化权重幅度尺度搜索','等价缩放与clipping','并非永久保留1%高精度权重'])
source('S-SMOOTHQUANT','references/S-SMOOTHQUANT/fulltext.txt',[[487,553]],False,
       ['O1/O2/O3各自激活量化粒度','线性/BMM与轻量算子的精度分别处理'])
source('S-PAGED','references/S-PAGED/fulltext.txt',[[546,651],[926,960]],False,
       ['逻辑token块与不连续物理块','层/head分开管理的可选布局','引用计数与末块COW'])
(REPORT/'source-read-ledger.json').write_text(json.dumps(source_reads,ensure_ascii=False,indent=2),encoding='utf-8')

affected=[]
for e in entries:
    p=KB/e['path']
    for n,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
        if 'S-HF-CACHE' in line and 'cache position' in line:
            affected.append({'entry_id':e['id'],'path':e['path'],'lines':[n,n+1], 'sha256':e['sha256']})
findings=[
 {'id':'KR1-001','severity':'P2','blocks_review_pass':True,'kind':'resource_example_correctness',
  'title':'MoE热点算例超过top-2路由的单专家负载上界',
  'locations':[{'entry_id':'K-MODEL-003','path':by_id['K-MODEL-003']['path'],'lines':[13,15],'sha256':by_id['K-MODEL-003']['sha256']}],
  'evidence':'正文设B=16、k=2，继而称一个热点专家拿到20项；同章33行及S-MIXTRAL原文35–36、108–111、133–135行采用每token选择两个不同专家。因此每个专家最多收到B=16项。总dispatch项32并不允许单专家重复收到同一token。',
  'impact':'该设定无法构造合法路由轨迹，会误导偏斜/慢专家容量或测试输入；其余MoE公式与并集算例成立。Qwen3-8B稠密Task不会直接采用该MoE算例。',
  'recommendation':'保留B=16、k=2、E=8，将热点20项改为合法数（如16项），明确其余16项分配到其他专家；同时保留sum_e n_e=B*k与0<=n_e<=B。按既有修订流程更新当前正文、catalog SHA和旧版本历史。',
  'verification':'构造16个token的distinct top-2映射，核对总32项与单专家上界16；复算该段。'},
 {'id':'KR1-002','severity':'P2','blocks_review_pass':True,'kind':'source_attribution',
  'title':'S-HF-CACHE固定原文没有来源说明所称的cache position定义',
  'locations':[{'path':'knowledge/sources/S-HF-CACHE.md','lines':[7,15],'sha256':sha(KB/'knowledge/sources/S-HF-CACHE.md')},
               {'path':'knowledge/sources.json','fields':['S-HF-CACHE.coverage','S-HF-CACHE.local_summary']}]+affected,
  'evidence':'完整复读references/S-HF-CACHE/fulltext.txt 1–130行及对应固定756dffe归档；它说明历史/新KV、mask的past+new长度和Dynamic/Static/StaticSlidingWindow存储，全文没有cache_position或cache position定义。来源卡11行及12篇provenance却明确归因于该文档。',
  'impact':'运行角色追到固定原文后无法验证这项具体接口归因。逻辑位置、position_ids、物理槽与可见性区分的机制推导本身没有被否定；不能据此删除允许的非代码机制正文。',
  'recommendation':'保留固定原文，将来源卡、registry摘要及上述12处provenance限缩到实际支持的KV/mask/存储机制；保留逻辑位置与物理槽推导，并将cache_position等API字段说明明确为实现相关、按Task固定实现核对。若要保留“该来源定义cache_position”则需另有截止前固定原文依据。旧历史不回写。',
  'verification':'逐一核对12处当前provenance和来源卡/registry，源原文SHA不变；结构审计和实际受限视图读取仍通过。',
  'source_reading_note':'原文softmax(score × mask)示意不能直接作为0/1布尔屏蔽算法。当前正文的合法集合/加性mask关系正确；可在该来源适用边界澄清此示意，不新增正文错误。'},
 {'id':'KR1-003','severity':'P3','blocks_review_pass':False,'kind':'documentation_consistency',
  'title':'knowledge/README仍停留在94篇的building状态',
  'locations':[{'path':'knowledge/README.md','lines':[1,5],'sha256':sha(KB/'knowledge/README.md')}],
  'evidence':'第3行称base-gpt-6-astra-2026-04-30-v2-building、已接入21篇新增、共94篇、仍在扩充；实际catalog为108篇/35新增，根README和plan为complete。',
  'impact':'独立仓库读者可能误判交付范围。新0802c3f运行时生成的选定README已使用实际目录计数，未复现此过时数字。',
  'recommendation':'更新这段为当前v2、108篇与35新增，继续保留seed不是目标Task实测的边界。',
  'verification':'与catalog、根README、VERSION及expansion-plan计数一致。'}]

all_affected={loc.get('entry_id') for f in findings for loc in f['locations'] if loc.get('entry_id')}
for e in entries:
    e['title']=metadata[e['id']]['title']
    e['review_scope']='完整机制正文，含公式、设定场景、适用边界、来源说明与延伸阅读；不是只读标题/摘要。'
    e['findings']=[f['id'] for f in findings if any(loc.get('entry_id')==e['id'] for loc in f['locations'])]
    e['assessment']='revision_required' if e['findings'] else 'no_material_issue_found_in_this_review'
    e['independent_calculation_checks']=sum(r['entry_id']==e['id'] for r in calc['checks'])
    e['sources_independently_cross_checked']=[s['source_id'] for s in source_reads if s['source_id'] in e['source_ids']]

task_path=ROOT/'phase0-tools/tasks/gpu-engine-qwen3-8b-rtx4090-continuous.toml'
data={
 'schema_version':1,'review_round':1,'reviewer':'independent sub-agent /root/knowledge_release_review',
 'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'verdict':'FAIL','approval_to_launch_gpu_task':False,
 'decision':'修复KR1-001与KR1-002后对实际新SHA复审；KR1-003建议同批收口。没有确认到P0/P1级正文缺陷，但已确认错误不应在本轮签发通过。',
 'severity_definition':{'P0':'无法维持基本正确性/边界的紧急阻断','P1':'会实质误导当前GPUTask或核心设计的重要缺陷','P2':'确定的局部技术或来源归因错误，需要修订后复审','P3':'文档一致性或非阻断改进'},
 'baseline':{'repository':str(KB),'branch':'gpt-6-astra-2026-04-30',
             'head':'0a5a0c0636101d0fbce7c3734636fd50be70e93a',
             'catalog_sha256':sha(KB/'knowledge/catalog.json'),'sources_sha256':sha(KB/'knowledge/sources.json'),
             'source_cutoff_inclusive':'2026-04-30','entry_count':108,'source_count':114},
 'scope':{'design':'docs/inference-coverage.md','plan':'docs/expansion-plan.json','domains':24,'scenario_families':12,
          'non_generative_subcontracts':5,'categories':struct['categories'],'complete_body_reads':len(entries),
          'body_characters':sum(e['chars'] for e in entries),
          'source_texts_fully_reread':sum(s['complete_archived_text_read'] for s in source_reads),
          'source_texts_partially_reread':sum(not s['complete_archived_text_read'] for s in source_reads),
          'all_114_sources_fully_reread':False,
          'source_scope_note':'本轮19项来源有独立原文阅读：9项完整归档文本、10项与具体论断有关的原文范围。其余95项未逐项复读全文；所有114项的版本/本地来源/归档完整性作机械核对。未把既有通过报告当成实质审查依据，未把原文图片/PDF版面称作已视觉复审。'},
 'findings':findings,
 'task_readiness':{'task_path':str(task_path),'task_sha256_at_probe':runtime['task_sha256'],
                  'current_task_sha256':sha(task_path),'runtime_probe':'PASS',
                  'semantic_contract':'PENDING_USER_CLARIFICATION',
                  'unresolved':'workload要求多种采样方式，quality要求每轮精确greedy token；root已向用户询问适用范围，本审查不擅自选择。该问题不计为知识库正文缺陷。',
                  'supported_design_path':['K-MODEL-001','K-FW-011','K-ALG-016','K-FW-002','K-SYS-007','K-WORK-010','K-ALG-001','K-ALG-007','K-ALG-002','K-ALG-003','K-SYS-002','K-SYS-004','K-FW-003','K-FW-005','K-WORK-001','K-WORK-002','K-WORK-004','K-TOOLS-006','K-TOOLS-007'],
                  'scope_judgment':'修复后的机制体系适合作为Qwen3-8B/4090在线多轮设计与实验的seed依据；不提供该checkpoint参数或精确greedy实现已正确、性能已合格的证明。BS<=16仍应是负载假设，示例ITL/SLO数字不覆盖Task，近似方案须受allow_approximation=false约束。'},
 'runtime':{'head':runtime['runtime_head'],'remote_kb_head':runtime['remote_kb_head'],
            'same_catalog_sources_as_local':all(sha(KB/k)==v for k,v in runtime['canonical_hashes_before'].items()),
            'full_chapters_read':len(runtime['chapter_reads']),'knowledge_uses_validated':runtime['validated_use_count'],
            'history_versions_read':runtime['history_versions_read'],'restricted_sources_denied':len(runtime['restricted_originals_checks']),
            'navigation_entry_counts':{k:v['entry_count'] for k,v in runtime['navigation'].items()},
            'selection':runtime['index']['selection'],'evidence':'runtime-probe.json',
            'probe_harness_correction':'首次探针将read_bytes解码结果与Path.read_text的换行归一化结果相比，遇K-FW-011的CRLF产生断言；修正探针为精确字节解码比较后全108通过。没有修改KB或运行时。',
            'limit':'实际RepositoryKnowledge接口及read_knowledge工具的直接委派已检查；没有运行真实Luna角色、模型或GPU，不据此声称模型已经理解或有效使用知识。'},
 'structural':{'result':'PASS','evidence':'structural-audit.json','body_links':struct['mechanism_body_links_checked'],
               'index_covers_all_entries':struct['navigation']['knowledge/INDEX.md']['linked_entry_count']==108,
               'connected_components':[len(c) for c in struct['connected_components']],
               'zero_incoming_from_mechanism_chapters':struct['zero_incoming_entries'],
               'isolation_judgment':'3项无其他机制正文/related入边，但有完整索引和分类导航，且新运行时设计/诊断直接列出全部108篇，不构成不可达孤立条目。'},
 'calculation_validation':{'checks':calc['count'],'passed':calc['passed'],'failed':len(calc['failed']),'evidence':'calculation-checks.json','interpretation':'设定算例的量纲/算术/可行性核对，不是实测性能。'},
 'source_location_validation':{'symbol_anchors':27,'passed':27,'evidence':'engine-symbol-checks.json','interpretation':'只证明固定源码符号坐标，不冒充完整引擎实现复审。'},
 'source_reads':source_reads,'entries':entries,
 'non_actions':['未修改canonical KB','未修改旧Task或旧实验','未修改shared runtime','未启动Agent/模型/GPUTask','未推送或发布','未联网引入新技术事实'],
 'report_artifacts':['review.json','review.md','entry-inventory.json','source-read-ledger.json','structural-audit.json','calculation-checks.json','engine-symbol-checks.json','runtime-probe.json'],
 'next_review_requirements':['提供修复后实际KB SHA','复读MoE受影响段及12篇HF provenance/来源卡/registry变化','确认原归档与既有历史未回写','重新核对catalog/hash/history和0802c3f真实受限视图读取','Task采样边界由用户澄清后才能独立结束launch readiness门槛']}
(REPORT/'review.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

lines=[
'# 知识库独立审查：round 1', '',
'**结论：FAIL，本轮不签发 GPU Task 启动通过。** 已完整阅读当前 108 篇机制正文，确认两项 P2 修订问题和一项 P3 文档一致性问题。没有确认到会直接破坏新 Qwen3-8B 稠密 GPU Task 的 P0/P1 正文缺陷；但两项确定错误应修复后对实际新 SHA 复审。新 Task 的采样质量边界仍由用户澄清，独立于知识库缺陷。', '',
'审查基线：`InfraAgents-know-astra`，分支 `gpt-6-astra-2026-04-30`，HEAD `0a5a0c0636101d0fbce7c3734636fd50be70e93a`。范围依据 `docs/inference-coverage.md` 与 `docs/expansion-plan.json`，固定来源版本上限为 2026-04-30（含）。本轮由新的独立审查者完成，不是此前内容作者。', '',
'## 启动门槛', '',
'用户要求审查通过后才启动真实 GPU Task，因此本轮 FAIL 意味着暂不启动。除此之外，Task 同时写了“需要提供多种采样方式”和“每轮输出精确 greedy token”，随机采样与所有输出都按 greedy 检查的适用范围尚未一致。root 已向用户询问，本审查没有修改 Task 或代替用户选择。', '',
'实际知识接入检查已通过：候选运行时 `0802c3f140e0b46e26bfdd686ae55662db39c042`，基于已验证 GPU 版本加导航补丁。它在报告目录的独立 bank 副本上，用五角色 `gpt-6-luna`、`full`、`seed_and_learned` 和九项原文限制完整读取 108 篇、验证 108 份知识使用记录和 191 个当前/历史版本。没有启动 Agent、模型或 GPU 作业。详见 [runtime-probe.json](runtime-probe.json)。', '',
'## 必须修订的问题', '',
'### KR1-001 · P2：MoE 热点算例违反单专家负载上界', '',
'位置：`knowledge/models/moe-routing.md`（K-MODEL-003）第 13–15 行，具体错误在第 15 行。正文设定 B=16、k=2、E=8，却称热点专家收到 20 项。同章第 33 行及 S-MIXTRAL 固定原文第 35–36、108–111、133–135 行采用每 token 选择两个不同专家。每个 token 对同一个专家最多贡献一项，因此任何专家都必须满足 `n_e ≤ 16`；总分派项 32 不放宽这个上界。', '',
'这会使偏斜/慢专家算例无法构造合法输入，但不否定该章其余路由、并集与容量推导。建议将热点项数改为 16 或其他合法值，同时保留总 32 项和其余专家的分配解释。复核应构造 distinct top-2 映射并检查总量与单专家上界。它不直接影响当前稠密 Qwen3-8B Task，仍是应收口的确定技术错误。', '',
'### KR1-002 · P2：HF 缓存来源归因超出固定原文', '',
'位置：`knowledge/sources/S-HF-CACHE.md` 第 11 行、`knowledge/sources.json` 的 `coverage`/`local_summary`，以及下列 12 篇的 provenance。完整复读 `references/S-HF-CACHE/fulltext.txt` 第 1–130 行（固定提交 `756dffe0d664419d2f620e131ed426373cfd6f12`），原文有逐层历史/新 KV、mask 的 past+new 长度、Dynamic/Static/StaticSlidingWindow 存储，全文没有 `cache_position` 或 `cache position` 定义。当前来源说明却称该文档区分 cache position。', '',
'| 条目 | 当前正文 | 受影响 provenance 行 |','|---|---|---|']
for a in affected:
    lines.append(f'| {a["entry_id"]} | `{a["path"]}` | {a["lines"][0]}–{a["lines"][1]} |')
lines += ['',
'建议保留固定原文，将来源卡、registry 摘要和这 12 处归因限缩到原文实际内容；保留逻辑位置、position_ids、物理槽与可见性区分的独立推导，明确具体 API 字段必须按 Task 固定实现核对。若坚持保留“此来源定义 cache_position”，则需要另一个截止前、实际包含该定义的固定依据。旧历史和原始归档不应回写。该问题不构成删除允许的非代码机制正文的理由。', '',
'原文另有 `softmax(score × mask)` 示意，不能直接解释为 0/1 数值乘法屏蔽：被乘为 0 的 score 经 softmax 仍可能得到正概率。当前 K-ALG-004、K-SYS-003、K-FW-006 的合法集合/加性 mask 解释正确。可在该来源适用边界澄清示意的读法，不将其扩成新的正文缺陷。', '',
'### KR1-003 · P3：独立入口 README 的交付状态过时', '',
'`knowledge/README.md` 第 3 行仍称 `v2-building`、新增 21 篇、共 94 篇、仍在扩充；实际目录是 108 篇、35 新增，根 README 与 plan 已完成。建议同步这段计数与状态。新运行时生成的选定 README 已按实际目录生成，此过时数字没有进入本轮接口探针。', '',
'## 实质阅读与验证结果', '',
'完整阅读覆盖 108 篇、约 40.7 万字符，包括机制、状态、公式、设定场景、来源说明及延伸阅读。逐篇 ID、SHA-256、实际行范围在本报告末尾和 [review.json](review.json)；不是只读标题、索引或已有审查结论。', '',
'独立复算了 105 个算术或可行性检查，104 通过，唯一失败为 KR1-001。范围包括 KV/共享容量、MLA 复制、LoRA 和图池、CPU/GPU 带宽下界、分块预算、传输与流水、统计区间、扩散/音视频和非生成推理资源。数值均按正文的设定复算，没有将设定当实测，见 [calculation-checks.json](calculation-checks.json)。', '',
'跨章节组合检查中，输入/输出/KV 一步偏移、前缀依赖闭包、异构状态共同恢复位置、取消后的多类引用、旧 DMA 与代次保护、图地址与内容生命周期、greedy/分布/近似质量、开放到达与会话反馈均能相互衔接。未发现与新 Task 直接相关的机制互相矛盾。量化、KV 裁剪、非前缀融合和激活稀疏均有质量前提，不能作为 `allow_approximation=false` 下的默认合法路线。', '',
'结构检查验证八类目录、schema、当前正文 SHA、旧修订连续性与 SHA、114 项来源卡/归档 manifest 和全部已登记成员，以及 1,921 个正文链接。实际运行时 audit 报告 1,020 个原文成员、313,942,499 字节，通过结果仅用于结构身份，不代替语义审查。五篇引擎索引的 27 个符号锚点与固定源码 AST 行号一致；这一项不声称完整引擎源码已经复读。', '',
'完整索引覆盖 108/108 篇；关联图在忽略方向后为单一连通分量。K-ENGINE-004、K-ENGINE-005、K-TOOLS-004 没有其他机制正文/related 入边，但均有完整索引和分类导航，新运行时 DESIGN/DIAGNOSIS 也各直接列出 108 条，故不能据此判成不可达孤立条目。', '',
'## 来源原文的真实核查范围', '',
'**本轮没有完整复读全部 114 项来源。** 独立原文阅读覆盖 19 项：9 项完整归档文本、10 项与具体论断有关的原文范围；其余 95 项未逐项复读全文。原文图片和 PDF 版面也没有全部视觉重审。完整文本/局部范围如下，SHA 见 [source-read-ledger.json](source-read-ledger.json)。没有从外网补入截止日后的技术事实，也没有把原有“审查通过”报告用作本文结论的依据。', '',
'| 来源 | 实际阅读文件 | 行范围 | 范围性质 |','|---|---|---|---|']
for s in source_reads:
    span=', '.join(str(a)+'–'+str(b) for a,b in s['read_ranges'])
    lines.append(f'| {s["source_id"]} | `{s["path"]}` | {span} | {"完整归档文本" if s["complete_archived_text_read"] else "具体论断相关范围"} |')
lines += ['',
'核对结果支持：SDPA/FlashAttention 的非方形 causal 对齐差异、FlashInfer plan 不可捕获及 split 图限制、PyTorch stream 生命周期与图池条件、SARATHI 的分块可见性、PagedAttention 的引用/COW、AWQ v1 的尺度空间、SmoothQuant 的 O1/O2/O3 区别、Trainium2 的 SBUF/PSUM 与 DGE 边界，以及 MLX 编译与旧惰性文档的区分。HF 的具体接口归因为本轮发现的例外。', '',
'## 对新 GPU 在线多轮 Task 的适配判断', '',
'修复上述内容后，知识库适合作为该 Task 的机制与实验设计 seed。它已经提供以下可连接的路径，但实际引擎仍须用固定 checkpoint、目标环境和独立检查建立证据。', '',
'| Task 需要 | 已有知识路径 | 本轮判断 |','|---|---|---|',
'| Qwen3-8B 固定制品、BF16 与精确输出 | K-MODEL-001、K-FW-011、K-ALG-016、K-FW-002、K-TOOLS-006 | 能还原函数/状态并区分浮点重排与 token 一致；不冒充该 checkpoint 参数或数值实现已核验。 |',
'| 完整历史、工具依赖、thinking 开关 | K-SYS-007、K-WORK-010、K-SYS-002、K-ALG-003 | 规范化输入、模板、共同前缀、末 token 未计算状态与回合身份已有机制；具体 thinking 模板应取 Task 制品。 |',
'| 取消终态与其他会话隔离 | K-SYS-004、K-FW-003、K-ALG-002、K-ALG-008、K-TOOLS-006 | 逻辑引用、执行引用、输出引用和旧 DMA 保护分开；有共享前缀与槽重用的组合检查。 |',
'| 在线交错、突发与 BS≤16 假设 | K-WORK-002、K-WORK-010、K-ALG-001、K-ALG-007、K-FW-005 | 请求到达、在途/排队与执行 batch 分开推导；不把例子批量当能力上限。 |',
'| p95 session completion、没有固定性能门槛 | K-WORK-001、K-WORK-004、K-WORK-010、K-TOOLS-007 | 会话/请求指标、开放到达、删失/失败和独立重复可连接；示例 ITL 数值不是新 Task 阈值。 |',
'| 4090 资源和受限 profiler | K-HW-001、K-HW-005、K-SYS-019、K-TOOLS-004 | 按实际能力和资源口径校准，允许计数器不可用时保留较窄结论，不将 Hopper/Trainium 能力套到 4090。 |', '',
'真实读取探针中，远端不可变 v2 的 `fdce576c5972c25b91ad81ba887c616d717ea3ca` 与本地 receipt 提交的 catalog/sources SHA 完全相同。九项受限来源的说明卡可读，`fulltext.txt`、`reader.html`、`manifest.json` 均被实际读接口拒绝；允许的 HF 原文可完整读取。导航分页能完整推进。词法检索的中文多轮、精确数值、图、CPU quota 等查询能找到相关条目；英文 `cancel` 与 `cancellation` 不做同义词保证，检索排名不代表适用性，仍需沿完整导航及相关链接读正文。', '',
'`full` 视图没有模型时间 profile 过滤，五角色 Luna 不会被 Astra profile 身份规则错误阻断。静态 seed 的 2026-04-30 来源边界与运行期 learned 的真实观察日期应分别保持。接口可读和知识使用记录有效，不证明实际 Luna 已理解、采用或受益；本轮没有进行这种模型实验。', '',
'## 修订与复审要求', '',
'root 修复 KR1-001、KR1-002，并建议同批修正 KR1-003；按既有 catalog/revision/history 方式发布新的内容身份，保留原始归档与旧历史。复审应绑定实际新 SHA，重读受影响正文和来源归因，核对数值上界、来源文本范围与实际受限视图接口。Task 的采样边界需要独立明确。完成这两类门槛之前不启动真实 GPU Task。', '',
'本轮只在指定报告目录生成审查产物与无 Agent 的临时 bank/Records；没有修改 canonical KB、shared runtime、旧 Task/实验，没有启动模型/实验，也没有推送。首次接口探针的 CRLF 与 `Path.read_text` 换行归一化比较产生了一次检查脚本断言；修正为精确字节解码比较后全部通过，未将此记为 KB 或 runtime 缺陷。', '',
'## 逐篇实际阅读记录', '',
'以下范围均为当前审查 SHA 下的完整文件行范围，包括 provenance 与延伸阅读。标记“未发现实质问题”限定于本轮阅读和已声明原文核查范围，不等于每项来源与所有实现路径都已证明。', '',
'| ID | 正文 | 实际行范围 | SHA-256 | 本轮结果 |','|---|---|---|---|---|']
for e in entries:
    result=', '.join(e['findings']) or '未发现实质问题'
    lines.append(f'| {e["id"]} | `{e["path"]}` | 1–{e["line_count"]} | `{e["sha256"]}` | {result} |')
(REPORT/'review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'verdict':data['verdict'],'entries':len(entries),'findings':[(f['id'],f['severity']) for f in findings],
                  'source_reads':len(source_reads),'reports':{'review.json':(REPORT/'review.json').stat().st_size,'review.md':(REPORT/'review.md').stat().st_size}},ensure_ascii=False))
