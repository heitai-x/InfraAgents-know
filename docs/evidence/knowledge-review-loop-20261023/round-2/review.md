# 知识库独立复审：round 2

**知识库结论：PASS。** 对实际提交 `69408e3fc62c8d34e576c2f1ce6d247488d568b9` 的本轮复审通过。KR1-001、KR1-002、KR1-003 均已解决，修订没有发现新的机制、资源算例或来源范围问题。该结论承接 round 1 的完整正文审查，并以本轮实际新内容、历史/原文身份及运行时检查为依据。

**Task 采样契约仍独立 pending，不改变上述 KB PASS。** 用户尚需明确“多种采样方式”和“每轮精确 greedy token”的适用范围；知识库通过不等于 Task 契约已解决，也不等于真实 GPU 引擎已正确或性能达成。本审查没有启动 Agent、模型或 GPU。

## 修订核查

| 前轮问题 | 结论 | 本轮实际证据 |
|---|---|---|
| KR1-001 · MoE 偏斜算例 | 已解决 | K-MODEL-003 第15行改为 `[16,4,3,3,2,2,1,1]`，总量32、最大16，并明确B为本轮token数及distinct top-2。 |
| KR1-002 · HF 来源归因 | 已解决 | 来源卡、registry摘要及全部12处provenance只归因于固定文档实际说明的KV、mask与存储形态；SYS003/007保留机制推导并限定API合同。 |
| KR1-003 · README过时 | 已解决 | 根与knowledge README均明确v2.1复审候选；73+35=108篇、114来源与实际目录一致。 |

MoE 复算不止检查 `16+4+3+3+2+2+1+1=32`：已构造16条token路由，每条均选择专家0及另一个非零专家；其他专家分别被选4、3、3、2、2、1、1次。每条路由有两个不同专家，所得计数与正文完全一致。正文也不再将热点项数直接等同全部关键路径，而保留专家放置、权重读取和调度条件。构造见 [repair-checks.json](repair-checks.json)。

HF 固定归档 `cache_explanation.md` 再次全文读取（fulltext第1–130行）。新的来源摘要与第46–58、78–95行相符；来源卡明确该文件未定义 `cache_position`/`position_ids` 的参数合同。K-SYS-003第9行、K-SYS-007第13行仍保留位置、可见性、padding与缓存推进的独立关系，没有因原文缺少字段定义而删掉必要机制。

关于原文 `score × mask` 示意的新增澄清也成立：scores `[2,5]` 数值乘mask `[1,0]`后为 `[2,0]`，被屏蔽项仍有 `1/(exp(2)+1)≈0.1192` 的softmax概率；合法集合或等价加性遮罩应使它为0。此说明纠正来源的读法，未修改原文，也未将当前正确的正文误判成缺陷。

## 阅读范围与证据边界

本轮完整复读了修改后的 **13篇 current 正文**，包括机制、公式、设定场景、provenance和延伸阅读；其余95篇与round 1逐篇SHA完全相同，继承前轮完整阅读结论。两轮合计覆盖全部108篇，不声称本轮又语义复读了未变的全部108篇。末尾逐篇表和 [review.json](review.json) 区分本轮重读与未变继承。

本轮原文阅读为 S-HF-CACHE全文1–130行，以及S-MIXTRAL的TopK/专家定义93–144行。累计原文阅读范围仍是round 1所列的19项（9项完整归档文本、10项相关范围），**没有宣称完整复读全部114项来源**。原文图片和PDF版面未全部重新视觉检查。来源计数、归档哈希和既有通过报告均未代替机制审阅。

与修订直接相关的MoE可行性和mask反例已重新计算。13篇中其他数值/公式未被本次provenance与API边界修订改变，完整复读后沿用前轮相应复算结果；不把这些数值当作目标硬件实测。

## 历史、固定来源与时间边界

本轮独立检查确认：

- 13篇revision均只增加1；旧current正文按精确字节保存在对应history，旧metadata与此前catalog记录一致。
- 原有166个history文件未改动；其余95篇current正文未改动。
- `references/` 相对round 1没有任何变更；114项来源的URL、固定版本、归档身份、日期与时间资格均未改变。
- temporal policy只将profile ID从v2改为v2.1；来源上限仍为2026-04-30，全部114项仍是eligible。编辑/复核日期没有伪装成来源日期。
- VERSION的108个章节版本与SHA均匹配catalog；八类组织、schema、1,921个正文链接和完整索引通过检查。

详见 [repair-checks.json](repair-checks.json) 和 [structural-audit.json](structural-audit.json)。这些机械结果限定在身份、引用和资格层；语义结论来自实际阅读与推导核查。

## 实际运行时复验

运行时为 `0802c3f140e0b46e26bfdd686ae55662db39c042`，远端候选知识库与本地均为 `69408e3fc62c8d34e576c2f1ce6d247488d568b9`；catalog/sources SHA精确一致。探针在本轮报告目录的独立bank副本和Records内使用真实RepositoryKnowledge API，避免恢复/锁文件触及canonical。

| 检查 | 本轮结果 |
|---|---|
| Task知识条件 | `full` + `seed_and_learned`；五角色模型绑定 `gpt-6-luna` |
| 完整正文读取 | 108/108，所有返回字节与当前catalog SHA一致 |
| 知识使用版本记录 | 108/108精确revision声明通过 |
| 当前与历史读取 | 204个版本通过 |
| DESIGN / DIAGNOSIS / INDEX | 各覆盖108篇，分页完整读取 |
| 九项受限原文 | 说明卡9项均可读；fulltext/reader/manifest共27次读取全部拒绝 |
| 允许原文 | S-HF-CACHE完整可读 |
| bank audit / temporal | 通过；114来源、1,020原文成员、313,942,499字节，来源上限不变 |

实际无模型读取调用在planner验证作用域进行；没有运行Luna角色进程或GPU。此验证证明接口可读、版本绑定和访问限制生效，不证明模型已理解或采用知识，也没有验证新的learned发布结果。完整输出见 [runtime-probe.json](runtime-probe.json)。

## 继承的v2证据与发布记录

当前VERSION/plan的待复审状态是准确的，不要求在复审前将其伪造为完成。以下字段仍属于继承的v2记录，**不能视为69408e3已经通过的当前证据**：

| 当前字段 | 实际范围 | 发布时处理 |
|---|---|---|
| `VERSION.completed_at`、`plan.completed_at` | 继承的v2建设完成日期 | 明确标为前版建设日期，或在最终发布记录中区分当前完成事件。 |
| `VERSION.validation`、`plan.final_validation` → `docs/evidence/final-linux-validation.json` | 该报告输入manifest绑定v2；catalog、sources、policy、VERSION、plan五项均与当前身份不同 | 保留历史文件，并将当前验证入口绑定本轮review/probe及受审内容身份。 |
| `VERSION.runtime_integration` → `runtime-navigation-published.json` | 旧导航发布 `aedff22` 的证据 | 保留历史定位；当前GPU runtime知识读取由本轮 `0802c3f` probe支持。 |

这属于已明确的候选到最终发布记录收口，不构成新的正文阻断。root可按既定安排单独提交round-2报告、状态及证据入口；后续校验应确认发布提交没有改变正文、catalog、sources、原始归档及其受审内容身份。若改变了机制内容，则应复审对应差异。

## 对新Task的范围判断

在已声明审查范围内，当前知识体系适合作为Qwen3-8B BF16、4090在线多轮、完整历史、取消隔离和独立验证的机制seed。它区分精确greedy、浮点数值关系、随机分布和近似质量；BS≤16保留为负载假设，章节设定的ITL/SLO数字不替换新Task的无固定性能阈值。固定模型制品与目标执行、实际资源峰值和最终服务表现仍需后续真实Task自己验证。

Task采样质量澄清、最后发布记录及实际启动准备分别处理；本轮KB PASS不宣称这些工作已完成。本轮未修改canonical、Task、shared runtime或旧实验，未启动Agent/模型/GPU，未推送，也未从外网引入新技术事实。

## 逐篇身份与阅读记录

“R2完整”表示本轮在当前SHA下完整复读；“R1完整，字节未变”表示此前完整阅读且本轮验证SHA完全相同。两者的实际行范围均记录，不能把机械接口全读混称语义复读。

| ID | 当前revision | SHA-256 | 阅读依据及行范围 |
|---|---|---|---|
| K-ALG-001 | 2 | `156a7dd70b2126dd00afe8fcde002c96ded6c6eb69483c94071f559fecbc7e99` | R1完整，字节未变：1–83 |
| K-ALG-002 | 2 | `a5e4216601b16b1c5f0a566eb4a2b663dafe9ffd21add75a23dd2ba755c8625f` | R1完整，字节未变：1–79 |
| K-ALG-003 | 3 | `fe888e513cd79ce5be2e5c4c8fc43d019b33bc9015e5d6358ddd79d101264e03` | R2完整：1–74 |
| K-ALG-004 | 2 | `a5e851ffddc0af450d425f6e29f3fbe667d245da61d65eefc81e12b563f35978` | R1完整，字节未变：1–65 |
| K-ALG-005 | 2 | `1ff1082b95610754539212473ba3a562ed9725713e5b882d264a56c79d713403` | R1完整，字节未变：1–101 |
| K-ALG-006 | 3 | `f4e638885bb276ea3e2db2a885c249975395b6a986b2db78c449e63fe6d5ac33` | R2完整：1–66 |
| K-ALG-007 | 3 | `942222412eb991203d528afc591b850f0a7f5e625120fcb1a7e69bf9ad909945` | R2完整：1–58 |
| K-ALG-008 | 2 | `9f3e037e2725193a3aefda22db0d7461c91ee737b40018a51e8e2f5afbe9fa5d` | R1完整，字节未变：1–75 |
| K-ALG-009 | 2 | `c9d7ef861fd1243cb4a605a38405fe1c7ddb2cfc24a653a97a83080ad065d02b` | R1完整，字节未变：1–69 |
| K-ALG-010 | 2 | `aa8313c1df241ae8dabe2bc7a5f57e7189bc5e1260e232eb30835d399f1499e3` | R1完整，字节未变：1–71 |
| K-ALG-011 | 2 | `a75584a814f07f661f8dc61148529e8e08d47fa05ff68ffa8d6b31562316e24a` | R1完整，字节未变：1–56 |
| K-ALG-012 | 2 | `cfa738aa99700ff90b96b3044c134897867e93792289fe62b21d611855139d0e` | R1完整，字节未变：1–54 |
| K-ALG-013 | 2 | `07d8beba67dd10960cd504dafad8cdf0baaf92c03ef368a2a97236690c6bc7e8` | R1完整，字节未变：1–78 |
| K-ALG-014 | 2 | `3cba63cc43866f40cca61ad911f32115e9beaa5c4adff9559b4deb592fd3d4d7` | R1完整，字节未变：1–73 |
| K-ALG-015 | 2 | `161a883561e39371ea22a1430c385546c402db2f993d8a37c6f37e03e1e4bd17` | R1完整，字节未变：1–103 |
| K-ENGINE-001 | 2 | `723d1f1b1bc65ca6fca28c83e9930a08d643469e7dec866b96a6d7de2f6afd95` | R1完整，字节未变：1–60 |
| K-ENGINE-002 | 2 | `60972ada3e249ab9ed5ee1dd1f38adae9a8a6219f3bfe53407bf93f7ec12027a` | R1完整，字节未变：1–63 |
| K-ENGINE-003 | 2 | `41c03c35b3489f1900311f13e943d955372e08b1286f02aaedab710540e91d1f` | R1完整，字节未变：1–57 |
| K-ENGINE-004 | 2 | `bbb648ee10b9ae8581809ea14fd0bd11396400c7c0a30d23443554ad440fa298` | R1完整，字节未变：1–59 |
| K-ENGINE-005 | 2 | `ec40e61a590a31703236965be35a0893af553c79f3f75657e5aa64ab2f899b8b` | R1完整，字节未变：1–58 |
| K-FW-001 | 2 | `683d5b7f5323ab43af402fbe4719539ab4ad0d20d290db019d041070e8887fff` | R1完整，字节未变：1–72 |
| K-FW-002 | 2 | `a70beff4c73683f897d9c0c1fb291539051e20b2c31adfd90a3f6677c411596f` | R1完整，字节未变：1–75 |
| K-FW-003 | 2 | `64427e69f6eb991d4029242e24e4edf56a331d233fdeab3c50f53f0f84a4df7b` | R1完整，字节未变：1–54 |
| K-FW-004 | 2 | `50c53e4310e29020b5897f9c069420f4933dd792a50d452e9ff2a3aa827d210c` | R1完整，字节未变：1–88 |
| K-FW-005 | 2 | `34232a876bf3cc6904ad796423659dc8b4bc250532e6d2213e0d0f5d1fd88e6c` | R1完整，字节未变：1–93 |
| K-FW-006 | 2 | `01009cca4251544f55e6478d8bb99f2af1aeb3a043d027ff3279692215917bac` | R1完整，字节未变：1–93 |
| K-FW-007 | 2 | `35f35dde0def4be2bc2e14e4c0a476d5b38e535d3a0b2284643fb52f16f1fdba` | R1完整，字节未变：1–80 |
| K-FW-008 | 2 | `c308a8ac651460ede6b6e53d3413bea4c4a7707b757516faf8a752b35872b6c1` | R1完整，字节未变：1–66 |
| K-FW-009 | 2 | `0a7be25e54019cb53d80e180594d1eb633cfdd9e9a058cb340a2c07dee7f3f0e` | R1完整，字节未变：1–63 |
| K-FW-010 | 2 | `abc8dbd0be6a05a8b52ed4686ab2d0e27f893c6e182656c36c514306c0799a16` | R1完整，字节未变：1–59 |
| K-HW-001 | 2 | `58e9e2f9292d24457d7281a22e646242ba607096f204ba7a30feea621353f3e3` | R1完整，字节未变：1–76 |
| K-HW-002 | 2 | `40d7b15c7a9b4148b95f44c2bdba861884112b59bb8c6fab44b96cf8e119c7ed` | R1完整，字节未变：1–64 |
| K-HW-003 | 2 | `5bfc321d4349e141d1787a360ab72106bc70af99af17d9e53bdd1b75e2a0980a` | R1完整，字节未变：1–67 |
| K-HW-004 | 2 | `7185979b8429b7d44fd34ec0910bbe85c617a457cc4d9d9055920e73d07bc5d3` | R1完整，字节未变：1–88 |
| K-HW-005 | 2 | `aef68ee4706e1ca36c7d2655ef8edf3535e9893a29d54947407beddc8465f2ed` | R1完整，字节未变：1–67 |
| K-HW-006 | 2 | `08c875daee94b818579b1f63e07543c3d8628177d61f3141a6f39727cc0fbf8a` | R1完整，字节未变：1–52 |
| K-HW-007 | 2 | `b4a1f38557eecf13e85f3a962f617c95ec86ef7c800a664471d6b1d2c3d31fd4` | R1完整，字节未变：1–49 |
| K-HW-008 | 2 | `0773594917538d150957f945b6c9268e1aca64bb83e041aec2315c1cb0984019` | R1完整，字节未变：1–55 |
| K-HW-009 | 2 | `2f48de03875b90417b29b039736a4b9532d5b8a8681dc6c28d7707b9271084e6` | R1完整，字节未变：1–78 |
| K-MODEL-001 | 4 | `f9dad0cf98d2372d9540ea04e47ebdec740337b3d7c77f7a77d6bb3895f74af4` | R2完整：1–104 |
| K-MODEL-002 | 3 | `f7a961c49e1774dc4bc2bf2300e2985eb9ded056a1600b5a9ec4a90efb84e5ab` | R2完整：1–71 |
| K-MODEL-003 | 3 | `1983c63c143c20159af90e0b7d66d4adbb0c8ac471df67d1ca1176599ed30bbc` | R2完整：1–79 |
| K-MODEL-004 | 3 | `b16b229713829787bffef07a62a08c0feca0a7b3a188aea2332a3689388c0417` | R2完整：1–68 |
| K-MODEL-005 | 2 | `147fa3bc49d6b6423d7ba03ecc311b87345ee7c2c934a4b0007ad59b2ed72bf4` | R1完整，字节未变：1–68 |
| K-MODEL-006 | 3 | `1f0c786d7dde16c9f946254f6ad1046c7d409e1c9a11e3a383f1750ffe1646e1` | R1完整，字节未变：1–80 |
| K-MODEL-007 | 3 | `1e53a46e31c96b8f99e06f88a107ec0df87becb9d064e31bed580854c0fa0aef` | R2完整：1–72 |
| K-MODEL-008 | 3 | `5ce873c82e975be23aedfebbdeeda5acc0cf55d35b51ef24f744af9d174fe55c` | R2完整：1–66 |
| K-MODEL-009 | 2 | `f08b493a7f620357ffd83d063ee0a797ffa886046659e08271c3face962000e6` | R1完整，字节未变：1–76 |
| K-MODEL-010 | 2 | `d6ebd708a7b9de12dfae97659345b1ccec8a694133be812e25a67f2e22246da7` | R1完整，字节未变：1–70 |
| K-MODEL-011 | 2 | `9cab040f27f77dc8040a6cc6f0ae96a175542e4379e2262ca2cdb53fb9aee3a4` | R1完整，字节未变：1–64 |
| K-MODEL-012 | 2 | `f1fc4dc6611ad1b53ad8cc7f2abf0cbf9b9b2022d2d22c64dc712a39943d9d15` | R1完整，字节未变：1–71 |
| K-SYS-001 | 2 | `411061489661a325939476348559d80df773a32cc19be2aef216e4d4a5b0b554` | R1完整，字节未变：1–78 |
| K-SYS-002 | 2 | `856dab119bd58b29a2a4a48826a05aab387df4101c185b142162aa5bdf65d4cb` | R1完整，字节未变：1–58 |
| K-SYS-003 | 3 | `1a764f9e8bfb9f1eb80d89abb97ebcba98318df838d10684dbb375162b621cc4` | R2完整：1–83 |
| K-SYS-004 | 2 | `919052c77f65a13c9dc09aad03341ed31f0820dbbdcb2750b5789f349f9e86c1` | R1完整，字节未变：1–81 |
| K-SYS-005 | 2 | `c24f552f27a912784d04e2a05bc3ee6849147ddf8655af5a7e67600f04c5145c` | R1完整，字节未变：1–54 |
| K-SYS-006 | 2 | `50b7c97de928be322713ef149fb734c0c20acda17c4a321c46a198b0019831be` | R1完整，字节未变：1–84 |
| K-SYS-007 | 3 | `1dea61a1f632805cb90873e747cb3079230a7b6bab76b17b4281662b82367565` | R2完整：1–64 |
| K-SYS-008 | 2 | `f68e94189e886407e217a22cea44bb4767d6611200d8b657d656e00c925c8eb8` | R1完整，字节未变：1–74 |
| K-SYS-009 | 2 | `425985efff21e4acbe091ddfa9043cf97b37e550b606f10c0e1560a3c02b1f6f` | R1完整，字节未变：1–69 |
| K-SYS-010 | 3 | `a769af3d7a1b18a662391160fd09ff7e4b2d0822ba88229f47bc6b765ad7b05c` | R1完整，字节未变：1–158 |
| K-TOOLS-001 | 2 | `6793e9f99d23ce6dea9c630e9b77a8b37954ef43b3307ef7ee52e5bc4205a557` | R1完整，字节未变：1–77 |
| K-TOOLS-002 | 2 | `cf4b01a8ca5cb35e653b65fbe2bc018d3ae5cc42788a77688524fb29c7da817d` | R1完整，字节未变：1–74 |
| K-TOOLS-003 | 2 | `9ecd4279422f94ccc42ffa70aca42d5fbec6e8f61a1182fb5983aebeaccd692d` | R1完整，字节未变：1–83 |
| K-TOOLS-004 | 2 | `bd04c303d02d55165a234b1735c31bb1259d3374143635fa593ce13576e493a8` | R1完整，字节未变：1–92 |
| K-TOOLS-005 | 2 | `23dfc65dcbd319c1789dcb80b62f6c843bb5db8c076e86a00643cff546700067` | R1完整，字节未变：1–76 |
| K-TOOLS-006 | 3 | `8ac6d2aadcdc7c303c16cb52ab5b11546236d107e4ece5fee09fca4a7a25bdff` | R2完整：1–75 |
| K-WORK-001 | 2 | `6aa75b8243c1e608d6d9a54e7754af49ac7b4c6ffc343b812a9a44db49dfba49` | R1完整，字节未变：1–67 |
| K-WORK-002 | 2 | `986f34774c8890dfc163176991b15f6f839683a588bc88f481b5510bc114fe6d` | R1完整，字节未变：1–62 |
| K-WORK-003 | 2 | `b1b68f0d2a6d640c4b64c2bc98560b9e9a79a6a60b445899949b5db15b96aa90` | R1完整，字节未变：1–69 |
| K-WORK-004 | 2 | `7e839f0863c25fd6ebc0653b878bebda20236e6280a23a837fef110858f1801e` | R1完整，字节未变：1–70 |
| K-WORK-005 | 2 | `6ca0d099f7cf89b51c6e6556628abdc873b99d01aaebd1db6ff4db4b148cf45a` | R1完整，字节未变：1–62 |
| K-WORK-006 | 2 | `d33782fcb76cdf6f05f294f82b539bad4c119292a8f27248fac2ac721794ca8c` | R1完整，字节未变：1–76 |
| K-FW-011 | 1 | `871e49626f320d579993063db039ab0a4b8e98394a54ffa2073daa9ed8bb2cb5` | R1完整，字节未变：1–64 |
| K-ALG-016 | 2 | `4f83a7b40dc7880348304cf8a3439d418fc9e911f6051e687488f8e83bdf6e03` | R1完整，字节未变：1–67 |
| K-HW-010 | 2 | `8ce38535a37fc19684a58b3c9243f28eccaa2af58205ae262a288e3c948c71cb` | R1完整，字节未变：1–71 |
| K-FW-012 | 1 | `8a02565f2a303b5d70825d1db0c08c9a6b9b2ac053b18cd18e4f0cf2d55376a2` | R1完整，字节未变：1–65 |
| K-SYS-012 | 1 | `8cfcf5b54fff448070d05207ad57958296c1613ac12acae1d1b0ec18a4d4d0e1` | R1完整，字节未变：1–71 |
| K-SYS-015 | 1 | `f8cf049cc597aa891b88075ac712e0f0f4be0f4c465cd6e9b0aa901e7dde39ed` | R1完整，字节未变：1–68 |
| K-WORK-010 | 2 | `01bc154d3bd07ebf4822ee074ab91a0998159e4fe5f7d14c4c36198442666be5` | R1完整，字节未变：1–63 |
| K-TOOLS-007 | 1 | `cafeb92ae753a362d3ab1eaad0387d3b5964fd0908c8d6195a60cb78429cb105` | R1完整，字节未变：1–65 |
| K-HW-011 | 2 | `c89333e36034b5a9ea27762b73f0fc487c202a7ce8dd7d38d72ec35fc67258ce` | R1完整，字节未变：1–78 |
| K-ALG-017 | 1 | `eb3ebc6320d5a32481cf645605d159860dc6d1edb97f1e98afd89ddb24ee82af` | R1完整，字节未变：1–77 |
| K-HW-012 | 1 | `417f087e6e2591e58ce3957b4947d3c2718f1373da31bd7f3e4947eb77e58352` | R1完整，字节未变：1–62 |
| K-ALG-018 | 2 | `a7088eba5c78cd9e96cf011d61946ad7dec4ecb6378ac16e7f2fc3820af38301` | R2完整：1–67 |
| K-ALG-019 | 1 | `cb230fafeec2d39fa40228ad44dbb41d34c62673fbac2325b6e797f206bc0fde` | R1完整，字节未变：1–64 |
| K-ALG-020 | 2 | `7d0e23df334c5550a832153ef298e5fb32f35156944453852480e4600b0f6c50` | R1完整，字节未变：1–75 |
| K-HW-013 | 1 | `9471f29d33a5821b00e11b6d08fb69fe31288087d558f05c5f9234350efe8b0d` | R1完整，字节未变：1–64 |
| K-ALG-021 | 3 | `b106cc93bcd9d5acfce5e7766eb25f5c1fbd4ba7844dd1fe6454fc08cc1d99f4` | R1完整，字节未变：1–106 |
| K-SYS-013 | 1 | `ecc4f667ff186b05fececba52c500e43d0460def524e6200b9cfa00fcdb984c0` | R1完整，字节未变：1–75 |
| K-HW-014 | 1 | `72bcf1d8948eae41c243e422f6b0370462dcc4bf33ef86f5664a09b4e46d3ba2` | R1完整，字节未变：1–78 |
| K-SYS-014 | 1 | `f551fa1f1e6213469c856a880839eefff504fb39f4dd692000b2df80a20a7f5e` | R1完整，字节未变：1–81 |
| K-ALG-023 | 1 | `1b9dcfd5fea499a91611459bb936e1c1c49ceb562171fb0f1ba360a7010f5d32` | R1完整，字节未变：1–64 |
| K-ALG-024 | 1 | `42157d4da17ab8ffd7a0aa2f5cd57a1cacdf1a989757261993f6671ae231de3a` | R1完整，字节未变：1–75 |
| K-SYS-011 | 1 | `120147b62064624928974d827f10e5e7ad748ae780025882ef1c458ee2ff8c48` | R1完整，字节未变：1–65 |
| K-WORK-007 | 1 | `d60397111d36bfb52045d741842175cc794fbb37af694406558bdbd992133171` | R1完整，字节未变：1–67 |
| K-WORK-008 | 1 | `782ba12ac9d0af5f8dbdd477e0c706230ae14e7f54d2795ab3b385eaeb8b39f4` | R1完整，字节未变：1–64 |
| K-WORK-009 | 1 | `bf9b2c1d084f562d584b391ec9e22b6893540691dd7686f65599e8c2a9fef157` | R1完整，字节未变：1–61 |
| K-SYS-016 | 1 | `0ca7501a80e750152a5fb3b9815c77f9e3f9aa393ff7546b3f05375958f88606` | R1完整，字节未变：1–73 |
| K-WORK-011 | 1 | `235de985ced67e21bfe6727ff2e91be680e8def602ad02a3f3e9de2b3ca1b8a1` | R1完整，字节未变：1–65 |
| K-SYS-018 | 1 | `283ceab23fbbf5c27bbb4486b56c2123459a19a00229c5baae70d09c18062661` | R1完整，字节未变：1–74 |
| K-SYS-019 | 1 | `1b47e39a7f36c9f1cba4dc1792ad9243fd69d083ae1d6a7bb13bfd80dae1a8a5` | R1完整，字节未变：1–67 |
| K-ALG-022 | 1 | `ddd8fa928d5ebd9876aca4971dc357b666ea999be9b148e3e88404c9e271ab13` | R1完整，字节未变：1–78 |
| K-MODEL-013 | 1 | `44f22a27c9b4a11579bed108952f18722d90cd0d4c326db189c865d7e51aef38` | R1完整，字节未变：1–71 |
| K-SYS-017 | 1 | `bd09460502715f214c20ffdd68bdbc6bb9373668a849ca602b75d5e933ff40d0` | R1完整，字节未变：1–83 |
| K-MODEL-014 | 1 | `a3b95c24c3838a54df90db945066257c0960e20a24929ed2812838fc5a3528a0` | R1完整，字节未变：1–75 |
| K-MODEL-015 | 1 | `a36b0addea05174460d8f51179643665f98c51c6aa4a4d9dec235535948dd06d` | R1完整，字节未变：1–68 |
| K-MODEL-016 | 1 | `343135d67c7fd61ed1e7fb82185df7a8f30ebfc86cb5427dfd9e6313046fe6fe` | R1完整，字节未变：1–71 |
