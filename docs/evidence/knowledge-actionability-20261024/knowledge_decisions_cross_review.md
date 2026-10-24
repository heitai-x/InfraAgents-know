# 知识决策与验证章节：独立交叉审查

**结论：三项可能影响研究结论的表述已由作者修正，并经过独立复读。另一项证据范围措辞建议也已修正并全文复读。四篇正文在以下来源和阅读范围内可作为方法内容交付；本轮范围内没有未关闭的实质发现。此结论不代表目标引擎、GPU 性能或知识科研收益已得到验证。**

审查者：`/root/communication_content`；作者：`/root`。基线：`f640148464929f803a53b55e27ee4cbb83dc8657`；来源截止：2026-04-30。审查者没有修改四篇作者正文，没有运行 GPU、NCCL、模型或目标引擎。

## 阅读范围与版本

四篇初稿均完整阅读；两篇既有章节的本轮 diff 完整对照基线。修订后再次阅读了统计章节第 32–66 行、oracle 第 40–80 行以及 design-decisions 全文，核对修订与上下文的衔接。以下 SHA 只绑定实际复核的正文。

| 文件 | 阅读范围 | 复核后 SHA-256 |
|---|---|---|
| `knowledge/system-semantics/design-decisions.md` | 全文 1–91；新文全文；修订后全文复读 | `a9552b50f3ad86305a93a3e8f880f5ba6822363c4a2c5a56b9f66fd1b84bbdf2` |
| `docs/knowledge-use-cases.md` | 全文 1–81；新文全文 | `9e36e3ea2232454f4d4969a0293817683d8ac13b11208b7a3f3cbaae351ca914` |
| `knowledge/tools/correctness-oracles.md` | 全文 1–101；本轮 diff；修订段复读 | `9c58f1980d2d51021f8415c36d7f951ad652504779c318b72ee617e0e11aa5a3` |
| `knowledge/tools/statistical-experiments.md` | 全文 1–81；本轮 diff；修订段复读 | `8774f23079086cb92966f437f326eee8d557011cc1cd4c3683921264e9f393cf` |

初稿 SHA、字节数与修订复读范围完整保存在同名 JSON 报告。

## 具体发现及修复复核

| 编号 | 严重度与状态 | 位置 | 判断 |
|---|---|---|---|
| R1 | P2，已修复关闭 | statistical-experiments 第 45 行 | 原稿将预算结束未完成与右删失并列，可能掩盖不可恢复失败。现稿在固定预算完成率中保留所有未交付；仅仍可能完成且因统一时间窗口截止的运行可作行政性右删失；崩溃和其他资源预算停止另行说明。限定足够，且没有声称某个生存分析估计器已获验证。 |
| R2 | P2，已修复关闭 | statistical-experiments 第 47 行 | 原稿把无效实验次数减少直接作为成本改善依据。10 次×1 成本与 5 次×3 成本构成反例。现稿要求同验收标准下实测总成本或到合格系统的成本下降，并明确次数不足以证明成本下降。 |
| R3 | P2，已修复关闭 | correctness-oracles 第 61 行 | 原稿“成功的分母”容易被解释为删除被阻断会话。现稿区分计划轮次、实际发起请求与预先定义会话集合；阻断不能计成功，也不能删会话；请求级条件率不能替代会话成功率。 |
| R4 | P3，已修复关闭 | design-decisions 第 61 行 | 原稿“指定输入域内实现满足组件契约”可能让有限样例被读成整个域的证明。作者已改为“已检查的输入、性质及证据范围内满足组件契约”；SHA 变化后重新阅读了全文，证据范围已清楚。 |

作者修订后的三段衔接自然，没有重复转折或因替换产生的语法断裂。统计章节对固定预算成功率、到达时间及成本的区分，与 oracle 的失败人口保留原则一致。

## 语义审查结果

使用场景围绕决定组织：触发症状、必要输入、未知条件、可交付工程产物和下一项证据都有具体含义。新章节不要求所有阅读机械填写同一表格，也没有把完整机制压成口号。布局转换的反例能区分 kernel 微基准改善与真实调用链改善；缓存例子要求区分少算、队列变化和预热差异。

从空工程构建完整推理引擎仍是主目标。局部组件能力只是阶段产物，完整任务验收保留原质量、历史、服务与资源要求。连续批处理、图重放及更多 stream 都是需要条件和证据的候选。数值路径差异没有被用来临时放宽精确输出契约。

请求身份反例可以识别错误合并、丢失或只保留正确子集；终止状态与内容分别核对。检查器修复只能恢复判断能力，旧候选仍须按修订后的规则复核；现稿没有把这种原则夸成已实现的自动证据传播系统。

运行时能力另外静态核对了本地 InfraAgents 源码：`knowledge.py` 的 search/read/validate_uses、`tools.py` 的分发和 knowledge_store 的模式限制，确有设计/诊断检索、冻结正文读取和使用版本绑定。这里只证明代码存在对应路径，没有执行这些接口。本文也没有声称新增自动场景路由或 workflow 执行器。

## 固定来源复核

| 来源及版本日期 | 已读原文范围 | 支持内容与边界 |
|---|---|---|
| S-ORCA，2022-07-13 | fulltext 第 570–630 行 | Supports cross-component scheduling example; not the seven-decision taxonomy. |
| S-ASGI，2026-02-23 | fulltext 第 180–276 行 | Send-buffer flush or response completion does not establish client consumption; wording remains conditional on protocol/version. |
| S-STATS，2020-07-21 | fulltext 第 395–488, 630–865, 1405–1478 行 | Supports hierarchical repetition and assumptions; Agent conditions, survival-analysis cautions, paired toy interval and tail example are author design/derivation, not attributed experiments. |
| S-TORCH-NUMERIC，2026-04-30 | fulltext 第 1–70 行 | Supports numerical path caution, not loosening immutable task quality; incorrect row/column source example is explicitly excluded. Lines 1–155 were also read in the preceding author task. |
| S-TORCH-REPRO，2026-04-30 | fulltext 第 1–181 行 | Seeds and deterministic algorithm settings control some paths; not cross-platform/version or concurrency coverage proof. |
| S-GENAI-PERF，2026-04-28 | fulltext 第 475–505 行 | Confirms metric event/window distinctions in surrounding statistical chapter; no target observation is inferred. |

S-MLPERF-SCENARIOS 只检查了固定版本和日期元信息，没有将其计作本轮语义全文复核。来源哈希、精确版本、原文段落主题与额外运行时代码范围见 JSON。S-STATS 的配对比较并非其本文方法，现稿明确区分固定论文的重复层级原则与本库新增研究设计。PyTorch 数值原文错误的行/列切片示例被新章节主动排除。

## 数值与反例复算

| 项目 | 结果 | 证据性质 |
|---|---|---|
| 转换成本 | 旧路径 R×1.0，新路径 0.5+R×0.6；严格获益要求 R>1.25，因此整数至少 2。R=1 时 1.1>1.0，R=2 时 1.7<2.0。 | 设定数字下的代数。正文 R>1 在已声明整数计数下与 R≥2 一致；若将来解释为跨对象平均复用率，应继续使用 1.25 阈值。 |
| 身份与空集 | A=17、B=99 的预期，拒绝 A=17、B=17、仅 A 及空结果；只比较保留子集的朴素检查会误接纳仅 A。 | 静态谓词反例，不是运行轨迹。 |
| t 区间 | n=16、均差 2、标准差 1、临界值 2.131，得到 [1.46725, 2.53275]。 | 使用正文给定临界值复算；不独立证明数据满足 t 方法假设。 |
| 零超限单侧界 | n=100、α=0.05，1−α^(1/n)=0.0295130496，约 2.95%。 | 二项独立观测假设下的推导；不是测得尾延迟。 |
| 成本反例 | 10×1=10；5×3=15。 | 证明减少实验数不足以推出成本下降。 |
| 来源错误示例边界 | 矩阵 [[1,2],[3,4]] 第一行和为 3，第一列和为 4。 | 说明排除该行/列等价例正确，不否定原文的一般浮点警告。 |

复算明细保存于 `knowledge_decisions_review_calculations.json`。它只核对算术和反例，不代表硬件实验。

## 条件化结论与未覆盖范围

本审查仅绑定所列正文 SHA。主线程仍需把 catalog、历史版本、导航和冻结视图注册绑定到实际发布正文，并执行真实知识接口读取探针。审查时新增 ID 的集成仍在进行，本文不把待完成的集成当作已通过。

实际构建成本、固定预算成功率、端到端质量/SLO 和跨任务收益必须由受控完整任务给出。本轮没有运行 GPU/NCCL、没有测量目标模型、没有宣布完整引擎交付或科研新颖性得到证明。机械结构通过不能替代这里逐段机制与反例复核；本审查本身也不是所有引用论文和全部未来输入的证明。
