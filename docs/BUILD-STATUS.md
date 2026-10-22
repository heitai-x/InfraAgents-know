# 推理知识库扩充进度

本轮在 Astra 日期分支持续建设完整的推理知识库。技术来源版本上限为 **2026-04-30，含当天**；编辑、归档和检查仍记录真实日期。新整理的内容属于 seed，不表示已经在 InfraAgents Task 上验证有效。

完整待办保存在 [扩充计划](expansion-plan.json)：逐一处理原有 73 篇、原方案 32 个新专题和 12 类组合场景。根据本轮用户确认的推理范围，增加 3 个平台专题：集群资源与放置、推理存储服务、部署环境。它们承接已有容量、拓扑、加载、恢复和隔离机制，继续放在现有八类目录中。

平台内容回答推理服务如何获得资源、读取稳定版本的模型与索引、管理共享带宽和故障域，以及部署限制怎样改变可用性能。训练作业调度、训练数据系统和通用平台运维百科不属于这次交付范围。

章节完成要有模型与状态、合法替代方案、资源账、组合条件、区分实验和选择反转条件；不能通过写出文件、达到篇数或通过格式检查宣称内容已经完整。组合场景的验证设计也不能冒充真实 Task 的性能收益实验。

当前已接入 21 篇新增专题，总数为 94 篇、95 个来源。前两批覆盖基础构建链与 GPU/分层存储/异构执行；第三批新增 KV 压缩、分布式 KV 存储、通信与 RDMA、服务恢复、路由迁移和分支生成。新篇在实施计划中记为 integrated_draft，仍需整库组合与最终语义复核。

原有 73 篇中已有 44 篇更新至修订 2。本批另外更新分页 KV、前缀缓存、投机、批量采样、结构化生成、P/D 分离、分布式状态迁移、集合顺序和通信拓扑。对应修订 1 的正文与 metadata 已保存在 history，和原始 Astra 提交逐字节/逐字段核对一致。其余 29 篇仍在待办中。

[系统设计推导](../knowledge/system-semantics/design-synthesis.md) 已加入 C01 CPU 多请求生成的完整场景：固定设定 → KV/容量 → 三种执行候选 → 带宽下界 → prefill/主机预算 → 验证 → 选择反转。它明确显示“容量容得下”和“ITL 达标”是不同判断，所有数值均为设定算例，未冒充真实引擎测试。

C02 单 GPU 交互场景在 [CUDA 平台](../knowledge/hardware/cuda-execution.md)，C04 三级存储场景在 [多级驻留](../knowledge/algorithms/tiered-residency.md)，C05 CPU–GPU 分工场景在 [异构执行](../knowledge/algorithms/heterogeneous-execution.md)。这三类也已形成完整条件链和可复算例子；性能均使用声明假设下的资源界限或比较，实际可达性仍需目标 Task 测量。

C03 长公共前缀与交错会话在 [前缀缓存](../knowledge/algorithms/prefix-cache-identity.md)，C07 P/D 与远端 KV 在 [P/D 分离](../knowledge/algorithms/disaggregated-serving.md)。前者区分物理共享容量和逻辑读取量，后者用首内容与第二 token 的完整时间轴解释 TTFT/ITL 的取舍；均含候选、状态、资源、验证和选择反转。

首批新增检查：[结构与时间资格](evidence/p0-integration-audit.json)、[检索与链接](evidence/p0-retrieval-check.json)、[实际读取接口](evidence/p0-runtime-read.json)。这些文件记录首批提交当时的 1425 个链接与新增章节查询结果。

基础链检查保存在 [基础链修订验证](evidence/p0-foundations-validation.json)，对应上一批的 81 篇、83 个来源与 1841 个本地链接。

多资源批次检查保存在 [上一批验证](evidence/p1-resource-validation.json)，对应 88 篇、87 个来源和 2055 个链接。

当前 [分布式与缓存批次验证](evidence/p1-distributed-validation.json) 覆盖 94 篇与 95 个来源的结构/归档/时间审计，无超期或未知来源；32 个历史修订与基线一致；本批 15 篇正文通过真实 RepositoryKnowledge 读取，8 份新来源全文通过分页核对，8 个定向查询找到目标专题，2280 个当前本地链接有效。压缩元数据、分支共享、迁移追赶、长前缀和 P/D 时间轴均已复算。这些检查不证明整库完备性或实际引擎收益。

该次读取检查的 Windows 原子目录提升曾返回 WinError 5。进程退出后确认目标尚未存在，核对已完整生成的隔离快照，在同一验证目录内完成提升并恢复检查，全部读取随后通过。没有修改运行时来吞掉错误；最终接入仍安排目标 Linux 环境的原生验证。

已完成 32 篇固定论文原文的采集，首批将引用的 7 篇纳入 source registry；未接入的材料留在编辑暂存区，不向运行时暴露。Goto 资料的公开镜像是作者稿，按 2016-11-30 固定提交绑定版本，明确没有冒用 2008 年最终出版日期。

第二批接入 FlexGen、PowerInfer、FlashAttention-3 与 Linux v6.12 I/O 文档。第三批接入 KIVI、H2O、CacheGen、Mooncake v1、Llumnix v1、EAGLE v1，以及 rdma-core v54.0 和 CUDA 12.8 GPUDirect 文档。至此，32 篇已采集论文中的 16 篇进入登记，另有 3 个新官方文档来源。Linux 与 rdma-core 均固定官方提交；GPUDirect 使用固定页面的 2025-01-21 更新坐标，和 toolkit 发布/其他文档更新日期分开记录。

来源阅读进度已从 [首批机制段核对](evidence/p0-source-review.json) 推进到 [整篇技术正文与附录复核](evidence/p0-source-full-review.json)：ServerlessLLM、RMSNorm、RoFormer、整数推理、vAttention 与此前完成的 Goto 作者稿均已完成相应阅读。STATS 的 §6 等引用机制段已读，其余证明与实验部分仍需继续核对。原文归档字节没有因阅读说明更新而改变。

[第二批来源记录](evidence/p1-resources-sources.json) 中，PowerInfer 全文和 Linux 所选文档已完成阅读，FlexGen 和 FA3 的剩余实验/附录仍待复核。FA3 v1 Algorithm 1 第 21 行的缩放问题已按原 PDF 视觉核对，正文使用独立在线 softmax 恒等式与凸包反例；没有修改原论文或声称其代码也存在同一问题。

[第三批来源记录](evidence/p1-distributed-sources.json) 中，KIVI、EAGLE 全文以及所选 verbs/GPUDirect 文档已完成阅读；H2O、CacheGen、Mooncake、Llumnix 已核对引用的机制段，剩余全文复核仍开放。新正文明确区分有损量化与无损熵编码、理论分布保证与实现验证，以及目录命中/CQE/应用提交。

后续继续完成剩余 29 篇既有章节的扩写/关联核查、14 个新专题、其余 6 类完整场景及整库复核。集群、存储服务和部署环境仍在这份完整计划内。源码中的 snapshot 导航替换问题仍记为框架限制，不冒称本轮已经修复。当前扩充工作尚未推送 GitHub 或替换远端实验使用的知识目录。

本次是用户授权的 base 内容整理，保留章节 ID、修订历史、catalog/source schema 和既有运行时读取逻辑。运行期 learned 知识仍由 Evolution 提案、独立复核和原发布流程管理；本轮不制造运行证据或伪造该流程的审核结论。

2026-10-23 已按用户授权启动三个并行编辑工作包，分别负责服务/平台、非生成推理/编译量化、模型/多模态/平台执行。它们在独立暂存目录准备正文与来源记录，catalog、版本历史与来源登记由主 agent 串行整合。当前暂存稿尚不计入上面的入库数量。

主 agent 本批完成 12 篇系统语义、观测和引擎导航修订，补跨节点时钟不确定性、位置/表示一致性、知识论断与受控评估、存储/网络/CPU 观测、计数器作用范围以及独立引擎的接口契约。新增历史保留在原 ID 下。当前[结构和时间检查](evidence/p2-observability-structural.json)通过；[Windows 读取检查](evidence/p2-observability-windows-read.json)再次在快照目录提升时返回 WinError 5，尚未完成该批实际读取。没有吞掉错误或改动运行时，将在目标 Linux 的隔离工作区进行原生验证。独立语义复核仍待交叉审阅。
