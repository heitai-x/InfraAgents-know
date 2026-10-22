# 推理知识库扩充进度

本轮在 Astra 日期分支持续建设完整的推理知识库。技术来源版本上限为 **2026-04-30，含当天**；编辑、归档和检查仍记录真实日期。新整理的内容属于 seed，不表示已经在 InfraAgents Task 上验证有效。

完整待办保存在 [扩充计划](expansion-plan.json)：逐一处理原有 73 篇、原方案 32 个新专题和 12 类组合场景。根据本轮用户确认的推理范围，增加 3 个平台专题：集群资源与放置、推理存储服务、部署环境。它们承接已有容量、拓扑、加载、恢复和隔离机制，继续放在现有八类目录中。

平台内容回答推理服务如何获得资源、读取稳定版本的模型与索引、管理共享带宽和故障域，以及部署限制怎样改变可用性能。训练作业调度、训练数据系统和通用平台运维百科不属于这次交付范围。

章节完成要有模型与状态、合法替代方案、资源账、组合条件、区分实验和选择反转条件；不能通过写出文件、达到篇数或通过格式检查宣称内容已经完整。组合场景的验证设计也不能冒充真实 Task 的性能收益实验。

当前已接入 15 篇新增专题，总数为 88 篇、87 个来源。首批 8 篇为权重加载、基础算子、CPU GEMM/GEMV、CPU 低位执行、内存分配、主机运行时、多轮负载与实验统计。第二批 7 篇为 GPU 矩阵/归约流水、多级驻留、存储 I/O、KV 卸载、权重/专家卸载、CPU–GPU 计算放置和共享内存路径。新篇在实施计划中记为 integrated_draft，仍需整库组合与最终语义复核。

原有 73 篇中已有 23 篇更新至修订 2。除首批 16 篇基础链外，新增扩写 attention IO、异步调度、stream/allocator、主机设备传输、CUDA、ROCm 和 Apple Silicon。对应修订 1 的正文与 metadata 已保存在 history，和原始 Astra 提交逐字节/逐字段核对一致。其余 50 篇仍在待办中。

[系统设计推导](../knowledge/system-semantics/design-synthesis.md) 已加入 C01 CPU 多请求生成的完整场景：固定设定 → KV/容量 → 三种执行候选 → 带宽下界 → prefill/主机预算 → 验证 → 选择反转。它明确显示“容量容得下”和“ITL 达标”是不同判断，所有数值均为设定算例，未冒充真实引擎测试。

C02 单 GPU 交互场景在 [CUDA 平台](../knowledge/hardware/cuda-execution.md)，C04 三级存储场景在 [多级驻留](../knowledge/algorithms/tiered-residency.md)，C05 CPU–GPU 分工场景在 [异构执行](../knowledge/algorithms/heterogeneous-execution.md)。这三类也已形成完整条件链和可复算例子；性能均使用声明假设下的资源界限或比较，实际可达性仍需目标 Task 测量。

首批新增检查：[结构与时间资格](evidence/p0-integration-audit.json)、[检索与链接](evidence/p0-retrieval-check.json)、[实际读取接口](evidence/p0-runtime-read.json)。这些文件记录首批提交当时的 1425 个链接与新增章节查询结果。

基础链检查保存在 [基础链修订验证](evidence/p0-foundations-validation.json)，对应上一批的 81 篇、83 个来源与 1841 个本地链接。

当前检查见 [多资源批次验证](evidence/p1-resource-validation.json)：88 篇和 87 个来源通过原框架审计，无时间排除项；2055 个当前正文、来源、索引和主要说明文件中的本地链接有效；23 个历史修订与基线一致，本批 14 篇新增/修订正文通过真实 RepositoryKnowledge 读取，4 份新增来源全文完成分页读取核对，8 个定向查询找到目标专题。各资源算例与在线 softmax 反例同时复算。检查不证明任意措辞的检索效果、整库完备性或 Agent/引擎实际性能。

已完成 32 篇固定论文原文的采集，首批将引用的 7 篇纳入 source registry；未接入的材料留在编辑暂存区，不向运行时暴露。Goto 资料的公开镜像是作者稿，按 2016-11-30 固定提交绑定版本，明确没有冒用 2008 年最终出版日期。

第二批另外接入 FlexGen、PowerInfer、FlashAttention-3 三篇固定论文和 Linux v6.12 的三个完整 I/O 文档源码文件。Linux 固定到官方提交 adc218676eef25575469234709c2d87185ca223a，日期为 2024-11-17。至此，32 篇已采集论文中的 10 篇进入登记，另有一个新官方文档来源。

来源阅读进度已从 [首批机制段核对](evidence/p0-source-review.json) 推进到 [整篇技术正文与附录复核](evidence/p0-source-full-review.json)：ServerlessLLM、RMSNorm、RoFormer、整数推理、vAttention 与此前完成的 Goto 作者稿均已完成相应阅读。STATS 的 §6 等引用机制段已读，其余证明与实验部分仍需继续核对。原文归档字节没有因阅读说明更新而改变。

[第二批来源记录](evidence/p1-resources-sources.json) 中，PowerInfer 全文和 Linux 所选文档已完成阅读，FlexGen 和 FA3 的剩余实验/附录仍待复核。FA3 v1 Algorithm 1 第 21 行的缩放问题已按原 PDF 视觉核对，正文使用独立在线 softmax 恒等式与凸包反例；没有修改原论文或声称其代码也存在同一问题。

后续继续完成剩余 50 篇既有章节的扩写/关联核查、20 个新专题、其余 8 类完整场景及整库复核。集群、存储服务和部署环境仍在这份完整计划内。源码中的 snapshot 导航替换问题仍记为框架限制，不冒称本轮已经修复。当前扩充工作尚未推送 GitHub 或替换远端实验使用的知识目录。

本次是用户授权的 base 内容整理，保留章节 ID、修订历史、catalog/source schema 和既有运行时读取逻辑。运行期 learned 知识仍由 Evolution 提案、独立复核和原发布流程管理；本轮不制造运行证据或伪造该流程的审核结论。
