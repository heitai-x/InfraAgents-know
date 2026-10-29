# K-MODEL-003：MoE：路由、Dispatch/Combine 与专家负载

MoE 的条件计算节省激活 FLOPs，但全部专家的驻留和路由后的不均衡决定真实服务成本。

## 一次 MoE 层的数据关系

输入 token hidden states 经过 router 得到专家分数，每 token 选择 k 个专家。每个 token 形成 k 个带权分派项；按专家重排后执行各专家 FFN，再将结果按原 token 聚合。典型形式 yₜ＝Σₑ gₜₑ Eₑ(xₜ)，其中只有选中专家具有非零权重；共享专家分支可以独立叠加。

重排必须保留原 token ID、专家 ID、路由权重和返回位置。combine 的语义通常是加权归约，不能用后到达的专家输出覆盖先到达结果。路由是否在 top-k 后重新归一化、是否有共享专家、是否限制设备组，都由模型定义。

## 激活参数不等于驻留参数

假设 8 个同尺寸专家，每 token 选 2 个。专家计算量接近 2 个专家的工作，但通常要保留 8 个专家权重以响应下一批任意路由。小 batch 下，每专家 token 很少，GEMM 变成许多碎片；增大 batch 可以增大各组，却提高排队和 KV 容量。

设本轮待路由 token 数 B＝16，每个 token 选择 k＝2 个互不重复的专家，则总 dispatch 项为 32，任一专家最多接收 16 项。均匀分给 8 个专家时各为 4 项；一种合法的偏斜分配是 [16,4,3,3,2,2,1,1]。热点可能形成慢端，但实际关键路径还取决于专家放置、权重读取与并行调度；其余专家仍各自承担所分到的工作。在线语料和小样本的路由分布可能与训练平均不同，不能只报告平均 tokens/expert。

## 专家并行的通信形态

EP 让 token hidden states 被发送到持有专家的 rank，再把结果发送回去。发送流量近似随远端分派项×hidden size×dtype 增长，另有索引和权重元数据。同 token 的多个专家在同设备时，可以减少重复网络传输，但仍需本地分派。拓扑感知路由只有在模型允许相应路由规则时才成立，不能运行时随意改 expert 选择。

容量因子与 token dropping 常见于训练或特定模型实现；推理是否允许丢弃必须由模型语义决定。为让固定形状 kernel 装下而静默丢弃溢出项，会改变函数。可变长 grouped GEMM、padding、分批执行是不同的实现选项。

## 可推导的优化路径

当主要成本是小专家 GEMM，融合 dispatch、分组计算和减少 padding 有意义；当主要是跨节点传输，调整 EP/TP 组合与专家布局更直接；当主要是权重容量，量化或卸载可能扩大可服务模型，但带来搬运成本。MLA 降低 attention 状态不会直接减少 MoE 权重，两个改进必须分别进入容量账本。

## Grouped GEMM 处理的是一组不同的矩阵形状

对专家 e，重排后的输入为 Xₑ[nₑ,d]，门控 FFN 的两次上投影为 [d,f]，逐元素门控后下投影为 [f,d]。路由决定 nₑ，权重和 hidden 维由模型决定。grouped GEMM 将这些不同 nₑ 的乘法统一组织，减少逐专家启动和小矩阵碎片，但不会让空专家获得有效工作，也不会自动消除热点专家的长队列。

三种可比较组织是：逐专家小矩阵；把每组补到固定行数后批量执行；按真实组长执行并按 tile 分配工作。第一种管理直接却易受启动开销限制；第二种利于固定图和规则 tile，却有 padding；第三种减少无效行但增加分组元数据、间接访问和调度。nₑ 很小时，许多专家表现为低复用 GEMV；prefill 的大组则更接近高复用 GEMM。路由排序、group offsets、输入重排、输出反排和加权 combine 都属于这一层的完整成本。

设 8 个专家、d=4096、f=14336、三矩阵 SwiGLU、BF16 权重，则每专家 3×d×f×2=336 MiB，整层专家共 2.625 GiB。每 token 选 2 个只描述本 token 的计算。若 16 个 token 独立均匀选择 2 个不同专家，预期触及专家数为 8×[1−(3/4)¹⁶]≈7.92；小批也可能读取几乎全部专家权重。这个独立均匀假设用于说明“激活比例”不能直接充当批级权重流量比例，真实路由须测量。

## 专家副本改变放置，不改变专家函数

热点专家可在多设备保留同一权重版本的副本，并把原来选择该专家的分派项分流到其中一个副本。原专家 ID、路由权重及每 token 应有的分派次数保持不变；把同一项发给两个副本再都 combine 会重复计算贡献。放置表切换要让在途派发使用同一代映射，旧副本等最后消费者完成后释放。

[DeepSeek-V3 v1](../sources/S-DEEPSEEKV3.md) 的部署段落分别讨论了 prefill/decode 的冗余专家和微批重叠。它提供了负载统计驱动复制的实例；不能把该集群的专家数、周期或并行配置视为普遍最优。正文还区分已采用的部署与仍在探索的动态冗余/部分 decode 重叠，知识库不将探索描述转成完成能力。

路由选择规则属于模型。该版本以带 bias 的 affinity 决定选哪些专家，但用于 FFN 加权的 gating 值来自原始 affinity；把用于选择的 bias 同时加入输出权重会改变函数。在线系统可以选同一专家的合法副本，不能为均衡负载随意选另一个专家。已有训练均衡也不保证在线短批、单领域或分布变化后的均衡。

## 驻留、卸载与通信重叠共同选择

对每层分别记常驻专家、DRAM/SSD 副本、当前需要但未到达的专家、在途输入和完成待合并输出。CPU 执行冷专家与把冷专家权重搬到 GPU 是两种候选；前者传 hidden/输出，后者传大权重，并有不同计算速度。预测下一层专家只能作为预取提示；真正路由选到未预取专家时仍要正确执行，不能静默丢弃。

按层流水可先加载后续层权重，但实际专家依赖尚未得到的 hidden 时，预取覆盖率和多余流量要进入账。等待 expert 到达期间仍需占有输入、返回位置和请求状态。若取消，只撤销尚未提交工作并等待在途读者完成，不能立即复用 DMA 目的缓冲。

通信和计算重叠需要两个有独立进展空间的微批或分块，以及匹配的 dispatch/combine 顺序。它同时占用更多 KV、路由缓冲、通信资源和计算单元；某个微批的输出没有完成 combine，不能进入下一依赖层。用同一 trace 比较每专家 nₑ、最慢 rank、padding、权重读取、网络字节和关键路径，才能判断瓶颈是小 GEMM、偏斜、传输还是容量。

验证保留独立的逐 token 逐专家参考，覆盖零 token 专家、全部请求命中同一热点、top-k 边界与平局、同专家不同副本、并行分流、卸载迟到和取消。精度检查先确认选中集合与 gating，再比较 FFN 与 combine；归约次序造成的舍入应单独记录。负载均衡改善不构成改变模型路由或丢 token 的授权。

<!-- infraagents:provenance -->

## 来源与推导

- [S-MIXTRAL：Mixtral of Experts](../sources/S-MIXTRAL.md)：Mixtral 采用稀疏专家结构：每个 token 根据路由选择部分专家参与计算，而专家总参数与每 token 激活参数是不同数量。
  [完整原文](../../references/S-MIXTRAL/reader.html) · [全文文本](../../references/S-MIXTRAL/fulltext.txt)

- [S-MLA：DeepSeek-V2：MLA 与解耦位置编码](../sources/S-MLA.md)：DeepSeek-V2 的 MLA 以低维 latent 联合表示内容 K/V，通过线性投影的吸收减少历史展开；位置相关旋转不能任意与投影交换，因此采用解耦 RoPE 分量。
  [完整原文](../../references/S-MLA/reader.html) · [全文文本](../../references/S-MLA/fulltext.txt)

- [S-NCCL-COLLECTIVE：NCCL — Collective Operations](../sources/S-NCCL-COLLECTIVE.md)：集合通信操作由 communicator、参与 ranks、数据类型、数量、归约方式或 root 等共同定义。参与者必须执行相互匹配的逻辑操作。
  [完整原文](../../references/S-NCCL-COLLECTIVE/reader.html) · [全文文本](../../references/S-NCCL-COLLECTIVE/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-DEEPSEEKV3：DeepSeek-V3 Technical Report](../sources/S-DEEPSEEKV3.md)：DeepSeek-V3 v1 的 MLA、细粒度/共享专家、路由选择与 gating 区分、MTP，以及分阶段推理部署中的专家复制、微批重叠和在线偏斜；只使用该固定版本。
  [完整原文](../../references/S-DEEPSEEKV3/reader.html) · [全文文本](../../references/S-DEEPSEEKV3/fulltext.txt)

## 延伸阅读

- [推理并行：张量分解、流水、专家与上下文](../algorithms/parallelism.md)。
- [通信拓扑：从 Rank 组到瓶颈链路](../hardware/topology-parallelism.md)。
- [工作负载分布：长度、复用、模态与相关性](../workloads/input-distribution.md)。
