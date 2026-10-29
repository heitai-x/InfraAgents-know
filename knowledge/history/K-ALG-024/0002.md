# K-ALG-024：多样本与搜索分支：父子状态、评分与共同提交

一个请求可以产生多个候选前缀，但 beam search、多样本采样和投机树具有不同目标。它们可以共享不可变前缀的计算与存储，不能因此共享可变输出、随机轨迹或未经确认的后续状态。

## 先定义分支的目标

| 方式 | 分支用于什么 | 不能据此推断 |
|---|---|---|
| 多样本采样 | 从规定分布得到多个样本 | 选出其中最高分结果仍服从原单样本分布 |
| Beam search | 按累计评分保留有限候选 | 它等价于普通随机采样或保证全局最优序列 |
| 投机树 | 提供候选以减少目标模型串行验证次数 | 验证多个节点后可以任意挑一条并保持目标分布 |

beam 的评分可包括累计 log 概率、长度处理和约束；平局、EOS 和停止条件都属于搜索定义。改变长度惩罚或保留宽度会改变结果，不能作为纯执行优化隐藏。多样本的重排/筛选目标也应在任务契约中写明。

## 父子映射与状态所有权

每个分支记录稳定身份、父分支/分叉位置、逻辑 token 路径、模型状态版本、累计评分、grammar/RNG 和终止状态。压紧 batch 后的槽位只是临时映射；所有采样参数、logits 行与结果都要反向映射到同一分支。

满页的共同前缀可共享只读 KV，追加需要私有尾页或写时复制。淘汰分支后，引用为零仍不代表设备/传输已结束。设定每 token 状态为 128 KiB，共同前缀 1024 token，4 个分支各追加 16 token；忽略碎片，分别完整复制需要 520 MiB，共享前缀需要 136 MiB。这个节省不消除每条不同路径的后续计算。

两个独立样本应有符合要求的独立随机子序列。仅复制相同 RNG 状态而不定义分叉规则，可能让多个样本重复相同随机轨迹。若要求调度无关可复现，随机身份还要绑定逻辑分支和采样位置，而非当前 batch 行。

## 树形 attention 的可见关系

验证树可以把多个候选位置放到一次前向，但每个 query 只能看到原始前缀及本路径的祖先，不能看到兄弟分支。假设 A/B 都是根前缀的下一 token，C 是 A 的子节点，则 C 可以依赖 A，不能依赖 B。物理行连续不等于它们属于同一序列。

一个节点作为输入被计算后，所得 logits 表示它之后的条件分布；根后的第一个候选由根前缀末行的分布验证。被输出或被接受的 token 自身是否已经具有 KV，要按实际计算位置判断。接受路径、有效状态与已分配容量分别管理，避免树压紧时引入一步偏移。

## 草稿来源和验证规则分开

[EAGLE v1](../sources/S-EAGLE.md) §3–4 研究在目标模型特征上生成草稿，并引入提前一步的已采样 token 来消除下一特征的不确定性；它也采用树形候选组织。这说明草稿可以来自不同表示，目标分布是否保持仍取决于验证与重采样协议。

对链式随机投机，接受与拒绝修正使用相同条件前缀下实际的 p、q；对多候选树，还要明确候选如何生成、选择顺序以及拒绝后分布怎样更新。不能把链式 p/q 规则直接套在任意 top-k 树，再任意选择获批分支。grammar、temperature 和截断变换也必须与这些分布对应。

## 提交与回滚跨过多个状态族

保留一条路径时，KV/递归状态、grammar、随机状态、评分和输出缓冲共同转移到相同边界。非保留节点的物理空间等待最后使用者结束后回收。SSM 等覆盖式状态不能仅通过缩短 token 长度回滚，需要快照、检查点或重算。

如果多个分支结果被拼接为新提示，新的上下文通常需要重新处理或按明确的复用算法构建；把两条独立路径的 KV 直接拼接不是一般有效的 join。流式输出也只应发布不会被后续搜索替换的内容；无法证明稳定时，应延迟提交或按协议处理结果修订。

## 资源预算与验证

宽度增加会扩展 logits、mask、候选状态、验证 token 和排序开销。深度增加可能改善一次接受长度，也可能耗费更多无效草稿；应按已确认输出衡量成本。分支与其他请求共享预算时，还要观察公平性和整体 goodput。

在小词表上穷举短序列作为 beam/采样参考，再对同一叶路径比较树形与独立前向。覆盖兄弟泄漏、平局、EOS、重排、部分取消和分支淘汰后复用；随机协议检验支持集与频率。最后在独立任务上检查质量与时延，不能用候选数量或共享比例代替收益。

<!-- infraagents:provenance -->

## 来源与推导

- [S-EAGLE：EAGLE](../sources/S-EAGLE.md)：§2–4 将特征草稿、提前一步的已采样 token、树形验证与投机重采样连接；§5 区分链式接受率与树形候选统计。
  [完整原文](../../references/S-EAGLE/reader.html) · [全文文本](../../references/S-EAGLE/fulltext.txt)

- [S-SPEC：Fast Inference from Transformers via Speculative Decoding](../sources/S-SPEC.md)：随机投机解码用草稿分布 q 产生候选，由目标分布 p 验证。候选 x 的接受概率为 min(1,p(x)/q(x))，拒绝时从正剩余概率质量中重新采样；greedy 采用自己的确定性验证规则。
  [完整原文](../../references/S-SPEC/reader.html) · [全文文本](../../references/S-SPEC/fulltext.txt)

- [S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention](../sources/S-PAGED.md)：PagedAttention 把每条序列的逻辑 KV 划分为固定 token 数的块，用映射表访问不连续物理存储。多个序列可以共享相同前缀的物理块，分叉写入需要保持各序列状态独立。
  [完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)

- [S-TORCH-REPRO：PyTorch — Reproducibility](../sources/S-TORCH-REPRO.md)：随机种子与确定性配置可以控制特定执行路径中的随机性，但不保证跨版本、平台和全部算法路径逐位一致。
  [完整原文](../../references/S-TORCH-REPRO/reader.html) · [全文文本](../../references/S-TORCH-REPRO/fulltext.txt)

- [S-SGLANG：SGLang: Efficient Execution of Structured Language Model Programs](../sources/S-SGLANG.md)：SGLang 的 RadixAttention 利用 token 前缀树组织复用状态，结构化执行场景可以复用重复前缀计算。
  [完整原文](../../references/S-SGLANG/reader.html) · [全文文本](../../references/S-SGLANG/fulltext.txt)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [投机解码：分布保持、状态提交与收益模型](../algorithms/speculation-budget.md)。
- [批量采样：概率变换、请求随机性与行映射](../algorithms/batched-sampling.md)。
- [结构化生成：语言状态、token 边界与约束采样](../algorithms/structured-output.md)。
- [分页 KV：逻辑位置、物理布局与共享所有权](../algorithms/paged-kv-capacity.md)。
- [请求身份：逻辑序列、批次视图与结果归属](../system-semantics/request-batch-identity.md)。
- [多轮与分支负载：依赖、状态驻留和到达过程](../workloads/session-workloads.md)。
