# K-FW-006：Attention Backend：语义、布局与执行契约的交集

选择 attention backend 是匹配可见性、张量几何、状态布局和执行模式，性能排名只对确定的运行点成立。

## 先写出共同函数

backend 接口必须表达 Q/K/V 的 head 数与通道、query 和 KV 各自长度、逻辑位置、mask/偏置、scale、dtype 和输出形状。分页、ragged 和稠密只改变存储表示；如果接口不能表达所需可见性，则速度再高也不是等价替换。

SDPA 是框架级算子，会按条件选择实现；FlashAttention 提供特定融合 attention 路径；FlashInfer 提供面向推理的分页、变长和计划接口；ROCm 上还可能选择 AITER 等目标平台实现。库名不等于模型和平台的完整支持矩阵，具体函数比库总览更有决定性。

## 非等长 causal 是一个具体反例

设 Q 长 2，K 长 5，Q 对应历史尾部位置 3、4。期望第一行可见前 4 个 K、第二行可见 5 个。FlashAttention 文档说明相关版本 causal 对齐改为右下；PyTorch SDPA 的非方形 is_causal 有其上左对齐定义。两者若直接互换同一个布尔参数，可能给第一行只看 1 个、第二行只看 2 个。应以绝对位置关系构造等价 mask，或选择明确支持相应对齐的接口。

另一个常见差别是布尔 mask 的 true 含义：不同算子可能分别表示保留或屏蔽。dropout_p 也可能由算子参数直接控制，不能假定外部 eval 状态自动将它清零。模型语义优先于同名参数。

## 用契约表组织选择

| 场景 | 必须能表达的内容 | 决定额外成本的项 |
|---|---|---|
| 稠密 prefill | 因果/非因果、变长 query | padding、tile、是否物化 mask |
| 分页 decode | 页表、tail、Hq/Hkv | plan、间接读取、split-K |
| Cross-attention | 独立 Q/KV 分段与源可见性 | 投影缓存、独立几何 |
| MLA | latent 与位置分量 | 是否展开历史、专用归约 |
| 图执行 | 稳定地址及允许更新的 metadata | 捕获外 plan、workspace |
| 窗口/稀疏 | 每 query 合法历史集合 | 索引和无效 tile |

## 适配器应承接什么

一个清晰的适配层可把引擎逻辑状态转换成 backend 输入，说明是否发生 gather、transpose、dtype conversion 或额外 copy。这样的转换可能比 kernel 本身更贵，benchmark 应覆盖它们。fallback 也应保持同一数学契约，不能在不支持窗口时静默改为全 attention。

不同 prefill/decode backend 可共享持久缓存，但布局、量化 scales 和写入约定必须一致。跨 backend 比较使用固定张量与可见关系，再评价端到端成本，才能区分数学接入错误和资源效率差异。

## 固定来源下的具体接口能力

下表绑定本知识版本已经归档的接口范围。能力属于某个函数和配置，不能从一行推出整个库的全部组合均受支持。

| 固定接口 | 已明确的输入/语义 | 接入时必须保留的边界 |
|---|---|---|
| PyTorch SDPA，提交 647b6a2，2026-04-30 | Q 为 …×Hq×L×E，K/V 的历史长 S 可不同；bool mask 的 true 参与 attention；显式 dropout_p；GQA 有 head 整除条件 | 普通张量参数不直接解释引擎页表；非方形 is_causal 左上对齐；具体 fused dispatch 还受 dtype、shape 与平台限制 |
| FlashAttention 的 flash_attn_func 及 varlen 接口，提交 c5f6ff4，2026-04-30 | 文档列出 MQA/GQA、窗口与 ALiBi；v2.1 起非等长 causal 右下对齐 | 常规入口与 KV-cache/分页入口不可混用参数；同名 deterministic 参数的文档针对 backward，不是任意跨批逐位一致性的承诺 |
| FlashInfer BatchPrefillWithPagedKVCacheWrapper，提交 070fabf，2026-04-30 | query indptr、KV page indptr/indices、末页长度，NHD/HND 布局；自定义 mask、窗口、位置模式和不同 QK/VO head 维 | plan 与 run 分离；plan 不可捕获；提供 custom mask 时普通 causal 选项不再替代它；某些 split 设置不保证图兼容 |
| FlashInfer 论文 2501.01005v1，2025-01-02 | BSR/ragged 表示、分区 attention 状态合并、动态工作队列与固定 workspace 的方案 | 属于论文中的具体设计和实验版本，不能用来覆盖后来固定源码中更具体的限制 |

不同条目之间需要有明确转换。SDPA 的 H、L 轴与常见 B×L×H×D 接口不同；只调整 shape 而未调整 stride/数据解释会读错。FlashInfer 固定源码说明未预打包的 custom mask 会额外 pack；这一项属于接入开销。AITER 等其他后端的项目总览不能补齐某个具体算子签名，应沿所选版本的实际函数建立同等细度的契约。

## 分区 attention 合并的额外不变量

若同一 query 的合法 K/V 集合拆成互不重叠的 I、J，且每一部分使用相同 score 定义和位置/数值解释，可以保留归一化输出 O 与 log-sum-exp l。取 m=max(l_I,l_J)，权重 a=exp(l_I−m)、b=exp(l_J−m)，合并输出为 (aO_I+bO_J)/(a+b)，新的 l=m+log(a+b)。用 max 做平移避免直接 exponent 大 l 溢出。

互不重叠是必要条件：共享前缀已经在 I 中出现，就不能在 J 的完整历史中再计一次，否则它的概率质量被重复增加。只有输出 O 而没有归一化尺度，通常不能通过普通平均恢复整体 attention。空分区、全 mask 行、LSE 是自然对数还是其他底、scale/softcap 的顺序也必须统一。

共享 prefix 的多个 query 仍有各自 Q，复用主要是 K/V 的读取与存储；不能共享本应不同的 attention 输出。分页表示本身只说明存储间接寻址，稀疏可见性若删除了原函数允许访问的位置，则是另一个质量敏感算法，不能从“同为块稀疏表示”推出精确等价。

## 转换成本与可复算反例

设一层 K/V 各有 L=8,192、Hkv=8、d=128、FP16，合计 2×8,192×8×128×2=32 MiB。若每次调用先完整 gather 到连续布局，需要读取旧布局并写入新布局，单是转换至少产生约 64 MiB 有效字节，之后 attention 还会读取新缓冲。更快的连续 kernel 只有在抵消这些转换、额外驻留与同步后才构成收益。

独立小例中，对同一 query 设两个 logits 都为 0，V 分别为 0、2，则整体输出为 1。各分区输出为 0 和 2、LSE 同为 0，合并得 1；若第二分区同时包含第一个位置，错误合并会重复计数，得到 2/3。对 Q 长 2、K 长 5 的因果输入，再显式验证两行可见集合，而不只比较输出 shape。

接入测试应覆盖 GQA 的 head 映射、QK 与 VO 维度差异、非整页、窗口边界、prefix/suffix 不重叠、无可见 key、位置偏置和 quantization scale。对同一逻辑状态依次记录转换、plan、kernel、合并和输出路由的耗时/内存，并用独立稠密数学参考验证。长历史、低 batch 时分区可能增加并行度；短历史或转换占主导时，简单路径可能更快。选择应由接口交集与完整执行成本决定。

<!-- infraagents:provenance -->

## 来源与推导

- [S-SDPA：PyTorch：scaled_dot_product_attention](../sources/S-SDPA.md)：已核对 SDPA 文档：布尔 mask 的 true 表示参与 attention；dropout_p 大于零时执行 dropout；非方形 is_causal 采用文档定义的左上对齐。Q、K/V heads 与通道形状有明确关系。
  [完整原文](../../references/S-SDPA/reader.html) · [全文文本](../../references/S-SDPA/fulltext.txt)

- [S-FA-IMPL：FlashAttention 官方实现说明](../sources/S-FA-IMPL.md)：FlashAttention 的版本说明记录：2.1 将非等长 Q/K 的 causal 对齐由左上改成右下。示例 Q 长 2、K 长 5 时，右下对齐使两行分别可见前 4、5 个位置。
  [完整原文](../../references/S-FA-IMPL/reader.html) · [全文文本](../../references/S-FA-IMPL/fulltext.txt)

- [S-FI：FlashInfer：Attention kernels](../sources/S-FI.md)：截止日前的 FlashInfer 固定源码区分 query 分段、KV 页表及末页长度。相关 wrapper 的 plan 安排计算与工作区，run 执行 attention；plan 的捕获限制与图模式缓冲须按该版本接口核对。
  [完整原文](../../references/S-FI/reader.html) · [全文文本](../../references/S-FI/fulltext.txt)

- [S-AITER：ROCm AITER 官方仓库](../sources/S-AITER.md)：AITER 是 AMD ROCm 的算子实现项目，提供面向相应硬件的 attention、矩阵或相关算子路径。项目覆盖不代表每种形状和设备都有相同实现。
  [完整原文](../../references/S-AITER/reader.html) · [全文文本](../../references/S-AITER/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-FLASHINFERPAPER：FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](../sources/S-FLASHINFERPAPER.md)：固定 v1 描述 BSR/ragged/可组合 KV 表示、attention 状态合并、JIT 变体、动态负载计划与 CUDA Graph workspace，以及 GQA 和 FP8 KV 路径。
  [完整原文](../../references/S-FLASHINFERPAPER/reader.html) · [全文文本](../../references/S-FLASHINFERPAPER/fulltext.txt)

## 延伸阅读

- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [交叉注意力缓存：固定源、独立几何与可见性](../algorithms/cross-attention-cache.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [MLA：压缩状态、矩阵吸收与解耦 RoPE](../models/mla.md)。
