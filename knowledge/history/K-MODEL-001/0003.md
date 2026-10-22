# K-MODEL-001：从模型计算图推导 Prefill、Decode 与持久状态

模型接入首先是一项计算图和状态语义工作：识别哪些量随 token、位置、请求和阶段变化，再选择执行机制。

## 自回归前向实际计算什么

给定前缀 x₀…xₙ₋₁，因果模型在各位置产生下一 token 的条件分布。prefill 同时处理已知输入，最后有效位置的 logits 用来选择第一个生成 token。decode 将上一次选出的 token 作为新输入，生成它的各层状态与下一个分布。输入 token、输出 token、KV 写入位置三者因此存在一步关系。

在普通 decoder Transformer 中，历史各层 K/V 可以保留；新 query 对它们读取。新 token 的隐藏表示还经过归一化、输出投影、残差和 FFN。缓存只消除了重复历史前向，并没有消除所有历史依赖。长上下文的 attention 工作仍随历史增长。

## 用算术强度解释两个阶段

对 dᵢ×dₒ 的权重，M 行输入的 GEMM 主要工作量约 2Mdᵢdₒ FLOPs。若权重只读一遍且忽略激活流量，算术强度约 2M/b，其中 b 为权重字节。M 小时难以摊薄权重读取，M 大时可提高复用；实际还受 tile、缓存与并发影响。

prefill 的 M 来自多个已知输入位置，decode 的 M 主要来自并发请求或并行验证位置。这解释了为什么小 batch decode 常受带宽与 launch 限制，而 prefill 更容易利用矩阵计算。但长上下文 attention、大 batch decode、MoE 和多模态 encoder 都能改变这一判断，阶段名称只是线索，张量形状才是依据。

## 模型档案应该还原哪些语义

| 结构维度 | 对系统的直接影响 |
|---|---|
| 层类型序列 | 全 KV、窗口 KV、递归状态分别管理 |
| Q/K/V heads、通道和 dtype | 缓存容量、投影宽度与 backend 适配 |
| 位置编码、mask 与特殊 token | 缓存等价性和增量前向是否成立 |
| FFN/专家结构 | 激活量、权重驻留、通信分布 |
| 编码器与模态桥接 | 前置计算、固定源状态和多阶段排队 |
| 输出头与采样协议 | logits 大小、取样位置及终止行为 |

配置文件给出参数，forward 代码和权重形状给出实现事实，tokenizer/processor 定义输入语义。三者应一致。例如配置说 8 KV heads，但权重加载后按 32 heads 展开，会改变真实容量；模型名字含“流式”也不足以证明每个 encoder 都因果。

## 具体家族作为阅读入口

Llama 类稠密 decoder 是理解 GQA、RoPE、SwiGLU 和分页状态的入口；Mistral 7B 提供窗口注意力例子；DeepSeek-V2 提供 MLA 与 MoE 的组合；Mamba 提供递归状态；Whisper/Moonshine 提供语音 encoder-decoder；Qwen2.5-VL 与 Omni 展示模态编码和时间对齐；DiT 类扩散展示按去噪步反复更新全局状态。家族名不替代具体 checkpoint 配置，同一家族不同版本也可能改变层组合。

从上述档案可以推导最小执行链：规范化输入 → 特征/嵌入 → 初始状态 → 增量执行或迭代求解 → 输出解释。每个箭头都对应具体数据形状和所有权。这比先选一个引擎再猜模型能否放入其接口更有解释力。

## 从固定制品还原最小 forward

接入从一份固定版本档案开始：模型配置、权重分片、tokenizer/processor、特殊 token 与生成配置分别绑定身份。[权重加载](../frameworks/checkpoint-loading.md) 将逻辑参数映射到执行布局；在并入服务调度前，先建立单请求、固定输入、无状态复用的 forward，再增加增量状态。这样可以区分“模型函数未还原”和“缓存或调度改变了函数”。

对一层常见的稠密 decoder，设有效 token 数 M、hidden 宽 H、Q head 数 Hq、KV head 数 Hkv、head 维 d、FFN 中间宽 I。Q/K/V 投影输出分别为 M×Hq×d、M×Hkv×d、M×Hkv×d；attention 输出通常还要通过输出投影回到 residual 形状。门控 FFN 的两个输入投影各为 M×I，逐点门控后再回到 M×H。上述是结构设定，不代表所有模型都满足 H=Hq·d、使用门控 FFN 或采用同一种 norm 顺序。

归一化在 residual 前还是后、是否有 bias、RoPE 作用的分量、logits 是否另作缩放，都应从固定 forward 与权重核对。先用少量具体张量检查轴排列，再用每层 hidden 和 logits 对照；只看模型最终能产生自然语言不能证明接入正确。[基础算子](../algorithms/operator-primitives.md) 给出这些接口的独立关系。

## 模型结构与状态的固定来源坐标

| 固定资料版本 | 结构入口 | 接入时特别核对 |
|---|---|---|
| Transformer 1706.03762v7 | encoder/decoder 与自注意力、交叉注意力 | 源与目标 mask、位置和不同投影的轴 |
| Mistral 7B 2310.06825v1 | GQA 与窗口注意力 | KV head、窗口可见集合与逻辑位置 |
| DeepSeek-V2 2405.04434v5 | MLA 与 MoE | latent/位置分量、专家参数及投影吸收条件 |
| Mamba 2312.00752v2 | 选择性状态空间模型 | recurrence 与卷积状态、更新步和恢复位置 |
| Whisper 2212.04356v1 | 音频 encoder-decoder | 特征窗口、固定源表示、cross KV 和 decoder 历史 |
| Qwen2.5-VL 2502.13923v1 | 视觉编码与语言生成 | processor、视觉 token 几何及位置对齐 |

这张表固定的是研究入口，不是某个未下载 checkpoint 的参数清单，也不声称同名家族的所有发行版相同。具体 Task 仍要把实际制品中的数值和层序列写入模型档案。

## 持久状态、临时激活与输出提交

普通 decoder 的全层 KV 容量可按 2·L·Hkv·d·s·N 估算，其中 L 为层数，s 为每元素字节，N 为实际计算的历史长度；量化元数据、页尾和并行副本另计。临时激活随 M、FFN 宽度和 attention 实现变化，不应重复当作每请求长期 KV。共享参数和可重用前缀又应按物理唯一对象计数。

设 prompt 已计算 100 个位置并从末行 logits 选出 token y。此时用户可收到 1 个生成 token，逻辑序列总长度为 101，但普通逐步路径的 KV 有效长度仍为 100，若按零基位置记则覆盖 0–99。下一步消费 y 后才写出它在位置 100 的状态，再选择下一个 token。前缀插入、迁移、会话延续和 usage 必须分别记录输入长度、生成输出数、逻辑序列总长度与已计算状态长度，不能由同一个计数器含混表示。

增量前向的验证从同一固定前缀出发，对照整段计算、分块计算和逐 token 计算。再改变 batch 排列与物理状态布局而保持逻辑输入，检查输出是否维持所要求的数值关系。若模型是双向 encoder、扩散迭代或图模型，应重新定义可复用状态与推进单位；它们的单次 forward 不天然具有 causal decode 的增量等价性。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TRANSFORMER：Attention Is All You Need](../sources/S-TRANSFORMER.md)：Transformer 的 attention 用 query 与 keys 的分数生成归一化权重，再对 values 加权；encoder-decoder attention 的 Q 来自 decoder，K/V 来自 encoder 表示。
  [完整原文](../../references/S-TRANSFORMER/reader.html) · [全文文本](../../references/S-TRANSFORMER/fulltext.txt)

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：Transformers 的缓存说明区分已存历史 K/V、新输入、attention mask 和 cache position。增量前向需要让新 token 的位置与已有缓存长度一致。
  [完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

- [S-SARATHI：SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](../sources/S-SARATHI.md)：SARATHI 将长 prefill 切块，并在合适的批处理中与 decode 组织在一起，以减少两类工作之间的干扰并利用计算资源。
  [完整原文](../../references/S-SARATHI/reader.html) · [全文文本](../../references/S-SARATHI/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-MISTRAL：Mistral 7B](../sources/S-MISTRAL.md)：Mistral 7B 提供滑动窗口 attention 的具体模型例子：局部可见历史与滚动状态可以降低相关层的直接历史成本。
  [完整原文](../../references/S-MISTRAL/reader.html) · [全文文本](../../references/S-MISTRAL/fulltext.txt)

- [S-MLA：DeepSeek-V2：MLA 与解耦位置编码](../sources/S-MLA.md)：DeepSeek-V2 的 MLA 以低维 latent 联合表示内容 K/V，通过线性投影的吸收减少历史展开；位置相关旋转不能任意与投影交换，因此采用解耦 RoPE 分量。
  [完整原文](../../references/S-MLA/reader.html) · [全文文本](../../references/S-MLA/fulltext.txt)

- [S-MAMBA：Mamba: Linear-Time Sequence Modeling with Selective State Spaces](../sources/S-MAMBA.md)：Mamba 使用输入相关的选择性状态空间机制。递归生成保留有限大小的状态，长序列处理可以采用专门的扫描/分块执行方式。
  [完整原文](../../references/S-MAMBA/reader.html) · [全文文本](../../references/S-MAMBA/fulltext.txt)

- [S-WHISPER：Robust Speech Recognition via Large-Scale Weak Supervision](../sources/S-WHISPER.md)：Whisper 采用语音 encoder-decoder 和任务条件 token。音频编码与目标文本自回归是不同阶段，decoder 使用源表示完成条件生成。
  [完整原文](../../references/S-WHISPER/reader.html) · [全文文本](../../references/S-WHISPER/fulltext.txt)

- [S-QWEN-VL：Qwen2.5-VL Technical Report](../sources/S-QWEN-VL.md)：Qwen2.5-VL 报告讨论动态视觉分辨率、窗口视觉编码及时间相关的多模态位置，视觉输入具有独立的网格和时间几何。
  [完整原文](../../references/S-QWEN-VL/reader.html) · [全文文本](../../references/S-QWEN-VL/fulltext.txt)

## 延伸阅读

- [MHA、GQA 与 MQA：头映射和带宽复用](../models/gqa-kv-heads.md)。
- [SSM 与混合模型：递归状态、扫描与恢复](../models/recurrent-hybrid-state.md)。
- [MLA：压缩状态、矩阵吸收与解耦 RoPE](../models/mla.md)。
- [输入输出语义：Tokenizer、Chat Template 与流式文本](../system-semantics/tokenization-io.md)。
