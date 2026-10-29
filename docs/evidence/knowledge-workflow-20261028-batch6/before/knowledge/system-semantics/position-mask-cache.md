# K-SYS-003：位置、Mask 与缓存：同一可见性关系的三种表示

缓存中的内容必须对应模型定义的逻辑位置；mask 决定读取集合，物理布局只决定这些内容放在哪里。

## 从绝对位置定义因果关系

因果 attention 的合法关系是 key 的逻辑位置不晚于 query。矩阵中的行列编号只有在 Q/K 起点一致时才直接代表这一关系。Q 是历史尾部的一小段而 K 包含全历史时，简单上三角或下三角布尔构造可能错位。

缓存推进或写入位置、模型的位置编码输入、padding 的有效范围以及 attention 可见性，是需要分别建模的关系。某些实现使用 cache_position、position_ids 等字段；具体维度、是否计入 padding 及更新方式要按 Task 固定模型与框架核对，不能从字段名推断它们等于逻辑长度或物理槽号。本库对这些关系作机制推导，未将字段合同归因于只说明历史 K/V 与 mask 的缓存概览。左 padding、前缀共享、滑动窗口和多模态会使这些关系进一步分离。

## 一个位置算例

历史有 5 个有效 token，本轮 query 长 2，其逻辑位置为 5、6，最终 K/V 长 7。第一行可见位置 0–5，第二行可见 0–6。若物理张量左侧有 3 个 padding，tensor 列号还要加偏移，但 RoPE 的逻辑位置不必跟着多加 3。明确逻辑→物理映射能直接得出正确 mask。

## KV 内容是否已经带位置

很多 RoPE 模型缓存旋转后的 K，另一些实现可能缓存未旋转内容并在使用时处理。两种表示需要不同读写约定。已经旋转的 K 再旋转一次会改变函数；窗口环形槽覆盖后沿用旧位置也会错误。prefix identity 应包含影响旋转的配置与起点。

## 用等价分区形成验证依据

对同一输入，整段前向、分两块 prefill、逐 token decode、前缀命中后继续应得到规定容差内的同一结果。它们改变执行分区而保留可见性，是很有解释力的变形关系。跨页、左 padding、窗口 rollover、cache 命中非整页和非等长 Q/K 是关系发生变化的边界。

SSM、cross-attention 和扩散需要各自的可见性定义，不能硬套同一个因果三角。系统统一的部分是显式表达逻辑依赖，而不是统一一张 mask。

## 接入时同时固定语义与表示

模型配置、checkpoint 布局、位置规则、mask 定义和缓存格式构成一条共同契约。权重名称相同并不能证明 Q/K head 的排列一致；张量形状相同也不能证明旋转维度、位置缩放或 KV 量化的分组轴一致。[权重加载](../frameworks/checkpoint-loading.md) 建立文件到运行布局的映射，本页进一步约束运行布局中的位置解释。

| 接入变化 | 必须保持或显式改变的关系 | 直接的区分输入 |
|---|---|---|
| 左 padding 或压紧 batch | 有效 token 的逻辑位置与请求归属 | 同一输入分别独立、补齐、混批执行 |
| 分块 prefill 或前缀命中 | 每个 query 能看到的历史与当前位置 | 非等长 Q/K，命中长度不在整页边界 |
| RoPE 或缩放配置变化 | K 的旋转状态、位置坐标和相关身份 | 非零起点、长位置及分块接续 |
| KV 量化或布局转换 | 分组轴、scale、有效范围与解码精度 | 组边界、尾组、常数通道和极值 |
| 混合状态或窗口覆盖 | 每个状态族的共同恢复位置 | 窗口换槽后迁移，再继续多个 token |

这些不是要求所有组合位级相同。不同归约顺序和低精度路径可能产生允许的数值差异，应先固定参考函数、容差和质量条件；数值差异不能用来掩盖读到未来 token、其他请求或错误位置的语义错误。

## 表示等价不意味着缓存可以直接互换

如果一个后端保存旋转后 K，另一个保存旋转前 K，跨后端转移需要明确是否存在受支持的转换及其数值误差。若只留下量化后的状态，往返转换通常不能恢复原高精度值。含滑动窗口或递归状态的模型还可能已经丢失重建所需历史，不能仅凭逻辑 token 数相等宣称两个缓存等价。

量化元数据、窗口起点和混合状态各自的有效范围应随缓存身份一起解释。页表只回答物理位置，不替代这些内容。例如同一物理槽先属于绝对位置 17，后属于位置 49，环形索引相同也不允许消费者把旧位置标签沿用到新内容。

## 将局部不变量连接到输出检查

先在小输入上逐项核对：有效 token 集合、逻辑位置、允许访问的 K 范围、实际写入槽和每种状态的恢复点。再比较相同输入的整段、分块、命中后续接、迁移后续接和取消后重新执行。所有参考应避免复用被测实现的同一页表或 mask 构造错误。

一个缓存系统只在 prefill 首 token 上对齐仍不足够：该 token 可能尚未消费压缩或迁移后的历史。继续执行至少覆盖实际读取新表示的阶段，并记录首次发生分歧的层、位置和状态族。对于近似注意力或裁剪，还必须说明参考函数已经变化，按任务质量评估，不能要求原精确 attention 的分区等价关系无条件成立。

<!-- infraagents:provenance -->

## 来源与推导

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：固定文档说明逐层复用历史 K/V、计算并合并新增 K/V，以及增量循环的 attention mask 应覆盖历史与新增 token；并区分 Dynamic、Static、StaticSlidingWindow 等缓存存储形态。
  [完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

- [S-TRANSFORMER：Attention Is All You Need](../sources/S-TRANSFORMER.md)：Transformer 的 attention 用 query 与 keys 的分数生成归一化权重，再对 values 加权；encoder-decoder attention 的 Q 来自 decoder，K/V 来自 encoder 表示。
  [完整原文](../../references/S-TRANSFORMER/reader.html) · [全文文本](../../references/S-TRANSFORMER/fulltext.txt)

- [S-SDPA：PyTorch：scaled_dot_product_attention](../sources/S-SDPA.md)：已核对 SDPA 文档：布尔 mask 的 true 表示参与 attention；dropout_p 大于零时执行 dropout；非方形 is_causal 采用文档定义的左上对齐。Q、K/V heads 与通道形状有明确关系。
  [算法与机制说明](../sources/S-SDPA.md)

- [S-FA-IMPL：FlashAttention 官方实现说明](../sources/S-FA-IMPL.md)：FlashAttention 的版本说明记录：2.1 将非等长 Q/K 的 causal 对齐由左上改成右下。示例 Q 长 2、K 长 5 时，右下对齐使两行分别可见前 4、5 个位置。
  [完整原文](../../references/S-FA-IMPL/reader.html) · [全文文本](../../references/S-FA-IMPL/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-ROPE：RoFormer: Enhanced Transformer with Rotary Position Embedding](../sources/S-ROPE.md)：§3.1–3.3 以二维旋转及分量对推广建立位置编码和相对位置内积关系。
  [完整原文](../../references/S-ROPE/reader.html) · [全文文本](../../references/S-ROPE/fulltext.txt)

- [S-KIVI：KIVI](../sources/S-KIVI.md)：§3 分析 K/V 的不同分组轴，使用分组量化与高精度残余区处理流式追加；§4 给出模型和任务相关的评估条件。
  [完整原文](../../references/S-KIVI/reader.html) · [全文文本](../../references/S-KIVI/fulltext.txt)

## 延伸阅读

- [分块 Prefill：服务时间切片与历史可见性](../algorithms/chunked-prefill.md)。
- [窗口与稀疏 Attention：可见性、位置和状态保留](../models/sliding-window-attention.md)。
- [Attention Backend：语义、布局与执行契约的交集](../frameworks/attention-backends.md)。
