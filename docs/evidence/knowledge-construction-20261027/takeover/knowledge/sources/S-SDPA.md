# S-SDPA：PyTorch：scaled_dot_product_attention

## 原文

[完整原文](../../references/S-SDPA/reader.html) · [全文文本](../../references/S-SDPA/fulltext.txt)

固定版本：Git 647b6a2ff9a83a886464932dfad78781ceb61cb6 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-27。可直接阅读全文；下面是本库核对后抽出的定义与关系。

## 这个算子定义了什么

文档用一段参考实现给出函数的精确含义，可以直接作为任何自写 attention 的数学参考：

1. L 为 query 长度，S 为 key 长度；缩放因子默认 1/√E，E 为 query 最后一维。
2. 偏置矩阵 attn_bias 初始为 L×S 的零矩阵。因果模式下取 L×S 的下三角（主对角线及以下）为可见，其余置 −∞；这意味着非方形时第 i 行可见前 i+1 个 key，即左上对齐。
3. 若给了 mask：布尔 mask 为 true 的位置参与计算，false 置 −∞；浮点 mask 直接加到偏置上。
4. 若开启分组查询：key、value 沿 head 轴按 Hq/Hkv 倍重复展开。
5. 权重 = softmax(QKᵀ·scale + bias)，经 dropout 后乘 V。

文档同时说明三件独立于数学定义的事：实际执行由三种实现之一完成（融合的 FlashAttention 路径、内存高效路径、与上述定义等价的朴素实现），选择由输入条件决定，融合路径各有输入限制，不可用时回退并给出原因；分组查询目前只在部分路径上支持，并要求 Hq 能被 Hkv 整除且 Hk = Hv；浮点融合可能让输出与朴素实现有数值差异。

## 本库据此推导

- 参考实现里的 repeat 展开是数学等价表达，不是融合路径一定物化扩展的证据；"接口语义"与"实际分派"是两层，见 [Attention 执行接口](../frameworks/attention-backends.md)。
- 因果左上对齐与另一类实现的右下对齐（见 [S-FA-IMPL](S-FA-IMPL.md)）是同名参数含义不同的具体反例；可见集合只能由绝对位置定义。
- 布尔 mask 的极性（true 表示参与）与浮点 mask 的加法语义，是适配层必须保持的契约，见 [位置、Mask 与缓存](../system-semantics/position-mask-cache.md)。
- 同一提交的 functional 接口还给出线性层权重按 N×K 存放并在算子内转置的约定，见 [基础算子](../algorithms/operator-primitives.md)。

## 适用边界

这是一个框架在一个提交上的定义；其他框架或库的同名参数不必相同。接入该版本时按归档原文核对。设计推导为独立分析，不是目标机器实测；版本日期不证明模型训练实际包含该资料。
