# S-GQA：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

## 原文

[原始 PDF](../../references/S-GQA/original.pdf) · [完整原文](../../references/S-GQA/reader.html) · [全文文本](../../references/S-GQA/fulltext.txt)

固定版本：2305.13245v3。来源版本日期：2023-12-23；整理核对日期：2026-10-27。可直接阅读全文；下面是本库核对后抽出的机制与结论。

## 机制

多头注意力有 H 个 query、key、value 头；多查询注意力让所有 query 头共享一个 key/value 头；分组查询注意力把 query 头分成 G 组，每组共享一个 key/value 头。GQA-1 等价于 MQA，GQA-H 等价于 MHA。从多头检查点转换时，一组内各 head 的 key/value 投影矩阵取平均（mean pooling）合成该组的单一头，再用原预训练配方的一小部分步数（论文用 5%）继续训练。

论文给出选择组数的理由：decode 的瓶颈是每步加载 key 与 value，MQA 把这部分带宽与容量削到最低，但大模型的 head 数多，MQA 是过激的削减；GQA 让带宽与容量按模型规模等比例缩减。大模型相对受 KV 带宽影响更小，因为 KV 随模型维度增长而 FLOPs 与参数随维度平方增长。GQA 不用于 encoder 自注意力，因为那里表示并行计算，带宽不是瓶颈。

## 结论

T5-XXL 上，5% 预训练计算量的继续训练得到的 GQA 质量接近 MHA、速度接近 MQA；MQA 质量明显下降。

## 本库据此推导

- Hq、Hkv 与组数 g = Hq/Hkv 是模型语义的一部分，决定 KV 容量与每步读取量；kernel 是否真正只读一份共享 K/V 而不物化扩展，是另一个独立事实，见 [MHA、GQA 与 MQA](../models/gqa-kv-heads.md) 与 [Attention 执行接口](../frameworks/attention-backends.md)。
- 张量并行下 KV 头如何分到各 rank、非整除时怎样映射，由 g 决定，见 [推理并行](../algorithms/parallelism.md)。

## 适用边界

实验是 T5 编码器-解码器族；组数与质量的关系随模型变化。版本日期不证明模型训练实际包含该资料。
