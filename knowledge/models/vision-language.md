# K-MODEL-008：视觉语言模型：Patch、模态桥接与联合位置

VLM 的真实输入规模由处理器生成的视觉 token 决定，文本长度只描述其中一部分。

## 两条主要接入路径

一种路径把视觉 encoder 的特征经 projector 变成语言模型维度，与文本 embedding 一起进入 decoder self-attention。另一种让语言 query 通过 cross-attention 读取视觉源。前者增加 decoder 序列长度及 KV；后者增加独立 context 状态及 cross 计算。两者对缓存容量和 prefix identity 的要求不同。

processor 决定 resize、crop、patch、归一化以及特殊 token 放置。动态分辨率可能让同一张图产生不同 patch 网格。视觉 token 数由输入尺寸、patch 大小、时空压缩与 merge 规则推导，不能只用图片张数衡量负载。

## 一次输入的几何账本

设图像经处理后为 H×W，patch 边长 p，不考虑裁剪和合并时约有 (H/p)(W/p) 个 patch。若空间分辨率每边增大 2 倍，patch 数增 4 倍，密集视觉 self-attention 的成对工作可增到 16 倍。窗口视觉 attention 则改变该增长规律。视频还多一个时间轴，抽帧率进入成本。

多模态位置可能有时间、高度、宽度多轴。MRoPE 类方案把不同通道分配给不同位置维，后续文本位置的推进规则由模型定义。把所有视觉 token 简单按单维文本位置编号可能保住形状却改变函数。

## Batch 与缓存

视觉 encoder 可按网格/分辨率分桶，decoder 可按剩余 token 工作聚批。阶段间特征需携带请求、媒体顺序、网格和位置元数据。多图输入的两个 image placeholders 与两个特征段对应，不能在压紧时交换顺序。

缓存视觉特征要求源内容和 processor 配置一致；缓存 decoder 前缀还要包含视觉特征身份、桥接参数及位置。图片 URL 相同不代表内容未变化，文本 placeholder 相同更不能作为视觉内容 key。缓存不同层产物的有效身份由各自依赖闭包决定。

## 从结构推导服务指标

首 token 路径包含媒体获取/解码、processor、视觉 encoder、桥接和语言 prefill。长图片任务的瓶颈可能在 CPU resize，也可能在视觉 attention；只看语言 decode tokens/s 会遗漏大部分工作。质量验证同时需要视觉定位、图文对应与文本生成，纯文本回归不能覆盖模态桥接。

<!-- infraagents:provenance -->

## 来源与推导

- S-QWEN-VL：[完整原文](../../references/S-QWEN-VL/reader.html) · [全文文本](../../references/S-QWEN-VL/fulltext.txt)
- S-TRANSFORMER：[完整原文](../../references/S-TRANSFORMER/reader.html) · [全文文本](../../references/S-TRANSFORMER/fulltext.txt)
- S-HF-CACHE：[完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

- [S-QWEN-VL：Qwen2.5-VL Technical Report](../sources/S-QWEN-VL.md)：Qwen2.5-VL 报告讨论动态视觉分辨率、窗口视觉编码及时间相关的多模态位置，视觉输入具有独立的网格和时间几何。

- [S-TRANSFORMER：Attention Is All You Need](../sources/S-TRANSFORMER.md)：Transformer 的 attention 用 query 与 keys 的分数生成归一化权重，再对 values 加权；encoder-decoder attention 的 Q 来自 decoder，K/V 来自 encoder 表示。

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：Transformers 的缓存说明区分已存历史 K/V、新输入、attention mask 和 cache position。增量前向需要让新 token 的位置与已有缓存长度一致。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 3，2026-09-20。

## 延伸阅读

- [交叉注意力缓存：固定源、独立几何与可见性](../algorithms/cross-attention-cache.md)。
- [输入输出语义：Tokenizer、Chat Template 与流式文本](../system-semantics/tokenization-io.md)。
- [工作负载分布：长度、复用、模态与相关性](../workloads/input-distribution.md)。
