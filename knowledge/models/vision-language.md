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

## 媒体、视觉和语言的三段容量

媒体阶段负责取得内容、解码、方向/色彩处理、resize/crop、归一化和帧选择；视觉阶段处理 patch 与空间/时间位置，再生成桥接特征；语言阶段按实际模态顺序组成 prompt，执行 prefill 和 decode。每个阶段的产物与工作区尺寸不同，固定“最多几张图”不足以约束峰值。

缓存原始媒体、processor 张量、视觉 encoder 输出、桥接特征或完整 decoder 前缀，各有不同命中条件和容量收益。只改变文本提问时，固定媒体的视觉特征可能继续可用，但 decoder 前缀是否相同还取决于模板、位置和此前文本。改变 resize、抽帧或图像内容后，旧特征即使形状相同也不能继续作为命中。

多阶段调度既要避免媒体获取阻塞就绪 GPU 任务，也要避免视觉 encoder 无限制领先而积累特征。队列以待处理像素、视觉 tokens、特征 bytes 和目标 KV 共同计量。高分辨率图在视觉全 attention 层可能昂贵，短文本请求不能因为图像张数相同就被分入同一成本桶。

## 场景：两种缓存与三阶段队列

采用 Qwen2.5-VL v1 报告中的 14 像素 patch、相邻 2×2 patch 合并，以及其 7B 配置的 28 层、4 KV heads、head_dim=128，构造以下资源算例；处理耗时和服务预算均为假设。处理后图像为 448×672，产生 32×48=1536 个视觉 patch，合并后为 384 个 decoder 视觉 token。512 个文本 token 与这些特征形成 896 个输入 token；另预留 256 个输出 token。忽略特殊 token 与页尾时，每 token KV 为 2×28×4×128×2=56 KiB，输入 KV 为 49 MiB，完整预留为 63 MiB/request。

视觉 patch 的 1280 维 BF16 特征为 3.75 MiB，合并后 3584 维特征为 2.625 MiB；FP32 三通道像素为约 3.445 MiB。若三个对象同时存在，仅这部分就约 9.820 MiB，另有权重、层中间量和媒体解码缓冲。8 个请求完整 KV 主项为 504 MiB；排队 20 份合并特征另占 52.5 MiB。以上不是整模型峰值显存估计，而是说明各阶段不能漏账。

候选 A 每次从媒体完整计算；B 缓存合法的合并视觉特征并保留阶段独立队列；C 缓存完整可共享的 decoder 前缀并使用亲和路由。设一次冷路径的媒体获取 20 ms、预处理 15 ms、视觉 encoder 35 ms、桥接 5 ms、语言 prefill 60 ms、所有队列合计 25 ms、首内容发送 5 ms，则 TTFT 为 165 ms。若 B 的特征查找/恢复总计 8 ms，且它消除预处理、视觉与桥接这 55 ms，其他项不变，则为 118 ms；若为判断内容身份仍须获取媒体，该 20 ms 仍保留。缓存成本如何计入必须说明，不能同时声称免获取又不验证内容版本。

C 还可能省 prefill，却会占更大的 KV 并产生路由热点；当同一图像配不同前置文本时，视觉特征可复用不代表完整前缀可复用。B 的主机特征若频繁搬入 GPU，恢复队列和带宽需与冷算比较。高命中且媒体稳定时 B/C 有机会；高分辨率但低重用时，应先改进预处理和视觉计算；热点排队超过省下的计算时，重新计算或复制缓存又可能更快。

验证分别检查媒体顺序、resize/crop 坐标反映射、patch 网格、桥接特征、MRoPE 时间/空间分量以及 decoder logits。质量用图文对应、OCR、计数和定位等与任务相关的标注，不以语言流畅性代替视觉正确性；定位结果需回到原图坐标再算 IoU 或点命中。相同图片不同处理配置、相同 URL 不同内容、两图互换、不同抽帧时间和取消中的迟到特征都进入组合验证。性能报告冷/热命中比例及每阶段排队、峰值和端到端 TTFT/ITL。

<!-- infraagents:provenance -->

## 来源与推导

- [S-QWEN-VL：Qwen2.5-VL Technical Report](../sources/S-QWEN-VL.md)：Qwen2.5-VL 报告讨论动态视觉分辨率、窗口视觉编码及时间相关的多模态位置，视觉输入具有独立的网格和时间几何。
  [完整原文](../../references/S-QWEN-VL/reader.html) · [全文文本](../../references/S-QWEN-VL/fulltext.txt)

- [S-TRANSFORMER：Attention Is All You Need](../sources/S-TRANSFORMER.md)：Transformer 的 attention 用 query 与 keys 的分数生成归一化权重，再对 values 加权；encoder-decoder attention 的 Q 来自 decoder，K/V 来自 encoder 表示。
  [完整原文](../../references/S-TRANSFORMER/reader.html) · [全文文本](../../references/S-TRANSFORMER/fulltext.txt)

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：固定文档说明逐层复用历史 K/V、计算并合并新增 K/V，以及增量循环的 attention mask 应覆盖历史与新增 token；并区分 Dynamic、Static、StaticSlidingWindow 等缓存存储形态。
  [完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [交叉注意力缓存：固定源、独立几何与可见性](../algorithms/cross-attention-cache.md)。
- [输入输出语义：Tokenizer、Chat Template 与流式文本](../system-semantics/tokenization-io.md)。
- [工作负载分布：长度、复用、模态与相关性](../workloads/input-distribution.md)。
