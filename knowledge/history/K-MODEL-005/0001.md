# K-MODEL-005：Encoder-decoder：源表示、两套 Attention 与流式边界

Encoder-decoder 将源编码和目标生成分开；固定源复用、流式追加和目标自回归具有不同的依赖结构。

## 三段计算

输入首先经过 modality processor 与 encoder，得到源表示 E。decoder 处理目标提示与生成前缀：self-attention 读取目标历史，cross-attention 读取 E 或其各层 K/V 投影。最后输出头和任务 token 规则决定文本、时间戳或其他符号。

encoder 可以是双向，decoder self-attention 通常因果，cross 可见性由任务定义。这使“所有 attention 都用 causal”成为结构性错误。源长度 S 与目标长度 T 独立，batch 中的 padding/mask 也应分别表达。

## 一次编码为何还不够描述缓存

缓存 E 可以避免重跑 encoder，但如果每个 decoder 步仍重复计算 EWₖ、EWᵥ，源投影成本仍存在。保存每层 cross KV 则减少投影，增加约 L 倍的状态规模。源编码时间、cross 投影时间、目标 self KV、cross KV 容量四项应分别建模。

不同请求源长差异大时，encoder 适合按长度或分辨率聚批；decode 适合连续批。让两段共享完全相同的 batch 队列可能引入队头阻塞。可以将源表示作为阶段间产物，但必须附上请求身份、源版本、有效长度和模态信息。

## 离线切块与真正流式

离线模型对完整输入编码后生成；将长音频分成独立小段，是另一种任务处理策略，会改变上下文和边界。真正流式模型通常定义可提交块、lookahead 和状态延续。若 encoder 是双向，新帧可能改变旧帧表示，先前 cross KV 就不能视为不可变前缀。

例如一段 20 秒音频被切成两个 10 秒窗口，重复上下文、时间戳偏移、说话者连续性和拼接规则都进入输出语义。每块识别结果正确不自动保证拼接后文本和时间对齐正确。模型提供的 streaming 或 chunked attention 规则决定可以保留哪些状态。

## 服务目标从用户时间尺度推导

ASR 可以关注最终转写延迟、首个稳定文本延迟、实时率和词错误率。实时率是处理耗时/音频时长，低于 1 说明平均可赶上输入，却不保证首段延迟小。若模型等待固定大窗口才编码，即使计算很快也可能交互迟钝。调度应区分采集等待、编码、解码和稳定结果发布。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TRANSFORMER：Attention Is All You Need](../sources/S-TRANSFORMER.md)：Transformer 的 attention 用 query 与 keys 的分数生成归一化权重，再对 values 加权；encoder-decoder attention 的 Q 来自 decoder，K/V 来自 encoder 表示。
  [完整原文](../../references/S-TRANSFORMER/reader.html) · [全文文本](../../references/S-TRANSFORMER/fulltext.txt)

- [S-WHISPER：Robust Speech Recognition via Large-Scale Weak Supervision](../sources/S-WHISPER.md)：Whisper 采用语音 encoder-decoder 和任务条件 token。音频编码与目标文本自回归是不同阶段，decoder 使用源表示完成条件生成。
  [完整原文](../../references/S-WHISPER/reader.html) · [全文文本](../../references/S-WHISPER/fulltext.txt)

- [S-MOONSHINE：Moonshine: Speech Recognition for Live Transcription and Voice Commands](../sources/S-MOONSHINE.md)：Moonshine 面向语音识别的 encoder-decoder 结构和变长音频处理，源音频表示与目标文本生成具有不同长度和计算阶段。
  [完整原文](../../references/S-MOONSHINE/reader.html) · [全文文本](../../references/S-MOONSHINE/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [交叉注意力缓存：固定源、独立几何与可见性](../algorithms/cross-attention-cache.md)。
- [语音生成：Codec Token、声学块与播放时钟](../models/speech-generation.md)。
- [输入输出语义：Tokenizer、Chat Template 与流式文本](../system-semantics/tokenization-io.md)。
