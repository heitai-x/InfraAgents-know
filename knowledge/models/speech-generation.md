# K-MODEL-011：语音生成：Codec Token、声学块与播放时钟

语音服务需要把模型生成时钟转换为音频播放时钟，首包和稳定供给由多个阶段共同决定。

## 常见生成链的状态

文本或语义条件驱动声学 token 生成，codec/vocoder 将其变成波形。不同模型可能使用自回归 codec、并行多个 codebook、扩散或 flow 解码。一个声学 token 不一定对应一个固定文本 token，也不一定能立即独立解码成可播放样本。

codec 的帧率、codebook 数、采样率和 chunk overlap 定义输出几何。例如声学帧率 50 Hz 时，10 帧代表 200 ms 音频；模型生成它们花费 40 ms，局部生成实时率为 0.2，但首包还要等待足够的声学上下文和 vocoder 处理。这个算例解释吞吐充裕仍可能首包很慢。

## 流式解码的边界

若 vocoder 需要右侧 lookahead，最前面一段样本只有拿到未来若干帧后才稳定。相邻 chunk 可能保留重叠用于衔接，发给客户端时必须裁去重复部分并维持累计样本位置。不能把网络 chunk 数当声学帧数，也不能在 EOS 前后任意补静音而不改变任务。

自回归声学分支的 KV、codec 状态、波形 overlap 与文本对齐进度组成请求状态。取消应停止后续生成并释放这些对象，但在途 audio chunk 仍可能已经送入播放队列。用户“停止说话”的体验因此还受客户端缓冲长度影响。

## 用缓冲模型推导服务目标

设客户端缓冲中尚未播放音频时长为 B(t)，网络可播放音频到达速率为 a(t)，正常播放消耗速率为 1，则缓冲变化近似 dB/dt＝a(t)−1。平均 a＞1 不排除瞬时断供；B 太小易卡顿，太大增加可感知延迟和取消迟滞。

服务目标可同时约束首段音频时间、实时率、缓冲欠载频率和最终质量。减少 chunk 大小降低初始等待，却增加调度、封包和 vocoder 边界成本。不同语言、语速与停顿分布会改变声学输出长度，负载模型应直接统计音频时长和块分布。

<!-- infraagents:provenance -->

## 来源与推导

- S-OMNI：[完整原文](../../references/S-OMNI/reader.html) · [全文文本](../../references/S-OMNI/fulltext.txt)
- S-MOONSHINE：[完整原文](../../references/S-MOONSHINE/reader.html) · [全文文本](../../references/S-MOONSHINE/fulltext.txt)
- S-WHISPER：[完整原文](../../references/S-WHISPER/reader.html) · [全文文本](../../references/S-WHISPER/fulltext.txt)

- [S-OMNI：Qwen2.5-Omni Technical Report](../sources/S-OMNI.md)：Qwen2.5-Omni 的报告描述 Thinker/Talker 分支、块式模态输入、时间对齐和流式音频输出，文本与音频不是同一执行时钟。

- [S-MOONSHINE：Moonshine: Speech Recognition for Live Transcription and Voice Commands](../sources/S-MOONSHINE.md)：Moonshine 面向语音识别的 encoder-decoder 结构和变长音频处理，源音频表示与目标文本生成具有不同长度和计算阶段。

- [S-WHISPER：Robust Speech Recognition via Large-Scale Weak Supervision](../sources/S-WHISPER.md)：Whisper 采用语音 encoder-decoder 和任务条件 token。音频编码与目标文本自回归是不同阶段，decoder 使用源表示完成条件生成。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 3，2026-09-20。

## 延伸阅读

- [Encoder-decoder：源表示、两套 Attention 与流式边界](../models/encoder-decoder-streaming.md)。
- [服务指标：事件边界、Token 间隔与用户时间](../workloads/metric-events.md)。
- [服务协议：接收、流式背压与终止语义](../system-semantics/service-protocol.md)。
