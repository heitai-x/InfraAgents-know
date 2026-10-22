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

## 各种生成链需要各自的执行预算

自回归声学 token、codec token 到 mel 的迭代 flow/扩散、mel 到 waveform 的 vocoder，分别消耗不同的状态、矩阵工作和窗口。直接因果 codec 解码、一次生成完整声学表示、以及分块迭代解码不能用同一个“每 token 耗时”比较。多 codebook 的生成顺序和延迟排列也决定何时凑齐一个可解码声学帧。

Qwen2.5-Omni 固定报告的具体路径是 Talker 生成语音 token，Flow-Matching DiT 生成 mel，再经修改的 BigVGAN 产生 waveform；DiT block 可见范围为当前块加两块 lookback 和一块 lookahead。需要未来 code block 不等于必须等待相同长度的墙钟时间：如果 code 生成快于实时，可以提前生成所需未来块，代价表现为额外模型步骤、状态和首次启动工作。

跨请求聚批可以分别发生在 Talker、flow step 和 vocoder，条件是 shape、所处求解阶段和上下文合法。为了合成大 batch 等待太久，会先耗尽客户端播放缓冲。调度按下一块最迟可播放时间与缺口安排工作，比只最大化整段音频/秒更贴合流式目标；不具备中途抢占能力的长 kernel 还会形成不可绕过的阻塞。

## 一个声学块的首包与稳态算例

以下为通用服务设定，不声称某个已发表模型的 codec 帧率。声学帧率 50 Hz，每块 10 帧对应 200 ms 音频，需要一块 lookahead；code 生成速度设为 500 帧/s，flow/vocoder 合计每块 25 ms，封包发送 5 ms，且文本条件已就绪。首块需 20 帧 code，共 40 ms，再解码和发送，首包为 70 ms。稳态每块只新增 10 帧，可串行用 20+25+5=50 ms 供给 200 ms 音频，对应这条设定路径的实时率 0.25。

若文本或语义条件还需等待 150 ms，首音频即变为 220 ms；若 flow 阶段与其他请求竞争产生 180 ms 排队，良好的平均实时率仍不能保证不断音。减少为 5 帧块可能降低首次 code 等待，却增大解码/封包调用次数和边界开销。完整声学表示一次解码可提高大批效率，但首包要等整段条件/声学生成；是否允许这种等待由任务决定。

24 kHz、单声道、16-bit PCM 的 200 ms 块有 4800 个样本、9600 字节；400 ms 客户端缓冲含 19200 字节。编码压缩可降低网络 bytes，却不取消声音所代表的播放时间。测客户端可播放块的到达时间和连续 sample offset，不能用服务器 token 完成时间代替声音实际开始。

## 打断、抖动与质量一起测

打断时使旧响应代次不再接收新块，取消未提交模型工作，并让在途 compute、传输、codec 状态和 overlap 安全结束。客户端也要丢弃尚未播放的旧代次数据；仅服务端停生成，已缓冲音频仍会继续播放。已播放样本无法撤销，协议要明确最后确认的 sample offset 和新响应从何处开始。

联合实验固定文本/语义输入与音色条件，扫描 chunk、lookahead、并发、突发、慢客户端和网络抖动。报告首可播放样本、平均与尾部实时率、缓冲欠载时长、打断到静音、最终音频时长、内容一致性和自然度。ASR 回转的 WER 只覆盖可懂内容的一部分，不能代替音色、韵律、接缝和噪声验证；裁去失败块、补静音或改变语速必须作为输出语义变化披露。

<!-- infraagents:provenance -->

## 来源与推导

- [S-OMNI：Qwen2.5-Omni Technical Report](../sources/S-OMNI.md)：Qwen2.5-Omni 的报告描述 Thinker/Talker 分支、块式模态输入、时间对齐和流式音频输出，文本与音频不是同一执行时钟。
  [完整原文](../../references/S-OMNI/reader.html) · [全文文本](../../references/S-OMNI/fulltext.txt)

- [S-MOONSHINE：Moonshine: Speech Recognition for Live Transcription and Voice Commands](../sources/S-MOONSHINE.md)：Moonshine 面向语音识别的 encoder-decoder 结构和变长音频处理，源音频表示与目标文本生成具有不同长度和计算阶段。
  [完整原文](../../references/S-MOONSHINE/reader.html) · [全文文本](../../references/S-MOONSHINE/fulltext.txt)

- [S-WHISPER：Robust Speech Recognition via Large-Scale Weak Supervision](../sources/S-WHISPER.md)：Whisper 采用语音 encoder-decoder 和任务条件 token。音频编码与目标文本自回归是不同阶段，decoder 使用源表示完成条件生成。
  [完整原文](../../references/S-WHISPER/reader.html) · [全文文本](../../references/S-WHISPER/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [Encoder-decoder：源表示、两套 Attention 与流式边界](../models/encoder-decoder-streaming.md)。
- [服务指标：事件边界、Token 间隔与用户时间](../workloads/metric-events.md)。
- [服务协议：接收、流式背压与终止语义](../system-semantics/service-protocol.md)。
