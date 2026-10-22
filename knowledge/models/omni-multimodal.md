# K-MODEL-012：全模态系统：多时钟、分支依赖与背压

全模态推理是多个有依赖的计算分支组成的系统；文本、视频和音频不能压成同一个 token 长度。

## 分支和时钟

音频输入有采样时间，视频有帧时间，文本有语义位置，模型内部还有编码块和生成步。多模态位置编码将这些关系映射进模型。接收顺序与事件发生时间可能不同，网络抖动使“先收到的帧”未必代表更早内容，因此对齐需要源时间戳和明确缓冲策略。

Qwen2.5-Omni 的 Thinker/Talker 是一个具体分支例：文本思考分支的中间表示为语音分支提供条件，音频还需进一步解码。分支有可重叠区间，也有必须等待的依赖。把两个分支各自 benchmark 的耗时相加可能高估延迟，取最大值又可能忽略串行依赖。

## 用数据流图安排资源

将每个阶段标出输入可用条件、输出块、持久状态和资源占用。视觉 encoder、音频 encoder、共享主干、文本采样、声学分支和波形解码可以有不同 batch 组织。阶段间队列有容量，慢分支会通过背压影响上游；若只让快分支无限生成，会积累无法及时消费的隐藏状态和输出。

例如文本持续产生而声学解码落后，系统可以限制两者领先距离，而不是让所有文本完成后堆积整段声学条件。限制会影响文本输出速度，属于服务策略，需要与“同步说写”还是“文字优先”的产品语义一致。

## 请求身份扩展到模态块

每个块可由请求 ID、模态、源时间区间、分支、生成代次和块序号定位。跨阶段迁移还要说明其依赖哪个源版本。用户打断后新请求的音频不得接在旧请求的尾部，旧分支的晚到结果要按代次丢弃并完成资源释放。

## 完整质量和性能关系

文字准确、语音自然和图像理解分别良好，不足以证明跨模态对齐正确。文本与语音内容一致、时间戳和帧内容一致、打断后状态连续性都来自数据流关系。首文本 token、首音频包、视频处理吞吐和最终完成延迟是不同指标，应沿真实用户消费的路径定义 SLO。

<!-- infraagents:provenance -->

## 来源与推导

- [S-OMNI：Qwen2.5-Omni Technical Report](../sources/S-OMNI.md)：Qwen2.5-Omni 的报告描述 Thinker/Talker 分支、块式模态输入、时间对齐和流式音频输出，文本与音频不是同一执行时钟。
  [完整原文](../../references/S-OMNI/reader.html) · [全文文本](../../references/S-OMNI/fulltext.txt)

- [S-QWEN-VL：Qwen2.5-VL Technical Report](../sources/S-QWEN-VL.md)：Qwen2.5-VL 报告讨论动态视觉分辨率、窗口视觉编码及时间相关的多模态位置，视觉输入具有独立的网格和时间几何。
  [完整原文](../../references/S-QWEN-VL/reader.html) · [全文文本](../../references/S-QWEN-VL/fulltext.txt)

- [S-OTEL：OpenTelemetry — Context propagation](../sources/S-OTEL.md)：OpenTelemetry context propagation 用上下文跨组件连接相关操作，trace/span 形成可追溯的执行关系。
  [完整原文](../../references/S-OTEL/reader.html) · [全文文本](../../references/S-OTEL/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [视觉语言模型：Patch、模态桥接与联合位置](../models/vision-language.md)。
- [语音生成：Codec Token、声学块与播放时钟](../models/speech-generation.md)。
- [服务协议：接收、流式背压与终止语义](../system-semantics/service-protocol.md)。
- [工作负载分布：长度、复用、模态与相关性](../workloads/input-distribution.md)。
