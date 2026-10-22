# K-SYS-008：服务协议：接收、流式背压与终止语义

推理服务将内部生成状态映射为可观察的网络事件；接收、执行完成和客户端收到结果是不同的完成点。

## 一条请求的外部状态

参数验证、排队接收、开始执行、产生首个有效内容、持续输出、最终终止构成可观察协议。返回 HTTP 成功状态通常只说明响应开始，后续流仍可能失败。每条流需要明确的终止事件和原因，否则客户端无法区分正常结束与连接中断。

流式 chunk 是传输单位，不一定是模型 token。一个 chunk 可包含多个 token、空角色信息或 usage 元数据；TTFT 应绑定协议定义的首个内容事件。压缩、代理缓冲和客户端解析还会改变收到事件的时间。

## 背压的资源含义

若生成快于客户端消费，输出队列增长。无限队列让慢客户端持有大量文本/音频和请求状态，最终影响其他用户。限制队列、暂停后续生成或断开超时消费者是不同服务策略，应该由延迟与可靠性目标决定。

暂停生成并不一定能释放 KV；完整请求状态仍需保留。对音频，过大队列还增加用户打断后的尾音。背压因此既是网络机制，也是调度和容量变量。

## 取消、错误和重试

客户端断开、发送失败、内部超时和模型异常都应进入同一请求终止账务，避免重复释放。重试是否允许复用请求 ID、是否可能重复输出、是否支持恢复已确认前缀，是协议明确承诺的功能，不能由客户端盲目重发推断。

请求追踪可以跨 API、tokenizer、scheduler、runner 和输出阶段传播上下文。batch 是多请求聚合，需额外记录请求与执行批的映射。敏感原始输入不必全部写 trace，状态身份和摘要即可支持许多因果分析。

## 兼容 API 的实际含义

字段名字相同只说明表面结构。采样默认值、token 计数、流式 delta、工具调用结构、错误格式、取消和结束原因共同定义行为兼容。服务验收应以具体协议的输入输出实例和状态转换为准。内部 engine 能生成文本，与一个完整可消费的服务接口之间还有这些确定的语义关系。

<!-- infraagents:provenance -->

## 来源与推导

- S-ASGI：[完整原文](../../references/S-ASGI/reader.html) · [全文文本](../../references/S-ASGI/fulltext.txt)
- S-OTEL：[完整原文](../../references/S-OTEL/reader.html) · [全文文本](../../references/S-OTEL/fulltext.txt)
- S-GENAI-PERF：[完整原文](../../references/S-GENAI-PERF/reader.html) · [全文文本](../../references/S-GENAI-PERF/fulltext.txt)

- [S-ASGI：ASGI — HTTP and WebSocket Message Format](../sources/S-ASGI.md)：ASGI HTTP/WebSocket 规范定义接收、发送、响应体和断开相关事件；断开通知与发送失败可能存在时间上的竞争。

- [S-OTEL：OpenTelemetry — Context propagation](../sources/S-OTEL.md)：OpenTelemetry context propagation 用上下文跨组件连接相关操作，trace/span 形成可追溯的执行关系。

- [S-GENAI-PERF：NVIDIA GenAI-Perf](../sources/S-GENAI-PERF.md)：GenAI-Perf 的指标围绕请求、响应、首个内容、流式间隔、输出长度和测量窗口定义。客户端 chunk 与模型 token 并非必然一一对应。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 3，2026-09-20。

## 延伸阅读

- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
- [服务指标：事件边界、Token 间隔与用户时间](../workloads/metric-events.md)。
- [语音生成：Codec Token、声学块与播放时钟](../models/speech-generation.md)。
