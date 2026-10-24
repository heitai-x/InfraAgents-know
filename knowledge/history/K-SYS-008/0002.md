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

## 接收协议要连接实际执行域

入口事件循环将请求交给分词线程/进程、调度器和模型 worker 时，每条队列都应携带请求版本、deadline、取消状态和缓冲所有权。[主机运行时](host-runtime.md)比较执行域组织；在协议层，关键是队列满时能明确等待、拒绝或终止，而不是已经接受后无限隐藏排队。

租户限额应同时覆盖输入字节/长度、待处理工作、活跃状态和输出积压。只限制连接数不能阻止单连接携带大输入，按请求数平均也不能限制长上下文 KV。配额不足和模型版本未就绪可以有不同的可重试语义，错误响应需说明这次请求是否已经被接收及是否已有对外结果。

## 滚动发布与连接生命周期

服务发现和路由将新请求绑定到具体就绪版本；已有流式连接保持其请求版本和提交顺序。滚动发布先撤销旧实例的新准入，再排空或按明确协议迁移；关闭监听端口不等于现有生成、IPC 和设备操作全部结束。遇到最长排空时间时，最终错误或恢复能力必须与客户端契约一致。

重试协议需要明确请求身份与去重范围。如果服务只能重新执行，就应披露可能重复已产生的内容；若支持从已确认位置恢复，还需保存对应模型、输入、采样/约束状态和输出序号。内部算子重新计算的幂等性不能自动提供网络交付的恰好一次语义。

## 服务故障状态验证

从参数验证前、进入队列后、首内容前、已交付若干内容后、最终结果已计算但未发送等边界分别断开连接或终止 worker。逐项核对客户端看到的事件、服务端提交记录、可否重试与最终资源账。慢客户端测试需要同时运行正常客户端，检查背压是否限定在预定队列和租户范围内。

发布测试保留一条旧版本长流，切换新请求到新版本，再注入新实例预热失败和旧实例排空超时。成功条件包括结果版本可识别、未就绪实例不被路由、每条流有可解释终态、没有迟到旧消息污染新连接。TTFT 继续以首个有效内容计算，不能因发布期间先发空 chunk 就宣称满足目标。

<!-- infraagents:provenance -->

## 来源与推导

- [S-ASGI：ASGI — HTTP and WebSocket Message Format](../sources/S-ASGI.md)：ASGI HTTP/WebSocket 规范定义接收、发送、响应体和断开相关事件；断开通知与发送失败可能存在时间上的竞争。
  [完整原文](../../references/S-ASGI/reader.html) · [全文文本](../../references/S-ASGI/fulltext.txt)

- [S-OTEL：OpenTelemetry — Context propagation](../sources/S-OTEL.md)：OpenTelemetry context propagation 用上下文跨组件连接相关操作，trace/span 形成可追溯的执行关系。
  [完整原文](../../references/S-OTEL/reader.html) · [全文文本](../../references/S-OTEL/fulltext.txt)

- [S-GENAI-PERF：NVIDIA GenAI-Perf](../sources/S-GENAI-PERF.md)：GenAI-Perf 的指标围绕请求、响应、首个内容、流式间隔、输出长度和测量窗口定义。客户端 chunk 与模型 token 并非必然一一对应。
  [完整原文](../../references/S-GENAI-PERF/reader.html) · [全文文本](../../references/S-GENAI-PERF/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-BORG：Large-scale cluster management at Google with Borg](../sources/S-BORG.md)：§2 任务状态、资源向量、优先级/配额；§3 可行性与评分、制品局部性；§4 相关故障；§5 装箱、碎片和资源回收；§6 隔离；§8 控制服务的组织经验。
  [完整原文](../../references/S-BORG/reader.html) · [全文文本](../../references/S-BORG/fulltext.txt)

- [S-NVIDIA-PLUGIN：NVIDIA Kubernetes device plugin v0.16.1 — deployment and sharing](../sources/S-NVIDIA-PLUGIN.md)：v0.16.1 README 定义设备暴露、UUID/index、运行时注入方式、MIG 策略、time-slicing 与 MPS 的配置边界和升级排空约定。
  [完整原文](../../references/S-NVIDIA-PLUGIN/reader.html) · [全文文本](../../references/S-NVIDIA-PLUGIN/fulltext.txt)

## 延伸阅读

- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
- [服务指标：事件边界、Token 间隔与用户时间](../workloads/metric-events.md)。
- [语音生成：Codec Token、声学块与播放时钟](../models/speech-generation.md)。
