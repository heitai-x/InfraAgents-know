# K-ENGINE-005：TensorRT-LLM：执行器、Overlap 与传输责任

TensorRT-LLM 的执行器将请求调度、模型执行、采样和资源管理组合成后台迭代，适合研究异步服务的责任划分。

## 两种层次的来源

本页核对固定提交 0343a9d9f2a8f69f9fe783834b042ff7687a28a4，提交日期为 2026-04-30。以下坐标由完整归档源码解析取得，只定位这一版本。

| 文件 | 符号 | 行号 |
|---|---|---|
| tensorrt_llm/_torch/pyexecutor/py_executor.py | PyExecutor | 272 |
| tensorrt_llm/_torch/pyexecutor/py_executor.py | AsyncTransferManager | 153 |
| tensorrt_llm/_torch/pyexecutor/py_executor.py | _prepare_and_schedule_batch | 1957 |
| tensorrt_llm/_torch/pyexecutor/py_executor.py | _schedule | 3110 |
| tensorrt_llm/_torch/pyexecutor/py_executor.py | _sample_async | 3632 |

[完整原文与本地阅读](../../references/S-TRTLLM-CODE/reader.html)；[源码定位证据](../../docs/evidence/astra-source-locations.json)。

架构文档与本文件来自同一历史提交，分别说明入口、执行器、scheduler、model engine 和 sampler 的职责及其实际连接。这里讨论 PyTorch 执行器，不把它与其他后端的接口混用。

## Overlap 的关键是设备端依赖连续

执行器可以让设备推进当前工作，同时 host 处理前一批结果。真正可重叠部分取决于前一批采样结果如何作为后续输入存在于设备，以及停止/取消怎样延迟作用。CPU 少等待并不消除自回归依赖，可能留下额外在途步，需要在资源和响应语义中处理。

## 传输责任的具体例子

源码中的 AsyncTransferManager 跟踪异步状态发送，多个传输使用者都完成后才解除相关保护。这展示了“请求业务上完成”与“KV 可以回收”的区别：发送、存储和设备工作可能仍持有它。围绕 _prepare_and_schedule_batch 还可以看到调度可用性与传输状态的交互。

## 如何利用这个索引

研究 graph padding 时连接静态容量与真实请求映射；研究 overlap 时连接两代 batch 状态；研究分离服务时连接预留、传输完成和回收。InfraAgents 抽取这些机制关系，避免把当前引擎的默认开关或一次性能宣传当成普遍最优策略。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TRTLLM：TensorRT-LLM Architecture Overview](../sources/S-TRTLLM.md)：已读 TensorRT-LLM 架构页将入口、PyExecutor、scheduler、model engine 与 sampler 连接成后台迭代，并介绍 graph padding 和 overlap 思路。
  [完整原文](../../references/S-TRTLLM/reader.html) · [全文文本](../../references/S-TRTLLM/fulltext.txt)

- [S-TRTLLM-CODE：TensorRT-LLM PyExecutor 固定提交](../sources/S-TRTLLM-CODE.md)：固定历史提交的 PyExecutor 包含调度准备、_schedule、_sample_async 和异步传输管理。AsyncTransferManager 在发送与保存的全部使用者完成后解除 KV 保护；加载状态由其他路径处理。
  [完整原文](../../references/S-TRTLLM-CODE/reader.html) · [全文文本](../../references/S-TRTLLM-CODE/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
- [Prefill/Decode 分离：状态传输与队列耦合](../algorithms/disaggregated-serving.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
