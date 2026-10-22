# K-ENGINE-005：TensorRT-LLM：执行器、Overlap 与传输责任

TensorRT-LLM 的执行器将请求调度、模型执行、采样和资源管理组合成后台迭代，适合研究异步服务的责任划分。

## 两种层次的来源

官方架构页说明 LLM 入口、执行器、scheduler、model engine 和 sampler 的关系。本次还实际读取 VibeSys 子模块记录指向的固定提交 0d2bea7c3c99b734a8e09c4c767820e03136a15b；并未初始化或引入它作为运行依赖。

该提交 tensorrt_llm/_torch/pyexecutor/py_executor.py：PyExecutor 第 256 行，_prepare_and_schedule_batch 第 1809 行，_schedule 第 2952 行，_sample_async 第 3467 行。文件哈希与符号见 [定位证据](../../docs/evidence/expert-source-locations-20260920.json)。该源码的组织不等于旧版 TensorRT C++ engine 路径，两者应按实际后端分别阅读。

## Overlap 的关键是设备端依赖连续

执行器可以让设备推进当前工作，同时 host 处理前一批结果。真正可重叠部分取决于前一批采样结果如何作为后续输入存在于设备，以及停止/取消怎样延迟作用。CPU 少等待并不消除自回归依赖，可能留下额外在途步，需要在资源和响应语义中处理。

## 传输责任的具体例子

源码中的 AsyncTransferManager 跟踪异步状态发送，多个传输使用者都完成后才解除相关保护。这展示了“请求业务上完成”与“KV 可以回收”的区别：发送、存储和设备工作可能仍持有它。围绕 _prepare_and_schedule_batch 还可以看到调度可用性与传输状态的交互。

## 如何利用这个索引

研究 graph padding 时连接静态容量与真实请求映射；研究 overlap 时连接两代 batch 状态；研究分离服务时连接预留、传输完成和回收。InfraAgents 抽取这些机制关系，避免把当前引擎的默认开关或一次性能宣传当成普遍最优策略。

<!-- infraagents:provenance -->

## 来源与推导

- S-TRTLLM：[完整原文](../../references/S-TRTLLM/reader.html) · [全文文本](../../references/S-TRTLLM/fulltext.txt)
- S-TRTLLM-CODE：[完整原文](../../references/S-TRTLLM-CODE/reader.html) · [全文文本](../../references/S-TRTLLM-CODE/fulltext.txt)

- [S-TRTLLM：TensorRT-LLM Architecture Overview](../sources/S-TRTLLM.md)：已读 TensorRT-LLM 架构页将入口、PyExecutor、scheduler、model engine 与 sampler 连接成后台迭代，并介绍 graph padding 和 overlap 思路。

- [S-TRTLLM-CODE：TensorRT-LLM PyExecutor 固定提交](../sources/S-TRTLLM-CODE.md)：实际读取固定提交 0d2bea7c 的 PyExecutor 文件，核对调度准备、_schedule、_sample_async 和异步传输责任。AsyncTransferManager 需在多个传输使用者完成后解除保护。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 3，2026-09-20。

## 延伸阅读

- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
- [Prefill/Decode 分离：状态传输与队列耦合](../algorithms/disaggregated-serving.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
