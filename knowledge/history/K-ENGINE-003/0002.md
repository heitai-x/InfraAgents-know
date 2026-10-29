# K-ENGINE-003：vLLM ModelRunner：调度视图到设备状态

ModelRunner 把逻辑调度结果转成设备输入与持久批状态，并将前向和采样结果按请求归还。

## 固定版本入口

本页核对固定提交 9c61864bf8a911a8369f35d79d538c7f11cf3dc2，提交日期为 2026-04-30。以下坐标由完整归档源码解析取得，只定位这一版本。

| 文件 | 符号 | 行号 |
|---|---|---|
| vllm/v1/worker/gpu_model_runner.py | GPUModelRunner | 394 |
| vllm/v1/worker/gpu_model_runner.py | execute_model | 3791 |
| vllm/v1/worker/gpu_model_runner.py | sample_tokens | 4144 |
| vllm/v1/worker/gpu_model_runner.py | initialize_kv_cache | 6812 |
| vllm/v1/worker/gpu_model_runner.py | get_kv_cache_spec | 6949 |

[完整原文与本地阅读](../../references/S-VLLM-RUNNER/reader.html)；[源码定位证据](../../docs/evidence/astra-source-locations.json)。

## 从请求错配追踪三段映射

第一段是请求到 persistent batch 行；第二段是本轮 query token 压紧与 KV 写入槽；第三段是 logits/采样结果回到请求。动态 batch、prefix hit 和 speculative lengths 会分别改变这些映射。若只有混合批失败，按这三段逐一比较比直接怀疑整个模型更有效。

## 从执行开销追踪边界

输入准备中的 host/device 同步、dtype/layout 转换、attention metadata 和 graph 路径决定实际开销。KV cache spec 由模型层结构产生，初始化再映射成物理 tensor。若改模型状态形态，应同时沿 spec、分配、metadata 和实际 attention consumer 阅读，单改缓存 tensor shape 不足以完成接入。

该文件是多个优化汇合处，适合验证组合条件。InfraAgents 的知识应抽取数据关系与生存期，再链接到对应版本位置；方法名和行号本身只是导航。

## 独立执行器需要保存的交接关系

从调度器到执行器的契约可以按语义记录：请求与分支身份、待计算范围、模型/adapter 版本、输入与缓存布局、允许写入范围，以及结果回传对应的批次代次。张量实现可以采用压紧、固定槽或图桶，但不得丢失这些关系。

把一次执行分成输入准备、计算、采样与输出消费之后，需要明确每个阶段最后使用哪些缓冲。图重放的地址稳定不代表上次内容已经消费；CPU–GPU 分工也需要在结果汇合前保护各自的分支状态。[异构执行](../algorithms/heterogeneous-execution.md) 与 [分支生成](../algorithms/branch-generation.md) 分别展开这些变化。

验证时让相同请求在独立执行、混合 batch、行重排、分支扩展、取消与恢复中保持相同的语义结果；每次比较固定参考和容差。源码索引用于定位一种实现如何承担责任，正文中的状态契约才是独立引擎可以直接用于推导的部分。

<!-- infraagents:provenance -->

## 来源与推导

- [S-VLLM-RUNNER：vLLM — GPUModelRunner source reference](../sources/S-VLLM-RUNNER.md)：GPUModelRunner 将调度输出映射为持久 batch、压紧输入、attention metadata、模型执行与采样结果。模型状态规格最终映射到物理缓存 tensor。
  [完整原文](../../references/S-VLLM-RUNNER/reader.html) · [全文文本](../../references/S-VLLM-RUNNER/fulltext.txt)

- [S-VLLM-CODE：vLLM V1 固定标签源码](../sources/S-VLLM-CODE.md)：固定历史提交的 Scheduler、KVCacheManager、GPUModelRunner、coordinator 和 block pool 展示 token 进度、分配、执行与采样、层组和回收之间的关系。接口含义以所归档的提交为准。
  [完整原文](../../references/S-VLLM-CODE/reader.html) · [全文文本](../../references/S-VLLM-CODE/fulltext.txt)

- [S-TORCH-CUDA：PyTorch — CUDA semantics](../sources/S-TORCH-CUDA.md)：PyTorch CUDA 运算通常异步提交；不同 stream 的依赖需显式表达，allocator 还需要知道跨 stream 的使用。CUDA Graph 的重放路径依赖稳定的地址与兼容的执行结构。
  [完整原文](../../references/S-TORCH-CUDA/reader.html) · [全文文本](../../references/S-TORCH-CUDA/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [请求身份：逻辑序列、批次视图与结果归属](../system-semantics/request-batch-identity.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [批量采样：概率变换、请求随机性与行映射](../algorithms/batched-sampling.md)。
- [vLLM V1：从 Token 进度到调度输出](../engine-index/vllm-scheduler.md)。
