# 引擎源码索引

本分类条目的适用条件、来源与未验证范围在各正文中说明。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-ENGINE-001：vLLM V1：从 Token 进度到调度输出](vllm-scheduler.md) | v4 · seed | vLLM V1 的调度可沿请求 token 进度理解，再追踪预算、缓存分配和执行结果怎样改变下一轮状态。 |
| [K-ENGINE-002：vLLM KV 管理：命中、分配和延迟释放](vllm-kv-manager.md) | v4 · seed | KVCacheManager 将逻辑 token 进度转换成层组块分配，命中长度、保留容量和可回收块各有独立含义。 |
| [K-ENGINE-003：vLLM ModelRunner：调度视图到设备状态](vllm-model-runner.md) | v4 · seed | ModelRunner 把逻辑调度结果转成设备输入与持久批状态，并将前向和采样结果按请求归还。 |
| [K-ENGINE-004：SGLang：Radix 复用与调度状态的连接](sglang-radix-scheduler.md) | v4 · seed | SGLang 将共享前缀组织成 Radix 结构，调度器围绕命中状态和可执行批次推进请求。 |
| [K-ENGINE-005：TensorRT-LLM：执行器、Overlap 与传输责任](tensorrt-llm.md) | v3 · seed | TensorRT-LLM 的执行器将请求调度、模型执行、采样和资源管理组合成后台迭代，适合研究异步服务的责任划分。 |
