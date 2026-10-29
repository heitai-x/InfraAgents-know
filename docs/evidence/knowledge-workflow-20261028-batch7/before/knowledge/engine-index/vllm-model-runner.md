# K-ENGINE-003：执行视图决策：压紧、固定槽与结果归属

动态请求集合需要转成设备能执行的张量，但逻辑身份不应随物理行号变化。执行器的核心问题是保持请求、query 位置、缓存写入与输出行之间的可逆对应，并管理这些表示的生存期。

## 从逻辑请求到本轮张量

设请求 i 本轮有 qᵢ 个 query，压紧偏移 p₀=0、pᵢ₊₁=pᵢ+qᵢ；请求 i 的第 k 个 query 位于 pᵢ+k，其绝对位置由对应请求的有效起点与 k 决定。仅需要末位置 logits 的普通生成中，候选行是 pᵢ₊₁−1；需要 prompt logprobs、分支或投机验证时，所需行集合另行表达，不能复用这个简化规则。

设 q=[2,5,3]，则 p=[0,2,7,10]，三个末位置行是 [1,6,9]。若请求被重排，必须同时重排身份映射、位置与结果路由；shape 正确不能发现把第 6 行交给另一请求的错误。

```pseudocode
更新本轮有效请求集合，保存每个物理槽对应的逻辑身份和代次
按所选表示生成 query 位置、输入 token 与缓存写入位置
建立输出行到请求/位置/分支的映射，再提交设备计算
按该轮映射解释 logits 和采样结果，校验返回代次
最后消费者完成后才覆盖本轮输入、metadata 或辅助缓冲
```

## 压紧与固定容量的交换

压紧减少无效行，却付出收集、前缀和、重排或 metadata 更新；固定槽与图桶方便稳定地址和复用，但可能计算填充行并增加多桶驻留。[已核对的执行器实例](../sources/S-VLLM-RUNNER.md) 维护持久批状态，再压紧本轮 query；它说明这两个选择可以组合，并不要求独立引擎复制同一类结构。

容量 16 的桶承载 10 个有效行时，有 6/16=37.5% 的槽位未承载有效行；这只是槽位比例，不直接等于浪费的运行时间。是否跳过无效行、kernel 效率和图重放成本决定实际收益。若每步都重新整理大块历史，压紧的准备成本也可能超过避免的填充工作。

## 用映射与完成证据区分解释

同一请求单独执行、混合批执行、行重排和取消后复用槽位，应得到相同语义结果。记录每轮身份、绝对位置、缓存槽和 logits 路由，可将错配与模型数值误差分开。延迟设备完成再安排下一轮，能检查固定地址是否被过早覆盖。比较准备、设备、采样与可见输出的完整时间线，避免把省下提交工作误当作数学计算变少。

<!-- infraagents:provenance -->

## 来源与推导

- [S-VLLM-RUNNER：vLLM — GPUModelRunner source reference](../sources/S-VLLM-RUNNER.md)：GPUModelRunner 将调度输出映射为持久 batch、压紧输入、attention metadata、模型执行与采样结果。模型状态规格最终映射到物理缓存 tensor。
  [算法与机制说明](../sources/S-VLLM-RUNNER.md)

- [S-VLLM-CODE：vLLM V1 固定标签源码](../sources/S-VLLM-CODE.md)：固定历史提交的 Scheduler、KVCacheManager、GPUModelRunner、coordinator 和 block pool 展示 token 进度、分配、执行与采样、层组和回收之间的关系。接口含义以所归档的提交为准。
  [算法与机制说明](../sources/S-VLLM-CODE.md)

- [S-TORCH-CUDA：PyTorch — CUDA semantics](../sources/S-TORCH-CUDA.md)：PyTorch CUDA 运算通常异步提交；不同 stream 的依赖需显式表达，allocator 还需要知道跨 stream 的使用。CUDA Graph 的重放路径依赖稳定的地址与兼容的执行结构。
  [完整原文](../../references/S-TORCH-CUDA/reader.html) · [全文文本](../../references/S-TORCH-CUDA/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [请求身份：逻辑序列、批次视图与结果归属](../system-semantics/request-batch-identity.md)。
- [CUDA Graph：捕获边界、计划状态与变长服务](../frameworks/cuda-graph-buffers.md)。
- [批量采样：概率变换、请求随机性与行映射](../algorithms/batched-sampling.md)。
- [vLLM V1：从 Token 进度到调度输出](../engine-index/vllm-scheduler.md)。
