# K-ENGINE-002：vLLM KV 管理：命中、分配和延迟释放

KVCacheManager 将逻辑 token 进度转换成层组块分配，命中长度、保留容量和可回收块各有独立含义。

## 固定坐标和职责

本页核对固定提交 9c61864bf8a911a8369f35d79d538c7f11cf3dc2，提交日期为 2026-04-30。以下坐标由完整归档源码解析取得，只定位这一版本。

| 文件 | 符号 | 行号 |
|---|---|---|
| vllm/v1/core/kv_cache_manager.py | KVCacheBlocks | 22 |
| vllm/v1/core/kv_cache_manager.py | KVCacheManager | 106 |
| vllm/v1/core/kv_cache_manager.py | get_computed_blocks | 183 |
| vllm/v1/core/kv_cache_manager.py | can_fit_full_sequence | 225 |
| vllm/v1/core/kv_cache_manager.py | allocate_slots | 265 |
| vllm/v1/core/kv_cache_manager.py | free | 437 |

[完整原文与本地阅读](../../references/S-VLLM-KV/reader.html)；[源码定位证据](../../docs/evidence/astra-source-locations.json)。

get_computed_blocks 将完整命中的最大长度限制在请求 token 数减一，以重算所需 logits；块对齐可能使重算超过一个 token。这说明前缀全部命中仍可能需要模型工作。

## 分配参数表达多种状态

allocate_slots 区分新算 token、新命中的本地块、外部已算状态、lookahead 和 encoder token。delay_cache_blocks 用于延后把尚未完成的传输状态发布为可复用缓存。读这些参数时，应分别还原已计算、待计算、预留和可共享状态，不能把全部长度相加后称为已完成前缀。

本版本另有 can_fit_full_sequence，按完整序列、已有命中与窗口规则计算需要的块，并与 block pool 空闲块比较；这是准入检查。它不把一次通过的容量检查自动变成之后每次分配都已成功的事实。

层组 coordinator 决定不同状态族怎样联合分配和命中。异构缓存设计文档与这些接口均绑定本库登记的同一历史提交；各状态族的共同恢复条件仍由正文中的状态模型解释。

## 与回收的接口

释放请求的逻辑持有、允许 block pool 复用、等待异步执行/传输完成是不同责任。要理解某块为何未回收，应沿 scheduler 的 deferred free 和 connector 路径继续追踪，不能只看 free 方法是否被调用。

对 InfraAgents 的价值在于一套可还原的资源账：请求想推进多少、哪些状态已存在、还需预留多少、何时可以发布和回收。它能指导独立实现及诊断，也能解释为什么单看空闲显存不足以判断某请求能否被调度。

<!-- infraagents:provenance -->

## 来源与推导

- [S-VLLM-KV：vLLM — KVCacheManager source reference](../sources/S-VLLM-KV.md)：KVCacheManager 的相关接口区分已命中状态、新 token 分配、lookahead、外部状态和释放。层组与实际可恢复状态参与分配判断。
  [完整原文](../../references/S-VLLM-KV/reader.html) · [全文文本](../../references/S-VLLM-KV/fulltext.txt)

- [S-VLLM-CODE：vLLM V1 固定标签源码](../sources/S-VLLM-CODE.md)：固定历史提交的 Scheduler、KVCacheManager、GPUModelRunner、coordinator 和 block pool 展示 token 进度、分配、执行与采样、层组和回收之间的关系。接口含义以所归档的提交为准。
  [完整原文](../../references/S-VLLM-CODE/reader.html) · [全文文本](../../references/S-VLLM-CODE/fulltext.txt)

- [S-VLLM-HYBRID：vLLM：Hybrid KV Cache Manager](../sources/S-VLLM-HYBRID.md)：固定历史设计文档区分 KV hidden bytes、token block size 和物理 page size，讨论异构层分组、统一分配以及各组可恢复前缀的交集。具体功能限制只适用于该文档版本。
  [完整原文](../../references/S-VLLM-HYBRID/reader.html) · [全文文本](../../references/S-VLLM-HYBRID/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [分页 KV：逻辑位置、物理布局与共享所有权](../algorithms/paged-kv-capacity.md)。
- [异构状态缓存：层组、恢复位置与联合分配](../algorithms/heterogeneous-state-cache.md)。
- [vLLM V1：从 Token 进度到调度输出](../engine-index/vllm-scheduler.md)。
