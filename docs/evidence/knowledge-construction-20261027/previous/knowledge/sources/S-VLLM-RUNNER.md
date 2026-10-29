# S-VLLM-RUNNER：vLLM — GPUModelRunner source reference

## 本地完整原文

[完整原文](../../references/S-VLLM-RUNNER/reader.html) · [全文文本](../../references/S-VLLM-RUNNER/fulltext.txt)

固定版本：Git 9c61864bf8a911a8369f35d79d538c7f11cf3dc2 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

GPUModelRunner 将调度输出映射为持久 batch、压紧输入、attention metadata、模型执行与采样结果。模型状态规格最终映射到物理缓存 tensor。

## InfraAgents 如何据此推导

本库用请求到 batch、query 到状态槽、logits 到输出三段映射定位组合问题；模型结构变化需要贯穿 spec、分配与消费者。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
