# S-SGLANG-CODE：SGLang — scheduler and radix cache source locations

## 本地完整原文

[完整原文](../../references/S-SGLANG-CODE/reader.html) · [全文文本](../../references/S-SGLANG-CODE/fulltext.txt)

固定版本：Git cf346bb15de90c447ed498ad5e99aae407e04ce2 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

固定历史提交中的 scheduler.py 与 radix_cache.py 展示 Scheduler、批次选择和执行，以及 RadixCache 的前缀匹配、插入与淘汰。精确符号和行号随本版本源码单独记录。

## InfraAgents 如何据此推导

源码入口用于连接前缀匹配、调度选择、执行与回收责任；本库的通用状态模型独立于这些方法名。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
