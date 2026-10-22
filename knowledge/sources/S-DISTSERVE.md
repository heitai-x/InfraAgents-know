# S-DISTSERVE：DistServe

## 本地完整原文

[完整原文阅读](../../references/S-DISTSERVE/reader.html) · [全文文本](../../references/S-DISTSERVE/fulltext.txt) · [原始文件](../../references/S-DISTSERVE/original.pdf)

完整 PDF：18 页，版本 2401.09670v3。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2024 paper。

## 来源支持的机制与定义

DistServe 将 prefill 与 decode 的资源分配和并行选择解耦，并把阶段间状态通信与链路条件纳入服务目标。

## InfraAgents 如何据此推导

本库将总路径拆成两侧排队、计算、KV 转换与传输，说明分离收益必须超过新增通信和接管成本。

## 阅读范围与出处身份

prefill/decode 资源解耦与通信代价。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
