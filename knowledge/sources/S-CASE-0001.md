# S-CASE-0001：InfraAgents CASE-0001 evidence manifest

## 本地完整原文

[完整原文阅读](../../references/S-CASE-0001/reader.html) · [全文文本](../../references/S-CASE-0001/fulltext.txt) · [原始文件](../../references/S-CASE-0001/original.json)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2026-09-19 / original implementation 1d828cb。

## 来源支持的机制与定义

本地证据清单记录微型 CUDA 采集、导出、缺表对照和事件关联。所记首次设备开始减 host API 结束为 −1116 ns；3 kernel 累计 4352 ns、2 memcpy 累计 4896 ns。

## InfraAgents 如何据此推导

这些事实支持本库对异步时序与观测覆盖的解释。其模型与负载范围仅为该微型案例，不能推导大模型吞吐。原始清单为 docs/evidence/nsys-feedback-20260919.json。

## 阅读范围与出处身份

微型 CUDA 采集、缺表对照、时间关系；非模型性能。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
