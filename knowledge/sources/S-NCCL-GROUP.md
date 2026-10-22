# S-NCCL-GROUP：NCCL — Group Calls

## 本地完整原文

[完整原文](../../references/S-NCCL-GROUP/reader.html) · [全文文本](../../references/S-NCCL-GROUP/fulltext.txt)

固定版本：NCCL 2.25.1。来源版本日期：2025-01-27；整理核对日期：2026-10-22。

## 来源支持的机制与定义

NCCL group calls 用于组织成组通信提交，调用次序与多设备使用需要符合其协议。组调用不替应用修复不匹配的通信顺序。

## InfraAgents 如何据此推导

本库将通信提交、设备完成和请求控制流连接起来，解释局部跳过计算为何可能破坏其他 ranks 等待的操作。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
