# S-CUDA-TOPOLOGY：CUDA Programming Guide — Programming Systems with Multiple GPUs

## 本地完整原文

[完整原文](../../references/S-CUDA-TOPOLOGY/reader.html) · [全文文本](../../references/S-CUDA-TOPOLOGY/fulltext.txt)

固定版本：CUDA 12.8.0。来源版本日期：2025-01-27；整理核对日期：2026-10-22。

## 来源支持的机制与定义

多 GPU 系统涉及设备上下文、peer access、主机与设备连接路径。可通信和具有相同有效带宽不是同一性质。

## InfraAgents 如何据此推导

本库用带延迟/带宽的拓扑图解释 TP、PP、EP、状态迁移和 NUMA 亲和，按消息大小与频率选择通信组。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
