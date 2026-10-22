# S-NKI：Trainium2 Architecture Guide for NKI

## 本地完整原文

[完整原文](../../references/S-NKI/reader.html) · [全文文本](../../references/S-NKI/fulltext.txt)

固定版本：Git ceca8d914d1c4ca531bd27d8e37581b039dcad2b (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

已读 Trainium2 NKI 资料区分 HBM、SBUF、PSUM、DMA 与计算引擎；片上张量的 partition/free 轴和转置路径具有具体硬件意义。

## InfraAgents 如何据此推导

本库用活跃 tile 集合与在线归约推导 attention 数据流，分析双缓冲、矩阵供数和片上存储生存期。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
