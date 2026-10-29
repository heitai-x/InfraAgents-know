# S-NEURON-EXPLORER：Neuron Explorer：Get started

## 本地完整原文

[完整原文](../../references/S-NEURON-EXPLORER/reader.html) · [全文文本](../../references/S-NEURON-EXPLORER/fulltext.txt)

固定版本：Git ceca8d914d1c4ca531bd27d8e37581b039dcad2b (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

Neuron Explorer 区分系统事件与设备指令 trace；系统视图覆盖 runtime、API、加载与主机，设备视图需要匹配的编译产物与 trace。

## InfraAgents 如何据此推导

本库从 executable 归属到 NeuronCore 数据流逐层定位，不能由系统调用区间直接推断片上算子瓶颈。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
