# S-TORCH-PROFILER：PyTorch — torch.profiler

## 本地完整原文

[完整原文](../../references/S-TORCH-PROFILER/reader.html) · [全文文本](../../references/S-TORCH-PROFILER/fulltext.txt)

固定版本：Git 647b6a2ff9a83a886464932dfad78781ceb61cb6 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

框架 profiler 可以采集算子、设备工作、shape、stack 和内存相关信息。更详细的采集会增加运行开销，部分选项也会延长张量引用。

## InfraAgents 如何据此推导

本库使用框架视图连接高层算子与具体 kernel，并将无扰动服务计时和短窗口诊断采集分开解释。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
