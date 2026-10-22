# S-MOONSHINE：Moonshine: Speech Recognition for Live Transcription and Voice Commands

## 本地完整原文

[完整原文](../../references/S-MOONSHINE/reader.html) · [全文文本](../../references/S-MOONSHINE/fulltext.txt)

固定版本：2410.15608v2。来源版本日期：2024-10-22；整理核对日期：2026-10-22。

## 来源支持的机制与定义

Moonshine 面向语音识别的 encoder-decoder 结构和变长音频处理，源音频表示与目标文本生成具有不同长度和计算阶段。

## InfraAgents 如何据此推导

本库将 processor、encoder、cross 源投影、目标自回归和结果交付分开建模。窗口等待与编码依赖决定流式延迟，不能从名称推断任意音频块均可独立追加。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
