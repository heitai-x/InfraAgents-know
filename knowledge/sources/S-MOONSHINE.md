# S-MOONSHINE：Moonshine: Speech Recognition for Live Transcription and Voice Commands

## 本地完整原文

[完整原文阅读](../../references/S-MOONSHINE/reader.html) · [全文文本](../../references/S-MOONSHINE/fulltext.txt) · [原始文件](../../references/S-MOONSHINE/original.pdf)

完整 PDF：7 页，版本 2410.15608v2。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2024 / v2。

## 来源支持的机制与定义

Moonshine 面向语音识别的 encoder-decoder 结构和变长音频处理，源音频表示与目标文本生成具有不同长度和计算阶段。

## InfraAgents 如何据此推导

本库将 processor、encoder、cross 源投影、目标自回归和结果交付分开建模。窗口等待与编码依赖决定流式延迟，不能从名称推断任意音频块均可独立追加。

## 阅读范围与出处身份

语音 encoder-decoder、变长输入与位置编码；非实时服务验收。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
