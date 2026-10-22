# S-WHISPER：Robust Speech Recognition via Large-Scale Weak Supervision

## 本地完整原文

[完整原文阅读](../../references/S-WHISPER/reader.html) · [全文文本](../../references/S-WHISPER/fulltext.txt) · [原始文件](../../references/S-WHISPER/original.pdf)

完整 PDF：28 页，版本 2212.04356v1。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2022 paper。

## 来源支持的机制与定义

Whisper 采用语音 encoder-decoder 和任务条件 token。音频编码与目标文本自回归是不同阶段，decoder 使用源表示完成条件生成。

## InfraAgents 如何据此推导

本库从固定源投影推导 cross KV 复用，再分析离线分段、源更新和真正流式提交的区别；缓存仅在相关依赖保持固定时有效。

## 阅读范围与出处身份

语音 encoder-decoder 与任务条件化。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
