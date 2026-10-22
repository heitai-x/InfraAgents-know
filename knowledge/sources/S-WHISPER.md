# S-WHISPER：Robust Speech Recognition via Large-Scale Weak Supervision

## 本地完整原文

[完整原文](../../references/S-WHISPER/reader.html) · [全文文本](../../references/S-WHISPER/fulltext.txt)

固定版本：2212.04356v1。来源版本日期：2022-12-06；整理核对日期：2026-10-22。

## 来源支持的机制与定义

Whisper 采用语音 encoder-decoder 和任务条件 token。音频编码与目标文本自回归是不同阶段，decoder 使用源表示完成条件生成。

## InfraAgents 如何据此推导

本库从固定源投影推导 cross KV 复用，再分析离线分段、源更新和真正流式提交的区别；缓存仅在相关依赖保持固定时有效。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
