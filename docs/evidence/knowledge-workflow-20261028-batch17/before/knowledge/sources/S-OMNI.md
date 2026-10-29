# S-OMNI：Qwen2.5-Omni Technical Report

## 本地完整原文

[完整原文](../../references/S-OMNI/reader.html) · [全文文本](../../references/S-OMNI/fulltext.txt)

固定版本：2503.20215v1。来源版本日期：2025-03-26；整理核对日期：2026-10-22。

## 来源支持的机制与定义

Qwen2.5-Omni 的报告描述 Thinker/Talker 分支、块式模态输入、时间对齐和流式音频输出，文本与音频不是同一执行时钟。

## InfraAgents 如何据此推导

本库构建分支依赖、领先距离、背压和播放缓冲模型，将首文本、首音频和最终完成分别定义。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
