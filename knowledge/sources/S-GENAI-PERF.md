# S-GENAI-PERF：NVIDIA GenAI-Perf

## 本地完整原文

[完整原文](../../references/S-GENAI-PERF/reader.html) · [全文文本](../../references/S-GENAI-PERF/fulltext.txt)

固定版本：Git 6c66ca6602371d2b79cc1b5352d5a4fa11f007c3 (2026-04-28)。来源版本日期：2026-04-28；整理核对日期：2026-10-22。

## 来源支持的机制与定义

GenAI-Perf 的指标围绕请求、响应、首个内容、流式间隔、输出长度和测量窗口定义。客户端 chunk 与模型 token 并非必然一一对应。

## InfraAgents 如何据此推导

本库明确 TTFT 起点、TPOT 分母、ITL 聚合顺序和窗口吞吐，连接内部执行与用户收到内容的时间。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
