# S-VLLM-PREFIX：vLLM — Automatic Prefix Caching

## 本地完整原文

[完整原文](../../references/S-VLLM-PREFIX/reader.html) · [全文文本](../../references/S-VLLM-PREFIX/fulltext.txt)

固定版本：Git 9c61864bf8a911a8369f35d79d538c7f11cf3dc2 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

所核对 vLLM 版本以块身份组织前缀复用，身份可包含父块、token 和额外条件。不同上下文的同 token 块不应自动视为相同计算。

## InfraAgents 如何据此推导

本库以计算依赖闭包推导模型/adapter/位置/模态内容的身份，区分逻辑命中、完整物理块与真正可跳过的计算。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
