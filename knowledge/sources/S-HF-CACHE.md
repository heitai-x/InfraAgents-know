# S-HF-CACHE：Transformers — Caching

## 本地完整原文

[完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

固定版本：Git 756dffe0d664419d2f620e131ed426373cfd6f12 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

Transformers 的缓存说明区分已存历史 K/V、新输入、attention mask 和 cache position。增量前向需要让新 token 的位置与已有缓存长度一致。

## InfraAgents 如何据此推导

本库将逻辑位置、可见性和物理槽位分开，构造整段、分块、逐步和合法前缀命中的等价关系，用于检查缓存接入。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
