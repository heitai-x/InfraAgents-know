# S-CHAT：Transformers：Chat templates

## 本地完整原文

[完整原文](../../references/S-CHAT/reader.html) · [全文文本](../../references/S-CHAT/fulltext.txt)

固定版本：Git 756dffe0d664419d2f620e131ed426373cfd6f12 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

Chat template 将消息序列变为包含角色边界、特殊 token 和生成提示的模型输入；tokenizer 自动添加特殊 token 的方式会影响最终序列。

## InfraAgents 如何据此推导

本库把模板和 tokenizer 版本纳入输入语义，分析重复 BOS/EOS、padding、截断与前缀缓存身份。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
