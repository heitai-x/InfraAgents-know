# S-SPEC：Fast Inference from Transformers via Speculative Decoding

## 本地完整原文

[完整原文](../../references/S-SPEC/reader.html) · [全文文本](../../references/S-SPEC/fulltext.txt)

固定版本：2211.17192v2。来源版本日期：2023-05-18；整理核对日期：2026-10-22。

## 来源支持的机制与定义

随机投机解码用草稿分布 q 产生候选，由目标分布 p 验证。候选 x 的接受概率为 min(1,p(x)/q(x))，拒绝时从正剩余概率质量中重新采样；greedy 采用自己的确定性验证规则。

## InfraAgents 如何据此推导

以实际接受前缀作为状态提交点，分别管理预留、写入与确认长度。单位确认 token 成本由草稿、验证、管理开销和平均产出共同决定。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
