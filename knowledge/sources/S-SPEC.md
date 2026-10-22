# S-SPEC：Fast Inference from Transformers via Speculative Decoding

## 本地完整原文

[完整原文阅读](../../references/S-SPEC/reader.html) · [全文文本](../../references/S-SPEC/fulltext.txt) · [原始文件](../../references/S-SPEC/original.pdf)

完整 PDF：13 页，版本 2211.17192v2。

资料形态：按已核对范围整理的本地非代码释义。原文版本：2023 / v2。

## 来源支持的机制与定义

随机投机解码用草稿分布 q 产生候选，由目标分布 p 验证。候选 x 的接受概率为 min(1,p(x)/q(x))，拒绝时从正剩余概率质量中重新采样；greedy 采用自己的确定性验证规则。

## InfraAgents 如何据此推导

以实际接受前缀作为状态提交点，分别管理预留、写入与确认长度。单位确认 token 成本由草稿、验证、管理开销和平均产出共同决定。

## 阅读范围与出处身份

草稿与目标模型验证的加速思路和输出语义。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
