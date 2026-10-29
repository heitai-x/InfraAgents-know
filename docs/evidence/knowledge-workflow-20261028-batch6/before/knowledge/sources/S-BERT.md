# S-BERT：BERT

## 本地完整原文

[完整原文](../../references/S-BERT/reader.html) · [全文文本](../../references/S-BERT/fulltext.txt)

固定版本：1810.04805v1。版本坐标日期：2018-10-11；整理日期：2026-10-23。

## 来源支持的机制与定义

固定 v1 描述双向 Transformer encoder、token/segment/position 输入、CLS 句级任务头及句对/序列标注任务。

## InfraAgents 如何据此推导

从完整编码函数推导 embedding 缓存身份、双向可见性、pooling/截断及句对重排的 batch 契约。

## 原文定位与阅读范围

PDF pp.1–13 至参考文献前全部正文；重点 §3.1–3.5、§4 任务输入/输出、§5.4 特征式使用。该 v1 没有额外技术附录。

## 适用边界

原始 BERT 与任意 pooling 不自动构成高质量检索模型；双塔检索及归一化作为需要训练制品定义的设计选择。论文任务结果不作为目标服务质量保证。

seed 不表示当前 Task 上已实测有效；归档完整与编辑复核完成是两项不同事实。来源时间不证明模型训练实际使用该资料。
