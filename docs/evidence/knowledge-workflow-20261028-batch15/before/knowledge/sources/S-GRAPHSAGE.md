# S-GRAPHSAGE：Inductive Representation Learning on Large Graphs

## 本地完整原文

[完整原文](../../references/S-GRAPHSAGE/reader.html) · [全文文本](../../references/S-GRAPHSAGE/fulltext.txt)

固定版本：1706.02216v1。版本坐标日期：2017-06-07；整理日期：2026-10-23。

## 来源支持的机制与定义

固定 v1 给出邻居采样、均值/pooling/LSTM 聚合与逐层表示，包含 minibatch 依赖、图数据预处理、实验和有条件的表达能力分析。

## InfraAgents 如何据此推导

从自身分支与邻居依赖推导逐层集合及容量上界，区分同层去重读取、边 multiplicity 和采样质量。

## 原文定位与阅读范围

PDF pp.1–8 正文、p.10 末至 p.18 附录 A–E 全部技术内容；重点 Algorithm 1、§3.1–3.3、Appendix A/C 与理论条件。

## 适用边界

采样不保证全邻域数值一致，LSTM 聚合不严格排列不变。附录实验存在度数预处理；表达能力定理是存在性结论而非给定 checkpoint 的质量保证。正文使用独立集合递推，不照搬附录伪代码的符号排版。

seed 不表示当前 Task 上已实测有效；归档完整与编辑复核完成是两项不同事实。来源时间不证明模型训练实际使用该资料。
