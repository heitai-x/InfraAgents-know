# S-FLEXGEN：High-throughput Generative Inference of Large Language Models with a Single GPU

## 本地完整原文

[完整原文](../../references/S-FLEXGEN/reader.html) · [全文文本](../../references/S-FLEXGEN/fulltext.txt)

固定版本：2303.06865v1。版本坐标日期：2023-03-13；整理日期：2026-10-22。

## 来源支持的机制与定义

按层/批遍历、权重/激活/KV 三层放置、CPU attention 委托与理想重叠成本模型；4-bit 表示、V 稀疏读取及吞吐/延迟评估。

## InfraAgents 如何据此推导

据合法执行依赖比较层间权重复用、迁移与重算，分别计算容量、每轮流量和每输出摊销；在线与离线目标分开。

## 原文定位与阅读范围

完整技术正文 PDF pp.1–11，以及附录 A.1–A.4 pp.15–25 已读；pp.12–14 为参考文献，未递归引入外链。核对 §4 搜索空间、§5 近似方法、§6 评估与附录理论/成本模型/额外实验。

本轮完整技术复核日期：2026-10-23。原始归档与版本日期保持不变。

## 适用边界

固定 v1 面向允许高延迟的吞吐任务。§4 成本模型假定理想重叠并放松离散放置，可能 OOM，作者需要手动调整；附录近优 I/O 结论限定无 CPU 委托、无限规则工作流与简化存储模型，并不证明端到端调度最优。§5 压缩格式需解回 FP16，因 CPU 解压成本该实现启用量化时关闭 CPU delegation；其 4-bit 与稀疏质量结果仅覆盖所测 OPT/任务。吞吐实验使用 dummy weights，部分长运行外推；质量实验使用真实权重。附录政策消融与主表的 OS page cache 设置不同，不能直接混合收益数字。

来源版本资格不证明训练集成员关系，seed 不表示目标 Task 已实测有效。

## 完整复核记录

[定位、条件与独立核验](../../docs/evidence/source-review-20261023/S-FLEXGEN/review.json)。
