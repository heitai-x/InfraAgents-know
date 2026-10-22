# S-FLEXGEN：High-throughput Generative Inference of Large Language Models with a Single GPU

## 本地完整原文

[完整原文](../../references/S-FLEXGEN/reader.html) · [全文文本](../../references/S-FLEXGEN/fulltext.txt)

固定版本：2303.06865v1。版本坐标日期：2023-03-13；整理日期：2026-10-22。

## 来源支持的机制与定义

§3–4 将生成、状态容量、层/批遍历与 GPU/CPU/disk 放置一起建模；§4.2 讨论把部分主存 KV 对应计算交给 CPU。

## InfraAgents 如何据此推导

据合法执行依赖比较层间权重复用、迁移与重算，分别计算容量、每轮流量和每输出摊销；在线与离线目标分开。

## 原文定位与阅读范围

PDF pp.3–6，§3、§4.1–4.3 与 Figure 3–4；机制引用已读，完整实验和附录复核仍待继续。

## 适用边界

固定 v1 面向吞吐导向和允许较高延迟的任务；理想重叠与线性成本需由目标形态校准，不移植论文速度或将某个历史长度阈值视为通用结论。

seed 不表示当前 Task 上已实测有效；归档完整与编辑复核完成是两项不同事实。来源时间不证明模型训练实际使用该资料。
