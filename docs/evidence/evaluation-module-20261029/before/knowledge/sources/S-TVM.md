# S-TVM：TVM

## 本地完整原文

[完整原文](../../references/S-TVM/reader.html) · [全文文本](../../references/S-TVM/fulltext.txt)

固定版本：1802.04799v1。版本坐标日期：2018-02-12；整理日期：2026-10-23。

## 来源支持的机制与定义

固定 v1 分离图级改写、张量计算、schedule 与硬件 intrinsic，描述融合、布局、协作加载、tensorization、显式同步和 code generation/runtime。

## InfraAgents 如何据此推导

把 IR/lowering 接回推理副作用、别名和在途生命周期，并按整个 producer/consumer 链计算转换成本。

## 原文定位与阅读范围

PDF pp.1–17 至参考文献前全部正文，重点 §2–4、Fig.4–18、§5 的实验边界；无额外技术附录。

## 适用边界

固定 2018 v1，不将后续 Relay/Relax 或其他 API 当作本篇已描述能力。其硬件与 baseline 属当时实验；自动优化仍需目标语义与实际 dispatch 验证。

seed 不表示当前 Task 上已实测有效；归档完整与编辑复核完成是两项不同事实。来源时间不证明模型训练实际使用该资料。
