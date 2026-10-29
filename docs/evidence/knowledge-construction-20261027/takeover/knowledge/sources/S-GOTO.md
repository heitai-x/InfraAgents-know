# S-GOTO：Anatomy of High-Performance Matrix Multiplication

## 本地完整原文

[完整原文](../../references/S-GOTO/reader.html) · [全文文本](../../references/S-GOTO/fulltext.txt)

固定版本：Author manuscript archived at tpn/pdfs commit 67ceef2b223ca15bdf6e257303df4fea63970aaa。来源版本坐标日期：2016-11-30；归档与整理日期：2026-10-22。

## 来源支持的机制与定义

§3–6 从多级存储模型推导 GEMM 的分解、panel 打包、cache/TLB 条件和实现参数。

## InfraAgents 如何据此推导

将分块原理应用到固定权重的重用、GEMV/GEMM 区别和线程输出分区；算例使用自行声明的 shape 与带宽。

## 原文定位与阅读范围

PDF pp.1–25 全文已读。重点核对 §3–6 的分块、打包、寄存器与缓存约束；§7 的历史平台参数与实验没有迁移为现代 CPU 的常量或 InfraAgents 的性能证据。p.8 的存储模型与公式已进行视觉核对。

该作者稿的完整阅读和引用范围记录在构建证据中；没有复现论文实验，也没有独立审核者的结论。

## 适用边界

完整归档的是带 20YY 占位印记的作者稿，非 ACM 最终排版版。日期绑定 2016 年历史镜像提交。文中旧处理器的 TLB、缓存和流水描述不作为现代 CPU 的无条件事实。 来源版本资格不证明模型训练集包含该资料；seed 不构成本项目实测有效性的声明。
