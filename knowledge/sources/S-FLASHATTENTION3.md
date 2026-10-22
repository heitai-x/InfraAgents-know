# S-FLASHATTENTION3：FlashAttention-3

## 本地完整原文

[完整原文](../../references/S-FLASHATTENTION3/reader.html) · [全文文本](../../references/S-FLASHATTENTION3/fulltext.txt)

固定版本：2407.08608v1。版本坐标日期：2024-07-11；整理日期：2026-10-22。

## 来源支持的机制与定义

§2–3 以 Hopper 的 TMA、异步 WGMMA、warp specialization 和多级缓冲组织 attention；讨论寄存器、布局和 FP8 数值限制。

## InfraAgents 如何据此推导

将 GEMM/attention 的数据、归约和缓冲生命周期分开推导，独立计算资源账与在线 softmax 缩放，不把伪代码直接当作实现。

## 原文定位与阅读范围

PDF pp.2–9，§2–3 以及 §4 开头；p.5 Algorithm 1 已视觉核对。剩余实验、附录与完整阅读尚待继续。

## 适用边界

固定 v1 的 Algorithm 1 第 21 行对旧输出的逆缩放与在线归一化推导不一致，已视觉核对。正文使用 exp(m_old−m_new) 对未归一化旧累加量缩放，并给出两块反例。此处不声称仓库代码有相同问题；硬件和 FP8 结果均有具体平台/形状边界。

seed 不表示当前 Task 上已实测有效；归档完整与编辑复核完成是两项不同事实。来源时间不证明模型训练实际使用该资料。
