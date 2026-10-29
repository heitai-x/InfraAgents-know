# S-FLASHATTENTION3：FlashAttention-3

## 本地完整原文

[完整原文](../../references/S-FLASHATTENTION3/reader.html) · [全文文本](../../references/S-FLASHATTENTION3/fulltext.txt)

固定版本：2407.08608v1。版本坐标日期：2024-07-11；整理日期：2026-10-22。

## 来源支持的机制与定义

Hopper TMA/WGMMA 的生产者消费者组织、跨 warpgroup 与组内 GEMM-softmax 流水、片上资源与 FP8 布局；block quantization/正交变换的数值条件及固定形状的性能评估。

## InfraAgents 如何据此推导

将 GEMM/attention 的数据、归约和缓冲生命周期分开推导，独立计算资源账与在线 softmax 缩放，不把伪代码直接当作实现。

## 原文定位与阅读范围

完整技术正文 PDF pp.1–12 与附录 A–C pp.17–22 已读；pp.12–16 参考文献不递归引入。Algorithm 1 p.5、Algorithm 2 p.7、Algorithm 4 p.20 的原始 PDF 已再次逐页视觉核对。

本轮完整技术复核日期：2026-10-23。原始归档与版本日期保持不变。

## 适用边界

固定 v1 的 Algorithm 1 第21行旧输出逆缩放与未归一化累加关系不一致；本库保留独立推导 exp(m_old−m_new)。Algorithm 2/4 的打印循环边界和部分状态交接也不能直接当完整程序：如 Algorithm 2 在 Tc=2 时循环不执行且未计算 K1 对应分数，Algorithm 4 打印范围漏掉末端 K 块。这里仅判断固定论文伪代码，不声称源码实现同样有错。三阶段变体受寄存器压力和编译重排影响，在所述实验中弱于两阶段；FP8误差结果基于合成异常值分布，不是端到端模型质量。主要速度比较绑定 H100 SXM5、固定时钟/软件与特殊形状，§5 仍把 LLM inference 优化列为后续方向。

来源版本资格不证明训练集成员关系，seed 不表示目标 Task 已实测有效。

## 完整复核记录

[定位、条件与独立核验](../../docs/evidence/source-review-20261023/S-FLASHATTENTION3/review.json)。
