# S-TRITON：Triton：Fused Attention

## 本地完整原文

[完整原文](../../references/S-TRITON/reader.html) · [全文文本](../../references/S-TRITON/fulltext.txt)

固定版本：Git 042b2cdcb41bb15821682cb6ba4bb15b1f15abb3 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

块级张量程序语言的 fused attention 教程展示了五个关系：一个 kernel 的执行域由几何假设（head 维相等、batch/head/token 共同展平、紧密的张量描述符）隐含在地址计算里；在线 softmax 可以在以 2 为底的域内维护最大值与归一化质量，持久化的统计量因此带有底数这一语义；概率在参与 PV 乘法前转换为低精度而累加保持 FP32，"FP32 累加"不描述整条数值路径；autotune 以 key 与候选剪枝共同定义合法域，测试配置、搜索候选与最优配置是三个不同集合；按设备调整 warp 分工、流水级数与寄存器上限是条件化的实现选择。

## InfraAgents 如何据此推导

本库从中抽出版本无关的关系：示例 kernel 先还原执行域再谈接入；跨 kernel 中间统计量的底数与精度是契约的一部分；编译变体、合法域与调优结果是三个层次；带副作用的 kernel 调优需要隔离初始状态。它们在 [Triton Kernel](../frameworks/triton-kernels.md)、[Attention 的 IO](../algorithms/attention-io.md) 与 [基础算子](../algorithms/operator-primitives.md) 中使用。

## 核对范围与适用边界

核对范围：教程的前向内核、描述符预处理、autotune 配置与剪枝、前向/反向封装及其测试。这是一份教程在一个提交上的实现选择，既不是块程序语言的能力上限，也不是任意 attention 的替换接口；接入时按归档原文核对几何假设、候选集合与特殊值处理。设计推导与数值例为独立分析，不是目标机器实测；版本日期不证明模型训练实际包含该资料。
