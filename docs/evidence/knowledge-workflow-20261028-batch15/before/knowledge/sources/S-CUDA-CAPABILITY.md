# S-CUDA-CAPABILITY：CUDA Programming Guide — Compute Capabilities

## 本地完整原文

[完整原文](../../references/S-CUDA-CAPABILITY/reader.html) · [全文文本](../../references/S-CUDA-CAPABILITY/fulltext.txt)

固定版本：CUDA 12.8.0。来源版本日期：2025-01-27；整理核对日期：2026-10-27。

## 来源支持的机制与定义

CUDA 的硬件功能与数值指令能力按 compute capability 和设备代际区分；一种 dtype 可被表示并不意味着所有算子形态都有高效实现。

## Warp矩阵操作的具体契约

归档包含完整CUDA 12.8.0编程指南。§7.24讨论warp矩阵操作：由整个warp协作执行D=A·B+C；本库实例选用文档支持的16×16×16、FP16乘数/FP32累加。矩阵加载、乘加和写回必须由全部warp线程以一致参数参加，条件执行时整个warp须作相同选择。

矩阵首地址须按256 bit（32字节）对齐，半精度leading dimension为8元素的倍数，FP32为4元素的倍数。row-major和col-major描述内存中的连续轴；它们不要求为了计算转置就一定复制矩阵。§7.24.1明确fragment内部的矩阵元素映射未公开保证，不能按数组下标臆造行列；需要逐行归约等非均匀操作时，可以先按定义布局写回共享/全局内存再访问。只有所有fragment元素统一执行的逐元素操作才不必先知道该映射。

本库[16×16 Attention实例](../frameworks/attention-backends.md)采用共享内存交接建立明确行列归属，并计入这份交通；这是独立算法选择，不是指南推荐的最佳kernel。warp内数据交接还需按§7.6的同步语义建立内存顺序；warp矩阵API中的同步不能代替任意前后共享内存依赖。原始fragment不能作为跨不同编译架构的稳定ABI传递，§7.24.5说明相应限制。

## InfraAgents 如何据此推导

本库分别分析存储、乘法、累加和输出精度，以及 tile、对齐、打包格式与真实 dispatch。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
