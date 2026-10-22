# S-ROCM：HIP：Hardware implementation

## 本地完整原文

[完整原文阅读](../../references/S-ROCM/reader.html) · [全文文本](../../references/S-ROCM/fulltext.txt) · [原始文件](../../references/S-ROCM/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：HIP 7.15.0 documentation snapshot。

## 来源支持的机制与定义

HIP 硬件说明介绍 CU、SIMD、向量寄存器、LDS 与矩阵运算资源。寄存器使用影响驻留和隐藏延迟能力，不同 AMD 架构的资源参数不同。

## InfraAgents 如何据此推导

本库重新推导 tile、wave 和数据布局，而不继承 CUDA 硬编码常量；同一模型函数可选择不同平台 kernel。

## 阅读范围与出处身份

CU、SIMD、寄存器压力、LDS 与 AMD 执行模型。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
