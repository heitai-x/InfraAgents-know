# S-CUDA-ALLOC：CUDA Programming Guide — Stream-Ordered Memory Allocator

## 本地完整原文

[完整原文](../../references/S-CUDA-ALLOC/reader.html) · [全文文本](../../references/S-CUDA-ALLOC/fulltext.txt)

固定版本：CUDA 12.8.0。来源版本日期：2025-01-27；整理核对日期：2026-10-22。

## 来源支持的机制与定义

stream-ordered allocator 将分配和释放纳入执行顺序。跨 stream 访问必须存在与分配、使用和释放相容的依赖。

## InfraAgents 如何据此推导

释放请求的逻辑引用并不证明地址可立刻重用。本库将所有权与最后一个设备/传输消费者的完成条件合并，推导安全回收点。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
