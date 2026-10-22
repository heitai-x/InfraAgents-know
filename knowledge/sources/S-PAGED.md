# S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention

## 本地完整原文

[完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)

固定版本：2309.06180v1。来源版本日期：2023-09-12；整理核对日期：2026-10-22。

## 来源支持的机制与定义

PagedAttention 把每条序列的逻辑 KV 划分为固定 token 数的块，用映射表访问不连续物理存储。多个序列可以共享相同前缀的物理块，分叉写入需要保持各序列状态独立。

## InfraAgents 如何据此推导

由层数、KV head 数、通道、dtype 和块长计算每块字节，再计算尾部碎片与共享收益。引用保护和设备完成共同决定块能否重用；分页本身不消除 attention 对可见历史的读取。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
