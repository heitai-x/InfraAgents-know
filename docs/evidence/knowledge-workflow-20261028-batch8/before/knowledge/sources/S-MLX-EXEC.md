# S-MLX-EXEC：MLX：Compilation 与 Streams（固定历史文档）

## 本地完整原文

[完整原文](../../references/S-MLX-EXEC/reader.html) · [全文文本](../../references/S-MLX-EXEC/fulltext.txt)

固定版本：Git e8ebdebeeb655feaa85a51f6b24ece5b6d5518d1 (2026-04-28)。版本坐标日期：2026-04-28；整理日期：2026-10-23。

## 来源支持的机制与定义

同一固定 MLX 提交的 Compilation 与 Using Streams 完整文档，说明 tracing、编译复用、输入与输出状态捕获、shapeless 的静态形状陷阱及 operation 的 stream/device 选择。

## InfraAgents 如何据此推导

据显式依赖和纯函数契约推导推理 KV/位置/随机状态的输入输出边界、求值完成和编译成本摊销；所有服务容量和时间算例为独立设定。

## 原文定位与阅读范围

已完整阅读 docs/src/usage/compile.rst 全部段落与示例，及 docs/src/usage/using_streams.rst 全文；固定提交 e8ebdebeeb655feaa85a51f6b24ece5b6d5518d1，2026-04-28。与现有 S-MLX、S-MLX-MEM 完整原文交叉核对。

## 适用边界

文档中的示例性能不作为目标机器结果；Using Streams 本文仅提供 stream/device 指定规则，不足以独立证明任意异步取消或外部内存写入协议。Lazy Evaluation 同提交中仍有早期不编译表述，编译能力据专门 Compilation 文档确认。

seed 不表示当前 Task 上已实测有效；归档完整与编辑复核完成是两项不同事实。来源时间不证明模型训练实际使用该资料。
