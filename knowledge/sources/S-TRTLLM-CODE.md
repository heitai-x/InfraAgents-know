# S-TRTLLM-CODE：TensorRT-LLM PyExecutor 固定提交

## 本地完整原文

[完整原文](../../references/S-TRTLLM-CODE/reader.html) · [全文文本](../../references/S-TRTLLM-CODE/fulltext.txt)

固定版本：Git 0343a9d9f2a8f69f9fe783834b042ff7687a28a4 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

固定历史提交的 PyExecutor 包含调度准备、_schedule、_sample_async 和异步传输管理。AsyncTransferManager 在发送与保存的全部使用者完成后解除 KV 保护；加载状态由其他路径处理。

## InfraAgents 如何据此推导

本库从该责任划分解释业务完成与 KV 回收的差别。文件 SHA 和方法位置存于 docs/evidence/expert-source-locations-20260920.json。

## 适用边界

所列机制与接口资料；本文设计推导和数值例为独立分析，不是目标机器实测。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
