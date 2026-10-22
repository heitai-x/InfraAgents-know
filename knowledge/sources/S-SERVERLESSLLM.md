# S-SERVERLESSLLM：ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models

## 本地完整原文

[完整原文](../../references/S-SERVERLESSLLM/reader.html) · [全文文本](../../references/S-SERVERLESSLLM/fulltext.txt)

固定版本：2401.14351v1。来源版本坐标日期：2024-01-25；归档与整理日期：2026-10-22。

## 来源支持的机制与定义

§4 的加载格式、块池和多级流水；§5 的 token 重算迁移；§6 的局部性与启动时间模型。

## InfraAgents 如何据此推导

从分片与目标张量布局推导加载阶段、峰值副本和就绪边界；用设定块大小分析流水，不移植论文实测加速比。

## 原文定位与阅读范围

PDF pp.1–8; §4、Figure 2 和加载依赖逐项核对。

正文当前引用的机制段已核对；整篇复核进度另记在构建证据中，不把下载成功当作全文理解。

## 适用边界

论文的加载路径与迁移假设属于所述系统，不能证明所有 mmap、直接 I/O 或迁移实现具有同样结果。 来源版本资格不证明模型训练集包含该资料；seed 不构成本项目实测有效性的声明。
