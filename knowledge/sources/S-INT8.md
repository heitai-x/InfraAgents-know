# S-INT8：Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference

## 本地完整原文

[完整原文](../../references/S-INT8/reader.html) · [全文文本](../../references/S-INT8/fulltext.txt)

固定版本：1712.05877v1。来源版本坐标日期：2017-12-15；归档与整理日期：2026-10-22。

## 来源支持的机制与定义

§2 给出仿射量化、整数点积、零点补偿、bias 和输出重新缩放；附录 A–B 讨论组合算子、舍入及 NEON 中间累加条件。

## InfraAgents 如何据此推导

将整数代数与权重低位存储、tile 解包和 CPU 成本账连接；组尺度与示例数值是明确设定。

## 原文定位与阅读范围

PDF pp.1–14 正文、实验与附录均已读；§2 仿射量化与零点补偿，§3 训练量化关系，附录 A–B 的融合、舍入和 NEON 中间累加条件重点复核。p.3 公式已视觉核对。

## 适用边界

论文的整数路径和 NEON 限值不是任意 INT4/INT8 硬件的通用能力保证；质量需要模型和校准分布验证。 来源版本资格不证明模型训练集包含该资料；seed 不构成本项目实测有效性的声明。


## 整篇复核后的使用限制

实验以 MobileNet/ResNet 等指定视觉模型和 ARM 平台为条件。权重量化文件变小不自动等于 CPU 计算更快；窄中间累加优化依赖极值范围限制。未将论文训练质量或速度值转写成 InfraAgents 实测。
