# S-INT8：Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference

## 本地完整原文

[完整原文](../../references/S-INT8/reader.html) · [全文文本](../../references/S-INT8/fulltext.txt)

固定版本：1712.05877v1。来源版本坐标日期：2017-12-15；归档与整理日期：2026-10-22。

## 来源支持的机制与定义

§2 给出仿射量化、整数点积、零点补偿、bias 和输出重新缩放；附录 A–B 讨论组合算子、舍入及 NEON 中间累加条件。

## InfraAgents 如何据此推导

将整数代数与权重低位存储、tile 解包和 CPU 成本账连接；组尺度与示例数值是明确设定。

## 原文定位与阅读范围

PDF pp.2–5、11–13; §2.1–2.4 与附录 B；p.3 公式视觉核对。

正文当前引用的机制段已核对；整篇复核进度另记在构建证据中，不把下载成功当作全文理解。

## 适用边界

论文的整数路径和 NEON 限值不是任意 INT4/INT8 硬件的通用能力保证；质量需要模型和校准分布验证。 来源版本资格不证明模型训练集包含该资料；seed 不构成本项目实测有效性的声明。
