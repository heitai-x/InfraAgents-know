# S-RESNET：Deep Residual Learning for Image Recognition

## 本地完整原文

[完整原文](../../references/S-RESNET/reader.html) · [全文文本](../../references/S-RESNET/fulltext.txt)

固定版本：1512.03385v1。版本坐标日期：2015-12-10；整理日期：2026-10-23。

## 来源支持的机制与定义

固定 v1 给出残差/投影 shortcut、卷积结构、推理视图与分类指标，并在附录给出基于 ResNet 的 proposal/RoI/分类与框回归检测路线。

## InfraAgents 如何据此推导

独立推导卷积/im2col 资源、固定 BN 折叠与残差存活期，连接图像前处理和检测坐标输出契约。

## 原文定位与阅读范围

PDF pp.1–8 正文、pp.10–12 附录 A–C 全部技术内容；重点 §3.2–3.4、Table 1 和检测附录。p.9 为参考文献。

## 适用边界

不同预处理、多尺度与分类/检测任务不可混作一个 baseline；附录中的历史 NMS 阈值/proposal 数不当作通用默认。正文设定算例与论文性能数据严格分开。

seed 不表示当前 Task 上已实测有效；归档完整与编辑复核完成是两项不同事实。来源时间不证明模型训练实际使用该资料。
