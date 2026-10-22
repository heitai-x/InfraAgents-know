# S-RMSNORM：Root Mean Square Layer Normalization

## 本地完整原文

[完整原文](../../references/S-RMSNORM/reader.html) · [全文文本](../../references/S-RMSNORM/fulltext.txt)

固定版本：1910.07467v1。来源版本坐标日期：2019-10-16；归档与整理日期：2026-10-22。

## 来源支持的机制与定义

§3–4 比较 LayerNorm 与 RMSNorm，给出不减均值的 RMS 归一化关系及其不变性边界。

## InfraAgents 如何据此推导

据归约域推导存储和融合选择；具体模型的 epsilon、bias 与 dtype 要独立核对。

## 原文定位与阅读范围

PDF pp.1–4; p.3 的公式和 p.4 的不变性条件。

正文当前引用的机制段已核对；整篇复核进度另记在构建证据中，不把下载成功当作全文理解。

## 适用边界

不把训练时可替换的架构选择解释为已训练模型推理时可任意替换；不引用论文速度数字作本项目收益。 来源版本资格不证明模型训练集包含该资料；seed 不构成本项目实测有效性的声明。
