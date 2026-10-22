# S-LDM：High-Resolution Image Synthesis with Latent Diffusion Models

## 本地完整原文

[完整原文阅读](../../references/S-LDM/reader.html) · [全文文本](../../references/S-LDM/fulltext.txt) · [原始文件](../../references/S-LDM/original.pdf)

完整 PDF：45 页，版本 2112.10752v2。

资料形态：按已核对范围整理的本地非代码释义。原文版本：CVPR 2022。

## 来源支持的机制与定义

Latent Diffusion 把去噪计算放在编码器形成的潜空间，通过条件机制引导生成，并最终解码到像素空间。

## InfraAgents 如何据此推导

本库区分固定文本条件与逐步变化的 latent，解释哪些 context 可以复用，哪些 self-attention 状态会随去噪步改变。

## 阅读范围与出处身份

潜空间扩散与条件交叉注意力。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
