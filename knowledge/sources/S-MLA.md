# S-MLA：DeepSeek-V2：MLA 与解耦位置编码

## 本地完整原文

[完整原文阅读](../../references/S-MLA/reader.html) · [全文文本](../../references/S-MLA/fulltext.txt) · [原始文件](../../references/S-MLA/original.pdf)

完整 PDF：52 页，版本 2405.04434v5。

资料形态：按已核对范围整理的本地非代码释义。原文版本：v5 / 2024。

## 来源支持的机制与定义

DeepSeek-V2 的 MLA 以低维 latent 联合表示内容 K/V，通过线性投影的吸收减少历史展开；位置相关旋转不能任意与投影交换，因此采用解耦 RoPE 分量。

## InfraAgents 如何据此推导

本库用 qᵀWk c＝((Wk)ᵀq)ᵀc 与线性值归约解释吸收，按 L(d_c＋d_R)b 计算缓存，并区分 prefill 与 decode 的执行路径。

## 阅读范围与出处身份

第 2.1 节压缩、矩阵吸收和解耦 RoPE。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
