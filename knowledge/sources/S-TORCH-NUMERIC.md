# S-TORCH-NUMERIC：PyTorch — Numerical accuracy

## 本地完整原文

[完整原文阅读](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt) · [原始文件](../../references/S-TORCH-NUMERIC/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：PyTorch 2.14 documentation。

## 来源支持的机制与定义

浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。

## InfraAgents 如何据此推导

本库区分逐位一致、容差一致、argmax、概率分布和任务质量。用 logits margin 与首次分歧位置解释为何小误差可能改变后续文本。

## 阅读范围与出处身份

浮点非结合性、批次差异和精度边界。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
