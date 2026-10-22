# S-CHAT：Transformers：Chat templates

## 本地完整原文

[完整原文阅读](../../references/S-CHAT/reader.html) · [全文文本](../../references/S-CHAT/fulltext.txt) · [原始文件](../../references/S-CHAT/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：4.57.1。

## 来源支持的机制与定义

Chat template 将消息序列变为包含角色边界、特殊 token 和生成提示的模型输入；tokenizer 自动添加特殊 token 的方式会影响最终序列。

## InfraAgents 如何据此推导

本库把模板和 tokenizer 版本纳入输入语义，分析重复 BOS/EOS、padding、截断与前缀缓存身份。

## 阅读范围与出处身份

消息到控制 token 与生成提示的序列化。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
