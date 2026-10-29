# S-HF-CACHE：Transformers — Caching

## 本地完整原文

[完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

固定版本：Git 756dffe0d664419d2f620e131ed426373cfd6f12 (2026-04-30)。来源版本日期：2026-04-30；整理核对日期：2026-10-22。

## 来源支持的机制与定义

固定文档说明逐层复用历史 K/V、计算并合并新增 K/V，以及增量循环的 attention mask 应覆盖历史与新增 token；并区分 Dynamic、Static、StaticSlidingWindow 等缓存存储形态。

## InfraAgents 如何据此推导

本库据历史与新增状态的依赖关系，将逻辑位置、可见性和物理槽位分开，构造整段、分块、逐步和合法前缀命中的等价检查。具体 API 字段名、维度与 padding 规则属于 Task 选定实现的合同，不能仅凭字段名推断。

## 适用边界

绑定归档的 cache_explanation.md 完整文档。该文件未定义 cache_position 或 position_ids 的参数合同，不能将这些字段的具体含义归因于本页；逻辑位置、模型位置编码与物理槽关系由机制推导并按 Task 固定实现核对。文中 score 与 mask 相乘的示意不构成布尔屏蔽实现规则：对不可见位置仅把 score 乘零，不能保证 softmax 后概率为零；应按实际可见集合或等价的加性遮罩定义。存储类型和编译描述不保证任意模型/形状组合，亦不构成目标性能证明。

原始资料和版本日期保持不变；本轮归因复核日期为 2026-10-23。来源资格不证明模型训练成员关系。
