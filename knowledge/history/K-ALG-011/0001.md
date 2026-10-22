# K-ALG-011：批量采样：概率变换、请求随机性与行映射

采样器消费每个请求自己的条件分布；向量化改变执行形态，不能改变参数、随机状态或输出归属。

## 采样之前已经发生了什么

logits 的行并不总与 batch 请求一一对应。prefill 可能输出所有输入行，但只需取最后有效位置；投机验证需要一段位置；并行候选又可能让一个请求占多行。先确定“请求、逻辑位置、分支 → logits 行”的映射，才能定义每行的采样参数。

temperature 缩放、重复惩罚、禁用 token、grammar mask、top-k 和 top-p 是有顺序的变换。top-p 根据排序后的归一化概率保留最小累计概率集合；它与 temperature 一般不交换。对所有 logits 同加常数不改变 softmax，乘常数则改变熵。temperature＝0 应按规定的 greedy 语义处理，不是直接除以零。

## 用一个小分布说明顺序

设原分布为 0.5、0.3、0.2。top-p＝0.6 通常保留前两项并重新归一化；若先用低温使第一项概率超过 0.6，则只保留第一项。若 grammar 禁止第一项，必须在合法集合中重新定义概率与截断，否则可能得到空集合或错误归一化。空合法集合应对应协议中的失败或恢复策略，不能静默抽取非法 token。

## 每请求的随机轨迹

共享一个按 batch 行连续消耗的随机流，会使请求输出受其他请求的加入、取消与重排影响。若任务要求调度无关的可复现性，可将随机流绑定请求种子、逻辑位置和采样次数；投机接受、拒绝重采样也要有明确的子序列定义。普通 seed 固定只保证某种执行路径下的随机初始化，并不自动提供这种更强性质。

在设备上批量完成筛选、采样和 EOS 判断可避免逐请求主机同步。不同 top-k、top-p、temperature 可以表现为逐行参数，不必每种参数拆一个 batch。若算法内部成本随参数变化，分组可能提高效率，但分组后的逆映射必须保持请求身份。

## 哪些验证支持哪种结论

greedy 的目标是匹配规定的 argmax 与平局规则。随机采样的目标是匹配分布，可以对固定小概率向量比较频率、支持集与置信区间。相同 seed 的单条文本不同，不能单独证明分布错误；非法 token、概率质量丢失或请求间随机状态串用，则是不同层面的明确问题。

<!-- infraagents:provenance -->

## 来源与推导

- [S-FI-SAMPLE：FlashInfer：Sampling](../sources/S-FI-SAMPLE.md)：FlashInfer sampling 接口提供从概率或 logits 的按行采样，以及 top-k、top-p 等相关操作；调用的输入表示和参数决定实际概率处理。
  [完整原文](../../references/S-FI-SAMPLE/reader.html) · [全文文本](../../references/S-FI-SAMPLE/fulltext.txt)

- [S-TORCH-REPRO：PyTorch — Reproducibility](../sources/S-TORCH-REPRO.md)：随机种子与确定性配置可以控制特定执行路径中的随机性，但不保证跨版本、平台和全部算法路径逐位一致。
  [完整原文](../../references/S-TORCH-REPRO/reader.html) · [全文文本](../../references/S-TORCH-REPRO/fulltext.txt)

- [S-SPEC：Fast Inference from Transformers via Speculative Decoding](../sources/S-SPEC.md)：随机投机解码用草稿分布 q 产生候选，由目标分布 p 验证。候选 x 的接受概率为 min(1,p(x)/q(x))，拒绝时从正剩余概率质量中重新采样；greedy 采用自己的确定性验证规则。
  [完整原文](../../references/S-SPEC/reader.html) · [全文文本](../../references/S-SPEC/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [请求身份：逻辑序列、批次视图与结果归属](../system-semantics/request-batch-identity.md)。
- [结构化生成：语言状态、token 边界与约束采样](../algorithms/structured-output.md)。
- [投机解码：分布保持、状态提交与收益模型](../algorithms/speculation-budget.md)。
