# K-SYS-006：工程认识如何成为可迁移的专家知识

知识的增量是对系统机制与设计关系的新增解释；运行结果为这些解释提供事实、区分和修正。

## 从现象走到解释

“改参数后更快”记录一个现象；“减少了哪些重复数据流、为何该运行点受到它限制”才形成可迁移解释。完整关系包含对象、依赖、干预、结果和竞争解释。例如开启前缀缓存后 TTFT 下降，可由少算 prompt、请求排队变化或预热差异导致，单个平均数不能区分。

系统知识可以来自算法定义、接口契约、代数推导和实际实验。来源支持一个机制存在，目标运行支持该机制在某组条件下的具体表现。将两者连起来，需要说明模型结构、硬件路径和 workload 如何满足推导前提。

## 什么值得进入正文

值得积累的是更准确的成本关系、更清晰的状态不变量、新发现的适用分支、可复现的反例及其原因，或原本分开的机制怎样组合。一次路径错误、固定文件名和临时命令通常属于运行记录；若它揭示更普遍的版本/所有权问题，再提炼其机制。

例如一次 graph 重放串请求的经历，可以推导“稳定地址与内容代次是两项独立契约”，并说明双缓冲何时足够、三个在途代次为什么还会冲突。这比增加一句“注意同步”提供更多设计能力。

## 独立核对的作用

实现者解释可能与其代码共享同一错误假设。诊断从任务函数、独立参考和原始执行观察出发，检验候选解释的可区分预测。局部 kernel 数值一致支持该 kernel 的某些输入范围，端到端 SLO 支持整个路径的一个运行点，各自都不能替代另一层证据。

RSI 的下一次使用是继续检验和扩展理解的机会。成功可能提示条件更广，失败可能揭示遗漏机制，也可能只是代码没有遵循知识。将后一种失败错误地当作知识反例，会让系统丢弃正确原理；将任务整体成功给所有被读条目记功，则会强化无效经验。

## 修订应保留推导能力

更新一个章节时，保留仍成立的对象模型、公式、布局和设计分支，把新认识放到它改变的地方。修订理由解释发生了何种认识变化。历史用于理解演化轨迹；正文应保持可独立阅读，而不是不断堆积时间顺序的日志。

<!-- infraagents:provenance -->

## 来源与推导

- S-ACE：[完整原文](../../references/S-ACE/reader.html) · [全文文本](../../references/S-ACE/fulltext.txt)
- S-GEPA：[完整原文](../../references/S-GEPA/reader.html) · [全文文本](../../references/S-GEPA/fulltext.txt)
- S-SELF-CORRECTION：[完整原文](../../references/S-SELF-CORRECTION/reader.html) · [全文文本](../../references/S-SELF-CORRECTION/fulltext.txt)
- S-REFLEXION：[完整原文](../../references/S-REFLEXION/reader.html) · [全文文本](../../references/S-REFLEXION/fulltext.txt)

- [S-ACE：Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](../sources/S-ACE.md)：ACE 的已读方法部分组织生成、反思和整理职责，并采用局部增量维护上下文条目，避免反复重写时丢失细节。

- [S-GEPA：GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](../sources/S-GEPA.md)：已核对摘要与官方项目说明：利用可评价轨迹中的反思反馈提出候选，并依据评价保留具有价值或互补性的候选。

- [S-SELF-CORRECTION：Large Language Models Cannot Self-Correct Reasoning Yet](../sources/S-SELF-CORRECTION.md)：该研究讨论其所考察设置下，仅依赖模型自身、缺少外部反馈的推理自纠正能力限制。结论具有实验设置范围。

- [S-REFLEXION：Reflexion: Language Agents with Verbal Reinforcement Learning](../sources/S-REFLEXION.md)：Reflexion 把任务反馈转为文字反思，并在后续尝试中使用这些记忆。改进载体是显式经验文本，而非必须修改模型权重。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 4，2026-09-20。

## 延伸阅读

- [正确性 Oracle：函数参考、状态变形与服务契约](../tools/correctness-oracles.md)。
- [失败分析：最早偏离、最小复现与机制归因](../tools/failure-evidence.md)。
- [系统设计推导：把模型、缓存、执行与目标组合起来](../system-semantics/design-synthesis.md)。
