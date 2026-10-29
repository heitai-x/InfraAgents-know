# K-ALG-012：结构化生成：语言状态、token 边界与约束采样

结构化生成将语法状态与语言模型状态同步推进，合法 token 是从当前语法状态仍可完成目标语言的延伸。

## 字符语言与 token 语言之间

JSON、正则和上下文无关语法通常描述字符或字节串，模型却按 token 生成。一个 token 可能跨越多个语法边界，也可能只是一段 UTF-8 字节。合法性判断要将 token 的展开内容作用到当前解析状态，而不能只检查其首字符。接受一个暂未完成字符串的 token 可以合法，只要后面仍存在可接受的完成路径。

语法状态可包括栈、字符串转义状态、未完成字节和已生成的结构字段。EOS 只有在当前状态允许结束时才合法。JSON 格式合法与 JSON Schema 的全部语义约束不是同一层：字段类型、范围、跨字段关系和业务含义需要各自对应的表达能力或后续验证。

## 约束怎样进入分布

设当前合法 token 集 A(s)，新分布 p′(x) 在 x∈A(s) 时与 p(x) 成比例，其余为零。这样改变了原模型分布，但满足规定语言。采样完成后，语法状态沿实际选中的 token 前进。将 grammar mask 提前计算并与 GPU 前向重叠，要求所用状态已确认；不能拿上一 token 之前的 mask 套在新 logits 上。

大量 token 的合法性与具体栈上下文无关，可预计算；剩余 token 才需要在线解析。XGrammar 的设计利用了这种分离。词表、tokenizer 和语法版本共同决定预计算结果，模型换 tokenizer 后沿用旧 mask 缓存不成立。

## 与投机、取消和多租户的组合

投机分支需要对应的语法分支状态；拒绝候选后回到最后接受位置。共享不可变的编译语法可以节省成本，但每请求的解析栈、已消费位置和输出不能共用可变对象。取消需要释放解析状态和设备 mask 缓冲区，同时保留在途执行的生命周期。

结构化输出性能取决于模型前向、mask 生成和传输的关键路径。预编译较复杂语法可能增加首次请求延迟，却提高后续吞吐。若把编译隐藏在预热里，应将冷启动与稳定服务分别建模。Agent 应从语言表达能力、逐 token 状态同步和缓存身份推导接入方式，再选择库提供的接口。

## 约束状态也是分支状态

一个树形候选需要自己的解析状态快照或可重放路径；兄弟分支不能共享可变解析栈。接受一条路径时，与 KV、随机状态和输出一起提交到同一位置，拒绝后撤销未确认的 mask/解析推进。预先生成 mask 的异步工作要携带状态代次，迟到 mask 不得用于新 token 的 logits。

应用 constraint mask 后的目标分布，与无约束原分布不同。若叠加投机，p/q 修正必须对应实际约束与截断顺序；某个草稿分支合法不等于它按目标概率被选择。grammar 编译缓存可以共享不可变内容，但 tokenizer、词表和规则版本变化应使不相容缓存退出可用集合。

## 验证语言与服务两层边界

用小词表枚举合法延伸，覆盖跨 token 的转义、UTF-8、嵌套结构、EOS 和空合法集合；再测试分支回滚、取消和模型切换。格式通过后还应检查 schema/业务层约束，不把语法解析成功当作内容正确。

性能区分首次规则编译、每 token mask 生成、传输与设备计算。将复杂约束预热移出计时，可以回答稳态问题，但不能支持冷启动同样快的结论。长输出、不同 grammar 与输入分布需要独立保留验证。

<!-- infraagents:provenance -->

## 来源与推导

- [S-XGRAMMAR：XGrammar](../sources/S-XGRAMMAR.md)：XGrammar 以语法状态决定合法 token，并利用可预计算与依赖上下文的部分分离，减少在线合法性判断开销。
  [完整原文](../../references/S-XGRAMMAR/reader.html) · [全文文本](../../references/S-XGRAMMAR/fulltext.txt)

- [S-FI-SAMPLE：FlashInfer：Sampling](../sources/S-FI-SAMPLE.md)：FlashInfer sampling 接口提供从概率或 logits 的按行采样，以及 top-k、top-p 等相关操作；调用的输入表示和参数决定实际概率处理。
  [算法与机制说明](../sources/S-FI-SAMPLE.md)

- [S-CHAT：Transformers：Chat templates](../sources/S-CHAT.md)：Chat template 将消息序列变为包含角色边界、特殊 token 和生成提示的模型输入；tokenizer 自动添加特殊 token 的方式会影响最终序列。
  [完整原文](../../references/S-CHAT/reader.html) · [全文文本](../../references/S-CHAT/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [批量采样：概率变换、请求随机性与行映射](../algorithms/batched-sampling.md)。
- [投机解码：分布保持、状态提交与收益模型](../algorithms/speculation-budget.md)。
- [输入输出语义：Tokenizer、Chat Template 与流式文本](../system-semantics/tokenization-io.md)。
