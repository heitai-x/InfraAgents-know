# K-SYS-007：输入输出语义：Tokenizer、Chat Template 与流式文本

模型处理的是规范化后的 token/特征序列，服务协议必须精确保存从用户输入到模型输入再到输出文本的映射。

## 文本序列化就是模型输入的一部分

chat messages 通过模板加入角色边界、回合终止和生成提示。BOS/EOS 是否由模板插入、是否还由 tokenizer 自动插入，会改变最终 token 序列。把 messages 直接拼接成普通字符串不等价于训练时的对话格式；工具调用和多模态 placeholder 也可能有专门语法。

tokenizer 版本、normalization、特殊 token 配置和 truncation 策略共同定义输入。Unicode 看起来相同的字符串可以有不同字节或规范化结果。前缀缓存应基于模型实际消费的规范表示，而不是仅以 UI 字符串或字符长度判断。

## Padding 与长度

不同模型/框架对左 padding、右 padding、position 和末 token logits 抽取有不同约定。attention mask 说明真实输入，cache_position 说明逻辑推进。长输入截断是任务语义选择：丢开头可能丢系统指令，丢尾部可能丢用户问题；服务不能把截断当成纯性能优化而不记录。

## 增量 detokenization

token 边界不等于字符边界，一个 token 也可能包含多个字符或不完整字节片段。逐 token 独立 decode 再串接未必等于整段 decode。流式解码器需要保留未稳定后缀，直到能输出有效文本片段。

stop string 可能跨多个 token 或多个网络 chunk，检测要保留足够后缀。若协议规定不把 stop string 发给用户，就必须延迟可能构成其前缀的字符，而不能先发出再撤回。EOS、最大 token 数与 stop string 是不同终止原因，对 usage 计数和 finish reason 有不同影响。

## 多模态沿用同一原则

图片/音频预处理版本、采样率、裁剪、patch 网格和文本占位符顺序一起定义模型输入。输出音频也需要样本位置、编码格式和 chunk 顺序。输入输出正确性可以在 CPU 上固定这些中间表示，成为后续内核优化共享的语义参考。

## 制品一致性跨过文本与模型边界

输入档案绑定 tokenizer/processor 的固定版本、模板、特殊 token 表、词表与模型 embedding/output head 的维度。两套 tokenizer 都能产生合法整数 ID，不代表它们对同一文本产生相同模型输入。输出解码还要使用与输入和模型一致的语义，不能在版本切换中混用旧 decoder 与新词表。

工具回合需要保留角色、工具调用身份、参数序列化和结果顺序。应用层把结果格式化为文本的方式可能改变 token 序列，即使业务上看起来表达同一个 JSON 对象。复用判断应从最终模型输入出发；会话 ID 只提供查找入口，不能独立证明 KV 等价。

## 增量输入与 tokenization 的边界

对新增文本片段单独分词再拼接，未必等于对完整文本分词：边界附近可能发生合并或 normalization 差异。只有实际 tokenizer 的增量契约或稳定边界得到证明时才能复用已分词前缀。多轮服务可以缓存经过完整序列化与分词的共同前缀，但模板新增的回合标记也要包含在对应身份中。

截断和输入限额应在明确的计量单位执行。字节、Unicode 字符、token、图像 patch 和音频采样点不能混用；模型最大上下文与服务可承受的资源预算也不一定相同。若拒绝、截断或分段，响应协议和测量记录要保留实际发生的处理。

## 优化 CPU 路径时保持的关系

分词批处理、缓存规范化结果、原生实现和进程池都可能减少输入开销，但必须保持 token IDs、offset/模态对齐和特殊 token 行为。缓存键包含会影响结果的配置；缓存容量和保存敏感输入的边界由服务隔离要求约束。并发分词不能让可变模板或 processor 缓冲在请求间串用。

对照输入包含空消息、非 ASCII、组合字符、特殊 token 样式文本、跨片段合并、多个工具结果及不同 padding。输出验证包含跨 token 的 UTF-8 片段、跨 chunk stop string、取消前最后一段内容与 usage/finish reason。测试比较规范 token/特征和协议事件，避免只比较屏幕上看起来相同的文本。

若 CPU 输入阶段已成为瓶颈，按 [主机运行时](host-runtime.md) 分配有界工作池；若模型计算支配总时延，复杂输入缓存可能仅增加容量和失效逻辑。测量时把输入排队、实际 CPU 服务与模型等待分开，才能判断优化是否改变用户时间。

<!-- infraagents:provenance -->

## 来源与推导

- [S-CHAT：Transformers：Chat templates](../sources/S-CHAT.md)：Chat template 将消息序列变为包含角色边界、特殊 token 和生成提示的模型输入；tokenizer 自动添加特殊 token 的方式会影响最终序列。
  [完整原文](../../references/S-CHAT/reader.html) · [全文文本](../../references/S-CHAT/fulltext.txt)

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：Transformers 的缓存说明区分已存历史 K/V、新输入、attention mask 和 cache position。增量前向需要让新 token 的位置与已有缓存长度一致。
  [完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

- [S-ASGI：ASGI — HTTP and WebSocket Message Format](../sources/S-ASGI.md)：ASGI HTTP/WebSocket 规范定义接收、发送、响应体和断开相关事件；断开通知与发送失败可能存在时间上的竞争。
  [完整原文](../../references/S-ASGI/reader.html) · [全文文本](../../references/S-ASGI/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [前缀缓存：计算等价类、Radix 与淘汰收益](../algorithms/prefix-cache-identity.md)。
- [视觉语言模型：Patch、模态桥接与联合位置](../models/vision-language.md)。
- [批量采样：概率变换、请求随机性与行映射](../algorithms/batched-sampling.md)。
