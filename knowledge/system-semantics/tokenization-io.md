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

<!-- infraagents:provenance -->

## 来源与推导

- S-CHAT：[完整原文](../../references/S-CHAT/reader.html) · [全文文本](../../references/S-CHAT/fulltext.txt)
- S-HF-CACHE：[完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)
- S-ASGI：[完整原文](../../references/S-ASGI/reader.html) · [全文文本](../../references/S-ASGI/fulltext.txt)

- [S-CHAT：Transformers：Chat templates](../sources/S-CHAT.md)：Chat template 将消息序列变为包含角色边界、特殊 token 和生成提示的模型输入；tokenizer 自动添加特殊 token 的方式会影响最终序列。

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：Transformers 的缓存说明区分已存历史 K/V、新输入、attention mask 和 cache position。增量前向需要让新 token 的位置与已有缓存长度一致。

- [S-ASGI：ASGI — HTTP and WebSocket Message Format](../sources/S-ASGI.md)：ASGI HTTP/WebSocket 规范定义接收、发送、响应体和断开相关事件；断开通知与发送失败可能存在时间上的竞争。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 3，2026-09-20。

## 延伸阅读

- [前缀缓存：计算等价类、Radix 与淘汰收益](../algorithms/prefix-cache-identity.md)。
- [视觉语言模型：Patch、模态桥接与联合位置](../models/vision-language.md)。
- [批量采样：概率变换、请求随机性与行映射](../algorithms/batched-sampling.md)。
