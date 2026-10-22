# K-WORK-003：工作负载分布：长度、复用、模态与相关性

系统成本由输入输出的联合分布决定；平均长度会丢失影响缓存、批次和尾部行为的结构。

## 不止两个平均数

prompt 长度影响 prefill、KV 初始容量和传输；输出长度影响请求驻留时长、decode 工作和尾部；两者相关性决定长输入是否还占用很长输出时间。相同均值可以由全为中长请求，或极短与极长混合组成，调度结果很不一样。

还应描述前缀共享长度与访问间隔、adapter 分布、grammar 类型、多样本分支、取消率和流式消费速度。视频帧数、分辨率、音频时长与 codec 设置形成额外几何，不能折算为文本字符数后丢失。

## 复用有时间结构

两条请求共享长前缀，只有在缓存尚未淘汰且被路由到可访问位置时才形成命中。把同一输入重复连续发送会得到近乎完美命中，与真实交错多租户的复用间隔不同。缓存容量、淘汰和亲和路由要在这个时间序列上一起分析。

## 合成负载怎样保持问题结构

可以用少量有含义的分层：短交互、长文档、长输出、共享系统提示、多模态与取消密集。每层保留长度范围、到达方式和质量目标，再按目标比例混合。这样能解释整体变化来自哪层，而不是只看一个总吞吐。

长度边界也有实现意义：页长前后、窗口 rollover、图桶边界、最大上下文和 tokenizer 特殊 token。为机制验证构造边界输入，为容量验证保留真实分布，两类输入回答不同问题。

## 输出长度是实现结果的一部分

不同 EOS、stop string、采样或错误可能让优化版本更早结束，造成表面 throughput 提高。比较时应同时看质量、输入条件与实际输出长度；若任务允许不同长度，采用与业务目标对应的 goodput，而不是假设每输出 token 同等有用。

<!-- infraagents:provenance -->

## 来源与推导

- S-GENAI-PERF：[完整原文](../../references/S-GENAI-PERF/reader.html) · [全文文本](../../references/S-GENAI-PERF/fulltext.txt)
- S-PAGED：[完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)
- S-QWEN-VL：[完整原文](../../references/S-QWEN-VL/reader.html) · [全文文本](../../references/S-QWEN-VL/fulltext.txt)

- [S-GENAI-PERF：NVIDIA GenAI-Perf](../sources/S-GENAI-PERF.md)：GenAI-Perf 的指标围绕请求、响应、首个内容、流式间隔、输出长度和测量窗口定义。客户端 chunk 与模型 token 并非必然一一对应。

- [S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention](../sources/S-PAGED.md)：PagedAttention 把每条序列的逻辑 KV 划分为固定 token 数的块，用映射表访问不连续物理存储。多个序列可以共享相同前缀的物理块，分叉写入需要保持各序列状态独立。

- [S-QWEN-VL：Qwen2.5-VL Technical Report](../sources/S-QWEN-VL.md)：Qwen2.5-VL 报告讨论动态视觉分辨率、窗口视觉编码及时间相关的多模态位置，视觉输入具有独立的网格和时间几何。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 4，2026-09-20。

## 延伸阅读

- [前缀缓存：计算等价类、Radix 与淘汰收益](../algorithms/prefix-cache-identity.md)。
- [编译与动态形状：Guard、分桶和变体成本](../frameworks/compile-dynamic-shapes.md)。
- [视觉语言模型：Patch、模态桥接与联合位置](../models/vision-language.md)。
