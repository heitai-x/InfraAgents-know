# K-TOOLS-006：正确性 Oracle：函数参考、状态变形与服务契约

正确性验证需要一组互补的 oracle：数值函数、状态转换、采样分布和外部协议分别有可检查的关系。

## 函数参考与实现独立性

简单稠密 attention、高精度矩阵路径或原模型 forward 可以作为数值参考，前提是输入、mask、位置和权重一致。参考若调用与被测实现相同的页表索引或共享同一个错误 processor，就失去对应维度的独立性。独立性应具体说明到逻辑组件，而不是仅说“另一个测试”。

## 变形关系覆盖状态

同一请求独立执行与放入不同顺序的 batch 应保持规定语义；整段 prefill 与分块 prefill 应一致；冷缓存与合法前缀命中后继续应一致；页的物理重新排列不应改变逻辑结果。这些关系改变执行组织而保持函数，适合在 CPU 小实例上穷举边界。

取消后，新请求使用回收槽应与全新槽执行一致；两个不同 adapter 的相同 prompt 应分别对应各自参考；window rollover 前后应对应同一绝对位置定义。每种关系都来自具体状态模型，可以发现“常见样例都对”时仍隐藏的错误。

## 概率与协议需要其他 Oracle

随机采样用固定小分布检验支持集、归一化和频率，投机检验目标分布关系；单条文本 exact match 不足以证明概率正确。服务协议检查 chunk 顺序、终止事件、stop string、usage、错误和背压，不能只看返回了 HTTP 200。

质量指标还包括 ASR 词错误率、视觉任务准确性、图像质量或语音一致性。数值近似通过小张量容差与任务质量合格各提供不同支持。一个完整工程验收把这些 oracle 组合起来，而不是让一个最终布尔值遮住具体覆盖。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

- [S-TORCH-REPRO：PyTorch — Reproducibility](../sources/S-TORCH-REPRO.md)：随机种子与确定性配置可以控制特定执行路径中的随机性，但不保证跨版本、平台和全部算法路径逐位一致。
  [完整原文](../../references/S-TORCH-REPRO/reader.html) · [全文文本](../../references/S-TORCH-REPRO/fulltext.txt)

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：Transformers 的缓存说明区分已存历史 K/V、新输入、attention mask 和 cache position。增量前向需要让新 token 的位置与已有缓存长度一致。
  [完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

- [S-ASGI：ASGI — HTTP and WebSocket Message Format](../sources/S-ASGI.md)：ASGI HTTP/WebSocket 规范定义接收、发送、响应体和断开相关事件；断开通知与发送失败可能存在时间上的竞争。
  [完整原文](../../references/S-ASGI/reader.html) · [全文文本](../../references/S-ASGI/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [数值参考：误差尺度、条件数与首次分歧](../frameworks/numerical-reference.md)。
- [位置、Mask 与缓存：同一可见性关系的三种表示](../system-semantics/position-mask-cache.md)。
- [服务协议：接收、流式背压与终止语义](../system-semantics/service-protocol.md)。
- [工程认识如何成为可迁移的专家知识](../system-semantics/evidence-and-independent-checks.md)。
