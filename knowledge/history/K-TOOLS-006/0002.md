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

## 组合覆盖来自相交的状态边界

各机制单独正确，不能推出组合正确。先列出它们共同触及的对象：分页与前缀共享共同管理尾块；投机与 grammar 共同管理未提交分支；异步执行与取消共同管理在途缓冲；量化与迁移共同管理尺度与布局。测试应覆盖交界上的状态转换，而不是机械地对全部参数做笛卡尔积。

例如“前缀共享＋异步取消”的判别场景：A/B 共享完整前缀，各自追加不同 token；取消 A 后立即把它可回收的私有槽分给 C；延迟 A 的完成通知；继续 B/C。参考分别从各自逻辑 token 全量重建，检查 B 未被污染、C 不接收 A 的迟到输出、共享页未提前回收。这个时序可以区分身份错误、引用错误和完成事件错误。

## 性质测试与并发调度

把物理页排列、batch 排序、分块边界和无关请求插入作为变形维度，保持模型函数与有效输入不变。生成器需要包含跨页、非整除尾部、零长度边界和最大合法范围，并在失败时保留最小可复现输入。变形关系本身也要有前提：改变采样种子不要求文本一致，改变模型版本不再保证缓存等价。

并发测试要控制完成顺序，包括旧工作晚于新工作返回、多个 stream 的不同事件次序、传输部分完成、取消与提交竞争。随机压力有发现价值，确定性的受控时序更便于解释；两者都不能以一次未出现错误证明所有竞态已消除。

## 长时间运行检查守恒关系

资源池满足总量=空闲+活跃占用+在途保护+保留但可解释的占用，各项不能重复计数。测试经过多轮分配、共享、驱逐、取消和失败恢复后，未完成工作清空时应回到预先声明的稳态基线。缓存刻意保留导致的驻留与不可达对象泄漏不同；观察 RSS 不下降时，需要先区分 allocator 保留和真实失去所有者。

同时检查请求终态唯一性、输出序号连续性、最多一次的外部提交，以及失败是否留下可复用却未初始化的块。故障注入应落在定义过的协议边界；若业务允许重试，参考需要包含重试身份与结果提交规则，不能用最终“有输出”掩盖重复响应。

## 独立保留验证检查什么

开发样例用于定位和调参，最终保留样例覆盖未参与选择的形状、长度、输入分层和状态时序。对低位路径，先用固定量化参数的独立公式排除实现错误，再检查原模型质量；对概率路径，先用小分布验证采样关系，再做统计检验。独立性说明到具体组件：参考可以共用权重文件，但不能在要验证页表索引时共用被测页表逻辑。

知识章节提出的是 oracle 和区分实验。只有实际任务执行这些检查，才能记录相应通过或失败；章节写得完整、结构检查通过，都不能被标记成目标引擎的正确性证据。

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



### 本次扩写补充来源

- [S-INT8：Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](../sources/S-INT8.md)：§2 给出仿射量化、整数点积、零点补偿、bias 和输出重新缩放；附录 A–B 讨论组合算子、舍入及 NEON 中间累加条件。
  [完整原文](../../references/S-INT8/reader.html) · [全文文本](../../references/S-INT8/fulltext.txt)

## 延伸阅读

- [数值参考：误差尺度、条件数与首次分歧](../frameworks/numerical-reference.md)。
- [位置、Mask 与缓存：同一可见性关系的三种表示](../system-semantics/position-mask-cache.md)。
- [服务协议：接收、流式背压与终止语义](../system-semantics/service-protocol.md)。
- [工程认识如何成为可迁移的专家知识](../system-semantics/evidence-and-independent-checks.md)。
