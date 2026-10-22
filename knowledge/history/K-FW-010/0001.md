# K-FW-010：NKI：Partition 轴、片上生存期与 Attention 数据流

NKI 的 tile 是带硬件轴语义的片上对象；布局与生存期共同决定 DMA、矩阵和归约引擎的供给。

## 布局不是抽象转置

SBUF 张量区分 partition 轴和 free 轴。Tensor Engine 对收缩维和 tile 有具体映射要求，HBM 连续维转为 partition 维可能需要物理转置。选择 HBM 布局时，应同时考虑 DMA 连续访问和片上矩阵布局，而不是分别优化两端。

Trainium2 的 DMA transpose 可以承担某些 HBM/SBUF 变换；其他路径可能通过 Tensor Engine 和 PSUM 再搬回 SBUF。它们占用不同引擎，最佳选择取决于当前哪个资源拥塞。仅比较转置字节数看不出引擎竞争。

## Tile 的生存期推导

Q tile 在处理多个 K/V tile 时持续使用；K/V 可流水装入；在线 softmax 的最大值、分母和输出累加需跨 tile 保留。若片上空间装不下这些活跃对象，就要缩小 tile、重算或溢出到 HBM。最大单张量能放下，不等于整个活跃集合能放下。

双缓冲可让 DMA 读取下一块与当前计算重叠，但占用两份输入空间。PSUM 累加区要在覆盖前完成向后续归约/输出路径转移。片上软件调度同样需要生产消费依赖，不能只看引擎支持并行。

## 对服务内核的意义

prefill 大 query tile 与 decode 小 query 的复用结构不同。长 KV decode 可能需要分区并合并统计，提高并行度却增加暂存和归约。动态请求长度应通过预先允许的形状与有效范围进入内核。模型语义和数据流正确后，再按实际架构资源选择 tile 与流水深度，能避免把某代推荐数字当成算法本身。

<!-- infraagents:provenance -->

## 来源与推导

- [S-NKI：Trainium2 Architecture Guide for NKI](../sources/S-NKI.md)：已读 Trainium2 NKI 资料区分 HBM、SBUF、PSUM、DMA 与计算引擎；片上张量的 partition/free 轴和转置路径具有具体硬件意义。
  [完整原文](../../references/S-NKI/reader.html) · [全文文本](../../references/S-NKI/fulltext.txt)

- [S-FLASH：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](../sources/S-FLASH.md)：FlashAttention 通过片上分块和在线 softmax，避免将完整 attention 分数与概率矩阵写入设备主存；目标函数仍是精确 attention 的数学形式。
  [完整原文](../../references/S-FLASH/reader.html) · [全文文本](../../references/S-FLASH/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [Trainium：HBM、SBUF、PSUM 与编译执行](../hardware/trainium-dataflow.md)。
- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [容量、算术强度与服务 Roofline](../hardware/memory-and-roofline.md)。
