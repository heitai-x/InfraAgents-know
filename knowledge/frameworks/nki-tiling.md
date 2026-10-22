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

## 一个矩阵 tile 的可复算资源账

以固定 Trainium2 架构文档的 BF16 矩阵数据流为依据，设逻辑运算为 A[M,K]B[K,N]，单块 M=128、K=128、N=512。stationary 在 SBUF 中按 [K,M] 存放，moving 按 [K,N] 存放，收缩 K 映射到 partition 轴。两个 BF16 输入分别占 32 KiB 和 128 KiB，FP32 的 M×N 累加占 256 KiB PSUM。对输入做双缓冲需要 320 KiB SBUF，尚未包含输出搬回 SBUF、scale、转置或后续算子。

Trainium2 文档给出每 NeuronCore-v3 的 SBUF 为 28 MiB、128 个 partition，每 partition 224 KiB；PSUM 为 2 MiB。上面的输入双缓冲平均每 partition 为 2.5 KiB，单输出累加为每 partition 2 KiB。总字节可容纳只是必要条件，实际分配还受 partition 分布、PSUM bank、指令 tile 限制和同时活跃对象约束。不能根据 2 MiB/256 KiB=8 就断言可以同时推进任意 8 组输出。

若完整 K=1024，则有 8 个收缩块，共执行 2×128×512×1024=134217728 次浮点运算，逻辑输入 A 为 256 KiB、B 为 1 MiB，BF16 输出为 128 KiB。保持同一输出累加直到 8 块完成，可以避免把中间和反复写回 HBM；若因片上冲突而提前回写，额外读写需要重新计账。复用 A 覆盖多个 N 块与复用 B 覆盖多个 M 块，分别要求保留不同片上对象。

## 归约与 attention 的活跃集合

矩阵相乘后做按行 softmax，需要让目标归约轴成为合法的数据布局。选择 M=128、N=512 的分数 tile 时，FP32 分数本身已占 256 KiB；保留分数、指数结果、每行最大值/分母、旧输出累加以及下一块 K/V，会扩大 SBUF 与 PSUM 同时活跃集合。可以按需复用已消费的分数空间，但不能在归约消费者完成前覆盖。

在线 attention 的 Q tile 跨多个 K/V tile 保留，每块依次形成局部分数、更新最大值和分母、重缩放旧累加并加入本块 PV。将最大值或分母转到较低精度是独立的数值选择；不能只因为输入 BF16 就让所有状态 BF16。若长历史被拆给多组计算，还需要合并每组的最大值、分母和未归一化输出，额外 workspace 与最终归约进入端到端成本。

DMA 装入下一块必须写入当前计算不再读取的槽，PSUM 转移完成后才能用于下一输出。扩大流水深度可能隐藏 DMA，也可能耗尽可用片上空间；应从最后使用位置推导缓冲数，再比较串行、双缓冲以及更小 tile 三种合法执行。

## 指令能力与形状分支

固定文档的 double-FP8 模式通过把 K=256 拆为 partition 128 和最外 free 维 2 来装入输入，累加为 FP32；这与直接增加 partition 数不同。文档同时列出不能与该模式组合的 column tiling、sparse matmul 和 transpose mode。不能把不同指令模式的峰值能力叠成一个不存在的内核。文档源码中被 TODO 注释的稀疏 API 段落也不构成已交付接口依据。

小 M 的 decode 可能浪费大 tile 的矩阵并行；prefill 大 M 更容易复用权重，长 K/V 则增加搬运和归约。非整除尾块要给合法 padding/mask，不把填充位置加入 softmax 或真实输出。验证含 M/K/N 边界、极端分数、NaN/Inf 转置行为、每个量化 scale 的归属和多 tile 累加误差。比较性能时说明有效计算、填充计算、SBUF/PSUM 占用、DMA/转置和归约，不以矩阵指令峰值替代完整内核时间。

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
