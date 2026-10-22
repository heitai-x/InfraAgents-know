# K-HW-001：容量、算术强度与服务 Roofline

容量决定系统能容纳哪些状态，带宽与计算能力决定搬运和处理这些状态的下界，关键路径决定哪些下界实际显现。

## 三本不同的账

容量账是权重＋持久状态＋激活峰值＋workspace＋图池＋通信缓冲＋分配碎片。流量账是一次服务实际跨某层内存边界读写的字节。计算账是该次服务的算术工作。一个对象驻留一次，可以在多个 token 步中被反复读取；分配字节不能直接当作总访存流量。

对工作量 F FLOPs、边界流量 D 字节、有效算力 P 和带宽 B，理想时间下界为 max(F/P,D/B)，算术强度 I＝F/D，屋顶吞吐为 min(P,BI)。这假定计算与搬运可充分重叠，实际还可能受 launch、依赖、占用率和尾部不均衡限制。使用可持续有效带宽比峰值广告参数更接近服务路径。

## 权重与 KV 谁主导

小 batch dense decode 中，权重可能每步读取较大部分，多个请求共用权重可摊薄每输出 token 成本。KV 则主要按各请求历史长度读取。batch 增大降低权重/token 项，却增加总 KV 和激活，使瓶颈可能从权重转向 attention。长上下文改变这两个项的交点。

假设一轮需要读 8 GiB 权重和 2 GiB KV，理想有效带宽 1 TiB/s，则这部分流量下界约 9.77 ms；并发 16 个 decode 请求时，单位输出 token 约摊到 0.61 ms，但每个请求仍等待整轮。吞吐和单请求延迟因此不能用同一个除法解释。

## 内存层级改变 D

寄存器、片上共享存储、L2 和设备主存各有不同带宽和容量。某层复用减少其上游流量，可能增加寄存器压力并降低 occupancy。attention tiling 减少 HBM 中间量，量化减少表示字节，前缀共享减少重复状态，三者作用于不同项，可组合却不能把加速比直接相乘。

容量瓶颈也会转化为时间：KV 不足导致抢占与重算，主存溢出导致换页，过多图桶挤出可服务请求。选择算法前应先还原当前状态在哪一层驻留、在哪一边界重复传输。

## 从局部模型回到完整服务

Amdahl 关系说明，若被优化部分占总关键路径比例 f，局部加速 s，则理想总加速为 1/((1−f)＋f/s)。流水重叠时不能从各 kernel 累计时长直接得到 f，应从依赖图中真正延迟输出的区间估计。Roofline 用来形成可解释的候选方向，再由时间线决定当前哪个资源确实卡住。

<!-- infraagents:provenance -->

## 来源与推导

- [S-CUDA-BEST：CUDA C++ Best Practices Guide](../sources/S-CUDA-BEST.md)：CUDA 性能资料区分有效带宽、访存组织、数据传输和执行资源。合并访问、复用和减少不必要搬运会改变实际流量。
  [完整原文](../../references/S-CUDA-BEST/reader.html) · [全文文本](../../references/S-CUDA-BEST/fulltext.txt)

- [S-PAGED：Efficient Memory Management for Large Language Model Serving with PagedAttention](../sources/S-PAGED.md)：PagedAttention 把每条序列的逻辑 KV 划分为固定 token 数的块，用映射表访问不连续物理存储。多个序列可以共享相同前缀的物理块，分叉写入需要保持各序列状态独立。
  [完整原文](../../references/S-PAGED/reader.html) · [全文文本](../../references/S-PAGED/fulltext.txt)

- [S-FLASH：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](../sources/S-FLASH.md)：FlashAttention 通过片上分块和在线 softmax，避免将完整 attention 分数与概率矩阵写入设备主存；目标函数仍是精确 attention 的数学形式。
  [完整原文](../../references/S-FLASH/reader.html) · [全文文本](../../references/S-FLASH/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [量化：误差传播、分组尺度与执行格式](../algorithms/quantization-contract.md)。
- [从模型计算图推导 Prefill、Decode 与持久状态](../models/prefill-decode-state.md)。
- [容量设计：驻留、服务率与阶段配比](../workloads/capacity-planning.md)。
