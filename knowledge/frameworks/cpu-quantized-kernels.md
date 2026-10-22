# K-FW-012：CPU 低位执行：解包、补偿与累加路径

低位权重首先改变存储表示。是否同时减少计算时间，取决于实际执行的是低位乘加、解包后的整数乘加，还是反量化后的浮点乘加。

## 三条合法执行路径

权重量化、浮点激活的路径可以逐 tile 解包 W、恢复尺度，再执行浮点点积；它省去主存读取的权重字节，但增加移位、符号扩展、scale 读取和转换。若每次先恢复一份完整 FP32 权重再算，稳态读取可能回到 FP32，压缩只保留了文件大小收益。

权重与激活都量化的路径把主体变为整数点积，再做补偿与重新缩放。激活 scale 可以预先固定或按 token/块动态求取；动态量化适应输入分布，却需要额外归约、读写和同步。第三条是保留敏感算子或通道的较高精度，代价是更多路径和合并成本。这些选择必须先满足 [量化契约](../algorithms/quantization-contract.md) 的质量条件。

## 从仿射量化推导点积

对同一量化组，xᵢ=sₓ(qₓᵢ−zₓ)、wᵢ=s𝓌(q𝓌ᵢ−z𝓌)，则点积为

sₓs𝓌[Σqₓᵢq𝓌ᵢ − z𝓌Σqₓᵢ − zₓΣq𝓌ᵢ + Kzₓz𝓌]。

固定权重的 Σq𝓌 可以随打包预计算；激活的 Σqₓ 随输入变化。对称权重量化令 z𝓌=0，消掉两项但不自动消除 zₓ 补偿。[整数推理论文](../sources/S-INT8.md) §2.1–2.4 展开了零点补偿、accumulator、bias 与输出缩放的关系。按 K 维分组时，不同组的 sₓs𝓌 可能不同，不能先把所有整数和相加再乘一个全局 scale。

设定 qₓ=(5,7)、zₓ=5、sₓ=0.5；q𝓌=(2,0)、z𝓌=1、s𝓌=2。真实向量为 (0,1) 与 (2,−2)，点积为 −2。展开后整数项为 10−12−10+10=−2，尺度乘积为 1。这个小例子同时检查零点、补偿符号和真实零的表示；padding 必须填量化零点而不总是字节零。

## 低位布局与算术宽度

4 bit 存储需要明确一个字节内的顺序、有符号编码、分组轴、尾部填充和输出通道打包。加载器、预打包器与内核必须使用同一契约；错误的 nibble 顺序仍会给出形状正确的输出。scale/zero point 的排列要跟随相同的置换。

累加宽度由最坏点积界决定。若 |xᵢ|≤A、|wᵢ|≤B，则 |Σxᵢwᵢ|≤KAB；bias 和补偿还会扩大中间范围。即便最终 int32 足够，某些向量路径可能先做更窄的中间求和。论文附录 B 的 NEON 优化明确依赖排除特定极值和控制窄 accumulator 累加次数；不能把它泛化成任意整数内核都不会饱和。

bias 若在整数 accumulator 上相加，其 scale 必须与 accumulator 对应。输出转换包含倍率、舍入、饱和与可选激活；负数半整数处的舍入规则也会影响系统性偏差。残差两支 scale 不同时，直接相加整数码没有正确的数值意义。固定点、浮点 epilogue 和保持较高精度输出都是可选路径，需要分别比较误差和成本。

## 字节减少的条件

设定一个 4096×4096 权重，4 bit 主体为 8 MiB。每 128 个权重一个组，每组 scale 与 zero point 合计假设为 4 字节，则元数据为 0.5 MiB，有效存储约 4.25 bit/weight，尚不含对齐。解包成完整 FP32 副本另需 64 MiB；tile 解包则只保留同时在用的片段，但可能重复解包。

性能模型应至少比较“压缩权重读取+解包/转换+点积+补偿”和原浮点路径，允许资源重叠，也计入 pack 的摊销。小 batch decode 的字节下降可能重要；大 prefill 能在多个 token 间重用权重，此时低位解包和不适配的 kernel 可能成为代价。线程扩大后若解包共享区发生写竞争或 NUMA 远端访问，容量优势不会自动变成吞吐优势。

## 区分实现错误与量化损失

先固定同一组量化码，用高精度公式参考检查解包、零点、分组累加和 epilogue；再将该量化模型与原浮点模型比较。第一层失败是执行错误，第二层差异才用于讨论表示损失。边界包括最小/最大码、负半值舍入、K 非整除、跨组 scale、全零输入、溢出界和多线程划分。

同时记录预打包时间、常驻副本、实测权重流量、解包计算占比和端到端延迟。若矩阵 shape、线程数或并发改变后收益反转，应重新选择路径，不能因存储标为 INT4 就强制所有算子使用同一个低位实现。

<!-- infraagents:provenance -->

## 来源与推导

- [S-INT8：Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](../sources/S-INT8.md)：§2 给出仿射量化、整数点积、零点补偿、bias 和输出重新缩放；附录 A–B 讨论组合算子、舍入及 NEON 中间累加条件。
  [完整原文](../../references/S-INT8/reader.html) · [全文文本](../../references/S-INT8/fulltext.txt)

- [S-GPTQ：GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](../sources/S-GPTQ.md)：GPTQ 是训练后权重量化方法，使用校准输入所反映的层输入结构，近似考虑量化误差对层输出的影响。它关注的目标不只是孤立权重元素的舍入误差。
  [完整原文](../../references/S-GPTQ/reader.html) · [全文文本](../../references/S-GPTQ/fulltext.txt)

- [S-CPU-TUNING：PyTorch Performance Tuning Guide](../sources/S-CPU-TUNING.md)：PyTorch CPU 调优资料涉及线程组织、NUMA、向量化与执行配置；线程数和内存归属影响真实可用资源。
  [完整原文](../../references/S-CPU-TUNING/reader.html) · [全文文本](../../references/S-CPU-TUNING/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

正文中的设定算例、资源账和机制组合是基于上述资料的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；本次编辑日期不改变来源时间。

## 延伸阅读

- [量化：误差传播、分组尺度与执行格式](../algorithms/quantization-contract.md)。
- [CPU GEMM/GEMV：寄存器、缓存与线程划分](../hardware/cpu-linear-kernels.md)。
- [数值格式、打包布局与矩阵指令](../hardware/precision-and-kernel-support.md)。
- [数值参考：误差尺度、条件数与首次分歧](../frameworks/numerical-reference.md)。
- [权重加载：从制品身份到可执行布局](../frameworks/checkpoint-loading.md)。
