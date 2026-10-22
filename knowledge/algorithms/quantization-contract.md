# K-ALG-005：量化：误差传播、分组尺度与执行格式

量化把数值表示、数据布局和执行内核一起改变；收益由减少的字节与新增的变换成本共同决定。

## 仿射表示与误差的来源

均匀量化可写为 q＝clip(round(x/s)＋z)，反量化值 x̂＝s(q−z)。不发生饱和且采用最近舍入时，单元素绝对误差至多约 s/2。增大范围会增大步长，缩小范围会使离群值饱和；分组尺度在两者间提供局部折衷。scale、zero point、分组轴和 rounding 是数值含义的一部分，不能只登记“INT4”。

线性层 y＝Wx 中，权重误差 ΔW 引入 Δy＝ΔWx；若激活也量化，还增加 WΔx 和 ΔWΔx。因此优化权重本身的均方误差不一定最小化实际输出误差。GPTQ 利用校准输入的结构近似层输出误差；校准分布变化会改变最应保留的方向。attention 中 Q/K 误差先改变 logits，再经 softmax 非线性放大或抑制，误差不会按 bit 数线性传到输出。

## 三种对象对应不同系统收益

| 对象 | 直接减少 | 新增代价 | 可能仍然占主导的部分 |
|---|---|---|---|
| 权重 W | 驻留与读取字节 | unpack、scale、布局转换 | 长上下文 KV 读取 |
| 激活 A | GEMM 输入/输出流量 | 动态尺度计算、异常值处理 | kernel launch、通信同步 |
| KV | 历史驻留与 decode 流量 | 每步写入量化、读取解量化 | 模型权重或采样开销 |

W4A16 的数学操作往往仍以高精度激活参与计算。若内核先把整个权重展开到高精度临时区，再做 GEMM，容量和带宽收益可能被副本削弱。有效实现通常让解包和乘法在适当粒度融合；具体是否使用低位矩阵指令由硬件和 kernel 决定。

## 把 scales 也算进容量

设每组 g 个权重，每权重 b bits，每组一个 s 字节 scale，不计 zero point 时平均每权重 b/8＋s/g 字节。例如 b＝4、g＝128、scale＝2 字节，得到 0.515625 字节，比只算 0.5 多约 3.1%。小分组能拟合局部分布，却提高元数据比例和加载成本。混合精度层、对齐 padding 和打包排列还会改变最终占用。

KV 量化的分组轴需匹配读写：按 token 动态尺度容易追加，按通道跨 token 分组可能要重新定标旧值或等待凑满块。前缀共享时量化格式和 scales 必须随内容身份固定，不能用另一个请求的校准状态解读同一物理页。

## 从任务质量导出选择

近似方案的合理选择是满足质量约束下减少服务成本。短确定性输入仅覆盖局部误差，长上下文、罕见 token、路由边界、多模态特征可能暴露不同敏感性。可以分别观察层输出、固定前缀的 logits 和任务级质量，找出误差从哪里开始增长。将某些敏感层保留高精度会牺牲部分容量，但有机会保住系统可用区间。

算例中若量化使权重读取减半，却使原本占 70% 的 attention 历史读取不变，端到端理想加速也只有 1/(0.7＋0.3/2)≈1.18。这个推导说明应先估计被量化对象在关键路径中的占比，再选择格式。

<!-- infraagents:provenance -->

## 来源与推导

- [S-GPTQ：GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](../sources/S-GPTQ.md)：GPTQ 是训练后权重量化方法，使用校准输入所反映的层输入结构，近似考虑量化误差对层输出的影响。它关注的目标不只是孤立权重元素的舍入误差。
  [完整原文](../../references/S-GPTQ/reader.html) · [全文文本](../../references/S-GPTQ/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

- [S-CUDA-CAPABILITY：CUDA Programming Guide — Compute Capabilities](../sources/S-CUDA-CAPABILITY.md)：CUDA 的硬件功能与数值指令能力按 compute capability 和设备代际区分；一种 dtype 可被表示并不意味着所有算子形态都有高效实现。
  [完整原文](../../references/S-CUDA-CAPABILITY/reader.html) · [全文文本](../../references/S-CUDA-CAPABILITY/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [数值格式、打包布局与矩阵指令](../hardware/precision-and-kernel-support.md)。
- [数值参考：误差尺度、条件数与首次分歧](../frameworks/numerical-reference.md)。
- [MHA、GQA 与 MQA：头映射和带宽复用](../models/gqa-kv-heads.md)。
