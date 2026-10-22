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

## 校准集决定优化面对的误差分布

校准需要固定 tokenizer、输入模板、有效长度、模态、adapter 和前处理，采集目标层实际接收的激活。短通用文本只覆盖一类分布；长上下文、稀疏专家、检索材料、视觉特征和真实 decode 位置可能激活不同通道。校准集应覆盖目标负载的重要部分，并与选超参数的验证集及最终质量测试集分开。不能用测试标签选择 clipping 或敏感层后再把同一结果称为独立验证。

静态尺度来自离线统计，运行时开销较小，但分布偏移会造成饱和或低有效精度；动态尺度根据当前输入求取，增加 reduction、同步和元数据，同时也改变批量之间的尺度关系。统计 max、分位数或重建误差会给出不同目标，应保留统计粒度和裁剪规则。全零通道、极少访问的专家、空批及异常数值需要定义尺度和错误处理，不能靠除零后自动得到一个任意值。

逐层校准还要说明输入来自高精度模型还是已部分量化模型。后者反映上游误差传播，前者便于隔离单层误差；二者可能产生不同制品。GPTQ 固定版本的实现说明按 block 量化并将量化后输出传给后续 block。本章把这作为一条具体校准路线，不要求所有方法机械复制同样流程。

## GPTQ、AWQ 与 SmoothQuant 各自改变什么

统一采用 y=Wx 的列向量约定。GPTQ 的层目标是减小 ‖(W−Ŵ)X‖²；输入相关结构可由 XXᵀ 表达。它逐步量化权重并调整未量化部分补偿输出误差，通过共同列顺序、分块更新和 Cholesky 形式降低成本、改善数值稳定性。阻尼、顺序、组大小、舍入网格和量化前后激活都影响结果。输入维度 K=4,096 时，一个 FP32 K×K 矩阵就需 64 MiB，逆/分解、副本和 block 工作区还需另外计入；“只量化一层”不等于峰值只有该层权重。

AWQ 利用激活统计寻找更值得保护的输入通道。令正对角矩阵 D=diag(s)，则 Wx=(WD)(D⁻¹x) 在量化前代数等价。量化 WD 后再与 D⁻¹x 相乘，改变的是量化网格相对重要通道的误差。固定 v1 在平均激活幅度与归一化权重幅度构成的尺度空间内搜索，并可搜索 clipping；它最终仍可把全部目标权重量化，不需要永久保留“最重要 1%”为 FP16。其 W3/W4A16 路线与执行内核应分别登记。

SmoothQuant 也使用等价通道缩放，但目标是让权重和激活共同适合 W8A8。按本章 W 的布局，可用 s_j=max|X_j|^α/max|W_:,j|^(1−α) 构造候选尺度，将激活离群程度转移一部分到权重。α 太大可能让权重难以量化，太小则保留激活困难；固定论文中 O1/O2/O3 的激活量化分别采用 per-token 动态、per-tensor 动态、per-tensor 静态，不能只写“使用 SmoothQuant”便认为执行路径相同。

这三类方法均依赖目标输入和实际数值路径。AWQ 论文中的分布鲁棒性观察不是不会过拟合的定理；SmoothQuant 的实数等价变换也不表示后续整数舍入无损。缩放并入上游 norm 或线性层时，需要补偿所有消费者；同一激活同时进入 Q/K/V 或残差分支时，只修改一条边会改变原函数。遇到非线性或共享参数，应先证明允许的变换位置。

## 从量化制品到真实低精度计算

W4A16 可以在 tile 内解包为半精度并进入矩阵单元，也可以走整数/查表或其他专用路径；它没有自动保证原生 INT4×FP16 指令。AWQ v1 的 kernel 说明采用在线解量化再进行矩阵计算；GPTQ 的固定论文也明确当时的硬件混合操作数限制。量化算法名称与 kernel 家族不是一一对应关系。

设一个 4,096×4,096 权重按输入维每 128 个分组，码为 4 bits，每组另有一个 2-byte scale 和 2-byte zero point。权重码 8 MiB，131,072 组元数据共 0.5 MiB，总计 8.5 MiB，未含 padding、索引与 workspace；FP16 权重为 32 MiB。加载时若同时保留原副本与打包制品，至少已占 40.5 MiB。若每轮先完整展开 32 MiB 权重再计算，额外写读会改变低位表示带来的收益。

量化参数沿输出行/输入 token 的尺度可以在矩阵乘后的 epilogue 中组合；若尺度随归约 K 组变化，则通常要分别缩放各组部分和再累加，不能只给最终结果乘一个公共尺度。zero point 还会引入修正项。尾部填充要表示实数零：仿射码应为对应 z，不能默认整数码 0。非整除组的 scale 所属范围和未使用 nibble 也必须明确。

FP8 需分别记录 E4M3/E5M2 等格式、缩放粒度、amax 来源、饱和/非有限值规则、乘法和累加精度。硬件能存 FP8，不证明全部计算用了 FP8 矩阵指令；FlashInfer 固定论文附录 F 的一条具体路径是 FP8 KV 存储、FP16 query/output，并做解量化。它主要减少缓存字节，不应被叙述为整条 attention 的所有算术翻倍。

## 质量回归与选择改变的条件

验证应依次区分打包解释错误、kernel 数值误差和算法近似：先比较同一量化码及参数的独立解码；再比较量化权重参考与实际 kernel；最后比较完整量化模型与高精度模型。第一层可用含极值、零点、负数、常数组、尾组和饱和的手算向量；后两层覆盖实际 M/N/K、prefill/decode、长上下文、稀有 token、路由与多模态分布。

敏感层恢复高精度时，同时重新计算驻留、传输、kernel 切换和工作区。质量评估既看层输出/logits，也看任务指标和允许下降，避免只用短贪心回答是否相同作判定。KV 量化与权重量化同时启用还需组合回归，其误差不会自动独立相加。

低 batch 权重带宽主导时，W4A16 的节省可能更有价值；大 batch 计算主导时，具有合适整数矩阵路径的 W8A8 可能更合适；长上下文被 KV 流量主导时，应比较 KV 优化而非继续降低权重位宽。质量阈值收紧、尺度统计漂移或目标设备缺少高效 dispatch 时，保留高精度或采用更粗粒度的成熟路径可能反而获得更低的合格请求成本。

<!-- infraagents:provenance -->

## 来源与推导

- [S-GPTQ：GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](../sources/S-GPTQ.md)：GPTQ 是训练后权重量化方法，使用校准输入所反映的层输入结构，近似考虑量化误差对层输出的影响。它关注的目标不只是孤立权重元素的舍入误差。
  [完整原文](../../references/S-GPTQ/reader.html) · [全文文本](../../references/S-GPTQ/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

- [S-CUDA-CAPABILITY：CUDA Programming Guide — Compute Capabilities](../sources/S-CUDA-CAPABILITY.md)：CUDA 的硬件功能与数值指令能力按 compute capability 和设备代际区分；一种 dtype 可被表示并不意味着所有算子形态都有高效实现。
  [完整原文](../../references/S-CUDA-CAPABILITY/reader.html) · [全文文本](../../references/S-CUDA-CAPABILITY/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-AWQ：AWQ](../sources/S-AWQ.md)：固定 v1 以激活感知通道缩放和 clipping 降低低位权重误差，说明 grouped quantization、在线解量化及特定 GPU kernel 路径。
  [完整原文](../../references/S-AWQ/reader.html) · [全文文本](../../references/S-AWQ/fulltext.txt)

- [S-SMOOTHQUANT：SmoothQuant](../sources/S-SMOOTHQUANT.md)：固定 v1 用通道等价缩放平衡权重/激活量化困难，区分三种激活量化设置，并通过具体 INT8 GEMM/BMM 路径评估。
  [完整原文](../../references/S-SMOOTHQUANT/reader.html) · [全文文本](../../references/S-SMOOTHQUANT/fulltext.txt)

- [S-FLASHINFERPAPER：FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](../sources/S-FLASHINFERPAPER.md)：固定 v1 描述 BSR/ragged/可组合 KV 表示、attention 状态合并、JIT 变体、动态负载计划与 CUDA Graph workspace，以及 GQA 和 FP8 KV 路径。
  [完整原文](../../references/S-FLASHINFERPAPER/reader.html) · [全文文本](../../references/S-FLASHINFERPAPER/fulltext.txt)

## 延伸阅读

- [数值格式、打包布局与矩阵指令](../hardware/precision-and-kernel-support.md)。
- [数值参考：误差尺度、条件数与首次分歧](../frameworks/numerical-reference.md)。
- [MHA、GQA 与 MQA：头映射和带宽复用](../models/gqa-kv-heads.md)。
