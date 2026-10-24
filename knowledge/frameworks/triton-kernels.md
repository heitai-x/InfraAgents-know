# K-FW-007：Triton Kernel：块程序、布局与融合边界

Triton 用块级程序描述张量工作，性能来自把复用、归约和线程布局映射到目标 GPU 资源。

## 程序实例与张量 tile

一个 program instance 通常负责输出 tile，内部对输入 tile 进行加载、计算和归约。tile 的逻辑形状不直接等同于硬件线程数，编译器布局决定元素分给哪些线程及寄存器。mask 保证尾部越界元素不参与错误读取或计算，但被 mask 的线程和资源未必完全免费。

对矩阵乘法，增加 M/N tile 提高数据复用，增加 K tile 改变循环与片上驻留。更大 tile 也增加寄存器和 shared memory，使可并发 blocks 变少。decode 的小 M 与 prefill 的大 M 需要不同候选配置。

## 融合从数据流推导

把 bias、激活、归一化或量化与相邻算子融合，可以减少中间 HBM 写读和 launch。若融合引入大归约或让原本矩阵高效路径失效，则收益可能消失。一个合理边界让中间量在寄存器/片上消费，同时保持足够并行度。

attention 使用在线 softmax 归约，数学统计量允许沿 KV 分块。全 mask 行、非整 tile、不同 query/KV 长度和 scale 都要遵循原函数。与普通逐元素 kernel 相比，它对归约次序和累计精度更敏感。

## Autotune 和编译成本

autotune key 应包含真实影响最优配置和合法性的 shape/dtype/布局条件。首见形状触发多配置试跑会增加冷延迟；固定 workload 可以预先形成少量候选，动态服务则应控制变体数量。调优目标应覆盖实际 padding 和 launch 路径，而非孤立最漂亮的 tile。

可移植语言不保证同一配置跨 CUDA/ROCm 最优。不同 warp/wave、矩阵指令、寄存器和 LDS/shared memory 约束需要各自调优。引擎应优先复用已支持的 kernel，在明确缺失的形态上才承担新内核的维护成本。

## 从矩阵形状推到程序与线程工作

以 C[M,N]=A[M,K]B[K,N] 为例，输出 program 坐标决定 M/N tile，归约沿 K 分段。A 的地址由逻辑行、K 索引及两个 stride 共同确定，B 同理；只有原始布局与公式吻合时才可把它简化为连续偏移。调度 program 的遍历顺序可以改变相邻块对 A/B 的 cache 复用，但不改变每个输出元素的归约集合。

设定 M=128、N=256、K=96。方案 A 用 BM=32、BN=64、BK=32，产生 16 个输出块、每块 3 次 K 迭代；方案 B 用 64×128×32，产生 4 个输出块。输入 FP16 时，A 每 stage 的 A/B tile 共 6 KiB，B 为 12 KiB；若双缓冲则分别为 12/24 KiB，尚未含 padding、barrier 和其他状态。FP32 输出累加 tile 分别含 2,048/8,192 个值，即 8/32 KiB 的逻辑累计存储。

若每个 program 使用 4 个 CUDA warp，共 128 个线程，均分累计值时平均分别为每线程 16/64 个 FP32 值。这只是逻辑下界式估计：矩阵指令 fragment 布局、临时输入、地址计算及编译器复制会改变实际寄存器需求，不能把 program 的张量维度直接当作线程维度或最终寄存器数。方案 B 复用更强，但块数更少、资源更多，在小矩阵上可能无法填满设备。

M=1 的 decode 将大量 M tile 行填空时，大 tile 的潜在复用并不存在。可以比较更小 M tile、GEMV、split-K 或跨请求聚批；split-K 会引入部分和、合并及不同归约次序。它适合补足并行度的条件需要与额外流量一起计算，不能只看 K 很大。

## 合并访问、片上流水与 spill

逻辑相邻元素应尽量映射到相邻线程的合并内存访问；shared/LDS 中的布局还要配合矩阵加载和 bank 组织。为了满足 tensor 指令而做的 transpose/swizzle 应与加载、消费及写回一致。移植到不同 warp/wave 或矩阵单元时，要重新核对线程布局，不能只把 num_warps 换一个数字。

流水多一个 stage，增加提前加载机会，也增加片上缓冲和活跃寄存器。资源过大可能减少驻留块，寄存器不够还会 spill 到 local memory，产生额外全局层级流量。“本地变量”并不保证只在片上。应从生成程序的寄存器/shared 使用、spill 和实际 occupancy 解释时延，再判断是否缩 tile、减 stage 或改变消费顺序。

尾块需要同时处理读写和归约身份。矩阵乘缺失的 K 元素可作为零贡献；attention 中越界 key 应从可见性中排除，全 mask query 需定义输出；归约最大值不能把无效元素无条件填零。预取越界地址也可能有问题，不能等到最终 store mask 才保护。非连续 stride、对齐变化、非整除维度是必要的验证点。

## 调优预算需要服务于完整工作负载

TVM 给出计算与 schedule 分离、tensorization 和显式同步的机制；Ansor 把结构选择、参数搜索、成本模型与实际硬件测量分开。它们说明搜索空间不仅是 BM/BN 两个数，还包括融合边界、缓存位置、归约划分和布局。候选首先满足合法性，再进入性能测量。

调优优先级可以由调用频率、当前关键路径占比和近期可获得的改善共同决定。设某 kernel 每请求出现 32 次，每次可节省 2 μs，则理想串行贡献为 64 μs；只出现一次的 kernel 即使快 20 μs，端到端贡献也更小。存在重叠时应看关键路径，不能机械相加。两个候选只差测量噪声时，应增加有意义的重复或保留更简单可靠的方案。

Autotune key 应包含影响合法性与性能的 dtype、shape、stride、mask模式和目标架构，不必加入与选择无关的请求 ID。试跑会写输出/状态的 kernel 需要隔离或恢复相同输入，避免后面的候选得到被前一个候选修改的 KV、随机状态或计数。首次编译、调优开销和缓存驻留应与稳定态分开报告。

最后对调优集之外的合法形状及真实服务混合验证，保留库实现或简单实现作为独立参考。大 tile/深流水在大批规则矩阵上可能占优，小 M、长尾、竞争性并发或寄存器紧张时则可能反转。通用编译语言、手写 CUDA 和厂商库都应按同一完整路径比较，不把历史论文中的某一次胜负变成永久能力排序。

## 教程给出一个执行域，不是任意 attention 的替换接口

固定提交 042b2cd 的 fused-attention 教程适合研究块程序与执行策略，但接入引擎前必须还原它实际实现的域。该示例的 forward 要求 Q/K/V head dimension 相等，限定一组 head dimension，并用 B×H×N_CTX×D 的共同几何建立 tensor descriptor。forward 的普通 Q/K 描述按紧密的 D 维与展平的 batch/head/token 关系构造；接口没有独立的 query/KV 长度、页表、ragged indptr 或不同 Hq/Hkv 参数。

因此，不能仅把一个模型层的任意 Q/K/V 指针传入就宣布接入成功。形状相同但 stride 不同的 view，也未必满足该 descriptor 的地址假设。若要用于分页 decode、GQA、cross-attention 或不同 value head dimension，需要扩展索引与状态契约，或者选择已支持该问题的后端。这里的限制属于归档示例的边界，不是 Triton 语言本身的能力上限。

示例还包含服务 forward 未必需要的工作。它为输出及每个 query/head 的 FP32 中间归一化量分配缓冲，并保存供 backward 使用的张量；实际推理路径是否需要输出 LSE、是否需要保存这些状态，应由下游消费关系决定。如果重构为仅推理的版本，可以研究减少无用状态，但必须保留被分区合并、其他验证或后续算法真实使用的统计量。

## 指针、对齐与整数索引也是内核输入

块程序中的每个地址都来自基址、逻辑索引和 stride，不能用 tile shape 代替这些量。对一个输出 tile，尾部 mask 需要同时约束输入加载和输出写入；若归约中使用填充值，还必须满足该归约的恒等关系。对于乘加，非法 K 槽的贡献可为零；对于 max，零可能大于所有合法负值；对于 RMSNorm，补零后分母也必须保留真实 hidden 长度。

索引类型影响这个关系能否被表达。若某个元素偏移达到 2³¹，以有符号 32-bit 整数保存已经不能表示其正值；FP16 对应的字节位置为 4 GiB。大缓存、跨 head 的 stride 或多页组合可能扩大偏移，即使单个 tile 很小。应在发生乘法和累加溢出之前采用足够宽的索引；对溢出结果做边界 mask，不能恢复原本应访问的地址。这个数值是整数范围推导，不能据此假设所有库或设备都具有相同索引上限。

基址对齐也会被合法 view 的存储偏移改变。用整齐的独立张量调优所得配置，不能直接证明带偏移的缓存 view 适合相同向量访问。固定教程的 descriptor 与后端矩阵指令有具体布局要求；若生产者不满足，应比较直接支持 stride 的实现、单次重排或改变生产布局的成本。主张“零拷贝”需要证明确实只建立视图并且消费者接受该表示，不能只看上层没有显式复制。

## 在线归约的底数、近似与输出精度

固定 fused-attention 的 forward 内核把 softmax scale 乘以 1/ln2，然后使用 exp2。原因是 2^(x/ln2)=exp(x)。中间最大值与分区质量也在同一以 2 为底的域内维护；最终写入 M 的量是 m₂+log₂(l)，即 log₂Σexp(score)，不是自然对数 LSE。若另一个合并接口期望 Lₑ=lnΣexp(score)，需要 Lₑ=ln2·M。直接将 M 代入自然指数，会赋予分区错误权重。

例如，两个不重叠分区的真实归一化质量为 1 和 4、归一化输出为 0 和 2，正确合并结果为 8/5=1.6。其 log₂ 质量为 0 与 2；若错误地计算 exp(0) 与 exp(2)，会得到约 1.7616。它不是舍入噪声，而是单位不同造成的函数变化。这个算例用精确设定说明跨内核中间量也有语义契约。

源码同时展示概率矩阵在参与 V 的矩阵乘法前被转换为内核选定 dtype，而累加输出以 FP32 维护。因而“FP32 accumulator”没有说明整条路径都是 FP32；QK 的乘法、指数近似、概率转换、PV 及最终输出各自有精度边界。固定 PyTorch 数值资料说明，相同存储类型也可能因 TF32 或 reduced-precision reduction 产生不同舍入过程。

教程的因果块用一个有限负常数压低不可见位置，其测试又选择了有限尺度的随机输入。这可以作为特定执行域的实现选择来研究，不能当作全 mask 行、任意极端 logits 或所有模型的通用语义保证。把示例改造成可交付引擎时，应先固定真实输入域和特殊值行为，再决定是否沿用这一表示。

## 编译变体、合法域与调优结果是不同层次

一份编译程序对应哪些常量和类型，一项配置在哪些输入上合法，以及一项合法配置在何种负载下更快，是三个相关但不同的问题。编译缓存避免重复生成同一程序；合法域排除越界、资源不足或不支持的表示；性能选择再比较 tile、流水和布局。只用 shape 相等判定三个层次都可复用，会漏掉 dtype、stride、mask、目标架构或数值策略变化。

固定教程的 autotune key 列出 N_CTX、HEAD_DIM、FP8_OUTPUT 和 warp_specialize，并另由编译参数和配置剪枝承接其他条件。它的候选 BM 为 64/128，剪枝删除 BM>N_CTX 的配置；当引擎需要更短长度时，现成候选可能全部被删去。合法做法是提供满足新域的配置或通用路径，不能把较短输入直接解释成已有长度。若采用 padding，还需要确保可见性、输出切片、资源成本和统计分母都正确。

同一教程在测试环境使用固定单配置，而性能搜索使用较大的候选集合。测试可复现性、候选空间和调优所得最优配置因此不能互相代替。也不能把教程中给某个 provider 的误差阈值直接继承为模型或服务的质量标准；目标任务的语义和数值要求优先。

autotune 还会多次运行候选。纯函数输出被覆盖通常容易隔离，带 KV append、scatter、随机状态或计数器更新的 kernel 则可能改变后续试跑的输入条件。若调优目标包括这些副作用，需要以相同初始状态比较候选，或者将可安全调优的计算段与一次性状态提交分开。否则最快配置可能只是执行了不同工作。

## 选择布局和流水时观察可区分的资源结果

tile 增大后变慢有多个解释：更少的输出块使设备并行不足，更大的片上状态减少驻留，寄存器 spill 增加内存流量，或者一次适配引入新的转置。它们产生的观测不同。改变 M 而保持 K/N 可以检验是否缺少输出并行；改变 stage 而保持数学工作量，可以检验流水隐藏延迟与片上容量的权衡；比较直接目标布局与带转换的路径，可以分离内核与适配成本。

寄存器数、shared/LDS 字节、实际 block 数、spill 和带宽是解释时延的证据，不能分别作为越大或越小越好的独立目标。CUDA 12.8.0 Best Practices 的 occupancy 章节明确指出更高 occupancy 不必然更快；减少寄存器可能引入 spill，保留更多寄存器又可能降低并行度。固定教程按设备和配置调整 warp specialization、num_stages 或 maxnreg，体现的是条件化的实现选择，不是跨平台可照搬的常数。

对多个同时活跃的推理请求，单 kernel 独占设备时的最佳配置也可能变化。更大的工作区、缓存占用或更长的不可抢占工作段，会影响其他工作能否重叠。应在目标服务的资源分配和并发模式中复核候选；若比较只在孤立微基准成立，就将结论限定在该条件。

## 调优回报与引擎验收

设调优与编译总成本为 C，每次真实调用节省 Δt，在其有效域内预计还会使用 N 次，则仅就运行时间而言，需要 NΔt>C 才能收回前期开销。设 C=1.2 s、Δt=2 μs，需要超过 600,000 次同等收益的调用。若每个 decode 步调用该内核 32 次，这对应超过 18,750 个这样的步骤；仍不包含版本维护、缓存驻留和形状不命中的成本。这是预算算例，不能当作实测收益。

调优结果还需要经过三层连接。第一层在明确支持域内确认数学输出、合法 stride、尾部和状态副作用；第二层纳入真实参数准备、工作区、布局转换和图捕获；第三层放回完整模型及请求流，观察质量、延迟、吞吐与容量。不能用训练 backward 的测试覆盖证明独立新增的分页 decode 路径，也不能以一个平方 attention 长度的成功替代变长服务。

冷编译、搜索试跑和稳定态应各自有明确口径。GPU 的异步返回时间并非执行时间；用于测量的完成依赖也不能任意串行化本应重叠的正常路径。保留一个语义正确的通用后端，可以在新形状未覆盖时维持任务能力，同时把 fallback 的比例和成本纳入结果。所有最终性能结论应来自目标条件下的真实运行，本章没有新增 GPU 实测。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TRITON：Triton：Fused Attention](../sources/S-TRITON.md)：Triton 的 fused attention 示例展示块级张量程序、在线 softmax 归约、tile 划分与相应配置选择。逻辑 tile 和实际线程/寄存器布局有映射关系。
  [完整原文](../../references/S-TRITON/reader.html) · [全文文本](../../references/S-TRITON/fulltext.txt)

- [S-CUDA-BEST：CUDA C++ Best Practices Guide](../sources/S-CUDA-BEST.md)：CUDA 性能资料区分有效带宽、访存组织、数据传输和执行资源。合并访问、复用和减少不必要搬运会改变实际流量。
  [完整原文](../../references/S-CUDA-BEST/reader.html) · [全文文本](../../references/S-CUDA-BEST/fulltext.txt)

- [S-ROCM：HIP：Hardware implementation](../sources/S-ROCM.md)：HIP 硬件说明介绍 CU、SIMD、向量寄存器、LDS 与矩阵运算资源。寄存器使用影响驻留和隐藏延迟能力，不同 AMD 架构的资源参数不同。
  [完整原文](../../references/S-ROCM/reader.html) · [全文文本](../../references/S-ROCM/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-TVM：TVM](../sources/S-TVM.md)：固定 v1 分离图级改写、张量计算、schedule 与硬件 intrinsic，描述融合、布局、协作加载、tensorization、显式同步和 code generation/runtime。
  [完整原文](../../references/S-TVM/reader.html) · [全文文本](../../references/S-TVM/fulltext.txt)

- [S-ANSOR：Ansor](../sources/S-ANSOR.md)：固定 v1 用结构草图、参数采样、进化搜索和学习成本模型优化完整张量程序，并按目标函数分配跨子图调优预算。
  [完整原文](../../references/S-ANSOR/reader.html) · [全文文本](../../references/S-ANSOR/fulltext.txt)

### 执行接口扩写的补充依据

- [S-TRITON 固定源码](../../references/S-TRITON/reader.html)：本次核对 _attn_fwd_inner、descriptor pre-hook、候选配置与剪枝、_attn_fwd、_attention.forward/backward 及 test_op。关于共同 N_CTX/HEAD_DIM、紧密 descriptor、exp2/log2 统计、概率转换和测试域的描述只覆盖提交 042b2cd 的这份教程。
- [S-TORCH-NUMERIC：PyTorch Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：固定提交 647b6a2 的 TF32、reduced-precision reduction 和批量计算说明，用于区分存储类型与真实数值路径。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)
- [S-TORCH-CUDA：PyTorch CUDA semantics](../sources/S-TORCH-CUDA.md)：固定提交 647b6a2 的异步执行、stream 和 CUDA Graph 约束，限定内核计时、输入完成和图模式接入。
  [完整原文](../../references/S-TORCH-CUDA/reader.html) · [全文文本](../../references/S-TORCH-CUDA/fulltext.txt)

整数偏移范围、LSE 底数反例和调优摊销数字为本库的可复算推导，不将教程测量或容差移植成 InfraAgents 的新性能与质量证据。

## 延伸阅读

- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [CUDA 平台：SM 数据流、占用率与执行粒度](../hardware/cuda-execution.md)。
- [ROCm 平台：CU、Wave 与 AMD 算子路径](../hardware/rocm-execution.md)。

- [张量地址与合法融合](../algorithms/operator-primitives.md)。
