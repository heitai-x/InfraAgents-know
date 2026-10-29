# S-FLASHINFERPAPER：FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving

## 原文

[原始 PDF](../../references/S-FLASHINFERPAPER/original.pdf) · [完整原文](../../references/S-FLASHINFERPAPER/reader.html) · [全文文本](../../references/S-FLASHINFERPAPER/fulltext.txt)

固定版本：2501.01005v1。版本坐标日期：2025-01-02；整理核对日期：2026-10-27。可直接阅读全文；下面是本库核对后抽出的机制、算法与结论。

## 问题

服务中的 attention 面临三种动态性：KV 的存储形态不一（分页、共享前缀、稀疏）、attention 变体不一（mask、softcap、窗口、无 softmax）、每步请求长度不一而 CUDA Graph 要求静态配置。论文把这三件事分别交给一种数据结构、一个编译时模板和一个运行时调度器。

## 机制

Attention 状态与合并算子（§2.2）。对 query q 与 key 索引集合 I，定义 LSE(I) = log Σ_{i∈I} exp(q·kᵢ)，O(I) = Σ_{i∈I} exp(q·kᵢ − LSE(I)) vᵢ。二元组 (O(I), LSE(I)) 称为 attention 状态。I 与 J 不相交时，
O(I∪J) = (e^{LSE(I)}O(I) + e^{LSE(J)}O(J)) / (e^{LSE(I)} + e^{LSE(J)})，LSE(I∪J) = log(e^{LSE(I)} + e^{LSE(J)})。
该算子满足结合律与交换律，因此任意顺序合并都得到同一结果；它被当作 attention 的标准归约算子（类比 GEMM 的加法）。

块稀疏行格式作为 KV 的统一表示（§2.3、§3.1）。分页 KV 的页表可以看成行块大小 Br、列块大小 Bc 的块压缩稀疏矩阵：行对应 query tile，列块对应物理页。Br 取 query tile 大小，Bc 取页大小；从稀疏索引数组计算 KV 地址，从稠密行偏移计算 Q 地址。

可组合格式（§3.1）。同一批请求的 KV 用多个块稀疏矩阵存：共享前缀部分用更大的 Br（多个请求的 query 行合并成一个 tile，共享前缀的 K/V 只从 HBM 读一次进入共享存储），各请求独有部分用 Br 较小的另一个矩阵。两部分各算出 attention 状态，再用合并算子合并。不需要搬动 KV 数据，只重新计算索引数组。

变体模板（§3.2.3）。把 attention 变体分解为几个可替换的函数：query/key/value 的变换、logits 变换、logits 掩码、输出变换，以及是否使用 softmax；JIT 把这些函数插入固定模板生成 kernel。query tile 为 1 时用标量核模板（矩阵指令最小行数 16 装不满），否则用矩阵核模板。

负载均衡调度（§3.3.1，Algorithm 1）。输入每个请求的 query 长度与 KV 长度，以及 query tile 大小 Tq：
1. 定义一个工作块的成本 cost(lq, lkv) = α·lq + β·lkv。
2. 取 KV 最大分块长 Lkv = Σᵢ ⌈lqo(i)/Tq⌉·lkv(i) / CTA 数。
3. 把每个 query tile 的 KV 按 Lkv 切成块，形成工作集合 W，按长度降序排序。
4. 用最小堆维护各 CTA 的累计成本；反复弹出成本最小的 CTA，把 W 中最长的块分配给它。
输出是每个 CTA 的工作队列和部分输出到最终输出的索引映射。合并顺序由调度决定，相同长度信息下结果确定；论文为此放弃了原子累加。

与 CUDA Graph 兼容的运行时（§3.3.1、附录 D）。调度器在 CPU 上计算计划，异步复制到设备 workspace 的固定区段；attention kernel 与合并 kernel 都是固定 grid 的持久 kernel，按工作队列执行。workspace 各区段按上界（元数据上界、部分输出最多 2×CTA 数）分配并保持地址不变，因此每步重放的 kernel 参数相同。短 KV 的请求不分块，直接写最终输出，省掉合并。计划每步生成一次，可被同一步的所有层复用。

## 结论

论文报告：相对编译器后端的服务基准，token 间延迟降低 29–69%；长上下文推理延迟降低 28–30%；并行生成加速 13–17%。比较对象、模型与硬件见 §4。

## 本库据此推导

- 合并算子是"分区 attention 必须互不重叠且共享 score 定义"这一不变量的数学形式，见 [Attention 执行接口](../frameworks/attention-backends.md)。
- 页表即块稀疏矩阵、共享前缀即更大行块，说明"状态表示"轴上的点可以统一到一种索引结构；代价是索引构造与可组合格式的额外合并，见 [分页 KV](../algorithms/paged-kv-capacity.md) 与 [前缀缓存](../algorithms/prefix-cache-identity.md)。
- 把调度结果写成固定缓冲中的数据、用固定 grid 消费，是"动态任务进入静态图"的一种表示；归约确定性与图兼容是两个独立性质，见 [CUDA Graph](../frameworks/cuda-graph-buffers.md)。
- 成本函数 α·lq + β·lkv 与按 CTA 数取 KV 分块上界，是 split-K 划分粒度的一个可参数化起点，见 [Attention 的 IO](../algorithms/attention-io.md)。

## 适用边界

论文描述的是 v0.2 设计；后续源码在选项与限制上有所不同，接入具体版本时按归档源码核对。性能数字针对 A100/H100 与 2024 年底的基线。版本日期不证明模型训练实际包含该资料。
