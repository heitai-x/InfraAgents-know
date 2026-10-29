# K-FW-005：CUDA Graph：捕获边界、计划状态与变长服务

CUDA Graph 重放一组已捕获的设备工作；服务动态性通过稳定缓冲中的值、分桶和捕获外控制面承接。

## 图实际固定了什么

capture 将设备工作及其依赖记录为图，实例化后重复 launch。普通框架 capture/replay 路径通常依赖稳定的输入、输出、workspace 与参数地址，以及可重放的拓扑和形状。图中读某个地址，并不意味着该地址内容不能改变；正是更新内容使不同请求能共享一张图。

需要区分宿主端值和设备端值：CPU 分支在 capture 时选择的路径不会在 replay 时重新执行；设备 kernel 从 length tensor 读取的值则可以变化，前提是仍满足图和 kernel 的容量约束。把一个 Python 整数改掉不等于图节点参数已更新。

## 完整捕获与分段捕获

完整捕获可以减少更多 host launch，但要求整段路径都兼容。动态调度、某些 attention plan、CPU grammar、数据相关输出处理可能留在图外。分段捕获把稳定区域分别图化，保留控制边界；这样增加图间提交，却降低对动态部分的约束。

选择依据是 host 开销在何处，以及跨段中间状态的大小。若主要开销集中在若干线性/归一化链，分段已经可能有价值；若每层 attention 都留在 host，分段数很大，收益可能受限。不能以“使用 CUDA Graph”这个布尔值描述捕获覆盖。

## 变长批怎样进入静态容量

选择容量为 B 的图处理实际 b≤B 个请求，需要为真实请求填入 token、position、KV 页表、有效长度和采样参数。填充行必须按 backend 的有效规则处理：长度为零未必被所有 kernel 接受，使用 dummy 页也必须保证写入不会污染真实状态。输出只路由真实行。

图桶 key 可能包括 batch/token 容量、prefill/decode 模式、dtype、head 几何、LoRA rank、投机长度和 attention backend。固定 batch 并不固定总 query tokens；混合 prefill 的打包布局需另行表达。多桶减少 padding，却增加图池及持久缓冲占用，从而挤压 KV 容量。

## Attention 的准备/执行边界

面向服务的 attention 接口常把依赖 metadata 的准备（读取长度与页表、安排 workspace、选择 kernel）与依赖张量的执行分开；准备段通常不能进入图，执行段在满足图模式、预分配元数据和 workspace 契约时可以。可在捕获外准备，再把受支持的执行段捕获进图。

源 KV 未变不保证准备结果未变：batch 成员、query 分段、总 token、页索引、tail 长度或 kernel 特化条件都可能变化。计划缓存 key 必须覆盖准备段真正依赖的项。某些 metadata 可在固定地址中修改，某些变化需要重新准备或另一张图；以所选接口的归档原文为准，两段的成本结构见 [Attention 执行接口](attention-backends.md)。

## 缓冲内容的时间所有权

图 replay 是异步的。replay n 还在读取页表时，CPU 或另一 stream 把它改成 n＋1，会把两代状态混合。稳定地址解决 capture 的引用问题，不解决内容竞争。双缓冲、代次槽或事件保护应覆盖输入、metadata、输出和 workspace 的整个在途区间。

多图共享同一私有内存池或同一 workspace，只有在其重放顺序和生存期兼容时才安全。并发 replay 与串行 replay 的容量模型不同；graph 输出张量若下一次 replay 原地覆盖，消费者需要在覆盖前读完或复制。取消请求也不能立即将图仍引用的 KV 页重新分配。

## 捕获成本与收益推导

若每轮有 K 个短 kernel，host 提交成本约 Kh，图 launch 成本 g，则可节省的主要项约 Kh−g；GPU 算术和数据流通常仍在。图 padding 增加的执行量与图池容量可能抵消收益。capture、warmup、分配和库初始化也应从冷路径单独计入。

一种完整的设计说明应能回答：哪些工作被捕获，哪些依赖值可变，哪些地址必须持续存在，何时切桶，谁写入下一代，谁确认上一代最后一次读取完成。这些答案决定图与连续批、交叉注意力、投机和取消能否正确组合。

## 卸载、LoRA 和多 stream 的组合条件

卸载权重时，图捕获的设备地址必须在每次 replay 读取时指向已经完成换入的正确层/版本。固定 staging 地址可以轮流装不同内容，但同一地址还被旧 replay 读取时不能覆盖。换入完成事件应在消费 stream 等待；最后一次读取完成后才允许下次填充。主机权重存在不代表设备数据已就绪，地址仍有效也不代表内容版本正确。

LoRA 路径若捕获了具体 adapter 指针，切换 adapter 需要使用受支持的图更新、另一图或固定槽填充；若 kernel 通过稳定 metadata 间接选择 adapter，则仍需保证目标槽、rank、布局与在途代次满足契约。固定 rank 桶可以承接较小 rank，前提是 padding 对低秩增量的数学贡献为零，scale 与 routing 一致。模型权重合并、adapter 切换和 cache 身份须同时保持兼容。

多 stream 捕获/replay 要在依赖图中包含输入、KV、通信、输出及释放关系。一个 event 保护输入并不自动保护共享 workspace；两个图单独执行都正确，不证明同时 replay 正确。共享图池只在生命周期与重放顺序满足条件时节省容量，需要检查并发最大活跃集合。

## 固定地址不等于任意计划都可重放

工作划分决定每个 CTA 处理哪些 query 行与哪些 key 块。若划分结果直接决定 launch 的 grid，KV 长度变化就会改变 grid，重放失效；若把划分结果写成固定容量工作队列中的数据，由固定 grid 的 kernel 读队列执行，动态任务便能进入静态图，代价是队列容量上界、空闲 CTA 与分区合并的额外输出。这是一种可行表示，不是所有接口选项的统一保证；来源实例见 [S-FLASHINFERPAPER](../sources/S-FLASHINFERPAPER.md)。

归约顺序是否确定与 launch 配置是否固定是两个独立性质。固定划分粒度可以让归约顺序确定，却不保证 CTA 数不随 KV 长度变化；固定 grid 也不保证合并顺序确定。从一个标志推出另一个是常见错误；两者和"准备段能否被捕获"一样，都按所选接口在归档原文中逐项核对。

图输入可以容纳的最大 query 行数、page index 数、tail 长度与 partial-output 数量，需要按具体路径建立上界。超界时切换合法图/普通执行或拒绝该计划；不应截断页表来让 replay 勉强运行。计划复用还需保持实现、head 几何、mask 与实际分段依赖一致。

## 图池应进入容量和收益账

设定四个桶各额外占 64 MiB，合计 256 MiB；若同一资源池中的 KV 每 token 占 128 KiB，这相当于 2,048 token 的容量。这个算例展示图池与并发/上下文容量的直接竞争，不表示真实图一定占该容量。捕获更多桶既能减少 padding，也可能降低可准入请求数，进而增大排队。

验证应包含反复捕获/释放、跨桶切换、并发 replay、取消后迟到完成、卸载槽复用、adapter 更换和超容量输入。检查最终输出与 reference、持久缓冲代次、峰值分配和稳定态基线，确认释放后回到预期占用。不能只以一次 replay 的 kernel 时间判断图策略；需要在相同 offered workload 下看完整时延、完成率以及图池挤占后的有效容量。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TORCH-CUDA：PyTorch — CUDA semantics](../sources/S-TORCH-CUDA.md)：PyTorch CUDA 运算通常异步提交；不同 stream 的依赖需显式表达，allocator 还需要知道跨 stream 的使用。CUDA Graph 的重放路径依赖稳定的地址与兼容的执行结构。
  [完整原文](../../references/S-TORCH-CUDA/reader.html) · [全文文本](../../references/S-TORCH-CUDA/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

- [S-FI：FlashInfer：Attention kernels](../sources/S-FI.md)：截止日前的 FlashInfer 固定源码区分 query 分段、KV 页表及末页长度。相关 wrapper 的 plan 安排计算与工作区，run 执行 attention；plan 的捕获限制与图模式缓冲须按该版本接口核对。
  [完整原文](../../references/S-FI/reader.html) · [全文文本](../../references/S-FI/fulltext.txt)

- [S-TRTLLM：TensorRT-LLM Architecture Overview](../sources/S-TRTLLM.md)：已读 TensorRT-LLM 架构页将入口、PyExecutor、scheduler、model engine 与 sampler 连接成后台迭代，并介绍 graph padding 和 overlap 思路。
  [完整原文](../../references/S-TRTLLM/reader.html) · [全文文本](../../references/S-TRTLLM/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-FLASHINFERPAPER：FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](../sources/S-FLASHINFERPAPER.md)：固定 v1 描述 BSR/ragged/可组合 KV 表示、attention 状态合并、JIT 变体、动态负载计划与 CUDA Graph workspace，以及 GQA 和 FP8 KV 路径。
  [完整原文](../../references/S-FLASHINFERPAPER/reader.html) · [全文文本](../../references/S-FLASHINFERPAPER/fulltext.txt)

## 延伸阅读

- [异步调度：依赖图、代次与双缓冲](../algorithms/async-scheduling.md)。
- [交叉注意力缓存：固定源、独立几何与可见性](../algorithms/cross-attention-cache.md)。
- [Attention Backend：语义、布局与执行契约的交集](../frameworks/attention-backends.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
