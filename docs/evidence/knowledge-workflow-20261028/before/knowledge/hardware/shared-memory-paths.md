# K-HW-013：共享内存路径：统一地址、物理驻留与显式迁移

CPU 和 GPU 能使用同一个指针，不足以说明它们读取同一份物理内存、拥有一致缓存，或不需要传输。设计首先区分地址空间、物理位置、访问能力和同步契约，再比较哪条路径适合实际重用。

## 四种不同关系

| 路径 | 数据怎样被访问 | 主要成本与条件 |
|---|---|---|
| 独立设备缓冲与显式复制 | CPU 与 GPU 各有目的缓冲，按依赖搬运 | 副本容量、DMA、传输完成与回收 |
| GPU 映射访问主机内存 | GPU 通过受支持的映射访问主机驻留页 | 链路访问、锁页/映射条件、访问粒度 |
| 可迁移的统一内存 | 运行时/系统按支持能力管理驻留与访问 | 缺页、迁移、预取、页粒度与来回迁移 |
| 物理统一内存 | CPU/GPU 使用共同内存池 | 共享带宽、缓存行为、执行顺序和系统内存压力 |

[CUDA 12.8 固定资料](../sources/S-CUDA-ASYNC.md) 按设备与系统能力区分 Unified Memory 支持级别，并说明数据移动仍会发生。[MLX 固定文档](../sources/S-MLX-MEM.md) 描述 Apple Silicon 的共同内存池与跨设备操作依赖。两者不能仅因名称相近就采用同一套迁移或 allocator 假设。

## 显式搬运和远端读取的临界点

设 D 字节数据被 GPU 完整读取 R 次，链路带宽 Bₗ、设备本地带宽 B𝒹。忽略 cache、同步和重叠，先复制再读的时间约 D/Bₗ+RD/B𝒹，直接远端读取约 RD/Bₗ。复制的价值随着本地重用增加；只访问对象很小一部分时，应改用实际访问字节和事务放大重新计算。

设定 D=64 MiB，Bₗ=16 GiB/s、B𝒹=512 GiB/s，R=4。显式复制加本地读取的理想成本约 4.3945 ms，四次远端读取约 15.625 ms。R=1 时，两者则约 4.0283 ms 与 3.90625 ms，复制优势消失。这个例子没有代入任何具体设备保证；实际 cache 命中、映射开销和可重叠传输可能改变结果。

对于物理共享内存，省掉显式副本不等于 RD 项消失。CPU 预处理、GPU 权重读取与其他进程会竞争相同资源；若同时工作使总需求超过带宽，简单把两个阶段时间取最大值就会低估执行时间。

## 页粒度与来回迁移

按页迁移的系统可能为少量被访问数据移动整页。CPU 修改一部分、GPU 随后读取，再由 CPU 访问，可能产生驻留反复变化。可比较批量交接访问权、独立只读副本、显式缓冲和适用的预取提示；选择应由实际系统的支持属性与访问轨迹决定。

预取是对未来使用的提示或显式工作，不能取代消费前的完成检查。提前过多会挤出其他对象；缺页后自动搬运也不一定满足交互 deadline。共享地址方案的工作区、页表、系统保留和临时副本仍应进入 [内存预算](../system-semantics/memory-allocators.md)。

## 可见性与所有权

生产者写完并建立所要求的可见性后，消费者才开始；消费者结束后，生产者才可覆盖。统一地址或物理共享均不取消这条依赖。框架可以为其管理的数组操作建立顺序，但通过原生指针、外部库或其他进程访问时，仍要核对相应互操作契约。CPU/GPU 同时更新同一可变对象是否支持特定原子或一致性语义，也必须按固定平台版本确认。

取消时保留已经提交的读写保护，迟到完成按对象与代次清理。地址稳定的 graph、共享输入视图和系统页迁移可以有不同约束；能够运行一个简单相加样例，不足以证明完整模型中的别名、生命周期与迁移组合正确。

## 如何选择并验证

分别观察初次访问、重复读取、CPU/GPU 交替写读、不同访问比例与系统内存压力。计时要包含真正求值完成，统计实际传输/缺页、带宽与峰值驻留；在支持相应观测的环境里，用它们区分传输减少、cache 命中和只是延后工作。

小而一次性的输入可能适合直接访问；反复读取的大权重通常值得本地驻留；大对象中稀疏少量访问需要比较事务与页放大；物理共享设备则更关注带宽竞争和有界容量。[异构计算放置](../algorithms/heterogeneous-execution.md) 再决定把数据交给 GPU，还是让某段计算留在 CPU。

<!-- infraagents:provenance -->

## 来源与推导

- [S-MLX-MEM：MLX：Unified Memory](../sources/S-MLX-MEM.md)：MLX 利用 Apple Silicon 的统一内存，CPU/GPU 可以对共享内存中的数组执行操作，设备选择发生在操作执行层。
  [完整原文](../../references/S-MLX-MEM/reader.html) · [全文文本](../../references/S-MLX-MEM/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

- [S-TORCH-CUDA：PyTorch — CUDA semantics](../sources/S-TORCH-CUDA.md)：PyTorch CUDA 运算通常异步提交；不同 stream 的依赖需显式表达，allocator 还需要知道跨 stream 的使用。CUDA Graph 的重放路径依赖稳定的地址与兼容的执行结构。
  [完整原文](../../references/S-TORCH-CUDA/reader.html) · [全文文本](../../references/S-TORCH-CUDA/fulltext.txt)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [主机准备、DMA 与流水缓冲](../hardware/host-device-transfer.md)。
- [Apple Silicon：统一内存、带宽与交互式推理](../hardware/metal-unified-memory.md)。
- [CPU–GPU 计算放置：传数据、传结果与共同提交](../algorithms/heterogeneous-execution.md)。
- [内存分配：地址、容量与在途所有权](../system-semantics/memory-allocators.md)。
- [Stream 与分配器：依赖顺序和对象生存期](../frameworks/streams-allocator.md)。
- [MLX 服务：惰性求值、编译与请求状态](../frameworks/mlx-serving.md)。
