# K-FW-001：PyTorch 推理：模块模式、Autograd 与状态突变

推理行为由模型模式、梯度记录、dtype/device、模块状态与执行路径共同决定。

## 两个独立开关

模块的 training/eval 状态影响 dropout、BatchNorm 等层的函数行为；关闭梯度记录影响 autograd 图和内存开销。inference_mode 不会自动把模型切换为 eval，eval 也不会自动关闭梯度。一个服务可能数值上仍在随机 dropout，同时又没有梯度，这并不矛盾。

inference_mode 进一步减少 view tracking 和 version counter 等推理开销，但它创建的张量有使用限制。若某个后处理或适配器流程还要参与 autograd，需明确在哪个边界产生与消费张量。服务线程各自的执行上下文也应符合框架语义，不能因为初始化线程进入某个模式就假定所有线程继承。

## 权重、buffer 和请求状态

parameters、模块 buffers、临时激活和每请求缓存有不同所有权。RoPE 查表可作为共享只读 buffer；请求 KV 是可变状态；BatchNorm 的统计在某些模式下会更新。把请求缓存藏进共享模型对象且不按请求隔离，会使单请求正确而并发错误。

权重加载还涉及 tied weights、dtype 转换、device 放置和自定义模块。shape 相同不能证明语义相同：转置权重、不同归一化 epsilon、不同激活近似都可能改变函数。模型迁移需要从原 forward 的操作顺序还原等价关系。

## 动态 Python 与设备执行

Python 分支读设备标量会产生同步；构造新的 tensor、隐式 dtype 转换或非连续张量复制会引入额外分配。把这些开销归到“模型慢”会掩盖执行路径。清晰的推理边界通常让输入规范化在外，设备前向消费明确 shape 与状态，输出解释再在定义好的完成点进行。

编译和 graph 可以改变调度成本，却不替代这些语义。在同一模块模式和请求状态条件下比较 eager/compiled 的函数关系，有助于区分模型错误与执行优化错误。

## 把模型接入分成可检查的边界

[加载契约](checkpoint-loading.md)、processor 与 forward、每请求状态、并批和优化后端分别提供可观察的函数边界。对同一模型输入比较相关边界的结果，可以把一次大的“引擎不一致”定位到最早发生分歧的位置。实现可以整体构建，也可以按当前问题拆分；这些验证边界不规定组件开发顺序。

模块初始化后还可能在第一次实际输入时生成位置表、选择算法或创建临时缓存。预热需要覆盖计划服务的执行路径，并确认新创建对象的 device、dtype 与共享方式。第一次请求触发共享 buffer 扩容时，应避免其他线程同时读取尚未发布完整的新对象；不可变版本替换或明确的同步边界比依赖调用顺序更容易验证。

## 原生扩展与执行上下文

原生算子接收的不只是一个数据指针，还应约定 shape、stride、dtype、device、当前执行 stream、输出别名和 workspace 生命周期。扩展内部偷偷切到另一个 stream、保存指向可重用临时区的地址，或者返回前没有建立必要依赖，都会让 eager 单线程成功而并发失败。同步返回、异步返回并建立事件依赖，是不同的合法接口，调用方必须知道采用哪一种。

执行上下文在实际运行模型的线程内建立。初始化线程的梯度模式不能代替 worker 的上下文；异步任务也可能把前处理、模型提交和后处理交给不同执行域。是否释放语言运行时锁、是否能并行执行 CPU 工作，需要由具体扩展实现确认，不应由“原生”这个名字推断。线程池和有界队列的资源关系见 [主机运行时](../system-semantics/host-runtime.md)。

## eager、编译与图执行的对照

无编译路径可在其语义独立且覆盖目标输入时作为参考，帮助区分编译、布局变换、融合和状态复用的影响。编译先行或整体原型也可以使用独立算子、状态和模型参考，不要求事先交付一套完整 eager 引擎。对照除了输出数值，还包括新旧状态、别名、错误行为以及 shape 边界。编译 guard 不满足后重新编译或回到其他路径会改变性能分布；只计命中已有 executable 的请求无法代表冷形状负载。

例：一个优化把 QKV 投影合并，并把 RoPE 与 KV 写入融合。分层对照可沿 Q/K/V 分段、旋转结果、写入位置与 attention/logits 定位首次分歧；直接比较最终 token 容易把轴错误误判成浮点差异。对量化或近似 kernel 另设质量验证，不把 eager 参考的单一容差扩大到任意近似路径。

当 Python 调度主导时，减少调用与提交次数有价值；当矩阵或内存路径主导时，复杂编译缓存可能只增加启动和容量成本。把编译命中率、首次成本、工作区峰值和真实服务延迟一起比较，才知道该 Task 是否值得引入更多执行模式。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TORCH-INFERENCE：PyTorch — inference_mode](../sources/S-TORCH-INFERENCE.md)：inference_mode 关闭推理不需要的 autograd 相关开销，并具有张量使用边界；它不会自动将模块切换到 eval 模式。
  [算法与机制说明](../sources/S-TORCH-INFERENCE.md)

- [S-TORCH-CUDA：PyTorch — CUDA semantics](../sources/S-TORCH-CUDA.md)：PyTorch CUDA 运算通常异步提交；不同 stream 的依赖需显式表达，allocator 还需要知道跨 stream 的使用。CUDA Graph 的重放路径依赖稳定的地址与兼容的执行结构。
  [完整原文](../../references/S-TORCH-CUDA/reader.html) · [全文文本](../../references/S-TORCH-CUDA/fulltext.txt)

- [S-TORCH-REPRO：PyTorch — Reproducibility](../sources/S-TORCH-REPRO.md)：随机种子与确定性配置可以控制特定执行路径中的随机性，但不保证跨版本、平台和全部算法路径逐位一致。
  [完整原文](../../references/S-TORCH-REPRO/reader.html) · [全文文本](../../references/S-TORCH-REPRO/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-TORCH-DYNAMIC：PyTorch — Dynamic Shapes](../sources/S-TORCH-DYNAMIC.md)：编译会基于 shape、stride、dtype、常量等条件特化，并用 guard 确定专门程序是否可继续使用；部分维度可以符号化，变化也可能触发重编译。
  [完整原文](../../references/S-TORCH-DYNAMIC/reader.html) · [全文文本](../../references/S-TORCH-DYNAMIC/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

## 延伸阅读

- [数值参考：误差尺度、条件数与首次分歧](../frameworks/numerical-reference.md)。
- [编译与动态形状：Guard、分桶和变体成本](../frameworks/compile-dynamic-shapes.md)。
- [位置、Mask 与缓存：同一可见性关系的三种表示](../system-semantics/position-mask-cache.md)。
