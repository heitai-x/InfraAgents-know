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

编译和 graph 可以改变调度成本，却不替代这些语义。先固定模块模式和请求状态关系，再验证 eager/compiled 的函数关系，能够将模型错误与执行优化错误分开定位。

<!-- infraagents:provenance -->

## 来源与推导

- S-TORCH-INFERENCE：[完整原文](../../references/S-TORCH-INFERENCE/reader.html) · [全文文本](../../references/S-TORCH-INFERENCE/fulltext.txt)
- S-TORCH-CUDA：[完整原文](../../references/S-TORCH-CUDA/reader.html) · [全文文本](../../references/S-TORCH-CUDA/fulltext.txt)
- S-TORCH-REPRO：[完整原文](../../references/S-TORCH-REPRO/reader.html) · [全文文本](../../references/S-TORCH-REPRO/fulltext.txt)

- [S-TORCH-INFERENCE：PyTorch — inference_mode](../sources/S-TORCH-INFERENCE.md)：inference_mode 关闭推理不需要的 autograd 相关开销，并具有张量使用边界；它不会自动将模块切换到 eval 模式。

- [S-TORCH-CUDA：PyTorch — CUDA semantics](../sources/S-TORCH-CUDA.md)：PyTorch CUDA 运算通常异步提交；不同 stream 的依赖需显式表达，allocator 还需要知道跨 stream 的使用。CUDA Graph 的重放路径依赖稳定的地址与兼容的执行结构。

- [S-TORCH-REPRO：PyTorch — Reproducibility](../sources/S-TORCH-REPRO.md)：随机种子与确定性配置可以控制特定执行路径中的随机性，但不保证跨版本、平台和全部算法路径逐位一致。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 4，2026-09-20。

## 延伸阅读

- [数值参考：误差尺度、条件数与首次分歧](../frameworks/numerical-reference.md)。
- [编译与动态形状：Guard、分桶和变体成本](../frameworks/compile-dynamic-shapes.md)。
- [位置、Mask 与缓存：同一可见性关系的三种表示](../system-semantics/position-mask-cache.md)。
