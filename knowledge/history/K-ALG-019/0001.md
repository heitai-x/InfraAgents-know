# K-ALG-019：权重与专家卸载：重用、路由与在途版本

权重卸载在容量不足时改变读取路径。稠密层大多按固定顺序使用，MoE 专家由输入路由决定，激活稀疏模型又有不同的选择条件；三者不能共享一个未经区分的“只加载需要的参数”假设。

## 稠密层的层间重用

稠密 forward 仍需执行规定的所有层。可以让当前层权重驻留，轮流处理多个微批后再换下一层，以摊薄读取；代价是更多激活、KV 和较长的单请求等待。[FlexGen](../sources/S-FLEXGEN.md) §4 将这种调度与三级存储放置一起研究，目标偏向吞吐而非任意交互时延。

设有 Wₒ 字节权重每个生成迭代必须从主存加载，链路 B，若一轮推进 n 个请求，权重传输下界是 Wₒ/B，摊到每输出 token 是 Wₒ/(Bn)。后一项不能代替每请求 ITL。扩大 n 同时增加状态和激活，必须先满足容量与质量/延迟约束。

固定执行顺序使预取下一层较容易，但主机布局转换、量化解包和同链路 KV 流量可能成为新瓶颈。预打包格式也绑定目标内核和模型版本，不能将文件中的压缩字节直接交给任意设备算子。

## MoE 路由改变工作集合

一个 token 只使用部分专家，并不意味着一个 batch 只访问很少专家。若 E 个专家、每个请求选 k 个，且请求路由独立、每个专家的边缘命中率均为 k/E，则 B 个请求触及的专家数期望为 E[1−(1−k/E)ᴮ]。非均匀热点和请求相关性需使用相应分布，不能沿用这个设定。

例如 E=64、k=2，B=1 时为 2 个，B=16 时约 25.5 个，B=64 时约 55.6 个。每专家假设 128 MiB、全部冷且需跨同一链路加载时，batch 扩大可能迅速接近全专家读取。按 token 稀疏和按 batch 的权重访问稀疏不是同一指标。

缓存策略可以比较固定热门专家、按近期路由更新、需求加载和预测预取。固定热点稳定时简单有效，变化时可能持续缺失；需求加载避免无用传输但阻塞当前 token；预测预取以错误带宽和容量换取潜在提前量。不同专家大小、重建成本、所在设备与目标 deadline 应共同进入策略。

## 不可变权重与部分输出的所有权

权重对象由模型版本、层/专家、表示和分片身份确定。旧版本在途计算结束前不能回收或覆盖，即使模型路由已经切到新版本。专家执行完成后，dispatch 的 token 映射、路由权重和合并顺序仍要与本轮快照一致；不能只按一个可复用专家槽位接收结果。

一个专家部分加载时不得执行未完成范围。异步加载可发布已经完整且相互独立的专家，但整轮是否能提交仍取决于所有必需路由输出。加载失败、取消与路由重算需要明确处理方式，不能默默丢掉缺失专家的贡献。

## 激活稀疏与专家稀疏的边界

[PowerInfer](../sources/S-POWERINFER.md) 研究的是输入相关激活与热点神经元的 CPU/GPU 分工，不能直接视为任意 MoE 的专家缓存实现。神经元预测如果省略实际非零贡献会改变函数；已知为零的计算跳过、预测性近似和标准 MoE 的确定路由需要分别验证。

对稠密或专家层，还可以把冷权重对应的计算留在 CPU，传输较小激活或结果。这是 [计算放置](heterogeneous-execution.md) 的选择，收益取决于 CPU 算力、冷工作量和合并开销，不等同于把权重搬到 GPU。

## 验证与策略反转

先用固定路由和固定权重版本验证常驻/卸载的一致性，再改变热点、batch、专家大小与预取准确性。记录每轮唯一加载字节、重复加载、未被使用的预取、在途权重峰值、慢专家和最终 ITL/吞吐。使用独立路由映射参考检查输出合并；质量约束禁止任意跳过低概率专家。

较大 batch 增加单次加载后的重用，也可能扩大专家并集；更高热点集中度有利于固定驻留，热点漂移则可能使更新或直接计算更划算。策略需要在这些相反作用下选择，而不应只按 GPU 占用率或缓存命中率决定。

<!-- infraagents:provenance -->

## 来源与推导

- [S-FLEXGEN：High-throughput Generative Inference of Large Language Models with a Single GPU](../sources/S-FLEXGEN.md)：§3–4 将生成、状态容量、层/批遍历与 GPU/CPU/disk 放置一起建模；§4.2 讨论把部分主存 KV 对应计算交给 CPU。
  [完整原文](../../references/S-FLEXGEN/reader.html) · [全文文本](../../references/S-FLEXGEN/fulltext.txt)

- [S-POWERINFER：PowerInfer](../sources/S-POWERINFER.md)：§3–6 以激活局部性、预测器、神经元放置和 CPU/GPU 结果合并组织混合执行；§8.1 明确评估模型包含特定 ReLU/ReGLU 变体。
  [完整原文](../../references/S-POWERINFER/reader.html) · [全文文本](../../references/S-POWERINFER/fulltext.txt)

- [S-MIXTRAL：Mixtral of Experts](../sources/S-MIXTRAL.md)：Mixtral 采用稀疏专家结构：每个 token 根据路由选择部分专家参与计算，而专家总参数与每 token 激活参数是不同数量。
  [完整原文](../../references/S-MIXTRAL/reader.html) · [全文文本](../../references/S-MIXTRAL/fulltext.txt)

- [S-CUDA-ASYNC：CUDA Programming Guide — Asynchronous Execution](../sources/S-CUDA-ASYNC.md)：CUDA stream 表达设备工作的顺序，event 可以在不同执行队列间建立完成依赖。host 调用返回、设备工作开始和设备完成属于不同事件。
  [完整原文](../../references/S-CUDA-ASYNC/reader.html) · [全文文本](../../references/S-CUDA-ASYNC/fulltext.txt)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [多级驻留：有效副本、预取、淘汰与重算](../algorithms/tiered-residency.md)。
- [CPU–GPU 计算放置：传数据、传结果与共同提交](../algorithms/heterogeneous-execution.md)。
- [MoE：路由、Dispatch/Combine 与专家负载](../models/moe-routing.md)。
- [权重加载：从制品身份到可执行布局](../frameworks/checkpoint-loading.md)。
- [推理并行：张量分解、流水、专家与上下文](../algorithms/parallelism.md)。
- [CPU 低位执行：解包、补偿与累加路径](../frameworks/cpu-quantized-kernels.md)。
