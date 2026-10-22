# K-ALG-021：KV 压缩：表示、近似保留与传输格式

KV 量化、丢弃历史和压缩传输减少的是不同成本。设计应说明模型实际消费什么状态、哪些信息改变，以及节省的字节发生在驻留、读取还是网络边界。

## 四种机制的语义边界

| 机制 | 改变的对象 | 需要验证的关系 |
|---|---|---|
| KV 低位量化 | K/V 的数值表示 | 量化执行正确性、误差与任务质量 |
| 重要 token 保留 | 当前可见历史的子集 | 新的近似 attention 与长程质量 |
| 压缩传输/存储 | 序列化后的字节流 | 解码语义、端到端质量与解码开销 |
| 模型原生窗口 | 模型定义的可见集合 | 实现是否保持既定窗口与位置语义 |

淘汰一个暂不使用的跨请求缓存副本，通常还可以在需要时重算；直接丢弃当前请求本应可见的历史，则改变本次模型函数。两者不能仅因都称为 eviction 就使用同一个正确性结论。

## 分组轴决定哪些数共享尺度

对一组数可定义 q=clip(round((x−b)/s))，重建值为 x′=sq+b，其中 s>0；常数组须另行定义表示，不能直接除以零。这里 b 是实数偏移；另一种格式写成 s(q−z)，其中 z 是整数零点。两种元数据的含义不同，加载、迁移与 kernel 不能混用。

[KIVI v1](../sources/S-KIVI.md) §3 研究了 K 与 V 的不同分组方式：K 固定 channel，在一组 token 间共享量化参数；V 固定 token，沿 channel 分组。名称“per-channel”描述参数属于哪个通道，不应误解成随意把不同通道混成同一组。文中的离群值与质量观察属于所评估的模型/输入，不证明所有架构都应使用同一分组。

K 的跨 token 分组需要等待一组元素齐备，尾部可以保留高精度后再封组；V 可按 token 追加，但也可以保留近期高精度区域。残余长度、组长和封组时机影响容量、计算与质量。物理分页、prefix 分叉和分组边界不一定重合，复制尾块时还要保留尺度、有效长度和是否已封组的状态。

## 有效位宽与峰值不能省略元数据

设一个组含 G 个值，每值 w bit，scale 与偏移合计 m 字节，则不计 padding 的有效位宽为 w+8m/G。高精度残余区、索引、对齐和解量化工作区另外计算。若先完整恢复 FP16 KV 再计算，网络可能省了字节，但显存峰值与 kernel 读取未必减少；tile 内解量化则把转换成本放进计算路径。

设定原始 FP16 KV 为 2 GiB、历史 4096 token，按 128 token 高精度残余上界计算，其余采用 2 bit、组长 32、每组元数据 4 字节。残余区为 64 MiB，量化码为 248 MiB，元数据为 124 MiB，合计 436 MiB，约为原始的 21.3%。这已经明显大于只按 2/16 比例得到的 256 MiB；实际格式还可能更大。该算例不声明这些参数达到任何模型的质量要求。

## 保留重要历史是一种近似策略

[H2O v1](../sources/S-H2O.md) §3–4 以累计 attention 统计与近期 token 组织有限预算。在线策略只能使用当前可得的统计，不能借用未来生成结果选择“重要 token”。被丢弃状态此后不可直接参与原全 attention；理论分析也有其评分函数与结构假设，不等价于任意模型输出无损。

实现应保留逻辑位置与物理槽的映射、评分更新、近期区和重用后的槽代次。压紧物理数组不意味着把原始位置重新编号；量化与保留同时使用时，分组参数必须跟随正确元素。质量对照需要长距离依赖、热点变化与多轮恢复，不能只测下一 token 或某个短文本集合。

## 传输压缩的临界点

[CacheGen v1](../sources/S-CACHEGEN.md) §4 将 KV 编成可传输字节流，并比较不同压缩等级与重新处理文本。其量化步骤可以有损，后续熵编码的无损性质不能扩展到整个管线。锚点差分还需与解码后的锚点一致，否则编码与解码使用不同参考。

令原始字节 D、压缩比例 r、编码/解码总时间 T𝒸、带宽 B。单次传输的简单比较为 T𝒸+rD/B 与 D/B；预先编码可改变在线计时范围，但离线成本仍存在。若 D=512 MiB、r=0.25、T𝒸=12 ms，20 GiB/s 链路上两者约为 18.25 ms 与 25 ms；100 GiB/s 时约为 13.25 ms 与 5 ms，选择反转。还须与传文本后重算比较，并保持相同前缀与恢复语义。

## 验证和组合

先固定量化码与元数据，使用独立公式检查 unpack、scale、偏移、残余拼接和封组；再与原模型比较固定前缀 logits、长生成和任务质量。压缩传输另测损坏/缺块、错误编码版本、解码峰值与吞吐。保留策略另测逻辑位置、评分更新和未来所需信息被丢弃的反例。

还要确认验证路径确实消费了压缩后的缓存。如果 prefill 使用高精度状态完成当前计算，而只把量化副本留给后续 decode，单测首 token 不能充分检验后续 KV 量化。短序列、空分组和残余区占比很高的输入应单独检查。

最终同时报告驻留字节、实际读取/传输字节、转换时间、有效并发与联合质量/SLO。低位表示腾出的容量可能提高并发，也可能把系统推入更严重排队；压缩比本身不足以决定系统方案。

<!-- infraagents:provenance -->

## 来源与推导

- [S-KIVI：KIVI](../sources/S-KIVI.md)：§3 分析 K/V 的不同分组轴，使用分组量化与高精度残余区处理流式追加；§4 给出模型和任务相关的评估条件。
  [完整原文](../../references/S-KIVI/reader.html) · [全文文本](../../references/S-KIVI/fulltext.txt)

- [S-H2O：H2O](../sources/S-H2O.md)：§2–4 定义有限 KV 保留集合，研究累计 attention 统计与近期 token，讨论带条件的动态子模分析。
  [完整原文](../../references/S-H2O/reader.html) · [全文文本](../../references/S-H2O/fulltext.txt)

- [S-CACHEGEN：CacheGen](../sources/S-CACHEGEN.md)：§4 以锚点差分、量化和熵编码产生可传输 KV 字节流，并根据带宽与计算估计选择压缩等级或文本重算。
  [完整原文](../../references/S-CACHEGEN/reader.html) · [全文文本](../../references/S-CACHEGEN/fulltext.txt)

- [S-INT8：Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](../sources/S-INT8.md)：§2 给出仿射量化、整数点积、零点补偿、bias 和输出重新缩放；附录 A–B 讨论组合算子、舍入及 NEON 中间累加条件。
  [完整原文](../../references/S-INT8/reader.html) · [全文文本](../../references/S-INT8/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [量化：误差传播、分组尺度与执行格式](../algorithms/quantization-contract.md)。
- [分页 KV：逻辑位置、物理布局与共享所有权](../algorithms/paged-kv-capacity.md)。
- [KV 卸载与预取：读集合、恢复点与迁移预算](../algorithms/kv-offload-prefetch.md)。
- [窗口与稀疏 Attention：可见性、位置和状态保留](../models/sliding-window-attention.md)。
- [数值参考：误差尺度、条件数与首次分歧](../frameworks/numerical-reference.md)。
- [性能实验：独立重复、效应量与尾部证据](../tools/statistical-experiments.md)。
