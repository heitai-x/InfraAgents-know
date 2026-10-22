# K-HW-008：Trainium：HBM、SBUF、PSUM 与编译执行

Trainium 的高效执行来自把张量分块映射到片上存储和专用引擎，同时让静态可执行程序承接动态服务状态。

## 三层存储与引擎分工

NKI 的 Trainium2 架构资料描述 HBM、SBUF 和 PSUM：HBM 容量大，SBUF 放输入和中间 tile，PSUM 支持矩阵累加。DMA 在 HBM/SBUF 间搬运，Tensor Engine 做矩阵运算，其他引擎处理逐元素、归约与数据转换。优化目标是让这些资源在依赖允许时重叠，避免每个中间量都回写 HBM。

片上张量的 partition 轴与 free 轴具有硬件含义。收缩维映射、tile 大小和转置路径影响矩阵引擎能否高效供数。Trainium2 支持的 DMA 转置路径与前代资源不同，所以“转置是免费视图”在这里通常不是物理成本模型。

## Attention 的数据流

QK 乘法得到分数 tile，softmax 需要行归约和缩放，PV 再产生输出 tile。保存在线 softmax 统计可以避免完整分数落 HBM；但归约轴、PSUM 占用和 SBUF 生存期要与 tile 组织一致。计算引擎能并行工作不表示依赖数据可提前读取，DMA 完成仍是消费者的前置条件。

## 编译形状与服务形状

静态编译的 executable 有固定形状边界，在线请求则有可变 batch、prompt 和历史。分桶加 padding 将动态值放进稳定容量；有效长度和 mask 区分语义数据与填充。桶过大浪费算术和缓存，桶太多增加编译、加载和驻留成本。分桶策略应来自目标 workload 的联合分布。

持久 KV 可以驻留设备，token 步只更新必要位置和输入元数据。若每步把全状态往返主机，算力优势会被搬运覆盖。模型并行时状态分片、集合操作和 executable 组必须一致。NKI 与 NxD 的具体 API、可用 dtype 和实例代际以各自版本为准，本文给出的是数据流推导。

## 存储层级需要两本账

HBM 账覆盖权重、持久状态、可执行程序及运行时保留区、跨核通信缓冲和溢出 workspace；SBUF/PSUM 账覆盖当前内核的活跃 tile、双缓冲和累加。片上空间足够不能保证整模型 HBM 可容纳，HBM 有剩余也不能消除一次 tile 的片上冲突。[NKI tile 推导](../frameworks/nki-tiling.md) 给出具体字节和生存期。

固定 Trainium2 文档中，HBM↔SBUF 主要由 DMA 完成，PSUM→SBUF 使用计算引擎指令；两者的“搬运”不消耗同一种资源。Tensor Engine transpose 还会暂占 PSUM 并需要移回 SBUF；DMA transpose 可直接到 SBUF，却竞争 DMA。对一个 Scalar/Vector 已成为慢端的 attention，DMA transpose 可能减少该慢端压力；当 DMA 已被 KV 读取占满，选择又可能相反。

该版本允许 Vector 与 GpSimd 并行访问 SBUF，但部分吞吐模式共享总线；Vector 与 Scalar 可并行访问 PSUM 的条件包括不冲突于同一个 bank。引擎图上两条独立连线不等于任意访问都能全速重叠。DGE 负责描述符生成，减少主机静态描述符或软件生成的开销；固定文档仍明确硬件 DGE 不支持 indirect gather/scatter，分页式间接访问需要单独核对路径。

## 从单算子接到一个服务实例

以下是容量设定，不是实例规格或实测。设一个执行组有 4 个逻辑分片，每片可用于本服务的 HBM 为 20 GiB；总权重 48 GiB 可均匀分为每片 12 GiB，持久状态每片 4 GiB，程序及运行时每片 1 GiB，通信/workspace 峰值每片 2 GiB，则每片为 19 GiB，仅剩 1 GiB。若前缀执行还需每片 2 GiB 的 flat KV 暂存，这个配置即超出所设预算。全组还有其他空闲空间不能弥补不可迁移的局部峰值。

若给每个分片再加载一套 0.5 GiB 的特化程序，稳态或许减少 padding，但原剩余量降到 0.5 GiB；并发加载新旧权重则显然需要另一种排空或放置安排。实际程序占用须测量，不能把此处 0.5 GiB 当作 NEFF 的普遍大小。

并行度增加时，必须同时重算每片权重、KV head/latent 复制、collective 缓冲和每层同步。设一层计算因分片从 3 ms 降为 1.8 ms，但新增不可隐藏通信 1.5 ms，层时间反升为 3.3 ms。若 prefill 有更大矩阵、可隐藏其中 1.0 ms，关键路径又可能降到 2.3 ms。两组数字仅说明阶段差异；是否重叠要由依赖、链路和共享 HBM 争用验证。

对照应包括合法的较粗桶、较细桶和不同并行布局，保持输入/质量相同，量测启动、程序驻留、每层计算/通信、片上溢出与请求尾延迟。在平台或编译器版本变化后重新确认资源限制和实际执行图，不把本版 Trainium2 的 SBUF、PSUM、LNC 与指令限制套给所有 Neuron 设备。

<!-- infraagents:provenance -->

## 来源与推导

- [S-NKI：Trainium2 Architecture Guide for NKI](../sources/S-NKI.md)：已读 Trainium2 NKI 资料区分 HBM、SBUF、PSUM、DMA 与计算引擎；片上张量的 partition/free 轴和转置路径具有具体硬件意义。
  [完整原文](../../references/S-NKI/reader.html) · [全文文本](../../references/S-NKI/fulltext.txt)

- [S-NXDI：NxD Inference Features Configuration Guide](../sources/S-NXDI.md)：NxD Inference 说明静态形状可执行程序通过分桶和 padding 承接不同长度，context encoding 与 token generation 可配置不同桶；持久状态与模型执行共同受配置约束。
  [完整原文](../../references/S-NXDI/reader.html) · [全文文本](../../references/S-NXDI/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [Neuron/NxD：静态程序承接动态生成](../frameworks/neuron-serving.md)。
- [NKI：Partition 轴、片上生存期与 Attention 数据流](../frameworks/nki-tiling.md)。
- [异构状态缓存：层组、恢复位置与联合分配](../algorithms/heterogeneous-state-cache.md)。
