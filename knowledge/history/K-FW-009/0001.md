# K-FW-009：Neuron/NxD：静态程序承接动态生成

Neuron 服务通过形状桶、设备驻留状态和分布式执行组，把可变请求映射到编译后的程序族。

## 编译产物的身份

模型计算、形状、dtype、并行度、编译选项与目标硬件共同决定 executable。持久编译缓存复用的是符合相同编译条件的产物，不能仅按模型名字判断命中。权重替换是否需要重编译与其形状和具体接口有关，必须遵守版本契约。

context encoding 和 token generation 有不同的 query/历史几何，通常需要独立桶。NxD 文档说明采用能容纳输入的桶并补齐形状。有效位置、attention mask 和 KV 写入位置仍按真实请求值执行，padding 只是物理容量。

## KV 驻留与状态别名

若每个 token 将全部历史作为主机输入再送入 executable，流量随历史增长并失去设备缓存价值。驻留状态使新步只更新新 token 和少量元数据。输入/输出状态别名可以减少复制，但必须遵守编译器与运行时关于可变状态的定义，不能让两个请求写同一逻辑槽。

不同模型层组的 KV/SSM 可能对应不同形状与更新规则，prefix hit 的可恢复位置仍是联合条件。bucket 切换时，需要说明状态如何沿同一逻辑位置延续，而非把较大桶视为一个全新请求。

## 编译与服务的联合成本

大量 batch×context×模型变体形成程序组合，增加构建、加载与驻留成本。对少量高频形态特化、对长尾使用更粗桶，是由 workload 分布推导的常见策略。冷启动时间与稳定 token latency 需分别度量，才能判断方案适合交互服务还是离线批处理。

NxD、torch-neuronx 与其他 Neuron 后端的职责随版本演进。这里的入口聚焦已读文档中的分桶和驻留机制，不把某一版本默认配置当作跨代通则。

<!-- infraagents:provenance -->

## 来源与推导

- [S-NXDI：NxD Inference Features Configuration Guide](../sources/S-NXDI.md)：NxD Inference 说明静态形状可执行程序通过分桶和 padding 承接不同长度，context encoding 与 token generation 可配置不同桶；持久状态与模型执行共同受配置约束。
  [完整原文](../../references/S-NXDI/reader.html) · [全文文本](../../references/S-NXDI/fulltext.txt)

- [S-NKI：Trainium2 Architecture Guide for NKI](../sources/S-NKI.md)：已读 Trainium2 NKI 资料区分 HBM、SBUF、PSUM、DMA 与计算引擎；片上张量的 partition/free 轴和转置路径具有具体硬件意义。
  [完整原文](../../references/S-NKI/reader.html) · [全文文本](../../references/S-NKI/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [Trainium：HBM、SBUF、PSUM 与编译执行](../hardware/trainium-dataflow.md)。
- [编译与动态形状：Guard、分桶和变体成本](../frameworks/compile-dynamic-shapes.md)。
- [异构状态缓存：层组、恢复位置与联合分配](../algorithms/heterogeneous-state-cache.md)。
