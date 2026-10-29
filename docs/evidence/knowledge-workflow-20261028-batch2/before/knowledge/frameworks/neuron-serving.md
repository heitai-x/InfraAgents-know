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

## 从制品到可接入执行组

固定 NxD 文档将编译分为样例输入 tracing 生成 HLO、Neuron Compiler 生成 NEFF 两步。依赖输入张量数值的动态 Python 分支只会记录样例走到的路径，因此服务调度分支不能直接依赖 tracing 得到自动覆盖。持久编译缓存要求匹配编译器版本、模型配置和 flags；目录里存在 NEFF 不足以证明当前模型可以直接使用。

一次加载至少经过制品校验、权重分片准备、NEFF 加载、并行组建立、持久状态和 workspace 预留、代表性执行验证，之后才进入就绪集合。这是服务设计的生命周期推导，不宣称 NxD 将全部步骤作为一个原子 API。固定文档还说明编译默认不序列化分片权重；只搬运编译目录而遗漏权重处理可能把冷启动成本留到部署时。

新版本加载要与旧版本的在途请求共同计峰值。若不能同时容纳两套权重、程序和状态，应选择受控排空或备用资源，而非在旧请求继续执行时覆盖权重。某个 rank 加载失败，则这一执行组尚不具备服务资格；失败前其他 rank 已分配的对象也需要回收。

## 桶选择不替代状态延续

设定一条请求当前已计算 1000 token，接下来经过 1024 边界，候选程序桶为 1024 与 2048。选择大桶只解决可表示形状；真正延续还要求同一请求的 KV 身份、有效长度、位置和槽映射正确。若两程序采用不同状态布局，必须显式转换或使用已验证的共享布局。改变分片度则还涉及跨 rank 重分布，不能只换 executable 名称。

固定版本的前缀缓存依赖外部管理器提供 block tables 和 slot mappings，NxD 本身不负责完整的前缀哈希、淘汰和容量管理。该文档的执行路径先把 block KV gather 为 flat KV，再计算并 scatter 回块布局；因此登记了 block cache 不等于内核原生按页读历史。布局转换、flat 暂存峰值和有效历史长度必须进入性能及容量账。

前缀缓存的 context encoding 使用前缀长度和 suffix 长度两个桶轴，token generation 仍按最大序列长度选一维桶。固定文档允许把前缀末尾移到 suffix 重算，以填充选定的 suffix 桶；实际跳过的前缀长度可能小于发现的命中长度。没有覆盖所需前缀的桶会失败，不能把所有输入都默认映射到最近配置。

## 并行、采样与异步的组合边界

固定指南对 attention heads 的整除、平台支持的 TP 度和逻辑 NeuronCore 配置有明确限制。容量判断应按实际物理设备、逻辑分组和每份对象归属展开，不能重复使用不同层级的“每 core 内存”数字。文档给出的 GQA 策略包括复制 KV 至 TP 度以及展开为 MHA 形式；复制改变存储和读取量，数值等价不代表保留了全部 GQA 容量收益。

该版本 Async mode 依赖设备端采样；叠加投机时还要求 fused speculation。它重叠 CPU 与 NEFF 工作，不证明任意主机采样、任意投机树或外部服务循环都具备相同路径。EAGLE 在此文档中特指 flat draft，不能把树形验证能力从论文直接转移到后端。对不满足条件的组合应明确选择合法执行配置并记录成本。

以同一输入分别经过最小桶、较大桶、边界切换和恢复重放，比较逐层或逐步状态与输出；再在一个 rank 加载失败、队列取消、旧组执行未完成时切换等位置注入失败。稳态时间之外，单独计编译、加载、首次执行、block/flat 转换和程序驻留。更多桶减少 padding，但增加程序数量、加载压力和验证组合；选择应由真实形状分布决定。

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
