# K-MODEL-002：MHA、GQA 与 MQA：头映射和带宽复用

GQA 让多个 query heads 读取同一组 K/V；逻辑共享可以减少状态，但是否减少实际访存取决于 kernel。

## 三个 head 数的关系

设 query heads 为 Hq，KV heads 为 Hkv，每 query head 通道 d。MHA 通常 Hq＝Hkv，MQA 的 Hkv＝1，GQA 位于二者之间。常见均匀分组要求 Hq 是 Hkv 的整数倍，每组 Hq/Hkv 个 query heads 读取同一 KV head。输出仍有 Hq 个 head，不能因为 KV 变少就缩小输出投影输入。

批量 query 的逻辑形状为总 query tokens×Hq×d；KV 为总历史 tokens×Hkv×d。若 backend 要求显式重复 KV，数值上可以等价，但重复后的持久存储和流量可能抵消 GQA 的系统收益。支持原生分组的 kernel 可以在片上复用同一 KV，实际效率还受布局与 tile 限制。

## 容量和访存的独立推导

普通同构模型的缓存字节为 2LN Hkv d b。Hq 从 32 保持不变，Hkv 从 32 改为 8，缓存理论降到四分之一；模型 QK/PV 仍针对 32 个 query heads，不能把 attention FLOPs 同样除以 4。读取共享 KV 能否被合并，是数据流和缓存层级问题。

32 层、32 query heads、8 KV heads、d＝128、b＝2 时，每 token KV 为 128 KiB；4096 token 是 512 MiB。MQA 同尺寸变为 16 KiB/token。这里是结构算例，不表示可将已有 MHA checkpoint 无损改成 GQA/MQA；权重训练方式决定模型本身的质量。

## TP 下的头归属

当 Hkv 能被 TP 数整除时，KV heads 可分片。TP 大于 Hkv 或不能整除时，实现可能复制部分头、改变映射或拒绝配置。以 Hkv＝2、TP＝4 为例，若每个 rank 至少持有一个 KV head，则每头复制两份，全系统状态不会按 4 倍减少。通信与本地 query 分组也要对应这张映射。

## 从错误形态回到映射

单 head 正确、多个 head 错误，常说明分组索引或布局轴不一致；单卡正确、多卡错误，则应比较每 rank 的 query/KV head 归属和输出拼接。展开后能通过只说明数学路径可行，并不能证明原生 GQA 优化已经实现。由 head 映射推导独立的小矩阵参考，比只比较最终生成文本更容易定位差异。

## 非整除分片先写完整头映射

设 Hq=24、Hkv=6，原模型每 4 个 query heads 对应一个 KV head，TP=4，每 rank 分 6 个连续 query heads。四个 rank 分别需要 KV heads {0,1}、{1,2}、{3,4}、{4,5}，全组实际存储 8 份 head 数据，相比逻辑 6 份多三分之一。局部“6 query heads/2 KV heads=3”并不是原模型分组：例如 rank 0 的两组实际有 4 与 2 个 query heads。

合法实现可以显式传 global-to-local head 映射，或在支持的后端中按原分组补齐 query 并屏蔽填充输出，再恢复输出投影所需的全局 head 顺序；也可以选择能整除的并行度。后端只支持均匀局部分组时，不得直接把 6/2 当成正确分组。固定 NxD 文档中的复制/展开策略展示了一种后端选择，但不能推断所有后端支持相同非整除配置。

按每 rank 实际 head 数而非 TP 除法计容量。上述分片在层数、长度、dtype 相同时，每 rank 是完整逻辑 KV 的 2/6；全组是 8/6。若某 head 复制到两个 rank，逻辑上只有一份状态，物理副本、传输和量化元数据却分别占资源。

## 分页和量化不能改变头身份

分页改变 token 到物理块的映射，GQA 改变 query 到 KV head 的映射，量化再添加 group 到 scale/offset 的映射，三张表必须可同时解释。设 token 页长 16、沿 token 的量化组长 32，一组可跨两页；复制共享尾页或迁移其中一页时，不能遗漏属于该组的参数，也不能用目的页号代替源逻辑组号。也可在设计中强制页与量化组共同对齐，但应计算因此增加的尾部浪费。

同一 KV head 供多个 query heads 读取，允许复用整数码及对应参数；只复制整数码但从另一 head 取 scale，会保住形状而改变结果。原生分组 kernel、显式展开为 MHA 的参考、分页重排后的读取，应在相同解量化值上逐 head 比较。量化误差另与高精度参考比较，不把分片错误与可接受量化误差混在一个总容差里。

组合验证至少覆盖非整除 head 分配、一个页内多个量化组、一个组跨页、仅最后一页有有效 token、请求 batch 重排以及迁移后 head 次序。性能记录实际持久 bytes、重复展开 bytes、每轮读取和转换；逻辑容量减少与真实带宽节约分别下结论。

<!-- infraagents:provenance -->

## 来源与推导

- [S-GQA：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](../sources/S-GQA.md)：GQA 让一组 query heads 共享 K/V heads；MQA 是只保留一组 K/V 的特殊形态。输出 query head 数与 KV head 数需要分别描述。
  [完整原文](../../references/S-GQA/reader.html) · [全文文本](../../references/S-GQA/fulltext.txt)

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：Transformers 的缓存说明区分已存历史 K/V、新输入、attention mask 和 cache position。增量前向需要让新 token 的位置与已有缓存长度一致。
  [完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

- [S-FI：FlashInfer：Attention kernels](../sources/S-FI.md)：截止日前的 FlashInfer 固定源码区分 query 分段、KV 页表及末页长度。相关 wrapper 的 plan 安排计算与工作区，run 执行 attention；plan 的捕获限制与图模式缓冲须按该版本接口核对。
  [完整原文](../../references/S-FI/reader.html) · [全文文本](../../references/S-FI/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-KIVI：KIVI](../sources/S-KIVI.md)：§3 分析 K/V 的不同分组轴，使用分组量化与高精度残余区处理流式追加；§4 给出模型和任务相关的评估条件。
  [完整原文](../../references/S-KIVI/reader.html) · [全文文本](../../references/S-KIVI/fulltext.txt)

- [S-NXDI：NxD Inference Features Configuration Guide](../sources/S-NXDI.md)：NxD Inference 说明静态形状可执行程序通过分桶和 padding 承接不同长度，context encoding 与 token generation 可配置不同桶；持久状态与模型执行共同受配置约束。
  [完整原文](../../references/S-NXDI/reader.html) · [全文文本](../../references/S-NXDI/fulltext.txt)

## 延伸阅读

- [分页 KV：逻辑位置、物理布局与共享所有权](../algorithms/paged-kv-capacity.md)。
- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [MLA：压缩状态、矩阵吸收与解耦 RoPE](../models/mla.md)。
- [容量、算术强度与服务 Roofline](../hardware/memory-and-roofline.md)。
