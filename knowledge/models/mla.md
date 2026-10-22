# K-MODEL-006：MLA：压缩状态、矩阵吸收与解耦 RoPE

MLA 通过缓存低维 latent 改变 KV 的表示；真正节省 decode 流量需要执行内核沿压缩表示计算。

## 压缩的对象

对 hidden state h，先得到低维 c＝Wᴰh，再由上投影产生内容 K＝Wᴷc、V＝Wⱽc。若每 token 缓存 c 而非展开后的所有 heads，状态由约 2Hd 个元素变成 d𝒸 个元素。DeepSeek-V2 还保留独立的 RoPE key 分量，因此总量按 d𝒸＋dᴿ 计算。

这与 GQA 不同：GQA 共享 KV heads，MLA 共享一份低维联合表示并通过不同投影恢复各 head 内容。用 Hkv 和普通 head_dim 强行描述 latent 页容易算错容量和 attention scale。

## 为什么可以不展开历史 K/V

单个内容分数 qᵀWᴷc 可以重排为 ((Wᴷ)ᵀq)ᵀc，即把上投影吸收到 query 侧。值聚合 Σⱼ aⱼWⱽcⱼ＝WⱽΣⱼ aⱼcⱼ，则可以先在 latent 空间归约，再做输出相关投影。这样避免每步将全部历史状态展开成大 K/V。

这种重排依赖线性关系。位置相关旋转若插在 q 和 K 上，会让投影与旋转矩阵的次序无法任意交换。DeepSeek-V2 将位置分量与内容分量解耦，使内容路径保留吸收性质，位置路径单独计算。将整个 latent 当普通 RoPE head 旋转并不等价。

## Prefill 与 Decode 可以采用不同实现

prefill 有较多 query，展开后的 attention 可能更适合现有高效矩阵 kernel；decode 更关心历史状态读取，沿 latent 执行更有价值。两阶段可以使用不同物理工作区，但对外持久状态格式与交接必须清楚。若 prefill 只留下展开 K/V，decode 期望 latent，则需要转换或同时保存，后者会改变容量收益。

算例中 L＝32、普通 H＝32、d＝128、b＝2，普通 KV/token 为 512 KiB；若 d𝒸＝512、dᴿ＝64，则 latent 路径为 36 KiB/token。比值来自所设结构，不是特定已部署模型的性能。实际还要计 padding、量化 scales 和分片复制。

## 验证吸收是否落实

可从驻留对象形状判断是否只保存 latent，从每步内存流量判断是否临时展开全部历史，从固定输入的 attention 输出判断代数重排是否正确。scale、RoPE 通道、head 输出组合和分片都参与等价关系。仅把缓存类命名为 MLA，无法保证压缩路径已经贯穿执行。

<!-- infraagents:provenance -->

## 来源与推导

- S-MLA：[完整原文](../../references/S-MLA/reader.html) · [全文文本](../../references/S-MLA/fulltext.txt)
- S-FI：[完整原文](../../references/S-FI/reader.html) · [全文文本](../../references/S-FI/fulltext.txt)
- S-TORCH-NUMERIC：[完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

- [S-MLA：DeepSeek-V2：MLA 与解耦位置编码](../sources/S-MLA.md)：DeepSeek-V2 的 MLA 以低维 latent 联合表示内容 K/V，通过线性投影的吸收减少历史展开；位置相关旋转不能任意与投影交换，因此采用解耦 RoPE 分量。

- [S-FI：FlashInfer：Attention kernels](../sources/S-FI.md)：已读 FlashInfer 0.6.18 接口说明区分 query 分段、KV 页表及末页长度。一些 wrapper 的 plan 安排计算/工作区，run 执行 attention；相关 plan 不能直接捕获入图。

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 3，2026-09-20。

## 延伸阅读

- [MHA、GQA 与 MQA：头映射和带宽复用](../models/gqa-kv-heads.md)。
- [Attention 的 IO、在线 softmax 与分块归约](../algorithms/attention-io.md)。
- [异构状态缓存：层组、恢复位置与联合分配](../algorithms/heterogeneous-state-cache.md)。
