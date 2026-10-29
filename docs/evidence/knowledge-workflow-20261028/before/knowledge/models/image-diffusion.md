# K-MODEL-009：图像扩散：迭代求解、条件复用与 CFG

扩散服务围绕不断变化的 latent 做多步求解；可复用对象由每步依赖决定，不能沿用自回归 KV 的全部规则。

## 状态和执行链

文本条件先编码，初始噪声 latent 进入多个时间步的去噪网络，scheduler 根据网络预测与当前 latent 计算下一 latent，最后由 VAE 解码。模型可能预测噪声、速度或其他参数化量；scheduler 的更新方程必须与预测类型和时间步设置相容。

去噪网络每一步的 self-attention 输入随 latent 改变，因此其历史 K/V 通常不能像 decoder 前缀一样跨步无条件复用。固定文本 encoder 输出可以复用；由固定文本和固定投影形成的 cross K/V 在结构允许时也可复用。任何跨步近似激活复用都引入另一个质量/成本方案。

## CFG 的计算形态

classifier-free guidance 的常见形式是 ε̂＝εᵤ＋w(ε𝒸−εᵤ)，其中有条件与无条件两条预测共享当前 latent。两条可沿 batch 维合并执行，减少调度开销却增加峰值激活；分开执行减小峰值却增加时间。某些模型采用蒸馏或不同 guidance 机制，不能按此公式给所有模型固定乘 2。

DiT 将 latent 切成 patch tokens。空间边长扩大使 token 数和 attention 工作快速增长；VAE 压缩、patch size、分辨率和 batch 共同决定内存。固定分辨率与固定步数的请求更易采用编译/图执行，混合分辨率与不同 scheduler 步数需要分桶或阶段调度。

## 步数、质量和缓存的共同目标

总耗时可近似为条件编码＋Σ各步去噪/更新＋VAE 解码。减少步数同时减少多次网络执行，却改变数值求解和质量；不能把它当成纯实现等价优化。seed、初始噪声、scheduler、步数、guidance 和解码设置一起定义比较条件。

输出是一张图而非 token 流，用户目标可能是整图延迟、图像吞吐、分辨率或质量。中间预览图的交付是附加服务语义，会增加解码与网络成本。VAE tiling 能降低峰值空间，但 tile overlap 与拼接也会影响边界质量，应从感受野解释其处理。

## Scheduler 是数值执行契约

以固定 Diffusers 源码中的 DDIM 为可检查例。记 aₜ 为累计 alpha，xₜ 为当前 latent，模型预测噪声 ε。无额外截断时，预测原样本为 x̂₀=(xₜ−√(1−aₜ)ε)/√aₜ；到所选前一时点 s 的更新为 xₛ=√aₛx̂₀+√(1−aₛ−σ²)ε+σz，其中 σ 由 eta 与 aₜ/aₛ 决定。s 属于 scheduler 的时间步安排，不能把循环序号或简单 t−1 直接代入。

若模型输出预测原样本或 v，源码采用不同换算；以 v 为例，x̂₀=√aₜxₜ−√(1−aₜ)v，ε=√aₜv+√(1−aₜ)xₜ。若应用 clipping 或 thresholding，是否重算 ε 也是明确定义的分支。该版本特别提醒动态 thresholding 不适用于其所述 latent-space 扩散情形。模型输出 shape 对得上，不代表 prediction_type、缩放、时间步与输出含义相容。

请求状态至少包括当前 latent、实际时间步位置、scheduler 配置、条件、guidance、随机生成器或每步噪声；使用多步求解器时，还需其历史预测。切换 batch 或暂停恢复必须按请求保留这些对象。让两个请求共享可变 scheduler 进度或随机流，可能使单请求参考通过而混合服务错误。

## 固定条件复用与跨步近似

固定文本与固定投影产生的 cross K/V 可以复用，但 timestep 调制、图像 latent self-attention 和随 latent 改变的中间激活通常不是固定对象。跨步复用 attention/FFN 输出、跳过部分 unconditional 计算或减少去噪步数，均需作为独立近似候选登记。

Wan 固定报告利用跨步 attention 与后期 CFG 相似性，按验证集选择缓存步并做补偿；其“lossless”措辞对应作者报告的评估条件，不构成数值恒等或所有 prompt 的质量保证。知识库在选择这类方法时，应保留未缓存参考、早/中/晚时间步、强 guidance、不同分辨率和稀有构图的质量对照，防止误差沿迭代累积。

CFG 两分支可以顺序运行、沿 batch 合并，或分设备执行后合成。分设备需要每步保持相同 latent 和时间条件，并传回合成所需预测；这不免除同步。DiT 的 tensor/context parallel 则切分单分支计算，两者的通信与内存不同。按 checkpoint 的条件方式检查是否确实存在可缓存 cross 投影：原始 DiT 论文同时研究 in-context、cross-attention、adaLN 等条件方式，不能给所有 DiT 假定同一 cross KV。

## 场景：24 GiB 预算的图像生成

以下为结构与时间设定。任务输出 1024×1024 图像，latent 空间压缩 8 倍、patch 边长 2，去噪 token 数为 64×64=4096；hidden 设 1536、16 heads、BF16。单份 hidden 为 12 MiB，显式完整 attention scores 为每 head 32 MiB、合计 512 MiB。分块 attention 可避免完整分数落存储，但其他激活、FFN、CFG 和 VAE 工作区仍需计算。

设权重常驻 8 GiB，单 CFG 分支峰值 workspace 为 6 GiB，设备可用 24 GiB。候选 A 顺序执行两分支，主项 14 GiB；B 合并两分支，按两份 workspace 的保守设定为 20 GiB。实际共享或额外 workspace 需测量。候选 C 在 B 上引入跨步缓存或更少求解步，属于质量可能变化的路径，不能与 A/B 的实现等价比较混在一起。

设条件编码 40 ms，单分支去噪 30 ms，合并双分支 45 ms，scheduler 每步 1 ms，30 步，最终 VAE 120 ms，则 A 为 40+30×61+120=1990 ms，B 为 40+30×46+120=1540 ms。数字仅用于复算选择：B 节省调用和更好聚批的假设若不成立，收益消失；若高分辨率或其他请求使 20 GiB 超预算，A 即使较慢也可能是合法选择。

验证固定初始 latent、条件、scheduler、guidance 和每步噪声，先比较逐步预测、更新后 latent 与最终图像；对 C 再用预先保留的 prompt/seed 集合评估质量、构图与分辨率，不挑选最好看的样本。首轮编译、稳态、排队和 VAE 分开报告。VAE 切块必须检查卷积 halo、归一化/attention 的全局依赖及拼接；只有覆盖全部依赖时才可主张等价，overlap 不能普遍保证无缝或无误差。

<!-- infraagents:provenance -->

## 来源与推导

- [S-LDM：High-Resolution Image Synthesis with Latent Diffusion Models](../sources/S-LDM.md)：Latent Diffusion 把去噪计算放在编码器形成的潜空间，通过条件机制引导生成，并最终解码到像素空间。
  [完整原文](../../references/S-LDM/reader.html) · [全文文本](../../references/S-LDM/fulltext.txt)

- [S-DIT：Scalable Diffusion Models with Transformers](../sources/S-DIT.md)：DiT 使用 Transformer 处理潜空间 patch，去噪计算的 token 几何由 latent 网格和 patch 化决定。
  [完整原文](../../references/S-DIT/reader.html) · [全文文本](../../references/S-DIT/fulltext.txt)

- [S-DIFFUSERS：Diffusers — Scheduler 与 DDIM（固定历史源码）](../sources/S-DIFFUSERS.md)：Diffusers 的固定 scheduler 源码定义 scheduler 配置、模型预测与时间步更新；DDIM 文件给出 latent 更新及 prediction_type 的不同分支。当前材料不把后续版本的默认参数当作通用规则。
  [算法与机制说明](../sources/S-DIFFUSERS.md)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。



### 本次扩写补充来源

- [S-WAN：Wan: Open and Advanced Large-Scale Video Generative Models](../sources/S-WAN.md)：Wan 的视频生成报告涉及时空建模和视频潜空间/解码结构；输出帧与压缩潜表示具有不同的时间和空间粒度。
  [完整原文](../../references/S-WAN/reader.html) · [全文文本](../../references/S-WAN/fulltext.txt)

## 延伸阅读

- [视频生成：时空 Token、全局耦合与分块解码](../models/video-generation.md)。
- [交叉注意力缓存：固定源、独立几何与可见性](../algorithms/cross-attention-cache.md)。
- [编译与动态形状：Guard、分桶和变体成本](../frameworks/compile-dynamic-shapes.md)。
- [Goodput：联合 SLO、拒绝和容量前沿](../workloads/slo-goodput.md)。
