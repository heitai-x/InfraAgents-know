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

<!-- infraagents:provenance -->

## 来源与推导

- S-LDM：[完整原文](../../references/S-LDM/reader.html) · [全文文本](../../references/S-LDM/fulltext.txt)
- S-DIT：[完整原文](../../references/S-DIT/reader.html) · [全文文本](../../references/S-DIT/fulltext.txt)
- S-DIFFUSERS：[完整原文](../../references/S-DIFFUSERS/reader.html) · [全文文本](../../references/S-DIFFUSERS/fulltext.txt)

- [S-LDM：High-Resolution Image Synthesis with Latent Diffusion Models](../sources/S-LDM.md)：Latent Diffusion 把去噪计算放在编码器形成的潜空间，通过条件机制引导生成，并最终解码到像素空间。

- [S-DIT：Scalable Diffusion Models with Transformers](../sources/S-DIT.md)：DiT 使用 Transformer 处理潜空间 patch，去噪计算的 token 几何由 latent 网格和 patch 化决定。

- [S-DIFFUSERS：Diffusers：Schedulers](../sources/S-DIFFUSERS.md)：Diffusers 的 scheduler 说明连接模型预测、当前 latent 和时间步更新；不同预测参数化与 scheduler 配置需要相容。

上文的公式展开、容量算例与组合设计是 InfraAgents 在这些机制基础上的推导，具体推导已写在本条正文中。来源入口均为本地释义，可继续读取其定义与使用关系。版本 3，2026-09-20。

## 延伸阅读

- [视频生成：时空 Token、全局耦合与分块解码](../models/video-generation.md)。
- [交叉注意力缓存：固定源、独立几何与可见性](../algorithms/cross-attention-cache.md)。
- [编译与动态形状：Guard、分桶和变体成本](../frameworks/compile-dynamic-shapes.md)。
- [Goodput：联合 SLO、拒绝和容量前沿](../workloads/slo-goodput.md)。
