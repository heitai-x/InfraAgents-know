# K-MODEL-010：视频生成：时空 Token、全局耦合与分块解码

视频生成把时间加入 latent 几何，时间压缩、时空 attention 和输出解码共同决定可服务规模。

## 时间轴为什么迅速放大成本

若压缩后的 latent 网格为 T′×H′×W′，patch 后 token 数约为这三个轴各自分块数的乘积。密集时空 attention 的成对工作随 token 数平方增长。视频长度翻倍，在其他条件不变时可能使这部分工作接近四倍；采用因子化、局部或稀疏 attention 的模型则有不同增长规律。

时空 VAE 的时间压缩使输出帧数与 latent 帧数不同，首尾边界也可能有特殊规则。帧率决定播放时间，分辨率决定像素规模，二者不能与模型生成的 latent 步数混淆。视频请求的资源单位应包含帧数、帧率、空间尺寸、去噪步数和条件输入。

## 计算分块与语义分块

按 tile 计算 attention 可以保持精确函数；把视频切成独立片段生成通常会改变时间上下文和一致性。若模型本身是因果或有明确窗口，可按其规则流式推进；若是全序列去噪，早期帧在后续去噪步仍会变化，不能把它们当已提交前缀直接释放。

VAE 分块解码降低峰值内存，但时间卷积需要前后文或状态。重复 overlap、边界融合与最终裁剪关系决定每帧只出现一次且时间顺序正确。视频编码器还会带来 CPU/GPU 编码、色彩空间和容器封装的后处理成本。

## 多卡和服务调度

长序列可以采用 context/sequence parallel，代价是时空特征或 attention 统计的通信。分辨率和帧数决定的峰值往往比请求数更适合做 admission control。不同去噪步数的请求可以在迭代边界重组，但时间步 embedding、guidance、latent 形状和随机轨迹必须按请求保存。

端到端延迟包含条件编码、所有去噪步、VAE 和视频封装。仅测一个 DiT forward 无法预测整段视频交付时间。质量目标还包括时间一致性和运动，不应从单帧图像得分直接推断。

## 时间压缩不能忽略首帧规则

Wan 固定报告的 VAE 将 (1+T) 帧映为 (1+T/4) 个时间 latent，空间每边压缩 8 倍、通道 16；首帧只做空间压缩。DiT 再按 (1,2,2) patch 化。因此 81 帧、720×1280 的设定输入有 21×90×160 latent 网格与 21×45×80=75600 个 DiT token。不能把 81/4 的非整数直接四舍五入而不遵守首帧和 padding 规则。

该 VAE 的因果卷积以历史 feature cache 连接 chunk，普通 kernel=3 的时间卷积与 stride=2 下采样采用不同保留规则。每层 feature cache、有效时间范围和首次 padding 都是状态；各 chunk 独立零填充会改动边界。VAE 因果不意味着全序列 DiT 已变成因果生成，未完成去噪的前部帧仍可能变化。

## 时空切分的两种通信路径

沿 token 轴切分时，每 rank 可保留局部 Q，轮流获得其他片的 K/V 并合并在线 softmax 统计；或者把 head 与 sequence 轴重新分布，用 all-to-all 改变局部布局。前者关注块传输与计算重叠，后者关注全组重排、head 整除和网络交换。还可以结合权重分片，但每层权重 gather 与 attention 交换会争用链路及 HBM。

对按时间或空间切出的逻辑区间，位置坐标仍来自原完整网格。把局部分片从零重新编号，会改变时空位置。完整 attention 的 query 仍需访问全局合法 key 集合，不能仅计算本片内部来冒充精确 CP。分片选择与物理拓扑共同决定跨节点字节，集合通信启动次数和最慢 rank 决定短片的收益。

## 场景：长视频跨四卡并完整交付

延续前述 81 帧、720×1280 的结构，假设 hidden=1536、BF16、四卡 context parallel，去噪步数固定，质量目标与完整上下文参考一致；hidden 和设备数据不是声称某一 Wan 型号的全部配置。单层完整 hidden 为约 221.48 MiB；显式 score 矩阵仅单 head 就约 10.65 GiB。因此避免完整 scores、核对 QKV/FFN 暂存和考虑分片是不同的资源措施。

候选 A 单卡精确分块 attention，保留全上下文；B 四卡分片并传输远端 K/V；C 独立生成多个短片后拼接。A 避免网络但可能超过计算/容量预算，B 减少每卡工作却增加通信，C 改变时间耦合，只有任务允许此类质量变化才可参与同一设计决策。

若 B 按 4 个等长 token 片切分，各 rank 18900 token，单层完整 K+V 主项为 442.97 MiB。每 rank 为看见其余三片需接收约 332.23 MiB，发送量另计；环形逐片交换可以只保留当前/下一块，而非一次物化全部接收量。假设有效链路 25 GiB/s，纯接收时间下界约 13 ms/层，另有启动、共享链路及合并。去噪多步和多层重复该过程，不能只报一次交换时间；计算可否覆盖它需要按实际 overlap 校准。

最终 latent 进入 VAE chunk 解码，再经颜色/像素转换、视频编码、封装与发送。解码和视频编码可有界流水；若编码落后，不能无限保留高精度帧。81 帧在 16 fps 下的首末帧时间戳跨度为 5 秒，而容器按每帧完整展示区间计算的持续时间可为 81/16 秒；帧数、时间基和最终时长应按交付协议明确，不能混用来报告 FPS。

组合约束包括每步所有分片使用同一时间条件与随机轨迹、CFG 两分支匹配、位置未重置、VAE feature cache 连续、每帧仅发布一次。故障恢复还要保留当前 latent/scheduler 步骤，或者从已知起点重算；部分 rank 的旧步结果不能与另一 rank 的新步拼接。

验证先比较小网格分片前后的逐层/逐步张量，再比较完整视频的帧数、时间戳、首尾与 chunk 边界。质量同时评估指令符合、对象身份、运动、时间一致性与帧间伪影，不能只测每帧清晰度。性能覆盖条件编码、全部去噪、VAE、封装和网络，报告慢帧和峰值，而不拿单个 DiT forward 代替完整交付。

短视频或慢网络下，A 可能快于 B；长序列计算足以隐藏通信且拓扑合适时，B 才更有机会；独立镜头或允许片段化任务可考虑 C，但需另测拼接与跨片一致性。混合分辨率服务应按 token 几何、步数和峰值预算聚批，既限制超大任务占用，也避免小任务无期限等待大视频结束。

<!-- infraagents:provenance -->

## 来源与推导

- [S-WAN：Wan: Open and Advanced Large-Scale Video Generative Models](../sources/S-WAN.md)：Wan 的视频生成报告涉及时空建模和视频潜空间/解码结构；输出帧与压缩潜表示具有不同的时间和空间粒度。
  [完整原文](../../references/S-WAN/reader.html) · [全文文本](../../references/S-WAN/fulltext.txt)

- [S-DIT：Scalable Diffusion Models with Transformers](../sources/S-DIT.md)：DiT 使用 Transformer 处理潜空间 patch，去噪计算的 token 几何由 latent 网格和 patch 化决定。
  [完整原文](../../references/S-DIT/reader.html) · [全文文本](../../references/S-DIT/fulltext.txt)

- [S-DIFFUSERS：Diffusers — Scheduler 与 DDIM（固定历史源码）](../sources/S-DIFFUSERS.md)：Diffusers 的固定 scheduler 源码定义 scheduler 配置、模型预测与时间步更新；DDIM 文件给出 latent 更新及 prediction_type 的不同分支。当前材料不把后续版本的默认参数当作通用规则。
  [完整原文](../../references/S-DIFFUSERS/reader.html) · [全文文本](../../references/S-DIFFUSERS/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [图像扩散：迭代求解、条件复用与 CFG](../models/image-diffusion.md)。
- [推理并行：张量分解、流水、专家与上下文](../algorithms/parallelism.md)。
- [容量、算术强度与服务 Roofline](../hardware/memory-and-roofline.md)。
