# K-ALG-010：交叉注意力缓存：固定源、独立几何与可见性

当 encoder 输出和投影权重固定时，cross-attention 的 K/V 与 decoder 步数无关，可以一次生成、多次读取。

## 从依赖关系推导复用

设固定源表示 E 为 S×dₑ，decoder 第 t 步状态 Hₜ 为 Q×d𝒹。某层 cross-attention 的 K＝EWₖ、V＝EWᵥ、Qₜ＝HₜWq。只要 E、Wₖ、Wᵥ 不变，K/V 不依赖 t。缓存因此消除重复源投影；每步的 QₜKᵀ、softmax 和加权求和仍要执行。

若输出 T 步，每层源投影总成本从约 T 倍降到 1 倍，而 attention 读源的成本仍随 TS 增长。缓存所有 cross 层的 K/V 可能大于只缓存 E：层数 L、KV heads H、维度 d、每元素 b 字节时约 2LSHdb。源很短或内存紧张时，只保留 E 并重投影是一种明确的空间换计算选择。

## 两套状态的几何独立

| 属性 | Decoder self-attention | Cross-attention |
|---|---|---|
| K/V 来源 | 已生成 decoder 前缀 | encoder 或其他源表示 |
| 时间变化 | 通常追加 | 固定源时只读，流式源可增长或替换 |
| 长度 | decoder 已计算长度 | 当前可见源长度 |
| head 与通道 | decoder self 配置 | 对应 cross 模块自己的配置 |
| 位置与 mask | 由 self 模块定义 | 由 cross 模块和源可见性定义 |

通常的离线 encoder-decoder 可以让一个 decoder query 看完整 encoder 输出；流式、块对齐、多图像可见性可能需要自定义 mask。cross-attention 也可能有位置编码或相对位置偏置。因而不能将“非因果且绝不使用 RoPE”作为通用定义；正确依据是模型的 forward 语义。

## 分页批处理如何表达

不同请求的 query 段长与源段长分别编码。例如 A 一行 query 读 100 个源 token，B 三行 query 读 40 个源 token，query 偏移与源页表拥有不同分段。输出仍按 query 行归属。两路 encoder 可各用独立池；几何相同的池可以共享分配器，但内容身份与所有权仍必须按源隔离。

源投影写入只发生在源版本变化时。运行 attention 的阶段只读已完成的源缓存。decoder batch 压紧会改变请求到源页表的选择，所以“源本身没变”不等于运行计划永不变化。计划缓存键至少覆盖 kernel 所特化的几何、batch 分段、源布局与相关版本；图兼容也要求 workspace、metadata 地址和实际 backend 契约成立。

## 流式源为何更难

新增音频可能让双向 encoder 重新解释之前的帧，旧 E 不一定是新 E 的稳定前缀。只有模型保证已经提交的源块不再变化，才能安全地只追加 cross KV。否则应以源版本区分缓存并让 decoder 采用定义明确的更新语义。这个差别决定了一个离线识别模型能否直接包装成真正的低延迟流式服务。

## 源产物在阶段间排队时也占容量

源表示 E、各层 cross KV、目标 self KV 的持有时间不同：E 在 encoder 完成后即可出现，cross 投影可能等到 decoder 准入才构建，目标 self KV 则随生成增长。若 encoder 无限制领先，排队中尚未生成文本的请求也会占满特征或 cross 缓存。阶段间队列应按产物 bytes 和等待时限准入，不能只限制活跃 decoder 数。

设 S=1000、dₑ=1024、decoder cross 层 L=12、每层 KV hidden=1024、BF16，则 E 为 1.953125 MiB，全部 cross KV 为 46.875 MiB。100 条已编码待解码请求若全部提前投影，主项为 4.57763671875 GiB；只保存 E 为约 195.31 MiB，但后续支付投影时间。该设定说明保留 E、提前投影、逐层构建或分层存储需要结合队列与复用次数比较。

## 分层驻留与 backend 转换

固定源 context 不随 decoder 步数改变，适合只读共享和预取；但每个目标步仍会读取它。若同一请求每步都把全部 cross KV 从 DRAM 取回，流量为该状态 bytes×目标步数，首次投影节约可能被传输覆盖。可比较 GPU 保留完整 cross KV、只保留 E 并按层投影、保留在主存按层换入，以及按任务放置 encoder/decoder。选择取决于源长度、目标长度、带宽、并发和空闲间隔。

不同 backend 可能用不同 head 顺序、页长、dtype、量化分组或 plan/workspace。迁移的对象身份仍由源内容/版本、processor、encoder、cross 投影权重及其精度定义，不能只依据“同一段音频”。转换是生成另一份可验证的物理表示；目标完成前，源表示和在途读者需要保持有效。

## 源更新和取消的组合验证

将源版本 v 更新到 v+1 时，先判断旧 E 的哪些部分真的稳定。双向重编码可以改动旧位置；即使 source token 数只增加，也不意味着 cross KV 仅追加。目标 decoder 之前产生的 self 状态可能已依赖旧源，换入新 cross KV 后继续原 self KV 只有在服务/模型契约明确允许时才成立；否则要从规定的目标起点重新计算，并区分暂定输出与已交付输出。

验证从相同源完整生成开始，分别在源编码后、投影中、decoder 首步后和迁移中取消；再在目标处理中追加或替换源、切换 backend、复用物理槽。检查源与目标版本匹配、引用只释放一次、旧投影迟到不会覆盖新请求、已交付文本是否需要修订。性能比较含排队、投影、传输、转换和首次正确继续步骤，而不只测 attention kernel。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TRANSFORMER：Attention Is All You Need](../sources/S-TRANSFORMER.md)：Transformer 的 attention 用 query 与 keys 的分数生成归一化权重，再对 values 加权；encoder-decoder attention 的 Q 来自 decoder，K/V 来自 encoder 表示。
  [完整原文](../../references/S-TRANSFORMER/reader.html) · [全文文本](../../references/S-TRANSFORMER/fulltext.txt)

- [S-WHISPER：Robust Speech Recognition via Large-Scale Weak Supervision](../sources/S-WHISPER.md)：Whisper 采用语音 encoder-decoder 和任务条件 token。音频编码与目标文本自回归是不同阶段，decoder 使用源表示完成条件生成。
  [完整原文](../../references/S-WHISPER/reader.html) · [全文文本](../../references/S-WHISPER/fulltext.txt)

- [S-FI：FlashInfer：Attention kernels](../sources/S-FI.md)：截止日前的 FlashInfer 固定源码区分 query 分段、KV 页表及末页长度。相关 wrapper 的 plan 安排计算与工作区，run 执行 attention；plan 的捕获限制与图模式缓冲须按该版本接口核对。
  [算法与机制说明](../sources/S-FI.md)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [Encoder-decoder：源表示、两套 Attention 与流式边界](../models/encoder-decoder-streaming.md)。
- [分页 KV：逻辑位置、物理布局与共享所有权](../algorithms/paged-kv-capacity.md)。
- [Attention Backend：语义、布局与执行契约的交集](../frameworks/attention-backends.md)。
- [视觉语言模型：Patch、模态桥接与联合位置](../models/vision-language.md)。
