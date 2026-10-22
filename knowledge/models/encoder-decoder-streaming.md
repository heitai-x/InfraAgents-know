# K-MODEL-005：Encoder-decoder：源表示、两套 Attention 与流式边界

Encoder-decoder 将源编码和目标生成分开；固定源复用、流式追加和目标自回归具有不同的依赖结构。

## 三段计算

输入首先经过 modality processor 与 encoder，得到源表示 E。decoder 处理目标提示与生成前缀：self-attention 读取目标历史，cross-attention 读取 E 或其各层 K/V 投影。最后输出头和任务 token 规则决定文本、时间戳或其他符号。

encoder 可以是双向，decoder self-attention 通常因果，cross 可见性由任务定义。这使“所有 attention 都用 causal”成为结构性错误。源长度 S 与目标长度 T 独立，batch 中的 padding/mask 也应分别表达。

## 一次编码为何还不够描述缓存

缓存 E 可以避免重跑 encoder，但如果每个 decoder 步仍重复计算 EWₖ、EWᵥ，源投影成本仍存在。保存每层 cross KV 则减少投影，增加约 L 倍的状态规模。源编码时间、cross 投影时间、目标 self KV、cross KV 容量四项应分别建模。

不同请求源长差异大时，encoder 适合按长度或分辨率聚批；decode 适合连续批。让两段共享完全相同的 batch 队列可能引入队头阻塞。可以将源表示作为阶段间产物，但必须附上请求身份、源版本、有效长度和模态信息。

## 离线切块与真正流式

离线模型对完整输入编码后生成；将长音频分成独立小段，是另一种任务处理策略，会改变上下文和边界。真正流式模型通常定义可提交块、lookahead 和状态延续。若 encoder 是双向，新帧可能改变旧帧表示，先前 cross KV 就不能视为不可变前缀。

例如一段 20 秒音频被切成两个 10 秒窗口，重复上下文、时间戳偏移、说话者连续性和拼接规则都进入输出语义。每块识别结果正确不自动保证拼接后文本和时间对齐正确。模型提供的 streaming 或 chunked attention 规则决定可以保留哪些状态。

## 服务目标从用户时间尺度推导

ASR 可以关注最终转写延迟、首个稳定文本延迟、实时率和词错误率。实时率是处理耗时/音频时长，低于 1 说明平均可赶上输入，却不保证首段延迟小。若模型等待固定大窗口才编码，即使计算很快也可能交互迟钝。调度应区分采集等待、编码、解码和稳定结果发布。

## 从声音采集到稳定文字的提交链

一次 ASR 请求应保存音频采样率、声道、源时间区间、特征处理器版本、encoder 可见范围、文本提示/语言/任务 token，以及已经对外提交的文字或时间戳边界。重采样和特征窗口有自己的历史与边缘处理；每个网络包独立重置处理器会破坏连续音频，即使包内样本合法。

encoder 阶段可按音频长度聚批，decoder 按目标已计算位置推进，发布阶段则判断文字是暂定还是稳定。三者具有不同 deadline。长录音的离线 encoder 不应无界阻塞短语音命令，decode 也不能因为一个源尚未准备好而让所有就绪请求等待。可以给两阶段独立有界队列，再由源产物和 decoder 状态的容量共同限流。

Whisper 的固定版本将转写/翻译、语言、是否时间戳等编码进任务 token，时间戳相对当前音频窗口；拼接窗口时需要恢复全局时间。Moonshine v2 的变长编码减少固定 padding 需求，但论文没有据此给出任意 chunk 的不可变源表示承诺。它对极短音频与超出训练长度的音频还报告了不同质量限制。变长输入能力、反复识别已采集音频与真正按稳定块延续，是三件需要分别验证的能力。

## 场景：八路低延迟转写

以下为设计设定。八条会话输入 16 kHz、单声道、16-bit PCM，每条每 200 ms 产生 3200 个样本，即 6400 字节；总输入为每秒 256000 字节。目标是在允许的文字稳定规则下，首个稳定片段于首个相关采样后 500 ms 内发布，同时在固定评测集上满足任务定义的 WER 门槛。输入网络字节很小，瓶颈仍可能在编码、目标解码和排队。

候选 A 等待完整 1 秒窗口后识别；候选 B 每 200 ms 重跑当前音频窗口并允许修订暂定文字；候选 C 使用明确支持 200 ms 可提交块、100 ms lookahead 的模型执行增量处理。A 的等待下界已经超过 500 ms，不能用低实时率弥补。B 可较早展示文字，但不能把首次猜测时间记作稳定提交时间，且重复编码成本随窗口增长。C 的合法性依赖模型与处理器契约，不能把这种块大小直接赋予 Whisper 或 Moonshine。

对 C，再设排队/聚批合计 25 ms、encoder 40 ms、decoder 35 ms、发布 10 ms，则首块完成的设定时序为 200+100+25+40+35+10=410 ms。该和式只是所设路径，没有证明真实设备达到目标。若八路每秒各产生 5 块，总计 40 块/s，逐块 encoder+decoder 串行占用 75 ms，则同一串行资源每秒需求 3 秒，队列必持续增长。若允许将八块聚批并把该组执行降为 110 ms，则每 200 ms 服务一组的等效占用为 55%，但实际到达相位、聚批等待和长尾仍需测量。

容量同时包含采集环形缓冲、特征、encoder 状态、源 E/cross KV、目标 self KV 和待发布文本。每块应携带会话、源区间与代次；断线恢复、重复包和迟到块不能重复推进。B 的源版本变化可能使已生成目标状态失效，C 的稳定源也必须按模型要求保留 lookback。取消时各阶段共同结束，已发布稳定文本保留明确终态。

验证以相同原始音频和标注比较三方案，固定文字标准化规则后计算 WER=(替换+删除+插入)/参考词数；对适用语言另定义 CER。加入静音、极短指令、噪声、说话速度、跨块词和长句，分别统计最终正确性、暂定文字修订、稳定片段延迟、音频丢失、队列与实时率。不能删去错误或超时片段后再报告平均 WER/时延。

若任务只要求整段录音最终结果，A 的简单批处理可能更合算；若稳定延迟严格且质量允许的原生流式模型可用，C 才有明确优势；若 B 的重复计算可接受且产品允许修订，它可能是有效中间选择。选择由稳定性承诺和负载共同决定，而不是由是否逐块返回 JSON 决定。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TRANSFORMER：Attention Is All You Need](../sources/S-TRANSFORMER.md)：Transformer 的 attention 用 query 与 keys 的分数生成归一化权重，再对 values 加权；encoder-decoder attention 的 Q 来自 decoder，K/V 来自 encoder 表示。
  [完整原文](../../references/S-TRANSFORMER/reader.html) · [全文文本](../../references/S-TRANSFORMER/fulltext.txt)

- [S-WHISPER：Robust Speech Recognition via Large-Scale Weak Supervision](../sources/S-WHISPER.md)：Whisper 采用语音 encoder-decoder 和任务条件 token。音频编码与目标文本自回归是不同阶段，decoder 使用源表示完成条件生成。
  [完整原文](../../references/S-WHISPER/reader.html) · [全文文本](../../references/S-WHISPER/fulltext.txt)

- [S-MOONSHINE：Moonshine: Speech Recognition for Live Transcription and Voice Commands](../sources/S-MOONSHINE.md)：Moonshine 面向语音识别的 encoder-decoder 结构和变长音频处理，源音频表示与目标文本生成具有不同长度和计算阶段。
  [完整原文](../../references/S-MOONSHINE/reader.html) · [全文文本](../../references/S-MOONSHINE/fulltext.txt)

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [交叉注意力缓存：固定源、独立几何与可见性](../algorithms/cross-attention-cache.md)。
- [语音生成：Codec Token、声学块与播放时钟](../models/speech-generation.md)。
- [输入输出语义：Tokenizer、Chat Template 与流式文本](../system-semantics/tokenization-io.md)。
