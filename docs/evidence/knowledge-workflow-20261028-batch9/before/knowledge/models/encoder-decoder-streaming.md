# K-MODEL-005：Encoder-decoder：源表示、两套 Attention 与流式边界

Encoder-decoder 将源编码与目标生成分开：encoder 形成条件表示，decoder 根据条件和已知目标前缀预测下一个 token。固定源可以复用，目标 self KV 随已消费输入增长，流式源更新又可能使目标状态失效。正确实现需要同时回答：本轮实际消费哪个输入、各层 KV 对应什么、何时还能继续工作，以及什么事件才允许发布整请求终态。

## 源、目标与状态的对应关系

一次请求具有请求身份、执行代次、原始源 A、processor 与模型版本、任务条件和源时间区间。processor 将 A 转成特征 X∈R^{F×d_in}，同时给出有效位置；encoder 将其映射为 E∈R^{S×d_e}。F、S、原始采样点数和网络包数是不同单位。源有效位置还应保留到原始样本或文本区间的映射，不能用 encoder 行号直接充当音频偏移。

令非空任务前缀为 P＝(p_0,…,p_{m−1})，其中包括模型规定的 BOS、语言、任务和可选前文 token；生成 token 为 y_0,y_1,…。decoder 实际输入序列记为 z：i<m 时 z_i＝p_i，i≥m 时 z_i＝y_{i−m}。位置从 0 开始，消费输入 z_i 后得到 H_i，输出头的 logits_i 预测 z_{i+1}。因此最后一个 prompt 位置 H_{m−1} 预测第一个生成 token y_0。

用 n 表示所有 decoder 层已经完成计算的输入位置数。各层 self KV 的有效长度都是 n，对应 z_0,…,z_{n−1}。选择出 y_0 时，n 仍为 m；y_0 的位置 m 尚无 KV。下一次消费 y_0，才计算各层位置 m 的 KV，并得到预测 y_1 的 logits。生成 EOS 后可以直接结束目标生成，无须为了结束请求再消费 EOS。

请求保存 known＝P＋已经选择的生成 token、n、逐层 self KV、尚未消费的 known 后缀，以及最后有效的 logits 和它所预测的位置。完整消费 known 后，必须保留末行 logits，或采用明确的末位置重算路径；只有 KV 而没有 logits，不能凭空选择下一 token。本文路径选择保存 logits，消费最后一个 prompt 后立即具备选择 y_0 的后继步骤。

还需持久化采样随机状态、任务约束状态、源依赖版本、cross 可见范围、目标停止原因、processor/encoder 流式状态、源 flush 状态、暂定输出、稳定文本与全局样本提交偏移。已选择、已消费、已稳定和已发布是不同边界。

## 逐层 forward：KV 在哪里产生

下面选用推理态、post-norm、每层包含 self、cross 和 FFN 的代表性 Transformer 展开机制。它是可重建的结构示例，不规定 Whisper 等 pre-norm 模型必须采用相同归一化顺序。实际模型的 embedding 缩放、位置编码、norm 顺序、bias、激活和 head 分组均属于函数定义。

位置 i 的层 0 输入为 u_i^0＝γ Embed(z_i)＋Pos(i)。对 decoder 层 l：

1. 从该层输入计算 q_i＝u_i^{l−1}W_Qs,l、k_i＝u_i^{l−1}W_Ks,l、v_i＝u_i^{l−1}W_Vs,l，应用模型规定的位置变换。k_i、v_i 追加到该层 self 缓存的位置 i，各位置只追加一次。
2. 每个 self query head 在允许的目标位置 j≤i 上计算分数 s_j＝q_i·k_j/√d_k＋bias(i,j)。只对合法位置做稳定 softmax：先减去合法分数最大值，再取指数并归一化；与对应 values 加权求和。多头输出拼接，经 W_Os,l 投影后做残差与归一化，得到 x_i^l。
3. cross 模块的 K_c,l＝EW_Kc,l、V_c,l＝EW_Vc,l 由源产生；query 为 x_i^l W_Qc,l。在当前目标位置允许的源集合 M_{l,i} 内计算缩放点积、softmax 和 values 加权和，拼接各 head，经 W_Oc,l 投影后做残差与归一化，得到 r_i^l。
4. FFN(r)＝φ(rW_1,l＋b_1,l)W_2,l＋b_2,l；u_i^l＝Norm(r_i^l＋FFN(r_i^l))。该结果是下一层输入。最后一层输出经过模型规定的末端处理成为 H_i，输出头产生 logits_i。

因此高层位置 i 的 self KV 来自低层已经完成 self、cross 和 FFN 后的隐藏状态，并非全部层都从原始 embedding 投影。源变化即使不影响第一层 self KV，也可能影响高层 self KV。

GQA/MQA 中，一个 query head 按模型给定映射读取相应 KV head；query head 数与 KV head 数不能混用。cross 还可能有自己的位置偏置或位置变换。可见集合为空时，本文服务路径返回等待或明确失败，不执行没有归一化定义的空 softmax。某个 kernel 约定全屏蔽行输出零，只是该接口的数值约定，不能替代模型允许空条件解码的证明。

## 可见性、缓存身份与源更新

| 对象 | Q 来源 | K/V 来源 | 合法可见性与变化 |
|---|---|---|---|
| encoder self | encoder 当前层输入 | 同层源输入 | 可双向、分块或受 lookahead 限制，由模型定义 |
| decoder self | 对应 decoder 层输入 | 已消费目标输入 | 通常因果；缓存只覆盖已计算位置 |
| decoder cross | self 子层之后的 decoder 状态 | encoder 表示的模块专属投影 | 当前 query 可见哪些源位置由模型与任务定义 |

把三者全部设为 causal 会改变模型函数。源长与目标长独立，batch 中的源分段、目标分段、padding、位置和页表也要分别表达。batch 压紧后，请求到源段的映射必须一起更新；源张量没有改变不能证明运行视图仍然正确。

固定 E、投影权重和位置/表示契约时，cross K/V 不依赖目标步数，可以复用。如果可见集合随目标位置变化，可以在缓存覆盖的源上改变合法读取 mask，但不能沿用只包含旧集合的 softmax 输出或归一化量。

只有旧 E 真正不变、旧位置坐标不变、投影是逐行的且权重和表示契约不变时，追加源才允许保留旧 cross K/V、仅投影新增行并拼接。双向 encoder 重新编码增长中的音频可能改变旧行；此时长度增长不构成追加合法性的证据。换 processor、权重、dtype、量化 scale 或 head 布局也可能使缓存失效。

源产物身份至少绑定请求或合法共享身份、源内容版本、processor/encoder、源位置范围、cross 投影参数及物理表示。任务前缀另属于 decoder 状态身份；若它不影响 encoder 和源投影，不必把它强加到独立源缓存键中，但更换前缀不能继续复用不等价的目标 self KV。

换入新的 cross K/V 后，旧目标 self KV 和 logits 是否可继续使用，需要单独判断。离线完整新源语义通常要求从规定的目标起点重算；只对旧目标前缀做 replay，得到的是“新源条件下强制保留此前 token”的状态，并不证明这些 token 会由新源重新生成。允许暂定文本修订时，应丢弃受影响的暂定后缀，再重建目标状态。原生流式模型若明确定义既有目标状态可延续，则按该契约推进，不能把它称为与完整新源重算无条件等价。

## 源构建、输入消费与请求推进

源状态先通过分层空检查。原始空源不能先送入不支持空输入的 processor；特征全无效不能进入 encoder；encoder 输出全无效不能进入 cross 投影。静音包含实际采样点，通常也有有效特征，不能等同于空源。

```pseudocode
ALGORITHM BuildSourceState(A, meta, source_version)
输入：一个待构建的源视图、元数据、版本
输出：READY 源产物、EMPTY_SOURCE 或明确失败
1. 校验请求、代次、采样率/声道、源区间与版本；失败则返回 FAILED_META；
2. 若 A 的原始有效长度为 0：返回 EMPTY_SOURCE_RAW；
3. X, feature_valid, source_mapping ← processor(A, processor_state)；
4. 若 feature_valid 没有有效位置：返回 EMPTY_SOURCE_FEATURES；
5. E, encoder_valid, encoder_state ← encoder(X, feature_valid, encoder_state)；
6. 若 encoder_valid 没有有效源位置：返回 EMPTY_SOURCE_ENCODER；
7. 保留有效 E 与位置映射；S ← 有效源位置数；
8. 对每个实际 decoder cross 模块 c：
       K_c ← E W_Kc；V_c ← E W_Vc；
9. 将版本、范围、S、E、各模块 K/V 与表示身份绑定；
10. 发布 READY 产物；发布前不得让 decoder 读取部分投影。
```

该算法描述一个源视图的正常构建。网络收到零长度结束包时，应对已有 processor/encoder 状态执行结束处理，不能用 EMPTY_SOURCE 替换已有非空源。若整个请求最终没有有效源，按任务契约返回空结果或输入错误，不进入本文 cross 解码路径。

原始结束标志只表示不会再到达输入，不等于 flush 已完成。flush 先让重采样器与特征窗按模型契约处理残留，再让 encoder 处理最后未提交位置和所需右上下文，最后完成源投影与发布。尾部不足 lookahead 时，只能采用模型明确允许的 padding、边界 mask 或末块规则；没有允许路径则返回 FAILED_INSUFFICIENT_CONTEXT。填充值不算真实采样，提交偏移不超过实际收到的源末端。flush 也可能耗尽时间片，应保存其阶段和残留后继续，而非提前发布完成。

```pseudocode
ALGORITHM ConsumeKnownInput(req, src)
前提：i=req.n < length(req.known)，src 为合法 READY 产物
输出：位置 i 的各层 KV、末层 logits，或失败
1. i ← req.n；h ← 模型规定的 Embed(req.known[i]) 与位置 i 的组合；
2. 为本次位置建立未提交的逐层写入区；
3. 对 decoder 层 l 从低到高：
       q, k, v ← 从 h 做 self 投影并应用规定的位置变换；
       在第 l 层位置 i 暂写 k, v；不得重新追加位置 0..i−1；
       a ← 对历史 0..i 和当前 k/v 做合法 self attention；
       x ← 按模型顺序处理 self 输出投影、残差、norm；
       若该层有 cross：
           检查 M_{l,i} 非空且已就绪；
           c ← 用 x 的 query 与该模块 src.K/V 做 cross attention；
           r ← 按模型顺序处理 cross 输出投影、残差、norm；
       否则 r ← x；
       h ← 按模型定义执行 FFN、残差、norm，作为下一层输入；
4. logits ← OutputHead(末端处理(h))；
5. 全部层成功后，原子提交位置 i，req.n ← i＋1；
6. 若 req.n＝length(req.known)：保存 logits，标记其预测位置为 req.n；
   否则丢弃此中间位置的生成 logits，继续消费已知 prompt/强制前缀；
7. 失败时不推进 n；撤销暂写或标记请求失败，不暴露部分层长度。
```

实际实现可以一次处理多个未计算的 prompt 位置，但各层 attention mask 必须同时覆盖历史与新增位置，新增位置内部仍因果。上述逐位置写法用来明确数据依赖，并非要求每个位置单独启动 kernel。

初始化要求 m≥1、n＝0、known＝P、各层 self KV 为空、待选 logits 为空、生成计数 g＝0。max_new_tokens 只计生成 token，不计任务前缀；是否将 EOS 计入 usage 按协议规定，本文 g 将它作为一个已选择的生成 token 计入。

```pseudocode
ALGORITHM RunDecoderSlice(req, src, execution_budget)
输入：已初始化或可恢复的请求、源产物、调度时间片
输出：可恢复暂停、等待事件、带原因的终态或失败
1. 若取消或代次过期：停止未来提交，返回 CANCELLED；
2. 若源依赖不兼容：返回 NEED_REBUILD；按源更新规则重建后才能继续；
3. 重复：
   a. 若已有失败：保存原因并返回 FAILED；
   b. 若源尾部契约已被判为不可满足：返回 FAILED_INSUFFICIENT_CONTEXT；
   c. 若 g 已到 max_new_tokens 且尚无目标停止原因：设置 TARGET_LENGTH；
   d. 若已有合法目标停止原因：
          若源 flush 未完成：返回 WAIT_SOURCE_FINAL；
          若停止依据仍适用于最终源且尾部已处理：
              提交允许的最终后缀，发布 END_OF_REQUEST(目标停止原因)；
              返回；
          否则返回 NEED_REBUILD；
   e. 若当前操作缺少合法源视图或所需 lookahead：返回 WAIT_MORE_SOURCE；
   f. 若时间片已耗尽：保存进度，返回 PAUSED_BUDGET；
   g. 若 n < length(known)：
          ConsumeKnownInput(req, src)；计入本片工作；继续循环；
   h. 要求 n＝length(known)，且保存的 logits 预测位置为 n；
      若没有该 logits：返回 FAILED_MISSING_LOGITS；
   i. 按任务约束从 logits 选择下一个 token y；
      原子保存选择结果、随机状态与任务约束状态；
      将 y 追加到 known 和暂定输出；g ← g＋1；清除待选 logits；
      若 y 为合法 EOS：设置 TARGET_EOS；
      否则若 g＝max_new_tokens：设置 TARGET_LENGTH；
      按稳定规则检查可以提交的文本及源样本边界；计入本片工作；
      继续循环。
```

消费输入和选择输出是两个可恢复步骤。在最后一个 prompt 消费后暂停，恢复时先选择 y_0；在选择 y_0 后暂停，恢复时只消费 known 的位置 m。不得重新采样同一 logits，也不得把刚选择的 token 直接记作已有 KV。

PAUSED_BUDGET 等待的是下一次调度，现有源上仍可继续；WAIT_MORE_SOURCE 等待的是输入或源发布事件；WAIT_SOURCE_FINAL 表示当前目标已停，但整请求完成条件尚未成立。最后源块到达而目标仍可生成时，时间片耗尽只产生 PAUSED_BUDGET。长度终止是请求策略造成的截断，应保留 finish reason，不能冒充 EOS。

非最终源上的局部 EOS 不天然是整请求 EOS。如果服务允许每个源块产生局部结束，应将它绑定到当前源版本；下一源块到来后，按模型定义延续或清除受影响的暂定后缀并重建。模型没有这种定义时，应采用独立窗口任务，或等待完整源后生成。

## 提交、暂停与取消的不变量

所有成功提交边界必须保持：各层 self KV 长度等于 n；缓存位置 i 对应已消费输入 z_i；待消费位置从 n 开始；可选择的 logits 只预测其标记的位置且只选择一次；源投影、mask 与目标状态依赖相容。

同一请求的所有兼容源版本中，全局原始样本提交偏移必须单调不减，稳定文本必须以前次稳定文本为前缀。窗口内相对时间戳先换算为全局时间/样本坐标，再检查单调性；源版本增加不能重置提交偏移。版本过期、同版本偏移回退、新版本偏移回退和文本回退是四种应分别拒绝的情况。暂定内容可修订，已发布稳定内容的修正需要协议明确表达。

发布动作还需要请求、代次和源依赖检查。取消撤销后续调度与业务发布资格；在途 processor、encoder、kernel 或传输仍可能持有缓冲。代次可以拒绝旧结果提交，却不能阻止旧 DMA 写入已重用地址；资源必须等待最后使用事件后才回收，具体引用汇合见[取消与完成](../system-semantics/cancellation-resource-completion.md)。

## 成本：模块几何、工作量与实际流量

容量先按真实模块求和。源表示容量为 S d_e b_e。令 C 为 decoder 中实际 cross 模块集合，模块 c 的 K/V 宽度分别为 D_Kc、D_Vc，每元素字节分别为 b_Kc、b_Vc，则 cross 主体容量为 S Σ_c(D_Kc b_Kc＋D_Vc b_Vc)。它与 encoder 层数没有直接的份数关系。

令 J 为实际 decoder self 模块集合，长度 n 时 self 主体容量为 n Σ_l(D_Ks,l b_Ks,l＋D_Vs,l b_Vs,l)。常见等宽 K/V 下，D_self,l＝H_kv,self,l d_h,self,l，D_cross,c＝H_kv,cross,c d_h,cross,c。只有各模块同宽同精度时，才分别简写为 2L_self n D_self b_self 和 2|C|S D_cross b_cross。页尾、对齐、scale、页表和工作区另计。

设定 S＝1000、d_e＝1024、|C|＝12、D_cross＝1024，E 与 cross 都采用每元素 2 字节，则 E＝2,048,000 字节＝1.953125 MiB；cross＝49,152,000 字节＝46.875 MiB。100 条已编码请求的 cross 合计为 4.57763671875 GiB，只保存 E 为 195.3125 MiB。另设 L_self＝12、D_self＝1024、n＝100、每元素 2 字节，则每请求 self＝4,915,200 字节＝4.6875 MiB。这是同宽配置算例，不表示 self、cross 和 encoder 必须同宽同层数。

计算以一次乘加记一个 MAC。逐行 dense cross 源投影约需 S Σ_c d_e(D_Kc＋D_Vc) MAC；固定源重复使用 R 次时，逐次重投影支付 R 倍，缓存只支付一次。目标 query 投影、attention 和 FFN 仍需执行。

对单个模块，设 query head 数为 H_q，key 维度 d_k，value 维度 d_v，可见 key 数为 N。每个新增 query 的 QK 与 PV 合计约为 H_q N(d_k＋d_v) MAC，softmax 另有 O(H_q N) 标量工作。GQA/MQA 按 H_kv 存储，但 query 侧运算仍按 H_q 组织；减少 KV heads 不自动按同比例减少 attention 算术。

若固定 prompt 长度为 m，新增消费 G 个位置、完整目标历史都可见，则每层 self 的 QK/PV 项为 Θ((mG＋G²)H_q d_h)，这里假设 d_k＝d_v＝d_h。固定源的 cross 项为 Θ(GS H_q,cross d_h,cross)。这些是各模块 attention 项，不是完整 decoder 成本：self/cross 的 query 投影、self 新 K/V 投影、输出投影、FFN、归一化、输出词表头和采样均另计。prompt prefill 本身也有工作，不能只计后续 G 步。

encoder 的 dense 双向 attention 每层有 O(S²H_q d_h) 项，FFN 和投影另计；若物理执行包含 padding，应以实际计算长度而非 S_valid 估算。变长 encoder 可减少短源的无效计算，但不能据此推断质量或任意分块能力。

逻辑读取上，每个 decoder query 需要访问其可见 cross K/V；如果每步都从 HBM 取回完整缓存，R 步流量约为 R 倍 cross 字节。该式是所设读取路径，真实 DRAM 事务还受片上复用、head 共享、batch、布局和重复加载影响。主存换入 GPU 时，还要计主存读取、链路传输、设备写入及后续设备读取，不能只用缓存容量代替全路径流量。

encoder/decoder 分设备时，可传 E 后在 decoder 侧投影，也可在源侧投影后传全部 cross K/V。上述算例后者主体字节是前者的 24 倍；是否值得仍取决于投影算力、链路、复用次数和重叠。同步发布后消费者才能读取，未完成传输不能作为有效缓存。

准备成本包括重采样、特征窗历史、tokenization、任务前缀和时间戳映射。同步包括源就绪、投影发布、目标位置事务和稳定结果提交。驻留包括原始缓冲、特征、encoder 状态、E、cross/self KV、暂定输出与在途副本；共享物理页只计一次，但每个读者生命周期独立。阶段队列有界才能限制 encoder 领先造成的源产物积压。

## 条件变化怎样改变方案

固定源、较长目标通常增加缓存投影的价值；短目标可能不足以摊薄构建与管理开销。只存 E、提前投影全部 cross、decoder 准入后投影、按层重投影或换入，是不同的计算、流量与驻留选择。先计算投影节省，再检查它是否被缓存读取、迁移或并发下降抵消。

源每次更新都改变旧表示时，应放弃精确追加缓存；必要时重建 cross 和目标状态。原生稳定源前缀只解决源投影追加问题，不自动证明目标 self 状态可按完整新源语义继续。更换精度、位置规则或 backend 后，重新检查数值和布局身份。

更小源块减少采集等待，却增加调用、边界处理和不稳定输出；更大块减少启动次数，却增加首段等待。lookahead 是模型契约和质量条件，不是可任意删减的性能参数。重采样与特征窗跨包延续，不能每个网络包重新初始化。

Whisper 固定论文描述 16 kHz 音频、80 通道 log-magnitude Mel、25 ms 窗、10 ms 步长和 30 秒音频块；任务、语言及时间戳规则通过 decoder token 表达，时间戳相对当前窗口并以 20 ms 量化。长音频 buffered transcription 根据预测时间戳推进窗口，并可使用前文条件。这条路径需要恢复全局时间和处理跨窗口词，不证明 cross KV 可随任意新包追加。

Moonshine 的固定论文版本采用变长音频和 strides 为 64、3、2 的卷积 stem，以 RoPE 表达位置。论文报告短输入可减少固定 padding 工作，同时指出极短音频的重复输出及超过训练长度的质量限制；其中训练样本组装规则采用约 4 到 30 秒区间。变长输入、反复识别已采集窗口、按稳定块持续推进是不同能力，不能由名称相互推出。

## 场景：八路低延迟转写

以下全部为设计设定。八条会话输入 16 kHz、单声道、16-bit PCM，每条每 200 ms 产生 3200 个样本、6400 字节，总输入为 256000 字节/s。目标是在任务允许的稳定规则下，首个稳定片段距首个相关采样不超过 500 ms，并满足固定评测集上的质量门槛。

A 等完整 1 秒窗口再识别；B 每 200 ms 重跑增长窗口并允许修订；C 使用明确支持 200 ms 提交块、100 ms lookahead 的模型。A 的采集等待已超过目标。B 可早给暂定文字，却支付重复编码且不能把首次猜测当稳定提交。C 的合法性依赖模型与 processor，不能直接赋予 Whisper 或 Moonshine。

设 C 的排队/聚批为 25 ms、encoder 40 ms、decoder 35 ms、发布 10 ms，且这条首块路径依次发生，则 200＋100＋25＋40＋35＋10＝410 ms。八路每秒共 40 块，每块在同一串行资源占用 75 ms，则资源需求为 3 秒/s，队列无法长期稳定。若八块合批后整组执行 110 ms，每 200 ms 服务一组的该资源占用为 55%；实际相位、聚批等待、质量和长尾仍待测量。可重叠阶段应按依赖与共享资源重新计算，不能无条件相加或取最大值。

一条组合轨迹是：采集区间及 lookahead 到齐，processor 延续历史并生成特征；encoder 发布合法源版本；cross 投影完成后 decoder 获得读取权；消费任务前缀末位置，保存预测 y_0 的 logits；时间片结束，状态进入 PAUSED_BUDGET；下一次调度在同一源上选择 y_0，再消费它，逐层追加 KV；稳定规则满足后提交文本与全局样本偏移。最后收到结束包时，先完成尾部处理；目标若未结束则继续调度，直到合法 EOS 或明确长度终止后才发布请求终态。

若新源改变旧表示，重建发生在旧目标状态继续使用之前。若途中取消，停止发布，旧代次在途结果不得写入新请求；其缓冲仍等最后使用者结束。慢客户端会延长目标状态驻留，应通过输出额度反向限制生成，而非无界领先。

验证固定相同原始音频、标注与文字标准化，计算 WER＝(替换＋删除＋插入)/参考词数；适用语言另定义 CER。静音、极短指令、噪声、跨块词、长句和慢客户端共同覆盖边界，记录暂定修订、稳定延迟、最终质量、失败、队列和实时率。处理耗时/音频时长小于 1 只是平均实时率条件，不能证明首段稳定延迟达标。只要求整段最终结果时 A 可能更合算；允许修订且重复计算可承受时 B 有价值；稳定延迟严格时再验证 C 的模型与服务联合条件。

## 诊断与可反驳检查

| 观察 | 竞争解释 | 区分实验与修正方向 |
|---|---|---|
| prompt 后没有输出 | 末行 logits 丢失、未选择首 token、错误等待源 | 固定已就绪源，在最后 prompt 消费后暂停再恢复；要求先选择 y_0，缓存长度仍为 m |
| 恢复后文本重复或 KV 过长 | 完整前缀重复追加、同一 logits 重采样 | 比较连续执行与不同时间片切分的消费位置、选择次数、随机状态和最终输出 |
| 最后源包到达后过早完成 | 把源结束或预算耗尽当目标终止 | 在最后源已 flush、目标尚未 EOS 时耗尽预算；应暂停并能在同一源继续 |
| 新版本输出出现旧内容 | cross 复用错误、上层 self KV/logits 仍依赖旧源 | 分别重建 cross 与完整目标状态；只有后者恢复一致时，修正目标依赖判断 |
| 缓存减少投影但总时延不降 | 带宽、驻留下降、排队或传输主导 | 固定输入比较投影时间、流量、在途字节和队列；计算项下降不能单独支持整体收益 |
| 边界 WER 变差 | 上下文不足、时间戳映射或拼接错误 | 分别改变 lookahead 和全局拼接；单窗正确而全局错误优先检查提交映射 |

随章检查使用标准库 CPU 小模型：以非对称矩阵的字面投影结果和多源非均匀 softmax 解析式核对算术；保留旧 K/V、仅投影新增行并与完整重算比较；用独立的完整前缀计算路径核对简化两层 decoder 的增量状态；比较连续执行与时间片暂停恢复。它还分别检查原始空、特征全无效、encoder 输出全无效，以及跨兼容版本的提交单调性。

负对照包括错误 causal-cross、未来目标可见、修改源后沿用旧缓存、把已选择 token 提前计为已计算、旧 ledger 放行新版本偏移回退、空检查后置和预算耗尽后直接结束。故障产生的状态或事件必须送入与正常路径相同的不变量判定器并被拒绝，不能仅打印“故障函数返回了错误值”。

这些检查覆盖小模型算术、缓存数据流和调度状态关系。真实模型的位置/norm/量化实现、GPU 内存可见性、kernel、WER、负载时延和知识使用效果均需独立证据；本章没有报告相应实测。

<!-- infraagents:provenance -->

## 来源与推导

- [S-TRANSFORMER：Attention Is All You Need](../sources/S-TRANSFORMER.md)：Transformer 的 attention 用 query 与 keys 的分数生成归一化权重，再对 values 加权；encoder-decoder attention 的 Q 来自 decoder，K/V 来自 encoder 表示。§3.1–3.4 支持 decoder 子层、移位目标输入、因果 mask、FFN 与输出头的展开；本文缓存和恢复状态是在这些依赖上的推导。
  [完整原文](../../references/S-TRANSFORMER/reader.html) · [全文文本](../../references/S-TRANSFORMER/fulltext.txt)

- [S-WHISPER：Robust Speech Recognition via Large-Scale Weak Supervision](../sources/S-WHISPER.md)：Whisper 采用语音 encoder-decoder 和任务条件 token。音频编码与目标文本自回归是不同阶段，decoder 使用源表示完成条件生成。§2.1–2.3、§3.8 与 §4.5 支持音频处理、30 秒块、任务/时间戳格式和长音频窗口推进的说明，不能据此推出任意块的稳定源缓存。
  [完整原文](../../references/S-WHISPER/reader.html) · [全文文本](../../references/S-WHISPER/fulltext.txt)

- [S-MOONSHINE：Moonshine: Speech Recognition for Live Transcription and Voice Commands](../sources/S-MOONSHINE.md)：Moonshine 面向语音识别的 encoder-decoder 结构和变长音频处理，源音频表示与目标文本生成具有不同长度和计算阶段。固定论文 v2 的 §2–4 支持 padding、卷积 stem、RoPE、训练长度及极短/超训练长度音频的边界；论文版本 v2 不表示本文另行核验了后续同名产品。
  [完整原文](../../references/S-MOONSHINE/reader.html) · [全文文本](../../references/S-MOONSHINE/fulltext.txt)

正文中的公式展开、容量算例、请求状态机与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期不作为事实发布日期。来源论文的设备与质量结果不等于本库设备实验；本文 CPU 检查也不证明真实模型或服务性能。

## 延伸阅读

- [交叉注意力缓存：固定源、独立几何与可见性](../algorithms/cross-attention-cache.md)。
- [语音生成：Codec Token、声学块与播放时钟](../models/speech-generation.md)。
- [输入输出语义：Tokenizer、Chat Template 与流式文本](../system-semantics/tokenization-io.md)。
- [服务主机运行时：执行域、队列与缓冲所有权](../system-semantics/host-runtime.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
