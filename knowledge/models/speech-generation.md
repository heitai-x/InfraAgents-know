# K-MODEL-011：语音生成：Codec Token、声学块与播放时钟

语音服务需要把模型生成时钟接到连续播放时钟上。已经选择 codec token，不表示该 token 自身的自回归状态已经计算，也不表示声学块已经可解码、波形已经稳定或客户端已经听到。平均 token/s 和音频秒/秒不能独立说明首音频、断音、打断残留和尾部完整性。

本章区分固定报告支持的模型事实、用于恢复数据流的参考网络，以及服务侧的提交与播放模型。参考网络不是 Qwen2.5-Omni checkpoint 的逐层复刻；逻辑事务也不等于 GPU 同步实现或音质实测。

## 生成对象、身份与样本边界

设采样率为 f_s Hz，声学帧率为 f_a 帧/s，每块包含 N_f 个声学帧，必需 codebook 数为 C。声学 token 写作 x_{c,j}，c 为 codebook，j 为声学帧；文本 track 的位置不计为声学帧。稳态块 b 的帧区间为 [bN_f,(b+1)N_f)，名义时长为 D=N_f/f_a。

当 f_s/f_a 非整数时，不能将每块长度独立取整后累加。定义

s_b = floor(bN_f f_s/f_a)。

完整块的逻辑样本区间为 [s_b,s_{b+1})。EOS 已知后，另保存最终真实帧数 F≥0、最终样本数 S_F=floor(F f_s/f_a)、最终块数 N=ceil(F/N_f)；F=0 时 N=0。真实块 b 的末端为

e_b = floor(min((b+1)N_f,F) f_s/f_a)。

EOS 未知时只封闭完整块，使用 e_b=s_{b+1}；EOS 已知后最后一个块可以只有 F mod N_f 帧。填充帧不计入 F，右侧 halo 和重复 overlap 不计入交付样本。这是参考几何；实际 codec 若规定另一种有效长度映射，应整体替换边界函数，不能混用两套映射。此完整块事件接口要求非空真实块对应正样本数；否则需先合并零样本区间或报告不支持的几何。

必须分开保存 eos_known 与 final_frames：前者表示不会再产生真实声学帧，后者只在前者为真时定义最终边界。未知长度流无需预先给出 code_end。合法 EOS 根据实际 track/codebook 排列封闭最后一个完整声学帧，一次性确定 F。若必需 codebook 尚未齐备，应按明确的模型规则处理，或返回 FAILED_INCOMPLETE_FRAME，不能伪造缺失 code。EOS 不得撤回已经提交的样本前缀。

| 状态或游标 | 已经成立的事实 | 不能由此推出 |
|---|---|---|
| selected token | token 已选择并保存，随机状态已推进 | 该 token 自身各层 KV 已计算 |
| CODE_READY(b) | 块的全部真实帧、必需 codebook 和身份已经封闭 | mel、波形或未来上下文已就绪 |
| next_decode | 尚未成功提交的最小块号 | 该块尚未开始声学计算 |
| emit | 已原子发布的连续样本前缀为 [0,emit) | 样本已经发送或播放 |
| pending | 一个未提交块的 solver、波形及暂定 overlap | 消费者可以读取这些结果 |
| SENT | 输出已交给声明的发送边界 | 远端已收到或播放 |
| PLAYED(P) | 客户端确认播放至 sample offset P | 在途旧块已消失 |
| DONE | 唯一合法 END 已排在全部音频之后进入输出队列 | 网络排空、播放或物理回收完成 |

所有块携带请求 ID、代次 epoch、块号、逻辑样本区间、模型/codec/格式身份及依赖版本。取消撤销该代次的未来提交权，不能撤销已经播放的样本，也不证明在途计算结束。

Qwen2.5-Omni 固定报告描述 Thinker 产生高层表示和文本，Talker 流式接收高层表示及已采样文本 token 的 embedding，产生语音 token；报告称 qwen-tts-tokenizer 可通过 causal audio decoder 流式解码。报告另展开 Flow-Matching DiT 将 code 转成 mel、修改的 BigVGAN 将 mel 转成 waveform 的路径，DiT 窗口包含两个 lookback 块、当前块和一个 lookahead 块。语音生成不要求与文本做词级或时间戳级对齐。这些事实不能确定任意 checkpoint 的 codebook 排列、层序、solver 或首尾策略。

## 可解码性与首尾契约

设左上下文块数为 L，右 lookahead 为 A。CODE_READY(b) 检查全部必需 codebook 和真实帧；DecodeReady 还检查窗口、首尾规则及声码器的稳定输出条件。

对窗口 k=b−L,...,b+A：

- k<0 必须有模型允许的 startup_policy，例如专用初始状态、边界 mask 或训练支持的填充。没有规则时是永久首部策略错误。
- EOS 未知且未来块尚未齐备时返回 WAIT_CONTEXT，不能把当前最大块号当作最终边界。
- EOS 已知且 k≥N 时，只能使用 end_policy 允许的终止上下文；缺少规则为 FAILED_END_POLICY，等待真实 code 不会解除它。
- 最后真实块不足 N_f 帧时，检查真实部分齐备，再按 end_policy 构造虚拟上下文和 mask，只输出真实样本区间。完整末块也可能因右 lookahead 不足而需要尾策略。
- 中间任何必需帧、codebook 或版本缺失都会阻塞，不能因更远块已经存在而跳过依赖。

声码器可能还需要相邻 mel、卷积历史和右侧 halo。DiT 的 code 窗口齐备不自动证明这些 mel 已齐备；适配器必须把它们加入 DecodeReady 或声码器暂定阶段，并把等待和工作计入成本。

本章服务接口要求：普通块只有能够给出整个 [s_b,e_b) 的稳定样本时才能提交；已知最终块可保留尚未稳定的尾样本到独立 FLUSH 阶段。若 codec 必须跨普通块扣留样本，应改用显式稳定样本 watermark，不能套用完整块提交接口。

尾策略须同时说明输入填充、mask、有效长度、声码器结束方式和裁剪。补零让张量形状合法，不等于模型允许补零。补出的右上下文不得变成新增静音输出。缺少合法尾策略时，可以交付已经提交的稳定前缀并报告失败，不能宣告完整成功。

## Talker：消费、选择与自回归状态

实际双 track 和 codebook 顺序由 checkpoint 契约 π 给出，它把预测位置映射到 track、codebook、帧和允许词表。参考算法不猜测这个排列。其位置事务与 [Encoder-decoder 的消费和推进算法](encoder-decoder-streaming.md) 相接，下面展开声学分支中的具体关系。

令 known 为已经确定的输入和已选择 token，n 为已完整计算各层状态的位置数。初始化至少包含模型规定的起始 token，n=0，各层 KV 和待选 logits 为空。参考输入为 h_i=Embed_track(known_i)+Pos_i+G_i，其中 G_i 是该位置允许消费的 Thinker 表示和已采样文本 embedding 经桥接投影形成的 d 维条件。其可见范围与版本必须固定；相加是参考接入方式，不声称等于真实 Talker 的条件融合。

TalkerStep 的结果是带类型的控制结果：COMPUTED、SELECTED、EOS、WAIT(reason)、CANCELLED(reason) 或 FAILED(reason)。WAIT 携带实际缺少的条件或事件；FAILED 携带不可继续的具体原因。调用方必须接收并分派结果，不能用“执行后继续”抹掉这些区别。

```pseudocode
ALGORITHM TalkerStep(req, π)
输入：known、计算游标 n、逐层 KV、条件视图、随机与约束状态
输出：COMPUTED、SELECTED、EOS、WAIT(reason)、CANCELLED(reason)、FAILED(reason)
1. 若取消或代次过期：返回 CANCELLED(实际原因)。
   若生成资格已撤销或已有失败：返回对应终态，不消费或选择。
2. 若 n < length(known)：
   i ← n；条件尚未到达时返回 WAIT(WAIT_CONDITION,条件身份)。
   条件版本不合法时返回 FAILED(具体身份原因)。
   h ← Embed_track(known[i]) + Pos[i] + G[i]。
   建立位置 i 的私有逐层 KV 写区。
   对层 l 从低到高：
       U ← LN_l(h)；q,k,v ← U W_Ql, U W_Kl, U W_Vl。
       按位置规则变换 q,k，暂写本层位置 i 的 k,v。
       每个 head 对合法历史 0..i 计算：
           p ← softmax(q Kᵀ/sqrt(d_h) + causal/track mask)。
           a ← p V。
       h ← h + ConcatHeads(a) W_Ol。
       h ← h + GELU(LN_Fl(h) W_1l) W_2l。
   logits ← LN_out(h) W_vocab(π 的下一位置)。
   全部层成功且再次校验身份后，原子提交全部当前位置 KV，n ← i+1。
   身份失效返回 CANCELLED；计算错误返回 FAILED，均不暴露部分层长度。
   仅当 n=length(known) 时保存预测位置为 n 的 logits；否则继续消费已知输入。
   返回 COMPUTED。
3. 要求 n=length(known)，且存在预测位置 n 的合法 logits；
   不满足则返回 FAILED(FAILED_MISSING_LOGITS)。
4. 按 π 的允许词表、约束和采样规则选择 y；在身份校验事务中
   保存 y、随机与约束状态，追加 y 到 known，清除 logits；n 不变。
   此选择记录不等于声学块或音频已经提交。
5. 若 y 是声学 code：
   检查 code 范围、π 位置及版本；不合法返回 FAILED(具体原因)。
   写入对应帧/codebook 暂定集合。
   全部必需真实 code 齐备且合法后，带身份发布 CODE_READY。
6. 若 y 是声学结束标志：
   由 π 封闭的完整帧计算候选 F。
   必需 codebook 未齐备且没有模型允许的处理规则：
       返回 FAILED(FAILED_INCOMPLETE_FRAME)，不设置 eos_known 或 final_frames。
   越界 code、非法结束位置或已提交前缀回退：返回 FAILED(具体原因)。
   全部合法后，原子设置 eos_known=true、final_frames=F，
       关闭新的真实声学选择，返回 EOS。
7. 返回 SELECTED。
```

矩阵采用 token 为行、通道为列。参考 FFN 的 W_1∈R^{d×d_ff}、W_2∈R^{d_ff×d}，输出仍为 d 维。LN 沿通道归一化并加正稳定项；softmax 沿合法 key 归一化，禁止没有合法 key 的行。真实模型的 norm、位置编码、门控 FFN 和 head 组织按模型替换。

选择 y 后，其自身 KV 尚未计算；正常恢复时先消费这个已知 y，再用新 logits 选择后继，不能重复采样旧 logits。已封闭 code 可交给声学解码，不要求为了交付再计算预测后继所需的 KV。没有后继预测需求的合法终止 token 无须额外前向。预算暂停保存 known、n、logits 预测位置、π 进度及随机状态。

第 4 步的选择记录可能先于第 6 步的语义失败。此时追加的 EOS、已经推进的 RNG、未封闭帧和其他未交付 code 必须随请求失败一起隔离，不能把它们当作下一次正常恢复的输入。单一推进拥有者在分派结果之前不启动新的下游工作；具体隔离与终态动作如下节所述。

## Flow 网络、solver 与波形数据流

为展开网络内部路径，定义参考速度场 Vθ。目标块的 mel 状态 z∈R^{M×m}，M 为本次求解的 mel 时间位置数，m 为 mel 通道数。窗口 code 与合法边界表示经 embedding 得到 E∈R^{S×d_e}：参考路径对同一声学帧各 codebook embedding 求和，并保留帧/块位置。S、M 和声学帧数是不同轴，不要求一一相等。

参考网络在目标块 M 个位置内做 self-attention，通过 cross-attention 读取 b−L..b+A 的 code 窗口；因此多层传播不会取得窗口外 code。这是用于恢复机制的参考结构，不声称是报告中 DiT 的真实排列。实际模型若在整个 mel 窗口做 self-attention，应按真实 token 数和 mask 重算依赖。

设 d=H d_h，W_z∈R^{m×d}、W_v∈R^{d×m}。每个 self head 的 Q/K/V 为 M×d_h，cross K/V 为 S×d_h。时间编码使用固定频率 sin/cos，经两层 MLP 得到 e_t；每层映射出长度 d 的 γ、β、gate。定义 Ada(X,γ,β)=(1+γ)⊙LN(X)+β，沿时间轴广播。self 四个投影为 d×d，cross 的 K/V 投影为 d_e×d，其余为 d×d；FFN 形状与上一节一致。

```pseudocode
FUNCTION ReferenceVelocity(z, t, E, masks)
输入：z[M,m]、时间 t、E[S,d_e]、合法 mask
输出：完整速度场 v[M,m]
1. X ← z W_z + PositionMel。
   e_t ← MLP(sin/cos(frequencies × t))。
2. 对 DiT 层 l 从低到高：
   γ1,β1,g1,γ2,β2,g2,γ3,β3,g3 ← Mod_l(e_t)。
   U ← Ada(X,γ1,β1)。
   Q,K,V ← SplitHeads(U W_Ql), SplitHeads(U W_Kl), SplitHeads(U W_Vl)。
   每个 head：P ← RowSoftmax(Q Kᵀ/sqrt(d_h)+self_mask)；A ← P V。
   X ← X + g1 ⊙ (ConcatHeads(A) W_Ol)。
   U ← Ada(X,γ2,β2)；Q_c ← SplitHeads(U W_Qcl)。
   K_c,V_c ← SplitHeads(E W_Kcl), SplitHeads(E W_Vcl)。
   每个 head：P_c ← RowSoftmax(Q_c K_cᵀ/sqrt(d_h)+code_mask)；C ← P_c V_c。
   X ← X + g2 ⊙ (ConcatHeads(C) W_Ocl)。
   U ← Ada(X,γ3,β3)。
   X ← X + g3 ⊙ (GELU(U W_1l) W_2l)。
3. v ← FinalAda(X,e_t) W_v；检查形状和数值合法性。
4. 全部层和输出投影完成后才返回 v。

ALGORITHM ReferenceFlowBlock(scratch, E, t[0..K])
输入：固定条件、K≥1 的时间表、请求私有初始噪声/RNG
输出：完整 mel 或可恢复暂停
1. 首次进入：按固定随机状态采得 z[M,m]，保存 k=0。
2. 当 k<K：
   若预算不足：保存一致 scratch，返回 PAUSED_BUDGET。
   v ← ReferenceVelocity(z,t[k],E,masks)。
   z_new ← z + (t[k+1]−t[k]) v。
   全部更新完成且合法后，成对保存 z←z_new、k←k+1 及 solver/RNG 状态。
3. mel ← 对 z 做模型规定的反归一化；参考路径可取恒等映射。
4. 返回完整 mel。
```

这里采用噪声到 mel 的时间方向及显式 Euler。其他 checkpoint 的方向、预测参数化和 solver 必须整体替换，不能直接套图像 DDIM 的噪声公式。层内残差不推进 k；未完成最后输出投影时不能更新 z。暂停只能保留完整 solver 边界，或具有完整身份的网络内部暂定进度。

若 E 不依赖 z 或 t，其逐层 cross K/V 可在相同身份下复用；self K/V 随 z、层和步改变，不能跨步沿用。若时间或噪声参与条件变换，固定 cross 缓存也失效。

声码器消费 mel 和已经提交的卷积/overlap 状态的私有快照。有限感受野路径可展开为输入卷积、逐级上采样及局部非线性/残差卷积、输出卷积。例如转置卷积的一层为 u′_a[s]=Σ_{r,c}u_c[r]w_{a,c}[s−r·stride]；逐层组合依赖才能确定所需 mel halo 和输出 origin。相位、padding 与裁剪都会改变全局索引。固定报告不足以核验修改 BigVGAN 的实际核宽、抗混叠、激活及 overlap 参数，本章不臆造这些细节。

适配器必须返回带全局 origin 的波形、稳定范围、暂定后继 overlap 和待 flush 残留。普通块不满足完整稳定范围时不能发布；最终块通过 FLUSH 完成残留，仍只提交真实区间。

## 可恢复推进与发布事务

请求保存 phase∈{CONDITIONING,RUN,FLUSH,END,DONE,FAILED,CANCELLED}、eos_known、final_frames、next_decode、emit、已提交 overlap 版本、最多一个 pending、flush_stage 和 flush_work_done。另保存生成资格、提交资格及不可覆盖的首个终态原因。pending 有 FLOW、VOCODER、READY 阶段及 solver 游标。初始化 next_decode=emit=0、无 pending、flush_work_done=false。

请求推进和 EOS 接受由单一逻辑拥有者串行化；取消和代次切换可以并发，必须与提交事务协调。推进期间不允许另一个拥有者同时修改 code、条件或 overlap。EOS 到来不改变已封闭完整块；若 EOS 在已稳定普通块提交后才被接受，结束路径处理零新增样本的 flush，不能追溯修改前缀。

输出队列同时限制事件数和字节数；音频额度包括样本、声道、格式和头部，END 也占事件额度。单个最大合法事件永远无法容纳时返回 FAILED_QUEUE_CONFIG，不能无限 WAIT_SEND。额度按本地所有权转移归还，不能用远端播放进度代替本地出队完成。

```pseudocode
FUNCTION StopSpeech(req, terminal, reason)
输入：FAILED 或 CANCELLED，以及具体原因
输出：带原因的吸收终态
1. 在与提交协调的状态域中：
   若已有终态，保留其首次原因并返回。
   保存 reason；撤销生成资格和未来提交资格；phase←terminal。
   冻结 Talker 的 known、n、logits、π、RNG 与约束状态供诊断。
   将未交付 code、未封闭帧、私有 KV 写区和 pending 标为不可推进、不可发布。
2. 已合法提交的 DATA、emit、next_decode 和已提交 overlap 不回写。
   不从失败后缀继续生成，也不排入正常成功 END。
3. 在途计算、传输和输出读取仍按各自最后使用证据结算物理责任。
4. 返回 (terminal,reason)。

FUNCTION TryPublish(req,p)
输入：READY 的不可变 pending 波形及暂定后继 overlap
输出：COMMITTED、WAIT_SEND、CANCELLED 或明确错误
1. 将原始位置 i 映射到 origin+i，只保留 [p.start,p.end) 的样本。
2. 校验完整长度、连续索引、稳定范围和真实尾边界；
   不符返回 BLOCK_LENGTH_MISMATCH 或 UNSTABLE_TAIL，游标不变。
3. 在与取消、代次切换及队列消费者协调的同一提交事务中：
   校验未来提交资格仍有效、p.epoch=当前 epoch 且请求未取消，
       否则返回 CANCELLED 或已有失败。
   校验请求、模型、格式、上下文身份，以及
       p.block=next_decode、p.start=emit、p.base_overlap_version=已提交版本。
   校验 p.end 仍符合当前真实边界；不符返回明确错误。
   若事件或字节额度暂时不足，返回 WAIT_SEND，不发布任何对象。
   原子发布完整 DATA，并同时执行：
       emit ← p.end；next_decode ← p.block+1；
       提交 p 的后继 overlap 并递增版本；记录 DECODED/QUEUED。
4. 返回 COMMITTED；调用方不得再次推进 next_decode。

ALGORITHM AdvanceSpeech(req,budget)
输入：初始化或可恢复请求、时间片预算
输出：等待、暂停、进展或带原因的终态
1. 若为 DONE/FAILED/CANCELLED，返回保存的终态及原因。
   若取消或代次过期，返回 StopSpeech(req,CANCELLED,实际原因)。
2. CONDITIONING：条件不足返回 WAIT_CONDITION；
   否则初始化 Talker、codec/overlap 和有界额度，phase←RUN。
3. 循环：
   先处理已确定终态和取消；预算只决定能否启动新的计算单元。
   已返回的失败、取消或等待必须先分派，不能被 PAUSED_BUDGET 覆盖。
   若 phase=FLUSH：
       若未完成 flush：在预算允许时从保存的 flush_stage 继续；暂停保留残留。
           缺少合法尾规则：返回 StopSpeech(req,FAILED,FAILED_END_POLICY)。
           完成后把最终 pending 置 READY，保存 flush_work_done=true。
       若有最终 pending：调用 TryPublish，并接收结果。
           COMMITTED：清除 pending，phase←END，继续。
           WAIT_SEND：保留结果与 flush_work_done，返回 WAIT_SEND。
           CANCELLED：返回 StopSpeech(req,CANCELLED,实际原因)。
           其他错误：返回 StopSpeech(req,FAILED,具体原因)，不推进游标。
       否则要求 emit=S_F 且 flush 没有新增逻辑样本，phase←END，继续。
   若 phase=END：
       要求 eos_known、next_decode=N、emit=S_F、flush_work_done 且无 pending。
       在身份/资格/取消/额度事务中把唯一 END(S_F,原因) 排在全部 DATA 之后。
       暂时无额度返回 WAIT_SEND；身份失效进入相应取消终态；
       其他错误保存原因并进入 FAILED；成功则 phase←DONE，返回 DONE。
   若 phase=RUN 且有 pending：
       若阶段为 FLOW：在预算允许时继续同一 scratch，完成后转 VOCODER。
       若阶段为 VOCODER：在预算允许时继续同一声码器快照；完成后：
           已知最终块转 FLUSH；普通块须完整稳定，转 READY。
       若阶段为 READY：调用 TryPublish，并接收结果。
           只有 COMMITTED 清除 pending；WAIT_SEND 保留结果返回；
           CANCELLED 返回 StopSpeech(req,CANCELLED,实际原因)；
           其他错误返回 StopSpeech(req,FAILED,具体原因)。
       继续循环，禁止为同一块再次初始化噪声或声码器快照。
   若 eos_known 且 next_decode=N：
       在访问任何下一块之前转 FLUSH，继续。
   b←next_decode；重新检查全部 code、窗口、首尾和声码器依赖。
   若永久错误：返回 StopSpeech(req,FAILED,具体原因)。
   若未就绪：
       若 EOS 未知且外层条件与 code ring 额度允许尝试 Talker：
           若预算不能启动下一 Talker 单元：保存一致状态，返回 PAUSED_BUDGET。
           r←TalkerStep(req,π)；计入本片工作。
           立即按 r 分派，期间不重入循环、不检查剩余预算：
               COMPUTED 或 SELECTED：继续到循环中的依赖重查。
               EOS：要求 eos_known 与 final_frames 已合法设置；
                   继续重查最终边界、N 和 DecodeReady，禁止再调用 TalkerStep。
               WAIT(reason)：保存一致状态，返回该真实等待原因与依赖身份。
               CANCELLED(reason)：返回 StopSpeech(req,CANCELLED,reason)。
               FAILED(reason)：返回 StopSpeech(req,FAILED,reason)。
               未知结果类型：返回 StopSpeech(req,FAILED,FAILED_TALKER_RESULT)。
       否则返回实际的 WAIT_CODE、WAIT_CONTEXT、WAIT_CONDITION 或 WAIT_CODE_CREDIT。
   取得 scratch 额度；不足返回 WAIT_SCRATCH。
   创建唯一 pending，捕获代次、区间、上下文和 overlap 基版本，阶段设为 FLOW。
   继续循环。
```

COMPUTED 或 SELECTED 之后的重查可能发现一个块已经齐备；EOS 之后的重查可能发现空流、已有完整前缀或一个部分尾块。它们都不能被当作相同的“再生成一步”。WAIT 则必须回到能等待对应事件的调度层；紧循环再次尝试既不能创造条件，也可能把真实等待误报成预算暂停。

失败分派先于时间片判断。例如两个必需 codebook 中只有一个已经提供帧 j 的 code，随后选择 EOS：该 EOS 虽已追加到 known，但返回 FAILED_INCOMPLETE_FRAME 后，外层立即保存原因、进入 FAILED 并撤销资格。即使此选择恰好耗尽预算，也返回 FAILED，而不是 PAUSED_BUDGET。之后重复调用 AdvanceSpeech 只返回同一终态，不消费 EOS、不选择后继、不解码剩余 code，也不新增 DATA 或成功 END。

隔离范围包含失败时所有尚未交付的候选，而不只包含最后一个 token：早先封闭但未提交的 code 块、未封闭帧、solver 和暂定 overlap 都不能继续进入下游。已经合法提交的稳定前缀不因失败回写，失败后的 RNG 记录也不能靠倒退并重采样来假装请求从未失败。若另行支持恢复或重试，应从明确可信的提交快照建立相应执行身份，不能直接重新开放失败后缀。逻辑隔离不删除在途读写责任。

WAIT_SEND 不继续无界生成。code ring 必须容纳合法窗口、未封闭帧和在途读者；只有不再被下一窗口或在途计算引用的旧块才能淘汰。若配置无法容纳最小合法依赖集合，应在配置阶段拒绝，不能让等待形成循环。

F=0 时 RUN 直接进入 FLUSH，不读取块 0；空输出契约完成零样本结束处理，再提交 END。若 A=0 的因果路径在 EOS 前已提交最后完整块，EOS 后 next_decode=N，也直接进入 FLUSH。该分支不能增加真实末端之外的样本；需要扣留尾样本的 codec 应事先采用相应接口。

WAIT_SEND 保留不可变波形、完整 solver 结果和暂定 overlap，额度恢复后只重试提交，不重采样或重复 flush。长度、稳定性和身份错误进入 FAILED，已提交 overlap 与游标不变，暂定结果不能成为下一块输入。取消或代次过期撤销结果资格；物理缓冲仍须等待最后设备、传输及输出使用者结束，见 [取消与完成](../system-semantics/cancellation-resource-completion.md)。

取消与发布必须有共同的线性化顺序：取消先发生，随后完成的结果被拒绝；音频先提交，则属于取消前前缀，客户端仍需丢弃旧代次中尚未播放的数据。入口检查不能替代计算后、提交前的检查。逻辑隔离不证明 GPU 地址可立即重用。

一条设定轨迹为 F=3、N_f=2、f_s=5、f_a=3，两个真实区间是 [0,3)、[3,5)。块 0 提交后队列满，块 1 进入 FLUSH；第一时间片保存残留，下一片完成尾部。发布返回 WAIT_SEND 时 next_decode=1、emit=3，已提交 overlap 仍属于块 0。发送者归还额度后才提交块 1，游标变为 2 和 5；END 无事件额度时停在 END，再恢复才 DONE。整个过程不请求块 2，也不交付满块补出的第 6 个样本。

若最后块少一个样本，发布失败且三个提交游标不变，不能进入 DONE；若声学计算完成后、事务之前发生取消，同样不能新增音频事件。

## 到达轨迹、启动缓冲与 deadline

令 A_b 为客户端获得第 b 个完整、连续可播放块的墙钟时刻。乱序数据先重排，只有连续样本前缀增长才计入到达。累计到达音频量为

R(t)=Σ_{b:A_b≤t}(e_b−s_b)/f_s，

单位是音频秒。音频量阈值 B_0 与固定墙钟等待 Δ_0 是两种策略：

- 音频量阈值：T_0 是 R(t) 首次达到 B_0 且存在音频的时刻。整块到达允许越过阈值，不能把块内音频秒变成额外墙钟等待。
- 短请求：若合法 END 已到达、已有非空音频且总量小于 B_0，可按声明策略在 END 到达时启动，否则阈值永远不能达到。空输出没有首个播放样本。
- 固定墙钟等待：收到首个非空块后等待 Δ_0，T_0=A_0+Δ_0；届时缓冲量由实际到达轨迹决定，不保证恰有 Δ_0 秒音频。

同时使用两种约束时，取满足二者的最早时刻。设备启动、重采样和播放器内部缓冲产生的额外延迟应另列。

由 sample offset 得到块 b 首样本 deadline：

d_b=T_0+s_b/f_s。

等长完整块时近似为 T_0+bD；非整数几何与部分尾块使用实际边界。若 A_b>d_b，且不允许时伸缩或重缓冲，则在该块之前欠载。等号在本章理想事件顺序下合法，真实服务应留 margin。

未欠载时，播放前缓冲为 R(t)，播放期间为 B(t)=R(t)−(t−T_0)。发生欠载后，消费时钟会暂停或按播放器策略改变，不能把负数当真实队列长度。连续近似 dB/dt=a(t)−1 只适用于正常播放区间，平均 a>1 不排除局部断供。

若 C_{b+A} 为所需 code 上下文完成时刻，Q_b、V_b、N_b 为 flow、vocoder 和传输的串行耗时，则需要

C_{b+A}+Q_b+V_b+N_b ≤ T_0+s_b/f_s−margin。

尾块用合法 EOS/终止上下文就绪事件替代不存在的 C_{b+A}。阶段可重叠时按依赖图和共享资源求关键路径，不能无条件相加，也不能忽略同设备争用而直接取最大值。聚批等待、不可抢占 kernel 和发送背压同样进入路径。

设计算例，非实测：f_s=24 kHz、f_a=50 Hz、N_f=10，块长 200 ms、4800 样本；code 生成 500 帧/s、A=1，flow/vocoder 合计 25 ms/块，发送 5 ms，条件已就绪且阶段串行。首块需要 20 帧，客户端到达时间为 40+25+5=70 ms；稳态新增一块需要 20+25+5=50 ms，供给 200 ms 音频，对应该串行资源 25% 占用。

若后续块在 120、170 ms 到达，100 ms 音频量阈值在 70 ms 已满足；300 ms 阈值在 120 ms 满足。固定等待 Δ_0=100 ms 则在 170 ms 开播，此时按该轨迹已有 600 ms 音频。170 ms 不能由“积累 100 ms 音频”直接推出。条件另等 150 ms 时，首块约在 220 ms 到达，再按移动后的轨迹计算启动。

24 kHz 单声道 16-bit PCM 每块为 9600 字节，400 ms 音频为 19200 字节，稳态为 384 kbit/s。压缩减少字节，不改变播放时长。某块额外排队 180 ms 是否断音，取决于当时 lead 与实际缓冲；平均 25% 占用和单块处理超过 200 ms 都不能单独决定。

## 成本与有界驻留

code 生成若每帧每 codebook 恰有一个 token，每秒约有 C f_a 次声学选择，文本与控制 token 另计。每次选择之外，还有消费 token 的完整前向、历史 self KV 读取、输出头和采样；已选择长度不能当作已计算长度。条件预处理和 prefill 属于准备成本。

参考 DiT 每层、每步的 self QK/PV 为 O(M²d)，投影为 O(Md²)，cross attention 为 O(MSd)，条件投影为 O(Sd_e d)，FFN 为 O(Md d_ff)。N_l 层、K 次完整前向再乘 N_l K。固定条件 K/V 复用只省条件投影，query 和 attention 仍执行。实际 DiT 若在整个窗口 T_W 个 mel token 上计算 self-attention，应使用 O(T_W²d)，不能用 M 替代。减少 solver 步改变数值轨迹和质量。

声码器按实际特征/样本位置、通道、核宽、上采样率、残差分支与 halo 计工作，重叠重算和尾部 flush 另计。波形字节不等于卷积激活流量。主机侧还有重采样、格式转换、封包、复制与队列管理。

驻留包括共享权重；每请求 Talker self KV 与条件；约 C N_f W_ring 个 code token；E 和可能缓存的 cross K/V；当前 z、速度场、solver 历史及层工作区；声码器已提交状态与暂定快照；pending 波形、发送和客户端缓冲。共享物理存储只计一次，额外副本另计。WAIT_SEND 减少重算，却延长 pending 和模型状态驻留，需受 scratch 额度和暂停期限约束。

访存须区分逻辑读取、片上复用和实际设备事务。flow 重复 K 次完整预测，其权重、激活和条件搬运受 batch、布局、缓存与 backend 影响，不能由参数容量直接声称 HBM 流量。通信账区分 PCM、编码主体、协议头、重传和跨设备条件/特征。

准备包括条件桥接、首部策略、随机初始化、codec 状态及预热；同步包括条件就绪、完整 code 发布、solver 更新、声码器稳定、DATA 提交、额度归还、END 和播放确认。取消或错误后的无效在途计算、隔离对象与回收滞后仍计资源；进入 FAILED 只关闭资格，不会把这些成本变成零。

共享串行设备上的占用相加；独立资源可流水重叠，稳态受瓶颈与争用限制，首块仍受启动依赖约束。单阶段计算下降而聚批、网络或驻留主导时，首音频和欠载不一定改善。

## 条件变化后的选择

| 变化 | 可以保留 | 必须调整或放弃 |
|---|---|---|
| N_f 减小 | 连续 sample offset、事务和 deadline | 调用数、halo、窗口时长和接缝；模型未支持时放弃该块形状 |
| f_s 提高 | 声学帧时长可以保留 | 样本边界、vocoder 工作、网络和缓冲字节 |
| f_a 改变 | 有理边界推导方法 | token 速率、块时长、训练契约和 code/mel 映射 |
| codebook 或延迟排列变化 | 全部必需 code 齐备才能封闭 | π、帧结束判定、EOS 合法位置、就绪时间、KV 和 code ring |
| lookahead 减少 | 提交与播放记账 | 模型可见函数、内容/音色/韵律与接缝；不能未经允许删依赖 |
| 少 flow 步或换 solver | 完整网络预测后再更新 | 时间表、误差、历史和质量，旧性能/质量结论需重验 |
| batch 增大 | 每请求身份和私有状态 | 聚批等待、shape/solver 阶段相容性、峰值及最早 deadline |
| EOS 提前或部分尾块 | 已提交稳定前缀 | F、完整帧条件、终止上下文和 FLUSH；无合法规则则失败 |
| 抖动或慢客户端 | 样本几何和音频时长 | 到达轨迹、启动策略、额度、暂停驻留和超时 |
| 模型/codec/条件版本变化 | 外部身份框架 | token、KV、条件投影、solver、overlap 和缓存；不相容状态重建 |

离线任务可允许等完整条件、较大块和较高求解步数；实时对话更重视首可播放样本、deadline 和打断。整段非因果模型把结果切成小网络包，不会因此获得合法的低延迟声学生成。vocoder 已是瓶颈时，提高 code 速度可能只增加领先驻留；设备饱和时大 batch 可能以排队换吞吐。

打断同时撤销服务端提交权并清除客户端尚未播放的旧代次数据。恢复需要共同边界：已选择/已计算 token、条件版本、solver、已提交 overlap、emit、发送和播放确认，以及请求是否已进入吸收终态。只有 KV 或最后块号不足以恢复；失败后发现还有未消费 token，也不构成恢复生成资格的理由。

## 诊断与可反驳检查

| 观察 | 竞争解释 | 区分实验与反证修正 |
|---|---|---|
| token/s 高而首音频晚 | 条件、lookahead、flow/vocoder、网络或启动策略 | 记录同请求各阶段事件，分别预置条件与改变启动策略；code 早而播放晚时撤回 token 生成慢的解释 |
| 半帧 EOS 后暂停或继续生成 | π 对 EOS 的规则不同；失败返回被忽略；预算判断覆盖失败 | 固定两 codebook 契约，生成一个未齐备帧后选择 EOS，分别给予恰好一步和更大预算；首次失败必须立即终止，重入不能再调用 Talker 或发布 DATA/END |
| EOS 后 WAIT_CODE | 终止检查后置、结束标志/边界混用、真实 code 缺失 | 空输出、因果完整前缀、部分尾块和 FLUSH 暂停分别测试；无下一块仍可结束时修正阶段顺序 |
| 队列满后少块或重复尾音 | 失败仍推进、重算/重复 flush、重复发送 | 在普通块、尾块、END 前耗尽额度，恢复后核对计算次数、样本和唯一 END |
| 计算后取消仍新增音频 | 只在入口查取消、代次复用或客户端旧队列 | 在计算完成与提交间注入取消/代次变化；仍提交则修正事务保护 |
| 边界重复、缺口或 click | overlap、取整、重采样相位或声学质量 | 先用递增样本/脉冲核对全局索引；索引正确仍有 click 再查上下文与音质 |
| 积 100 ms 却又等 100 ms | 音频量/墙钟混用、整块越阈值、播放器延迟 | 固定到达轨迹分别计算阈值与固定等待；剩余延迟用播放器事件定位 |
| 平均供给快仍断音 | 突发、聚批、长 kernel、缓冲不足 | 逐块检查 A_b−d_b 与连续缓冲，固定质量/负载扫描等待上限 |
| 小块或少步后质量下降 | 上下文、边界或 solver 误差 | 合法范围内固定步数改块，再固定块改步数；按首次分歧缩小解释 |

随章独立检查为 `tools/check_k_model_011_full.py`。标准库 CPU 小模型用独立事件 ledger 核对提交样本恰为规范序列前缀、块号/emit/overlap 只随成功事务推进、依赖与代次合法、队列不越额度、成功 END 覆盖最终边界。

声学提交模型覆盖空输出、未知 EOS、部分尾块、完整因果前缀后 EOS、两段 flush、普通块/尾块/END 背压及恢复、长度失败、计算后取消和槽位复用。故障产生的实际事件交给同一个 ledger：终止判断后置、发布失败仍推进、缺少取消/代次保护、忽略额度、漏 lookahead、只检查首 codebook、发布重复 overlap，以及无合法尾策略却补零解码。合法 pad-and-crop 路径只提交真实长度。

另一路合成生成器直接执行已知 token 消费、保存预测位置、选择新 token、构造多 codebook 帧、校验 EOS 和外层结果分派；这些回归不通过外部 feed/seal 注入待测失败。独立控制事件判据从选择与消费记录重建前沿，并规定失败/取消返回后立即进入吸收终态，等待返回真实原因，COMPUTED/SELECTED/EOS 返回后先重查依赖或最终边界。半帧 EOS 在新时间片的第一次 Talker 调用返回失败时，预算恰好耗尽与仍有余量两种情况都检查；终态重入不得增加 Talker 调用、DATA 或 END。已有稳定前缀后的半帧失败也检查前缀不回写。移除结果分派的故障变体产生的状态与事件由同一判据拒绝。

几何检查逐边界对照 Fraction 参考，播放检查使用逐样本可用时间与累计到达事件，区分音频量阈值和墙钟延迟。检查中的 token 选择由确定性输入序列代替学习网络的 logits；消费只模拟位置事务，不执行多层 Talker 数值。flow/声码器使用规范索引样本和两段合成 flush，不验证真实速度场或波形。

这些检查不证明 GPU 可见性、物理最后使用完成、kernel 时间、声学质量、WER 或知识使用效果，也未穷举所有调度交错、code ring 淘汰和 scratch 分配。真实实验应固定文本/语义和音色，联合报告内容、音色、韵律、接缝、噪声、首样本、欠载、打断残留与最终长度。ASR 回转 WER 只覆盖部分可懂内容；裁去失败块、补静音或改变语速须披露为输出语义变化。

<!-- infraagents:provenance -->

## 来源与推导

- [S-OMNI：Qwen2.5-Omni Technical Report](../sources/S-OMNI.md)：报告描述 Thinker/Talker 分支、块式模态输入、时间对齐和流式音频输出，文本与音频不是同一执行时钟。§2.1、§2.3 与 §2.4 支持 Talker 接收高层表示和已采样文本 embedding、产生语音 token，qwen-tts-tokenizer 的 causal audio decoder、Flow-Matching DiT 生成 mel 及修改 BigVGAN 生成波形；§2.4 给出四块窗口，即两个 lookback、当前块和一个 lookahead，并分解首包延迟。本文据此推导窗口与服务依赖；参考层序、Euler、提交状态机和首尾策略不冒充 checkpoint 事实。
  [完整原文](../../references/S-OMNI/reader.html) · [全文文本](../../references/S-OMNI/fulltext.txt)

- [S-MOONSHINE：Moonshine: Speech Recognition for Live Transcription and Voice Commands](../sources/S-MOONSHINE.md)：Moonshine 面向语音识别的 encoder-decoder 结构和变长音频处理，源音频表示与目标文本生成具有不同长度和计算阶段。固定论文 v2 描述 strides 为 64、3、2 的卷积 stem、RoPE、约 4 到 30 秒的训练实例，以及极短输入重复输出和超过训练长度的质量边界。它支持窗口、长度与边界的对照，不是 codec TTS 或声码器能力的来源。
  [完整原文](../../references/S-MOONSHINE/reader.html) · [全文文本](../../references/S-MOONSHINE/fulltext.txt)

- [S-WHISPER：Robust Speech Recognition via Large-Scale Weak Supervision](../sources/S-WHISPER.md)：Whisper 采用语音 encoder-decoder 和任务条件 token；音频编码与目标文本自回归是不同阶段，decoder 使用源表示完成条件生成。固定论文 v1 描述 16 kHz 音频、80 通道 log-magnitude Mel、25 ms 窗、10 ms 步长、30 秒块，以及相对当前窗口、20 ms 量化的时间戳。它提供源表示、条件生成和窗口推进参照，不能证明任意 codec token 可独立流式解码。
  [完整原文](../../references/S-WHISPER/reader.html) · [全文文本](../../references/S-WHISPER/fulltext.txt)

正文公式、参考网络、样本几何、容量与时延算例、状态机及区分实验为基于来源机制的独立推导，不是目标模型或机器实测。来源事实受 2026-04-30 上限约束，编辑日期不改变事实日期。实际排列、条件融合、solver、首尾规则和 BigVGAN 参数仍需针对具体模型核验并转写为对应算法。

## 延伸阅读

- [Encoder-decoder：源表示、两套 Attention 与流式边界](encoder-decoder-streaming.md)。
- [图像扩散：迭代求解、条件复用与 CFG](image-diffusion.md)。
- [服务指标：事件边界、Token 间隔与用户时间](../workloads/metric-events.md)。
- [服务协议：接收、流式背压与终止语义](../system-semantics/service-protocol.md)。
- [服务主机运行时：执行域、队列与缓冲所有权](../system-semantics/host-runtime.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
- [推理服务恢复：提交边界、执行组与重试身份](../system-semantics/serving-recovery.md)。
- [Goodput：联合 SLO、拒绝和容量前沿](../workloads/slo-goodput.md)。
- [全模态系统：多时钟、分支依赖与背压](omni-multimodal.md)。
- [系统设计推导：把模型、缓存、执行与目标组合起来](../system-semantics/design-synthesis.md)。
