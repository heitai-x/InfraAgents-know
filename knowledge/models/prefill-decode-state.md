# K-MODEL-001：从模型计算图推导 Prefill、Decode 与持久状态

模型的计算图和状态语义说明哪些量随 token、位置、请求和执行阶段变化，为执行机制的选择与验证提供依据。本章以普通因果 decoder 的全注意力、GQA 和窗口 KV 为核心，展开前向、分块、提交与前缀命中恢复。MLA、递归模型、encoder-decoder 和多模态模型用于说明哪些依赖改变后不能直接套用这条路径。prefill 与 decode 是同一因果函数的不同执行形状；阶段名称本身不决定缓存是否合法，也不决定设备瓶颈。

## 自回归函数、输入和充分状态

设模型参数及配置为 θ，规范化输入为 x₀…xₙ₋₁。在普通零基文本序列中，位置 t 的前向产生 logits ℓ_t，其 softmax 表示下一个 token x_{t+1} 的条件分布：

ℓ_t = F_θ(x₀…x_t)，p_θ(x_{t+1}|x₀…x_t) = softmax(ℓ_t)。

prefill 同时消费若干已知输入，最后有效位置的 logits 用于选择第一个生成 token。decode 消费上次选出的 token，产生它的各层状态和后继分布。已知 prompt 行可以并行计算，并不意味着位置 t 可以读取未来行；它仍须遵守模型的因果 mask。

以 S(n) 表示恰好消费前 n 个有效输入后的持久状态。充分性要求：从 S(n) 消费后续输入得到的结果，与相同模型从头计算相同完整序列在数学上相同。普通 decoder 的主要摘要是每层历史 K/V；MLA 保存 latent 内容分量和解耦位置分量；递归模型保存递归张量和卷积历史；encoder-decoder 还可能保存固定源表示及其 cross KV。

S(n) 不是完整的生成会话。继续生成还可能需要末行 logits、已采样但尚未消费的 pending token、采样随机状态、停止条件和输出提交记录。仅有 KV 不保证立刻存在可采样的分布。

必须分别记录：

- consumed_len：已消费并成功提交的有效输入数，本章记作 state.len=n。
- next_position：下一个待消费行的逻辑序号；算法 A 假定零基连续序列，因此等于 n。
- sampled_count：已经选出的生成 token 数。
- emitted_count：按服务协议已提交的输出数。
- logical_len：已知序列长度；普通单 pending 路径中等于 n 加尚未消费的 pending 数。

例如 prompt 已消费 100 个位置，末行 ℓ₉₉ 选出 y。此时 KV 覆盖 0–99，state.len=100；y 的位置是 100，尚无对应 KV。消费 y 后，状态覆盖到 100，ℓ₁₀₀ 用于选择位置 101 的 token。用户可能在消费 y 之前已经收到它，因此输出计数不能代替状态长度。

上述位置首先是序列中的逻辑序号。模型位置编码所用坐标另记为 φ(t,metadata)：普通文本常有 φ(t)=t，多模态模型可能使用时间、高度和宽度坐标。连续性检查约束逻辑序号，不能强迫多维模型坐标等于 state.len；物理页、环形槽和 batch 行号也不是模型坐标。

## 从固定制品还原计算图

同一次函数对照应绑定相容的配置、权重、tokenizer/processor、特殊 token 与生成配置。配置给出声明，权重形状和经过作者核验的 forward 给出实际计算关系，processor 决定输入及位置元数据。三者应共同进入模型档案。

| 档案对象 | 必须还原的关系 | 改变后的影响 |
|---|---|---|
| 层类型与次序 | 全注意力、窗口、递归、cross attention、FFN 的组合 | 状态类型、恢复位置、通信和执行形状改变 |
| 投影与 head 轴 | Q/K/V 宽度、query 到 KV head 的映射、输出拼接顺序 | 容量与 FLOPs 使用不同 head 数 |
| 位置与 mask | 输入位置、逐层旋转、分数偏置、窗口边界、padding | 决定增量结果是否还是同一函数 |
| norm、激活与残差 | pre/post norm、ε、门函数、bias、缩放、活跃 residual | 不能因算子名称相似而省略或交换 |
| 输入及源状态 | 模板、词表、媒体处理、源表示及其版本 | 决定前缀等价类和 cross KV 是否有效 |
| 输出头及协议 | 哪些行需要 logits、最终 norm、采样与终止 | 决定首 token 就绪性、输出数与临时容量 |

设一层有 M 个有效输入，hidden 宽 H，query heads 为 Hq，KV heads 为 Hkv，每 head 维度为 d。Q 的形状是 M×Hq×d，K/V 各为 M×Hkv×d；attention 输出仍有 Hq 个 heads，再经输出投影回到 residual 宽度。不能因为 KV heads 少而缩小 query 输出轴，也不默认所有模型都满足 H=Hq·d。

门控 FFN 可写成 U=XWᵤ、G=φ(XW𝗀)、FFN(X)=(G⊙U)W𝒹；具体门函数、bias、缩放和 norm 顺序由制品确定。[基础算子](../algorithms/operator-primitives.md)展开这些局部契约，[权重加载](../frameworks/checkpoint-loading.md)解释参数轴到执行布局的对应。对照时逐层比较 hidden、K/V 和 logits；能生成自然语言不是计算图正确性的证明。

| 固定研究入口 | 本章使用的结构事实 | 不能据此推断 |
|---|---|---|
| Transformer 1706.03762v7 | 因果 decoder、输入加性位置、encoder-decoder attention | 所有 decoder 都使用相同 norm 与位置机制 |
| Mistral 7B 2310.06825v1 | GQA、窗口、滚动缓冲与分块输入 | 同名后续发行版的全部层配置 |
| DeepSeek-V2 2405.04434v5 | MLA、latent 与解耦 RoPE | latent 可以当普通 K/V head 处理 |
| Mamba 2312.00752v2 | 选择性状态、递推及扫描 | 覆盖后的旧状态可由长度截断恢复 |
| Whisper 2212.04356v1 | 音频 encoder-decoder 与任务条件 token | 音频窗口移动后旧源状态仍有效 |
| Qwen2.5-VL 2502.13923v1 | 动态视觉输入与时间相关多模态位置 | 相同文本 token 足以证明前缀等价 |

这些是固定资料的机制入口，不是未核验 checkpoint 的参数清单。双向 encoder、扩散迭代和图模型需要重新定义推进单位与充分状态，不能直接使用因果 decoder 的逐 token 论证。

## 因果性、位置变换与合法复用

### 增量等价为什么成立

普通因果注意力中，位置 t 的每层输入只依赖 x_≤t。固定参数、位置规则和源条件后，未来输入不改变已经计算的历史 K/V。对层数和消费位置作归纳：若旧状态与完整参考一致，新位置读取相同可见历史和本行 K/V，便得到相同 attention、残差和 FFN 输出；因此新增状态仍与参考一致。

这个论证需要以下条件：

1. 可见集合正确。查询 t 只读取本请求且模型允许的位置；不可见项从 softmax 域中排除，或用等价加性遮罩处理。把 score 乘零不能保证其概率为零。
2. 位置身份正确。逻辑序号、模型坐标 φ(t)、物理槽和 batch 行分别保存；重排与迁移不能改变模型坐标。
3. 模型函数固定。权重、adapter、norm、激活、位置配置、源条件及数值策略必须相容。改变依赖后要重新判断旧状态是否有效。
4. 摘要完整。每层都保留下一步所需内容；窗口、递归和混合状态按各自依赖判断，不能只核对一个长度字段。
5. 状态提交一致。计算失败不能发布部分层；共享前缀只读，物理回收等待最后一个消费者结束。

实数等价不保证浮点逐位相同。分块、GEMM 和 softmax 归约次序可改变舍入；数值容差应结合精度和中间量定义。提高精度后差异缩小支持舍入解释，但不能单凭最终误差小就排除位置或 mask 错误。

### 三条位置路径及缓存格式

对本层归一化输入 u_t，先定义 q_raw,t=W_Q u_t、k_raw,t=W_K u_t、v_t=W_V u_t。

输入加性位置路径在网络入口使用 h⁰_t=embed(x_t)+PE(φ(t))；之后各层直接使用投影后的 q/k。它不等于逐层 RoPE。

标准 RoPE 在每层 Q/K 投影后作用：

q_t=R(φ(t))q_raw,t，k_t=R(φ(t))k_raw,t。

对一对分量 (a,b)，R(p) 的结果为 (a cos(pω)−b sin(pω), a sin(pω)+b cos(pω))。固定频率、配对方式及旋转维度时，R(p)ᵀR(q)=R(q−p)。普通一维坐标下，分数因此可以写成

score_tj=q_raw,tᵀR(j−t)k_raw,j/√d。

缓存可以选择以下两种格式，算法必须贯彻其中一种：

- K-rot：保存 R(φ(j))k_raw,j。新增 Q/K 旋转一次；读取历史 K 时直接使用，不能再次旋转。
- K-raw：保存 k_raw,j 及模型坐标 φ(j)。消费时按该历史条目自己的坐标旋转 K；不能用当前 query 的坐标旋转全部历史。

分数偏置路径则使用 score_tj=q_raw,tᵀk_raw,j/√d+b_h(φ(t),φ(j))，缓存原始 K，偏置在每个 query-key 分数上产生。加性位置、RoPE 和偏置是否组合由具体模型决定，不能默认互相替代或同时打开。算法 A 用独立配置显式表示这些操作。

标准 RoPE 的相对内积关系还给检查增加一个约束：若没有加性绝对位置、没有改变频率或其他绝对位置依赖，并将全部 Q/K 坐标一起平移相同常数，则分数可以保持不变。因此“改了绝对位置但输出相同”不自动证明检查失败。需要检出的是历史保持原坐标、仅新增块归零造成的跨块相对位置变化；同时应检查缓存 K 的位置身份。

### 窗口和其他状态类型的边界

本章设定窗口 W 包含当前行，故 V_t={j|max(0,t−W+1)≤j≤t}；全注意力的 V_t={0,…,t}。这是明确的边界约定，实际 checkpoint 必须核对是否相同。物理槽可取 t mod W，但 φ(t) 仍按模型规则推进。

MLA 的内容状态 c_j 与位置状态 kᴿ_j 不应走普通 K 的旋转分支。若内容 K_h=Wᴷ_h c_j、V_h=Wⱽ_h c_j，则分数可写成 ((Wᴷ_h)ᵀqᶜ_t,h)ᵀc_j+(qᴿ_t,h)ᵀkᴿ_j，再使用模型规定的联合维度缩放；值可先归约 Σ_j a_j c_j，再上投影。实际 latent 的 norm、缩放和 query 压缩必须保留，不能跨非线性吸收。具体完整路径见 [MLA 的压缩、吸收与制品边界](mla.md)。

递归层的推进形如 s_{t+1}=A_t s_t+b_t，系数由该层输入产生；同请求多行必须从上一行的新递归状态继续，不能每行都从入口旧状态开始。扫描可以组合这些仿射变换，但递归状态和卷积历史的最终版本必须共同提交。这里不把递归层塞进普通 KV attention 循环；[递归与混合状态](recurrent-hybrid-state.md)给出扫描、快照及恢复算法。

cross attention 的 Q 来自 decoder，K/V 来自源表示 E。当 E 和投影固定时，cross KV 不随 decoder 步数改变，可以缓存；源编码、音频窗口或媒体预处理改变后，旧 cross KV 及依赖它的 decoder 状态都要重新判断。跨窗口仅保留旧 cross KV 不足以维持同一条件生成函数。

## 前向、提交和首 token 就绪性

### 普通 KV decoder 的完整消费算法

算法 A 的核心范围是普通稠密因果 decoder，允许 MHA/GQA、全历史或上述窗口，以及显式位置路径。每个 query head h 通过模型给定映射 g(h) 读取 KV head。MLA、递归与 cross attention 的完整 block 不能仅靠修改 layer_type 名称接入此算法。

同请求有效行按输入出现顺序编号 j；padding 不参与编号、状态推进或 logits 选择。入口冻结 n_r，要求第 j 行位置为 n_r+j。不同请求可交错出现；不静默排序同请求的乱序输入，因为排序可能掩盖调用者身份错误。

```pseudocode
算法 A：consume_rows(model, state_handle, rows)
输入:
  rows[i] = (request_id, token_or_feature, logical_position, model_position, valid, want_logits)
  committed root: 每请求长度、各层状态目录、模型与源身份
  每层配置: input_additive、use_rope、K_storage、score_bias、mask、head_map
输出:
  各输入行对应的 logits 或空；成功后的完整新 root

入口:
  固定旧 root = state_handle.root
  touched = 有有效输入行的请求
  若没有有效行: 返回空结果，root 不变
  验证模型/源身份及每请求写入权；同请求至多一个推进事务在途
  对 touched 中每个 r:
    n[r] = root.requests[r].len
    按 rows 中的出现顺序列出该请求有效行 i[r,0..k[r]-1]
    对 j=0..k[r]-1:
      要求 rows[i[r,j]].logical_position = n[r]+j
      要求 model_position 与该请求的模型位置规则相符
  预留全层新增状态、必要写时复制、目录和工作区
  若预留失败: 撤销本次预留，旧 root 保持完整，返回失败

初始化:
  对每个有效行 i:
    H[i] = embed(rows[i].token_or_feature)
    若模型使用输入加性位置: H[i] += PE(rows[i].model_position)
  新状态和新目录均为私有对象；旧 KV 内容只读

对层 l = 0..L-1:
  对每个有效行 i:
    r = rows[i].request_id; p = rows[i].logical_position
    u[i] = norm_in_l(H[i])
    q_raw[i] = Wq_l u[i]       // Hq×d
    k_raw[i] = Wk_l u[i]       // Hkv×d
    v_new[i] = Wv_l u[i]       // Hkv×d
    q_use[i] = q_raw[i]
    若 use_rope:
      q_use[i] = R(rows[i].model_position) q_raw[i]
    若 use_rope 且 K_storage == K-rot:
      k_saved[i] = R(rows[i].model_position) k_raw[i]
    否则:
      k_saved[i] = k_raw[i]
    staged[l][r] 加入 (p, model_position, k_saved[i], v_new[i])

  对每个有效行 i:
    r = rows[i].request_id; p = rows[i].logical_position
    candidates = root.requests[r].entries[l] ∪ staged[l][r]
    visible = candidates 中本请求、位置 <= p 且满足模型 mask 的条目
    // 窗口约定为 p-W+1 <= entry.position <= p
    // 当前行恰好出现一次；同 chunk 的逻辑未来被排除
    对 query head h:
      对 visible 中每个 e:
        k = e.K[g(h)]
        若 use_rope 且 K_storage == K-raw:
          k = R(e.model_position) k
        score[e] = dot(q_use[i,h], k) / sqrt(d)
        若模型使用分数偏置:
          score[e] += b_l,h(rows[i].model_position, e.model_position)
      weights = softmax(score，归一化域仅为 visible)
      a[i,h] = Σ_e weights[e] * e.V[g(h)]
    z[i] = H[i] + Wo_l concat_heads(a[i])
    H_next[i] = z[i] + FFN_l(norm_ffn_l(z[i]))
  H = H_next
  // 本层 staged 不对外发布，也不覆盖旧窗口槽

对 want_logits 的有效行 i:
  logits[i] = lm_head(final_norm(H[i]))

完成检查与准备提交:
  确认全部层、logits 和必要异步生产者已成功完成
  对每个 touched r、每层 l:
    private_entries = 旧条目 ∪ staged[l][r]
    new_len[r] = n[r] + k[r]
    全注意力保留全部有效条目
    窗口路径保留位置 >= max(0,new_len[r]-W) 的条目
    // 这里保留最近 W 个已消费位置，也支持算法 C 恢复最新末行
    为未保留旧条目记录待退休引用，不立即释放物理内容
  构造同时包含全部请求新长度、全层目录和身份的新 root

提交:
  在写入权保护下核对旧 root 仍为入口版本
  若核对失败或任何准备步骤失败:
    丢弃私有结果；等待私有在途使用结束后归还本次预留；旧 root 不变
  否则用一次不可分割、发布后不再回退的 root 引用切换提交
  logits 仅随成功提交返回给调用者
  旧目录不再接受新读者；其引用和已提交物理读者全部结束后才回收旧槽
返回 logits, new_root
```

此处采用整次调用的 root 切换，满足“全部行成功提交”的输出保证。也可以按请求分别提交，但接口必须报告每请求成功或失败；不能继续声称整个 batch 全有或全无。算法中的原子切换是实现要求，不能用“末尾 publish”字样代替实际同步或目录协议。单线程 CPU 检查只能验证软件更新顺序，不能证明 GPU 可见性。

空状态的同请求两行使用位置 0、1，成功后长度为 2；已有 n=3 的三行 chunk 使用 3、4、5，成功后长度为 6。整个层循环期间 root 中的长度仍为 3，但第二、三行通过 staged 和位置 mask 读取块内前驱，所以不需要提前提交长度。

窗口尤其不能在投影完全部新 K/V 后，先按最终窗口裁剪整个候选集合。若 chunk 跨过窗口，前面 query 仍可能需要较旧条目。算法先完成所有 query 的读取，再裁剪将来目录；新条目占独立预留槽，不能提前覆盖旧目录仍引用的环形槽。

### 第二层失败的完整轨迹

设两层窗口 W=3，入口 S(3) 的两层均保留位置 0、1、2。消费位置 3、4：

1. 旧目录继续指向两层的 0、1、2，长度为 3。
2. 第一层产生私有 K/V 3、4，并完成两行输出；其查询 3 读取 1、2、3，查询 4 读取 2、3、4。此时位置 0、1 的物理槽仍未覆盖。
3. 第二层失败。丢弃第一层私有条目和其他暂存；旧两层仍为 0、1、2，长度仍为 3。异步暂存若仍在使用，须等其结束再释放。
4. 重试从同一 S(3) 开始。若全部成功，新目录两层均保留 2、3、4，长度为 5。旧目录的 0、1 只有在旧读者完成后才能物理回收。

如果第一层已经原地追加或淘汰，第三步就无法仅靠末尾不发布来恢复；这是本章禁止逐层更新共享状态的原因。共享前缀的另一个请求继续读取旧 root 时，也应始终看到完整旧版本。

### 完整 KV 命中与已有 pending 是两条轨迹

完整 prompt KV 命中、state.len=N 且没有 pending 时，首个生成 token 可能尚未选出。此时需要位置 N−1 的 logits：可以在缓存对象中保存 logits 或最终 hidden，也可以安全重算最新末行。

已有 pending token y、其位置为 state.len=N 时，输入已就绪。即使没有保留旧 prompt logits，也可以直接消费 y。包含当前待消费行的区间 [start,end] 的行数是 end−start+1；start=end=N 时工作量是 1。只算 end−start 得到零属于工作范围错误。

下面的恢复限定为普通 KV decoder；各层必须仍保存末行查询所需的全部 K/V，包括末行本身。它通过已有 K/V 重算 query 路径，不再次追加当前位置。

```pseudocode
算法 C：recover_last_logits_no_append(model, committed_request, last_input)
前置:
  N = committed_request.len; N > 0; p = N-1
  last_input 与缓存前缀中位置 p 的规范输入及模型坐标完全一致
  每层保留 V_p 所需的全部 K/V；条目无重复、身份一致
  本操作只读，且没有与其冲突的状态写入

H = embed(last_input)
若模型使用输入加性位置: H += PE(model_position[p])
对层 l=0..L-1:
  u = norm_in_l(H)
  q = Wq_l u
  若 use_rope: q = R(model_position[p]) q
  visible = 该层缓存中满足位置 p 的模型 mask 的全部条目
  对每个 query head h、每个 visible 条目 e:
    k = 按算法 A 的 K-raw/K-rot 契约读取 e.K[g(h)]
    score[e] = dot(q[h],k)/sqrt(d) + 模型规定的分数偏置
  a[h] = Σ_e softmax_visible(score)[e] * e.V[g(h)]
  z = H + Wo_l concat_heads(a)
  H = z + FFN_l(norm_ffn_l(z))
返回 lm_head(final_norm(H))；长度、目录、KV 和代次全部不变
```

按层归纳，恢复出的 H 与首次计算该末行时相同：每层 query 输入相同，缓存中的当前和历史 K/V 也相同。若窗口实现只保留下一 query 最少需要的 W−1 个旧位置，可能已经缺少恢复上一行所需的最老条目，算法 C 必须拒绝。算法 A 特意保留最近 W 个位置。更早位置恢复、递归状态或缺少必要历史时，应使用完整检查点重放；不能猜测缺失内容，也不能只修改长度。

一条普通生成循环可以明确保留最后一个尚未消费的输出，从而不为预算已耗尽的请求额外做一步前向：

```pseudocode
算法 B：generate_plain_causal(prompt, state, pending, remaining_output_budget)
若 remaining_output_budget == 0: 直接返回，不采样
若 prompt 为空且无可恢复会话:
  按模型定义加入起始输入，或拒绝；不索引不存在的末行

若 pending 不存在:
  若 state.len < prompt 有效长度:
    用算法 A 消费未计算后缀，仅请求末行 logits
  否则若已经完整命中 prompt:
    读取身份相符的末行 logits，或执行算法 C
  从所得分布选择一次 token，记录为 pending，位置=state.len

循环:
  按协议检查 pending 的 EOS、stop 和取消
  若应终止: 返回状态与终止记录
  若 pending 尚未输出:
    提交该输出一次，标记 emitted，减少剩余预算
  若剩余预算 == 0:
    返回 state 与尚未消费的 pending
  要求 pending.position == state.len
  用算法 A 消费 pending 一行并请求该行 logits
  若失败: 保留原 state、pending 及其 emitted 标记，等待重试
  若成功:
    清除已消费 pending
    从新 logits 选择下一 token，记录新 pending，位置=state.len
```

输出提交、采样和状态推进是不同事件；失败重试不能重复输出或重新抽取同一 pending。更复杂的 stop string、流式 detokenization 和 usage 规则由[输入输出语义](../system-semantics/tokenization-io.md)限定。上述循环不处理投机接受、多分支生成或任意 pending 队列。

## 成本：轴、资源边界和提交峰值

### 计算主项

以下按一次乘加计两次 FLOP，假定普通 attention 的 K/Q/V head 维相同为 d。M 是本次有效输入行数，不是 batch 请求数。

一层 Q/K/V 与输出投影的主项为

F_proj≈4MHd(Hq+Hkv)。

Q 与输出投影各产生 2MH·Hq·d，K 与 V 各产生 2MH·Hkv·d。标准两矩阵 FFN 为约 4MHI；三矩阵门控 FFN 为约 6MHI，另计逐元素门函数。norm、RoPE、bias 和 softmax 不在这些矩阵乘主项中。若只有 M_log 行需要 logits，词表宽 V_vocab 的输出头另需约 2M_log H V_vocab FLOP，不能默认对所有 prefill 行执行。

令 P_pair 为单个 query head 的有效 query-key 对数，按请求及本次 query 行求和，不包含 head 轴。若各 heads 可见集合相同，则

F_attn≈4Hq dP_pair。

每对的 QK 点积为约 2d FLOP，PV 加权累加约 2d FLOP。三个常见形状是：

- 空历史 prefill N 行：P_pair=N(N+1)/2。
- 每请求一行 decode、旧长度 n_r：P_pair=Σ_r(n_r+1)。
- 旧长度 h、新 chunk q 行：P_pair=qh+q(q+1)/2。

一般形式为 P_pair=Σ_rΣ_i|V_{r,i}|；窗口只需改为 |V_{r,i}|=min(W,h_r+i)，其中 i 从 1 到 q_r。若 head 的 mask 或维度不同，应直接按 head 求和 2(d_k,h+d_v,h)|V_{r,i,h}|，不能再套统一 Hq 的乘法。

例如 N=2、Hq=2，每 head 有 3 对，跨 head 共 6 对。正确计算为 4×2×d×3=24d，也等于 4d×6。把跨 head 的 6 再代入 4Hq dP 会错误得到 48d。

把完整 prompt 切块，各块的 qh+q(q+1)/2 相加仍等于 N(N+1)/2。分块改变形状、准备次数及复用，不增加合法数学对数；实际 kernel 若为 padding 或不可见 tile 做运算，物理执行 FLOPs 可以高于这一有效主项。

算法 C 只重算一行 Q、输出投影、FFN、最终 norm 和输出头，读取末行的可见 K/V；已有完整当前 K/V 时可省略新的 K/V 投影及持久写入。因此全层仍有一次窄形状前向和 attention 读取，完整命中但未保存 logits 并不是零计算。

### 容量与读取流量

同构全注意力的逻辑 KV 容量为

B_KV=2LHkv d sN，

L 为层数、s 为每元素字节、N 为保留位置数。异构层应逐层求和；窗口层以实际保留数替换 N。MLA 的逻辑状态约为 L(d_c+d_R)sN，递归状态通常不随 N 线性增长，但应另计卷积历史和快照。量化参数、页尾、目录、并行复制及临时展开另列。

设定 L=32、Hkv=8、d=128、s=2，则普通 KV 为 128 KiB/token；4096 token 为 512 MiB，32768 token 为 4 GiB。这是结构算例，未代表实际部署占用。

一次追加 M 行的全层普通 KV 新写入量约为 2LHkv d sM。事务实现需让这些新字节与旧状态同时存活；私有目录可以引用旧只读页，不必复制全部旧 KV。共享尾页的写时复制、窗口的新旧槽并存、工作区和旧读者延迟退休会增加峰值。为维持全有或全无而使用全量深拷贝也是合法软件方案，但成本随完整历史增长，不能将其报告成仅追加 M 行的生产成本。

对单层 decode，每请求一个 query，若每个 KV head 的历史在该调用中只读取一次，历史有效读取量为

D_KV≈2Hkv d sΣ_r n_r。

这是明确边界下的有效流量估计：片上或 L2 命中可以减少 HBM 事务；query head 之间缺少复用、split 分区和布局转换可以增加事务。共享前缀容量只计一次物理对象，不表示不同请求的读取会自动合并。

在长历史、忽略当前行和 softmax、并采用上述读取假设时，attention 强度近似为

I_attn≈4Hq dΣn_r/(2Hkv d sΣn_r)=2Hq/(Hkv s)。

因此单行 decode 的这部分强度不会仅因历史更长而提高。GQA 减少 Hkv，QK/PV 仍按 Hq 计算；是否兑现带宽复用需看 kernel，详见[头映射与带宽复用](gqa-kv-heads.md)。

K-rot 将旋转放在新增 K 的生产端；K-raw 在历史消费端旋转，可能每步重复 O(Hkv dN) 的逐元素工作。后者不一定增加完整中间 K 的 HBM 写回，因为可在读取时转换，但这取决于实现。两条路径数值等价不意味着成本相同。

线性层若权重 W 的元素字节为 b_w，一次理想读取权重、忽略激活流量时，M 行 GEMM 的强度约为 2M/b_w。prefill 由已知位置提供 M，decode 主要由并发请求提供 M。较大的 M 能摊薄权重读取，但会增加状态、激活和请求等待；长历史 attention 又可能成为主导。

### 准备、通信、同步与驻留

准备包括 tokenization/processor、行映射、位置坐标、mask、页表和容量预留。每块还可能重复 kernel launch、目录构造和调度；采样与 detokenization 位于输出链。算法 A 的暂存与提交另外支付元数据和完成检查成本，算法 C 则以恢复计算换取不常驻末行 logits。只看模型矩阵乘时间无法解释完整 TTFT 或 ITL。

算法 A 的单设备版本没有设备间通信。若采用每层 attention 输出及 FFN 输出各一次 all-reduce 的普通 TP 组织，两次通信的有效张量通常各为 M×H；实际网络字节及延迟还取决于分片、collective 算法和拓扑。PP 传递层边界激活，P/D 分离传递状态包及位置、格式和 pending 等元数据。迁移时不能只发送长度而漏掉各层状态或末行就绪信息。

同步包括层间依赖、生产者完成、提交与旧读者回收。资源下界可按 max(F/P_eff,D/B_eff) 建模，但有顺序依赖的区间须沿关键路径累加；共享带宽需求须合并，不能把重叠阶段的累计时间直接相加。

驻留按物理对象与时间记账：权重随模型版本存在；KV 随请求或前缀存在；末行 hidden/logits 是可选的就绪缓存；workspace 随执行存在；图和通信缓冲可能跨迭代。保存最终 hidden 大约需要 H 个元素，保存完整 logits 需要 V_vocab 个元素；前者恢复分布仍要执行输出头。若只保存特定采样配置下的结果，它不能代替支持不同采样要求的 logits。

## 条件变化后的选择

序列增长时，全 KV 容量线性增长，新 query 的历史计算与读取也增长。窗口只有在模型定义允许时才能减少直接历史；换成窗口不是对全注意力模型的无条件等价优化。最近窗口还不足以任意回滚，分叉需要对应位置的完整检查点或重放。

MLA 的 latent 降低持久容量；若每步又把完整历史展开，流量和暂存收益会改变。递归状态保持固定大小，但检查点、投机边界和回放成本可能主导恢复。混合模型需要所有层处于同一个消费边界，不能将各层独立命中的最大长度直接拼成可用状态。

batch 增大可以提高线性权重复用，却同时增加 KV、attention 及排队。chunk 变小降低单次不可抢占区间，但增加调用、准备和潜在权重重复读取。SARATHI v1 在其固定装置中展示将 prefill 块与 decode 行共同送入线性运算、分别执行 attention 的组合；历史 h 增大时，同样 q 的 attention 对数 qh+q(q+1)/2 仍会增长，所以固定 chunk token 数不是严格等长时间片。

举例，将 N=6 分成 q=2 的三块，对数依次为 3、7、11，总和为 21，与整段相同。理想线性权重读取却可能从一次变成三次，跨块历史 KV 也再次读取。若这些边界使正在服务的 decode 更快获得执行机会，交互 SLO 可能改善；若没有并发就绪工作，额外边界通常只增加成本。反转应比较同负载下的 TTFT、ITL、服务时间与拒绝情况。

位置频率、adapter、dtype、量化组、页布局或 TP head 映射变化时，应判断是可验证的表示转换，还是模型函数改变。窗口移动只改变物理存储时应保留逻辑位置；改变源音频、视觉 patch 网格、帧采样或模型位置元数据时，则可能需要放弃旧复用。

设备改变后，权重带宽、KV 读取、矩阵形状、launch 和同步的相对成本会改变。CPU 算术检查能验证下述状态和代数关系，不能给出 GPU 的带宽、可见性、重叠或最优 chunk。实际选择依赖目标设备的可持续参数和完整时间线。

## 反例与区分实验

物理槽冒充位置会在环形回绕或 batch 重排后改变模型函数。正确的槽位映射可以任意置换，φ(t) 必须随条目身份保存。标准 RoPE 下，若只测整段统一位置平移，结果相同还可能是正确的相对位置不变量；负对照应保留历史、只错改新增位置，并选择不同 K/V 贡献可被观测的输入。

逐层提交的反例是第一层已淘汰窗口、第二层失败。长度即使没有增加，旧目录已不再代表原来的前缀。只比较最终长度不能检出，必须比较全部层的旧条目及共享读者所见内容。

完整 KV 无 logits 的反例发生在首 token 尚未选出时；已有 pending 却计算零行的反例发生在输入已经就绪时。两者都可能表现为停滞，但需要不同下一事件。算法 C 解决前者，包含当前行的范围计数和算法 A 解决后者。

窗口 W=4 时，写入位置 10 可以覆盖位置 6 的物理槽；之后从较早前缀分叉，缩短长度不会恢复被覆盖的内容。递归状态原地推进也有同样边界。MLA 若仍持久保存展开 K/V，却按 latent 报告容量，则是成本对象错误；若漏掉位置分量，则是模型函数错误。

| 观察 | 竞争解释 | 区分实验 | 反证后的修正 |
|---|---|---|---|
| 同请求 chunk 第二行被拒绝 | 每行错误绑定入口长度、输入乱序、重复位置 | 空状态输入 0、1；已有长度 n 输入 n、n+1；另输入重复和间隙 | 合法两行应成功且只在末尾加 2；非法行应在任何写入前拒绝 |
| 块边界 logits 不同 | 局部位置归零、遗漏旋转、重复旋转、缓存缺失、舍入 | 固定历史比较 K-raw/K-rot、整段参考及逐行结果；用解析位置样例 | 分数或 K 首先分歧支持位置错误；提高精度仅缩小末端误差才支持舍入解释 |
| 改位置却没有差异 | 负对照幅度弱、样例对称、标准 RoPE 共同平移不变量 | 固定旧 K，仅更改新块坐标；另单独检验共同平移 | 若跨块相对距离确实改变而样例仍无区分力，重选可解析输入，不能据此放行位置重置 |
| 第二层失败后重试异常 | 逐层发布、旧窗口提前覆盖、共享对象被修改 | 注入第二层及 logits 阶段失败，比较所有层、长度、代次及共享旧视图 | 任何已提交对象变化都要求修复提交顺序；不能只恢复计数器 |
| 完整命中不推进 | 无 pending/无 logits、已有 pending 零行、有 logits 未采样 | 分别构造三种状态；恢复后与整段末行对照并检查 KV 无重复 | 无 logits 时恢复；有 pending 时消费一行；已有分布时检查采样与唤醒 |
| 窗口边界不同 | off-by-one、过早裁剪、环形槽位置混用 | W=1、2、3，对照显式 V_t；chunk 跨 W；检查恢复所需最老项 | 缺少早 query 所需旧项说明裁剪过早；只在回绕后错说明位置或槽身份混用 |
| batch 重排后输出错位 | 行归属、请求状态、位置用行号 | 交错两个不同历史请求，改变跨请求顺序但保持各请求内部顺序 | 逐请求、逐位置输出应保持；只比较整个输出列表会掩盖归属错误 |
| decode 或分块变慢 | 权重重复读取、KV 增长、准备、tile、同步与排队 | 固定历史改变 M；固定总输入改变 chunk；分别记录字节、调用及时间线 | 未观察到预期字节下降时应撤回带宽解释；随块数增长的固定成本不能归为 attention FLOPs |

交付检查以不依赖缓存状态的整段计算作参考，用相对位置分数公式独立核对显式旋转路径；覆盖多行、非零历史、窗口、请求置换、失败和无 logits 恢复。位置负对照另有闭式分母的确定性样例，避免只凭微小最终 logits 与固定阈值判断。提前发布和重复追加分别由状态不变性与位置唯一性检出。

这些检查限定为两层、单 head、小宽度的 CPU 数学模型，不能替代真实 GQA kernel、MLA、递归或跨设备验证。其 norm 和 FFN 用于检验层间数据流，不代表某个真实 checkpoint。真实设备仍需检查状态格式、同步、数值策略和资源成本；没有这些结果时不声称 GPU 实测或知识带来的 Agent 使用效果。

<!-- infraagents:provenance -->

## 来源与推导

以下按固定版本及本任务提供的材料范围使用。S-TRANSFORMER、S-HF-CACHE、S-MISTRAL 提供了完整抽取文本；其余五份原文仅提供指定字符范围的摘录，不据此声称读完整论文。正文的事务协议、成本展开、恢复证明、组合算例及负对照为本库推导，不归为来源实测。

- [S-TRANSFORMER：Attention Is All You Need](../sources/S-TRANSFORMER.md)：§3.2 定义用 query-key 分数归一化后加权 values，decoder 通过 softmax 前的遮罩阻止读取未来；encoder-decoder attention 的 Q 来自 decoder，K/V 来自 encoder。§3.5 的位置编码加在网络入口。本章据此区分因果可见集合、输入加性位置与固定源投影。
  [完整原文](../../references/S-TRANSFORMER/reader.html) · [全文文本](../../references/S-TRANSFORMER/fulltext.txt)

- [S-HF-CACHE：Transformers — Caching](../sources/S-HF-CACHE.md)：固定文档说明逐层复用历史 K/V、计算并合并新增 K/V，增量 mask 应覆盖历史及新增输入，并区分 Dynamic、Static、StaticSlidingWindow 存储形态。其示意中的 score 乘 mask 不足以定义布尔屏蔽；本章按合法集合定义 softmax。文档没有定义具体 cache_position/position_ids 合同，本章不把字段语义归因给它。
  [完整原文](../../references/S-HF-CACHE/reader.html) · [全文文本](../../references/S-HF-CACHE/fulltext.txt)

- [S-SARATHI：SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](../sources/S-SARATHI.md)：所给 v1 §3–4 等摘录从 query 行数、算术强度和批组成解释 prefill/decode 差异；chunk 后续部分继续读取此前 KV，混合批合并线性运算而分别处理 attention。材料说明 chunk、P:D 比例、tile 与首尾供给改变收益。本章的对数守恒、事务及恢复协议为独立推导，论文装置上的报告不是本章设备结果。
  [完整原文](../../references/S-SARATHI/reader.html) · [全文文本](../../references/S-SARATHI/fulltext.txt)

- [S-MISTRAL：Mistral 7B](../sources/S-MISTRAL.md)：§2 提供 GQA、窗口与滚动缓冲，逻辑位置 i 的物理槽可为 i mod W；分块图同时遮罩旧历史、块内未来与窗口外位置。原文对窗口文字和示意的计数不能替代具体 checkpoint 边界，本章明确采用包含当前行的 W 个位置约定，并按层核对状态。
  [完整原文](../../references/S-MISTRAL/reader.html) · [全文文本](../../references/S-MISTRAL/fulltext.txt)

- [S-MLA：DeepSeek-V2：MLA 与解耦位置编码](../sources/S-MLA.md)：所给 §2.1、附录 C 等摘录说明低维联合 KV latent、内容投影吸收与独立 RoPE key；位置旋转不能任意穿过内容投影。所给 §3.1.2 还说明实际 latent 后的额外 RMSNorm 与缩放。本章保留这些制品边界，普通 KV 的算法 A 不冒充完整 MLA forward。
  [完整原文](../../references/S-MLA/reader.html) · [全文文本](../../references/S-MLA/fulltext.txt)

- [S-MAMBA：Mamba: Linear-Time Sequence Modeling with Selective State Spaces](../sources/S-MAMBA.md)：所给机制与附录 D 摘录说明输入相关选择性状态、并行扫描，以及长序列分块传递边界状态；融合避免完整扩展状态反复写入 HBM。本章据此区分递归状态的驻留、临时展开和旧位置恢复，不将缩短长度视为覆盖状态的回滚。
  [完整原文](../../references/S-MAMBA/reader.html) · [全文文本](../../references/S-MAMBA/fulltext.txt)

- [S-WHISPER：Robust Speech Recognition via Large-Scale Weak Supervision](../sources/S-WHISPER.md)：所给 §2.2–2.3 及长音频相关摘录说明音频 encoder-decoder、任务条件 token 和分段音频窗口。固定源投影可缓存是由计算依赖作出的推导；窗口改变后不能无条件延用旧源状态及其 decoder 历史。
  [完整原文](../../references/S-WHISPER/reader.html) · [全文文本](../../references/S-WHISPER/fulltext.txt)

- [S-QWEN-VL：Qwen2.5-VL Technical Report](../sources/S-QWEN-VL.md)：所给 §2.1 等摘录说明动态视觉分辨率、视觉窗口、空间 patch 合并和时间相关 MRoPE。本章据此将 processor、网格、帧采样及时间坐标纳入输入与复用身份，不把所有模型位置坐标简化成消费长度。
  [完整原文](../../references/S-QWEN-VL/reader.html) · [全文文本](../../references/S-QWEN-VL/fulltext.txt)

RoPE 的分量对公式和位置内积关系还沿用本任务完整提供的 [K-ALG-016 归一化与位置旋转](../algorithms/operator-primitives.md)中的机制说明；本轮未读取额外 RoFormer 原文，不另声称其全文阅读。

正文中的公式展开、容量算例与组合设计是基于这些来源的推导；设定算例不冒充实测。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [MHA、GQA 与 MQA：头映射和带宽复用](gqa-kv-heads.md)。
- [SSM 与混合模型：递归状态、扫描与恢复](recurrent-hybrid-state.md)。
- [MLA：压缩状态、矩阵吸收与解耦 RoPE](mla.md)。
- [输入输出语义：Tokenizer、Chat Template 与流式文本](../system-semantics/tokenization-io.md)。
- [权重加载：从制品身份到可执行布局](../frameworks/checkpoint-loading.md)。
- [基础算子：从张量契约到合法融合](../algorithms/operator-primitives.md)。
- [内存分配：地址、容量与在途所有权](../system-semantics/memory-allocators.md)。
- [容量、算术强度与服务 Roofline](../hardware/memory-and-roofline.md)。
