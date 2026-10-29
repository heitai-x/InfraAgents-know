# K-ALG-011：批量采样：概率变换、请求随机性与行映射

批量采样把多个条件分布排成设备可执行的行，再把选出的 token 交给对应的请求、分支和逻辑位置。向量化可以减少启动、传输和主机同步，但不能改变概率变换顺序、合法支持集、随机轨迹或输出归属。采样器的输入是 logits 或具有明确语义的概率行；它不负责证明这些行由正确的模型前向产生。

## 对象、不变量与状态

设请求为 r，目标分支为 b，要生成的位置为 t，同一位置的样本序号为 s。一次执行快照用 β 标识；请求的当前授权代次为 g_r，分支的当前授权代次为 h_rb。采样事件携带 o=(r,b,t,s,g_r,h_rb)，执行结果另带 β。β 标识哪份行映射和缓冲，g_r、h_rb 标识哪个对象仍有提交权，三者不能相互替代。不同请求可以分别处于代次 2 和 9，并合法地参加同一个 β=77 的 batch。

| 描述符字段 | 含义与校验 |
|---|---|
| 请求、目标分支、逻辑位置、样本序号 | 输出归属与随机事件身份，不使用物理行号替代 |
| 请求代次、分支代次 | 分别与所属请求、分支的当前授权比较 |
| 分支状态版本、预期长度 | 防止同一代次内重试、重复提交或 grammar 更新后继续消费旧结果 |
| 快照 β、来源概率行 | 在本轮不可变映射中定位分布，不与授权代次比较 |
| 来源路径、来源位置 | 说明哪个条件前缀产生了 logits；可以与分叉后的目标分支不同 |
| 参数、历史、grammar 快照 | 确定本次概率变换；包含影响合法性和数值的版本 |
| 输出槽 | 用于反向路由，不授予槽位当前占用者接收旧结果的权利 |

设最终分布为 q_o(k)，词表大小为 V，k∈{0,…,V−1}。它在合法支持之外为零，在支持内非负，总质量为一；输出 X_o∼q_o。需要保持以下不变量：

1. 每个输出事件的身份唯一。padding、取消、终止和失去授权的行不产生正式输出。
2. 分布按声明的变换顺序构造；空合法集合和零总质量必须失败，零概率 token 不能被选中。
3. 要求调度无关复现时，随机数按逻辑事件定位；重排物理行不改变同一事件使用的随机数。
4. 构造分布、读取随机数、暂存结果和正式提交分开。失败或被丢弃的结果不推进正式随机账本、grammar、历史或输出长度。
5. token、grammar 后继、长度、随机消费记录和分支终止在同一个提交边界更新。提交成功不表示该 token 自身的 KV 已计算。
6. 快照、logits、mask 和 scratch 的最后一个设备或传输消费者结束后，才可覆盖其缓冲。业务丢弃结果不等于设备工作已经结束。

正式状态按分支保存 token 历史、grammar、随机账本、长度、授权代次、状态版本和终止原因。请求保存必需分支集合及完成策略。本章参考策略是：非空的必需分支集合中的所有分支均结束，才结束请求。建立本轮快照时冻结该集合；修改集合须经过请求所有者并使旧描述符失效。取消整个请求是另一个显式事件。若采用首个成功样本即返回等策略，应明确取消其余分支，不能由 staged 遍历顺序隐式决定。

## 概率变换：顺序、合法集合与数值范围

对事件 o，设原始 logits 为 z，历史变换为 H，合法集合为 A，温度为 T。参考路径先计算 a=H(z)，再屏蔽 A 之外的项。H 可以是具有明确公式的重复惩罚或存在惩罚；作用对象、公式和参数版本属于输入契约。没有历史项时 H 为恒等变换。历史变换必须读取快照，不能在构造分布时修改正式历史。

本章允许合法项的分数为有限数或 −∞，拒绝 NaN 和 +∞；至少一个合法项必须为有限数。T 必须为有限非负数。T>0 时，以 m=max_{k∈A}a(k) 构造

w(k)=exp((a(k)−m)/T)，p(k)=w(k)/Σ_{j∈A}w(j)。

非法项的 w、p 均为零。先减最大值再除温度，避免把很大的正 logits 直接缩放到 +∞。极端负差值在有限精度中可能成为 −∞，指数可能下溢为零；这意味着数值分布的支持可能小于实数 softmax 的支持。比较 GPU 与 CPU 时，应说明是在比较实数目标、某个浮点参考，还是允许误差的分布。

T=0 使用 greedy：在合法集合的有限分数中取最大值，平局选最小 token id。其支持为单元素集合 {token}，等价分布为该点质量一，随机消费为零。本章不在 greedy 路径应用 top-k/top-p；它仍检查参数格式和合法集合。全为 −∞ 时失败，不能把任意合法 id 当作成功结果。

没有历史项时，给所有 logits 加同一常数不改变实数 softmax；依赖分数绝对值的历史变换未必保留此性质。同乘常数会改变熵。temperature、历史项、mask 和截断不能任意交换。

令 top-k 参数 k≥1，top-p 参数 ρ∈(0,1]。对当前数值分布的正质量项，按概率降序、同概率按 token id 升序排序。K 为前 min(k,正质量项数) 个 token。截断顺序是分布定义的一部分：

- k_then_p：先得到 K，把 p 在 K 上归一化为 p^K，再取累计质量达到 ρ 的最小前缀 C，最终支持 S=C。
- p_then_k：先在完整 p 上确定 top-p 前缀 C，再取 S=C∩K。

最后把 p 在 S 上重新归一化得到 q。两种路径使用的 top-p 质量基准不同，不能只凭接口名字判定等价。ρ=1 必须保留该阶段的全部正质量项，即使累计舍入已提前达到一。由于 K 和 C 都包含按同一规则选出的首项，合法非零输入的交集非空；参考仍检查最终质量，以捕获错误输入或实现偏离。

设原分布为 0.5、0.3、0.2，k=2、ρ=0.6。k_then_p 在 K 上归一化后为 0.625、0.375，只保留 token 0；p_then_k 先在原分布保留 token 0、1，再与 K 相交，仍保留两项。若低温使第一项质量超过 0.6，top-p 的支持也会改变。

合法集合必须先于归一化和阈值确定。对原分布 0.4、0.35、0.25，禁止 token 0，ρ=0.4：先在合法集合归一化得到约 0.583、0.417，前缀为 {1}；先按原分布截断得到 {0} 再屏蔽，会留下空集合。空集合应返回错误，由上层选择恢复或失败策略；不得偷偷发出非法 token。EOS 只有在当前 grammar 允许结束时才属于合法集合。

## 逆 CDF 与 top-p 的有限精度契约

实数逆 CDF 返回第一个满足 u<C_j 的类别，u∈[0,1)，C_j 是按 token id 顺序的累计质量。u 恰好等于某项累计端点时，进入下一个非空区间。零质量类别没有区间。

本章 CPU 数值参考使用 binary64：归一化总量用准确求和，除法和显式逐项累计使用 binary64；累计端点限制在 [0,1]，最后一个正质量项的端点显式设为 1。逆 CDF 跳过零质量项。因此尾部舍入剩余只能落到最后一个正质量类别，不能回退到数组末尾。极小正质量的区间仍可能因累计舍入而塌缩；这份参考没有宣称精确实现所有实数概率。这里的正质量是归一化后的数值质量，归一化下溢造成的支持变化应单独记录。

top-p 使用同样的归一化和累计规则，但按概率降序遍历。ρ<1 时以 C_j≥ρ 判定，不减去任意容差；ρ=1 单独返回全部正质量项。若目标实现采用不同归约或比较精度，应另行声明差异，不能把支持变化藏在未声明容差中。

```pseudocode
FUNCTION NORMALIZE(weights):
    if weights empty or any weight nonfinite or negative:
        return ERROR(invalid_mass)
    total = ACCURATE_SUM(weights)
    if total is not finite or total <= 0: return ERROR(invalid_mass)
    return OK([ROUND64(weight / total) for weight in weights])

FUNCTION ENDPOINTS(normalized_weights, positive_order):
    cumulative = 0
    for j in 0..length(positive_order)-1:
        token = positive_order[j]
        cumulative = min(1, ROUND64(cumulative + normalized_weights[token]))
        if j == length(positive_order)-1: endpoint = 1
        else: endpoint = cumulative
        yield token, endpoint

FUNCTION INVERSE_CDF(weights, u):
    if not 0 <= u < 1: return ERROR(invalid_uniform)
    normalized = NORMALIZE(weights)
    if normalized is ERROR: return normalized
    order = positive entries of normalized.value, by increasing token id
    for token, endpoint in ENDPOINTS(normalized.value, order):
        if u < endpoint: return OK(token)
    return ERROR(internal_cdf_violation)

FUNCTION PREFIX_UNTIL(weights, rho):
    if not 0 < rho <= 1: return ERROR(invalid_top_p)
    normalized = NORMALIZE(weights)
    if normalized is ERROR: return normalized
    order = positive entries sorted by (-probability, token_id)
    if rho == 1: return OK(order)
    prefix = empty
    for token, endpoint in ENDPOINTS(normalized.value, order):
        prefix.append(token)
        if endpoint >= rho: return OK(prefix)
    return ERROR(internal_prefix_violation)
```

两条具体反例检验这个契约。第一，从零开始用显式 binary64 左折叠累计 [0.1]×10+[0]，末值为 0.9999999999999999。取 u 为小于一的最大 binary64 值，即同一数值，旧逆 CDF 的循环不命中，末索引回退选中零质量 token 10；修订参考把最后正质量 token 9 的端点封为一，返回 token 9。

这个反例不能用语言内置求和的历史行为代替显式左折叠。Python 3.12 改进了内置 sum 的浮点求和，sum([0.1]×10+[0]) 可能得到 1.0；把它直接用作 u，既不再构成原反例，也违反 u<1。检查应分别记录内置求和与显式左折叠，并用固定合法端点构造输入，不把内置 sum 必须返回旧值作为验收条件。

第二，对 [1−5×10⁻¹³,5×10⁻¹³]，ρ=1 时必须保留两项；用累计值≥ρ−10⁻¹² 提前停止会错误删除第二项。ρ 稍小于一但仍大于第一项时，同一容差也会改变支持集。检查同时验证这两个阈值，避免只给 ρ=1 加特判后继续隐藏普通阈值的容差。

## 行映射：从输入 query 到独立后继

本轮按输入路径分段；路径可以是请求内的不同分支。第 j 段有 q_j 个 query 行，有效长度为 v_j，0≤v_j≤q_j；偏移 o_j=Σ_{l<j}q_l，总输入行数 Q=Σ_jq_j。右 padding 下，末有效来源行是 o_j+v_j−1，而不是 o_j+q_j−1。左 padding 必须显式记录有效起点，不能套用这个简化式。零有效长度没有可直接采样的末行。

最后输入位置的 logits 预测下一位置。映射同时记录来源输入位置和目标生成位置，避免把“输出了 token”误写成“该 token 的 KV 已存在”。prompt logprob 行和投机验证行不自动成为普通采样事件。

同一概率行可以服务多个样本，但多个样本不能都追加到同一个正式分支。在建立快照前，为每个样本登记独立的目标子分支，保存父路径和分叉位置：共享不可变前缀，分别拥有可变 token 尾部、grammar、长度、随机子流、状态版本和终止位。每个描述符引用父前缀的同一来源行，指向不同子分支。分支建立属于准备工作，不是成功采样的状态推进。

```pseudocode
BUILD_ROW_MAP(beta, input_segments, needed_events, authority_snapshot):
    validate unique segment IDs and each segment's valid/query lengths
    input_rows = empty
    row_of_event = empty
    events_of_row = empty
    offset = 0
    for segment in input_segments in fixed order:
        for k in 0..segment.query_count-1:
            input_rows.append(source_identity(beta, segment.path,
                              segment.logical_start + k,
                              valid=(k < segment.valid_length)))
        offset_of_segment[segment.id] = offset
        offset = offset + segment.query_count
    seen_events = empty set
    seen_targets = empty set
    for event in needed_events:
        if event.id in seen_events: return ERROR(duplicate_event)
        seen_events.add(event.id)
        target = (event.request, event.target_branch)
        if target in seen_targets: return ERROR(duplicate_target_in_snapshot)
        seen_targets.add(target)
        segment = input_segments[event.source_segment]
        if not 0 <= event.source_offset < segment.query_count:
            return ERROR(invalid_source_offset)
        row = offset_of_segment[segment.id] + event.source_offset
        if not input_rows[row].valid: return ERROR(padding_source)
        if source path/position does not match event's declared parent prefix:
            return ERROR(source_identity_mismatch)
        if not AUTHORIZED(event, authority_snapshot): return ERROR(stale_event)
        descriptor = attach beta, row, target authority, state version,
                     expected length, context snapshot and output slot to event
        row_of_event[event.id] = descriptor
        events_of_row[row].append(event.id)
    return OK(input_rows, row_of_event, events_of_row)
```

这个参考 batch 对每个目标分支最多生成一个后继 token。需要同一路径连续多个位置时，后一个分布依赖前一个已选 token，应逐步执行，或使用具有独立候选状态和接受协议的投机算法。不能把多个相互依赖的位置当作普通独立行提交。beam 的候选评分、全局剪枝和停止条件也不由独立分类采样替代。

压紧 A、B、C 为 A、C 时，C 的物理行可能从 2 变成 1；参数、grammar、随机身份和输出映射必须与它一起迁移。取消不能抹掉在途快照记录。旧结果仍带原 β、请求、分支和授权代次，完成后可释放旧资源，但不能写入已经复用的输出槽。

## 批量算法：失败、暂存与提交

下列返回值均为带状态的结果，不将 ERROR 当作 token 或概率向量解包。构造函数不修改正式状态；合法集合、参数和分数失败均发生在读取随机数之前。上下文包含词表长度、历史变换契约、grammar 快照、长度限制及终止规则；输入行的长度和 token id 范围也要先验证。

```pseudocode
FUNCTION PREPARE_ROW(z, context):
    validate z length, legal token IDs and context parameter formats
    if validation fails: return ERROR(invalid_input)
    if T is not finite or T < 0: return ERROR(invalid_temperature)
    if k is not an integer or k < 1: return ERROR(invalid_top_k)
    if rho is not finite or not 0 < rho <= 1: return ERROR(invalid_top_p)
    if mode not in {k_then_p, p_then_k}: return ERROR(unknown_mode)
    A = LEGAL_SET(context.grammar_snapshot, context.vocab_size)
    if A is empty: return ERROR(empty_legal_set)
    a = HISTORY_TERMS(z, context.history_snapshot, context.history_contract)
    if history transform failed: return ERROR(history_transform_failed)
    if any legal score is NaN or +inf: return ERROR(invalid_score)
    if no legal score is finite: return ERROR(no_finite_legal_mass)
    if T == 0:
        token = finite legal argmax of a, tie=lowest_token_id
        return OK(kind=greedy, token=token, q=point_mass(token),
                  support={token}, draws=0)
    m = maximum legal score
    weights = zero vector of length V
    for token in A:
        weights[token] = exp((a[token] - m) / T)
    p = NORMALIZE(weights)
    if p is ERROR: return p
    K = first min(k, number of positive entries) entries of p.value,
        sorted by (-probability, token_id)
    if mode == k_then_p:
        restricted = p.value with entries outside K set to zero
        C = PREFIX_UNTIL(restricted, rho)
        if C is ERROR: return C
        S = set(C.value)
    else:
        C = PREFIX_UNTIL(p.value, rho)
        if C is ERROR: return C
        S = set(C.value) intersect K
    final_mass = p.value with entries outside S set to zero
    q = NORMALIZE(final_mass)
    if q is ERROR: return q
    return OK(kind=random, token=NONE, q=q.value, support=S, draws=1)

FUNCTION STILL_CURRENT(d, authority, beta):
    if request or target branch no longer exists: return false
    r = authority.request[d.request]
    b = r.branch[d.target_branch]
    return d.beta == beta and not r.cancelled and not r.terminal
           and d.request_generation == r.current_generation
           and d.branch_generation == b.current_generation
           and d.state_version == b.state_version
           and d.expected_length == b.length and not b.terminal
           and b.length < b.length_limit

PROCEDURE BATCH_SAMPLE(beta, descriptors, logits, authority):
    require immutable snapshot beta and no duplicate event or target branch
    staged = empty
    outcomes = empty
    for d in descriptors:
        if not STILL_CURRENT(d, authority, beta):
            outcomes[d.id] = DISCARDED(stale_or_terminal)
            continue
        prepared = PREPARE_ROW(logits[d.source_row], d.context_snapshot)
        if prepared is ERROR:
            outcomes[d.id] = ROW_ERROR(prepared.reason)
            continue
        if prepared.kind == greedy:
            token = prepared.token
        else:
            u = LEDGER_PEEK(d.logical_random_key, purpose=sample, draw_index=0)
            selected = INVERSE_CDF(prepared.q, u)
            if selected is ERROR:
                outcomes[d.id] = ROW_ERROR(selected.reason)
                continue
            token = selected.value
        successor = PURE_SUCCESSOR(d.context_snapshot, token)
        if successor is ERROR:
            outcomes[d.id] = ROW_ERROR(successor.reason)
            continue
        staged.append(SUCCESS(d, token, prepared.support, prepared.draws,
                              successor.grammar, successor.history,
                              successor.length, successor.finish_reason))
    touched_attempts = empty set
    for result in staged:
        d = result.descriptor
        acquire commit ownership for d.request and d.target_branch
        if not STILL_CURRENT(d, authority, beta):
            outcomes[d.id] = DISCARDED(lost_commit_authority)
            release commit ownership
            continue
        atomically publish result.token, grammar, history, length,
            ledger event and actual draw count to this target branch
        increment this branch's state_version
        if result.finish_reason exists:
            terminate this branch with result.finish_reason
        outcomes[d.id] = COMMITTED(result)
        touched_attempts.add((d.request, d.request_generation))
        release commit ownership
    for (r, generation) in touched_attempts:
        under request commit ownership:
            if r.current_generation != generation or r.cancelled: continue
            if all frozen required branches of r are terminal:
                terminate request r with aggregate branch outcomes
    return outcomes routed by original event identity
```

PURE_SUCCESSOR 根据所选 token 计算 grammar 转移、追加历史和长度，再判定 EOS、长度限制等分支终止原因。对于合法集合构造后仍出现的 grammar 转移错误，返回行错误；已经读取的随机数只是无副作用的 PEEK，不推进正式账本。终止原因的优先级和 EOS 是否计入 token usage 由输出协议声明；EOS 不必作为可见文本发给用户，但逻辑提交及 finish reason 必须一致。

错误行直接返回 ROW_ERROR，不进入成功暂存列表；其 ledger、grammar、历史、长度和状态版本保持不变。其他合法分支仍可成功，这是逐分支提交语义，不是整个 batch 的事务。若业务要求失败后结束分支，应由独立错误策略处理，不能把失败当成采样成功。若请求已失去授权，迟到错误也不得改变新尝试。

检查与写入必须共享锁、串行所有者或等价的版本比较发布机制。仅先检查、随后无保护地写入，仍会与取消或接管竞争。一个版本只允许一次正式提交；重放同一 staged 结果会因状态版本或长度改变而被丢弃。请求聚合也重新检查所属尝试，避免旧 batch 的聚合动作结束新尝试。物理缓冲的设备排空另行完成，不由上述业务锁替代。

## 分支 EOS、随机子流与组合轨迹

设请求 R 有必需分支 b0、b1，另一个请求 C 只有 c0。β=77 中，R 的请求代次为 2，C 为 9；三个描述符分别匹配各自分支代次和状态版本。b0 greedy 产生 EOS，b1 产生普通 token，c0 随机采样产生普通 token。

先提交 b0：只更新 b0 的 token、grammar、长度、版本和终止位，其随机消费为零；R 的 b1 尚未结束，因此 R 保持活动。随后 b1、c0 正常提交。交换 b0 与 b1 的 staged 次序，得到相同的分支状态。若 c0 返回前 C 被接管到代次 10，c0 的旧结果被丢弃，C 新代次的账本、grammar 和长度均不改变。快照 77 的设备消费者完成后才能复用它的缓冲。

如果 b0、b1 是从同一父分布产生的两个样本，二者从分叉点开始拥有独立后继状态。一个分支遇到 EOS，不会阻止另一个继续；两个分支恰好抽到同一 token，也不表示可以合并可变状态。参考逐分支提交不要求成功事件的外部到达顺序一致，但按逻辑身份整理的结果和最终状态应一致。

要求调度无关复现时，可以使用逻辑定位的随机账本：

```pseudocode
KEY = (request_seed, stable_request_id, stable_branch_path,
       logical_position, sample_ordinal, purpose, draw_index,
       random_algorithm_version)
UNIFORM = KEYED_PRNG(KEY)
```

β、物理行号和输出槽不进入该逻辑键。授权代次用于隔离提交；若恢复策略承诺重放同一逻辑事件，则恢复代次也不改变随机键。若策略有意重采样，应创建新的样本身份并明确放弃逐事件重放承诺。实际 PRNG 的统计质量和子流隔离需单独验证；键不同本身不是数学独立性的证明。

逆 CDF 每个随机事件消耗一个 uniform，greedy 消耗零个。LEDGER_PEEK 是按键读取或克隆游标后读取，只有提交成功才记入正式账本。拒绝采样可能使用可变次数随机数；应记录实际 draw_index 和算法版本。共享概率行只能复用分布，不能复用某次抽样结果。投机接受、拒绝后重采样和正式采样应使用不同 purpose，废弃候选的消费不能推进保留路径的账本。

随机投机解码把这些规则组合起来：对相同条件前缀下的实际目标分布 p 和草稿分布 q，候选 x∼q 的接受概率为 min(1,p(x)/q(x))；拒绝后从与 max(p−q,0) 成比例的残差分布抽样。候选来自 q 时 q(x)>0，接受比有定义。令 α=Σ_x min(p(x),q(x))，拒绝质量为 1−α；当它为正时，残差归一化为 r(x)=(p(x)−min(p(x),q(x)))/(1−α)。最终每类质量为 min(p(x),q(x))+(1−α)r(x)=p(x)。当 α=1 时不会拒绝，无需对零残差归一化。

p、q 应包含各自实际的 temperature、历史项、mask 和截断。接受前缀、grammar、随机状态及模型状态的确认边界必须一致。独立采样伪代码不能直接替代多位置投机的接受、回滚和共同提交协议；链式修正也不能直接证明任意树形搜索合法。

固定 seed 不保证跨版本、平台或 CPU/GPU 逐位一致，也不保证不同采样算法使用相同随机数轨迹。相同 seed 的文本不同不能单独证明分布错误，文本相同也不能证明映射或随机子流正确。

## 计算、访存、通信与驻留成本

设输入路径段数为 N，总 query 行数为 Q，有效输出事件数为 B，词表大小为 V。令 M≤B 为不能进一步复用的概率变换数：只有来源行、历史项、合法集合、温度和截断参数均相同的事件才能共享一份 q。仅来源 logits 相同而 grammar 不同，不能按同一概率变换计费。设 K 是 top-k 上界，S 是最终支持规模上界。

给出的 BUILD_ROW_MAP 显式遍历所有 query，再遍历事件，因此准备时间和存储均为 O(N+Q+B)，不是 O(B)。长 prefill 可以 Q 很大而每个请求只输出一个 token。若只存分段偏移和必要事件，可把映射存储降为 O(N+B)，但这是另一种构造；通过事件给定的段号和局部位置可 O(1) 定位，通过全局行反查段则通常需搜索。不能保留逐 query 算法却使用压缩表示的成本。

参数上传约 O(Bd)，d 是每事件参数和身份元数据大小；grammar 编译、历史整理和子分支建立另计。显式复制长度 L 的 grammar/历史快照，每个子分支可付 O(L)；共享不可变前缀与私有尾部能避免全部复制，但增加引用管理和写时复制。dense mask 构造需 O(MV) 写入，稀疏合法集合可按其表示规模计费；缓存必须计入驻留容量和失效检查。

计算账依所选路径展开：

| 路径 | 条件化工作量 |
|---|---|
| greedy | 扫描合法分数和归约 argmax，dense 情况 O(BV)，无指数、排序或随机抽取 |
| 稳定 softmax | 最大值、指数、质量归约与归一化 O(MV)，历史项另加 O(nnz_history) |
| 堆式 top-k | O(MV log max(2,K)) 时间、O(MK) 候选存储；线性选择的平均或最坏界按具体算法声明 |
| k_then_p | 选 K 后在 K 上排序 O(MK log K)，扫描 O(MK)；ρ=1 时无需这一步排序 |
| p_then_k | 本参考先对全词表正质量项排序，最坏 O(MV log V)，再扫描确定 C 和相交；小 K 不能消除全局 top-p 的工作 |
| 最终抽样 | 一次构造 CDF O(MS)，B 次线性扫描 O(BS)；存储 CDF 后二分为 O(B log S) |
| 多次复用同一 q | alias 表构建 O(MS)、抽样 O(B)，但需要声明随机消费与数值契约 |

伪代码为了清晰可重复归一化、排序及创建 dense 向量；照字面实现会重复这些遍历。上表允许缓存排序次序、用支持 id 列表构建 CDF，属于显式优化后的阶段账；若每次对 V 长向量执行 NORMALIZE 或复制，须把额外 O(V) 工作计入。共享 q 不等于每个样本免费，B 次随机读取、抽样和私有后继提交仍然存在。

若采用无需全排序的全局阈值算法或融合拒绝采样，应按实际遍历次数、候选规模和接受率记成本。多轮全局质量扫描可写为 O(MVI)，I 是数据相关的扫描轮数，不能当作固定常数。精确 top-p 还需处理阈值边界和平局，使保留集合对应最小排序前缀。本章参考选择全排序，不声称来源归档中的融合设备循环就是这个实现。

若模型保留全部 query 的 logits，logits 容量为 QVb_z，其中 b_z 是每元素字节数；若 output head 只计算选中的来源行，或先收集来源行再释放原矩阵，则可按保留来源行数计费，但还须计入收集或生命周期成本。最朴素的不复用路径中，概率、CDF 和排序索引各需 BV 个元素；复用后可按 M 计费。

以选中来源行数=M=B=64、V=128000 为设定：fp16 logits 为 15.625 MiB；一份 fp32 数组为 31.25 MiB。若同时保留概率、CDF 和 4 字节索引各一份，加上 logits，共 109.375 MiB。若此前 Q≫64 的完整 logits 尚未释放，峰值还包括它，不能用这个数替代总峰值。此处是容量推导，没有 GPU 实测。

访存账要区分容量与搬运。未融合的最大值、指数、归一化、截断和 CDF 可能多次读取分数、读写概率及索引；每个 dense pass 是 Θ(MV) 个元素事务的逻辑规模。一次写出概率再读回至少增加相应数组的一写一读。融合可以减少中间全局数组，实际 DRAM 字节数仍受 cache、合并访问、排序重排和 spill 影响。

一种可理解的设备映射是每个工作组处理一条概率变换行，线程按 k=lane+j×组宽遍历 token；地址为 base+row×row_stride+k×元素字节数。先局部求最大值并做组内归约、广播，再扫描计算指数及质量并归约，随后归一化或输出候选。组内阶段之间需要完成与可见性保证；跨工作组的全局排序、候选合并或分片归约通过后续阶段及依赖连接。最后按事件索引读取共享 q，分别抽样、写入独立暂存槽。过长行需要多个工作组时，最大值和质量还要跨组合并，不能把组内屏障当作全设备屏障。该映射是设计推导，不是对归档 GPU kernel 线程布局的声明。

词表沿 D 个 rank 分片时，全局 softmax 至少需要全局最大值和指数质量；collective 的网络量、延迟由拓扑与算法决定。全量概率合并时，每个接收者需取得约 MV(1−1/D) 个远端元素；top-k 合并候选约为 MD×min(K,V/D) 个带全局 id 的项。不能把每个接收者的数据量与整个集群总量混记。p_then_k 的全局 top-p 仍需完整排序信息或可证明的全局阈值过程，各片独立归一化或 top-p 后拼接通常改变分布。

当 ρ=1 且 top-k 不缩小支持时，可先按 shard 质量选择 shard，再在其中按条件分布采样，以 O(MD) 个质量元数据避免全量概率合并。将同一 uniform 的区间重映射到被选 shard，或使用另一个明确 purpose 的 uniform，都能形成分布参考，但随机轨迹承诺不同。若需 top-k/top-p，必须先形成正确的全局支持，不能直接复用这个简化。

同步成本包括 kernel 启动、阶段依赖、token 回读、提交所有权及下一轮准备。逐请求主机同步可形成 B 次往返，设备批量筛选、采样和终止标记能减少往返，但不能取消最终授权检查。参数、账本、grammar 和私有分支状态随活跃对象增长；概率、排序 scratch 和在途快照随 M×V、流水深度增长。计算、传输和主机准备可能重叠，墙钟时间应按依赖关键路径测量，不能把所有局部耗时直接相加。吞吐分子应为满足任务条件的已提交合法 token。

## 条件变化后的保留、调整与放弃

- ρ=1 且 k 覆盖全部正质量项时，保留批量分类采样，跳过截断排序。T=0 时直接使用合法 argmax。
- K 很小直接帮助 k_then_p，或只需 top-k 的路径。p_then_k 在 ρ<1 时仍承担全局 top-p；把它换成 k_then_p 是改变分布，需要业务允许。
- 同一来源行和上下文服务许多样本时，保留共享 q 或 CDF，分别使用随机键和后继状态；每个样本 grammar 不同后，应拆分概率变换缓存。
- 参数异构可以保留一个 batch；当排序长度、mask 密度或执行分歧造成的浪费超过分组、重排和逆映射成本时再分组。改变分组不应改变事件随机键。
- Q≫B 的长 prefill 应考虑分段映射和只计算必要 output-head 行。若消费者需要全部 prompt logprob，则不能使用只保留末行的简化。
- grammar 稀疏时可以采用位图、区间或合法 id 列表；共享编译规则仍须为各分支保留独立解析状态。tokenizer、词表或规则版本变化后，旧 mask 缓存失效。
- 分片且无截断时可以采用 shard 质量采样；需要精确 top-p 时保留全局约束。若选择近似阈值，必须声明支持或质量误差，不能继续声称相同分布。
- 精度、归约顺序或 PRNG 算法变化后，重新核对平局、CDF 端点、尾部正质量和随机消费。语言运行时的求和实现变化也应与所声明的显式累计契约区分。
- 若不要求逐事件复现，可以选择更简单的游标流，但应明确它对调度的依赖。不能一面允许共享游标，一面承诺重排不改变逐事件结果。
- 分支、投机或异步流水增加驻留与失败管理成本。接受率低、目标 batch 已饱和或状态复制昂贵时，可以减少宽度、深度或关闭投机；不能只按采样 kernel 时间决定。
- 取消和接管后，保留旧快照到最后消费者完成，撤销其业务提交权。释放与提交权是两个判断，不能通过重用行号合并。

## 竞争解释、区分实验与反证修正

| 观察 | 竞争解释 | 可区分的输入或干预 |
|---|---|---|
| 首行 greedy 出错或支持集沿用上一行 | 成功字段未初始化、分支返回类型不统一 | 分别运行首行 greedy、随机后接 greedy；验证 support={token}、draws=0 |
| 全屏蔽后长度或账本变化 | 错误被当作 token、提前消费正式 RNG | greedy 与随机路径都输入空合法集合；比较失败前后的完整分支状态和随机读取记录 |
| 一个样本 EOS 后另一个没有输出 | 把分支终止提升为请求终止 | 同一请求一支 EOS、一支普通 token，交换提交次序；活动兄弟必须继续且请求未完成 |
| 一个 batch 无故丢失部分请求 | 公共轮次被用作每请求代次 | 同一 β 放入代次 2、9 的两个合法请求，再让其中一个在返回前升代；只有旧授权结果被丢弃 |
| 重试或取消后重复推进 | 检查与发布非原子、缺少状态版本 | 暂存后取消、升代或改变版本，再提交；重复提交同一 staged 结果应无额外推进 |
| 长 prefill 准备时间随输入长度增长 | row map、完整 logits 或 mask 构造支配 | 固定 B、V，改变 Q；分别记录映射访问数、logits 产出行数和阶段时间 |
| 小 K 仍有高排序成本 | 实际路径为 p_then_k、全局阈值扫描 | 固定 K、改变 V 和 ρ，分别运行两种顺序；核对支持集及全局读取规模 |
| 偶发零概率 token 或 p=1 丢尾项 | CDF 末索引回退、未声明阈值容差 | 使用两条 binary64 反例；验证选中质量严格为正，p=1 包含所有正质量项 |
| 反例在换 Python 后首先失败 | 内置 sum 与显式逐项累计不同 | 同时记录两种求和，但固定 u 为小于一的最大 binary64 值；直接检查旧算法是否选中零质量项 |
| 行重排后结果变化 | 行映射错或共享行游标 RNG | 分别固定 uniform、固定逻辑键，再交换物理行；先区分分布归属与随机定位 |
| 分片后支持集增加 | shard 局部归一化或局部 top-p | 对 0.6、0.3、0.1 分两片、ρ=0.5；全局支持为 {0}，局部并集会含 token 2 |

检查应主动检测错误实现。等宽切分 [0,1) 的伪逆 CDF 在 [0.9,0.05,0.05]、u=0.5 上会返回 token 1，质量参考返回 token 0。行游标 RNG、共享同一概率行的随机抽取、EOS 立即终止整个请求、把请求代次与 β 比较，以及失败时推进账本，都应各有能够触发差异的负对照。

独立有理数参考可以在质量和阈值均为可精确表示的二进制分数时穷举 CDF 区间和 top-p 支持；这能核对算法，不依赖再次调用同一个浮点实现。对一般浮点输入，还需验证所声明的舍入契约和支持不变量。大量随机样本的频率检验可补充 PRNG 与分布检查，但固定 seed 的一条文本不能替代它。

若区分实验发现差异，应按最早偏离修正判断：同一 uniform 已选错类别时先检查分布和 CDF；uniform 随行重排改变时检查随机定位；暂存正确而正式状态不同则检查授权、版本和分支终止。反例构造首先失败时，先核对数值输入，不能把尚未执行到算法断言的失败记为算法通过。设备更快不能抵消支持、身份或提交语义错误。

## 验证范围与未测边界

配套检查 tools/check_k_alg_011_full.py 使用标准库、确定性小输入和独立有理数参考，覆盖 CDF、top-p、顺序差异、合法集合、行映射、逻辑随机定位及分支提交模型，并核对单步投机修正的质量恒等式。它以显式左折叠构造尾部反例，不依赖内置 sum 的版本行为。

状态模型使用恒等历史项、追加式 grammar 和所有必需分支结束的请求策略；它没有实现具体 grammar 解析器、beam 评分或多位置投机协议。串行状态模型只验证授权和提交的逻辑效果，不能证明真实锁、CAS、设备屏障、内存可见性或缓冲排空。

CPU 模型不能证明 GPU 的浮点归约、PRNG 质量、排序 kernel、真实通信量、同步或性能。设备验证应绑定环境、shape、dtype、版本和失败样本，并比较最终支持与提交事件；本章没有目标 GPU 实测、端到端收益或 Agent 构建能力的测量结论。设定容量和复杂度为推导；脚本是否实际执行及结果由本轮证据记录，不由正文的覆盖说明代替。

<!-- infraagents:provenance -->

## 来源与推导

- [S-FI-SAMPLE：FlashInfer：Sampling](../sources/S-FI-SAMPLE.md)：FlashInfer sampling 接口提供从概率或 logits 的按行采样，以及 top-k、top-p 等相关操作；调用的输入表示和参数决定实际概率处理。按行采样要明确每个输出对应的分布行；索引允许多个输出复用同一概率行，但不等于复用一次随机结果。截断顺序及阈值所依赖的归一化分布影响语义。显式随机种子与偏移由调用方维护，重复传入同一状态不会自动形成新随机流。来源说明将归档融合 GPU 路径描述为无需显式排序的拒绝采样，同时明确包装文件不足以核验全部设备内循环。本章使用其机制说明，没有读取或冒称还原上游设备源码。
  [算法与机制说明](../sources/S-FI-SAMPLE.md)

- [S-TORCH-REPRO：PyTorch — Reproducibility](../sources/S-TORCH-REPRO.md)：随机种子与确定性配置可以控制特定执行路径中的随机性，但不保证跨版本、平台和全部算法路径逐位一致，包括 CPU/GPU 差异；确定性算法可能更慢。原文还讨论不同库的随机源及未初始化内存。按逻辑事件组织请求随机流是本章据此提出的设计选择，不是来源承诺 seed 自动提供调度无关性。
  [完整原文](../../references/S-TORCH-REPRO/reader.html) · [全文文本](../../references/S-TORCH-REPRO/fulltext.txt)

- [S-SPEC：Fast Inference from Transformers via Speculative Decoding](../sources/S-SPEC.md)：随机投机解码用草稿分布 q 产生候选，由目标分布 p 验证。候选 x 的接受概率为 min(1,p(x)/q(x))，拒绝时从正剩余概率质量中重新采样；greedy 采用自己的确定性验证规则。§2.2–2.3 将采样方法表示为调整后的概率分布，附录 A.1 推导输出分布保持。正文只在随机子流和共同提交边界引用该机制，不将论文的 T5/TPU 测量当作本库实测，也不把链式规则直接当成任意树形搜索规则。
  [完整原文](../../references/S-SPEC/reader.html) · [全文文本](../../references/S-SPEC/fulltext.txt)

正文中的公式展开、行映射、分支提交协议、数值端点修正、复杂度和容量算例是基于这些来源的推导；设定算例不冒充实测。k_then_p 与 p_then_k 是本章明确区分的两种可声明语义，不据此推断某个归档实现采用哪一种。内置求和变化用于解释本轮检查失败，不作为 cutoff 之后新增的上游采样实现事实。来源版本受 2026-04-30 上限约束，编辑日期 2026-10-22 不作为事实发布日期。此独立 base 版本内的章节修订号从 1 开始，原章节身份与版本映射保存在版本清单。

## 延伸阅读

- [请求身份：逻辑序列、批次视图与结果归属](../system-semantics/request-batch-identity.md)。
- [结构化生成：语言状态、token 边界与约束采样](../algorithms/structured-output.md)。
- [投机解码：分布保持、状态提交与收益模型](../algorithms/speculation-budget.md)。
- [多样本与搜索分支：父子状态、评分与共同提交](../algorithms/branch-generation.md)。
- [推理服务恢复：提交边界、执行组与重试身份](../system-semantics/serving-recovery.md)。
- [执行视图决策：压紧、固定槽与结果归属](../engine-index/vllm-model-runner.md)。
- [性能实验：独立重复、效应量与尾部证据](../tools/statistical-experiments.md)。
- [输入输出语义：Tokenizer、Chat Template 与流式文本](../system-semantics/tokenization-io.md)。
