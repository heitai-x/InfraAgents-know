# K-MODEL-013：Embedding 与重排：编码函数、批量与评分契约

Embedding 服务把一次输入变成固定维度表示，重排服务把一个查询与候选集合变成分数或顺序。它们的主要状态是模型、输入变换和中间张量，不能直接套用自回归 decode 的逐 token KV 生命周期。

## 输出向量由整个编码函数定义

完整链路是文本规范化与 tokenizer、特殊 token/segment/position、padding mask、编码器、pooling 或任务头、可选归一化、输出序列化。模型权重相同而 pooling、截断或 normalize 不同，会得到不同向量空间。索引侧与查询侧应遵循训练好的双塔契约；双塔权重可以共享也可以不同，但不能任意混配。

BERT 固定论文描述双向 encoder：一个有效 token 在每层可看到同一输入中的左右文。句对作为一个序列输入，借助分隔符和 segment 区分；分类头读取特定聚合位置。它为 encoder 与句对评分提供结构依据，并不证明未经相应训练的 CLS 或任意平均池化天然适合语义检索。

将同一文档从长度 128 截断到 64，会改变剩余 token 可见上下文，因而不只是丢掉后半部分输出。双向编码器通常不能像因果 decoder 那样把“前 64 个 token 的全部深层状态”作为未来追加内容的精确前缀缓存。可缓存的独立文档向量必须绑定完整有效输入与编码函数。

## 双塔检索与交叉编码器重排

双塔分别计算 e_q=f_q(q)、e_d=f_d(d)，文档向量可离线构建，查询只做一次编码并进行向量搜索。评分函数可以是内积、余弦或训练指定的其他形式。使用余弦时需要明确零范数处理；只有双方按同一规则归一化后，内积排序才与余弦一致。

交叉编码器将 q、d 放入同一次 attention，并由任务头输出 s(q,d)。每个候选都要经过这条计算链，成本随候选数及句对长度增长，文档深层表示通常不能脱离查询独立复用。两者是不同质量/成本方案，不能把交叉编码器的分数当作普通向量内积接口。

一个请求含 k 个候选时，系统可以将句对聚批，但必须保留查询 ID、候选 ID、原顺序及各自截断结果。分数归还错误的候选会表现成模型质量下降。若输出概率，须使用该任务头的校准与归一化定义；任意跨请求 softmax 会让同一候选的分数依赖一起排队的其他请求。

## Padding、packing 与计算形状

设一批有 B 个长度 nᵢ 的序列，隐藏维 d。投影和 MLP 的主要工作随 Σnᵢ 变化，密集 attention 的逻辑配对数随 Σnᵢ² 变化。统一 padding 到 nmax 后，相应工作变成 Bnmax 和 Bnmax²。按长度分组能减少浪费，但等待聚批增加在线延迟，并可能让长输入饥饿。

Packing 把多个序列放入一个连续缓冲时，必须保留序列边界和各自合法的位置、segment 与 attention mask。仅拼接 token 后做一次全连接的双向 attention，会让不同请求相互影响。可用 ragged 或块对角可见性减少无效计算，前提是 backend 能表达该函数。普通 padding mask 也只屏蔽无效位置，不等于跨序列隔离。

Pooling 应按有效 token 集合计算。平均池化的分母、特殊 token 是否参与、最后有效位置如何确定，都属于模型制品。向量归一化的 epsilon 和输出 dtype 同样需要固定。取 tensor 的最后一列，在右 padding 的批中可能读到 padding；对左 padding 或位置重建也不能凭形状猜测其含义。

## 驻留、缓存与阶段预算

单次执行的主要资源为权重、输入、层间激活、attention 工作区和输出向量。融合 attention 可避免显式保存完整分数矩阵，但不能因此把算术工作当作线性。输出尺寸为 B d_out s；如果只需要 pooled embedding，尽早释放逐 token 隐藏状态比向宿主复制全部 token 表示更节省流量。

文档向量缓存 key 至少包含编码器/任务头版本、tokenizer 与前处理、完整有效输入、截断与 pooling、normalize 和输出格式。重排分数缓存还依赖有序查询—文档对及评分头；候选序列整体归一化的评分还依赖集合本身。热点输入缓存与长度聚批之间有相互影响：命中后留下的冷请求长度分布可能改变，调优不能只看原始流量。

端到端延迟由排队、分词/图像等输入处理、设备搬运、编码、pooling/评分、归还组成。小批短句可能由 tokenizer 或提交成本主导；长句 attention 与激活更重；大量文档离线建库则更重视吞吐、输出持久化和失败重试。三个负载不宜共用一个“每秒 token”指标。

## C12-A：Embedding 与重排的完整验证场景

设定一个已固定输入和 pooling 契约的 encoder，输出 768 维 FP32；在线批含四条 64 token 与四条 512 token 序列。统一 512 桶的 attention 配对数为 8×512²=2,097,152，按两类分批后为 4×64²+4×512²=1,064,960。约 1.97 倍只是逻辑配对数之比，不包含权重、投影、等待与 kernel 效率，不能直接作为时延加速。八个输出向量合计 8×768×4=24,576 字节。

候选 A 使用统一桶，在有限等待窗口内组成一个批；候选 B 使用 64/512 两桶并分别设置最大等待时间。质量目标是保持每条输入的编码函数，服务目标是固定 offered workload 下的 p95/p99 响应和完成率。离线建库可放宽等待来提高利用率，在线稀疏到达时则可能选择 A 或更小批。

先用单样本参考逐条计算，再对照两种批处理的完整向量、余弦/内积和候选排序。一个独立 pooling 单元算例取有效向量 (1,0)、(0,1)，padding 向量 (100,100)，规定均值只含前两行，结果应为 (0.5,0.5)，再 L2 归一化为 (1/√2,1/√2)。这是测试输入变换的设定算例，不代表任意 checkpoint 应采用均值池化。

重排部分为每个查询固定 20 个候选，分别运行串行句对与聚批句对。先验证每个 score 与候选身份，再测 nDCG/MRR 或任务规定的相关性指标，不能用向量数值接近代替排序质量。覆盖空文本、只有特殊 token、超长截断、重复候选、同分排序、混合长度、取消一个子请求和部分失败。批量置换后还原顺序应保持语义。

最后同时报告编码延迟、完整查询重排延迟、候选数分布、内存峰值与质量。若缩小候选数带来加速，需要把召回损失列为算法变化；若只改变聚批，则使用同一候选集合比较。这样才能区分系统调度收益和更少工作带来的表面收益。

<!-- infraagents:provenance -->

## 来源与推导

- [S-BERT：BERT](../sources/S-BERT.md)：固定 v1 描述双向 Transformer encoder、token/segment/position 输入、CLS 句级任务头及句对/序列标注任务。
  [完整原文](../../references/S-BERT/reader.html) · [全文文本](../../references/S-BERT/fulltext.txt)

- [S-FAISS：The Faiss library](../sources/S-FAISS.md)：固定 v1 系统描述距离与 embedding contract、精确/近似搜索、PQ/IVF/图索引、精排、更新过滤、CPU/GPU 实现及外部存储接口。
  [完整原文](../../references/S-FAISS/reader.html) · [全文文本](../../references/S-FAISS/fulltext.txt)

- [S-TORCH-NUMERIC：PyTorch — Numerical accuracy](../sources/S-TORCH-NUMERIC.md)：浮点运算不满足实数运算的全部结合性质，批量方式、归约顺序和平台路径可以产生数值差异。数值格式、累加精度和极端输入影响误差。
  [完整原文](../../references/S-TORCH-NUMERIC/reader.html) · [全文文本](../../references/S-TORCH-NUMERIC/fulltext.txt)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [输入输出语义：Tokenizer、Chat Template 与流式文本](../system-semantics/tokenization-io.md)。
- [基础算子：从张量契约到合法融合](../algorithms/operator-primitives.md)。
- [编译与动态形状：Guard、分桶和变体成本](../frameworks/compile-dynamic-shapes.md)。
- [工作负载分布：长度、复用、模态与相关性](../workloads/input-distribution.md)。
- [向量检索与 RAG：索引、版本与端到端质量](../system-semantics/retrieval-pipeline.md)。
- [正确性 Oracle：函数参考、状态变形与服务契约](../tools/correctness-oracles.md)。
