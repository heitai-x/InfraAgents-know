# 知识库独立审查：round 1

**结论：FAIL，本轮不签发 GPU Task 启动通过。** 已完整阅读当前 108 篇机制正文，确认两项 P2 修订问题和一项 P3 文档一致性问题。没有确认到会直接破坏新 Qwen3-8B 稠密 GPU Task 的 P0/P1 正文缺陷；但两项确定错误应修复后对实际新 SHA 复审。新 Task 的采样质量边界仍由用户澄清，独立于知识库缺陷。

审查基线：`InfraAgents-know-astra`，分支 `gpt-6-astra-2026-04-30`，HEAD `0a5a0c0636101d0fbce7c3734636fd50be70e93a`。范围依据 `docs/inference-coverage.md` 与 `docs/expansion-plan.json`，固定来源版本上限为 2026-04-30（含）。本轮由新的独立审查者完成，不是此前内容作者。

## 启动门槛

用户要求审查通过后才启动真实 GPU Task，因此本轮 FAIL 意味着暂不启动。除此之外，Task 同时写了“需要提供多种采样方式”和“每轮输出精确 greedy token”，随机采样与所有输出都按 greedy 检查的适用范围尚未一致。root 已向用户询问，本审查没有修改 Task 或代替用户选择。

实际知识接入检查已通过：候选运行时 `0802c3f140e0b46e26bfdd686ae55662db39c042`，基于已验证 GPU 版本加导航补丁。它在报告目录的独立 bank 副本上，用五角色 `gpt-6-luna`、`full`、`seed_and_learned` 和九项原文限制完整读取 108 篇、验证 108 份知识使用记录和 191 个当前/历史版本。没有启动 Agent、模型或 GPU 作业。详见 [runtime-probe.json](runtime-probe.json)。

## 必须修订的问题

### KR1-001 · P2：MoE 热点算例违反单专家负载上界

位置：`knowledge/models/moe-routing.md`（K-MODEL-003）第 13–15 行，具体错误在第 15 行。正文设定 B=16、k=2、E=8，却称热点专家收到 20 项。同章第 33 行及 S-MIXTRAL 固定原文第 35–36、108–111、133–135 行采用每 token 选择两个不同专家。每个 token 对同一个专家最多贡献一项，因此任何专家都必须满足 `n_e ≤ 16`；总分派项 32 不放宽这个上界。

这会使偏斜/慢专家算例无法构造合法输入，但不否定该章其余路由、并集与容量推导。建议将热点项数改为 16 或其他合法值，同时保留总 32 项和其余专家的分配解释。复核应构造 distinct top-2 映射并检查总量与单专家上界。它不直接影响当前稠密 Qwen3-8B Task，仍是应收口的确定技术错误。

### KR1-002 · P2：HF 缓存来源归因超出固定原文

位置：`knowledge/sources/S-HF-CACHE.md` 第 11 行、`knowledge/sources.json` 的 `coverage`/`local_summary`，以及下列 12 篇的 provenance。完整复读 `references/S-HF-CACHE/fulltext.txt` 第 1–130 行（固定提交 `756dffe0d664419d2f620e131ed426373cfd6f12`），原文有逐层历史/新 KV、mask 的 past+new 长度、Dynamic/Static/StaticSlidingWindow 存储，全文没有 `cache_position` 或 `cache position` 定义。当前来源说明却称该文档区分 cache position。

| 条目 | 当前正文 | 受影响 provenance 行 |
|---|---|---|
| K-ALG-003 | `knowledge/algorithms/prefix-cache-identity.md` | 57–58 |
| K-ALG-006 | `knowledge/algorithms/speculation-budget.md` | 46–47 |
| K-ALG-007 | `knowledge/algorithms/chunked-prefill.md` | 49–50 |
| K-MODEL-001 | `knowledge/models/prefill-decode-state.md` | 72–73 |
| K-MODEL-002 | `knowledge/models/gqa-kv-heads.md` | 48–49 |
| K-MODEL-004 | `knowledge/models/recurrent-hybrid-state.md` | 59–60 |
| K-MODEL-007 | `knowledge/models/sliding-window-attention.md` | 50–51 |
| K-MODEL-008 | `knowledge/models/vision-language.md` | 57–58 |
| K-SYS-003 | `knowledge/system-semantics/position-mask-cache.md` | 55–56 |
| K-SYS-007 | `knowledge/system-semantics/tokenization-io.md` | 52–53 |
| K-TOOLS-006 | `knowledge/tools/correctness-oracles.md` | 55–56 |
| K-ALG-018 | `knowledge/algorithms/kv-offload-prefetch.md` | 51–52 |

建议保留固定原文，将来源卡、registry 摘要和这 12 处归因限缩到原文实际内容；保留逻辑位置、position_ids、物理槽与可见性区分的独立推导，明确具体 API 字段必须按 Task 固定实现核对。若坚持保留“此来源定义 cache_position”，则需要另一个截止前、实际包含该定义的固定依据。旧历史和原始归档不应回写。该问题不构成删除允许的非代码机制正文的理由。

原文另有 `softmax(score × mask)` 示意，不能直接解释为 0/1 数值乘法屏蔽：被乘为 0 的 score 经 softmax 仍可能得到正概率。当前 K-ALG-004、K-SYS-003、K-FW-006 的合法集合/加性 mask 解释正确。可在该来源适用边界澄清示意的读法，不将其扩成新的正文缺陷。

### KR1-003 · P3：独立入口 README 的交付状态过时

`knowledge/README.md` 第 3 行仍称 `v2-building`、新增 21 篇、共 94 篇、仍在扩充；实际目录是 108 篇、35 新增，根 README 与 plan 已完成。建议同步这段计数与状态。新运行时生成的选定 README 已按实际目录生成，此过时数字没有进入本轮接口探针。

## 实质阅读与验证结果

完整阅读覆盖 108 篇、约 40.7 万字符，包括机制、状态、公式、设定场景、来源说明及延伸阅读。逐篇 ID、SHA-256、实际行范围在本报告末尾和 [review.json](review.json)；不是只读标题、索引或已有审查结论。

独立复算了 105 个算术或可行性检查，104 通过，唯一失败为 KR1-001。范围包括 KV/共享容量、MLA 复制、LoRA 和图池、CPU/GPU 带宽下界、分块预算、传输与流水、统计区间、扩散/音视频和非生成推理资源。数值均按正文的设定复算，没有将设定当实测，见 [calculation-checks.json](calculation-checks.json)。

跨章节组合检查中，输入/输出/KV 一步偏移、前缀依赖闭包、异构状态共同恢复位置、取消后的多类引用、旧 DMA 与代次保护、图地址与内容生命周期、greedy/分布/近似质量、开放到达与会话反馈均能相互衔接。未发现与新 Task 直接相关的机制互相矛盾。量化、KV 裁剪、非前缀融合和激活稀疏均有质量前提，不能作为 `allow_approximation=false` 下的默认合法路线。

结构检查验证八类目录、schema、当前正文 SHA、旧修订连续性与 SHA、114 项来源卡/归档 manifest 和全部已登记成员，以及 1,921 个正文链接。实际运行时 audit 报告 1,020 个原文成员、313,942,499 字节，通过结果仅用于结构身份，不代替语义审查。五篇引擎索引的 27 个符号锚点与固定源码 AST 行号一致；这一项不声称完整引擎源码已经复读。

完整索引覆盖 108/108 篇；关联图在忽略方向后为单一连通分量。K-ENGINE-004、K-ENGINE-005、K-TOOLS-004 没有其他机制正文/related 入边，但均有完整索引和分类导航，新运行时 DESIGN/DIAGNOSIS 也各直接列出 108 条，故不能据此判成不可达孤立条目。

## 来源原文的真实核查范围

**本轮没有完整复读全部 114 项来源。** 独立原文阅读覆盖 19 项：9 项完整归档文本、10 项与具体论断有关的原文范围；其余 95 项未逐项复读全文。原文图片和 PDF 版面也没有全部视觉重审。完整文本/局部范围如下，SHA 见 [source-read-ledger.json](source-read-ledger.json)。没有从外网补入截止日后的技术事实，也没有把原有“审查通过”报告用作本文结论的依据。

| 来源 | 实际阅读文件 | 行范围 | 范围性质 |
|---|---|---|---|
| S-HF-CACHE | `references/S-HF-CACHE/fulltext.txt` | 1–130 | 完整归档文本 |
| S-TORCH-NUMERIC | `references/S-TORCH-NUMERIC/fulltext.txt` | 1–167 | 完整归档文本 |
| S-TORCH-REPRO | `references/S-TORCH-REPRO/fulltext.txt` | 1–181 | 完整归档文本 |
| S-CHAT | `references/S-CHAT/fulltext.txt` | 1–229 | 完整归档文本 |
| S-ASGI | `references/S-ASGI/fulltext.txt` | 1–584 | 完整归档文本 |
| S-NKI | `references/S-NKI/fulltext.txt` | 1–369 | 完整归档文本 |
| S-MLX-EXEC | `references/S-MLX-EXEC/fulltext.txt` | 1–539 | 完整归档文本 |
| S-MLX | `references/S-MLX/fulltext.txt` | 1–146 | 完整归档文本 |
| S-MLX-MEM | `references/S-MLX-MEM/fulltext.txt` | 1–80 | 完整归档文本 |
| S-TORCH-CUDA | `references/S-TORCH-CUDA/fulltext.txt` | 279–383, 1336–1479, 1651–1688, 1720–1792 | 具体论断相关范围 |
| S-SDPA | `references/S-SDPA/upstream/torch/nn/functional.py` | 6138–6300 | 具体论断相关范围 |
| S-FA-IMPL | `references/S-FA-IMPL/upstream/README.md` | 210–270, 408–451 | 具体论断相关范围 |
| S-FI | `references/S-FI/upstream/flashinfer/prefill.py` | 1741–1892 | 具体论断相关范围 |
| S-MIXTRAL | `references/S-MIXTRAL/fulltext.txt` | 1–150 | 具体论断相关范围 |
| S-ORCA | `references/S-ORCA/fulltext.txt` | 650–810 | 具体论断相关范围 |
| S-SARATHI | `references/S-SARATHI/fulltext.txt` | 750–835 | 具体论断相关范围 |
| S-AWQ | `references/S-AWQ/fulltext.txt` | 312–386 | 具体论断相关范围 |
| S-SMOOTHQUANT | `references/S-SMOOTHQUANT/fulltext.txt` | 487–553 | 具体论断相关范围 |
| S-PAGED | `references/S-PAGED/fulltext.txt` | 546–651, 926–960 | 具体论断相关范围 |

核对结果支持：SDPA/FlashAttention 的非方形 causal 对齐差异、FlashInfer plan 不可捕获及 split 图限制、PyTorch stream 生命周期与图池条件、SARATHI 的分块可见性、PagedAttention 的引用/COW、AWQ v1 的尺度空间、SmoothQuant 的 O1/O2/O3 区别、Trainium2 的 SBUF/PSUM 与 DGE 边界，以及 MLX 编译与旧惰性文档的区分。HF 的具体接口归因为本轮发现的例外。

## 对新 GPU 在线多轮 Task 的适配判断

修复上述内容后，知识库适合作为该 Task 的机制与实验设计 seed。它已经提供以下可连接的路径，但实际引擎仍须用固定 checkpoint、目标环境和独立检查建立证据。

| Task 需要 | 已有知识路径 | 本轮判断 |
|---|---|---|
| Qwen3-8B 固定制品、BF16 与精确输出 | K-MODEL-001、K-FW-011、K-ALG-016、K-FW-002、K-TOOLS-006 | 能还原函数/状态并区分浮点重排与 token 一致；不冒充该 checkpoint 参数或数值实现已核验。 |
| 完整历史、工具依赖、thinking 开关 | K-SYS-007、K-WORK-010、K-SYS-002、K-ALG-003 | 规范化输入、模板、共同前缀、末 token 未计算状态与回合身份已有机制；具体 thinking 模板应取 Task 制品。 |
| 取消终态与其他会话隔离 | K-SYS-004、K-FW-003、K-ALG-002、K-ALG-008、K-TOOLS-006 | 逻辑引用、执行引用、输出引用和旧 DMA 保护分开；有共享前缀与槽重用的组合检查。 |
| 在线交错、突发与 BS≤16 假设 | K-WORK-002、K-WORK-010、K-ALG-001、K-ALG-007、K-FW-005 | 请求到达、在途/排队与执行 batch 分开推导；不把例子批量当能力上限。 |
| p95 session completion、没有固定性能门槛 | K-WORK-001、K-WORK-004、K-WORK-010、K-TOOLS-007 | 会话/请求指标、开放到达、删失/失败和独立重复可连接；示例 ITL 数值不是新 Task 阈值。 |
| 4090 资源和受限 profiler | K-HW-001、K-HW-005、K-SYS-019、K-TOOLS-004 | 按实际能力和资源口径校准，允许计数器不可用时保留较窄结论，不将 Hopper/Trainium 能力套到 4090。 |

真实读取探针中，远端不可变 v2 的 `fdce576c5972c25b91ad81ba887c616d717ea3ca` 与本地 receipt 提交的 catalog/sources SHA 完全相同。九项受限来源的说明卡可读，`fulltext.txt`、`reader.html`、`manifest.json` 均被实际读接口拒绝；允许的 HF 原文可完整读取。导航分页能完整推进。词法检索的中文多轮、精确数值、图、CPU quota 等查询能找到相关条目；英文 `cancel` 与 `cancellation` 不做同义词保证，检索排名不代表适用性，仍需沿完整导航及相关链接读正文。

`full` 视图没有模型时间 profile 过滤，五角色 Luna 不会被 Astra profile 身份规则错误阻断。静态 seed 的 2026-04-30 来源边界与运行期 learned 的真实观察日期应分别保持。接口可读和知识使用记录有效，不证明实际 Luna 已理解、采用或受益；本轮没有进行这种模型实验。

## 修订与复审要求

root 修复 KR1-001、KR1-002，并建议同批修正 KR1-003；按既有 catalog/revision/history 方式发布新的内容身份，保留原始归档与旧历史。复审应绑定实际新 SHA，重读受影响正文和来源归因，核对数值上界、来源文本范围与实际受限视图接口。Task 的采样边界需要独立明确。完成这两类门槛之前不启动真实 GPU Task。

本轮只在指定报告目录生成审查产物与无 Agent 的临时 bank/Records；没有修改 canonical KB、shared runtime、旧 Task/实验，没有启动模型/实验，也没有推送。首次接口探针的 CRLF 与 `Path.read_text` 换行归一化比较产生了一次检查脚本断言；修正为精确字节解码比较后全部通过，未将此记为 KB 或 runtime 缺陷。

## 逐篇实际阅读记录

以下范围均为当前审查 SHA 下的完整文件行范围，包括 provenance 与延伸阅读。标记“未发现实质问题”限定于本轮阅读和已声明原文核查范围，不等于每项来源与所有实现路径都已证明。

| ID | 正文 | 实际行范围 | SHA-256 | 本轮结果 |
|---|---|---|---|---|
| K-ALG-001 | `knowledge/algorithms/iteration-batching.md` | 1–83 | `156a7dd70b2126dd00afe8fcde002c96ded6c6eb69483c94071f559fecbc7e99` | 未发现实质问题 |
| K-ALG-002 | `knowledge/algorithms/paged-kv-capacity.md` | 1–79 | `a5e4216601b16b1c5f0a566eb4a2b663dafe9ffd21add75a23dd2ba755c8625f` | 未发现实质问题 |
| K-ALG-003 | `knowledge/algorithms/prefix-cache-identity.md` | 1–74 | `c42c9c2c1b01bc17851914fef6899eef828e1c46c98ea954a7f25ebc911287ac` | KR1-002 |
| K-ALG-004 | `knowledge/algorithms/attention-io.md` | 1–65 | `a5e851ffddc0af450d425f6e29f3fbe667d245da61d65eefc81e12b563f35978` | 未发现实质问题 |
| K-ALG-005 | `knowledge/algorithms/quantization-contract.md` | 1–101 | `1ff1082b95610754539212473ba3a562ed9725713e5b882d264a56c79d713403` | 未发现实质问题 |
| K-ALG-006 | `knowledge/algorithms/speculation-budget.md` | 1–66 | `431a1def479bd548131e59b154d5f1ec9c814128e401bbd7a2d3e7a76392066d` | KR1-002 |
| K-ALG-007 | `knowledge/algorithms/chunked-prefill.md` | 1–58 | `86c0b343390920574f6679da9f0eff8e24537991a8e0919a735aadc2ac7c043f` | KR1-002 |
| K-ALG-008 | `knowledge/algorithms/async-scheduling.md` | 1–75 | `9f3e037e2725193a3aefda22db0d7461c91ee737b40018a51e8e2f5afbe9fa5d` | 未发现实质问题 |
| K-ALG-009 | `knowledge/algorithms/heterogeneous-state-cache.md` | 1–69 | `c9d7ef861fd1243cb4a605a38405fe1c7ddb2cfc24a653a97a83080ad065d02b` | 未发现实质问题 |
| K-ALG-010 | `knowledge/algorithms/cross-attention-cache.md` | 1–71 | `aa8313c1df241ae8dabe2bc7a5f57e7189bc5e1260e232eb30835d399f1499e3` | 未发现实质问题 |
| K-ALG-011 | `knowledge/algorithms/batched-sampling.md` | 1–56 | `a75584a814f07f661f8dc61148529e8e08d47fa05ff68ffa8d6b31562316e24a` | 未发现实质问题 |
| K-ALG-012 | `knowledge/algorithms/structured-output.md` | 1–54 | `cfa738aa99700ff90b96b3044c134897867e93792289fe62b21d611855139d0e` | 未发现实质问题 |
| K-ALG-013 | `knowledge/algorithms/disaggregated-serving.md` | 1–78 | `07d8beba67dd10960cd504dafad8cdf0baaf92c03ef368a2a97236690c6bc7e8` | 未发现实质问题 |
| K-ALG-014 | `knowledge/algorithms/lora-serving.md` | 1–73 | `3cba63cc43866f40cca61ad911f32115e9beaa5c4adff9559b4deb592fd3d4d7` | 未发现实质问题 |
| K-ALG-015 | `knowledge/algorithms/parallelism.md` | 1–103 | `161a883561e39371ea22a1430c385546c402db2f993d8a37c6f37e03e1e4bd17` | 未发现实质问题 |
| K-ENGINE-001 | `knowledge/engine-index/vllm-scheduler.md` | 1–60 | `723d1f1b1bc65ca6fca28c83e9930a08d643469e7dec866b96a6d7de2f6afd95` | 未发现实质问题 |
| K-ENGINE-002 | `knowledge/engine-index/vllm-kv-manager.md` | 1–63 | `60972ada3e249ab9ed5ee1dd1f38adae9a8a6219f3bfe53407bf93f7ec12027a` | 未发现实质问题 |
| K-ENGINE-003 | `knowledge/engine-index/vllm-model-runner.md` | 1–57 | `41c03c35b3489f1900311f13e943d955372e08b1286f02aaedab710540e91d1f` | 未发现实质问题 |
| K-ENGINE-004 | `knowledge/engine-index/sglang-radix-scheduler.md` | 1–59 | `bbb648ee10b9ae8581809ea14fd0bd11396400c7c0a30d23443554ad440fa298` | 未发现实质问题 |
| K-ENGINE-005 | `knowledge/engine-index/tensorrt-llm.md` | 1–58 | `ec40e61a590a31703236965be35a0893af553c79f3f75657e5aa64ab2f899b8b` | 未发现实质问题 |
| K-FW-001 | `knowledge/frameworks/inference-eval.md` | 1–72 | `683d5b7f5323ab43af402fbe4719539ab4ad0d20d290db019d041070e8887fff` | 未发现实质问题 |
| K-FW-002 | `knowledge/frameworks/numerical-reference.md` | 1–75 | `a70beff4c73683f897d9c0c1fb291539051e20b2c31adfd90a3f6677c411596f` | 未发现实质问题 |
| K-FW-003 | `knowledge/frameworks/streams-allocator.md` | 1–54 | `64427e69f6eb991d4029242e24e4edf56a331d233fdeab3c50f53f0f84a4df7b` | 未发现实质问题 |
| K-FW-004 | `knowledge/frameworks/compile-dynamic-shapes.md` | 1–88 | `50c53e4310e29020b5897f9c069420f4933dd792a50d452e9ff2a3aa827d210c` | 未发现实质问题 |
| K-FW-005 | `knowledge/frameworks/cuda-graph-buffers.md` | 1–93 | `34232a876bf3cc6904ad796423659dc8b4bc250532e6d2213e0d0f5d1fd88e6c` | 未发现实质问题 |
| K-FW-006 | `knowledge/frameworks/attention-backends.md` | 1–93 | `01009cca4251544f55e6478d8bb99f2af1aeb3a043d027ff3279692215917bac` | 未发现实质问题 |
| K-FW-007 | `knowledge/frameworks/triton-kernels.md` | 1–80 | `35f35dde0def4be2bc2e14e4c0a476d5b38e535d3a0b2284643fb52f16f1fdba` | 未发现实质问题 |
| K-FW-008 | `knowledge/frameworks/mlx-serving.md` | 1–66 | `c308a8ac651460ede6b6e53d3413bea4c4a7707b757516faf8a752b35872b6c1` | 未发现实质问题 |
| K-FW-009 | `knowledge/frameworks/neuron-serving.md` | 1–63 | `0a7be25e54019cb53d80e180594d1eb633cfdd9e9a058cb340a2c07dee7f3f0e` | 未发现实质问题 |
| K-FW-010 | `knowledge/frameworks/nki-tiling.md` | 1–59 | `abc8dbd0be6a05a8b52ed4686ab2d0e27f893c6e182656c36c514306c0799a16` | 未发现实质问题 |
| K-HW-001 | `knowledge/hardware/memory-and-roofline.md` | 1–76 | `58e9e2f9292d24457d7281a22e646242ba607096f204ba7a30feea621353f3e3` | 未发现实质问题 |
| K-HW-002 | `knowledge/hardware/topology-parallelism.md` | 1–64 | `40d7b15c7a9b4148b95f44c2bdba861884112b59bb8c6fab44b96cf8e119c7ed` | 未发现实质问题 |
| K-HW-003 | `knowledge/hardware/host-device-transfer.md` | 1–67 | `5bfc321d4349e141d1787a360ab72106bc70af99af17d9e53bdd1b75e2a0980a` | 未发现实质问题 |
| K-HW-004 | `knowledge/hardware/precision-and-kernel-support.md` | 1–88 | `7185979b8429b7d44fd34ec0910bbe85c617a457cc4d9d9055920e73d07bc5d3` | 未发现实质问题 |
| K-HW-005 | `knowledge/hardware/cuda-execution.md` | 1–67 | `aef68ee4706e1ca36c7d2655ef8edf3535e9893a29d54947407beddc8465f2ed` | 未发现实质问题 |
| K-HW-006 | `knowledge/hardware/rocm-execution.md` | 1–52 | `08c875daee94b818579b1f63e07543c3d8628177d61f3141a6f39727cc0fbf8a` | 未发现实质问题 |
| K-HW-007 | `knowledge/hardware/metal-unified-memory.md` | 1–49 | `b4a1f38557eecf13e85f3a962f617c95ec86ef7c800a664471d6b1d2c3d31fd4` | 未发现实质问题 |
| K-HW-008 | `knowledge/hardware/trainium-dataflow.md` | 1–55 | `0773594917538d150957f945b6c9268e1aca64bb83e041aec2315c1cb0984019` | 未发现实质问题 |
| K-HW-009 | `knowledge/hardware/cpu-serving.md` | 1–78 | `2f48de03875b90417b29b039736a4b9532d5b8a8681dc6c28d7707b9271084e6` | 未发现实质问题 |
| K-MODEL-001 | `knowledge/models/prefill-decode-state.md` | 1–104 | `fa02b9460240604dc99171c61dbb62c78f99bc2a8afcd2079ba69ed78d387864` | KR1-002 |
| K-MODEL-002 | `knowledge/models/gqa-kv-heads.md` | 1–71 | `d381f8374bdc2df2a4e3096f5ab9a93e3bd2493db369cb23339ab61e8c5b5da2` | KR1-002 |
| K-MODEL-003 | `knowledge/models/moe-routing.md` | 1–79 | `f7b00bde22fbeada937690805ad5f1c45e7a4a51b802e3164241505cc7ddeed1` | KR1-001 |
| K-MODEL-004 | `knowledge/models/recurrent-hybrid-state.md` | 1–68 | `b05d62f616ddc062975f04ccba7f505a85b2090f0839473d67e734fcbd64ae26` | KR1-002 |
| K-MODEL-005 | `knowledge/models/encoder-decoder-streaming.md` | 1–68 | `147fa3bc49d6b6423d7ba03ecc311b87345ee7c2c934a4b0007ad59b2ed72bf4` | 未发现实质问题 |
| K-MODEL-006 | `knowledge/models/mla.md` | 1–80 | `1f0c786d7dde16c9f946254f6ad1046c7d409e1c9a11e3a383f1750ffe1646e1` | 未发现实质问题 |
| K-MODEL-007 | `knowledge/models/sliding-window-attention.md` | 1–72 | `3e3111991f3800dfd1c40559f585403c1af37ddab120399f9b34aaa2541f681d` | KR1-002 |
| K-MODEL-008 | `knowledge/models/vision-language.md` | 1–66 | `c45ceef4d245f3de46f6d3ca219883f495deddd0d5355f19dd64a6ba9e14bd25` | KR1-002 |
| K-MODEL-009 | `knowledge/models/image-diffusion.md` | 1–76 | `f08b493a7f620357ffd83d063ee0a797ffa886046659e08271c3face962000e6` | 未发现实质问题 |
| K-MODEL-010 | `knowledge/models/video-generation.md` | 1–70 | `d6ebd708a7b9de12dfae97659345b1ccec8a694133be812e25a67f2e22246da7` | 未发现实质问题 |
| K-MODEL-011 | `knowledge/models/speech-generation.md` | 1–64 | `9cab040f27f77dc8040a6cc6f0ae96a175542e4379e2262ca2cdb53fb9aee3a4` | 未发现实质问题 |
| K-MODEL-012 | `knowledge/models/omni-multimodal.md` | 1–71 | `f1fc4dc6611ad1b53ad8cc7f2abf0cbf9b9b2022d2d22c64dc712a39943d9d15` | 未发现实质问题 |
| K-SYS-001 | `knowledge/system-semantics/host-device-timing.md` | 1–78 | `411061489661a325939476348559d80df773a32cc19be2aef216e4d4a5b0b554` | 未发现实质问题 |
| K-SYS-002 | `knowledge/system-semantics/request-batch-identity.md` | 1–58 | `856dab119bd58b29a2a4a48826a05aab387df4101c185b142162aa5bdf65d4cb` | 未发现实质问题 |
| K-SYS-003 | `knowledge/system-semantics/position-mask-cache.md` | 1–83 | `6d86052f0fa6aaeafa11b8a533ada433b3aa0c728e8ee8510ee47071218aa65f` | KR1-002 |
| K-SYS-004 | `knowledge/system-semantics/cancellation-resource-completion.md` | 1–81 | `919052c77f65a13c9dc09aad03341ed31f0820dbbdcb2750b5789f349f9e86c1` | 未发现实质问题 |
| K-SYS-005 | `knowledge/system-semantics/collective-ordering.md` | 1–54 | `c24f552f27a912784d04e2a05bc3ee6849147ddf8655af5a7e67600f04c5145c` | 未发现实质问题 |
| K-SYS-006 | `knowledge/system-semantics/evidence-and-independent-checks.md` | 1–84 | `50b7c97de928be322713ef149fb734c0c20acda17c4a321c46a198b0019831be` | 未发现实质问题 |
| K-SYS-007 | `knowledge/system-semantics/tokenization-io.md` | 1–64 | `89f4bad055aa5169e14d460ca7b4fc96c0f70a7e4d046510ced32b6bf9e2c198` | KR1-002 |
| K-SYS-008 | `knowledge/system-semantics/service-protocol.md` | 1–74 | `f68e94189e886407e217a22cea44bb4767d6611200d8b657d656e00c925c8eb8` | 未发现实质问题 |
| K-SYS-009 | `knowledge/system-semantics/distributed-state-transfer.md` | 1–69 | `425985efff21e4acbe091ddfa9043cf97b37e550b606f10c0e1560a3c02b1f6f` | 未发现实质问题 |
| K-SYS-010 | `knowledge/system-semantics/design-synthesis.md` | 1–158 | `a769af3d7a1b18a662391160fd09ff7e4b2d0822ba88229f47bc6b765ad7b05c` | 未发现实质问题 |
| K-TOOLS-001 | `knowledge/tools/trace-observability.md` | 1–77 | `6793e9f99d23ce6dea9c630e9b77a8b37954ef43b3307ef7ee52e5bc4205a557` | 未发现实质问题 |
| K-TOOLS-002 | `knowledge/tools/timeline-attribution.md` | 1–74 | `cf4b01a8ca5cb35e653b65fbe2bc018d3ae5cc42788a77688524fb29c7da817d` | 未发现实质问题 |
| K-TOOLS-003 | `knowledge/tools/kernel-counter-replay.md` | 1–83 | `9ecd4279422f94ccc42ffa70aca42d5fbec6e8f61a1182fb5983aebeaccd692d` | 未发现实质问题 |
| K-TOOLS-004 | `knowledge/tools/framework-profiler-scope.md` | 1–92 | `bd04c303d02d55165a234b1735c31bb1259d3374143635fa593ce13576e493a8` | 未发现实质问题 |
| K-TOOLS-005 | `knowledge/tools/failure-evidence.md` | 1–76 | `23dfc65dcbd319c1789dcb80b62f6c843bb5db8c076e86a00643cff546700067` | 未发现实质问题 |
| K-TOOLS-006 | `knowledge/tools/correctness-oracles.md` | 1–75 | `4a9633cae7fe6a67174776a45e259faed2bfef3bd50385a6867f3339037bd0de` | KR1-002 |
| K-WORK-001 | `knowledge/workloads/metric-events.md` | 1–67 | `6aa75b8243c1e608d6d9a54e7754af49ac7b4c6ffc343b812a9a44db49dfba49` | 未发现实质问题 |
| K-WORK-002 | `knowledge/workloads/arrival-and-concurrency.md` | 1–62 | `986f34774c8890dfc163176991b15f6f839683a588bc88f481b5510bc114fe6d` | 未发现实质问题 |
| K-WORK-003 | `knowledge/workloads/input-distribution.md` | 1–69 | `b1b68f0d2a6d640c4b64c2bc98560b9e9a79a6a60b445899949b5db15b96aa90` | 未发现实质问题 |
| K-WORK-004 | `knowledge/workloads/measurement-comparability.md` | 1–70 | `7e839f0863c25fd6ebc0653b878bebda20236e6280a23a837fef110858f1801e` | 未发现实质问题 |
| K-WORK-005 | `knowledge/workloads/slo-goodput.md` | 1–62 | `6ca0d099f7cf89b51c6e6556628abdc873b99d01aaebd1db6ff4db4b148cf45a` | 未发现实质问题 |
| K-WORK-006 | `knowledge/workloads/capacity-planning.md` | 1–76 | `d33782fcb76cdf6f05f294f82b539bad4c119292a8f27248fac2ac721794ca8c` | 未发现实质问题 |
| K-FW-011 | `knowledge/frameworks/checkpoint-loading.md` | 1–64 | `871e49626f320d579993063db039ab0a4b8e98394a54ffa2073daa9ed8bb2cb5` | 未发现实质问题 |
| K-ALG-016 | `knowledge/algorithms/operator-primitives.md` | 1–67 | `4f83a7b40dc7880348304cf8a3439d418fc9e911f6051e687488f8e83bdf6e03` | 未发现实质问题 |
| K-HW-010 | `knowledge/hardware/cpu-linear-kernels.md` | 1–71 | `8ce38535a37fc19684a58b3c9243f28eccaa2af58205ae262a288e3c948c71cb` | 未发现实质问题 |
| K-FW-012 | `knowledge/frameworks/cpu-quantized-kernels.md` | 1–65 | `8a02565f2a303b5d70825d1db0c08c9a6b9b2ac053b18cd18e4f0cf2d55376a2` | 未发现实质问题 |
| K-SYS-012 | `knowledge/system-semantics/memory-allocators.md` | 1–71 | `8cfcf5b54fff448070d05207ad57958296c1613ac12acae1d1b0ec18a4d4d0e1` | 未发现实质问题 |
| K-SYS-015 | `knowledge/system-semantics/host-runtime.md` | 1–68 | `f8cf049cc597aa891b88075ac712e0f0f4be0f4c465cd6e9b0aa901e7dde39ed` | 未发现实质问题 |
| K-WORK-010 | `knowledge/workloads/session-workloads.md` | 1–63 | `01bc154d3bd07ebf4822ee074ab91a0998159e4fe5f7d14c4c36198442666be5` | 未发现实质问题 |
| K-TOOLS-007 | `knowledge/tools/statistical-experiments.md` | 1–65 | `cafeb92ae753a362d3ab1eaad0387d3b5964fd0908c8d6195a60cb78429cb105` | 未发现实质问题 |
| K-HW-011 | `knowledge/hardware/gpu-kernel-pipelines.md` | 1–78 | `c89333e36034b5a9ea27762b73f0fc487c202a7ce8dd7d38d72ec35fc67258ce` | 未发现实质问题 |
| K-ALG-017 | `knowledge/algorithms/tiered-residency.md` | 1–77 | `eb3ebc6320d5a32481cf645605d159860dc6d1edb97f1e98afd89ddb24ee82af` | 未发现实质问题 |
| K-HW-012 | `knowledge/hardware/storage-io.md` | 1–62 | `417f087e6e2591e58ce3957b4947d3c2718f1373da31bd7f3e4947eb77e58352` | 未发现实质问题 |
| K-ALG-018 | `knowledge/algorithms/kv-offload-prefetch.md` | 1–67 | `a02821cd67e2d6d3c90ee5720cf2d2804d7a9562508f83a2ae0955132904e5cc` | KR1-002 |
| K-ALG-019 | `knowledge/algorithms/weight-expert-offload.md` | 1–64 | `cb230fafeec2d39fa40228ad44dbb41d34c62673fbac2325b6e797f206bc0fde` | 未发现实质问题 |
| K-ALG-020 | `knowledge/algorithms/heterogeneous-execution.md` | 1–75 | `7d0e23df334c5550a832153ef298e5fb32f35156944453852480e4600b0f6c50` | 未发现实质问题 |
| K-HW-013 | `knowledge/hardware/shared-memory-paths.md` | 1–64 | `9471f29d33a5821b00e11b6d08fb69fe31288087d558f05c5f9234350efe8b0d` | 未发现实质问题 |
| K-ALG-021 | `knowledge/algorithms/kv-compression.md` | 1–106 | `b106cc93bcd9d5acfce5e7766eb25f5c1fbd4ba7844dd1fe6454fc08cc1d99f4` | 未发现实质问题 |
| K-SYS-013 | `knowledge/system-semantics/distributed-kv-store.md` | 1–75 | `ecc4f667ff186b05fececba52c500e43d0460def524e6200b9cfa00fcdb984c0` | 未发现实质问题 |
| K-HW-014 | `knowledge/hardware/transport-and-rdma.md` | 1–78 | `72bcf1d8948eae41c243e422f6b0370462dcc4bf33ef86f5664a09b4e46d3ba2` | 未发现实质问题 |
| K-SYS-014 | `knowledge/system-semantics/serving-recovery.md` | 1–81 | `f551fa1f1e6213469c856a880839eefff504fb39f4dd692000b2df80a20a7f5e` | 未发现实质问题 |
| K-ALG-023 | `knowledge/algorithms/routing-and-migration.md` | 1–64 | `1b9dcfd5fea499a91611459bb936e1c1c49ceb562171fb0f1ba360a7010f5d32` | 未发现实质问题 |
| K-ALG-024 | `knowledge/algorithms/branch-generation.md` | 1–75 | `42157d4da17ab8ffd7a0aa2f5cd57a1cacdf1a989757261993f6671ae231de3a` | 未发现实质问题 |
| K-SYS-011 | `knowledge/system-semantics/model-lifecycle.md` | 1–65 | `120147b62064624928974d827f10e5e7ad748ae780025882ef1c458ee2ff8c48` | 未发现实质问题 |
| K-WORK-007 | `knowledge/workloads/multi-model-placement.md` | 1–67 | `d60397111d36bfb52045d741842175cc794fbb37af694406558bdbd992133171` | 未发现实质问题 |
| K-WORK-008 | `knowledge/workloads/elastic-serving.md` | 1–64 | `782ba12ac9d0af5f8dbdd477e0c706230ae14e7f54d2795ab3b385eaeb8b39f4` | 未发现实质问题 |
| K-WORK-009 | `knowledge/workloads/cost-and-energy.md` | 1–61 | `bf9b2c1d084f562d584b391ec9e22b6893540691dd7686f65599e8c2a9fef157` | 未发现实质问题 |
| K-SYS-016 | `knowledge/system-semantics/tenant-isolation.md` | 1–73 | `0ca7501a80e750152a5fb3b9815c77f9e3f9aa393ff7546b3f05375958f88606` | 未发现实质问题 |
| K-WORK-011 | `knowledge/workloads/cluster-resource-placement.md` | 1–65 | `235de985ced67e21bfe6727ff2e91be680e8def602ad02a3f3e9de2b3ca1b8a1` | 未发现实质问题 |
| K-SYS-018 | `knowledge/system-semantics/storage-services.md` | 1–74 | `283ceab23fbbf5c27bbb4486b56c2123459a19a00229c5baae70d09c18062661` | 未发现实质问题 |
| K-SYS-019 | `knowledge/system-semantics/deployment-environment.md` | 1–67 | `1b47e39a7f36c9f1cba4dc1792ad9243fd69d083ae1d6a7bb13bfd80dae1a8a5` | 未发现实质问题 |
| K-ALG-022 | `knowledge/algorithms/context-reuse.md` | 1–78 | `ddd8fa928d5ebd9876aca4971dc357b666ea999be9b148e3e88404c9e271ab13` | 未发现实质问题 |
| K-MODEL-013 | `knowledge/models/embedding-reranking.md` | 1–71 | `44f22a27c9b4a11579bed108952f18722d90cd0d4c326db189c865d7e51aef38` | 未发现实质问题 |
| K-SYS-017 | `knowledge/system-semantics/retrieval-pipeline.md` | 1–83 | `bd09460502715f214c20ffdd68bdbc6bb9373668a849ca602b75d5e933ff40d0` | 未发现实质问题 |
| K-MODEL-014 | `knowledge/models/vision-inference.md` | 1–75 | `a3b95c24c3838a54df90db945066257c0960e20a24929ed2812838fc5a3528a0` | 未发现实质问题 |
| K-MODEL-015 | `knowledge/models/recommendation-inference.md` | 1–68 | `a36b0addea05174460d8f51179643665f98c51c6aa4a4d9dec235535948dd06d` | 未发现实质问题 |
| K-MODEL-016 | `knowledge/models/graph-inference.md` | 1–71 | `343135d67c7fd61ed1e7fb82185df7a8f30ebfc86cb5427dfd9e6313046fe6fe` | 未发现实质问题 |
