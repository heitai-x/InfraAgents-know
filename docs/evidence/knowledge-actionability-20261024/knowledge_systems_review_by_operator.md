# Engine/API 组独立审查

**结论：在所列版本与审查范围内通过；两个发现均已修复并复核，无未关闭发现。** 审查者 knowledge_scenarios；正文作者 knowledge_use_audit。审查者未修改被审查正文。

基准提交为 f640148464929f803a53b55e27ee4cbb83dc8657。完整阅读新增的 engine-construction 与 api-contracts；完整阅读 service-protocol、host-runtime 及其相对基准的本轮 diff。修复后重新完整阅读 api-contracts，并复读 service-protocol 的算例和上下文。对应结构化记录为 knowledge_systems_review_by_operator.json。

## 发现与处理

1. **SYS-REVIEW-01 / P1，已解决。** 原慢客户端算例把远端消费速率直接作为中间队列出队率。root 已改成同一有界队列的实际入队 8,000 B/s、实际出队 2,000 B/s，并明确下游缓冲与背压传导条件；审查者已复读修订段落。
2. **SYS-REVIEW-02 / P2，已解决。** 原文将 ASGI 的 flush-before-return 保证泛化到任意 send。root 已将其限定为成功处理 http.response.body 的 send 返回，补充 http.response.start 可以等待首次 Response Body，且保留不确认远端消费的限制。审查者复读修后全文与 www.rst 第 179–245 行，修订与固定原文一致。

## 实际原文阅读范围

下列均为仓库固定原文；表中范围是实际读取范围，不代表全文审查。源文件 SHA 完整登记于 JSON。

| 来源 | 固定原文路径 | 实际读取行范围 |
|---|---|---|
| S-ASGI | references/S-ASGI/upstream/specs/www.rst | 34–90, 179–282 |
| S-ORCA | references/S-ORCA/fulltext.txt | 577–811 |
| S-PAGED | references/S-PAGED/fulltext.txt | 461–734 |
| S-VLLM-SCHED | references/S-VLLM-SCHED/upstream/vllm/v1/core/sched/scheduler.py | 352–475, 1303–1394, 1741–1873 |
| S-GENAI-PERF | references/S-GENAI-PERF/upstream/genai-perf/README.md | 475–511 |
| S-GENAI-PERF | references/S-GENAI-PERF/upstream/genai-perf/docs/goodput.md | 27–52 |
| S-LINUX-RESOURCES | references/S-LINUX-RESOURCES/upstream/Documentation/admin-guide/cgroup-v2.rst | 1080–1162, 2297–2353 |
| S-PY-SUBPROCESS | references/S-PY-SUBPROCESS/upstream/Doc/library/subprocess.rst | 735–885 |
| S-CPU-TUNING | references/S-CPU-TUNING/upstream/recipes_source/recipes/tuning_guide.py | 208–320 |
| S-TORCH-CUDA | references/S-TORCH-CUDA/upstream/docs/source/notes/cuda.rst | 277–381 |

S-ASGI 第 179–245 行在修订后再次阅读。S-TORCH-CUDA 第 277–381 行是在同一任务前半段算子编写时已完整读取的固定来源，此次复用该阅读结果；未另称新读全文。

## 独立算术复核

| 算例 | 复算结果与条件 |
|---|---|
| 每 token KV | 2 × 32 × 8 × 128 × 2 = 131,072 B = 128 KiB |
| 最坏 KV 与总量 | 8 × 8,192 token 对应 8 GiB KV；14 + 3 + 2 + 8 = 27 GiB |
| 当前 KV 与总量 | 长度 512、1,024、2,048、4,096 各出现两次，总 15,360 token；KV 1.875 GiB，总量 20.875 GiB |
| 分页增量与空尾 | 一页 32 token 为 4 MiB；仅写一个 token 时 31 个空位为 3.875 MiB |
| 串行路径局部加速 | 20 + 6 + 2 = 28 ms；模型时间减半后 10 + 6 + 2 = 18 ms；比值 1.5556 |
| 局部队列填满 | 65,536 B / (8,000 − 2,000) B/s = 10.9227 s；入、出流量必须属于同一队列边界 |
| 慢客户端占用 | 128 × 64 KiB = 8 MiB 输出缓冲；128 × 6 MiB = 768 MiB 私有状态 |
| CPU 配额与需求 | 400 ms / 100 ms = 4 CPU 秒/s；200 请求/s × 15 ms CPU/请求 + 1.2 CPU 秒/s = 4.2 CPU 秒/s |
| 主机缓冲 | 512 × 64 KiB = 32 MiB；512 × 4,096 × 4 B = 8 MiB；128 × 64 KiB = 8 MiB；互不重叠且同时存活时合计 48 MiB |
| 串行排队 | 20 × 5 ms = 100 ms |
| 合格完成率与比例 | 72 / 10 s = 7.2 请求/s；72 / 100 offered = 72%，不以 80 completed 偷换分母 |

这些均为给定假设的算术核对，不是实测。

## 已核对的关键结论

- 架构阶段属于条件化设计推导，不强制分页、连续批、多卡或固定进程布局；从零构建先明确允许复用的范围。
- 当前页式 KV 占用与最坏准入承诺清楚区分，分页没有被宣称可消除最坏总容量不足。
- CPU 时间额度与服务需求的单位一致，正文保留祖先约束、竞争、有效 CPU 集合与持续容量的条件。
- 取消确认不证明设备排空；slot 代次只能约束结果提交权，不能替代在途 DMA 的真实完成。
- GenAI-Perf 固定原文将 TTFT 解释为首 response；正文保留任务需要首个有效内容时的单独口径和核对责任。
- 子进程管道死锁、communicate 超时不自动杀进程及长日志内存风险均与所读固定原文相符。
- 来源机制、原创设计推导与假设算例明确区分，没有冒充真实 GPU、真实服务或 Agent 构建结果。

## 审查限制

这是来源、逻辑与文档审查，没有运行模型、GPU、真实网络服务、调度器或 Agent 实验。审查仅覆盖以上实际原文范围；未重新完整审计未改动的 Borg、OpenTelemetry 和 device-plugin 历史来源。通过结论仅适用于下列文件 SHA 及本轮语义范围，不证明性能、Agent 实现完整性或科研新颖性。目录、索引、来源总账与版本整合由 root 统一负责，本报告未替代其整体验证。

## 最终正文 SHA-256

| 文件 | SHA-256 |
|---|---|
| knowledge/system-semantics/engine-construction.md | d3254439c8dea9f29a2efc5d9242a3134a9a2a57dd2e87976634db838d43d285 |
| knowledge/system-semantics/api-contracts.md | e52b88a4dd7163259658dfaf9ebd053cc9cd74e06f5c72d8b3bc027fb10c8a96 |
| knowledge/system-semantics/service-protocol.md | 5e7ac3914b91d8b6273d189deae4e77662ee1f11f78ac201a588237d30c2da38 |
| knowledge/system-semantics/host-runtime.md | 82daa88b6e9b3110a232eb86786b3e0fce93c5e2e3dcf83a65715dd23d682305 |
