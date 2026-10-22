# 时间边界与离线知识库

InfraAgents 的知识阅读必须与联网资料准备分开。运行时只读取仓库内已经归档的正文和原始资料，不通过 WebSearch、在线文档或动态源码补定义。离线还不等于没有数据泄漏：一个在截止日之后抓取的网页快照，即使今天保存在本地，也可能把模型当时不可能知道的版本带进推理。

## 固定比较时间

GPT-6 Astra 官方模型元数据给出的 knowledge cutoff 是 **2026-04-30**，本地核查记录在 [模型截止时间证据](https://github.com/heitai-x/InfraAgents/blob/a102858e3bae595a4af88be6c3a1cd2d7a0f37f1/docs/evidence/gpt-6-astra-cutoff-20260927.json)；官方出处保留在该记录的 `source_url` 中。知识库使用同一个允许上限，配置为 [temporal-policy.json](https://github.com/heitai-x/InfraAgents/blob/a102858e3bae595a4af88be6c3a1cd2d7a0f37f1/knowledge/temporal-policy.json)。两个上限的交集才是运行时比较时间：

\[
  T_{effective}=\min(T_{model},T_{knowledge})
\]

取最大日期会允许知识库越过模型能力边界，因此不适合作为无泄漏比较时间。当前 profile 的两者都是 2026-04-30，`effective_cutoff` 也必须是这一天。

## 三种时间不能混用

| 时间 | 含义 | 能否证明截止日前可用 |
|---|---|---|
| `available_at` | 论文版本、协议版本、固定源码或网页版本实际可获得的日期 | 可以，前提是有明确版本依据 |
| `checked_at`/`archived_at` | 本地阅读链抓取和归档的日期 | 不能；它只说明何时保存 |
| `updated_at`/`observed_at` | 正文编辑日期或项目运行观测日期 | 不能替代来源版本日期；观测还要按 profile 的规则单独处理 |

严格 profile 只接受有明确 `available_at` 且不晚于 2026-04-30 的来源。没有日期证据的来源是 `unknown`，明确晚于截止日的是 `post_cutoff`，两者都保留在归档和审计中，但不进入角色工作快照。一个专题引用的来源必须全部 `eligible`，专题才可用；单个来源被挡住就不能用整章的推导链。

当前逐来源、逐专题的结果在 [temporal-audit.json](https://github.com/heitai-x/InfraAgents/blob/a102858e3bae595a4af88be6c3a1cd2d7a0f37f1/knowledge/temporal-audit.json)。本次审计覆盖 80 项来源：33 项有截止日前版本证据，21 项明确在截止日后，26 项仍缺少足够的版本日期。73 个专题中只有 5 个能进入严格 profile，68 个被阻断。这是保守的清理结果，不把“内容看起来很基础”当作日期证据。

本次阅读也确认了知识库的设计主线：模型专题先确定计算图、逻辑位置和持久状态；算法专题把状态变成缓存、调度、采样、量化或并行关系；硬件和框架专题说明这些关系怎样落到存储层级、执行资源与编译边界；系统语义专题维护身份、完成、通信和状态迁移；工作负载专题把到达、分布、指标和容量接回用户目标；工具专题则限定哪些观测能支持哪种解释；引擎索引把通用机制定位到固定版本源码。`DESIGN.md` 和 `DIAGNOSIS.md` 是这条推导链的两个入口，RSI 只把经过独立核对的机制增量写回原章节。

## 五个逻辑职责怎样自动构建

时间门禁属于运行和知识合同，不增加额外的时间审查角色；Experimenter、Diagnostician
和其他逻辑职责都读取同一份冻结 profile：

1. 准备阶段为每个来源登记固定版本、`available_at`、判定依据、完整本地归档和哈希。只有网页抓取日期而没有版本日期的材料先放入 `unknown`。
2. `knowledge temporal-audit` 对 80 个来源和 73 个专题逐条计算状态。规则是确定性的，不让模型自行猜日期；`--write` 生成可复查的审计 JSON。
3. Runner 创建工作快照时加载 `gpt-6-astra-2026-04-30`，只复制 `eligible` 专题、其来源说明和对应原始归档。Planner、Executor、Experimenter、Diagnostician 和 Evolution 看到的是同一份冻结视图；被挡内容的正文仍可在仓库中追溯，但运行时读取会拒绝。
4. Evolution 只能从截止日前来源和符合 `observed_at` 规则的运行证据提出更新。Diagnostician 独立核对来源日期、版本哈希、机制和范围；发布器在共享写入前再次拒绝未知或越界来源。
5. 需要加入新专题时，先补齐来源的版本日期和本地原文，再重新审计。补日期不能修改原文哈希，也不能把后来的源码快照声称成旧版本。

项目在截止日之后产生的运行观测属于工程证据，不自动变成模型时代的背景知识。严格 profile 只接受 `observed_at` 不晚于截止日的观测；较新的观测继续保存在 `runs/` 和证据目录中，供现实工程使用，但不会被该 profile 的 Evolution 角色写回可迁移知识。

## 离线检查

```text
PYTHONPATH=src python3 -m infraagents knowledge --repository . temporal-audit --write
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

`knowledge audit` 仍检查正文、目录、来源归档和历史哈希；`knowledge temporal-audit` 负责时间资格。两者都不证明专题语义正确，也不证明目标模型、硬件或 SLO 已完成。时间 profile 只约束“这个事实能否在该模型时代被读到”。

后续 profile 可以选择更早的模型或更早的知识上限，但必须生成新的 profile ID 和审计报告；不能在同一个 profile 中静默改变截止日。
