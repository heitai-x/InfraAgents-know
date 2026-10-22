# 知识库与 RSI 使用

本页说明已经实现的接口。研究取舍及主循环见 [RSI 设计](https://github.com/heitai-x/InfraAgents/blob/a102858e3bae595a4af88be6c3a1cd2d7a0f37f1/docs/11-rsi-design.md)；知识内容覆盖见 [矩阵](../knowledge/COVERAGE.md)。所有下列检查均可使用 CPU，不启动 GPU workload 或模型调用。

## 检索与查看

离线运行默认使用仓库中的时间 profile；严格 profile 的共同上限是模型 cutoff 与知识上限的较早者。当前 profile、逐来源判定和被阻断章节见 [时间边界与离线知识库](23-time-bounded-knowledge.md)。完整目录可审计不代表所有内容都能进入角色快照。

从仓库根目录使用：

```bash
PYTHONPATH=src python3 -m infraagents knowledge search "前缀命中后输出错误" --mode diagnosis --limit 3
PYTHONPATH=src python3 -m infraagents knowledge search "动态形状 重编译" --runtime pytorch
PYTHONPATH=src python3 -m infraagents knowledge show K-SYS-004
PYTHONPATH=src python3 -m infraagents knowledge show K-SYS-004 --revision 1
PYTHONPATH=src python3 -m infraagents knowledge audit
PYTHONPATH=src python3 -m infraagents knowledge temporal-audit --write
PYTHONPATH=src python3 -m infraagents rsi-report
```

在其他位置操作时，将 `--repository /path/to/InfraAgents` 放在 `knowledge` 与子命令之间；`rsi-report` 直接接受该参数。输出为 JSON。检索支持 design/diagnosis、类别、model-family、platform、runtime、workload 和结果数量限制。条件过滤使用目录中的精确名称，返回已检查字段；未提供的维度不假定已验证。

默认查询 seed/supported，`--include-inactive` 可以显式查看候选、冲突与弃用条目。结果给出版本、状态、来源、范围、匹配项与相关性分数。分数是本地词项导航分数，不是可信度或适用性概率。中文双字词项与英文术语都参与匹配，诊断入口额外关注诊断线索。

存在 temporal profile 时，`search` 和 `show` 默认也过滤 blocked 专题；只有明确使用 `--include-blocked` 做审计时才读取保留材料。

## 运行中的主循环

`run`/`resume` 直接接入 RSI。RSI 运行目录位于仓库内，使证据引用可追溯；知识库、运行记录和发布器使用同一套当前协议。

每轮先保存目录版本与正文哈希，并生成简短的知识入口。规划选择相关知识；每轮最多另推荐一条候选供考虑，上一轮新经验可立即进入下一轮修复。候选不会因为被推荐或在原任务重复使用就晋升，规划也可以跳过它。已声明的 CPU/CUDA accelerator 会用于基本平台过滤，其他条件仍需角色检查。

代码执行与诊断通过 knowledge_used 声明使用的 ID、版本和具体用途；运行器将正文冻结到本轮目录。进化看到实际验证结果后提出逐条反馈和局部增量。诊断对每项反馈与更新作 approve、reject 或 needs_evidence 判断。

新认识与修订均为 candidate。后续不同模型/workload 身份的任务中，确切版本被声明使用、反馈支持且位于适用范围、实际检查合格并通过独立核对时，生成新的 supported 版本。原任务重复、只改变预算、未真正采用、检查失败或反馈未通过，都不能促成晋升。

严格时间 profile 还要求引用的运行结果带 `observed_at`，且不晚于有效 cutoff；当前运行产生的较新观测继续保留，但不能借此发布模型时代的可迁移知识。

经过核对的范围内反例会把相同当前版本标为 contested；范围外失败不改变原结论状态。已有候选可以继续修订、明确弃用或恢复历史正文，恢复操作生成新候选版本。当前规则不自动优化模型权重、Agent 运行器或权限。


## 专家正文与章节修订

mechanism 容纳完整非代码 Markdown 正文，允许公式、表格、算例和主题自选章节。add 提供完整论证；revise 提供需要替换或新增的完整二级章节，以“## ”后的原标题精确匹配。原正文未出现的新标题会追加章节，重复标题或只有一段摘要的 revise 会被拒绝。未受影响的章节原样保留。

claim 继续用于目录摘要，conditions 表达范围，diagnostic_clues 和 validation_plan 表达由机制推出的预测与检验。来源与证据由发布器合并保留。contest/retire 不再重写主体，完整原文保留并附状态依据。写作标准见 [专家知识约定](../knowledge/templates/entry.md)。

## 主要产物

| 位置 | 内容 |
|---|---|
| knowledge/catalog.json | 条目身份、状态、范围、版本、正文和证据哈希 |
| knowledge/sources.json | 来源种类、版本、核对范围与限制 |
| knowledge/history/ | 旧版本正文与元数据，追加保留 |
| knowledge/events/ | 每轮暴露、候选、声明使用、核对反馈和状态变化 |
| runs/…/rounds/…/knowledge/ | 本轮快照、简短入口、冻结正文、选择与使用声明 |
| runs/…/rounds/…/evolution/ | 原提案与审查前冻结的证据清单 |
| runs/…/rounds/…/publish/ | 带前后哈希的可重放发布清单 |
| runs/…/rsi-report.json | 本次运行的知识反馈汇总 |

证据在提案后冻结，核对与发布期间发生变化时停止采用。正式正文、目录、历史、事件与索引按同一写入清单更新，恢复时继续已完成部分；发生外部编辑冲突时保留原内容并报告。没有引入数据库或向量服务。

每次知识使用都必须有明确声明和对应 revision；没有声明的运行不会自动补记。运行恢复使用保存的任务、工作项、知识快照和发布清单。

## 怎样解读效果

报告分别统计检索轮次、候选试用轮次、声明使用轮次、经核对的支持与反例、范围外反馈及未采纳反馈。保持原始任务和证据引用，成功、失败和不确定结果都可形成经验。

可比较运行点的指标变化只报告为观察。causal_improvement 保持 not_estimated；没有受控对照，不能从计数或分数变化宣称知识造成能力提升。目录审计检查结构、哈希与可访问性，也不代替语义审查。

## 验证入口

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

集成测试使用脚本化角色决策和真实 CPU 子进程/Git/文件操作，覆盖候选生成、不同任务试用、晋升、反例隔离、历史恢复，以及错误证据和重复发布。它们验证控制逻辑，不冒充模型自主学习效果实验。
