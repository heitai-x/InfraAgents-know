# 选择和验证知识分支

仓库根目录保持 `knowledge/`、`references/` 和必要 `docs/`，可以直接作为已有 `--knowledge-repository` 的参数。分支的 `VERSION.json` 保存来源和内容身份。当前运行保持已记录的知识选择；更换内容版本使用新的任务或独立检出。

## main：完整 base

```toml
[knowledge]
view = "full"
condition = "seed"
```

main 保留 73 个原始 seed 专题、80 项原始来源和这些章节的历史版本。它没有活动的模型时间 profile；较晚的来源、原 base 已引用的工程微型案例仍明确保留原日期。后续运行新增的学习章节和运行状态没有混入 base。

## Astra：2026-04-30 内容版本

```toml
[knowledge]
view = "attributable"
condition = "seed"
profile = "knowledge/temporal-policy.json"
freeze_initial = true
publication = "deferred"
```

五角色实际模型身份必须与 profile 中的 `gpt-6-astra` 一致。此日期是官方知识截止日期，不是已披露的训练完成时间。源码版本日期使用固定提交时间，不能据此证明模型训练集成员关系。最初 v1 含 73 篇，65 篇机制正文保留、8 篇修订具体内容；当前扩充版的计数与状态以根目录 VERSION.json 和构建记录为准。

原始实现的访问限制继续由 Task 的 `restricted_originals` 决定；分支或时间资格本身不替代这些权限。

## 验证

在已有 InfraAgents 工作树中执行，替换知识仓库的实际路径：

```bash
PYTHONPATH=src python3 -m infraagents knowledge --repository /path/to/InfraAgents-know audit
```

Astra 分支另外可以执行：

```bash
PYTHONPATH=src python3 -m infraagents knowledge --repository /path/to/InfraAgents-know temporal-audit
```

仓库没有复制另一套 Controller、角色实现或运行时。正文更新沿用现有章节增量和版本历史；正式发布前仍需独立复核。Git 属性保留知识与原文的精确字节，防止换行转换破坏已有归档校验。

## v2 的读取入口与已验证运行时（历史）

v2 保留全部 108 篇机制正文与 114 项固定来源。完整选择/读取验收见 [Linux 记录](evidence/final-linux-validation.json)。选定视图的导航修复位于 [运行时分支](https://github.com/heitai-x/InfraAgents/tree/codex/knowledge-navigation-20261023)，提交 `aedff22c0885c4fc56db6058b9904e23d5e189b9`；使用修复后的框架可取得按冻结目录生成的设计与诊断入口。其 src/tests 基线与实际验证版本一致，详见 [发布身份](evidence/runtime-navigation-published.json)。

更换知识版本应创建新任务或新的受控运行；既有冻结运行继续使用原内容和模型身份。知识库更新没有修改正在运行的实验，也没有将当前运行学习伪装成截止日前 seed。

## v2.1 的运行期反馈与发布（历史）

v2.1 当时的 RSI 反馈和章节修订接入使用包含提交 `6d205853b6135adbc35b43525608ff919eb4dee6` 的运行时，见 [已验证的 GPU 兼容运行时分支](https://github.com/heitai-x/InfraAgents/tree/codex/knowledge-publication-bytes-20261023)。该版本同时包含选定视图导航修复，以及发布、历史保存和中断恢复中的字节保持读取。

真实运行曾发现：旧发布器把 CRLF 正文读取成 LF 后再计算哈希，导致未修改章节也无法通过发布校验。知识正文与登记哈希本身一致。修复保留所有校验和原始换行，已通过 56 项相关回归、真实 108 篇知识库的批准反馈回放与幂等检查，以及状态修订和历史恢复检查，见 [独立审查](evidence/publication-runtime-fix-20261023/review.md)。此前读取验证没有覆盖这条发布路径。

如使用同一份限时 seed 配合 Luna 等其他实际模型，可以按 Task 选择 `full` 与 `seed_and_learned`；这不声明这些模型具有 Astra 的训练或知识截止日期。为运行建立独立可写知识仓库并将 runs 放在其中，运行期反馈和 learned 保留真实观察日期；已发布的限时 seed 标签保持固定。此次实际接入使用的初始知识内容仍为 `astra-2026-04-30-v2.1`，没有为运行修复改写章节或原始资料。

## 当前 v2.2-r2 的内容与接口检查

当前探针对应框架已采用的准确 ID / 路径读取接口。角色的采用声明只提交 `entry_id` 和 `purpose`；探针调用所提供运行时的 `KnowledgeUse` 解析器与 `bind_handoff`，由框架取得冻结修订号，再调用内部 `validate_uses`。调用者提供 `revision` 或使用 `ID@revision` 会被拒绝。内部归档和版本检查继续保留准确身份。

七组问题是无序、非穷尽的阅读示例，不要求角色检索、提交统一产物或按固定阶段探索，见 [使用问题与探索边界](knowledge-use-cases.md)。

在知识仓库根目录执行：

```bash
python3 tools/validate_knowledge.py --report /tmp/knowledge-structure.json
python3 tools/check_knowledge_examples.py --report /tmp/knowledge-examples.json
python3 tools/probe_knowledge_runtime.py --runtime /path/to/InfraAgents --repository . --report /tmp/knowledge-runtime.json
```

内容快照不带 Git 历史，默认结构检查使用已保存的前版清单与正文。只有在真实 Git 检出中具备相应历史对象时，才额外使用 `--baseline f640148464929f803a53b55e27ee4cbb83dc8657` 检查最初基线。

正文或时间 profile 改变后，用所选框架的现有审计函数重新生成根目录报告，再运行结构检查：

```bash
PYTHONPATH=/path/to/InfraAgents/src python3 -c "from pathlib import Path; from infra_agent.temporal import audit; audit(Path('.'), write_report=True)"
```

结构检查核对审计的 profile、章节集合、修订号、来源集合与截止日，避免旧版本的通过记录继续冒充当前审计。来源说明中的本地 Markdown 链接也会检查；修正来源说明时保留前版原文，固定来源版本和原始归档不变。

探针在独立副本中验证四种知识视图，完整读取当前正文和历史，经过框架绑定检查采用声明，并核对未读、重复、撤回及受限原文的拒绝行为。模型名仅是选择条件 fixture，没有调用模型。查询命中只作为检索观察，不是引擎任务门禁。

本轮报告及使用的实际运行时模块身份见 [修改与验证](evidence/knowledge-cleanup-20261027/review.md)。上述历史运行时曾验证的发布路径不自动成为当前验证结论；本轮不执行知识发布、真实 Agent 构建、模型请求或 GPU 性能实验。
