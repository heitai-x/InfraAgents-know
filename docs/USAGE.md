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

## v2 的读取入口与已验证运行时

v2 保留全部 108 篇机制正文与 114 项固定来源。完整选择/读取验收见 [Linux 记录](evidence/final-linux-validation.json)。选定视图的导航修复位于 [运行时分支](https://github.com/heitai-x/InfraAgents/tree/codex/knowledge-navigation-20261023)，提交 `aedff22c0885c4fc56db6058b9904e23d5e189b9`；使用修复后的框架可取得按冻结目录生成的设计与诊断入口。其 src/tests 基线与实际验证版本一致，详见 [发布身份](evidence/runtime-navigation-published.json)。

更换知识版本应创建新任务或新的受控运行；既有冻结运行继续使用原内容和模型身份。知识库更新没有修改正在运行的实验，也没有将当前运行学习伪装成截止日前 seed。

## v2.1 的运行期反馈与发布

使用这份知识库执行 RSI 反馈或章节修订时，采用包含提交 `6d205853b6135adbc35b43525608ff919eb4dee6` 的运行时，见 [已验证的 GPU 兼容运行时分支](https://github.com/heitai-x/InfraAgents/tree/codex/knowledge-publication-bytes-20261023)。该版本同时包含选定视图导航修复，以及发布、历史保存和中断恢复中的字节保持读取。

真实运行曾发现：旧发布器把 CRLF 正文读取成 LF 后再计算哈希，导致未修改章节也无法通过发布校验。知识正文与登记哈希本身一致。修复保留所有校验和原始换行，已通过 56 项相关回归、真实 108 篇知识库的批准反馈回放与幂等检查，以及状态修订和历史恢复检查，见 [独立审查](evidence/publication-runtime-fix-20261023/review.md)。此前读取验证没有覆盖这条发布路径。

如使用同一份限时 seed 配合 Luna 等其他实际模型，可以按 Task 选择 `full` 与 `seed_and_learned`；这不声明这些模型具有 Astra 的训练或知识截止日期。为运行建立独立可写知识仓库并将 runs 放在其中，运行期反馈和 learned 保留真实观察日期；已发布的限时 seed 标签保持固定。此次实际接入使用的初始知识内容仍为 `astra-2026-04-30-v2.1`，没有为运行修复改写章节或原始资料。

## 修订版的内容检查

v2.2-r1仍通过原有knowledge search/read与knowledge_used版本绑定使用。七组问题只是无序、非穷尽的阅读示例，不是阶段状态机；它们不要求角色检索、提交统一产物或先完成某层验证才准许探索。详见 [使用问题与探索边界](knowledge-use-cases.md)。

在知识仓库根目录执行结构与算例检查：

```bash
python3 tools/validate_knowledge.py --baseline f640148464929f803a53b55e27ee4cbb83dc8657 --report /tmp/knowledge-structure.json
python3 tools/check_knowledge_examples.py --report /tmp/knowledge-examples.json
```

本节探针通过章节 ID、`ID@revision` 和正文路径核对同一冻结版本，需要包含读取地址修复提交 [`07d114a451e42d382381bee4cd215f1ebd8630a5`](https://github.com/heitai-x/InfraAgents/commit/07d114a451e42d382381bee4cd215f1ebd8630a5) 的运行时。本次发布前补验直接使用该提交，其历史包含前述 `6d20585` 的字节保持发布修复。仅使用旧 `6d20585` 时，探针的 `ID@revision` 请求会因未解析为正文路径而被后缀检查拒绝；它在 v2.1 的发布验证不代表支持这项读取形式。四视图补验结果与内容身份见 [发布前验证回执](evidence/knowledge-publication-check-20261024.json)。

用包含上述修复的运行时源码执行实际冻结读取与访问边界检查：

```bash
python3 tools/probe_knowledge_runtime.py --runtime /path/to/InfraAgents --repository . --report /tmp/knowledge-runtime.json
```

探针在独立副本上读取当前正文和历史，校验引用，并检查撤回ID与受限原文不进入当前视图。模型名是选择条件fixture，没有调用模型。查询命中仅观察可找到哪些入口，不要求实现按入口方案构建；新问题或未命中不构成引擎任务失败。

随包原复审按源码模块SHA记录当时所用运行时；当时推荐分支获取要求GitHub凭据，未重新验证其发布器。[原复审报告](evidence/knowledge-open-exploration-20261024/runtime-probe.json)只支持其实际检查的接口。本次发布前补验另以精确的 `07d114a` 源码验证读取，仍未重跑知识发布或真实Agent构建实验。正文去除固定流程、目录去重和读取通过，都不能代替真实Agent行为实验。
