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
