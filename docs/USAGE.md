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

五角色实际模型身份必须与 profile 中的 `gpt-6-astra` 一致。此日期是官方知识截止日期，不是已披露的训练完成时间。源码版本日期使用固定提交时间，不能据此证明模型训练集成员关系。该分支保留 73 个专题；65 篇机制正文保留，8 篇修订具体实现、观察或来源范围。

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
