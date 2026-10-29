# 构建 workflow 的隔离前向测试

测试者：`/root/workflow_forward_test`；日期：2026-10-28。按新 skill `C:/Users/26815/.codex/skills/infraagents-knowledge-build/SKILL.md` 与[工作流合同](../../knowledge-build-workflow.md)，使用独立 tempfile 小知识库执行队列、审查和真实 Python 检查；没有修改真实111章队列、正文或登记状态。测试者继承父对话并可检查工具实现，属于独立执行复核，不是盲测或知识效果对照。

## 已实际跑通

[行为测试](../../../tools/test_knowledge_build.py)最终执行25项，全部通过；当前机器可复跑：`python -X utf8 tools/test_knowledge_build.py --report docs/evidence/knowledge-workflow-20261028/workflow-tests.json`。结果与工具/测试SHA保存在[当前报告](workflow-tests.json)。

端到端命令行测试真实调用 `init → next → review → check → status`，其中检查在子进程中执行算术断言并留下stdout/退出码。一个条目进入ready而另一条仍queued时，状态没有报告全库完成。

失败恢复测试让检查真实退出7，确认进入checks_failed；领取下一轮、修复检查、重新审查与运行后进入ready，原失败轮的审查与日志字节保持不变。检查超时、脚本在执行中自修改也会保留失败证据。当前正文、来源阅读、具体引用、审查者身份和检查脚本变化的拒绝路径均有实际执行。

剩余队列测试核对：预算耗尽的单条任务进入needs_attention，其他任务仍可领取；登记状态registered没有被当成verified；继续下一基线时，继承剩余条目的修订轮、问题和实际状态。release实现仅在协调者的登记前guard测试中用具有真实返回合同的stub替代；发布登记、结构验证和冻结读取由各自专项检查承担，本报告不冒充已经验证这些实现。

## 测试真正发现并促成的修复

第一次23项测试有1项失败，保留在[第一次报告](workflow-tests-initial.json)：下一批继承revise状态后，返回摘要仍沿用initialize时的queued计数。协调者将返回值改为继承完成后的summary；后续测试通过。

补充反例后，第二次24项测试有1项失败，保留在[第二次报告](workflow-tests-second.json)：checks_failed状态可重新提交完全相同的原审查，重置为reviewed并清空checks，从而允许原轮重跑并覆盖失败日志、绕开轮数预算。协调者修订为失败检查必须进入新轮；已有reviewed状态的相同审查只幂等返回，不倒退状态或清空证据。该反例与原恢复路径在最终测试中均通过。

测试期工具的登记摘要接口新增必需返回字段后，测试stub曾与真实合同不一致。测试改为提供真实最小返回结构，保留工具的严格字段访问，没有要求工具吞掉错误或增加无依据兜底。这个fixture修正不列为workflow缺陷。

## 能够支持的结论与限制

当前证据支持上述队列、输入绑定、真实检查、失败恢复和剩余工作管理在这些隔离场景中按合同执行。审查中的六项判断仍由实际Agent负责；具备六项字段和原句并不证明技术论证为真。作者/审查者名字不同也不能证明上下文隔离。

测试没有调用模型写作，没有在真实全库完成所有条目，没有运行GPU、graph、并发allocator或受控知识收益实验。未来工具实现改变后应重新运行这些行为测试，不能沿用本次SHA绑定冒称新版本已通过。
