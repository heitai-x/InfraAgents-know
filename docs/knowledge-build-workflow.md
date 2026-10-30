# 知识库构建 workflow

当前维护和执行入口已独立为项目级 [knowledge-workflow](../../../knowledge-workflow/README.md)。该目录包含模型调用、并发协调、确定性队列、隔离检查和登记代码，通过配置读取本知识库与 runtime。统一命令为 `python workflow.py --config workflow.local.toml <命令>`；旧 dated ops 执行器保留为历史证据并已拒绝启动。以下队列、审查和证据合同继续适用；日常启动、恢复与环境检查按新入口执行。

目标是把已经验证有用的构建过程变成可以连续执行、失败可定位、下一轮可恢复的工作。质量依据仍是[构建方法](knowledge-method.md)与[六项审查标准](knowledge-review.md)。工具负责队列、产物、实际检查和登记；Agent 负责技术推导、内容审查及根据反证修订。脚本接受一份审查记录，不意味着脚本能判断技术论证为真。

## 固定什么

每个专题的输入是当前正文、必要的来源说明和原文、相关知识、当前明确缺口。交付是修订后的知识、对六项标准的独立审查、适合该主题的实际检查，以及明确的验证边界。硬件映射、数值尺度或跨机制生命周期是实际问题需要时才展开，不规定所有章节复制同样例子或篇幅。

完成的含义是：六项标准在声明范围内有充分解释，没有未解决的阻塞问题，作者检查真实执行且对应当前字节，修订历史与导航登记完成。设备测量和 Agent 使用效果分别记录；未测 GPU 或未做受控知识对照不能写成已验证，也不自动否定已成立的理论内容。

## 执行一轮

从知识库根目录运行 `python -X utf8 tools/knowledge_build.py --run <本轮名称> <命令>`。

1. `init --entries all` 从 catalog 建立全库队列并保存准确基线。`--entries pending` 仅选择扩展状态仍待深化的条目；也可给逗号分隔的明确 ID。已有未登记正文修改时拒绝建立基线，先核清其归属。不要复用绑定旧 draft 的一次性登记脚本。
2. `next --owner <作者身份> --entry <ID>` 领取任务；`--limit` 可用于同一作者的批次。协调者独占队列写入，多个作者只改各自正文和产物。来源说明有共享调用方，由协调者分配单一责任方。
3. `prompt --entry <ID> --role author` 输出本次范围、来源、六项标准和遗留问题。作者先读当前内容，再做实质修订，保存算法检查及负对照，报告真正未解决的边界。不能为了通过流程只补标题或改状态。
4. 非作者用 `prompt --entry <ID> --role reviewer` 审查当前阅读链。审查 JSON 逐项引用当前文本；`review --entry <ID> --file <JSON相对路径>` 校验并保留该轮记录。`revise` 回到修订队列；`accept` 进入检查。旧轮次记录不覆盖。
5. `check --entry <ID>` 真正执行审阅者指定的本库 Python 检查，保留命令、退出码、输出、脚本及输入哈希。检查失败进入下一轮修订，不能只重标 PASS。内容或检查脚本改变后，旧通过记录失效。
6. 对代表性机制链做非作者使用练习：先保留正文未演示的任务、初答/实现和执行结果，再给反证并另存修正。`exercise --entry <ID> --file <JSON相对路径>` 记录有边界的使用证据。不是每个导航或说明条目都必须重复同类练习；未做的条目明确保留 `unassessed`。使用效果不能由结构检查替代。
7. `register --version <新的版本ID> --runtime-source <现有runtime的src>` 统一登记本批 `ready` 条目，保存准确旧正文/元数据，更新 catalog、来源说明哈希、导航、时间审计、VERSION、变更清单和扩展状态。没有实际正文变化不虚增 revision。随后 `verify-release --runtime-source <现有runtime的src>` 实际执行结构检查和本批完整正文/二级节/来源说明的冻结读取，绑定当前报告，状态由 `registered` 变为 `verified`。失败时解决具体原因，未完成验证不能称作本批验收完毕。
8. 全库仍有剩余任务时，`continue --next-run <下一批名称>` 以已登记版本建立下一批基线并继承未完成 ID、问题和轮数。继续领取、修订、复核和登记，直到队列全部完成，或遇到明确外部阻塞。`status` 始终报告实际状态，不把队列创建当作内容完成。

自动执行由调用本 workflow 的 Agent 协调：在授权范围内连续推进上述工作，不逐章请求确认。可用并行 Agent 时安排互不冲突的作者和非作者审阅；没有并行能力时按相同产物合同串行推进。该工具不内置模型账号、外部模型调用或驻留服务，单独运行 `init` 不会启动后台写作。

全量构建的历史执行记录保留在 `ops/knowledge-full-build-20261028/`，其旧执行器已停用。当前统一入口是项目级 `knowledge-workflow/workflow.py`：按所选 TOML 的 `repository.run` 和 `execution.state_dir` 定位队列、实际 PID、heartbeat、并发与活动数。作者、独立审阅和隔离检查使用有界并发流水，同一批选择互不改变彼此阅读材料的专题；失败保留候选与完整反馈，检查在真实隔离环境中运行。同一知识库只有一个协调者写队列。首次前向检验可限一章，完整执行沿选定队列推进。

当前登记器处理已有条目的修订与来源说明更新。增加或撤回来源、改变时间政策、扩展条目集合时，应先完成相应资料登记与资格核验，再建立匹配的新基线；不能把这些身份变化夹入已有修订批次。

每条任务默认最多三轮自动修订，目的在于暴露没有收敛的真实问题。达到轮数后记为 `needs_attention`，报告具体阻塞和证据；其他可独立任务继续执行。知识不足但可继续查证时应处理，不把难度当成需要用户批准的理由。若要继续该任务，应先说明失败原因，再显式调整本轮预算或重建任务，不能无条件重试直到出现接受结果。

## 审查记录

审查者实际完整阅读本章，并按需要阅读来源/相关章节。`documents` 记录真正读取的文件和 SHA，不用“已读全库”代替阅读范围。身份字段记录实际参与者，工具只能检查作者与审查者名称不同，不能由两个字符串证明隔离。

```json
{
  "entry_id": "K-ALG-002",
  "body_sha256": "当前正文SHA256",
  "reviewer": "/root/independent_reviewer",
  "decision": "accept",
  "documents": {"knowledge/algorithms/paged-kv-capacity.md": "当前正文SHA256"},
  "criteria": {
    "definition": {"assessment": "sufficient", "reason": "当前范围内的具体判断", "evidence": [{"path": "knowledge/algorithms/paged-kv-capacity.md", "quote": "正文中可核对的实际原句"}]},
    "legality": {},
    "algorithm": {},
    "cost": {},
    "adaptation": {},
    "diagnosis": {}
  },
  "findings": [],
  "limitations": ["实际仍未验证的设备或使用效果边界"],
  "checks": [{"script": "tools/check_paged_state.py", "args": []}]
}
```

示意中的后五项应各自填写与第一项相同的结构和真实证据，不能留空。`assessment` 为 `sufficient`、`partial` 或有具体理由的 `not_applicable`。任何未解决的 `partial` 或 `blocking: true` 的问题都会阻止接受。无需算法检查的主题应给 `checks: []` 和具体 `no_checks_reason`，不编造无意义的测试。

问题保存在每轮 `findings`：使用可追踪 ID、所属六项标准之一、当前证据、影响、阻塞与否和修订要求。新版本应判断已关闭、仍存在或被证据否定；不能机械沿用旧缺口。可以明确保留限制，不把期望中的最佳参数、未测设备或无关扩展都列为内容阻塞。审查后又改了正文或检查脚本时，用 `reopen --entry <ID> --reason <具体原因>` 保留旧轮并回到修订队列，再领取新一轮；已登记的修订由新基线任务处理。

## 使用练习记录

JSON 记录 `entry_id`、当前 `body_sha256`、`participant`、`task`、`initial_answer`、`feedback`、`revision`、`observed_outcome`、`limitations` 和 `documents`。四个产物字段是本库相对路径，均须出现在 `documents` 的哈希映射中。结果描述应覆盖真实实现与执行范围；共享上下文、未固定预算、没有操作系统隔离等限制如实保留。初答在反馈前保存，不能事后改写成一直正确。

## 本次启用

[首轮队列](evidence/knowledge-workflow-20261028/workflow.json)从已交付 draft5 的111章建立，首先处理分页 KV 的两个具体缺口。已经有深度证据的章节仍保留此前状态，不因加入队列而撤销既有结论。全库工作继续按当前字节验收，不能把一个例子跑通写成111章均已完成。
