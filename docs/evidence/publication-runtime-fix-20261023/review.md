# RSI发布字节一致性修复独立审查

**结论：PASS。** 绑定修复运行时 `6d205853b6135adbc35b43525608ff919eb4dee6`，路径 `/mnt/infra_agents/worktrees/gpu-knowledge-publication-20261023`。真实108篇bank和w001冻结批准反馈的隔离发布/幂等验证通过；四项CRLF状态与历史边界验证也通过。没有发现该补丁新的阻断问题。

本轮范围是RSI发布一致性。此前只读probe覆盖读取、版本声明、导航和原文访问限制，**没有覆盖发布**，不能把此前PASS追溯成已验证这条发布路径。

## 根因与修复范围

真实K-FW-011有64个CRLF，原始字节SHA为 `871e49626f320d579993063db039ab0a4b8e98394a54ffa2073daa9ed8bb2cb5`，与catalog一致；`read_text().encode()`通用换行后变为 `85e664ad5cc056b4ab3f082d230a9baca4d377cfc9f70a2816fe9843298f8ab3`。全库共8篇存在这种转换差异。

发布入口validate_catalog按read_bytes验证，随后candidate全库循环却将未修改正文用read_text读取并再次算SHA。因此updates为空的feedback-only发布也会失败，K-FW-011只是第一个触发者。同样的隐式转换还会破坏archive历史或使replay忽略外部仅换行变化。

补丁增加精确UTF-8解码helper，并替换8处身份/正文读取；未移除或放宽SHA、审查、证据或replay冲突检查。JSON语义解析保留原行为，计划写出仍是明确UTF-8字节。新revision的章节格式化与原revision身份是不同边界：可以按既有section编辑器生成新正文，但必须精确保留旧history，不能批量改写seed换行来掩盖问题。

## 独立验证

| 场景 | 结果与关键检查 |
|---|---|
| 0802c3f + 真实108篇bank/w001 | 精确复现K-FW-011 candidate SHA错误；失败前后knowledge字节未变，pending/event均未产生。 |
| 6d20585 + 同一真实冻结提案/独立review | 两项原批准反馈成功应用，changes为空；108正文和全部既有历史字节保持，204个当前/历史版本完整性通过。 |
| 同一真实发布重复调用 | 回执相同，knowledge无第二次写入，无新增revision/重复反馈event；Graph、selection及冻结证据保持。 |
| contest / retire | 旧CRLF正文和metadata精确进入history；状态前缀后的原正文逐字节保留；其他107篇未改，重复发布幂等。 |
| restore | CRLF原始历史和LF+CRLF中间历史均保持；恢复正文含精确原始后缀，新revision/hash正确，重复调用幂等。 |
| feedback导致contested | 旧历史及原正文后缀精确保持、正确增加一个revision，其他107篇未改，重复调用幂等。 |

真实提案从Graph已提交的Evolution/独立review提取，确认两者相邻且与冻结proposal一致，13份冻结证据哈希全部吻合。合成状态操作只验证发布器字节/生命周期，使用明确标记的synthetic fixture，不冒充真实推理结论。详细结果见 [基线复现](baseline0802-replay.json)、[真实反馈及首次fixture记录](fixed6d20585-replay.json)、[四项状态操作最终结果](fixed6d20585-date-fixture-replay.json)。

四项synthetic首轮因审查脚本将observed_at写成完整时间戳，被既有ISO日期合同在写入前拒绝；改为2026-10-23后仅重跑这四项，全部通过。该次fixture失败原样保留，不归因于产品。作者另执行的56项相关回归日志为OK；已审查其4个focused测试，包括partial replay和newline-only外部冲突，未把作者测试宣称为审查者另行执行。

## 实际run与恢复边界

实际bank检查到没有pending-plan、w001发布event或Controller publication.json，失败发生在写计划创建前。隔离回放前后，实际bank知识文件、Graph和全部冻结证据均未变。Graph SHA为 `4c0b9580fe60481334ea1cfd5f075efa8211c2bb945fc066460b1fb6a0c60f02`；已消耗6 calls、7,591,048 tokens，截止时间为 `1792759425.0301414`，这些值没有重置。

root可以按既定流程以6d20585恢复同一个run的框架发布步骤，继续使用原Task、w001、冻结提案、独立批准和实验结果，不重跑模型实验。发生中途发布时，只允许精确before/after字节状态恢复；不得通过换行归一化绕过冲突。恢复历史、失败记录、事件幂等和独立证据边界必须保持。

本审查没有执行真实Controller恢复或随后跨Task资产记账；这两项仍由root串行操作并核对发布回执/event。审查者只在报告目录及独立副本写入，没有改实际run bank、canonical、shared runtime或源码，也没有启动Agent/模型/GPU或推送。
