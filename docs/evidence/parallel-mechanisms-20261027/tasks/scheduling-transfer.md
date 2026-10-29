# 调度模块的迁移与诊断练习

这是一次交叉阅读与独立算例，不是受控知识增益实验。允许当前连续批处理正文、来源说明和相邻知识；不要读作者检查脚本、报告、其他答卷或上游源码。先完成第一阶段答卷，再等待反证。

本题采用单请求最多一轮在途。n是已知且确认的token长度，k是已完成连续KV长度，输入区间[k,n)已知。无投机、无前缀共享。A:(n=5,k=3,已分配1页)，B:(8,7,3页)，新请求C:(4,0,0页)。P=3，物理池共7页，现有映射独占，未在途。全层每token KV=96KiB。每轮Q≤5，同时要求设定时间预测T=20+2Q+0.5·pairs≤36μs，pairs=Σ[qk+q(q+1)/2]，本题忽略额外准备时间。

候选A2+B1+C2是否满足所有约束？推导至少两个合法候选的query数、pairs、时间和增量页数。选择A2+B1作为一次具体轨迹，明确每个packed row的请求身份和逻辑位置、哪些行允许采样；不能让未知的后续自回归token提前作为本轮输入。

该轮提交后B被取消，A正常完成并采到非EOS token。描述A的n/k怎样变化、B的页什么时候能回收、结果如何按原快照归属。随后重用B的外部名字建立新请求，注入原B的迟到及重复完成，解释怎样防止误提交和双重释放。若无法为A/C保留所需页，允许怎样回滚，不能只减少计数却继续执行原视图。

诊断题：“相同Q的两批耗时明显不同，所以调度器CPU准备一定有问题”。给出两个以上竞争解释，指出需要分开观测什么，并设计至少一次能推翻该单一解释的干预。

第一阶段保存 transfers/scheduling-answer.md、scheduling-answer.json，可写 scheduling-solution.py独立模型，不导入作者脚本。JSON含 rejected{Q,pairs,time_us,incremental_pages}，legal_candidates数组（每项alloc{A,B,C},Q,pairs,time_us,incremental_pages），selected_rows数组{request,position,sample}，A_after{n,k}，hypotheses，documents（文件SHA），scope。后续反证修改另存 scheduling-revision.md。
