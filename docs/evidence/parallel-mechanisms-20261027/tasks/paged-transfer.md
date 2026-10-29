# 分页模块的迁移与诊断练习

这是一次作者之外的交叉阅读练习，不能用于声称受控知识增益。允许阅读当前分页正文、其来源说明及正文链接的知识；不要阅读作者检查脚本、作者报告、其他答卷或上游实现。保留本任务原文。先写第一阶段答卷，再等待审阅者提供反证。

设单层，P=3，Hkv=2，dk=3，dv=5，K/V 均 FP16，两个独立的紧密物理池，页从0编号；页内不含额外padding。请求 A/B 长度均为5，共享页表[5,2]，第二页仅前2个token有效。先分别对 NHD=[page,token,head,channel]、HND=[page,head,token,channel]推导 t=4、head=1 的 K通道2和V通道4相对池基址的字节偏移；解释只改描述符会发生什么。

旧 A 快照 R0 已取得[5,2]的在途读取权。A 得到一个合法已知输入 token，选物理页7复制尾页有效内容并追加，发布新快照[5,7]、长度6，R1取得新快照读取权；然后取消 A。B 继续拥有[5,2]。没有其他所有者或读者，复制写者已完成。给出每个物理页的所有者数、在途快照读者数、何时可回收；比较 R0/R1 两种完成次序及重复完成。算清 COW 有效前缀的读取、写入字节，以及追加 token 写入字节，区分物理容量与这些流量。

另一个容量题：13条独占请求长度都为5，每token全层KV=96KiB。比较P=3与P=8的有效、分配、浪费字节和页表项数。页表项4字节，每条请求每次处理页表项的设定成本0.3μs；忽略其他准备差异。不能据此声称最佳页长，因为读取流量、碎片和COW仍可能改变选择。

对“取消与快速复用时偶发读到另一请求的尾块”至少提出两个竞争解释及不同预测，给有区分力的干预。不要在证据不足时选唯一原因。

将第一阶段答案写到本目录上一级的 transfers/paged-answer.md 和 paged-answer.json，可另写 paged-solution.py独立小实现（不导入作者检查脚本）。JSON请含：offsets{NHD_K,NHD_V,HND_K,HND_V}，after_cancel{5:{owners,pins},2:{owners,pins},7:{owners,pins}}，cow{read_bytes,write_bytes,append_bytes}，capacity{P3:{allocated_bytes,wasted_bytes,table_entries},P8:{同字段}}，hypotheses，documents（实际阅读文件及SHA），scope。等待反证后，另存 paged-revision.md，不覆盖初始判断。
