# 流水模块的迁移与诊断练习

这是一次交叉阅读与新shape推导，不是GPU实测或受控知识增益实验。允许当前GPU流水正文、来源说明及相邻知识；不要读作者检查脚本、报告、其他答卷或上游实现。先保存第一阶段答卷，再等待反证。

为一个FP32 SIMT GEMM设计输出tile8×24、阶段K=5的映射，CTA64线程。使用i=tid+64t，t=0,1,2，row=floor(i/24)，col=i mod24分配192个输出；每个线程维护其3个FP32累加量。A共享tile8×5，B共享tile5×24，两warp都作为输入缓冲读者。推导全局/共享地址、搬运覆盖、尾K零填、输出尾行列保护，以及每级输入共享字节和持久逻辑累加字节。允许普通同步搬运实现；如果采用异步，要指出可见性和槽复用的完成条件，不能把软件事件当作CUDA语义证明。

设固定延迟L=7、连续copy发起最小间隔I=2、单CTA每阶段compute时间C=4，单位为设定时间；每个槽的上次compute完成才可写新代。N=4阶段。第i次copy发起s_i=max(s_(i-1)+I,e_(i-d))，不存在的索引限制取0；完成f_i=s_i+L；compute完成e_i=max(f_i,e_(i-1))+C，e_-1=0。分别算d=1/2/3的总时间及事件表，包含N=0/1/2的边界。不能只用稳态下界估计N=4。

合成设备仅1个SM、每SM最多驻留2个CTA，忽略其他资源上限；每级共享由上面tile确定。共享上限先为2048B，再为4096B。网格4个CTA，采用简单不交叠的wave模型T_grid=ceil(4/R)·T_CTA，R=min(2,floor(shared_limit/(d*shared_per_stage)))。给两个容量下的驻留数、网格时间与选择反转。这是限定的合成模型，真实共享带宽与CTA间争用未建模。

诊断题：“d=3较慢，一定是首尾成本”。提出竞争解释、不同预测和能区分它们的干预，保留未确定部分。

保存 transfers/pipeline-answer.md、pipeline-answer.json，可写 pipeline-solution.py不导入作者脚本。JSON含 shared_per_stage_bytes,accumulator_bytes,cta_times{1,2,3},grid{2048:{residency:{1,2,3},times:{1,2,3}},4096:{同字段}},hypotheses,documents（实际文件SHA）,scope。反证阶段另存 pipeline-revision.md。
