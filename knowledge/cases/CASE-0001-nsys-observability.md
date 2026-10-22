# CASE-0001：执行成功但 CUDA 观测缺失

## 条件与问题

2026-09-19，在 RTX 3090、驱动 580.95.05、Nsight Systems 2025.3.2 上，验证 InfraAgents 是否能够从原始采集准确区分执行情况和观测情况。

用例处理 4096 个整数，分配 16 KiB 设备内存，执行三次加一并检查全部结果。单进程、单设备、默认 stream；未涉及模型推理。

## 干预与观察

| 条件 | 程序结果 | 采集与分析结果 |
|---|---|---|
| 无 profiler | 结果全部正确 | 仅作为正确性对照 |
| 开启 CUDA tracing | 结果全部正确 | 3 次 kernel、2 次拷贝、12 次主机 API、1 次同步记录 |
| 关闭 tracing | 结果全部正确 | 采集成功，CUDA 事件表缺失 |
| 在第一份报告内缩小时间窗口 | 不重新运行程序 | 找到一个 kernel 的 1400 ns 区间，保留对应主机 API |

原生统计与 InfraAgents 对 kernel、拷贝及主机 API 的次数和累计时长计算一致。5 个设备事件均获得唯一的同进程 API 候选。首个 kernel 的开始时间比其主机调用结束早 1116 ns。

## 得到支持的认识

观测缺失不能直接推断工作未发生；同一份报告中的主机与设备执行区间可能重叠；局部窗口查询需要保留窗口外的关联依据。

本例没有测量 profiler 扰动，没有评估模型数值、TTFT、TPOT、吞吐或 SLO。没有从微型用例推导优化收益，也没有把合成测试中的多进程关联称为真实多进程验证。

## 原始证据

[工程证据清单](../../docs/evidence/nsys-feedback-20260919.json) 保存原始采集、导出、分析、原生统计、测试和源文件的路径、大小及 SHA-256。原始产物留在远端项目的 runs/nsys-feedback-20260919T154328Z 下，未放入知识正文或提交大型 trace。

本次采集方法、执行边界和复现入口见 [工程实现说明](../../docs/09-nsys-feedback-implementation.md)。

## 知识更新

[P-0001](../proposals/P-0001-nsys-feedback.md) 将本例归入 [K-TOOLS-001](../tools/trace-observability.md) 与 [K-SYS-001](../system-semantics/host-device-timing.md)。本次核对由工程实现过程完成，自动四角色调度尚未参与。
