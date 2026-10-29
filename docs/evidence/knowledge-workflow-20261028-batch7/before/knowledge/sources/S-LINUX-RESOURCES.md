# S-LINUX-RESOURCES：Linux 6.12 — cgroup v2 and powercap

## 本地完整原文

[完整原文](../../references/S-LINUX-RESOURCES/reader.html) · [全文文本](../../references/S-LINUX-RESOURCES/fulltext.txt)

固定版本：Linux v6.12 commit adc218676eef25575469234709c2d87185ca223a。版本坐标日期：2024-11-17；整理日期：2026-10-23。

## 来源支持的机制与定义

Linux v6.12 的完整 cgroup v2 文档定义资源层级、CPU quota/权重、内存压力/上限、cpuset、进程和设备/RDMA 控制；完整 powercap 文档定义计量域、能量单位/范围和功率约束窗口。

## InfraAgents 如何据此推导

推导推理主机实际可用 CPU、加载峰值、租户资源边界与能量统计，区分资源暴露、配额、独占及请求归因。

## 原文定位与阅读范围

已完整阅读 Documentation/admin-guide/cgroup-v2.rst 与 Documentation/power/powercap/powercap.rst，重点 CPU/Memory/Cpuset/Device/RDMA、ownership、writeback 和 power-zone/energy-counter 定义。

## 适用边界

绑定官方提交 adc218676eef25575469234709c2d87185ca223a，日期取 2024-11-17，不能把 cgroup 页面头部 October 2015 当作该版本日期。正文采用具体接口定义，不传播文档残存的泛化旧句（例如 IO 总述与后续 io.cost/io.max 支持说明并不一致）。具体控制依赖内核配置、驱动、层级和后端；RAPL 不是逐请求或全机能量计量。

seed 不表示当前 Task 上已实测有效；归档完整与编辑复核完成是两项不同事实。来源时间不证明模型训练实际使用该资料。
