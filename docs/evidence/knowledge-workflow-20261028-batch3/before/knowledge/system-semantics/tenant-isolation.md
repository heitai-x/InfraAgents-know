# K-SYS-016：租户隔离：资源份额、缓存命名空间与故障边界

一个推理租户可以同时消耗 CPU、设备计算、KV 容量、adapter 驻留、网络、I/O 与响应缓冲。隔离设计应分别说明谁能够访问哪些对象、谁最多占用多少资源、谁在竞争时优先推进，以及一个故障影响哪些请求。限制其中一项不会自动建立其他保证。

## 从入口身份传到状态拥有者

入口确定的租户、模型/adapter 版本和请求身份，需要传递到调度、缓存、传输与输出。不同租户即使提交相同文本，也不默认取得对方的 KV、检索内容、adapter 或 trace。共享不可变公共前缀应以明确的公共对象契约和版本身份为前提，而不是通过发现相同 hash 就绕过授权。

请求 ID 本身不是访问凭证。对共享缓存，授权范围与数值兼容性同时满足才允许命中；页句柄、远端地址和日志引用也不能只依赖可猜的槽号。过期版本退出目录后，在途访问仍要结束，最后回收时清除对新拥有者不应暴露的内容或重新初始化全部可读范围。

trace 保存排队、形状、长度、代次与资源事件通常已足以做归因。确需记录原始输入或输出时，把存取范围、保留时间和删除责任纳入同一对象生命周期。租户隔离不是要求所有状态复制一份，而是要求共享的条件和可见范围可验证。

## 主机限额的实际含义

[Linux 6.12 cgroup v2](../sources/S-LINUX-RESOURCES.md) 区分权重、上限、保护与独占分配。CPU 权重决定竞争时的相对份额，CPU 带宽上限约束一段时间可消耗的执行时间，cpuset 约束可运行的 CPU/内存节点；它们不能互相替代。CPU 配额不保证独占 cache、内存带宽或确定的尾延迟。

memory.high 会引入回收压力和节流，memory.max 在无法回收时可触发 cgroup 内 OOM；文件页缓存、匿名内存与部分内核/网络缓冲也可能计入预算。只统计模型 tensor 会遗漏这些消耗。内存控制器不是所有 GPU 显存对象的自动配额器，GPU 容量仍需执行器和实际设备机制约束。RDMA 对象数、线程/进程数量与存储流量也可能先于字节预算耗尽。

## GPU 共享必须绑定具体机制

下列界限取自固定的 [NVIDIA device plugin v0.16.1](../sources/S-NVIDIA-PLUGIN.md) README。这里说明该版本公开的资源与部署契约，不把插件文档扩展成所有设备、驱动或攻击场景的安全证明。

| 机制 | 该来源明确支持的理解 | 不能由资源名称推出的保证 |
|---|---|---|
| 整卡分配 | 将指定设备暴露给容器，区分 UUID/index 等身份传递方式 | 其他宿主进程是否仍可访问、整机故障是否隔离 |
| CUDA time-slicing | 多个访问份额指向同一底层 GPU；文档明确没有额外的显存与故障域隔离 | 每份额对应固定显存或固定吞吐；申请更多份额按比例得到算力 |
| 插件的 MPS 共享 | 控制进程按份额限制客户端可用显存与计算能力；该版本只支持完整 GPU，并不支持启用 MIG 的设备 | 与独占物理设备相同的尾延迟、性能无干扰或覆盖所有错误的隔离 |
| MIG 资源暴露 | 在硬件支持前提下，none/single/mixed 策略决定暴露方式；mixed 可暴露按 profile 命名的资源 | 任意 profile 组合均合法、profile 吞吐线性缩放，或共享宿主/网络/供电故障自动消失 |

MIG、MPS 与 time-slicing 不应被合并成一个“虚拟 GPU”容量数字。MIG profile 需要对应实际硬件支持与配置，多个访问份额也可能继续共享同一 MIG 资源。更细的硬件隔离承诺需要相应固定硬件文档和故障测试；本章不从插件的资源暴露推断未记录的保证。

## 容量、公平与故障预算

设定共享设备的 KV 池为 8 GiB，租户 A/B 各保留 3 GiB，另有 2 GiB 可借用池。借用可在空闲时增加利用率，但 A 已有活跃请求占用的借用状态，不能在 B 到来时立即按账面扣回；必须有合法结束、迁移、重算或拒绝路线。仅设“每租户 10 请求”的限额，会让长上下文租户占据远多于短请求租户的容量。

再设一张卡被 time-slicing 暴露为 8 份。接收 8 个各需要 10 GiB 的服务进程，不会把物理显存变成 80 GiB。共享份额数量只允许更多访问者，模型加载和峰值容量仍需真实字节账。同样，空闲时一个租户借到更多设备时间，不代表竞争时能继续获得相同服务率。

## 验证与选择改变

构造两租户相同输入但不同权限/adapter 的请求，检查缓存、结果、日志与迟到消息无越界归属；再让一个租户持续增长 KV、生成超长响应、启动大量冷 adapter、消耗 CPU 配额与存储流量。观察另一个租户的准入、尾延迟和可完成工作量，不能只检查进程没有退出。

故障测试按已承诺的边界设计：终止一个 worker、制造配额内存不足、使一个共享设备不可用或让网络中断，确认影响范围、可重试语义和清理责任。对全卡故障、宿主故障与租户进程错误分别记录，不能把某一种测试通过推广到全部错误。

严格隔离需求可能要求独占设备或独立执行域；低占用且允许干扰的服务可采用共享。统一基座内的多租户执行节省内存，却把应用调度器变成主要容量和身份边界，必须同时有明确的资源账与独立污染测试。

<!-- infraagents:provenance -->

## 来源与推导

- [S-LINUX-RESOURCES：Linux 6.12 — cgroup v2 and powercap](../sources/S-LINUX-RESOURCES.md)：Linux v6.12 的完整 cgroup v2 文档定义资源层级、CPU quota/权重、内存压力/上限、cpuset、进程和设备/RDMA 控制；完整 powercap 文档定义计量域、能量单位/范围和功率约束窗口。
  [完整原文](../../references/S-LINUX-RESOURCES/reader.html) · [全文文本](../../references/S-LINUX-RESOURCES/fulltext.txt)

- [S-NVIDIA-PLUGIN：NVIDIA Kubernetes device plugin v0.16.1 — deployment and sharing](../sources/S-NVIDIA-PLUGIN.md)：v0.16.1 README 定义设备暴露、UUID/index、运行时注入方式、MIG 策略、time-slicing 与 MPS 的配置边界和升级排空约定。
  [完整原文](../../references/S-NVIDIA-PLUGIN/reader.html) · [全文文本](../../references/S-NVIDIA-PLUGIN/fulltext.txt)

- [S-BORG：Large-scale cluster management at Google with Borg](../sources/S-BORG.md)：§2 任务状态、资源向量、优先级/配额；§3 可行性与评分、制品局部性；§4 相关故障；§5 装箱、碎片和资源回收；§6 隔离；§8 控制服务的组织经验。
  [完整原文](../../references/S-BORG/reader.html) · [全文文本](../../references/S-BORG/fulltext.txt)

- [S-VLLM-PREFIX：vLLM — Automatic Prefix Caching](../sources/S-VLLM-PREFIX.md)：所核对 vLLM 版本以块身份组织前缀复用，身份可包含父块、token 和额外条件。不同上下文的同 token 块不应自动视为相同计算。
  [完整原文](../../references/S-VLLM-PREFIX/reader.html) · [全文文本](../../references/S-VLLM-PREFIX/fulltext.txt)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [请求身份：逻辑序列、批次视图与结果归属](../system-semantics/request-batch-identity.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
- [服务协议：接收、流式背压与终止语义](../system-semantics/service-protocol.md)。
- [前缀缓存：计算等价类、Radix 与淘汰收益](../algorithms/prefix-cache-identity.md)。
- [多模型放置：共同容量、混合执行与租户公平](../workloads/multi-model-placement.md)。
- [集群放置：资源向量、拓扑、执行组与故障域](../workloads/cluster-resource-placement.md)。
- [部署环境：版本契约、可见设备与真实资源边界](../system-semantics/deployment-environment.md)。
- [失败分析：最早偏离、最小复现与机制归因](../tools/failure-evidence.md)。
