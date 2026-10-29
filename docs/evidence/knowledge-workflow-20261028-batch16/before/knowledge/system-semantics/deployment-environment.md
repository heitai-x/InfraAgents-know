# K-SYS-019：部署环境：版本契约、可见设备与真实资源边界

部署描述必须能够解释模型实际运行在哪些资源上。镜像、宿主内核/驱动、设备暴露、用户态库、CPU/内存限制、文件与网络路径共同决定可执行性和性能。容器能够启动，不代表这些条件已经形成有效的推理环境。

## 环境身份与相容关系

记录模型制品、执行器和编译器版本，用户态计算/通信库，宿主驱动与内核，以及实际设备能力。兼容关系按目标平台的固定文档和验证确定，不能由版本号大小或某个历史 README 的最低要求推断所有组合都可用。编译缓存还需绑定目标能力和相关编译条件；拷贝已编译文件不保证另一类设备能执行。

镜像携带的用户态依赖与宿主提供的设备驱动、设备文件和资源控制属于不同层。问题可能表现为设备完全不可见、库初始化失败、某个 kernel 路径不支持，或通信路径降为另一种传输。应记录实际选择，不能只写“使用 GPU”或“启用 RDMA”。

[NVIDIA device plugin v0.16.1](../sources/S-NVIDIA-PLUGIN.md) 显示设备清单可以经环境变量、挂载或 CDI 方式交给容器运行时，且设备身份有 UUID/index 选择。这说明部署分配、容器可见性与执行器里的逻辑序号需要形成映射。重启后逻辑 device 0 仍叫 0，不证明仍是之前那张物理卡或同一 MIG 资源。

## CPU、内存与拓扑限制

CPU 线程池大小需要结合实际可用 CPU 集与带宽 quota。设宿主有 64 个逻辑 CPU，但该服务的 CPU 预算为每 100 ms 最多消耗 200 ms 执行时间，长期上限相当于 2 个 CPU 的时间；把线程池设成 64 不会获得 64 核持续计算。[Linux cgroup v2](../sources/S-LINUX-RESOURCES.md) 还规定限制沿层级收紧，子组设置不能突破祖先限制。

若设定分词每请求消耗 10 ms CPU、其他主机工作消耗 6 ms CPU，150 请求/s 需要 2.4 CPU 秒/s，已超出上述 2 CPU 预算。即使 GPU 很空闲，也不能先推断模型 kernel 是瓶颈。需要测 cpu.stat 的节流、实际队列和线程时间，并考虑增大主机份额或减少主机工作。

内存预算包含匿名缓冲、文件缓存、共享内存和相关内核资源。设加载时源权重 24 GiB、转换后权重 24 GiB、staging 4 GiB 同时存活，则主项为 52 GiB；宿主拥有 512 GiB 也无法使一个上限 48 GiB 的服务自动加载成功。可采用已验证的流式转换降低峰值，或调整合法资源配额。

cpuset 的 CPU 与内存节点设置、实际 NUMA 分配、NIC/SSD/GPU 亲和共同决定数据路径。指定核集合并不自动保证页面已在期望节点；改变内存节点可能触发昂贵且不完全的页面迁移。部署验证应读取有效值和实际路径，而不是只保留期望配置。

## 健康、就绪与可接收容量

健康探测回答进程是否能够推进其必要工作；就绪探测回答该模型版本和执行组是否已满足接收契约。端口能连接、HTTP 返回成功或插件进程没有退出，都不能单独证明权重完成加载、通信组完整、设备可用和代表路径正确。

就绪还可以有资源范围：允许短文本但尚未准备长图像路径，或者 KV 已满而需要暂停准入。若把暂时过载当成进程失效并不断重启，就会清空热状态、重复加载并放大存储压力。无法恢复的设备或执行组错误则应退出相应可用集合，按[恢复](serving-recovery.md)协议重建。

共享内存容量、文件描述符、进程/线程限制、I/O 限额和网络访问范围也需要进入环境检查。只提升 GPU 数量可能同时增加 tokenizer 进程、IPC 缓冲和并发制品读取，从而先耗尽这些资源。限制的数值要绑定实际配置，不在知识正文中写成通用默认值。

## 更新与独立验证

更换驱动、设备插件或运行时可能影响整个节点，而不只是一个服务进程。固定插件文档明确建议更新插件前排空 GPU 任务，并不保证任务在滚动升级中存活。具体升级策略应由目标版本与设备支持决定，先停止新准入，再完成或迁移已有工作，确认最后访问结束后释放旧环境。

验收分三层：核对真实环境身份与有效资源；运行覆盖目标 shape、dtype、通信与状态路径的最小模型参考；重放代表负载验证服务质量和资源峰值。再测试 CPU 限额收紧、加载超出主存、设备映射变化、部分 rank 未就绪和实例排空。每次失败都应能区分环境不相容、配置不足与实现错误。

当任务只在单一受控机器运行时，直接部署可以减少组织成本；多租户、多版本与弹性环境需要更明确的资源和生命周期契约。无论部署载体是什么，实验都要在与目标一致的有效资源范围内比较，避免把环境改变当成内核优化收益。

<!-- infraagents:provenance -->

## 来源与推导

- [S-NVIDIA-PLUGIN：NVIDIA Kubernetes device plugin v0.16.1 — deployment and sharing](../sources/S-NVIDIA-PLUGIN.md)：v0.16.1 README 定义设备暴露、UUID/index、运行时注入方式、MIG 策略、time-slicing 与 MPS 的配置边界和升级排空约定。
  [完整原文](../../references/S-NVIDIA-PLUGIN/reader.html) · [全文文本](../../references/S-NVIDIA-PLUGIN/fulltext.txt)

- [S-LINUX-RESOURCES：Linux 6.12 — cgroup v2 and powercap](../sources/S-LINUX-RESOURCES.md)：Linux v6.12 的完整 cgroup v2 文档定义资源层级、CPU quota/权重、内存压力/上限、cpuset、进程和设备/RDMA 控制；完整 powercap 文档定义计量域、能量单位/范围和功率约束窗口。
  [完整原文](../../references/S-LINUX-RESOURCES/reader.html) · [全文文本](../../references/S-LINUX-RESOURCES/fulltext.txt)

- [S-TORCH-CUDA：PyTorch — CUDA semantics](../sources/S-TORCH-CUDA.md)：PyTorch CUDA 运算通常异步提交；不同 stream 的依赖需显式表达，allocator 还需要知道跨 stream 的使用。CUDA Graph 的重放路径依赖稳定的地址与兼容的执行结构。
  [完整原文](../../references/S-TORCH-CUDA/reader.html) · [全文文本](../../references/S-TORCH-CUDA/fulltext.txt)

- [S-BORG：Large-scale cluster management at Google with Borg](../sources/S-BORG.md)：§2 任务状态、资源向量、优先级/配额；§3 可行性与评分、制品局部性；§4 相关故障；§5 装箱、碎片和资源回收；§6 隔离；§8 控制服务的组织经验。
  [完整原文](../../references/S-BORG/reader.html) · [全文文本](../../references/S-BORG/fulltext.txt)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [CPU 推理：线程、缓存层级与 NUMA 分区](../hardware/cpu-serving.md)。
- [通信拓扑：从 Rank 组到瓶颈链路](../hardware/topology-parallelism.md)。
- [通信传输：注册、完成、可见与回收](../hardware/transport-and-rdma.md)。
- [权重加载：从制品身份到可执行布局](../frameworks/checkpoint-loading.md)。
- [模型生命周期：加载、就绪、排空与版本切换](../system-semantics/model-lifecycle.md)。
- [服务主机运行时：执行域、队列与缓冲所有权](../system-semantics/host-runtime.md)。
- [租户隔离：资源份额、缓存命名空间与故障边界](../system-semantics/tenant-isolation.md)。
- [推理服务恢复：提交边界、执行组与重试身份](../system-semantics/serving-recovery.md)。
- [失败分析：最早偏离、最小复现与机制归因](../tools/failure-evidence.md)。
