# K-SYS-011：模型生命周期：加载、就绪、排空与版本切换

模型服务的版本是一组共同定义输入到输出函数的制品和执行约束。权重、配置、tokenizer/processor、adapter、量化参数以及编译布局必须在同一次请求中保持相容。文件存在和进程存活，都不足以证明这个版本可以接收任务。

## 从制品到可接收请求的状态

将一个服务版本的生命期分为已登记、验证制品、预留资源、加载转换、编译与预热、就绪、接收请求、排空、可卸载。状态名称可以合并，完成条件不能省略。加载器检查分片完整和张量关系；执行器建立指定设备与布局；预热覆盖将要公开的形状和路径；就绪记录这个版本目前可接受的模型、请求形状和资源范围。

预热产生的临时 KV、随机状态和测试请求必须释放或隔离，不能成为正式请求的历史。编译缓存命中还要核对模型表示、算子语义、目标设备与编译条件，不能只比较模型名称。某个短输入完成 forward，只证明该输入路径可执行，不证明长上下文、另一 rank 的 adapter 或多卡通信都已准备好。

| 转换 | 公开之前应当成立的条件 | 失败后的处理 |
|---|---|---|
| 制品登记 → 加载 | 全部必要分片与配置属于同一版本，目标资源已预留 | 保留具体缺片、格式或容量错误，释放未发布的预留 |
| 加载 → 就绪 | 必要张量、表示转换、执行组和代表路径验证完成 | 不把部分成功的 rank 或 tensor 暴露给路由 |
| 就绪 → 接收 | 路由可识别版本，当前资源仍满足准入 | 暂停准入并报告具体限制，而非重复启动掩盖错误 |
| 接收 → 排空 | 新请求不再取得该版本；已接收请求仍有责任方 | 明确继续、迁移或失败的协议，保留已交付位置 |
| 排空 → 卸载 | 请求、graph、kernel、传输与响应的相关引用均结束 | 未证明完成的资源继续隔离，不能立即重用地址 |

## 三种切换策略

停止旧服务后加载新版本，峰值容量最低，但必然包含服务空窗。新旧并存后切流，可在新版本验证期间继续服务，代价是双份权重、执行缓存与尚未排空的旧 KV。滚动更换多个副本，把峰值容量和暂时减少的有效容量分摊到多个阶段，但要求剩余副本在发布期间仍能满足流量与故障余量。

请求在准入时绑定不可变的版本身份。路由别名从 v1 切到 v2，只改变之后的绑定；已接收请求不能在一次生成中途无条件换权重。会话下一轮是否沿用 v1 是协议选择；若改用 v2，需要重新解释输入并重新计算不相容的状态。版本回退也不能直接复用 v2 产生的 KV，或假定撤销路由配置会撤销已交付的回答。

adapter 更新遵循同样的规则。以新对象发布 v2，再让新请求引用它；v1 的最后一轮计算与传输结束后才回收。对旧地址原地覆写会让图重放或异步 kernel 读到混合版本，即使请求层的版本字段已经更新也无法补救。

## 启动与容量账

启动时间由排队取得资源、取得制品、读取/转换/复制、编译、预热和路由生效构成。独立阶段能流水时按实际依赖求关键路径；共享 SSD、NIC、CPU 或 HBM 的阶段不能各取独占带宽后声称完全重叠。[权重加载](../frameworks/checkpoint-loading.md) 给出文件到执行布局的峰值副本账，[弹性服务](../workloads/elastic-serving.md) 继续分析启动时延对请求等待的影响。

设定设备可用 24 GiB，旧权重 12 GiB、旧请求 KV 5 GiB、共同运行与临时区 3 GiB。新权重另需 12 GiB，即使忽略新图和转换峰值，并存需求也已达 32 GiB。单卡原地进行无中断更新在这组条件下不可行。把旧 KV 排空只降到 27 GiB，仍需其他设备、降低相容表示的容量，或选择有空窗的更新。容量问题不能通过提前报告就绪解决。

## 验证与选择改变的条件

独立参考绑定制品版本与规范化输入，在新版本发布前比较指定路径的输出；发布测试还要覆盖旧请求长时间生成、新请求已进入 v2、某个 rank 预热失败、制品读取中断、回退以及迟到输出。检查每个结果只归属一个版本，未就绪实例不接收任务，旧引用结束之前权重地址不重用。

持续记录目标副本数、已分配数、加载中数、真正就绪数和可接收容量。发布失败首先说明失败的状态转换，不把重试次数当成恢复证据。低流量、可接受维护空窗时，串行替换可以最简单；严格连续服务或长请求多时，并存或滚动更合适，但必须支付实际容量和排空时间。

<!-- infraagents:provenance -->

## 来源与推导

- [S-SERVERLESSLLM：ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models](../sources/S-SERVERLESSLLM.md)：§4 的加载格式、块池和多级流水；§5 的 token 重算迁移；§6 的局部性与启动时间模型。
  [完整原文](../../references/S-SERVERLESSLLM/reader.html) · [全文文本](../../references/S-SERVERLESSLLM/fulltext.txt)

- [S-BORG：Large-scale cluster management at Google with Borg](../sources/S-BORG.md)：§2 任务状态、资源向量、优先级/配额；§3 可行性与评分、制品局部性；§4 相关故障；§5 装箱、碎片和资源回收；§6 隔离；§8 控制服务的组织经验。
  [完整原文](../../references/S-BORG/reader.html) · [全文文本](../../references/S-BORG/fulltext.txt)

- [S-NVIDIA-PLUGIN：NVIDIA Kubernetes device plugin v0.16.1 — deployment and sharing](../sources/S-NVIDIA-PLUGIN.md)：v0.16.1 README 定义设备暴露、UUID/index、运行时注入方式、MIG 策略、time-slicing 与 MPS 的配置边界和升级排空约定。
  [完整原文](../../references/S-NVIDIA-PLUGIN/reader.html) · [全文文本](../../references/S-NVIDIA-PLUGIN/fulltext.txt)

- [S-TORCH-INFERENCE：PyTorch — inference_mode](../sources/S-TORCH-INFERENCE.md)：inference_mode 关闭推理不需要的 autograd 相关开销，并具有张量使用边界；它不会自动将模块切换到 eval 模式。
  [算法与机制说明](../sources/S-TORCH-INFERENCE.md)

正文中的资源账、比较与设定算例是基于固定来源的推导，未作为 InfraAgents 实测结果。来源版本上限为 2026-04-30；编辑日期不改变来源时间。

## 延伸阅读

- [权重加载：从制品身份到可执行布局](../frameworks/checkpoint-loading.md)。
- [取消与完成：业务终止、设备排空和资源回收](../system-semantics/cancellation-resource-completion.md)。
- [服务协议：接收、流式背压与终止语义](../system-semantics/service-protocol.md)。
- [推理服务恢复：提交边界、执行组与重试身份](../system-semantics/serving-recovery.md)。
- [弹性服务：启动延迟、可用容量与排空](../workloads/elastic-serving.md)。
- [推理存储服务：制品、缓存、索引与提交记录](../system-semantics/storage-services.md)。
- [部署环境：版本契约、可见设备与真实资源边界](../system-semantics/deployment-environment.md)。
