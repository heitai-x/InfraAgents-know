# Nsight 性能反馈：从摘要回到原始事件

状态：已实现 SQLite 分析入口，已在 RTX 3090 / Nsight Systems 2025.3.2 上完成微型 CUDA 采集、原生统计对照和窗口查询验证。尚未运行模型推理、请求级关联或性能优化实验。

## 本次解决的问题

环境探查已经能够保存工具的完整输出。但诊断还需要识别一份 trace 实际包含什么、热点对应哪些事件、设备事件由哪个主机调用触发，以及时间区间是否重叠。

继续使用原生 nsys 采集、导出和统计；InfraAgents 增加一个只读分析入口，集中处理反复需要的信号状态、时间窗口、进程内关联和证据回查。运行时只有 Python 标准库，没有新增 Agent 框架、MCP 服务或通用进程管理器。

## 入口与输入

`analyze-nsys` 接收已完成的原生 SQLite 导出。可选参数包括上下文 JSON、起止纳秒、每类返回数量和新的结果目录。时间使用导出文件内的原始时间轴，不能直接填入另一份报告的相对时间。

```bash
PYTHONPATH=src python3 -m infraagents analyze-nsys runs/example/capture.sqlite \
  --context runs/example/context.json --output-dir runs/example/analysis

PYTHONPATH=src python3 -m infraagents analyze-nsys runs/example/capture.sqlite \
  --start-ns 556483329 --end-ns 556484729 \
  --output-dir runs/example/window-analysis --json
```

第二条命令的时间值只是本次已保存 fixture 的具体窗口，新的采集应从自己的事件中选取。窗口按相交关系选取事件，聚合时长裁剪到窗口，返回的事件保留原始起止时间。主机 API 的关联查找不受窗口限制，避免丢掉窗口之前的提交。

上下文示例见 [profile-context.json](https://github.com/heitai-x/InfraAgents/blob/a102858e3bae595a4af88be6c3a1cd2d7a0f37f1/examples/profile-context.json)。填写模型、硬件、workload、运行点、代码版本、采集工具版本、实际命令、信号、覆盖范围、原始执行记录和采集扰动。这些是调用者提供的描述；分析器保留为 declared_not_verified，不把提供了字段视为事实已验证。未提供时明确标记 not_provided。

## 采集复用原生工具

下面是 Linux 上的微型用例示例，会实际使用少量 GPU 资源。工作目录为仓库根目录；每次创建新目录，实际命令、退出状态和完整输出应随运行记录保存。本次已保存的采集证据见验证清单。

```bash
run_dir=$(mktemp -d runs/nsys-XXXXXX)
timeout --signal=TERM --kill-after=5 45 nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --duration=15 --kill=sigterm \
  --force-overwrite=false --output="$run_dir/capture" \
  python3 examples/cuda_feedback_smoke.py
nsys export --type=sqlite --force-overwrite=false \
  --output="$run_dir/capture.sqlite" "$run_dir/capture.nsys-rep"
PYTHONPATH=src python3 -m infraagents analyze-nsys "$run_dir/capture.sqlite" \
  --output-dir "$run_dir/analysis"
```

微型用例只分配 16 KiB，执行三次整数加一并检查 4096 个结果。它验证工具链，不代表模型性能。它不依赖 VibeSys、PyTorch 或模型权重。实际任务由代码执行角色按目标系统生成采集命令和 workload，诊断角色检查范围与干扰。

## 输出与解释

每次生成 report.json、schema.json 和 queries.jsonl。报告记录源文件路径、SHA-256、大小、上下文、窗口和未验证项。SQL 文本及参数可重放；事件通过原始表和 source_rowid 定位。大型原始 trace 留在原处。

| 输出 | 能够回答的问题 | 解释边界 |
|---|---|---|
| signal 状态、全表行数和窗口内区间数 | 导出中有哪些可用记录 | 表缺失不能解释为零工作量 |
| 热点及累计时长 | 哪些名称或线程值得进一步查看 | 不是端到端时间占比；拷贝方向等细节继续读事件字段 |
| 限量原始事件 | 具体时间、stream、shape、传输字节、返回值 | 数量上限不截断全量计数或区间并集 |
| 设备事件与主机 API 候选 | 当前事件可能由哪个主机调用触发 | 使用完整进程作用域与 correlationId；多个候选保留歧义 |
| 同进程同设备区间并集 | 已观察到的 kernel/copy/memset 如何重叠、哪些区间未覆盖 | 不是 SM 利用率，也不是设备全局空闲时间 |
| 原始 schema、SQL 与哈希 | 摘要是否可以追溯、源文件是否变化 | 需要原始产物仍然可访问 |

CUDA runtime、driver、kernel、memcpy、memset、同步和 NVTX 分别呈现。只分析已完成且具有正时长的区间；NVTX 点事件、未闭合范围和异常时间戳计入排除行数，原始库仍保留这些记录。首版不自动恢复请求、批次、整图内部、分布式依赖或完整关键路径。

API 返回时刻到 GPU 开始时刻的差值保留正负号，字段名只描述这两个时间点，不直接命名为真实排队耗时。进程标识缺失的记录不混入同设备的其他进程区间。

## 失败与状态

- table_absent：导出里未观察到对应表，原因需结合采集记录。
- unsupported_schema：表存在，但当前分析所需字段缺失。
- no_completed_intervals_in_selection：当前窗口没有可分析的完成区间，不推断整个应用没有工作。
- observed：至少有一个符合条件的区间。

CLI 返回 0 只表示分析完成，包括无事件的导出。SQLite 损坏或读取失败时，返回 1，并在新目录保留原始异常和已有查询日志。参数无效或结果目录已经存在时返回 2；既有报告拒绝覆盖。拒绝带事务侧文件的数据库，以分析已结束的稳定导出。

## 实际验证与边界

验证清单：[CASE-0001 的证据清单](evidence/nsys-feedback-20260919.json)。原始产物位于远端 /mnt/InfraAgents/runs/nsys-feedback-20260919T154328Z；清单保存每个产物的相对位置、大小和 SHA-256，原始 trace 不进入 Git。

- 开启 CUDA tracing：3 次 kernel、2 次 memcpy、12 次主机 API 和 1 次同步记录；5 个设备事件均找到唯一的同进程 API 候选。
- 与 nsys 原生统计对照：kernel 累计 4352 ns，memcpy 累计 4896 ns，次数与 API 汇总一致。
- 关闭 tracing：程序正确执行，CUDA 表缺失；分析器保留为未知观测，不输出“GPU 工作为零”。
- 缩小到首个 kernel 内部：窗口得到 1400 ns，并保留原始 kernel 和对应主机 API。
- Windows 与远端 Linux 的 23 项测试通过。覆盖重叠区间、窗口裁剪、跨进程及跨 VM 编号冲突、歧义关联、负时间差、缺失 schema、原始库只读、长输出及失败保存等行为。

这些数值是采集与分析校验值，没有作为性能成绩。无插桩运行只检查正确性，没有测量 profiling overhead。真实多进程、多卡、模型推理、TTFT/TPOT 和 SLO 仍未验证。知识提炼已落入条目和提案；本次采集没有使用自动角色调度或自动知识更新。后续代码入口见 [五个逻辑职责 MVP](https://github.com/heitai-x/InfraAgents/blob/a102858e3bae595a4af88be6c3a1cd2d7a0f37f1/docs/10-mvp-runtime.md)。历史清单中的实现哈希对应当时版本，可从 Git 提交 `1d828cb` 回查，不要求后续 CLI 版本保持相同哈希。

## VibeSys 对照与资料

参考其 [Profiler 扩展规范](https://github.com/uw-syfi/vibesys/blob/bc8937bab9ef57f207ba688babf6b11f42421d14/docs/contributing/extending-profilers.md) 和 [Nsight 工具入口](https://github.com/uw-syfi/vibesys/blob/bc8937bab9ef57f207ba688babf6b11f42421d14/resources/profilers/nsys/server.py)：沿用原生产物、采集与评分分离、按需分析的思想。本次独立实现只读 SQLite 查询，不引入其运行时或复制分析实现。

NVIDIA 的 [原生统计与时间解释](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#cuda-kern-exec-sum-nvtx-name-base-mangled-cuda-kernel-launch-exec-time-summary) 说明 API 与设备时间可能重叠；本项目同时保留原始区间和查询依据，便于诊断角色继续核对。
