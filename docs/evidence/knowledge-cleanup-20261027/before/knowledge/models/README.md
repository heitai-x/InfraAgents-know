# 模型

本分类条目的适用条件、来源与未验证范围在各正文中说明。

| 条目 | 版本与状态 | 核心认识 |
|---|---|---|
| [K-MODEL-001：从模型计算图推导 Prefill、Decode 与持久状态](prefill-decode-state.md) | v4 · seed | 模型接入首先是一项计算图和状态语义工作：识别哪些量随 token、位置、请求和阶段变化，再选择执行机制。 |
| [K-MODEL-002：MHA、GQA 与 MQA：头映射和带宽复用](gqa-kv-heads.md) | v3 · seed | GQA 让多个 query heads 读取同一组 K/V；逻辑共享可以减少状态，但是否减少实际访存取决于 kernel。 |
| [K-MODEL-003：MoE：路由、Dispatch/Combine 与专家负载](moe-routing.md) | v3 · seed | MoE 的条件计算节省激活 FLOPs，但全部专家的驻留和路由后的不均衡决定真实服务成本。 |
| [K-MODEL-004：SSM 与混合模型：递归状态、扫描与恢复](recurrent-hybrid-state.md) | v3 · seed | 递归状态是前缀的压缩摘要；它支持低增长成本的 decode，却改变了前缀复用、回滚和批次管理方式。 |
| [K-MODEL-005：Encoder-decoder：源表示、两套 Attention 与流式边界](encoder-decoder-streaming.md) | v2 · seed | Encoder-decoder 将源编码和目标生成分开；固定源复用、流式追加和目标自回归具有不同的依赖结构。 |
| [K-MODEL-006：MLA：压缩状态、矩阵吸收与解耦 RoPE](mla.md) | v3 · seed | MLA 通过缓存低维 latent 改变 KV 的表示；真正节省 decode 流量需要执行内核沿压缩表示计算。 |
| [K-MODEL-007：窗口与稀疏 Attention：可见性、位置和状态保留](sliding-window-attention.md) | v3 · seed | 窗口限制的是每个 query 的直接可见历史；逻辑位置与跨层信息传播仍然持续演进。 |
| [K-MODEL-008：视觉语言模型：Patch、模态桥接与联合位置](vision-language.md) | v3 · seed | VLM 的真实输入规模由处理器生成的视觉 token 决定，文本长度只描述其中一部分。 |
| [K-MODEL-009：图像扩散：迭代求解、条件复用与 CFG](image-diffusion.md) | v2 · seed | 扩散服务围绕不断变化的 latent 做多步求解；可复用对象由每步依赖决定，不能沿用自回归 KV 的全部规则。 |
| [K-MODEL-010：视频生成：时空 Token、全局耦合与分块解码](video-generation.md) | v2 · seed | 视频生成把时间加入 latent 几何，时间压缩、时空 attention 和输出解码共同决定可服务规模。 |
| [K-MODEL-011：语音生成：Codec Token、声学块与播放时钟](speech-generation.md) | v2 · seed | 语音服务需要把模型生成时钟转换为音频播放时钟，首包和稳定供给由多个阶段共同决定。 |
| [K-MODEL-012：全模态系统：多时钟、分支依赖与背压](omni-multimodal.md) | v2 · seed | 全模态推理是多个有依赖的计算分支组成的系统；文本、视频和音频不能压成同一个 token 长度。 |
| [K-MODEL-013：Embedding 与重排：编码函数、批量与评分契约](embedding-reranking.md) | v1 · seed | 编码器、pooling、normalize 与句对评分共同定义 embedding/重排函数；长度聚批、padding 与输入版本决定成本及结果。 |
| [K-MODEL-014：视觉识别与卷积：空间布局、预处理与任务输出](vision-inference.md) | v1 · seed | 视觉识别把图像变换、卷积布局、残差生命周期与分类/检测输出连接；按分辨率、工作区和几何契约验证完整请求。 |
| [K-MODEL-015：推荐推理：稀疏表、特征交互与请求批量](recommendation-inference.md) | v1 · seed | 推荐推理分别计量稀疏表 lookup、bag 聚合、特征交互与 dense 计算，并以特征/参数版本约束分片、缓存和更新。 |
| [K-MODEL-016：图模型推理：邻居依赖、采样与不规则执行](graph-inference.md) | v1 · seed | 图推理先展开邻居依赖再逐层聚合；采样、图/特征版本、去重语义和不规则搬运共同决定质量与资源上界。 |
