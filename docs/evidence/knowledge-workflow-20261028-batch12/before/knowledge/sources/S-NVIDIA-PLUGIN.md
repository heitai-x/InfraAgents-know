# S-NVIDIA-PLUGIN：NVIDIA Kubernetes device plugin v0.16.1 — deployment and sharing

## 本地完整原文

[完整原文](../../references/S-NVIDIA-PLUGIN/reader.html) · [全文文本](../../references/S-NVIDIA-PLUGIN/fulltext.txt)

固定版本：v0.16.1 commit cb6e45ec357705e6458f753eb04a4bde31ec18cf。版本坐标日期：2024-07-23；整理日期：2026-10-23。

## 来源支持的机制与定义

v0.16.1 README 定义设备暴露、UUID/index、运行时注入方式、MIG 策略、time-slicing 与 MPS 的配置边界和升级排空约定。

## InfraAgents 如何据此推导

区分访问份额、实际显存、计算限额、物理身份与故障域，将设备暴露接到服务就绪、租户准入和环境验证。

## 原文定位与阅读范围

已完整阅读固定 README.md 及随附 LICENSE，重点 Configuration Option Details、Shared Access to GPUs、MPS、MIG 与 Upgrading Kubernetes with the Device Plugin。

## 适用边界

绑定官方提交 cb6e45ec357705e6458f753eb04a4bde31ec18cf，2024-07-23。该版本的 MPS 共享只支持完整 GPU、与 MIG 启用不相容；time-slicing 无额外显存与故障域隔离。README 只能支持所述资源/部署契约，不能推断所有 GPU 硬件细节、确定吞吐、全面安全隔离或后续版本能力；外链 latest 文档未作为来源。

seed 不表示当前 Task 上已实测有效；归档完整与编辑复核完成是两项不同事实。来源时间不证明模型训练实际使用该资料。
