# S-RDMA：rdma-core v54.0 — registration, work requests, completion queues and events

## 本地完整原文

[完整原文](../../references/S-RDMA/reader.html) · [全文文本](../../references/S-RDMA/fulltext.txt)

固定版本：rdma-core v54.0 commit 77128e6b6d32a1aad7c1bec9d6eb1f061b63cf39。版本坐标日期：2024-10-06；整理日期：2026-10-22。

## 来源支持的机制与定义

固定 verbs 文档给出 MR 注册、WR 提交、CQE、事件通知及本地缓冲复用规则；包含 inline 例外、错误字段与一次性通知。

## InfraAgents 如何据此推导

将传输完成与模型状态发布分离，按对象/代次/范围跟踪在途读写，区分 CQE、通道事件与应用接管确认。

## 原文定位与阅读范围

完整阅读 libibverbs/man 的 ibv_post_send、post_recv、poll_cq、reg_mr、get_cq_event 和 req_notify_cq 固定版本文件。

## 适用边界

绑定 rdma-core v54.0 的六份完整 man-page 源码与许可文件，具体 transport/flags/设备能力仍有条件；不将本地完成解释为远端应用已处理或 GPU 任意并发可见。

seed 不表示当前 Task 上已实测有效；归档完整与编辑复核完成是两项不同事实。来源时间不证明模型训练实际使用该资料。
