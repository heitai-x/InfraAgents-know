# S-ASGI：ASGI — HTTP and WebSocket Message Format

## 本地完整原文

[完整原文](../../references/S-ASGI/reader.html) · [全文文本](../../references/S-ASGI/fulltext.txt)

固定版本：Git f41c0c77cc9502dae68e2b8cb84a5171e9e687f7 (2026-02-23)。来源版本日期：2026-02-23；整理核对日期：2026-10-22。

## 来源支持的机制与定义

ASGI HTTP/WebSocket 规范定义接收、发送、响应体和断开相关事件；断开通知与发送失败可能存在时间上的竞争。

## InfraAgents 如何据此推导

本库将协议完成、客户端消费、业务取消和设备排空分开，推导背压、幂等清理和晚到结果的处理。

## 适用边界

机制与接口说明；不构成当前目标的性能或可达性证明。 原始版本、日期依据与归档范围见本地 provenance.json；版本日期不证明模型训练实际包含该资料。
