# S-ASGI：ASGI — HTTP and WebSocket Message Format

## 本地完整原文

[完整原文阅读](../../references/S-ASGI/reader.html) · [全文文本](../../references/S-ASGI/fulltext.txt) · [原始文件](../../references/S-ASGI/original.html)

原始文件、完整正文及正文所需本地资源均已保存。

资料形态：按已核对范围整理的本地非代码释义。原文版本：ASGI 3.0 documentation。

## 来源支持的机制与定义

ASGI HTTP/WebSocket 规范定义接收、发送、响应体和断开相关事件；断开通知与发送失败可能存在时间上的竞争。

## InfraAgents 如何据此推导

本库将协议完成、客户端消费、业务取消和设备排空分开，推导背压、幂等清理和晚到结果的处理。

## 阅读范围与出处身份

连接断开通知及与发送错误的竞争。

核对日期：2026-09-20。原文标题、版本与原始地址保留在来源登记元数据中；上述内容是来源释义与本库推导的区分，不冒充原文逐字副本。
