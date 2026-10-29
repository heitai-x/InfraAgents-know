# S-LINUX-IO：Linux 6.12 — Block multi-queue and iomap I/O documentation

## 本地完整原文

[完整原文](../../references/S-LINUX-IO/reader.html) · [全文文本](../../references/S-LINUX-IO/fulltext.txt)

固定版本：Linux v6.12 commit adc218676eef25575469234709c2d87185ca223a。版本坐标日期：2024-11-17；整理日期：2026-10-22。

## 来源支持的机制与定义

Linux v6.12 的 blk-mq 文档区分软件/硬件队列与乱序完成；iomap design/operations 区分文件映射、pagecache、直接 I/O、部分完成和写回。

## InfraAgents 如何据此推导

据文件范围、队列、完成和持久性边界推导加载缓冲生命周期；将 SSD→DRAM→GPU 的阶段分别记账，内核版本作为具体示例。

## 原文定位与阅读范围

完整阅读 Documentation/block/blk-mq.rst、Documentation/filesystems/iomap/design.rst、operations.rst；重点是 Tag-based completion、Buffered I/O、Direct I/O、Return Values、Direct Reads/Writes。

## 适用边界

仅绑定 Linux v6.12 的三个完整文档源码；不是全部文件系统或所有存储设备的统一性能保证，也未递归导入 kernel-doc 引用的整个源代码树。内核完成、应用发布与设备持久性需按所用路径区分。

seed 不表示当前 Task 上已实测有效；归档完整与编辑复核完成是两项不同事实。来源时间不证明模型训练实际使用该资料。
