# InfraAgents-know v2.2-r1：Windows PowerShell 手动推送

目标：把这次交付提交推送到原仓库的 `gpt-6-astra-2026-04-30` 分支。以下命令由你在 Windows PowerShell 中执行，GitHub 登录和推送权限使用你本机的 Git 配置。

## 1. 包内内容与提交身份

| 内容 | 用途 |
| --- | --- |
| `InfraAgents-know/` | 完整内容快照，包含 1,821 个提交内文件及全部 `references/`，可直接阅读或另存。它不是带有完整 Git 历史的克隆仓库。 |
| `git/InfraAgents-know-v2.2-r1-commit.bundle` | 保留原提交身份的增量 Git bundle，供下面的 fetch/merge 流程使用。 |
| 随包 `.patch` 文件 | 补丁形式的备选交付；本说明的主流程使用 bundle，无须再应用 patch 或复制快照覆盖克隆目录。 |

| 身份 | 精确值 |
| --- | --- |
| 仓库 | `https://github.com/heitai-x/InfraAgents-know.git` |
| 目标远端分支 | `gpt-6-astra-2026-04-30` |
| bundle 所需 baseline | `f640148464929f803a53b55e27ee4cbb83dc8657` |
| 本次交付 commit | `ba75abc1259966f0a6fd08c790ae819093c10fa1` |
| 本次交付 tree | `715bac132b48daf253d1d36f7b15f0b50f19efe6` |

**bundle 是依赖 baseline 的增量包，不是全部 Git 历史。** 接收它的本地仓库必须已拥有上述 `f640148…` baseline。下面从原仓库正常 clone，未使用浅克隆参数；如果远端历史仍包含该 baseline，就具备前提。`git bundle verify` 会核对这个前提。

正文和来源使用原始字节与 SHA 绑定，因此 clone 命令带有 `--config core.autocrlf=false`：它在首次 checkout 前关闭这个新仓库的自动换行转换，**不会修改你的全局 Git 配置**。

## 2. 主流程：保留这次交付的原 commit

先把整个 ZIP 解压。在 PowerShell 中进入**解压包根目录**，这个目录应同时包含 `InfraAgents-know/` 和 `git/`。下面会在这里新建 `InfraAgents-know-push/` 作为真正的 Git 仓库；不要在内容快照目录里直接执行 push。

每段命令成功后再执行下一段；保留 `$LASTEXITCODE` 检查。目录 `InfraAgents-know-push` 若已存在，请先换一个空的解压工作目录，避免覆盖已有工作。

### 2.1 克隆原指定分支

```powershell
git clone --config core.autocrlf=false --branch gpt-6-astra-2026-04-30 https://github.com/heitai-x/InfraAgents-know.git InfraAgents-know-push
if ($LASTEXITCODE -ne 0) { throw "克隆失败；停止后续步骤。" }

Set-Location -LiteralPath .\InfraAgents-know-push
git status --short
```

新克隆仓库的 `git status --short` 应没有输出。后续命令都在这个 `InfraAgents-know-push/` 目录执行，因此 bundle 的相对路径是 `../git/...`。

### 2.2 检查 bundle、读取交付提交并快进合并

```powershell
git bundle verify ../git/InfraAgents-know-v2.2-r1-commit.bundle
if ($LASTEXITCODE -ne 0) { throw "bundle 验证失败；先确认完整解压及 baseline 历史，不继续合并。" }

git fetch ../git/InfraAgents-know-v2.2-r1-commit.bundle HEAD
if ($LASTEXITCODE -ne 0) { throw "读取 bundle 失败；停止后续步骤。" }

$expectedCommit = "ba75abc1259966f0a6fd08c790ae819093c10fa1"
$bundleCommit = (git rev-parse FETCH_HEAD).Trim()
if ($bundleCommit -ne $expectedCommit) { throw "bundle HEAD 与交付 commit 不一致；停止。" }

git merge --ff-only FETCH_HEAD
if ($LASTEXITCODE -ne 0) { throw "无法快进合并；先停止主流程，按第 3 节保留远端更新。" }
```

`--ff-only` 只允许不产生分叉的快进，不会为了覆盖远端工作而改写历史。如果提示缺少 prerequisite commit，说明本地尚不具备 bundle 需要的 baseline；应先确认原仓库历史，不能通过忽略这个错误继续。

### 2.3 核对本地 commit 与 tree

```powershell
$expectedCommit = "ba75abc1259966f0a6fd08c790ae819093c10fa1"
$expectedTree = "715bac132b48daf253d1d36f7b15f0b50f19efe6"
$localCommit = (git rev-parse HEAD).Trim()
$localTree = (git rev-parse "HEAD^{tree}").Trim()

if ($localCommit -ne $expectedCommit) { throw "当前 HEAD 不是本次原交付提交；先检查远端是否已有后续更新，见第 3 节。" }
if ($localTree -ne $expectedTree) { throw "提交树与本次交付不一致；停止推送。" }

"本地 commit 与 tree 均与交付一致。"
```

### 2.4 推送并核对远端 SHA

```powershell
git push origin HEAD:refs/heads/gpt-6-astra-2026-04-30
if ($LASTEXITCODE -ne 0) { throw "推送失败；若提示 non-fast-forward，按第 3 节处理；其他错误按 Git 的认证或权限提示处理。" }

$localCommit = (git rev-parse HEAD).Trim()
$remoteLine = git ls-remote --exit-code origin refs/heads/gpt-6-astra-2026-04-30
if ($LASTEXITCODE -ne 0) { throw "无法读取远端分支，暂不能确认推送结果。" }

$remoteCommit = ($remoteLine -split "\s+")[0]
if ($remoteCommit -ne $localCommit) { throw "核对时远端 SHA 与本地不同；先检查远端是否又有新提交。" }

"已确认远端分支 SHA：$remoteCommit"
```

正常快进路径最后应显示 `ba75abc1259966f0a6fd08c790ae819093c10fa1`。

## 3. 远端已经前进：保留远端更新再应用本次改动

如果第 2.2 节的 `--ff-only` 失败，或推送时远端又前进导致 `non-fast-forward`，不要使用 force。先保留并处理好自己的未提交修改；以下步骤要求工作区干净，也没有进行中的 merge/cherry-pick。

仍在 `InfraAgents-know-push/` 中执行。先检查工作区，再更新远端引用：

```powershell
git status
git fetch origin
if ($LASTEXITCODE -ne 0) { throw "更新远端引用失败；停止。" }

git fetch ../git/InfraAgents-know-v2.2-r1-commit.bundle HEAD
if ($LASTEXITCODE -ne 0) { throw "读取交付 bundle 失败；停止。" }

$integrationBranch = "manual-push-v22r1-" + (Get-Date -Format "yyyyMMdd-HHmmss")
git switch -c $integrationBranch origin/gpt-6-astra-2026-04-30
if ($LASTEXITCODE -ne 0) { throw "无法从最新远端创建本地整合分支；停止。" }

git cherry-pick ba75abc1259966f0a6fd08c790ae819093c10fa1
```

新分支以**刚获取的远端最新提交**为起点，cherry-pick 只把本次提交的改动应用上去，保留远端已经发生的更新。

- **成功且没有冲突：** 检查合并后的内容，再执行第 2.4 节的普通 push 和远端 SHA 核对。此时当前本地分支可以是 `manual-push-v22r1-...`，命令中的显式目标仍然推送到 `gpt-6-astra-2026-04-30`。
- **发生冲突：** 先用 `git status` 查看冲突文件，逐个编辑以保留双方需要的改动；然后对处理完的文件执行 `git add`，再执行 `git cherry-pick --continue`。有新的冲突就继续处理，完成前不要 push。
- **决定停止这次整合：** 执行 `git cherry-pick --abort`，回到新建整合分支开始 cherry-pick 前的状态。
- **提示改动已经存在、cherry-pick 为空：** 先确认远端确实已经包含对应改动，再执行 `git cherry-pick --skip`，避免重复提交。

例如，处理冲突后可按实际文件路径执行：

```powershell
git status
# 将下一行的路径替换为你已经解决冲突的具体文件；可以重复多次。
git add -- path/to/resolved-file
git cherry-pick --continue
```

整合后的 commit SHA 通常不同于 `ba75abc…`；保留远端新内容后，tree 也可能不同于 `715bac…`。这条路径应核对**最终实际提交的内容和相应知识身份记录**，再确认远端 SHA 等于你刚推送的本地 HEAD；不要再用第 2.3 节的原提交断言来判定整合版本。若修改了知识正文，还需要同步对应 catalog/revision/history 等记录，不能仅删除冲突标记就把旧验证结果当作新内容的验证。

如果整合期间远端再次前进，普通 push 会再次拒绝；重新获取最新远端并保留其更新后再整合即可，无须 force。

## 4. 验证范围

原交付 commit 与 tree 已按上表标识。本说明没有执行远端 push。

已在隔离的本地临时克隆中，从原 baseline 验证 `git bundle verify`、`git fetch … HEAD` 和 `git merge --ff-only FETCH_HEAD`。结果恢复到上表的 commit 和 tree，工作区干净。演练使用本地源仓库并保留其浅历史边界；没有联网，也没有修改正式仓库。较早一次直接提取 baseline 的尝试因浅历史未取得 `FETCH_HEAD` 而停止，随后用临时克隆完成了上述验证。

这证明随包 bundle 能恢复本次提交；没有声称在你的 Windows 环境执行过 PowerShell 或验证过你的 GitHub 写入凭据。实际推送后仍按第 2.4 节核对远端 SHA。
