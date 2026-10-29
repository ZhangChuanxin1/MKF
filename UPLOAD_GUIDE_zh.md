# 后续 GitHub 发布步骤

本次按用户要求只整理本地发布包，未创建 Git 提交、未上传远端。
本指南不绑定仓库或账户，也不会自动改写旧历史。

## 1. 发布前检查

先决定原创 MKF/硬件代码的许可证，保留第三方许可证和必要归属声明。
确认 README 中所有“未完成/推算/作者提供”的边界均保留。

```powershell
py -3 scripts/publication_scan.py --verify-manifest
py -3 -m unittest discover -s tests -v
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile smoke
```

如果你有意改过源代码/文档，先审查改动，再运行
`py -3 scripts/publication_scan.py --write-manifest` 更新清单并重新验证。
不要用更新清单来掩盖未知改动。

生成新源码 ZIP（输出必须在发布目录之外，拒绝覆盖已有压缩包）：

```powershell
py -3 scripts/make_source_zip.py --output ../MKFHE_FPGA_U55C_sources.zip
```

ZIP 自动排除 `.local`、缓存、生成 fixture 等；不要把运行日志或大型数据
直接加入普通 Git。不要通过网页逐个上传文件，使用 Git 可避免网页文件数限制。

## 2. 后续推送到新的空仓库

只有你确认账户、目标仓库、许可和公开范围后，才执行下列命令。
将 `<REMOTE_URL>` 换成你自己的仓库 URL，不要直接复制占位符：

```powershell
git init -b main
git add .
git status --short
git diff --cached --check
git commit -m "Organize MKF experiments and U55C source artifact"
git remote add origin <REMOTE_URL>
git push -u origin main
```

在提交前检查暂存区没有 `.local/`、数据集、生成 fixture、板卡镜像、个人
路径和密钥。如果命令报错，先保留输出并检查；不要强制推送来绕过错误。

## 3. 已有仓库

先另行 clone 到新的目录，把本包文件复制进去，排除 `.git`、`.local`、
缓存和大型数据。查看 `git status` / `git diff` 后再提交；不要覆盖你正在
使用的旧工作区，也不要自动删除远端已有文件。

旧仓库历史、账户和提交者可能暴露身份。新分支并不能清理旧历史；双盲
投稿应另行审查账户/历史/Release。此包的字符串扫描不是匿名性证明，第三方
法律归属也不能为匿名而抹除。登录时不要将令牌或密码贴到聊天或 README。

## 4. 上传后

下载全新副本，重跑 manifest、unit tests 和 smoke。只在具备对应工具和
资源的机器上运行重实验。不得把 HLS 配置、请求时钟或投影图写成板测通过。
