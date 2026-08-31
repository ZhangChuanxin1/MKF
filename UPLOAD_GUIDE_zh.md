# GitHub 上传指南（U55C 专用发布）

## 先确认两个边界

1. 当前发布目录是 U55C 目标的 HLS/软件复现包，不是已经完成板测的
   U55C 加速器。README 中的证据状态不要改成板测通过。
2. 双盲投稿应使用与作者身份无关的中性账户或组织。仓库名称匿名并不足够；
   账户名、提交作者、推送者和旧 Git 历史都可能暴露身份。

本机在未认证状态下无法读取你给出的远端（返回 404 或要求凭据），因此
本文档不猜测默认分支，也不会自动推送。

## 1. 发布前检查

在本目录运行：

```powershell
py -3 scripts/publication_scan.py --write-manifest
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile smoke
```

必须得到 publication scan `PASS` 和 smoke `PASS`。如果你已经生成了大型
fixture，请保持它们在被 `.gitignore` 排除的 `fixtures/generated/` 中，且
不要用普通 Git 提交。

还需由仓库所有者决定并添加原创 MKF/硬件代码的许可证。第三方 KKLSS 的
CC BY-NC 2.0 许可证必须保留。

## 2. 推荐方案：新建全新的中性空仓库

这是双盲和彻底隔离旧历史时最安全的方案。先在 GitHub 中创建空仓库，不要
自动生成 README、LICENSE 或 `.gitignore`，然后在本目录执行：

```powershell
git init -b main
git config user.name "Anonymous Authors"
git config user.email "anonymous@users.noreply.github.com"
git add .gitattributes .gitignore README.md THIRD_PARTY_NOTICES.md `
  UPLOAD_GUIDE_zh.md requirements-software.txt PUBLICATION_MANIFEST.sha256 `
  config docs experiments fixtures hardware scripts software
git status --short
git diff --cached --check
git commit -m "Release U55C reproducibility artifact"
git remote add origin <REMOTE_URL>
git ls-remote --symref origin HEAD
git push -u origin HEAD:main
```

逐项检查 `git status --short`，确认没有 `.local/`、生成 fixture、综合输出、
旧报告或个人路径后再提交。不要直接把整个论文工作区设为 remote。

## 3. 如果必须使用已有仓库

先认证并确认远端，不要猜默认分支：

```powershell
gh auth status
gh repo view <OWNER/REPOSITORY>
git ls-remote --symref <REMOTE_URL> HEAD
```

建议把本发布目录作为一条新分支推送：

```powershell
git remote add origin <REMOTE_URL>
git push -u origin HEAD:u55c-artifact
```

这不会清除旧提交中的敏感对象。如果旧仓库历史、tag 或 Release 资产中已经
包含不应公开的内容，新分支并不能完成匿名化。此时优先新建空仓库；历史重写
和强制推送具有破坏性，必须先备份并由你明确决定，本指南不自动执行。

## 4. 推送后核对

- 在网页上确认 README 的硬件状态仍是 HLS-only。
- 确认仓库中不存在 `.local/`、`fixtures/generated/` 和工具输出。
- 下载一个全新副本，重新运行 publication scan 与 smoke。
- 只有在装有 Vitis/Vivado 2022.2 和精确 U55C 平台的主机上，才运行
  `u55c-csim` 或 `u55c-csynth`。
- 不要把 C-sim、C-synth 或请求时钟写成板上实测结果。
