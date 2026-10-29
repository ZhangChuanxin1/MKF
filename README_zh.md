# MKF 论文实验整理包（U55C）

整理日期：2026-10-28。本次只整理本地代码，**未上传 GitHub**。

首版工程目录已按 CPU、FPGA、MNIST 和 ResNet-20 建立，见
[工程结构](docs/PROJECT_STRUCTURE.md)。缺失实现的目录只有英文说明，
明确标为 `PENDING_IMPLEMENTATION`，不是可执行的占位实现。

## 内容与边界

整理了正确性、CPU profiling、噪声/深度、gadget/阶段计时、flooding、
安全估计、MNIST、ResNet-20、rotation、U55C HLS/RTL 和最新绘图代码。
见 [15 类实验总表](docs/EXPERIMENT_CATALOG.md)。

“全部整理”指实验类别均有代码索引和缺口说明，不表示所有实验已完成，
也不表示论文中的每个数字都能通过本包重新实测。

## 第一次运行

在本目录打开 PowerShell 7，安装 Go 和 Python 3.10+ 后执行：

```powershell
py -3 scripts/publication_scan.py --verify-manifest
py -3 -m unittest discover -s tests -v
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile smoke
```

`smoke` 只编译和检查轻量合约，不启动完整 FHE 网络、综合或板卡任务。
软件正确性检查：

```powershell
pwsh -NoProfile -File scripts/run_suite.ps1 -Profile software
```

flooding 需要 NumPy，绘图需要 NumPy/Matplotlib，明文 ResNet 需要 PyTorch。
按需安装 `requirements-software.txt`、`requirements-figures.txt` 或
`requirements-resnet.txt`。虚拟环境可用 `-PythonExe .venv/Scripts/python.exe`。

## 从哪里看

- [逐项运行命令](docs/RUNBOOK.md)：CPU、noise、flooding、安全估计等。
- [数据准备](datasets/README.md)：MNIST 转换、CIFAR/模型/权重说明。
- [构造身份](docs/CONSTRUCTION_IDENTITIES.md)：历史 profiling 与完整论文构造不能混用。
- [U55C 说明](hardware/u55c/README.md)：HLS/RTL 与板级门禁。
- [绘图数据来源](figures/README.md)：明确推算/作者提供数据的边界。
- [GitHub 上传指南](UPLOAD_GUIDE_zh.md)：后续由你决定何时发布。
- [本次验证记录](docs/VALIDATION_20261028.md)：实际通过与未执行的检查。

新输出保存在被忽略的 `.local/`，不覆盖旧证据。重任务必须显式加
`-AllowHeavy` / `-AllowHeavySynthesis`，串行运行。

## 必须保留的限制

1. 硬件目标固定 U55C，但目前是源代码/HLS 包，不是完整可直接上板的加速器。
   XRT host、HBM 连接、路由镜像和合格板测尚未包含。
2. 3.333 ns 是请求时钟，不是已达到 300 MHz；OOC 时序也不是完整加速器时序。
3. 历史 `cmd/mkfbench` 不能自动代表 `true_mkf_*` 的完整论文构造。
4. MKF 图 a、BFV n=16 保留 **1.290 ms**。这是作者指定绘图值，不是本次板测。
5. 噪声历史 CSV 是原样副本，未在本次重建完整运行清单和构造匹配关系。
6. ResNet-20 matched CPU 与生产 U55C 性能仍缺失，不能填入结果表。
7. 第三方许可证必须保留；原创代码的许可证仍需仓库所有者决定。

检查结论以本次真实日志为准；未执行的 Vitis/Sage/完整网络/板卡实验不宣称通过。
