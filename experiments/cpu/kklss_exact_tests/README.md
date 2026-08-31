# KKLSS CPU 分阶段实验包

本目录用于复现论文中 `KKLSS CPU-ST` 一列。测试直接运行公开
`SNUCP/MKHE-KKLSS` 实现的 `MulRelinNew`，配置固定为
`PN14QP439`（论文中的 `P1`）、参与方数 `n=16`、单线程、5 次预热和
10 次正式测量。

实验代码不会用 MKF 实现替代 KKLSS。补丁只在 KKLSS 原有执行路径中
加入计时钩子，并增加两个测试入口；密码学参数、密钥、运算顺序和
`MulRelinNew` 算法逻辑不变。

## 1. 目录结构

```text
kklss_exact_tests/
├── README.md
├── patches/
│   └── kklss_stage_instrumentation.patch
├── stageprof/
│   └── stageprof.go
├── mkckks/
│   ├── profile_exact_test.go
│   └── profile_stages_exact_test.go
├── mkbfv/
│   ├── profile_exact_test.go
│   └── profile_stages_exact_test.go
├── scripts/
│   ├── install_into_kklss.sh
│   ├── run_p1_n16.sh
│   ├── summarize_stages.py
│   └── summarize_totals.py
└── sample_results/
    ├── ENVIRONMENT.md
    ├── kklss_ckks_p1_n16_stage_raw.csv
    ├── kklss_bfv_p1_n16_stage_raw.csv
    ├── kklss_p1_n16_stage_summary.csv
    ├── kklss_p1_n16_uninstrumented_crosscheck_raw.csv
    └── kklss_p1_n16_total_summary.csv
```

`patches/` 修改 KKLSS 的原有函数；`stageprof/` 提供轻量计时器；两个
scheme 目录分别包含端到端交叉检查和分阶段测试。`sample_results/` 是
本机预跑结果，不应替代论文声明的服务器环境结果。

## 2. 获取并安装

先准备一份干净的公开 KKLSS 源码：

```bash
git clone https://github.com/SNUCP/MKHE-KKLSS.git
```

然后在本目录执行：

```bash
./scripts/install_into_kklss.sh /absolute/path/to/MKHE-KKLSS
```

如果 Go 不在 `PATH`，可显式指定：

```bash
GO_BIN=/absolute/path/to/go \
  ./scripts/install_into_kklss.sh /absolute/path/to/MKHE-KKLSS
```

安装脚本会先用 `patch --dry-run` 检查源码版本是否兼容，再执行以下操作：

1. 应用 `kklss_stage_instrumentation.patch`；
2. 安装 `stageprof`；
3. 安装 CKKS/BFV 的端到端和分阶段测试；
4. 对新增及修改的 Go 文件执行 `gofmt`。

请对干净源码运行安装脚本。脚本发现已有同名测试或计时包时会停止，
防止覆盖现有修改。

## 3. 一键运行

```bash
./scripts/run_p1_n16.sh \
  /absolute/path/to/MKHE-KKLSS \
  /absolute/path/to/output
```

`run_p1_n16.sh` 同样支持 `GO_BIN=/absolute/path/to/go`。

如果省略第二个参数，结果默认写入本目录下带时间戳的 `runs/` 子目录。
脚本依次执行：

1. KKLSS 原有 CKKS/BFV 正确性测试；
2. CKKS 端到端交叉检查；
3. BFV 端到端交叉检查；
4. CKKS 分阶段测试；
5. BFV 分阶段测试；
6. CSV 完整性检查和统计汇总。

运行需要：Go、Python 3、标准 Unix shell。脚本固定
`GOMAXPROCS=1`，但不修改 `GOGC`。每次正式测量前调用 `runtime.GC()`，
GC 不在计时区间内。官方仓库没有提交完整的 `go.sum`，因此脚本会先运行
`go mod download`；新环境首次执行时需要连接 Go module 下载源。

## 4. 测量边界

密钥生成、加密、输入构造和结果验证均在计时区间之外。计时区间是一次
完整的：

```go
evaluator.MulRelinNew(...)
```

测试先执行 5 次不记录的预热，再执行 10 次正式测量并取算术平均值。
端到端测试将结果写入 `*_total_raw.csv`；分阶段测试将结果写入
`*_stage_raw.csv`。

## 5. 阶段定义

六个阶段互斥，同一次 trial 中严格满足：

```text
Total = Tensor + Decomposition + External product
      + Modulus management + Memory movement + Other
```

各行含义如下：

- `Tensor-product generation`：密文张量积中的 NTT、MForm、逐点多项式乘法
  和对应的逆 NTT；BFV 的 `Quantize` 不计入本项。
- `Gadget decomposition`：`Decompose`、`DecomposeBFV` 以及非 hoisted
  external product 内的分解。
- `External-product accumulation`：evaluation-key 逐点乘加和结果累加。
- `ModUp/ModDown and rescaling`：QP/RNS basis conversion、ModUp、ModDown；
  CKKS 另含 rescale，BFV 另含 rescale、quantize/scale-and-round。
- `Memory and data movement`：`MulRelinNew` 输出密文分配及初始化。
- `Other operations`：用 `Total` 减去前五项得到的剩余控制开销。

计时器使用单个活动 sample，不支持并发 benchmark，因此必须保持
`GOMAXPROCS=1`，也不要给测试添加 `t.Parallel()`。

## 6. 输出文件

一次完整运行会生成：

```text
kklss_ckks_p1_n16_total_raw.csv
kklss_bfv_p1_n16_total_raw.csv
kklss_p1_n16_total_summary.csv
kklss_ckks_p1_n16_stage_raw.csv
kklss_bfv_p1_n16_stage_raw.csv
kklss_p1_n16_stage_summary.csv
environment.txt
```

`summarize_stages.py` 会检查：

- 每个 scheme 恰好有 10 个 trial；
- trial 编号为 0--9 且不重复；
- 所有时间非负；
- 每个 trial 的六个阶段之和与 Total 完全相等。

任一检查失败时脚本返回非零状态，不生成可用于论文的汇总结果。

## 7. 单独运行和汇总

安装后也可以在 KKLSS 源码根目录手动运行：

```bash
GOMAXPROCS=1 KKLSS_STAGE_OUT=/tmp/ckks_stage.csv \
  go test ./mkckks -run '^TestProfileStagesExactP1N16$' -count=1 -v

GOMAXPROCS=1 KKLSS_STAGE_OUT=/tmp/bfv_stage.csv \
  go test ./mkbfv -run '^TestProfileStagesExactP1N16$' -count=1 -v
```

汇总：

```bash
python3 scripts/summarize_stages.py \
  --ckks /tmp/ckks_stage.csv \
  --bfv /tmp/bfv_stage.csv \
  --out /tmp/stage_summary.csv
```

## 8. 论文使用注意事项

`sample_results/` 是 macOS/arm64 本地预检数据。正式稿若声明 Intel Xeon、
Ubuntu 或其他 Go 版本，必须在论文声明的同一 CPU、操作系统和软件环境下
重新运行，并同时重跑 MKF CPU-ST。不要把不同机器上的 KKLSS 和 MKF 时间
直接用于 speedup 比值。
