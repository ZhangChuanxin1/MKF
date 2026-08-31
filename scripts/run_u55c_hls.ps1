[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('kklss_ckks', 'mkf_ckks', 'kklss_bfv', 'mkf_bfv')]
    [string]$Target,

    [ValidateSet('csim', 'csynth', 'export')]
    [string]$Mode = 'csim',

    [ValidateSet('smoke', 'release300')]
    [string]$ClockProfile = 'smoke',

    [string]$RunDir,
    [switch]$AllowHeavySynthesis
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($Mode -in @('csynth', 'export') -and -not $AllowHeavySynthesis) {
    throw 'csynth/export is resource intensive. Re-run with -AllowHeavySynthesis.'
}

$vitis = Get-Command vitis_hls -ErrorAction SilentlyContinue
if (-not $vitis) {
    throw 'vitis_hls is unavailable. Activate Vitis/Vivado 2022.2.'
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$config = Get-Content -LiteralPath (Join-Path $repoRoot 'config\u55c.json') -Raw | ConvertFrom-Json
if ($config.device_part -ne 'xcu55c-fsvh2892-2L-e' -or
    $config.platform_name -ne 'xilinx_u55c_gen3x16_xdma_3_202210_1') {
    throw 'U55C configuration identity mismatch.'
}

if (-not $RunDir) {
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
    $RunDir = Join-Path $repoRoot ".local\${stamp}_u55c_${Target}_${Mode}"
}
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$clockNs = if ($ClockProfile -eq 'release300') { '3.333' } else { '10.000' }
$buildDir = Join-Path $RunDir 'hls-build'
$logPath = Join-Path $RunDir "${Target}_${Mode}_${ClockProfile}.log"
$tcl = Join-Path $repoRoot 'hardware\u55c\hls\run_hls.tcl'

$previousBuild = $env:MKF_HLS_BUILD_DIR
try {
    $env:MKF_HLS_BUILD_DIR = $buildDir
    & $vitis.Source -f $tcl -tclargs $Target $Mode $clockNs 2>&1 |
        Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "Vitis HLS failed for $Target/$Mode. See $logPath"
    }
}
finally {
    $env:MKF_HLS_BUILD_DIR = $previousBuild
}

Write-Host "U55C HLS $Mode PASS for $Target. Raw log: $logPath"
if ($ClockProfile -eq 'release300') {
    Write-Warning '3.333 ns is requested only. Do not report 300 MHz without routed U55C WNS/WHS and achieved-clock evidence.'
}
