[CmdletBinding()]
param(
    [ValidateSet('smoke', 'software', 'u55c-csim', 'u55c-csynth')]
    [string]$Profile = 'smoke',

    [ValidateSet('smoke', 'release300')]
    [string]$ClockProfile = 'smoke',

    [string]$PythonExe = 'py',
    [switch]$AllowHeavySynthesis
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$runDir = Join-Path $repoRoot ".local\${stamp}_${Profile}"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

function Invoke-GoTest([string]$WorkingDirectory, [string]$Package, [string]$Pattern, [string]$Name) {
    $log = Join-Path $runDir "${Name}.log"
    Push-Location $WorkingDirectory
    try {
        & go test $Package -p 1 -parallel 1 -run $Pattern -count=1 2>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) {
            throw "Go test failed: $Name. See $log"
        }
    }
    finally {
        Pop-Location
    }
}

$lockPath = Join-Path $repoRoot '.local/experiment.lock'
try {
    $suiteLock = [IO.File]::Open($lockPath,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
} catch { throw 'Another artifact experiment holds the local lock. Do not interrupt it.' }
try {
$requireU55C = $Profile -in @('u55c-csim', 'u55c-csynth')
& (Join-Path $PSScriptRoot 'preflight.ps1') -RunDir $runDir -PythonExe $PythonExe -RequireU55C:$requireU55C

$previousMax = $env:GOMAXPROCS
try {
    $env:GOMAXPROCS = '1'
    if ($Profile -in @('smoke', 'software')) {
        $mkfModule = Join-Path $repoRoot 'software\mkf'
        $kklssModule = Join-Path $repoRoot 'software\kklss'
        Invoke-GoTest $mkfModule './...' '^$' 'go_compile_mkf'
        Invoke-GoTest $kklssModule './...' '^$' 'go_compile_kklss'
        Invoke-GoTest $mkfModule './mkf_cpu' '^TestP2GN16LattigoLevelSemanticsNoCrypto$' 'go_level_contract'
        Invoke-GoTest $mkfModule './mkf_cpu' '^TestP2GN16MatchedCPUContractNoFullRun$' 'go_matched_contract'
    }

    if ($Profile -eq 'software') {
        $mkfModule = Join-Path $repoRoot 'software\mkf'
        $kklssModule = Join-Path $repoRoot 'software\kklss'
        Invoke-GoTest $mkfModule './mkf_cpu' '^TestCKKSCorrectnessP1$' 'go_mkf_ckks_p1_correctness'
        Invoke-GoTest $mkfModule './mkf_cpu' '^TestBFVCorrectnessP1$' 'go_mkf_bfv_p1_correctness'
        Invoke-GoTest $kklssModule './mkckks' '^TestCKKS$' 'go_kklss_ckks_correctness'
        Invoke-GoTest $kklssModule './mkbfv' '^TestMKBFV$' 'go_kklss_bfv_correctness'

        $pythonLeaf = [IO.Path]::GetFileName($PythonExe)
        $useLauncher = $pythonLeaf -in @('py', 'py.exe')
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        if ($useLauncher) {
            & $PythonExe -3 -c 'import numpy' 2>$null
        }
        else {
            & $PythonExe -c 'import numpy' 2>$null
        }
        $numpyExit = $LASTEXITCODE
        $ErrorActionPreference = $savedPreference
        if ($numpyExit -eq 0) {
            $floodDir = Join-Path $runDir 'flooding-output'
            $floodScript = Join-Path $repoRoot 'experiments\flooding\simulate_uniform_flooding.py'
            $floodInput = Join-Path $repoRoot 'experiments\flooding\preflood_calibration_canonical_p1p2_raw.csv'
            if ($useLauncher) {
                & $PythonExe -3 $floodScript --input $floodInput --out-dir $floodDir --mc-trials 4 2>&1 |
                    Tee-Object -FilePath (Join-Path $runDir 'flooding.log')
            }
            else {
                & $PythonExe $floodScript --input $floodInput --out-dir $floodDir --mc-trials 4 2>&1 |
                    Tee-Object -FilePath (Join-Path $runDir 'flooding.log')
            }
            if ($LASTEXITCODE -ne 0) {
                throw 'Flooding replay failed.'
            }
        }
        else {
            Write-Warning 'Flooding replay skipped because NumPy is not installed.'
        }
    }

    if ($Profile -eq 'u55c-csim') {
        foreach ($target in @('kklss_ckks', 'mkf_ckks', 'kklss_bfv', 'mkf_bfv')) {
            & (Join-Path $PSScriptRoot 'run_u55c_hls.ps1') `
                -Target $target -Mode csim -ClockProfile $ClockProfile -RunDir $runDir
        }
    }

    if ($Profile -eq 'u55c-csynth') {
        if (-not $AllowHeavySynthesis) {
            throw 'HLS synthesis requires -AllowHeavySynthesis.'
        }
        foreach ($target in @('kklss_ckks', 'mkf_ckks', 'kklss_bfv', 'mkf_bfv')) {
            & (Join-Path $PSScriptRoot 'run_u55c_hls.ps1') `
                -Target $target -Mode csynth -ClockProfile $ClockProfile `
                -RunDir $runDir -AllowHeavySynthesis
        }
    }
}
finally {
    $env:GOMAXPROCS = $previousMax
}

$summary = [ordered]@{
    schema = 'mkf-u55c-local-run-v1'
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
    profile = $Profile
    clock_profile = $ClockProfile
    status = 'PASS'
    note = 'PASS applies only to the selected profile and is not a U55C board-performance result.'
}
$summary | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $runDir 'RUN_SUMMARY.json') -Encoding UTF8
Write-Host "Profile $Profile PASS. Local evidence: $runDir"
}
finally {
    $suiteLock.Dispose()
}
