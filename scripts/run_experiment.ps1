[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('cpu-mkf','cpu-kklss-stages','noise','mnist','figures')]
    [string]$Experiment,
    [ValidateSet('ckks','bfv')][string]$Scheme = 'ckks',
    [ValidateSet('mkf','kklss')][string]$Implementation = 'mkf',
    [ValidateSet('P1','P2')][string]$Parameter = 'P1',
    [ValidateRange(2,64)][int]$Parties = 2,
    [ValidateRange(1,1000)][int]$Trials = 10,
    [ValidateRange(0,1000)][int]$Warmup = 5,
    [ValidateRange(1,64)][int]$MaxDepth = 1,
    [string]$PythonExe = 'py',
    [switch]$AllowHeavy
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($Experiment -ne 'figures' -and -not $AllowHeavy) {
    throw 'Crypto/noise/CNN jobs require -AllowHeavy. Run resource-heavy jobs serially.'
}
if ($Parties -notin @(2,4,8,16,32,64)) { throw 'Select a supported power-of-two party count.' }
if ($Experiment -eq 'cpu-kklss-stages' -and
    ($Parameter -ne 'P1' -or $Parties -ne 16 -or $Trials -ne 10 -or $Warmup -ne 5)) {
    throw 'The included stage harness is fixed to P1/n=16, 5 warmups, 10 trials.'
}
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$localRoot = Join-Path $repoRoot '.local'
New-Item -ItemType Directory -Force -Path $localRoot | Out-Null
$lockPath = Join-Path $localRoot 'experiment.lock'
try {
    $runLock = [IO.File]::Open($lockPath,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
} catch { throw 'Another artifact experiment holds the local lock. Do not interrupt it.' }
$runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '_' + [Guid]::NewGuid().ToString('N').Substring(0,8)
$runDir = Join-Path $localRoot "${runId}_${Experiment}"
New-Item -ItemType Directory -Path $runDir | Out-Null
$status = 'FAIL'
$failure = $null
$savedEnv = @{}
$envNames = @('GOMAXPROCS','KKLSS_STAGE_OUT','NOISE_OUT','NOISE_PARAMS','NOISE_PARTIES','NOISE_SCHEMES','NOISE_TRIALS','NOISE_MAX_DEPTH','NOISE_TARGET_PRECISION','NOISE_CKKS_CHAIN_MODE')
foreach ($name in $envNames) { $savedEnv[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
function Run-Native([string]$Executable,[string[]]$Arguments,[string]$Directory,[string]$LogName) {
    Write-Host "$Executable $($Arguments -join ' ')"
    Push-Location -LiteralPath $Directory
    try {
        & $Executable @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $runDir $LogName)
        if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $LogName" }
    } finally { Pop-Location }
}
try {
    $env:GOMAXPROCS = '1'
    $manifest = Join-Path $repoRoot 'PUBLICATION_MANIFEST.sha256'
    $cpu = if ($IsWindows) { @(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors) } else { @() }
    $environment = [ordered]@{
        schema='mkf-local-experiment-environment-v1'
        generated_utc=(Get-Date).ToUniversalTime().ToString('o')
        experiment=$Experiment; scheme=$Scheme; implementation=$Implementation
        parameter=$Parameter; parties=$Parties; trials=$Trials; warmup=$Warmup; max_depth=$MaxDepth
        gomaxprocs=1; cpu_affinity='not set by this wrapper'; cpu=$cpu
        os=[Runtime.InteropServices.RuntimeInformation]::OSDescription
        source_manifest_sha256=if(Test-Path -LiteralPath $manifest){(Get-FileHash -LiteralPath $manifest).Hash.ToLowerInvariant()}else{$null}
        note='Local run only. Check construction, correctness and timing eligibility separately.'
    }
    $environment | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $runDir 'environment.json') -Encoding utf8
    & (Join-Path $PSScriptRoot 'preflight.ps1') -RunDir $runDir -PythonExe $PythonExe
    if ($Experiment -ne 'figures') {
        Run-Native 'go' @('version') $repoRoot 'go-version.log'
    }
    $mkfRoot = Join-Path $repoRoot 'software/mkf'
    $kklssRoot = Join-Path $repoRoot 'software/kklss'
    switch ($Experiment) {
        'cpu-mkf' {
            Write-Warning 'Historical aggregate-output profiler, not the matched full paper construction.'
            Run-Native 'go' @('run','./cmd/mkfbench','-scheme',$Scheme,'-param',$Parameter,'-parties',"$Parties",'-warmup',"$Warmup",'-runs',"$Trials",'-out',(Join-Path $runDir 'cpu_raw.csv')) $mkfRoot 'cpu.log'
        }
        'cpu-kklss-stages' {
            $env:KKLSS_STAGE_OUT = Join-Path $runDir 'kklss_stage_raw.csv'
            Run-Native 'go' @('test',"./mk$Scheme",'-p','1','-parallel','1','-run','^TestProfileStagesExactP1N16$','-count=1','-timeout=24h','-v') $kklssRoot 'stages.log'
        }
        'noise' {
            $env:NOISE_OUT=Join-Path $runDir 'noise_raw.csv'
            $env:NOISE_PARAMS=$Parameter; $env:NOISE_PARTIES="$Parties"
            $env:NOISE_SCHEMES=$Scheme.ToUpperInvariant(); $env:NOISE_TRIALS="$Trials"
            $env:NOISE_MAX_DEPTH="$MaxDepth"; $env:NOISE_TARGET_PRECISION='20'
            $env:NOISE_CKKS_CHAIN_MODE=''
            if ($Implementation -eq 'mkf') {
                Run-Native 'go' @('test','./mkf_cpu','-p','1','-parallel','1','-run','^TestConcreteNoiseMKF$','-count=1','-timeout=24h','-v') $mkfRoot 'noise.log'
            } else {
                $pattern = if($Scheme -eq 'ckks'){'^TestConcreteNoiseKKLSSCKKS$'}else{'^TestConcreteNoiseKKLSSBFV$'}
                Run-Native 'go' @('test',"./mk$Scheme",'-p','1','-parallel','1','-run',$pattern,'-count=1','-timeout=24h','-v') $kklssRoot 'noise.log'
            }
        }
        'mnist' {
            $data = Join-Path $kklssRoot 'cnn/data/mnist_test.csv'
            if (-not (Test-Path -LiteralPath $data)) { throw 'MNIST CSV missing. Follow datasets/README.md before this heavy test.' }
            Run-Native 'go' @('test','./cnn','-p','1','-parallel','1','-run','^TestCNN$','-count=1','-timeout=24h','-v') $kklssRoot 'mnist.log'
        }
        'figures' {
            $launcher = [IO.Path]::GetFileName($PythonExe) -in @('py','py.exe')
            foreach ($spec in @(@('mkf','MKF-multi-CPU-FPGA.py'),@('kklss','KKLSS-multi-CPU-FPGA.py'))) {
                $dest = Join-Path $runDir $spec[0]
                New-Item -ItemType Directory -Path $dest | Out-Null
                Copy-Item -LiteralPath (Join-Path $repoRoot ('figures/'+$spec[0]+'/'+$spec[1])) -Destination $dest
                $pythonArgs = @()
                if ($launcher) { $pythonArgs += '-3' }
                $pythonArgs += (Join-Path $dest $spec[1])
                Run-Native $PythonExe $pythonArgs $dest ($spec[0]+'_figure.log')
            }
        }
    }
    $status='PASS'
} catch {
    $failure=$_.Exception.Message
    throw
} finally {
    foreach ($name in $envNames) { [Environment]::SetEnvironmentVariable($name,$savedEnv[$name],'Process') }
    [ordered]@{schema='mkf-local-experiment-v1';status=$status;experiment=$Experiment;failure=$failure;note='PASS applies only to this selected run, not publication or board qualification.'} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runDir 'RUN_SUMMARY.json') -Encoding utf8
    $runLock.Dispose()
    Write-Host "Experiment $status; local output: $runDir"
}
