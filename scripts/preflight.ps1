[CmdletBinding()]
param(
    [string]$RunDir,
    [string]$PythonExe = 'py',
    [switch]$RequireU55C
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $RunDir) {
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
    $RunDir = Join-Path $repoRoot ".local\${stamp}_preflight"
}
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

function Get-CommandRecord([string]$Name) {
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    return [ordered]@{
        available = [bool]$command
        source = if ($command) { $command.Source } else { $null }
    }
}

$commands = [ordered]@{}
foreach ($name in @('git', 'go', 'vitis_hls', 'vivado', 'v++', 'platforminfo', 'xbutil')) {
    $commands[$name] = Get-CommandRecord $name
}

$pythonCommand = Get-Command $PythonExe -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    throw "Python launcher '$PythonExe' was not found."
}
$pythonLeaf = [IO.Path]::GetFileName($PythonExe)
$useLauncher = $pythonLeaf -in @('py', 'py.exe')
$scanOut = Join-Path $RunDir 'publication-scan.json'
if ($useLauncher) {
    & $PythonExe -3 (Join-Path $PSScriptRoot 'publication_scan.py') --verify-manifest --out $scanOut
}
else {
    & $PythonExe (Join-Path $PSScriptRoot 'publication_scan.py') --verify-manifest --out $scanOut
}
if ($LASTEXITCODE -ne 0) {
    throw "Publication scan failed. See $scanOut"
}

$platform = [ordered]@{
    expected = 'xilinx_u55c_gen3x16_xdma_3_202210_1'
    listing_available = $commands['platforminfo'].available
    expected_visible = $false
    listing_file = $null
}
if ($commands['platforminfo'].available) {
    $listingPath = Join-Path $RunDir 'platform-list.txt'
    & platforminfo -l 2>&1 | Tee-Object -FilePath $listingPath | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $listing = Get-Content -LiteralPath $listingPath -Raw
        $platform.expected_visible = $listing.Contains($platform.expected)
        $platform.listing_file = $listingPath
    }
}

$report = [ordered]@{
    schema = 'mkf-u55c-local-preflight-v1'
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
    commands = $commands
    u55c = [ordered]@{
        part = 'xcu55c-fsvh2892-2L-e'
        toolchain_target = 'Vitis/Vivado 2022.2'
        platform = $platform
        hls_ready_on_this_host = ($commands['vitis_hls'].available -and $platform.expected_visible)
    }
    publication_scan = $scanOut
}
$reportPath = Join-Path $RunDir 'preflight.json'
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8

Write-Host "Preflight PASS: $reportPath"
if (-not $commands['vitis_hls'].available) {
    Write-Warning 'Vitis HLS is absent; U55C C simulation and synthesis cannot run on this host.'
}
if ($RequireU55C -and (-not $commands['vitis_hls'].available -or -not $platform.expected_visible)) {
    throw 'U55C preflight failed: activate Vitis/Vivado 2022.2 and the exact U55C platform.'
}
