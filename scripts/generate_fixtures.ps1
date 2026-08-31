[CmdletBinding()]
param(
    [string]$OutputDirectory,
    [switch]$AllowLargeOutput
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $AllowLargeOutput) {
    throw 'Fixture generation is large and resource intensive. Re-run with -AllowLargeOutput.'
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $repoRoot 'fixtures\generated'
}
$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
$fixtureRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'fixtures'))
if (-not $outputPath.StartsWith($fixtureRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'OutputDirectory must be below the repository fixtures directory.'
}
if (Test-Path -LiteralPath $outputPath) {
    $existing = Get-ChildItem -LiteralPath $outputPath -Force
    if ($existing.Count -ne 0) {
        throw "Refusing to overwrite a non-empty fixture directory: $outputPath"
    }
}
else {
    New-Item -ItemType Directory -Path $outputPath | Out-Null
}

$drive = Get-PSDrive -PSProvider FileSystem |
    Where-Object { $outputPath.StartsWith($_.Root, [StringComparison]::OrdinalIgnoreCase) } |
    Sort-Object { $_.Root.Length } -Descending |
    Select-Object -First 1
if ($drive -and $drive.Free -lt 4GB) {
    throw 'At least 4 GiB of free disk space is required before fixture generation.'
}

function Invoke-GoFixtureTest([string]$Directory, [string]$Package, [string]$Pattern) {
    Push-Location $Directory
    try {
        & go test $Package -run $Pattern -count=1
        if ($LASTEXITCODE -ne 0) {
            throw "Fixture generation failed in $Directory for $Package/$Pattern"
        }
    }
    finally {
        Pop-Location
    }
}

$previousMax = $env:GOMAXPROCS
$previousRoot = $env:FPGA_FIXTURE_ROOT
$previousDir = $env:FPGA_FIXTURE_DIR
try {
    $env:GOMAXPROCS = '1'
    $env:FPGA_FIXTURE_ROOT = $outputPath
    Invoke-GoFixtureTest (Join-Path $repoRoot 'software\mkf') './mkf_cpu' '^(TestExportP1N16RingConstants|TestExportP1N16Fixtures)$'

    $env:FPGA_FIXTURE_DIR = Join-Path $outputPath 'kklss_ckks_p1_n16'
    Invoke-GoFixtureTest (Join-Path $repoRoot 'software\kklss') './mkckks' '^TestExportP1N16Fixture$'

    $env:FPGA_FIXTURE_DIR = Join-Path $outputPath 'kklss_bfv_p1_n16'
    Invoke-GoFixtureTest (Join-Path $repoRoot 'software\kklss') './mkbfv' '^TestExportP1N16Fixture$'
}
finally {
    $env:GOMAXPROCS = $previousMax
    $env:FPGA_FIXTURE_ROOT = $previousRoot
    $env:FPGA_FIXTURE_DIR = $previousDir
}

$records = Get-ChildItem -LiteralPath $outputPath -Recurse -File |
    Sort-Object FullName |
    ForEach-Object {
        [ordered]@{
            path = [IO.Path]::GetRelativePath($outputPath, $_.FullName).Replace('\', '/')
            bytes = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
$manifest = [ordered]@{
    schema = 'mkf-u55c-generated-fixtures-v1'
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
    files = $records
}
$manifest | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath (Join-Path $outputPath 'SHA256_MANIFEST.json') -Encoding UTF8
Write-Host "Fixture generation PASS: $outputPath"
