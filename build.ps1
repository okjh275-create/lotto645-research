param(
    [ValidateSet("quick", "full", "clean")]
    [string]$Mode = "quick"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host ""
Write-Host "LRP Build System"
Write-Host "Mode: $Mode"
Write-Host ""

& $Python ".\tools\lrp_build.py" $Mode

if ($LASTEXITCODE -ne 0) {
    throw "LRP build failed with exit code $LASTEXITCODE"
}
