param(
    [switch]$CompileOnly,
    [switch]$FullOutput
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (
    Join-Path $PSScriptRoot "..\.."
)

Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Python virtual environment not found: $Python"
}

Write-Host ""
Write-Host "===== PYTHON ====="
& $Python --version

if ($LASTEXITCODE -ne 0) {
    throw "Python execution failed"
}

Write-Host ""
Write-Host "===== COMPILE CHECK ====="

$compileTargets = @(
    "lrp",
    "engine",
    "model",
    "cli"
)

foreach ($target in $compileTargets) {
    $path = Join-Path $RepoRoot $target

    if (-not (Test-Path $path)) {
        continue
    }

    Write-Host "Compiling: $target"

    & $Python -m compileall -q $path

    if ($LASTEXITCODE -ne 0) {
        throw "Compile check failed: $target"
    }
}

Write-Host "Compile check passed"

if ($CompileOnly) {
    exit 0
}

Write-Host ""
Write-Host "===== PYTEST ====="

if ($FullOutput) {
    & $Python -m pytest
}
else {
    & $Python -m pytest -q
}

if ($LASTEXITCODE -ne 0) {
    throw "Pytest failed"
}

Write-Host ""
Write-Host "All tests passed"
