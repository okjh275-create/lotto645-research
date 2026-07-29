param(
    [switch]$SkipTests
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
Write-Host "===== GIT DIFF CHECK ====="

git --no-pager diff --check

if ($LASTEXITCODE -ne 0) {
    throw "Working tree whitespace check failed"
}

git --no-pager diff --cached --check

if ($LASTEXITCODE -ne 0) {
    throw "Staged whitespace check failed"
}

Write-Host "Git diff check passed"

Write-Host ""
Write-Host "===== CHANGED PYTHON BOM CHECK ====="

$ChangedPythonFiles = @(
    git diff --name-only --diff-filter=ACMR
    git diff --cached --name-only --diff-filter=ACMR
    git ls-files --others --exclude-standard
) |
    Where-Object {
        $_ -and $_.EndsWith(".py")
    } |
    Sort-Object -Unique

$BomFiles = New-Object System.Collections.Generic.List[string]

foreach ($RelativePath in $ChangedPythonFiles) {
    $FullPath = Join-Path $RepoRoot $RelativePath

    if (-not (Test-Path $FullPath)) {
        continue
    }

    $Bytes = [System.IO.File]::ReadAllBytes($FullPath)

    if (
        $Bytes.Length -ge 3 -and
        $Bytes[0] -eq 239 -and
        $Bytes[1] -eq 187 -and
        $Bytes[2] -eq 191
    ) {
        $BomFiles.Add($RelativePath)
    }
}

if ($BomFiles.Count -gt 0) {
    Write-Host "UTF-8 BOM detected in changed Python files:"

    foreach ($BomFile in $BomFiles) {
        Write-Host "- $BomFile"
    }

    throw "Changed Python BOM check failed"
}

if ($ChangedPythonFiles.Count -eq 0) {
    Write-Host "No changed Python files to inspect"
}
else {
    Write-Host (
        "Changed Python BOM check passed: " +
        $ChangedPythonFiles.Count +
        " file(s)"
    )
}

Write-Host ""
Write-Host "===== E005D PUBLIC API ====="

$ApiCheck = @(
    "from lrp.learning import LearningCoordinator, LearningCoordinatorConfig, LearningCoordinatorResult"
    "required = {"
    "    'LearningCoordinator': LearningCoordinator,"
    "    'LearningCoordinatorConfig': LearningCoordinatorConfig,"
    "    'LearningCoordinatorResult': LearningCoordinatorResult,"
    "}"
    "for expected, value in required.items():"
    "    if value.__name__ != expected:"
    "        raise RuntimeError(f'Unexpected public API name: {value.__name__}')"
    "print('E005D public API import passed')"
) -join "`n"

& $Python -c $ApiCheck

if ($LASTEXITCODE -ne 0) {
    throw "E005D public API verification failed"
}

Write-Host ""
Write-Host "===== COMPILE CHECK ====="

& $Python -m compileall -q `
    (Join-Path $RepoRoot "lrp")

if ($LASTEXITCODE -ne 0) {
    throw "LRP compile check failed"
}

Write-Host "Compile check passed"

if (-not $SkipTests) {
    Write-Host ""
    Write-Host "===== TESTS ====="

    & $Python -m pytest -q

    if ($LASTEXITCODE -ne 0) {
        throw "Test verification failed"
    }
}

Write-Host ""
Write-Host "Contract verification completed successfully"
