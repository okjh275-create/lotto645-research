param(
    [switch]$AllowDirty,
    [string]$ExpectedBranch = "feature/project-i-adaptive-automation"
)

$ErrorActionPreference = "Stop"

function Section {
    param([string]$Name)
    Write-Host ""
    Write-Host ("===== {0} =====" -f $Name)
}

function Require-Success {
    param([string]$Name)
    if ($LASTEXITCODE -ne 0) {
        throw ("{0} failed with exit code {1}" -f $Name, $LASTEXITCODE)
    }
}

try {
    Section "REPOSITORY"

    $branch = (git branch --show-current).Trim()
    Require-Success "git branch"

    $commit = (git log -1 --oneline).Trim()
    Require-Success "git log"

    Write-Host ("Branch: {0}" -f $branch)
    Write-Host ("Commit: {0}" -f $commit)

    if ($ExpectedBranch -and $branch -ne $ExpectedBranch) {
        throw ("Unexpected branch. Expected '{0}', actual '{1}'." -f $ExpectedBranch, $branch)
    }

    git diff --check
    Require-Success "git diff --check"

    $status = @(git status --short)
    Require-Success "git status"

    if (-not $AllowDirty -and $status.Count -gt 0) {
        Write-Host "Working tree changes:"
        $status | ForEach-Object { Write-Host $_ }
        throw "Working tree is not clean."
    }

    if ($AllowDirty) {
        Write-Host "Dirty-tree check: SKIPPED"
    }
    else {
        Write-Host "Dirty-tree check: PASS"
    }

    Section "DOCUMENTATION"

    $documents = @(
        ".\docs\adaptive_automation.md",
        ".\docs\adaptive_doctor.md",
        ".\docs\adaptive_rollback.md",
        ".\docs\repository_layout.md",
        ".\docs\approval_workflow.md",
        ".\docs\troubleshooting.md",
        ".\PROJECT_J_PERFORMANCE_BASELINE.md",
        ".\PROJECT_J_RELEASE_CANDIDATE.md"
    )

    foreach ($document in $documents) {
        if (-not (Test-Path $document)) {
            throw ("Required document missing: {0}" -f $document)
        }

        $bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $document).Path)

        if (
            $bytes.Length -ge 3 -and
            $bytes[0] -eq 0xEF -and
            $bytes[1] -eq 0xBB -and
            $bytes[2] -eq 0xBF
        ) {
            throw ("UTF-8 BOM detected: {0}" -f $document)
        }

        if ($bytes.Length -eq 0 -or $bytes[$bytes.Length - 1] -ne 0x0A) {
            throw ("Missing final newline: {0}" -f $document)
        }

        Write-Host ("PASS: {0}" -f $document)
    }

    Section "PUBLIC API"

    .\.venv\Scripts\python.exe -c "from lrp.evolution.feedback import AdaptiveAutomationDoctor, AdaptiveAutomationDoctorMarkdownRenderer, AdaptiveAutomationDoctorReportWriter, AdaptiveProfileIntegrityDoctor, AdaptiveRepositoryStatusAnalyzer; print('public_api = PASS')"
    Require-Success "public API import"

    Section "CLI MODULES"

    .\.venv\Scripts\python.exe -c "import tools.validation.run_adaptive_automation, tools.validation.run_adaptive_rollback, tools.validation.run_adaptive_doctor; print('cli_modules = PASS')"
    Require-Success "CLI module import"

    Section "FOCUSED VALIDATION"

    .\.venv\Scripts\python.exe -m pytest `
        .\tests\integration\test_adaptive_automation_end_to_end.py `
        .\tests\integration\test_cli_smoke_scenario.py `
        .\tests\performance\test_adaptive_automation_performance.py `
        -q `
        --durations=10

    Require-Success "focused validation"

    Section "FULL REGRESSION"

    powershell.exe `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File .\tools\dev\run_tests.ps1

    Require-Success "full regression"

    Section "RELEASE CANDIDATE RESULT"

    Write-Host "Status: PASS"
    Write-Host "Documentation: PASS"
    Write-Host "Public API: PASS"
    Write-Host "CLI modules: PASS"
    Write-Host "Focused validation: PASS"
    Write-Host "Full regression: PASS"
    Write-Host ("Branch: {0}" -f $branch)
    Write-Host ("Commit: {0}" -f $commit)

    exit 0
}
catch {
    Section "RELEASE CANDIDATE RESULT"
    Write-Host "Status: FAIL"
    Write-Host ("Reason: {0}" -f $_.Exception.Message)
    exit 1
}
