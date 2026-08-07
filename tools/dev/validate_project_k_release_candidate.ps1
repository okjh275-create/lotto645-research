param(
    [switch]$AllowDirty,
    [string]$ExpectedBranch = "feature/project-k-adaptive-learning"
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("===== {0} =====" -f $Title)
}

function Assert-LastExitCode {
    param([string]$Description)

    if ($LASTEXITCODE -ne 0) {
        throw (
            "{0} failed with exit code {1}." -f
            $Description,
            $LASTEXITCODE
        )
    }
}

function Assert-PathExists {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        throw ("Required path is missing: {0}" -f $Path)
    }
}

try {
    Write-Section "REPOSITORY"

    $branch = (git branch --show-current).Trim()
    Assert-LastExitCode "Read current branch"

    $commit = (git log -1 --oneline).Trim()
    Assert-LastExitCode "Read latest commit"

    Write-Host ("Branch: {0}" -f $branch)
    Write-Host ("Commit: {0}" -f $commit)

    if ($ExpectedBranch -and $branch -ne $ExpectedBranch) {
        throw (
            "Unexpected branch. Expected '{0}', actual '{1}'." -f
            $ExpectedBranch,
            $branch
        )
    }

    git diff --check
    Assert-LastExitCode "Git whitespace check"

    $status = @(git status --short)
    Assert-LastExitCode "Git status"

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

    Write-Section "RELEASE DOCUMENT"

    Assert-PathExists ".\PROJECT_K_RELEASE_CANDIDATE.md"
    Write-Host "Project K release document: PASS"

    Write-Section "PUBLIC IMPORTS"

    .\.venv\Scripts\python.exe -c "from lrp.outcomes import OutcomeBridge, OutcomeLearningBridge, OutcomeImporter; from lrp.operations import RoundCompletionRepository, RoundCompletionSummary, summarize_round_completions; from lrp.pipelines.round_completion import RoundCompletionPipeline, RoundCompletionResult; print('project_k_public_imports = PASS')"
    Assert-LastExitCode "Project K public imports"

    Write-Section "CLI IMPORTS"

    .\.venv\Scripts\python.exe -c "from lrp.cli.round_complete import main as round_complete_main; from lrp.cli.status import main as status_main; from lrp.cli.doctor import main as doctor_main; print('project_k_cli_imports = PASS')"
    Assert-LastExitCode "Project K CLI imports"

    Write-Section "FOCUSED OPERATIONAL VALIDATION"

    .\.venv\Scripts\python.exe -m pytest `
        .\tests\integration\test_round_completion_operational_flow.py `
        .\tests\cli\test_round_complete_cli.py `
        .\tests\cli\test_status_round_completion.py `
        .\tests\cli\test_doctor_round_completion.py `
        .\tests\operations\test_round_completion_repository.py `
        -q `
        --durations=10

    Assert-LastExitCode "Focused operational validation"

    Write-Section "FULL REGRESSION"

    powershell.exe `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File .\tools\dev\run_tests.ps1

    Assert-LastExitCode "Full regression"

    Write-Section "PROJECT K RELEASE CANDIDATE RESULT"

    Write-Host "Status: PASS"
    Write-Host "Release document: PASS"
    Write-Host "Public imports: PASS"
    Write-Host "CLI imports: PASS"
    Write-Host "Focused operational validation: PASS"
    Write-Host "Full regression: PASS"
    Write-Host ("Branch: {0}" -f $branch)
    Write-Host ("Commit: {0}" -f $commit)

    exit 0
}
catch {
    Write-Section "PROJECT K RELEASE CANDIDATE RESULT"
    Write-Host "Status: FAIL"
    Write-Host ("Reason: {0}" -f $_.Exception.Message)
    exit 1
}
