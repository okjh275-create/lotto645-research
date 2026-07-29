param(
    [string]$OutputPath = ""
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

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputPath = Join-Path `
        $RepoRoot `
        "project_inventory_$Timestamp.txt"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $RepoRoot $OutputPath
}

$Lines = New-Object System.Collections.Generic.List[string]

function Add-Line {
    param([string]$Text = "")

    $script:Lines.Add($Text)
}

function Add-Section {
    param(
        [string]$Title,
        [scriptblock]$Command
    )

    Add-Line ""
    Add-Line "===== $Title ====="

    try {
        $Result = & $Command 2>&1

        if ($null -eq $Result) {
            Add-Line "(no output)"
            return
        }

        foreach ($Item in $Result) {
            Add-Line (($Item | Out-String).TrimEnd())
        }
    }
    catch {
        Add-Line "ERROR: $($_.Exception.Message)"
    }
}

Add-Line "LRP Development Inventory"
Add-Line "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Add-Line "Repository: $RepoRoot"

Add-Section "GIT STATUS" {
    git status --short
}

Add-Section "RECENT COMMITS" {
    git log --oneline -10
}

Add-Section "PYTHON VERSION" {
    & $Python --version
}

Add-Section "LEARNING PACKAGE FILES" {
    $LearningPath = Join-Path $RepoRoot "lrp\learning"

    if (Test-Path $LearningPath) {
        Get-ChildItem $LearningPath -File -Filter *.py |
            Sort-Object Name |
            ForEach-Object {
                "{0} ({1} bytes)" -f $_.Name, $_.Length
            }
    }
}

Add-Section "PREDICTION-RELATED FILES" {
    $SearchRoots = @(
        "lrp",
        "engine",
        "model",
        "cli"
    )

    foreach ($Root in $SearchRoots) {
        $Path = Join-Path $RepoRoot $Root

        if (-not (Test-Path $Path)) {
            continue
        }

        Get-ChildItem $Path -Recurse -File -Filter *.py |
            Where-Object {
                $_.FullName -match `
                    "predict|pipeline|candidate|score|weight|strategy|snapshot|learning"
            } |
            Sort-Object FullName |
            ForEach-Object {
                $_.FullName.Replace("$RepoRoot\", "")
            }
    }
}

Add-Section "CORE CLASSES AND FUNCTIONS" {
    $Patterns = @(
        "^class .*Predict",
        "^class .*Pipeline",
        "^class .*Candidate",
        "^class .*Scor",
        "^class .*Learning",
        "^\s*def .*predict",
        "^\s*def .*score",
        "^\s*def .*candidate",
        "LearningSnapshot",
        "AdaptiveWeight",
        "strategy_type"
    )

    $SearchRoots = @(
        "lrp",
        "engine",
        "model",
        "cli"
    )

    foreach ($Root in $SearchRoots) {
        $Path = Join-Path $RepoRoot $Root

        if (-not (Test-Path $Path)) {
            continue
        }

        Get-ChildItem $Path -Recurse -File -Filter *.py |
            Select-String -Pattern $Patterns |
            ForEach-Object {
                "{0}:{1}: {2}" -f `
                    $_.Path.Replace("$RepoRoot\", ""),
                    $_.LineNumber,
                    $_.Line.Trim()
            }
    }
}

Add-Section "TEST FILES" {
    $TestPath = Join-Path $RepoRoot "tests"

    if (Test-Path $TestPath) {
        Get-ChildItem $TestPath -Recurse -File -Filter *.py |
            Sort-Object FullName |
            ForEach-Object {
                $_.FullName.Replace("$RepoRoot\", "")
            }
    }
}

Add-Section "PYTEST COLLECTION" {
    & $Python -m pytest --collect-only -q
}

$Parent = Split-Path $OutputPath -Parent

if ($Parent -and -not (Test-Path $Parent)) {
    New-Item -ItemType Directory -Force -Path $Parent |
        Out-Null
}

$Lines |
    Set-Content `
        -Path $OutputPath `
        -Encoding UTF8

Write-Host ""
Write-Host "Inventory completed:"
Write-Host $OutputPath
