# Project J Release Candidate Validation

## Purpose

This document defines the release-candidate validation gate for the
adaptive automation, rollback, Doctor, reporting, integration, and
operational documentation work completed through Project J.

Passing this gate confirms engineering consistency and operational
readiness of the current branch. It does not prove predictive gain or
future lottery performance.

## Candidate Scope

The release candidate includes:

- cross-window validation reporting,
- adaptive feedback analysis,
- recommendation generation,
- safety guards,
- profile-update planning,
- revision-aware persistence,
- explicit approval controls,
- rollback planning and persistence,
- repository status analysis,
- profile-integrity inspection,
- Doctor JSON and Markdown reports,
- automation, rollback, and Doctor CLIs,
- end-to-end integration tests,
- subprocess-level CLI smoke tests,
- performance regression baselines,
- operating documentation.

## Starting Point

- Branch: `feature/project-i-adaptive-automation`
- Documentation commit before RC tooling: `f6322a5`
- Full regression before J-05: `953 passed`
- Python: `3.13.5`
- Primary environment: Windows 10 and PowerShell

## Validation Command

After the RC validation files are committed and the working tree is
clean, run:

```powershell
powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File .\tools\dev\validate_release_candidate.ps1
```

During initial development of the validation script, the dirty-tree
gate may be skipped explicitly:

```powershell
powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File .\tools\dev\validate_release_candidate.ps1 `
    -AllowDirty
```

`-AllowDirty` is for script development only. Final release-candidate
acceptance requires a clean working tree.

## Required Gates

### Repository Gate

- current directory is a Git work tree,
- branch equals `feature/project-i-adaptive-automation`,
- unstaged and staged whitespace checks pass,
- final validation runs on a clean working tree,
- current commit and subject are recorded.

### Documentation Gate

Required documents:

- `docs/adaptive_automation.md`
- `docs/adaptive_doctor.md`
- `docs/adaptive_rollback.md`
- `docs/repository_layout.md`
- `docs/approval_workflow.md`
- `docs/troubleshooting.md`

Each document must:

- exist,
- decode as UTF-8,
- contain no UTF-8 BOM,
- use LF line endings,
- end with a newline,
- contain a top-level Markdown heading,
- contain substantive content.

The documentation generator must also exist.

### Public API Gate

Critical adaptive feedback services must be present in
`lrp.evolution.feedback.__all__` and import successfully.

### CLI Gate

The following modules must load and return help successfully:

- `tools.validation.run_adaptive_automation`
- `tools.validation.run_adaptive_rollback`
- `tools.validation.run_adaptive_doctor`

### Focused Validation Gate

The following suites must pass:

- adaptive automation end-to-end integration,
- subprocess-level CLI smoke flow,
- adaptive automation performance baseline.

### Full Regression Gate

The standard development test runner must pass:

```text
compile check
full pytest suite
```

## Performance Interpretation

Performance limits are conservative regression safety thresholds. A
failure requires investigation against the previous known-good commit,
filesystem cache state, antivirus activity, CPU load, Python version,
and storage characteristics.

A passing performance gate is not a benchmark claim.

## Release Candidate Decision

The candidate is accepted only when the validation script reports:

```text
Status: PASS
Documentation: PASS
Public API: PASS
CLI modules: PASS
Focused validation: PASS
Full regression: PASS
```

Any failed gate blocks release-candidate acceptance.

## Post-Validation Records

Record at minimum:

- branch,
- commit hash,
- commit subject,
- Python version,
- focused-test result,
- full regression count and duration,
- validation date in KST,
- whether the working tree was clean,
- any environmental exceptions.

## Non-Goals

This gate does not:

- merge the branch automatically,
- create a Git tag,
- publish a package,
- deploy files,
- alter adaptive weights,
- claim statistical significance,
- claim predictive improvement.

## Next Step After PASS

After a clean-tree PASS:

1. record the final validation output,
2. review the branch history,
3. merge according to the repository's integration policy,
4. run the validation gate again on the integration branch,
5. create a release tag only after integration-branch PASS.
