# Project J Performance Baseline

## Purpose

This document records the initial performance safety baseline for the
adaptive automation repository, doctor service, and report generation
workflow.

The baseline is intended to detect major regressions, infinite loops,
unexpected filesystem amplification, and severely degraded repository
scanning.

It is not evidence of predictive performance and is not a benchmark
claim against other systems.

## Reference Environment

- Operating system: Windows 10
- Python: 3.13.5
- Test runner: pytest
- Repository: lotto645-research
- Branch at baseline creation:
  `feature/project-i-adaptive-automation`
- Starting commit: `8a420a0`
- Full regression state before Project J-03:
  949 tests passed

## Covered Operations

The performance suite measures:

1. Adaptive doctor inspection of 100 profile revisions.
2. Adaptive doctor inspection of 500 profile revisions.
3. Repeated deterministic JSON and Markdown report generation.
4. Repeated doctor inspection stability.
5. Repository growth handling for profile and automation records.

## Initial Regression Thresholds

### Doctor Repository Scan

- 100 profile revisions: less than 5 seconds
- 500 profile revisions: less than 15 seconds

### Report Generation

- 100 JSON reports and 100 Markdown reports:
  less than 10 seconds

### Repeated Doctor Inspection

- 20 inspections of a 100-revision repository:
  less than 10 seconds

## Interpretation

These limits are intentionally conservative because elapsed time varies
with CPU load, antivirus scanning, filesystem caching, storage type,
virtual environments, and background processes.

A threshold failure means the performance result must be investigated.
It does not automatically prove a code defect.

The investigation should compare:

- the same test on the previous known-good commit,
- cold and warm filesystem-cache runs,
- background process activity,
- generated file count,
- repository record size,
- Python and operating-system versions.

## Correctness Requirements

Performance results are valid only when all functional assertions also
pass.

The performance tests must continue to verify:

- latest revision detection,
- revision count,
- recommendation count,
- zero integrity errors for valid fixtures,
- deterministic doctor results,
- expected JSON and Markdown file counts.

## Non-Goals

This baseline does not currently measure:

- prediction generation performance,
- historical replay duration,
- candidate sampling throughput,
- database query performance,
- network performance,
- cross-machine benchmark comparability,
- peak resident memory with an external profiler.

## Future Extensions

Possible later extensions include:

- tracemalloc-based peak allocation checks,
- subprocess-level CLI timing,
- 1,000- and 5,000-revision stress fixtures,
- Windows and Linux CI comparison,
- percentile-based repeated benchmark recording,
- JSON report size growth tracking.
