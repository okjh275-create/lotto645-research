# Project H Validation Reporting Baseline

## Starting Point

- Parent project: Project G Evolution Foundation
- Starting commit: 2dc2662
- Adaptive adjustment scale: 0.25
- Adaptive minimum weight: 0.03
- Full regression status: PASS
- Working tree at branch creation: clean

## Project G Results Carried Forward

### Core Integration

- Prediction probability components are serialized.
- Feature attribution signals are calculated from winning-number components.
- Review learning persists feature signals.
- Adaptive profiles consume seven signals:
  - hot
  - cold
  - gap
  - trend
  - transition
  - learning
  - adaptive

### Validation Infrastructure

- Historical replay runner
- Replay effectiveness analysis
- Feature attribution effectiveness analysis
- Feature attribution stability analysis
- Adaptive policy A/B comparison runner

### Policy Validation

Three independent 100-round windows were evaluated:

- 932 through 1031
- 1032 through 1131
- 1132 through 1231

The minimum-weight floor policy used:

- adjustment_scale: 0.25
- minimum_weight: 0.03

The floor policy prevented learning and adaptive weights from
collapsing near 0.01 while preserving normalized total weight 1.0.

The observed hit-performance differences were small and were not
statistically significant. The floor promotion is therefore treated
as an engineering stability decision, not proof of predictive gain.

## Project H Goal

Project H will convert replay and policy-validation artifacts into
reproducible, comparable, and explainable reports.

The first implementation scope is:

1. Discover validation runs and verify their contracts.
2. Aggregate replay, effectiveness, attribution, stability, and policy data.
3. Compare multiple windows without manual transcription.
4. Separate factual metrics from statistical interpretation.
5. Detect missing, duplicate, stale, or incompatible artifacts.
6. Produce deterministic JSON and Markdown reports.

## Non-Goals

Project H will not initially:

- change prediction probability logic,
- add new number-selection features,
- tune adaptive weights,
- claim predictive improvement without statistical evidence,
- commit local replay output directories.

## Initial Deliverables

- ValidationRunDiscovery
- ValidationReportModel
- CrossWindowPolicyAggregator
- Project H JSON report
- Project H Markdown report
- Contract and regression tests
