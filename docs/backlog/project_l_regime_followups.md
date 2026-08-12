# Project L — Regime Follow-up Backlog

Status: POST-PROJECT-L
Source: Project L completion / L-10F empirical validation

## Completed baseline

Project L prediction-intelligence regime learning is complete.

Validated:
- Regime public API
- Historical replay integration
- Calibration learning
- Bayesian learning
- Calibration + Bayesian combined scenario
- Regime learning provenance
- Empirical comparison runner
- Real-history recent-100 replay
- Full regression

Completion evidence:
- Target regression: 285 passed
- Full suite collection: 1343 tests
- Recent-100 substantive regime rate: 61%
- high_band_expansion: 26
- low_band_expansion: 35
- neutral: 39
- calibration revisions: 61
- Bayesian revisions: 61

## Follow-up 1 — gap_recovery calibration

Observed:
- gap_recovery primary count = 0 in recent-100 empirical validation.

Candidate cause:
- Feature-scale mismatch, especially variance-related inputs.

Required future work:
1. Measure gap_recovery feature distributions across longer historical windows.
2. Compare detector fixture ranges with extractor-produced ranges.
3. Do not tune thresholds against only the recent-100 sample.
4. Add out-of-sample / rolling-window validation before changing production scoring.
5. Require empirical improvement or justified regime coverage before activation.

Classification:
- Non-blocking follow-up.
- Do not reopen Project L solely for this item.

## Follow-up 2 — cluster_rotation calibration

Observed:
- cluster_rotation primary count = 0 in recent-100 empirical validation.

Candidate cause:
- Feature-scale mismatch in dispersion / pair-variance inputs.

Required future work:
1. Measure historical cluster_rotation signal ranges.
2. Validate expected pair-density / pair-variance geometry.
3. Rebuild detector tests using extractor-realistic feature ranges.
4. Evaluate multiple non-overlapping historical windows.
5. Avoid increasing regime frequency merely to achieve artificial class balance.

Classification:
- Non-blocking follow-up.
- Do not reopen Project L solely for this item.

## Guardrails

Future calibration must:
- preserve deterministic replay,
- preserve fail-open behavior,
- preserve neutral handling,
- retain provenance,
- pass the complete regression suite,
- be evaluated out-of-sample,
- avoid tuning solely to improve regime occurrence counts.

## Project L closure decision

Project L is considered COMPLETE.

The two items above are model-calibration research backlog items,
not missing Project L implementation contracts.
