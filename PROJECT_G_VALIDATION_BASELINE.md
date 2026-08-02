# Project G Evolution Validation Baseline

## Branch

feature/project-g-evolution-foundation

## Baseline Commit

1adc1d8 feat(cli): add review learning workflow

## Completion Scope

- Adaptive weight calculation
- Bayesian posterior and state signals
- Weighted signal aggregation
- Reward calculation
- UCB1, Thompson Sampling, Epsilon-Greedy
- Reinforcement feedback update
- Learning cycle and persistent learning
- Learning and adaptive-profile snapshots
- Snapshot serialization and repositories
- Evolution public APIs
- Prediction probability weight adapters
- Adaptive profile provider and adapter factory
- PredictionPipeline integration
- Prediction review reward mapping
- Review learning service
- Review signal extraction
- Review-to-profile evolution
- Review learning end-to-end flow
- `lrp review --learn` CLI workflow

## Validation Commands

powershell.exe -ExecutionPolicy Bypass -File .\tools\dev\run_tests.ps1

powershell.exe -ExecutionPolicy Bypass -File .\tools\dev\verify_contracts.ps1

## Closed-Loop Flow

Prediction
→ Review
→ RewardFeedback
→ LearningContext
→ Learning Snapshot
→ Review Signals
→ AdaptiveWeightProfile
→ Profile Snapshot
→ Next Prediction

## Operational Policy

- Default profile confidence threshold: 0.60
- Default minimum sample size: 20
- Profile rejection below policy thresholds is expected.
- Prediction without a profile snapshot uses NoOpEvolutionWeightAdapter.
- Review learning is enabled only with `--learn`.

## Validation-001 Closed Loop Smoke Test

### Test Round

- Prediction round: 1231
- Seed: 20260802
- Candidate count: 1000
- Selected sets: 20
- Winning numbers: 4, 13, 14, 18, 31, 38
- Bonus: 15

### Review Result

- Best main hits: 3
- Best set IDs: S7, S9
- Practical best hits: 3
- Feedback count: 2
- Learning context version: 3

### Adaptive Profile

- Applied: true
- Revision: 1
- Confidence: 0.8
- Sample size: 20
- Learning weight: 0.05179104477611941
- Adaptive weight: 0.05179104477611941

### Probability Vector Comparison

- Changed numbers: 45
- Maximum absolute delta: 0.000036142187536475356
- Total absolute delta: 0.0006002718810181502
- Changed final Top20 sets: 0

### Result

The adaptive profile was loaded and applied successfully.
All 45 number probabilities changed.
The adjustment magnitude was not large enough to change the
final candidate portfolio for this seed.
