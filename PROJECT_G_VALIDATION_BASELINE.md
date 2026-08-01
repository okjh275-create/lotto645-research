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
