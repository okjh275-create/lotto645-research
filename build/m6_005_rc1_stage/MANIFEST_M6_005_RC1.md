# M6-005 Adaptive Weight RC1

## Scope

- Adds a derived, memory-only adaptive-weight layer.
- Preserves the append-only SQLite schema.
- Reuses the M6-004 `(event_count, maximum_rowid)` revision.
- Keeps M6-001 through M6-004 public APIs compatible.
- Adds `LearningService.get_adaptive_weights()`.

## Formula

Target unit score:

`0.60 * rank_score + 0.25 * confidence + 0.15 * stability`

Target weight is mapped to `[0.50, 1.50]`.

EMA:

`current = 0.80 * previous + 0.20 * target`

Final current weights are normalized so their sum is 1.0.

## Intentionally excluded

- SQLite adaptive-weight tables
- Persistent snapshots/history
- Entropy and drift persistence
- Prediction hints
- Regime weighting
- Project A integration

## Files

New:

- `lrp/learning/adaptive_models.py`
- `lrp/learning/adaptive_engine.py`
- `lrp/learning/adaptive_repository.py`
- `lrp/learning/learning_facade.py`
- `tests/test_m6_adaptive_weight.py`

Full replacements:

- `lrp/learning/service.py`
- `lrp/learning/__init__.py`
