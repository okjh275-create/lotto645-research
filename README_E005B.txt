Project E-005B Adaptive Weight Reporting & Validation

Files:
- lrp/learning/adaptive_reporting.py
- lrp/learning/service.py
- lrp/learning/__init__.py
- tests/test_e005b_adaptive_weight_reporting.py

Purpose:
- Reuse the existing AdaptiveWeightEngine.
- Validate normalized totals and revision consistency.
- Explain weight changes as RAISED / LOWERED / UNCHANGED.
- Add LearningService.get_adaptive_weight_report().
- Preserve the existing SQLite schema and public APIs.
