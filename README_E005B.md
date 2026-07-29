# Project E-005B — Adaptive Weight Reporting & Validation

## 목적

기존 `AdaptiveWeightEngine`이 계산한 가중치를 변경하지 않고,
사람이 검토할 수 있는 설명형 리포트로 변환합니다.

## 추가 파일

- `lrp/learning/adaptive_report.py`
- `tests/test_e005b_adaptive_weight_report.py`

## 교체 파일

- `lrp/learning/__init__.py`

## 적용

ZIP을 프로젝트 루트에 풀어 덮어씁니다.

```powershell
Expand-Archive `
  ".\LRP_Project_E005B_Adaptive_Weight_Report.zip" `
  "." `
  -Force
```

## 단독 회귀 테스트

```powershell
.\.venv\Scripts\python.exe `
  ".\tests\test_e005b_adaptive_weight_report.py"
```

## 전체 회귀 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

pytest를 사용하지 않는 프로젝트라면 기존 테스트 실행 스크립트를 그대로 사용하십시오.

## 설계 제약

- 기존 Adaptive Weight 계산 공식을 변경하지 않습니다.
- SQLite 스키마를 변경하지 않습니다.
- 리포트 계층은 읽기 전용입니다.
- 서로 다른 repository revision의 가중치를 한 리포트에 혼합하지 않습니다.
- 입력 순서와 무관하게 rank position 기준으로 결정론적 정렬을 수행합니다.
