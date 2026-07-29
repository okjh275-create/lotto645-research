# Project E-005C — Learning Snapshot Writer

## 목적

현재 Learning 상태를 회차별 디렉터리에 읽기 전용 JSON 스냅샷으로 저장합니다.

생성 파일:

```text
snapshots/<round_no>/
  rankings.json
  adaptive_weights.json
  performance.json
  adaptive_report.json
  metadata.json
  SHA256SUMS.txt
```

## 전제

E-005B가 먼저 적용되어 `lrp/learning/adaptive_report.py`가 존재해야 합니다.

## 추가 파일

- `lrp/learning/snapshot.py`
- `tests/test_e005c_learning_snapshot.py`

## 교체 파일

- `lrp/learning/__init__.py`

## 적용 방법 A — ZIP

ZIP을 먼저 이 채팅에서 다운로드한 뒤 프로젝트 루트로 복사합니다.

```powershell
Copy-Item `
  "$HOME\Downloads\LRP_Project_E005C_Learning_Snapshot_Writer.zip" `
  "C:\Users\PC2303\Documents\GitHub\lotto645-research\"

Set-Location `
  "C:\Users\PC2303\Documents\GitHub\lotto645-research"

Expand-Archive `
  ".\LRP_Project_E005C_Learning_Snapshot_Writer.zip" `
  "." `
  -Force
```

## 적용 방법 B — Git Patch

`LRP_Project_E005C.patch` 파일을 프로젝트 루트에 둔 다음 실행합니다.

```powershell
git apply --check ".\LRP_Project_E005C.patch"
git apply ".\LRP_Project_E005C.patch"
```

## 단독 테스트

```powershell
.\.venv\Scripts\python.exe `
  ".\tests\test_e005c_learning_snapshot.py"
```

## 전체 회귀 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 설계 보장

- Learning DB를 수정하지 않는 읽기 전용 수집
- Ranking, Adaptive Weight, Performance, Report revision 일치 검증
- 전략 키 집합 일치 검증
- JSON 파일 원자적 교체
- SHA-256 무결성 기록
- 기존 스냅샷 덮어쓰기 방지
- 동일 입력과 동일 시각에 대한 결정론적 출력
