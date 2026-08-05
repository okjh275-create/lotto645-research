# Adaptive Automation

## 목적

Adaptive Automation은 cross-window 검증 보고서와 현재 adaptive profile을 입력으로 받아 revision-aware 업데이트 계획을 생성합니다. 예측력 향상을 증명하는 기능이 아니라, 검증 결과를 추적 가능한 추천과 profile revision으로 연결하는 운영 계층입니다.

## Dry Run

```powershell
.\.venv\Scripts\python.exe `
    -m tools.validation.run_adaptive_automation `
    --report ".\reports\cross_window_policy_report.json" `
    --profile ".\profiles\current_profile.json" `
    --repository ".\out_adaptive_automation" `
    --policy "floor" `
    --recommendation-id "auto-1232-preview" `
    --dry-run
```

Dry-run에서는 repository 파일을 생성하지 않습니다.

## 승인 실행

```powershell
.\.venv\Scripts\python.exe `
    -m tools.validation.run_adaptive_automation `
    --report ".\reports\cross_window_policy_report.json" `
    --profile ".\profiles\current_profile.json" `
    --repository ".\out_adaptive_automation" `
    --policy "floor" `
    --recommendation-id "auto-1232" `
    --approve
```

저장 실행에는 `--approve`가 필수입니다. `--dry-run`과 `--approve`는 동시에 사용할 수 없습니다.

## Revision 규칙

- 현재 profile revision은 repository head와 일치해야 합니다.
- target revision은 source revision보다 정확히 1 커야 합니다.
- 기존 revision 파일은 덮어쓰지 않습니다.
- stale profile과 revision collision은 거부됩니다.
- 7개 weight의 합은 1.0이어야 합니다.
