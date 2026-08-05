# Adaptive Rollback

## 목적

Adaptive Rollback은 과거 revision의 weight를 새 revision으로 복원합니다. 기존 revision을 덮어쓰거나 삭제하지 않습니다.

## Dry Run

```powershell
.\.venv\Scripts\python.exe `
    -m tools.validation.run_adaptive_rollback `
    --profile ".\out_adaptive_automation\profiles\revision-00000014.json" `
    --repository ".\out_adaptive_automation" `
    --rollback-revision 13 `
    --rollback-id "rollback-preview-15" `
    --dry-run
```

## 승인 Rollback

```powershell
.\.venv\Scripts\python.exe `
    -m tools.validation.run_adaptive_rollback `
    --profile ".\out_adaptive_automation\profiles\revision-00000014.json" `
    --repository ".\out_adaptive_automation" `
    --rollback-revision 13 `
    --rollback-id "rollback-15" `
    --approve-rollback
```

`--approve-rollback` 없이 저장할 수 없습니다.
