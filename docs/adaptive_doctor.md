# Adaptive Automation Doctor

## 목적

Adaptive Automation Doctor는 repository 상태, profile 무결성, revision 연속성, rollback metadata, automation record를 점검하고 JSON과 Markdown 보고서를 생성합니다.

## 실행

```powershell
.\.venv\Scripts\python.exe `
    -m tools.validation.run_adaptive_doctor `
    --repository ".\out_adaptive_automation" `
    --output ".\out_adaptive_doctor" `
    --stem "adaptive_doctor"
```

`--fail-on-issues`를 사용하면 최종 상태가 FAIL일 때 exit code 1을 반환합니다.

## 검사 항목

- filename revision과 `target_revision` 일치
- nested `profile.revision` 일치
- source-to-target lineage
- revision 연속성
- timezone-aware timestamp
- weight 합계 1.0
- confidence와 sample_size
- rollback lineage
- automation identifier
