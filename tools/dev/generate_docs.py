from pathlib import Path

DOCS = {
    "adaptive_automation.md": '''# Adaptive Automation

## 목적

Adaptive Automation은 cross-window 검증 보고서와 현재 adaptive profile을 입력으로 받아 revision-aware 업데이트 계획을 생성합니다. 예측력 향상을 증명하는 기능이 아니라, 검증 결과를 추적 가능한 추천과 profile revision으로 연결하는 운영 계층입니다.

## Dry Run

```powershell
.\\.venv\\Scripts\\python.exe `
    -m tools.validation.run_adaptive_automation `
    --report ".\\reports\\cross_window_policy_report.json" `
    --profile ".\\profiles\\current_profile.json" `
    --repository ".\\out_adaptive_automation" `
    --policy "floor" `
    --recommendation-id "auto-1232-preview" `
    --dry-run
```

Dry-run에서는 repository 파일을 생성하지 않습니다.

## 승인 실행

```powershell
.\\.venv\\Scripts\\python.exe `
    -m tools.validation.run_adaptive_automation `
    --report ".\\reports\\cross_window_policy_report.json" `
    --profile ".\\profiles\\current_profile.json" `
    --repository ".\\out_adaptive_automation" `
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
''',
    "adaptive_doctor.md": '''# Adaptive Automation Doctor

## 목적

Adaptive Automation Doctor는 repository 상태, profile 무결성, revision 연속성, rollback metadata, automation record를 점검하고 JSON과 Markdown 보고서를 생성합니다.

## 실행

```powershell
.\\.venv\\Scripts\\python.exe `
    -m tools.validation.run_adaptive_doctor `
    --repository ".\\out_adaptive_automation" `
    --output ".\\out_adaptive_doctor" `
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
''',
    "adaptive_rollback.md": '''# Adaptive Rollback

## 목적

Adaptive Rollback은 과거 revision의 weight를 새 revision으로 복원합니다. 기존 revision을 덮어쓰거나 삭제하지 않습니다.

## Dry Run

```powershell
.\\.venv\\Scripts\\python.exe `
    -m tools.validation.run_adaptive_rollback `
    --profile ".\\out_adaptive_automation\\profiles\\revision-00000014.json" `
    --repository ".\\out_adaptive_automation" `
    --rollback-revision 13 `
    --rollback-id "rollback-preview-15" `
    --dry-run
```

## 승인 Rollback

```powershell
.\\.venv\\Scripts\\python.exe `
    -m tools.validation.run_adaptive_rollback `
    --profile ".\\out_adaptive_automation\\profiles\\revision-00000014.json" `
    --repository ".\\out_adaptive_automation" `
    --rollback-revision 13 `
    --rollback-id "rollback-15" `
    --approve-rollback
```

`--approve-rollback` 없이 저장할 수 없습니다.
''',
    "repository_layout.md": '''# Adaptive Automation Repository Layout

## 구조

```text
out_adaptive_automation/
├── automation/
└── profiles/
```

`automation/`에는 recommendation과 실행 결과가 저장됩니다. `profiles/`에는 `revision-XXXXXXXX.json` 형식의 profile revision이 저장됩니다.

filename revision, `target_revision`, `profile.revision`은 일치해야 합니다. 일반 update는 `source_revision + 1 == target_revision`을 만족해야 합니다.

기존 revision 파일을 rename, overwrite, edit, delete하지 않습니다.
''',
    "approval_workflow.md": '''# Adaptive Approval Workflow

## 원칙

Automation 저장에는 `--approve`, rollback 저장에는 `--approve-rollback`이 필요합니다.

## Automation 승인 순서

1. report와 current profile 확인
2. Doctor 실행
3. Dry-run 실행
4. decisions, violations, revision, weight 변화 검토
5. `--approve` 실행
6. Doctor 재실행
7. 산출물 보관

## Rollback 승인 순서

1. repository health 확인
2. historical revision 선택
3. rollback dry-run 실행
4. difference 검토
5. `--approve-rollback` 실행
6. Doctor 재실행
''',
    "troubleshooting.md": '''# Adaptive Automation Troubleshooting

## No module named

CLI 파일이 없거나 잘못된 repository에서 실행한 경우입니다.

## File not found

예시 경로를 실제 경로로 바꾸지 않았을 가능성이 큽니다.

## UTF-8 BOM

JSON에서 BOM 오류가 발생하면 UTF-8 without BOM으로 다시 저장합니다.

## Approval required

Automation은 `--dry-run` 또는 `--approve` 중 하나가 필요합니다. Rollback은 `--dry-run` 또는 `--approve-rollback` 중 하나가 필요합니다.

## Stale profile

입력 profile revision이 repository head보다 오래된 경우입니다.

## PowerShell `>>`

PowerShell이 미완성 expression이나 here-string 입력을 기다리는 상태입니다. `Ctrl+C`로 취소합니다.
''',
}

root = Path("docs")
root.mkdir(parents=True, exist_ok=True)
for name, content in DOCS.items():
    path = root / name
    normalized = content.strip() + "\n"
    path.write_text(normalized, encoding="utf-8", newline="\n")
    print(f"created: {path} ({len(normalized.encode('utf-8'))} bytes)")
print("documentation_status = CREATED")
