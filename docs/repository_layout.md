# Adaptive Automation Repository Layout

## 구조

```text
out_adaptive_automation/
├── automation/
└── profiles/
```

`automation/`에는 recommendation과 실행 결과가 저장됩니다. `profiles/`에는 `revision-XXXXXXXX.json` 형식의 profile revision이 저장됩니다.

filename revision, `target_revision`, `profile.revision`은 일치해야 합니다. 일반 update는 `source_revision + 1 == target_revision`을 만족해야 합니다.

기존 revision 파일을 rename, overwrite, edit, delete하지 않습니다.
