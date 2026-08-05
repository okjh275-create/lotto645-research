# Adaptive Automation Troubleshooting

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
