# Adaptive Approval Workflow

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
