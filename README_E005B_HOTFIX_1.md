# E-005B Contract Hotfix 1

- delta 소수점 6자리 안정화
- normalized_total 속성 및 JSON 출력 추가
- 비어 있지 않은 리포트의 normalized weight 합계 1.0 검증
- 빈 리포트 호환성 유지

## 적용
```powershell
Expand-Archive `.\LRP_Project_E005B_Contract_Hotfix_1.zip` `.` -Force
```

## 테스트
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
