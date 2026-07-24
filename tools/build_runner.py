"""Dependency-free script regression runner for LRP."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import subprocess, sys, time
from typing import Iterable, Sequence

@dataclass(frozen=True, slots=True)
class ScriptTestResult:
    path: str
    return_code: int
    elapsed_seconds: float
    @property
    def passed(self) -> bool: return self.return_code == 0

@dataclass(frozen=True, slots=True)
class ScriptTestSummary:
    discovered: int
    passed: int
    failed: int
    elapsed_seconds: float
    results: tuple[ScriptTestResult, ...]
    @property
    def success(self) -> bool: return self.discovered > 0 and self.failed == 0
    @property
    def status(self) -> str:
        if self.discovered == 0: return "skipped:no_tests"
        if self.failed: return f"failed:script:{self.failed}/{self.discovered}"
        return f"passed:script:{self.passed}/{self.discovered}"

def discover_script_tests(test_directories: Iterable[Path]) -> tuple[Path, ...]:
    found: set[Path] = set()
    for directory in test_directories:
        if not directory.is_dir(): continue
        for pattern in ("test_*.py", "*_test.py"):
            found.update(p.resolve() for p in directory.rglob(pattern) if p.is_file())
    return tuple(sorted(found, key=lambda p: p.as_posix().casefold()))

def run_script_tests(*, project_root: Path, test_directories: Sequence[Path], python_executable: str|None=None, fail_fast: bool=True) -> ScriptTestSummary:
    started=time.perf_counter(); root=project_root.resolve(); exe=python_executable or sys.executable
    tests=discover_script_tests(test_directories)
    if not tests: return ScriptTestSummary(0,0,0,round(time.perf_counter()-started,6),())
    print("\n-------------------------------------\nRunning Regression Tests\n-------------------------------------")
    results=[]
    for path in tests:
        rel=path.relative_to(root); print("\nRUN:", exe, rel)
        tick=time.perf_counter(); cp=subprocess.run([exe,str(rel)],cwd=root,check=False)
        item=ScriptTestResult(rel.as_posix(),cp.returncode,round(time.perf_counter()-tick,6)); results.append(item)
        print(("PASS" if item.passed else "FAIL")+f": {item.path} ({item.elapsed_seconds:.3f}s)")
        if not item.passed and fail_fast: break
    passed=sum(x.passed for x in results); failed=len(results)-passed
    out=ScriptTestSummary(len(tests),passed,failed,round(time.perf_counter()-started,6),tuple(results))
    print(f"\n-------------------------------------\n{out.passed} / {out.discovered} PASSED\n-------------------------------------")
    return out
