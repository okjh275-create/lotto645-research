"""Fast and reproducible build system for Lotto645 Research Platform.

Modes
-----
quick:
    Compile recently changed Python files and run import smoke tests.

full:
    Compile all project Python files, run available tests, create a
    SHA-256 manifest and produce a reproducible ZIP archive.

Usage
-----
    python tools/lrp_build.py quick
    python tools/lrp_build.py full
"""

from __future__ import annotations

import argparse
import compileall
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
CACHE_PATH = BUILD_DIR / ".lrp_build_cache.json"
MANIFEST_PATH = BUILD_DIR / "lrp_build_manifest.json"
LATEST_ZIP_PATH = BUILD_DIR / "lrp_latest.zip"

PYTHON_DIRS = (
    ROOT / "lrp",
    ROOT / "tools",
)

OPTIONAL_TEST_DIRS = (
    ROOT / "tests",
    ROOT / "lrp" / "tests",
)

SMOKE_MODULES = (
    "lrp",
    "lrp.contracts",
    "lrp.core",
    "lrp.adapters",
    "lrp.pipelines",
)

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
}

ARCHIVE_ROOTS = (
    "lrp",
    "tools",
    "tests",
)

ARCHIVE_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "README.md",
    "build.ps1",
)


@dataclass(slots=True)
class BuildResult:
    mode: str
    started_at_utc: str
    elapsed_seconds: float
    compiled_files: int
    smoke_modules: int
    tests_status: str
    manifest_files: int
    archive_path: str | None
    archive_sha256: str | None
    success: bool


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def is_excluded(path: Path) -> bool:
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return True

    return any(part in EXCLUDED_PARTS for part in relative.parts)


def discover_python_files() -> list[Path]:
    files: list[Path] = []

    for directory in PYTHON_DIRS:
        if not directory.exists():
            continue

        for path in directory.rglob("*.py"):
            if path.is_file() and not is_excluded(path):
                files.append(path)

    return sorted(set(files))


def file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_cache() -> dict[str, dict[str, int]]:
    if not CACHE_PATH.exists():
        return {}

    try:
        value = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(value, dict):
        return {}

    files = value.get("files", {})
    return files if isinstance(files, dict) else {}


def save_cache(files: Iterable[Path]) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_utc": utc_now(),
        "files": {
            path.relative_to(ROOT).as_posix(): file_signature(path)
            for path in files
        },
    }

    CACHE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def changed_python_files(
    files: list[Path],
    cache: dict[str, dict[str, int]],
) -> list[Path]:
    changed: list[Path] = []

    for path in files:
        key = path.relative_to(ROOT).as_posix()
        if cache.get(key) != file_signature(path):
            changed.append(path)

    return changed


def compile_file(path: Path) -> None:
    import py_compile

    py_compile.compile(
        str(path),
        doraise=True,
    )


def compile_quick(files: list[Path]) -> int:
    cache = load_cache()
    targets = changed_python_files(files, cache)

    if not targets:
        print("PASS: compile cache hit — 변경된 Python 파일 없음")
        return 0

    for path in targets:
        compile_file(path)

    print(f"PASS: quick compile — {len(targets)} files")
    return len(targets)


def compile_full() -> int:
    compiled_count = 0

    for directory in PYTHON_DIRS:
        if not directory.exists():
            continue

        success = compileall.compile_dir(
            str(directory),
            force=True,
            quiet=1,
        )
        if not success:
            raise RuntimeError(
                f"compileall failed: {directory.relative_to(ROOT)}"
            )

        compiled_count += sum(
            1
            for path in directory.rglob("*.py")
            if path.is_file() and not is_excluded(path)
        )

    print(f"PASS: full compile — {compiled_count} files")
    return compiled_count


def run_smoke_imports() -> int:
    imported = 0

    for module_name in SMOKE_MODULES:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                print(f"SKIP: optional smoke module unavailable — {module_name}")
                continue
            raise

        imported += 1
        print(f"PASS: import {module_name}")

    if imported == 0:
        raise RuntimeError("no LRP smoke modules could be imported")

    return imported


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def run_subprocess(command: list[str]) -> int:
    print("RUN:", " ".join(command))

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
    )
    return completed.returncode


def has_test_files() -> bool:
    for directory in OPTIONAL_TEST_DIRS:
        if not directory.exists():
            continue

        if any(directory.rglob("test_*.py")):
            return True
        if any(directory.rglob("*_test.py")):
            return True

    return False


def run_tests() -> str:
    if not has_test_files():
        print("SKIP: 테스트 파일 없음")
        return "skipped:no_tests"

    pytest_available = (
        importlib.util.find_spec("pytest") is not None
        or command_exists("pytest")
    )

    if pytest_available:
        code = run_subprocess(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--disable-warnings",
                "--maxfail=1",
            ]
        )
        if code != 0:
            raise RuntimeError(f"pytest failed with exit code {code}")

        print("PASS: pytest")
        return "passed:pytest"

    code = run_subprocess(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ]
    )
    if code != 0:
        raise RuntimeError(f"unittest failed with exit code {code}")

    print("PASS: unittest")
    return "passed:unittest"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def archive_candidates() -> list[Path]:
    candidates: list[Path] = []

    for root_name in ARCHIVE_ROOTS:
        root = ROOT / root_name
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if (
                path.is_file()
                and not is_excluded(path)
                and path.suffix not in {".pyc", ".pyo"}
            ):
                candidates.append(path)

    for file_name in ARCHIVE_FILES:
        path = ROOT / file_name
        if path.is_file():
            candidates.append(path)

    return sorted(set(candidates))


def create_manifest(files: list[Path]) -> dict[str, object]:
    entries = []

    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "python_version": sys.version.split()[0],
        "file_count": len(entries),
        "files": entries,
    }

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        "PASS: manifest — "
        f"{MANIFEST_PATH.relative_to(ROOT)} "
        f"({len(entries)} files)"
    )
    return payload


def create_reproducible_zip(files: list[Path]) -> tuple[Path, str]:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    temporary = BUILD_DIR / ".lrp_latest.tmp.zip"
    if temporary.exists():
        temporary.unlink()

    fixed_timestamp = (2020, 1, 1, 0, 0, 0)

    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()

            info = zipfile.ZipInfo(
                filename=relative,
                date_time=fixed_timestamp,
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16

            archive.writestr(info, data)

        manifest_data = MANIFEST_PATH.read_bytes()
        manifest_info = zipfile.ZipInfo(
            filename="build/lrp_build_manifest.json",
            date_time=fixed_timestamp,
        )
        manifest_info.compress_type = zipfile.ZIP_DEFLATED
        manifest_info.external_attr = 0o644 << 16
        archive.writestr(manifest_info, manifest_data)

    temporary.replace(LATEST_ZIP_PATH)
    digest = sha256_file(LATEST_ZIP_PATH)

    print(
        "PASS: archive — "
        f"{LATEST_ZIP_PATH.relative_to(ROOT)}"
    )
    print(f"PASS: archive SHA-256 — {digest}")

    return LATEST_ZIP_PATH, digest


def write_result(result: BuildResult) -> None:
    path = BUILD_DIR / "lrp_build_result.json"
    path.write_text(
        json.dumps(
            asdict(result),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def clean() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    removed = 0
    for path in ROOT.rglob("__pycache__"):
        if path.is_dir() and not is_excluded(path.parent):
            shutil.rmtree(path, ignore_errors=True)
            removed += 1

    print(f"PASS: clean — removed {removed} cache directories")


def build(mode: str) -> BuildResult:
    started = time.perf_counter()
    started_at = utc_now()

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    python_files = discover_python_files()

    if not python_files:
        raise RuntimeError("no Python files found under lrp/ or tools/")

    if mode == "quick":
        compiled = compile_quick(python_files)
        smoke_count = run_smoke_imports()
        tests_status = "skipped:quick_mode"
        manifest_count = 0
        archive_path = None
        archive_sha256 = None

    elif mode == "full":
        compiled = compile_full()
        smoke_count = run_smoke_imports()
        tests_status = run_tests()

        files = archive_candidates()
        manifest = create_manifest(files)
        archive, archive_sha256 = create_reproducible_zip(files)

        manifest_count = int(manifest["file_count"])
        archive_path = str(archive.relative_to(ROOT))

    else:
        raise ValueError(f"unsupported build mode: {mode}")

    save_cache(python_files)

    result = BuildResult(
        mode=mode,
        started_at_utc=started_at,
        elapsed_seconds=round(time.perf_counter() - started, 3),
        compiled_files=compiled,
        smoke_modules=smoke_count,
        tests_status=tests_status,
        manifest_files=manifest_count,
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        success=True,
    )

    write_result(result)

    print("")
    print("BUILD SUCCESS")
    print(f"mode: {result.mode}")
    print(f"elapsed: {result.elapsed_seconds:.3f}s")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LRP fast build and release utility"
    )
    parser.add_argument(
        "mode",
        choices=("quick", "full", "clean"),
        nargs="?",
        default="quick",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.mode == "clean":
            clean()
            return 0

        build(args.mode)
        return 0

    except Exception as exc:
        BUILD_DIR.mkdir(parents=True, exist_ok=True)

        failure = {
            "generated_at_utc": utc_now(),
            "mode": args.mode,
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

        (BUILD_DIR / "lrp_build_failure.json").write_text(
            json.dumps(
                failure,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        print("")
        print("BUILD FAILED")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
