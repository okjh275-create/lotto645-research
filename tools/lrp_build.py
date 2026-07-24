"""Fast, reproducible build system for Lotto645 Research Platform."""
from __future__ import annotations
import argparse, compileall, hashlib, importlib, json, shutil, sys, time, zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from build_runner import run_script_tests

ROOT=Path(__file__).resolve().parents[1]; BUILD_DIR=ROOT/'build'; CACHE_PATH=BUILD_DIR/'.lrp_build_cache.json'; MANIFEST_PATH=BUILD_DIR/'lrp_build_manifest.json'; LATEST_ZIP_PATH=BUILD_DIR/'lrp_latest.zip'
PYTHON_DIRS=(ROOT/'lrp',ROOT/'tools'); OPTIONAL_TEST_DIRS=(ROOT/'tests',ROOT/'lrp'/'tests')
SMOKE_MODULES=('lrp','lrp.contracts','lrp.core','lrp.adapters','lrp.pipelines')
EXCLUDED_PARTS={'.git','.venv','venv','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','build','dist'}
ARCHIVE_ROOTS=('lrp','tools','tests'); ARCHIVE_FILES=('pyproject.toml','requirements.txt','requirements-dev.txt','README.md','build.ps1')

@dataclass(slots=True)
class BuildResult:
    mode:str; started_at_utc:str; elapsed_seconds:float; compiled_files:int; smoke_modules:int; tests_status:str; manifest_files:int; archive_path:str|None; archive_sha256:str|None; success:bool

def utc_now(): return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
def is_excluded(path):
    try: rel=path.relative_to(ROOT)
    except ValueError: return True
    return any(p in EXCLUDED_PARTS for p in rel.parts)
def discover_python_files():
    return sorted({p for d in PYTHON_DIRS if d.exists() for p in d.rglob('*.py') if p.is_file() and not is_excluded(p)})
def sig(path):
    s=path.stat(); return {'size':s.st_size,'mtime_ns':s.st_mtime_ns}
def load_cache():
    try:
        data=json.loads(CACHE_PATH.read_text(encoding='utf-8')); return data.get('files',data) if isinstance(data,dict) else {}
    except (OSError,json.JSONDecodeError): return {}
def save_cache(files):
    BUILD_DIR.mkdir(parents=True,exist_ok=True); CACHE_PATH.write_text(json.dumps({'schema_version':1,'generated_at_utc':utc_now(),'files':{p.relative_to(ROOT).as_posix():sig(p) for p in files}},indent=2,sort_keys=True),encoding='utf-8')
def compile_quick(files):
    cache=load_cache(); changed=[p for p in files if cache.get(p.relative_to(ROOT).as_posix())!=sig(p)]
    if not changed: print('PASS: compile cache hit — 변경된 Python 파일 없음'); return 0
    for p in changed:
        if not compileall.compile_file(str(p),force=True,quiet=1): raise RuntimeError(f'compile failed: {p.relative_to(ROOT)}')
    print(f'PASS: quick compile — {len(changed)} files'); return len(changed)
def compile_full():
    count=0
    for d in PYTHON_DIRS:
        if not d.exists(): continue
        if not compileall.compile_dir(str(d),force=True,quiet=1): raise RuntimeError(f'compileall failed: {d.relative_to(ROOT)}')
        count+=sum(1 for p in d.rglob('*.py') if p.is_file() and not is_excluded(p))
    print(f'PASS: full compile — {count} files'); return count
def run_smoke_imports():
    n=0
    for name in SMOKE_MODULES:
        try: importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name==name: print(f'SKIP: optional smoke module unavailable — {name}'); continue
            raise
        n+=1; print(f'PASS: import {name}')
    if n==0: raise RuntimeError('no LRP smoke modules could be imported')
    return n
def run_tests():
    s=run_script_tests(project_root=ROOT,test_directories=OPTIONAL_TEST_DIRS,python_executable=sys.executable,fail_fast=True)
    if s.discovered==0: print('SKIP: 테스트 파일 없음'); return s.status
    if not s.success:
        failed=next((x for x in s.results if not x.passed),None); detail='unknown' if failed is None else f'{failed.path} (exit code {failed.return_code})'
        raise RuntimeError(f'script regression test failed: {detail}')
    print(f'PASS: script regression tests — {s.passed}/{s.discovered}'); return s.status
def sha256_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def archive_candidates():
    out=[]
    for name in ARCHIVE_ROOTS:
        d=ROOT/name
        if d.exists(): out.extend(p for p in d.rglob('*') if p.is_file() and not is_excluded(p) and p.suffix not in {'.pyc','.pyo'})
    out.extend(ROOT/name for name in ARCHIVE_FILES if (ROOT/name).is_file()); return sorted(set(out))
def create_manifest(files):
    entries=[{'path':p.relative_to(ROOT).as_posix(),'size':p.stat().st_size,'sha256':sha256_file(p)} for p in files]
    payload={'schema_version':1,'generated_at_utc':utc_now(),'python_version':sys.version.split()[0],'file_count':len(entries),'files':entries}
    BUILD_DIR.mkdir(parents=True,exist_ok=True); MANIFEST_PATH.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True),encoding='utf-8'); print(f'PASS: manifest — {MANIFEST_PATH.relative_to(ROOT)} ({len(entries)} files)'); return payload
def create_zip(files):
    BUILD_DIR.mkdir(parents=True,exist_ok=True); tmp=BUILD_DIR/'.lrp_latest.tmp.zip'; tmp.unlink(missing_ok=True); stamp=(2020,1,1,0,0,0)
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:
            info=zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),stamp); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o644<<16; z.writestr(info,p.read_bytes())
        info=zipfile.ZipInfo('build/lrp_build_manifest.json',stamp); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o644<<16; z.writestr(info,MANIFEST_PATH.read_bytes())
    tmp.replace(LATEST_ZIP_PATH); digest=sha256_file(LATEST_ZIP_PATH); print(f'PASS: archive — {LATEST_ZIP_PATH.relative_to(ROOT)}\nPASS: archive SHA-256 — {digest}'); return LATEST_ZIP_PATH,digest
def write_result(result): (BUILD_DIR/'lrp_build_result.json').write_text(json.dumps(asdict(result),ensure_ascii=False,indent=2,sort_keys=True),encoding='utf-8')
def clean():
    if BUILD_DIR.exists(): shutil.rmtree(BUILD_DIR)
    removed=0
    for p in ROOT.rglob('__pycache__'):
        if p.is_dir() and not is_excluded(p.parent): shutil.rmtree(p,ignore_errors=True); removed+=1
    print(f'PASS: clean — removed {removed} cache directories')
def build(mode):
    started=time.perf_counter(); started_at=utc_now(); BUILD_DIR.mkdir(parents=True,exist_ok=True); files=discover_python_files()
    if not files: raise RuntimeError('no Python files found under lrp/ or tools/')
    tests='not_run'; mcount=0; apath=None; adigest=None
    if mode=='quick': compiled=compile_quick(files); smoke=run_smoke_imports()
    elif mode=='full':
        compiled=compile_full(); smoke=run_smoke_imports(); tests=run_tests(); afiles=archive_candidates(); manifest=create_manifest(afiles); archive,adigest=create_zip(afiles); mcount=int(manifest['file_count']); apath=str(archive.relative_to(ROOT))
    else: raise ValueError(f'unsupported build mode: {mode}')
    save_cache(files); result=BuildResult(mode,started_at,round(time.perf_counter()-started,3),compiled,smoke,tests,mcount,apath,adigest,True); write_result(result); print(f'\nBUILD SUCCESS\nmode: {mode}\nelapsed: {result.elapsed_seconds:.3f}s'); return result
def parse_args():
    p=argparse.ArgumentParser(description='LRP fast build and release utility'); p.add_argument('mode',choices=('quick','full','clean'),nargs='?',default='quick'); return p.parse_args()
def main():
    args=parse_args()
    try:
        if args.mode=='clean': clean(); return 0
        build(args.mode); return 0
    except Exception as exc:
        BUILD_DIR.mkdir(parents=True,exist_ok=True); (BUILD_DIR/'lrp_build_failure.json').write_text(json.dumps({'generated_at_utc':utc_now(),'mode':args.mode,'success':False,'error_type':type(exc).__name__,'error':str(exc)},ensure_ascii=False,indent=2,sort_keys=True),encoding='utf-8'); print('\nBUILD FAILED'); print(f'{type(exc).__name__}: {exc}',file=sys.stderr); return 1
if __name__=='__main__': raise SystemExit(main())
