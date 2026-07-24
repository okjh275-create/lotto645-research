"""M6-005 RC2 build-runner regression test."""
from pathlib import Path
import sys, tempfile
from tools.build_runner import discover_script_tests, run_script_tests

def write_script(path,code):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(f"print('fixture:{path.name}')\nraise SystemExit({code})\n",encoding='utf-8')
def main():
    with tempfile.TemporaryDirectory() as t:
        root=Path(t); tests=root/'tests'; write_script(tests/'test_02_pass.py',0); write_script(tests/'test_01_pass.py',0); write_script(tests/'helper.py',9)
        assert [p.name for p in discover_script_tests((tests,))]==['test_01_pass.py','test_02_pass.py']
        ok=run_script_tests(project_root=root,test_directories=(tests,),python_executable=sys.executable); assert ok.success and ok.status=='passed:script:2/2'
        write_script(tests/'test_03_fail.py',7); bad=run_script_tests(project_root=root,test_directories=(tests,),python_executable=sys.executable,fail_fast=True)
        assert not bad.success and bad.status=='failed:script:1/3' and bad.results[-1].return_code==7
        print('PASS: M6-005 RC2 build runner')
if __name__=='__main__': main()
