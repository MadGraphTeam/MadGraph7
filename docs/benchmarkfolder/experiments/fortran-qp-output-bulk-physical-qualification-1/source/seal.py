"""Pure blocked-API test receipt and immutable source-only phase proposal seal."""
from contextlib import ExitStack,redirect_stdout,redirect_stderr
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
import pytest

HERE=Path(__file__).resolve().parent


def forbidden(*args,**kwargs):raise RuntimeError('Compiler/native/process API forbidden in source-only qualification')
def identity(path):
    raw=Path(path).read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def dump(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def main():
    assert not (HERE/'source-index.json').exists()
    output=io.StringIO()
    with ExitStack() as stack:
        for name in ('Popen','run','check_output','check_call','call'):
            stack.enter_context(patch.object(subprocess,name,forbidden))
        for name in ('system','fork','posix_spawn','posix_spawnp'):
            if hasattr(os,name):stack.enter_context(patch.object(os,name,forbidden))
        stack.enter_context(patch.object(ctypes,'CDLL',forbidden))
        with redirect_stdout(output),redirect_stderr(output):
            result=pytest.main(['-q','-p','no:cacheprovider','-c','/dev/null',str(HERE)])
    if int(result)!=0:
        failure=HERE.parent/'fortran-qp-output-bulk-physical-source-tests-failure-1.txt'
        with failure.open('x') as stream:stream.write(output.getvalue())
        raise RuntimeError('Source-only tests failed; retained '+str(failure))
    dump(HERE/'pure-checks.json',{'schema':'madloop-tirust-bulk-physical-source-tests-v1',
         'exit_code':0,'output':output.getvalue(),'subprocess_process_native_apis_blocked':True,
         'compiler_executions':0,'native_executions':0,'python':sys.executable})
    files={str(p.relative_to(HERE)):identity(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
    seal={'schema':'madloop-tirust-bulk-physical-source-v1','status':'source-only-uncompiled-unexecuted',
          'files':files,'file_count':len(files),'total_bytes':sum(v['bytes'] for v in files.values()),
          'compiler_executions':0,'native_executions':0,'Rust_builds':0,'candidate_promoted':False,
          'performance_acceptance':False,'caps_seconds':{'build':300,'parity':600,'timing':480},
          'cleanup_reserve_seconds':60,'numerical_cohort':'Unchanged100 saved QCDn3points; exactanchors1DP219 and16autoQP323',
          'timing_scope':'Onlypoints1and16,targetedanchors,separatefromfullprocess/baselinechain/localacceptance'}
    dump(HERE/'source-index.json',seal)
    print(json.dumps({'source_index':identity(HERE/'source-index.json'),'payloads':len(files),
                      'compiler_executions':0,'native_executions':0,'tests_output':output.getvalue()},indent=2,sort_keys=True))


if __name__=='__main__':main()
