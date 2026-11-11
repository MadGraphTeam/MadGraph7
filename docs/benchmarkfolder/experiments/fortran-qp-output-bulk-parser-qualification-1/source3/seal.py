"""Pure test receipt and source seal; refuses an existing final seal."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent


def identity(path):
    b=Path(path).read_bytes()
    return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def dump(path,value):
    with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def forbidden(*args,**kwargs):raise RuntimeError('Subprocess forbidden in source-only receipt')


def main():
    assert not (HERE/'source-index.json').exists()
    spec=importlib.util.spec_from_file_location('bulk_source_receipt_tests',HERE/'test_source.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    suite=unittest.defaultTestLoader.loadTestsFromModule(module)
    spec2=importlib.util.spec_from_file_location('bulk_runtime_receipt_tests',HERE/'test_unit_runtime.py')
    module2=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(module2)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module2))
    stream=io.StringIO()
    with patch.object(subprocess,'Popen',forbidden),patch.object(subprocess,'run',forbidden),patch.object(subprocess,'check_output',forbidden):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    assert result.wasSuccessful(),stream.getvalue()
    dump(HERE/'pure-checks.json',{'schema':'madloop-tirust-bulk-unit-source-tests-v1',
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'subprocess_apis_blocked':True,'compiler_executions':0,'native_executions':0,
        'output':stream.getvalue()})
    files={str(p.relative_to(HERE)):identity(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
    seal={'schema':'madloop-tirust-bulk-unit-controller-source-v1','status':'source-only-uncompiled',
          'files':files,'file_count':len(files),'total_bytes':sum(v['bytes'] for v in files.values()),
          'proposal_index_sha256':'600fb55cae10c8f105998c6ae689c09992526765734f33036dd79936261f5150',
          'compiler_executions':0,'native_executions':0,'production_modified':False,
          'build_cap_seconds':180,'run_cap_seconds':600,'cleanup_reserve_seconds':60,
          'physical_amplitude_calls':0,'actual_tirust_calls':0,'performance_acceptance':False}
    dump(HERE/'source-index.json',seal)
    print(json.dumps({'source_index':identity(HERE/'source-index.json'),'files':len(files),'tests_passed':result.testsRun,
                      'compiler_executions':0,'native_executions':0},sort_keys=True,indent=2))


if __name__=='__main__':main()
