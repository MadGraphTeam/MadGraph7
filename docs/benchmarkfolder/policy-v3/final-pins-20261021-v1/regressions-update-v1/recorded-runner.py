"""Record the affected final --update software tests; no real engines."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT=Path.cwd().resolve()
OUT=ROOT/'.benchmark-work/final-dependency-update-regression-20261021-v1'
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
MODULES=(
    'tests.unit_tests.loop.test_ninjars_fortran',
    'tests.unit_tests.various.test_ninjars_install',
    'tests.unit_tests.various.test_ninjars_statistics',
    'tests.unit_tests.various.test_ninjars_integration_boundaries',
    'tests.unit_tests.various.test_ninjars_upstream',
)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def files():
    paths=subprocess.check_output(['git','ls-files','-z','madgraph','aloha',
        'Template','vendor/ninjars','flake.nix','flake.lock'],text=True).split('\0')
    paths.extend(name.replace('.','/')+'.py' for name in MODULES)
    paths.extend(['tests/__init__.py','tests/unit_tests/__init__.py',
        'tests/unit_tests/loop/__init__.py','tests/unit_tests/various/__init__.py',
        'vendor/ninjars/src/upstream_master_tests.rs',
        str(Path(__file__).relative_to(ROOT))])
    return {p:sha(ROOT/p) for p in sorted(set(paths)) if p and (ROOT/p).is_file()}

def ids(suite):
    for test in suite:
        if isinstance(test,unittest.TestSuite):
            yield from ids(test)
        else:
            yield test.id()

def main():
    result_path=OUT/'focused-result.json'
    assert not result_path.exists()
    before=files()
    start=time.monotonic()
    utc=datetime.now(timezone.utc).isoformat()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    suite=unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    names=list(ids(suite))
    assert names and len(names)==len(set(names))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    after=files()
    changes=sorted(p for p in before.keys()|after.keys() if before.get(p)!=after.get(p))
    record={'schema':'ninjars-update-focused-software-regression-v1',
        'source_revision':revision,'source_files_before':before,'source_files_after':after,
        'source_unchanged':not changes,'changed_source_files':changes,
        'modules':list(MODULES),'test_ids':names,'tests':result.testsRun,
        'failures':[(t.id(),message) for t,message in result.failures],
        'errors':[(t.id(),message) for t,message in result.errors],
        'skipped':[(t.id(),message) for t,message in result.skipped],
        'passed':result.wasSuccessful() and not result.skipped and not changes,
        'actual_reduction_engine_evaluations':0,
        'started_utc':utc,'elapsed_seconds':time.monotonic()-start,
        'affinity':sorted(os.sched_getaffinity(0)),'python':sys.version,
        'gfortran':subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],
        'dev_env_sha256':sha(ROOT/'.ninjars-build/dev-env.sh'),
        'command':sys.argv}
    assert result.testsRun==len(names)
    with result_path.open('x') as f:
        json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:record[k] for k in ('tests','passed','source_unchanged',
        'changed_source_files','elapsed_seconds','actual_reduction_engine_evaluations')},sort_keys=True))
    return 0 if record['passed'] else 1

if __name__=='__main__':
    raise SystemExit(main())
