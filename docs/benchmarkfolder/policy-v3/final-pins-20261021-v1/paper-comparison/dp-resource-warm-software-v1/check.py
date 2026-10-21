"""Software/actual-source checks only; no compilers or numerical engines."""
from pathlib import Path
import hashlib
import os
import runpy
import sys
import unittest

ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
if (HERE/"software-final.json").exists():raise ValueError("Preserve previous software receipt")
paths=[HERE/"ninjars_dp_stability.py",HERE/"test_ninjars_dp_stability.py",
       ROOT/"tools/ninjars_dp_stability.py",ROOT/"tests/unit_tests/loop/test_ninjars_dp_stability.py",
       ROOT/"tools/ninjars_dp_stability_driver.f"]
def hashes():return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
before=hashes();tests=HERE/"test_ninjars_dp_stability.py"
code=tests.read_text().replace('ROOT = Path(__file__).resolve().parents[3]',f'ROOT = Path({str(ROOT)!r})')
code=code.replace('ROOT/"tools/ninjars_dp_stability.py"',repr(str(HERE/"ninjars_dp_stability.py")))
namespace={"__file__":str(tests),"__name__":"private_dp_tests"}
exec(compile(code,str(tests),"exec"),namespace);namespace["dp"].ROOT=ROOT
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(value) for value in namespace.values()
    if isinstance(value,type) and issubclass(value,unittest.TestCase))
with (HERE/"unittest-final.log").open("x") as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
if not result.wasSuccessful():raise ValueError("Software tests failed")
runpy.run_path(str(HERE/"check_renderer.py"),run_name="__source_preflight__")
after=hashes()
if before!=after:raise ValueError("Source changed during checks")
dp=namespace["dp"]
dp.dump(HERE/"software-final.json",{"schema":"dp-random-custody-warm-software-check-v1",
    "source_before":before,"source_after":after,"software_tests":result.testsRun,
    "failures":len(result.failures),"errors":len(result.errors),"python":sys.version,
    "executable":sys.executable,"cpu":sorted(os.sched_getaffinity(0)),"pid":os.getpid(),
    "renderer_preflight":{"path":str((HERE/"renderer-preflight-final.json").relative_to(ROOT)),"sha256":dp.digest(HERE/"renderer-preflight-final.json")},
    "amplitude_evaluations":0,"compilations":0,"physical_random_campaign_complete":False})
