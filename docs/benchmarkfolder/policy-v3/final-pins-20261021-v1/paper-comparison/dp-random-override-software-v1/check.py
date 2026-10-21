"""Bounded source-only validation of the optional random measurement stream."""
import hashlib
import importlib.util
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path.cwd()
WORK = pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"tools"))
source = WORK/"ninjars_dp_stability.py"
tests = WORK/"test_ninjars_dp_stability.py"
hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,tests)}
code = tests.read_text().replace('ROOT = Path(__file__).resolve().parents[3]',f'ROOT = Path({str(ROOT)!r})')
code = code.replace('ROOT/"tools/ninjars_dp_stability.py"',repr(str(source)))
namespace = {"__file__":str(tests),"__name__":"private_dp_tests"}
exec(compile(code,str(tests),"exec"),namespace)
namespace["dp"].ROOT = ROOT
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(value)
        for value in namespace.values() if isinstance(value,type) and issubclass(value,unittest.TestCase))
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise SystemExit(1)
dp = namespace["dp"]
import ninjars_benchmark_policy_v3 as policy_v3
release = policy_v3.validate_cohort(ROOT/"docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/candidate-cohort.json",{})
manifest_path = ROOT/"docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/dp-random-points-v2/manifest.json"
manifest = json.loads(manifest_path.read_text())
config = json.loads((ROOT/manifest["configuration"]["path"]).read_text())
bound_manifest = {"path":str(manifest_path.relative_to(ROOT)),"sha256":dp.digest(manifest_path)}
rows = []
inputs = set()
for cid in config["case_order"]:
    case = config["cases"][cid]
    run = {"case":cid,"N":100,"origin_authority":case["origin_authority"],
           "measurement_authority":{"kind":"random-dp-v1","manifest":bound_manifest,"case":cid}}
    origin = dp.resolve_origin(run,release,inputs)
    measured = dp.resolve_measurement(run,release,origin,inputs)
    rows.append({"case":cid,"generation_case":measured["generation_case"],"points":measured["point_record"],
        "N":run["N"],"seed":measured["seed"],"cuts":measured["cuts"],
        "sample_kind":measured["sample_kind"],"center_of_mass_energy_gev":measured["center_of_mass_energy_gev"]})
after = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,tests)}
if after != hashes:
    raise ValueError("Source changed during source-only validation")
dp.dump(WORK/"validation.json",{"schema":"dp-random-measurement-source-validation-v1", "source_before":hashes,
    "source_after":after,"software_tests":result.testsRun,"failures":len(result.failures),
    "errors":len(result.errors),"cases":rows,"bound_inputs":len(inputs),
    "measurement_manifest":bound_manifest,"amplitude_evaluations":0,"compilations":0})
