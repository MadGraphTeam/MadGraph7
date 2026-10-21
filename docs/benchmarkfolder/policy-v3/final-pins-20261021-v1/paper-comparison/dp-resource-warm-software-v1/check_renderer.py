"""Actual-source/renderer checks only; no compiler or native evaluations."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import sys

ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"tools"))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
dp=load("private_dp",HERE/"ninjars_dp_stability.py");dp.ROOT=ROOT
helper=ROOT/".benchmark-work/paper-dp-stability-20261021-v1/generic-random-parity-held-v2/checkpoint/checkpoint_driver.py"
if dp.digest(helper)!="e4edb61d75429395f08b22b3cfab0460db76efb5ac74407d1805b6a725e65792":
    raise ValueError("Checkpoint renderer source changed")
checkpoint=load("audited_checkpoint",helper)
base=ROOT/"docs/benchmarkfolder/policy-v3/final-pins-20261021-v1"
release=json.loads((base/"candidate-cohort.json").read_text())
manifest_path=base/"dp-random-points-v2/manifest.json"
manifest=json.loads(manifest_path.read_text());config=json.loads((ROOT/manifest["configuration"]["path"]).read_text())
authority={"path":str(manifest_path.relative_to(ROOT)),"sha256":dp.digest(manifest_path)}
rows=[]
for cid in config["case_order"]:
    case=config["cases"][cid]
    run={"case":cid,"N":100,"origin_authority":case["origin_authority"],
         "measurement_authority":{"kind":"random-dp-v1","manifest":authority,"case":cid}}
    resolved=dp.resolve_origin(run,release)
    source=next(resolved["origin"].glob("SubProcesses/P*_*"))
    _,receipt=dp.instrument_interface((source/"CT_interface.f").read_text())
    prefix=receipt["prefix"];nx=case["definition"]["nfinal"]+2
    nso=int(re.search(r"NSQUAREDSO\s*=\s*(\d+)",(source/"nsquaredSO.inc").read_text(),re.I)[1])
    original=dp.render_driver(prefix,nx,nso)
    augmented,proof=checkpoint.augment_driver(original,prefix)
    if augmented!=dp.render_custody_driver(prefix,nx,nso) or checkpoint.strip_driver(augmented)!=original:
        raise ValueError("Reader/runner custody transformations differ")
    resources=resolved["origin"]/"SubProcesses/MadLoop5_resources"
    table=resources/(prefix+"HelConfigs.dat")
    helicities=[list(map(int,line.split())) for line in table.read_text().splitlines()]
    if any(len(h)!=nx for h in helicities):raise ValueError("Helicity dimensions differ")
    physical=[-1,-1]+[0]*(nx-2) if "sqrvirt=" in case["definition"]["process"] else None
    selection=[i+1 for i,h in enumerate(helicities) if h==physical] if physical else None
    if physical and len(selection)!=1:raise ValueError("Ambiguous loop-induced all-minus tuple")
    prefix_bytes=dp.point_prefix_bytes(resolved["points"],nx,3)
    rows.append({"case":cid,"nexternal":nx,"nso":nso,"prefix":prefix,
        "helicity":selection[0] if physical else -1,"physical_tuple":physical,"table_sha256":dp.digest(table),
        "prefix_points_sha256":hashlib.sha256(prefix_bytes).hexdigest(),"prefix_N":3,"science_N":100,
        "driver_transformation":proof,
        "static_resources":{name:dp.digest(resources/(prefix+name)) for name in ("HelConfigs.dat","ColorNumFactors.dat","ColorDenomFactors.dat")},
        "initial_filters":{name:{"available":(resources/(prefix+name)).exists()} for name in ("HelFilter.dat","LoopFilter.dat")}})
dp.dump(HERE/"renderer-preflight-final.json",{"schema":"dp-random-custody-renderer-source-preflight-v1",
    "producer_sha256":dp.digest(HERE/"ninjars_dp_stability.py"),"tests_sha256":dp.digest(HERE/"test_ninjars_dp_stability.py"),
    "checkpoint_renderer_sha256":dp.digest(helper),"measurement_authority":authority,"cases":rows,
    "amplitude_evaluations":0,"compilations":0})
