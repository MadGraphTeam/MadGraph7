"""Reproduce only the frozen n1 driver and header-only repair instrumentation."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent


def load(name):
    spec=importlib.util.spec_from_file_location('host_repair_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def derive():
    pin=json.loads((HERE/'input-pins.json').read_text())
    for name in ('improve_ps.f','check_sa.f'):
        expected=pin['generated_sources']['SubProcesses/P0_uux_z/'+name]
        if hashlib.sha256((HERE/'original-source'/name).read_bytes()).hexdigest()!=expected:raise ValueError('Actual CLI source binding differs:'+name)
    repair=(HERE/'original-source/improve_ps.f').read_text()
    if hashlib.sha256(repair.encode()).hexdigest()!=load('instrument').SOURCE_SHA256:raise ValueError('Frozen repair source changed')
    driver=load('driver').derive((HERE/'original-source/check_sa.f').read_text())
    baseline={'improve_ps.f':repair,'check_sa.f':driver}
    observed={'improve_ps.f':load('instrument').instrument(repair),'check_sa.f':driver}
    if load('instrument').reconstruct(observed['improve_ps.f'])!=repair:raise ValueError('Original repair body differs')
    return baseline,observed


def verify_materialized():
    baseline,observed=derive()
    for name,files in (('baseline-source',baseline),('observed-source',observed)):
        for filename,text in files.items():
            if (HERE/name/filename).read_text()!=text:raise ValueError('Source derivation changed:'+name+'/'+filename)
    return {'frozen_scientific_body_sha256':load('instrument').SOURCE_SHA256,'original_bodies_reconstructed_exactly':True,
            'public_ABIs_preserved':True,'driver_common':baseline['check_sa.f']==observed['check_sa.f'],
            'numerical_flags_changed':False,'native_calls':0}


if __name__=='__main__':
    baseline,observed=derive()
    for name,files in (('baseline-source',baseline),('observed-source',observed)):
        target=HERE/name;target.mkdir(exist_ok=False)
        for filename,text in files.items():(target/filename).write_text(text)
