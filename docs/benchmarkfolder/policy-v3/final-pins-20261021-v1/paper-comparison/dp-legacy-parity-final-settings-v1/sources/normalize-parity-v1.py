from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
sys.path.insert(0,str(ROOT/'tools'))
import ninjars_dp_stability as dp
work=Path(sys.argv[1]).resolve()
def bound(p):return {"path":str(p.relative_to(ROOT)),"sha256":dp.digest(p)}
start=json.loads((work/'started.json').read_text())
cohort=json.loads((ROOT/start['cohort']['path']).read_text())
origin=ROOT/'.benchmark-work/policy-v3/final-pins-20261021-v1/generation/outputs/candidate/qcd-1'
points=ROOT/start['points']['path'];runs=[]
for tool in (1,6,7):
 resultpath=work/f'{tool}-observed-result.json';result=json.loads(resultpath.read_text())
 directory=ROOT/result['execution']['cwd'];output=directory.parents[1]
 normalized_execution={'source_record':bound(resultpath),'requested_tool':tool,'helicity':-1,
  'exit_code':result['execution']['exit_code'],'cpu':start['cpu'][0],'thread_environment':start['thread_environment']}
 normalized_build={'source_record':bound(resultpath),'driver_sha256':result['driver_sha256'],
  'instrumented_interface_sha256':result['interface_sha256'],'executable_sha256':result['executable_sha256']}
 dp.dump(work/f'{tool}-normalized-execution.json',normalized_execution)
 dp.dump(work/f'{tool}-normalized-build.json',normalized_build)
 artifacts={'original_interface':bound(origin/'SubProcesses/P0_uux_z/CT_interface.f'),
  'instrumented_interface':bound(directory/'CT_interface.f'),'driver':bound(directory/'check_sa.f'),
  'executable':bound(directory/'check'),'points':bound(points),'parameter_card':bound(output/'Cards/param_card.dat'),
  'madloop_card':bound(output/'Cards/MadLoopParams.dat'),'execution':bound(work/f'{tool}-normalized-execution.json'),
  'build':bound(work/f'{tool}-normalized-build.json'),**{name:result['execution'][name] for name in ('stdout','stderr')}}
 runs.append({'case':'qcd-1','process':'u u~ > z [virt=QCD]','group':'QCD','sample_kind':'recorder-parity-control',
  'N':100,'nso':1,'nexternal':3,'helicity':-1,'tool':tool,'artifacts':artifacts,'raw':result['raw'],
  'qualification':{'start':bound(work/'started.json'),'parity_receipt':bound(work/f'{tool}-parity.json'),
   'plain_result':bound(work/f'{tool}-plain-result.json'),'observed_result':bound(resultpath)}})
manifest={'schema':dp.RAW_SCHEMA,'stream':'forced_dp_self_consistency','rescue_enabled':False,
 'instrumentation':'per-dispatch-observer-v1','cohort':{key:cohort[key] for key in ('numerical_revision','generation_revision','generator_identity_sha256','runtime_identity_sha256','archive_sha256')},
 'cohort_record':start['cohort'],'bindings':[bound(work/'started.json'),bound(work/'run.py'),bound(work/'guard.c')],
 'thresholds':[1e-12,1e-9,1e-6,1e-3],'runs':runs}
dp.dump(work/'qualified-manifest.json',manifest)
try:
 inputs=set();summary=dp.report_snapshot(work/'qualified-manifest.json',expected={'cohort':manifest['cohort']},inputs=inputs)
 dp.dump(work/'qualified-summary.json',summary)
 dp.dump(work/'semantic-validation.json',{'status':'PASS','complete':summary['complete'],'physical_qualified':summary['physical_qualified'],
  'observations_complete':summary['observations_complete'],'tools':{tool:entry['physical_qualified'] for tool,entry in summary['cases']['qcd-1']['tools'].items()},'inputs':{str(p):dp.digest(p) for p in sorted(inputs)}})
 print('PASS', {tool:entry['physical_qualified'] for tool,entry in summary['cases']['qcd-1']['tools'].items()},'allfour_complete',summary['complete'])
except Exception as error:
 dp.dump(work/'semantic-validation.json',{'status':'REJECTED','error_type':type(error).__name__,'error':str(error),'complete':False,'physical_qualified':False})
 print('REJECTED',str(error));raise
