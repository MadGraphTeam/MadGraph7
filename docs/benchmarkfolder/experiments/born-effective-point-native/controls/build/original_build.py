#!/usr/bin/env python3
"""Compile/link prepared runtime fixtures only, CPU14, one shared600s limit."""
import hashlib,importlib.util,json,os,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
HELPER=ROOT/'docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/attempt2/control.py'
COMPILER=Path('/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/bin/gfortran')
HELPER_SHA='acf00431793415817c29f26d997ae272da968f4b024482cfb72c60333b61ed50'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while data:=f.read(1024*1024):h.update(data)
 return h.hexdigest()
def dump(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');t.replace(p)
def source_map(p):
 return {str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file() and (f.suffix.lower() in ('.f','.f90','.h','.inc') or f.name in ('makefile','make_opts','makeinc.inc'))}
def include_map(proc):
 out={}
 for rel in ('.','../../Source/MODEL','../../Source/DHELAS'):
  for f in sorted((proc/rel).iterdir()):
   if f.is_file() and f.suffix.lower() in ('.inc','.h'):out[rel+'/'+f.name]=sha(f)
 out['make_opts']=sha(proc/'../../Source/make_opts');out['makefile']=sha(proc/'makefile')
 return out
def snapshot():
 result={}
 for base in [HERE/'exports',HERE/'runs',HERE/'inputs',HERE.parent/'control-stub/sources']:
  for p in sorted(base.rglob('*')):
   if p.is_file():result[str(p.relative_to(ROOT))]=sha(p)
 for base in [ROOT/'HEPTools/ninja/lib',ROOT/'HEPTools/ninja/include',ROOT/'HEPTools/ninjars/lib',ROOT/'HEPTools/collier',ROOT/'HEPTools/include',ROOT/'.tirust-build/qp-transport-install/mg/HEPTools/tirust/lib',ROOT/'vendor/CutTools/includects']:
  for p in sorted(base.rglob('*')):
   if p.is_file() and (p.parent==base or p.suffix.lower() in ('.h','.mod','.smod','.inc')):result[str(p.relative_to(ROOT))]=sha(p)
 return result
def main():
 assert Path.cwd()==ROOT and not (HERE/'build.json').exists()
 assert sha(HELPER)==HELPER_SHA and sha(COMPILER)=='30c6495dcb18cc324629fc230302d9cd8d81143f0ca1635d752b3f8552d48cae'
 spec=importlib.util.spec_from_file_location('runtime_owned_helper',HELPER);owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner);owner.CPU=14
 material=json.loads((HERE/'materialized.json').read_text());runtime=json.loads((HERE/'runtime-prepared.json').read_text())
 assert material['status']=='complete' and runtime['state']=='prepared-no-native'
 pinned=snapshot();dump(HERE/'build-inputs.json',pinned)
 env=os.environ.copy()
 for key in ('LD_PRELOAD','LD_AUDIT','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','GFORTRAN_INCLUDE_PATH','FFLAGS','LDFLAGS','MAKEFLAGS'):assert not env.get(key),key
 env.update(FC=str(COMPILER),MAKEFLAGS='-j1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 report={'state':'running','cpu':14,'max_compiler_jobs':1,'total_budget_seconds':600,'native_executed':False,'commands':[],'reused':[],
  'helper_sha256':HELPER_SHA,'build_inputs_sha256':sha(HERE/'build-inputs.json'),'controller_sha256':sha(Path(__file__)),
  'environment':{k:v for k,v in sorted(env.items()) if k in ('PATH','FC','MAKEFLAGS','FFLAGS','LDFLAGS') or k.startswith(('NIX_','OMP_','OPENBLAS_','LD_'))}}
 dump(HERE/'build.json',report)
 donors=[]
 # Prior exact sources/objects are eligible only after their frozen build bindings.
 prior=ROOT/'.tirust-build/born-effective-build/attempt7';pb=json.loads((prior/'build.json').read_text())
 for name,expected in pb['outputs'].items():assert sha(ROOT/name)==expected,name
 for case in json.loads((prior/'materialized.json').read_text())['cases']:
  donors.append({'export':ROOT/case['export'],'proc':ROOT/case['proc'],'binding':sha(prior/'build.json')})
 start=time.monotonic();deadline=start+600
 def run(argv,cwd,name,log):
  env['PWD']=str(cwd);log.write(json.dumps({'name':name,'argv':argv,'cwd':str(cwd)})+'\n');log.flush()
  event=owner.bounded(argv,cwd,log,deadline,env);event['name']=name;report['commands'].append(event);dump(HERE/'build.json',report)
  assert owner.successful(event),name
 try:
  with owner.signal_cleanup(),(HERE/'build.log').open('x') as log:
   order={'full':0,'graph':1,'mass':2,'original':3}
   for case in sorted(material['cases'],key=lambda c:(['real_uux_z','cms_epem_ttx','ew_epem_mupmum','qcd_uux_zgg'].index(c['case']),order[c['arm']])):
    owner.check_interrupt();out=ROOT/case['export'];proc=ROOT/case['proc'];key=case['case']+'/'+case['arm']
    smap=source_map(out/'Source');inc=include_map(proc)
    for donor in donors:
     if source_map(donor['export']/'Source')!=smap:continue
     copied=[]
     for sub in ('MODEL','DHELAS'):
      for p in sorted((donor['export']/'Source'/sub).iterdir()):
       if p.suffix.lower() in ('.o','.mod','.smod'):
        target=out/'Source'/sub/p.name
        if not target.exists():shutil.copyfile(p,target);copied.append({'from':str(p.relative_to(ROOT)),'to':str(target.relative_to(ROOT)),'sha256':sha(p)})
     for name in ('libmodel.a','libdhelas.a'):
      p=donor['export']/'lib'/name
      if p.exists():shutil.copyfile(p,out/'lib'/name);copied.append({'from':str(p.relative_to(ROOT)),'to':str((out/'lib'/name).relative_to(ROOT)),'sha256':sha(p)})
     report['reused'].append({'case':key,'scope':'complete MODEL+ALOHA source/flags identity','source_map':smap,'files':copied,'donor_binding':donor['binding']})
     if include_map(donor['proc'])==inc:
      reused=[]
      for f in sorted(proc.glob('*.f')):
       old=donor['proc']/f.name;obj=old.with_suffix('.o');target=f.with_suffix('.o')
       if old.exists() and obj.exists() and sha(f)==sha(old) and not target.exists():
        shutil.copyfile(obj,target);reused.append({'source':str(f.relative_to(ROOT)),'source_sha256':sha(f),'from':str(obj.relative_to(ROOT)),'to':str(target.relative_to(ROOT)),'sha256':sha(obj)})
        if f.name=='polynomial.f':
         for m in donor['proc'].glob('*.mod'):shutil.copyfile(m,proc/m.name)
      report['reused'].append({'case':key,'scope':'identical process unit and all local/model/ALOHA textual includes; same makefile/flags','include_map':inc,'files':reused,'donor_binding':donor['binding']})
     break
    make=[shutil.which('make'),'-j1','SHELL='+shutil.which('bash')]
    run(make+['all'],out/'Source',key+'/model-aloha',log)
    run(make+['-f','runtime.mk','runtime_probe'],proc,key+'/runtime-link',log)
    donors.append({'export':out,'proc':proc,'binding':'owned jobs in this report; sources and generated artifacts bound at completion'})
   stub=json.loads((HERE.parent/'control-stub/compiler-jobs.json').read_text())
   for job in stub['jobs']:
    argv=job['compile_argv'];position=argv.index('gfortran');argv=[str(COMPILER)]+argv[position+1:]
    for directory in job['create_directories']:Path(directory).mkdir(parents=True,exist_ok=True)
    run(argv,ROOT,'control-stub/'+job['name'],log)
  report['state']='complete'
 except BaseException as error:report['state']='failed';report['error']={'type':type(error).__name__,'message':str(error)}
 finally:
  report['elapsed_seconds']=time.monotonic()-start
  report['changed_inputs']={name:{'before':expected,'after':sha(ROOT/name)} for name,expected in pinned.items() if sha(ROOT/name)!=expected}
  if report['changed_inputs']:report['state']='failed';report['input_failure']='Bound source/dependency bytes changed'
  report['artifacts']={str(p.relative_to(ROOT)):sha(p) for base in (HERE/'exports',HERE.parent/'control-stub/work') for p in sorted(base.rglob('*')) if p.is_file() and not p.is_symlink() and (p.suffix.lower() in ('.o','.a','.mod','.smod','.exe') or p.name=='runtime_probe')}
  report['build_log_sha256']=sha(HERE/'build.log');dump(HERE/'build.json',report)
 print(json.dumps({'state':report['state'],'seconds':report['elapsed_seconds'],'jobs':len(report['commands']),'error':report.get('error')}))
 return int(report['state']!='complete')
if __name__=='__main__':sys.exit(main())
