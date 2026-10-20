#!/usr/bin/env python3
"""Held internal-copy export control; explicit prepare/run phases, no reinstall."""
import sys
sys.dont_write_bytecode = True
import argparse,hashlib,itertools,json,math,os,re,shutil,subprocess,time
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
REPORT=Path(__file__).resolve().parent
WORK=ROOT/'.benchmark-work/policy-v2/install-export-abi3-tensor-factor-v1'
PRIVATE=WORK/'private/HEPTools/ninjars'
HIDDEN=PRIVATE.with_name('ninjars-unavailable')
OUTPUT=WORK/'standalone'
RELOCATED=WORK/'standalone-relocated'
MAIN=ROOT/'HEPTools/ninjars'
REVISION='f4bffc61ed67e5d083b80ff9901768d4d5779989'
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import ninjars_benchmark as bench
from madgraph.various.banner import MadLoopParam
from madgraph.various.ninjars_install import read_config


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(name,data):
    path=REPORT/name
    assert not path.exists(), 'Refuse to overwrite '+str(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def files(root):return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
def environment():
    env=os.environ.copy()
    for key in ('LD_LIBRARY_PATH','LIBRARY_PATH'):env.pop(key,None)
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XDG_CONFIG_HOME=str(WORK/'config'),
               MAKEFLAGS='SHELL='+shutil.which('bash'),TMPDIR=str(ROOT/'.ninjars-build/tmp'),
               SYMBOLICA_LOCK_DIR=str(ROOT/'.ninjars-build/symbolica-locks'))
    return env

def command(args,log,cwd=ROOT,timeout=600):
    path=REPORT/log;assert not path.exists()
    start=time.monotonic()
    with path.open('w') as stream:
        result=subprocess.run(args,cwd=cwd,env=environment(),stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
    record=dict(command=list(map(str,args)),cwd=str(cwd.relative_to(ROOT)),returncode=result.returncode,
                seconds_diagnostic=time.monotonic()-start,log_sha256=sha(path))
    if result.returncode:raise RuntimeError(str(record))
    return record

def logical(path):
    rows=[]
    for line in path.read_text().splitlines():
        if len(line)<7 or line[0] in 'cC*!#':continue
        body=re.sub(r'\s+','',line[6:]).upper()
        if not body:continue
        if line[5] not in ' 0':rows[-1]+=body
        else:rows.append(body)
    return rows

def verify_main():
    expected=read(REPORT/'installation-input.json')
    assert files(MAIN)==expected['files'], 'Main installed prefix changed'
    assert read_config(str(MAIN/'lib'))['abi_version']==3
    return expected

def verify_source():
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert revision==REVISION, (revision,REVISION)
    subprocess.run(['git','diff','--quiet',REVISION,'--','madgraph','aloha','Template','vendor/ninjars'],cwd=ROOT,check=True)
    return bench.generator_tree_state()

def configure(mode):
    card=MadLoopParam(str(RELOCATED/'Cards/MadLoopParams.dat'))
    settings=read(REPORT/'reference-input.json')['shared_numerical_settings']
    for key,value in settings.items():
        if key!='selected_backend_qp_integrand':card[key]=value
    for key,value in dict(MLReductionLib='8',CTModeRun=mode,UseQPIntegrandForNinjaRS=True).items():card[key]=value
    for path in {p.resolve() for p in [RELOCATED/'Cards/MadLoopParams.dat',*(RELOCATED/'SubProcesses').rglob('MadLoopParams.dat')]}:
        assert path.is_relative_to(RELOCATED)
        card.write(str(path),template=str(ROOT/'Template/loop_material/StandAlone/Cards/MadLoopParams.dat'))

def factor_sites():
    total=0;records={}
    pattern=re.compile(r'CALLMP_ML5_0_(?:UPDATE|TERMINAL)_WL_[2-4]_1_FACTOR_FFV1L[12]_[12]\(')
    for p in (RELOCATED/'SubProcesses').rglob('mp_coef_construction_*.f'):
        rows=logical(p);indices=[i for i,r in enumerate(rows) if pattern.match(r)]
        if not indices:continue
        assert 'USE,INTRINSIC::IEEE_EXCEPTIONS' in rows and 'LOGICALWL_ENTRY_INEXACT' in rows
        for i in indices:assert rows[i-1]=='CALLIEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)'
        records[str(p.relative_to(RELOCATED))]=dict(calls=len(indices),sha256=sha(p));total+=len(indices)
    assert total==136,total
    return records

def diagnostic_driver(text):
    assert 'PARAMETER (READPS = .FALSE.)' in text and 'PARAMETER (NPSPOINTS = 4)' in text
    text=text.replace('PARAMETER (READPS = .FALSE.)','PARAMETER (READPS = .TRUE.)').replace('PARAMETER (NPSPOINTS = 4)','PARAMETER (NPSPOINTS = 20)')
    needle='C       CALL ML5_0_FORCE_STABILITY_CHECK(.TRUE.)';assert text.count(needle)==1
    text=text.replace(needle,"      CALL GET_COMMAND_ARGUMENT(1,BUFF(1))\n      IF (TRIM(BUFF(1)).EQ.'stable') THEN\n        CALL ML5_0_FORCE_STABILITY_CHECK(.TRUE.)\n      ENDIF")
    needle='        IF (K.EQ.NPSPOINTS) THEN';assert text.count(needle)==1
    text=text.replace(needle,needle+"\n          DO J=0,NSQUAREDSO_LOOP\n            WRITE(*,'(A,1X,I4,1X,4ES26.17E3)')\n     $       'RELOCATION_ROW',J,(MATELEM(I,J),I=0,3)\n          ENDDO")
    return text

def prepare(cpu):
    assert not WORK.exists() and not (REPORT/'prepared.json').exists()
    os.sched_setaffinity(0,{cpu});state=verify_source();installation=verify_main()
    WORK.mkdir(parents=True)
    bench.REPORT=REPORT
    identity=bench.capture_generation_identity('candidate')
    assert identity['revision']==REVISION
    result=dict(complete=False,acceptance_eligible=False,revision=REVISION,source_identity=identity,cpu=cpu)
    try:
        PRIVATE.parent.mkdir(parents=True)
        shutil.copytree(MAIN,PRIVATE,symlinks=False)
        assert files(PRIVATE)==installation['files']
        lines=['set automatic_html_opening False','set auto_update 0','set nb_core 1',
               'set ninja None','set collier None','set ninjars '+str(PRIVATE/'lib'),
               'set output_dependencies internal','set loop_optimized_output True',
               'set complex_mass_scheme False','import model loop_sm',
               'generate u u~ > z g g [virt=QCD]','output standalone_fortran '+str(OUTPUT),'quit']
        card=REPORT/'standalone.mg5';assert not card.exists();card.write_text('\n'.join(lines)+'\n')
        result['export']=command([sys.executable,str(ROOT/'bin/madgraph'),'--plain',str(card)],'export.log')
        assert verify_source()==state
        assert OUTPUT.exists() and not RELOCATED.exists()
        OUTPUT.rename(RELOCATED)
        assert not HIDDEN.exists();PRIVATE.rename(HIDDEN)
        assert not OUTPUT.exists() and not PRIVATE.exists()
        assert files(HIDDEN)==installation['files']
        copied={}
        for name in ('libninjars.a','mninjars.mod','mninjars.f90','ninjars.h','ninjars-config.json','ninjars-link-before.txt','ninjars-native-libs.txt'):
            p=RELOCATED/'lib'/name
            assert p.is_file() and not p.is_symlink(),p
            sources=[q for q in HIDDEN.rglob(name) if q.is_file()]
            assert len(sources)==1 and sha(p)==sha(sources[0]),name
            copied[name]=dict(sha256=sha(p),bytes=p.stat().st_size,regular_file=True)
        link=RELOCATED/'SubProcesses/MadLoop_makefile_definitions'
        text=link.read_text()
        assert str(PRIVATE) not in text and str(MAIN) not in text and '-lninja ' not in text
        assert text.index('-lgcc_s')<text.index('-lninjars')
        shutil.copy2(link,REPORT/'MadLoop_makefile_definitions')
        result['copied_assets']=copied;result['factored_batches']=factor_sites()
        directory=RELOCATED/'SubProcesses/P0_uux_zgg'
        rows=logical(directory/'loop_matrix.f')
        availability=next(r for r in rows if r.startswith('DATALOOPLIBS_AVAILABLE/'))
        flags=re.findall(r'\.TRUE\.|\.FALSE\.',availability)
        assert len(flags)==8 and flags[5]=='.FALSE.' and flags[7]=='.TRUE.',availability
        result['library_availability']=flags
        original=directory/'check_sa.f';text=original.read_text();(REPORT/'check_sa.generated.f').write_text(text)
        text=diagnostic_driver(text)
        original.write_text(text);(REPORT/'check_sa.diagnostic.f').write_text(text)
        reference=read(REPORT/'reference-input.json')
        assert sha(ROOT/reference['source'])==reference['source_sha256']
        points=ROOT/'docs/benchmarkfolder/points/qcd-3.dat'
        assert sha(points)==reference['points_sha256']
        point_lines=points.read_text().splitlines()
        assert point_lines[0]=='100';point='\n'.join(point_lines[1:6])+'\n'
        (directory/'PS.input').write_text(point);(REPORT/'PS.input').write_text(point)
        reference=read(REPORT/'reference-input.json')
        # Match the benchmark's explicit physical-width policy and canonical card writer.
        from madgraph.models.check_param_card import ParamCard
        param_path=RELOCATED/'Cards/param_card.dat'
        shutil.copy2(param_path,REPORT/'param_card.generated.dat')
        param=ParamCard(str(param_path))
        for decay in param.get('decay',[]):decay.value=0.0
        param.write(str(param_path))
        assert sha(param_path)==reference['parameter_card_sha256']
        shutil.copy2(RELOCATED/'Cards/param_card.dat',REPORT/'param_card.dat')
        configure(-1)
        result['build']=command(['make','-C',str(directory),'-j1','SHELL='+shutil.which('bash'),'check'],'build.log')
        assert not PRIVATE.exists() and verify_source()==state
        raw=subprocess.check_output(['nm','-D',str(directory/'check')],cwd=ROOT,text=True,env=environment())
        (REPORT/'dynamic-symbols.txt').write_text(raw)
        helpers=[r for r in raw.splitlines() if re.search(r'__(?:add|sub|mul|div)tf3(?:@|$)',r)]
        assert helpers and all(re.search(r'\bU __\w+@GCC_',r) for r in helpers)
        result['linkage']=command(['ldd',str(directory/'check')],'ldd.txt')
        linked=(REPORT/'ldd.txt').read_text();assert 'not found' not in linked and 'libgcc_s.so' in linked
        for line in linked.splitlines():assert str(PRIVATE) not in line and str(MAIN) not in line
        result['executable_sha256']=sha(directory/'check')
        result['sources']=bench.generated_source_manifest(RELOCATED)
        result['hidden_installation_sha256']=bench.content_digest(files(HIDDEN))
        result['compiled_source_flags']=sha(RELOCATED/'Source/make_opts')
        shutil.copy2(RELOCATED/'Source/make_opts',REPORT/'make_opts')
        verify_main();result['complete']=True
    except Exception as error:
        result['error']=repr(error);raise
    finally:write('prepared.json',result)

def run(cpu):
    os.sched_setaffinity(0,{cpu})
    assert not (REPORT/'runtime-results.json').exists()
    prepared=read(REPORT/'prepared.json');assert prepared['complete']
    assert not PRIVATE.exists() and not OUTPUT.exists()
    assert bench.content_digest(files(HIDDEN))==prepared['hidden_installation_sha256']
    assert bench.generated_source_manifest(RELOCATED)==prepared['sources']
    directory=RELOCATED/'SubProcesses/P0_uux_zgg'
    assert sha(directory/'check')==prepared['executable_sha256']
    reference=read(REPORT/'reference-input.json')
    assert sha(ROOT/reference['source'])==reference['source_sha256']
    assert sha(RELOCATED/'Cards/param_card.dat')==reference['parameter_card_sha256']
    assert (directory/'PS.input').read_bytes()==(REPORT/'PS.input').read_bytes()
    result=dict(complete=False,acceptance_eligible=False,cpu=cpu,measurements={})
    try:
        for label,mode,expected in (('dp',1,{110}),('qp',4,{120}),('stable',-1,{218,322})):
            configure(mode)
            shutil.copy2(RELOCATED/'Cards/MadLoopParams.dat',REPORT/f'{label}-MadLoopParams.dat')
            for pattern in ('*HelFilter.dat','*LoopFilter.dat'):
                for p in (RELOCATED/'SubProcesses').rglob(pattern):
                    assert p.resolve().is_relative_to(RELOCATED);p.unlink()
            args=[str(directory/'check')]+(['stable'] if label=='stable' else [])
            record=command(args,label+'.log',cwd=directory,timeout=600)
            text=(REPORT/(label+'.log')).read_text()
            codes=re.findall(r'MadLoop return code\s*=\s*(\d+)',text);assert codes and int(codes[-1]) in expected,codes
            rows=[]
            for match in re.finditer(r'(?m)^\s*RELOCATION_ROW\s+(\d+)\s+(.+)$',text):
                values=[float(v.replace('D','E')) for v in match[2].split()]
                assert len(values)==4 and all(math.isfinite(x) for x in values)
                rows.append(dict(order=int(match[1]),coefficients=values))
            assert [r['order'] for r in rows]==[0,1]
            record.update(return_code=int(codes[-1]),rows=rows);result['measurements'][label]=record
        reference=read(REPORT/'reference-input.json')
        compared={label:record['rows'] for label,record in result['measurements'].items()}
        compared['retained_Ninja6']=reference['rows']
        scale=max(abs(v) for rows in compared.values() for row in rows for v in row['coefficients'])
        failures=[]
        for left,right in itertools.combinations(compared,2):
            for a,b in zip(compared[left],compared[right]):
                assert a['order']==b['order']
                for k,(x,y) in enumerate(zip(a['coefficients'],b['coefficients'])):
                    tol=1e-7*max(abs(x),abs(y))+1e-10*scale
                    if abs(x-y)>tol:failures.append(dict(pair=[left,right],order=a['order'],coefficient=k,values=[x,y],tolerance=tol))
        result['failures']=failures;assert not failures,failures
        assert bench.generated_source_manifest(RELOCATED)==prepared['sources']
        assert sha(directory/'check')==prepared['executable_sha256']
        verify_main();result['complete']=True
    except Exception as error:
        result['error']=repr(error);raise
    finally:write('runtime-results.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run']);parser.add_argument('--cpu',type=int,default=2)
    args=parser.parse_args();globals()[args.phase](args.cpu)
