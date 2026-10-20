#!/usr/bin/env python3
"""Bounded NLO forced-QP control with diagnostic-driver-only policy override."""
import sys
sys.dont_write_bytecode = True
import difflib,json,os,re,shutil,subprocess,time
from pathlib import Path
import run
ROOT=run.ROOT
PARENT=run.OUTPUT
REPORT=run.REPORT/'forced-qp-driver-v2'
OUTPUT=run.WORK/'forced-qp-driver-v2'


def main():
    os.sched_setaffinity(0,{2})
    assert not REPORT.exists() and not OUTPUT.exists()
    REPORT.mkdir()
    original_sources=run.nlo_source_manifest('after-build')
    shutil.copytree(PARENT,OUTPUT,symlinks=True)
    run.OUTPUT=OUTPUT
    before=run.nlo_source_manifest('after-build')
    assert before['files']==original_sources['files']
    driver=OUTPUT/'SubProcesses/check_poles.f'
    assert driver.resolve().is_relative_to(OUTPUT)
    old=driver.read_text()
    new=old.replace('      integer npointsChecked\n','      integer npointsChecked\n      integer factor_attempts\n')
    assert new!=old
    new=new.replace('      npointsChecked = 0\n','      npointsChecked = 0\n      factor_attempts = 0\n')
    needle='          call binothlha_frame(p_born, born, virt_wgt)'
    replacement='''          factor_attempts = factor_attempts + 1
          if (factor_attempts.gt.100) stop 97
          call force_stability_check(.false.)
          call binothlha_frame(p_born, born, virt_wgt)
          write(*,*) 'FACTOR_DIAGNOSTIC_RETURN',ret_code_ml'''
    assert new.count(needle)==1
    new=new.replace(needle,replacement)
    driver.write_text(new)
    (REPORT/'check_poles.original.f').write_text(old)
    (REPORT/'check_poles.diagnostic.f').write_text(new)
    (REPORT/'driver.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original/check_poles.f',tofile='diagnostic/check_poles.f')))
    after=run.nlo_source_manifest('after-build')
    changed={name for name in before['files'] if before['files'][name]!=after['files'][name]}
    assert changed=={'SubProcesses/check_poles.f','SubProcesses/P0_uux_zg/check_poles.f'},changed
    result=dict(complete=False,acceptance_eligible=False,cpu=2,scope='Diagnostic driver only; force unchecked QP and disable the wrapper stability retry before each call, avoiding its perpetual initialization label for H1. First wrapper entry may override this once. No stability acceptance claim; all numerical model/reduction/polynomial code unchanged. Built-in analytic pole tolerance1e-5.',changed_sources=sorted(changed),source_sha256=after['files_sha256'],maximum_attempts=100,timeout_per_backend_seconds=300,runs={})
    directory=OUTPUT/'SubProcesses/P0_uux_zg'
    try:
        with (REPORT/'compile.log').open('w') as stream:
            child=subprocess.run(['make','-C',str(directory),'check_poles','SHELL='+shutil.which('bash')],cwd=ROOT,env=run.environment(),stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        assert child.returncode==0
        result['executable_sha256']=run.sha(directory/'check_poles')
        for backend in (6,8):
            card=run.MadLoopParam(str(OUTPUT/'Cards/MadLoopParams.dat'))
            for key,value in dict(MLReductionLib=str(backend),CTModeRun=4,MLStabThres=1e-9,COLLIERRequiredAccuracy=1e-10,UseQPIntegrandForNinja=True,UseQPIntegrandForNinjaRS=True).items():card[key]=value
            for path in {p.resolve() for p in [OUTPUT/'Cards/MadLoopParams.dat',*(OUTPUT/'SubProcesses').rglob('MadLoopParams.dat')]}:
                assert path.is_relative_to(OUTPUT)
                card.write(str(path),template=str(ROOT/'Template/loop_material/StandAlone/Cards/MadLoopParams.dat'))
            (REPORT/f'MadLoopParams-{backend}.dat').write_bytes((OUTPUT/'Cards/MadLoopParams.dat').read_bytes())
            (REPORT/f'input-{backend}.txt').write_text('20\n-1\n')
            start=time.monotonic()
            with (REPORT/f'backend-{backend}.log').open('w') as stream:
                child=subprocess.run([str(directory/'check_poles')],cwd=directory,env=run.environment(),input='20\n-1\n',text=True,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
            text=(REPORT/f'backend-{backend}.log').read_text()
            codes=list(map(int,re.findall(r'FACTOR_DIAGNOSTIC_RETURN\s+(\d+)',text)))
            counted=[c for c in codes if c//10%10 not in (3,4)]
            result['runs'][str(backend)]={'returncode':child.returncode,'elapsed_seconds_diagnostic':time.monotonic()-start,'return_codes':codes,'counted_qp_return_codes':counted,'log_sha256':run.sha(REPORT/f'backend-{backend}.log')}
            assert child.returncode==0
            assert re.search(r'>\s*MLReductionLib\s*=\s*'+str(backend)+r'\b',text)
            assert re.search(r'>\s*CTModeRun\s*=\s*4\b',text)
            assert len(counted)==20 and all(c//10%10==2 and c//100==1 for c in counted),codes
            assert re.search(r'NUMBER OF POINTS PASSING THE CHECK\s+20\b',text)
            assert re.search(r'NUMBER OF POINTS FAILING THE CHECK\s+0\b',text)
            assert run.nlo_source_manifest('after-build')==after
            (REPORT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print('Forced QP driver control',backend,'passed20 poles; statuses',counted,flush=True)
        result['complete']=True
    except Exception as error:
        result['error']=repr(error)
        raise
    finally:
        run.OUTPUT=PARENT
        result['parent_source_unchanged']=run.nlo_source_manifest('after-build')==original_sources
        assert result['parent_source_unchanged']
        (REPORT/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
