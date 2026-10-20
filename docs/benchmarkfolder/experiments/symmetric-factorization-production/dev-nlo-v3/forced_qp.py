#!/usr/bin/env python3
"""Bounded forced-QP controls of the unchanged fresh NLO executable."""
import sys
sys.dont_write_bytecode = True
import hashlib,json,os,subprocess,time
from pathlib import Path
import run

ROOT, OUTPUT = run.ROOT, run.OUTPUT
REPORT = run.REPORT / 'forced-qp'


def main():
    os.sched_setaffinity(0,{2})
    assert not REPORT.exists(), 'Preserve earlier forced-QP evidence'
    REPORT.mkdir()
    initial=json.loads((run.REPORT/'result.json').read_text())
    assert initial['complete'] and initial['backend']==6
    directory=OUTPUT/'SubProcesses/P0_uux_zg'
    executable=directory/'check_poles'
    digest=run.sha(executable)
    assert digest==initial['controls']['P0_uux_zg']['executables']['check_poles']
    originals={p.resolve():p.read_bytes() for p in [OUTPUT/'Cards/MadLoopParams.dat',*(OUTPUT/'SubProcesses').rglob('MadLoopParams.dat')]}
    before=run.nlo_source_manifest('after-build')
    result=dict(complete=False,acceptance_eligible=False,cpu=2,executable_sha256=digest,
                executable=str(executable.relative_to(ROOT)),source_manifest_sha256=before['files_sha256'],
                subprocess_timeout_seconds=300,scope='Unmodified actual NLO executable; forced QP CTModeRun4 '
                'with QP integrands for both backends. The analytic pole tolerance is the existing1e-5 '
                'FKS control, not the full benchmark coefficient gate. Factor call fast/fallback counts '
                'are not instrumented or claimed.',runs={})
    try:
        for backend in (6,8):
            card=run.MadLoopParam(str(OUTPUT/'Cards/MadLoopParams.dat'))
            for key,value in dict(MLReductionLib=str(backend),CTModeRun=4,MLStabThres=1e-9,
                                  COLLIERRequiredAccuracy=1e-10,UseQPIntegrandForNinja=True,
                                  UseQPIntegrandForNinjaRS=True).items(): card[key]=value
            for path in originals:
                card.write(str(path),template=str(ROOT/'Template/loop_material/StandAlone/Cards/MadLoopParams.dat'))
            retained=REPORT/f'MadLoopParams-{backend}.dat'
            retained.write_bytes((OUTPUT/'Cards/MadLoopParams.dat').read_bytes())
            (REPORT/f'input-{backend}.txt').write_text('20\n-1\n')
            start=time.monotonic()
            with (REPORT/f'backend-{backend}.log').open('w') as stream:
                child=subprocess.run([str(executable)],cwd=directory,input='20\n-1\n',text=True,
                     stdout=stream,stderr=subprocess.STDOUT,env=run.environment(),timeout=300)
            log=(REPORT/f'backend-{backend}.log').read_text()
            entry=dict(returncode=child.returncode,elapsed_seconds_diagnostic=time.monotonic()-start,
                 log_sha256=run.sha(REPORT/f'backend-{backend}.log'),card_sha256=run.sha(retained))
            result['runs'][str(backend)]=entry
            assert child.returncode==0
            import re
            assert re.search(r'>\s*MLReductionLib\s*=\s*'+str(backend)+r'\b',log)
            assert re.search(r'>\s*CTModeRun\s*=\s*4\b',log)
            assert re.search(r'NUMBER OF POINTS PASSING THE CHECK\s+20\b',log)
            assert re.search(r'NUMBER OF POINTS FAILING THE CHECK\s+0\b',log)
            assert run.sha(executable)==digest
            assert run.nlo_source_manifest('after-build')==before
            (REPORT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print('Forced QP backend',backend,'20/20 analytic pole controls passed',flush=True)
        result['complete']=True
    except Exception as error:
        result['error']=repr(error)
        raise
    finally:
        for path,content in originals.items(): path.write_bytes(content)
        result['original_cards_restored']=all(path.read_bytes()==content for path,content in originals.items())
        assert result['original_cards_restored']
        result['source_and_executable_unchanged']=(run.nlo_source_manifest('after-build')==before and run.sha(executable)==digest)
        assert result['source_and_executable_unchanged']
        (REPORT/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
