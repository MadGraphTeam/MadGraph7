#!/usr/bin/env python3
"""Bounded real GNU make fixture; no compiler, loader or amplitude invocation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import time

ROOT=Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE=Path(__file__).resolve().parent
SOURCE=HERE.with_name('physical-stress-current-supervisor-1')
WORK=ROOT/'.tirust-build/stress-make-fixture-1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as stream:json.dump(v,stream,indent=2,sort_keys=True);stream.write('\n')
def main():
    assert Path.cwd()==ROOT and not WORK.exists()
    spec=importlib.util.spec_from_file_location('stress_real_make_fixture',SOURCE/'run.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);r.seal()
    WORK.mkdir();(WORK/'lib').mkdir();(WORK/'include').mkdir()
    (WORK/'lib/libfixture.a').write_text('synthetic archive-name data; never passed to a linker\n')
    (WORK/'include/fixture.mod').write_text('synthetic module-name data; never passed to a compiler\n')
    base=("FC = fixture-compiler-MUST-NOT-EXECUTE\nFFLAGS = -O2 -fno-fast-math\n"
          "LDFLAGS = -Wl,--as-needed\nLINKLIBS = -L{w}/lib -lfixture $(LDFLAGS)\n"
          "LOOP_INCLUDE = -I{w}/include\nPROCESS = tensor.o denominator.o\n"
          "check_sa.o tensor.o denominator.o:\n\t@printf 'synthetic make fixture %s\\n' '$@' > $@\n").format(w=WORK)
    (WORK/'makefile').write_text(base);(WORK/'stress.mk').write_text(r.MAKE_ADDENDUM)
    (HERE/'fixture-makefile.txt').write_text(base);(HERE/'emitted-stress.mk').write_text(r.MAKE_ADDENDUM)
    r.L.REPORT=HERE;r.L.CURRENT_STAGE='make-fixture';r.L.CPU='12-15'
    with r.L.cleanup.SignalRequests() as requests:
        r.L.REQUESTS=requests
        with r.L.Deadline(30):
            info=r.L.command(['make','-s','--no-print-directory','-f','stress.mk','stress-info'],'stress-info',WORK,10)
            names=r.L.command(['make','-s','--no-print-directory','-f','stress.mk','stress-list'],'stress-list',WORK,10).split()
            r.L.command(['make','--always-make','-j4','-f','stress.mk','stress-objects'],'stress-objects',WORK,10)
    fields={line.split()[0]:shlex.split(line)[1:] for line in info.splitlines() if line.strip()}
    assert fields=={'FC':['fixture-compiler-MUST-NOT-EXECUTE'],'FFLAGS':['-O2','-fno-fast-math'],'LDFLAGS':['-Wl,--as-needed'],'LINKLIBS':['-L'+str(WORK/'lib'),'-lfixture','-Wl,--as-needed'],'LOOP_INCLUDE':['-I'+str(WORK/'include')]}
    assert names==['check_sa.o','tensor.o','denominator.o']
    for name in names:assert (WORK/name).read_text()=='synthetic make fixture '+name+'\n'
    # Exercise q.external_inputs parsing using the actual make output, with only
    # compiler queries mocked. No fake compiler executable exists or is run.
    queries=[]
    def capture(args,cwd=None,timeout=None):
        assert args[0]=='fixture-compiler-MUST-NOT-EXECUTE' and args[1].startswith('-print-file-name=')
        queries.append(list(args));name=args[1].split('=',1)[1]
        return str(WORK/'lib'/name) if name=='libfixture.a' else name
    r.q.capture=capture
    bound=r.q.external_inputs(WORK,info,fields['FC'],time.monotonic()+10)
    expected={str((WORK/name).resolve()):sha(WORK/name) for name in ('lib/libfixture.a','include/fixture.mod')}
    assert bound==expected and len(queries)>0
    for target in ('stress-info','stress-list','stress-objects'):
        record=json.loads((HERE/'execution'/(target+'.json')).read_bytes())
        assert record['returncode']==0 and record['cleanup_exception'] is None and not record['cleanup']['survivors']
    summary={'passed':True,'source_index_sha256':sha(SOURCE/'source-index.json'),'source_run_sha256':sha(SOURCE/'run.py'),'make_info':info,'make_object_names':names,'external_inputs':bound,'mocked_compiler_queries':queries,'actual_make_commands':3,'compiler_loader_amplitude_commands':0,'owned_cleanup_all_passed':True,'source_unchanged':True}
    r.seal();save(HERE/'result.json',summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
