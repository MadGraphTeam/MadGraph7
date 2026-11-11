"""Bounded synthetic qualification of the compiled integer-only kernel.

No MadLoop process, tensor library, phase-space fixture, or amplitude is used.
The exact rational oracle checks a sufficient mathematical error enclosure;
this does not qualify source-DAG bounds or an amplitude caller.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'kinematic-invariant-general-proposal-1'
spec=importlib.util.spec_from_file_location('sealed_reference',SOURCE/'certificate_reference.py')
ref=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
ZERO=-1000000

def power(n):
    return F(2**n) if n>=0 else F(1,2**(-n))

def bits_power(exponent,width,negative=False):
    nf,ne,bias=(52,11,1023) if width==64 else (112,15,16383)
    assert 1-bias<=exponent<=bias
    return ((int(negative)<<(width-1)) | ((exponent+bias)<<nf))

def decode(bits,width):
    nf,ne,bias=(52,11,1023) if width==64 else (112,15,16383)
    if bits>>width:
        return None
    ex=(bits>>nf)&((1<<ne)-1)
    if ex==(1<<ne)-1:
        return None
    mantissa=(bits&((1<<nf)-1)) + ((1<<nf) if ex else 0)
    return (-1 if bits>>(width-1) else 1)*mantissa*power((ex or 1)-bias-nf)

def oracle(case):
    width=case['width']
    if width not in (64,128) or case['cls']!=2 or case['mass'] or not case['have'] or not case['env']:
        return False,None
    if any(e!=ZERO and not -100000<=e<=100000 for e in case['errors']):
        return False,None
    values=[decode(b,width) for b in case['bits']]
    if any(v is None for v in values):
        return False,None
    q,raw=values[:4],values[4]
    p,emin=(53,-1022) if width==64 else (113,-16382)
    expected=ref.certificate(q,raw,case['errors'],p,emin)
    exact=ref.exact_error_bound(q,case['errors'],p,emin)
    return expected,(raw,exact)

def main():
    assert not (HERE/'summary.json').exists(), 'Do not overwrite a completed attempt'
    rng=random.Random(20261111)
    cases=[]
    def add(width,q,raw,e=None,**kwargs):
        c=dict(width=width,cls=2,mass=0,have=1,env=1,seed=len(cases)%32,
               errors=[ZERO]*4 if e is None else e,bits=[*q,raw])
        c.update(kwargs)
        c['id']=len(cases)+1
        cases.append(c)
    for width in (64,128):
        nf,ne,bias=(52,11,1023) if width==64 else (112,15,16383)
        one=bits_power(0,width)
        negone=bits_power(0,width,True)
        q=[0,one,0,0]
        # Accepted controls under every combination of sticky IEEE flags.
        for seed in range(32):
            add(width,q,negone,e=[-100]*4,seed=seed)
        for exponent in (-400,-40,-18,0,40,400):
            q=[0,bits_power(exponent,width),0,0]
            raw=bits_power(2*exponent,width,True)
            for error in (ZERO,exponent-130,exponent-60,exponent-30,exponent-3,exponent+3):
                add(width,q,raw,e=[error]*4)
        # Nonfinite/signed-zero/subnormal and maximum finite encodings.
        special=[0,1,(1<<nf)-1,1<<(width-1),((1<<ne)-1)<<nf,
                 (((1<<ne)-1)<<nf)|1,(((1<<ne)-1)<<nf)|(1<<(nf-1)),
                 ((1<<width)-1),((((1<<ne)-2)<<nf)|((1<<nf)-1))]
        for value in special:
            add(width,[value,one,0,0],negone)
            add(width,[0,one,0,0],value)
        for kw in (dict(cls=0),dict(cls=1),dict(cls=3),dict(mass=1),
                   dict(have=0),dict(env=0),dict(errors=[100001]*4),dict(errors=[-100001]*4)):
            add(width,[0,one,0,0],negone,**kw)
        for _ in range(2048):
            exponent=rng.randint(-400,400)
            q=[bits_power(exponent+rng.randint(-3,3),width,bool(rng.getrandbits(1)))
               if rng.randrange(5) else 0 for _ in range(4)]
            raw=bits_power(2*exponent+rng.randint(-12,12),width,bool(rng.randrange(5)))
            errors=[exponent+rng.choice([-160,-120,-65,-60,-50,-30,-4,4]) for _ in range(4)]
            add(width,q,raw,e=errors)
    # Invalid width, DP sign extension, and nonzero upper DP payload must reject.
    add(32,[0,0,0,0],0)
    add(64,[0,bits_power(0,64),0,0],(1<<128)-(1<<64)+bits_power(0,64,True))
    add(64,[0,(1<<64)|bits_power(0,64),0,0],bits_power(0,64,True))
    inputs=[]
    for c in cases:
        words=[b-(1<<128) if b>=(1<<127) else b for b in c['bits']]
        fields=[c['id'],c['width'],c['cls'],c['mass'],c['have'],c['env'],c['seed'],*c['errors'],*words]
        inputs.append(' '.join(map(str,fields)))
    raw=('\n'.join(inputs)+'\n').encode()
    (HERE/'inputs.txt').write_bytes(raw)
    compiler=shutil.which('gfortran')
    assert compiler
    records=[]
    def command(argv,**kwargs):
        start=time.monotonic()
        r=subprocess.run(argv,cwd=HERE,capture_output=True,**kwargs)
        records.append(dict(argv=argv,returncode=r.returncode,elapsed_accountability_seconds=time.monotonic()-start,
                            stdout_sha256=hashlib.sha256(r.stdout).hexdigest(),stderr_sha256=hashlib.sha256(r.stderr).hexdigest()))
        return r
    version=command([compiler,'--version'],timeout=10)
    (HERE/'compiler-version.txt').write_bytes(version.stdout+version.stderr)
    build=command([compiler,'-O2','-Wall','-Wextra','-fcheck=all','-fno-fast-math',
                   'certificate.f90','check.f90','-o','check'],timeout=60)
    (HERE/'build.stdout').write_bytes(build.stdout)
    (HERE/'build.stderr').write_bytes(build.stderr)
    assert build.returncode==0,build.stderr.decode()
    run=command([str(HERE/'check')],input=raw,timeout=30)
    (HERE/'run.stdout').write_bytes(run.stdout)
    (HERE/'run.stderr').write_bytes(run.stderr)
    (HERE/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
    assert run.returncode==0,run.stderr.decode()
    lines=run.stdout.decode().splitlines()
    assert len(lines)==len(cases)
    accepted=0
    for c,line in zip(cases,lines):
        idx,result,flags=line.split()
        assert int(idx)==c['id'] and flags=='T',(c,line)
        expected,enclosure=oracle(c)
        assert (result=='T')==expected,(c,line,expected)
        if result=='T':
            accepted+=1
            rawvalue,bound=enclosure
            assert rawvalue<0 and -rawvalue>bound,(c,line)
    report=dict(scope='compiled integer certificate, synthetic bits only',
                cases=len(cases),accepted=accepted,rejected=len(cases)-accepted,
                exact_enclosure_false_positives=0,reference_mismatches=0,ieee_flag_changes=0,
                widths=[64,128],all_32_flag_combinations=True,
                compiler=compiler,compiler_sha256=hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
                input_sha256=hashlib.sha256(raw).hexdigest(),
                kernel_sha256=hashlib.sha256((HERE/'certificate.f90').read_bytes()).hexdigest(),
                executable_sha256=hashlib.sha256((HERE/'check').read_bytes()).hexdigest(),
                source_dag_bound_factory_qualified=False,amplitude_accuracy_qualified=False,
                tensor_calls=0,performance_measurement=False)
    (HERE/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
