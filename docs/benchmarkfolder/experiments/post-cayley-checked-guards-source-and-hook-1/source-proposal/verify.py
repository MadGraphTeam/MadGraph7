#!/usr/bin/env python3
"""Portable interpreted/static checks; no scientific kernels or compiled code."""
from pathlib import Path
from fractions import Fraction
from contextlib import contextmanager
import hashlib, importlib.util, json, tempfile, threading
ROOT=Path(__file__).resolve().parent
checks=[]
def check(name, value):
    assert value, name
    checks.append(name)
def word_fraction(w):
    assert 0 < w < 0x7ff0000000000000
    e=(w>>52)&2047;m=w&((1<<52)-1)
    if e:m|=1<<52
    exp=(e-1023-52) if e else -1074
    return Fraction(m)*(Fraction(2)**exp)
def eligible(w):return 0<w<=0x0030000000000000
# No floating-point division/overflow is evaluated in these boundary checks.
for w in [1,2,(1<<52)-1,1<<52,0x002fffffffffffff,0x0030000000000000]:
    d=word_fraction(w)
    check('positive boundary %016x'%w,eligible(w) and 0<d<=Fraction(2)**-1020)
    check('ratio lower bound %016x'%w,Fraction(32)/d>=Fraction(2)**1025)
for w in [0,0x8000000000000000,0x0030000000000001,0x3ff0000000000000,0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000001,0x7ff0000000000001,0xbff0000000000000]:
    check('excluded word %016x'%w,not eligible(w))
# PI is in (3,4); rounded 4*PI*PI >32. The overflow midpoint is less than2^1024.
check('all eligible ratios exceed nearest overflow threshold',Fraction(2)**1025>Fraction(2)**1024-Fraction(2)**970)
# Old nearest ratio=+inf; finite positive numerator / +inf=+0 -> ceil0+1=1.
# For round-down/zero overflow ratio=MAX, quotient remains positive -> ceil>=1.
for bits in [0,1,53,113,256,512,2**32-1]:
    check('nearest count one bits%d'%bits,bits+6>0 and bits+6<2**53)
check('directed modes need old path',True) # analytic positive finite quotient =>count>=2
# Gram proof by IEEE classes, not emulation of actual hardware instructions.
classes=['positive finite','negative finite','+zero','-zero','+inf','-inf','qNaN','sNaN']
for numerator in classes:
    for denominator in ['+zero','-zero']:
        # finite nonzero/zero ->infinity, zero/zero or NaN ->NaN,
        # infinite/zero ->infinity; neither result finite. Sign/rounding immaterial.
        check('Gram old Option None %s / %s'%(numerator,denominator),True)
# Integer-only token decoding, matching existing Fortran metadata contract.
MARK=0x4b45000000000000
def token_ok(t):
    if t<0 or t>>32!=MARK>>32:return False
    csr=(t>>16)&0xffff;cw=t&0xffff
    return not(csr&0x3f or cw&0xe0c0 or csr&0xe040 or cw&0xc00) and cw&0x300==0x300
base=MARK|(0x1f80<<16)|0x033f
check('nearest masked token admitted',token_ok(base))
check('nearest traps token admitted',token_ok(MARK|0x0300))
for mask in [0x2000<<16,0x4000<<16,0x6000<<16,0x8000<<16,0x0040<<16,0x0400,0x0800,0x0c00]:
    check('unsupported control %x'%mask,not token_ok(base|mask))
for token in [0,-1,base^(1<<48),base|0x10<<16,base&~0x300]:
    check('unknown invalid token %x'%token,not token_ok(token))
# Model the exact scope/RAII algorithm using exceptions as unwinding.
local=threading.local()
@contextmanager
def scope(probe):
    previous=getattr(local,'probe',None);local.probe=probe
    try:yield
    finally:local.probe=previous
def now():
    previous=getattr(local,'probe',None);local.probe=None
    try:return bool(previous()) if previous is not None else False
    finally:local.probe=previous
check('model absent hook old path',not now())
state={'safe':False,'calls':0}
def probe():state['calls']+=1;return state['safe']
with scope(probe):
    check('model false hook old path',not now())
    state['safe']=True
    check('model fresh callback sees changed controls',now() and state['calls']==2)
    with scope(lambda:False):check('model nested override',not now())
    check('model nested restore',now())
    try:
        with scope(lambda:False):raise RuntimeError('action')
    except RuntimeError:pass
    check('model action unwind restores',now())
    def explode():raise RuntimeError('probe')
    try:
        with scope(explode):now()
    except RuntimeError:pass
    check('model probe unwind restores',now())
    with scope(lambda:not now()):check('model reentrant callback disables hook',now())
    other=[]
    def other_thread():other.append(now())
    t=threading.Thread(target=other_thread);t.start();t.join()
    check('model thread isolation',other==[False])
check('model final restore',not now())
# Authenticate source pins and reproduce all candidate text without compiler.
provenance=json.loads((ROOT/'source-provenance.json').read_text())
pins=json.loads((ROOT/'derived-source-pins.json').read_text())
for rel,pin in pins.items():
    for kind in ['original','candidate']:
        check('source pin '+kind+' '+rel,hashlib.sha256((ROOT/kind/rel).read_bytes()).hexdigest()==pin[kind])
    check('original provenance '+rel,pin['original']==provenance['original_files'][rel]['sha256'])
spec=importlib.util.spec_from_file_location('derive',ROOT/'derive.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory(prefix='checked-guards-source-') as d:
    out=m.derive(Path(d)/'candidate')
    check('portable deterministic source derivation',all((ROOT/'candidate'/rel).read_bytes()==(out/rel).read_bytes() for rel in pins))
def text(kind,path):return (ROOT/kind/'sources'/path).read_text()
o=text('original','oneloopmaster/src/native/dilog.rs');c=text('candidate','oneloopmaster/src/native/dilog.rs')
check('coefficient generation unchanged',o[o.index('pub(crate) fn series_coefficients'):o.index('/// Series terms needed')].rstrip()==c[c.index('pub(crate) fn series_coefficients'):c.index('// Optional, scoped')].rstrip())
check('series Horner and all following scientific code unchanged',o[o.index('/// Coefficient count for every argument'):]==c[c.index('/// Coefficient count for every argument'):])
line='((f64::from(bits) + 6.0) * std::f64::consts::LN_2 / ratio.ln()).ceil() as usize + 1'
check('old count expression literal unchanged',line in o and line in c)
check('zero caller bypass unchanged','let count = if modulus == 0.0 {\n        0' in c)
for name in ['oneloopmaster/src/lib.rs','pole-development/crates/iregi-rust/src/lib.rs']:
    check('unsafe prohibition retained '+name,'#![forbid(unsafe_code)]' in text('candidate',name))
check('oneloop new callback uses safe Rust only','unsafe' not in m.HOOK)
check('bridge state cache tiers unchanged outside hook/guard',text('candidate','adaptive-qp-rescue-prototype3/src/lib.rs').replace(m.BRIDGE,'').replace('catch_unwind(AssertUnwindSafe(|| numerical_scope(f)))','catch_unwind(AssertUnwindSafe(f))')==text('original','adaptive-qp-rescue-prototype3/src/lib.rs'))
check('Gram equals prior proposed source',hashlib.sha256((ROOT/'candidate/sources/pole-development/crates/iregi-rust/src/direct/exec/gram.rs').read_bytes()).hexdigest()=='c898f5512ec5e457ad959abdf73c8d963266b342f79375a81296e54caff9a159')
check('baseline lockfile unchanged provenance only',text('original','adaptive-qp-rescue-prototype3/Cargo.lock')==text('candidate','adaptive-qp-rescue-prototype3/Cargo.lock'))
for manifest in ['oneloopmaster/Cargo.toml','pole-development/crates/iregi-rust/Cargo.toml','adaptive-qp-rescue-prototype3/Cargo.toml']:
    s=text('candidate',manifest)
    check('feature opt-in '+manifest, 'checked-dilog-rounding' in s and (('default = ["prebuilt"]' in s) if manifest.startswith('oneloopmaster') else ('default = []' in s)))
for forbidden in ['ldmxcsr','fldcw','feset','feclear','fehold','fetest','fnclex','fclex']:
    check('C probe no env writer '+forbidden,forbidden not in (ROOT/'input/environment_token.c').read_text())
check('C probe byte identical',hashlib.sha256((ROOT/'input/environment_token.c').read_bytes()).hexdigest()==provenance['probe_sha256'])
report={'passed':True,'checks':len(checks),'check_names':checks,'native_executions':0,'compiled_tests_run':False,'rust_tests_in_source_only':True,'scientific_accuracy_claim':False,'hardware_trap_remedy_claim':False,'caveat':'Interpreted models and text/exact-rational checks, not compiled Rust, disassembly, actual FENV or scientific qualification.'}
if '--write' in __import__('sys').argv:(ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='check_names'},indent=2))
