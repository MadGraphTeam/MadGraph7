#!/usr/bin/env python3
"""Prepare a same-input private QP Born diagnostic; no reducer/native calls."""
from pathlib import Path
import hashlib,json,re,difflib,shutil
ROOT=Path.cwd().resolve()
REPORT=ROOT/'docs/benchmarkfolder/experiments/qcd-born-cancellation-audit'
WORK=ROOT/'.tirust-build/qcd-born-cancellation-audit'
PROC=ROOT/'.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg'
SP=PROC/'SubProcesses/P0_uux_zgg'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def name(p):return str(p.relative_to(ROOT))
if WORK.exists():raise SystemExit('Refusing to overwrite diagnostic')
WORK.mkdir()
src=(SP/'born_matrix.f').read_text()
def routine(start,end):return src[src.index('      SUBROUTINE '+start):src.index('      SUBROUTINE '+end)]
parts=[]
originals=[]
for old,nxt,new in [('ML5_0_GET_AMP','ML5_0_GET_JAMP','AUDIT_QP_GET_AMP'),('ML5_0_GET_JAMP','ML5_0_GET_MATRIX','AUDIT_QP_GET_JAMP'),('ML5_0_GET_MATRIX','ML5_0_GET_INTER','AUDIT_QP_GET_MATRIX')]:
 s=routine(old,nxt);originals.append(s)
 t=s.replace(old,new).replace('REAL*8','REAL*16').replace('COMPLEX*16','COMPLEX*32').replace('DOUBLE PRECISION','REAL*16').replace('TYPE(ALOHA)','TYPE(MP_ALOHA)').replace('DCONJG(','CONJG(')
 t=re.sub(r'(?<=[0-9.])D([+-]?[0-9]+)',r'Q\1',t)
 if old=='ML5_0_GET_AMP':
  t=t.replace('(P,NHEL,IC,FLAV_IDX,AMP)','(P,NHEL,IC,FLAV_IDX,AMP,MDL_MZ,GC_4,GC_5,GC_22,GC_23)')
  t=t.replace("      INCLUDE 'coupl.inc'","      REAL*16 MDL_MZ\n      COMPLEX*32 GC_4,GC_5,GC_22,GC_23")
  t=re.sub(r'(?m)^(      CALL )([A-Z0-9_]+)',r'\1MP_\2',t)
 parts.append(t)
qp=''.join(parts)
(REPORT/'qp_born.f').write_text(qp);(WORK/'qp_born.f').write_text(qp)
(REPORT/'derivation.patch').write_text(''.join(difflib.unified_diff(''.join(originals).splitlines(True),qp.splitlines(True),fromfile='original-born-routines.f',tofile='qp_born.f')))
for fn in ['driver.f']:
 shutil.copy2(REPORT/fn,WORK/fn)
points=ROOT/'docs/benchmarkfolder/experiments/physical-stress/points/qcd_n3_uux_zgg/selected.dat'
card=PROC/'Cards/param_card.dat'
shutil.copy2(points,WORK/'points.dat');shutil.copy2(card,WORK/'param_card.dat')
paths=[SP/'born_matrix.f',SP/'born_matrix.o',SP/'loop_matrix.f',SP/'mp_helas_calls_ampb_1.f',SP/'mp_compute_loop_coefs.f',PROC/'lib/libdhelas.a',PROC/'lib/libmodel.a',PROC/'Source/DHELAS/FFV1_2.f',PROC/'Source/DHELAS/MP_FFV1_2.f',PROC/'Source/DHELAS/aloha_functions.f',PROC/'Source/MODEL/coupl.inc',PROC/'Source/MODEL/model_object.mod',PROC/'Source/DHELAS/aloha_object.mod',points,card,REPORT/'prepare.py',REPORT/'driver.f',REPORT/'qp_born.f',REPORT/'derivation.patch']
meta={'schema':1,'scope':'Private Born-only arithmetic diagnostic; no reducers, no phase-space repair or production modification. QP model parameters are exact lifts of current DP values.','input_policy':'Read original DAT into binary64 once then exact REAL(...,KIND=16) promotion before all QP arithmetic; no direct decimal-to-QP reinterpretation.','source_base_sha256':sha(SP/'born_matrix.f'),'source_and_link_inputs':{name(p):sha(p) for p in paths},'derived_routines':['GET_AMP','GET_JAMP','GET_MATRIX'],'aggregation':'All48 helicities, unpolarized, fullsingleSO; originalIDEN72,color matrix(16,-4,16)/3 andbroken-symmetryfactor; ordinary originalDP SMATRIX control retained.','symmetry_diagnostics':['exchange identical finalgluons4and5','cyclic spatialaxispermutation x,y,z -> y,z,x'],'limits':['No native trace yet identifies the first nonfinite wavefunction in the original executable.','Standalone QP is not an automatically checked full-loop amplitude or proof all loop groups are served.']}
(REPORT/'preparation.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
print('Prepared3precision-lifted Bornroutines; source-bound; no execution.')
