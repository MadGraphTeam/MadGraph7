#!/usr/bin/env python3
"""Private compact DP experiment: no production source changes."""
from pathlib import Path
import hashlib,json
from madgraph.various.q_polynomial import FortranPolynomialRoutines,get_number_of_coefs_for_rank
from madgraph.iolibs.file_writers import FortranWriter
ROOT=Path.cwd()
HERE=ROOT/'docs/benchmarkfolder/experiments/dp-higher-rank/kernels'
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b,1)
def fixed(path,s):
    with FortranWriter(str(path)) as w:
        w.downcase=False
        w.writelines(s)
g=FortranPolynomialRoutines(6,updater_max_rank=1,proc_prefix='TEST_',sub_prefix='TEST_')
(HERE/'coef_specs.inc').write_text('      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=5)\n')
(HERE/'loop_max_coefs.inc').write_text('      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=210)\n')
fixed(HERE/'constants.f',g.write_polynomial_constant_module())
prototype=g.write_dp_terminal_wl_updater()
guard=prototype[prototype.index('C Private diagonal'):prototype.index('DO I=1,LCUT_SIZE')]
classifier=prototype[prototype.index('CONTAINS\n'):]
publics=[]
for rank in (2,3):
    public=g.write_compact_wl_updater(rank,1,_sparse=False)
    publics.append(public)
    private=once(public,'TEST_UPDATE_WL_%d_1('%rank,'TEST_DP_TERMINAL_WL_%d_1('%rank)
    private=once(private,'OUT_SIZE,OUT)\n','OUT_SIZE,OUT,WL_ENTRY_INEXACT)\n')
    private=once(private,'  implicit none','  USE, INTRINSIC :: IEEE_ARITHMETIC\n  USE, INTRINSIC :: IEEE_EXCEPTIONS\n  implicit none')
    private=once(private,'  INTEGER I,J,K,L,M','  INTEGER I,J,K,L,M,WL_H\n  LOGICAL WL_ENTRY_INEXACT,WL_HALTING(SIZE(IEEE_ALL))\n  TYPE(IEEE_ROUND_TYPE) WL_ROUNDING')
    active=get_number_of_coefs_for_rank(rank)-1
    rank_guard=guard.replace('A(:,0:4,:)','A(:,0:%d,:)'%active)
    private=once(private,'OUT(:,:,:)='+g.czero,rank_guard+'OUT(:,:,:)='+g.czero)
    private=once(private,'          DO I=1,LCUT_SIZE','          DO I=J,J')
    assert private.endswith('\nEND')
    private=private[:-3]+'RETURN\n900 CONTINUE\nCALL TEST_UPDATE_WL_%d_1(A,LCUT_SIZE,B,IN_SIZE,OUT_SIZE,OUT)\n'%rank+classifier
    (HERE/('private-r%d.txt'%rank)).write_text(private+'\n')
    fixed(HERE/('private-r%d.f'%rank),private)
fixed(HERE/'public.f','\n'.join(publics))
fixed(HERE/'merge.f',g.write_wl_merger())
sources=[Path('madgraph/various/q_polynomial.py'),Path(__file__),*HERE.glob('*.f'),*HERE.glob('*.inc'),*HERE.glob('*.txt')]
(HERE/'preparation.json').write_text(json.dumps({'scope':'Private synthetic compact DP experiment; no production promotion; numerical qualification awaits separate COMB/consumer proof','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')
print('PRIVATE_KERNELS_READY',HERE)
