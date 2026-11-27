"""Loop coefficients against the contraction for groups 44 and 49: the QP pass prints its loop
coefficients (LCQ); the DP pass then contracts groups 44, 49 with LC_SUBST mode
1: QP coefficients rounded to DP, DP contraction; 2: QP coefficients, QP contraction;
3: DP coefficients, QP contraction (with the DP tensor integrals in every mode)."""
import json, os, subprocess, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
OUT = '/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad/trace/uux'
pdir = olc.subprocess_dir(OUT)
GROUPS = (44, 49)
# QP pass: print the coefficients of the groups
p = os.path.join(pdir, 'TIRUST_interface.f'); s = open(p).read()
if 'LCQ' not in s:
    a = """      RESULT=0
      DO I=1,3
        DO J=0,NCOEF_R(RANK)-1
          RESULT(I)=RESULT(I)+LOOPCOEFS(J,SQSOINDEX,ID)*COEFS(J,I)"""
    assert s.count(a) == 1
    s = s.replace(a, """      IF(ID.EQ.44.OR.ID.EQ.49) THEN
        DO J=0,NCOEF_R(RANK)-1
          WRITE(*,'(A,2I5,2ES46.36)') 'LCQ',ID,J,
     $     REAL(LOOPCOEFS(J,SQSOINDEX,ID),KIND=16),
     $     AIMAG(LOOPCOEFS(J,SQSOINDEX,ID))
        ENDDO
      ENDIF
""" + a)
    open(p, 'w').write(s)
# DP pass: the alternative contractions
p = os.path.join(pdir, 'TIR_interface.f'); s = open(p).read()
if 'LC_SUBST' not in s:
    a = "      COMMON/ML5_0_LCOEFS/LOOPCOEFS\n"
    assert s.count(a) == 1
    s = s.replace(a, a + """      COMPLEX*32 LCS_Q(0:LOOPMAXCOEFS-1),LCS_R(3)
      REAL*16 LCS_RE,LCS_IM
      INTEGER LCS_MODE,LCS_ID,LCS_J,LCS_K
      CHARACTER*16 LCS_ENV
      LOGICAL LCS_INIT
      SAVE LCS_MODE,LCS_INIT
      DATA LCS_INIT/.FALSE./
""")
    a = """      DO I=1,3
        RES(I)=(0.0D0,0.0D0)
        DO J=0,NLOOPCOEFS-1
          RES(I)=RES(I)+LOOPCOEFS(J,I_SQSO,I_LOOPGROUP)*TIRCOEFS(J,I)
        ENDDO
      ENDDO
"""
    assert s.count(a) == 1
    s = s.replace(a, """      IF(.NOT.LCS_INIT) THEN
        LCS_MODE=0
        LCS_ENV=' '
        CALL GET_ENVIRONMENT_VARIABLE('LC_SUBST',LCS_ENV)
        IF(LEN_TRIM(LCS_ENV).GT.0) READ(LCS_ENV,*) LCS_MODE
        LCS_INIT=.TRUE.
      ENDIF
      IF(LCS_MODE.GT.0.AND.(I_LOOPGROUP.EQ.44.OR.I_LOOPGROUP.EQ.49))
     $  THEN
        LCS_Q=0
        OPEN(77,FILE='lcq.txt',STATUS='OLD')
        DO LCS_K=1,100000
          READ(77,*,END=91) LCS_ID,LCS_J,LCS_RE,LCS_IM
          IF(LCS_ID.EQ.I_LOOPGROUP) LCS_Q(LCS_J)=CMPLX(LCS_RE,LCS_IM
     $     ,KIND=16)
        ENDDO
 91     CLOSE(77)
        DO I=1,3
          LCS_R(I)=0
          DO J=0,NLOOPCOEFS-1
            IF(LCS_MODE.EQ.1) THEN
              LCS_R(I)=LCS_R(I)+CMPLX(CMPLX(LCS_Q(J),KIND=8)
     $         *TIRCOEFS(J,I),KIND=16)
            ELSEIF(LCS_MODE.EQ.2) THEN
              LCS_R(I)=LCS_R(I)+LCS_Q(J)*CMPLX(TIRCOEFS(J,I),KIND=16)
            ELSE
              LCS_R(I)=LCS_R(I)+CMPLX(LOOPCOEFS(J,I_SQSO,I_LOOPGROUP)
     $         ,KIND=16)*CMPLX(TIRCOEFS(J,I),KIND=16)
            ENDIF
          ENDDO
          RES(I)=CMPLX(LCS_R(I),KIND=8)
        ENDDO
      ELSE
""" + a + "      ENDIF\n")
    open(p, 'w').write(s)
olc.build_driver(pdir)
sp = json.load(open(OUT + '_split.json')); q = sp['4']
pt = json.load(open(OUT + '_modes.json'))['point']
lines = ['1', '%.17e' % 11.0] + [' '.join('%.17e' % c for c in v) for v in pt]
def run(mode, env):
    olc.set_reduction(OUT, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d99', 'ImprovePSPoint': '2'})
    return subprocess.run([os.path.join(pdir, 'ml_points_driver')], cwd=pdir, input='\n'.join(lines) + '\n',
                          capture_output=True, text=True, check=True, env=dict(os.environ, **env)).stdout.splitlines()
o = run('4', {})
lcq = [l.split()[1:] for l in o if l.startswith('LCQ')]
# the last QP evaluation's coefficients (one block per group per evaluation)
last = {}
for idx, j, re_, im_ in lcq:
    last[(int(idx), int(j))] = (re_, im_)
with open(os.path.join(pdir, 'lcq.txt'), 'w') as f:
    for (idx, j), (re_, im_) in sorted(last.items()):
        f.write('%d %d %s %s\n' % (idx, j, re_.replace('E', 'Q') if False else re_, im_))
print('dumped %d QP coefficients of groups 44, 49' % len(last))
for m in ('0', '1', '2', '3'):
    for mode in ('1', '2'):
        o = run(mode, {'LC_SUBST': m})
        ct = [l.split() for l in o if l.startswith('SPLITDBG CT')][-1]
        g = {}
        for l in o:
            if l.startswith('SPLITDBG LG') and l.split()[2] == ct[2]:
                f = l.split(); g[int(f[3])] = [float(x) for x in f[4:]]
        ans = float(ct[6])
        print('LC_SUBST %s CTModeRun %s: groups 44, 49 vs QP %.2e %.2e | total vs QP %.1e rel' % (
            m, mode, g[44][0] - q['groups']['44'][0], g[49][0] - q['groups']['49'][0], (ans - q['ans']) / abs(q['ans'])))
olc.set_reduction(OUT, ['9'], extra={'CTModeRun': '-1', 'MLStabThres': '1.0d-3', 'ImprovePSPoint': '2'})
