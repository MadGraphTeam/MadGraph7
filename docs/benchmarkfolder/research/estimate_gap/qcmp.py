"""Per TIRust call of the double-precision pass (calls with TIR_QCMP_N lines, -1 all): the
same call in quadruple precision from the same inputs, the largest relative difference of the
coefficients (finite, 1/eps, 1/eps^2 parts), printed as QSBCMP; with TIR_QSUBST_USE=1 the
quadruple-precision coefficients replace the double-precision ones."""
import json, os, re, subprocess, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
OUT = '/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad/trace/uux'
pdir = olc.subprocess_dir(OUT)
path = os.path.join(pdir, 'TIRUST_interface.f')
s = open(path).read()
i = s.index('      SUBROUTINE ML5_0_TIRUST_TENSOR_DIAGNOSTIC(')
j = s.index('\n      END\n', i)
body = s[i:j]
if 'TIR_QSUBST_N' in body:   # drop the earlier substitution block
    a = body.index('      IF(.NOT.QSB_INIT) THEN')
    b = body.index('        RETURN\n      ENDIF\n', a) + len('        RETURN\n      ENDIF\n')
    body = body[:a] + body[b:]
    body = body.replace("""      REAL*16 QSB_PL(0:3,NLOOPLINE)
      COMPLEX*32 QSB_COEFS(0:LOOPMAXCOEFS-1,3)
      INTEGER QSB_N
      CHARACTER*16 QSB_ENV
      LOGICAL QSB_INIT
      SAVE QSB_N,QSB_INIT
      DATA QSB_INIT/.FALSE./
""", '')
if 'TIR_QCMP_N' not in body:
    a = "      INCLUDE 'coupl.inc'\n"
    body = body.replace(a, """      REAL*16 QSB_PL(0:3,NLOOPLINE)
      COMPLEX*32 QSB_COEFS(0:LOOPMAXCOEFS-1,3)
      INTEGER QSB_N,QSB_USE,QI,QJ
      CHARACTER*16 QSB_ENV
      LOGICAL QSB_INIT,QSB_ON,QSB_OK
      REAL*8 QSB_D(3),QSB_R(3)
      SAVE QSB_N,QSB_USE,QSB_INIT
      DATA QSB_INIT/.FALSE./
""" + a, 1)
    a = "      STABLE=.FALSE.\n      TIRCOEFS=0\n"
    assert body.count(a) == 1
    body = body.replace(a, a + """      IF(.NOT.QSB_INIT) THEN
        QSB_N=0
        QSB_USE=0
        QSB_ENV=' '
        CALL GET_ENVIRONMENT_VARIABLE('TIR_QCMP_N',QSB_ENV)
        IF(LEN_TRIM(QSB_ENV).GT.0) READ(QSB_ENV,*) QSB_N
        QSB_ENV=' '
        CALL GET_ENVIRONMENT_VARIABLE('TIR_QSUBST_USE',QSB_ENV)
        IF(LEN_TRIM(QSB_ENV).GT.0) READ(QSB_ENV,*) QSB_USE
        QSB_INIT=.TRUE.
      ENDIF
      QSB_ON=QSB_N.EQ.-1.OR.QSB_N.EQ.NLOOPLINE
      IF(QSB_ON) THEN
        QSB_PL=REAL(PL,KIND=16)
        CALL ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE,RANK,
     $   QSB_PL,M2L,QSB_COEFS,QSB_OK)
        STABLE=.FALSE.
        TIRCOEFS=0
      ENDIF
""")
    a = "      STABLE=.TRUE."
    assert body.endswith(a), body[-80:]
    body = body + """
      IF(QSB_ON) THEN
        QSB_D=0
        QSB_R=0
        DO QI=0,NC-1
          DO QJ=1,3
            QSB_D(QJ)=MAX(QSB_D(QJ),DBLE(ABS(TIRCOEFS(QI,QJ)
     $       -CMPLX(QSB_COEFS(QI,QJ),KIND=8))))
            QSB_R(QJ)=MAX(QSB_R(QJ),DBLE(ABS(QSB_COEFS(QI,QJ))))
          ENDDO
        ENDDO
        WRITE(*,'(A,4I4,L2,6ES12.3)') 'QSBCMP',CTMODE,NLOOPLINE,RANK,
     $   NC,QSB_OK,(QSB_D(QJ)/MAX(QSB_R(QJ),1D-300),QJ=1,3),QSB_R
        IF(QSB_USE.EQ.1.AND.QSB_OK) THEN
          DO QI=0,NC-1
            DO QJ=1,3
              TIRCOEFS(QI,QJ)=CMPLX(QSB_COEFS(QI,QJ),KIND=8)
            ENDDO
          ENDDO
        ENDIF
      ENDIF"""
s = s[:i] + body + s[j:]
open(path, 'w').write(s)
olc.build_driver(pdir)
sp = json.load(open(OUT + '_split.json')); q = sp['4']
pt = json.load(open(OUT + '_modes.json'))['point']
lines = ['1', '%.17e' % 11.0] + [' '.join('%.17e' % c for c in v) for v in pt]
for n in ('5', '-1'):
    for use in ('0', '1'):
        olc.set_reduction(OUT, ['9'], extra={'CTModeRun': '1', 'MLStabThres': '1.0d99', 'ImprovePSPoint': '2'})
        env = dict(os.environ, TIR_QCMP_N=n, TIR_QSUBST_USE=use)
        o = subprocess.run([os.path.join(pdir, 'ml_points_driver')], cwd=pdir, input='\n'.join(lines) + '\n',
                           capture_output=True, text=True, check=True, env=env).stdout.splitlines()
        cmp_ = [l.split() for l in o if l.startswith('QSBCMP')]
        ct = [l.split() for l in o if l.startswith('SPLITDBG CT')][-1]
        g = {}
        for l in o:
            if l.startswith('SPLITDBG LG') and l.split()[2] == ct[2]:
                f = l.split(); g[int(f[3])] = [float(x) for x in f[4:]]
        ans = float(ct[6])
        nok = sum(1 for c in cmp_ if c[5] == 'T')
        worst = sorted(cmp_, key=lambda c: -float(c[6]))[:6]
        print('N=%-2s use=%s: %d calls compared (%d with QP served); total diff vs QP %.2e rel; groups 44, 49 %.1e %.1e' % (
            n, use, len(cmp_), nok, (ans - q['ans']) / abs(q['ans']), g[44][0] - q['groups']['44'][0], g[49][0] - q['groups']['49'][0]))
        if use == '0':
            for c in worst:
                print('   call N=%s rank=%s QPok=%s: rel diff finite %s 1/eps %s 1/eps2 %s  (max|QP| %s %s %s)' % (c[2], c[3], c[5], c[6], c[7], c[8], c[9], c[10], c[11]))
olc.set_reduction(OUT, ['9'], extra={'CTModeRun': '-1', 'MLStabThres': '1.0d-3', 'ImprovePSPoint': '2'})
