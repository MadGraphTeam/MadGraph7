"""Phase 1 prototype of the extended-precision (80-bit) P-wave mode: retarget
the quadruple-precision path of a COPY of a standalone output to kind 10
(x87 extended), so that its CTModeRun 4 runs MadLoop's whole quadruple-
precision chain (phase-space improvement, Born, loop coefficients,
contractions, projections) in 80-bit arithmetic, with TIRust's double-
precision entry in place of its binary128 one.

    ep_retarget.py OUTPUT_COPY      (then rebuild Source/DHELAS, Source/MODEL
                                     and the driver)

- REAL*16 -> REAL*10, COMPLEX*32 -> COMPLEX*20, KIND=16 -> KIND=10, the
  _16 suffix of numeric literals -> _10 and Q exponents (1Q0) -> E.._10, in Source/DHELAS, Source/MODEL and
  the SubProcesses directories (TIRust's own kind KI_QTIR is left alone);
- the two binary128 TIRust routines of TIRUST_interface.f become wrappers
  of the double-precision ones (inputs rounded to double, arrays widened).
  If lowwords.py was applied first (TIRust with low words), the raised-
  power wrapper takes ML5_0_TIRUST_TENSOR_POWERS_XP (RAW + RAW_LO, summed
  in kind 10) when ONIA_EP_LOWWORDS is set at run time, else RAW only.
  After lowwords.py --ext, ONIA_EP_LOWWORDS=2 takes
  ML5_0_TIRUST_TENSOR_POWERS_EXT instead: shifts and invariants formed in
  kind 10, passed as hi + lo words with INPUT_BITS = 64; =3 sends the
  unit-power calls through it as well (powers all 1).

Prototype only: the generator support comes in Phase 2."""
import glob
import os
import re
import sys

root = os.path.abspath(sys.argv[1])
files = set()
for pattern in ('Source/DHELAS/*.f', 'Source/DHELAS/*.inc', 'Source/MODEL/*.f', 'Source/MODEL/*.inc',
                'SubProcesses/P*/*.f', 'SubProcesses/P*/*.inc'):
    for f in glob.glob(os.path.join(root, pattern)):
        files.add(os.path.realpath(f))

SPLIT = re.compile(r'([EeDdQq])(\n {5}\S\s*[+-]?\d+)_16\b')
QLIT = re.compile(r'(?<![A-Za-z0-9_])(\d+\.?\d*|\.\d+)[Qq]([+-]?\d+)(?![A-Za-z0-9_])')
LIT = re.compile(r'(?<![A-Za-z0-9_])((?:\d+\.\d*|\.\d+)(?:[EeDdQq][+-]?\d+)?|\d+[EeDdQq][+-]?\d+)_16\b')
SUBS = [(re.compile(r'REAL\*16', re.I), 'REAL*10'),
        (re.compile(r'COMPLEX\*32', re.I), 'COMPLEX*20'),
        (re.compile(r'KIND\s*=\s*16\b', re.I), 'KIND=10'),
        (re.compile(r'(integer\s*,\s*parameter\s*::\s*qp\s*=\s*)16', re.I), r'\g<1>10')]

changed = 0
for f in sorted(files):
    if not f.startswith(root):
        continue
    src = open(f).read()
    out = src
    for rx, rep in SUBS:
        out = rx.sub(rep, out)
    out = LIT.sub(r'\1_10', out)
    # a literal split across a continuation line: 1.18E / $ +00_16
    out = SPLIT.sub(r'\1\2_10', out)
    # gfortran's Q exponent (ALOHA's quadruple-precision literals: 1Q0)
    out = QLIT.sub(r'\1E\2_10', out)
    if out != src:
        open(f, 'w').write(out)
        changed += 1
print('retargeted', changed, 'files')

# the binary128 TIRust routines -> wrappers of the double-precision ones
for tif in glob.glob(os.path.join(root, 'SubProcesses/P*/TIRUST_interface.f')):
    src = open(tif).read()

    def replace(src, name, body):
        start = src.index('      SUBROUTINE %s(' % name)
        end = src.index('\n      END\n', start) + len('\n      END\n')
        return src[:start] + body + src[end:]

    src = replace(src, 'ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC', '''      SUBROUTINE ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE
     $ ,RANK,PL,M2L,TIRCOEFS,STABLE)
C     ep_retarget.py: the double-precision TIRust entry, widened
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      INTEGER CTMODE,NLOOPLINE,RANK
      REAL*10 PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*20 TIRCOEFS(0:LOOPMAXCOEFS-1,3)
      LOGICAL STABLE
      REAL*8 DPL(0:3,NLOOPLINE)
      COMPLEX*16 DC(0:LOOPMAXCOEFS-1,3)
''' + ('''      CHARACTER*8 LWENV
      INTEGER UPOW(NLOOPLINE)
      CALL GET_ENVIRONMENT_VARIABLE('ONIA_EP_LOWWORDS',LWENV)
      IF(LWENV(1:1).EQ.'3') THEN
        UPOW=1
        CALL ML5_0_TIRUST_TENSOR_POWERS_EXT(CTMODE,NLOOPLINE,RANK,UPOW
     $   ,PL,M2L,TIRCOEFS,STABLE,64)
        RETURN
      ENDIF
''' if 'ML5_0_TIRUST_TENSOR_POWERS_EXT' in src else '') + '''      DPL=REAL(PL,KIND=8)
      CALL ML5_0_TIRUST_TENSOR_DIAGNOSTIC(CTMODE,NLOOPLINE,RANK,DPL
     $ ,M2L,DC,STABLE)
      TIRCOEFS=DC
      END
''')
    src = replace(src, 'ML5_0_MP_TIRUST_TENSOR_POWERS', '''      SUBROUTINE ML5_0_MP_TIRUST_TENSOR_POWERS(CTMODE,NLOOPLINE,RANK
     $ ,POWERS,PL,M2L,TIRCOEFS,STABLE)
C     ep_retarget.py: the double-precision TIRust entry, widened (RAW, or
C     RAW + RAW_LO with ONIA_EP_LOWWORDS after lowwords.py)
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      INTEGER CTMODE,NLOOPLINE,RANK
      INTEGER POWERS(NLOOPLINE)
      REAL*10 PL(0:3,NLOOPLINE)
      COMPLEX*16 M2L(NLOOPLINE)
      COMPLEX*20 TIRCOEFS(0:LOOPMAXCOEFS-1,3)
      LOGICAL STABLE
      REAL*8 DPL(0:3,NLOOPLINE)
      COMPLEX*16 DC(0:LOOPMAXCOEFS-1,3)
      CHARACTER*8 LWENV
      DPL=REAL(PL,KIND=8)
''' + ('''      CALL GET_ENVIRONMENT_VARIABLE('ONIA_EP_LOWWORDS',LWENV)
''' if 'ML5_0_TIRUST_TENSOR_POWERS_XP' in src else '') + ('''      IF(LWENV(1:1).EQ.'2'.OR.LWENV(1:1).EQ.'3') THEN
        CALL ML5_0_TIRUST_TENSOR_POWERS_EXT(CTMODE,NLOOPLINE,RANK,POWERS
     $   ,PL,M2L,TIRCOEFS,STABLE,64)
        RETURN
      ENDIF
''' if 'ML5_0_TIRUST_TENSOR_POWERS_EXT' in src else '') + ('''      IF(LEN_TRIM(LWENV).GT.0) THEN
        CALL ML5_0_TIRUST_TENSOR_POWERS_XP(CTMODE,NLOOPLINE,RANK,POWERS
     $   ,DPL,M2L,TIRCOEFS,STABLE)
        RETURN
      ENDIF
''' if 'ML5_0_TIRUST_TENSOR_POWERS_XP' in src else '') + '''      CALL ML5_0_TIRUST_TENSOR_POWERS(CTMODE,NLOOPLINE,RANK,POWERS
     $ ,DPL,M2L,DC,STABLE)
      TIRCOEFS=DC
      END
''')
    open(tif, 'w').write(src)
    print('wrapped the binary128 TIRust routines in', tif)

# the quadruple-precision Ninja and NinjaRS glue cannot take kind 10 (typed
# module interfaces); a TIRust output never calls it: stop if it does
CALL = re.compile(r'      CALL NINJA(?:RS)?_TENSOR_EVALUATE\(.*?\n(?:     [$&].*\n)*')
for ctf in glob.glob(os.path.join(root, 'SubProcesses/P*/CT_interface.f')):
    src = open(ctf).read()
    for name in ('ML5_0_MP_NINJA_LOOP', 'ML5_0_MP_NINJARS_LOOP'):
        if '      SUBROUTINE %s(' % name not in src:
            continue
        start = src.index('      SUBROUTINE %s(' % name)
        end = src.index('\n      END\n', start)
        body, n = CALL.subn("      STOP 'ep_retarget: no quadruple-precision Ninja'\n", src[start:end])
        src = src[:start] + body + src[end:]
        print('stubbed', n, 'Ninja call(s) in', name)
    open(ctf, 'w').write(src)
