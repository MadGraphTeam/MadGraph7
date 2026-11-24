"""Fixtures for TIRust: every raised-power call of a two-P-wave output (the
integrals of the second-order dual terms), with the inputs MadLoop passes,
whether TIRust's capability check accepts the call, its status and, when
served, its binary64 array.

    fixtures.py OUTDIR OUT.json.gz ENERGY:NPOINTS [ENERGY:NPOINTS ...] [--unit]

OUTDIR is patched in place (pass a copy):
- onia_loop.f: a refused raised-power call no longer ends the loop group
  (PW_REFUSED is still set), so every call of the point is reached;
- TIRUST_interface.f: the double-precision raised-power entry writes its
  inputs (FX_DUMP=FILE) before the capability check, then the status and the
  array.
With --unit, the unit-power entry (TIRUST_TENSOR_DIAGNOSTIC) writes its
calls too, in the same format with every power 1 (OUTDIR patched before is
patched again only for the part it lacks).
Both passes are dumped: the primary one (CTModeRun 1) and the reversed loop
of the stability test (CTModeRun 2, inputs after SWITCH_ORDER; TIRust shifts
the invariants of a check pass itself). All numbers are hex doubles."""
import gzip
import json
import os
import re
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', '..', 'tools'))
import onia_loop_check as olc

argv = [a for a in sys.argv[1:] if a != '--unit']
unit = '--unit' in sys.argv[1:]
outdir, out = os.path.abspath(argv[0]), argv[1]
runs = [(float(a.split(':')[0]), int(a.split(':')[1])) for a in argv[2:]]
pdir = olc.subprocess_dir(outdir)


def routine(src, name):
    start = src.index('      SUBROUTINE %s(' % name)
    end = src.index('\n      END\n', start)
    return start, end


# onia_loop.f: continue past refusals
path = os.path.join(pdir, 'onia_loop.f')
src = open(path).read()
if 'fixtures.py' not in src:
    s, e = routine(src, 'ML5_0_ONIA_PW_DOUBLED')
    body, n = re.subn(r'(\n( +)PW_REFUSED=\.TRUE\.\n)\2RETURN\n',
                      r'\1\2CYCLE\n', src[s:e])
    assert n >= 2, n
    src = src[:s] + body.replace('      IMPLICIT NONE\n', '      IMPLICIT NONE\nC     fixtures.py: refusals do not end the group\n', 1) + src[e:]
    open(path, 'w').write(src)
    print('onia_loop.f: %d refusals continue' % n)

# TIRUST_interface.f: dump inputs, capability, status and array
path = os.path.join(pdir, 'TIRUST_interface.f')
src = open(path).read()
if 'FX_DUMP' not in src:
    s, e = routine(src, 'ML5_0_TIRUST_TENSOR_POWERS')
    body = src[s:e]
    supp = '''      IF(.NOT.TIRUST_SUPPORTS_POWERS(NLOOPLINE,RANK,POW
     $ ,TIRUSTCOMPUTEFINITE)) RETURN
'''
    assert body.count(supp) == 1
    body = body.replace(supp, '''      FX_SUPP=TIRUST_SUPPORTS_POWERS(NLOOPLINE,RANK,POW
     $ ,TIRUSTCOMPUTEFINITE)
''')
    body = body.replace('      LOGICAL STABLE,CACHE_ENABLED,CHECK_PASS\n',
                        '      LOGICAL STABLE,CACHE_ENABLED,CHECK_PASS\n      LOGICAL FX_SUPP\n', 1)
    call = '      CALL TIRUST_TENSOR_EVALUATE_POWERS(NLOOPLINE,RANK,POW,SHIFTS\n'
    assert body.count(call) == 1
    body = body.replace(call, '''      CALL FX_INPUTS(CTMODE,NLOOPLINE,RANK,POW,SHIFTS,INVARIANTS,MASSES
     $ ,SCALE,FX_SUPP)
      IF(.NOT.FX_SUPP) RETURN
''' + call)
    ret = '      IF(STATUS.NE.TIRUST_SUCCESS) RETURN\n      DO I=0,NC-1\n'
    assert body.count(ret) == 1
    body = body.replace(ret, '      CALL FX_RESULT(NC,STATUS,RAW)\n' + ret)
    src = src[:s] + body + src[e:] + '''
      SUBROUTINE FX_INPUTS(CTMODE,N,RANK,POW,SHIFTS,INV,MASSES,SCALE
     $ ,SUPP)
C     fixtures.py: the inputs of a raised-power call, as hex doubles
      IMPLICIT NONE
      INTEGER CTMODE,N,RANK,POW(N),I,J
      REAL*8 SHIFTS(0:3,N),INV(N,N),SCALE
      COMPLEX*16 MASSES(N)
      LOGICAL SUPP
      INTEGER U
      COMMON/FX_UNIT/U
      CALL FX_OPEN()
      IF (U.LE.0) RETURN
      WRITE(U,'(A,4I4,L2,1X,8I2)') 'CALL',CTMODE,N,RANK,0,SUPP,
     $ (POW(I),I=1,N)
      WRITE(U,'(A,1X,Z16.16)') 'MU2',SCALE
      DO J=1,N
        WRITE(U,'(A,I3,2(1X,Z16.16))') 'M',J,MASSES(J)
      ENDDO
      DO J=1,N
        WRITE(U,'(A,I3,8(1X,Z16.16))') 'I',J,(INV(J,I),I=1,N)
      ENDDO
      DO J=1,N
        WRITE(U,'(A,I3,4(1X,Z16.16))') 'S',J,(SHIFTS(I,J),I=0,3)
      ENDDO
      IF (.NOT.SUPP) WRITE(U,'(A)') 'END'
      FLUSH(U)
      END

      SUBROUTINE FX_RESULT(NC,STATUS,RAW)
      IMPLICIT NONE
      INCLUDE 'loop_max_coefs.inc'
      INTEGER NC,STATUS,I,J
      COMPLEX*16 RAW(LOOPMAXCOEFS,4)
      INTEGER U
      COMMON/FX_UNIT/U
      IF (U.LE.0) RETURN
      WRITE(U,'(A,2I6)') 'RES',STATUS,NC
      IF (STATUS.EQ.0) THEN
        DO I=1,NC
          WRITE(U,'(8(Z16.16,1X))') (REAL(RAW(I,J)),AIMAG(RAW(I,J)),J=1,4)
        ENDDO
      ENDIF
      WRITE(U,'(A)') 'END'
      FLUSH(U)
      END

      SUBROUTINE FX_OPEN()
      IMPLICIT NONE
      INTEGER U,L
      LOGICAL, SAVE :: DONE=.FALSE.
      CHARACTER*512 FNAME
      COMMON/FX_UNIT/U
      IF (DONE) RETURN
      DONE=.TRUE.
      U=0
      CALL GET_ENVIRONMENT_VARIABLE('FX_DUMP',FNAME,L)
      IF (L.GT.0) THEN
        U=73
        OPEN(UNIT=U,FILE=FNAME(1:L),STATUS='REPLACE')
      ENDIF
      END
'''
    open(path, 'w').write(src)
    print('TIRUST_interface.f: raised-power calls dumped')
    for f in ('onia_loop.o', 'TIRUST_interface.o', 'ml_points_driver'):
        if os.path.exists(os.path.join(pdir, f)):
            os.unlink(os.path.join(pdir, f))
# TIRUST_interface.f, with --unit: the unit-power entry dumps its calls too
path = os.path.join(pdir, 'TIRUST_interface.f')
src = open(path).read()
if unit and 'FX_UNIT_POW' not in src:
    s0, e0 = routine(src, 'ML5_0_TIRUST_TENSOR_DIAGNOSTIC')
    body = src[s0:e0]
    body = body.replace('      LOGICAL STABLE,CACHE_ENABLED,CHECK_PASS\n',
                        '      LOGICAL STABLE,CACHE_ENABLED,CHECK_PASS\n      INTEGER FX_UNIT_POW(NLOOPLINE)\n', 1)
    call = '      CALL TIRUST_TENSOR_EVALUATE(NLOOPLINE,RANK,SHIFTS,INVARIANTS'
    assert body.count(call) == 1
    body = body.replace(call, '''      FX_UNIT_POW=1
      CALL FX_INPUTS(CTMODE,NLOOPLINE,RANK,FX_UNIT_POW,SHIFTS,
     $ INVARIANTS,MASSES,SCALE,.TRUE.)
''' + call)
    ret = '      IF(STATUS.NE.TIRUST_SUCCESS) RETURN\n      DO I=0,NC-1\n'
    assert body.count(ret) == 1
    body = body.replace(ret, '      CALL FX_RESULT(NC,STATUS,RAW)\n' + ret)
    src = src[:s0] + body + src[e0:]
    open(path, 'w').write(src)
    print('TIRUST_interface.f: unit-power calls dumped')
    for f in ('TIRUST_interface.o', 'ml_points_driver'):
        if os.path.exists(os.path.join(pdir, f)):
            os.unlink(os.path.join(pdir, f))
olc.build_driver(pdir)
masses = olc.driver_masses(pdir)

hexd = lambda h: float.hex(struct.unpack('>d', bytes.fromhex(h))[0])


def parse(path):
    calls, cur = [], None
    for line in open(path):
        f = line.split()
        if f[0] == 'CALL':
            n = int(f[2])
            cur = dict(ctmode=int(f[1]), n=n, rank=int(f[3]), supported=f[5] == 'T',
                       powers=[int(x) for x in f[6:6 + n]], masses2=[], invariants=[], shifts=[])
            calls.append(cur)
        elif f[0] == 'MU2':
            cur['mu2'] = hexd(f[1])
        elif f[0] == 'M':
            cur['masses2'].append([hexd(x) for x in f[2:4]])
        elif f[0] == 'I':
            cur['invariants'].append([hexd(x) for x in f[2:]])
        elif f[0] == 'S':
            cur['shifts'].append([hexd(x) for x in f[2:]])
        elif f[0] == 'RES':
            cur['status'], cur['nc'], cur['dp'] = int(f[1]), int(f[2]), []
        elif f[0] == 'END':
            cur = None
        elif cur is not None and 'dp' in cur:
            v = [hexd(x) for x in f]
            cur['dp'].append([[v[2 * j], v[2 * j + 1]] for j in range(4)])
    return calls


records = []
work = os.path.dirname(outdir)
for energy, npts in runs:
    pts = [p['momenta'] for p in olc.points_lib.generate_points(masses[2:], count=npts, energy=energy, seed=1,
                                                            incoming_masses=tuple(masses[:2]))]
    for k, p in enumerate(pts):
        for mode in ('1', '2'):
            olc.set_reduction(outdir, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d99'})
            dump = os.path.join(work, 'fx_%g_%d_%s.txt' % (energy, k, mode))
            os.environ['FX_DUMP'] = dump
            res = olc.run_points(pdir, [p], None, warmup=0, timeout=3600)[0]
            os.environ.pop('FX_DUMP')
            calls = parse(dump)
            records.append(dict(energy=energy, point=k, ctmode=int(mode), momenta=[[float.hex(c) for c in v] for v in p],
                                ret_code=res['ret_code'], born=res['born'], calls=calls))
            nref = sum(1 for c in calls if not c['supported'] or c.get('status', 0) != 0)
            print('sqrt(s) %g point %d CTMODE %s: %d raised-power calls, %d refused, code %d'
                  % (energy, k, mode, len(calls), nref, res['ret_code']), flush=True)
meta = dict(output=os.path.basename(pdir), masses=masses, note=__doc__)
with gzip.open(out, 'wt') as f:
    json.dump(dict(meta=meta, records=records), f)
print('wrote', out)
