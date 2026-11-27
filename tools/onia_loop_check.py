#!/usr/bin/env python3
################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""Evaluate MadLoop one-loop virtuals at deterministic phase-space points.

    tools/onia_loop_check.py --model loop_sm_onia-c_mass \\
        --process "g g > etac(1|1S01) [virt=QCD]" --out OUTDIR \\
        --libs 9 3 --npoints 10 [--energy 20] [--mu 3.1] [--json res.json]

The process is generated and written out with the MadGraph CLI (once: an
existing OUTDIR is reused unless --regenerate), a driver reading the points
on its standard input is linked against the MadLoop objects of its P*
directory, and the virtual is evaluated with each requested reduction
library (MLReductionLib). The points are those of tools/tirust_benchmark_points.py
(2 -> 1: longitudinal boosts of the rest frame; 2 -> n: RAMBO at the given
energy), with the masses of the physical legs (a bound state of mass m1+m2).
The output lists, per library and point, the Born, the finite part and the
poles, also normalised to Born * alpha_s/(2 pi), MadLoop's return code and
the evaluation time.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import tirust_benchmark_points as points_lib

DRIVER = """\
C     Reads NPOINTS, then per point MU_R (<= 0: sqrt(s)) and the momenta of
C     the NEXTERNAL physical legs; writes one RESULT line per point.
C     With --masses, writes the masses of pmass.inc. With --jz K N JZ1 ...
C     JZN (bound states), only those J_z of bound state K (ONIA_SELECT_JZ).
      PROGRAM ML_POINTS_DRIVER
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=@NEXTERNAL@)
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)
      INCLUDE 'coupl.inc'
      REAL*8 PMASS(NEXTERNAL)
      REAL*8 P(0:3,NEXTERNAL), PIN(0:3), MU, PI, AO2PI, TIME, POLES(2)
      REAL*8, ALLOCATABLE :: ANS(:,:), PREC(:)
      INTEGER NDIM, NSO, NPOINTS, K, I, J, RET
      INTEGER*8 T0, T1, RATE
      CHARACTER*64 ARG
      REAL*8 QREL(3)
      INTEGER KJZ, NJZ, JZS(32)
      PI = 4D0*DATAN(1D0)
      CALL SETPARA('param_card.dat')
@SETONIA@
      INCLUDE 'pmass.inc'
      CALL GET_COMMAND_ARGUMENT(1,ARG)
@RELMOM@
@SELJZ@
      IF (TRIM(ARG).EQ.'--masses') THEN
        DO I=1,NEXTERNAL
          WRITE(*,'(A,I4,ES26.17E3)') 'MASS',I,PMASS(I)
        ENDDO
        STOP
      ENDIF
      CALL @PREFIX@GET_ANSWER_DIMENSION(NDIM)
      CALL @PREFIX@GET_NSQSO_LOOP(NSO)
      ALLOCATE(ANS(0:3,0:NDIM),PREC(0:NSO))
      READ(*,*) NPOINTS
      DO K=1,NPOINTS
        READ(*,*) MU
        DO I=1,NEXTERNAL
          READ(*,*) (P(J,I),J=0,3)
        ENDDO
        IF (MU.LE.0D0) THEN
          PIN(0:3)=P(0:3,1)+P(0:3,2)
          MU=DSQRT(DABS(PIN(0)**2-PIN(1)**2-PIN(2)**2-PIN(3)**2))
        ENDIF
        MU_R=MU
        CALL UPDATE_AS_PARAM()
        AO2PI=G**2/(8D0*PI**2)
        CALL SYSTEM_CLOCK(T0,RATE)
        CALL @PREFIX@SLOOPMATRIX_THRES(P,ANS,-1D0,PREC,RET)
        CALL SYSTEM_CLOCK(T1)
        TIME=DBLE(T1-T0)/DBLE(RATE)
        POLES(1)=0D0
        POLES(2)=0D0
@IRPOLES@
        WRITE(*,'(A,I7,4ES26.17E3,ES12.4,I5,2ES26.17E3,ES14.6,2ES26.17E3)')
     &   'RESULT',K,ANS(0,0),ANS(1,0),ANS(2,0),ANS(3,0),PREC(0),RET,
     &   AO2PI,MU,TIME,POLES(1),POLES(2)
      ENDDO
      END
"""

DRIVER_MAKEFILE = """\
include makefile
ml_points_driver: ml_points_driver.o $(PROCESS) $(LIBS)
\t$(FC) $(FFLAGS) -o ml_points_driver ml_points_driver.o $(PROCESS) $(LINKLIBS)
"""


def run_madgraph(commands, log):
    """Run the MadGraph CLI on a command list."""
    script = log + '.mg'
    with open(script, 'w') as f:
        f.write('\n'.join(commands) + '\n')
    with open(log, 'w') as out:
        status = subprocess.call([os.path.join(ROOT, 'bin', 'madgraph'), script],
                                 stdout=out, stderr=subprocess.STDOUT, cwd=ROOT)
    text = open(log).read()
    if status or 'Error detected' in text or 'Traceback' in text:
        raise RuntimeError('MadGraph failed, see %s' % log)


def output_process(model, process, outdir, extra=(), regenerate=False):
    """Generate and write out the process (standalone MadLoop output)."""
    outdir = os.path.abspath(outdir)
    if os.path.isdir(outdir) and not regenerate:
        return outdir
    os.makedirs(os.path.dirname(outdir) or '.', exist_ok=True)
    commands = ['import model %s' % model] + list(extra) + \
               ['generate %s' % process, 'output standalone %s -f' % outdir]
    run_madgraph(commands, outdir + '.log')
    return outdir


def subprocess_dir(outdir):
    sub = os.path.join(outdir, 'SubProcesses')
    dirs = sorted(d for d in os.listdir(sub) if d.startswith('P') and
                  os.path.isdir(os.path.join(sub, d)))
    if len(dirs) != 1:
        raise RuntimeError('Expected one P* directory in %s, found %s' % (sub, dirs))
    return os.path.join(sub, dirs[0])


def make(pdir, target, makefile=None):
    cmd = ['make', '-j', '8', 'SHELL=%s' % shutil.which('bash')]
    if makefile:
        cmd += ['-f', makefile]
    cmd.append(target)
    with open(os.path.join(pdir, 'make_%s.log' % target), 'w') as out:
        status = subprocess.call(cmd, cwd=pdir, stdout=out, stderr=subprocess.STDOUT)
    if status:
        raise RuntimeError('make %s failed in %s' % (target, pdir))


def build_driver(pdir):
    """Compile ml_points_driver in the P* directory."""
    prefix = open(os.path.join(pdir, 'proc_prefix.txt')).read().strip()
    onia = os.path.isfile(os.path.join(pdir, 'onia_loop_data.inc'))
    if onia:
        nexternal = int(re.search(r'NEXTERNAL_PHYS=(\d+)',
                                  open(os.path.join(pdir, 'onia_loop.f')).read()).group(1))
    else:
        nexternal = int(re.search(r'NEXTERNAL=(\d+)',
                                  open(os.path.join(pdir, 'nexternal.inc')).read()).group(1))
    source = DRIVER.replace('@NEXTERNAL@', str(nexternal)).replace('@PREFIX@', prefix)
    source = source.replace('@SETONIA@', "      CALL SETONIA('onia_card.dat')" if onia else '')
    source = source.replace('@IRPOLES@', ("        CALL %sONIA_IR_POLES(P,POLES)" % prefix)
                            if onia else '')
    relmom = '''      IF (TRIM(ARG).EQ.'--relmom') THEN
        DO I=1,3
          CALL GET_COMMAND_ARGUMENT(I+1,ARG)
          READ(ARG,*) QREL(I)
        ENDDO
        CALL %sONIA_SET_RELATIVE_MOMENTUM(QREL)
      ENDIF''' % prefix
    source = source.replace('@RELMOM@', relmom if onia else '')
    seljz = '''      IF (TRIM(ARG).EQ.'--jz') THEN
        CALL GET_COMMAND_ARGUMENT(2,ARG)
        READ(ARG,*) KJZ
        CALL GET_COMMAND_ARGUMENT(3,ARG)
        READ(ARG,*) NJZ
        DO I=1,NJZ
          CALL GET_COMMAND_ARGUMENT(3+I,ARG)
          READ(ARG,*) JZS(I)
        ENDDO
        CALL %sONIA_SELECT_JZ(KJZ,NJZ,JZS)
      ENDIF''' % prefix
    # outputs generated before ONIA_SELECT_JZ existed have no --jz
    has_jz = onia and 'ONIA_SELECT_JZ' in open(os.path.join(pdir, 'onia_loop.f')).read()
    source = source.replace('@SELJZ@', seljz if has_jz else '')
    with open(os.path.join(pdir, 'ml_points_driver.f'), 'w') as f:
        f.write(source)
    with open(os.path.join(pdir, 'ml_points_driver.mk'), 'w') as f:
        f.write(DRIVER_MAKEFILE)
    make(pdir, 'ml_points_driver', 'ml_points_driver.mk')
    return nexternal


def driver_masses(pdir):
    out = subprocess.check_output([os.path.join(pdir, 'ml_points_driver'), '--masses'],
                                  cwd=pdir, text=True)
    return [float(l.split()[2]) for l in out.splitlines() if l.startswith('MASS')]


def set_reduction(outdir, libs, extra=None):
    """Set MLReductionLib (and other MadLoopParams.dat entries) of the output."""
    card = os.path.join(outdir, 'Cards', 'MadLoopParams.dat')
    text = open(card).read()
    values = dict(extra or {})
    values['MLReductionLib'] = '|'.join(str(l) for l in libs)
    for key, value in values.items():
        text, n = re.subn(r'(#%s\n)[^\n]*' % re.escape(key), r'\g<1>%s' % value, text)
        if n != 1:
            raise RuntimeError('No parameter %s in %s' % (key, card))
    with open(card, 'w') as f:
        f.write(text)


def set_masses(outdir, masses):
    """Set entries of Block MASS of Cards/param_card.dat, {pdg: value}."""
    card = os.path.join(outdir, 'Cards', 'param_card.dat')
    lines = open(card).read().splitlines(True)
    block = None
    done = set()
    for i, line in enumerate(lines):
        f = line.split()
        if f and f[0].lower() == 'block':
            block = f[1].lower()
            continue
        if block == 'mass' and f and f[0].isdigit() and int(f[0]) in masses:
            lines[i] = '  %d %.10e # set by onia_loop_check\n' % (int(f[0]), masses[int(f[0])])
            done.add(int(f[0]))
    if done != set(masses):
        raise RuntimeError('masses %s not found in %s' % (set(masses) - done, card))
    open(card, 'w').writelines(lines)


def run_points(pdir, momenta, mu=None, warmup=3, timeout=None, relmom=None, jz=None):
    """Evaluate the virtual at the given points (lists of 4-vectors); the
    first point is repeated `warmup` times first (MadLoop's initialisation
    of its filters), and those results are dropped. jz = (k, [J_z, ...])
    restricts bound state k to those J_z (ONIA_SELECT_JZ)."""
    lines = [str(len(momenta) + warmup)]
    for p in [momenta[0]] * warmup + list(momenta):
        lines.append('%.17e' % (mu if mu else -1.0))
        lines.extend(' '.join('%.17e' % c for c in v) for v in p)
    cmd = [os.path.join(pdir, 'ml_points_driver')]
    if relmom is not None:
        cmd += ['--relmom'] + ['%.17e' % q for q in relmom]
    elif jz is not None:
        # (bound state, [J_z, ...]): only those J_z of that bound state
        cmd += ['--jz', str(jz[0]), str(len(jz[1]))] + [str(j) for j in jz[1]]
    proc = subprocess.run(cmd, cwd=pdir,
                          input='\n'.join(lines) + '\n', capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode:
        raise RuntimeError('ml_points_driver failed:\n%s\n%s' % (proc.stdout[-2000:],
                                                                 proc.stderr[-2000:]))
    results = []
    for line in proc.stdout.splitlines():
        if not line.startswith('RESULT'):
            continue
        f = line.split()
        born, fin, sp, dp = (float(x) for x in f[2:6])
        ao2pi = float(f[8])
        norm = born * ao2pi
        pred1, pred2 = float(f[11]), float(f[12])
        results.append(dict(born=born, finite=fin, single=sp, double=dp,
                            accuracy=float(f[6]), ret_code=int(f[7]), ao2pi=ao2pi,
                            mu=float(f[9]), time=float(f[10]),
                            predicted_single_norm=pred1 / norm if norm else float('nan'),
                            predicted_double_norm=pred2 / norm if norm else float('nan'),
                            finite_norm=fin / norm if norm else float('nan'),
                            single_norm=sp / norm if norm else float('nan'),
                            double_norm=dp / norm if norm else float('nan')))
    return results[warmup:]


HEL_DRIVER = """\
C     Reads NPOINTS, then per point MU_R (<= 0: sqrt(s)) and the momenta of
C     the NEXTERNAL physical legs. Per point: the summed SLOOPMATRIX_THRES
C     (configuration 0), then SLOOPMATRIXHEL_THRES for every physical
C     helicity configuration (onia_loop_data.inc), one HEL line each.
      PROGRAM ML_HEL_DRIVER
      IMPLICIT NONE
      INTEGER NEXTERNAL
      PARAMETER (NEXTERNAL=@NEXTERNAL@)
      INCLUDE 'coupl.inc'
      INCLUDE 'onia_loop_data.inc'
      REAL*8 P(0:3,NEXTERNAL), PIN(0:3), MU
      REAL*8, ALLOCATABLE :: ANS(:,:), PREC(:)
      INTEGER NDIM, NSO, NPOINTS, K, I, J, H, RET, USERHEL
      COMMON/@PREFIX@USERCHOICE/USERHEL
      CALL SETPARA('param_card.dat')
      CALL SETONIA('onia_card.dat')
      CALL @PREFIX@GET_ANSWER_DIMENSION(NDIM)
      CALL @PREFIX@GET_NSQSO_LOOP(NSO)
      ALLOCATE(ANS(0:3,0:NDIM),PREC(0:NSO))
      READ(*,*) NPOINTS
      DO K=1,NPOINTS
        READ(*,*) MU
        DO I=1,NEXTERNAL
          READ(*,*) (P(J,I),J=0,3)
        ENDDO
        IF (MU.LE.0D0) THEN
          PIN(0:3)=P(0:3,1)+P(0:3,2)
          MU=DSQRT(DABS(PIN(0)**2-PIN(1)**2-PIN(2)**2-PIN(3)**2))
        ENDIF
        MU_R=MU
        CALL UPDATE_AS_PARAM()
C       SLOOPMATRIXHEL leaves USERHEL set, as in MadLoop
        USERHEL=-1
        CALL @PREFIX@SLOOPMATRIX_THRES(P,ANS,-1D0,PREC,RET)
        WRITE(*,'(A,2I6,4ES26.17E3,ES12.4,I5)') 'HEL',K,0,ANS(0,0),
     &   ANS(1,0),ANS(2,0),ANS(3,0),PREC(0),RET
        DO H=1,ONIA_PHYS_NCOMB
          CALL @PREFIX@SLOOPMATRIXHEL_THRES(P,H,ANS,-1D0,PREC,RET)
          WRITE(*,'(A,2I6,4ES26.17E3,ES12.4,I5)') 'HEL',K,H,ANS(0,0),
     &     ANS(1,0),ANS(2,0),ANS(3,0),PREC(0),RET
        ENDDO
        USERHEL=-1
      ENDDO
      END
"""


def run_points_hel(pdir, momenta, mu=None, timeout=None):
    """Per point (lists of 4-vectors) of a bound-state output: the summed
    result and that of every physical helicity configuration
    (SLOOPMATRIXHEL_THRES), as {'sum': r, 'configs': [r1, r2, ...], 'helavg': n}
    with r = dict(born, finite, single, double, accuracy, ret_code). As in
    MadLoop, a configuration is not averaged over the initial helicities: the
    configurations add up to helavg times the sum."""
    prefix = open(os.path.join(pdir, 'proc_prefix.txt')).read().strip()
    nexternal = int(re.search(r'NEXTERNAL_PHYS=(\d+)',
                              open(os.path.join(pdir, 'onia_loop.f')).read()).group(1))
    helavg = int(re.search(r'ONIA_HELAVG=(\d+)',
                           open(os.path.join(pdir, 'onia_loop_data.inc')).read()).group(1))
    with open(os.path.join(pdir, 'ml_hel_driver.f'), 'w') as f:
        f.write(HEL_DRIVER.replace('@NEXTERNAL@', str(nexternal)).replace('@PREFIX@', prefix))
    with open(os.path.join(pdir, 'ml_hel_driver.mk'), 'w') as f:
        f.write(DRIVER_MAKEFILE.replace('ml_points_driver', 'ml_hel_driver'))
    make(pdir, 'ml_hel_driver', 'ml_hel_driver.mk')
    lines = [str(len(momenta))]
    for p in momenta:
        lines.append('%.17e' % (mu if mu else -1.0))
        lines.extend(' '.join('%.17e' % c for c in v) for v in p)
    proc = subprocess.run([os.path.join(pdir, 'ml_hel_driver')], cwd=pdir,
                          input='\n'.join(lines) + '\n', capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode:
        raise RuntimeError('ml_hel_driver failed:\n%s\n%s' % (proc.stdout[-2000:],
                                                              proc.stderr[-2000:]))
    out = [dict(sum=None, configs=[], helavg=helavg) for _ in momenta]
    for line in proc.stdout.splitlines():
        if not line.startswith('HEL'):
            continue
        f = line.split()
        k, h = int(f[1]), int(f[2])
        r = dict(born=float(f[3]), finite=float(f[4]), single=float(f[5]), double=float(f[6]),
                 accuracy=float(f[7]), ret_code=int(f[8]))
        if h == 0:
            out[k - 1]['sum'] = r
        else:
            out[k - 1]['configs'].append(r)
    return out


def relative_momentum_scan(pdir, masses, onium_legs, velocities, npoints=1, energy=1000.,
                           seed=None, mu=None, direction=(0.48, 0.36, 0.8), timeout=None):
    """The virtual with a relative momentum +-q of the constituents of the
    bound states (equal constituent masses): for each velocity v of the
    constituents in the rest frame, q = m v/sqrt(1-v^2) along `direction`,
    the bound states of mass 2 sqrt(m^2+q^2) = 2m/sqrt(1-v^2), the same
    random phase-space point for every v (RAMBO is smooth in the masses).
    Returns {v: [mean of the +q and -q results (finite_norm, single_norm,
    double_norm) per point]}. The finite part is a/v + b + O(v): the Coulomb
    singularity and the threshold value b (tools: fit_threshold)."""
    seed = points_lib.DEFAULT_SEED if seed is None else seed
    norm = math.sqrt(sum(d * d for d in direction))
    direction = [d / norm for d in direction]
    out = {}
    for v in velocities:
        mv = list(masses)
        q = None
        for leg in onium_legs:
            m = masses[leg] / 2.
            mv[leg] = 2. * m / math.sqrt(1. - v * v)
            q = m * v / math.sqrt(1. - v * v)
        final = mv[2:]
        e = energy if len(final) > 1 else final[0]
        pts = points_lib.generate_points(final, count=npoints, energy=e, seed=seed,
                                         incoming_masses=tuple(mv[:2]))
        moms = [p['momenta'] for p in pts]
        res = []
        for sign in (1., -1.):
            res.append(run_points(pdir, moms, mu, timeout=timeout,
                                  relmom=[sign * q * d for d in direction]))
        out[v] = [dict((k, 0.5 * (a[k] + b[k])) for k in
                       ('finite_norm', 'single_norm', 'double_norm', 'born'))
                  for a, b in zip(*res)]
    return out


def fit_threshold(velocities, values, powers=(-1, 0, 1, 2)):
    """Least-squares fit of values(v) = sum_k c_k v^k over `powers`;
    returns {power: c}."""
    rows = [[v ** k for k in powers] for v in velocities]
    # normal equations
    n = len(powers)
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    aty = [sum(r[i] * y for r, y in zip(rows, values)) for i in range(n)]
    # Gaussian elimination
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(ata[r][i]))
        ata[i], ata[piv] = ata[piv], ata[i]
        aty[i], aty[piv] = aty[piv], aty[i]
        for r in range(i + 1, n):
            f = ata[r][i] / ata[i][i]
            for c in range(i, n):
                ata[r][c] -= f * ata[i][c]
            aty[r] -= f * aty[i]
    x = [0.] * n
    for i in reversed(range(n)):
        x[i] = (aty[i] - sum(ata[i][c] * x[c] for c in range(i + 1, n))) / ata[i][i]
    return dict(zip(powers, x))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', required=True)
    parser.add_argument('--process', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--extra', action='append', default=[],
                        help='additional MadGraph commands before generate')
    parser.add_argument('--libs', nargs='+', default=['9'],
                        help='reduction libraries, each a MLReductionLib value (e.g. 9 3 "9|1")')
    parser.add_argument('--npoints', type=int, default=5)
    parser.add_argument('--energy', type=float, default=1000.0)
    parser.add_argument('--seed', type=int, default=points_lib.DEFAULT_SEED)
    parser.add_argument('--mu', type=float, default=None, help='MU_R (default sqrt(s))')
    parser.add_argument('--param', action='append', default=[],
                        help='MadLoopParams.dat entry NAME=VALUE')
    parser.add_argument('--mass', action='append', default=[],
                        help='param_card mass PDG=VALUE (e.g. 5=1000)')
    parser.add_argument('--timeout', type=float, default=None,
                        help='seconds allowed per library (failures are recorded)')
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--json')
    args = parser.parse_args(argv)

    outdir = output_process(args.model, args.process, args.out, args.extra, args.regenerate)
    pdir = subprocess_dir(outdir)
    if args.mass:
        set_masses(outdir, dict((int(m.split('=')[0]), float(m.split('=')[1]))
                                for m in args.mass))
    nexternal = build_driver(pdir)
    masses = driver_masses(pdir)
    final = masses[2:]
    energy = args.energy if len(final) > 1 else final[0]
    pts = points_lib.generate_points(final, count=args.npoints, energy=energy,
                                     seed=args.seed, incoming_masses=tuple(masses[:2]))
    momenta = [p['momenta'] for p in pts]
    extra = dict(p.split('=', 1) for p in args.param)
    report = dict(model=args.model, process=args.process, outdir=outdir,
                  nexternal=nexternal, masses=masses, points=momenta, results={})
    for lib in args.libs:
        set_reduction(outdir, lib.split('|'), extra)
        try:
            res = run_points(pdir, momenta, args.mu, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print('lib %-8s TIMEOUT after %s s' % (lib, args.timeout))
            report['results'][lib] = 'timeout'
            continue
        except RuntimeError as error:
            print('lib %-8s FAILED: %s' % (lib, str(error)[:300]))
            report['results'][lib] = 'failed'
            continue
        report['results'][lib] = res
        for i, r in enumerate(res):
            print('lib %-8s pt %3d  born %.10e  fin/B %.12f  1eps/B %.12f  2eps/B %.12f'
                  '  acc %.1e  ret %d  t %.2e s  | IR pred 1eps/B %.12f 2eps/B %.12f'
                  % (lib, i, r['born'], r['finite_norm'], r['single_norm'], r['double_norm'],
                     r['accuracy'], r['ret_code'], r['time'], r['predicted_single_norm'],
                     r['predicted_double_norm']))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
