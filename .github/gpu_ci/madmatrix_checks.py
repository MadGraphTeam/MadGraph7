#!/usr/bin/env python3
"""The GPU backend of madmatrix against its scalar CPU backend, through umami.

Run by madmatrix_gpu.sh (job madmatrix of gpu_runner_ci.yml) in its run directory.

1. mssm_gpu_vs_cpu: MSSM ``p p > go go`` (both subprocesses), whose squark-quark-gluino
   couplings are running flavour couplings (nDPF > 0, gathered per event into
   dpf_value), on random points with a random alpha_s and flavour per event: the GPU
   |M|^2 equal the CPU ones event by event (the GPU gather used to address its one-event
   dpf_value with the offset of the event in the all-events coupling buffer, out of
   bounds past the first event page), and scale as alpha_s^2.
2. zero_me: ``b b~ > ta+ ta- / z a`` (s-channel Higgs only) with the tau Yukawa coupling
   set to 0 in the param card, so that no helicity survives the helicity filtering: the
   GPU library returns |M|^2 = 0 and no helicity (-1), and for the colour either none
   (-1) or the only colour flow (0, what a -ffast-math build makes of 0/0, as the CPU
   one), never stale memory; it used to abort on the invalid launch configuration
   (gridDim.y = 0) of the helicity loop.
3. nonblocking_stream: ``g g > t t~ g`` with umami handed a non-blocking stream (as a
   torch side stream, which does not synchronise with the default stream) gives the same
   |M|^2 and helicities as on the default stream, call after call: every memset, kernel
   and copy of the matrix element must go to the stream umami is given.

Each check runs in a process of its own (--only NAME): an abort of the GPU runtime
fails that check, not the others. Writes summary.txt (key=value lines) in the current
directory; exit status 1 if a check fails.
"""

import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
import time

from umami_harness import (GPU_BACKENDS, OUT_COLOR_INDEX, OUT_HELICITY_INDEX,
                           OUT_MATRIX_ELEMENT, Umami, rambo)

CPU_BACKEND = 'scalar'
CHECKS = ('mssm_gpu_vs_cpu', 'zero_me', 'nonblocking_stream')

results = {}


def section(title):
    print('\n=================== %s (%s)' % (title, time.strftime('%H:%M:%S')), flush=True)


def check(name, ok, detail):
    print('%s %s: %s' % ('PASS' if ok else 'FAIL', name, detail), flush=True)
    results[name] = (ok, detail)


def run(cmd, cwd=None, log=None):
    with open(log or os.devnull, 'w') as out:
        return subprocess.call(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT)


def output(repo, name, model, process, backends):
    """output standalone + make for these backends in every subprocess: the output
    directory and its subprocess names."""
    with open(name + '.mg5', 'w') as f:
        f.write('import model %s\ngenerate %s\noutput standalone %s -f\n' % (model, process, name))
    if run([sys.executable, os.path.join(repo, 'bin', 'madgraph'), name + '.mg5'],
           log=name + '.log') != 0:
        raise RuntimeError('madgraph failed, see %s.log' % name)
    subprocesses = sorted(os.path.basename(d)
                          for d in glob.glob(os.path.join(name, 'SubProcesses', 'P*')))
    if not subprocesses:
        raise RuntimeError('no subprocess directory for %s' % process)
    jobs = os.environ.get('SLURM_CPUS_PER_TASK', '4')
    for sub in subprocesses:
        for backend in backends:
            log = '%s_%s_%s_make.log' % (name, sub, backend)
            if run(['make', '-j', jobs, 'BACKEND=' + backend, 'FPTYPE=d'],
                   cwd=os.path.join(name, 'SubProcesses', sub), log=log) != 0:
                raise RuntimeError('make failed, see %s' % log)
    return subprocesses


def library(name, sub, backend):
    return os.path.join(name, 'lib', 'libmadmatrix_%s_%s.so' % (sub, backend))


def card(name):
    return os.path.join(name, 'Cards', 'param_card.dat')


# ---------------------------------------------------------------------------
# 1. running flavour couplings (MSSM p p > go go): GPU against CPU
# ---------------------------------------------------------------------------

def check_mssm_gpu_vs_cpu(repo, backend, npoints=1024):
    section('MSSM p p > go go, running flavour couplings: %s against %s' % (backend, CPU_BACKEND))
    name = 'mssm_gogo'
    subprocesses = output(repo, name, 'MSSM_SLHA2', 'p p > go go', (CPU_BACKEND, backend))
    rng = random.Random(11)
    problems, details = [], []
    for sub in subprocesses:
        data = open(os.path.join(name, 'SubProcesses', sub, 'ProcessData.h')).read()
        ndpf = int(re.search(r'constexpr int nDPF = (\d+);', data).group(1))
        nflavor = int(re.search(r'constexpr int nmaxflavor = (\d+);', data).group(1))
        if ndpf == 0:
            problems.append('%s: nDPF = 0, no running flavour coupling to check' % sub)
        cpu = Umami(CPU_BACKEND, library(name, sub, CPU_BACKEND), card(name))
        gpu = Umami(backend, library(name, sub, backend), card(name))
        points = [rambo(gpu.masses[2:], 3000., rng, gpu.masses[:2]) for _ in range(npoints)]
        # the first two events: one point at two values of alpha_s
        points[1] = points[0]
        alpha_s = [0.10, 0.12] + [rng.uniform(0.09, 0.14) for _ in range(npoints - 2)]
        flavor = [rng.randrange(nflavor) for _ in range(npoints)]
        (me_cpu,) = cpu(points, alpha_s=alpha_s, flavor=flavor)
        (me_gpu,) = gpu(points, alpha_s=alpha_s, flavor=flavor)
        worst, iworst = 0., -1
        for ievt, (a, b) in enumerate(zip(me_cpu, me_gpu)):
            rel = abs(a - b) / abs(a) if a != 0 else (0. if b == 0 else float('inf'))
            if not rel <= worst:  # also catches nan
                worst, iworst = rel, ievt
        if not worst < 1e-9:
            problems.append('%s: event %d %s %.10e vs %s %.10e (rel %.1e)'
                            % (sub, iworst, backend, me_gpu[iworst], CPU_BACKEND,
                               me_cpu[iworst], worst))
        if not any(v != 0 for v in me_cpu):
            problems.append('%s: all |M|^2 vanish' % sub)
        scaling = me_gpu[1] / me_gpu[0] / (0.12 / 0.10) ** 2 - 1 if me_gpu[0] else float('inf')
        if not abs(scaling) < 1e-9:
            problems.append('%s: |M|^2 not ~ alpha_s^2 (ratio - 1 = %.1e)' % (sub, scaling))
        details.append('%s nDPF %d, %d flavours, max rel diff %.1e' % (sub, ndpf, nflavor, worst))
        print('  ' + details[-1])
        cpu.free()
        gpu.free()
    check('mssm_gpu_vs_cpu', not problems, '; '.join(problems[:3]) or '; '.join(details))


# ---------------------------------------------------------------------------
# 2. a subprocess with no good helicity
# ---------------------------------------------------------------------------

def check_zero_me(repo, backend, npoints=64):
    section('no good helicity: b b~ > ta+ ta- / z a with ymtau = 0 (%s)' % backend)
    name = 'zero_me'
    (sub,) = output(repo, name, 'sm', 'b b~ > ta+ ta- / z a', (CPU_BACKEND, backend))
    text = open(card(name)).read()
    zero_card, count = re.subn(r'(?mi)^(\s*15\s+)\S+(\s*#\s*ymtau)', r'\g<1>0.000000e+00\2', text)
    if count != 1:
        raise RuntimeError('no ymtau in %s' % card(name))
    zero_path = os.path.join(name, 'Cards', 'param_card_ymtau0.dat')
    open(zero_path, 'w').write(zero_card)
    rng = random.Random(5)
    # the helicity filtering runs once per library (and process): the GPU library only
    # ever sees the card with ymtau = 0, the CPU library the default one (|M|^2 != 0)
    gpu = Umami(backend, library(name, sub, backend), zero_path)
    cpu = Umami(CPU_BACKEND, library(name, sub, CPU_BACKEND), card(name))
    points = [rambo(gpu.masses[2:], 500., rng, gpu.masses[:2]) for _ in range(npoints)]
    rnd = [rng.random() for _ in range(npoints)]
    outputs = (OUT_MATRIX_ELEMENT, OUT_HELICITY_INDEX, OUT_COLOR_INDEX)
    me, hel, col = gpu(points, random_helicity=rnd, random_color=rnd, outputs=outputs)
    (me_default,) = cpu(points)
    problems = []
    if any(v != 0 for v in me):
        problems.append('|M|^2 not 0: %s' % me[:4])
    if set(hel) != {-1} or not set(col) <= {-1, 0}:
        problems.append('helicities %s, colours %s (expected -1, and -1 or 0)'
                        % (sorted(set(hel))[:4], sorted(set(col))[:4]))
    if not all(v > 0 for v in me_default):
        problems.append('|M|^2 with the default card not > 0: %s' % me_default[:4])
    gpu.free()
    cpu.free()
    check('zero_me', not problems, '; '.join(problems) or
          '%d events: |M|^2 = 0, helicity -1, colour %s (default card: %.3e)'
          % (npoints, sorted(set(col)), me_default[0]))


# ---------------------------------------------------------------------------
# 3. a non-blocking stream
# ---------------------------------------------------------------------------

def check_nonblocking_stream(repo, backend, npoints=16384, ncalls=4):
    section('g g > t t~ g on a non-blocking stream (%s)' % backend)
    name = 'ggttxg'
    (sub,) = output(repo, name, 'sm', 'g g > t t~ g', (backend,))
    rng = random.Random(3)
    gpu = Umami(backend, library(name, sub, backend), card(name))
    points = [rambo(gpu.masses[2:], 1000., rng) for _ in range(npoints)]
    rnd = [rng.random() for _ in range(npoints)]
    outputs = (OUT_MATRIX_ELEMENT, OUT_HELICITY_INDEX)
    stream = gpu.memory.nonblocking_stream()
    # the first call on the non-blocking stream: it also runs the helicity filtering
    calls = [gpu(points, random_helicity=rnd, outputs=outputs, stream=stream)]
    reference = gpu(points, random_helicity=rnd, outputs=outputs)  # default stream
    calls += [gpu(points, random_helicity=rnd, outputs=outputs, stream=stream)
              for _ in range(ncalls - 1)]
    gpu.memory.stream_destroy(stream)
    gpu.free()
    me_ref, hel_ref = reference
    problems = []
    for icall, (me, hel) in enumerate(calls):
        bad_me = sum(1 for a, b in zip(me, me_ref) if not abs(a - b) <= 1e-12 * abs(b))
        bad_hel = sum(1 for a, b in zip(hel, hel_ref) if a != b)
        if bad_me or bad_hel:
            problems.append('call %d: %d |M|^2 and %d helicities differ' % (icall, bad_me, bad_hel))
    if not all(v > 0 for v in me_ref) or min(hel_ref) < 0:
        problems.append('default stream: |M|^2 <= 0 or no helicity')
    check('nonblocking_stream', not problems, '; '.join(problems[:3]) or
          '%d calls of %d events identical to the default stream' % (ncalls, npoints))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--repo', required=True)
    parser.add_argument('--backend', required=True)
    parser.add_argument('--only', choices=CHECKS,
                        help='run this check here and write its result to NAME.json')
    args = parser.parse_args()
    if args.only:
        step = globals()['check_' + args.only]
        try:
            step(args.repo, args.backend)
        except Exception as error:
            check(args.only, False, 'error: %s' % error)
        json.dump(results, open(args.only + '.json', 'w'))
        return 0
    if args.backend not in GPU_BACKENDS:
        parser.error('the checks compare a GPU backend (%s) with the CPU one'
                     % ', '.join(GPU_BACKENDS))
    start = time.time()
    summary, failures = {}, []
    for name in CHECKS:
        if os.path.exists(name + '.json'):
            os.remove(name + '.json')
        status = subprocess.call([sys.executable, os.path.abspath(__file__), '--repo', args.repo,
                                  '--backend', args.backend, '--only', name])
        try:
            outcome = json.load(open(name + '.json'))
        except (OSError, ValueError):
            outcome = {name: (False, 'crashed (exit status %d)' % status)}
            print('FAIL %s: %s' % (name, outcome[name][1]), flush=True)
        for key, (ok, detail) in outcome.items():
            summary[key] = ('ok' if ok else 'FAILED') + ' (%s)' % detail
            if not ok:
                failures.append(key)
    summary['node'] = os.uname().nodename
    summary['gpu'] = os.environ.get('GPU_NAME', 'none')
    summary['backend'] = args.backend
    summary['walltime'] = '%ds' % (time.time() - start)
    with open('summary.txt', 'w') as f:
        for key, value in summary.items():
            f.write('%s=%s\n' % (key, value))
    print('\n%d check(s) failed: %s' % (len(failures), ', '.join(failures)) if failures
          else '\nall checks passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
