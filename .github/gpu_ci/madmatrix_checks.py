#!/usr/bin/env python3
"""The GPU backend of madmatrix against its scalar CPU backend, with check_sa.exe.

Run by madmatrix_gpu.sh (job madmatrix of gpu_runner_ci.yml) in its run directory.
Every evaluation is a ``check_sa.exe perf`` run of the standalone output, one executable
per backend (check_sa_<backend>.exe), on the same events: the momenta, alpha_s, flavour
and random numbers of every event go in as raw binary files (--momenta, --alphas,
--flavors, --rnd-hel, --rnd-col) and the |M|^2, helicities and colours come out the
same way (--dump-me, --dump-hel, --dump-col). An abort of the GPU runtime fails the run
of check_sa.exe, so only the check it belongs to.

1. mssm_gpu_vs_cpu: MSSM ``p p > go go`` (both subprocesses), whose squark-quark-gluino
   couplings are running flavour couplings (nDPF > 0, gathered per event into
   dpf_value), on RAMBO points with a random alpha_s and flavour per event: the GPU
   |M|^2 equal the CPU ones event by event (the GPU gather used to address its one-event
   dpf_value with the offset of the event in the all-events coupling buffer, out of
   bounds past the first event page), and scale as alpha_s^2 (--alphas 0.10 and 0.12).
2. zero_me: ``b b~ > ta+ ta- / z a`` (s-channel Higgs only): |M|^2 > 0 with the default
   param card, then the tau Yukawa coupling set to 0, so that no helicity survives the
   helicity filtering: both backends return |M|^2 = 0, no helicity and no colour (-1),
   never stale memory; the GPU used to abort on the invalid launch configuration
   (gridDim.y = 0) of the helicity loop.
   zero_me_good_hel: ``p p > w+ j``, the subprocess that folds the crossings
   (P1_gQ_wpQ), with a good-helicity list: a first call at alpha_s = 0.118 (|M|^2 > 0,
   a helicity and a colour for every extended flavour id), then a call at alpha_s = 0,
   where |M|^2 = 0 exactly and no helicity or colour (-1) may be reported on either
   backend: a -ffast-math build let the 0/0 of the selection pass and reported the first
   good helicity (relabelled for the crossed ids) and the first colour flow.
3. mssm_crossed_gpu_vs_cpu: MSSM ``p p > go go j``, the subprocess that folds crossings
   (P1_gQ_gogoQ: g q > go go q also serves q q~ > go go g, ...), with running flavour
   couplings: events drawn over every extended flavour id K*nmaxflavor + flavour
   (crossing-table row K), a random alpha_s and random numbers per event: the GPU |M|^2,
   helicities and colours equal the CPU ones event by event (the crossed helicity code
   of a moved leg included).
   crossed_massive_gpu_vs_cpu: SM ``u g > u z`` and ``u u~ > z g`` in one generation,
   folded into P1_ug_uz, whose crossing moves the Z into a massless slot (xonshell_tab 0):
   umami_initialize chooses the good helicities of that crossing on points of its own
   masses (choose_good_helicities, getGoodHel). The events of every row are on shell for
   that row, in a randomly boosted frame: the GPU |M|^2, helicities and colours equal
   the CPU ones event by event, so the GPU kept the same good-helicity list.
4. nonblocking_stream: ``g g > t t~ g``, 4 calls of 16384 events, with umami handed a
   non-blocking stream (--nonblocking-stream, as a torch side stream, which does not
   synchronise with the default stream) gives the same |M|^2 and helicities as on the
   default stream: every memset, kernel and copy of the matrix element must go to the
   stream umami is given, the helicity filtering of the first call included.

Any CPU backend but scalar (simd_128, ...) works too, to try the script without a GPU;
check 4 is then skipped. Writes summary.txt (key=value lines) in the current directory;
exit status 1 if a check fails.
"""

import argparse
import array
import glob
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time

GPU_BACKENDS = ('cuda', 'hip')
CPU_BACKEND = 'scalar'
CHECKS = ('mssm_gpu_vs_cpu', 'zero_me', 'zero_me_good_hel', 'mssm_crossed_gpu_vs_cpu',
          'crossed_massive_gpu_vs_cpu', 'nonblocking_stream')

results = {}


def section(title):
    print('\n=================== %s (%s)' % (title, time.strftime('%H:%M:%S')), flush=True)


def check(name, ok, detail):
    print('%s %s: %s' % ('PASS' if ok else 'FAIL', name, detail), flush=True)
    results[name] = (ok, detail)


def run(cmd, cwd=None, log=None):
    with open(log or os.devnull, 'w') as out:
        return subprocess.call(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT)


def write(path, typecode, values):
    """A raw binary file of check_sa.exe: 'd' double, 'I' 32-bit unsigned int."""
    with open(path, 'wb') as f:
        array.array(typecode, values).tofile(f)
    return os.path.abspath(path)


def read(path, typecode):
    """A file written by check_sa.exe: 'd' (--dump-me), 'i' (--dump-hel, --dump-col)."""
    values = array.array(typecode)
    with open(path, 'rb') as f:
        values.frombytes(f.read())
    return values


def ncross(proc_dir):
    """The number of rows of the crossing table of a subprocess (1: no crossing)."""
    tables = open(os.path.join(proc_dir, 'ProcessTables.h')).read()
    return int(re.search(r'constexpr int ncross = (\d+);', tables).group(1))


def table(proc_dir, name):
    """A host table of ProcessTables.h (xonshell_tab, xmass_slot_tab), as a list."""
    tables = open(os.path.join(proc_dir, 'ProcessTables.h')).read()
    match = re.search(r'constexpr int %s\[[^]]*\] = \{([^}]*)\};' % name, tables)
    if not match:
        raise RuntimeError('no %s in %s/ProcessTables.h' % (name, proc_dir))
    return [int(v) for v in match.group(1).split(',')]


def boosted_two_to_two(rng, masses, energy):
    """An on-shell 2 > 2 point, legs (E, px, py, pz): the beams along z in the partonic
    rest frame at sqrt(s) = energy, then boosted along a random axis (|beta| < 0.9)."""
    def pcm(m_a, m_b):
        s = energy ** 2
        return math.sqrt(max((s - (m_a + m_b) ** 2) * (s - (m_a - m_b) ** 2), 0.)) / (2 * energy)
    def leg(m, p):
        return [math.sqrt(m * m + sum(c * c for c in p))] + p
    p_in, p_out = pcm(masses[0], masses[1]), pcm(masses[2], masses[3])
    cos, phi = rng.uniform(-1, 1), rng.uniform(0, 2 * math.pi)
    sin = math.sqrt(1 - cos * cos)
    out = [p_out * sin * math.cos(phi), p_out * sin * math.sin(phi), p_out * cos]
    legs = [leg(masses[0], [0., 0., p_in]), leg(masses[1], [0., 0., -p_in]),
            leg(masses[2], out), leg(masses[3], [-c for c in out])]
    cos, phi, beta = rng.uniform(-1, 1), rng.uniform(0, 2 * math.pi), rng.uniform(0, 0.9)
    sin = math.sqrt(1 - cos * cos)
    b = [beta * sin * math.cos(phi), beta * sin * math.sin(phi), beta * cos]
    gamma = 1 / math.sqrt(1 - beta * beta)
    boosted = []
    for e, *p in legs:
        bp = sum(bi * pi for bi, pi in zip(b, p))
        boosted.append([gamma * (e + bp)] +
                       [pi + bi * (gamma * gamma / (gamma + 1) * bp + gamma * e) for bi, pi in zip(b, p)])
    return boosted


def output(repo, name, model, process, backends, select=None, defines=()):
    """output standalone + make for these backends in every subprocess (those for which
    select(proc_dir) is true, if given), keeping the check_sa.exe of each as
    check_sa_<backend>.exe: the subprocess directories. defines: multiparticle
    definitions ('b1 = g u~') made before the generation."""
    with open(name + '.mg5', 'w') as f:
        f.write('import model %s\n' % model)
        f.write(''.join('define %s\n' % define for define in defines))
        f.write('generate %s\noutput standalone %s -f\n' % (process, name))
    if run([sys.executable, os.path.join(repo, 'bin', 'madgraph'), name + '.mg5'],
           log=name + '.log') != 0:
        raise RuntimeError('madgraph failed, see %s.log' % name)
    proc_dirs = sorted(glob.glob(os.path.join(name, 'SubProcesses', 'P*')))
    if not proc_dirs:
        raise RuntimeError('no subprocess directory for %s' % process)
    if select:
        proc_dirs = [proc_dir for proc_dir in proc_dirs if select(proc_dir)]
        if not proc_dirs:
            raise RuntimeError('no selected subprocess directory for %s' % process)
    jobs = os.environ.get('SLURM_CPUS_PER_TASK', '4')
    for proc_dir in proc_dirs:
        for backend in backends:
            log = '%s_%s_%s_make.log' % (name, os.path.basename(proc_dir), backend)
            # check_sa.exe is relinked for the backend of each make (its libraries carry
            # the backend in their name)
            if run(['make', '-j', jobs, 'BACKEND=' + backend, 'FPTYPE=d'],
                   cwd=proc_dir, log=log) != 0:
                raise RuntimeError('make failed, see %s' % log)
            shutil.copy(os.path.join(proc_dir, 'check_sa.exe'),
                        os.path.join(proc_dir, 'check_sa_%s.exe' % backend))
    return proc_dirs


def check_sa(proc_dir, backend, args, log):
    """check_sa_<backend>.exe perf <args> in proc_dir (it reads ../../Cards/param_card.dat)."""
    cmd = ['./check_sa_%s.exe' % backend, 'perf'] + [str(a) for a in args]
    status = run(cmd, cwd=proc_dir, log=log)
    if status != 0:
        raise RuntimeError('%s failed (exit status %d), see %s' % (' '.join(cmd), status, log))


def dump(prefix, what):
    return os.path.abspath('%s_%s.bin' % (prefix, what))


# ---------------------------------------------------------------------------
# 1. running flavour couplings (MSSM p p > go go): GPU against CPU
# ---------------------------------------------------------------------------

def check_mssm_gpu_vs_cpu(backend, repo, nevt=1024):
    section('MSSM p p > go go, running flavour couplings: %s against %s' % (backend, CPU_BACKEND))
    name = 'mssm_gogo'
    proc_dirs = output(repo, name, 'MSSM_SLHA2', 'p p > go go', (CPU_BACKEND, backend))
    rng = random.Random(11)
    problems, details = [], []
    for proc_dir in proc_dirs:
        sub = os.path.basename(proc_dir)
        data = open(os.path.join(proc_dir, 'ProcessData.h')).read()
        ndpf = int(re.search(r'constexpr int nDPF = (\d+);', data).group(1))
        nflavor = int(re.search(r'constexpr int nmaxflavor = (\d+);', data).group(1))
        if ndpf == 0:
            problems.append('%s: nDPF = 0, no running flavour coupling to check' % sub)
        prefix = '%s_%s' % (name, sub)
        # one batch of nevt events, the same on both backends
        batch = [1, nevt, 1]
        momenta = dump(prefix, 'momenta')
        check_sa(proc_dir, CPU_BACKEND, ['--energy', 3000, '--dump-momenta', momenta] + batch,
                 prefix + '_rambo.log')
        alphas = write(prefix + '_alphas.bin', 'd', [rng.uniform(0.09, 0.14) for _ in range(nevt)])
        flavors = write(prefix + '_flavors.bin', 'I', [rng.randrange(nflavor) for _ in range(nevt)])
        me = {}
        for tag, run_backend, alphas_arg in (('cpu', CPU_BACKEND, alphas), ('gpu', backend, alphas),
                                             ('gpu010', backend, '0.10'), ('gpu012', backend, '0.12')):
            out = dump(prefix, 'me_' + tag)
            check_sa(proc_dir, run_backend, ['--momenta', momenta, '--alphas', alphas_arg,
                                             '--flavors', flavors, '--dump-me', out] + batch,
                     '%s_%s.log' % (prefix, tag))
            me[tag] = read(out, 'd')
        if not len(me['cpu']) == len(me['gpu']) == nevt:
            problems.append('%s: %d CPU and %d GPU |M|^2 for %d events'
                            % (sub, len(me['cpu']), len(me['gpu']), nevt))
        rels = [abs(a - b) / abs(a) if a != 0 else (0. if b == 0 else float('inf'))
                for a, b in zip(me['cpu'], me['gpu'])]
        # every event that is not within the tolerance, a nan included (it fails any
        # comparison, so it must not go through a running maximum, which would drop it)
        bad = [ievt for ievt, rel in enumerate(rels) if not rel < 1e-9]
        worst = max((rel for rel in rels if not math.isnan(rel)), default=0.)
        if bad:
            first = bad[0]
            problems.append('%s: %d of %d events differ, first event %d %s %.10e vs %s %.10e (rel %.1e)'
                            % (sub, len(bad), nevt, first, backend, me['gpu'][first],
                               CPU_BACKEND, me['cpu'][first], rels[first]))
        if not any(v != 0 for v in me['cpu']):
            problems.append('%s: all |M|^2 vanish' % sub)
        # |M|^2 ~ alpha_s^2, event by event
        ratio = (0.12 / 0.10) ** 2
        scaling = [b / a / ratio - 1 if a != 0 else float('inf')
                   for a, b in zip(me['gpu010'], me['gpu012'])]
        bad = [ievt for ievt, s in enumerate(scaling) if not abs(s) < 1e-9]
        if bad or len(scaling) != nevt:
            problems.append('%s: |M|^2 not ~ alpha_s^2 in %d of %d events (first: ratio - 1 = %.1e)'
                            % (sub, len(bad), nevt, scaling[bad[0]] if bad else float('nan')))
        details.append('%s nDPF %d, %d flavours, max rel diff %.1e' % (sub, ndpf, nflavor, worst))
        print('  ' + details[-1])
    check('mssm_gpu_vs_cpu', not problems, '; '.join(problems[:3]) or '; '.join(details))


# ---------------------------------------------------------------------------
# 2. a subprocess with no good helicity
# ---------------------------------------------------------------------------

def check_zero_me(backend, repo, nevt=64):
    section('no good helicity: b b~ > ta+ ta- / z a with ymtau = 0 (%s, %s)' % (backend, CPU_BACKEND))
    name = 'zero_me'
    (proc_dir,) = output(repo, name, 'sm', 'b b~ > ta+ ta- / z a', (CPU_BACKEND, backend))
    card = os.path.join(name, 'Cards', 'param_card.dat')
    text = open(card).read()
    zero_card, count = re.subn(r'(?mi)^(\s*15\s+)\S+(\s*#\s*ymtau)', r'\g<1>0.000000e+00\2', text)
    if count != 1:
        raise RuntimeError('no ymtau in %s' % card)
    rng = random.Random(5)
    batch = [1, nevt, 1]
    momenta = dump(name, 'momenta')
    check_sa(proc_dir, CPU_BACKEND, ['--energy', 500, '--dump-momenta', momenta,
                                     '--dump-me', dump(name, 'me_default')] + batch,
             name + '_default.log')
    me_default = read(dump(name, 'me_default'), 'd')
    rnd = write(name + '_rnd.bin', 'd', [rng.random() for _ in range(nevt)])
    # check_sa.exe reads the param card at every run: from here on ymtau = 0 (the
    # helicity filtering runs at the first call of each run)
    open(card, 'w').write(zero_card)
    problems, seen = [], []
    for run_backend in (backend, CPU_BACKEND):
        prefix = '%s_%s' % (name, run_backend)
        check_sa(proc_dir, run_backend, ['--momenta', momenta, '--rnd-hel', rnd, '--rnd-col', rnd,
                                         '--dump-me', dump(prefix, 'me'),
                                         '--dump-hel', dump(prefix, 'hel'),
                                         '--dump-col', dump(prefix, 'col')] + batch,
                 prefix + '.log')
        me, hel, col = (read(dump(prefix, 'me'), 'd'), read(dump(prefix, 'hel'), 'i'),
                        read(dump(prefix, 'col'), 'i'))
        if len(me) != nevt or any(v != 0 for v in me):
            problems.append('%s: |M|^2 not 0: %s' % (run_backend, list(me[:4])))
        if set(hel) != {-1} or set(col) != {-1}:
            problems.append('%s: helicities %s, colours %s (expected -1)'
                            % (run_backend, sorted(set(hel))[:4], sorted(set(col))[:4]))
    if len(me_default) != nevt or not all(v > 0 for v in me_default):
        problems.append('|M|^2 with the default card not > 0: %s' % list(me_default[:4]))
    check('zero_me', not problems, '; '.join(problems) or
          '%d events: |M|^2 = 0, helicity and colour -1 on %s and %s (default card: %.3e)'
          % (nevt, backend, CPU_BACKEND, me_default[0]))


def check_zero_me_good_hel(backend, repo, nevt=64):
    section('|M|^2 = 0 with good helicities: p p > w+ j at alpha_s = 0 (%s, %s)'
            % (backend, CPU_BACKEND))
    name = 'zero_me_wj'
    (proc_dir,) = output(repo, name, 'sm', 'p p > w+ j', (CPU_BACKEND, backend),
                         select=lambda proc_dir: ncross(proc_dir) > 1)
    sub = os.path.basename(proc_dir)
    data = open(os.path.join(proc_dir, 'ProcessData.h')).read()
    nflavor = int(re.search(r'constexpr int nmaxflavor = (\d+);', data).group(1))
    nids = nflavor * ncross(proc_dir)
    rng = random.Random(23)
    # two calls (iterations) of nevt events in one run: the helicity filtering of the
    # first call, at alpha_s = 0.118, keeps its good helicities for the second, at 0
    batch = [1, nevt, 2]
    momenta = dump(name, 'momenta')
    check_sa(proc_dir, CPU_BACKEND, ['--energy', 1000, '--dump-momenta', momenta] + batch,
             name + '_rambo.log')
    ids = [ievt % nids for ievt in range(nevt)] * 2
    inputs = ['--momenta', momenta,
              '--alphas', write(name + '_alphas.bin', 'd', [0.118] * nevt + [0.] * nevt),
              '--flavors', write(name + '_flavors.bin', 'I', ids),
              '--rnd-hel', write(name + '_rndhel.bin', 'd', [rng.random() for _ in range(2 * nevt)]),
              '--rnd-col', write(name + '_rndcol.bin', 'd', [rng.random() for _ in range(2 * nevt)])]
    problems = []
    for run_backend in (backend, CPU_BACKEND):
        prefix = '%s_%s' % (name, run_backend)
        files = {what: dump(prefix, what) for what in ('me', 'hel', 'col')}
        check_sa(proc_dir, run_backend, inputs + ['--dump-me', files['me'], '--dump-hel', files['hel'],
                                                  '--dump-col', files['col']] + batch,
                 prefix + '.log')
        me, hel, col = read(files['me'], 'd'), read(files['hel'], 'i'), read(files['col'], 'i')
        if not len(me) == len(hel) == len(col) == 2 * nevt:
            problems.append('%s: output sizes %d/%d/%d for %d events'
                            % (run_backend, len(me), len(hel), len(col), 2 * nevt))
            continue
        first = [ievt for ievt in range(nevt) if not (me[ievt] > 0 and hel[ievt] >= 0 and col[ievt] >= 0)]
        if first:
            ievt = first[0]
            problems.append('%s: %d events of the alpha_s = 0.118 call without |M|^2 > 0, a helicity '
                            'and a colour, first event %d (id %d): %.3e, helicity %d, colour %d'
                            % (run_backend, len(first), ievt, ids[ievt], me[ievt], hel[ievt], col[ievt]))
        zero = [ievt for ievt in range(nevt, 2 * nevt) if not (me[ievt] == 0 and hel[ievt] == -1 and col[ievt] == -1)]
        if zero:
            ievt = zero[0]
            problems.append('%s: %d events of the alpha_s = 0 call not |M|^2 = 0, helicity -1 and '
                            'colour -1, first event %d (id %d): %.3e, helicity %d, colour %d'
                            % (run_backend, len(zero), ievt - nevt, ids[ievt], me[ievt], hel[ievt], col[ievt]))
    check('zero_me_good_hel', not problems, '; '.join(problems[:3]) or
          '%s, %d extended ids: |M|^2 > 0 at alpha_s = 0.118, then |M|^2 = 0, helicity and colour -1 '
          'at alpha_s = 0 on %s and %s' % (sub, nids, backend, CPU_BACKEND))


# ---------------------------------------------------------------------------
# 3. crossed subprocesses with running flavour couplings (MSSM p p > go go j)
# ---------------------------------------------------------------------------

def crossed_gpu_vs_cpu(check_name, backend, name, proc_dirs, energy, seed, nevt, own_points=False):
    """Events over every extended flavour id K*nmaxflavor + flavour of each subprocess
    (crossing-table row K), a random alpha_s and random numbers per event: the GPU |M|^2,
    helicities and colours equal the CPU ones event by event. own_points: the momenta of
    each event on shell for its own row (2 > 2 only), else the RAMBO points of the base."""
    rng = random.Random(seed)
    problems, details = [], []
    for proc_dir in proc_dirs:
        sub = os.path.basename(proc_dir)
        data = open(os.path.join(proc_dir, 'ProcessData.h')).read()
        ndpf = int(re.search(r'constexpr int nDPF = (\d+);', data).group(1))
        nflavor = int(re.search(r'constexpr int nmaxflavor = (\d+);', data).group(1))
        nids = nflavor * ncross(proc_dir)
        prefix = '%s_%s' % (name, sub)
        batch = [1, nevt, 1]
        momenta = dump(prefix, 'momenta')
        check_sa(proc_dir, CPU_BACKEND, ['--energy', energy, '--dump-momenta', momenta] + batch,
                 prefix + '_rambo.log')
        # every extended id at least once, then random ones
        ids = list(range(nids)) + [rng.randrange(nids) for _ in range(nevt - nids)]
        rng.shuffle(ids)
        if own_points:
            # the base masses from the RAMBO points, then the masses of every row
            base = read(momenta, 'd')
            npar = len(base) // nevt // 4
            if npar != 4:
                raise RuntimeError('%s: own_points needs a 2 > 2, not %d legs' % (sub, npar))
            masses = [math.sqrt(max(base[4 * k] ** 2 - sum(c * c for c in base[4 * k + 1:4 * k + 4]), 0.))
                      for k in range(npar)]
            masses = [m if m > 1e-6 * energy else 0. for m in masses]
            slots = table(proc_dir, 'xmass_slot_tab')
            points = []
            for xid in ids:
                row = xid // nflavor
                for p in boosted_two_to_two(rng, [masses[slots[row * npar + k]] for k in range(npar)], energy):
                    points.extend(p)
            momenta = write(prefix + '_own_momenta.bin', 'd', points)
        inputs = ['--momenta', momenta,
                  '--alphas', write(prefix + '_alphas.bin', 'd', [rng.uniform(0.09, 0.14) for _ in range(nevt)]),
                  '--flavors', write(prefix + '_flavors.bin', 'I', ids),
                  '--rnd-hel', write(prefix + '_rndhel.bin', 'd', [rng.random() for _ in range(nevt)]),
                  '--rnd-col', write(prefix + '_rndcol.bin', 'd', [rng.random() for _ in range(nevt)])]
        out = {}
        for tag, run_backend in (('cpu', CPU_BACKEND), ('gpu', backend)):
            files = {what: dump(prefix, '%s_%s' % (what, tag)) for what in ('me', 'hel', 'col')}
            check_sa(proc_dir, run_backend, inputs + ['--dump-me', files['me'], '--dump-hel', files['hel'],
                                                      '--dump-col', files['col']] + batch,
                     '%s_%s.log' % (prefix, tag))
            out[tag] = (read(files['me'], 'd'), read(files['hel'], 'i'), read(files['col'], 'i'))
        (me_cpu, hel_cpu, col_cpu), (me_gpu, hel_gpu, col_gpu) = out['cpu'], out['gpu']
        if not len(me_cpu) == len(me_gpu) == len(hel_gpu) == len(col_gpu) == nevt:
            problems.append('%s: output sizes %d/%d/%d/%d for %d events'
                            % (sub, len(me_cpu), len(me_gpu), len(hel_gpu), len(col_gpu), nevt))
            continue
        rels = [abs(a - b) / abs(a) if a != 0 else (0. if b == 0 else float('inf'))
                for a, b in zip(me_cpu, me_gpu)]
        bad = [ievt for ievt, rel in enumerate(rels) if not rel < 1e-9]
        if bad:
            first = bad[0]
            problems.append('%s: %d of %d events differ, first event %d (id %d) %s %.10e vs %s %.10e'
                            % (sub, len(bad), nevt, first, ids[first], backend, me_gpu[first],
                               CPU_BACKEND, me_cpu[first]))
        bad_hel = [ievt for ievt in range(nevt) if hel_cpu[ievt] != hel_gpu[ievt]]
        bad_col = [ievt for ievt in range(nevt) if col_cpu[ievt] != col_gpu[ievt]]
        if bad_hel or bad_col:
            first = (bad_hel or bad_col)[0]
            problems.append('%s: %d helicities and %d colours differ, first event %d (id %d): '
                            'helicity %d vs %d, colour %d vs %d'
                            % (sub, len(bad_hel), len(bad_col), first, ids[first], hel_gpu[first],
                               hel_cpu[first], col_gpu[first], col_cpu[first]))
        # every crossing row evaluated: some event of each row has |M|^2 > 0
        rows = {ids[ievt] // nflavor for ievt in range(nevt) if me_cpu[ievt] > 0}
        if len(rows) != nids // nflavor:
            problems.append('%s: |M|^2 > 0 only for crossing rows %s of %d'
                            % (sub, sorted(rows), nids // nflavor))
        worst = max((rel for rel in rels if not math.isnan(rel)), default=0.)
        details.append('%s nDPF %d, %d extended ids, max rel diff %.1e' % (sub, ndpf, nids, worst))
        print('  ' + details[-1])
    check(check_name, not problems, '; '.join(problems[:3]) or '; '.join(details))


def check_mssm_crossed_gpu_vs_cpu(backend, repo, nevt=1024):
    section('MSSM p p > go go j, crossed subprocesses: %s against %s' % (backend, CPU_BACKEND))
    name = 'mssm_gogoj'
    # only the subprocesses that fold crossings (the others are check 1 with one more jet)
    proc_dirs = output(repo, name, 'MSSM_SLHA2', 'p p > go go j', (CPU_BACKEND, backend),
                       select=lambda proc_dir: ncross(proc_dir) > 1)
    crossed_gpu_vs_cpu('mssm_crossed_gpu_vs_cpu', backend, name, proc_dirs, 3000, 17, nevt)


def check_crossed_massive_gpu_vs_cpu(backend, repo, nevt=1024):
    section('u g > u z with u u~ > z g, a massive leg crossed into a massless slot: %s against %s'
            % (backend, CPU_BACKEND))
    name = 'crossed_massive'
    # u u~ > z g is folded onto u g > u z: its Z sits in the slot of the outgoing u
    proc_dirs = output(repo, name, 'sm', 'u b1 > c1 d1 QED=1 QCD=1', (CPU_BACKEND, backend),
                       select=lambda proc_dir: 0 in table(proc_dir, 'xonshell_tab'),
                       defines=('b1 = g u~', 'c1 = u z', 'd1 = z g'))
    crossed_gpu_vs_cpu('crossed_massive_gpu_vs_cpu', backend, name, proc_dirs, 1000, 29, nevt,
                       own_points=True)


# ---------------------------------------------------------------------------
# 4. a non-blocking stream
# ---------------------------------------------------------------------------

def check_nonblocking_stream(backend, repo, ncalls=4):
    section('g g > t t~ g on a non-blocking stream (%s)' % backend)
    if backend not in GPU_BACKENDS:
        check('nonblocking_stream', True, 'skipped: %s is not a GPU backend' % backend)
        return
    name = 'ggttxg'
    (proc_dir,) = output(repo, name, 'sm', 'g g > t t~ g', (backend,))
    # ncalls calls (iterations) of 64 x 256 = 16384 events, the RAMBO points of check_sa.exe
    # (the same in both runs: one seed per iteration)
    batch = [64, 256, ncalls]
    nevt = 64 * 256 * ncalls
    rng = random.Random(3)
    rnd = write(name + '_rnd.bin', 'd', [rng.random() for _ in range(nevt)])
    me, hel = {}, {}
    for tag, extra in (('default', []), ('nonblocking', ['--nonblocking-stream'])):
        prefix = '%s_%s' % (name, tag)
        check_sa(proc_dir, backend, extra + ['--energy', 1000, '--rnd-hel', rnd,
                                             '--dump-me', dump(prefix, 'me'),
                                             '--dump-hel', dump(prefix, 'hel')] + batch,
                 prefix + '.log')
        me[tag], hel[tag] = read(dump(prefix, 'me'), 'd'), read(dump(prefix, 'hel'), 'i')
    problems = []
    if not len(me['default']) == len(me['nonblocking']) == nevt:
        problems.append('%d and %d |M|^2 for %d events' % (len(me['default']), len(me['nonblocking']), nevt))
    for icall in range(ncalls):
        events = slice(icall * nevt // ncalls, (icall + 1) * nevt // ncalls)
        pairs = list(zip(me['nonblocking'][events], me['default'][events]))
        bad_me = sum(1 for a, b in pairs if not abs(a - b) <= 1e-12 * abs(b))
        bad_hel = sum(1 for a, b in zip(hel['nonblocking'][events], hel['default'][events]) if a != b)
        if bad_me or bad_hel:
            problems.append('call %d: %d |M|^2 and %d helicities differ' % (icall, bad_me, bad_hel))
    if not all(v > 0 for v in me['default']) or min(hel['default'], default=-1) < 0:
        problems.append('default stream: |M|^2 <= 0 or no helicity')
    check('nonblocking_stream', not problems, '; '.join(problems[:3]) or
          '%d calls of %d events identical to the default stream' % (ncalls, nevt // ncalls))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--repo', required=True)
    parser.add_argument('--backend', required=True)
    parser.add_argument('--only', choices=CHECKS, action='append',
                        help='run this check only (repeatable)')
    args = parser.parse_args()
    if args.backend == CPU_BACKEND:
        parser.error('the checks compare a backend with %s: give another one' % CPU_BACKEND)
    start = time.time()
    for name in args.only or CHECKS:
        try:
            globals()['check_' + name](args.backend, args.repo)
        except Exception as error:
            check(name, False, 'error: %s' % error)
    summary = {key: ('ok' if ok else 'FAILED') + ' (%s)' % detail
               for key, (ok, detail) in results.items()}
    failures = [key for key, (ok, _detail) in results.items() if not ok]
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
