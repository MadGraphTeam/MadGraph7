#!/usr/bin/env python3
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Sep 2026) for project MadVfold (arXiv:2609.23500).

"""Timing tables for the Road B runs of one step, all modes side by side.

Usage:
    ./vfold/myTimingTable.py [-g GROUPING] [-f FORMAT] PREFIX

Example:
    ./vfold/myTimingTable.py Events/REF3/n200f111k32/step9h
    ./vfold/myTimingTable.py -g ctmc Events/REF3/n200f111k32/step9h
    ./vfold/myTimingTable.py --skip nonstd -f tex2 Events/REF7/GOLD/n200f841k64g1/step15d
    ./vfold/myTimingTable.py --skip nonstd --vecsizes 64,512,1024 -f tex2 Events/REF7/CUDA/n200f841k64g1/step15d
    ./vfold/myTimingTable.py --branch vfold5363 -f tex2 Events/REF2/CUDA/n200f841k64g1/step10b

PREFIX is the common path prefix of the per-mode run directories, e.g.
Events/REF3/n200f111k32/step9h for step9hZ, step9hS, ..., step9hCavx2.
The stdout format (default) writes the text table to stdout only; the txt
format writes the same table only to <dir>/timingTable_<step>.txt, e.g.
Events/REF3/n200f111k32/timingTable_step9h.txt, without printing it (only
errors and warnings are printed); the tex formats (tex1, tex2) are written
to a .tex file in Events/TEXn for a prefix in Events/REFn
(TEX/TEX2/TEX3 for REF/REF2/REF3). With --branch, e.g. vfold5363 (the code
branch of the runs), the branch is printed first in the title (underlined in
tex) and, after the short process name (the process directory name without
_vfgen, _vfalt or _stock), it prefixes the tex file name, e.g. in eebbbb_vfgen
eebbbb_vfold5363_CUDA_..._step10b.tex.

The script has two independent parts:
    1. READING: read_step(prefix) parses the logs into plain dictionaries;
    2. WRITING: build_table(...) turns them into a format-neutral table
        (header lines, column names, rows of numbers), which a renderer
        (render_text, render_tex) then writes out.
Rows are defined in GROUPINGS below as lists of raw MG5 timing labels, so
categories can be merged freely (MISC means 'every label not used by any
other row except Total').
"""

import argparse
import glob
import os
import re
import sys

#----------------------------------------------------------------------------
# Configuration
#----------------------------------------------------------------------------

# Directory suffix -> mode name, in column order. Missing modes are skipped.
MODES = [
    ('Z', 'F0'), ('S', 'F1'),
    ('Sn', 'F1[n]'), ('Sc', 'F1[c]'),  # step 14h: F1 also with the other MadLoop reduction library
    ('Su', 'F1[u]'),  # step 15e: F1 unoptimized, without the stock MadGraph7 Born cache and gscale
    ('T', 'F2'), ('F', 'F'),
    ('Cnone', 'C/none'), ('Csse4', 'C/sse4'), ('Cavx2', 'C/avx2'),
    ('C512y', 'C/512y'), ('C512z', 'C/512z'), ('Ccuda', 'C/cuda'),
    ('Ccudb', 'C/cudb'),  # cuda with BLAS colour sums (CUDACPP_RUNTIME_BLASCOLORSUM=1)
    ('Ccudbr', 'C/cudbr'),  # step 14n: BLAS colour sums for the Real libraries only
]

# Step 15e: directory suffixes of the non-standard columns (--skip nonstd)
NONSTANDARD = ('Sn', 'Sc', 'Su', 'Ccudb', 'Ccudbr')

# GPU columns: the only ones that accept vecsize scans
MODES_GPU = ('C/cuda', 'C/cudb', 'C/cudbr')

# Mode whose run is used for the kpoint and batching sub-headers
MODE_BATCHED = 'F'
MODE_BASELINE = 'F0'

MISC = 'MISC'  # placeholder: all raw labels not used by another row (except Total)

# Baseline rescaling: show MODE_BASELINE as 'F0*', i.e. rescaled to the
# kpoints of MODE_BATCHED, because F0 draws different random numbers and
# does a different amount of work for the same NEVENTS. Each raw label is
# rescaled by the kpoints that actually pay for it: 'virt' for the virtual
# (only virt kpoints compute it) or 'novi' (the default) for all others, as
# almost all of their time is in novi kpoints (step 14h: this is measured,
# also for the Born, and F0* then agrees with F1 if F0 and F have a different
# fraction of virt kpoints). The rescaled Total is the sum of the rescaled labels.
RESCALE_BASELINE = True
RESCALE_BY = {
    'Virtuals': 'virt',
}

# Row definitions: (row label, list of raw labels from 'Time spent in X : ...')
GROUPINGS = {
    'noctmc': [
        ('TOTAL', ['Total']),
        ('Born', ['Born']),
        ('Real', ['Reals_evaluation']),
        ('Virtual', ['Virtuals']),
        ('PS_Generation', ['PS_Generation']),
        ('MCsubtraction', ['MCsubtraction']),
        ('Delta_Pythia', ['Delta_Pythia']),
        ('Counter_terms', ['Counter_terms']),
        ('Integrated_CT', ['Integrated_CT']),
        ('Nbody_prefactor', ['Nbody_prefactor']),
        ('Bridge_crt_del', ['Bridge_create_delete']),
        ('Misc', MISC),
    ],
    'default': [
        ('TOTAL', ['Total']),
        ('Born', ['Born']),
        ('Real', ['Reals_evaluation']),
        ('Virtual', ['Virtuals']),
        ('PS_Generation', ['PS_Generation']),
        ('CT_and_MCsub', ['Counter_terms', 'Integrated_CT', 'MCsubtraction']),
        ('Delta_Pythia', ['Delta_Pythia']),
        ('Nbody_prefactor', ['Nbody_prefactor']),
        ('Bridge_crt_del', ['Bridge_create_delete']),
        ('Misc', MISC),
    ],
    'all': None,  # one row per raw label, in the order of the log
}

# Rows shown only if ALL mode C columns have one of their raw labels: the bridge
# create/delete timer (cudacpp GpuInit etc) only exists from step 11j on (before
# that its time is inside Misc), and some older C logs do not have it either.
# If the row is shown, the other columns print their value (0 in F0/F1/F2/F).
OPTIONAL_ROWS = ['Bridge_create_delete']

# Rows shown only if some column has a nonzero value for one of their raw
# labels: the MC@NLO-Delta timer (Pythia8 and Sudakov factors in
# complete_xmcsubt) only exists from step 11q on, and is 0 without Delta
# (before step 11q, or without Delta, that time is inside MCsubtraction)
NONZERO_ROWS = ['Delta_Pythia']

# Shorter row labels in tex, to keep the label column narrow
TEXLABELS = {'Bridge_create_delete': 'Bridge', 'Delta_Pythia': 'Delta'}

# vfold_cpu_time mode (step 11a, ENV_VFOLD_CPUTIME) -> short name for the cpu_timer row:
# 0 = getrusage (MG5 default, one syscall per timer call), 1 = clock_gettime
# monotonic (vDSO, no syscall), 2 = rdtsc
CPUTIMERS = {'0': 'rusage', '1': 'mono', '2': 'rdtsc'}

#----------------------------------------------------------------------------
# Part 1: READING
#----------------------------------------------------------------------------

RE_TIME = re.compile(r'^\s*Time spent in (\w+)\s*:\s*([-0-9.Ee+]+)', re.M)


def _grep_int(text, pattern, default=0):
    m = re.search(pattern, text)
    return int(m.group(1)) if m else default


def read_channel_log(path):
    """Parse one log_MINT2.txt: timings and generation/batching counters."""
    text = open(path, errors='replace').read()
    times = {}
    order = []
    for name, value in RE_TIME.findall(text):
        if name not in times:
            order.append(name)
        times[name] = times.get(name, 0.0) + float(value)
    c = {}
    c['ncand'] = _grep_int(text, r'another call to the function:\s*(\d+)')
    c['ev_novi'] = _grep_int(text, r'events generated, novi:\s*(\d+)')
    c['ev_virt'] = _grep_int(text, r'events generated, virt:\s*(\d+)')
    c['ev_born'] = _grep_int(text, r'events generated, born:\s*(\d+)')
    # 'Generation efficiencies: eff eff_virt' with eff = accepted/candidates
    # for novi(+born) and eff_virt for virt (-1 when there were none):
    # this is the only place the per-type candidate counts are recoverable.
    m = re.search(r'Generation efficiencies:\s*([-0-9.Ee+]+)\s+([-0-9.Ee+]+)', text)
    eff, eff_virt = (float(m.group(1)), float(m.group(2))) if m else (-1.0, -1.0)
    ev_nv = c['ev_novi'] + c['ev_born']
    c['cand_novi'] = round(ev_nv / eff) if eff > 0 else 0
    c['cand_virt'] = round(c['ev_virt'] / eff_virt) if eff_virt > 0 else 0
    m = re.search(r'born_batch_avg=\s*([-0-9.Ee+]+)\s+born_batch_ncalls=\s*(\d+)\s+'
                  r'real_batch_avg=\s*([-0-9.Ee+]+)\s+real_batch_ncalls=\s*(\d+)', text)
    if m:
        c['born_calls'] = int(m.group(2))
        c['born_entries'] = float(m.group(1)) * int(m.group(2))
        c['real_calls'] = int(m.group(4))
        c['real_entries'] = float(m.group(3)) * int(m.group(4))
    else:
        c['born_calls'] = c['real_calls'] = 0
        c['born_entries'] = c['real_entries'] = 0.0
    m = re.search(r'AV: (?:av|vfold)_kpoint_batch=\s*(\d+)', text)
    c['kbatch_log'] = m.group(1) if m else None
    # (vfold_bsf_vec and ENV_VFOLD_BSF_VEC were renamed vfold_bsf_fused/ENV_VFOLD_BSF_FUSED)
    m = re.search(r'AV: (?:av|vfold)_bsf_(?:fused|vec)=\s*([TF])', text)
    c['bsffused_log'] = {'T': '1', 'F': '0'}[m.group(1)] if m else None
    m = re.search(r'AV: (?:av|vfold)_sborn_bridge=\s*(\d+)', text)
    c['sborn_log'] = m.group(1) if m else None
    # any 'AV:' line at all means the log is from a Road B build, so a missing
    # 'AV: vfold_bsf_fused='/'vfold_sborn_bridge=' line means the switch did not exist
    c['has_av_log'] = bool(re.search(r'AV: (?:av|vfold)_', text))
    m = re.search(r'AV: (?:av|vfold)_sborn_gscale=\s*(\d+)', text)
    c['gscale_log'] = m.group(1) if m else None
    m = re.search(r'AV: (?:av|vfold)_vecsize_used=\s*(\d+)', text)
    c['vecsize_log'] = m.group(1) if m else None
    m = re.search(r'AV: (?:av|vfold)_cpu_time mode=\s*(\d+)', text)
    c['cputime_log'] = m.group(1) if m else None
    return times, order, c


# Names of the test variables in the logs before step 14s part 10/10 (now ENV_VFOLDTEST_*)
OLD_TEST_ENV = r'^(?:ENV_|ENV_VFOLD_)(NEVENTS|FOLDING|MCATNLODELTA|PHYSICS|REQACC|ISEED|MLREDUCTION|POLYCACHE|CHANNELS|MINT[01]_\w+|NOSHUFFLE)(?==)'


def read_env(logtxt):
    """ENV_xxx=value and CUDACPP_xxx=value lines dumped by myLaunch.sh into LOG.txt (first occurrence)."""
    env = {}
    if os.path.isfile(logtxt):
        for line in open(logtxt, errors='replace'):
            line = re.sub(r'^ENV_AV_', 'ENV_VFOLD_', line.rstrip('\n'))  # old logs: ENV_AV_
            line = re.sub(OLD_TEST_ENV, r'ENV_VFOLDTEST_\1', line)  # old logs: test variables before step 14s part 10/10
            m = re.match(r'^((?:ENV|CUDACPP)_\w+)=(.*)$', line)
            if m and m.group(1) not in env:
                env[m.group(1)] = m.group(2)
    return env


def read_mode(rundir):
    """Sum timings and counters over all channels (P*/GF*) of one run."""
    logs = sorted(glob.glob(os.path.join(rundir, 'P*', 'GF*', 'log_MINT2.txt')))
    if not logs:
        return None
    times, order, cnt = {}, [], {}
    for log in logs:
        t, o, c = read_channel_log(log)
        for k in o:
            if k not in times:
                order.append(k)
            times[k] = times.get(k, 0.0) + t[k]
        for k, v in c.items():
            if isinstance(v, (int, float)):
                cnt[k] = cnt.get(k, 0) + v
            else:
                cnt.setdefault(k, v)
    cnt['nchannels'] = len(logs)
    return {'dir': rundir, 'times': times, 'order': order, 'counters': cnt,
            'env': read_env(os.path.join(rundir, 'LOG.txt'))}


def read_step(prefix, skip=(), vecsizes=None):
    """Read all the modes of one step: returns an ordered dict name -> data.
    If PREFIX is itself a single run directory (e.g. Events/run_01), it is
    read as one column, named after its ENV_VFOLD_ME_MODE (and backend)."""
    data = {}
    for suffix, name in MODES:
        if suffix in skip:  # step 15e (without Sn and Sc, the column F1 keeps its plain name)
            continue
        d = read_mode(prefix + suffix)
        if d is not None:
            data[name] = d
        # Scans of the cudacpp_vecsize (ENV_VFOLD_VECSIZE_USED) for the cuda
        # backend only (with or without BLAS), e.g. <prefix>Ccuda32, <prefix>Ccudb1024:
        # one column each, keyed 'C/cuda#32' etc (shown as 'C/cuda', the value being
        # printed in the cudacpp_vecsize line), in increasing order. Numbered directories
        # of the other backends (e.g. <prefix>Cavx232) are refused: on CPUs the
        # vecsize only adds padding, and such names would be ambiguous.
        if name.startswith('C/'):
            scan = []
            for path in glob.glob(glob.escape(prefix + suffix) + '[0-9]*'):
                tail = path[len(prefix + suffix):]
                if re.fullmatch(r'\d+', tail):
                    scan.append((int(tail), path))
            if scan and name not in MODES_GPU:
                sys.exit('ERROR! vecsize scans are only supported for cuda, not for %s: %s'
                         % (name, ', '.join(sorted(p for n, p in scan))))
            for num, path in sorted(scan):
                if vecsizes is not None and num not in vecsizes:  # step 15e: --vecsizes
                    continue
                d = read_mode(path)
                if d is not None:
                    data['%s#%d' % (name, num)] = d
    # Step 14h: F1 with both MadLoop reduction libraries: show them as F1[n] and F1[c], in this order
    if 'F1' in data and ('F1[n]' in data or 'F1[c]' in data):
        lib = _mlreduction(data['F1'])
        data['F1[%s]' % lib[0]] = data.pop('F1')
        order = [name for _, name in MODES if name not in ('F1[n]', 'F1[c]')]
        first = order.index('F1')
        names = sorted(data, key=lambda n: first + (0.1 if n == 'F1[n]' else 0.2) if n in ('F1[n]', 'F1[c]')
                       else (order.index(n.split('#')[0])))
        data = {n: data[n] for n in names}
    if not data:
        d = read_mode(prefix)
        if d is not None:
            mode = d['env'].get('ENV_VFOLD_ME_MODE') or os.path.basename(prefix)
            backend = d['env'].get('ENV_VFOLD_CUDACPP_BACKEND')
            if mode == 'C' and backend == 'cuda' and d['env'].get('CUDACPP_RUNTIME_BLASCOLORSUM'):
                backend = 'cudb'  # cuda with BLAS colour sums
                if d['env'].get('CUDACPP_RUNTIME_BLASCOLORSUM_VFOLDBORN') == '':
                    backend = 'cudbr'  # step 14n: only for the Real libraries
            data[mode + ('/' + backend if mode == 'C' and backend else '')] = d
    return data


#----------------------------------------------------------------------------
# Part 2: WRITING
#----------------------------------------------------------------------------

def _noswitch(cnt, key):
    """True if the code of this step has no such switch: its MINT2 log has
    'AV:' lines (so it is a Road B build) but none for this switch. myLaunch.sh
    exports ENV_VFOLD_BSF_FUSED and ENV_VFOLD_SBORN_BRIDGE for every step, while the
    Fortran only reads them from steps 9f and 9i on."""
    return cnt.get(key) is None and bool(cnt.get('has_av_log'))


def _run_params(data):
    """Step-level parameters for the header, from the batched mode (or any)."""
    ref = data.get(MODE_BATCHED) or next(iter(data.values()))
    env, cnt = ref['env'], ref['counters']
    nev = env.get('ENV_VFOLDTEST_NEVENTS') or str(cnt['ev_novi'] + cnt['ev_virt'] + cnt['ev_born'])
    return {
        'nevents': nev,
        'folding': env.get('ENV_VFOLDTEST_FOLDING') or '?',
        # MC@NLO-Delta (ENV_VFOLDTEST_MCATNLODELTA, ENV_MCATNLODELTA in older logs)
        'delta': env.get('ENV_VFOLDTEST_MCATNLODELTA', '0') not in ('', '0'),
        # (N/A before step 9a: that code evaluates one kpoint at a time and
        # never reads ENV_VFOLD_KPOINT_BATCH, which myLaunch.sh exports anyway)
        'kbatch': 'N/A' if _noswitch(cnt, 'kbatch_log') else
                  (env.get('ENV_VFOLD_KPOINT_BATCH') or cnt.get('kbatch_log') or '?'),
        # What the code actually used, i.e. the value it logged: myLaunch.sh
        # exports ENV_VFOLD_BSF_FUSED and ENV_VFOLD_SBORN_BRIDGE for every step, but
        # the Fortran only reads them from step 9f and 9i on, so the ENV value
        # alone would claim 'bsffused=-1 sborn=1' for e.g. step 8n, whose code
        # has neither switch. 'N/A' = no such switch in that code.
        # (the value REQUESTED for the step, i.e. the ENV one: what each mode
        # then does with it is mode-dependent, e.g. -1 logs as vfold_bsf_fused=T in
        # F2/F/C and =F in F0/F1, and vfold_sborn_bridge is forced to 0 outside C)
        'bsffused': 'N/A' if _noswitch(cnt, 'bsffused_log') else
                    (env.get('ENV_VFOLD_BSF_FUSED') or env.get('ENV_VFOLD_BSF_VEC') or
                     cnt.get('bsffused_log') or '0'),
        'sborn': 'N/A' if _noswitch(cnt, 'sborn_log') else
                 (env.get('ENV_VFOLD_SBORN_BRIDGE') or cnt.get('sborn_log') or '0'),
    }


def _mlreduction(d):
    """MadLoop reduction library of a column (step 14h, ENV_VFOLDTEST_MLREDUCTION):
    older runs do not have this variable, and they all used the MG5 default, ninja."""
    return d['env'].get('ENV_VFOLDTEST_MLREDUCTION') or 'ninja'


def _vecsize(name, d):
    """Bridge chunk size (ENV_VFOLD_VECSIZE_USED) of a mode C column, i.e. the
    number of events per cudacpp call; '-' for the other modes, which never
    call cudacpp (mode F loops in Fortran over the whole batch)."""
    if not name.startswith('C'):
        return '-'
    return d['env'].get('ENV_VFOLD_VECSIZE_USED') or d['counters'].get('vecsize_log') or '?'


def _batch_avg(d, what):
    """Mean number of entries per Born or Real batch ('born' or 'real') in a
    run, from the born/real_batch_avg and _ncalls counters of the MINT2 logs;
    '-' for the modes that do not batch (F0, F1, F2)."""
    c = d['counters']
    ncalls = c.get(what + '_calls', 0)
    return '%.1f' % (c[what + '_entries'] / ncalls) if ncalls else '-'


def _switch_na(p, key):
    return p[key] == 'N/A'


def _gscale(name, d):
    """ENV_VFOLD_SBORN_GSCALE of a mode C column, i.e. whether the Borns at the
    real-emission coupling g' are served by rescaling those at g (step 10b);
    '-' for the other modes, which do not serve SBORN from cudacpp. It is 0
    before step 10b, when there was no such switch at all."""
    if not name.startswith('C'):
        return '-'
    return d['counters'].get('gscale_log') or ('N/A' if d['counters'].get('has_av_log')
                                               else d['env'].get('ENV_VFOLD_SBORN_GSCALE') or '0')


def _cputimer(d):
    """Timer used by the MG5 'Time spent in' timers of a column (step 11b,
    vfold_cpu_time): F0/F1 keep MG5's getrusage, the other modes use
    clock_gettime with ENV_VFOLD_CPUTIME=-1 (myLaunch.sh default). Without
    vfold_cpu_time (before step 11a, or stock MG5) it is gfortran's cpu_time,
    i.e. getrusage."""
    c = d['counters']
    if c.get('cputime_log') is not None:
        return CPUTIMERS.get(c['cputime_log'], c['cputime_log'])
    return CPUTIMERS['0']


def _kpoint_line(name, c, isTex):
    # The same line in tex as in txt/stdout (virt and novi back in tex since --branch)
    return ('mode %s: %d kpoints = %d virt + %d novi -> %d events (%d virt + %d novi)'
            % (name, c['ncand'], c['cand_virt'], c['cand_novi'],
               c['ev_novi'] + c['ev_virt'] + c['ev_born'], c['ev_virt'],
               c['ev_novi'] + c['ev_born']))


def _rescale_factors(cbase, cref):
    """Ratios of kpoints (reference/baseline) for each RESCALE_BY type."""
    ratio = lambda a, b: float(a) / b if b else 1.0
    return {'all': ratio(cref['ncand'], cbase['ncand']),
            'novi': ratio(cref['cand_novi'], cbase['cand_novi']),
            'virt': ratio(cref['cand_virt'], cbase['cand_virt'])}


def _rescaled_times(times, factors):
    """Rescale each raw label; the new Total is the sum of the rescaled labels."""
    out = {k: v * factors[RESCALE_BY.get(k, 'novi')] for k, v in times.items() if k != 'Total'}
    out['Total'] = sum(out.values())
    return out


def build_table(data, step, grouping='default', rescale=RESCALE_BASELINE, isTex=False):
    """Format-neutral table: {'title', 'subtitles', 'columns', 'rows'}."""
    columns = list(data.keys())
    p = _run_params(data)
    if not isTex:
        title = ('%s | NEVENTS=%s | folding=%s | kbatch=%s | bsffused=%s | sborn=%s'
                 % (step, p['nevents'], p['folding'], p['kbatch'], p['bsffused'], p['sborn']))
    else:
        title = ('%s | NEVENTS=%s | folding=%s' % (step, p['nevents'], p['folding']))
        if p['delta']:
            title += r' | MC@NLO-$\Delta$'
        if p['kbatch'] != 'N/A':
            title += ' | kbatch=%s' % p['kbatch']
    vecsizes = [_vecsize(n, d) for n, d in data.items()]
    gscales = [_gscale(n, d) for n, d in data.items()]
    bornavgs = [_batch_avg(d, 'born') for d in data.values()]
    realavgs = [_batch_avg(d, 'real') for d in data.values()]
    cputimers = [_cputimer(d) for d in data.values()]
    mlreductions = [_mlreduction(d) for d in data.values()]
    subtitles = []
    if not isTex:
        if p['bsffused'] == 'N/A' or p['sborn'] == 'N/A':
            subtitles.append('N/A: that switch does not exist in the code of this step')
    if p['bsffused'] == '-1':
        subtitles.append('bsffused -1: fused single-pass BSF kernel OFF for F0/F1, ON for F2/F/C')
        ###subtitles.append('bsfvec -1: BSF vectorization OFF for F0/F1, ON for F2/F/C')
    if p['sborn'] not in ('0', 'N/A'):
        subtitles.append('sborn %s: SBORN served from cudacpp, in mode C only%s'
                         % (p['sborn'],
                            ' (and validated against the scalar SBORN)' if p['sborn'] == '2' else ''))
    if any(n.split('#')[0] == 'C/cudbr' for n in data):
        subtitles.append('cudbr: cuda backend with BLAS colour sums for the Real libraries only')
    if any(n.split('#')[0] == 'C/cudb' for n in data):
        subtitles.append('cudb: cuda backend with BLAS colour sums (CUDACPP_RUNTIME_BLASCOLORSUM=1)')
    # In tex there is no sborn_gscale row, so say it here (step 10b: a second,
    # separate optimization on top of sborn, see vfold_mode.f90): the Borns at the
    # real-emission coupling are not evaluated at all, but rescaled from those
    # at the n-body coupling.
    if isTex and '1' in gscales:
        subtitles.append("gscale 1: Borns at the real-emission coupling $g'$ obtained"
                         " by rescaling those at $g$, in mode C only")
    # Different timers in different columns (F0/F1 getrusage, others clock_gettime):
    # the getrusage syscalls inflate the timings of those columns
    if len(set(cputimers)) > 1:
        bytimer = {}
        for n, t in zip(columns, cputimers):
            bytimer.setdefault(t, []).append(n.split('#')[0] + ('*' if n == MODE_BASELINE else ''))
        names = {'rusage': 'getrusage', 'mono': 'clock_gettime', 'rdtsc': 'rdtsc'}
        parts = ['%s in %s' % (names.get(t, t), ', '.join(dict.fromkeys(ns)))
                 for t, ns in bytimer.items()]
        if not isTex:
            subtitles.append('timers: %s' % '; '.join(parts))
        else:
            # Shorter in tex, where the subtitle must fit on the page: name the
            # getrusage columns only (the usual case is F0*, F1 vs all others)
            others = [names.get(t, t) for t in bytimer if t != 'rusage']
            if 'rusage' in bytimer and len(others) == 1:
                subtitles.append('timers: getrusage in %s (a syscall per timer call),'
                                 ' %s in all other modes'
                                 % (', '.join(dict.fromkeys(bytimer['rusage'])), others[0]))
            else:
                subtitles.append('timers: %s' % '; '.join(parts))
    elif isTex:
        # Same timer in all columns (no cpu_timer head row in tex): say which one
        names = {'rusage': 'getrusage', 'mono': 'clock_gettime', 'rdtsc': 'rdtsc'}
        subtitles.append('timers: %s in all modes' % names.get(cputimers[0], cputimers[0]))
    if MODE_BASELINE in data:
        subtitles.append(_kpoint_line(MODE_BASELINE, data[MODE_BASELINE]['counters'], isTex))
    if MODE_BATCHED in data:
        cb = data[MODE_BATCHED]['counters']
        key = lambda c: (c['ncand'], c['cand_virt'], c['cand_novi'])
        others = [n for n in columns if n not in (MODE_BASELINE, MODE_BATCHED)]
        same = [n for n in others if key(data[n]['counters']) == key(cb)]
        diff = [n for n in others if n not in same]
        if not isTex:
            line = _kpoint_line(MODE_BATCHED, cb, isTex)
        else:
            line = _kpoint_line('%s, F1, F2, Cs'%MODE_BATCHED, cb, isTex)
        ###if same:
        ###    line += '  [same in %s]' % ', '.join(same)
        if diff:
            line += ' [DIFFERENT in %s]' % ', '.join(diff)
        else:
            if not isTex: line += ' [same in F1, F2, Cs]'
        subtitles.append(line)
        bavg = cb['born_entries'] / cb['born_calls'] if cb['born_calls'] else 0.0
        ravg = cb['real_entries'] / cb['real_calls'] if cb['real_calls'] else 0.0
        if not isTex:
            subtitles.append('mode %s: Born %d batched calls, mean batch %.3f | '
                             'Real %d batched calls, mean batch %.3f'
                             % (MODE_BATCHED, cb['born_calls'], bavg, cb['real_calls'], ravg))
        else:
            subtitles.append('mode %s: Born / Real %d / %d batched calls, mean batch %.1f / %.1f'
                             % ('%s, Cs'%MODE_BATCHED,
                                cb['born_calls'], cb['real_calls'], bavg, ravg))
    # Per-column timings, with the baseline optionally rescaled (column 'F0*')
    times = {n: data[n]['times'] for n in columns}
    if rescale and MODE_BASELINE in data and MODE_BATCHED in data:
        f = _rescale_factors(data[MODE_BASELINE]['counters'], data[MODE_BATCHED]['counters'])
        times[MODE_BASELINE] = _rescaled_times(data[MODE_BASELINE]['times'], f)
        ###subtitles.append('%s*: %s rescaled to the kpoints of %s: Virtual x %.4f (virt),'
        ###                 ' Real/MCsubtraction/Counter_terms x %.4f (novi), rest x %.4f (all)'
        ###                 % (MODE_BASELINE, MODE_BASELINE, MODE_BATCHED,
        ###                    f['virt'], f['novi'], f['all']))
        subtitles.append('%s*: %s rescaled to the kpoints of %s (separately for virt and novi)'
                         % (MODE_BASELINE, MODE_BASELINE, MODE_BATCHED))
        columns = [n + '*' if n == MODE_BASELINE else n for n in columns]
        times = {(n + '*' if n == MODE_BASELINE else n): t for n, t in times.items()}
    # Rows
    allraw = []
    for n in data:
        for k in data[n]['order']:
            if k not in allraw:
                allraw.append(k)
    rowdefs = GROUPINGS[grouping]
    if rowdefs is None:
        rowdefs = [(k, [k]) for k in ['Total'] + [k for k in allraw if k != 'Total']]
    ccols = [n for n in data if '/' in n]  # mode C columns (C/<backend>)
    hasraw = lambda raws: ccols and all(any(k in data[n]['times'] for k in raws) for n in ccols)
    rowdefs = [(label, raws) for label, raws in rowdefs
               if not (any(k in OPTIONAL_ROWS for k in raws) and not hasraw(raws))]
    nonzero = lambda raws: any(data[n]['times'].get(k, 0.0) for n in data for k in raws)
    rowdefs = [(label, raws) for label, raws in rowdefs
               if not (any(k in NONZERO_ROWS for k in raws) and not nonzero(raws))]
    used = set()
    for label, raws in rowdefs:
        if raws != MISC:
            dup = used.intersection(raws)
            if dup:
                sys.stderr.write('WARNING! %s counted in more than one row\n' % sorted(dup))
            used.update(raws)
    rows = []
    for label, raws in rowdefs:
        if raws == MISC:
            raws = [k for k in allraw if k not in used]
        rows.append((label, [sum(times[n].get(k, 0.0) for k in raws)
                             for n in columns]))
    # Sanity check: the rows other than Total should add up to Total
    # (MG5's Other_tasks is itself the residual that closes the sum)
    istot = [raws == ['Total'] for label, raws in rowdefs]
    for i, n in enumerate(columns):
        tot = sum(v[i] for (label, v), t in zip(rows, istot) if t)
        parts = sum(v[i] for (label, v), t in zip(rows, istot) if not t)
        if tot and abs(parts - tot) > 1e-3 * max(1.0, tot):
            sys.stderr.write('WARNING! %s: rows sum to %.4f but Total is %.4f\n'
                             % (n, parts, tot))
    if not isTex:
        headrows = [('cudacpp_vecsize', vecsizes), ('sborn_gscale', gscales),
                    ('ml_reduction', mlreductions),
                    ('born_batch_avg', bornavgs),
                    ('real_batch_avg', realavgs), ('cpu_timer', cputimers)]
    else:
        headrows = [('cudacpp_vecsize', vecsizes)]
        # Step 15e: a header line instead of a row, if all columns use the same reduction library
        if len(set(mlreductions)) == 1:
            subtitles.append('ml_reduction: %s for all' % mlreductions[0])
        else:
            headrows.append(('ml_reduction', mlreductions))
    return {'title': title, 'subtitles': subtitles, 'columns': columns, 'rows': rows,
            'vecsizes': vecsizes, 'colnames': [n.split('#')[0] for n in columns],
            'headrows': headrows}


def render_text(table, fmt='%7.3f', prefix=None):
    """Markdown-style ascii table."""
    modebknd = 'mode/bknd'
    llen = max([len(modebknd)] + [len(label) for label, values in table['headrows']])
    for label, values in table['rows']: llen = max(llen, len(label))
    lfmt = '%%-%is'%llen
    title = table['title'] if not table.get('branch') else table['branch'] + ' | ' + table['title']
    out = ['', '='*20 + ' ' + prefix + ' ' + '='*20, '### ' + title]
    out += ['- ' + s for s in table['subtitles']]
    # Column width: as many as needed for the header rows (e.g. 7 for 'collier' as of step 14h)
    width = max([6] + [len(str(v)) for label, values in table['headrows'] for v in values])
    numfmt = '%%%d%s' % (width + 1, fmt[fmt.index('.'):])
    # Step 15e: as in the rows of numbers, a name one character longer (C/cudbr) touches the left wall
    columns = [('%%%ds'%(width + 1))%col for col in table['colnames']]
    hline = '|-%s-|'%(llen*'-') + '-%s-|'%(width*'-') * len(columns)
    out += [hline, '| %s |'%(lfmt%modebknd) + ' |'.join(columns) + ' |']
    for label, values in table['headrows']:
        out.append('| %s | '%(lfmt%label) + ' | '.join(('%%%ds'%width)%v for v in values) + ' |')
    out += [hline]
    for label, values in table['rows']:
        out.append('| %s |%s |' % (lfmt%label, ' |'.join(numfmt % v for v in values)))
    out += [hline, '']
    return '\n'.join(out) + '\n'


def render_tex1(table, prefix=None):
    return render_tex(table, fmt='%.1f', prefix=prefix)

def render_tex2(table, prefix=None):
    return render_tex(table, fmt='%.2f', prefix=prefix)

def render_tex(table, fmt='%.2f', prefix=None):
    """Minimal LaTeX tabular (starting point for the paper tables)."""
    esc = lambda s: s.replace('_', r'\_').replace('|', r'$|$').replace('->', r'$\to$')
    ncol = len(table['columns'])
    # The title and subtitle rows span the whole table and are wider than the
    # columns: a plain \multicolumn would give all that excess width to the LAST
    # column of the span (a visible gap before the last column). Typeset them in
    # a zero-width left-aligned box instead, so that they overflow to the right
    # without contributing to any column width.
    span = lambda txt: (r'\multicolumn{%d}{@{}l@{}}{\makebox[0pt][l]{%s}} \\'
                        % (ncol + 1, txt))
    # Two header rows instead of one 'mode/bknd' row with e.g. 'C/none', for
    # more compact tables: 'mode' (F0*, F1, ..., C) and 'cudacpp_backend'
    # (none, sse4, ..., cuda; '-' for the modes that do not call cudacpp)
    modes = [c.split('/')[0] for c in table['colnames']]
    bknds = [c.split('/')[1] if '/' in c else '-' for c in table['colnames']]
    # Equally spaced columns: every cell of a column of numbers is typeset in a
    # right-aligned box of width \tmtwidth, so that a '-' takes as much space as
    # a '409.2'. \tmtwidth is NOT defined here: set it in the main tex file, e.g.
    # \newlength{\tmtwidth}\setlength{\tmtwidth}{8mm}
    title = esc(table['title'])
    if table.get('branch'):  # --branch: the branch first in the title, underlined
        title = r'\underline{%s} $|$ %s' % (esc(table['branch']), title)
    out = [r'% ' + ((table['branch'] + ' | ') if table.get('branch') else '') + table['title']]
    out += [r'\providecommand{\tmtcell}[1]{\makebox[\tmtwidth][r]{#1}}',
            r'\begin{tabular}{l' + 'r' * ncol + '}',
            span('{\\bf %s}'%title)]
    for sub in table['subtitles']:
        out.append(span(r'\footnotesize ' + esc('{\\em %s}'%sub)))
    cell = lambda x: r'\tmtcell{%s}' % x
    out += [r'\hline', 'mode & ' + ' & '.join(cell(esc(m)) for m in modes) + r' \\',
            r'cudacpp\_backend & ' + ' & '.join(cell(esc(b)) for b in bknds) + r' \\',
            ] + [esc(label) + ' & ' + ' & '.join(cell(v) for v in values) + r' \\'
                 for label, values in table['headrows']] + [
            r'\hline']
    for label, values in table['rows']:
        out.append(esc(TEXLABELS.get(label, label)) + ' & '
                   + ' & '.join(cell(fmt % v) for v in values) + r' \\')
    out += [r'\hline', r'\end{tabular}']
    return '\n'.join(out) + '\n'


RENDERERS = {'stdout': render_text, 'txt': render_text, 'tex1': render_tex1, 'tex2': render_tex2}


#----------------------------------------------------------------------------
# MAIN
#----------------------------------------------------------------------------

def main():
    # Run from the process directory (this script is in its vfold subdirectory)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p0s = glob.glob(os.path.join(os.getcwd(), 'SubProcesses', "P0_*"))
    if len(p0s) != 1:
        sys.exit('ERROR! more than one P0 directories?\n%s'%p0s)
    proc = os.path.basename(p0s[0]).replace('P0_','')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('prefix', help="run directory prefix, e.g. Events/REF3/n200f111k1/step9h")
    ap.add_argument('-g', '--grouping', default='default', choices=sorted(GROUPINGS))
    ap.add_argument('-f', '--format', default='stdout', choices=sorted(RENDERERS))
    ap.add_argument('--raw-baseline', action='store_true',
                    help='show %s as measured, not rescaled to the kpoints of %s'
                    % (MODE_BASELINE, MODE_BATCHED))
    ap.add_argument('--skip', default='',
                    help="columns to skip, as comma-separated directory suffixes (e.g. Sn,Sc,Su,Ccudb,Ccudbr); "
                    "'nonstd' means all of these (not with -f txt, to keep the full table)")
    ap.add_argument('--vecsizes', default=None,
                    help='cudacpp_vecsize values to keep in the GPU scans, comma-separated, '
                    'e.g. 64,512,1024 (not with -f txt, to keep the full table)')
    ap.add_argument('--branch', default=None,
                    help='code branch of the runs, e.g. vfold5363: printed first in the title, '
                    'and a prefix of the tex file name')
    args = ap.parse_args()
    prefix = args.prefix.rstrip('/')
    if args.branch is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.branch):
        sys.exit('ERROR! invalid --branch (letters, digits, dots, dashes, underscores): %s' % args.branch)
    skip = set()
    for s in filter(None, args.skip.split(',')):
        if s == 'nonstd':
            skip.update(NONSTANDARD)
        elif s in [suffix for suffix, _ in MODES]:
            skip.add(s)
        else:
            sys.exit('ERROR! unknown column to skip: %s' % s)
    vecsizes = None
    if args.vecsizes is not None:
        if not re.fullmatch(r'\d+(,\d+)*', args.vecsizes):
            sys.exit('ERROR! invalid --vecsizes (it must be comma-separated numbers): %s' % args.vecsizes)
        vecsizes = set(int(v) for v in args.vecsizes.split(','))
    if (skip or vecsizes is not None) and args.format == 'txt':
        sys.exit('ERROR! --skip and --vecsizes are not allowed with -f txt: they would overwrite the full timing table')
    data = read_step(prefix, skip, vecsizes)
    if vecsizes is not None:
        found = set(int(n.split('#')[1]) for n in data if '#' in n)
        if found and vecsizes - found:  # e.g. the same list for two tests (no warning without GPU scans)
            sys.stderr.write('WARNING! --vecsizes: no column for %s\n' % sorted(vecsizes - found))
    if not data:
        sys.exit('ERROR! no run directories with P*/GF*/log_MINT2.txt found for prefix %s' % prefix)
    isTex = ( args.format == 'tex1' ) or ( args.format == 'tex2' )
    table = build_table(data, os.path.basename(prefix), args.grouping,
                        rescale=RESCALE_BASELINE and not args.raw_baseline, isTex=isTex)
    table['title'] = proc + ' | ' + table['title']
    table['branch'] = args.branch
    if isTex:
        m = re.match(r'^Events/REF(\d*)/(.*)$', prefix)  # Events/REF<n>/... -> Events/TEX<n>/...
        if not m:
            sys.exit('ERROR! tex output needs a prefix in Events/REF, Events/REF2 or Events/REF3: %s' % prefix)
        os.makedirs('Events/TEX'+m.group(1), exist_ok=True)  # step 15e
        texname = m.group(2).replace('/','_')+'.tex'
        if args.branch:  # e.g. in ggttg_vfgen: ggttg_vfold7cg_CUDA_n20f111k512g1_step15d.tex
            procdir = re.sub(r'_(vfgen|vfalt|stock)$', '', os.path.basename(os.getcwd()))
            texname = procdir + '_' + args.branch + '_' + texname
        with open('Events/TEX'+m.group(1)+'/'+texname, 'w') as f:
            f.write(RENDERERS[args.format](table, prefix=prefix))
    elif args.format == 'txt':
        # Write <dir>/timingTable_<step>.txt, e.g. for prefix Events/REF3/GOLD/n200f841k1/step8r
        # write Events/REF3/GOLD/n200f841k1/timingTable_step8r.txt (no output to stdout)
        with open(os.path.join(os.path.dirname(prefix), 'timingTable_'+os.path.basename(prefix)+'.txt'), 'w') as f:
            f.write(RENDERERS[args.format](table, prefix=prefix))
    else:
        sys.stdout.write(RENDERERS[args.format](table, prefix=prefix))

if __name__ == '__main__':
    main()
