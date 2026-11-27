#! /usr/bin/env python3
"""Generate the MC@NLO-Delta Sudakov tables with an unmodified Pythia8.

Every row of every table (dipole type, radiator, dipole mass) is one run of
sudgen_fast, which multiplies the trial-shower no-emission probabilities of
adjacent scale intervals (see sudgen_fast.cc). The rows run in parallel. The
Pythia settings are those of the MC@NLO shower interface (sudgen_settings.py).
The result is written as a standalone Fortran provider in the format of the
shipped sudakov_pdfmode1.f: its lookup functions and colour-line conversions
are kept, the numerical grids and the coupling metadata are replaced.

Usage:
  sudgen_tables.py --pythia8 /path/to/pythia8 --workdir DIR --output sudakov_new.f
                   [--jobs 120] [--nrep-fsr 50] [--nrep-isr 1] [--seed 1000]
"""
import argparse
import concurrent.futures
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sudgen_settings  # noqa: E402

TYPES = (1, 2, 3, 4)            # II, FF, IF, FI as Merging::generateSingleSudakov
PARTONS = (1, 2, 3, 4, 5, 6, 21)
NNST, NNXM = 100, 50
STLOW, STUPP, XMLOW, XMUPP = 1., 7000., 1., 7000.
TEMPLATE = os.path.join(HERE, 'sudakov_pdfmode1.f')


def node(i, n, low, upp):
    return low * (upp / low) ** ((i - 1.) / (n - 1.))


def build(pythia8, workdir):
    exe = os.path.join(workdir, 'sudgen_fast')
    config = os.path.join(pythia8, 'bin', 'pythia8-config')
    flags = subprocess.check_output([config, '--cxxflags', '--libs']).decode().split()
    subprocess.check_call(['g++', '-O2', '-std=c++17', os.path.join(HERE, 'sudgen_fast.cc'),
                           '-o', exe] + flags + ['-ldl'])
    return exe


def run_row(args):
    exe, cmd, itype, pid, ixm, nrep, seed, out = args
    if os.path.isfile(out) and sum(1 for l in open(out) if not l.startswith('#')) == NNST:
        return out
    with open(out + '.log', 'w') as log:
        subprocess.check_call([exe, cmd, str(itype), str(pid), str(ixm), str(ixm), str(nrep),
                               str(seed), out], stdout=log, stderr=subprocess.STDOUT)
    return out


def read_rows(paths):
    grid, error, headers = {}, {}, {}
    for path in paths:
        for line in open(path):
            if line.startswith('#'):
                if 'alphaSvalue' in line:
                    headers[path] = line[1:].split()
                continue
            ixm, ist, st, xm, delta, err = line.split()
            key = tuple(int(v) for v in os.path.basename(path)[4:-4].split('_'))[:2]
            grid[key + (int(ist), int(ixm))] = float(delta)
            error[key + (int(ist), int(ixm))] = float(err)
    return grid, error, headers


def fortran(value):
    """0.12345678D+01 style, as the shipped providers."""
    if value == 0.:
        return '0.00000000D+00'
    exponent = int(math.floor(math.log10(abs(value)))) + 1
    mantissa = value / 10. ** exponent
    if abs(mantissa) >= 0.999999995:
        mantissa /= 10.
        exponent += 1
    return '%.8fD%+03d' % (mantissa, exponent)


def write_provider(output, grid, metadata):
    text = open(TEMPLATE).read()
    # Header and coupling metadata.
    start = text.index('      function pysudakov(st,xm,id,itype,xmpart)')
    s = metadata['settings']
    header = [
        'c     MC@NLO-Delta no-emission tables generated with Pythia %s and the'
        % metadata['pythia8_version'],
        'c     settings of the MC@NLO shower interface (vendor/SudGen/sudgen_tables.py;',
        'c     sudgen_settings.py lists the settings). Product of adjacent-interval',
        'c     trial-shower probabilities: monotone in the scale, no statistical zeros.',
        'c     Raw FF probabilities cover two radiators. Other types cover one.',
        'c     Each gluon radiator has two colour lines. Convert the raw return',
        'c     to one emitter and one directed colour line in pysudakov below.',
        '      subroutine pysudakov_isr_parameters(alpha_mz,order,cmw,nfmax,',
        '     $     thresholds,pdfmode,status)',
        'c     Initialised generation settings of the initial-state tables',
        'c     (SpaceShower:alphaSvalue, alphaSorder, alphaSuseCMW,',
        'c     StandardModel:alphaSnfmax, Pythia AlphaStrong thresholds,',
        'c     SpaceShower:pdfMode).',
        '      implicit none',
        '      real*8 alpha_mz,thresholds(2)',
        '      integer order,nfmax,pdfmode,status',
        '      logical cmw',
        '      alpha_mz=%sd0' % ('%.6f' % metadata['alpha_mz']).rstrip('0'),
        '      order=%d' % metadata['order'],
        '      cmw=.%s.' % ('true' if metadata['cmw'] else 'false'),
        '      nfmax=%d' % metadata['nfmax'],
        '      thresholds=(/1.5d0,4.8d0/)',
        '      pdfmode=%d' % metadata['pdfmode'],
        '      status=0',
        '      end subroutine pysudakov_isr_parameters',
        '',
        '',
    ]
    body = text[start:]
    # Grids of each lookup function.
    def replace_function(match):
        pid, itype = int(match.group(1)), int(match.group(2))
        function = match.group(0)
        def replace_column(col):
            ixm = int(col.group(1))
            values = [grid[(itype, pid, ist, ixm)] for ist in range(1, NNST + 1)]
            lines = []
            for i in range(0, NNST, 4):
                chunk = ', '.join(fortran(v) for v in values[i:i + 4])
                lines.append('     # %s%s' % (chunk, '/' if i + 4 >= NNST else ','))
            return '      data (gridv(inst,%3d),inst=1,100)/\n%s\n' % (ixm, '\n'.join(lines))
        new, n = re.subn(r'      data \(gridv\(inst,\s*(\d+)\),inst=1,100\)/\n(?:     #[^\n]*\n){25}',
                         replace_column, function)
        if n != NNXM:
            raise SystemExit('pysudakov_%d_%d: replaced %d columns' % (pid, itype, n))
        return new
    body, nfun = re.subn(r'      function pysudakov_(\d+)_(\d+)\(st,xm\)\n.*?\n      end\n',
                         replace_function, body, flags=re.S)
    if nfun != len(TYPES) * len(PARTONS):
        raise SystemExit('replaced %d table functions' % nfun)
    with open(output, 'w') as out:
        out.write('\n'.join(header) + body)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--pythia8', required=True)
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--jobs', type=int, default=os.cpu_count())
    parser.add_argument('--nrep-fsr', type=int, default=50)
    parser.add_argument('--nrep-isr', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1000)
    args = parser.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    lines, metadata = sudgen_settings.table_settings(args.pythia8)
    cmd = os.path.join(args.workdir, 'sudgen.cmd')
    with open(cmd, 'w') as out:
        out.write('\n'.join(lines) + '\n')
    exe = build(args.pythia8, args.workdir)
    jobs = []
    for itype in TYPES:
        nrep = args.nrep_isr if itype in (1, 3) else args.nrep_fsr
        for pid in PARTONS:
            for ixm in range(1, NNXM + 1):
                seed = args.seed + len(jobs)
                out = os.path.join(args.workdir, 'row_%d_%d_%d.dat' % (itype, pid, ixm))
                jobs.append((exe, cmd, itype, pid, ixm, nrep, seed, out))
    # Longest first: initial-state rows.
    jobs.sort(key=lambda j: j[2] not in (1, 3))
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for _ in pool.map(run_row, jobs):
            done += 1
            if done % 100 == 0:
                print('%d / %d rows' % (done, len(jobs)), flush=True)
    grid, error, headers = read_rows([j[-1] for j in jobs])
    if len(grid) != len(TYPES) * len(PARTONS) * NNST * NNXM:
        raise SystemExit('incomplete grid: %d entries' % len(grid))
    # Initialised settings of the initial-state tables, from the run headers.
    isr = [h for path, h in headers.items() if os.path.basename(path).startswith('row_1_')]
    values = dict(zip(isr[0][0::2], isr[0][1::2]))
    meta = {'pythia8_version': metadata['pythia8_version'],
            'alpha_mz': float(values['alphaSvalue']), 'order': int(values['alphaSorder']),
            'cmw': bool(int(values['alphaSuseCMW'])), 'nfmax': int(values['alphaSnfmax']),
            'pdfmode': int(values['pdfMode']), 'settings': metadata['settings'],
            'nrep_fsr': args.nrep_fsr, 'nrep_isr': args.nrep_isr, 'seed': args.seed}
    for path, h in headers.items():
        v = dict(zip(h[0::2], h[1::2]))
        if os.path.basename(path).startswith(('row_1_', 'row_3_')) and (
                float(v['alphaSvalue']) != meta['alpha_mz'] or int(v['alphaSnfmax']) != meta['nfmax']
                or int(v['pdfMode']) != meta['pdfmode']):
            raise SystemExit('inconsistent initial-state settings in %s' % path)
    write_provider(args.output, grid, meta)
    with gzip.open(os.path.splitext(args.output)[0] + '_grid.dat.gz', 'wt') as out:
        out.write('# type pdg ist ixm st xm delta relative_error\n')
        for (itype, pid, ist, ixm), value in sorted(grid.items()):
            out.write('%d %d %d %d %.8e %.8e %.10e %.4e\n' % (
                itype, pid, ist, ixm, node(ist, NNST, STLOW, STUPP), node(ixm, NNXM, XMLOW, XMUPP),
                value, error[(itype, pid, ist, ixm)]))
    with open(os.path.splitext(args.output)[0] + '_settings.json', 'w') as out:
        json.dump(meta, out, indent=1, sort_keys=True)
    print('wrote', args.output)


if __name__ == '__main__':
    main()
