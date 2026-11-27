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
"""Paired timing of MadLoop outputs of two MadGraph checkouts.

    tools/onia_loop_timing.py --root base=/path/to/baseline --root cand=/path \\
        --model loop_sm --process "g g > t t~ [virt=QCD]" --work DIR \\
        [--libs "6|7|1"] [--npoints 50] [--pairs 7] [--energy 1000] [--json out]

Each checkout generates the process with its own bin/madgraph. The same
points driver (tools/onia_loop_check.py) is linked against each output, the
MadLoopParams.dat settings are made identical, and the same deterministic
points are evaluated. A run times every call with SYSTEM_CLOCK, after the
initialisation calls, and its metric is the median time per point. The runs
alternate (A B, B A, ...); the ratio of each pair is cand/base. The report
gives the median ratio and the upper bound of the 95% confidence interval
of the geometric-mean ratio (t distribution on the log ratios), and checks
that both outputs give identical results.
"""

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import onia_loop_check as olc
import tirust_benchmark_points as points_lib

# two-sided 95% quantiles of Student's t, by degrees of freedom
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t975(dof):
    keys = sorted(T975)
    for k in keys:
        if dof <= k:
            return T975[k]
    return 1.96


def generate(root, model, process, outdir, extra=()):
    """The standalone output of `process` written by the checkout `root`."""
    if os.path.isdir(outdir):
        return outdir
    script = outdir + '.mg'
    with open(script, 'w') as f:
        f.write('\n'.join(['import model %s' % model] + list(extra) +
                          ['generate %s' % process, 'output standalone %s -f' % outdir]) + '\n')
    with open(outdir + '.log', 'w') as out:
        status = subprocess.call([os.path.join(root, 'bin', 'madgraph'), script],
                                 cwd=root, stdout=out, stderr=subprocess.STDOUT)
    if status or not os.path.isdir(outdir):
        raise RuntimeError('generation failed, see %s.log' % outdir)
    return outdir


def run_timed(pdir, momenta, mu, warmup):
    res = olc.run_points(pdir, momenta, mu, warmup=warmup)
    return statistics.median(r['time'] for r in res), res


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--root', action='append', required=True,
                        help='NAME=PATH of a MadGraph checkout (two of them: baseline first)')
    parser.add_argument('--model', required=True)
    parser.add_argument('--process', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--libs', default='6|7|1')
    parser.add_argument('--param', action='append', default=[])
    parser.add_argument('--npoints', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--pairs', type=int, default=7)
    parser.add_argument('--energy', type=float, default=1000.)
    parser.add_argument('--mu', type=float, default=None)
    parser.add_argument('--json')
    args = parser.parse_args(argv)

    roots = [r.split('=', 1) for r in args.root]
    if len(roots) != 2:
        raise SystemExit('two --root NAME=PATH are needed (baseline, candidate)')
    os.makedirs(args.work, exist_ok=True)
    pdirs = []
    for name, root in roots:
        outdir = generate(root, args.model, args.process, os.path.join(args.work, name))
        pdir = olc.subprocess_dir(outdir)
        params = dict(p.split('=', 1) for p in args.param)
        olc.set_reduction(outdir, args.libs.split('|'), params)
        olc.build_driver(pdir)
        pdirs.append(pdir)
    masses = olc.driver_masses(pdirs[0])
    final = masses[2:]
    energy = args.energy if len(final) > 1 else final[0]
    pts = points_lib.generate_points(final, count=args.npoints, energy=energy,
                                     incoming_masses=tuple(masses[:2]))
    momenta = [p['momenta'] for p in pts]

    # identical results
    _, ra = run_timed(pdirs[0], momenta, args.mu, args.warmup)
    _, rb = run_timed(pdirs[1], momenta, args.mu, args.warmup)
    keys = ('born', 'finite', 'single', 'double')
    identical = all(a[k] == b[k] for a, b in zip(ra, rb) for k in keys)
    maxrel = max(abs(a[k] - b[k]) / max(abs(a[k]), 1e-300) for a, b in zip(ra, rb)
                 for k in keys if a[k] != 0)
    pairs = []
    for k in range(args.pairs):
        order = (0, 1) if k % 2 == 0 else (1, 0)
        t = {}
        for i in order:
            t[i], _ = run_timed(pdirs[i], momenta, args.mu, args.warmup)
        pairs.append((t[0], t[1]))
        print('pair %d: %s %.4e s  %s %.4e s  ratio %.4f' % (
            k + 1, roots[0][0], t[0], roots[1][0], t[1], t[1] / t[0]), flush=True)
    logs = [math.log(b / a) for a, b in pairs]
    mean = statistics.mean(logs)
    sd = statistics.stdev(logs) if len(logs) > 1 else 0.
    upper = math.exp(mean + t975(len(logs) - 1) * sd / math.sqrt(len(logs)))
    report = dict(process=args.process, model=args.model, libs=args.libs,
                  npoints=args.npoints, pairs=pairs,
                  median_ratio=statistics.median(b / a for a, b in pairs),
                  geomean_ratio=math.exp(mean), upper95=upper,
                  base_ms=1e3 * statistics.median(a for a, _ in pairs),
                  cand_ms=1e3 * statistics.median(b for _, b in pairs),
                  identical=identical, max_relative_difference=maxrel,
                  return_codes=sorted(set(r['ret_code'] for r in rb)))
    print('%s: base %.4g ms cand %.4g ms  median ratio %.4f  geomean %.4f  upper95 %.4f  '
          'identical %s (max rel diff %.1e)' % (args.process, report['base_ms'],
          report['cand_ms'], report['median_ratio'], report['geomean_ratio'], upper,
          identical, maxrel))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=1)


if __name__ == '__main__':
    main()
