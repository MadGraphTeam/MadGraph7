#!/usr/bin/env python3
"""Validate madspace's dressed-lepton chain against the MadEvent Fortran.

Everything the chain is made of is compared with the code it was ported from,
at the level where a disagreement would actually change a cross section:

  1. the luminosity, sum_n f1_n(x1) f2_n(x2), against `ee_comp_prod` built
     from the Fortran `compute_eepdf` -- this is the factor that multiplies
     the matrix element, so it is the one that matters;

  2. the beamstrahlung split, against `eepdf_sample_bstrahl`;

  3. the physics the two produce together -- how often a beam radiates and
     how much it loses -- sampled through the madspace mapping and compared
     with the *same* quantity computed from the Fortran densities on the same
     points.

Note on what that third check is not. Event-level numbers from a full MadEvent
run (5.9% of beams radiating at FCC-ee 240, 23.1% at ILC 500) are a different
quantity: they are shaped by the matrix element and the cuts, which reweight x
before any beam radiates. Comparing the two directly reads like a factor-two
disagreement at 365 GeV and is not one -- the Fortran density alone gives 12.6%
there. The reference for this code is the Fortran, and that is what is used.

What is *not* here is the cross section itself. That needs the components
wired into Integrand/DifferentialCrossSection and the mg7 launcher, which is
the remaining work; these checks cover the physics that wiring would carry.

Usage:
    python3 vendor/eepdfgrid/validate_madspace.py <MG root> [--events N]

Needs the Fortran references built first (see build_references below).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# MadEvent's get_ee_expo(); the mapping and the density must share it
PS_EXPO = 0.96

# Beam energy per set, and -- for context only -- the event-level fractions
# measured from full MadEvent runs behind vendor/py8main/validation. Those are
# post matrix element and post cuts, so they are NOT what this script checks
# against; see the module docstring.
SETS = {
    'fcce240ll': {'e_beam': 120.0, 'madevent_events': 0.059},
    'fcce365ll': {'e_beam': 182.5, 'madevent_events': 0.060},
    'ilc500ll': {'e_beam': 250.0, 'madevent_events': 0.231},
}

QTERM_NAMES = {'sqrt': 'sqrt_u', 'u15': 'u_three_halves',
               'ym15': 'y_minus_three_halves'}


def build_references(root, sets):
    """Compile the Fortran dumpers this script compares against."""
    src = os.path.join(root, 'Template', 'Common', 'Source', 'PDF')
    built = {}
    for name in sets:
        density = os.path.join(src, 'lep_densities', name)
        out = os.path.join(tempfile.gettempdir(), 'validate_ce_' + name)
        cmd = ['gfortran', '-w', '-ffixed-line-length-132', '-O1', '-o', out,
               os.path.join(root, 'vendor', 'eepdfgrid', 'dump_compute_eepdf.f'),
               os.path.join(density, 'eepdf.f'),
               os.path.join(density, 'gridpdfaux.f'),
               os.path.join(src, 'dfint.f'), os.path.join(src, 'kerset.f')]
        abend = os.path.join(tempfile.gettempdir(), 'validate_abend.f')
        with open(abend, 'w') as fsock:
            fsock.write('      subroutine abend\n      stop 999\n      end\n')
        cmd.append(abend)
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            built[name] = out
    return built


def fortran_density(binary, x, q2, workdir):
    """compute_eepdf for all 16 (component, partonid, beamid) at each point."""
    points = os.path.join(workdir, 'points.txt')
    with open(points, 'w') as fsock:
        fsock.write('%d\n' % len(x))
        for xi, q2i in zip(x, q2):
            fsock.write('%.17e %.17e\n' % (xi, q2i))
    out = subprocess.run([binary, points], capture_output=True, text=True,
                         check=True).stdout
    return np.array([[float(v) for v in line.split()[2:]]
                     for line in out.strip().split('\n')])


FULL_PIDS = [1000 * n + 10 * a + b
             for n in (1, 2, 3, 4) for a in (1, 2) for b in (1, 2)]


def kappa_const(meta, n, beamid):
    k = meta['kappa']['%d_%d' % (n, beamid)]
    return k['c_b'] * meta['b_exp'] + k['c_1']


def check_luminosity(ms, meta, dat, binary, workdir, n_points=4000, seed=5):
    """madspace's EELuminosity against sum_n f1_n f2_n from the Fortran."""
    pids1 = [1000 * n + 22 for n in (1, 2, 3, 4)]   # electron beam
    pids2 = [1000 * n + 11 for n in (1, 2, 3, 4)]   # positron beam
    off1 = [PS_EXPO - kappa_const(meta, n, 11) for n in (1, 2, 3, 4)]
    off2 = [PS_EXPO - kappa_const(meta, n, -11) for n in (1, 2, 3, 4)]

    ctx = ms.default_context()
    grid = ms.PdfGrid(dat)
    prefix = 'val_%s_' % meta['set']
    grid.initialize_globals(ctx, prefix)
    lum = ms.EELuminosity(grid, pids1, pids2, off1, off2, prefix=prefix)

    xlo, xhi = meta['usable_x_range']
    qlo, qhi = meta['usable_q_range']
    rng = np.random.default_rng(seed)
    x1 = np.clip(1 - 10 ** rng.uniform(-8, 0, n_points), xlo, xhi)
    x2 = np.clip(1 - 10 ** rng.uniform(-8, 0, n_points), xlo, xhi)
    q = 10 ** rng.uniform(np.log10(qlo) * (1 - 1e-9),
                          np.log10(qhi) * (1 - 1e-9), n_points)

    got = np.asarray(lum(x1, 1 - x1, x2, 1 - x2, q, rng.uniform(0, 1, n_points))
                     .luminosity)
    f1 = fortran_density(binary, x1, q * q, workdir)
    f2 = fortran_density(binary, x2, q * q, workdir)
    ref = sum(f1[:, FULL_PIDS.index(p1)] * f2[:, FULL_PIDS.index(p2)]
              for p1, p2 in zip(pids1, pids2))
    keep = np.abs(ref) > 1e-6 * np.abs(ref).max()
    rel = np.abs(got[keep] - ref[keep]) / np.abs(ref[keep])
    return lum, dict(median=np.median(rel), p99=np.percentile(rel, 99),
                     maximum=rel.max(), n=int(keep.sum()))


def build_sampler_reference(root, name):
    """Compile eepdf_sample_bstrahl of one set as a standalone dumper."""
    out = os.path.join(tempfile.gettempdir(), 'validate_bs_' + name)
    cmd = ['gfortran', '-w', '-ffixed-line-length-132', '-O1', '-o', out,
           os.path.join(root, 'vendor', 'eepdfgrid', 'dump_bstrahl_sample.f'),
           os.path.join(root, 'Template', 'Common', 'Source', 'PDF',
                        'lep_densities', name, 'bstrahl.f')]
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        return None
    return out


def check_split(ms, meta, binary, workdir, e_beam, n_points=3000, seed=13):
    """madspace's BeamstrahlungSampler against eepdf_sample_bstrahl."""
    bs = meta['beamstrahlung']
    pars = bs['parameters']

    def beam_params(beamid):
        out = []
        for n in (1, 2, 3, 4):
            key = '%d_%d' % (n, beamid)
            if key in pars:
                p = pars[key]
                out.append(ms.BeamstrahlungParameters(p['a'], p['p'], p['q'], True))
            else:
                out.append(ms.BeamstrahlungParameters())
        return out

    sampler = ms.BeamstrahlungSampler(
        beam_params(11), beam_params(-11),
        getattr(ms.BeamstrahlungQTerm, QTERM_NAMES[bs['q_term']]))

    rng = np.random.default_rng(seed)
    x = np.clip(1 - 10 ** rng.uniform(-7, -0.05, n_points), 1e-6, 1 - 1e-12)
    component = rng.integers(0, 4, n_points)
    beam = rng.choice([11, -11], n_points)
    u1 = rng.uniform(0, 1, n_points)
    u2 = rng.uniform(0, 1, n_points)

    points = os.path.join(workdir, 'bs_points.txt')
    with open(points, 'w') as fsock:
        fsock.write('%d\n' % n_points)
        for i in range(n_points):
            fsock.write('%.17e %.17e %d %d %.17e %.17e\n'
                        % (x[i], e_beam * e_beam, component[i] + 1, beam[i],
                           u1[i], u2[i]))
    out = subprocess.run([binary, points], capture_output=True, text=True,
                         check=True).stdout
    ref = np.array([float(line.split()[0]) for line in out.strip().split('\n')])

    got = np.array([
        sampler.sample(
            float(x[i]) if beam[i] == 11 else 1.0,
            1.0 if beam[i] == 11 else float(x[i]),
            e_beam * e_beam, int(component[i]),
            u1[i], u2[i], u1[i], u2[i])[0 if beam[i] == 11 else 1]
        for i in range(n_points)])
    rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)
    radiating = ref < 1 - 1e-15
    return dict(maximum=rel.max(), radiating=int(radiating.sum()),
                n=n_points, q_term=bs['q_term'])


def radiating_fraction(ms, meta, lum, binary, workdir, e_beam, n_events, seed=11):
    """How often a beam radiates and how much it loses, both ways.

    Fractions come from the madspace mapping; the component weights are then
    formed once by madspace and once from the Fortran densities at the same
    points, so the two differ only in the code under test.
    """
    pids1 = [1000 * n + 22 for n in (1, 2, 3, 4)]
    pids2 = [1000 * n + 11 for n in (1, 2, 3, 4)]
    mapping = ms.LeptonLuminosity((2 * e_beam) ** 2, 0.0, PS_EXPO)
    rng = np.random.default_rng(seed)
    f = mapping.map_forward([rng.uniform(0, 1, n_events),
                             rng.uniform(0, 1, n_events)])
    x1 = np.asarray(f.x1); omx1 = np.asarray(f.omx1)
    x2 = np.asarray(f.x2); omx2 = np.asarray(f.omx2)
    det = np.asarray(f.det)
    q = np.full(n_events, float(e_beam))

    fort1 = fortran_density(binary, x1, q * q, workdir)
    fort2 = fortran_density(binary, x2, q * q, workdir)
    products = np.stack(
        [fort1[:, FULL_PIDS.index(a)] * fort2[:, FULL_PIDS.index(b)]
         for a, b in zip(pids1, pids2)], axis=1)
    absolute = np.abs(products)
    total = absolute.sum(axis=1)
    # the Fortran extrapolates outside its grid and can return an all-zero
    # row there; use the same points for both sides rather than silently
    # comparing different samples
    usable = (total > 0) & np.isfinite(total)

    out = lum(x1, omx1, x2, omx2, q, rng.uniform(0, 1, n_events))
    weight = np.asarray(out.luminosity) * det
    component = np.asarray(out.component_index)
    usable &= np.isfinite(weight) & (weight != 0)

    # the electron radiates in components n = 2, 4 and the positron in n = 3, 4
    # (0-based 1,3 and 2,3): which ones is not symmetric between the beams
    w = weight[usable]
    ms_frac = ((w * np.isin(component[usable], [1, 3])).sum()
               + (w * np.isin(component[usable], [2, 3])).sum()) / (2 * w.sum())

    probs = absolute[usable] / total[usable, None]
    wf = (products.sum(axis=1) * det)[usable]
    fort_frac = ((wf * probs[:, [1, 3]].sum(axis=1)).sum()
                 + (wf * probs[:, [2, 3]].sum(axis=1)).sum()) / (2 * wf.sum())
    return ms_frac, fort_frac, int(usable.sum())


def weighted(values, weights):
    total = weights.sum()
    return float((values * weights).sum() / total) if total else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--events', type=int, default=200000)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.root, 'madspace', 'install'))
    sys.path.insert(0, os.path.join(args.root, 'vendor', 'eepdfgrid'))
    import madspace as ms
    import gen_eepdf_lhagrid as gen

    sets = list(SETS)
    binaries = build_references(args.root, sets)
    workdir = tempfile.mkdtemp()
    try:
        print('=' * 78)
        print('1. luminosity  sum_n f1_n f2_n   (madspace vs Fortran compute_eepdf)')
        print('=' * 78)
        print('%-12s %7s %11s %11s %11s' % ('set', 'points', 'median', 'p99', 'max'))
        handles = {}
        for name in sets:
            if name not in binaries:
                print('%-12s  no Fortran reference could be built' % name)
                continue
            work = os.path.join(workdir, name)
            os.makedirs(work, exist_ok=True)
            shutil.copy(os.path.join(args.root, 'Template', 'Common', 'Source',
                                     'PDF', 'lep_densities', name, 'eepdf.f'), work)
            dat, meta_path, _, _ = gen.generate(work, name)
            with open(meta_path) as fsock:
                meta = json.load(fsock)
            lum, stats = check_luminosity(ms, meta, dat, binaries[name], work)
            print('%-12s %7d %11.2e %11.2e %11.2e'
                  % (name, stats['n'], stats['median'], stats['p99'],
                     stats['maximum']))
            handles[name] = (lum, meta, work)

        print()
        print('=' * 78)
        print('2. beamstrahlung split   (madspace vs eepdf_sample_bstrahl)')
        print('=' * 78)
        print('%-12s %8s %8s %12s %14s'
              % ('set', 'q_term', 'points', 'radiating', 'max rel diff'))
        for name in sets:
            if name not in handles:
                continue
            _lum, meta, work = handles[name]
            if 'beamstrahlung' not in meta:
                print('%-12s  pure ISR, nothing to split' % name)
                continue
            binary = build_sampler_reference(args.root, name)
            if binary is None:
                print('%-12s  no Fortran reference could be built' % name)
                continue
            stats = check_split(ms, meta, binary, work, SETS[name]['e_beam'])
            print('%-12s %8s %8d %8d/%d %14.2e'
                  % (name, stats['q_term'], stats['n'], stats['radiating'],
                     stats['n'], stats['maximum']))

        print()
        print('=' * 78)
        print('3. how often a beam radiates   (same points, madspace vs Fortran)')
        print('=' * 78)
        print('%-12s %8s %12s %12s %10s %14s'
              % ('set', 'points', 'madspace', 'Fortran', 'rel diff',
                 'full MG5 run'))
        for name in sets:
            if name not in handles:
                continue
            lum, meta, work = handles[name]
            spec = SETS[name]
            got, ref, used = radiating_fraction(
                ms, meta, lum, binaries[name], work, spec['e_beam'],
                args.events)
            print('%-12s %8d %11.2f%% %11.2f%% %9.1f%% %13.1f%%'
                  % (name, used, 100 * got, 100 * ref,
                     100 * abs(got / ref - 1), 100 * spec['madevent_events']))
        print()
        print('The last column is a full MadEvent run of e- e+ > mu- mu+ with cuts.')
        print('It is a different quantity -- the matrix element and the cuts reweight')
        print('x before any beam radiates -- and is shown only so the two are not')
        print('mistaken for each other. The Fortran column is the reference here.')
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == '__main__':
    main()
