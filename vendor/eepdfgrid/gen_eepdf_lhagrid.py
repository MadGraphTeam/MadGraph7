#!/usr/bin/env python3
"""Re-tabulate a dressed-lepton density as an LHAPDF-format grid file.

madspace carries no dressed-lepton density: its PartonDensity is an LHAPDF
grid interpolator and `beam.leptonic` means "no parton luminosity" outright.
The grids inside eepdf.f are ordinary 2D tables, so writing them in the LHAPDF
`lhagrid1` format lets madspace read them with no new loader and no new
interpolation kernel.

The catch, and the reason this is not a transcription
-----------------------------------------------------
eepdf.f interpolates *multilinearly in (y, z)* on knots that are roughly
uniform in y. madspace interpolates *log-bicubically in (log x, log Q2)*.
Handing madspace the shipped 100x20 table therefore reproduces it only to
about 1.4e-2 at the 95th percentile -- fine nowhere near good enough for a
density that multiplies a cross section. The disagreement is entirely in the
x direction and entirely on the four *continuum* components, because the
density's structure lives in log(1-x) while the shipped knots resolve y.

So this script does not transcribe the table, it *resamples* it: the shipped
grid is evaluated with an exact reimplementation of CERNLIB's dfint (verified
to 5e-16 against the Fortran) on a mesh that is dense in log(x) at small x
*and* in log(1-x) near x -> 1. With the default mesh madspace then reproduces
the Fortran density to ~3e-5 at the 95th percentile.

Two further properties of madspace's kernel that the mesh has to respect:

- It returns exactly 0 on the outermost knots of each axis, so the physical
  range must sit strictly inside the tabulated one; both axes are padded.
- Only the eight grids with `partonid == beamid` carry data; the other eight
  are identically zero in every shipped set and are dropped.

What is *not* tabulated
-----------------------
The generator-side density is

    compute_eepdf = eepdf_tilde(x, Q2, n, id, idbeam) * omx**(-kappa + ps_expo)

with kappa = eepdf_tilde_power(Q2, n, id, idbeam). Only eepdf_tilde is
tabulated; the `(1-x)**(-kappa)` reduction stays analytic and belongs in the
consumer, which is the whole point of the reduction -- baking it in would
destroy the endpoint resolution it exists to preserve. The kappa coefficients
and the component types go to a companion JSON, since the LHAPDF format
cannot carry them.

Usage:
    python3 gen_eepdf_lhagrid.py <MG root> [<set> ...]
"""
import json
import math
import os
import re
import sys

try:
    import numpy as np
except ImportError:
    np = None

# z = log(Q/me) is the grid's scale variable; this is the only place the
# electron mass enters the axis definition, and it must match eepdf.f.
ME = 0.511e-3
# Gmu scheme, as in gridpdfaux.f
ALPHAEM = 0.007562397
# the (1-x) exponent b of eepdf_tilde_power
B_EXP = -2.0 / 3.0

# Mesh defaults. N_SMALL/N_TAIL set the resampled x mesh (geometric in x below
# X_SPLIT, geometric in 1-x above it); Q_SUBDIV subdivides each shipped z cell.
# See the module docstring for the accuracy these buy.
N_SMALL = 400
N_TAIL = 500
X_SPLIT = 0.5
Q_SUBDIV = 3
VALUE_FMT = '%.9e'

class AnalyticDensity(Exception):
    """The set defines eepdf_tilde in closed form; there is no grid."""


FUNC_RE = re.compile(r'^      function eepdf_(\d)_(\d)_(\d)\(y,z\)', re.M)
DATA_RE = re.compile(
    r'      data\s+(yv|zv|\(gridv\(iny,\s*(\d+)\),iny=1,\d+\))\s*/(.*?)/',
    re.S)
NUM_RE = re.compile(r'[-+]?\d*\.\d+[DdEe][-+]?\d+')

# index 1 -> -11, index 2 -> +11, for both partonid and beamid
PDG_OF_INDEX = {1: -11, 2: 11}


def parse_eepdf(path):
    """{(n, a, b): grid[iz][iy]} plus the shared y and z knots."""
    with open(path) as fsock:
        text = fsock.read()
    starts = [(m.start(), tuple(int(g) for g in m.groups()))
              for m in FUNC_RE.finditer(text)]
    if not starts:
        # isrbetll / isrbetmll are analytic reduced structure functions with
        # no table at all -- nothing to resample, and a much easier target for
        # madspace than a grid. Signalled, not guessed at.
        raise AnalyticDensity(path)

    yv = zv = None
    grids = {}
    for i, (pos, key) in enumerate(starts):
        body = text[pos:starts[i + 1][0] if i + 1 < len(starts) else len(text)]
        columns = {}
        for name, index, blob in DATA_RE.findall(body):
            values = [float(t.replace('D', 'E').replace('d', 'e'))
                      for t in NUM_RE.findall(blob)]
            if name == 'yv':
                if yv is not None and yv != values:
                    raise ValueError('y knots differ between grids in %s' % path)
                yv = values
            elif name == 'zv':
                if zv is not None and zv != values:
                    raise ValueError('z knots differ between grids in %s' % path)
                zv = values
            else:
                columns[int(index)] = values
        if sorted(columns) != list(range(1, len(zv) + 1)):
            raise ValueError('eepdf_%d_%d_%d: expected %d z columns, got %d'
                             % (key + (len(zv), len(columns))))
        grid = [columns[iz + 1] for iz in range(len(zv))]
        if any(len(row) != len(yv) for row in grid):
            raise ValueError('eepdf_%d_%d_%d: ragged column' % key)
        grids[key] = grid

    return yv, zv, grids


def _cell(knots, t):
    """CERNLIB dfint's bracketing: binary search, then clamp the *cell*, not
    the interpolation parameter -- so an argument outside the table
    extrapolates linearly from the edge cell rather than being clipped."""
    i = np.clip(np.searchsorted(knots, t, side='right') - 1, 0, len(knots) - 2)
    return i, (t - knots[i]) / (knots[i + 1] - knots[i])


def dfint2d(yv, zv, grid, y, z):
    """Exact reimplementation of the dfint() call in eepdf.f.

    Verified against the compiled Fortran to 5e-16 relative; that check is the
    whole licence for resampling in Python instead of linking the Fortran.
    """
    iy, ey = _cell(yv, y)
    iz, ez = _cell(zv, z)
    return ((1 - ey) * (1 - ez) * grid[iz, iy]
            + ey * (1 - ez) * grid[iz, iy + 1]
            + (1 - ey) * ez * grid[iz + 1, iy]
            + ey * ez * grid[iz + 1, iy + 1])


def x_mesh(yv, n_small=N_SMALL, n_tail=N_TAIL, split=X_SPLIT):
    """Dense in log(x) below `split`, in log(1-x) above it, shipped knots kept.

    Keeping the shipped knots puts the kinks of the piecewise-linear source on
    mesh points, where both interpolation schemes are exact.
    """
    lo, hi = yv[0], yv[-1]
    small = np.geomspace(lo, split, n_small)
    tail = 1.0 - np.geomspace(1.0 - split, 1.0 - hi, n_tail)
    xs = np.unique(np.concatenate([np.asarray(yv), small, tail]))
    return xs[(xs >= lo) & (xs <= hi)]


def q_mesh(zv, subdiv=Q_SUBDIV):
    """Shipped z knots, each cell subdivided `subdiv` times, as Q = me*exp(z)."""
    z = np.asarray(zv, float)
    out = [z[0]]
    for a, b in zip(z[:-1], z[1:]):
        out.extend(a + (b - a) * np.arange(1, subdiv + 1) / subdiv)
    return ME * np.exp(np.asarray(out))


def pad(knots, factor=1.001):
    """madspace's bicubic returns 0 on the outermost knots, so add one knot
    outside each end and keep the physical range strictly interior."""
    k = np.asarray(knots, float)
    return np.concatenate([[k[0] / factor], k, [k[-1] * factor]])


def pid_of(n, a, b):
    """Synthetic id for one (component, partonid, beamid) grid."""
    return 1000 * n + 10 * a + b


def physical_keys(grids):
    """The eight grids with partonid == beamid; the rest are identically zero
    in every shipped set."""
    return [k for k in sorted(grids) if k[1] == k[2]]


def write_lhagrid(path, yv, zv, grids, xs, qs, fmt=VALUE_FMT):
    """One `lhagrid1` block, values with the q index running fastest."""
    keys = physical_keys(grids)
    arrays = {k: np.asarray(grids[k]) for k in keys}
    zs = np.log(qs / ME)
    xg = np.repeat(xs, len(qs))
    zg = np.tile(zs, len(xs))
    cols = [dfint2d(np.asarray(yv), np.asarray(zv), arrays[k], xg, zg)
            for k in keys]

    out = ['PdfType: central', 'Format: lhagrid1', '---',
           ' '.join('%.16e' % x for x in xs),
           ' '.join('%.16e' % q for q in qs),
           ' '.join(str(pid_of(*k)) for k in keys)]
    for row in zip(*cols):
        out.append(' '.join(fmt % v for v in row))
    out.append('---')
    with open(path, 'w') as fsock:
        fsock.write('\n'.join(out) + '\n')
    return [pid_of(*k) for k in keys]


def kappa_table():
    """eepdf_tilde_power as data: kappa = c_beta*beta + c_b*b + c_1, with
    beta = alphaem/pi*(log(Q2/me^2)-1) and b = -2/3 (gridpdfaux.f)."""
    table = {}
    for beamid, reduced in ((11, (2, 4)), (-11, (3, 4))):
        for n in (1, 2, 3, 4):
            table['%d_%d' % (n, beamid)] = (
                {'c_beta': -1., 'c_b': -1., 'c_1': 0.} if n in reduced
                else {'c_beta': -1., 'c_b': 0., 'c_1': 1.})
    return table


def type_table():
    """eepdf_tilde_type: 1 = delta(1-x) peak, 2 = continuum."""
    return {'%d_%d' % (n, beamid): (2 if n in continuum else 1)
            for beamid, continuum in ((11, (2, 4)), (-11, (3, 4)))
            for n in (1, 2, 3, 4)}


def write_meta(path, name, yv, zv, xs, qs, pids):
    meta = {
        'set': name,
        'comment': 'companion to %s.dat, generated by '
                   'vendor/eepdfgrid/gen_eepdf_lhagrid.py from eepdf.f' % name,
        'pid_encoding': '1000*component + 10*partonid_index + beamid_index; '
                        'index 1 = -11, index 2 = +11',
        'pids': pids,
        'components': 4,
        'scale_variable': 'z = log(Q/me)',
        'me': ME,
        'alphaem': ALPHAEM,
        'b_exp': B_EXP,
        'source_knots': {'y': len(yv), 'z': len(zv)},
        'grid_knots': {'x': len(xs), 'q': len(qs)},
        # the outermost knot of each axis is padding: madspace's bicubic
        # returns 0 there, so the usable range is the interior
        'usable_x_range': [xs[1], xs[-2]],
        'usable_q_range': [qs[1], qs[-2]],
        'reduction': 'density = grid * (1-x)**(-kappa + ps_expo)',
        'kappa': kappa_table(),
        'component_type': type_table(),
        'nonzero_only_when': 'partonid == beamid',
    }
    with open(path, 'w') as fsock:
        json.dump(meta, fsock, indent=2, sort_keys=True)
        fsock.write('\n')


def generate(set_dir, name, **mesh_kwargs):
    """Write <name>.dat and <name>.json into `set_dir`; return their paths."""
    if np is None:
        raise RuntimeError('numpy is required to generate the lhagrid tables')
    yv, zv, grids = parse_eepdf(os.path.join(set_dir, 'eepdf.f'))
    xs = pad(x_mesh(yv, **{k: v for k, v in mesh_kwargs.items()
                           if k in ('n_small', 'n_tail', 'split')}))
    qs = pad(q_mesh(zv, **{k: v for k, v in mesh_kwargs.items()
                           if k in ('subdiv',)}))
    dat = os.path.join(set_dir, '%s.dat' % name)
    js = os.path.join(set_dir, '%s.json' % name)
    pids = write_lhagrid(dat, yv, zv, grids, xs, qs)
    write_meta(js, name, yv, zv, xs, qs, pids)
    return dat, js, len(xs), len(qs)


def main(root, names=None):
    base = os.path.join(root, 'Template', 'Common', 'Source', 'PDF',
                        'lep_densities')
    if not names:
        names = sorted(d for d in os.listdir(base)
                       if os.path.isfile(os.path.join(base, d, 'eepdf.f')))
    for name in names:
        try:
            dat, _, nx, nq = generate(os.path.join(base, name), name)
        except AnalyticDensity:
            print('%-12s analytic density, no grid to resample -- skipped'
                  % name)
            continue
        print('%-12s %4d x-knots x %3d q-knots  %5.1f MB  -> %s'
              % (name, nx, nq, os.path.getsize(dat) / 1e6,
                 os.path.basename(dat)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
