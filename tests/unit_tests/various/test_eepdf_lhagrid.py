################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Re-tabulation of the dressed-lepton densities as LHAPDF grids.

vendor/eepdfgrid/gen_eepdf_lhagrid.py resamples the grids of eepdf.f onto a
mesh that madspace's log-bicubic kernel can reproduce. Two things have to hold
for that to be worth anything, and both are checked here:

  * the script's numpy reimplementation of CERNLIB's dfint must agree with the
    table it was read from -- it is the reference the resampling is done
    against, so an error there is invisible everywhere else;

  * madspace, reading the emitted grid, must reproduce that reference. This is
    the accuracy gate: a plain transcription of the shipped 100x20 table fails
    it at the 1.4e-2 level, which is why the script resamples at all.
"""
import os
import shutil
import sys
import tempfile
import unittest

_file_path = os.path.dirname(os.path.realpath(__file__))
_root = os.path.abspath(os.path.join(_file_path, '..', '..', '..'))
sys.path.append(os.path.join(_root, 'vendor', 'eepdfgrid'))

try:
    import numpy as np
except ImportError:
    np = None

try:
    import gen_eepdf_lhagrid as gen
except ImportError:
    gen = None

SET = 'fcce240ll'
SET_DIR = os.path.join(_root, 'Template', 'Common', 'Source', 'PDF',
                       'lep_densities', SET)


def _madspace():
    """madspace is an optional build; the gate is skipped without it.

    madspace/install is a vendored site-packages -- it carries its own yaml,
    packaging, pathspec and scikit_build_core -- so it is put on sys.path only
    for the duration of the import. Leaving it there would shadow those
    modules for every test that runs after this one in the same process.
    """
    install = os.path.join(_root, 'madspace', 'install')
    if not os.path.isdir(install):
        return None
    saved = list(sys.path)
    sys.path.insert(0, install)
    try:
        import madspace
    except ImportError:
        return None
    finally:
        sys.path[:] = saved
    # A separate, usually older madspace is often installed in site-packages.
    # It must not be what the gate measures, and if something imported it
    # before us it is already cached under this name, so check rather than
    # assume -- a silently stale build would give a wrong answer, not an error.
    origin = getattr(madspace, '__file__', None) or ''
    if os.path.realpath(install) not in os.path.realpath(origin):
        return None
    return madspace


@unittest.skipIf(np is None or gen is None, 'numpy and the generator required')
@unittest.skipIf(not os.path.isdir(SET_DIR), '%s not shipped' % SET)
class TestEEPdfLhaGrid(unittest.TestCase):
    """Emitting a dressed-lepton density in LHAPDF grid format."""

    @classmethod
    def setUpClass(cls):
        cls.yv, cls.zv, cls.grids = gen.parse_eepdf(
            os.path.join(SET_DIR, 'eepdf.f'))

    def test_parse_shape(self):
        """The shipped set has 16 grids on a shared (y, z) mesh."""
        self.assertEqual(len(self.grids), 16)
        self.assertEqual(len(self.yv), 100)
        self.assertEqual(len(self.zv), 20)
        for grid in self.grids.values():
            self.assertEqual(len(grid), len(self.zv))
            self.assertEqual(len(grid[0]), len(self.yv))

    def test_only_diagonal_grids_carry_data(self):
        """partonid != beamid is identically zero, so those grids are dropped.

        If a future set ever puts data there, dropping them would silently
        lose it -- hence the check rather than the assumption.
        """
        for key, grid in self.grids.items():
            arr = np.asarray(grid)
            if key[1] == key[2]:
                self.assertTrue(np.any(arr != 0.),
                                'diagonal grid %s is empty' % (key,))
            else:
                self.assertTrue(np.all(arr == 0.),
                                'off-diagonal grid %s carries data' % (key,))
        self.assertEqual(len(gen.physical_keys(self.grids)), 8)

    def test_dfint_reproduces_the_table_at_its_knots(self):
        """The numpy dfint is exact on the knots it was built from."""
        yv, zv = np.asarray(self.yv), np.asarray(self.zv)
        key = (2, 2, 2)
        grid = np.asarray(self.grids[key])
        iy = np.arange(len(yv))
        for iz in (0, 7, len(zv) - 1):
            got = gen.dfint2d(yv, zv, grid, yv[iy], np.full(len(iy), zv[iz]))
            np.testing.assert_allclose(got, grid[iz, iy], rtol=1e-13, atol=0)

    def test_dfint_extrapolates_like_cernlib(self):
        """dfint clamps the cell, not the interpolation parameter: outside the
        table it continues the edge cell linearly rather than clipping."""
        yv, zv = np.asarray(self.yv), np.asarray(self.zv)
        grid = np.asarray(self.grids[(2, 2, 2)])
        z = np.array([zv[0]])
        below = gen.dfint2d(yv, zv, grid, np.array([yv[0] - 0.5 * (yv[1] - yv[0])]), z)
        edge = grid[0, 0] - 0.5 * (grid[0, 1] - grid[0, 0])
        np.testing.assert_allclose(below, [edge], rtol=1e-12)

    def test_madspace_reproduces_the_density(self):
        """The accuracy gate, on the grid the generator actually emits."""
        ms = _madspace()
        if ms is None:
            self.skipTest('madspace build required for the accuracy gate')

        tmp = tempfile.mkdtemp()
        try:
            shutil.copy(os.path.join(SET_DIR, 'eepdf.f'), tmp)
            dat, meta_path, _, _ = gen.generate(tmp, SET)

            import json
            with open(meta_path) as fsock:
                meta = json.load(fsock)
            xlo, xhi = meta['usable_x_range']
            qlo, qhi = meta['usable_q_range']

            ctx = ms.default_context()
            grid = ms.PdfGrid(dat)
            prefix = 'test_eepdf_'
            grid.initialize_globals(ctx, prefix)
            pdf = ms.PartonDensity(grid, grid.pids, False, prefix)

            rng = np.random.default_rng(5)
            n = 4000
            x = np.clip(1 - 10 ** rng.uniform(-8, 0, n), xlo, min(xhi, 1 - 1e-8))
            q = 10 ** rng.uniform(np.log10(qlo) * (1 - 1e-9),
                                  np.log10(qhi) * (1 - 1e-9), n)

            got = np.asarray(pdf(x, q))
            yv, zv = np.asarray(self.yv), np.asarray(self.zv)
            z = np.log(q / gen.ME)
            keys = gen.physical_keys(self.grids)
            self.assertEqual([gen.pid_of(*k) for k in keys], list(grid.pids))
            ref = np.stack(
                [gen.dfint2d(yv, zv, np.asarray(self.grids[k]), x, z)
                 for k in keys], axis=1)

            scale = np.abs(ref).max(axis=0, keepdims=True)
            mask = np.abs(ref) > 1e-6 * scale
            rel = np.abs(got[mask] - ref[mask]) / np.abs(ref[mask])
            # measured on this set: median 9e-8, p95 2e-5, max 3e-3. The
            # thresholds are loose enough not to be noise-sensitive and tight
            # enough that a regression to a plain transcription (p95 1.4e-2)
            # fails them.
            self.assertLess(np.median(rel), 1e-5)
            self.assertLess(np.percentile(rel, 95), 1e-3)
            self.assertLess(rel.max(), 1e-2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
