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
"""Dressed-lepton beams in madspace: the density and its sampling.

Three layers, each checked against the one below it:

  * vendor/eepdfgrid/gen_eepdf_lhagrid.py resamples the grids of eepdf.f onto
    a mesh madspace's log-bicubic kernel can reproduce. Its numpy
    reimplementation of CERNLIB's dfint must agree with the table it was read
    from -- it is the reference the resampling is done against, so an error
    there is invisible everywhere else -- and madspace, reading the emitted
    grid, must reproduce that reference. The latter is the accuracy gate: a
    plain transcription of the shipped 100x20 table fails it at the 1.4e-2
    level, which is why the script resamples at all.

  * EEPartonDensity restores the (1-x) endpoint the grid has factored out.

  * LeptonLuminosity samples x from that endpoint, leaving the (1-x)**p of the
    transformation out of its Jacobian because the density carries it. Neither
    piece can be checked alone, so the last test integrates the two together
    against a quadrature that uses neither.
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

    def test_madspace_applies_the_reduction(self):
        """EEPartonDensity restores the endpoint the grid has factored out.

        The grid holds the reduced density; the generator-side quantity is

            compute_eepdf = eepdf_tilde * (1-x)**(-kappa + ps_expo)

        with kappa analytic in Q. This checks the whole product, which is what
        a cross section would use, against the same reduction applied in numpy
        from the companion metadata -- so a wrong kappa table, a wrong sign or
        a wrong scale convention all show up here.
        """
        ms = _madspace()
        if ms is None:
            self.skipTest('madspace build required')
        if not hasattr(ms, 'EEPartonDensity'):
            self.skipTest('madspace build predates EEPartonDensity')

        ps_expo = 0.96  # get_ee_expo() of Template/LO/Source/PDF/pdg2pdf.f
        tmp = tempfile.mkdtemp()
        try:
            shutil.copy(os.path.join(SET_DIR, 'eepdf.f'), tmp)
            dat, meta_path, _, _ = gen.generate(tmp, SET)
            import json
            with open(meta_path) as fsock:
                meta = json.load(fsock)
            pids = meta['pids']
            xlo, xhi = meta['usable_x_range']
            qlo, qhi = meta['usable_q_range']

            def beam_of(pid):
                return 11 if pid % 10 == 2 else -11

            # kappa = c_beta*beta + c_b*b + c_1; the Q-dependent c_beta*beta
            # part is what EEPartonDensity applies internally, so only the
            # constant remainder is handed to it.
            offsets, kappa_const = [], []
            for pid in pids:
                k = meta['kappa']['%d_%d' % (pid // 1000, beam_of(pid))]
                self.assertEqual(k['c_beta'], -1.)
                const = k['c_b'] * meta['b_exp'] + k['c_1']
                kappa_const.append(const)
                offsets.append(ps_expo - const)

            ctx = ms.default_context()
            grid = ms.PdfGrid(dat)
            prefix = 'test_eepdf_red_'
            grid.initialize_globals(ctx, prefix)
            dens = ms.EEPartonDensity(grid, pids, offsets, prefix=prefix)
            dens.initialize_globals(ctx)

            rng = np.random.default_rng(17)
            n = 2000
            x = np.clip(1 - 10 ** rng.uniform(-8, 0, n), xlo, min(xhi, 1 - 1e-8))
            q = 10 ** rng.uniform(np.log10(qlo) * (1 - 1e-9),
                                  np.log10(qhi) * (1 - 1e-9), n)
            omx = 1.0 - x

            yv, zv = np.asarray(self.yv), np.asarray(self.zv)
            z = np.log(q / gen.ME)
            beta = meta['alphaem'] / np.pi * (2 * z - 1)
            keys = gen.physical_keys(self.grids)

            for j, (pid, key) in enumerate(zip(pids, keys)):
                self.assertEqual(gen.pid_of(*key), pid)
                reduced = gen.dfint2d(yv, zv, np.asarray(self.grids[key]), x, z)
                ref = reduced * omx ** (beta - kappa_const[j] + ps_expo)
                # me_int_t is 32-bit; an int64 index is rejected, and the
                # binding reports that as a shape error
                got = np.asarray(dens(x, omx, q, np.full(n, j, dtype=np.int32)))
                mask = np.abs(ref) > 1e-6 * np.abs(ref).max()
                rel = np.abs(got[mask] - ref[mask]) / np.abs(ref[mask])
                # measured against the compiled Fortran compute_eepdf:
                # p99 2.3e-4, max 2.5e-3, the same tail the grid alone has,
                # so the reduction itself contributes nothing visible
                self.assertLess(np.median(rel), 1e-5, 'pid %d' % pid)
                self.assertLess(np.percentile(rel, 99), 1e-3, 'pid %d' % pid)
                self.assertLess(rel.max(), 1e-2, 'pid %d' % pid)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipIf(np is None or gen is None, 'numpy and the generator required')
@unittest.skipIf(not os.path.isdir(SET_DIR), '%s not shipped' % SET)
class TestLeptonLuminosity(unittest.TestCase):
    """Sampling the momentum fractions for dressed-lepton beams.

    The mapping draws each beam from its (1-x) endpoint and deliberately
    leaves the (1-x)**p of the transformation out of its Jacobian, because the
    density carries exactly that factor. That cancellation is the one thing
    here that cannot be checked by looking at either piece alone, so the last
    test integrates the two together against an independent quadrature.
    """

    @classmethod
    def setUpClass(cls):
        cls.ms = _madspace()
        if cls.ms is None or not hasattr(cls.ms, 'LeptonLuminosity'):
            raise unittest.SkipTest('madspace build with LeptonLuminosity required')
        cls.s_lab = 240.0 ** 2
        cls.power = 0.96   # get_ee_expo()

    def _forward(self, tau_min, n=4000, seed=23):
        lum = self.ms.LeptonLuminosity(self.s_lab, tau_min * self.s_lab, self.power)
        rng = np.random.default_rng(seed)
        r1 = rng.uniform(1e-6, 1.0, n)
        r2 = rng.uniform(1e-6, 1.0, n)
        return lum, r1, r2, lum.map_forward([r1, r2])

    def test_matches_generate_x_ee(self):
        """The closed form of genps.f's generate_x_ee, both beams."""
        p = self.power
        for tau_min in (0.0, 1e-4, 0.25):
            _, r1, r2, f = self._forward(tau_min)
            span1 = 1.0 - tau_min
            omx1 = r1 ** (1.0 / (1.0 - p)) * span1
            x1 = 1.0 - omx1
            # the second beam's lower end is fixed by the first draw
            span2 = 1.0 - tau_min / x1
            omx2 = r2 ** (1.0 / (1.0 - p)) * span2
            x2 = 1.0 - omx2
            det = (span1 ** (1.0 - p) / (1.0 - p)) * (span2 ** (1.0 - p) / (1.0 - p))
            for name, ref in (('x1', x1), ('omx1', omx1), ('x2', x2),
                              ('omx2', omx2), ('det', det)):
                got = np.asarray(getattr(f, name))
                np.testing.assert_allclose(
                    got, ref, rtol=1e-13,
                    err_msg='%s at tau_min=%g' % (name, tau_min))
            np.testing.assert_allclose(
                np.asarray(f.s_hat), x1 * x2 * self.s_lab, rtol=1e-13)

    def test_inverse_round_trips(self):
        for tau_min in (0.0, 1e-4, 0.25):
            lum, r1, r2, f = self._forward(tau_min, seed=31)
            inv = lum.map_inverse([f.x1, f.omx1, f.x2, f.omx2, f.s_hat])
            np.testing.assert_allclose(np.asarray(inv.r1), r1, rtol=1e-12)
            np.testing.assert_allclose(np.asarray(inv.r2), r2, rtol=1e-12)
            # both directions report the same Jacobian, as Luminosity does
            np.testing.assert_allclose(
                np.asarray(inv.det), np.asarray(f.det), rtol=1e-13)

    def test_samples_the_requested_density(self):
        """omx**(1-p) is the CDF, so it must come out uniform."""
        lum = self.ms.LeptonLuminosity(self.s_lab, 0.0, self.power)
        rng = np.random.default_rng(5)
        n = 200000
        f = lum.map_forward([rng.uniform(0, 1, n), rng.uniform(0, 1, n)])
        u = np.asarray(f.omx1) ** (1.0 - self.power)
        self.assertAlmostEqual(u.mean(), 0.5, places=2)
        self.assertAlmostEqual(u.std(), 1.0 / np.sqrt(12.0), places=2)

    def test_rejects_a_power_that_cannot_be_normalised(self):
        """p >= 1 makes (1-x)**-p non-integrable; p < 0 flees the endpoint."""
        for bad in (1.0, 1.5, -0.1):
            self.assertRaises(
                ValueError, self.ms.LeptonLuminosity, self.s_lab, 0.0, bad)

    def test_jacobian_and_density_integrate_correctly(self):
        """The (1-x)**p left out of the Jacobian is the one the density has.

        Compared against a quadrature that uses neither: substituting
        1-x = exp(-s) turns the endpoint into an exponential tail. The tail
        has to be followed far enough -- the delta-like components fall only
        as exp(-beta s) with beta ~ 0.06, so truncating at s = 80 loses a
        full percent and looks exactly like a broken Jacobian.
        """
        tmp = tempfile.mkdtemp()
        try:
            shutil.copy(os.path.join(SET_DIR, 'eepdf.f'), tmp)
            dat, meta_path, _, _ = gen.generate(tmp, SET)
            import json
            with open(meta_path) as fsock:
                meta = json.load(fsock)
            pids = meta['pids']
            yv, zv, grids = gen.parse_eepdf(os.path.join(tmp, 'eepdf.f'))
            yv, zv = np.asarray(yv), np.asarray(zv)
            keys = gen.physical_keys(grids)

            def beam_of(pid):
                return 11 if pid % 10 == 2 else -11

            kappa_const, offsets = [], []
            for pid in pids:
                k = meta['kappa']['%d_%d' % (pid // 1000, beam_of(pid))]
                const = k['c_b'] * meta['b_exp'] + k['c_1']
                kappa_const.append(const)
                offsets.append(self.power - const)

            q_val = 120.0
            z = np.log(q_val / gen.ME)
            beta = meta['alphaem'] / np.pi * (2 * z - 1)

            s = np.linspace(0.0, 700.0, 300001)
            u = np.exp(-s)
            x_ref = 1.0 - u

            ms = self.ms
            ctx = ms.default_context()
            grid = ms.PdfGrid(dat)
            prefix = 'test_lumi_'
            grid.initialize_globals(ctx, prefix)
            dens = ms.EEPartonDensity(grid, pids, offsets, prefix=prefix)
            dens.initialize_globals(ctx)
            lum = ms.LeptonLuminosity((2 * q_val) ** 2, 0.0, self.power)

            rng = np.random.default_rng(99)
            n = 300000
            f = lum.map_forward([rng.uniform(0, 1, n), rng.uniform(0, 1, n)])
            x1 = np.asarray(f.x1)
            omx1 = np.asarray(f.omx1)
            q = np.full(n, q_val)
            # tau_min = 0, so beam 2 contributes a constant factor; drop it to
            # compare against the single-beam integral
            det1 = 1.0 / (1.0 - self.power)

            for j, pid in enumerate(pids):
                g = gen.dfint2d(yv, zv, np.asarray(grids[keys[j]]), x_ref,
                                np.full_like(x_ref, z))
                ref = np.trapezoid(g * u ** (beta - kappa_const[j]) * u, s)
                w = np.asarray(
                    dens(x1, omx1, q, np.full(n, j, dtype=np.int32))) * det1
                mc = w.mean()
                err = w.std() / np.sqrt(n)
                self.assertLess(
                    abs(mc - ref), max(5 * err, 1e-3 * abs(ref)),
                    'pid %d: MC %.8e vs quadrature %.8e' % (pid, mc, ref))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipIf(np is None, 'numpy required')
class TestLuminosityChannels(unittest.TestCase):
    """The two sampling channels, and the Jacobian each of them omits.

    A dressed-lepton density is stored reduced by ps_expo, so a channel's
    Jacobian has to end up supplying exactly (1-x)**-p per beam: the drawn
    fractions get it by *omission* (the density cancels it), the derived one
    by an explicit factor. Multiplying a channel's Jacobian by
    (omx1*omx2)**p therefore has to integrate the unit square to 1 -- a check
    that needs no density at all, and the only one that pins the convention
    down for both channels at once.
    """

    @classmethod
    def setUpClass(cls):
        cls.ms = _madspace()
        if cls.ms is None or not hasattr(cls.ms, 'LeptonLuminosityInvariant'):
            raise unittest.SkipTest('madspace build with the channels required')
        cls.s_lab = 240.0 ** 2
        cls.power = 0.96

    def _unit_square(self, f):
        w = (np.asarray(f.det)
             * (np.asarray(f.omx1) * np.asarray(f.omx2)) ** self.power)
        return w.mean(), w.std() / np.sqrt(len(w))

    def test_endpoint_channel_covers_the_unit_square(self):
        lum = self.ms.LeptonLuminosity(self.s_lab, 0.0, self.power)
        rng = np.random.default_rng(3)
        n = 400000
        mean, err = self._unit_square(
            lum.map_forward([rng.uniform(0, 1, n), rng.uniform(0, 1, n)]))
        self.assertLess(abs(mean - 1.0), 4 * err + 1e-3)

    def test_invariant_channel_covers_the_unit_square(self):
        """Same measure through a completely different parametrisation."""
        rng = np.random.default_rng(3)
        n = 400000
        for power in (0.0, 1.0):
            lum = self.ms.LeptonLuminosityInvariant(
                self.s_lab, 0.0, self.s_lab, self.power, 0.0, 0.0, power)
            mean, err = self._unit_square(lum.map_forward(
                [rng.uniform(0, 1, n), rng.uniform(0, 1, n),
                 rng.uniform(0, 1, n)]))
            self.assertLess(abs(mean - 1.0), 4 * err + 1e-3,
                            'invariant_power=%g gave %.5f' % (power, mean))

    def test_invariant_channel_geometry(self):
        """The drawn fraction is the larger one, and either beam can carry it."""
        lum = self.ms.LeptonLuminosityInvariant(
            self.s_lab, 0.0, self.s_lab, self.power)
        rng = np.random.default_rng(11)
        n = 100000
        f = lum.map_forward([rng.uniform(0, 1, n), rng.uniform(0, 1, n),
                             rng.uniform(0, 1, n)])
        x1, x2 = np.asarray(f.x1), np.asarray(f.x2)
        tau = x1 * x2
        self.assertTrue((np.maximum(x1, x2) >= np.sqrt(tau) - 1e-12).all())
        np.testing.assert_allclose(np.asarray(f.s_hat), tau * self.s_lab,
                                   rtol=1e-12)
        np.testing.assert_allclose(np.asarray(f.omx1), 1.0 - x1, atol=1e-12)
        # the swap has to be even, or one ordering is sampled twice as often
        self.assertAlmostEqual((x1 > x2).mean(), 0.5, places=2)

    def test_invariant_channel_has_no_inverse(self):
        """Forward-only, and it says so rather than inventing an answer."""
        lum = self.ms.LeptonLuminosityInvariant(
            self.s_lab, 0.0, self.s_lab, self.power)
        ones = np.full(3, 0.5)
        self.assertRaises(Exception, lum.map_inverse,
                          [ones, ones, ones, ones, ones])

    def test_hadronic_luminosity_still_builds(self):
        """Regression: Luminosity read Invariant's output under the wrong name.

        Invariant returns "invariant" and "random"; Luminosity asked for "s"
        and "r", so building its forward map threw as soon as anyone called
        it. Nothing exercised it -- test_doc_convention only reads the
        docstring -- so it had rotted unseen.
        """
        lum = self.ms.Luminosity(13000.0 ** 2, 100.0 ** 2)
        r_s = np.array([0.3, 0.7, 0.5])
        r_x = np.array([0.4, 0.6, 0.5])
        f = lum.map_forward([r_s, r_x])
        np.testing.assert_allclose(
            np.asarray(f.s_hat),
            np.asarray(f.x1) * np.asarray(f.x2) * 13000.0 ** 2, rtol=1e-12)
        inv = lum.map_inverse([f.x1, f.x2, f.s_hat])
        np.testing.assert_allclose(np.asarray(inv.r_s), r_s, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(inv.r_x), r_x, rtol=1e-12)


@unittest.skipIf(np is None, 'numpy required')
class TestBeamstrahlungSplit(unittest.TestCase):
    """Splitting x into its beamstrahlung and ISR parts, and writing it out.

    The sampler is a port of eepdf_sample_bstrahl (bstrahl.f), and it was
    checked against the compiled Fortran to 1e-16 for all three collider
    q-term forms -- vendor/eepdfgrid/dump_bstrahl_sample.f is that reference,
    kept because it is the only thing that pins the spectrum down. What is
    checked here are the properties that a wrong port would break without
    anyone noticing: the collider-dependent factor actually mattering, the
    non-radiating components staying exactly 1, and the split being a split.
    """

    @classmethod
    def setUpClass(cls):
        cls.ms = _madspace()
        if cls.ms is None or not hasattr(cls.ms, 'BeamstrahlungSampler'):
            raise unittest.SkipTest('madspace build with the sampler required')
        if gen is None or not os.path.isdir(SET_DIR):
            raise unittest.SkipTest('the generator and %s are required' % SET)
        import json
        tmp = tempfile.mkdtemp()
        try:
            shutil.copy(os.path.join(SET_DIR, 'eepdf.f'), tmp)
            _, meta_path, _, _ = gen.generate(tmp, SET)
            with open(meta_path) as fsock:
                cls.meta = json.load(fsock)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        cls.bs = cls.meta.get('beamstrahlung')
        if cls.bs is None:
            raise unittest.SkipTest('%s carries no beamstrahlung' % SET)

    def _params(self, beamid):
        ms = self.ms
        out = []
        for n in (1, 2, 3, 4):
            key = '%d_%d' % (n, beamid)
            if key in self.bs['parameters']:
                p = self.bs['parameters'][key]
                out.append(ms.BeamstrahlungParameters(p['a'], p['p'], p['q'], True))
            else:
                out.append(ms.BeamstrahlungParameters())
        return out

    def _sampler(self, q_term=None):
        ms = self.ms
        names = {'sqrt': ms.BeamstrahlungQTerm.sqrt_u,
                 'u15': ms.BeamstrahlungQTerm.u_three_halves,
                 'ym15': ms.BeamstrahlungQTerm.y_minus_three_halves}
        return ms.BeamstrahlungSampler(
            self._params(11), self._params(-11),
            names[q_term or self.bs['q_term']])

    def test_metadata_carries_the_collider_dependent_factor(self):
        """Which q-term a set uses has to be recorded, not guessed.

        It differs between machines and getting it wrong is silent: the
        normalisation of the split drifts and nothing fails.
        """
        self.assertIn(self.bs['q_term'], ('sqrt', 'u15', 'ym15'))
        # the beams do not radiate in the same components
        self.assertEqual(sorted(self.bs['parameters']),
                         ['2_11', '3_-11', '4_-11', '4_11'])

    def test_non_radiating_components_return_exactly_one(self):
        """The delta(1-x_BS) pieces: the beam did not radiate at all."""
        s = self._sampler()
        # component 0 radiates on neither beam, 1 only on the electron
        for comp, expect in ((0, (1.0, 1.0)), (1, (None, 1.0))):
            got = s.sample(0.9, 0.9, 120.0 ** 2, comp, 0.3, 0.4, 0.5, 0.6)
            for value, want in zip(got, expect):
                if want is not None:
                    self.assertEqual(value, want, 'component %d' % comp)

    def test_the_split_is_a_split(self):
        s = self._sampler()
        rng = np.random.default_rng(3)
        for _ in range(300):
            x = float(np.clip(1 - 10 ** rng.uniform(-7, -0.05), 1e-6, 1 - 1e-12))
            xbs, _ = s.sample(x, x, 120.0 ** 2, 3,
                              *rng.uniform(0, 1, 4).tolist())
            self.assertGreater(xbs, 0.0)
            self.assertLessEqual(xbs, 1.0)
            # x = x_BS * x_ISR with x_ISR <= 1, so x_BS can never fall below x
            self.assertGreaterEqual(xbs, x - 1e-12)

    def test_q_term_changes_the_answer(self):
        """A wrong collider table must not be silent."""
        base = self._sampler('sqrt')
        others = [self._sampler('u15'), self._sampler('ym15')]
        args = (0.97, 0.97, 120.0 ** 2, 3, 0.31, 0.62, 0.17, 0.48)
        ref = base.sample(*args)[0]
        for other in others:
            self.assertNotAlmostEqual(other.sample(*args)[0], ref, places=6)

    def test_cdf_inversion_is_monotone(self):
        """u2 walks the CDF, so x_BS has to move monotonically with it."""
        s = self._sampler()
        vals = [s.sample(0.98, 0.98, 120.0 ** 2, 3, 0.2, u2, 0.5, 0.5)[0]
                for u2 in np.linspace(0.01, 0.99, 40)]
        diffs = np.diff(vals)
        self.assertTrue((diffs >= -1e-12).all() or (diffs <= 1e-12).all(), vals)

    def test_lhe_element_round_trips(self):
        """Written by madspace, read by the tools that consume it."""
        ms = self.ms
        sys.path.insert(0, _root)
        import madgraph.various.lhe_parser as lhe_parser
        particle = ms.LHEParticle(
            pdg_id=11, status_code=-1, mother1=0, mother2=0, color=0,
            anti_color=0, px=0., py=0., pz=120., energy=120., mass=0.,
            lifetime=0., spin=9.)
        event = ms.LHEEvent(process_id=1, weight=1.0, scale=120.0,
                            alpha_qed=0.0075, alpha_qcd=0.118,
                            particles=[particle])
        self.assertIsNone(
            lhe_parser.Event(event.format()).parse_beamstrahlung(),
            'the element must be absent unless it was set')
        event.beamstrahlung = (0.99987654, 1.0)
        text = event.format()
        self.assertIn('<beamstrahlung>', text)
        parsed = lhe_parser.Event(text).parse_beamstrahlung()
        self.assertEqual(len(parsed), 2)
        self.assertAlmostEqual(parsed[0], 0.99987654)
        self.assertAlmostEqual(parsed[1], 1.0)
        # and it survives being read and written back out
        again = lhe_parser.Event(str(lhe_parser.Event(text))).parse_beamstrahlung()
        self.assertAlmostEqual(again[0], 0.99987654)


@unittest.skipIf(np is None, 'numpy required')
class TestDressedPhaseSpace(unittest.TestCase):
    """PhaseSpaceMapping with dressed-lepton beams.

    madspace samples s_hat first -- the topology's root invariant carries the
    resonance structure -- and only then splits it into x1, x2. The hadronic
    split x1 = tau^r puts essentially no weight at x -> 1, which is exactly
    where a dressed-lepton density lives, so the split is replaced rather than
    the whole mapping.
    """

    @classmethod
    def setUpClass(cls):
        cls.ms = _madspace()
        if cls.ms is None:
            raise unittest.SkipTest('madspace build required')
        try:
            cls.ms.PhaseSpaceMapping(
                [0.] * 4, 240.0, False, 0.8,
                cls.ms.PhaseSpaceMapping.TChannelMode.rambo, None, None, 0.96)
        except TypeError:
            raise unittest.SkipTest(
                'madspace build predates lepton_sampling_power')
        cls.power = 0.96
        cls.e_cm = 240.0

    def _mappings(self, masses):
        M = self.ms.PhaseSpaceMapping
        mode = M.TChannelMode.rambo
        return (M(masses, self.e_cm, False, 0.8, mode),
                M(masses, self.e_cm, False, 0.8, mode, None, None, self.power))

    def test_the_split_costs_one_extra_random_number(self):
        """It also has to decide which beam carries the drawn fraction."""
        had, lep = self._mappings([0.] * 4)
        self.assertEqual(lep.random_dim(), had.random_dim() + 1)

    def test_same_phase_space_volume(self):
        """The two splits cover the same space, so they must agree on it.

        The dressed Jacobian omits (1-x)**p on the drawn beam and supplies
        (1-x)**-p on the derived one, so the integrand carries (omx1*omx2)**p
        -- which is what the density does. Getting the s_hat -> tau conversion
        wrong shows up here as a factor of s_lab.
        """
        for masses in ([0.] * 4, [0., 0., 5., 5., 0.]):
            had, lep = self._mappings(masses)
            n = 400000
            rng = np.random.default_rng(9)
            fh = had.map_forward([rng.uniform(0, 1, (n, had.random_dim()))])
            wh = np.nan_to_num(np.asarray(fh.det))
            fl = lep.map_forward([rng.uniform(0, 1, (n, lep.random_dim()))])
            wl = np.nan_to_num(
                np.asarray(fl.det)
                * ((1 - np.asarray(fl.x1)) * (1 - np.asarray(fl.x2))) ** self.power)
            err = np.hypot(wh.std(), wl.std()) / np.sqrt(n)
            self.assertLess(abs(wh.mean() - wl.mean()),
                            max(5 * err, 0.02 * abs(wh.mean())),
                            'masses %s: %.6e vs %.6e' % (masses, wh.mean(), wl.mean()))

    def test_weight_sits_at_the_endpoint(self):
        """The whole point: the dressed split samples where the density is."""
        had, lep = self._mappings([0.] * 4)
        n = 100000
        rng = np.random.default_rng(1)
        fracs = {}
        for tag, m in (('hadronic', had), ('dressed', lep)):
            f = m.map_forward([rng.uniform(0, 1, (n, m.random_dim()))])
            omx = np.minimum(1 - np.asarray(f.x1), 1 - np.asarray(f.x2))
            fracs[tag] = (omx < 1e-3).mean()
        self.assertLess(fracs['hadronic'], 0.05)
        self.assertGreater(fracs['dressed'], 0.5)

    def test_no_inverse_with_dressed_beams(self):
        """Forward only, and it says so. Integrand never asks for this one."""
        _had, lep = self._mappings([0.] * 4)
        n = 8
        rng = np.random.default_rng(2)
        f = lep.map_forward([rng.uniform(0, 1, (n, lep.random_dim()))])
        self.assertRaises(Exception, lep.map_inverse, [f.momenta, f.x1, f.x2])

    def test_s_hat_is_sampled_at_its_upper_endpoint(self):
        """A dressed density piles its weight where neither beam radiated.

        Integrating out the beam split at fixed tau leaves
        (1-tau)**(2*beta-2*kappa0+1), about (1-tau)**-0.886 for the sharply
        peaked components. The hadronic root sampling spreads s_hat over its
        whole range and puts 0.001% of its points where 7% of the weight is,
        which is what made the unweighting efficiency round to zero.
        """
        had, lep = self._mappings([0.] * 4)
        n = 100000
        rng = np.random.default_rng(9)
        fracs = {}
        for tag, m in (('hadronic', had), ('dressed', lep)):
            f = m.map_forward([rng.uniform(0, 1, (n, m.random_dim()))])
            omtau = 1.0 - np.asarray(f.x1) * np.asarray(f.x2)
            fracs[tag] = (omtau < 1e-4).mean()
        self.assertLess(fracs['hadronic'], 0.05)
        self.assertGreater(fracs['dressed'], 0.4)

    def test_no_infinite_weights_at_the_endpoint(self):
        """x_b = tau/x_a can round above 1 once tau sits at its endpoint.

        The derived beam's Jacobian is written as omx_b**-p so that such a
        point gets weight zero; as 1/omx_b**p it divided by the zero the pow
        guard returns and poisoned the batch with an infinity.
        """
        _had, lep = self._mappings([0.] * 4)
        n = 200000
        rng = np.random.default_rng(9)
        f = lep.map_forward([rng.uniform(0, 1, (n, lep.random_dim()))])
        det = np.asarray(f.det)
        omx1, omx2 = np.asarray(f.omx1), np.asarray(f.omx2)
        self.assertEqual(int((~np.isfinite(det)).sum()), 0)
        self.assertEqual(int((~np.isfinite(omx1) | ~np.isfinite(omx2)).sum()), 0)
        self.assertEqual(int(((omx1 < 0) | (omx2 < 0)).sum()), 0)


@unittest.skipIf(np is None or gen is None, 'numpy and the generator required')
class TestLheAnnotation(unittest.TestCase):
    """Sampling the beamstrahlung split onto a written event file.

    It runs once per written event, after unweighting, because the fractions
    do not enter the weight. x comes back from the initial-state momenta, so
    nothing has to be threaded through the integrand or the event record.
    """

    @classmethod
    def setUpClass(cls):
        cls.ms = _madspace()
        if cls.ms is None or not hasattr(cls.ms, 'BeamstrahlungSampler'):
            raise unittest.SkipTest('madspace build with the sampler required')
        sys.path.insert(0, os.path.join(
            _root, 'madgraph', 'iolibs', 'template_files', 'mg7'))
        try:
            import lepton_density
        except ImportError:
            raise unittest.SkipTest('lepton_density helper not importable')
        cls.mod = lepton_density
        cls.source = os.path.join(_root, 'Template', 'Common', 'Source')

    def _density(self, name, tmp):
        return self.mod.LeptonDensity(
            name, self.source, _root, self.ms, self.ms.default_context(),
            240.0, beam_pdgs=(-11, 11),
            cache_dir=os.path.join(tmp, name), prefix='ann_%s_' % name)

    @staticmethod
    def _lhe(path, fractions):
        """A minimal LHE whose initial-state momenta encode the given x."""
        events = []
        for x1, x2 in fractions:
            e1, e2 = 120.0 * x1, 120.0 * x2
            events.append(
                '<event>\n   4    1 +1.0e+00 9.1188e+01 7.5e-03 1.2e-01\n'
                '  11   -1    0    0    0    0 +0.0e+00 +0.0e+00 %+.16e %.16e 0.0 0.0 9.0\n'
                ' -11   -1    0    0    0    0 +0.0e+00 +0.0e+00 %+.16e %.16e 0.0 0.0 9.0\n'
                '  13    1    1    2    0    0 +0.0e+00 +0.0e+00 +0.0e+00 %.16e 0.0 0.0 9.0\n'
                ' -13    1    1    2    0    0 +0.0e+00 +0.0e+00 +0.0e+00 %.16e 0.0 0.0 9.0\n'
                '</event>\n' % (e1, e1, -e2, e2, (e1 + e2) / 2, (e1 + e2) / 2))
        with open(path, 'w') as fsock:
            fsock.write('<LesHouchesEvents version="3.0">\n<init>\n'
                        ' 11 -11 1.2e+02 1.2e+02 0 0 0 0 -4 1\n'
                        ' 1.0 1.0e-03 1.0 1\n</init>\n'
                        + ''.join(events) + '</LesHouchesEvents>\n')

    def test_annotates_every_event(self):
        import madgraph.various.lhe_parser as lhe_parser
        tmp = tempfile.mkdtemp()
        try:
            density = self._density('fcce240ll', tmp)
            if not density.has_beamstrahlung:
                self.skipTest('fcce240ll carries no beamstrahlung')
            path = os.path.join(tmp, 'events.lhe')
            fractions = [(1.0, 1.0), (0.9, 1.0), (0.5, 0.7), (1.0, 0.3)]
            self._lhe(path, fractions)
            self.assertEqual(density.annotate_lhe(path, seed=3), len(fractions))

            text = open(path).read()
            parsed = [lhe_parser.Event('<event>' + ev.split('</event>')[0]
                                       + '</event>').parse_beamstrahlung()
                      for ev in text.split('<event>')[1:]]
            self.assertEqual(len(parsed), len(fractions))
            for (x1, x2), value in zip(fractions, parsed):
                self.assertIsNotNone(value)
                xbs1, xbs2 = value
                # x = x_BS * x_ISR with x_ISR <= 1, so x_BS cannot fall below x
                self.assertGreaterEqual(xbs1, x1 - 1e-9)
                self.assertGreaterEqual(xbs2, x2 - 1e-9)
                self.assertLessEqual(xbs1, 1.0)
                self.assertLessEqual(xbs2, 1.0)
            # a beam that kept all its energy cannot have radiated
            self.assertEqual(parsed[0], [1.0, 1.0])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_pure_isr_writes_nothing(self):
        """isronlyll has no beamstrahlung, so there is no element to write."""
        tmp = tempfile.mkdtemp()
        try:
            density = self._density('isronlyll', tmp)
            self.assertFalse(density.has_beamstrahlung)
            path = os.path.join(tmp, 'events.lhe')
            self._lhe(path, [(0.9, 0.8)])
            self.assertEqual(density.annotate_lhe(path), 0)
            self.assertNotIn('<beamstrahlung>', open(path).read())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
