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
"""The Effective Vector boson Approximation (pdlabel eva) in mg7/madspace.

  * madspace's EvaFlux (src/kernels/eva.hpp) is a port of madevent's
    ElectroweakFlux.f + ElectroweakFluxDriver.f: compiled together with a
    driver, the Fortran is the reference, to 1e-12, over the vector bosons,
    beam leptons, helicities, polarisations, orders, evolution variables, the
    x cut and scales below the floor.
  * The compute-graph kernel (PartonDensity built from an EvaFlux) gives the
    helicity sum of the same densities.
  * The run card: the madevent settings translate into [beam] keys, and a
    W/Z-initiated process gets EVA beams by default.
"""
import itertools
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_file_path = os.path.dirname(os.path.realpath(__file__))
_root = os.path.abspath(os.path.join(_file_path, '..', '..', '..'))
sys.path.insert(0, _root)

import madgraph.various.misc as misc
import madgraph.various.banner as banner_mod

pjoin = os.path.join

try:
    import numpy as np
except ImportError:
    np = None

# the electroweak inputs of the sm model (restrict_default): MZ, aEWM1 = 132.507,
# Gf = 1.16639e-5 give MW and ee
MZ = 91.188
AEWM1 = 132.507
GF = 1.16639e-5
EE = math.sqrt(4 * math.pi / AEWM1)
MW = math.sqrt(MZ ** 2 / 2 + math.sqrt(
    MZ ** 4 / 4 - (math.pi / AEWM1 * MZ ** 2) / (GF * math.sqrt(2))))


def _madspace():
    """madspace from madspace/install, None without that build (see
    test_eepdf_lhagrid._madspace)."""
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
    origin = getattr(madspace, '__file__', None) or ''
    if os.path.realpath(install) not in os.path.realpath(origin):
        return None
    if not hasattr(madspace, 'EvaFlux'):
        return None
    return madspace


class TestEvaFortranReference(unittest.TestCase):
    """EvaFlux.density against eva_get_pdf_by_PID of the compiled Fortran."""

    driver = """
      program eva_driver
      implicit none
      integer vpid, fpid, vpol, ievo, order, xcut, n, i
      double precision flpol, x, q, ebeam, mu2, pdf
      double precision eva_get_pdf_by_PID
      double precision gal(2)
      common/eva_test_gal/gal
      integer ievo_eva, evaorder, eva_xcut
      common/to_eva/ievo_eva, evaorder, eva_xcut
      read(*,*) gal(1)
      gal(2) = gal(1)
      read(*,*) n
      do i = 1, n
         read(*,*) vpid, fpid, vpol, flpol, x, q, ebeam, ievo, order, xcut
         ievo_eva = ievo
         evaorder = order
         eva_xcut = xcut
         mu2 = q*q
         pdf = eva_get_pdf_by_PID(vpid, fpid, vpol, flpol, x, mu2, ebeam)
         write(*,'(A,ES26.17E3)') 'RES ', pdf
      enddo
      end
"""

    @classmethod
    def setUpClass(cls):
        cls.ms = _madspace()
        if cls.ms is None:
            raise unittest.SkipTest('madspace build with EvaFlux required')
        cls.fc = misc.which('gfortran')
        if not cls.fc:
            raise unittest.SkipTest('gfortran not available')
        cls.tmpdir = tempfile.mkdtemp(prefix='eva_fortran')
        pdf_dir = pjoin(_root, 'Template', 'LO', 'Source', 'PDF')
        for name in ('ElectroweakFlux.f', 'ElectroweakFluxDriver.f'):
            shutil.copy(pjoin(pdf_dir, name), cls.tmpdir)
        # the include madevent writes at output (export_v4 create_ewa), with
        # the masses as numbers and gal(1) = e from a common block instead of
        # the model's coupl.inc
        with open(pjoin(_root, 'madgraph', 'iolibs', 'template_files',
                        'madevent_electroweakFlux.inc')) as fsock:
            inc = fsock.read() % {'MW': '%.17gd0' % MW, 'MZ': '%.17gd0' % MZ}
        inc = inc.replace("\tinclude '../vector.inc' ! defines VECSIZE_MEMMAX\n", '')
        inc = inc.replace(
            "\tinclude '../MODEL/coupl.inc' ! needs VECSIZE_MEMMAX (defined in vector.inc)\n",
            "\tdouble precision gal(2)\n\tcommon/eva_test_gal/gal\n")
        assert 'coupl.inc' not in inc
        with open(pjoin(cls.tmpdir, 'ElectroweakFlux.inc'), 'w') as fsock:
            fsock.write(inc)
        with open(pjoin(cls.tmpdir, 'eva_driver.f'), 'w') as fsock:
            fsock.write(cls.driver)
        out = subprocess.run(
            [cls.fc, '-O0', '-ffixed-line-length-132', '-o', 'eva_driver',
             'eva_driver.f', 'ElectroweakFlux.f', 'ElectroweakFluxDriver.f'],
            cwd=cls.tmpdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        if out.returncode:
            raise RuntimeError('compiling the EVA driver failed:\n' + out.stdout)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, 'tmpdir', None):
            shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def fortran(self, points):
        """eva_get_pdf_by_PID at each point (vpid, fpid, vpol, flpol, x, q,
        ebeam, ievo, order, xcut)."""
        lines = ['%.17g' % EE, str(len(points))]
        for vpid, fpid, vpol, flpol, x, q, ebeam, ievo, order, xcut in points:
            lines.append('%d %d %d %.17g %.17g %.17g %.17g %d %d %d' % (
                vpid, fpid, vpol, flpol, x, q, ebeam, ievo, order, xcut))
        out = subprocess.run([pjoin(self.tmpdir, 'eva_driver')],
                             input='\n'.join(lines) + '\n', cwd=self.tmpdir,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
        self.assertEqual(out.returncode, 0, out.stdout[-2000:])
        values = [float(line.split()[1]) for line in out.stdout.splitlines()
                  if line.startswith('RES ')]
        self.assertEqual(len(values), len(points))
        return values

    def test_densities_match_fortran(self):
        ms = self.ms
        ebeam = 5000.
        # x: small, around M_V / E_beam (the x cut), central, near 1;
        # q: below the floors (10 GeV < M_W), around them, high
        xs = [2e-4, 0.0159, 0.0185, 0.05, 0.3, 0.71, 0.999]
        qs = [0.05, 10., 85., 91.188, 500., 4000.]
        vectors = [24, -24, 23, 22]
        leptons = [11, -11, 13, -13]
        points, mine = [], []
        for vpid, lepton, flpol, order, ievo, xcut in itertools.product(
                vectors, leptons, (0., 0.5, 0.9, 1.), (0, 1, 2), (0, 1), (0, 1)):
            flux = ms.EvaFlux([lepton, lepton], [ebeam, ebeam], MW, MZ, EE ** 2,
                              [flpol, flpol], order, bool(ievo), bool(xcut))
            if not flux.has_density(vpid, 0):
                # madevent stops (stop 24) where mg7 has no density
                self.assertEqual(flux.density(vpid, 0, 1, 0.3, 100.), 0.)
                continue
            pols = (-1, 1) if vpid == 22 else (-1, 0, 1)
            for vpol, x, q in itertools.product(pols, xs, qs):
                # madevent's photon is pdg 7
                points.append((7 if vpid == 22 else vpid, lepton, vpol, flpol,
                               x, q, ebeam, ievo, order, xcut))
                mine.append(flux.density(vpid, 0, vpol, x, q))
        ref = self.fortran(points)
        self.assertGreater(len(points), 10000)
        worst = 0.
        for point, a, b in zip(points, mine, ref):
            scale = max(abs(b), 1e-300)
            if b == 0.:
                self.assertEqual(a, 0., point)
                continue
            rel = abs(a - b) / scale
            worst = max(worst, rel)
            self.assertLess(rel, 1e-12, '%s: %r vs %r' % (point, a, b))
        # the densities do vanish below the x cut and are nonzero elsewhere
        self.assertTrue(any(v == 0. for v in ref))
        self.assertTrue(any(v != 0. for v in ref))

    def test_charge_violating_w_has_no_density(self):
        ms = self.ms
        flux = ms.EvaFlux([-11, 13], [7500., 5000.], MW, MZ, EE ** 2)
        self.assertTrue(flux.has_density(24, 0))     # W+ from e+
        self.assertFalse(flux.has_density(-24, 0))   # W- from e+
        self.assertTrue(flux.has_density(-24, 1))    # W- from mu-
        self.assertFalse(flux.has_density(24, 1))    # W+ from mu-
        self.assertTrue(flux.has_density(23, 1))
        self.assertTrue(flux.has_density(22, 0))
        for hel in (-1, 0, 1, ms.EvaFlux.helicity_sum):
            self.assertEqual(flux.density(-24, 0, hel, 0.3, 100.), 0.)
        with self.assertRaises(Exception):
            flux.parameters(21, 0)


@unittest.skipIf(np is None, 'numpy required')
class TestEvaKernel(unittest.TestCase):
    """The compute-graph kernel against the host densities."""

    def test_parton_density_is_helicity_sum(self):
        ms = _madspace()
        if ms is None:
            self.skipTest('madspace build with EvaFlux required')
        flux = ms.EvaFlux([-13, 13], [5000., 5000.], MW, MZ, EE ** 2,
                          [0.9, 0.1], 2, True, True)
        pids = [24, -24, 23, 22]
        rng = np.random.default_rng(3)
        n = 500
        x = 10 ** rng.uniform(-4, 0, n) * (1 - 1e-6)
        q = 10 ** rng.uniform(0, 3.5, n)
        for beam in (0, 1):
            got = np.asarray(ms.PartonDensity(flux, pids, beam, False)(x, q))
            self.assertEqual(got.shape, (n, len(pids)))
            for j, pid in enumerate(pids):
                ref = np.array([
                    xi * flux.density(pid, beam, ms.EvaFlux.helicity_sum, xi, qi)
                    for xi, qi in zip(x, q)])
                np.testing.assert_allclose(got[:, j], ref, rtol=1e-13, atol=0)
            # a charge-violating W is zero, an allowed one is not
            wrong = 1 if beam == 0 else 0
            self.assertTrue(np.all(got[:, wrong] == 0.))
            self.assertTrue(np.any(got[:, 1 - wrong] > 0.))
            # one flavour per event
            index = rng.integers(0, len(pids), n).astype(np.int32)
            dyn = np.asarray(ms.PartonDensity(flux, pids, beam, True)(x, q, index))
            np.testing.assert_allclose(dyn, got[np.arange(n), index], rtol=1e-15)


class TestEvaRunCard(unittest.TestCase):
    """The [beam] EVA keys of the mg7 run card and the madevent translation."""

    @staticmethod
    def proc_def(*initials):
        class PC(dict):
            def __init__(self):
                super().__init__(ninitial=2, loop_induced=False, colored_pdgs=[])

        procs = []
        for initial in initials:
            legs = [{'state': False, 'id': i} for i in initial]
            legs += [{'state': True, 'id': 6}, {'state': True, 'id': -6}]
            procs.append({'legs': legs})
        return PC(), [procs]

    def test_default_card_of_a_w_w_process_has_eva_beams(self):
        """w+ w- > t t~ used to run silently with monochromatic W beams; the
        default is now madevent's: e+ e- at 2 x 15 TeV, pdlabel eva, fixed
        mu_F"""
        charac, proc_def = self.proc_def([24, -24])
        card = banner_mod.RunCardMG7()
        card.create_default_for_process(charac, '', proc_def)
        beam = card['beam']
        self.assertIs(beam['eva'], True)
        self.assertIs(beam['leptonic'], True)
        self.assertEqual((beam['lepton1'], beam['lepton2']), (-11, 11))
        self.assertEqual((beam['ebeam1'], beam['ebeam2']), (15000., 15000.))
        self.assertIs(beam['fixed_fact_scale'], True)
        card.check_validity()
        # the EVA settings are written out
        import io
        out = io.StringIO()
        card.write(out)
        self.assertIn('eva = true', out.getvalue())
        self.assertIn('eva_order = 0', out.getvalue())
        # madevent's rule: a W or Z on each beam, photons allowed next to them
        for initials, eva in ((([23, 23],), True), (([22, 24], [24, 22]), True),
                              (([22, 22],), False), (([12, -12],), False),
                              (([24, 11],), False)):
            charac, proc_def = self.proc_def(*initials)
            card = banner_mod.RunCardMG7()
            card.create_default_for_process(charac, '', proc_def)
            self.assertIs(card['beam']['eva'], eva, initials)

    def test_set_madevent_beam_settings(self):
        card = banner_mod.RunCardMG7()
        card.set_lo_beam_param('lpp1', '-4')
        card.set_lo_beam_param('lpp2', '4')
        self.assertEqual((card['beam']['lepton1'], card['beam']['lepton2']), (-13, 13))
        self.assertIs(card['beam']['leptonic'], True)
        self.assertIs(card['beam']['eva'], False)
        card.set_lo_beam_param('pdlabel', 'eva')
        self.assertIs(card['beam']['eva'], True)
        card.set_lo_beam_param('polbeam1', '-80')
        card.set_lo_beam_param('polbeam2', '30')
        self.assertEqual(card['beam']['polarization1'], -80.)
        self.assertEqual(card['beam']['polarization2'], 30.)
        card.set_lo_beam_param('evaorder', '2')
        card.set_lo_beam_param('ievo_eva', '1')
        card.set_lo_beam_param('eva_xcut', '0')
        self.assertEqual(card['beam']['eva_order'], 2)
        self.assertEqual(card['beam']['eva_evolution'], 'pt')
        self.assertIs(card['beam']['eva_xcut'], False)
        card.check_validity()
        for name, value in (('evaorder', '3'), ('ievo_eva', '2'),
                            ('polbeam1', '120'), ('eva_xcut', 'x')):
            with self.assertRaises(banner_mod.InvalidRunCard):
                card.set_lo_beam_param(name, value)
        # another label switches EVA off
        card.set_lo_beam_param('pdlabel', 'none')
        self.assertIs(card['beam']['eva'], False)
        # and a polarisation without EVA is refused
        with self.assertRaises(banner_mod.InvalidRunCard):
            card.check_validity()

    def test_invalid_eva_cards(self):
        card = banner_mod.RunCardMG7()
        card.set('beam.eva', True)
        card.set('beam.leptonic', False)
        with self.assertRaises(banner_mod.InvalidRunCard):
            card.check_validity()  # EVA needs lepton beams
        card.set('beam.leptonic', True)
        card.check_validity()
        card.set('beam.lepton_density', 'isronlyll')
        with self.assertRaises(banner_mod.InvalidRunCard):
            card.check_validity()
        card.set('beam.lepton_density', '')
        card.set('beam.dynamical_scale_choice', 'mlm')
        with self.assertRaises(banner_mod.InvalidRunCard):
            card.check_validity()

    def test_from_lo_card(self):
        lo = banner_mod.RunCardLO()
        for name, value in (('lpp1', -4), ('lpp2', 4), ('ebeam1', 5000.),
                            ('ebeam2', 5000.), ('pdlabel', 'eva'),
                            ('polbeam1', -80.), ('evaorder', 2),
                            ('ievo_eva', 1), ('eva_xcut', 0),
                            ('fixed_fac_scale', True)):
            lo.set(name, value, user=True)
        mg7, dropped = banner_mod.RunCardMG7.from_LO(lo, warn=False)
        beam = mg7['beam']
        self.assertIs(beam['eva'], True)
        self.assertIs(beam['leptonic'], True)
        self.assertEqual((beam['lepton1'], beam['lepton2']), (-13, 13))
        self.assertEqual(beam['polarization1'], -80.)
        self.assertEqual(beam['eva_order'], 2)
        self.assertEqual(beam['eva_evolution'], 'pt')
        self.assertIs(beam['eva_xcut'], False)
        self.assertFalse([d for d in dropped if 'eva' in d or 'polbeam' in d
                          or 'lpp' in d or 'pdlabel' in d], dropped)
        # EVA on one beam only has no mg7 equivalent: reported
        lo = banner_mod.RunCardLO()
        for name, value in (('lpp1', -3), ('lpp2', 3), ('pdlabel1', 'eva'),
                            ('pdlabel2', 'none')):
            lo.set(name, value, user=True)
        mg7, dropped = banner_mod.RunCardMG7.from_LO(lo, warn=False)
        self.assertIs(mg7['beam']['eva'], False)
        self.assertTrue(any('EVA' in d for d in dropped), dropped)


if __name__ == '__main__':
    unittest.main()
