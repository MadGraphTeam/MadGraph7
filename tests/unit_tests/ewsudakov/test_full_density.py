"""Full linear spin interference, including spectator and colour correlations."""

from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

from madgraph import MadGraph5Error
from madgraph.iolibs import export_fks
from madgraph.various import fxfx_ewsudakov as fx
from tests.unit_tests.ewsudakov.test_sudakov_factor_limits import FactorDriver, density_result
from tests.unit_tests.ewsudakov.test_fxfx_decay_transport import ww_event
from tests.unit_tests.ewsudakov.test_upstream_compatibility import exec_dispatcher


@unittest.skipIf(np is None, "NumPy is required for full density checks")
class TestFullInterference(unittest.TestCase):
    def weight(self, amplitude, correction, decay, metric):
        born = np.einsum('sha,ab,skb->hk', amplitude, metric, amplitude.conj())
        mixed = np.einsum('sha,ab,skb->hk', correction, metric, amplitude.conj())
        db = mixed + mixed.T.conj()
        b = fx.DensityMatrix.from_full_matrix(born, [23])
        c = fx.DensityMatrix.from_full_matrix(np.outer(decay, decay.conj()), [23])
        return fx.compute_density_interference_weight(b, c, db)

    def test_matches_coherent_amplitudes_for_each_spectator_and_colour(self):
        rng = np.random.default_rng(20261022)
        amplitude = rng.normal(size=(4, 3, 2)) + 1j*rng.normal(size=(4, 3, 2))
        correction = .05*(rng.normal(size=amplitude.shape) + 1j*rng.normal(size=amplitude.shape))
        decay = rng.normal(size=3) + 1j*rng.normal(size=3)
        metric = np.array([[3, -1], [-1, 3]])
        a = np.einsum('sha,h->sa', amplitude, decay)
        da = np.einsum('sha,h->sa', correction, decay)
        rate = np.einsum('sa,ab,sb->', a, metric, a.conj()).real
        change = 2*np.einsum('sa,ab,sb->', da, metric, a.conj()).real
        expected = 1 + change/rate
        self.assertAlmostEqual(self.weight(amplitude, correction, decay, metric), expected, places=14)
        # The weight is linear in delta M, with no accidental O(alpha^2).
        self.assertAlmostEqual(self.weight(amplitude, 2*correction, decay, metric), 2*expected-1, places=14)

    def test_colour_orthogonal_correction_survives(self):
        amplitude = np.array([[[1,0], [0,1], [0,0]]], dtype=complex)
        correction = np.array([[[0,.2j], [.1,0], [0,0]]], dtype=complex)
        decay = np.array([1, 1j, 0])
        # Every same-helicity Born/correction product is zero. Their
        # off-diagonal interference is nonzero and must not be projected out.
        self.assertAlmostEqual(self.weight(amplitude, correction, decay, np.eye(2)), 1.2)

    def test_zero_born_helicity_can_interfere_with_another_helicity(self):
        amplitude = np.array([[[1], [0], [0]]], dtype=complex)
        correction = np.array([[[0], [.1j], [0]]], dtype=complex)
        decay = np.array([1, -1j, 0])
        self.assertAlmostEqual(self.weight(amplitude, correction, decay, np.eye(1)), 1.2)

    def test_spectators_are_incoherent_with_each_other(self):
        amplitude = np.array([[[1], [1], [0]], [[1], [-1], [0]]], complex)
        correction = np.array([[[.2], [0], [0]], [[0], [.1], [0]]], complex)
        decay = np.array([1, 1, 0])
        self.assertAlmostEqual(self.weight(amplitude, correction, decay, np.eye(1)), 1.2)

    def test_weight_is_invariant_under_small_amplitude_normalisation(self):
        amplitude = np.array([[[1], [2j], [1]]], complex)
        correction = .1*amplitude
        decay = np.array([1, 1, 1])
        for scale in (1., 1e-40, 1e-120):
            self.assertAlmostEqual(self.weight(scale*amplitude, scale*correction, decay, np.eye(1)), 1.2)

    def test_contraction_cancelling_within_rounding_gives_unit_weight(self):
        # sum_h M_h D_h = 2 eps, so <B,C> = 4 eps^2 while its terms are of
        # size 1, and <dB,C> = 0.4 eps (1 + eps). A small contraction keeps its
        # large factor, at any normalisation; one at rounding level gives 1.
        amplitude = np.array([[[1], [1], [0]]], complex)
        correction = np.array([[[.1], [0], [0]]], complex)
        fx.fks_reset_accounting()
        self.addCleanup(fx.fks_reset_accounting)
        with mock.patch.object(fx, 'CURRENT_EVENT_ID', 1):
            for scale in (1., 1e-100):
                eps = 1e-3
                weight = self.weight(scale*amplitude, scale*correction,
                                     np.array([1 + eps, -1 + eps, 0]), np.eye(1))
                self.assertAlmostEqual(weight / (1 + .1*(1 + eps)/eps), 1., places=6)
            self.assertNotIn('events_density_vanishing_contraction', fx.FKS_QUALITY_COUNTS)
            eps = 1e-6
            self.assertEqual(self.weight(amplitude, correction,
                                         np.array([1 + eps, -1 + eps, 0]), np.eye(1)), 1.)
        self.assertEqual(fx.FKS_QUALITY_COUNTS['events_density_vanishing_contraction'], 1)


@unittest.skipIf(np is None, "NumPy is required for the generated Sudakov dispatcher")
class TestFullDispatcher(unittest.TestCase):
    def setUp(self):
        self.ns = exec_dispatcher(self, '/tmp/process/bin/internal/ewsud_pydispatcher.py')
        self.key = ((2,-2), (23,21,21))
        self.kernel = mock.Mock(return_value=(np.eye(36,dtype=complex),
            np.zeros((36,36,5),complex),np.zeros((36,36,5),complex),0,0))
        self.ns['pdg2ewsud_dict'][self.key] = SimpleNamespace(density_interference_py=self.kernel)
        self.p = [[100,0,0,100], [100,0,0,-100], [100,0,60,0], [50,50,-30,0], [50,-50,-30,0]]

    def test_interface_has_full_matrices_and_all_variants(self):
        result = self.ns['density_sudakov'](self.key,self.p,1.,[3],[3])
        self.assertEqual(result['version'],fx.DENSITY_KERNEL_VERSION)
        self.assertEqual(self.kernel.call_args.args[5:],(5,1))
        self.assertEqual(result['density_born'].shape,(3,3))
        self.assertEqual(result['density_correction'].shape,(3,3,5))
        self.assertNotIn('density_delta', result)

    def test_kernel_evaluates_only_the_requested_leading_variants(self):
        result = self.ns['density_sudakov'](self.key,self.p,1.,[3],[3],nvariants=3)
        self.assertEqual(self.kernel.call_args.args[5:],(3,1))
        self.assertEqual(result['density_correction'].shape,(3,3,3))
        self.assertEqual(result['density_relative'].shape,(3,3,3))

    def test_relative_comparison_only_on_request(self):
        result = self.ns['density_sudakov'](self.key,self.p,1.,[3],[3],
                                            goldstone_relative=False)
        self.assertEqual(self.kernel.call_args.args[5:],(5,0))
        self.assertIsNone(result['density_relative'])

    def test_invalid_parameters_and_polarised_legs_stop_the_run(self):
        for status,message in ((3,'YMT = MT'),(4,'polarised')):
            values=list(self.kernel.return_value)
            values[-1]=status
            self.kernel.return_value=tuple(values)
            with self.subTest(status=status), self.assertRaisesRegex(NotImplementedError,message):
                self.ns['density_sudakov'](self.key,self.p,1.,[3],[3])

    def test_relative_overflow_leaves_the_physical_matrices_usable(self):
        relative=np.full((36,36,5),np.inf,complex)
        self.kernel.return_value=(np.eye(36,dtype=complex),np.zeros((36,36,5),complex),
                                  relative,0,0)
        result=self.ns['density_sudakov'](self.key,self.p,1.,[3],[3])
        self.assertTrue(np.isfinite(result['density_correction']).all())

    def test_invalid_dimensions_never_reach_fortran(self):
        for positions,dims in (([3],[4]),([3,3],[3,3]),([3,4,5,2],[3,3,3,3]),([0],[3]),([3],[])):
            with self.assertRaises(ValueError):
                self.ns['density_sudakov'](self.key,self.p,1.,positions,dims)
        for nvariants in (0,6):
            with self.assertRaises(ValueError):
                self.ns['density_sudakov'](self.key,self.p,1.,[3],[3],nvariants=nvariants)
        self.kernel.assert_not_called()

    def test_old_compiled_library_requires_regeneration(self):
        self.ns['pdg2ewsud_dict'][self.key] = SimpleNamespace(density_sudakov_py=mock.Mock())
        with self.assertRaisesRegex(NotImplementedError,'Regenerate'):
            self.ns['density_sudakov'](self.key,self.p,1.,[3],[3])

    def test_failed_kernel_is_not_returned_as_a_physical_matrix(self):
        values=list(self.kernel.return_value)
        values[-1]=2
        self.kernel.return_value=tuple(values)
        with self.assertRaisesRegex(RuntimeError,'status 2'):
            self.ns['density_sudakov'](self.key,self.p,1.,[3],[3])



class TestSudakovExportBeams(unittest.TestCase):
    @staticmethod
    def matrix_element(beams):
        masses={24:'MDL_MW',23:'MDL_MZ',6:'MDL_MT',2:'ZERO',21:'ZERO',22:'ZERO'}
        model=SimpleNamespace(get_particle=lambda pdg: {'mass':masses[abs(pdg)]})

        class Process(dict):
            def base_string(self):
                return '%d %d > t t~' % beams

        legs=[{'id':pdg,'state':False} for pdg in beams]+[
            {'id':6,'state':True},{'id':-6,'state':True}]
        return SimpleNamespace(born_me={'processes':[Process(legs=legs,model=model)]})

    def test_massive_beams_stop_every_sudakov_export(self):
        # Neither Sudakov kernel supports them, in any output.
        exporter=export_fks.ProcessExporterFortranFKS.__new__(
            export_fks.ProcessExporterFortranFKS)
        for beams in ((24,-24),(23,2),(6,-6)):
            with self.subTest(beams=beams), self.assertRaisesRegex(MadGraph5Error,
                                                                   'massless beams'):
                exporter.check_sudakov_beams(self.matrix_element(beams))
        for beams in ((2,-2),(21,21),(22,22)):
            exporter.check_sudakov_beams(self.matrix_element(beams))

    def test_only_the_reweighting_library_has_the_density_kernel(self):
        self.assertFalse(export_fks.ProcessExporterFortranFKS.sudakov_density_kernel)
        self.assertTrue(export_fks.ProcessExporterEWSudakovSA.sudakov_density_kernel)


class DensityEventFixture:
    """Clustered WW event, driver and mocked kernel of the density event path.

    Not a TestCase, so that test classes can share it without inheriting tests.
    """
    def setUp(self):
        self.driver=FactorDriver()
        self.driver._sudakov_ickkw=3
        self.event=ww_event([])
        self.event.wgt=2.
        self.event.parse_reweight()
        self.event.reweight_data={'1001':2.}
        self.event.reweight_order=['1001']
        _,tag=self.driver._cluster_fxfx_event(self.event,record_tag=True)
        self.result=density_result(9,.2,.2,.2,relative=(.1,.1,.1))
        self.module=SimpleNamespace(original_pdg_list_dict={tag:tag},
                                    density_sudakov=mock.Mock(return_value=self.result),
                                    ewsudakov=mock.Mock())
        self.driver._get_decay_density_matrix=mock.Mock(return_value=
            fx.DensityMatrix.from_full_matrix(np.eye(9),[24,-24]))


@unittest.skipIf(np is None, "NumPy is required for density prescription checks")
class TestDensityPrescriptions(DensityEventFixture, unittest.TestCase):
    def test_explicit_comparison_selects_its_own_matrix_and_metadata(self):
        for prescription,factor in [('amplitude',1.2),('goldstone_relative',1.1)]:
            self.driver.sudakov_density_prescription=prescription
            weights=self.driver._compute_density_ewsudakov_reweight(self.event,self.module)
            for key in self.driver._sudakov_weight_metadata:
                self.assertAlmostEqual(weights[key],2*factor)
            self.assertIn('density_prescription="'+prescription+'"',
                          self.driver._sudakov_weight_header())
        self.module.ewsudakov.assert_not_called()

    def test_zero_goldstone_projection_cannot_silently_use_scalar(self):
        self.driver.sudakov_density_prescription='goldstone_relative'
        self.result['zero_gold']=1
        with self.assertRaisesRegex(NotImplementedError,'zero Goldstone Born'):
            self.driver._compute_density_ewsudakov_reweight(self.event,self.module)
        self.module.ewsudakov.assert_not_called()

    def test_stale_density_result_cannot_silently_use_scalar(self):
        self.result['version']=1
        with self.assertRaisesRegex(NotImplementedError,'Regenerate rw_me'):
            self.driver._compute_density_ewsudakov_reweight(self.event,self.module)
        self.module.ewsudakov.assert_not_called()
