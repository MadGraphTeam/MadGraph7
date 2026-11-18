"""Open input-spin coherences, their physical contraction, and persisted events."""

import io
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

from madgraph.interface.reweight_interface import ReweightInterface
from madgraph.various import fxfx_ewsudakov as fx
from madgraph.various import lhe_parser
from madgraph.various import process_spin_tensor as pst
from tests.unit_tests.ewsudakov.test_full_density import DensityEventFixture
from tests.unit_tests.ewsudakov.test_fxfx_decay_transport import ww_event
from tests.unit_tests.ewsudakov.test_upstream_compatibility import exec_dispatcher


def example():
    rng = np.random.default_rng(20261023)
    amplitude = rng.normal(size=(9, 4)) + 1j*rng.normal(size=(9, 4))
    delta = .03*(rng.normal(size=(9, 4, 5)) + 1j*rng.normal(size=(9, 4, 5)))
    vector = amplitude.reshape(36)
    correction = np.empty((36, 36, 5), complex)
    for v in range(5):
        mixed = np.outer(delta[:, :, v].reshape(36), vector.conj())
        correction[:, :, v] = mixed + mixed.conj().T
    tensor = dict(version=1, incoming_dim=4, outgoing_dim=9,
                  born=np.outer(vector, vector.conj()), correction=correction,
                  relative=2*correction, leg_indices=[1, 2, 3, 4], spin_dims=[2, 2, 3, 3])
    return amplitude, delta, tensor


def fxfx_tensor(relative=False):
    """The tensor the dispatcher returns for the FxFx event path: the three
    leading variants, and the relative arrays only if requested."""
    unused, unused2, tensor = example()
    tensor = dict(tensor, correction=tensor['correction'][:, :, :3],
                  relative=tensor['relative'][:, :, :3],
                  variants=['qed', 'weak', 'sr_off'])
    if not relative:
        del tensor['relative']
    return tensor


def assert_hermitian_copy(restored, original):
    """Stored upper triangles are exact; the rest is their conjugate."""
    upper = np.triu_indices(original.shape[0])
    np.testing.assert_array_equal(restored[upper], original[upper])
    np.testing.assert_allclose(restored, original, rtol=0, atol=1e-14 * abs(original).max())


@unittest.skipIf(np is None, "NumPy is required for process tensor checks")
class TestOpenTensorAlgebra(unittest.TestCase):
    def test_unpolarised_reduction_matches_sum_of_amplitudes(self):
        amplitude, delta, tensor = example()
        result = pst.contract_process_tensor(tensor)
        np.testing.assert_allclose(result['density_born'], amplitude @ amplitude.conj().T / 4)
        for v in range(5):
            mixed = delta[:, :, v] @ amplitude.conj().T
            np.testing.assert_allclose(result['density_correction'][:, :, v],
                                       (mixed+mixed.conj().T)/4, atol=1e-15)

    def test_complex_prepared_input_matches_coherent_amplitudes(self):
        amplitude, delta, tensor = example()
        vector = np.array([1, 1j, -.3j, .2], complex)
        vector /= np.linalg.norm(vector)
        result = pst.contract_process_tensor(tensor, np.outer(vector, vector.conj()))
        outgoing = amplitude @ vector
        np.testing.assert_allclose(result['density_born'], np.outer(outgoing, outgoing.conj()), atol=1e-14)
        for v in range(5):
            mixed = np.outer(delta[:, :, v] @ vector, outgoing.conj())
            np.testing.assert_allclose(result['density_correction'][:, :, v],
                                       mixed+mixed.conj().T, atol=1e-14)
        # Removing a!=a' would destroy this information, although the
        # unpolarised reduction would still pass.
        diagonal_input = np.diag(abs(vector)**2)
        self.assertGreater(np.linalg.norm(result['density_born'] -
                           pst.contract_process_tensor(tensor, diagonal_input)['density_born']), 1.)

    def test_fixed_helicity_selects_a_full_outgoing_block(self):
        amplitude, unused, tensor = example()
        for a in range(4):
            rho = np.zeros((4, 4)); rho[a, a] = 1
            np.testing.assert_allclose(pst.contract_process_tensor(tensor, rho)['density_born'],
                                       np.outer(amplitude[:, a], amplitude[:, a].conj()))

    def test_unphysical_input_and_malformed_tensor_rejected(self):
        unused, unused2, tensor = example()
        for rho in [np.eye(4), np.diag([1.1, -.1, 0, 0]), np.ones((3, 3)),
                    np.eye(4)/4 + 1j*np.eye(4), np.full((4, 4), np.nan)]:
            with self.assertRaises(ValueError):
                pst.contract_process_tensor(tensor, rho)
        with self.assertRaises(ValueError):
            pst.contract_process_tensor(dict(tensor, born=np.zeros((9, 9))))


@unittest.skipIf(np is None, "NumPy is required for process tensor checks")
class TestOpenTensorPersistence(unittest.TestCase):
    def test_npz_roundtrip_is_lossless_without_object_arrays(self):
        unused, unused2, tensor = example()
        stream = io.BytesIO()
        pst.save_process_tensor(stream, tensor)
        stream.seek(0)
        restored = pst.load_process_tensor(stream)
        for name in ('born', 'correction', 'relative'):
            assert_hermitian_copy(restored[name], tensor[name])
        self.assertEqual(restored['leg_indices'], tensor['leg_indices'])
        self.assertNotIn('packing', restored)

    def test_records_keep_only_the_upper_triangles(self):
        unused, unused2, tensor = example()
        stream = io.BytesIO()
        pst.save_process_tensor(stream, tensor)
        stream.seek(0)
        with np.load(stream) as data:
            self.assertEqual(data['born'].shape, (36 * 37 // 2,))
            self.assertEqual(data['correction'].shape, (36 * 37 // 2, 5))

    def test_tensor_with_three_variants(self):
        tensor = fxfx_tensor(relative=True)
        stream = io.BytesIO()
        pst.save_process_tensor(stream, tensor)
        stream.seek(0)
        restored = pst.load_process_tensor(stream)
        self.assertEqual(restored['variants'], ['qed', 'weak', 'sr_off'])
        self.assertEqual(pst.contract_process_tensor(restored)['density_correction'].shape,
                         (9, 9, 3))

    def test_relative_array_is_optional(self):
        unused, unused2, tensor = example()
        del tensor['relative']
        stream = io.BytesIO()
        pst.save_process_tensor(stream, tensor)
        stream.seek(0)
        restored = pst.load_process_tensor(stream)
        self.assertNotIn('relative', restored)
        np.testing.assert_array_equal(restored['correction'], tensor['correction'])
        self.assertEqual(set(pst.contract_process_tensor(restored)),
                         {'density_born', 'density_correction'})

    def test_lhe_roundtrip_retains_all_xi_and_replaces_repeated_xi(self):
        unused, unused2, tensor = example()
        event = ww_event([])
        event.tag = '<test_tag>preserve me</test_tag>'
        pst.set_event_process_tensor(event, tensor, 0, 1.)
        pst.set_event_process_tensor(event, None, 1, 1.5)
        pst.set_event_process_tensor(event, tensor, 0, 1., prescription='amplitude')
        event = lhe_parser.Event(str(event))
        records = pst.read_event_process_tensors(event)
        self.assertEqual(len(records), 2)
        self.assertIn('<test_tag>preserve me</test_tag>', event.tag)
        self.assertEqual(records[0]['status'], 'available')
        self.assertEqual(records[1]['status'], 'not_evaluated')
        self.assertEqual(records[0]['prescription'], 'amplitude')
        assert_hermitian_copy(records[0]['correction'], tensor['correction'])
        pst.set_event_process_tensor(event, None, 0, 1.)
        self.assertEqual(pst.read_event_process_tensors(event)[0]['status'], 'not_evaluated')


@unittest.skipIf(np is None, "NumPy is required for the generated Sudakov dispatcher")
class TestOpenTensorDispatcher(unittest.TestCase):
    def setUp(self):
        self.ns = exec_dispatcher(self, '/tmp/tensor/bin/internal/ewsud_pydispatcher.py')
        self.amplitude, unused, self.tensor = example()
        self.kernel = mock.Mock(return_value=(self.tensor['born']/4,
            self.tensor['correction']/4, self.tensor['relative']/4, 0, 0))
        self.key = ((2, -2), (-24, 24))
        self.ns['pdg2ewsud_dict'][self.key] = SimpleNamespace(density_interference_py=self.kernel)
        self.p = [[500, 0, 0, 500], [500, 0, 0, -500],
                  [500, 400, 0, 200], [500, -400, 0, -200]]

    def test_open_indices_and_normalisation(self):
        result = self.ns['density_sudakov'](self.key, self.p, 1., [3, 4], [3, 3], keep_initial=True)
        tensor = result['process_tensor']
        self.assertEqual(tensor['born'].shape, (36, 36))
        np.testing.assert_array_equal(tensor['born'], self.tensor['born'])
        np.testing.assert_allclose(result['density_born'], self.amplitude @ self.amplitude.conj().T / 4)
        self.assertEqual(tensor['helicities'][0], [-1, -1, -1, -1])
        self.assertEqual(tensor['helicities'][1], [1, -1, -1, -1])
        self.assertEqual(tensor['helicities'][4], [-1, -1, 0, -1])
        np.testing.assert_array_equal(self.kernel.call_args.args[3][:4], [1, 2, 3, 4])
        np.testing.assert_array_equal(self.kernel.call_args.args[4][:4], [2, 2, 3, 3])
        # By default the tensor keeps all five variants and the comparison.
        self.assertEqual(self.kernel.call_args.args[5:], (5, 1))
        self.assertEqual(result['total_dim'], 9)
        self.assertEqual(result['nres'], 2)
        for key, expected in pst.contract_process_tensor(tensor).items():
            np.testing.assert_allclose(result[key], expected)

    def test_tensor_holds_the_requested_variants_and_arrays(self):
        result = self.ns['density_sudakov'](self.key, self.p, 1., [3, 4], [3, 3],
                                            keep_initial=True, nvariants=3,
                                            goldstone_relative=False)
        tensor = result['process_tensor']
        self.assertEqual(self.kernel.call_args.args[5:], (3, 0))
        self.assertEqual(tensor['variants'], ['qed', 'weak', 'sr_off'])
        self.assertNotIn('relative', tensor)
        self.assertIsNone(result['density_relative'])
        self.assertEqual(tensor['correction'].shape, (36, 36, 3))

    def test_unsupported_open_spaces_do_not_fall_back(self):
        for key, pos, dims, error in [(((12, -12), (-24, 24)), [3, 4], [3, 3], NotImplementedError),
                (self.key, [3, 4, 5], [3, 3, 3], NotImplementedError),
                (self.key, [1, 4], [3, 3], ValueError),
                (self.key, [3, 3], [3, 3], ValueError)]:
            with self.assertRaises(error):
                self.ns['density_process_tensor'](key, self.p, 1., pos, dims)
        self.kernel.assert_not_called()


@unittest.skipIf(np is None, "NumPy is required for process tensor checks")
class TestOpenTensorEventPath(DensityEventFixture, unittest.TestCase):
    def test_option_retains_tensor_with_same_weights(self):
        previous = self.driver._compute_density_ewsudakov_reweight(self.event, self.module)
        tensor = fxfx_tensor()
        self.result['process_tensor'] = tensor
        self.driver.sudakov_density_keep_initial = True
        self.module.PROCESS_TENSOR_VERSION = 1
        result = self.driver._compute_density_ewsudakov_reweight(self.event, self.module)
        self.assertEqual(previous, result)
        # The amplitude prescription needs neither the floor-off variants nor
        # the Goldstone-relative comparison.
        self.assertEqual(self.module.density_sudakov.call_args.kwargs,
                         {'keep_initial': True, 'nvariants': 3, 'goldstone_relative': False})
        records = pst.read_event_process_tensors(lhe_parser.Event(str(self.event)))
        assert_hermitian_copy(records[0]['born'], tensor['born'])
        self.assertNotIn('relative', records[0])
        self.assertEqual(records[0]['variant_aliases'], fx.FXFX_VARIANT_ALIASES)
        self.assertIn('process_tensor="initial_spin_v1"', self.driver._sudakov_weight_header())

    def test_goldstone_relative_tensor_keeps_its_relative_array(self):
        tensor = fxfx_tensor(relative=True)
        self.result['process_tensor'] = tensor
        self.driver.sudakov_density_prescription = 'goldstone_relative'
        self.driver.sudakov_density_keep_initial = True
        self.module.PROCESS_TENSOR_VERSION = 1
        self.driver._compute_density_ewsudakov_reweight(self.event, self.module)
        self.assertTrue(self.module.density_sudakov.call_args.kwargs['goldstone_relative'])
        record = pst.read_event_process_tensors(self.event)[0]
        self.assertEqual(record['prescription'], 'goldstone_relative')
        assert_hermitian_copy(record['relative'], tensor['relative'])

    def test_each_xi_writes_its_record_once(self):
        self.result['process_tensor'] = fxfx_tensor()
        self.driver.sudakov_density_keep_initial = True
        self.module.PROCESS_TENSOR_VERSION = 1
        xis = [1.5, 3.]
        self.driver._configure_sudakov_weight_ids(self.event.parse_reweight(), xis)
        with mock.patch.object(fx, 'set_event_process_tensor',
                               wraps=pst.set_event_process_tensor) as write:
            for k, xi in enumerate(xis):
                self.driver._current_xi_idx = k
                self.driver.mW2_cluster_scale = xi
                self.driver._compute_density_ewsudakov_reweight(self.event, self.module)
        self.assertEqual([call.args[2] for call in write.call_args_list], [0, 1])
        self.assertEqual([r['status'] for r in pst.read_event_process_tensors(self.event)],
                         ['available', 'available'])

    def test_recomputed_weights_drop_tensors_of_an_earlier_run(self):
        unused, unused2, tensor = example()
        pst.set_event_process_tensor(self.event, tensor, 0, 1.5)
        self.driver._current_xi_idx = 0
        self.driver._compute_density_ewsudakov_reweight(self.event, self.module)
        self.assertEqual(pst.read_event_process_tensors(self.event), [])

    def test_old_dispatcher_rejected_if_tensor_requested(self):
        self.driver.sudakov_density_keep_initial = True
        with self.assertRaisesRegex(NotImplementedError, 'dispatcher'):
            self.driver._compute_density_ewsudakov_reweight(self.event, self.module)
        self.module.ewsudakov.assert_not_called()

    def test_new_scan_removes_tensors_from_previous_xi_scan(self):
        tensor = fxfx_tensor()
        pst.set_event_process_tensor(self.event, tensor, 2, 2.)
        self.result['process_tensor'] = tensor
        self.driver.sudakov_density_keep_initial = True
        self.driver._current_xi_idx = 0
        self.module.PROCESS_TENSOR_VERSION = 1
        self.driver._compute_density_ewsudakov_reweight(self.event, self.module)
        self.assertEqual([r['xi_index'] for r in pst.read_event_process_tensors(self.event)], [0])

    def test_scalar_weight_without_decay_matrix_keeps_no_tensor(self):
        self.result['process_tensor'] = fxfx_tensor()
        self.driver.sudakov_density_keep_initial = True
        self.module.PROCESS_TENSOR_VERSION = 1
        self.module.ewsudakov.return_value = [2., -.2, -.4, -.3, -.5, -.6]
        self.driver._get_decay_density_matrix.return_value = None
        weights = self.driver._compute_density_ewsudakov_reweight(self.event, self.module)
        self.module.ewsudakov.assert_called_once()
        central = [key for key, (source, k, xi, variant)
                   in self.driver._sudakov_weight_metadata.items() if variant == 'central']
        self.assertTrue(central)
        for key in central:
            self.assertAlmostEqual(weights[key], 2*.8)
        records = pst.read_event_process_tensors(lhe_parser.Event(str(self.event)))
        self.assertEqual([r['status'] for r in records], ['not_evaluated'])

    def test_export_failure_cannot_silently_use_scalar(self):
        self.driver.sudakov_density_keep_initial = True
        self.module.PROCESS_TENSOR_VERSION = 1
        self.module.density_sudakov.side_effect = ValueError('invalid tensor')
        with self.assertRaisesRegex(ValueError, 'invalid tensor'):
            self.driver._compute_density_ewsudakov_reweight(self.event, self.module)
        self.module.ewsudakov.assert_not_called()

    def test_card_option(self):
        interface = SimpleNamespace(split_arg=lambda line: line.split(), InvalidCmd=ValueError)
        for text, value in [('True', True), ('False', False)]:
            ReweightInterface.do_change(interface, 'sudakov_density_keep_initial ' + text)
            self.assertEqual(interface.sudakov_density_keep_initial, value)
        with self.assertRaisesRegex(ValueError, 'True or False'):
            ReweightInterface.do_change(interface, 'sudakov_density_keep_initial')

    def test_lhe_option_rejects_modes_without_tensor_export(self):
        self.driver.sudakov_density_keep_initial = True
        self.driver.banner = SimpleNamespace(get=lambda *args: 0)
        with self.assertRaisesRegex(NotImplementedError, 'ickkw=3'):
            self.driver._prepare_sudakov_weight_ids()

    def test_lhe_option_rejects_per_weight_event_files(self):
        self.driver.sudakov_density_keep_initial = True
        self.driver.banner = SimpleNamespace(get=lambda *args: 3)
        for output_type in ('2.0', 'unweight'):
            self.driver.output_type = output_type
            with self.subTest(output_type=output_type), \
                    self.assertRaisesRegex(NotImplementedError, 'default reweighting output'):
                self.driver._prepare_sudakov_weight_ids()


if __name__ == '__main__':
    unittest.main()
