"""Required density setup errors and observable per-event scalar fallbacks."""

import importlib
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from madgraph.interface import reweight_interface
from madgraph.interface.reweight_interface import ReweightInterface
from madgraph.various import fxfx_ewsudakov as fx
from madgraph.various import misc
from tests.unit_tests.ewsudakov import test_sudakov_factor_limits as limits


class TestDensityLibraries(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.interface = ReweightInterface.__new__(ReweightInterface)
        self.interface.exitted = True
        self.interface.inc_sudakov = True
        self.interface.me_dir = str(self.root)
        self.interface.rwgt_dir = None
        self.interface.keep_ordering = True
        self.interface.f2py_suffix = {}
        names = mock.patch.object(reweight_interface, 'loaded_f2py_names', set())
        names.start()
        self.addCleanup(names.stop)
        self.interface.banner = SimpleNamespace(get=lambda *args: 3)
        self.interface.fxfx_clustered_decay_processes = {(24, (-13, 14))}
        self.interface.density_prod_dirname = 'rw_me_density_prod'
        self.interface.density_decay_dirname = 'rw_me_density_decay'
        self.modules = {
            'rw_me_density_prod': self.module([2, -2, 24, -24]),
            'rw_me_density_decay': self.module([24, -13, 14]),
        }

    @staticmethod
    def module(pdgs):
        return SimpleNamespace(get_pdg_order=lambda: ([pdgs], [1]),
                               get_prefix=lambda: [b'p0_'])

    def make_directories(self):
        for name in self.modules:
            (self.root / name / 'SubProcesses').mkdir(parents=True)

    def import_module(self, name):
        if name.endswith('all_matrix3py'):
            raise ImportError('optional loop module absent')
        return self.modules[name.split('.')[0]]

    def load(self, importer=None):
        with mock.patch.object(importlib, 'import_module', side_effect=importer or self.import_module), \
                mock.patch.object(importlib, 'reload', side_effect=lambda module: module):
            self.interface.load_density_modules()

    def test_required_missing_directory_stops_setup_with_path(self):
        with self.assertRaisesRegex(fx.DensityLibraryError,
                                    str(self.root / 'rw_me_density_prod')):
            self.load()

    def test_required_tree_import_preserves_original_error(self):
        self.make_directories()
        error = OSError('undefined symbol: density_model')
        with self.assertRaises(fx.DensityLibraryError) as caught:
            self.load(mock.Mock(side_effect=error))
        self.assertIs(caught.exception.__cause__, error)
        self.assertIn('all_matrix2py', str(caught.exception))
        self.assertIn(str(self.root), str(caught.exception))
        self.assertIn('OSError: undefined symbol: density_model', str(caught.exception))

    def test_optional_loop_module_may_be_absent(self):
        self.make_directories()
        self.load()
        self.assertEqual(set(self.interface.density_prod_f2pylib), {('rw_me_density_prod', 2)})
        self.assertIn(((2, -2), (24, -24)), self.interface.density_prod_id_to_path)
        self.assertIn(((24,), (-13, 14)), self.interface.density_decay_id_to_path)

    def test_density_rebuild_uses_new_module_and_library_names(self):
        self.make_directories()
        reweight_interface.loaded_f2py_names.update(self.modules)
        with mock.patch.object(self.interface, 'compile_SubProcess_dir') as compile_dir:
            self.interface._compile_density_subprocesses()
        self.assertEqual(compile_dir.call_args_list, [
            mock.call(str(self.root / name / 'SubProcesses'), '_r2') for name in self.modules])
        imports = []

        def importer(name):
            imports.append(name)
            if not name.endswith('all_matrix2py_r2'):
                raise ImportError('optional loop module absent')
            return self.modules[name.split('.')[0]]

        self.load(importer)
        for name, module in self.modules.items():
            self.assertIn(name + '.SubProcesses.all_matrix2py_r2', imports)
            self.assertIn(name + '_r2', reweight_interface.loaded_f2py_names)
        self.assertIs(self.interface.density_prod_f2pylib['rw_me_density_prod', 2],
                      self.modules['rw_me_density_prod'])

    def test_mappings_come_from_tree_even_if_loop_module_loads(self):
        self.make_directories()
        loop = self.module([21, 21, 23, 23])
        self.load(lambda name: loop if name.endswith('all_matrix3py')
                  else self.import_module(name))
        self.assertIn(((2, -2), (24, -24)), self.interface.density_prod_id_to_path)
        self.assertNotIn(((21, 21), (23, 23)), self.interface.density_prod_id_to_path)

    def test_required_tree_cannot_be_replaced_by_loop_module(self):
        self.make_directories()

        def importer(name):
            if name.endswith('all_matrix2py'):
                raise ImportError('tree module absent')
            return self.module([21, 21, 23, 23])

        with self.assertRaisesRegex(fx.DensityLibraryError, 'tree module absent'):
            self.load(importer)

    def test_missing_decay_library_stops_setup(self):
        (self.root / 'rw_me_density_prod' / 'SubProcesses').mkdir(parents=True)
        with self.assertRaisesRegex(fx.DensityLibraryError, 'rw_me_density_decay'):
            self.load()

    def test_scalar_only_samples_need_no_density_libraries(self):
        for decays, ickkw in ((set(), 3), ({(24, (-13, 14))}, 0)):
            with self.subTest(ickkw=ickkw, decays=decays):
                self.interface.fxfx_clustered_decay_processes = decays
                self.interface.banner = SimpleNamespace(get=lambda *args: ickkw)
                self.load()
                self.assertEqual(self.interface.density_prod_id_to_path, {})
                self.assertEqual(self.interface.density_decay_id_to_path, {})

    def test_optional_import_failure_is_allowed(self):
        self.make_directories()
        self.interface.fxfx_clustered_decay_processes = set()
        self.load(mock.Mock(side_effect=ImportError('unused library absent')))
        self.assertEqual(self.interface.density_prod_f2pylib, {})
        self.assertEqual(self.interface.density_prod_id_to_path, {})

    def test_invalid_required_metadata_retains_cause(self):
        self.make_directories()
        error = ValueError('corrupt process table')
        for get_order, message in ((mock.Mock(side_effect=error), 'corrupt process table'),
                                   (lambda: ([], []), 'No density subprocesses')):
            with self.subTest(message=message):
                self.modules['rw_me_density_prod'].get_pdg_order = get_order
                with self.assertRaisesRegex(fx.DensityLibraryError, message) as caught:
                    self.load()
                self.assertIsInstance(caught.exception.__cause__, ValueError)


@unittest.skipIf(limits.np is None, 'NumPy is required for density evaluation')
class TestDensityFallback(unittest.TestCase):
    def setUp(self):
        self.fixture = limits.TestSudakovFactorLimits()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        event_id = mock.patch.object(fx, 'CURRENT_EVENT_ID', 1)
        event_id.start()
        self.addCleanup(event_id.stop)
        self.event = self.fixture.event
        self.driver = limits.FactorDriver()
        self.module = self.fixture.scalar_module(self.driver, [1.2, 1.1, 1.05])
        self.module.density_sudakov = mock.Mock(return_value={
            'density_delta': limits.np.full(9, .1),
            'density_delta_var': limits.np.full(9, .05),
            'born_diag': limits.np.ones(9), 'total_dim': 9, 'nres': 2})
        self.driver.density_prod_id_to_path = {'process': True}
        self.driver.density_prod_f2pylib = {'module': True}
        self.driver.density_decay_id_to_path = {'decay': True}
        self.driver.density_decay_f2pylib = {'module': True}

    @staticmethod
    def density_matrix(event, pdgs, dims):
        return limits.np.eye(math.prod(dims))

    def evaluate(self):
        return self.driver._compute_density_ewsudakov_reweight(self.event, self.module)

    def assert_fallback(self, stage, message):
        with self.assertLogs(fx.reweight_logger, level='WARNING') as logs:
            result = self.evaluate()
        self.fixture.assert_factors(self.driver, result, [1.2, 1.1, 1.05])
        self.assertIn(stage, '\n'.join(logs.output))
        self.assertIn(message, '\n'.join(logs.output))
        self.assertIn('event 1, xi=1.5', '\n'.join(logs.output))
        self.assertEqual(fx.FKS_QUALITY_COUNTS['events_density_scalar_fallback'], 1)
        self.assertEqual(fx.FKS_QUALITY_COUNTS['events_density_fallback_' + stage], 1)

    def test_production_failure_counted_once_per_event_across_xis(self):
        with mock.patch.object(fx.DensityHelper, 'compute_density_matrix',
                               side_effect=ArithmeticError('production pole failed')):
            for event_id in (0, 1, 2):
                fx.CURRENT_EVENT_ID = event_id
                for xi in (1., 1.5, 3.):
                    self.driver.mW2_cluster_scale = xi
                    with self.assertLogs(fx.reweight_logger, level='WARNING') as logs:
                        result = self.evaluate()
                    self.fixture.assert_factors(self.driver, result, [1.2, 1.1, 1.05])
                    self.assertIn('ArithmeticError: production pole failed', '\n'.join(logs.output))
                    self.assertEqual(fx.FKS_QUALITY_COUNTS['events_density_scalar_fallback'], event_id)
                    self.assertEqual(fx.FKS_QUALITY_COUNTS['events_density_fallback_production'], event_id)
        self.assertIn('events_density_scalar_fallback', fx.fks_accounting_summary())
        self.assertIn('events_density_fallback_production', fx.fks_accounting_summary())

    def test_decay_exception_reaches_fallback_with_original_cause(self):
        with mock.patch.object(fx.DensityHelper, 'compute_density_matrix',
                               side_effect=[limits.np.eye(9), ArithmeticError('decay boost failed')]):
            self.assert_fallback('decay', 'ArithmeticError: decay boost failed')

    def test_kernel_event_error_uses_scalar_factors(self):
        self.module.density_sudakov.side_effect = ValueError('bad event kinematics')
        self.assert_fallback('kernel', 'ValueError: bad event kinematics')

    def test_diagnostic_is_visible_while_launch_mutes_madgraph_logging(self):
        self.module.density_sudakov.side_effect = ValueError('bad event kinematics')

        @misc.mute_logger()
        def launch(interface):
            return self.evaluate()

        with self.assertLogs('decay.stdout', level='WARNING') as logs:
            result = launch(self.driver)
        self.assertIn('ValueError: bad event kinematics', '\n'.join(logs.output))
        self.fixture.assert_factors(self.driver, result, [1.2, 1.1, 1.05])

    def test_contraction_error_uses_scalar_factors(self):
        with mock.patch.object(fx.DensityHelper, 'compute_density_matrix',
                               side_effect=self.density_matrix), \
                mock.patch.object(fx, 'compute_density_weight',
                                  side_effect=ZeroDivisionError('zero density norm')):
            self.assert_fallback('contraction', 'ZeroDivisionError: zero density norm')

    def test_unavailable_matrix_is_reported(self):
        for stage, matrices, message in (
                ('production', [None], 'B matrix'),
                ('decay', [limits.np.eye(9), None], 'Decay density matrix')):
            with self.subTest(stage=stage), \
                    mock.patch.object(fx.DensityHelper, 'compute_density_matrix', side_effect=matrices):
                fx.fks_reset_accounting()
                self.assert_fallback(stage, message)

    def test_required_kernel_or_libraries_are_fatal(self):
        for missing in ('density_sudakov', 'density_prod_id_to_path', 'density_decay_id_to_path'):
            target = self.module if missing == 'density_sudakov' else self.driver
            previous = getattr(target, missing)
            delattr(target, missing)
            try:
                with self.subTest(missing=missing), \
                        mock.patch.object(fx.DensityHelper, 'compute_density_matrix',
                                          side_effect=self.density_matrix):
                    with self.assertRaises(fx.DensityLibraryError):
                        self.evaluate()
                    self.module.ewsudakov.assert_not_called()
                    self.assertEqual(fx.FKS_QUALITY_COUNTS['events_density_scalar_fallback'], 0)
            finally:
                setattr(target, missing, previous)

    def test_valid_density_evaluation_is_not_a_fallback(self):
        with mock.patch.object(fx.DensityHelper, 'compute_density_matrix',
                               side_effect=self.density_matrix):
            result = self.evaluate()
        self.fixture.assert_factors(self.driver, result, [1.2, 1.1, 1.05])
        self.assertEqual(self.driver._density_logger.log_event.call_args.kwargs['mode'], 'density')
        self.assertEqual(fx.FKS_QUALITY_COUNTS['events_density_scalar_fallback'], 0)

    def test_no_decayed_resonance_is_an_ordinary_scalar_evaluation(self):
        with mock.patch.object(fx.ResonanceIdentifier, 'find_resonances', return_value=[]):
            result = self.evaluate()
        self.fixture.assert_factors(self.driver, result, [1.2, 1.1, 1.05])
        self.module.density_sudakov.assert_not_called()
        self.assertEqual(fx.FKS_QUALITY_COUNTS['events_density_scalar_fallback'], 0)


if __name__ == '__main__':
    unittest.main()
