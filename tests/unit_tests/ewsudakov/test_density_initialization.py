"""Density helpers share initialization while continuing to evaluate each event."""

import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from madgraph.interface.reweight_interface import ReweightInterface
from madgraph.interface import reweight_interface
from madgraph.various import fxfx_ewsudakov as fx
from tests.unit_tests.ewsudakov import test_sudakov_factor_limits as limits


@unittest.skipIf(limits.np is None, 'NumPy is required for density evaluation')
class TestDensityInitialization(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = limits.TestSudakovFactorLimits()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.event = self.fixture.event
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
        self.interface.options = {}
        self.interface.banner = SimpleNamespace(get=lambda *args: 3)
        self.interface.model = None
        self.interface.mW2_cluster_scale = 1.5
        self.interface.fxfx_clustered_final_states = set()
        self.interface.fxfx_clustered_decay_processes = {(24, (-13, 14)), (-24, (11, -12))}
        self.interface.density_prod_dirname = 'rw_me_density_prod'
        self.interface.density_decay_dirname = 'rw_me_density_decay'
        self.interface._density_logger = mock.Mock()
        self.interface._init_pole_masses_from_banner = lambda: None
        self.rows = {
            'rw_me_density_prod': [[2, -2, 24, -24], [1, -1, 24, -24]],
            'rw_me_density_decay': [[24, -13, 14], [-24, 11, -12]],
        }
        self.modules = {name: self.module(rows) for name, rows in self.rows.items()}
        for name in self.modules:
            (self.root / name / 'SubProcesses').mkdir(parents=True)
            (self.root / name / 'Cards').mkdir()
            self.write_card(name, 2. if name.endswith('prod') else 3.)
        self.load()
        self.clustered = self.interface._cluster_fxfx_event(self.event)[0]

    def module(self, rows):
        """A standalone module whose initializer reads a card used by its matrices."""
        module = SimpleNamespace()
        module.get_pdg_order = lambda: (rows, list(range(1, len(rows) + 1)))
        module.get_prefix = lambda: [('p%d_' % i).encode() for i in range(len(rows))]
        parameters = [None] * len(rows)

        def initialise(path, indices):
            value = float(Path(path).read_text().split()[-1])
            for index in indices:
                parameters[index] = value

        module.initialise = mock.Mock(side_effect=lambda path: initialise(path, range(len(rows))))
        for i in range(len(rows)):
            setattr(module, 'p%d_initialisemodel' % i, mock.Mock(
                side_effect=lambda path, index=i: initialise(path, [index])))

        def density(pdgs, procid, momenta, positions, helicities, alphas, scale2):
            index = rows.index(list(pdgs))
            if parameters[index] is None:
                raise AssertionError('Subprocess model was not initialized')
            dimension = len(helicities) // len(positions)
            matrix = limits.np.eye(dimension) * parameters[index]
            return matrix[limits.np.triu_indices(dimension)]

        module.py_get_density = mock.Mock(side_effect=density)
        return module

    def write_card(self, name, value):
        path = self.root / name / 'Cards' / 'param_card.dat'
        path.write_text('BLOCK TEST\n 1 %.12g\n' % value)
        return str(path)

    def load(self):
        def importer(name):
            if name.endswith('all_matrix3py'):
                raise ImportError('optional loop module absent')
            return self.modules[name.split('.')[0]]

        with mock.patch.object(importlib, 'import_module', side_effect=importer), \
                mock.patch.object(importlib, 'reload', side_effect=lambda module: module):
            self.interface.load_density_modules()

    def production(self, event=None):
        helper = fx.DensityHelper(self.interface, self.interface.density_prod_id_to_path,
                                   self.interface.density_prod_f2pylib)
        return helper.compute_density_matrix(event if event is not None else self.clustered,
                                             [24, -24], [3, 3])

    def density_scan(self):
        scalar = self.fixture.scalar_module(self.interface, [1.7, 1.4, 1.05])
        scalar.density_sudakov = mock.Mock(return_value={
            'density_delta': limits.np.full(9, .1),
            'density_delta_var': limits.np.full(9, .05),
            'born_diag': limits.np.ones(9), 'total_dim': 9, 'nres': 2})
        for _ in range(2):
            for xi in (1., 1.5, 3.):
                self.interface.mW2_cluster_scale = xi
                result = self.interface._compute_density_ewsudakov_reweight(self.event, scalar)
                self.fixture.assert_factors(self.interface, result, [1.2, 1.1, 1.05])
                self.assertEqual(self.interface._density_logger.log_event.call_args.kwargs['mode'],
                                 'density')

    def test_production_and_decay_cards_read_once_across_events_and_xis(self):
        self.density_scan()
        for name, module in self.modules.items():
            module.initialise.assert_called_once_with(
                str(self.root / name / 'Cards' / 'param_card.dat'))
            module.p0_initialisemodel.assert_not_called()
            module.p1_initialisemodel.assert_not_called()
        self.assertEqual(self.modules['rw_me_density_prod'].py_get_density.call_count, 6)
        self.assertEqual(self.modules['rw_me_density_decay'].py_get_density.call_count, 12)

    def test_subprocess_only_initializers_are_cached_separately(self):
        for module in self.modules.values():
            del module.initialise
        self.density_scan()
        prod = self.modules['rw_me_density_prod']
        decay = self.modules['rw_me_density_decay']
        prod.p0_initialisemodel.assert_called_once()
        prod.p1_initialisemodel.assert_not_called()
        decay.p0_initialisemodel.assert_called_once()
        decay.p1_initialisemodel.assert_called_once()
        other_channel = self.interface._cluster_fxfx_event(self.event)[0]
        other_channel[0].pid, other_channel[1].pid = 1, -1
        for _ in range(2):
            limits.np.testing.assert_array_equal(self.production(other_channel), limits.np.eye(9) * 2.)
        prod.p0_initialisemodel.assert_called_once()
        prod.p1_initialisemodel.assert_called_once()

    def test_explicit_library_reload_resets_initialization_even_for_same_module_object(self):
        self.density_scan()
        self.write_card('rw_me_density_prod', 5.)
        self.write_card('rw_me_density_decay', 7.)
        self.load()
        self.density_scan()
        limits.np.testing.assert_array_equal(self.production(), limits.np.eye(9) * 5.)
        for module in self.modules.values():
            self.assertEqual(module.initialise.call_count, 2)

    def test_replacement_module_is_initialized(self):
        self.production()
        old = self.modules['rw_me_density_prod']
        new = self.module(self.rows['rw_me_density_prod'])
        self.write_card('rw_me_density_prod', 5.)
        self.interface.density_prod_f2pylib['rw_me_density_prod', 2] = new
        for _ in range(2):
            limits.np.testing.assert_array_equal(self.production(), limits.np.eye(9) * 5.)
        old.initialise.assert_called_once()
        new.initialise.assert_called_once()

    def test_initialization_cache_does_not_leak_to_another_interface(self):
        self.production()
        other = SimpleNamespace(keep_ordering=True, me_dir=str(self.root), options={})
        for _ in range(2):
            helper = fx.DensityHelper(other, self.interface.density_prod_id_to_path,
                                      self.interface.density_prod_f2pylib)
            limits.np.testing.assert_array_equal(helper.compute_density_matrix(
                self.clustered, [24, -24], [3, 3]), limits.np.eye(9) * 2.)
        self.assertEqual(self.modules['rw_me_density_prod'].initialise.call_count, 2)

    def test_failed_initialization_is_retried(self):
        module = self.modules['rw_me_density_prod']
        initialise = module.initialise.side_effect
        module.initialise.side_effect = OSError('card could not be read')
        with self.assertRaisesRegex(OSError, 'card could not be read'):
            self.production()
        module.initialise.side_effect = initialise
        for _ in range(2):
            limits.np.testing.assert_array_equal(self.production(), limits.np.eye(9) * 2.)
        self.assertEqual(module.initialise.call_count, 2)

    def test_event_evaluation_failure_does_not_reread_the_card(self):
        module = self.modules['rw_me_density_prod']
        density = module.py_get_density.side_effect
        module.py_get_density.side_effect = ArithmeticError('bad event kinematics')
        with self.assertRaisesRegex(ArithmeticError, 'bad event kinematics'):
            self.production()
        module.py_get_density.side_effect = density
        limits.np.testing.assert_array_equal(self.production(), limits.np.eye(9) * 2.)
        module.initialise.assert_called_once()


if __name__ == '__main__':
    unittest.main()
