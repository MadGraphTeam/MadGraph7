"""Density helpers share initialization while continuing to evaluate each event."""

import importlib
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from madgraph.interface.reweight_interface import ReweightInterface
from madgraph.interface import reweight_interface
from madgraph.various import fxfx_ewsudakov as fx
from tests.unit_tests.ewsudakov import test_sudakov_factor_limits as limits
from tests.unit_tests.ewsudakov.test_fxfx_decay_transport import make_event, two_body


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
        self.interface.density_decay_dirname = 'rw_me_density_decay'
        self.interface._density_logger = mock.Mock()
        self.interface._init_pole_masses_from_banner = lambda: None
        self.rows = [[24, -13, 14], [-24, 11, -12]]
        self.module_object = self.module(self.rows)
        (self.root / 'rw_me_density_decay' / 'SubProcesses').mkdir(parents=True)
        (self.root / 'rw_me_density_decay' / 'Cards').mkdir()
        self.write_card(3.)
        self.load()

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

    def write_card(self, value):
        path = self.root / 'rw_me_density_decay' / 'Cards' / 'param_card.dat'
        path.write_text('BLOCK TEST\n 1 %.12g\n' % value)
        return str(path)

    def load(self):
        def importer(name):
            if name.endswith('all_matrix3py'):
                raise ImportError('optional loop module absent')
            return self.module_object

        with mock.patch.object(importlib, 'import_module', side_effect=importer), \
                mock.patch.object(importlib, 'reload', side_effect=lambda module: module):
            self.interface.load_density_modules()

    def decay(self, row=0, owner=None):
        """Decay matrix of a W at rest through the decay library."""
        pdgs = self.rows[row]
        parent = limits.np.array([fx.MW_POLE, 0., 0., 0.])
        daughters = two_body(parent, (0, 0), [.3, .8, math.sqrt(.27)])
        event = make_event([(pdgs[0], -1, parent)] +
                           [(pid, 1, p) for pid, p in zip(pdgs[1:], daughters)])
        helper = fx.DensityHelper(owner or self.interface,
                                  self.interface.density_decay_id_to_path,
                                  self.interface.density_decay_f2pylib)
        return helper.compute_density_matrix(event, pdgs[:1], [3])

    def density_scan(self):
        scalar = self.fixture.scalar_module(self.interface, [1.7, 1.4, 1.05])
        scalar.density_sudakov = mock.Mock(return_value=limits.density_result(9, .2, .1, .05))
        for _ in range(2):
            for xi in (1., 1.5, 3.):
                self.interface.mW2_cluster_scale = xi
                result = self.interface._compute_density_ewsudakov_reweight(self.event, scalar)
                self.fixture.assert_factors(self.interface, result, [1.2, 1.1, 1.05])
                self.assertEqual(self.interface._density_logger.log_event.call_args.kwargs['mode'],
                                 'density')

    def test_decay_card_read_once_across_events_and_xis(self):
        self.density_scan()
        self.assertEqual(self.module_object.py_get_density.call_count, 12)
        for _ in range(2):
            self.decay()
        self.module_object.initialise.assert_called_once_with(
            str(self.root / 'rw_me_density_decay' / 'Cards' / 'param_card.dat'))
        self.module_object.p0_initialisemodel.assert_not_called()
        self.module_object.p1_initialisemodel.assert_not_called()

    def test_subprocess_only_initializers_are_cached_separately(self):
        module = self.module_object
        del module.initialise
        for _ in range(2):
            limits.np.testing.assert_array_equal(self.decay(), limits.np.eye(3) * 3.)
        module.p0_initialisemodel.assert_called_once()
        module.p1_initialisemodel.assert_not_called()
        for _ in range(2):
            limits.np.testing.assert_array_equal(self.decay(row=1), limits.np.eye(3) * 3.)
        self.density_scan()
        module.p0_initialisemodel.assert_called_once()
        module.p1_initialisemodel.assert_called_once()

    def test_explicit_library_reload_resets_initialization_even_for_same_module_object(self):
        self.density_scan()
        self.decay()
        self.write_card(7.)
        self.load()
        self.density_scan()
        limits.np.testing.assert_array_equal(self.decay(), limits.np.eye(3) * 7.)
        self.assertEqual(self.module_object.initialise.call_count, 2)

    def test_replacement_module_is_initialized(self):
        self.decay()
        old = self.module_object
        new = self.module(self.rows)
        self.write_card(5.)
        self.interface.density_decay_f2pylib['rw_me_density_decay', 2] = new
        for _ in range(2):
            limits.np.testing.assert_array_equal(self.decay(), limits.np.eye(3) * 5.)
        old.initialise.assert_called_once()
        new.initialise.assert_called_once()

    def test_initialization_cache_does_not_leak_to_another_interface(self):
        self.decay()
        other = SimpleNamespace(keep_ordering=True, me_dir=str(self.root), options={})
        for _ in range(2):
            limits.np.testing.assert_array_equal(self.decay(owner=other), limits.np.eye(3) * 3.)
        self.assertEqual(self.module_object.initialise.call_count, 2)

    def test_failed_initialization_is_retried(self):
        module = self.module_object
        initialise = module.initialise.side_effect
        module.initialise.side_effect = OSError('card could not be read')
        with self.assertRaisesRegex(OSError, 'card could not be read'):
            self.decay()
        module.initialise.side_effect = initialise
        for _ in range(2):
            limits.np.testing.assert_array_equal(self.decay(), limits.np.eye(3) * 3.)
        self.assertEqual(module.initialise.call_count, 2)

    def test_event_evaluation_failure_does_not_reread_the_card(self):
        module = self.module_object
        density = module.py_get_density.side_effect
        module.py_get_density.side_effect = ArithmeticError('bad event kinematics')
        with self.assertRaisesRegex(ArithmeticError, 'bad event kinematics'):
            self.decay()
        module.py_get_density.side_effect = density
        limits.np.testing.assert_array_equal(self.decay(), limits.np.eye(3) * 3.)
        module.initialise.assert_called_once()


if __name__ == '__main__':
    unittest.main()
