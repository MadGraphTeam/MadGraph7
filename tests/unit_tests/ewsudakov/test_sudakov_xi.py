"""The Sudakov clustering threshold must be finite and at least M_W^2."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from madgraph.interface.reweight_interface import ReweightInterface
from madgraph.various import fxfx_ewsudakov as fx, lhe_parser
from tests.unit_tests.ewsudakov.test_fks_mappings import _mk_event


class TestSudakovXi(unittest.TestCase):
    def interface(self):
        interface = ReweightInterface.__new__(ReweightInterface)
        interface.exitted = True
        interface.sudakov_xi_scan = [1., 1.5, 3.]
        return interface

    def test_card_accepts_one_and_larger_values(self):
        interface = self.interface()
        for values, expected in (('1', [1.]), ('1 1.5 3', [1., 1.5, 3.]),
                                 ('1e0 2e0', [1., 2.])):
            with self.subTest(values=values):
                interface.do_change('sudakov_xi ' + values)
                self.assertEqual(interface.sudakov_xi_scan, expected)

    def test_invalid_card_raises_and_preserves_previous_scan(self):
        interface = self.interface()
        for values in ('0.5', '0', '-1', 'nan', 'inf', '-inf', '1 0.999999 3',
                       '1 invalid', ''):
            with self.subTest(values=values):
                with self.assertRaisesRegex(interface.InvalidCmd, 'sudakov_xi.*>= 1'):
                    interface.do_change('sudakov_xi ' + values)
                self.assertEqual(interface.sudakov_xi_scan, [1., 1.5, 3.])

    def test_layout_rejects_invalid_thresholds_before_allocating_weights(self):
        for xi in (.5, 0., -1., float('nan'), float('inf')):
            with self.subTest(xi=xi):
                interface = fx.FxFxEWSudakovMixin()
                with self.assertRaisesRegex(ValueError, 'sudakov_xi.*>= 1'):
                    interface._configure_sudakov_weight_ids(['1001'], [1., xi])
                self.assertFalse(hasattr(interface, '_sudakov_weight_map'))

    def test_clustering_rejects_invalid_programmatic_thresholds(self):
        interface = fx.FxFxEWSudakovMixin()
        interface.model = None
        event = _mk_event([(2, -1, 100., 0., 0., 100.),
                           (-2, -1, 100., 0., 0., -100.),
                           (21, 1, 100., 100., 0., 0.),
                           (21, 1, 100., -100., 0., 0.)])
        event.tag = '<clustering type="born"></clustering>'
        for xi in (.5, 0., -1., float('nan'), float('inf')):
            with self.subTest(xi=xi):
                interface.mW2_cluster_scale = xi
                with self.assertRaisesRegex(ValueError, 'sudakov_xi.*>= 1'):
                    interface._cluster_fxfx_event(event)
        interface.mW2_cluster_scale = 1.
        self.assertIsNotNone(interface._cluster_fxfx_event(event))

    def test_catalogue_rejects_invalid_scan_before_opening_event_stream(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'events.lhe'
            path.write_text('<LesHouchesEvents>\n</LesHouchesEvents>\n')
            interface = fx.FxFxEWSudakovMixin()
            interface.inc_sudakov = True
            interface.banner = SimpleNamespace(get=lambda *args: 3)
            interface.lhe_input = SimpleNamespace(path=str(path))
            for xi in (.5, 0., -1., float('nan'), float('inf')):
                with self.subTest(xi=xi), \
                        mock.patch.object(lhe_parser, 'EventFile',
                                          side_effect=AssertionError('Unexpected event scan')):
                    interface.sudakov_xi_scan = [1., xi]
                    with self.assertRaisesRegex(ValueError, 'sudakov_xi.*>= 1'):
                        interface._ensure_fxfx_cluster_catalogue()

    def test_default_threshold_remains_one_point_five(self):
        interface = fx.FxFxEWSudakovMixin()
        interface.banner = SimpleNamespace(get=lambda *args: 3)
        interface.sudakov_xi_scan = None
        interface._sudakov_source_ids = {'1001'}
        ids = interface._prepare_sudakov_weight_ids()
        self.assertEqual(len(ids), 3)
        self.assertEqual(interface._sudakov_xi_list, [1.5])


if __name__ == '__main__':
    unittest.main()
