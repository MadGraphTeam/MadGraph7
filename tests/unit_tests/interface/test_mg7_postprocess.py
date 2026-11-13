"""The guard of run_lhe_postprocessing against legacy systematics on top of
the madspace scale/PDF weights."""

import os
import tempfile
import types
import unittest
from unittest import mock

from madgraph.mg7 import postprocess


class TestNativeSystematicsGuard(unittest.TestCase):

    def run_postprocessing(self, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        open(os.path.join(tmp.name, 'events.lhe'), 'w').close()
        process = types.SimpleNamespace(
            run_path=tmp.name, param_card_path='',
            run_card={'postprocessing': {'systematics': True, 'time_of_flight': -1.0}})
        calls = []
        with mock.patch.object(postprocess, '_run_systematics',
                               lambda *args: calls.append(args)):
            postprocess.run_lhe_postprocessing(process, **kwargs)
        return calls

    def test_legacy_systematics_without_native_weights(self):
        self.assertEqual(len(self.run_postprocessing()), 1)
        self.assertEqual(len(self.run_postprocessing(native_weight_count=0)), 1)

    def test_native_weights_skip_legacy_systematics(self):
        # the steering process of a cluster run has no systematics calculator
        self.assertEqual(self.run_postprocessing(native_weight_count=9), [])


if __name__ == '__main__':
    unittest.main()
