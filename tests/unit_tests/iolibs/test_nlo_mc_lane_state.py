"""Independent native draws own the Born selection, cuts and signed records."""

import subprocess

from tests.unit_tests.fks.test_born_flow_rejection import NativeIntegrandFixture


class TestMCLaneState(NativeIntegrandFixture):
    def test_new_native_draws_start_with_empty_weights_and_a_fresh_born(self):
        # Alternate passing and failing cuts after deliberately poisoning the
        # previous lane's n-body flag, cached Born, and MINT cut indicator.
        for width in (1, 4):
            with self.subTest(width=width):
                data = (f'{width} F 0\n1 1 1 1 1 1 1 1\n' +
                        ' '.join([str(1. / width)] * width + ['0'] * (8-width)) + '\n')
                result = subprocess.run([self.executable, 'lane_reset'], input=data,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('PASS lane_reset', result.stdout)
