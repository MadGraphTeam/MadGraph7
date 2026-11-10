import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('appendix_c_points', Path(__file__).with_name('points.py'))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


class PointTests(unittest.TestCase):
    def test_massive_samples_are_physical_cut_qualified_and_deterministic(self):
        inputs = dict(masses=[173., 173.], incoming_masses=[0., 0.], seed=721, count=8)
        first, second = M.generate_cut_points(**inputs), M.generate_cut_points(**inputs)
        self.assertEqual(first, second)
        self.assertTrue(all(M.passes_cuts(p['momenta']) for p in first['points']))
        M.helper().validate_points(first['points'], inputs['masses'], incoming_masses=inputs['incoming_masses'])

    def test_longer_stream_preserves_prefix_and_rng_state(self):
        state = random.getstate()
        inputs = dict(masses=[91.188, 0., 0.], incoming_masses=[0., 0.], seed=722)
        short = M.generate_cut_points(**inputs, count=5)
        long = M.generate_cut_points(**inputs, count=9)
        self.assertEqual(short['points'], long['points'][:5])
        self.assertEqual(state, random.getstate())

    def test_strict_cut_boundary_and_phi_wrap(self):
        initial = [[500.,0,0,500.],[500.,0,0,-500.]]
        self.assertFalse(M.passes_cuts(initial + [[100.,50.,0.,0.],[100.,-100.,0.,0.]]))
        # Azimuths near -pi and +pi are close, rather than 2*pi apart.
        near = [[100.,-100.,1e-6,0.],[100.,-100.,-1e-6,0.]]
        self.assertFalse(M.passes_cuts(initial + near))

    def test_finite_draw_cap_does_not_admit_partial_sample(self):
        with self.assertRaisesRegex(ValueError, 'draw cap'):
            M.generate_cut_points(masses=[0.,0.], incoming_masses=[0.,0.], seed=1,
                                  count=3, max_draws=3, pt_min=501.)

    def test_fresh_freeze_binds_card_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample'
            args = dict(masses=[0.,0.], incoming_masses=[0.,0.], seed=723,
                        count=4, card_sha256='a'*64)
            record = M.freeze_cut_points(path, **args)
            self.assertEqual(hashlib.sha256(path.with_suffix('.dat').read_bytes()).hexdigest(), record['sha256'])
            self.assertEqual(json.loads(path.with_suffix('.json').read_text())['inputs']['resolved_physical_card_sha256'], 'a'*64)
            before = path.with_suffix('.dat').read_bytes()
            with self.assertRaises(FileExistsError): M.freeze_cut_points(path, **args)
            self.assertEqual(before, path.with_suffix('.dat').read_bytes())

    def test_bad_inputs_and_unbound_card_fail(self):
        for count in (0, True, -1):
            with self.assertRaises(ValueError):
                M.generate_cut_points(masses=[0.,0.], incoming_masses=[0.,0.], seed=1, count=count)
        with self.assertRaises(ValueError):
            M.generate_cut_points(masses=[600.,600.], incoming_masses=[0.,0.], seed=1)
        with self.assertRaises(ValueError):
            M.freeze_cut_points('/unreachable', masses=[0.,0.], incoming_masses=[0.,0.], seed=1, card_sha256='bad')


if __name__ == '__main__': unittest.main()
