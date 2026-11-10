"""Synthetic metrics and mocked artifacts only; no raw data or matplotlib."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('stability_figures', Path(__file__).with_name('figures.py'))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
PLOTTER_ROOT = Path(__file__).resolve().parents[1] / 'plotter-source-6'


def dataset():
    return {'schema': 'madloop-stability-point-metrics-v1',
            'metric': 'raw-dp-finite-consistency',
            'provenance': {'closed_index_sha256': 'a' * 64,
                           'qualification': 'synthetic test; not measured'},
            'cases': [{'id': case_id, 'label': case_id, 'point_ids': list(range(1, 101)),
                       'lanes': [{'backend': backend,
                                  'values': [None] * 100 if backend == 9 else [0] * 99 + [1e-5]}
                                 for backend in (1, 6, 7, 8, 9)]} for case_id in M.CASES]}


class FigureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_root = self.root / 'data'
        self.data_root.mkdir()
        self.figure_root = self.root / 'figures'
        self.data = dataset()
        self.plotter = M._load_plotter(PLOTTER_ROOT)
        self.real_load_plotter = M._load_plotter
        self.calls = []
        patch = mock.patch.object(M, '_load_plotter', return_value=self.plotter)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(self.plotter, 'render', side_effect=self.fake_render)
        patch.start()
        self.addCleanup(patch.stop)
        self.write_capsule(self.data)

    def write_capsule(self, data, *, kind='file'):
        raw = M._json(data)
        (self.data_root / 'point-metrics.json').write_bytes(raw)
        binding = {'sha256': M._sha(raw), 'bytes': len(raw)}
        index = {'schema': 'madloop-raw-stability-capsule-v1',
                 'payloads': {'point-metrics.json': dict(binding, kind=kind, path='point-metrics.json')},
                 'files': {'point-metrics.json': binding}}
        (self.data_root / 'index.json').write_bytes(M._json(index))
        self.data_index_sha256 = M._sha((self.data_root / 'index.json').read_bytes())

    def fake_render(self, data, prefix, *, input_sha256):
        self.calls.append((copy.deepcopy(data), input_sha256))
        prefix = Path(prefix)
        prefix.with_suffix('.pdf').write_bytes(b'%PDF synthetic, not rendered\n')
        prefix.with_suffix('.png').write_bytes(b'PNG synthetic, not rendered\n')
        summary = self.plotter.summarize(data, self.plotter.threshold_grid(data))
        summary['input_sha256'] = input_sha256
        prefix.with_suffix('.json').write_bytes(M._json(summary))
        return summary

    def render(self, **overrides):
        kwargs = {'data_root': self.data_root, 'plotter_root': PLOTTER_ROOT,
                  'expected_data_index_sha256': self.data_index_sha256}
        kwargs.update(overrides)
        return M.render(self.figure_root, **kwargs)

    def verify(self):
        with mock.patch.object(self.plotter, 'render', side_effect=AssertionError('verification rendered')):
            return M.verify(self.figure_root, data_root=self.data_root, plotter_root=PLOTTER_ROOT)

    def rebind_artifact(self, path):
        index_path = self.figure_root / 'index.json'
        index = json.loads(index_path.read_bytes())
        raw = (self.figure_root / path).read_bytes()
        index['files'][path] = {'sha256': M._sha(raw), 'bytes': len(raw)}
        index_path.write_bytes(M._json(index))

    def test_all_four_single_case_figures_keep_complete_input_sha_and_provenance(self):
        result = self.render()
        self.assertTrue(result['verified'])
        self.assertEqual(result['artifacts'], 12)
        self.assertEqual(len(self.calls), 4)
        full_sha = M._sha((self.data_root / 'point-metrics.json').read_bytes())
        for (subset, sha), case_id in zip(self.calls, M.CASES):
            self.assertEqual(sha, full_sha)
            self.assertEqual(subset['provenance'], self.data['provenance'])
            self.assertEqual([c['id'] for c in subset['cases']], [case_id])
            self.assertEqual(len(subset['cases'][0]['lanes']), 5)
        self.assertEqual((self.figure_root / 'point-metrics.json').read_bytes(),
                         (self.data_root / 'point-metrics.json').read_bytes())
        self.assertEqual(len(json.loads((self.figure_root / 'index.json').read_bytes())['files']), 16)
        self.assertEqual(self.verify(), result)

    def test_wholly_unavailable_lane_keeps_fixed_denominator_bounds_and_masked_ratios(self):
        self.render()
        summary = json.loads((self.figure_root / 'figures' / f'{M.CASES[0]}.json').read_bytes())
        missing = summary['cases'][0]['lanes'][-1]
        self.assertTrue(all((r['n'], r['unavailable'], r['lower'], r['upper']) == (100, 100, 0, 1)
                            for r in missing['tail']))
        self.assertTrue(all(r is None for r in missing['ratio_over_Ninja']))

    def test_verify_does_not_import_matplotlib_or_call_render(self):
        self.render()
        with mock.patch.dict(sys.modules, {'matplotlib': None, 'matplotlib.pyplot': None}), \
                mock.patch.object(M, '_load_plotter', side_effect=self.real_load_plotter):
            self.assertTrue(self.verify()['verified'])

    def test_wrong_input_index_is_rejected_before_creating_outputs(self):
        with self.assertRaisesRegex(ValueError, 'Data index mismatch'):
            self.render(expected_data_index_sha256='0' * 64)
        self.assertFalse(self.figure_root.exists())
        self.assertEqual(self.calls, [])

    def test_verify_rejects_changed_data_index_or_payload(self):
        self.render()
        index_path = self.data_root / 'index.json'
        saved = index_path.read_bytes()
        index_path.write_bytes(saved + b'\n')
        with self.assertRaisesRegex(ValueError, 'Data index mismatch'):
            self.verify()
        index_path.write_bytes(saved)
        with (self.data_root / 'point-metrics.json').open('ab') as output:
            output.write(b'\n')
        with self.assertRaisesRegex(ValueError, 'File binding mismatch'):
            self.verify()

    def test_payload_file_binding_and_direct_storage_are_required(self):
        self.write_capsule(self.data, kind='gzip')
        with self.assertRaisesRegex(ValueError, 'Unsupported point-metrics storage'):
            self.render()
        self.assertFalse(self.figure_root.exists())
        self.write_capsule(self.data)
        path = self.data_root / 'index.json'
        index = json.loads(path.read_bytes())
        index['files']['point-metrics.json']['bytes'] += 1
        path.write_bytes(M._json(index))
        self.data_index_sha256 = M._sha(path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'binding disagree'):
            self.render()

    def test_missing_cases_lanes_or_points_are_rejected(self):
        for change in ('case', 'lane', 'point', 'metric'):
            data = dataset()
            if change == 'case':
                data['cases'].pop()
            elif change == 'lane':
                data['cases'][0]['lanes'].pop()
            elif change == 'point':
                data['cases'][0]['point_ids'][0] = 101
            else:
                data['metric'] = 'accepted-composite-internal-estimate'
            self.write_capsule(data)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.render()
            self.assertFalse(self.figure_root.exists())

    def test_existing_directory_is_never_overwritten(self):
        self.figure_root.mkdir()
        marker = self.figure_root / 'keep'
        marker.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            self.render()
        self.assertEqual(marker.read_bytes(), b'keep')
        self.assertEqual(self.calls, [])

    def test_pdf_and_png_byte_tampering_are_detected(self):
        self.render()
        for suffix in ('pdf', 'png'):
            path = self.figure_root / 'figures' / f'{M.CASES[0]}.{suffix}'
            saved = path.read_bytes()
            path.write_bytes(saved + b'changed')
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, 'File binding mismatch'):
                self.verify()
            path.write_bytes(saved)

    def test_rebound_statistical_json_tampering_is_detected_by_recomputation(self):
        self.render()
        name = f'figures/{M.CASES[0]}.json'
        path = self.figure_root / name
        summary = json.loads(path.read_bytes())
        summary['cases'][0]['lanes'][0]['tail'][0]['lower'] = .75
        path.write_bytes(M._json(summary))
        self.rebind_artifact(name)
        with self.assertRaisesRegex(ValueError, 'Statistical figure summary mismatch'):
            self.verify()

    def test_rebound_dataset_provenance_and_inventory_tampering_are_detected(self):
        self.render()
        path = self.figure_root / 'provenance.json'
        path.write_bytes(b'{}\n')
        self.rebind_artifact('provenance.json')
        with self.assertRaisesRegex(ValueError, 'Figure provenance mismatch'):
            self.verify()
        index_path = self.figure_root / 'index.json'
        index = json.loads(index_path.read_bytes())
        index['files'].pop(f'figures/{M.CASES[0]}.pdf')
        index_path.write_bytes(M._json(index))
        with self.assertRaisesRegex(ValueError, 'Unexpected figure inventory'):
            self.verify()

    def test_symlink_payload_artifact_and_output_parent_are_rejected(self):
        self.render()
        path = self.figure_root / 'figures' / f'{M.CASES[0]}.pdf'
        stored = self.root / 'stored.pdf'
        path.rename(stored)
        path.symlink_to(stored)
        with self.assertRaises((OSError, ValueError)):
            self.verify()
        output_parent = self.root / 'output-parent'
        output_parent.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink figure root or parent'):
            M.render(output_parent / 'fresh', data_root=self.data_root, plotter_root=PLOTTER_ROOT,
                     expected_data_index_sha256=self.data_index_sha256)
        self.assertFalse((self.root / 'fresh').exists())
        payload = self.data_root / 'point-metrics.json'
        stored = self.root / 'stored-data.json'
        payload.rename(stored)
        payload.symlink_to(stored)
        with self.assertRaises((OSError, ValueError)):
            self.render()

    def test_actual_frozen_plotter_pins_are_enforced(self):
        copied = self.root / 'plotter'
        shutil.copytree(PLOTTER_ROOT, copied)
        index = copied / 'index.json'
        index.write_bytes(index.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'Frozen plotter index mismatch'):
            self.real_load_plotter(copied)

    def test_extra_file_and_unindexed_top_level_symlink_are_rejected(self):
        self.render()
        extra = self.figure_root / 'unindexed.txt'
        extra.write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'Physical figure inventory mismatch'):
            self.verify()
        extra.unlink()
        extra.symlink_to(self.data_root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Unexpected symlink'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
