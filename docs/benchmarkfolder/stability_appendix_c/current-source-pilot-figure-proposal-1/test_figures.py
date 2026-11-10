"""Synthetic source-only qualification; no campaign data or actual rendering."""
import copy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('current_source_figure', Path(__file__).with_name('figures.py'))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
PLOTTER_ROOT = Path(__file__).resolve().parents[1] / 'plotter-source-6'


def dataset():
    return {'schema': 'madloop-stability-point-metrics-v1',
            'metric': 'raw-dp-finite-consistency', 'provenance': {'fixture': 'synthetic only'},
            'cases': [{'id': M.CASE, 'point_ids': list(range(1, 101)),
                       'lanes': [{'backend': b, 'values': [None] * 100 if b == 9 else [0] * 99 + [1e-5]}
                                 for b in (1, 6, 7, 8, 9)]}]}


class FakeFigure:
    def __init__(self):
        self.calls = []

    def savefig(self, filename, *args, **kwargs):
        self.calls.append((Path(filename), args, copy.deepcopy(kwargs)))
        Path(filename).write_bytes(('synthetic ' + Path(filename).suffix).encode())


class FigureProposalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_file = self.root / 'synthetic-point-metrics.json'
        self.data = dataset()
        self.write_data()
        self.output = self.root / 'figure'
        self.plotter = M._load_plotter(PLOTTER_ROOT)
        self.real_load_plotter = M._load_plotter
        self.real_render_with_svg = M._render_with_svg
        self.received = []
        patch = mock.patch.object(M, '_load_plotter', return_value=self.plotter)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(M, '_render_with_svg', side_effect=self.fake_render)
        patch.start()
        self.addCleanup(patch.stop)

    def write_data(self):
        raw = (json.dumps(self.data, sort_keys=True) + '\n').encode()
        self.data_file.write_bytes(raw)
        self.data_sha256 = M._sha(raw)

    def fake_render(self, plotter, data, prefix, *, input_sha256):
        self.received.append((copy.deepcopy(data), input_sha256))
        figure = FakeFigure()
        with M._svg_export(FakeFigure, prefix.with_suffix('.pdf')):
            figure.savefig(prefix.with_suffix('.pdf'), bbox_inches='tight',
                           metadata={'CreationDate': None, 'ModDate': None})
            figure.savefig(prefix.with_suffix('.png'), dpi=180, bbox_inches='tight')
        self.figure_calls = figure.calls
        summary = plotter.summarize(data, plotter.threshold_grid(data))
        summary['input_sha256'] = input_sha256
        prefix.with_suffix('.json').write_text(json.dumps(summary) + '\n')
        return summary

    def render(self, **kwargs):
        args = {'point_metrics_file': self.data_file, 'expected_point_metrics_sha256': self.data_sha256,
                'plotter_root': PLOTTER_ROOT}
        args.update(kwargs)
        with mock.patch.dict(sys.modules, {'matplotlib': None, 'matplotlib.pyplot': None}):
            return M.render(self.output, **args)

    def test_single_source_labelled_figure_keeps_statistics_provenance_and_full_input_sha(self):
        result = self.render()
        subset, sha = self.received[0]
        self.assertEqual(sha, self.data_sha256)
        self.assertEqual(subset['provenance'], self.data['provenance'])
        self.assertIn('Current-source QCD pilot', subset['cases'][0]['label'])
        self.assertEqual(subset['cases'][0]['lanes'], self.data['cases'][0]['lanes'])
        expected = self.plotter.summarize(self.data, self.plotter.threshold_grid(self.data))
        expected['input_sha256'] = self.data_sha256
        self.assertEqual(result['summary'], expected)
        self.assertEqual((self.output / 'point-metrics.json').read_bytes(), self.data_file.read_bytes())
        self.assertEqual([p.suffix for p, _, _ in self.figure_calls], ['.pdf', '.svg', '.png'])
        index = json.loads((self.output / 'index.json').read_bytes())
        self.assertEqual(len(index['files']), 6)
        for term in ('110', '1%', 'unavailable', 'not independent true error'):
            self.assertIn(term, index['caption'])
        self.assertEqual(index['dependencies']['plotter_index_sha256'], M.PLOTTER_INDEX_SHA256)

    def test_svg_hook_uses_same_figure_and_preserves_pdf_and_png_arguments(self):
        self.render()
        pdf, svg, png = self.figure_calls
        self.assertEqual(pdf[2]['metadata'], {'CreationDate': None, 'ModDate': None})
        self.assertEqual(svg[2]['metadata'], {'Date': None, 'Creator': 'MadLoop current-source pilot'})
        self.assertEqual(svg[2]['bbox_inches'], pdf[2]['bbox_inches'])
        self.assertEqual(png[2], {'dpi': 180, 'bbox_inches': 'tight'})

    def test_actual_export_helper_uses_fixed_svg_context_without_real_matplotlib(self):
        matplotlib = types.ModuleType('matplotlib')
        matplotlib.use = mock.Mock()
        matplotlib.rc_context = mock.Mock(return_value=nullcontext())
        figure_module = types.ModuleType('matplotlib.figure')
        figure_module.Figure = FakeFigure
        calls = []

        def fake_frozen_render(data, prefix, *, input_sha256):
            calls.append((data, input_sha256))
            FakeFigure().savefig(prefix.with_suffix('.pdf'), bbox_inches='tight')
            return {'synthetic': True}

        plotter = types.SimpleNamespace(render=fake_frozen_render)
        with mock.patch.dict(sys.modules, {'matplotlib': matplotlib, 'matplotlib.figure': figure_module}):
            result = self.real_render_with_svg(plotter, self.data, self.root / 'export',
                                               input_sha256=self.data_sha256)
        self.assertEqual(result, {'synthetic': True})
        self.assertEqual(calls, [(self.data, self.data_sha256)])
        matplotlib.use.assert_called_once_with('Agg')
        matplotlib.rc_context.assert_called_once_with({'svg.hashsalt': M.BASENAME})
        self.assertTrue((self.root / 'export.svg').is_file())

    def test_svg_hook_restores_on_exception_and_ignores_other_pdfs(self):
        original = FakeFigure.savefig
        target = self.root / 'target.pdf'
        other = self.root / 'other.pdf'
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with M._svg_export(FakeFigure, target):
                FakeFigure().savefig(other)
                raise RuntimeError('test failure')
        self.assertIs(FakeFigure.savefig, original)
        self.assertTrue(other.exists())
        self.assertFalse(other.with_suffix('.svg').exists())

    def test_svg_collision_is_rejected_without_overwriting_pdf_or_svg(self):
        target = self.root / 'target.pdf'
        svg = target.with_suffix('.svg')
        svg.write_bytes(b'keep')
        with M._svg_export(FakeFigure, target), self.assertRaises(FileExistsError):
            FakeFigure().savefig(target)
        self.assertFalse(target.exists())
        self.assertEqual(svg.read_bytes(), b'keep')

    def test_wrong_hash_missing_lanes_bad_points_and_nonraw_metric_fail_before_output(self):
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.render(expected_point_metrics_sha256='0' * 64)
        for change in ('lane', 'points', 'metric', 'case'):
            self.data = dataset()
            if change == 'lane':
                self.data['cases'][0]['lanes'].pop()
            elif change == 'points':
                self.data['cases'][0]['point_ids'][0] = 101
            elif change == 'metric':
                self.data['metric'] = 'accepted-composite-internal-estimate'
            else:
                self.data['cases'][0]['id'] = 'other'
            self.write_data()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.render()
            self.assertFalse(self.output.exists())
        self.assertEqual(self.received, [])

    def test_existing_output_and_symlink_inputs_are_rejected(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.render()
        self.assertEqual(self.received, [])
        self.output.rmdir()
        link = self.root / 'data-link.json'
        link.symlink_to(self.data_file)
        with self.assertRaises(ValueError):
            self.render(point_metrics_file=link)
        self.assertFalse(self.output.exists())

    def test_frozen_plotter_index_and_source_bindings_are_enforced(self):
        copied = self.root / 'plotter'
        shutil.copytree(PLOTTER_ROOT, copied)
        source = copied / 'plot_metrics.py'
        source.write_bytes(source.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'Frozen plotter file mismatch'):
            self.real_load_plotter(copied)
        index = copied / 'index.json'
        index.write_bytes(index.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'Frozen plotter index mismatch'):
            self.real_load_plotter(copied)


if __name__ == '__main__':
    unittest.main()
