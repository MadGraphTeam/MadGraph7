"""Tests of Template/NLO/Utilities/lhe_utilities.py."""

import contextlib
import gzip
import importlib.util
import io
import json
import os
import re
import tempfile
import unittest

import madgraph

pjoin = os.path.join
MG5DIR = madgraph.MG5DIR
EVENTS = pjoin(MG5DIR, 'tests', 'input_files', 'ttbar_nlo.lhe.gz')


def load_module():
    path = pjoin(MG5DIR, 'Template', 'NLO', 'Utilities', 'lhe_utilities.py')
    spec = importlib.util.spec_from_file_location('lhe_utilities', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = load_module()


def events_of(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as stream:
        text = stream.read()
    return text[text.index('<event'):text.rindex('</LesHouchesEvents>')]


class TestLHEUtilities(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='mg5_lheutil_')
        self.addCleanup(self.directory.cleanup)
        self.root = self.directory.name

    def run_command(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            status = L.main(list(args))
        self.assertEqual(status, 0)
        return out.getvalue()

    def test_split_merge(self):
        self.run_command('split', EVENTS, '3', '-d', self.root, '--no-gzip')
        parts = [pjoin(self.root, 'ttbar_nlo_%03d.lhe' % i) for i in (1, 2, 3)]
        self.assertEqual([L.count_events(p) for p in parts], [34, 33, 33])
        for p, n in zip(parts, (34, 33, 33)):
            self.assertEqual(L.LHEHeader(L.open_events(p).banner).nevents, n)
        merged = pjoin(self.root, 'merged.lhe')
        self.run_command('merge', *parts + ['-o', merged])
        self.assertEqual(events_of(merged), events_of(EVENTS))
        self.assertEqual(L.LHEHeader(L.open_events(merged).banner).nevents,
                         100)

    def test_select(self):
        out = {}
        for keep in ('S', 'H', 'pos', 'neg'):
            out[keep] = pjoin(self.root, keep + '.lhe')
            self.run_command('select', EVENTS, '--keep', keep, '-o',
                             out[keep])
        n = dict((k, L.count_events(p)) for k, p in out.items())
        self.assertEqual(n['S'] + n['H'], 100)
        self.assertEqual(n['pos'] + n['neg'], 100)
        self.assertEqual((n['S'], n['H']), (70, 30))
        rng = pjoin(self.root, 'range.lhe')
        self.run_command('select', EVENTS, '--keep', 'range', '--range', '5',
                         '7', '-o', rng)
        self.assertEqual(L.count_events(rng), 3)

    def test_check(self):
        text = self.run_command('check', EVENTS, '-d', self.root)
        self.assertIn('The total number of S-events is: 70', text)
        self.assertIn('Total number of errors found: 0', text)
        for name in ('LHEF.errors', 'LHEF.stats', 'LHEF.rwgt',
                     'check_events.HwU', 'res_wgt'):
            self.assertTrue(os.path.isfile(pjoin(self.root, name)), name)

    def test_quoted_run_card_values_preserve_cross_section_normalization(self):
        with gzip.open(EVENTS, 'rt') as stream:
            contents = stream.read()
        ef = L.open_events(EVENTS)
        weights = [L.LHEEvent(text).wgt for text in L.iter_raw(ef)]
        ef.close()
        analysis = pjoin(self.root, 'rates.py')
        with open(analysis, 'w') as stream:
            stream.write("def book(book): book('total rate', 1, 0., 1.)\n"
                         "def analyse(event, fill): fill('total rate', .5)\n")
        for index, value in enumerate(("'average'", '"average"', "'bias'",
                                        '"bias"', "'sum'")):
            with self.subTest(event_norm=value):
                directory = pjoin(self.root, str(index))
                os.mkdir(directory)
                path = pjoin(directory, 'events.lhe')
                text, count = re.subn(r'(?m)^\s*\S+[^\S\n]*=[^\S\n]*event_norm\b[^\n]*',
                                      '  %s = event_norm' % value, contents)
                self.assertEqual(count, 1)
                with open(path, 'w') as stream:
                    stream.write(text)
                expected = sum(weights)
                if 'sum' not in value:
                    expected /= len(weights)
                tolerance = 1e-6 * max(1., abs(expected))
                self.run_command('check', path, '-d', directory)
                with open(pjoin(directory, 'res_wgt')) as stream:
                    cross_section = float(stream.readline().split('=')[1].split()[0])
                self.assertAlmostEqual(cross_section, expected, delta=tolerance)
                histogram = pjoin(directory, 'rates.HwU')
                self.run_command('plot', path, '--analysis', analysis, '-o', histogram)
                _, histograms = L.read_hwu(histogram)
                self.assertAlmostEqual(histograms[0][1][0][2], expected, delta=tolerance)
                report = pjoin(directory, 'uncertainties.json')
                self.run_command('uncertainties', path, '--pdf-error', 'replicas',
                                 '-o', report)
                with open(report) as stream:
                    result = json.load(stream)
                self.assertAlmostEqual(result['cross_section'], expected, delta=tolerance)
        header = L.LHEHeader("  'PYTHIA8' = parton_shower\n")
        self.assertEqual(header.parton_shower, 'PYTHIA8')

    def test_rescale(self):
        ef = L.open_events(EVENTS)
        text = next(L.iter_raw(ef))
        ef.close()
        event = L.LHEEvent(text)
        scaled = L.LHEEvent(L.rescale_event(text, 0.25))
        self.assertAlmostEqual(scaled.wgt, 0.25 * event.wgt)
        for wid, value in event.rwgt().items():
            self.assertAlmostEqual(scaled.rwgt()[wid], 0.25 * value,
                                   delta=1e-6 * abs(value))
        self.assertEqual(scaled.pup, event.pup)

    def test_pdf_uncertainty(self):
        members = [[1.], [1.02], [0.97], [1.01], [1.03]]
        c, minus, plus = L.pdf_uncertainty(members, 'hessian')
        self.assertAlmostEqual(plus[0], (0.02 ** 2 + 0.03 ** 2) ** 0.5)
        self.assertAlmostEqual(minus[0], 0.03)
        c, minus, plus = L.pdf_uncertainty(members, 'symmhessian')
        self.assertAlmostEqual(plus[0], (0.02 ** 2 + 0.03 ** 2 + 0.01 ** 2 +
                                         0.03 ** 2) ** 0.5)
        c, minus, plus = L.pdf_uncertainty([[1.], [1.], [2.], [3.]],
                                           'replicas')
        self.assertEqual(c, [2.])
        self.assertAlmostEqual(plus[0], 1.)
        # +as: the last two members are ignored
        c, minus, plus = L.pdf_uncertainty(members + [[5.], [5.]],
                                           'hessian+as')
        self.assertAlmostEqual(minus[0], 0.03)
        # 90% CL to 68% CL
        c, minus, plus = L.pdf_uncertainty(members, 'hessian', 90., 68.27)
        self.assertAlmostEqual(minus[0], 0.03 / 1.6449, places=4)

    def test_replica_uncertainty_intervals_use_replica_means(self):
        for error_type in ('replicas', 'replicas+as'):
            with self.subTest(error_type=error_type):
                values = [[1., 5.], [1., 2.], [2., 4.], [3., 6.]]
                if error_type.endswith('+as'):
                    values += [[-100., 400.], [1000., -500.]]
                infos = [L.WeightInfo(str(i), 'PDF= %d Example_mc' % (300000 + i))
                         for i in range(len(values))]
                groups = L.VariationGroups(infos)
                args = L.build_parser().parse_args(
                    ['uncertainties', 'unused.lhe', '--pdf-error', error_type])
                result, = L.UncertaintyEngine(args).compute(
                    groups, lambda w: values[int(w.id)])
                self.assertEqual(result['central'], [2., 4.])
                self.assertEqual(result['min'], [1., 2.])
                self.assertEqual(result['max'], [3., 6.])
                self.assertEqual(values[0], [1., 5.])
        self.assertEqual(L.pdf_uncertainty([[9.], [3.]], 'replicas'),
                         ([3.], [0.], [0.]))

    def test_scale_uncertainty(self):
        points = {(1., 1.): [1.], (2., 1.): [0.9], (0.5, 1.): [1.1],
                  (2., 0.5): [1.5], (0.5, 2.): [0.5]}
        res = L.scale_uncertainty(points)
        self.assertEqual(res['envelope'][1:3], ([0.5], [1.5]))
        self.assertEqual(res['7-point'][1:3], ([0.9], [1.1]))

    def test_weight_labels(self):
        w = L.WeightInfo('1', 'tag= 400 dyn= -1 muR=0.20000E+01 '
                         'muF=0.50000E+00')
        self.assertEqual((w.kind, w.tag, w.dyn, w.muR, w.muF),
                         ('scale', 400, -1, 2., 0.5))
        w = L.WeightInfo('2', 'PDF= 260001 NNPDF31_nlo_as_0118')
        self.assertEqual((w.kind, w.lhaid, w.pdfset),
                         ('pdf', 260001, 'NNPDF31_nlo_as_0118'))
        self.assertEqual(w.hwu_label(), 'PDF=260001 NNPDF31_nlo_as_0118')

    def test_colour_checks(self):
        ef = L.open_events(EVENTS)
        event = L.LHEEvent(next(L.iter_raw(ef)))
        ef.close()
        self.assertEqual(L.colour_messages(event, event.external()), [])
        i = event.external()[0]
        event.icolup[i] = (event.icolup[i][0] + 77, event.icolup[i][1])
        self.assertTrue(L.colour_messages(event, event.external()))


if __name__ == '__main__':
    unittest.main()
