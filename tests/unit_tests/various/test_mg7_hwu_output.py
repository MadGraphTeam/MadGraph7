##############################################################################
#
# Copyright (c) 2010 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""The mg7 event-sample histograms written in the HwU format (MADatLO.HwU).

The point of the format is that the rest of MadGraph reads it, so the tests
that matter here parse the result back with madgraph.various.histograms --
the very code that plots an aMC@NLO MADatNLO.HwU -- rather than only checking
the text against itself.
"""

from __future__ import absolute_import
import os
import tempfile
import unittest

import madgraph.various.histograms as histograms
from madgraph.iolibs.template_files.mg7 import hwu_output


def make_histogram(name='jet-pt', bin_count=2, weights=True):
    """One histogram in the shape EventHistograms::to_json() writes: the
    arrays carry the underflow first and the overflow last."""
    out = {
        'name': name,
        'min': 0.0,
        'max': 100.0,
        'bin_count': bin_count,
        #        under  bin1  bin2  over
        'bin_values': [0.5, 10.0, 30.0, 1.5],
        'bin_errors': [0.05, 1.0, 3.0, 0.15],
        'scale_envelope': {'low': [0.4, 8.0, 24.0, 1.2],
                           'high': [0.6, 12.0, 36.0, 1.8]},
        'pdf_uncertainty': [{'pdf_set': 'NNPDF40_lo_as_01180',
                             'pdf_lhaid': 331900,
                             'error_type': 'replicas',
                             'central': [0.5, 10.0, 30.0, 1.5],
                             'uncertainty_down': [0.05, 1.0, 2.0, 0.1],
                             'uncertainty_up': [0.05, 2.0, 3.0, 0.1]}],
        'weights': [],
    }
    if weights:
        out['weights'] = [
            {'id': 1, 'bin_values': [0.4, 8.0, 24.0, 1.2], 'bin_errors': [0.] * 4},
            {'id': 2, 'bin_values': [0.6, 12.0, 36.0, 1.8], 'bin_errors': [0.] * 4},
            {'id': 3, 'bin_values': [0.5, 10.0, 30.0, 1.5], 'bin_errors': [0.] * 4},
            {'id': 4, 'bin_values': [0.5, 11.0, 29.0, 1.5], 'bin_errors': [0.] * 4},
        ]
    return out


SUMMARY = {
    'scale': {'weight_ids': [1, 2]},
    'pdf': [{'pdf_lhaid': 331900, 'error_type': 'replicas',
             'weight_ids': [3, 4]}],
    'variations': [
        {'id': 1, 'mur': 0.5, 'muf': 2.0, 'dyn': -1,
         'pdf_lhaid': 331900, 'pdf_member': 0},
        {'id': 2, 'mur': 2.0, 'muf': 0.5, 'dyn': -1,
         'pdf_lhaid': 331900, 'pdf_member': 0},
        {'id': 3, 'mur': 1.0, 'muf': 1.0, 'dyn': -1,
         'pdf_lhaid': 331900, 'pdf_member': 0},
        {'id': 4, 'mur': 1.0, 'muf': 1.0, 'dyn': -1,
         'pdf_lhaid': 331900, 'pdf_member': 1},
    ],
}


class TestHwUOutput(unittest.TestCase):
    """hwu_output.to_hwu: the text it writes"""

    def header(self, text):
        return [c.strip() for c in text.splitlines()[0][4:].split('&')]

    def bin_rows(self, text, index=0):
        rows, inside = [], False
        for line in text.splitlines():
            if line.startswith('<histogram>'):
                inside = True
                continue
            if line.startswith('<\\histogram>'):
                if inside and not index:
                    return rows
                inside = False
                index -= 1
                rows = []
                continue
            if inside and line.strip():
                rows.append([float(x) for x in line.split()])
        return rows

    def test_header_columns(self):
        text = hwu_output.to_hwu([make_histogram()], SUMMARY)
        self.assertEqual(self.header(text),
                         ['xmin', 'xmax', 'central value', 'dy',
                          'delta_mu_min @aux', 'delta_mu_max @aux',
                          'delta_pdf_min @aux', 'delta_pdf_max @aux',
                          'muR=0.50 muF=2.00', 'muR=2.00 muF=0.50',
                          'PDF= 331900', 'PDF= 331901'])

    def test_pdf_central_member_is_not_a_scale_variation(self):
        """the central member carries mur = muf = 1: the group decides"""
        labels = hwu_output.weight_labels(SUMMARY)
        self.assertEqual(labels[3], 'PDF= 331900')
        self.assertEqual(labels[4], 'PDF= 331901')

    def test_dynamical_scale_label(self):
        summary = {'scale': {'weight_ids': [1]},
                   'variations': [{'id': 1, 'mur': 0.5, 'muf': 0.5, 'dyn': 3}]}
        self.assertEqual(hwu_output.weight_labels(summary)[1],
                         'dyn=3 muR=0.50 muF=0.50')

    def test_values_are_differential(self):
        """HwU holds dsigma/dx: sum(value * width) is the cross section"""
        text = hwu_output.to_hwu([make_histogram()], SUMMARY)
        rows = self.bin_rows(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r[0] for r in rows], [0.0, 50.0])
        self.assertEqual([r[1] for r in rows], [50.0, 100.0])
        # 10 pb in a 50 GeV wide bin
        self.assertAlmostEqual(rows[0][2], 10.0 / 50.0)
        self.assertAlmostEqual(rows[1][2], 30.0 / 50.0)
        self.assertAlmostEqual(sum(r[2] * (r[1] - r[0]) for r in rows), 40.0)

    def test_underflow_and_overflow_are_dropped(self):
        """HwU has no such bins; an aMC@NLO analysis drops them too"""
        rows = self.bin_rows(hwu_output.to_hwu([make_histogram()], SUMMARY))
        self.assertNotIn(0.5 / 50.0, [r[2] for r in rows])   # the underflow
        self.assertNotIn(1.5 / 50.0, [r[2] for r in rows])   # the overflow

    def test_no_systematics(self):
        """without variations the file still holds the distribution"""
        histogram = make_histogram(weights=False)
        del histogram['scale_envelope']
        del histogram['pdf_uncertainty']
        text = hwu_output.to_hwu([histogram], None)
        self.assertEqual(self.header(text),
                         ['xmin', 'xmax', 'central value', 'dy'])
        self.assertEqual(len(self.bin_rows(text)), 2)

    def test_nothing_to_write(self):
        self.assertEqual(hwu_output.to_hwu([], SUMMARY), '')
        self.assertEqual(hwu_output.to_hwu(None, None), '')


class TestHwUOutputReadBack(unittest.TestCase):
    """the file is read by madgraph.various.histograms, which is the point"""

    def setUp(self):
        text = hwu_output.to_hwu(
            [make_histogram('jet-pt'), make_histogram('sqrt_s')], SUMMARY)
        handle, self.path = tempfile.mkstemp(suffix='.HwU')
        with os.fdopen(handle, 'w') as stream:
            stream.write(text)

    def tearDown(self):
        os.remove(self.path)

    def test_parsed_by_the_hwu_reader(self):
        parsed = histograms.HwUList(self.path, run_id=0)
        self.assertEqual([h.title for h in parsed], ['jet-pt', 'sqrt_s'])
        self.assertEqual(parsed[0].type, 'LO')
        self.assertEqual(len(parsed[0].bins), 2)
        self.assertAlmostEqual(parsed[0].bins[0].wgts['central'], 10.0 / 50.0)

    def test_weight_labels_are_recognised(self):
        """the labels must read as scale/PDF, or there is no band to draw"""
        parsed = histograms.HwUList(self.path, run_id=0)
        kinds = {}
        for label in parsed[0].bins.weight_labels:
            kind = histograms.HwU.get_HwU_wgt_label_type(label)
            kinds.setdefault(kind, []).append(label)
        self.assertEqual(sorted(kinds['scale']), [('scale', 0.5, 2.0),
                                                  ('scale', 2.0, 0.5)])
        self.assertEqual(sorted(kinds['pdf']), [('pdf', 331900), ('pdf', 331901)])

    def test_scale_envelope_survives_the_round_trip(self):
        """the envelope histograms.py rebuilds is the one madspace computed"""
        parsed = histograms.HwUList(self.path, run_id=0)
        histogram = parsed[0]
        histogram.set_uncertainty(type='all_scale')
        low = [l for l in histogram.bins[0].wgts
               if isinstance(l, str) and l.startswith('delta_mu_min')][0]
        high = [l for l in histogram.bins[0].wgts
                if isinstance(l, str) and l.startswith('delta_mu_max')][0]
        self.assertAlmostEqual(histogram.bins[0].wgts[low], 8.0 / 50.0)
        self.assertAlmostEqual(histogram.bins[0].wgts[high], 12.0 / 50.0)
        self.assertAlmostEqual(histogram.bins[1].wgts[low], 24.0 / 50.0)
        self.assertAlmostEqual(histogram.bins[1].wgts[high], 36.0 / 50.0)
