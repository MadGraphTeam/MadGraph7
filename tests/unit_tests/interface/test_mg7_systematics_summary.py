################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""The scan summary reads scale/PDF variations from madspace's own report.

madspace computes the systematics while writing the events and puts them in
Events/<run>/info.json under "systematics". The base MadEventCmd parser reads
the parton_systematics.log that systematics.py used to write, which an mg7 run
no longer produces unless the legacy post-processing path was used.
"""

from __future__ import absolute_import

import json
import os
import shutil
import tempfile
import unittest

import madgraph.iolibs.template_files.mg7.run_interface as run_interface
import madgraph.iolibs.template_files.mg7.systematics_summary as systematics_summary

pjoin = os.path.join


class _Parser(run_interface.MG7RunCmd):
    """Just the parser: the real __init__ wants a whole output directory."""

    def __init__(self):
        pass


class TestNativeSystematicsSummary(unittest.TestCase):

    NOMINAL = 500.0

    def setUp(self):
        self.parser = _Parser()
        self.me_dir = tempfile.mkdtemp()
        self.run = 'run_01'
        os.makedirs(pjoin(self.me_dir, 'Cards'))
        os.makedirs(pjoin(self.me_dir, 'Events', self.run))
        self.kpath = pjoin(self.me_dir, 'Cards', 'param_card.dat')
        open(self.kpath, 'w').close()

    def tearDown(self):
        shutil.rmtree(self.me_dir, ignore_errors=True)

    def write_info(self, systematics):
        info = {'process': 'p p > t t~', 'status': 'done'}
        if systematics is not None:
            info['systematics'] = systematics
        with open(pjoin(self.me_dir, 'Events', self.run, 'info.json'), 'w') as f:
            json.dump(info, f)

    def parse(self):
        return self.parser.getSysSummaryFromLog(kpath=self.kpath,
                                                knext_name=self.run)

    def full_summary(self):
        """The shape SystematicsCalculator::summary() writes."""

        return {
            'nominal': {'pdf_set': 'NNPDF23_lo', 'pdf_lhaid': 247000,
                        'cross_section': self.NOMINAL},
            'event_count': 10000,
            'warnings': [],
            'scale': {'weight_ids': ['1', '2'],
                      'min': 0.785 * self.NOMINAL,      # -21.5%
                      'max': 1.296 * self.NOMINAL},     # +29.6%
            'pdf': [{'pdf_set': 'NNPDF23_lo', 'pdf_lhaid': 247000,
                     'error_type': 'replicas', 'weight_ids': ['11', '12'],
                     'central': self.NOMINAL,
                     'uncertainty_up': 0.02 * self.NOMINAL,      # +2%
                     'uncertainty_down': 0.02 * self.NOMINAL}],  # -2%
        }

    #-- the happy path ------------------------------------------------------

    def test_scale_envelope_becomes_signed_percentages(self):
        self.write_info(self.full_summary())
        scale, _pdf = self.parse()
        self.assertEqual(scale, ['+29.6', '-21.5'])

    def test_pdf_uncertainty_is_relative_to_the_sets_central_value(self):
        """uncertainty_up/down are absolute, and the percentage is against the
        set's own central value -- which is what log_systematics_summary
        prints in the same run. For a replicas set that is the replica mean,
        not the nominal member."""

        self.write_info(self.full_summary())
        _scale, pdf = self.parse()
        self.assertEqual(pdf, ['+2', '-2'])

    def test_it_agrees_with_what_the_run_printed(self):
        """The scan column and the run's own Systematics box must not differ.

        Mirrors launch.py's log_systematics_summary arithmetic.
        """

        summary = self.full_summary()
        # a replicas set whose mean sits away from the nominal member
        summary['pdf'][0]['central'] = 0.9 * self.NOMINAL
        self.write_info(summary)
        _scale, pdf = self.parse()

        entry = summary['pdf'][0]
        printed_up = entry['uncertainty_up'] / entry['central'] * 100
        printed_down = entry['uncertainty_down'] / entry['central'] * 100
        self.assertAlmostEqual(float(pdf[0]), printed_up, places=2)
        self.assertAlmostEqual(float(pdf[1]), -printed_down, places=2)

    def test_an_entry_without_a_central_is_skipped(self):
        """summary() omits "central" when it could not be computed, and both
        readers skip such an entry rather than measuring it against something
        else. One rule, shared -- see systematics_summary.pdf_percentages."""

        summary = self.full_summary()
        summary['pdf'][0].pop('central')
        self.write_info(summary)
        scale, pdf = self.parse()
        self.assertEqual(scale, ['+29.6', '-21.5'])
        self.assertEqual(pdf, [])

    def test_it_matches_what_the_legacy_log_would_have_given(self):
        """Same numbers, either path -- the scan column must not shift.

        systematics.py writes '+(max-nom)/nom' and '-(nom-min)/nom', which is
        the convention reproduced here.
        """

        summary = self.full_summary()
        legacy_scale_hi = (summary['scale']['max'] - self.NOMINAL) / self.NOMINAL * 100
        legacy_scale_lo = (self.NOMINAL - summary['scale']['min']) / self.NOMINAL * 100
        self.write_info(summary)
        scale, _pdf = self.parse()
        self.assertAlmostEqual(float(scale[0]), legacy_scale_hi, places=2)
        self.assertAlmostEqual(float(scale[1]), -legacy_scale_lo, places=2)

    #-- everything that can be missing --------------------------------------

    def test_no_info_file_falls_back(self):
        """No info.json: the base parser runs, and raises for the missing log
        exactly as it does for a madevent run. store_scan_result catches that
        and records no variation -- keeping the behaviour identical either
        side is the point, so this asserts the exception rather than papering
        over it."""

        self.assertRaises(OSError, self.parse)

    def test_info_without_systematics_falls_back(self):
        """A run that computed none: the key is simply absent, and the legacy
        parton_systematics.log is still the place to look (the legacy
        post-processing path writes both)."""

        self.write_info(None)
        self.assertRaises(OSError, self.parse)

    def test_a_run_with_no_events_reports_nothing(self):
        """summary() omits cross_section when event_count is 0."""

        summary = self.full_summary()
        summary['event_count'] = 0
        del summary['nominal']['cross_section']
        summary.pop('scale')
        self.write_info(summary)
        self.assertEqual(self.parse(), ([], []))

    def test_missing_scale_block_still_gives_pdf(self):
        """A run varying only the PDF has no "scale" key."""

        summary = self.full_summary()
        summary.pop('scale')
        self.write_info(summary)
        scale, pdf = self.parse()
        self.assertEqual(scale, [])
        self.assertEqual(pdf, ['+2', '-2'])

    def test_a_one_member_set_gives_no_pdf_numbers(self):
        """errorset on a single-member set yields no uncertainty -- the entry
        is there but carries no uncertainty_up/down."""

        summary = self.full_summary()
        summary['pdf'] = [{'pdf_set': 'NNPDF40MC_lo_as_01180',
                           'pdf_lhaid': 335000, 'error_type': '',
                           'weight_ids': []}]
        self.write_info(summary)
        scale, pdf = self.parse()
        self.assertEqual(scale, ['+29.6', '-21.5'])
        self.assertEqual(pdf, [])

    def test_a_corrupt_info_file_falls_back(self):
        """A truncated info.json must not be mistaken for "no systematics"."""

        with open(pjoin(self.me_dir, 'Events', self.run, 'info.json'), 'w') as f:
            f.write('{ not json')
        self.assertRaises(OSError, self.parse)


class TestLegacyFallback(TestNativeSystematicsSummary):
    """With no native summary, the inherited parton_systematics.log parser
    still has to work -- the legacy path is still selectable."""

    def test_the_legacy_log_is_still_read(self):
        """[postprocessing] systematics writes the log and no native block."""

        self.write_info(None)
        log = pjoin(self.me_dir, 'Events', self.run, 'parton_systematics.log')
        with open(log, 'w') as handle:
            handle.write('# original cross-section: 574.45\n'
                         '#     scale variation: +29.6% -21.5%\n'
                         '# PDF variation: + 2% - 2%\n')
        scale, pdf = self.parse()
        self.assertEqual(scale, ['+29.6', '-21.5'])
        self.assertEqual(pdf, ['+2', '-2'])


class TestOneImplementation(unittest.TestCase):
    """The run's Systematics box and the scan column must not drift apart.

    They used to be two copies of the same arithmetic kept in step by a test.
    Now systematics_summary owns it and both callers format what it returns --
    these check that neither has quietly grown its own copy back.
    """

    def summary(self, nominal=500.0):
        return {
            'nominal': {'cross_section': nominal},
            'event_count': 1000,
            'scale': {'min': 0.785 * nominal, 'max': 1.296 * nominal},
            'pdf': [{'pdf_set': 'NNPDF23_lo', 'error_type': 'replicas',
                     'central': 0.9 * nominal,
                     'uncertainty_up': 0.02 * nominal,
                     'uncertainty_down': 0.03 * nominal}],
        }

    def test_the_box_and_the_scan_column_agree(self):
        summary = self.summary()

        # what the scan column reports
        scale = systematics_summary.scale_percentages(summary)
        pdf = systematics_summary.pdf_percentages(summary)
        column = (run_interface.MG7RunCmd._signed(scale),
                  run_interface.MG7RunCmd._signed(pdf[0][1:]))

        # what the box prints, through launch.py's own formatter
        def format_variation(up, down):
            return "%s%%   %s%%" % ('+%.3g' % up, '-%.3g' % down)

        box_scale = format_variation(*scale)
        box_pdf = format_variation(pdf[0][1], pdf[0][2])

        self.assertEqual(box_scale, '%s%%   %s%%' % (column[0][0], column[0][1]))
        self.assertEqual(box_pdf, '%s%%   %s%%' % (column[1][0], column[1][1]))

    def test_neither_caller_does_the_arithmetic_itself(self):
        """A grep-level guard: the divisions used to live in both files."""

        import inspect

        import madgraph.iolibs.template_files.mg7.launch as launch

        box = inspect.getsource(launch.MadgraphProcess.log_systematics_summary)
        column = inspect.getsource(run_interface.MG7RunCmd.getSysSummaryFromLog)
        for name, source in (('the Systematics box', box),
                             ('the scan column', column)):
            self.assertNotIn('* 100', source,
                             '%s computes percentages itself again' % name)
            self.assertIn('systematics_summary', source,
                          '%s no longer uses the shared helper' % name)

    def test_the_helper_needs_nothing_heavy(self):
        """It sits between two modules that must not import each other."""

        import ast

        tree = ast.parse(inspect_source())
        imported = [n for n in ast.walk(tree)
                    if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = []
        for node in imported:
            if isinstance(node, ast.ImportFrom):
                names.append(node.module)
            else:
                names.extend(alias.name for alias in node.names)
        self.assertEqual([n for n in names if n and n != '__future__'], [])


def inspect_source():
    import inspect

    return inspect.getsource(systematics_summary)
