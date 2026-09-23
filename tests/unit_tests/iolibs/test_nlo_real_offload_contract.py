################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""Phase-04 contracts for MadMatrix NLO real-amplitude offloading.

The expected failures are the red tests for Phase 04.01.  They intentionally
describe interfaces which do not exist at the Phase-04.00 reference checkpoint.
Remove each ``expectedFailure`` only when its complete contract is implemented.
"""

from __future__ import absolute_import

import json
import math
import os
import re
import unittest

from madgraph import MG5DIR
from madmatrix import output as madmatrix_output


MANIFEST_TOP_LEVEL_KEYS = {
    'version',
    'process_id',
    'nexternal',
    'split_order_names',
    'global_squared_orders',
    'real_matrix_elements',
    'fks_rows',
}

MANIFEST_REAL_ME_KEYS = {
    'id',
    'library_process_id',
    'local_flavours',
    'local_squared_orders',
    'backend_capabilities',
    'process_fingerprint',
}

MANIFEST_FKS_ROW_KEYS = {
    'fks_row',
    'topology',
    'real_me_id',
    'fortran_flavour_index',
    'madmatrix_flavour_index',
    'pdgs',
    'local_to_global',
}


class TestNLORealOffloadContract(unittest.TestCase):
    """Freeze the Python, generated-adapter, and UMAMI extension contracts."""

    def test_phase00_reference_fixtures_are_complete(self):
        """The committed baseline identifies all selected points and rows."""

        input_dir = os.path.join(MG5DIR, 'tests', 'input_files')
        with open(os.path.join(
                input_dir, 'nlo_real_amplitudes_oracle.json')) as stream:
            amplitudes = json.load(stream)
        metadata = amplitudes['metadata']
        self.assertEqual(metadata['phase'], '04.00')
        self.assertEqual(metadata['point_count'], 16)
        self.assertEqual(
            metadata['source_commit'],
            'bd512c29939ca0fd6f92e73453ba965b5745d005')
        for case in amplitudes['cases'].values():
            self.assertEqual(len(case['points']), metadata['point_count'])
            self.assertEqual(
                case['selected_rows'],
                list(range(1, len(case['selected_rows']) + 1)))
            self.assertEqual(
                len(case['records']),
                len(case['points']) * len(case['selected_rows']))
            self.assertEqual(
                len(set((record['point'], record['fks_row'])
                        for record in case['records'])),
                len(case['records']))

        with open(os.path.join(
                input_dir, 'nlo_real_workflow_reference.json')) as stream:
            workflow = json.load(stream)
        self.assertEqual(workflow['metadata']['phase'], '04.00')
        self.assertEqual(workflow['metadata']['source_commit'],
                         metadata['source_commit'])
        self.assertEqual(workflow['fixed_order']['executable'],
                         'madevent_mintFO')
        self.assertEqual(workflow['mint_event_path']['executable'],
                         'madevent_mintMC')
        self.assertEqual(
            workflow['mint_event_path']['input']['vector_size'], 1)
        for key in ('fixed_order', 'mint_event_path'):
            result = workflow[key]['result']
            self.assertTrue(math.isfinite(result['cross_section']))
            self.assertTrue(math.isfinite(result['uncertainty']))
            self.assertNotEqual(result['cross_section'], 0.)

    @staticmethod
    def _umami_headers():
        return [
            os.path.join(
                MG5DIR, 'madgraph', 'iolibs', 'template_files', 'madmatrix',
                'umami.h'),
            os.path.join(
                MG5DIR, 'madspace', 'include', 'madspace', 'umami.h'),
        ]

    def test_generated_and_public_umami_headers_are_synchronized(self):
        """MadSpace and generated process libraries must use one ABI."""

        generated, public = self._umami_headers()
        with open(generated, 'rb') as stream:
            generated_text = stream.read()
        with open(public, 'rb') as stream:
            public_text = stream.read()
        self.assertEqual(generated_text, public_text)

    @unittest.expectedFailure
    def test_lo_and_nlo_real_exporters_are_factory_selected(self):
        """LO and NLO own distinct lifecycles without duplicating selection."""

        registry = madmatrix_output.MADMATRIX_EXPORTER_REGISTRY
        lo_exporter = registry['lo']
        nlo_exporter = registry['nlo_real']
        self.assertIs(lo_exporter,
                      madmatrix_output.ProcessExporterMadMatrix)
        self.assertIs(nlo_exporter,
                      madmatrix_output.ProcessExporterMadMatrixNLOReal)
        self.assertIsNot(lo_exporter, nlo_exporter)
        self.assertIs(
            madmatrix_output.MadMatrixExporterFactory.get_exporter_class('lo'),
            lo_exporter)
        self.assertIs(
            madmatrix_output.MadMatrixExporterFactory.get_exporter_class(
                'nlo_real'),
            nlo_exporter)

    @unittest.expectedFailure
    def test_nlo_real_manifest_schema_is_versioned(self):
        """The generated row/ME/order map has a stable first-version schema."""

        self.assertEqual(madmatrix_output.NLO_REAL_MANIFEST_VERSION, 1)
        self.assertEqual(
            set(madmatrix_output.NLO_REAL_MANIFEST_TOP_LEVEL_KEYS),
            MANIFEST_TOP_LEVEL_KEYS)
        self.assertEqual(
            set(madmatrix_output.NLO_REAL_MANIFEST_REAL_ME_KEYS),
            MANIFEST_REAL_ME_KEYS)
        self.assertEqual(
            set(madmatrix_output.NLO_REAL_MANIFEST_FKS_ROW_KEYS),
            MANIFEST_FKS_ROW_KEYS)

    @unittest.expectedFailure
    def test_umami_declares_direct_g_and_squared_order_output(self):
        """The ABI extension appends all keys required by the NLO adapter."""

        required = (
            'UMAMI_META_ABI_MAJOR_VERSION',
            'UMAMI_META_ABI_MINOR_VERSION',
            'UMAMI_META_PROCESS_FINGERPRINT',
            'UMAMI_META_SQUARED_ORDER_COUNT',
            'UMAMI_IN_G_STRONG',
            'UMAMI_OUT_SQUARED_ORDERS',
        )
        for path in self._umami_headers():
            with open(path) as stream:
                header = stream.read()
            self.assertRegex(
                header, r'#define\s+UMAMI_MINOR_VERSION\s+1\b')
            for symbol in required:
                self.assertIn(symbol, header)

    @unittest.expectedFailure
    def test_generated_nlo_bridge_has_the_frozen_c_abi(self):
        """Fortran calls one exception-safe, dynamically loaded C adapter."""

        path = os.path.join(
            MG5DIR, 'madgraph', 'iolibs', 'template_files', 'madmatrix',
            'nlo_real_bridge.h')
        self.assertTrue(os.path.isfile(path), path)
        with open(path) as stream:
            header = ' '.join(stream.read().split())

        prototypes = (
            r'int\s+mg7_nlo_real_initialize\s*\(\s*void\s*\*\*\s*context\s*,'
            r'\s*const\s+char\s*\*\s*param_card\s*,'
            r'\s*const\s+char\s*\*\s*library_dir\s*,'
            r'\s*const\s+char\s*\*\s*backend\s*\)',
            r'int\s+mg7_nlo_real_evaluate\s*\(\s*void\s*\*\s*context\s*,'
            r'\s*int\s+real_me_id\s*,\s*size_t\s+event_count\s*,'
            r'\s*const\s+double\s*\*\s*momenta\s*,'
            r'\s*const\s+double\s*\*\s*g_strong\s*,'
            r'\s*const\s+int32_t\s*\*\s*flavour\s*,'
            r'\s*double\s*\*\s*squared_orders\s*\)',
            r'int\s+mg7_nlo_real_finalize\s*\(\s*void\s*\*\*\s*context'
            r'\s*\)',
            r'const\s+char\s*\*\s*mg7_nlo_real_last_error\s*\('
            r'\s*void\s*\*\s*context\s*\)',
        )
        for prototype in prototypes:
            self.assertRegex(header, prototype)


if __name__ == '__main__':
    unittest.main()
