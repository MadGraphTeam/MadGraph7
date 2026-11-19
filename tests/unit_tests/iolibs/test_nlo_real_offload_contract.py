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
import ctypes
import math
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from madgraph import MG5DIR
from madgraph.interface import amcatnlo_run_interface
from madgraph.various import banner
from madmatrix import output as madmatrix_output


MANIFEST_TOP_LEVEL_KEYS = {
    'version',
    'library_model',
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

    def test_nlo_real_manifest_schema_is_versioned(self):
        """The generated row/ME/order map has a stable first-version schema."""

        self.assertEqual(madmatrix_output.NLO_REAL_MANIFEST_VERSION, 2)
        self.assertEqual(
            set(madmatrix_output.NLO_REAL_MANIFEST_TOP_LEVEL_KEYS),
            MANIFEST_TOP_LEVEL_KEYS)
        self.assertEqual(
            set(madmatrix_output.NLO_REAL_MANIFEST_REAL_ME_KEYS),
            MANIFEST_REAL_ME_KEYS)
        self.assertEqual(
            set(madmatrix_output.NLO_REAL_MANIFEST_FKS_ROW_KEYS),
            MANIFEST_FKS_ROW_KEYS)

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
                header, r'#define\s+UMAMI_MINOR_VERSION\s+4\b')
            for symbol in required:
                self.assertIn(symbol, header)

    @unittest.skipUnless(shutil.which('gcc'), 'requires a C11 compiler')
    def test_umami_extensions_preserve_upstream_numeric_keys(self):
        """Appended NLO keys must never alias upstream minor-version2 keys."""
        source = '''#include "umami.h"
_Static_assert(UMAMI_IN_PARTICLE_COUNT==9,"upstream inputs");
_Static_assert(UMAMI_IN_BEAM_INDEX==14,"upstream inputs");
_Static_assert(UMAMI_IN_G_STRONG==15,"NLO input");
_Static_assert(UMAMI_OUT_CUT==6,"upstream outputs");
_Static_assert(UMAMI_OUT_MATRIX_ELEMENT_RATIO==12,"upstream outputs");
_Static_assert(UMAMI_OUT_SQUARED_ORDERS==13,"NLO output");
_Static_assert(UMAMI_IN_G_EW==16,"optional EW coupling");
_Static_assert(UMAMI_META_SQUARED_ORDER_POWERS==10,"coupling powers");
_Static_assert(UMAMI_MINOR_VERSION==4,"additive ABI");
'''
        for path in self._umami_headers():
            compiled = subprocess.run(['gcc', '-std=c11', '-x', 'c',
                '-I'+os.path.dirname(path), '-fsyntax-only', '-'],
                input=source, text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)

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

    @unittest.skipUnless(shutil.which('g++') and shutil.which('make'),
                         'requires C++ and make')
    def test_bridge_rebuild_binds_model_precision_and_library_tag(self):
        """Changing build identity must load the newly selected actual DSO."""
        exporter = object.__new__(madmatrix_output.ProcessExporterMadMatrixNLOReal)
        manifest = {'library_model': 'fixture_plus_model', 'nexternal': 4,
                    'split_order_names': ['QCD', 'QED'],
                    'real_matrix_elements': [{
                        'library_process_id': 'PMM_B1_R1',
                        'process_fingerprint': 'fixture',
                        'local_squared_orders': [[4, 2]],
                        'local_flavours': [[1, -1, 21, 21]],
                        'backend_capabilities': {'cuda': True}}]}
        with tempfile.TemporaryDirectory() as root:
            directory = os.path.join(root, 'SubProcesses', 'P0_fixture')
            os.makedirs(directory)
            exporter._write_bridge_config(manifest, directory)
            exporter._write_nlo_build_fragment(manifest, directory)
            process = os.path.join(root, 'SubProcesses', 'PMM_B1_R1')
            os.makedirs(process)
            with open(os.path.join(process, 'makefile'), 'w') as stream:
                stream.write('.PHONY: all\nall:\n\t@:\n')
            source = os.path.join(directory, 'fixture.cc')
            with open(source, 'w') as stream:
                stream.write('''#include "umami.h"
extern "C" {
UmamiStatus umami_get_meta(UmamiMetaKey key, void* value) {
  switch(key) {
    case UMAMI_META_ABI_MAJOR_VERSION: *static_cast<int*>(value)=UMAMI_MAJOR_VERSION; break;
    case UMAMI_META_ABI_MINOR_VERSION: *static_cast<int*>(value)=UMAMI_MINOR_VERSION; break;
    case UMAMI_META_PROCESS_FINGERPRINT: *static_cast<const char**>(value)="fixture"; break;
    case UMAMI_META_COUPLING_NAMES: *static_cast<const char**>(value)="QCD,QED"; break;
    case UMAMI_META_SQUARED_ORDER_POWERS:
      static_cast<int*>(value)[0]=4; static_cast<int*>(value)[1]=2; break;
    case UMAMI_META_PARTICLE_COUNT: *static_cast<int*>(value)=4; break;
    case UMAMI_META_SQUARED_ORDER_COUNT: *static_cast<int*>(value)=1; break;
    default: return UMAMI_ERROR_UNSUPPORTED_META;
  }
  return UMAMI_SUCCESS;
}
UmamiStatus umami_initialize(UmamiHandle* handle, const char*) {
  *handle=reinterpret_cast<void*>(1); return UMAMI_SUCCESS;
}
UmamiStatus umami_free(UmamiHandle) { return UMAMI_SUCCESS; }
UmamiStatus umami_matrix_element(UmamiHandle, size_t, size_t, size_t,
  size_t n_inputs, const UmamiInputKey* keys, const void* const* inputs, size_t,
  const UmamiOutputKey*, void* const* outputs) {
  if (n_inputs==4 && (keys[1]!=UMAMI_IN_G_STRONG || keys[3]!=UMAMI_IN_G_EW ||
      *static_cast<const double*>(inputs[1])!=1. || *static_cast<const double*>(inputs[3])!=1.))
    return UMAMI_ERROR;
  *static_cast<double*>(outputs[0])=FIXTURE_VALUE; return UMAMI_SUCCESS;
}
}
''')
            for iteration, (fptype, tag, value) in enumerate((
                    ('d', '_a', 1), ('m', '_b', 2), ('d', '_a', 1))):
                library = os.path.join(directory,
                    'libmadmatrix%s_fixture_plus_model_PMM_B1_R1_scalar_%s.so' %
                    (tag, fptype))
                subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', '-I.',
                                '-DFIXTURE_VALUE=%d' % value, source, '-o', library],
                               cwd=directory, check=True, capture_output=True)
                command = ['make', '-f', 'nlo_real_offload.mk',
                           '.nlo_real_scalar_%s%s_cpp.stamp' % (fptype, tag),
                           'CXX=g++', 'NLO_REAL_FPTYPE='+fptype,
                           'NLO_REAL_LIBTAG='+tag]
                built = subprocess.run(command, cwd=directory, text=True,
                                       capture_output=True)
                self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
                bridge_path = os.path.join(directory, 'libnlo_real_bridge.so')
                saved = os.path.join(directory, 'bridge%d.so' % iteration)
                shutil.copy2(bridge_path, saved)
                bridge = ctypes.CDLL(saved)
                bridge.mg7_nlo_real_initialize.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                    ctypes.c_char_p, ctypes.c_char_p]
                bridge.mg7_nlo_real_evaluate.argtypes = [
                    ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double)]
                bridge.mg7_nlo_real_finalize.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
                context = ctypes.c_void_p()
                self.assertEqual(bridge.mg7_nlo_real_initialize(
                    ctypes.byref(context), b'unused', directory.encode(), b'scalar'), 0)
                try:
                    momenta = (ctypes.c_double * 16)()
                    coupling = (ctypes.c_double * 1)(1.)
                    flavour = (ctypes.c_int32 * 1)(0)
                    result = (ctypes.c_double * 1)()
                    self.assertEqual(bridge.mg7_nlo_real_evaluate(
                        context, 1, 1, momenta, coupling, flavour, result), 0)
                    self.assertEqual(result[0], value)
                    bridge.mg7_nlo_real_evaluate_ew.argtypes = [
                        ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
                        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                        ctypes.c_double, ctypes.POINTER(ctypes.c_int32),
                        ctypes.POINTER(ctypes.c_double)]
                    coupling[0] = 2.
                    self.assertEqual(bridge.mg7_nlo_real_evaluate_ew(
                        context, 1, 1, momenta, coupling, 3., flavour, result), 0)
                    self.assertEqual(result[0], value * 2.**4 * 3.**2)
                finally:
                    self.assertEqual(bridge.mg7_nlo_real_finalize(ctypes.byref(context)), 0)
                previous = os.stat(bridge_path).st_mtime_ns
                subprocess.run(command, cwd=directory, check=True,
                               capture_output=True)
                self.assertEqual(os.stat(bridge_path).st_mtime_ns, previous)

    def test_run_card_selects_nlo_real_backend(self):
        """The real-ME backend is a run-card setting (design section 4.1)
        with the safe Fortran default; unknown backends are rejected."""
        run_card = banner.RunCardNLO()
        self.assertEqual(run_card['nlo_real_backend'], 'fortran')
        for backend in ('scalar', 'avx512y', 'cuda'):
            run_card['nlo_real_backend'] = backend
            self.assertEqual(run_card['nlo_real_backend'], backend)
        with self.assertRaises(Exception):
            run_card.set('nlo_real_backend', 'no_such_backend',
                         raiseerror=True)

    def test_nlo_real_backend_has_no_runtime_override(self):
        """The generated wrapper takes the backend only from the compiled
        run-card value, never from the environment of the job."""

        path = os.path.join(MG5DIR, 'madmatrix', 'output.py')
        with open(path) as stream:
            source = stream.read()
        self.assertNotIn('MG7_NLO_REAL_BACKEND', source)
        self.assertIn("include 'nlo_real_backend.inc'", source)
        self.assertIn('backend = default_backend', source)

    def test_fks_standalone_launch_uses_run_card_backend(self):
        """'launch' of an 'output standalone_fortran --fks' made with
        --me_exporter=mg7 compiles the run-card backend, like aMC@NLO."""

        from madgraph.interface import launch_ext_program

        root = tempfile.mkdtemp(prefix='nlo-fkssa-backend-')
        try:
            for path in ('Cards', 'Source', 'SubProcesses/P0_test'):
                os.makedirs(os.path.join(root, path))
            make_opts = os.path.join(root, 'Source', 'make_opts')
            with open(make_opts, 'w') as stream:
                stream.write('DEFAULT_F_COMPILER=gfortran\n'
                             '#end_of_make_opts_variables\n\n'
                             'FC=$(DEFAULT_F_COMPILER)\nlibext=a\n')
            launcher = launch_ext_program.FKSSALauncher

            # Ordinary FKS standalone output: nothing is written.
            launcher._set_nlo_real_backend(root)
            with open(make_opts) as stream:
                self.assertNotIn('NLO_REAL_RUNTIME_BACKEND', stream.read())

            open(os.path.join(root, 'SubProcesses', 'P0_test',
                              'nlo_real_offload.mk'), 'w').close()
            run_card = banner.RunCardNLO()
            run_card.set('nlo_real_backend', 'avx512y', user=True)
            run_card.write(os.path.join(root, 'Cards', 'run_card.dat'))
            launcher._set_nlo_real_backend(root)
            with open(make_opts) as stream:
                content = stream.read()
            self.assertIn('NLO_REAL_RUNTIME_BACKEND=avx512y', content)
            self.assertIn('FC=$(DEFAULT_F_COMPILER)', content)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_nlo_event_jobs_honor_run_card_vector_size(self):
        """The run layer must not force event-generation jobs to one lane."""

        root = tempfile.mkdtemp(prefix='nlo-vector-size-')
        try:
            channels_path = os.path.join(
                root, 'SubProcesses', 'P0_test', 'channels.txt')
            os.makedirs(os.path.dirname(channels_path))
            with open(channels_path, 'w') as stream:
                stream.write('1\n')
            command = amcatnlo_run_interface.aMCatNLOCmd.__new__(
                amcatnlo_run_interface.aMCatNLOCmd)
            command.me_dir = root
            command.stop_for_runweb = True
            command.force_run = False
            command.run_card = banner.RunCardNLO()
            command.run_card['npoints_FO_grid'] = 8
            command.run_card['niters_FO_grid'] = 1
            command.run_card['vector_size'] = 7
            jobs, _, _ = command.create_jobs_to_run(
                {'only_generation': False}, ['P0_test'], -1, 'all', 1,
                'aMC@NLO', fixed_order=False)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]['vecsize'], 7)
        finally:
            shutil.rmtree(root)

    def test_native_checkpoint_detects_changed_grids_and_preserves_legacy_scalar(self):
        error = amcatnlo_run_interface.aMCatNLOError
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdir = root / 'SubProcesses/P0'
            grid = pdir / 'GF1.0'
            grid.mkdir(parents=True)
            cards = root / 'Cards'
            cards.mkdir()
            for p in [pdir/'madevent_mintMC', grid/'mint_grids', grid/'grid.MC_integer',
                      cards/'param_card.dat', cards/'MadLoopParams.dat', cards/'FKS_params.dat']:
                p.write_text(p.name)
            seed = root / 'SubProcesses/randinit'
            seed.write_text('r=42')
            command = amcatnlo_run_interface.aMCatNLOCmd.__new__(amcatnlo_run_interface.aMCatNLOCmd)
            command.me_dir = str(root)
            command.stop_for_runweb = True
            command.force_run = False
            command.run_card = banner.RunCardNLO()
            job = dict(p_dir='P0', dirname=str(grid), channel='1.0', vecsize=1)
            command.bind_native_mint_checkpoint([job], verify=True)
            self.assertNotIn('native_mc_checkpoint', job)
            command.bind_native_mint_checkpoint([job])
            command.bind_native_mint_checkpoint([job], verify=True)
            self.assertEqual(job.pop('native_mc_checkpoint')['binding']['integration_width'], 1)
            command.run_card['vector_size'] = 3
            job['vecsize'] = 3
            with self.assertRaisesRegex(error, 'lack a compatible'):
                command.bind_native_mint_checkpoint([job], verify=True)
            command.bind_native_mint_checkpoint([job])
            seed.write_text('r=43')
            command.run_card['nevents'] = 200
            command.bind_native_mint_checkpoint([job], verify=True)
            self.assertEqual(job['native_mc_checkpoint']['integration_seed'], 42)
            cut = command.run_card['ptj']
            command.run_card['ptj'] = cut + 1
            with self.assertRaisesRegex(error, 'checkpoint differs'):
                command.bind_native_mint_checkpoint([job], verify=True)
            command.run_card['ptj'] = cut
            command.run_card['nlo_gen_vector_size'] = 2
            with self.assertRaisesRegex(error, 'checkpoint differs'):
                command.bind_native_mint_checkpoint([job], verify=True)
            command.run_card['nlo_gen_vector_size'] = 3
            (grid/'mint_grids').write_text('different grid')
            with self.assertRaisesRegex(error, 'checkpoint differs'):
                command.bind_native_mint_checkpoint([job], verify=True)

            (pdir/'nlo_real_backend.inc').write_text('scalar')
            command.run_card['nlo_real_backend'] = 'scalar'
            custom = grid/'custom'
            custom.mkdir()
            library = custom/'libmadmatrix_sm_PMM_B1_R1_scalar_d.so'
            library.write_bytes(b'A')
            (custom/'my_card.dat').write_text('first model')
            with patch.dict(os.environ, {'MG7_NLO_REAL_LIBRARY_DIR':'custom',
                                         'MG7_NLO_REAL_PARAM_CARD':'custom/my_card.dat'}):
                command.bind_native_mint_checkpoint([job])
                command.bind_native_mint_checkpoint([job], verify=True)
                library.write_bytes(b'B')
                with self.assertRaisesRegex(error, 'checkpoint differs'):
                    command.bind_native_mint_checkpoint([job], verify=True)
                library.write_bytes(b'A')
                (custom/'my_card.dat').write_text('different model')
                with self.assertRaisesRegex(error, 'checkpoint differs'):
                    command.bind_native_mint_checkpoint([job], verify=True)

    def test_nlo_generation_width_keeps_integration_batch_and_gpu_default(self):
        """Only MC generation uses the capped CPU width in native input."""

        root = tempfile.mkdtemp(prefix='nlo-generation-width-')
        try:
            command = amcatnlo_run_interface.aMCatNLOCmd.__new__(
                amcatnlo_run_interface.aMCatNLOCmd)
            command.run_card = banner.RunCardNLO()
            command.stop_for_runweb = True
            command.force_run = False
            command.cross_sect_dict = {
                'nevents': 1, 'xseca': 1., 'erra': .1,
                'xsect': 1., 'errt': .1, 'p_labels': ['P0'],
                'P0': {'p_label': 'P0', 'xseca': 1., 'erra': .1,
                       'xsect': 1., 'errt': .1},
            }
            job = {'dirname': root, 'mint_mode': 1, 'accuracy': .1,
                   'channel': '1.0', 'run_mode': 'all', 'vecsize': 256,
                   'p_label': 'P0', 'nevents': 1}

            def native_width():
                with open(os.path.join(root, 'input_app.txt')) as stream:
                    return int(next(line for line in stream
                                    if '! vector size' in line).split()[0])

            command.write_input_file(job, fixed_order=False)
            self.assertEqual(native_width(), 256)
            job['mint_mode'] = 0
            command.write_input_file(job, fixed_order=False)
            self.assertEqual(native_width(), 256)
            job['mint_mode'] = 2
            command.write_input_file(job, fixed_order=False)
            self.assertEqual(native_width(), 8)
            self.assertEqual(job['vecsize'], 256)

            command.run_card['nlo_real_backend'] = 'cuda'
            command.write_input_file(job, fixed_order=False)
            self.assertEqual(native_width(), 256)
            command.run_card['nlo_real_backend'] = 'hip'
            command.write_input_file(job, fixed_order=False)
            self.assertEqual(native_width(), 256)

            command.run_card.set('nlo_gen_vector_size', 4, user=True)
            command.write_input_file(job, fixed_order=False)
            self.assertEqual(native_width(), 4)
            card_path = os.path.join(root, 'run_card.dat')
            command.run_card.write(card_path)
            self.assertEqual(
                banner.RunCardNLO(card_path)['nlo_gen_vector_size'], 4)

            command.run_card.set('nlo_gen_vector_size', -1, user=True)
            with self.assertRaisesRegex(
                    amcatnlo_run_interface.aMCatNLOError,
                    'nlo_gen_vector_size'):
                command.write_input_file(job, fixed_order=False)
            with self.assertRaisesRegex(
                    amcatnlo_run_interface.aMCatNLOError,
                    'nlo_gen_vector_size'):
                command.create_jobs_to_run(
                    {'only_generation': False}, [], -1, 'all', 1,
                    'aMC@NLO', fixed_order=False)
        finally:
            shutil.rmtree(root)


if __name__ == '__main__':
    unittest.main()
