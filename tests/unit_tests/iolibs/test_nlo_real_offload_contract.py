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
import sys
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
            os.path.join(
                MG5DIR, 'madgraph', 'iolibs', 'template_files', 'mg7', 'api.h'),
        ]

    def test_generated_and_public_umami_headers_are_synchronized(self):
        """MadSpace and generated process libraries must use one ABI."""

        generated, public, native = self._umami_headers()
        with open(generated, 'rb') as stream:
            generated_text = stream.read()
        with open(public, 'rb') as stream:
            public_text = stream.read()
        self.assertEqual(generated_text, public_text)
        with open(native, 'rb') as stream:
            self.assertEqual(stream.read(), public_text)

    def test_nlo_full_order_tables_leave_lo_selection_unchanged(self):
        from madmatrix.model_handling import OneProcessExporterMadMatrix
        table = dict(sqsoindex=[[0, 1], [1, 2]],
                     names=['QCD=4', 'QCD=2 QED=2', 'QED=4'],
                     chosen=[False, True, False])
        lo = object.__new__(OneProcessExporterMadMatrix)
        nlo = object.__new__(OneProcessExporterMadMatrix)
        for exporter, full_block in ((lo, False), (nlo, True)):
            exporter.matrix_elements = [object()]
            exporter.nlo_real_order_blocks = full_block
        with patch('madgraph.iolibs.export_v4.split_order_tables', return_value=table):
            self.assertIn('{ false, true, false }', lo.get_sqso_table_lines())
            self.assertIn('{ true, true, true }', nlo.get_sqso_table_lines())
            self.assertEqual(table['chosen'], [False, True, False])
            self.assertIn('{ false, true, false }', lo.get_sqso_table_lines())

    @unittest.skipUnless(shutil.which('g++'), 'requires a C++ compiler')
    def test_fortran_provider_reports_real_extensions_unsupported(self):
        stub = r"""
#pragma once
#include <vector>
struct Parameters {
  double aS = 0.118;
  bool setParameterByName(const char*, double, double) { return false; }
  bool getParameterByName(const char*, double*, double*) { return false; }
};
struct CPPProcess {
  static constexpr int nexternal=2, ndiagrams=1, ncomb=1;
  Parameters params;
  std::vector<double*> momenta;
  std::vector<double> masses{0.,0.};
  explicit CPPProcess(const char*) {}
  std::vector<double*>& getMomenta() { return momenta; }
  const std::vector<double>& getMasses() { return masses; }
  Parameters& getParameters() { return params; }
  void refreshMasses() {}
  double sigmaKin(int) { return 1.; }
  const double* getAmp2() { static const double a[1]={1.}; return a; }
};
"""
        check = r"""
#include "api.h"
int main() {
  int value=0, count=0;
  const bool* supported=nullptr;
  if(umami_get_meta(UMAMI_META_ABI_MAJOR_VERSION,&value)!=UMAMI_SUCCESS || value!=UMAMI_MAJOR_VERSION) return 1;
  if(umami_get_meta(UMAMI_META_ABI_MINOR_VERSION,&value)!=UMAMI_SUCCESS || value!=UMAMI_MINOR_VERSION) return 2;
  if(umami_supported_inputs(&supported,&count)!=UMAMI_SUCCESS || count!=UMAMI_INPUT_KEY_COUNT) return 3;
  if(supported[UMAMI_IN_G_EW] || supported[UMAMI_IN_G_STRONG]) return 4;
  if(umami_supported_outputs(&supported,&count)!=UMAMI_SUCCESS || count!=UMAMI_OUTPUT_KEY_COUNT || supported[UMAMI_OUT_SQUARED_ORDERS]) return 5;
  double p[8]={}, output=0.;
  const void* inputs[]={p}; void* outputs[]={&output};
  UmamiInputKey input=UMAMI_IN_MOMENTA; UmamiOutputKey key=UMAMI_OUT_SQUARED_ORDERS;
  if(umami_matrix_element(nullptr,0,0,0,1,&input,inputs,1,&key,outputs)!=UMAMI_ERROR_UNSUPPORTED_OUTPUT) return 6;
  input=UMAMI_IN_G_EW;
  if(umami_matrix_element(nullptr,0,0,0,1,&input,inputs,0,nullptr,nullptr)!=UMAMI_ERROR_UNSUPPORTED_INPUT) return 7;
  return 0;
}
"""
        with tempfile.TemporaryDirectory(prefix='nlo-fortran-abi-') as work:
            work = Path(work)
            template = Path(MG5DIR) / 'madgraph/iolibs/template_files/mg7'
            (work / 'CPPProcess.h').write_text(stub)
            (work / 'check.cc').write_text(check)
            shutil.copyfile(template / 'api.h', work / 'api.h')
            shutil.copyfile(template / 'api.cpp', work / 'api.cpp')
            subprocess.check_call([shutil.which('g++'), '-std=c++20',
                str(work / 'api.cpp'), str(work / 'check.cc'), '-o', str(work / 'check')])
            subprocess.check_call([str(work / 'check')])

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
                header, r'#define\s+UMAMI_MINOR_VERSION\s+5\b')
            for symbol in required:
                self.assertIn(symbol, header)

    @unittest.skipUnless(shutil.which('gcc'), 'requires a C11 compiler')
    def test_umami_extensions_preserve_upstream_numeric_keys(self):
        """Appended NLO keys must never alias released minor-version3 keys."""
        source = '''#include "umami.h"
_Static_assert(UMAMI_IN_PARTICLE_COUNT==9,"upstream inputs");
_Static_assert(UMAMI_IN_BEAM_INDEX==14,"upstream inputs");
_Static_assert(UMAMI_IN_RANDOM_FLAVOR==15,"released flavour input");
_Static_assert(UMAMI_IN_PARTON_DENSITIES==16,"released PDF input");
_Static_assert(UMAMI_IN_G_STRONG==17,"NLO input");
_Static_assert(UMAMI_OUT_CUT==6,"upstream outputs");
_Static_assert(UMAMI_OUT_MATRIX_ELEMENT_RATIO==12,"upstream outputs");
_Static_assert(UMAMI_OUT_FLAVOR_INDEX==13,"released flavour output");
_Static_assert(UMAMI_OUT_SQUARED_ORDERS==14,"NLO output");
_Static_assert(UMAMI_IN_G_EW==18,"optional EW coupling");
_Static_assert(UMAMI_META_SQUARED_ORDER_POWERS==10,"coupling powers");
_Static_assert(UMAMI_MINOR_VERSION==5,"additive ABI");
'''
        for path in self._umami_headers():
            compiled = subprocess.run(['gcc', '-std=c11', '-x', 'c',
                '-I'+os.path.dirname(path), '-fsyntax-only', '-'],
                input=source.replace('"umami.h"', json.dumps(os.path.basename(path))),
                text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)

    @unittest.skipUnless(shutil.which('g++'), 'requires a C++ compiler')
    def test_backend_capabilities_do_not_alias_flavour_switch_keys(self):
        """Compile the actual capability functions against the merged header."""
        templates = os.path.join(MG5DIR, 'madgraph', 'iolibs', 'template_files',
                          'madmatrix', 'backend')
        header = self._umami_headers()[0]
        for backend in ('cpu', 'simd', 'gpu'):
            with open(os.path.join(templates, backend, 'umami.cc')) as stream:
                source = stream.read()
            start = source.index('  UmamiStatus umami_supported_inputs(')
            end = source.index('  UmamiStatus umami_initialize(', start)
            functions = source[start:end]
            for orders, ew in ((1, True), (3, True), (1, False)):
                with self.subTest(backend=backend, orders=orders, ew=ew), \
                        tempfile.TemporaryDirectory() as directory:
                    probe = '''#include "umami.h"
#include <array>
struct UmamiOrders { static constexpr bool ewResolved = EW; };
struct ProcessData { static constexpr int nsqampso = ORDERS; };
FUNCTIONS
int main() {
  bool const *inputs, *outputs; int ni, no;
  if(umami_supported_inputs(&inputs,&ni)!=UMAMI_SUCCESS ||
     umami_supported_outputs(&outputs,&no)!=UMAMI_SUCCESS) return 1;
  if(ni!=UMAMI_INPUT_KEY_COUNT || no!=UMAMI_OUTPUT_KEY_COUNT) return 2;
  if(inputs[UMAMI_IN_RANDOM_FLAVOR] || inputs[UMAMI_IN_PARTON_DENSITIES] ||
     outputs[UMAMI_OUT_FLAVOR_INDEX]) return 3;
  if(!inputs[UMAMI_IN_G_STRONG]) return 4;
  if(inputs[UMAMI_IN_G_EW] != EXPECT_EW) return 5;
  if(outputs[UMAMI_OUT_SQUARED_ORDERS] != EXPECT_ORDERS) return 6;
  if(outputs[UMAMI_OUT_GPU_STREAM] != GPU) return 7;
}
'''.replace('FUNCTIONS', functions).replace('ORDERS;', str(orders)+';')
                    probe = probe.replace('= EW;', '= '+str(ew).lower()+';')
                    gpu = backend == 'gpu'
                    probe = probe.replace('EXPECT_EW', str(ew and (not gpu or orders == 1)).lower())
                    probe = probe.replace('EXPECT_ORDERS', str(not gpu or orders == 1).lower())
                    probe = probe.replace('!= GPU)', '!= '+str(gpu).lower()+')')
                    path = os.path.join(directory, 'capabilities.cc')
                    with open(path, 'w') as stream:
                        stream.write(probe)
                    executable = os.path.join(directory, 'capabilities')
                    compiled = subprocess.run([
                        'g++', '-std=c++17', '-I'+os.path.dirname(header),
                        path, '-o', executable], capture_output=True, text=True)
                    self.assertEqual(compiled.returncode, 0, compiled.stderr)
                    result = subprocess.run([executable], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

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
                if sys.platform == 'darwin':
                    # dyld returns an already loaded image with the same
                    # install name (@rpath/libnlo_real_bridge.so) instead of
                    # this copy; give each copy its own name and re-sign it.
                    # (no longer than the original name: it must fit the header)
                    subprocess.run(['install_name_tool', '-id',
                                    '@rpath/bridge%d.so' % iteration, saved],
                                   check=True, capture_output=True)
                    subprocess.run(['codesign', '-f', '-s', '-', saved],
                                   check=True, capture_output=True)
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
