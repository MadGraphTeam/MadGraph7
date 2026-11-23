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
"""The MadMatrix common library of a --me_exporter=mg7 NLO output is built
once, before the P* directories are compiled in parallel.

Every P* directory links the same library from src/. When each one rebuilt it,
parallel compilations raced on the same objects and the link failed with
'HelRecyclingScratch.o: file not recognized'.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from madgraph.interface import amcatnlo_run_interface
from madmatrix import output as madmatrix_output


def real(process_id, cuda):
    return {'library_process_id': process_id,
            'process_fingerprint': 'fixture',
            'local_squared_orders': [[4, 2]],
            'local_flavours': [[1, -1, 21, 21]],
            'backend_capabilities': {'cuda': cuda}}


# Records how each MadMatrix process directory was asked to build.
PROCESS_MAKEFILE = '''.PHONY: all commonlib
all:
\t@echo "lib $(BACKEND) $(MADMATRIX_COMMONLIB_EXTERNAL)" >> ../make.log
commonlib:
\t@echo "commonlib $(notdir $(CURDIR)) $(BACKEND) $(FPTYPE)" >> ../make.log
'''


class TestNLORealCommonlibOnce(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.subprocesses = os.path.join(self.root, 'SubProcesses')
        exporter = object.__new__(madmatrix_output.ProcessExporterMadMatrixNLOReal)
        # P0_a has no GPU-capable real emission; P0_b has one.
        for p_dir, reals in (('P0_a', [real('PMM_B1_R1', False)]),
                             ('P0_b', [real('PMM_B2_R1', True),
                                       real('PMM_B2_R2', True)])):
            manifest = {'library_model': 'fixture_plus_model', 'nexternal': 4,
                        'split_order_names': ['QCD', 'QED'],
                        'real_matrix_elements': reals}
            directory = os.path.join(self.subprocesses, p_dir)
            os.makedirs(directory)
            exporter._write_bridge_config(manifest, directory)
            exporter._write_nlo_build_fragment(manifest, directory)
            for entry in reals:
                process = os.path.join(self.subprocesses,
                                       entry['library_process_id'])
                os.makedirs(process)
                with open(os.path.join(process, 'makefile'), 'w') as stream:
                    stream.write(PROCESS_MAKEFILE)
        os.makedirs(os.path.join(self.subprocesses, 'P0_fortran'))

    def tearDown(self):
        shutil.rmtree(self.root)

    def make_log(self):
        path = os.path.join(self.subprocesses, 'make.log')
        if not os.path.exists(path):
            return []
        with open(path) as stream:
            return stream.read().split('\n')[:-1]

    def test_commonlib_directory_follows_the_backend(self):
        p_dirs = ['P0_fortran', 'P0_a', 'P0_b']
        find = amcatnlo_run_interface.nlo_real_commonlib_dir
        self.assertEqual(find(self.subprocesses, p_dirs, 'fortran'), 'P0_a')
        self.assertEqual(find(self.subprocesses, p_dirs, 'avx512y'), 'P0_a')
        self.assertEqual(find(self.subprocesses, p_dirs, 'cuda'), 'P0_b')
        self.assertIsNone(find(self.subprocesses, ['P0_fortran'], 'avx512y'))

    def test_generated_targets_build_commonlib_once_then_skip_it(self):
        directory = os.path.join(self.subprocesses, 'P0_b')
        make = ['make', '--no-print-directory', '-f', 'nlo_real_offload.mk',
                'NLO_REAL_RUNTIME_BACKEND=avx512y']
        built = subprocess.run(make + ['nlo_real_commonlib'], cwd=directory,
                               capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        # Built through one real-emission directory only.
        self.assertEqual(self.make_log(), ['commonlib PMM_B2_R1 avx512y d'])
        # The flag set on the P* make reaches every process directory.
        built = subprocess.run(
            make + ['MADMATRIX_COMMONLIB_EXTERNAL=1', 'CXX=g++',
                    '.nlo_real_avx512y_d_cpp.stamp'],
            cwd=directory, capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        self.assertEqual(self.make_log()[1:], ['lib avx512y 1', 'lib avx512y 1'])

    def test_parallel_directory_builds_leave_commonlib_to_the_interface(self):
        me_dir = self.root
        with open(os.path.join(me_dir, 'test_ME_input.txt'), 'w'):
            pass
        calls = []

        def compile(arg, cwd=None, **opt):
            calls.append((os.path.basename(cwd), list(arg)))

        with patch.object(amcatnlo_run_interface.misc, 'compile', compile), \
             patch.object(amcatnlo_run_interface.misc, 'call', return_value=0):
            for p_dir in ('P0_a', 'P0_fortran'):
                amcatnlo_run_interface.compile_dir(
                    me_dir, p_dir, 'aMC@NLO', {'reweightonly': False},
                    ['test_ME'], 'madevent_mintMC', 2)
        targets = {'test_soft_col_limits', 'gensym', 'madevent_mintMC',
                   'reweight_xsec_events'}
        for p_dir, external in (('P0_a', ['MADMATRIX_COMMONLIB_EXTERNAL=1']),
                                ('P0_fortran', [])):
            made = [arg for directory, arg in calls if directory == p_dir]
            self.assertEqual({arg[0] for arg in made}, targets)
            for arg in made:
                self.assertEqual(arg[1:], external, arg)


if __name__ == '__main__':
    unittest.main()
