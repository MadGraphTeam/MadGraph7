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
"""Fortran reference oracle for NLO real-amplitude offloading.

This is the executable Phase-04.00 reference.  It generates ordinary grouped
FKS output, compiles the unmodified scalar Fortran real matrix elements and
records the complete contract that a later MadMatrix adapter must reproduce:
physical FKS row, distinct real ME, local flavour, direct G, momenta, raw local
squared orders, mapped global split orders and summed result.
"""

from __future__ import absolute_import

import glob
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import unittest

from madgraph import MG5DIR
import madgraph.interface.master_interface as MGCmd
import madgraph.various.misc as misc
from madgraph.various import banner


pjoin = os.path.join
ORACLE_PATH = pjoin(
    MG5DIR, 'tests', 'input_files', 'nlo_real_amplitudes_oracle.json')


class TestNLORealFortranOracle(unittest.TestCase):
    """Generate and replay scalar Fortran real-ME reference records."""

    maxDiff = None

    @classmethod
    def setUpClass(cls):
        with open(ORACLE_PATH) as stream:
            cls.oracle = json.load(stream)
        metadata = cls.oracle['metadata']
        if metadata['phase'] != '04.00':
            raise AssertionError('unexpected oracle phase')
        if metadata['source_commit'] != (
                'bd512c29939ca0fd6f92e73453ba965b5745d005'):
            raise AssertionError('unexpected oracle source commit')
        if metadata['point_count'] != 16:
            raise AssertionError('the real-ME oracle is not many-point')
        for case in cls.oracle['cases'].values():
            if len(case['points']) != metadata['point_count']:
                raise AssertionError('incomplete point set')

    def setUp(self):
        fast_root = None
        if os.path.isdir('/scratch') and os.access('/scratch', os.W_OK):
            fast_root = '/scratch'
        self.tmpdir = tempfile.mkdtemp(prefix='m4-real-oracle-',
                                       dir=fast_root)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _new_cmd():
        cmd = MGCmd.MasterCmd()
        cmd.no_notification()
        cmd.run_cmd('set automatic_html_opening False --no_save')
        return cmd

    @staticmethod
    def _run(cmd, line):
        cmd.exec_cmd(line, errorhandling=False, printcmd=False,
                     precmd=True, postcmd=True)

    def _generate_case(self, name, madmatrix=False, low_memory=False,
                       vector_size=None):
        case = self.oracle['cases'][name]
        output_path = pjoin(self.tmpdir, name)
        cmd = self._new_cmd()
        self._run(cmd, 'set apply_flavor_grouping True --no_save')
        self._run(
            cmd, 'set low_mem_multicore_nlo_generation %s --no_save' %
            ('True' if low_memory else 'False'))
        if low_memory:
            self._run(cmd, 'set nb_core 2 --no_save')
        if name == 'grouped_mixed_wj':
            self._run(cmd, 'set nlo_mixed_expansion True --no_save')

        model = case['model']
        if model.startswith('tests/'):
            model = pjoin(MG5DIR, model)
        self._run(cmd, 'import model %s' % model)
        self._run(cmd, 'generate %s' % case['process'])
        output = 'output standalone_fortran --fks --limits %s -f' % \
            output_path
        if madmatrix:
            output += ' --me_exporter=mg7'
        if vector_size is not None:
            output += ' --vector_size=%d' % vector_size
        self._run(cmd, output)
        if vector_size is not None:
            run_card = banner.RunCardNLO(
                pjoin(output_path, 'Cards', 'run_card.dat'))
            self.assertEqual(run_card['vector_size'], vector_size)
        if madmatrix:
            # The backend is a run-card setting compiled into the wrapper.
            run_card = banner.RunCardNLO(
                pjoin(output_path, 'Cards', 'run_card.dat'))
            self.assertIn('nlo_real_backend', run_card.user_set)
            self.assertEqual(run_card['nlo_real_backend'], 'fortran')
            subprocess_path = pjoin(output_path, 'SubProcesses',
                                    case['subprocess'])
            with open(pjoin(subprocess_path, 'nlo_real_offload.mk')) as stream:
                self.assertIn('NLO_REAL_RUNTIME_BACKEND ?= fortran',
                              stream.read())
            with open(pjoin(subprocess_path,
                            'nlo_real_offload.f90')) as stream:
                self.assertIn("include 'nlo_real_backend.inc'", stream.read())
        param_card = pjoin(output_path, 'Cards', 'param_card.dat')
        with open(param_card, 'rb') as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        self.assertEqual(digest, case['param_card']['sha256'])
        return output_path

    def _real_me_contract(self, subproc):
        """Read the authoritative FKS-row -> real-ME and local-size maps."""

        chooser_path = pjoin(subproc, 'real_me_chooser.f')
        with open(chooser_path) as stream:
            chooser = stream.read()
        scalar = chooser.split(
            'RECURSIVE SUBROUTINE SMATRIX_REAL_VEC', 1)[0]
        pairs = re.findall(
            r'NFKSPROCESS\.EQ\.(\d+).*?SMATRIX(\d+)_AMP\(',
            scalar, re.I | re.S)
        row_to_me = {int(row): int(real_me) for row, real_me in pairs}
        self.assertTrue(row_to_me, chooser_path)
        self.assertEqual(sorted(row_to_me),
                         list(range(1, max(row_to_me) + 1)))

        local_counts = {}
        for real_me in sorted(set(row_to_me.values())):
            matrix_path = pjoin(subproc, 'matrix_%d.f' % real_me)
            with open(matrix_path) as stream:
                matrix = stream.read()
            match = re.search(
                r'PARAMETER\s*\(\s*NSQAMPSO\s*=\s*(\d+)\s*\)',
                matrix, re.I)
            self.assertIsNotNone(match, matrix_path)
            local_counts[real_me] = int(match.group(1))
        return row_to_me, local_counts

    def _driver_source(self, row_to_me, local_counts, vector=False):
        """Build a fixed-form driver without modifying production wrappers."""

        real_mes = sorted(local_counts)
        lines = [
            '      PROGRAM CHECK_REAL_ORACLE',
            '      IMPLICIT NONE',
            "      INCLUDE 'nexternal.inc'",
            "      INCLUDE 'orders.inc'",
            "      INCLUDE 'nFKSconfigs.inc'",
            "      INCLUDE 'coupl.inc'",
            '      INTEGER I,J,K,NPOINTS,NME,NLOCAL,NFKSPROCESS',
            '      DOUBLE PRECISION P(0:3,NEXTERNAL),GIN,WGT',
            '      DOUBLE PRECISION LOCAL(0:AMP_SPLIT_SIZE)',
            '      DOUBLE PRECISION GLOBAL(AMP_SPLIT_SIZE)',
            '      COMMON/C_NFKSPROCESS/NFKSPROCESS',
            "      INCLUDE 'fks_info.inc'",
            "      INCLUDE 'amp_split_orders.inc'",
        ]
        for real_me in real_mes:
            lines.append(
                '      INTEGER GETORDPOWFROMINDEX%d' % real_me)
        lines.extend([
            "      CALL SETPARA('param_card.dat')",
            '      READ(*,*) NPOINTS',
            '      DO K=1,NPOINTS',
            '        READ(*,*) GIN',
            '        DO J=1,NEXTERNAL',
            '          READ(*,*) (P(I,J),I=0,3)',
            '        ENDDO',
            '        G=GIN',
            '        CALL UPDATE_AS_PARAM()',
            '        DO NFKSPROCESS=1,FKS_CONFIGS',
            '          NME=0',
            '          NLOCAL=0',
        ])
        for row, real_me in sorted(row_to_me.items()):
            lines.append(
                '          IF (NFKSPROCESS.EQ.%d) NME=%d' %
                (row, real_me))
        for real_me, count in sorted(local_counts.items()):
            lines.append(
                '          IF (NME.EQ.%d) NLOCAL=%d' %
                (real_me, count))
        real_call = ['          CALL SMATRIX_REAL(P,GLOBAL,WGT)']
        if vector:
            real_call = [
                '          CALL SMATRIX_REAL_VEC(P,GLOBAL,WGT,1,',
                '     $      NFKSPROCESS,',
                '     $      REAL_FLAVOR_INDEX_D(NFKSPROCESS))']
        lines.extend([
            '          GLOBAL(:)=0D0',
            '          LOCAL(:)=0D0',
        ] + real_call)
        for real_me in real_mes:
            lines.extend([
                '          IF (NME.EQ.%d) THEN' % real_me,
                '            CALL SMATRIX%d_SPLITORDERS(P,LOCAL)' %
                real_me,
                '          ENDIF',
            ])
        lines.extend([
            "          WRITE(*,*) 'ORACLE_BEGIN',K,NFKSPROCESS,NME,",
            '     $      REAL_FLAVOR_INDEX_D(NFKSPROCESS),NLOCAL,',
            '     $      FKS_TOPOLOGY_D(NFKSPROCESS)',
            "          WRITE(*,*) 'ORACLE_G',G",
            "          WRITE(*,*) 'ORACLE_PDGS',",
            '     $      (PDG_TYPE_D(NFKSPROCESS,J),J=1,NEXTERNAL)',
            '          DO J=1,NEXTERNAL',
            "            WRITE(*,*) 'ORACLE_MOMENTUM',J,",
            '     $        (P(I,J),I=0,3)',
            '          ENDDO',
            "          WRITE(*,*) 'ORACLE_LOCAL_SUM',LOCAL(0)",
        ])
        for real_me in real_mes:
            lines.extend([
                '          IF (NME.EQ.%d) THEN' % real_me,
                '            DO J=1,%d' % local_counts[real_me],
                "              WRITE(*,*) 'ORACLE_LOCAL',J,",
                '     $          (GETORDPOWFROMINDEX%d(I,J),' %
                real_me,
                '     $          I=1,NSPLITORDERS),LOCAL(J)',
                '            ENDDO',
                '          ENDIF',
            ])
        lines.extend([
            '          DO J=1,AMP_SPLIT_SIZE',
            "            WRITE(*,*) 'ORACLE_GLOBAL',J,",
            '     $        (AMP_SPLIT_ORDERS(J,I),I=1,NSPLITORDERS),',
            '     $        GLOBAL(J)',
            '          ENDDO',
            "          WRITE(*,*) 'ORACLE_SUM',WGT",
            "          WRITE(*,*) 'ORACLE_END'",
            '        ENDDO',
            '      ENDDO',
            '      END',
        ])
        for line in lines:
            self.assertLessEqual(len(line), 72, line)
        return '\n'.join(lines) + '\n'

    @staticmethod
    def _driver_input(points):
        lines = [str(len(points))]
        for point in points:
            lines.append('%.17g' % point['g_strong'])
            lines.extend(
                ' '.join('%.17g' % component for component in momentum)
                for momentum in point['momenta'])
        return ('\n'.join(lines) + '\n').encode()

    def _compile_and_run(self, output_path, case):
        subproc = pjoin(output_path, 'SubProcesses', case['subprocess'])
        row_to_me, local_counts = self._real_me_contract(subproc)
        selected_rows = case['selected_rows']
        self.assertEqual(selected_rows, sorted(row_to_me))

        driver = self._driver_source(row_to_me, local_counts)
        driver_path = pjoin(subproc, 'check_real_oracle.f')
        with open(driver_path, 'w') as stream:
            stream.write(driver)

        misc.compile(cwd=pjoin(output_path, 'Source'))
        matrix_objects = [
            os.path.basename(path)[:-2] + '.o' for path in
            sorted(glob.glob(pjoin(subproc, 'matrix_*.f')))]
        self.assertEqual(
            sorted(int(re.search(r'(\d+)', obj).group(1))
                   for obj in matrix_objects),
            sorted(local_counts))
        objects = matrix_objects + [
            'real_me_chooser.o', 'splitorders_stuff.o',
            'check_real_oracle.o']
        misc.compile(objects, cwd=subproc)

        executable = pjoin(subproc, 'check_real_oracle')
        subprocess.check_call(
            ['gfortran', '-o', executable] + objects +
            ['-L%s' % pjoin(output_path, 'lib'), '-ldhelas', '-lmodel'],
            cwd=subproc)
        output = subprocess.check_output(
            [executable], input=self._driver_input(case['points']),
            cwd=subproc, stderr=subprocess.STDOUT)
        return self._parse_driver_output(
            output.decode(errors='replace'),
            len(case['split_order_names']))

    def _compile_routed_driver(self, output_path, case, vector=False,
                               backend=None):
        """Build a frozen-oracle driver through the production wrapper."""
        subproc = pjoin(output_path, 'SubProcesses', case['subprocess'])
        row_to_me, local_counts = self._real_me_contract(subproc)
        suffix = '_vec' if vector else ''
        source_name = 'check_real_oracle%s.f' % suffix
        object_name = 'check_real_oracle%s.o' % suffix
        executable = pjoin(subproc, 'check_real_oracle%s' % suffix)
        with open(pjoin(subproc, source_name), 'w') as stream:
            stream.write(self._driver_source(
                row_to_me, local_counts, vector=vector))

        misc.compile(cwd=pjoin(output_path, 'Source'))
        make_command = ['make', 'nlo_real_offload.o', '-j2']
        if backend:
            make_command.append('NLO_REAL_BACKEND=%s' % backend)
        subprocess.check_call(make_command, cwd=subproc)
        matrix_objects = [
            os.path.basename(path)[:-2] + '.o' for path in
            sorted(glob.glob(pjoin(subproc, 'matrix_*.f')))]
        objects = matrix_objects + [
            'real_me_chooser.o', 'splitorders_stuff.o', object_name,
            'nlo_real_offload.o']
        compile_env = os.environ.copy()
        if backend:
            compile_env['NLO_REAL_BACKEND'] = backend
        misc.compile(objects, cwd=subproc, env=compile_env)
        subprocess.check_call(
            ['gfortran', '-o', executable] + objects + [
                '-L%s' % pjoin(output_path, 'lib'), '-ldhelas', '-lmodel',
                '-L%s' % subproc, '-lnlo_real_bridge',
                '-Wl,-rpath,$ORIGIN', '-lstdc++', '-ldl'],
            cwd=subproc)
        return executable

    def _run_routed_driver(self, executable, case, environment=None):
        env = os.environ.copy()
        if environment:
            for name, value in environment.items():
                if value is None:
                    env.pop(name, None)
                else:
                    env[name] = value
        output = subprocess.check_output(
            [executable], input=self._driver_input(case['points']),
            cwd=self.tmpdir, env=env, stderr=subprocess.STDOUT)
        text = output.decode(errors='replace')
        records = self._parse_driver_output(
            text, len(case['split_order_names']))
        self._assert_case(case, records)
        return text

    def _run_production_case(self, name, test_fallback=False):
        case = self.oracle['cases'][name]
        output_path = self._generate_case(name, madmatrix=True)
        scalar = self._compile_routed_driver(output_path, case)
        scalar_environment = {'MG7_NLO_REAL_BACKEND': 'scalar'}
        scalar_output = self._run_routed_driver(
            scalar, case, scalar_environment)
        self.assertIn('MG7 NLO real offload initialized: backend=scalar',
                      scalar_output)

        vector = self._compile_routed_driver(output_path, case, vector=True)
        vector_output = self._run_routed_driver(
            vector, case, scalar_environment)
        self.assertIn('MG7 NLO real offload initialized: backend=scalar',
                      vector_output)

        if test_fallback:
            default_fallback = self._run_routed_driver(
                scalar, case, {'MG7_NLO_REAL_BACKEND': None})
            self.assertIn('using Fortran fallback', default_fallback)
            fallback = self._run_routed_driver(
                scalar, case, {'MG7_NLO_REAL_BACKEND': 'fortran'})
            self.assertIn('using Fortran fallback', fallback)

            process_path = pjoin(
                output_path, 'SubProcesses', case['subprocess'])
            with open(pjoin(process_path, 'nlo_real_manifest.json')) as stream:
                manifest = json.load(stream)
            real = manifest['real_matrix_elements'][1]
            library = pjoin(
                output_path, 'lib',
                'libmadmatrix_%s_scalar.so' % real['library_process_id'])
            missing = library + '.missing'
            os.rename(library, missing)
            try:
                partial = self._run_routed_driver(
                    scalar, case, scalar_environment)
            finally:
                os.rename(missing, library)
            self.assertIn('real ME %d is unavailable' % real['id'], partial)
            self.assertNotIn('real ME 1 is unavailable', partial)

            launch = self._new_cmd()
            old_backend = os.environ.get('MG7_NLO_REAL_BACKEND')
            os.environ['MG7_NLO_REAL_BACKEND'] = 'scalar'
            try:
                self._run(launch, 'launch %s -f' % output_path)
            finally:
                if old_backend is None:
                    del os.environ['MG7_NLO_REAL_BACKEND']
                else:
                    os.environ['MG7_NLO_REAL_BACKEND'] = old_backend
            born_logs = []
            for manifest_path in glob.glob(pjoin(
                    output_path, 'SubProcesses', 'P*',
                    'nlo_real_manifest.json')):
                log_path = pjoin(os.path.dirname(manifest_path), 'test_ME.log')
                self.assertTrue(os.path.isfile(log_path), log_path)
                with open(log_path) as stream:
                    log = stream.read()
                self.assertIn(
                    'MG7 NLO real offload initialized: backend=scalar', log)
                born_logs.append(log_path)
            self.assertTrue(born_logs)

    @staticmethod
    def _fortran_float(value):
        return float(value.replace('D', 'E').replace('d', 'e'))

    @classmethod
    def _parse_driver_output(cls, output, split_order_count):
        records = []
        current = None
        for line in output.splitlines():
            tokens = line.split()
            if not tokens or not tokens[0].startswith('ORACLE_'):
                continue
            tag = tokens[0][7:].lower()
            if tag == 'begin':
                values = [int(value) for value in tokens[1:]]
                current = {
                    'point': values[0],
                    'fks_row': values[1],
                    'real_me_id': values[2],
                    'fortran_flavour_index': values[3],
                    'local_order_count': values[4],
                    'topology': values[5],
                    'momenta': [],
                    'local_squared_orders': [],
                    'global_squared_orders': [],
                }
            elif tag == 'g':
                current['g_strong'] = cls._fortran_float(tokens[1])
            elif tag == 'pdgs':
                current['pdgs'] = [int(value) for value in tokens[1:]]
            elif tag == 'momentum':
                index = int(tokens[1])
                if index != len(current['momenta']) + 1:
                    raise AssertionError('non-contiguous momentum records')
                current['momenta'].append([
                    cls._fortran_float(value) for value in tokens[2:]])
            elif tag == 'local_sum':
                current['local_sum'] = cls._fortran_float(tokens[1])
            elif tag in ('local', 'global'):
                order_end = 2 + split_order_count
                component = {
                    'index': int(tokens[1]),
                    'orders': [int(value) for value in
                               tokens[2:order_end]],
                    'value': cls._fortran_float(tokens[order_end]),
                }
                current['%s_squared_orders' % tag].append(component)
            elif tag == 'sum':
                current['summed'] = cls._fortran_float(tokens[1])
            elif tag == 'end':
                records.append(current)
                current = None
        if current is not None:
            raise AssertionError('unterminated ORACLE record')
        return records

    def _assert_close(self, actual, expected, scale=0.0, label='value'):
        self.assertTrue(math.isfinite(actual), '%s is not finite' % label)
        delta = max(abs(expected) * 1.e-12,
                    abs(scale) * 1.e-12, 1.e-30)
        self.assertAlmostEqual(actual, expected, delta=delta, msg=label)

    def _assert_case(self, case, actual_records):
        expected_records = sorted(
            case['records'], key=lambda item: (item['point'], item['fks_row']))
        actual_records = sorted(
            actual_records,
            key=lambda item: (item['point'], item['fks_row']))
        self.assertEqual(len(actual_records), len(expected_records))
        points = {point['id']: point for point in case['points']}
        tiny_filter_seen = False

        exact_fields = (
            'point', 'fks_row', 'real_me_id', 'fortran_flavour_index',
            'local_order_count', 'topology', 'pdgs')
        for actual, expected in zip(actual_records, expected_records):
            label = 'point %d FKS row %d' % (
                expected['point'], expected['fks_row'])
            for field in exact_fields:
                self.assertEqual(actual[field], expected[field],
                                 '%s: %s' % (label, field))

            point = points[expected['point']]
            self._assert_close(actual['g_strong'], point['g_strong'],
                               label='%s: G' % label)
            self.assertEqual(len(actual['momenta']), len(point['momenta']))
            for particle, (got, reference) in enumerate(
                    zip(actual['momenta'], point['momenta']), 1):
                self.assertEqual(len(got), 4)
                for component, (value, target) in enumerate(
                        zip(got, reference)):
                    self._assert_close(
                        value, target, scale=1000.,
                        label='%s: p(%d,%d)' %
                        (label, component, particle))

            self.assertEqual(
                len(actual['local_squared_orders']),
                expected['local_order_count'])
            local_scale = max(
                [abs(component['value']) for component in
                 expected['local_squared_orders']] or [0.])
            for kind in ('local_squared_orders', 'global_squared_orders'):
                self.assertEqual(len(actual[kind]), len(expected[kind]))
                for got, reference in zip(actual[kind], expected[kind]):
                    self.assertEqual(got['index'], reference['index'])
                    self.assertEqual(got['orders'], reference['orders'])
                    self._assert_close(
                        got['value'], reference['value'], scale=local_scale,
                        label='%s: %s %s' %
                        (label, kind, reference['orders']))

            self._assert_close(
                actual['local_sum'], expected['local_sum'],
                scale=local_scale, label='%s: local sum' % label)
            self._assert_close(
                actual['summed'], expected['summed'],
                scale=local_scale, label='%s: summed' % label)
            self._assert_close(
                math.fsum(component['value'] for component in
                          actual['local_squared_orders']),
                actual['local_sum'], scale=local_scale,
                label='%s: raw local sum invariant' % label)
            self._assert_close(
                math.fsum(component['value'] for component in
                          actual['global_squared_orders']),
                actual['summed'], scale=local_scale,
                label='%s: filtered global sum invariant' % label)

            global_by_order = {
                tuple(component['orders']): component['value']
                for component in actual['global_squared_orders']}
            for component in actual['local_squared_orders']:
                mapped = global_by_order[tuple(component['orders'])]
                if abs(component['value']) > local_scale * 1.e-12:
                    self._assert_close(
                        mapped, component['value'], scale=local_scale,
                        label='%s: local/global map' % label)
                elif component['value'] != 0. and mapped == 0.:
                    tiny_filter_seen = True

        flavours_by_me = {}
        for record in actual_records:
            flavours_by_me.setdefault(record['real_me_id'], set()).add(
                record['fortran_flavour_index'])
        self.assertTrue(any(len(flavours) > 1 for flavours in
                            flavours_by_me.values()))
        if case['subprocess'] == 'P0_gQ_wpQ':
            self.assertTrue(tiny_filter_seen)

    def _run_case(self, name):
        case = self.oracle['cases'][name]
        output_path = self._generate_case(name)
        actual = self._compile_and_run(output_path, case)
        self._assert_case(case, actual)

    @staticmethod
    def _bridge(process_path):
        bridge = ctypes.CDLL(pjoin(process_path, 'libnlo_real_bridge.so'))
        handle = ctypes.c_void_p
        bridge.mg7_nlo_real_initialize.argtypes = [
            ctypes.POINTER(handle), ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_char_p]
        bridge.mg7_nlo_real_initialize.restype = ctypes.c_int
        bridge.mg7_nlo_real_evaluate.argtypes = [
            handle, ctypes.c_int, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double)]
        bridge.mg7_nlo_real_evaluate.restype = ctypes.c_int
        bridge.mg7_nlo_real_finalize.argtypes = [ctypes.POINTER(handle)]
        bridge.mg7_nlo_real_finalize.restype = ctypes.c_int
        bridge.mg7_nlo_real_last_error.argtypes = [handle]
        bridge.mg7_nlo_real_last_error.restype = ctypes.c_char_p
        return bridge

    @staticmethod
    def _host_simd_backend():
        """Return a host SIMD backend supported by current MadMatrix rules."""
        flags = ''
        try:
            with open('/proc/cpuinfo') as stream:
                flags = stream.read()
        except IOError:
            pass
        if 'avx512vl' in flags:
            return 'avx512y', 4
        if 'avx2' in flags:
            return 'simd_256', 4
        if 'sse4_2' in flags:
            return 'simd_128', 2
        raise unittest.SkipTest('no supported host SIMD backend')

    def _compile_batch_wrapper_driver(self, output_path, case, backend,
                                      row, records, active):
        """Compile a compacting generated-Fortran batch-wrapper probe."""
        subproc = pjoin(output_path, 'SubProcesses', case['subprocess'])
        vector_size = len(records)
        source = """
      PROGRAM CHECK_REAL_BATCH
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      INTEGER VECTOR_SIZE
      PARAMETER (VECTOR_SIZE=%(vector_size)d)
      DOUBLE PRECISION P(0:3,NEXTERNAL,VECTOR_SIZE)
      DOUBLE PRECISION G_STRONG(VECTOR_SIZE)
      DOUBLE PRECISION RET(AMP_SPLIT_SIZE,VECTOR_SIZE),WGT(VECTOR_SIZE)
      LOGICAL ACTIVE(VECTOR_SIZE)
      INTEGER COUP_INDEX(VECTOR_SIZE),I,J,K
      DATA ACTIVE / %(active)s /
      DO I=1,VECTOR_SIZE
        READ(*,*) G_STRONG(I)
        DO J=1,NEXTERNAL
          READ(*,*) (P(K,J,I),K=0,3)
        ENDDO
        COUP_INDEX(I)=I
      ENDDO
      CALL SMATRIX_REAL_VEC_BATCH(P,G_STRONG,RET,WGT,ACTIVE,
     $ COUP_INDEX,VECTOR_SIZE,%(fks_row)d,%(flavour)d)
      DO I=1,VECTOR_SIZE
        WRITE(*,*) 'BATCH_LANE',I,WGT(I)
        DO J=1,AMP_SPLIT_SIZE
          WRITE(*,*) 'BATCH_GLOBAL',I,J,RET(J,I)
        ENDDO
      ENDDO
      END
""" % {
            'vector_size': vector_size,
            'active': ','.join('.TRUE.' if value else '.FALSE.'
                               for value in active),
            'fks_row': row['fks_row'],
            'flavour': row['fortran_flavour_index'],
        }
        source_path = pjoin(subproc, 'check_real_batch.f')
        with open(source_path, 'w') as stream:
            stream.write(source)

        misc.compile(cwd=pjoin(output_path, 'Source'))
        misc.compile(['couplmod'], cwd=subproc)
        subprocess.check_call([
            'make', '-f', 'nlo_real.mk', 'BACKEND=%s' % backend,
            'FPTYPE=d', '-j2'], cwd=subproc)
        subprocess.check_call([
            'make', 'nlo_real_offload.o',
            'NLO_REAL_BACKEND=%s' % backend, '-j2'], cwd=subproc)
        matrix_objects = [
            os.path.basename(path)[:-2] + '.o' for path in
            sorted(glob.glob(pjoin(subproc, 'matrix_*.f')))]
        objects = matrix_objects + [
            'real_me_chooser.o', 'splitorders_stuff.o',
            'check_real_batch.o', 'nlo_real_offload.o']
        misc.compile(objects[:-1], cwd=subproc)
        executable = pjoin(subproc, 'check_real_batch')
        subprocess.check_call(
            ['gfortran', '-o', executable] + objects + [
                '-L%s' % pjoin(output_path, 'lib'), '-ldhelas', '-lmodel',
                '-L%s' % subproc, '-lnlo_real_bridge',
                '-Wl,-rpath,$ORIGIN', '-lstdc++', '-ldl'], cwd=subproc)

        points = {point['id']: point for point in case['points']}
        values = []
        for record in records:
            point = points[record['point']]
            values.append('%.17e' % point['g_strong'])
            values.extend(' '.join('%.17e' % component
                                   for component in momentum)
                          for momentum in point['momenta'])
        environment = os.environ.copy()
        environment.update({
            'MG7_NLO_REAL_BACKEND': backend,
            'MG7_NLO_REAL_TRACE': '1',
        })
        output = subprocess.check_output(
            [executable], input=('\n'.join(values) + '\n').encode(),
            cwd=self.tmpdir, env=environment, stderr=subprocess.STDOUT)
        return output.decode(errors='replace')

    def _run_simd_case(self, name, check_compaction=False):
        case = self.oracle['cases'][name]
        output_path = self._generate_case(name, madmatrix=True)
        process_path = pjoin(
            output_path, 'SubProcesses', case['subprocess'])
        with open(pjoin(process_path, 'nlo_real_manifest.json')) as stream:
            manifest = json.load(stream)
        rows = {row['fks_row']: row for row in manifest['fks_rows']}
        backend, width = self._host_simd_backend()
        subprocess.check_call([
            'make', '-f', 'nlo_real.mk', 'BACKEND=%s' % backend,
            'FPTYPE=d', '-j2'], cwd=process_path)
        bridge = self._bridge(process_path)
        context = ctypes.c_void_p()
        self.assertEqual(
            bridge.mg7_nlo_real_initialize(
                ctypes.byref(context),
                pjoin(output_path, 'Cards', 'param_card.dat').encode(),
                pjoin(output_path, 'lib').encode(), backend.encode()), 0,
            bridge.mg7_nlo_real_last_error(context).decode())
        points = {point['id']: point for point in case['points']}
        try:
            # Alternate physical rows sharing a real ME to catch stale SIMD
            # flavour pages, then cover short/full/nonmultiple page counts.
            selected_rows = [manifest['fks_rows'][0]]
            first_real = selected_rows[0]['real_me_id']
            for candidate in manifest['fks_rows'][1:]:
                if (candidate['real_me_id'] == first_real and
                        candidate['madmatrix_flavour_index'] !=
                        selected_rows[0]['madmatrix_flavour_index']):
                    selected_rows.append(candidate)
                    break
            counts = sorted(set((1, max(1, width - 1), width,
                                 width + 1, 2 * width - 1)))
            for row in selected_rows:
                row_records = [record for record in case['records']
                               if record['fks_row'] == row['fks_row']]
                for count in counts:
                    records = row_records[:count]
                    flat_momenta = []
                    for component in range(4):
                        for particle in range(manifest['nexternal']):
                            for record in records:
                                flat_momenta.append(points[record['point']][
                                    'momenta'][particle][component])
                    momenta = (ctypes.c_double * len(flat_momenta))(
                        *flat_momenta)
                    g_strong = (ctypes.c_double * count)(*[
                        points[record['point']]['g_strong']
                        for record in records])
                    flavour = (ctypes.c_int32 * count)(*[
                        row['madmatrix_flavour_index']] * count)
                    nlocal = len(records[0]['local_squared_orders'])
                    output = (ctypes.c_double * (count * nlocal))()
                    self.assertEqual(
                        bridge.mg7_nlo_real_evaluate(
                            context, row['real_me_id'], count, momenta,
                            g_strong, flavour, output), 0,
                        bridge.mg7_nlo_real_last_error(context).decode())
                    for order in range(nlocal):
                        for event, record in enumerate(records):
                            expected = record['local_squared_orders'][order][
                                'value']
                            scale = max(abs(component['value']) for component
                                        in record['local_squared_orders'])
                            self._assert_close(
                                output[count * order + event], expected,
                                scale=scale,
                                label='%s SIMD count %d event %d order %d' %
                                (backend, count, event, order))
        finally:
            self.assertEqual(
                bridge.mg7_nlo_real_finalize(ctypes.byref(context)), 0)

        row = manifest['fks_rows'][0]
        records = [record for record in case['records']
                   if record['fks_row'] == row['fks_row']][:width + 1]
        active = [True] * len(records)
        if check_compaction and len(active) >= 3:
            active[1] = False
            active[-1] = False
        output = self._compile_batch_wrapper_driver(
            output_path, case, backend, row, records, active)
        self.assertIn('backend=%s' % backend, output)
        self.assertIn('event_count=%d' % sum(active), output)
        weights = {}
        globals_by_lane = {}
        for line in output.splitlines():
            fields = line.split()
            if 'BATCH_LANE' in fields:
                at = fields.index('BATCH_LANE')
                weights[int(fields[at + 1])] = float(fields[at + 2])
            elif 'BATCH_GLOBAL' in fields:
                at = fields.index('BATCH_GLOBAL')
                globals_by_lane.setdefault(int(fields[at + 1]), []).append(
                    float(fields[at + 3]))
        self.assertEqual(sorted(weights), list(range(1, len(records) + 1)))
        for lane, (record, enabled) in enumerate(
                zip(records, active), start=1):
            expected_weight = record['summed'] if enabled else 0.
            scale = max([abs(component['value']) for component in
                         record['local_squared_orders']] or [0.])
            self._assert_close(weights[lane], expected_weight, scale=scale,
                               label='packed wrapper weight lane %d' % lane)
            expected_global = ([component['value'] for component in
                                record['global_squared_orders']]
                               if enabled else
                               [0.] * len(record['global_squared_orders']))
            for position, expected in enumerate(expected_global):
                self._assert_close(
                    globals_by_lane[lane][position], expected, scale=scale,
                    label='packed wrapper lane %d global %d' %
                    (lane, position + 1))

    @staticmethod
    def _require_cuda():
        if not shutil.which('nvcc') or not shutil.which('nvidia-smi'):
            raise unittest.SkipTest('CUDA compiler or NVIDIA tooling absent')
        try:
            subprocess.check_call(
                ['nvidia-smi', '-L'], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            raise unittest.SkipTest('no usable NVIDIA device')

    def _run_cuda_qcd_case(self):
        """Execute every QCD flavour row and CUDA rounding boundaries."""
        self._require_cuda()
        case = self.oracle['cases']['grouped_qcd_ttx']
        output_path = self._generate_case(
            'grouped_qcd_ttx', madmatrix=True, vector_size=5)
        process_path = pjoin(
            output_path, 'SubProcesses', case['subprocess'])
        with open(pjoin(process_path, 'nlo_real_manifest.json')) as stream:
            manifest = json.load(stream)
        subprocess.check_call([
            'make', '-f', 'nlo_real.mk', 'BACKEND=cuda', 'FPTYPE=d',
            '-j2'], cwd=process_path)
        bridge = self._bridge(process_path)
        context = ctypes.c_void_p()
        self.assertEqual(
            bridge.mg7_nlo_real_initialize(
                ctypes.byref(context),
                pjoin(output_path, 'Cards', 'param_card.dat').encode(),
                pjoin(output_path, 'lib').encode(), b'cuda'), 0,
            bridge.mg7_nlo_real_last_error(context).decode())
        points = {point['id']: point for point in case['points']}

        def evaluate(row, records, label):
            count = len(records)
            flat = [
                points[record['point']]['momenta'][particle][component]
                for component in range(4)
                for particle in range(manifest['nexternal'])
                for record in records]
            momenta = (ctypes.c_double * len(flat))(*flat)
            g_strong = (ctypes.c_double * count)(*[
                points[record['point']]['g_strong'] for record in records])
            flavour = (ctypes.c_int32 * count)(*[
                row['madmatrix_flavour_index']] * count)
            output = (ctypes.c_double * count)()
            self.assertEqual(
                bridge.mg7_nlo_real_evaluate(
                    context, row['real_me_id'], count, momenta, g_strong,
                    flavour, output), 0,
                bridge.mg7_nlo_real_last_error(context).decode())
            for event, record in enumerate(records):
                expected = record['local_squared_orders'][0]['value']
                self._assert_close(
                    output[event], expected, scale=abs(expected),
                    label='%s event %d' % (label, event))

        try:
            for row in manifest['fks_rows']:
                records = [record for record in case['records']
                           if record['fks_row'] == row['fks_row']]
                evaluate(row, records, 'CUDA row %d' % row['fks_row'])
            row = manifest['fks_rows'][0]
            base = [record for record in case['records']
                    if record['fks_row'] == row['fks_row']]
            for count in (255, 256, 257, 513):
                records = [base[index % len(base)]
                           for index in range(count)]
                evaluate(row, records, 'CUDA boundary %d' % count)
        finally:
            self.assertEqual(
                bridge.mg7_nlo_real_finalize(ctypes.byref(context)), 0)

        row = manifest['fks_rows'][0]
        records = [record for record in case['records']
                   if record['fks_row'] == row['fks_row']][:5]
        packed = self._compile_batch_wrapper_driver(
            output_path, case, 'cuda', row, records,
            [True, False, True, True, False])
        self.assertIn('backend=cuda', packed)
        self.assertIn('event_count=3', packed)
        weights = {}
        for line in packed.splitlines():
            fields = line.split()
            if 'BATCH_LANE' in fields:
                at = fields.index('BATCH_LANE')
                weights[int(fields[at + 1])] = float(fields[at + 2])
        active = [True, False, True, True, False]
        for lane, (record, enabled) in enumerate(
                zip(records, active), start=1):
            expected = record['summed'] if enabled else 0.
            self._assert_close(
                weights[lane], expected, scale=abs(record['summed']),
                label='CUDA compacted lane %d' % lane)

        subprocess.check_call(['make', 'clean'], cwd=process_path)
        self.assertFalse(os.path.exists(
            pjoin(process_path, 'libnlo_real_bridge.so')))
        for real in manifest['real_matrix_elements']:
            self.assertFalse(glob.glob(pjoin(
                output_path, 'lib', 'libmadmatrix_%s_*.so' %
                real['library_process_id'])))

    def _run_cuda_mixed_case(self):
        """Use CUDA only for one-order reals and log all split-order fallback."""
        self._require_cuda()
        case = self.oracle['cases']['grouped_mixed_wj']
        output_path = self._generate_case('grouped_mixed_wj', madmatrix=True)
        process_path = pjoin(
            output_path, 'SubProcesses', case['subprocess'])
        with open(pjoin(process_path, 'nlo_real_manifest.json')) as stream:
            manifest = json.load(stream)
        supported = [real for real in manifest['real_matrix_elements']
                     if real['backend_capabilities']['cuda']]
        unsupported = [real for real in manifest['real_matrix_elements']
                       if not real['backend_capabilities']['cuda']]
        self.assertTrue(supported)
        self.assertTrue(unsupported)
        executable = self._compile_routed_driver(
            output_path, case, backend='cuda')
        text = self._run_routed_driver(
            executable, case, {'MG7_NLO_REAL_BACKEND': 'cuda'})
        self.assertIn('MG7 NLO real offload initialized: backend=cuda', text)
        for real in unsupported:
            self.assertIn(
                'MG7 NLO real GPU fallback: real_me_id=%d' % real['id'],
                text)
            library = pjoin(
                output_path, 'lib', 'libmadmatrix_%s_cuda.so' %
                real['library_process_id'])
            self.assertFalse(os.path.exists(library), library)
        for real in supported:
            library = pjoin(
                output_path, 'lib', 'libmadmatrix_%s_cuda.so' %
                real['library_process_id'])
            self.assertTrue(os.path.isfile(library), library)

    def _run_madmatrix_case(self, name, low_memory=False):
        """Generate, build, relocate, and replay every scalar bridge row."""

        case = self.oracle['cases'][name]
        output_path = self._generate_case(
            name, madmatrix=True, low_memory=low_memory)
        process_path = pjoin(
            output_path, 'SubProcesses', case['subprocess'])
        manifest_path = pjoin(process_path, 'nlo_real_manifest.json')
        with open(manifest_path) as stream:
            manifest = json.load(stream)

        self.assertEqual(manifest['version'], 1)
        self.assertEqual(manifest['nexternal'], len(case['points'][0]['momenta']))
        self.assertEqual(manifest['split_order_names'],
                         case['split_order_names'])
        rows = {row['fks_row']: row for row in manifest['fks_rows']}
        self.assertEqual(sorted(rows), case['selected_rows'])
        reals = {real['id']: real for real in
                 manifest['real_matrix_elements']}
        for row in rows.values():
            self.assertEqual(
                row['madmatrix_flavour_index'],
                row['fortran_flavour_index'] - 1)
            self.assertEqual(
                reals[row['real_me_id']]['local_flavours'][
                    row['madmatrix_flavour_index']], row['pdgs'])

        subprocess.check_call(
            ['make', '-f', 'nlo_real.mk', 'BACKEND=scalar', 'FPTYPE=d',
             '-j2'], cwd=process_path)

        # Runtime paths must survive moving the complete generated process tree.
        generated_path = output_path
        relocated = output_path + '-relocated'
        os.rename(output_path, relocated)
        output_path = relocated
        process_path = pjoin(
            output_path, 'SubProcesses', case['subprocess'])
        bridge = self._bridge(process_path)
        context = ctypes.c_void_p()
        status = bridge.mg7_nlo_real_initialize(
            ctypes.byref(context),
            pjoin(output_path, 'Cards', 'param_card.dat').encode(),
            pjoin(output_path, 'lib').encode(), b'scalar')
        self.assertEqual(
            status, 0,
            bridge.mg7_nlo_real_last_error(context).decode())

        points = {point['id']: point for point in case['points']}
        first_values = {}
        try:
            for expected in sorted(
                    case['records'],
                    key=lambda record: (record['point'], record['fks_row'])):
                row = rows[expected['fks_row']]
                real = reals[row['real_me_id']]
                local_orders = expected['local_squared_orders']
                self.assertEqual(
                    real['local_squared_orders'],
                    [component['orders'] for component in local_orders])
                self.assertEqual(
                    row['local_to_global'],
                    [manifest['global_squared_orders'].index(
                        component['orders']) + 1
                     for component in local_orders])

                point = points[expected['point']]
                flat_momenta = [
                    point['momenta'][particle][component]
                    for component in range(4)
                    for particle in range(manifest['nexternal'])]
                momenta = (ctypes.c_double * len(flat_momenta))(
                    *flat_momenta)
                g_strong = (ctypes.c_double * 1)(point['g_strong'])
                flavour = (ctypes.c_int32 * 1)(
                    row['madmatrix_flavour_index'])
                values = (ctypes.c_double * len(local_orders))()
                status = bridge.mg7_nlo_real_evaluate(
                    context, row['real_me_id'], 1, momenta, g_strong,
                    flavour, values)
                self.assertEqual(
                    status, 0,
                    bridge.mg7_nlo_real_last_error(context).decode())
                scale = max(
                    [abs(component['value']) for component in local_orders]
                    or [0.])
                for index, component in enumerate(local_orders):
                    self._assert_close(
                        values[index], component['value'], scale=scale,
                        label='MadMatrix point %d FKS row %d order %s' % (
                            expected['point'], expected['fks_row'],
                            component['orders']))

                # Point one naturally alternates among real libraries. Retain
                # the first value for each real and prove repeated calls agree.
                key = (expected['point'], row['real_me_id'],
                       row['madmatrix_flavour_index'])
                result = tuple(values)
                if key in first_values:
                    for got, reference in zip(result, first_values[key]):
                        self._assert_close(got, reference, scale=scale,
                                           label='repeated real-library call')
                else:
                    first_values[key] = result

            first = case['records'][0]
            row = rows[first['fks_row']]
            point = points[first['point']]
            flat_momenta = [
                point['momenta'][particle][component]
                for component in range(4)
                for particle in range(manifest['nexternal'])]
            momenta = (ctypes.c_double * len(flat_momenta))(*flat_momenta)
            g_strong = (ctypes.c_double * 1)(point['g_strong'])
            output = (ctypes.c_double * len(first['local_squared_orders']))()
            good_flavour = (ctypes.c_int32 * 1)(
                row['madmatrix_flavour_index'])
            bad_flavour = (ctypes.c_int32 * 1)(
                len(reals[row['real_me_id']]['local_flavours']))
            self.assertNotEqual(
                bridge.mg7_nlo_real_evaluate(
                    context, 0, 1, momenta, g_strong, good_flavour, output),
                0)
            self.assertNotEqual(
                bridge.mg7_nlo_real_evaluate(
                    context, row['real_me_id'], 1, momenta, g_strong,
                    bad_flavour, output), 0)
            # A contained bad call must not poison the context.
            self.assertEqual(
                bridge.mg7_nlo_real_evaluate(
                    context, row['real_me_id'], 1, momenta, g_strong,
                    good_flavour, output), 0)
            # Non-finite results are returned unchanged, as the retained
            # Fortran real matrix element gives at degenerate points; the
            # Fortran wrapper must never switch backends in a run because of
            # them.
            nan_momenta = (ctypes.c_double * len(flat_momenta))(*flat_momenta)
            nan_momenta[0] = float('nan')
            self.assertEqual(
                bridge.mg7_nlo_real_evaluate(
                    context, row['real_me_id'], 1, nan_momenta, g_strong,
                    good_flavour, output), 0)
            self.assertFalse(all(math.isfinite(value) for value in output))
            self.assertEqual(
                bridge.mg7_nlo_real_evaluate(
                    context, row['real_me_id'], 1, momenta, g_strong,
                    good_flavour, output), 0)
            self.assertTrue(all(math.isfinite(value) for value in output))

            probes = []
            seen_reals = set()
            for record in case['records']:
                probe_row = rows[record['fks_row']]
                if probe_row['real_me_id'] not in seen_reals:
                    probes.append(record)
                    seen_reals.add(probe_row['real_me_id'])
                if len(probes) == 2:
                    break
            self.assertEqual(len(probes), 2)

            def probe(handle, record):
                probe_row = rows[record['fks_row']]
                probe_point = points[record['point']]
                flat = [
                    probe_point['momenta'][particle][component]
                    for component in range(4)
                    for particle in range(manifest['nexternal'])]
                probe_momenta = (ctypes.c_double * len(flat))(*flat)
                probe_g = (ctypes.c_double * 1)(probe_point['g_strong'])
                probe_flavour = (ctypes.c_int32 * 1)(
                    probe_row['madmatrix_flavour_index'])
                probe_output = (ctypes.c_double * len(
                    record['local_squared_orders']))()
                self.assertEqual(
                    bridge.mg7_nlo_real_evaluate(
                        handle, probe_row['real_me_id'], 1, probe_momenta,
                        probe_g, probe_flavour, probe_output), 0,
                    bridge.mg7_nlo_real_last_error(handle).decode())
                return tuple(probe_output)

            first_before = probe(context, probes[0])
            probe(context, probes[1])
            first_after = probe(context, probes[0])
            self.assertEqual(first_after, first_before)

            second_context = ctypes.c_void_p()
            self.assertEqual(
                bridge.mg7_nlo_real_initialize(
                    ctypes.byref(second_context),
                    pjoin(output_path, 'Cards', 'param_card.dat').encode(),
                    pjoin(output_path, 'lib').encode(), b'scalar'), 0)
            try:
                self.assertEqual(probe(second_context, probes[0]),
                                 first_before)
            finally:
                self.assertEqual(
                    bridge.mg7_nlo_real_finalize(
                        ctypes.byref(second_context)), 0)
                self.assertFalse(second_context.value)

            invalid_context = ctypes.c_void_p()
            self.assertNotEqual(
                bridge.mg7_nlo_real_initialize(
                    ctypes.byref(invalid_context),
                    pjoin(output_path, 'Cards', 'param_card.dat').encode(),
                    pjoin(output_path, 'lib').encode(),
                    b'unsupported-backend'), 0)
            self.assertFalse(invalid_context.value)
            self.assertIn(
                b'no NLO real libraries are available',
                bridge.mg7_nlo_real_last_error(None))
        finally:
            self.assertEqual(
                bridge.mg7_nlo_real_finalize(ctypes.byref(context)), 0)
            self.assertFalse(context.value)
            os.rename(relocated, generated_path)

    def test_grouped_qcd_ttx_real_components(self):
        self._run_case('grouped_qcd_ttx')

    def test_grouped_mixed_wj_real_components(self):
        self._run_case('grouped_mixed_wj')

    def test_grouped_qcd_ttx_madmatrix_scalar_components(self):
        self._run_madmatrix_case('grouped_qcd_ttx')

    def test_grouped_mixed_wj_madmatrix_scalar_components(self):
        self._run_madmatrix_case('grouped_mixed_wj')

    def test_grouped_qcd_ttx_madmatrix_low_memory(self):
        self._run_madmatrix_case('grouped_qcd_ttx', low_memory=True)

    def test_grouped_qcd_ttx_production_scalar_and_fallback(self):
        self._run_production_case('grouped_qcd_ttx', test_fallback=True)

    def test_grouped_mixed_wj_production_scalar_components(self):
        self._run_production_case('grouped_mixed_wj')

    def test_grouped_qcd_ttx_madmatrix_cpu_simd_batches(self):
        self._run_simd_case('grouped_qcd_ttx', check_compaction=True)

    def test_grouped_mixed_wj_madmatrix_cpu_simd_components(self):
        self._run_simd_case('grouped_mixed_wj')

    def test_grouped_qcd_ttx_madmatrix_cuda_batches(self):
        self._run_cuda_qcd_case()

    def test_grouped_mixed_wj_madmatrix_cuda_fallback(self):
        self._run_cuda_mixed_case()


if __name__ == '__main__':
    unittest.main()
