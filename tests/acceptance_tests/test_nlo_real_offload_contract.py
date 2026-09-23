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

    def _generate_case(self, name):
        case = self.oracle['cases'][name]
        output_path = pjoin(self.tmpdir, name)
        cmd = self._new_cmd()
        self._run(cmd, 'set apply_flavor_grouping True --no_save')
        if name == 'grouped_mixed_wj':
            self._run(cmd, 'set nlo_mixed_expansion True --no_save')

        model = case['model']
        if model.startswith('tests/'):
            model = pjoin(MG5DIR, model)
        self._run(cmd, 'import model %s' % model)
        self._run(cmd, 'generate %s' % case['process'])
        self._run(
            cmd, 'output standalone_fortran --fks --limits %s -f' %
            output_path)
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

    def _driver_source(self, row_to_me, local_counts):
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
        lines.extend([
            '          GLOBAL(:)=0D0',
            '          LOCAL(:)=0D0',
            '          CALL SMATRIX_REAL(P,GLOBAL,WGT)',
        ])
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

    def test_grouped_qcd_ttx_real_components(self):
        self._run_case('grouped_qcd_ttx')

    def test_grouped_mixed_wj_real_components(self):
        self._run_case('grouped_mixed_wj')


if __name__ == '__main__':
    unittest.main()
