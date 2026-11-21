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
import ctypes.util
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import copy

from madgraph import MG5DIR
import madgraph.interface.master_interface as MGCmd
import madgraph.various.misc as misc
from madgraph.various import banner


pjoin = os.path.join
# UMAMI 1.5 appends NLO keys after the released 1.3 flavour-switch keys.
IN_G_STRONG, IN_G_EW, OUT_SQUARED_ORDERS = 17, 18, 14


def native_section_link_flag():
    return '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections'


def native_bridge_link_flags(directory="."):
    if sys.platform == 'darwin':
        return ['-L'+str(directory), '-lnlo_real_bridge', '-Wl,-rpath,@loader_path', '-lc++', '-ldl']
    return ['-L'+str(directory), '-lnlo_real_bridge', '-Wl,-rpath,$ORIGIN', '-lstdc++', '-ldl']
ORACLE_PATH = pjoin(
    MG5DIR, 'tests', 'input_files', 'nlo_real_amplitudes_oracle.json')


class TestNLORealFortranOracle(unittest.TestCase):
    """Generate and replay scalar Fortran real-ME reference records."""

    maxDiff = None

    @staticmethod
    def _library_path(output_path, manifest, real, backend, fptype='d', libtag=''):
        return pjoin(output_path, 'lib', 'libmadmatrix%s_%s_%s_%s_%s.so' % (
            libtag, manifest['library_model'], real['library_process_id'],
            backend, fptype))

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
        cmd._experimental_nlo_flavor_grouping = True
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
            contents = stream.read()
        # Upstream corrected this comment's grammar after the oracle was
        # recorded. Preserve the exact byte-level physics fixture otherwise.
        contents = contents.replace(
            b'## to external programs such as Pythia.\n',
            b'## to external program such as Pythia.\n')
        digest = hashlib.sha256(contents).hexdigest()
        self.assertEqual(digest, case['param_card']['sha256'])
        return output_path

    def _case_for_output(self, name, output_path):
        """Rebind transient row numbers without changing the trusted oracle.

        Ordered NLO-real flavours restore six W+j sectors that were absent
        from the original fixture. Every original numerical value remains a
        fixed reference; only its generated row/local index is resolved anew.
        Added sectors use a separately recorded native Fortran control on the
        same fixed points, in addition to independent all-sector QCD/limit tests.
        """
        original = self.oracle['cases'][name]
        if name != 'grouped_mixed_wj':
            return original
        case = copy.deepcopy(original)
        subproc = pjoin(output_path, 'SubProcesses', case['subprocess'])
        with open(pjoin(subproc, 'fks_info.inc')) as stream:
            flat = ' '.join(line[6:].strip() for line in stream
                            if line and line[0].upper() != 'C')
        def array_values(name):
            match = re.search(r'DATA\s+' + name + r'\s*/([^/]+)/', flat, re.I)
            if match:
                return [int(x) for x in re.findall(r'-?\d+', match.group(1))]
            # Native-context exports initialize each alias separately; the
            # original physical real rows remain the leading entries.
            indexed = {int(index): int(value) for index, value in re.findall(
                r'DATA\s+' + name + r'\s*\(\s*(\d+)\s*\)\s*/\s*(-?\d+)\s*/',
                flat, re.I)}
            self.assertTrue(indexed, name)
            self.assertEqual(sorted(indexed), list(range(1, max(indexed) + 1)))
            return [indexed[index] for index in sorted(indexed)]
        topology = array_values('FKS_TOPOLOGY_D')
        flavours = array_values('REAL_FLAVOR_INDEX_D')
        pdgs = {int(row): tuple(map(int, re.findall(r'-?\d+', values)))
                for row, values in re.findall(
                    r'DATA\s*\(PDG_TYPE_D\((\d+),\s*IPOS\).*?/([^/]+)/',
                    flat, re.I)}
        row_to_me, unused_counts = self._real_me_contract(subproc)
        rows = {(me, topology[row - 1], pdgs[row]): row
                for row, me in row_to_me.items()}
        self.assertEqual(len(rows), len(row_to_me))
        old_keys = {(r['real_me_id'], r['topology'], tuple(r['pdgs']))
                    for r in case['records']}
        restored_pdgs = {(1, 4, 24, 3, 1), (2, 4, 24, 3, 2),
                         (3, 2, 24, 1, 3), (4, 2, 24, 1, 4),
                         (5, 2, 24, 1, 5), (5, 4, 24, 3, 5)}
        restored_keys = {(3, 5, p) for p in restored_pdgs}
        self.assertEqual(set(rows), old_keys | restored_keys)
        self.assertEqual(len(row_to_me), 36)
        for record in case['records']:
            row = rows[(record['real_me_id'], record['topology'],
                        tuple(record['pdgs']))]
            record['fks_row'] = row
            record['fortran_flavour_index'] = flavours[row - 1]
            self.assertGreater(record['fortran_flavour_index'], 0)
        case['selected_rows'] = sorted(row_to_me)
        manifest_path = pjoin(subproc, 'nlo_real_manifest.json')
        if os.path.isfile(manifest_path):
            with open(manifest_path) as stream:
                manifest = json.load(stream)
            reals = {r['id']: r for r in manifest['real_matrix_elements']}
            for row, me in row_to_me.items():
                self.assertEqual(reals[me]['local_flavours'][flavours[row - 1] - 1],
                                 list(pdgs[row]))
        if os.path.isfile(pjoin(subproc, 'nlo_real_offload.f90')):
            executable = self._compile_routed_driver(
                output_path, case, backend='fortran')
            output = subprocess.check_output([executable],
                input=self._driver_input(case['points']), cwd=subproc,
                stderr=subprocess.STDOUT).decode(errors='replace')
            self.assertIn('using Fortran fallback', output)
            self.assertNotIn('offload initialized', output)
            native = self._parse_driver_output(output, len(case['split_order_names']))
        else:
            native = self._compile_and_run(output_path, case)
        retained = [r for r in native if
                    (r['real_me_id'], r['topology'], tuple(r['pdgs'])) in old_keys]
        self._assert_case(case, retained)
        added = [r for r in native if
                 (r['real_me_id'], r['topology'], tuple(r['pdgs'])) in restored_keys]
        self.assertEqual(len(added), 6 * len(case['points']))
        self.assertEqual(len(native), 36 * len(case['points']))
        added_keys = [(r['point'], r['real_me_id'], r['topology'], tuple(r['pdgs']))
                      for r in added]
        self.assertEqual(len(set(added_keys)), len(added_keys))
        self.assertEqual(set(added_keys), {
            (point['id'], me, top, physical)
            for point in case['points'] for me, top, physical in restored_keys})
        case['records'].extend(added)
        # This applies finite-value, physical input and every local/global sum
        # invariant to the restored controls before they can become references.
        self._assert_case(case, native)
        with open(pjoin(output_path, 'restored_wj_native_reference.json'), 'w') as stream:
            json.dump({'scope': 'Six restored sectors; original JSON/numerical '
                                'references unchanged; same-point native control',
                       'records': added}, stream, indent=2)
        return case

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

    @staticmethod
    def _physical_scalar_name(subproc):
        with open(pjoin(subproc, 'real_me_chooser.f')) as stream:
            chooser = stream.read()
        return ('SMATRIX_REAL_LOCAL' if re.search(
            r'(?i)\bsubroutine\s+smatrix_real_local\s*\(', chooser)
            else 'SMATRIX_REAL')

    def _driver_source(self, row_to_me, local_counts, vector=False,
                       fortran_vector=False, scalar_name='SMATRIX_REAL'):
        """Build a fixed-form driver without modifying production wrappers.

        Physical real rows are distinct from the extra native Born/history
        aliases. Exercise the retained physical provider leaf (including its
        MadMatrix wrapper when generated), not the surrounding alias router.

        With fortran_vector, the batched Fortran real ME of lane 1 is
        recorded, after the scalar one on the same point and FKS row (which
        sets the split-order and helicity filters, as set_goodhel does), for
        an exact comparison."""

        real_mes = sorted(local_counts)
        vector = vector or fortran_vector
        lines = ['      PROGRAM CHECK_REAL_ORACLE']
        if fortran_vector:
            lines.append('      USE COUPLINGS')
        lines.extend([
            '      IMPLICIT NONE',
            "      INCLUDE 'nexternal.inc'",
            "      INCLUDE 'orders.inc'",
            "      INCLUDE 'nFKSconfigs.inc'",
            "      INCLUDE 'coupl.inc'",
            '      INTEGER I,J,K,IROW,NPOINTS,NME,NLOCAL,NFKSPROCESS',
            '      DOUBLE PRECISION P(0:3,NEXTERNAL),GIN,WGT',
            '      DOUBLE PRECISION LOCAL(0:AMP_SPLIT_SIZE)',
            '      DOUBLE PRECISION GLOBAL(AMP_SPLIT_SIZE)',
            '      COMMON/C_NFKSPROCESS/NFKSPROCESS',
            "      INCLUDE 'fks_info.inc'",
            "      INCLUDE 'amp_split_orders.inc'",
        ])
        for real_me in real_mes:
            lines.append(
                '      INTEGER GETORDPOWFROMINDEX%d' % real_me)
        if fortran_vector:
            lines.append(
                '      DOUBLE PRECISION SWGT,SGLOBAL(AMP_SPLIT_SIZE)')
        lines.append("      CALL SETPARA('param_card.dat')")
        if fortran_vector:
            lines.append('      CALL ALLOCATE_COUPLINGS(1)')
        lines.extend([
            '      READ(*,*) NPOINTS',
            '      DO K=1,NPOINTS',
            '        READ(*,*) GIN',
            '        DO J=1,NEXTERNAL',
            '          READ(*,*) (P(I,J),I=0,3)',
            '        ENDDO',
            '        G=GIN',
            '        CALL UPDATE_AS_PARAM()',
        ])
        if fortran_vector:
            lines.append('        CALL UPDATE_AS_PARAM_VEC(1)')
        lines.extend([
            '        DO IROW=1,%d' % max(row_to_me),
            '          NFKSPROCESS=IROW',
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
        real_call = ['          CALL %s(P,GLOBAL,WGT)' % scalar_name]
        if vector:
            real_call = [
                '          CALL SMATRIX_REAL_VEC(P,GLOBAL,WGT,1,',
                '     $      NFKSPROCESS,',
                '     $      REAL_FLAVOR_INDEX_D(NFKSPROCESS))']
        if fortran_vector:
            real_call.insert(0, '          CALL %s(P,SGLOBAL,SWGT)' % scalar_name)
        lines.extend([
            '          GLOBAL(:)=0D0',
            '          LOCAL(:)=0D0',
        ] + real_call + ['          NFKSPROCESS=IROW'])
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
        ])
        if fortran_vector:
            lines.extend([
                '          DO J=1,AMP_SPLIT_SIZE',
                "            WRITE(*,*) 'ORACLE_SCALAR_GLOBAL',J,SGLOBAL(J)",
                '          ENDDO',
                "          WRITE(*,*) 'ORACLE_SCALAR_SUM',SWGT",
            ])
        lines.extend([
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

    def _compile_and_run(self, output_path, case, fortran_vector=False):
        subproc = pjoin(output_path, 'SubProcesses', case['subprocess'])
        row_to_me, local_counts = self._real_me_contract(subproc)
        selected_rows = case['selected_rows']
        self.assertEqual(selected_rows, sorted(row_to_me))

        name = 'check_real_oracle%s' % ('_fvec' if fortran_vector else '')
        driver = self._driver_source(
            row_to_me, local_counts, fortran_vector=fortran_vector,
            scalar_name=self._physical_scalar_name(subproc))
        driver_path = pjoin(subproc, name + '.f')
        with open(driver_path, 'w') as stream:
            stream.write(driver)

        misc.compile(cwd=pjoin(output_path, 'Source'))
        if fortran_vector:
            misc.compile(['couplmod'], cwd=subproc)
        matrix_objects = [
            os.path.basename(path)[:-2] + '.o' for path in
            sorted(glob.glob(pjoin(subproc, 'matrix_*.f')))]
        self.assertEqual(
            sorted(int(re.search(r'(\d+)', obj).group(1))
                   for obj in matrix_objects),
            sorted(local_counts))
        objects = matrix_objects + [
            'real_me_chooser.o', 'splitorders_stuff.o', name + '.o']
        objects += self._native_real_link_objects(output_path, subproc)
        misc.compile(objects, cwd=subproc)

        executable = pjoin(subproc, name)
        subprocess.check_call(
            ['gfortran', '-o', executable] + objects +
            ['-L%s' % pjoin(output_path, 'lib'), '-ldhelas', '-lmodel']+
            self._native_real_link_flags(output_path, subproc),
            cwd=subproc)
        output = subprocess.check_output(
            [executable], input=self._driver_input(case['points']),
            cwd=subproc, stderr=subprocess.STDOUT)
        return self._parse_driver_output(
            output.decode(errors='replace'),
            len(case['split_order_names']))

    def _native_real_link_objects(self, output_path, subproc):
        """Link the actual native router/state into isolated real probes.

        Function sections discard unused MC activation callbacks; these probes
        exercise amplitudes and never initialize full matching/event tables.
        """
        if not os.path.isfile(pjoin(subproc, 'native_real_me_chooser.f')):
            return []
        misc.compile(['native_real_me_chooser.o', 'born_support.o', 'driver.o'], cwd=subproc)
        subprocess.check_call([
            'gfortran', '-O0', '-fPIC', '-ffunction-sections', '-fdata-sections',
            '-ffree-line-length-none', '-I'+pjoin(output_path, 'lib'),
            '-c', 'mc_native_context.f90', '-o', 'native_context_probe.o'], cwd=subproc)
        subprocess.check_call(['make', 'timing.o'], cwd=pjoin(output_path, 'Source'))
        return ['native_real_me_chooser.o', 'born_support.o', 'mc_decision_ledger.o', 'mc_batch_trace.o',
                'native_context_probe.o', 'driver.o', '../../Source/timing.o']

    def _native_real_link_flags(self, output_path, subproc):
        if not os.path.isfile(pjoin(subproc, 'native_real_me_chooser.f')):
            return []
        return ['-lmc_born_support', native_section_link_flag(),
                '-Wl,-rpath,'+pjoin(output_path, 'lib')]

    def _compile_routed_driver(self, output_path, case, vector=False,
                               backend='scalar'):
        """Check the production physical-provider wrapper before alias routing.

        The backend is compiled into the wrapper exactly as the run-card
        nlo_real_backend is (there is no run-time override), so each backend
        gets its own executable."""
        subproc = pjoin(output_path, 'SubProcesses', case['subprocess'])
        row_to_me, local_counts = self._real_me_contract(subproc)
        suffix = '_vec' if vector else ''
        source_name = 'check_real_oracle%s.f' % suffix
        object_name = 'check_real_oracle%s.o' % suffix
        executable = pjoin(subproc, 'check_real_oracle%s_%s' % (
            suffix, backend))
        with open(pjoin(subproc, source_name), 'w') as stream:
            stream.write(self._driver_source(
                row_to_me, local_counts, vector=vector,
                scalar_name=self._physical_scalar_name(subproc)))

        misc.compile(cwd=pjoin(output_path, 'Source'))
        backend_option = 'NLO_REAL_RUNTIME_BACKEND=%s' % backend
        subprocess.check_call(
            ['make', 'nlo_real_offload.o', backend_option, '-j2'],
            cwd=subproc)
        matrix_objects = [
            os.path.basename(path)[:-2] + '.o' for path in
            sorted(glob.glob(pjoin(subproc, 'matrix_*.f')))]
        objects = matrix_objects + [
            'real_me_chooser.o', 'splitorders_stuff.o', object_name,
            'nlo_real_offload.o']
        objects += self._native_real_link_objects(output_path, subproc)
        misc.compile(objects + [backend_option], cwd=subproc)
        subprocess.check_call(
            ['gfortran', '-o', executable] + objects + [
                '-L%s' % pjoin(output_path, 'lib'), '-ldhelas', '-lmodel']+
                native_bridge_link_flags(subproc)+
                self._native_real_link_flags(output_path, subproc),
            cwd=subproc)
        return executable

    def _run_routed_driver(self, executable, case):
        output = subprocess.check_output(
            [executable], input=self._driver_input(case['points']),
            cwd=self.tmpdir, stderr=subprocess.STDOUT)
        text = output.decode(errors='replace')
        records = self._parse_driver_output(
            text, len(case['split_order_names']))
        self._assert_case(case, records)
        return text

    def _run_production_case(self, name, test_fallback=False):
        case = self.oracle['cases'][name]
        output_path = self._generate_case(name, madmatrix=True)
        case = self._case_for_output(name, output_path)
        scalar = self._compile_routed_driver(output_path, case)
        scalar_output = self._run_routed_driver(scalar, case)
        self.assertIn('MG7 NLO real offload initialized: backend=scalar',
                      scalar_output)

        vector = self._compile_routed_driver(output_path, case, vector=True)
        vector_output = self._run_routed_driver(vector, case)
        self.assertIn('MG7 NLO real offload initialized: backend=scalar',
                      vector_output)

        if test_fallback:
            fortran = self._compile_routed_driver(
                output_path, case, backend='fortran')
            fallback = self._run_routed_driver(fortran, case)
            self.assertIn('using Fortran fallback', fallback)
            self.assertNotIn('offload initialized', fallback)

            process_path = pjoin(
                output_path, 'SubProcesses', case['subprocess'])
            with open(pjoin(process_path, 'nlo_real_manifest.json')) as stream:
                manifest = json.load(stream)
            real = manifest['real_matrix_elements'][1]
            library = self._library_path(output_path, manifest, real, 'scalar')
            missing = library + '.missing'
            os.rename(library, missing)
            try:
                partial = self._run_routed_driver(scalar, case)
            finally:
                os.rename(missing, library)
            self.assertIn('real ME %d is unavailable' % real['id'], partial)
            self.assertNotIn('real ME 1 is unavailable', partial)

            # The launched run takes its backend from the run card only.
            run_card_path = pjoin(output_path, 'Cards', 'run_card.dat')
            with open(run_card_path) as stream:
                run_card = stream.read()
            run_card, count = re.subn(
                r'(?m)^\s*\S+\s*=\s*nlo_real_backend\b',
                ' scalar = nlo_real_backend', run_card)
            self.assertEqual(count, 1)
            with open(run_card_path, 'w') as stream:
                stream.write(run_card)
            launch = self._new_cmd()
            self._run(launch, 'launch %s -f' % output_path)
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
            elif tag == 'scalar_global':
                current.setdefault('scalar_global', []).append(
                    cls._fortran_float(tokens[2]))
            elif tag == 'scalar_sum':
                current['scalar_summed'] = cls._fortran_float(tokens[1])
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
        case = self._case_for_output(name, output_path)
        actual = self._compile_and_run(output_path, case)
        self._assert_case(case, actual)

    def _run_fortran_vector_case(self, name):
        """The batched Fortran physical provider reproduces the oracle
        and, bit for bit, the scalar physical provider.

        Once the scalar routine has set the helicity filter, the batched one
        must apply it, including the reuse of a helicity identical to an
        earlier one: a batched sum which recomputes those helicities differs
        from the scalar one in the last bits. On the first point the scalar
        routine still computes every helicity, so the exact comparison
        starts at the second point."""
        case = self.oracle['cases'][name]
        output_path = self._generate_case(name)
        case = self._case_for_output(name, output_path)
        actual = self._compile_and_run(output_path, case,
                                       fortran_vector=True)
        self._assert_case(case, actual)
        for record in actual:
            if record['point'] == 1:
                continue
            label = 'point %d FKS row %d' % (
                record['point'], record['fks_row'])
            self.assertEqual(record['summed'], record['scalar_summed'],
                             label)
            self.assertEqual(
                [component['value'] for component in
                 record['global_squared_orders']],
                record['scalar_global'], label)

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
    def _host_simd_backends():
        """Exercise every available split SIMD width, including odd batches."""
        flags = ''
        try:
            with open('/proc/cpuinfo') as stream:
                flags = stream.read()
        except IOError:
            pass
        backends = []
        if 'avx2' in flags:
            backends.append(('simd_256', 4))
        if 'avx512f' in flags:
            backends.append(('simd_512', 8))
        if not backends and 'sse4_2' in flags:
            backends.append(('simd_128', 2))
        if not backends:
            raise unittest.SkipTest('no supported host SIMD backend')
        return backends

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
C     Initialize the native EW coupling for the caller-side order weights.
      CALL SETPARA('param_card.dat')
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
            'NLO_REAL_RUNTIME_BACKEND=%s' % backend, '-j2'], cwd=subproc)
        matrix_objects = [
            os.path.basename(path)[:-2] + '.o' for path in
            sorted(glob.glob(pjoin(subproc, 'matrix_*.f')))]
        objects = matrix_objects + [
            'real_me_chooser.o', 'splitorders_stuff.o',
            'check_real_batch.o', 'nlo_real_offload.o']
        misc.compile(objects[:-1], cwd=subproc)
        objects += self._native_real_link_objects(output_path, subproc)
        executable = pjoin(subproc, 'check_real_batch')
        subprocess.check_call(
            ['gfortran', '-o', executable] + objects + [
                '-L%s' % pjoin(output_path, 'lib'), '-ldhelas', '-lmodel']+
                native_bridge_link_flags(subproc)+
                self._native_real_link_flags(output_path, subproc), cwd=subproc)

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
            'MG7_NLO_REAL_TRACE': '1',
        })
        output = subprocess.check_output(
            [executable], input=('\n'.join(values) + '\n').encode(),
            cwd=self.tmpdir, env=environment, stderr=subprocess.STDOUT)
        return output.decode(errors='replace')

    def _run_simd_case(self, name, check_compaction=False):
        case = self.oracle['cases'][name]
        output_path = self._generate_case(name, madmatrix=True)
        case = self._case_for_output(name, output_path)
        for backend, width in self._host_simd_backends():
            with self.subTest(backend=backend):
                self._check_simd_case(output_path, case, backend, width,
                                      check_compaction)

    def _check_simd_case(self, output_path, case, backend, width,
                         check_compaction=False):
        process_path = pjoin(
            output_path, 'SubProcesses', case['subprocess'])
        with open(pjoin(process_path, 'nlo_real_manifest.json')) as stream:
            manifest = json.load(stream)
        rows = {row['fks_row']: row for row in manifest['fks_rows']}
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
            selected_rows = manifest['fks_rows']
            # Include restored four-quark sectors as well as the first real
            # ME; every physical row and all sixteen points need SIMD coverage.
            counts = sorted(set((1, max(1, width - 1), width,
                                 width + 1, 2 * width - 1, len(case['points']))))
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

    def _check_cuda_dense_strided(self, output_path, case, manifest):
        """Compare the dense NLO route with generic strided UMAMI calls.

        Reuse the generated libraries and trusted per-flavour oracle. Vary G,
        physical flavour, and growing/shrinking batch sizes on each handle;
        padding sentinels catch writes outside the requested output slice.
        """
        points = {point['id']: point for point in case['points']}
        for real in manifest['real_matrix_elements']:
            records = [record for record in case['records']
                       if record['real_me_id'] == real['id']]
            library = ctypes.CDLL(self._library_path(output_path, manifest, real, 'cuda'))
            library.umami_initialize.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
            library.umami_initialize.restype = ctypes.c_int
            library.umami_free.argtypes = [ctypes.c_void_p]
            library.umami_free.restype = ctypes.c_int
            library.umami_matrix_element.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                ctypes.c_size_t, ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t, ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_void_p)]
            library.umami_matrix_element.restype = ctypes.c_int
            ldd = subprocess.check_output(['ldd', library._name], text=True)
            runtimes = [line.split('=>')[1].split()[0]
                        for line in ldd.splitlines()
                        if 'libcudart' in line and '=>' in line]
            runtime = ctypes.CDLL(runtimes[0] if runtimes else
                                  ctypes.util.find_library('cudart') or
                                  '/usr/local/cuda/lib64/libcudart.so')
            handle = ctypes.c_void_p()
            self.assertEqual(library.umami_initialize(ctypes.byref(handle),
                pjoin(output_path, 'Cards', 'param_card.dat').encode()), 0)
            try:
                for sequence, count in enumerate((1, 3, 255, 256, 257, 513, 3, 1)):
                    batch = [records[(i + 7 * sequence) % len(records)]
                             for i in range(count)]
                    answers = []
                    # Dense Fortran host bridge, generic dense device buffers,
                    # strided default stream, and explicit asynchronous stream.
                    for stride, offset, device, explicit_stream in (
                            (count, 0, False, False), (count, 0, True, False),
                            (count + 7, 3, True, False), (count + 7, 3, True, True)):
                        nexternal = manifest['nexternal']
                        momenta = (ctypes.c_double * (4 * nexternal * stride))()
                        coupling = (ctypes.c_double * stride)()
                        flavour = (ctypes.c_uint32 * stride)()
                        sentinel = -123456789.25
                        output = (ctypes.c_double * stride)(*([sentinel] * stride))
                        for event, record in enumerate(batch):
                            point = points[record['point']]
                            coupling[offset + event] = point['g_strong']
                            flavour[offset + event] = record['fortran_flavour_index'] - 1
                            for particle in range(nexternal):
                                for component in range(4):
                                    momenta[stride * (nexternal * component + particle)
                                            + offset + event] = point['momenta'][particle][component]
                        keys = (ctypes.c_int * 3)(0, IN_G_STRONG, 2)
                        inputs = (ctypes.c_void_p * 3)(*[
                            ctypes.cast(array, ctypes.c_void_p)
                            for array in (momenta, coupling, flavour)])
                        output_keys = (ctypes.c_int * 1)(OUT_SQUARED_ORDERS)
                        outputs = (ctypes.c_void_p * 1)(
                            ctypes.cast(output, ctypes.c_void_p))
                        pointers = []
                        stream = ctypes.c_void_p()
                        try:
                            if device:
                                # The generic GPU ABI reads/writes device
                                # buffers and returns before kernels finish.
                                for host in (momenta, coupling, flavour, output):
                                    pointer = ctypes.c_void_p()
                                    size = ctypes.c_size_t(ctypes.sizeof(host))
                                    self.assertEqual(runtime.cudaMalloc(
                                        ctypes.byref(pointer), size), 0)
                                    pointers.append(pointer)
                                    self.assertEqual(runtime.cudaMemcpy(
                                        pointer, ctypes.cast(host, ctypes.c_void_p),
                                        size, 1), 0)
                                inputs = (ctypes.c_void_p * 3)(
                                    *[pointer.value for pointer in pointers[:3]])
                                outputs[0] = pointers[3].value
                            if explicit_stream:
                                self.assertEqual(runtime.cudaStreamCreateWithFlags(
                                    ctypes.byref(stream), 1), 0)
                                output_keys = (ctypes.c_int * 2)(OUT_SQUARED_ORDERS, 5)
                                outputs = (ctypes.c_void_p * 2)(
                                    pointers[3].value, stream.value)
                            self.assertEqual(library.umami_matrix_element(
                                handle, count, stride, offset, 3, keys, inputs,
                                len(output_keys), output_keys, outputs), 0)
                            if device:
                                self.assertEqual(runtime.cudaStreamSynchronize(stream), 0)
                                self.assertEqual(runtime.cudaMemcpy(
                                    ctypes.cast(output, ctypes.c_void_p), pointers[3],
                                    ctypes.c_size_t(ctypes.sizeof(output)), 2), 0)
                        finally:
                            if stream:
                                self.assertEqual(runtime.cudaStreamDestroy(stream), 0)
                            for pointer in pointers:
                                self.assertEqual(runtime.cudaFree(pointer), 0)
                        values = list(output)[offset:offset + count]
                        for actual, record in zip(values, batch):
                            expected = record['local_squared_orders'][0]['value']
                            self._assert_close(actual, expected, scale=abs(expected),
                                label='CUDA dense/strided real %d count %d' % (real['id'], count))
                        self.assertTrue(all(value == sentinel
                            for i, value in enumerate(output)
                            if not offset <= i < offset + count))
                        answers.append(values)
                    for answer in answers[1:]:
                        self.assertEqual(answers[0], answer)
            finally:
                self.assertEqual(library.umami_free(handle), 0)

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

        self._check_cuda_dense_strided(output_path, case, manifest)

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
            self.assertFalse(glob.glob(self._library_path(output_path, manifest, real, '*', fptype='*', libtag='*')))

    def _run_cuda_mixed_case(self):
        """Use CUDA only for one-order reals and log all split-order fallback."""
        self._require_cuda()
        case = self.oracle['cases']['grouped_mixed_wj']
        output_path = self._generate_case('grouped_mixed_wj', madmatrix=True)
        case = self._case_for_output('grouped_mixed_wj', output_path)
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
        text = self._run_routed_driver(executable, case)
        self.assertIn('MG7 NLO real offload initialized: backend=cuda', text)
        for real in unsupported:
            self.assertIn(
                'MG7 NLO real GPU fallback: real_me_id=%d' % real['id'],
                text)
            library = self._library_path(output_path, manifest, real, 'cuda')
            self.assertFalse(os.path.exists(library), library)
        for real in supported:
            library = self._library_path(output_path, manifest, real, 'cuda')
            self.assertTrue(os.path.isfile(library), library)

    def _run_madmatrix_case(self, name, low_memory=False):
        """Generate, build, relocate, and replay every scalar bridge row."""

        case = self.oracle['cases'][name]
        output_path = self._generate_case(
            name, madmatrix=True, low_memory=low_memory)
        case = self._case_for_output(name, output_path)
        process_path = pjoin(
            output_path, 'SubProcesses', case['subprocess'])
        manifest_path = pjoin(process_path, 'nlo_real_manifest.json')
        with open(manifest_path) as stream:
            manifest = json.load(stream)

        self.assertEqual(manifest['version'], 2)
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

    def _check_real_umami_order_blocks(self, output, manifest, references, backend='scalar'):
        # Generated common libraries have a model/backend SONAME shared across
        # outputs, but their coupling layouts can differ. A fresh interpreter
        # prevents ctypes handles from another test binding this output's DSO.
        subprocess.run([
            sys.executable, '-c',
            'import json,sys; from tests.acceptance_tests.test_nlo_real_offload_contract '
            'import TestNLORealFortranOracle as T; '
            'T()._check_real_umami_order_blocks_inprocess(*json.load(sys.stdin))'],
            input=json.dumps([output, manifest, references, backend]).encode(),
            check=True, cwd=MG5DIR)

    def _check_real_umami_order_blocks_inprocess(self, output, manifest, references, backend):
        """Compare raw strided order blocks at independent couplings to native Fortran."""
        real = manifest['real_matrix_elements'][0]
        library = ctypes.CDLL(self._library_path(output, manifest, real, backend))
        library.umami_get_meta.argtypes = [ctypes.c_int, ctypes.c_void_p]
        library.umami_matrix_element.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p)]
        names, orders = ctypes.c_char_p(), ctypes.c_int()
        self.assertEqual(library.umami_get_meta(11, ctypes.byref(names)), 0)
        self.assertEqual(names.value, b'QCD,QED')
        self.assertEqual(library.umami_get_meta(9, ctypes.byref(orders)), 0)
        powers = (ctypes.c_int * (2 * orders.value))()
        selected = (ctypes.c_bool * orders.value)()
        self.assertEqual(library.umami_get_meta(10, powers), 0)
        self.assertEqual(library.umami_get_meta(17, selected), 0)
        self.assertTrue(all(selected), 'NLO real providers expose all local orders')
        self.assertEqual(list(zip(powers[:orders.value], powers[orders.value:])),
                         list(map(tuple, real['local_squared_orders'])))
        nlegs = manifest['nexternal']
        charges = (ctypes.c_double * nlegs)()
        colors = (ctypes.c_int * nlegs)()
        self.assertEqual(library.umami_get_meta(12, colors), 0)
        self.assertEqual(library.umami_get_meta(13, charges), 0)
        self.assertEqual(list(colors), [3, -3, 3, -3, 8])
        self.assertEqual(list(charges), [2/3, -2/3, 2/3, -2/3, 0.])
        card = pjoin(output, 'Cards', 'param_card.dat')
        handle = ctypes.c_void_p()
        self.assertEqual(library.umami_initialize(ctypes.byref(handle), card.encode()), 0)
        try:
            aewm1 = ctypes.c_double()
            self.assertEqual(library.umami_get_parameter(handle, b'aEWM1', ctypes.byref(aewm1), None), 0)
            card_e = math.sqrt(4. * math.pi / aewm1.value)
            # Only lanes 1 and 2 belong to the batch. Fill the borders with sentinels.
            stride, count, offset = 4, 2, 1
            momenta = (ctypes.c_double * (4 * nlegs * stride))()
            gs = (ctypes.c_double * stride)(0., 1., 0.7, 0.)
            ew = (ctypes.c_double * stride)(0., 1., 0.27, 0.)
            total = (ctypes.c_double * stride)(*([-1.] * stride))
            blocks = (ctypes.c_double * (stride * orders.value))(*([-1.] * stride * orders.value))
            for lane, point in enumerate(references, offset):
                for ipart, vector in enumerate(point['momenta']):
                    for mu, value in enumerate(vector):
                        momenta[(mu * nlegs + ipart) * stride + lane] = value
            ins = (ctypes.c_void_p * 3)(*map(ctypes.addressof, (momenta, gs, ew)))
            outs = (ctypes.c_void_p * 2)(*map(ctypes.addressof, (total, blocks)))
            self.assertEqual(library.umami_matrix_element(
                handle, count, stride, offset, 3, (ctypes.c_int * 3)(0, IN_G_STRONG, IN_G_EW), ins,
                2, (ctypes.c_int * 2)(0, OUT_SQUARED_ORDERS), outs), 0)
            for lane, point in enumerate(references, offset):
                expected = [component['value'] * (gs[lane] / point['g_strong']) ** qs
                            * (ew[lane] / card_e) ** qe for component, (qs, qe)
                            in zip(point['local_squared_orders'], real['local_squared_orders'])]
                scale = max(map(abs, expected))
                for order, value in enumerate(expected):
                    self._assert_close(blocks[order * stride + lane], value, scale=scale)
                self._assert_close(total[lane], sum(expected), scale=scale)
            # Installing a cached narrow mask must not narrow a later order
            # decomposition, even after this instance has already kept all helicities.
            library.madmatrix_set_helicity_choice.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            self.assertEqual(library.madmatrix_set_helicity_choice(
                handle, (ctypes.c_int * 2)(0, 0), 2), 0)
            saved_blocks, saved_total = list(blocks), list(total)
            self.assertEqual(library.umami_matrix_element(
                handle, count, stride, offset, 3, (ctypes.c_int * 3)(0, IN_G_STRONG, IN_G_EW), ins,
                2, (ctypes.c_int * 2)(0, OUT_SQUARED_ORDERS), outs), 0)
            for actual, expected in zip(blocks, saved_blocks):
                self._assert_close(actual, expected)
            for actual, expected in zip(total, saved_total):
                self._assert_close(actual, expected)
            # The ordinary total-only path (no G_EW) must agree with the
            # unscaled full block sum after a component request as well.
            self.assertEqual(library.umami_matrix_element(
                handle, count, stride, offset, 2, (ctypes.c_int * 2)(0, IN_G_STRONG), ins,
                1, (ctypes.c_int * 1)(0), outs), 0)
            for lane, point in enumerate(references, offset):
                expected = [component['value'] * (gs[lane] / point['g_strong']) ** qs
                            for component, (qs, _) in zip(
                                point['local_squared_orders'], real['local_squared_orders'])]
                self._assert_close(total[lane], sum(expected), scale=max(map(abs, expected)))
            self.assertEqual(list(total)[::3], [-1., -1.])
            for order in range(orders.value):
                self.assertEqual(blocks[order * stride], -1.)
                self.assertEqual(blocks[order * stride + stride - 1], -1.)
        finally:
            self.assertEqual(library.umami_free(handle), 0)

    def test_restricted_orders_scalar_and_batch(self):
        """The bridge returns all local orders; production must select NLO orders."""
        case = dict(self.oracle['cases']['grouped_qcd_ttx'])
        case['subprocess'] = 'P0_uux_ttx'
        case['points'] = case['points'][:2]
        output_path = pjoin(self.tmpdir, 'restricted_orders')
        cmd = self._new_cmd()
        for line in ('import model loop_sm',
                     'set apply_flavor_grouping False --no_save',
                     'generate u u~ > t t~ QCD^2=2 QED^2=2 [real=QCD]',
                     'output standalone_fortran --fks --limits %s --me_exporter=mg7' % output_path):
            self._run(cmd, line)
        subproc = pjoin(output_path, 'SubProcesses', case['subprocess'])
        records = {}
        for backend in ('fortran', 'scalar'):
            executable = self._compile_routed_driver(output_path, case, backend=backend)
            raw = subprocess.check_output(
                [executable], input=self._driver_input(case['points']),
                cwd=subproc, stderr=subprocess.STDOUT).decode()
            if backend == 'scalar':
                self.assertIn('offload initialized: backend=scalar', raw)
                self.assertNotIn('Fortran fallback', raw)
            records[backend] = self._parse_driver_output(raw, 2)
        self.assertEqual(len(records['fortran']), 24)
        self.assertEqual(len(records['scalar']), 24)
        for reference, actual in zip(records['fortran'], records['scalar']):
            scale = max(abs(c['value']) for c in reference['local_squared_orders'])
            self.assertNotEqual(reference['local_squared_orders'][0]['value'], 0.)
            self.assertEqual(reference['global_squared_orders'][-3]['value'], 0.)
            self.assertEqual(reference['global_squared_orders'][-1]['value'], 0.)
            self._assert_close(actual['summed'], reference['summed'], scale=scale)
            for a, b in zip(actual['global_squared_orders'], reference['global_squared_orders']):
                self._assert_close(a['value'], b['value'], scale=scale)

        with open(pjoin(subproc, 'nlo_real_manifest.json')) as stream:
            manifest = json.load(stream)
        row = manifest['fks_rows'][0]
        reference = [r for r in records['fortran'] if r['fks_row'] == row['fks_row']]
        self._check_real_umami_order_blocks(output_path, manifest, reference)
        for backend, _ in self._host_simd_backends():
            with self.subTest(backend=backend):
                subprocess.check_call([
                    'make', '-f', 'nlo_real.mk', 'BACKEND=%s' % backend,
                    'FPTYPE=d', '-j2'], cwd=subproc)
                self._check_real_umami_order_blocks(
                    output_path, manifest, reference, backend=backend)
        # Include a masked lane: compacted multi-order batches must also filter.
        reference.insert(1, reference[0])
        text = self._compile_batch_wrapper_driver(
            output_path, case, 'scalar', row, reference, [True, False, True])
        self.assertIn('backend=scalar', text)
        self.assertIn('event_count=2', text)
        self.assertNotIn('Fortran fallback', text)
        weights, components = {}, {}
        for line in text.splitlines():
            fields = line.split()
            if fields and fields[0] == 'BATCH_LANE':
                weights[int(fields[1])] = float(fields[2])
            elif fields and fields[0] == 'BATCH_GLOBAL':
                components.setdefault(int(fields[1]), []).append(float(fields[3]))
        self.assertEqual(sorted(weights), [1, 2, 3])
        for lane, ref in enumerate(reference, 1):
            scale = max(abs(c['value']) for c in ref['local_squared_orders'])
            self._assert_close(weights[lane], ref['summed'] if lane != 2 else 0., scale=scale)
            self.assertEqual(len(components[lane]), len(ref['global_squared_orders']))
            for value, component in zip(components[lane], ref['global_squared_orders']):
                self._assert_close(value, component['value'] if lane != 2 else 0., scale=scale)

    def test_grouped_qcd_ttx_real_components(self):
        self._run_case('grouped_qcd_ttx')

    def test_grouped_mixed_wj_real_components(self):
        self._run_case('grouped_mixed_wj')

    def test_grouped_qcd_ttx_fortran_vector_real(self):
        self._run_fortran_vector_case('grouped_qcd_ttx')

    def test_grouped_mixed_wj_fortran_vector_real(self):
        self._run_fortran_vector_case('grouped_mixed_wj')

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
