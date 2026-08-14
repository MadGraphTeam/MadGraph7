################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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

from __future__ import absolute_import

import os
import re
import shutil
import subprocess
import tempfile
import unittest
import logging
logger = logging.getLogger('madgraph.madevent')

import madgraph.interface.master_interface as cmd_interface
import madgraph.various.process_checks as process_checks


pjoin = os.path.join


def _sanitize_process_name(process):
    return re.sub(r'[^A-Za-z0-9]+', '_', process).strip('_').lower()


def matrix_element_consistency_test_factory(process, model='sm', tolerance=1e-6):
    def test(self):
        self.check_process(process, model=model, tolerance=tolerance)
    test.__name__ = 'test_%s' % _sanitize_process_name(process)
    test.__doc__ = 'Check standalone and madevent matrix elements agree for %s.' % process
    return test


class StandaloneMadeventMatrixElementConsistency(unittest.TestCase):

    debugging = getattr(unittest, 'debug', False)

    def setUp(self):
        self.cmd = cmd_interface.MasterCmd()
        self.cmd.no_notification()
        if not self.debugging:
            self.tmpdir = tempfile.mkdtemp(prefix='amc')
        else:
            self.tmpdir = tempfile.mkdtemp(prefix='amc_debug_')
        self.standalone_dir = pjoin(self.tmpdir, 'StandaloneProcess')
        self.madevent_dir = pjoin(self.tmpdir, 'MadEventProcess')

    def tearDown(self):
        if not self.debugging and os.path.isdir(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def do(self, line):
        self.cmd.exec_cmd(line)

    def check_process(self, process, model='sm', tolerance=1e-6):
        """Every backend must return the same matrix element per flavor.

        The reference is the plain (--use_crossing=False) fortran standalone.
        Every other backend is compared to it flavor by flavor, matched by the
        PDG tuple it prints (not by index -- the flavor ordering differs between
        backends, and a crossing-folded backend may expose extra flavors):

          - fortran madevent, ungrouped, --use_crossing=False (the original
            check; only the ungrouped ME exporter does not support crossing);
          - fortran standalone WITH crossing (the crossing-aware SMATRIX must
            reproduce the plain per-flavor matrix element);
          - fortran madevent, grouped, WITH crossing
            (ProcessExporterFortranMEGroup, which does support crossing);
          - standalone_mg7 (the madmatrix / cudacpp CPU-SIMD backend).
        """
        self.do('set automatic_html_opening False')
        self.do('set group_subprocesses False')
        self.do('set apply_flavor_grouping True')
        self.do('set zerowidth_tchannel False')
        self.do('import model %s' % model)

        # -- Reference: plain fortran standalone (crossing machinery off) -------
        self.do('generate %s --use_crossing=False' % process)
        generated_process = self.cmd._curr_amps[0].get('process')
        seeded_phase_space = self._get_seeded_phase_space(generated_process)

        ref_root = pjoin(self.tmpdir, 'standalone_plain')
        self.do('output standalone %s -f' % ref_root)
        ref_sub = self._get_single_subprocess_dir(pjoin(ref_root, 'SubProcesses'))
        ref_rows, printed_phase_space = self._run_standalone(ref_sub)
        self._assert_phase_space_reasonable(
            printed_phase_space, seeded_phase_space, ref_sub)
        reference = self._rows_by_pdg(ref_rows, ref_sub)

        # -- (1) fortran madevent, ungrouped, crossing off (the original check) -
        # madevent enumerates flavors in the same order as the standalone check
        # (its GET_FLAVOR returns group indices, not PDGs, so it is matched to
        # the reference by that shared IFLAV order rather than by PDG).
        me_root = pjoin(self.tmpdir, 'madevent_plain')
        self.do('output madevent %s -f' % me_root)
        me_sub = self._get_single_subprocess_dir(pjoin(me_root, 'SubProcesses'))
        me_by_iflav = self._run_hacked_madevent(me_root, me_sub, seeded_phase_space)
        self._compare_by_iflav(
            process, 'madevent (ungrouped, crossing off)',
            ref_rows, me_by_iflav, tolerance)

        # -- (2) fortran standalone WITH crossing -------------------------------
        self.do('generate %s --use_crossing=True' % process)
        sacross_root = pjoin(self.tmpdir, 'standalone_crossing')
        self.do('output standalone %s -f' % sacross_root)
        sacross_sub = self._get_single_subprocess_dir(
            pjoin(sacross_root, 'SubProcesses'))
        sacross_rows, _ = self._run_standalone(sacross_sub)
        self._compare_to_reference(
            process, 'standalone (crossing on)',
            reference, self._rows_by_pdg(sacross_rows, sacross_sub), tolerance)

        # -- (3) fortran madevent, grouped, WITH crossing (MEGroup) -------------
        self.do('set group_subprocesses True')
        self.do('generate %s --use_crossing=True' % process)
        meg_root = pjoin(self.tmpdir, 'madevent_group_crossing')
        self.do('output madevent %s -f' % meg_root)
        self.do('set group_subprocesses False')
        meg_sub = self._get_single_subprocess_dir(pjoin(meg_root, 'SubProcesses'))
        meg_by_iflav = self._run_hacked_madevent(
            meg_root, meg_sub, seeded_phase_space,
            smatrix_name='SMATRIX1', make_target='madevent_forhel')
        self._compare_by_iflav(
            process, 'madevent (grouped, crossing on)',
            ref_rows, meg_by_iflav, tolerance)

        # -- (4) standalone_mg7 (madmatrix / cudacpp CPU-SIMD) ------------------
        # Skipped (not failed) if no C++ compiler or the madmatrix build
        # toolchain is unavailable. Matched by flavor order like madevent: the
        # extended flavor id is cross*nflav+flav, so the base flavors are ids
        # 0..nflav-1, in the same order as the standalone check.
        mg7_by_iflav = self._run_standalone_mg7(process, seeded_phase_space, ref_rows)
        if mg7_by_iflav is not None:
            self._compare_by_iflav(
                process, 'standalone_mg7', ref_rows, mg7_by_iflav, tolerance)

    def _rows_by_pdg(self, rows, subproc_dir):
        """{PDG tuple -> matrix element} from _extract_standalone_flavors rows."""
        by_pdg = {}
        for row in rows:
            by_pdg[tuple(row['pdg'])] = row['value']
        self.assertEqual(len(by_pdg), len(rows),
                         'Duplicate PDG flavor rows in %s' % subproc_dir)
        return by_pdg

    def _compare_to_reference(self, process, label, reference, other, tolerance):
        """Assert `other` reproduces every reference flavor (matched by PDG)."""
        self.assertTrue(other, 'No matrix elements produced by %s for %s'
                        % (label, process))
        for pdg, ref_me in reference.items():
            self.assertIn(pdg, other,
                          'Flavor %s missing from %s for %s' % (pdg, label, process))
            other_me = other[pdg]
            scale = max(abs(ref_me), abs(other_me), 1e-99)
            rel = abs(ref_me - other_me) / scale
            logger.debug('%s flavor=%s: diff=%f%%', label, pdg, 100 * rel)
            self.assertLessEqual(
                rel, tolerance,
                'Incompatible matrix elements for %s flavor=%s (%s): '
                'reference=%s %s=%s'
                % (process, pdg, label, ref_me, label, other_me))

    def _get_single_subprocess_dir(self, root_dir):
        subproc_dirs = [pjoin(root_dir, name) for name in sorted(os.listdir(root_dir))
                        if name.startswith('P') and os.path.isdir(pjoin(root_dir, name))]
        self.assertEqual(len(subproc_dirs), 1,
                         'Expected a single subprocess directory in %s, got %s'
                         % (root_dir, subproc_dirs))
        return subproc_dirs[0]

    def _run_standalone(self, subproc_dir):
        retcode = self._call_with_optional_redirection(['make', 'check'], subproc_dir)
        self.assertEqual(retcode, 0, 'Failed to compile standalone check in %s' % subproc_dir)

        output = subprocess.Popen(['./check', '1000'],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  cwd=subproc_dir).communicate()[0].decode()

        ps_pattern = re.compile(
            r'^\s*\d+\s+'
            r'(?P<e>[\d\.eE\+-]+)\s+'
            r'(?P<px>[\d\.eE\+-]+)\s+'
            r'(?P<py>[\d\.eE\+-]+)\s+'
            r'(?P<pz>[\d\.eE\+-]+)',
            re.MULTILINE)
        phase_space = [[float(match.group(name)) for name in ('e', 'px', 'py', 'pz')]
                       for match in ps_pattern.finditer(output)]
        self.assertTrue(phase_space, 'No phase-space point found in %s' % subproc_dir)
        return self._extract_standalone_flavors(output, subproc_dir), phase_space

    def _get_seeded_phase_space(self, process_obj, energy=1000.0):
        evaluator = process_checks.MatrixElementEvaluator(
            process_obj.get('model'), cmd=self.cmd)
        phase_space = process_checks._get_seeded_python_momenta(
            process_obj, evaluator, energy)
        self.assertTrue(phase_space,
                        'Failed to generate seeded phase-space point for %s'
                        % process_obj.nice_string())
        return phase_space

    def _assert_phase_space_reasonable(self, printed, seeded, subproc_dir):
        self.assertEqual(len(printed), len(seeded),
                         'Mismatch in particle count for printed/seeded phase-space in %s'
                         % subproc_dir)
        for ipart, (printed_vec, seeded_vec) in enumerate(zip(printed, seeded), start=1):
            for icomp, (printed_val, seeded_val) in enumerate(zip(printed_vec, seeded_vec)):
                tolerance = max(1e-3, 1e-6 * max(abs(seeded_val), 1.0))
                self.assertLessEqual(
                    abs(printed_val - seeded_val), tolerance,
                    'Printed phase-space seems inconsistent in %s at particle=%s component=%s: '
                    'printed=%s seeded=%s'
                    % (subproc_dir, ipart, icomp, printed_val, seeded_val))

    def _compare_by_iflav(self, process, label, ref_rows, by_iflav, tolerance):
        """Assert a madevent backend reproduces the reference, matched by IFLAV.

        The standalone check loops flavors in the same order that the madevent
        driver loops IFLAV, so reference row i (1-based) is madevent IFLAV i.
        A grouped/crossing madevent may expose extra flavors past the reference
        count; only the reference flavors are required to agree.
        """
        self.assertTrue(by_iflav, 'No matrix elements produced by %s for %s'
                        % (label, process))
        for iflav, row in enumerate(ref_rows, start=1):
            self.assertIn(iflav, by_iflav,
                          'Missing IFLAV=%s (flavor %s) from %s for %s'
                          % (iflav, row['pdg'], label, process))
            ref_me = row['value']
            other_me = by_iflav[iflav]
            scale = max(abs(ref_me), abs(other_me), 1e-99)
            rel = abs(ref_me - other_me) / scale
            logger.debug('%s flavor=%s: diff=%f%%', label, row['pdg'], 100 * rel)
            self.assertLessEqual(
                rel, tolerance,
                'Incompatible matrix elements for %s flavor=%s iflav=%s (%s): '
                'reference=%s %s=%s'
                % (process, row['pdg'], iflav, label, ref_me, label, other_me))

    def _run_hacked_madevent(self, madevent_root, subproc_dir, phase_space,
                             smatrix_name='SMATRIX', make_target='madevent'):
        # The grouped exporter names its per-subprocess routine SMATRIX1 and
        # hides it behind helicity recycling (SMATRIX1 lives only in
        # matrix1_orig.f -> the 'madevent_forhel' target). The test processes
        # all group into a single subprocess (MAXSPROC=1), required by the
        # single-SMATRIX driver below.
        maxamps = pjoin(subproc_dir, 'maxamps.inc')
        if os.path.isfile(maxamps):
            match = re.search(r'MAXSPROC\s*=\s*(\d+)', open(maxamps).read())
            if match:
                self.assertEqual(int(match.group(1)), 1,
                                 'Driver assumes MAXSPROC=1 in %s' % subproc_dir)
        source_dir = pjoin(madevent_root, 'Source')
        retcode = self._call_with_optional_redirection(['make'], source_dir)
        self.assertEqual(retcode, 0, 'Failed to compile MadEvent source in %s' % source_dir)

        self._write_hacked_driver(pjoin(subproc_dir, 'driver.f'), phase_space,
                                  smatrix_name)

        retcode = self._call_with_optional_redirection(['make', make_target], subproc_dir)
        self.assertEqual(retcode, 0,
                         'Failed to compile hacked madevent (%s) in %s'
                         % (make_target, subproc_dir))

        output = subprocess.Popen(['./' + make_target],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  cwd=subproc_dir).communicate()[0].decode()
        return self._extract_madevent_by_iflav(output, subproc_dir)

    def _run_standalone_mg7(self, process, phase_space, ref_rows):
        """{IFLAV -> matrix element} for standalone_mg7 at the seeded momenta.

        Returns None (skip) if there is no C++ compiler or the madmatrix build
        toolchain cannot build check_sa.exe. check_sa.exe reads the external
        momenta from an LHE file (-e), so the same seeded point is used as for
        the fortran backends; the base flavors are the extended ids 0..nflav-1.
        """
        if not shutil.which(os.environ.get('CXX', 'g++')):
            return None
        outdir = pjoin(self.tmpdir, 'standalone_mg7')
        self.do('generate %s --use_crossing=True' % process)
        try:
            self.do('output standalone_mg7 %s -f' % outdir)
        except Exception:
            return None
        pdir = self._get_single_subprocess_dir(pjoin(outdir, 'SubProcesses'))

        nevt = 8
        lhe = pjoin(pdir, 'seeded.lhe')
        self._write_lhe_events(lhe, phase_space, nevt)

        rc = self._call_with_optional_redirection(
            ['make', '-j2', 'check_sa.exe'], pdir)
        if rc != 0:
            return None

        by_iflav = {}
        for iflav in range(1, len(ref_rows) + 1):
            flavor_id = iflav - 1  # extended id, cross=0 -> id = flavor (0-based)
            output = subprocess.Popen(
                ['./check_sa.exe', 'perf', '-v', '-f', str(flavor_id),
                 '-e', lhe, '1', str(nevt), '1'],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=pdir).communicate()[0].decode()
            values = re.findall(r'Matrix element =\s*([-\d.eE+]+)', output)
            self.assertTrue(values,
                            'No matrix element from standalone_mg7 flavor id %s '
                            'for %s:\n%s' % (flavor_id, process, output))
            by_iflav[iflav] = float(values[0])
        return by_iflav

    def _write_lhe_events(self, path, phase_space, nevents):
        """Write `nevents` identical minimal LHE events at `phase_space`.

        check_sa.exe only reads (E, px, py, pz) from each particle line; the
        pdg/status/colour columns are placeholders. The momenta are replicated
        across the SIMD page so every lane evaluates the seeded point.
        """
        def as_float(value):
            if isinstance(value, str):
                return float(value.replace('d', 'e').replace('D', 'E'))
            return float(value)

        npar = len(phase_space)
        lines = []
        for _ in range(nevents):
            lines.append('<event>')
            lines.append('%d 0 0.0 0.0 0.0 0.0' % npar)
            for momentum in phase_space:
                e, px, py, pz = (as_float(v) for v in momentum)
                lines.append('1 1 0 0 0 0 %.17E %.17E %.17E %.17E 0.0'
                             % (px, py, pz, e))
            lines.append('</event>')
        with open(path, 'w') as fsock:
            fsock.write('\n'.join(lines) + '\n')

    def _call_with_optional_redirection(self, command, cwd):
        if logger.isEnabledFor(logging.INFO):
            return subprocess.call(command, cwd=cwd)
        with open(os.devnull, 'w') as devnull:
            return subprocess.call(command, stdout=devnull, stderr=devnull, cwd=cwd)

    def _extract_standalone_flavors(self, output, subproc_dir):
        lines = output.splitlines()
        # The standalone driver may append a crossing-symmetry demonstration
        # (its own 'PDG ... / Matrix element = ...' lines for crossed
        # processes). Those are not the primary per-flavor output this test
        # compares against madevent, so stop at that section's header.
        for cut, line in enumerate(lines):
            if 'Crossing-symmetry example' in line:
                lines = lines[:cut]
                break
        standalone_rows = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith('PDG'):
                continue
            pdg_values = tuple(int(token) for token in re.findall(r'-?\d+', stripped))
            me_value = None
            for next_line in lines[index + 1:]:
                match = re.search(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)',
                                  next_line)
                if match:
                    me_value = float(match.group('value'))
                    break
                if next_line.strip().startswith('PDG'):
                    break
            if me_value is not None:
                standalone_rows.append({'pdg': pdg_values, 'value': me_value})
        self.assertTrue(standalone_rows, 'No flavor matrix elements found in %s' % subproc_dir)
        return standalone_rows

    def _extract_madevent_by_iflav(self, output, subproc_dir):
        lines = output.splitlines()
        by_iflav = {}
        current_iflav = None
        for line in lines:
            iflav_match = re.search(r'IFLAV\s*=\s*(\d+)', line)
            if iflav_match:
                current_iflav = int(iflav_match.group(1))
                continue
            me_match = re.search(r'Matrix element\s*=\s*(?P<value>[\d\.eE\+-]+)', line)
            if me_match and current_iflav is not None:
                by_iflav[current_iflav] = float(me_match.group('value'))
                current_iflav = None
        self.assertTrue(by_iflav, 'No madevent flavor matrix elements found in %s' % subproc_dir)
        return by_iflav

    def _write_hacked_driver(self, driver_path, phase_space, smatrix_name='SMATRIX'):
        lines = [
            '      PROGRAM DRIVER',
            '      use model_object',
            '      IMPLICIT NONE',
            "      INCLUDE 'genps.inc'",
            "      INCLUDE 'nexternal.inc'",
            "      INCLUDE 'maxamps.inc'",
            "      INCLUDE 'coupl.inc'",
            '      REAL*8 ZERO',
            '      PARAMETER (ZERO=0D0)',
            '      INTEGER SELECTED_HEL, SELECTED_COL, IFLAV, IVEC, J',
            '      INTEGER FLAVOR(NEXTERNAL)',
            '      REAL*8 P(0:3,NEXTERNAL), ANS',
            '      REAL*8 POL(2)',
            '      COMMON/TO_POLARIZATION/POL',
            '      INTEGER ISUM_HEL',
            '      LOGICAL MULTI_CHANNEL',
            '      COMMON/TO_MATRIX/ISUM_HEL, MULTI_CHANNEL',
            '      LOGICAL INIT_MODE',
            '      COMMON /TO_DETERMINE_ZERO_HEL/INIT_MODE',
            '      LOGICAL ALLOW_HELICITY_GRID_ENTRIES',
            '      COMMON/TO_ALLOW_HELICITY_GRID_ENTRIES/ALLOW_HELICITY_GRID_ENTRIES',
            '      INTEGER MINCFIG, MAXCFIG',
            '      COMMON/TO_CONFIGS/MINCFIG, MAXCFIG',
            '      INTEGER NB_SPIN_STATE(2)',
            '      COMMON /NB_HEL_STATE/ NB_SPIN_STATE',
            '      CHARACTER*30 PARAM_CARD_NAME',
            '      COMMON/TO_PARAM_CARD_NAME/PARAM_CARD_NAME',
            '      REAL*8 PMASS(NEXTERNAL)',
            '      COMMON/TO_MASS/PMASS',
            "      PARAM_CARD_NAME='param_card.dat'",
            '      CALL SETRUN',
            '      CALL SETPARA(PARAM_CARD_NAME)',
            "      INCLUDE 'pmass.inc'",
            '      POL(1)=1D0',
            '      POL(2)=1D0',
            '      ISUM_HEL=0',
            '      MULTI_CHANNEL=.FALSE.',
            '      HEL_PICKED=0',
            '      HEL_JACOBIAN=1D0',
            '      INIT_MODE=.FALSE.',
            '      ALLOW_HELICITY_GRID_ENTRIES=.FALSE.',
            '      MINCFIG=1',
            '      MAXCFIG=1',
            '      NB_SPIN_STATE(1)=2',
            '      NB_SPIN_STATE(2)=2',
            '      IVEC=1']

        for index, momentum in enumerate(phase_space):
            iparticle = index + 1
            for component, value in enumerate(momentum):
                if isinstance(value, str):
                    formatted_value = value.replace('e', 'd').replace('E', 'D')
                else:
                    formatted_value = ('%.17E' % value).replace('E', 'D')
                lines.append('      P(%d,%d)=%s' %
                             (component, iparticle, formatted_value))

        lines.extend([
            # The per-flavor PDG is read from leshouche.inc in python (madevent's
            # GET_FLAVOR returns group indices, and its signature differs between
            # the plain and grouped exporters), so the driver only emits IFLAV.
            '      DO IFLAV=1,MAXFLAVPERPROC',
            '         CALL %s(P, IFLAV, 0.5D0, 0.5D0, 1, IVEC, ANS,' % smatrix_name,
            '     $    SELECTED_HEL, SELECTED_COL)',
            "         WRITE(*,*) 'IFLAV = ', IFLAV",
            "         WRITE(*,*) 'Matrix element = ', ANS, ' GeV^',-(2*NEXTERNAL-8)",
            '      ENDDO',
            '      END',
            '',
            '      SUBROUTINE OPEN_FILE_LOCAL(LUN,FILENAME,FOPENED)',
            '      IMPLICIT NONE',
            '      INTEGER LUN',
            '      LOGICAL FOPENED',
            '      CHARACTER*(*) FILENAME',
            '      FOPENED=.FALSE.',
            "      OPEN(UNIT=LUN,FILE=FILENAME,STATUS='OLD',ERR=10)",
            '      FOPENED=.TRUE.',
            '      RETURN',
            ' 10   CONTINUE',
            '      RETURN',
            '      END',
            ''])

        with open(driver_path, 'w') as driver:
            driver.write('\n'.join(lines))


class TestStandaloneMadeventMatrixElementConsistency(
        StandaloneMadeventMatrixElementConsistency):
    pass    

    test_standalone_madevent_consistency_ee_ee = matrix_element_consistency_test_factory(
        'e+ e- > e+ e-', model='sm', tolerance=1e-6)

    test_standalone_madevent_consistency_ll_ll = matrix_element_consistency_test_factory(
        'l+ l- > l+ l-', model='sm', tolerance=1e-6)

    test_standalone_madevent_consistency_VBFZ_qqx = matrix_element_consistency_test_factory(
        '_quark _anti_quark > Z _quark _anti_quark QCD=0', model='sm', tolerance=1e-5)

    test_standalone_madevent_consistency_VBFZ_qq = matrix_element_consistency_test_factory(
        '_quark _quark > Z _quark _quark QCD=0', model='sm', tolerance=1e-5)
    
    test_standalone_madevent_consistency_VBFZ_qxqx = matrix_element_consistency_test_factory(
        '_anti_quark _anti_quark > Z _anti_quark _anti_quark QCD=0', model='sm', tolerance=1e-5)

    test_standalone_madevent_consistency_VBF_WW = matrix_element_consistency_test_factory(
        '_quark _quark > W+ W- _quark _quark QCD=0', model='sm', tolerance=1e-5)
    
    test_standalone_madevent_consistency_VBFH = matrix_element_consistency_test_factory(
        '_quark _anti_quark > H _quark _anti_quark QCD=0', model='sm', tolerance=1e-5)
    
    test_standalone_madevent_consistency_VBFHu = matrix_element_consistency_test_factory(
        'u u  > H u u QCD=0', model='sm', tolerance=1e-5)
    
    test_standalone_madevent_consistency_qq = matrix_element_consistency_test_factory(
        'u _quark  > u _quark QCD=0', model='sm', tolerance=1e-5)