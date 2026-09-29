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


def cpp_blas_colour_sum_test_factory(process, model='sm', tolerance=1e-6):
    def test(self):
        self.check_cpp_blas_colour_sum(process, model=model, tolerance=tolerance)
    test.__name__ = 'test_cpp_blas_%s' % _sanitize_process_name(process)
    test.__doc__ = ('Check the madmatrix colour sum agrees with the fortran '
                    'standalone with and without BLAS for %s.' % process)
    return test


def cpp_blas_crossed_colour_sum_test_factory(process, base_dir, defines=(),
                                             model='sm', tolerance=1e-6,
                                             color_basis=None):
    def test(self):
        self.check_cpp_blas_crossed_colour_sum(
            process, base_dir, defines=defines, model=model,
            tolerance=tolerance, color_basis=color_basis)
    test.__name__ = 'test_cpp_blas_crossed_%s' % _sanitize_process_name(base_dir)
    test.__doc__ = ('Check the crossed madmatrix colour sum of the %s base of '
                    '%s agrees with the fortran standalone with and without '
                    'BLAS.' % (base_dir, process))
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
          - standalone (the madmatrix CPU-SIMD backend).
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
        self.do('output standalone_fortran %s -f' % ref_root)
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
        self.do('output standalone_fortran %s -f' % sacross_root)
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

        # -- (4) standalone (madmatrix CPU-SIMD) --------------------------------
        # Skipped (not failed) if no C++ compiler or the madmatrix build
        # toolchain is unavailable. Matched by flavor order like madevent: the
        # base flavors are ids 0..nflav-1, in the same order as the standalone
        # check. Generated with the crossing on, but a single process folds no
        # crossed subprocess in, so this compiles the plain helicity loop: the
        # crossed one (extended ids cross*nflav+flav, with nflav > 1 on a
        # multi-flavor base) is compared to the fortran standalone by
        # check_cpp_blas_crossed_colour_sum.
        mg7_by_iflav = self._run_standalone_mg7(process, seeded_phase_space, ref_rows)
        if mg7_by_iflav is not None:
            self._compare_by_iflav(
                process, 'standalone', ref_rows, mg7_by_iflav, tolerance)

    def check_cpp_blas_colour_sum(self, process, model='sm', tolerance=1e-6):
        """standalone (madmatrix) must reproduce the fortran standalone with BLAS or without.

        The BLAS colour sum (CPPBLAS=hasBlas, which is the default wherever a
        host BLAS can be linked) is a SECOND copy of the helicity loop: it keeps
        the jamps of every good helicity and sums the colour for all of them in
        one SYMM call after the loop. That copy has to carry everything the
        scalar loop carries, and in particular the C-parity de-duplication
        weight: the good helicities are halved to one representative per mirror
        pair, so a path that counts each representative once returns exactly
        HALF of |M|^2 -- silently, since nothing else about the answer looks wrong.

        Hence all three ingredients below are load bearing:
          - a C-symmetric process (pure QCD, so |M(h)|^2 == |M(-h)|^2 and the
            de-duplication actually fires). On a process where it stays off the
            two variants agree without the weight ever being exercised;
          - BOTH CPPBLAS settings, since only one of them is the batch;
          - the fortran standalone as the reference, so that a weight lost from
            BOTH paths at once would still be caught.

        The BLAS colour sum is only selected above blas_min_ncolor, a
        performance threshold that no process cheap enough for a test reaches,
        so it is lowered for the duration of the output -- the code path is the
        one the big processes get, only the "is it worth the call" gate moves.

        This is the plain helicity loop only. backend/<variant>/SigmaKin.cc runs
        another one, per lane, when the crossing machinery is compiled in, and
        that is only written for a base that folds a crossed subprocess in: a
        bare process folds nothing, so --use_crossing=True would compile the
        very same plain loop again. The crossed loop is
        check_cpp_blas_crossed_colour_sum.
        """
        from madmatrix.model_handling import OneProcessExporterMadMatrix
        if not OneProcessExporterMadMatrix.blas_is_available():
            self.skipTest('no host BLAS to link the C++ colour sum against')

        self.do('set automatic_html_opening False')
        self.do('set group_subprocesses False')
        self.do('set apply_flavor_grouping True')
        self.do('set zerowidth_tchannel False')
        self.do('import model %s' % model)

        # -- Reference: plain fortran standalone, as in check_process ----------
        self.do('generate %s --use_crossing=False' % process)
        generated_process = self.cmd._curr_amps[0].get('process')
        seeded_phase_space = self._get_seeded_phase_space(generated_process)
        ref_root = pjoin(self.tmpdir, 'standalone_plain')
        self.do('output standalone_fortran %s -f' % ref_root)
        ref_sub = self._get_single_subprocess_dir(pjoin(ref_root, 'SubProcesses'))
        ref_rows, printed_phase_space = self._run_standalone(ref_sub)
        self._assert_phase_space_reasonable(
            printed_phase_space, seeded_phase_space, ref_sub)

        saved = OneProcessExporterMadMatrix.blas_min_ncolor
        OneProcessExporterMadMatrix.blas_min_ncolor = 1
        try:
            pdir = self._output_standalone_mg7(
                process, '--use_crossing=False', 'standalone_madmatrix_blas')
        finally:
            OneProcessExporterMadMatrix.blas_min_ncolor = saved
        if pdir is None:
            self.skipTest('standalone (madmatrix) output unavailable')

        # Without the BLAS colour sum selected for this process both
        # variants below run the very same scalar loop and the check is vacuous.
        self._assert_blas_selected(pdir, process)

        for label, make_args in self.CPPBLAS_VARIANTS:
            by_iflav = self._run_check_sa(
                pdir, process, seeded_phase_space, ref_rows, make_args)
            if by_iflav is None:
                self.skipTest('cannot build check_sa.exe (CPPBLAS=%s)' % label)
            self._compare_by_iflav(
                process, 'standalone CPPBLAS=%s' % label,
                ref_rows, by_iflav, tolerance)

    # The two C++ colour sums: the BLAS batch (the default wherever a host BLAS
    # can be linked) and the scalar per-helicity loop.
    CPPBLAS_VARIANTS = (('hasBlas (default)', ()),
                        ('hasNoBlas', ('CPPBLAS=hasNoBlas',)))

    def _assert_blas_selected(self, pdir, label):
        # assertTrue, not assertIn: the latter would print the whole file.
        with open(pjoin(pdir, 'ColorData.h')) as fsock:
            emitted = fsock.read()
        self.assertTrue(
            'shouldUseBlas = true' in emitted,
            'The BLAS colour sum was not selected for %s: the CPPBLAS '
            'comparison would not test anything' % label)

    def check_cpp_blas_crossed_colour_sum(self, process, base_dir, defines=(),
                                          model='sm', tolerance=1e-6,
                                          color_basis=None):
        """The crossed helicity loop of standalone (madmatrix) must reproduce the
        fortran standalone lane by lane, with BLAS or without.

        With the crossing machinery compiled in (ProcessTables::use_crossing),
        backend/<variant>/SigmaKin.cc runs its own copy of the helicity loop:
        cNGoodMaxCross iterations in which every lane evaluates the ighel-th
        good helicity of ITS OWN crossing (picked per lane inside
        calculate_jamps), with a per-lane C-parity weight (csym_lane_on, per
        crossing). The BLAS batch carries a second copy of all of it, and is
        where a lost weight once gave exactly half of |M|^2. The machinery is
        only written for a base that folds a crossed subprocess in, so `process`
        has to be a multiprocess whose `base_dir` subprocess really does; a bare
        process folds nothing and compiles the plain loop
        (check_cpp_blas_colour_sum).

        Every extended flavor id (cross*nmaxflavor + flavor) of a crossing the
        base recorded is evaluated at the seeded point of its own crossed PDG
        signature. It is compared, matched by that PDG, to the plain
        (--use_crossing=False) fortran standalone of the same multiprocess,
        where it is a subprocess of its own. Each id is run once with every
        lane the same, then all of them together, one per event. In that run
        the lanes of one SIMD page carry different crossings, which is what the
        per-lane helicity and C-parity weight are for. With nmaxflavor > 1,
        umami also regroups the reduced flavors into pages, and the id is decoded
        as cross = id / nmaxflavor, flavor = id % nmaxflavor. An id whose PDG
        signature the reference does not print (it printed another
        representative of the same flavor class) is not compared. Every
        recorded crossing must still be compared at least once, and on a
        multi-flavor base so must a crossed id with a non-zero reduced flavor.
        """
        from madmatrix.model_handling import OneProcessExporterMadMatrix
        if not OneProcessExporterMadMatrix.blas_is_available():
            self.skipTest('no host BLAS to link the C++ colour sum against')
        if not shutil.which(os.environ.get('CXX', 'g++')):
            self.skipTest('no C++ compiler')

        self.do('set automatic_html_opening False')
        self.do('set group_subprocesses False')
        self.do('set apply_flavor_grouping True')
        self.do('set zerowidth_tchannel False')
        if color_basis:
            self.do('set color_basis %s' % color_basis)
        self.do('import model %s' % model)
        for line in defines:
            self.do(line)

        # -- The folded base, with the BLAS colour sum selected ----------------
        self.do('generate %s --use_crossing=True' % process)
        mg_root = pjoin(self.tmpdir, 'standalone_madmatrix_crossed')
        saved = OneProcessExporterMadMatrix.blas_min_ncolor
        OneProcessExporterMadMatrix.blas_min_ncolor = 1
        try:
            self.do('output standalone %s -f' % mg_root)
        finally:
            OneProcessExporterMadMatrix.blas_min_ncolor = saved
        base_me, pdir = None, None
        for matrix_element in self.cmd._curr_matrix_elements.get_matrix_elements():
            name = process_checks._crossing_dir_name(matrix_element)
            if name.split('_', 1)[-1] == base_dir:
                base_me = matrix_element
                pdir = pjoin(mg_root, 'SubProcesses', name)
        self.assertTrue(base_me is not None and os.path.isdir(pdir),
                        'no %s directory written for %s' % (base_dir, process))
        label = '%s of %s' % (base_dir, process)
        # Vacuity guards: the crossed loop is compiled in, the batch is selected
        with open(pjoin(pdir, 'ProcessTables.h')) as fsock:
            self.assertTrue('use_crossing = true' in fsock.read(),
                            '%s was written without the crossing machinery: '
                            'the crossed loop is not compiled' % label)
        self._assert_blas_selected(pdir, label)
        recorded = process_checks._mg7_compiled_crossings(pdir)
        entries = [entry for entry in process_checks._crossing_pdg_entries(base_me)
                   if entry[1] in recorded]
        model_obj = self.cmd._curr_model
        ninitial = base_me.get_nexternal_ninitial()[1]

        seeded_by_pdg = {}
        def seeded(pdg):
            if pdg not in seeded_by_pdg:
                seeded_by_pdg[pdg] = process_checks._crossing_momenta(
                    pdg, ninitial, model_obj, None, 1000.0, self.cmd)
                self.assertTrue(seeded_by_pdg[pdg],
                                'no seeded phase-space point for %s' % (pdg,))
            return seeded_by_pdg[pdg]

        # -- Reference: each subprocess on its own, plain fortran standalone ---
        wanted = set(entry[3] for entry in entries)
        self.do('generate %s --use_crossing=False' % process)
        ref_root = pjoin(self.tmpdir, 'standalone_plain')
        self.do('output standalone_fortran %s -f' % ref_root)
        reference = {}
        for ref_me in self.cmd._curr_matrix_elements.get_matrix_elements():
            identities = set(entry[3] for entry in
                             process_checks._crossing_pdg_entries(
                                 ref_me, identity_only=True))
            if not identities & wanted:
                continue  # the base reaches none of its flavors: not built
            ref_sub = pjoin(ref_root, 'SubProcesses',
                            process_checks._crossing_dir_name(ref_me))
            rows, printed = self._run_standalone(ref_sub)
            # Every flavor of a directory is printed at the one point, the
            # seeded point of any of them (they share the masses)
            self._assert_phase_space_reasonable(
                printed, seeded(rows[0]['pdg']), ref_sub)
            for pdg, value in self._rows_by_pdg(rows, ref_sub).items():
                self.assertNotIn(pdg, reference,
                                 'flavor %s printed by two reference '
                                 'directories' % (pdg,))
                reference[pdg] = value

        lanes = [(idx, cross, flav, pdg, seeded(pdg))
                 for (idx, cross, flav, pdg) in entries if pdg in reference]
        compared = set(lane[1] for lane in lanes)
        self.assertEqual(compared, recorded,
                         'no reference for the crossings %s recorded by %s'
                         % (sorted(recorded - compared), label))
        if any(cross and flav for (_idx, cross, flav, _pdg) in entries):
            self.assertTrue(any(lane[1] and lane[2] for lane in lanes),
                            'no crossed id with a non-zero reduced flavor of '
                            '%s is compared' % label)
        for lane in lanes:
            self.assertGreater(abs(reference[lane[3]]), 0.,
                               'degenerate: the reference of %s is 0'
                               % (lane[3],))

        self._lift_check_sa_flavor_cap(pdir)
        for blas_label, make_args in self.CPPBLAS_VARIANTS:
            if not self._build_check_sa(pdir, make_args):
                self.skipTest('cannot build check_sa.exe (CPPBLAS=%s)'
                              % blas_label)
            for run in [[lane] for lane in lanes] + [lanes]:
                values = self._run_check_sa_lanes(
                    pdir, [(lane[0], lane[4]) for lane in run])
                for ievt, value in enumerate(values):
                    idx, cross, flav, pdg, _momenta = run[ievt % len(run)]
                    ref_me = reference[pdg]
                    rel = abs(ref_me - value) / max(abs(ref_me), abs(value), 1e-99)
                    logger.debug('%s id=%s event %d CPPBLAS=%s: diff=%f%%',
                                 label, idx, ievt, blas_label, 100 * rel)
                    self.assertLessEqual(
                        rel, tolerance,
                        'Incompatible matrix elements for %s id=%s (cross=%s '
                        'flavor=%s, PDG %s), event %d of a run of %s: '
                        'reference=%s standalone CPPBLAS=%s=%s'
                        % (label, idx, cross, flav, pdg, ievt,
                           'one id' if len(run) == 1 else 'mixed ids',
                           ref_me, blas_label, value))

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

    def _output_standalone_mg7(self, process, options='--use_crossing=True',
                               outdir_name='standalone_madmatrix'):
        """Write a standalone (madmatrix) output for `process`, return its P* dir.

        Returns None (skip) if there is no C++ compiler or the exporter refuses
        the process.
        """
        if not shutil.which(os.environ.get('CXX', 'g++')):
            return None
        outdir = pjoin(self.tmpdir, outdir_name)
        self.do('generate %s %s' % (process, options))
        try:
            self.do('output standalone %s -f' % outdir)
        except Exception:
            return None
        return self._get_single_subprocess_dir(pjoin(outdir, 'SubProcesses'))

    def _run_check_sa(self, pdir, process, phase_space, ref_rows, make_args=()):
        """{IFLAV -> matrix element} from a check_sa.exe built with `make_args`.

        Returns None (skip) if the madmatrix build toolchain cannot build
        check_sa.exe. check_sa.exe reads the external momenta from an LHE file
        (-e), so the same seeded point is used as for the fortran backends; the
        base flavors are the extended ids 0..nflav-1.
        """
        nevt = 8
        lhe = pjoin(pdir, 'seeded.lhe')
        self._write_lhe_events(lhe, phase_space, nevt)

        if not self._build_check_sa(pdir, make_args):
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
                            'No matrix element from standalone (madmatrix) flavor id %s '
                            'for %s:\n%s' % (flavor_id, process, output))
            by_iflav[iflav] = float(values[0])
        return by_iflav

    def _build_check_sa(self, pdir, make_args=()):
        """Build check_sa.exe with `make_args`; False if the toolchain cannot."""
        # cleanall first: the objects of a previous variant were compiled with
        # that variant's flags and the makefile has no way to notice.
        self._call_with_optional_redirection(['make', 'cleanall'], pdir)
        rc = self._call_with_optional_redirection(
            ['make', '-j2'] + list(make_args) + ['check_sa.exe'], pdir)
        return rc == 0

    # The CPU branch of run_perf_mode, where the per-event flavor ids are set
    _FLVVEC_FROM = '    std::vector<unsigned int> flvVec( nevt, flavorID );\n#endif\n'
    _FLVVEC_TO = (
        '    std::vector<unsigned int> flvVec( nevt, flavorID );\n'
        '    if( const char* mgfl = getenv( "MG_FLVLIST" ) )\n'
        '    {\n'
        '      std::vector<unsigned int> mgids;\n'
        '      std::istringstream mgin( mgfl );\n'
        '      std::string mgtok;\n'
        '      while( std::getline( mgin, mgtok, \',\' ) ) mgids.push_back( (unsigned int)std::stoul( mgtok ) );\n'
        '      for( unsigned int ievt = 0; ievt < nevt; ievt++ ) flvVec[ievt] = mgids[ievt % mgids.size()];\n'
        '    }\n'
        '#endif\n')

    def _lift_check_sa_flavor_cap(self, pdir):
        """Patch the shipped check_sa.cc (before _build_check_sa) so that it
        takes (a) the extended flavor ids of the crossings -- the shipped cap
        stops at nmaxflavor, see process_checks' crossing backend -- and (b)
        with MG_FLVLIST=id0,id1,... set, a different one per event: event i
        gets id[i % n]."""
        check = pjoin(pdir, 'check_sa.cc')
        with open(check) as fsock:
            src = fsock.read()
        for old, new in ((process_checks._MG7_CAP_FROM, process_checks._MG7_CAP_TO),
                         (self._FLVVEC_FROM, self._FLVVEC_TO)):
            self.assertEqual(src.count(old), 1,
                             'check_sa.cc changed, cannot patch %r' % old)
            src = src.replace(old, new)
        with open(check, 'w') as fsock:
            fsock.write(src)

    def _run_check_sa_lanes(self, pdir, lanes):
        """Per-event |M|^2 of the patched check_sa.exe for `lanes`, a list of
        (flavor id, momenta): event i is lanes[i % len(lanes)], over enough
        events to fill whole SIMD pages (two of them in mixed precision) on
        any vector width."""
        nevt = 32 * ((len(lanes) + 31) // 32)
        lhe = pjoin(pdir, 'lanes.lhe')
        self._write_lhe_points(
            lhe, [lanes[ievt % len(lanes)][1] for ievt in range(nevt)])
        env = dict(os.environ,
                   MG_FLVLIST=','.join(str(lane[0]) for lane in lanes))
        output = subprocess.Popen(
            ['./check_sa.exe', 'perf', '-v', '-f', str(lanes[0][0]),
             '-e', lhe, '1', str(nevt), '1'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=pdir, env=env).communicate()[0].decode()
        values = [float(value) for value in
                  re.findall(r'Matrix element =\s*([-\d.eE+]+)', output)]
        self.assertEqual(len(values), nevt,
                         'expected %d matrix elements from check_sa.exe in %s '
                         '(flavor ids %s), got:\n%s'
                         % (nevt, pdir, env['MG_FLVLIST'], output))
        return values

    def _run_standalone_mg7(self, process, phase_space, ref_rows):
        """{IFLAV -> matrix element} for standalone (madmatrix) at the seeded momenta."""
        pdir = self._output_standalone_mg7(process)
        if pdir is None:
            return None
        return self._run_check_sa(pdir, process, phase_space, ref_rows)

    def _write_lhe_events(self, path, phase_space, nevents):
        """Write `nevents` identical minimal LHE events at `phase_space`.

        check_sa.exe only reads (E, px, py, pz) from each particle line; the
        pdg/status/colour columns are placeholders. The momenta are replicated
        across the SIMD page so every lane evaluates the seeded point.
        """
        self._write_lhe_points(path, [phase_space] * nevents)

    def _write_lhe_points(self, path, points):
        """Write one minimal LHE event per phase-space point of `points`."""
        def as_float(value):
            if isinstance(value, str):
                return float(value.replace('d', 'e').replace('D', 'E'))
            return float(value)

        lines = []
        for phase_space in points:
            lines.append('<event>')
            lines.append('%d 0 0.0 0.0 0.0 0.0' % len(phase_space))
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


class TestMadMatrixCppBlasColourSum(
        StandaloneMadeventMatrixElementConsistency):
    """The two C++ colour sums (scalar loop and BLAS batch) must agree.

    Pure QCD on purpose: these are the processes where the C-parity helicity
    de-duplication fires, and the weight it owes each surviving representative
    is what the BLAS batch once dropped (giving exactly half of |M|^2).

    Both helicity loops are covered: the plain one (test_cpp_blas_<process>)
    and the per-lane crossed one, which only a base folding crossed
    subprocesses compiles (test_cpp_blas_crossed_<base>).
    """

    test_cpp_blas_gg_ttx = cpp_blas_colour_sum_test_factory(
        'g g > t t~', model='sm', tolerance=1e-6)

    # MHV-vanishing gluon configurations sit ~30 orders of magnitude below the
    # largest |M|^2 here, which is what the de-duplication's noise floor has to
    # cope with before it can be on at all.
    test_cpp_blas_uux_gg = cpp_blas_colour_sum_test_factory(
        'u u~ > g g', model='sm', tolerance=1e-6)

    # The crossed helicity loop, on bases that really fold crossings in.
    # g g > u u~ folds, among others, g u~ > g u~ and u u~ > g g (the process
    # above, here a crossed lane). The trace basis is forced because the
    # all-gluon sibling of this multiprocess cannot be written in the DDM
    # default.
    test_cpp_blas_crossed_gg_qqx = cpp_blas_crossed_colour_sum_test_factory(
        'pq pq > pq pq', 'gg_QQx', defines=('define pq = g u u~',),
        model='sm', tolerance=1e-6, color_basis='trace')

    # A multi-flavor base (nmaxflavor = 2: two equal and two different quark
    # flavors): the extended ids decode a non-zero reduced flavor and umami
    # regroups the lanes by reduced flavor.
    test_cpp_blas_crossed_qq_qq = cpp_blas_crossed_colour_sum_test_factory(
        'q q > q q', 'QQ_QQ', defines=('define q = u d u~ d~',),
        model='sm', tolerance=1e-6)

    # Massive and 2->3 (nexternal = 5): g q > t t~ q folds g q~ > t t~ q~ and
    # q~ q > t t~ g, the crossed twin of test_cpp_blas_gg_ttx.
    test_cpp_blas_crossed_gq_ttxq = cpp_blas_crossed_colour_sum_test_factory(
        'p p > t t~ j', 'gQ_ttxQ', model='sm', tolerance=1e-6)
