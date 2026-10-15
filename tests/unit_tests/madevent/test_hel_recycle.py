##############################################################################
#
# Copyright (c) 2010 The MadGraph Development team and Contributors
#
# This file is a part of the MadGraph 5 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph license which should accompany this
# distribution.
#
# For more information, please visit: http://madgraph.phys.ucl.ac.be
#
################################################################################
"""How the helicity-recycled color stage reads the amplitudes.

AMP is helicity major in the recycled matrix element, so a color flow line
either indexes it in place, AMP(K,i), or reads a row that has been gathered out
of it contiguously, AMPK(i,HRL). Which one it is has to be the same decision in
the rewritten lines and in the loop the driver template opens around them, so
both come from set_gather_lines and are checked together here.

Also the fixed-form line wrapping the recycled matrix element is emitted
through, which is what decides whether those lines are legal fortran at all."""

from __future__ import absolute_import
import os
import shutil
import tempfile
import unittest

import madgraph.madevent.hel_recycle as hel_recycle


class TestAmpGather(unittest.TestCase):
    """The size gate of the contiguous amplitude gather, and the two shapes of
    generated code that hang off it."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='hr_gather')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def recycler(self, ngraphs, ncomb):
        """A recycler whose template announces ngraphs amplitudes and whose
        good helicity list is ncomb long -- the two the gate is a function
        of."""

        template = os.path.join(self.tmpdir, 'template_matrix1.f')
        with open(template, 'w') as fsock:
            fsock.write('      PARAMETER (NGRAPHS=%d) \n' % ngraphs)
        obj = hel_recycle.HelicityRecycler(
            [str(i + 1) for i in range(ncomb)])
        obj.set_template(template)
        return obj

    def test_template_ngraphs(self):
        self.assertEqual(self.recycler(510, 8).template_ngraphs(), 510)
        # no template to read: the gate has nothing to decide on
        obj = self.recycler(510, 8)
        obj.set_template(os.path.join(self.tmpdir, 'absent.f'))
        self.assertEqual(obj.template_ngraphs(), 0)

    def test_gate_is_on_the_size_of_amp(self):
        """AMP is NCOMB x NGRAPHS complex*16, and only its total size decides:
        either factor can be the large one."""

        limit = hel_recycle.GATHER_MIN_BYTES // 16
        for ngraphs, ncomb in [(limit // 8, 8), (8, limit // 8)]:
            obj = self.recycler(ngraphs, ncomb)
            obj.set_gather_lines()
            self.assertTrue(obj.amp_gather, (ngraphs, ncomb))
            obj = self.recycler(ngraphs, ncomb - 1)
            obj.set_gather_lines()
            self.assertFalse(obj.amp_gather, (ngraphs, ncomb - 1))

    def test_no_template_never_gathers(self):
        obj = self.recycler(0, 4096)
        obj.set_gather_lines()
        self.assertFalse(obj.amp_gather)

    def test_plain_loop_reads_amp_in_place(self):
        """Below the gate nothing changes: the holes render the very loop the
        template used to carry, and the color flows index AMP by helicity."""

        obj = self.recycler(45, 20)
        obj.set_gather_lines()
        self.assertEqual(obj.template_dict['hr_gather_decl'], '')
        self.assertEqual(obj.template_dict['hr_gather_open'], 'K = 1, NCOMB')
        self.assertEqual(obj.template_dict['hr_gather_close'], 'K')
        self.assertEqual(obj.add_indices('JAMP(1,1) = AMP(31) - AMP(1)'),
                         'JAMP(1,1) = AMP( K,31) - AMP( K,1)')

    def test_gathered_loop_reads_the_lane(self):
        """Above it the block loop wraps a row loop, and every amplitude read
        moves to the gathered lane -- the color flows and the AMP2 lines
        alike."""

        obj = self.recycler(4096, 128)
        obj.set_gather_lines()
        decl = obj.template_dict['hr_gather_decl']
        self.assertIn('PARAMETER (NHRBLK=%d)' % hel_recycle.GATHER_BLOCK, decl)
        self.assertIn('COMPLEX*16 AMPK(NGRAPHS,NHRBLK)', decl)
        # same storage class as AMP, so that it stays thread private if the
        # !$OMP PARALLEL of SMATRIX1_MULTI is ever compiled in
        self.assertNotIn('SAVE', decl)
        opened = obj.template_dict['hr_gather_open']
        self.assertTrue(opened.startswith('KB = 1, NCOMB, NHRBLK'))
        self.assertIn('AMPK(I,HRL) = AMP(KB+HRL-1,I)', opened)
        # K is no longer a loop variable but still what every rewritten line
        # uses, so the row loop has to assign it
        self.assertIn('K = KB + HRL - 1', opened)
        # one ENDDO is the template's own, closing the row loop
        self.assertEqual(obj.template_dict['hr_gather_close'],
                         'HRL\n      ENDDO ! KB')
        self.assertEqual(obj.add_indices('JAMP(1,1) = AMP(31) - AMP(1)'),
                         'JAMP(1,1) = AMPK(31,HRL) - AMPK(1,HRL)')
        self.assertEqual(
            obj.add_indices('AMP2(1)=AMP2(1)+AMP(1)*DCONJG(AMP(1))'),
            'AMP2(1)=AMP2(1)+AMPK(1,HRL)*DCONJG(AMPK(1,HRL))')

    def test_only_amp_is_rewritten(self):
        """TMP_JAMP, JAMP and the AMPBUF of the table-emitted color flows all
        contain the three letters, and none of them is the amplitude array."""

        for obj in (self.recycler(45, 20), self.recycler(4096, 128)):
            obj.set_gather_lines()
            for line in ['TMP_JAMP(3) = TMP_JAMP(1) + TMP_JAMP(2)',
                         'JAMP(1,1) = JAMP(2,1)',
                         'AMPBUF(NGRAPHS+ITMP) = AMPBUF(TMP_JAMP_A(ITMP))']:
                self.assertEqual(obj.add_indices(line), line)

    def test_a_bracketed_index_survives(self):
        """With the color flow definitions emitted as operand tables, the
        gather the exporter writes into matrix<i>_orig.f reads AMP at a table
        lookup rather than at a literal."""

        obj = self.recycler(45, 20)
        obj.set_gather_lines()
        self.assertEqual(obj.add_indices('AMPBUF(ITMP) = AMP(IDX(ITMP))'),
                         'AMPBUF(ITMP) = AMP( K,IDX(ITMP))')
        obj = self.recycler(4096, 128)
        obj.set_gather_lines()
        self.assertEqual(obj.add_indices('AMPBUF(ITMP) = AMP(IDX(ITMP))'),
                         'AMPBUF(ITMP) = AMPK(IDX(ITMP),HRL)')


class TestDoMultiline(unittest.TestCase):
    """do_multiline breaks a statement over fixed-form continuation lines.

    A continuation line is attached to whatever statement precedes it, so a
    physical line which holds nothing but blanks must never be emitted: the
    continuation after it would silently extend the previous statement."""

    # the Kleiss-Kuijf color flow JAMPs of g g > g g g on the DDM basis: long
    # enough to wrap and with no space of their own to wrap at, so the only
    # candidate split point is inside the statement's indentation
    JAMPF = '        JAMPF(2,1)=+2D0*(-IMAG1*JAMP(3,1)-IMAG1*JAMP(4,1)' \
            '-IMAG1*JAMP(6,1))'

    def physical_lines(self, line):
        return hel_recycle.do_multiline(line).split('\n')

    def assertWrapIsValid(self, line):
        lines = self.physical_lines(line)
        for i, physical in enumerate(lines):
            self.assertLessEqual(len(physical), 132,
                                 'physical line %d is too long' % i)
            if i:
                self.assertTrue(physical.lstrip().startswith('$'),
                                'continuation line %d lost its marker' % i)
        for i, physical in enumerate(lines[:-1]):
            self.assertTrue(physical.strip(),
                            'physical line %d is blank, so the continuation '
                            'which follows it joins the previous statement' % i)
        # and nothing of the statement is lost on the way
        rebuilt = ''.join(p.lstrip()[1:] if i else p
                          for i, p in enumerate(lines))
        self.assertEqual(rebuilt.replace(' ', ''), line.replace(' ', ''))

    def test_no_blank_physical_line_without_a_space_to_wrap_at(self):
        """A statement whose only space is its indentation wraps mid-token"""

        self.assertWrapIsValid(self.JAMPF)
        self.assertEqual(len(self.physical_lines(self.JAMPF)), 2)

    def test_short_line_is_untouched(self):
        """Nothing happens below the limit"""

        short = '        JAMPF(1,1)=+2D0*(+IMAG1*JAMP(6,1))'
        self.assertEqual(hel_recycle.do_multiline(short), short)

    def test_wraps_at_a_space_when_there_is_one(self):
        """The usual case still breaks at the last space that fits"""

        line = '        JAMP(2,1) = (-1.000000000000000D+00)*AMP( K,8)+' \
               '(-1.000000000000000D+00)*AMP( K,11)+(-1.0D+00)*TMP_JAMP(20)'
        self.assertWrapIsValid(line)
        self.assertTrue(self.physical_lines(line)[0].endswith(' '))

    def test_every_wrap_width_around_the_limit_is_valid(self):
        """Sweep the statement length across the wrap width"""

        for pad in range(40):
            line = '        JAMPF(2,1)=+2D0*(' + 'X' * pad + ')'
            self.assertWrapIsValid(line)


class TestCppDialect(unittest.TestCase):
    """Reading and re-writing the C++ the madmatrix backend emits.

    The recycler's algorithm is language independent; what a dialect owns is
    only how a call is taken apart and put back together. The C++ one has to
    cope with three things fortran never shows it: a template argument list
    before the real one, a result that is not the last argument, and an
    amplitude whose name cannot say which diagram it belongs to."""

    EXTERNAL = ('      ixxxxx<M_ACCESS, W_ACCESS>( momenta, m_pars->MT,'
                ' cHel[ihel][2], -1, cFlavors[iflavor][2], aloha_obj[5], 2 );')
    SCALAR = ('      sxxxxx<M_ACCESS, W_ACCESS>( momenta, +1,'
              ' cFlavors[iflavor][3], aloha_obj[6], 3 );')
    CURRENT = ('      FFV1P0_3<W_ACCESS, CD_ACCESS>( aloha_obj[0],'
               ' aloha_obj[1], COUPs[0], 1.0, 0., 0., aloha_obj[4] );')
    AMPLITUDE = ('      FFV1_0<W_ACCESS, A_ACCESS, CD_ACCESS>( aloha_obj[2],'
                 ' aloha_obj[3], aloha_obj[4], COUPs[0], 1.0,'
                 ' &amp_fp[0] ); //HRAMP 7')

    def setUp(self):
        self.dialect = hel_recycle.CppDialect()
        self.tmpdir = tempfile.mkdtemp(prefix='hr_cpp')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_the_template_arguments_are_not_arguments(self):
        """A call's arguments start after the <...>, and the commas inside it
        are not separators"""

        self.assertEqual(self.dialect.called_function(self.CURRENT), 'FFV1P0_3')
        self.assertEqual(self.dialect.arguments(self.CURRENT),
                         ['aloha_obj[0]', 'aloha_obj[1]', 'COUPs[0]', '1.0',
                          '0.', '0.', 'aloha_obj[4]'])

    def test_an_arrow_is_not_a_closing_bracket(self):
        """m_pars->MT would end the template argument list if '>' were read
        without looking at what precedes it"""

        self.assertEqual(self.dialect.called_function(self.EXTERNAL), 'ixxxxx')
        self.assertEqual(self.dialect.arguments(self.EXTERNAL)[1], 'm_pars->MT')

    def test_the_result_is_not_the_last_argument(self):
        """An external ends with its leg index, a current with its output"""

        external = self.dialect.arguments(self.EXTERNAL)
        self.assertEqual(self.dialect.output_name(external), 'aloha_obj[5]')
        self.assertEqual(self.dialect.output_position(external),
                         len(external) - 2)
        current = self.dialect.arguments(self.CURRENT)
        self.assertEqual(self.dialect.output_name(current), 'aloha_obj[4]')

    def test_the_leg_a_call_builds(self):
        """Read off the trailing index, and 1-based like the fortran one"""

        self.assertEqual(self.dialect.external_leg(
                                    self.dialect.arguments(self.EXTERNAL)), 3)
        self.assertEqual(self.dialect.external_helicity_leg(
                                    self.dialect.arguments(self.EXTERNAL)), 3)

    def test_an_axial_gauge_external_ends_with_its_reference(self):
        """vxxxxxr carries its reference leg after its own: the leg it builds
        is the argument after the output slot, not the last one"""

        line = ('      vxxxxxr<M_ACCESS, W_ACCESS>( momenta, 0., cHel[ihel][2],'
                ' +1, cFlavors[iflavor][2], aloha_obj[2], 2, 0 );')
        args = self.dialect.arguments(line)
        self.assertEqual(self.dialect.called_function(line).upper(), 'VXXXXXR')
        self.assertIn('VXXXXXR', self.dialect.external_routines)
        self.assertEqual(self.dialect.external_leg(args), 3)
        self.assertEqual(self.dialect.external_helicity_leg(args), 3)
        self.assertEqual(self.dialect.output_name(args), 'aloha_obj[2]')

    def test_a_scalar_has_no_helicity_to_fix(self):
        """Unlike fortran, where the slot exists and is overwritten anyway"""

        args = self.dialect.arguments(self.SCALAR)
        self.assertIsNone(self.dialect.external_helicity_leg(args))
        self.assertEqual(self.dialect.set_external_helicity(list(args), 0),
                         args)

    def test_fixing_a_helicity_replaces_the_table_read(self):
        args = self.dialect.arguments(self.EXTERNAL)
        fixed = self.dialect.set_external_helicity(list(args), -1)
        self.assertEqual(fixed[2], '-1')
        self.assertEqual(fixed[:2] + fixed[3:], args[:2] + args[3:])

    def test_an_amplitude_says_which_diagram_it_is(self):
        """Every amplitude writes the same scratch, so the name cannot"""

        self.assertTrue(self.dialect.is_amplitude(
                            self.dialect.called_function(self.AMPLITUDE)))
        self.assertEqual(self.dialect.amplitude_diagram(
            self.AMPLITUDE, self.dialect.arguments(self.AMPLITUDE)), 7)

    def test_a_fold_is_stamped_out_once_per_row(self):
        """The exporter writes what an amplitude contributes with a ${row}
        hole; the dialect fills it, once per row the amplitude survives in"""

        self.dialect.folds = {7: ['      jampAll_sv[${row} * ncolor] += amp_sv[0];']}

        class FakeAmp(object):
            def __init__(self, row, args):
                self.numbers, self.diag_num, self.args = (row,), 7, args

        args = self.dialect.arguments(self.AMPLITUDE)
        rendered = self.dialect.render_amplitudes(
            self.AMPLITUDE, [FakeAmp(1, args), FakeAmp(4, args)],
            gauge='U', amp_splt=False)
        self.assertEqual(rendered.count('FFV1_0'), 2)
        self.assertIn('jampAll_sv[0 * ncolor]', rendered)
        self.assertIn('jampAll_sv[3 * ncolor]', rendered)

    def test_a_line_that_is_not_a_call_is_not_one(self):
        for line in ('      // Amplitude(s) for diagram number 2',
                     '//HRFOLD 7',
                     '      jampAll_sv[( 0 * nParity + iParity ) * ncolor + 1]'
                     ' += amp_sv[0];',
                     '      if( storeChannelWeights )'):
            self.assertIsNone(self.dialect.called_function(line), line)

    def test_the_fold_block_is_read_out_and_then_skipped(self):
        """It is not a call and must not reach the unfolding, but it is needed
        before the pass that has to skip it -- hence the separate read"""

        path = os.path.join(self.tmpdir, 'orig.cc')
        with open(path, 'w') as fsock:
            fsock.write('\n'.join([self.AMPLITUDE, '//HRFOLD 7',
                                   '      jampAll_sv[${row}] += amp_sv[0];',
                                   '//HRENDFOLD']) + '\n')
        self.assertEqual(self.dialect.read_folds(path),
                         {7: ['      jampAll_sv[${row}] += amp_sv[0];']})
        skipped = [line for line in open(path)
                   if self.dialect.skip_line(line)]
        self.assertEqual(len(skipped), 3)


class TestCppCrossedExternal(unittest.TestCase):
    """A crossed external of a madmatrix output with crossing, in the form the
    recycler reads (MadMatrixUFOHelasCallWriter._crossing_external_orig).

    Every lane evaluates the BASE helicity row; a lane whose leg changed side
    takes the opposite helicity with the opposite NSF flag (the all-outgoing
    configuration). So each unfolded copy of helicity h is the head call
    (+nsf, h) followed by the tail call (-nsf, -h) and the icsign blend into
    the copy's own slot."""

    def setUp(self):
        from madmatrix.model_handling import MadMatrixUFOHelasCallWriter
        self.dialect = hel_recycle.CppDialect()
        self.tmpdir = tempfile.mkdtemp(prefix='hr_cpp_x')

        def one_call(sign, obj, hel=None):
            return ('ixxxxx<M_ACCESS, W_ACCESS>( xmom, m_pars->MT, %s, %+d,'
                    ' cFlavors[iflavor][2], %s, 2 );' % (hel, sign, obj))
        self.text = MadMatrixUFOHelasCallWriter._crossing_external_orig(
            None, one_call, 2, 2, 5, -1)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_the_head_is_an_ordinary_external(self):
        head = self.text.split('\n')[0]
        args = self.dialect.arguments(head)
        self.assertEqual(self.dialect.called_function(head), 'ixxxxx')
        self.assertEqual(self.dialect.external_leg(args), 3)
        self.assertEqual(self.dialect.external_helicity_leg(args), 3)
        self.assertEqual(self.dialect.output_name(args), 'aloha_obj[5]')

    def test_each_copy_gets_the_opposite_helicity_and_its_own_slot(self):
        path = os.path.join(self.tmpdir, 'orig.cc')
        with open(path, 'w') as fsock:
            fsock.write(self.text + '\n')
        self.dialect.read_folds(path)
        self.assertEqual(sorted(self.dialect.externals), [2])
        streamed = [line for line in open(path)
                    if not self.dialect.skip_line(line)]
        self.assertEqual(len(streamed), 1)  # the head only

        class FakeExternal(object):
            def __init__(self, hel, slot):
                self.mg, self.hel, self.numbers = 3, hel, (slot,)
                self.args = None
        head = streamed[0].rstrip('\n')
        for hel, slot in ((-1, 7), (1, 8)):
            wav = FakeExternal(hel, slot)
            wav.args = self.dialect.set_external_helicity(
                self.dialect.arguments(head), hel)
            wav.args[self.dialect.output_position(wav.args)] = 'aloha_obj[%d]' % (slot - 1)
            text = self.dialect.render_external(head, wav)
            self.assertIn('m_pars->MT, %d, -1,' % hel, text)
            self.assertIn('m_pars->MT, -( %d ), +1,' % hel, text)
            self.assertIn('aloha_obj[%d].pvec[_k]' % (slot - 1), text)
            self.assertIn('icsign[2]', text)
            self.assertNotIn('$', text)


class TestCppAmplitudeSplit(unittest.TestCase):
    """The P1N amplitude split, rendered for C++.

    The decision -- which leg to peel, and which unfolded amplitudes then share
    a partial contraction -- is the fortran one (plan_amp_split); what is
    checked here is what the C++ dialect makes of it, and in particular the
    flavour test, which is NOT uniform across peel positions and whose failure
    mode is silent."""

    LINE = ('      FFV1_0<W_ACCESS, A_ACCESS, CD_ACCESS>( aloha_obj[0],'
            ' aloha_obj[1], aloha_obj[2], COUPs[0], 1.0,'
            ' &amp_fp[0] ); //HRAMP 1')

    class FakeAmp(object):
        def __init__(self, row, args):
            self.numbers, self.diag_num, self.args = (row,), 1, list(args)

    def setUp(self):
        self.dialect = hel_recycle.CppDialect()
        self.dialect.folds = {1: ['      jamp[${row}] += amp_sv[0];']}

    def amplitudes(self, peeled_slots, shared=('aloha_obj[1]', 'aloha_obj[2]')):
        """One amplitude per peeled wavefunction, all sharing the other legs"""
        return [self.FakeAmp(row + 1, [slot, shared[0], shared[1],
                                       'COUPs[0]', '1.0', '&amp_fp[0]'])
                for row, slot in enumerate(peeled_slots)]

    def render(self, amps, **env):
        old = hel_recycle.CppDialect.P1N_MIN_SHARING
        hel_recycle.CppDialect.P1N_MIN_SHARING = env.get('minshare', 2)
        try:
            return self.dialect.render_amplitudes(self.LINE, amps,
                                                  gauge='U', amp_splt=True)
        finally:
            hel_recycle.CppDialect.P1N_MIN_SHARING = old

    def test_the_column_with_most_wavefunctions_is_peeled(self):
        """Column 0 varies, the others do not, so column 0 is the one to peel
        and P1N_1 is the routine that leaves it off"""

        out = self.render(self.amplitudes(['aloha_obj[%d]' % i
                                           for i in (10, 11, 12, 13)]))
        self.assertIn('FFV1P1N_1<W_ACCESS, CD_ACCESS>( aloha_obj[1],'
                      ' aloha_obj[2], COUPs[0], 1.0, _p1n );', out)
        # one partial contraction for the four amplitudes, four contractions
        self.assertEqual(out.count('P1N_1<'), 1)
        self.assertEqual(out.count('amp_sv[0] ='), 4)

    def test_a_fermion_peel_re_applies_the_flavour_test(self):
        """A P1N whose output is a fermion propagates the partner's flv_index
        and defers the test, so the contraction has to make it"""

        out = self.render(self.amplitudes(['aloha_obj[%d]' % i
                                           for i in (10, 11, 12, 13)]))
        self.assertIn('_p1n.flv_index != aloha_obj[10].flv_index', out)
        self.assertIn('_p1n.flv_index == -1 ) ? cxzero_sv()', out)

    def test_a_boson_peel_does_not(self):
        """A P1N whose output is a boson applies the test itself and zeroes the
        current; testing again would read an flv_index it never set"""

        amps = [self.FakeAmp(row + 1, ['aloha_obj[0]', 'aloha_obj[1]', slot,
                                       'COUPs[0]', '1.0', '&amp_fp[0]'])
                for row, slot in enumerate('aloha_obj[%d]' % i
                                           for i in (10, 11, 12, 13))]
        out = self.render(amps)
        self.assertIn('FFV1P1N_3<W_ACCESS, CD_ACCESS>( aloha_obj[0],'
                      ' aloha_obj[1], COUPs[0], 1.0, _p1n );', out)
        self.assertNotIn('flv_index', out)

    def test_each_amplitude_still_gets_its_own_row_fold(self):
        out = self.render(self.amplitudes(['aloha_obj[%d]' % i
                                           for i in (10, 11, 12, 13)]))
        for row in range(4):
            self.assertIn('jamp[%d] += amp_sv[0];' % row, out)

    def test_a_group_below_the_threshold_is_not_split(self):
        """Shallow grouping is a measured net loss, so it falls back to
        evaluating the whole vertex per row"""

        amps = self.amplitudes(['aloha_obj[10]', 'aloha_obj[11]'])
        self.assertNotIn('P1N', self.render(amps, minshare=4))
        self.assertIn('P1N', self.render(amps, minshare=2))

    def test_the_split_is_declined_rather_than_guessed(self):
        """A spin the fortran renderer refuses too"""

        amps = self.amplitudes(['aloha_obj[10]', 'aloha_obj[11]',
                                'aloha_obj[12]', 'aloha_obj[13]'])
        line = self.LINE.replace('FFV1_0', 'TTT1_0')
        self.assertIsNone(self.dialect._p1n_group(line, amps, 0))


class TestRecycledChunkBoundaries(unittest.TestCase):
    """Where the recycled call block may be cut.

    The block goes into ordinary functions so that the compiler is not handed
    one body of a hundred thousand lines. A cut in the wrong place is not a
    subtle problem -- it does not compile -- but WHERE the wrong places are is
    subtle: brace depth alone says nothing about `if( x )`, whose body opens on
    the next line."""

    def setUp(self):
        from madmatrix.model_handling import OneProcessExporterMadMatrix
        self.split = OneProcessExporterMadMatrix.hr_split_statements

    @staticmethod
    def fold(i):
        """What one amplitude turns into: the call, a guarded multichannel
        block, then the color flows"""
        return ['      FFV1_0<W_ACCESS>( a, b, &amp_fp[0] ); //HRAMP %d' % i,
                '      if( storeChannelWeights )',
                '      {',
                '        numAll_sv[%d] += cxabs2( amp_sv[0] );' % i,
                '      }',
                '      jampAll_sv[%d] += amp_sv[0];' % i]

    def assertChunksAreWholeStatements(self, lines, per_chunk):
        chunks = self.split(lines, per_chunk)
        self.assertEqual([l for c in chunks for l in c], lines,
                         'the block must come back out unchanged')
        for n, chunk in enumerate(chunks):
            depth = sum(l.count('{') - l.count('}') for l in chunk)
            self.assertEqual(depth, 0, 'chunk %d leaves a brace open' % n)
            last = chunk[-1].split('//')[0].rstrip()
            self.assertTrue(last.endswith(';') or last.endswith('}'),
                            'chunk %d ends mid-statement: %r' % (n, last))
        return chunks

    def test_a_guard_is_never_split_from_its_body(self):
        """`if( storeChannelWeights )` leaves the brace depth at zero, so depth
        alone would allow a cut between it and the block it guards"""

        lines = sum((self.fold(i) for i in range(6)), [])
        for per_chunk in range(1, 8):
            self.assertChunksAreWholeStatements(lines, per_chunk)

    def test_a_braced_group_is_never_split(self):
        """A P1N group holds declarations its contractions read"""

        lines = ['      {',
                 '        FFV1P1N_1<W_ACCESS>( a, b, _p1n );',
                 '        const cxtype_sv* _pt = access( _p1n.w );',
                 '        { const cxtype_sv* _pw = access( w.w );',
                 '          amp_sv[0] = _pt[0] * _pw[0]; }',
                 '      }'] * 4
        for per_chunk in (1, 2, 3, 5):
            chunks = self.assertChunksAreWholeStatements(lines, per_chunk)
            for chunk in chunks:
                opens = ''.join(chunk).count('{')
                self.assertEqual(opens, ''.join(chunk).count('}'))

    def test_the_whole_block_survives_every_chunk_size(self):
        lines = sum((self.fold(i) for i in range(10)), [])
        for per_chunk in range(1, 30):
            self.assertChunksAreWholeStatements(lines, per_chunk)

class TestRecycledChunkPolicy(unittest.TestCase):
    """When the recycled block is cut up at all. Chunks cost run time on a small
    block and save the build of a big one, so a block stays inline up to
    HR_INLINE_MAX statements unless --hel_recycling_chunk says otherwise."""

    def setUp(self):
        from madmatrix.model_handling import OneProcessExporterMadMatrix
        self.cls = OneProcessExporterMadMatrix

    def exporter(self, options):
        class Writer(object):
            cmd_options = options
        exporter = self.cls.__new__(self.cls)
        exporter.helas_call_writer = Writer()
        return exporter

    def test_a_small_block_stays_inline(self):
        exporter = self.exporter({})
        self.assertEqual(exporter.hel_recycling_chunk_size(self.cls.HR_INLINE_MAX), 0)
        self.assertEqual(exporter.hel_recycling_chunk_size(self.cls.HR_INLINE_MAX + 1),
                         self.cls.HR_CHUNK_STMTS)

    def test_the_option_wins(self):
        self.assertEqual(self.exporter({'hel_recycling_chunk': '50'})
                         .hel_recycling_chunk_size(10), 50)
        self.assertEqual(self.exporter({'hel_recycling_chunk': '0'})
                         .hel_recycling_chunk_size(10**6), 0)

    def test_chunks_take_the_state_and_are_called_in_order(self):
        exporter = self.exporter({'hel_recycling_chunk': '2'})
        calls = '\n'.join('      FFV1_0<W_ACCESS>( a, b, &amp_fp[0] ); //HRAMP %d' % i
                          for i in range(5)) + '\n'
        defs, body = exporter.hr_chunk_calls(calls)
        self.assertEqual(defs.count('noinline'), 3)
        self.assertEqual([line.split('(')[0].strip() for line in body.split('\n') if line],
                         ['hr_chunk_0', 'hr_chunk_1', 'hr_chunk_2'])
        for decl, name in self.cls.HR_CHUNK_STATE:
            self.assertIn(decl, defs)
            self.assertIn(name, body)
        self.assertEqual(self.exporter({}).hr_chunk_calls(calls), ('', calls))


class TestHelRecyclingWarmup(unittest.TestCase):
    """What the probe prints, read back: madevent's helicity-filter protocol."""

    def test_parse(self):
        from madmatrix.output import ProcessExporterMadMatrixStandalone
        output = '\n'.join(['Matrix Element/Good Helicity: 1 3',
                            'Matrix Element/Good Helicity: 1 1',
                            'HEL/ZEROAMP: 1 3 7',
                            'HEL/ZEROAMP: 1 1 2',
                            'Matrix element = 1.0 GeV^-2'])
        self.assertEqual(ProcessExporterMadMatrixStandalone._parse_hel_warmup(output),
                         ([1, 3], [(1, 2), (3, 7)]))


class TestCrossingHelicityFactor(unittest.TestCase):
    """The helicity baked into an external of a crossing-aware fortran matrix
    element keeps the IC(i) factor of its NSF flag.

    A crossed call evaluates the base row with every leg that changed side
    negated together with its NSF flag (APPLY_CROSSING_ROWS), so one optim
    serves every crossing only if the baked value is h*IC(i). The C++ dialect
    has no IC: its crossed external carries the sign through its //HREXT tail."""

    FERMION = '      CALL IXXXXX(P(0,2),ZERO,NHEL(2),+1*IC(2), FLAVOR(2),W(2))'
    VECTOR = '      CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1,W(1))'
    SCALAR = '      CALL SXXXXX(P(0,3),+1*IC(3),W(3))'

    def setUp(self):
        hel_recycle.HelicityRecycler(['1'])  # resets the shared class state

    def fortran_args(self, line, hel):
        dialect = hel_recycle.Dialect()
        return dialect.set_external_helicity(dialect.arguments(line), hel)

    def test_the_ic_factor_follows_the_helicity(self):
        """NHEL(2) with +1*IC(2) as its NSF becomes -1*IC(2)"""

        self.assertEqual(self.fortran_args(self.FERMION, -1)[2], '-1*IC(2)')
        self.assertEqual(self.fortran_args(self.FERMION, 1)[2], '+1*IC(2)')

    def test_no_ic_no_factor(self):
        """A leg whose NSF carries no IC gets the bare value"""

        self.assertEqual(self.fortran_args(self.VECTOR, -1)[2], '-1')

    def test_a_scalar_takes_no_factor(self):
        """A scalar has no helicity: the slot is its output, renamed later"""

        self.assertEqual(self.fortran_args(self.SCALAR, 0)[2], ' 0')

    def test_the_unfolded_copies_carry_it(self):
        """generate_wavfuncs writes one copy per helicity, each h*IC(i)"""

        hel_recycle.External.hel_ranges = [set(), {-1, 1}]
        dag = hel_recycle.DAG()
        copies = hel_recycle.External.generate_wavfuncs(self.FERMION + '\n', dag)
        self.assertEqual(sorted(copy.args[2] for copy in copies),
                         ['+1*IC(2)', '-1*IC(2)'])
        self.assertEqual(sorted(copy.hel for copy in copies), [-1, 1])

    def test_cpp_takes_no_factor(self):
        """The C++ dialect writes the bare value"""

        dialect = hel_recycle.CppDialect()
        line = ('      ixxxxx<M_ACCESS, W_ACCESS>( xmom, m_pars->MT, cHel[ihel][2],'
                ' -1, cFlavors[iflavor][2], aloha_obj[5], 2 );')
        args = dialect.set_external_helicity(dialect.arguments(line), -1)
        self.assertEqual(args[2], '-1')


class TestGoodHelIds(unittest.TestCase):
    """NHEL(0,k) of the recycled matrix element is the 1-based id of that
    helicity in the original NHEL table.

    SMATRIX returns it as the selected helicity and get_nhel reads the LHE
    spin column back from it, where row 0 is the number of spin states: a
    0-based id puts every event on the previous combination."""

    # e+ e- > mu+ mu- ordering, only the four helicity conserving ones survive
    ALL_HEL = [(-1, 1, 1, -1), (-1, 1, 1, 1), (-1, 1, -1, -1), (-1, 1, -1, 1),
               (1, -1, 1, -1), (1, -1, 1, 1), (1, -1, -1, -1), (1, -1, -1, 1)]
    GOOD = ['1', '4', '5', '8']

    def recycled_nhel_lines(self, hel_filt):
        recycler = hel_recycle.HelicityRecycler(self.GOOD)
        recycler.hel_filt = hel_filt
        recycler.prepare_bools()
        for i, hel in enumerate(self.ALL_HEL, 1):
            recycler.get_good_hel('      DATA (NHEL(I,%4d),I=1,4) /%s/\n'
                                  % (i, ','.join('%2d' % h for h in hel)))
        recycler.get_good_hel('C     ----------\n')
        lines = recycler.template_dict['helicity_lines'].split('\n')
        return dict((tuple(int(h) for h in l.split('/')[1].split(',')[1:]),
                     int(l.split('/')[1].split(',')[0]))
                    for l in lines if 'NHEL' in l)

    def assertOriginalIds(self, nhel, expected_hel):
        self.assertEqual(sorted(nhel), sorted(expected_hel))
        for hel, old_id in nhel.items():
            self.assertEqual(self.ALL_HEL[old_id - 1], hel)

    def test_hel_recycling_ids_with_filtering(self):
        """hel_filtering = True keeps the good helicities with their own id"""

        nhel = self.recycled_nhel_lines(True)
        self.assertOriginalIds(nhel,
                               [self.ALL_HEL[int(i) - 1] for i in self.GOOD])

    def test_hel_recycling_ids_without_filtering(self):
        """hel_filtering = False keeps them all, ids still start at 1"""

        nhel = self.recycled_nhel_lines(False)
        self.assertOriginalIds(nhel, self.ALL_HEL)
        self.assertEqual(min(nhel.values()), 1)
