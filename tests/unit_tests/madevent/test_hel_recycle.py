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
""" Fixed-form line wrapping of the helicity recycled matrix element """

from __future__ import absolute_import
import os
import shutil
import tempfile
import unittest

import madgraph.madevent.hel_recycle as hel_recycle


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
