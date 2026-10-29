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
