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
