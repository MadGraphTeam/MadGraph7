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
