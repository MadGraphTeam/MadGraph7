################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""Unit tests for the madmatrix ALOHA writer (madmatrix/model_handling.py)."""

from __future__ import absolute_import

import tests.unit_tests as unittest


class TestMomentaInDenomPrecision(unittest.TestCase):
    """The TMPs that only involve momenta are computed from the momenta in
    denominator precision (double for FPTYPE=v), and a custom propagator
    denominator made of them is evaluated from those double versions."""

    def setUp(self):
        from madmatrix import model_handling
        # the helpers only need the two sets filled by define_expression
        self.writer = object.__new__(model_handling.MadMatrixALOHAWriter)
        self.writer.dp_needed = set()
        self.writer.pure_momentum_tmps = set()

    def test_momenta_only_expression(self):
        """p^2 and p.q are rewritten on dP, the constants on their d version,
        and the legs they need are recorded."""
        writer = self.writer
        self.assertEqual(
            writer.momenta_in_denom_precision(
                '( P1[0] * P1[0] - P1[1] * P1[1] - P1[2] * P1[2] - P1[3] * P1[3] )'),
            '( dP1[0] * dP1[0] - dP1[1] * dP1[1] - dP1[2] * dP1[2] - dP1[3] * dP1[3] )')
        self.assertEqual(writer.dp_needed, set([1]))
        self.assertEqual(
            writer.momenta_in_denom_precision('two * ( P2[0] * P3[0] + 0.5 * P2[1] * P3[1] )'),
            'twod * ( dP2[0] * dP3[0] + 0.5 * dP2[1] * dP3[1] )')
        self.assertEqual(writer.dp_needed, set([1, 2, 3]))

    def test_not_momenta_only(self):
        """A wavefunction, cI, a mass or no momentum at all: left alone, and no
        dP array is asked for."""
        writer = self.writer
        for expr in ('( wV3[0] * P1[0] - wV3[1] * P1[1] )',
                     '( P1[0] * P1[0] - cI * P1[3] )',
                     '( P1[0] * P1[0] - M1 * M1 )',
                     '( one + two )',
                     '( dP1[0] * dP1[0] )'):
            self.assertIsNone(writer.momenta_in_denom_precision(expr), expr)
        self.assertEqual(writer.dp_needed, set())

    def test_loop_mode_untouched(self):
        """Loop mode has complex momenta: nothing is rewritten."""
        import aloha
        old = aloha.loop_mode
        aloha.loop_mode = True
        try:
            self.assertIsNone(self.writer.momenta_in_denom_precision('( P1[0] * P1[0] )'))
        finally:
            aloha.loop_mode = old

    def test_denominator_momentum_tmps(self):
        """Only the TMPs of a custom denominator made of momenta-only TMPs are
        selected; a numerator-only momenta TMP, a denominator involving anything
        else, a standard propagator or an amplitude routine select nothing."""
        import types
        writer = self.writer
        writer.write_obj = lambda obj: obj          # contracted holds C++ text here
        writer.offshell = 1
        writer.tag = []
        contracted = {'TMP9': '( P1[0] * P1[0] - P1[1] * P1[1] )',
                      'TMP10': '( P1[0] * P1[0] + P1[1] * P1[1] )',
                      'TMP13': '( P1[0] * P2[0] - P1[1] * P2[1] )',   # numerator only
                      'TMP6': '( wV3[0] * wV2[0] - wV3[1] * wV2[1] )'}

        def select(denominator, offshell=1, tag=()):
            writer.routine = types.SimpleNamespace(denominator=denominator,
                                                   contracted=contracted)
            writer.offshell = offshell
            writer.tag = list(tag)
            return writer.denominator_momentum_tmps()

        self.assertEqual(select('(TMP9 * TMP10)'), set(['TMP9', 'TMP10']))
        self.assertEqual(select('(TMP9 * TMP6)'), set())          # a wavefunction TMP
        self.assertEqual(select('(TMP9 - M1 * M1)'), set())       # a mass
        self.assertEqual(select('(TMP9 * TMP99)'), set())         # unknown TMP
        self.assertEqual(select(None), set())                     # standard propagator
        self.assertEqual(select('1'), set())
        self.assertEqual(select('(TMP9 * TMP10)', offshell=0), set())
        self.assertEqual(select('(TMP9 * TMP10)', tag=['L']), set())

    def test_custom_denominator(self):
        """The axial gauge denominator (P.PBar) P^2 uses the double TMPs; a
        denominator with anything else keeps the amplitude precision ones."""
        writer = self.writer
        writer.pure_momentum_tmps = set(['TMP9', 'TMP10'])
        self.assertEqual(writer.denominator_in_denom_precision('(TMP9 * TMP10)'),
                         '(dTMP9 * dTMP10)')
        self.assertEqual(writer.denominator_in_denom_precision('(half * TMP9 * TMP10)'),
                         '(halfd * dTMP9 * dTMP10)')
        for denominator in ('(TMP9 * TMP11)',              # TMP11 is not momenta only
                            '(TMP9 - M1 * M1)',            # a mass
                            '(TMP9 + cI * M1 * W1)',       # a width
                            '1'):                          # no TMP at all
            self.assertEqual(writer.denominator_in_denom_precision(denominator),
                             denominator)
