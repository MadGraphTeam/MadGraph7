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
both come from set_gather_lines and are checked together here."""

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
