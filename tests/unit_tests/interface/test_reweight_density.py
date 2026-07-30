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
""" Test of the average density matrix helpers of the density mode.

Those helpers are shared by DensityInterface (which accumulates the average
while it reweights the events) and by CommonRunCmd.do_reweight (which has to
re-build the average from the recombined event file after a multicore run).
"""

from __future__ import absolute_import
import os
import shutil
import tempfile
import unittest

import madgraph.interface.reweight_interface as rwgt_interface

pjoin = os.path.join


class TestAverageDensityMatrix(unittest.TestCase):
    """check the average density matrix helpers"""

    # a 2x2 density matrix is stored as its upper triangle: (00, 01, 11)
    events = [(1.0, [0.6+0j, 0.1+0.2j, 0.4+0j]),
              (2.0, [0.5+0j, 0.0-0.1j, 0.5+0j]),
              (1.0, [0.7+0j, 0.2+0.0j, 0.3+0j])]

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='rwgt_density')
        self.lhe_path = pjoin(self.tmpdir, 'unweighted_events.lhe')
        self.write_lhe(self.lhe_path, self.events)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @staticmethod
    def write_lhe(path, events):
        """write a minimal event file where each event carries a <density> tag,
        exactly as DensityInterface does"""

        text = ['<LesHouchesEvents version="3.0">', '<init>', '</init>']
        for wgt, density in events:
            text.append('<event>')
            text.append(' 2  1 %+13.7e 1.0000000e+02 7.5000000e-03 1.2000000e-01' % wgt)
            text.append('       21 -1 0 0 501 502 0. 0. 0. 0. 0. 0. 1.')
            text.append('        6  1 1 2 501   0 0. 0. 0. 0. 0. 0. 1.')
            text.append('<density> %s</density>' % \
                        ''.join('%s ' % complex(value) for value in density))
            text.append('</event>')
        text.append('</LesHouchesEvents>')
        with open(path, 'w') as fsock:
            fsock.write('\n'.join(text) + '\n')

    def test_average_normalised(self):
        """with matrix_normalisation the average is weighted by the event weight"""

        rho_avg = rwgt_interface.average_density_matrix_from_lhe(self.lhe_path, True)

        total_wgt = sum(wgt for wgt, _ in self.events)
        solution = [sum(wgt * density[i] for wgt, density in self.events) / total_wgt
                                          for i in range(len(self.events[0][1]))]
        self.assertEqual(len(rho_avg), len(solution))
        for value, expected in zip(rho_avg, solution):
            self.assertAlmostEqual(value.real, expected.real, places=12)
            self.assertAlmostEqual(value.imag, expected.imag, places=12)

    def test_average_not_normalised(self):
        """without matrix_normalisation the average is a plain event average"""

        rho_avg = rwgt_interface.average_density_matrix_from_lhe(self.lhe_path, False)

        nb_event = len(self.events)
        solution = [sum(density[i] for _, density in self.events) / nb_event
                                          for i in range(len(self.events[0][1]))]
        for value, expected in zip(rho_avg, solution):
            self.assertAlmostEqual(value.real, expected.real, places=12)
            self.assertAlmostEqual(value.imag, expected.imag, places=12)

    def test_average_no_density(self):
        """an event file without density matrix returns nothing"""

        path = pjoin(self.tmpdir, 'no_density.lhe')
        with open(path, 'w') as fsock:
            fsock.write("""<LesHouchesEvents version="3.0">
<init>
</init>
<event>
 2  1 +1.0000000e+00 1.0000000e+02 7.5000000e-03 1.2000000e-01
       21 -1 0 0 501 502 0. 0. 0. 0. 0. 0. 1.
        6  1 1 2 501   0 0. 0. 0. 0. 0. 0. 1.
</event>
</LesHouchesEvents>
""")
        self.assertEqual(rwgt_interface.average_density_matrix_from_lhe(path), None)

    def test_label_and_path(self):
        """the canonical name does not depend on the file being gzipped or not"""

        self.assertEqual(rwgt_interface.average_density_matrix_label(self.lhe_path),
                         'unweighted_events')
        self.assertEqual(rwgt_interface.average_density_matrix_label(self.lhe_path + '.gz'),
                         'unweighted_events')
        self.assertEqual(rwgt_interface.average_density_matrix_path(self.lhe_path + '.gz'),
                         pjoin(self.tmpdir, 'Average_density_matrix_unweighted_events.txt'))

    def test_write_average(self):
        """the values are written as plain complex, not as numpy repr"""

        rho_avg = rwgt_interface.average_density_matrix_from_lhe(self.lhe_path, True)
        path = rwgt_interface.write_average_density_matrix(rho_avg, self.lhe_path)

        self.assertEqual(path, pjoin(self.tmpdir,
                                'Average_density_matrix_unweighted_events.txt'))
        text = open(path).read()
        self.assertTrue(text.startswith(
            'Average density matrix of LHE file unweighted_events:\n'))
        self.assertNotIn('np.complex', text)

        # the consumer parser reads the file line by line as a list of complex
        rho_square = []
        for line in text.split('\n')[1:]:
            if not line.strip():
                continue
            rho_square.append([complex(value.strip(' ()'))
                               for value in line.strip('\t[]').split(',')])
        self.assertEqual(len(rho_square), 2)
        for row in rho_square:
            self.assertEqual(len(row), 2)
        # hermitian, and the trace is the sum of the (normalised) diagonal
        self.assertAlmostEqual(rho_square[0][1].real, rho_square[1][0].real, places=12)
        self.assertAlmostEqual(rho_square[0][1].imag, -rho_square[1][0].imag, places=12)
        self.assertAlmostEqual(rho_square[0][0].real, rho_avg[0].real, places=12)
        self.assertAlmostEqual(rho_square[1][1].real, rho_avg[2].real, places=12)

    def test_combine_density_matrix(self):
        """the multicore recombination writes the canonical file and removes the
        per chunk ones"""

        # simulate what the multicore jobs leave behind: the recombined event file
        # plus one average density matrix per chunk of events
        chunks = [self.lhe_path + '.gz_%s.lhe' % i for i in range(3)]
        for chunk in chunks:
            with open(rwgt_interface.average_density_matrix_path(chunk), 'w') as fsock:
                fsock.write('average of a single chunk of events\n')

        canonical = rwgt_interface.combine_density_matrix(self.lhe_path, chunks)

        self.assertEqual(canonical, pjoin(self.tmpdir,
                                 'Average_density_matrix_unweighted_events.txt'))
        self.assertEqual(sorted(name for name in os.listdir(self.tmpdir)
                                if name.startswith('Average_density_matrix_')),
                         ['Average_density_matrix_unweighted_events.txt'])

        # and the content is the one of a single core run over the same events
        rho_avg = rwgt_interface.average_density_matrix_from_lhe(self.lhe_path, True)
        reference = pjoin(self.tmpdir, 'reference')
        os.mkdir(reference)
        rwgt_interface.write_average_density_matrix(rho_avg, self.lhe_path,
                                                   output_dir=reference)
        self.assertEqual(open(canonical).read(),
                open(pjoin(reference,
                     'Average_density_matrix_unweighted_events.txt')).read())

    def test_matrix_normalisation_from_card(self):
        """the option is read from the reweight card as DensityInterface does"""

        card = pjoin(self.tmpdir, 'reweight_card.dat')
        def write_card(*lines):
            with open(card, 'w') as fsock:
                fsock.write('\n'.join(lines) + '\n')

        # default value of DensityInterface when the option is absent
        write_card('# change matrix_normalisation False',
                   'change particle_in_density_matrix [6, -6]')
        self.assertEqual(rwgt_interface.get_matrix_normalisation(card), True)

        write_card('change matrix_normalisation True')
        self.assertEqual(rwgt_interface.get_matrix_normalisation(card), True)

        write_card('change matrix_normalisation False')
        self.assertEqual(rwgt_interface.get_matrix_normalisation(card), False)

        # anything else is refused, as in do_change_matrix_normalisation
        write_card('change matrix_normalisation garbage')
        self.assertEqual(rwgt_interface.get_matrix_normalisation(card), False)

        # last occurence wins, as when the card is executed line by line
        write_card('change matrix_normalisation True',
                   'change matrix_normalisation False')
        self.assertEqual(rwgt_interface.get_matrix_normalisation(card), False)

        self.assertEqual(rwgt_interface.get_matrix_normalisation(
                                        pjoin(self.tmpdir, 'no_such_card.dat')), True)

    def test_combine_density_matrix_uses_the_card(self):
        """matrix_normalisation False switches to the plain event average"""

        card = pjoin(self.tmpdir, 'reweight_card.dat')
        with open(card, 'w') as fsock:
            fsock.write('change matrix_normalisation False\n')

        canonical = rwgt_interface.combine_density_matrix(self.lhe_path,
                                                          reweight_card=card)
        rho_avg = rwgt_interface.average_density_matrix_from_lhe(self.lhe_path, False)
        rho_square = [[complex(value.strip(' ()'))
                       for value in line.strip('\t[]').split(',')]
                      for line in open(canonical).read().split('\n')[1:] if line.strip()]

        self.assertAlmostEqual(rho_square[0][0].real, rho_avg[0].real, places=12)
        self.assertAlmostEqual(rho_square[1][1].real, rho_avg[2].real, places=12)
