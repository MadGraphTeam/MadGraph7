"""A fixed-order run whose cross section vanishes exactly stops with a message."""

import unittest

from madgraph.interface.amcatnlo_run_interface import aMCatNLOCmd, aMCatNLOError


class TestZeroCrossSection(unittest.TestCase):
    def command(self, xsect, xseca):
        command = object.__new__(aMCatNLOCmd)
        command.cross_sect_dict = {'xsect': xsect, 'errt': 0., 'xseca': xseca, 'erra': 0.}
        return command

    def jobs(self, result):
        return [{'resultABS': result, 'errorABS': 0.1 * result, 'niters_done': 2,
                 'npoints_done': 1000, 'mint_mode': 0}]

    def test_vanishing_born_stops_with_a_message(self):
        # e.g. "launch LO" of p p > chic1(1|3P11): every channel is zero (it divided by zero)
        with self.assertRaises(aMCatNLOError) as ctx:
            self.command(0., 0.).update_jobs_to_run(0.01, 0, self.jobs(0.))
        self.assertIn('exactly zero', str(ctx.exception))

    def test_cancelling_cross_section_is_refined(self):
        # a total that nearly cancels with non-zero channels is refined as usual
        command = self.command(1e-3, 2.)
        command.cross_sect_dict['errt'] = 0.5
        jobs = command.update_jobs_to_run(0.01, 1, self.jobs(2.))
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['mint_mode'], -1)


if __name__ == '__main__':
    unittest.main()
