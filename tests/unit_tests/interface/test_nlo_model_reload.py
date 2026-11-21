"""Physical aliases survive the grouped-to-NLO model transition."""
import unittest

from madgraph.interface.master_interface import MasterCmd


class TestNLOModelReload(unittest.TestCase):
    def setUp(self):
        self.cmd = MasterCmd()
        self.cmd.no_notification()
        self.cmd.exec_cmd('set apply_flavor_grouping True --no_save',
                          errorhandling=False, printcmd=False)
        self.cmd.exec_cmd('import model loop_sm', errorhandling=False,
                          printcmd=False)

    def test_decay_all_after_loop_production_has_only_physical_ids(self):
        old_groups = self.cmd._curr_model.merged_particles
        self.assertTrue(old_groups)
        self.assertTrue(any(abs(pid) in old_groups
                            for pid in self.cmd._multiparticles['all']))
        self.cmd.validate_model('virtual', 'QCD')
        self.assertFalse(self.cmd._curr_model.merged_particles)
        # MadSpin generates its tree decays after a loop-induced production.
        # A stale merged ID in "all" used to crash MultiProcess mass lookup.
        self.cmd.exec_cmd('generate w+ > all all', errorhandling=False,
                          printcmd=False)
        self.assertTrue(self.cmd._curr_amps)
        for amplitude in self.cmd._curr_amps:
            for leg in amplitude.get('process').get('legs'):
                self.assertIsNotNone(self.cmd._curr_model.get_particle(leg.get('id')))
        self.assertTrue(self.cmd.options['apply_flavor_grouping'])

    def test_custom_signed_aliases_keep_members_and_alternatives(self):
        groups = self.cmd._curr_model.merged_particles
        quarks = next(pid for pid, members in groups.items() if 2 in members)
        members = groups[quarks]
        self.cmd._multiparticles['signed'] = [quarks, -quarks, 21]
        self.cmd._multiparticles['alternatives'] = [[quarks, 21], [-quarks, 22]]
        self.cmd._multiparticles['physical'] = [2, -2]
        self.cmd.validate_model('virtual', 'QCD')
        self.assertEqual(self.cmd._multiparticles['signed'],
                         members + [-pid for pid in members] + [21])
        self.assertEqual(self.cmd._multiparticles['alternatives'],
                         [members + [21], [-pid for pid in members] + [22]])
        self.assertEqual(self.cmd._multiparticles['physical'], [2, -2])
