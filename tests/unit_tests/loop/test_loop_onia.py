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
"""Unit tests of the one-loop amplitudes of processes with bound states
(quarkonia) in MadLoop: the loop_sm_onia model and its generator."""

import ast
import filecmp
import os
import shutil
import sys
import tempfile

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
mg_root = os.path.join(root_path, os.path.pardir, os.path.pardir)
sys.path.append(mg_root)
sys.path.append(os.path.join(mg_root, 'tools'))

import tests.unit_tests as unittest

import make_onia_model
import madgraph.models.import_boundstates as import_boundstates

_model_dir = os.path.join(os.path.abspath(mg_root), 'models')


class TestLoopSmOniaModel(unittest.TestCase):
    """models/loop_sm_onia is models/loop_sm plus the Fock states of sm_onia,
    as written by tools/make_onia_model.py."""

    loop_sm = os.path.join(_model_dir, 'loop_sm')
    loop_sm_onia = os.path.join(_model_dir, 'loop_sm_onia')

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='onia_model_')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def tracked_files(self, path):
        return sorted(f for f in os.listdir(path)
                      if not f.endswith(('.pkl', '.log', '.pyc')) and f != '__pycache__')

    def test_generator_reproduces_the_shipped_model(self):
        """Regenerating loop_sm_onia from loop_sm gives the shipped files."""
        out = os.path.join(self.tmpdir, 'loop_sm_onia')
        kept, dropped = make_onia_model.make_onia_model(
            self.loop_sm, out, os.path.join(self.loop_sm_onia, 'boundstates.py'))
        self.assertEqual((kept, dropped), (192, 0))
        self.assertEqual(self.tracked_files(out), self.tracked_files(self.loop_sm_onia))
        for name in self.tracked_files(out):
            new = open(os.path.join(out, name)).read().splitlines()
            old = open(os.path.join(self.loop_sm_onia, name)).read().splitlines()
            if name == 'boundstates.py':
                # the two provenance lines name the source of the Fock states
                new, old = new[2:], old[2:]
            self.assertEqual(new, old, name)

    def test_only_the_onia_files_differ_from_loop_sm(self):
        """Particles, couplings and counterterms are those of loop_sm."""
        common = [f for f in self.tracked_files(self.loop_sm)]
        differing = [f for f in common if not filecmp.cmp(
            os.path.join(self.loop_sm, f), os.path.join(self.loop_sm_onia, f), shallow=False)]
        self.assertEqual(sorted(differing), ['__init__.py', 'object_library.py'])
        self.assertEqual(sorted(set(self.tracked_files(self.loop_sm_onia)) - set(common)),
                         ['boundstates.py'])
        init = open(os.path.join(self.loop_sm_onia, '__init__.py')).read()
        self.assertIn('dual_mass_scheme = True', init)
        self.assertIn('arXiv:2510.26773', init)

    def test_constituents_are_particles_of_the_model(self):
        """Every constituent is a particle of loop_sm, the leptonia renamed by
        PDG code (loop_sm calls the muon and tau m- and tt-)."""
        names = set(make_onia_model.particle_pdgs(os.path.join(self.loop_sm, 'particles.py')))
        header, blocks = make_onia_model.boundstate_blocks(
            os.path.join(self.loop_sm_onia, 'boundstates.py'))
        self.assertEqual(len(blocks), 192)
        constituents = set(p for c, _, _ in blocks for p in c)
        self.assertTrue(constituents <= names, constituents - names)
        self.assertIn('m-', constituents)
        self.assertIn('tt+', constituents)

    def test_boundstates_load(self):
        """The Fock states load from the UFO directory, with their quantum
        numbers."""
        states = import_boundstates.get_boundstates_ufo(self.loop_sm_onia)
        self.assertEqual(len(states), 192)
        etac = [s for s in states if s.name == 'etac(1|1S01)'][0]
        self.assertEqual((etac.pdg_code, etac.particles, etac.spin, etac.orbital, etac.J,
                          etac.color), (441, ['c', 'c~'], 1, 0, 0, 1))
        psi8 = [s for s in states if s.name == 'Jpsi(1|3S18)'][0]
        self.assertEqual((psi8.spin, psi8.orbital, psi8.J, psi8.color), (3, 0, 1, 8))


if __name__ == '__main__':
    unittest.main()
