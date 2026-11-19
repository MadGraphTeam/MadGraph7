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
"""Generation of exclusive processes with generalised parton distributions

    generate a gpd > mu+ mu- / z      (gpd = gpd_g gpd_u gpd_d gpd_s)

A GPD channel gpd_X is the colour-singlet pair of partons X that leaves the
probed proton and returns to it: an incoming leg ('role': 'in') and an
outgoing one ('role': 'out'), with the elastic proton implicit. Tested here:
the leg property, the expansion of the 'gpd' label (with the options of the
line applied to every channel), the process names, the identical-particle
factor, the refusals, the removal of the diagrams in which the pair meets at
one vertex (light-like 2 xi P exchange) or a single gluon attaches to the
quark pair (zero by colour), the colour-singlet projector and the madevent
shadow process (madgraph/iolibs/gpd_madevent.py).
"""

from __future__ import absolute_import

import fractions
import os
import sys

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph.core.base_objects as base_objects
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color_algebra
import madgraph.interface.master_interface as MGCmd
import madgraph.iolibs.gpd_madevent as gpd_madevent

from madgraph import InvalidCmd


class GPDLegTest(unittest.TestCase):
    """The 'gpd' property of Leg and MultiLeg."""

    def info(self, **change):
        info = {'name': 'gpd_u', 'role': 'in', 'parton': 2, 'index': 0}
        info.update(change)
        return info

    def test_default_is_empty(self):
        self.assertEqual(base_objects.Leg()['gpd'], {})
        self.assertEqual(base_objects.MultiLeg()['gpd'], {})
        self.assertIn('gpd', base_objects.Leg().get_sorted_keys())

    def test_valid(self):
        for cls in (base_objects.Leg, base_objects.MultiLeg):
            obj = cls()
            obj.set('gpd', self.info())
            self.assertEqual(obj.get('gpd')['role'], 'in')
            obj.set('gpd', self.info(name='gpd_g', parton=21, role='out'))
            self.assertEqual(obj.get('gpd')['parton'], 21)

    def test_invalid(self):
        for bad in ([self.info()], self.info(name='gpd_x'), self.info(role='both'),
                    self.info(parton=1)):
            for cls in (base_objects.Leg, base_objects.MultiLeg):
                self.assertRaises(base_objects.PhysicsObject.PhysicsObjectError,
                                  cls().filter, 'gpd', bad)
                # set() refuses it (logged) and keeps the old value
                obj = cls()
                self.assertFalse(obj.set('gpd', bad))
                self.assertEqual(obj.get('gpd'), {})

    def test_channels(self):
        self.assertEqual(base_objects.GPD_DEFAULT_LABEL, ['gpd_g', 'gpd_u', 'gpd_d', 'gpd_s'])
        self.assertEqual(base_objects.GPD_CHANNELS['gpd_g'], 21)
        self.assertEqual(base_objects.GPD_CHANNELS['gpd_s'], 3)


class GPDGenerationTest(unittest.TestCase):
    """Generation in the sm model."""

    interface = None

    def setUp(self):
        if GPDGenerationTest.interface is None:
            interface = MGCmd.MasterCmd()
            interface.exec_cmd('import model sm', printcmd=False, precmd=True)
            GPDGenerationTest.interface = interface
        self.cmd = GPDGenerationTest.interface

    def generate(self, line):
        self.cmd.do_generate(line)
        return self.cmd._curr_amps

    def test_pair_legs(self):
        """The pair is an incoming and an outgoing leg of the same parton,
        named once in the process strings; the returning parton is no
        final-state particle of the identical-particle factor."""
        amps = self.generate('a gpd_u > a a')
        self.assertEqual(len(amps), 1)
        proc = amps[0].get('process')
        legs = [(l.get('id'), l.get('state'), l.get('gpd').get('role')) for l in proc.get('legs')]
        self.assertEqual(legs, [(22, False, None), (2, False, 'in'), (22, True, None), (22, True, None),
                                (2, True, 'out')])
        self.assertEqual(proc.nice_string().split('\n')[0].split(' WEIGHTED')[0], 'Process: a gpd_u > a a')
        self.assertTrue(proc.shell_string().endswith('_agpdu_aa'))
        self.assertEqual(proc.identical_particle_factor(), 2)
        amps = self.generate('a gpd_u > mu+ mu-')
        self.assertEqual(amps[0].get('process').identical_particle_factor(), 1)

    def test_label_expansion_with_options(self):
        """'gpd' expands into its channels (here gpd_g has no tree diagram);
        forbidden particles and the process number apply to all of them."""
        amps = self.generate('a gpd > mu+ mu- / z @3')
        names = [a.get('process').get('legs')[1].get('gpd').get('name') for a in amps]
        self.assertEqual(names, ['gpd_u', 'gpd_d', 'gpd_s'])
        for amp in amps:
            proc = amp.get('process')
            self.assertEqual(proc.get('forbidden_particles'), [23])
            self.assertEqual(proc.get('id'), 3)
            self.assertEqual(len(amp.get('diagrams')), 2)

    def test_refusals(self):
        for line, text in [('gpd gpd > mu+ mu-', 'both beams'),
                           ('gpd_u gpd_d > a a', 'both beams'),
                           ('a > gpd_u mu+ mu-', 'initial state'),
                           ('a gpd > mu+ mu- [QCD]', 'leading order')]:
            try:
                self.cmd.do_generate(line)
            except InvalidCmd as error:
                self.assertIn(text, str(error), line)
            else:
                self.fail("no refusal of '%s'" % line)

    def test_pair_vertex_filter(self):
        """Diagrams where the pair meets at one vertex and continues as one
        massless propagator (the light-like 2 xi P) are removed: the
        Bethe-Heitler photon (gamma gamma fusion) and the gluon (zero by
        colour). A massive propagator (Z) is kept."""
        self.assertEqual(len(self.generate('a u > u mu+ mu- / z')[0].get('diagrams')), 4)
        self.assertEqual(len(self.generate('a gpd_u > mu+ mu- / z')[0].get('diagrams')), 2)
        # with the Z: the two Z-exchange Bethe-Heitler graphs stay
        self.assertEqual(len(self.generate('a gpd_u > mu+ mu-')[0].get('diagrams')), 6)
        self.assertEqual(len(self.generate('a g > g t t~')[0].get('diagrams')), 8)
        self.assertEqual(len(self.generate('a gpd_g > t t~')[0].get('diagrams')), 6)

    def test_single_gluon_on_quark_pair(self):
        """One gluon on the line of a colour-singlet quark pair is Tr T^a = 0:
        every O(alpha_s) graph of a gpd_u > c c~ vanishes, and generation
        moves to the photon-exchange graphs of the next order."""
        plain = self.generate('a u > u c c~ / z')[0]
        self.assertEqual(plain.get('process').get('orders'), {'WEIGHTED': 4})
        self.assertEqual(len(plain.get('diagrams')), 4)
        amp = self.generate('a gpd_u > c c~ / z')[0]
        self.assertEqual(len(amp.get('diagrams')), 2)
        for diag in amp.get('diagrams'):
            self.assertEqual(diag.get('orders')['QCD'], 0)
            self.assertEqual(diag.get('orders')['QED'], 3)
        # of the four photon-exchange graphs, the two with the photon on the
        # c line are the Bethe-Heitler ones
        self.assertEqual(len(self.generate('a u > u c c~ / z QCD=0')[0].get('diagrams')), 4)

    def test_colour_projector(self):
        basis = color_amp.ColorBasis()
        proc = self.generate('a gpd_g > t t~')[0].get('process')
        projector = basis.GPDColorString(proc)
        expected = color_algebra.ColorString([color_algebra.Tr(2, 5)])
        expected.coeff = fractions.Fraction(2, 1)
        self.assertEqual(projector, expected)
        proc = self.generate('a gpd_u > a a')[0].get('process')
        projector = basis.GPDColorString(proc)
        self.assertEqual(projector, color_algebra.ColorString([color_algebra.T(2, 5)]))
        self.assertEqual(projector.coeff, 1)
        proc = self.generate('a u > u a a')[0].get('process')
        self.assertIsNone(basis.GPDColorString(proc))

    def test_colour_basis_singlet(self):
        """After the projector, the colour of a gpd_g > t t~ is that of the
        t t~ pair alone: one colour flow, delta(t, t~)."""
        amp = self.generate('a gpd_g > t t~')[0]
        basis = color_amp.ColorBasis(amp)
        self.assertEqual(len(basis), 1)

    def test_shadow_process(self):
        """madevent: the channels of one physical process become one shadow
        a z > X (the pair as a massless pdg-23 leg), with the mirror."""
        amps = self.generate('a gpd > mu+ mu- / z @3')
        self.assertTrue(all(gpd_madevent.is_gpd_amplitude(a) for a in amps))
        keys = set(gpd_madevent.physical_key(a) for a in amps)
        self.assertEqual(keys, set([((22, 'pair'), ((-13, 0, -1), (13, 0, -1)), 3)]))
        shadows = gpd_madevent.make_shadows(amps, self.cmd._curr_model)
        self.assertEqual(len(shadows), 1)
        proc = shadows[0].get('process')
        self.assertEqual([l.get('id') for l in proc.get('legs')], [22, gpd_madevent.SHADOW_PDG, -13, 13])
        self.assertTrue(shadows[0].get('has_mirror_process'))
        self.assertEqual(len(gpd_madevent.SHADOWS[proc.nice_string()]), 3)
        self.assertFalse(gpd_madevent.is_gpd_amplitude(shadows[0]))


if __name__ == '__main__':
    unittest.main()
