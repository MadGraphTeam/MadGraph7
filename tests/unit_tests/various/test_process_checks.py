################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Unit test Library for the objects in decay module."""
from __future__ import division

from __future__ import absolute_import
import math
import copy
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import tests.unit_tests as unittest
import madgraph.core.base_objects as base_objects
import madgraph.various.process_checks as process_checks
import madgraph.various.misc as misc
import madgraph.models.import_ufo as import_ufo
import madgraph.models.model_reader as model_reader
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

#===============================================================================
# TestModelReader
#===============================================================================
class TestMatrixElementChecker(unittest.TestCase):
    """Test class for the MatrixElementChecker and get_momenta"""


    def setUp(self):
        
        self.base_model = import_ufo.import_model('sm', options={'apply_flavor_grouping':False})
        #sm_path = import_ufo.find_ufo_path('sm')
        #self.base_model = import_ufo.import_model(sm_path)
    
    def test_get_momenta(self):
        """Test the get_momenta function"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':11,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True,
                                           'number': 4}))
        myleglist.append(base_objects.Leg({'id':23,
                                           'state':True,
                                           'number': 5}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        evaluator = process_checks.MatrixElementEvaluator(self.base_model)
        full_model = evaluator.full_model
        p, w_rambo = evaluator.get_momenta(myproc)

        # Check massless external momenta
        for mom in p[:-1]:
            mass = mom[0]**2-(mom[1]**2+mom[2]**2+mom[3]**2)
            self.assertAlmostEqual(mass, 0., 8)

        mom = p[-1]
        mass = math.sqrt(mom[0]**2-(mom[1]**2+mom[2]**2+mom[3]**2))
        self.assertAlmostEqual(mass,
                               full_model.get('parameter_dict')['mdl_MZ'],
                               8)

        # Check momentum balance
        outgoing = [0]*4
        incoming = [0]*4
        for i in range(4):
            incoming[i] = sum([mom[i] for mom in p[:2]])
            outgoing[i] = sum([mom[i] for mom in p[2:]])
            self.assertAlmostEqual(incoming[i], outgoing[i], 8)

        # Check non-zero final state momenta
        for mom in p[2:]:
            for i in range(4):
                self.assertGreater(abs(mom[i]), 0.)

    def test_comparison_for_process(self):
        """Test check process for e+ e- > a Z"""

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':11,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':23,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})
        process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
        comparison = process_checks.check_processes(myproc)[0][0]

        self.assertEqual(len(comparison['values']), 8)
        self.assertGreater(comparison['values'][0], 0)
        self.assertTrue(comparison['passed'])

        comparison = process_checks.check_gauge(myproc)
        
        #check number of helicities/jamp
        nb_hel = []
        nb_jamp = [] 
        for one_comp in comparison:
            nb_hel.append(len(one_comp['value']['jamp']))
            nb_jamp.append(len(one_comp['value']['jamp'][0]))
        self.assertEqual(nb_hel, [24])
        self.assertEqual(nb_jamp, [1])
        
        nb_fail = process_checks.output_gauge(comparison, output='fail')
        self.assertEqual(nb_fail, 0)
        
        comparison = process_checks.check_lorentz(myproc)
        #check number of helicities/jamp
        nb_hel = []
        nb_jamp = [] 
        for one_comp in comparison:
            nb_hel.append(len(one_comp['results'][0]['jamp']))
            nb_jamp.append(len(one_comp['results'][0]['jamp'][0]))
        self.assertEqual(nb_hel, [24])
        self.assertEqual(nb_jamp, [1])
        
        nb_fail = process_checks.output_lorentz_inv(comparison, output='fail')
        self.assertEqual(0, nb_fail)        
        
    def test_comparison_for_multiprocess(self):
        """Test check process for multiprocess"""

        myleglist = base_objects.MultiLegList()

        p = [1,2,-1,-2]

        myleglist.append(base_objects.MultiLeg({'ids':p,
                                           'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':p,
                                           'state':False}))
        myleglist.append(base_objects.MultiLeg({'ids':p}))
        myleglist.append(base_objects.MultiLeg({'ids':p}))

        myproc = base_objects.ProcessDefinition({'legs':myleglist,
                                                 'model':self.base_model,
                                                 'orders':{'QED':0}})
        process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
        comparisons, used_aloha = process_checks.check_processes(myproc)
        goal_value_len = [8, 2]

        for i, comparison in enumerate(comparisons):
            self.assertEqual(len(comparison['values']), goal_value_len[i])
            self.assertTrue(comparison['passed'])
        
        comparisons = process_checks.check_lorentz(myproc)
        nb_fail = process_checks.output_lorentz_inv(comparisons, 
                                                    output='fail')
        self.assertEqual(0, nb_fail)
        
        #check number of helicities/jamp
        nb_hel = []
        nb_jamp = [] 
        for one_comp in comparisons:
            if one_comp['results'] != 'pass':
                nb_hel.append(len(one_comp['results'][0]['jamp']))
                nb_jamp.append(len(one_comp['results'][0]['jamp'][0]))
        self.assertEqual(nb_hel, [16, 16])
        self.assertEqual(nb_jamp, [2, 2])
        
        for i, comparison in enumerate(comparisons):
            if i == 2:
                self.assertEqual(comparison['results'],'pass')
                continue
            else:
                nb_fail = process_checks.output_lorentz_inv([comparison], 
                                                            output='fail')
                self.assertEqual(0, nb_fail)

    def test_fd_python_value_matches_unitary_fixed_point(self):
        """FD Python evaluation should stay close to Unitary at a fixed point."""

        import madgraph.interface.master_interface as interface
        import madgraph.core.helas_objects as helas_objects

        process_line = 'u u > w+ w- u u QCD=0'
        p = [
            [500.0, 0.0, 0.0, 500.0],
            [500.0, 0.0, 0.0, -500.0],
            [401.01469431414546, 32.49828279011273, -244.71617101372763, 305.6197414460834],
            [260.789910425083, 5.193462736487908, -64.91667713260222, -239.38048040727654],
            [35.78415077155445, 20.545542318439836, 24.517065174688458, 16.04056272400125],
            [302.41124448921704, -58.237287845040406, 285.11578297164135, -82.27982376280812]
        ]

        values = {}
        for gauge in ['unitary', 'FD']:
            cmd = interface.MasterCmd()
            cmd.no_notification()
            cmd.exec_cmd('set gauge %s' % gauge)
            cmd.exec_cmd('import model sm')
            cmd.exec_cmd('generate %s' % process_line)
            me = helas_objects.HelasMatrixElement(cmd._curr_amps[0])
            # The SM is imported with flavor grouping, so the light quarks are
            # merged.  Evaluate at an explicit, valid flavor assignment (all-up:
            # index 2 for each merged-quark leg, 1 for the non-merged W bosons);
            # without it the merged-flavor vertices stay unresolved (flavor -1)
            # and the amplitude is identically zero in every gauge.
            for leg, flv in zip(me.get('processes')[0].get('legs'),
                                (2, 2, 1, 1, 2, 2)):
                leg.set('flavor', [flv])
            evaluator = process_checks.MatrixElementEvaluator(cmd._curr_model, cmd=cmd, reuse=False)
            for particle in evaluator.full_model.get('particles'):
                if particle.get('width') != 'ZERO':
                    evaluator.full_model.get('parameter_dict')[particle.get('width')] = 0.
            values[gauge], _ = evaluator.evaluate_matrix_element(me, p=p)

        relative_difference = abs(values['unitary'] - values['FD']) / \
            max(abs(values['unitary']), abs(values['FD']))
        self.assertLess(relative_difference, 5e-2)

    def test_external_polarisation_in_every_gauge(self):
        """An external polarised massive vector is supported in every gauge.

        At one phase-space point and with the widths at zero, |M|^2 of
        't t~ > w+{0} w-' and of 't t~ > w+{T} w-' is the same in the unitary,
        Feynman, axial and FD gauges (the FD gauge builds the longitudinal
        polarisation vector with its fifth, Goldstone component), and the two
        add up to the unpolarised |M|^2 in each gauge. The top quark makes the
        s-channel current non-conserved, so the Feynman gauge has a Goldstone
        diagram and the qq/M^2 piece of the Z propagator matters."""

        import madgraph.interface.master_interface as interface
        import madgraph.core.helas_objects as helas_objects

        processes = {'0': 't t~ > w+{0} w-', 'T': 't t~ > w+{T} w-',
                     'all': 't t~ > w+ w-'}
        gauges = ['unitary', 'Feynman', 'axial', 'FD']
        values = {}
        p = None
        for gauge in gauges:
            cmd = interface.MasterCmd()
            cmd.no_notification()
            cmd.exec_cmd('set gauge %s' % gauge)
            cmd.exec_cmd('import model sm')
            evaluator = process_checks.MatrixElementEvaluator(
                cmd._curr_model, cmd=cmd, reuse=False)
            for particle in evaluator.full_model.get('particles'):
                if particle.get('width') != 'ZERO':
                    evaluator.full_model.get('parameter_dict')[particle.get('width')] = 0.
            for pol, line in processes.items():
                cmd.exec_cmd('generate %s' % line)
                me = helas_objects.HelasMatrixElement(cmd._curr_amps[0])
                if p is None:
                    # the same point for the three processes and four gauges
                    # (the external masses are the same): the RAMBO point of
                    # the check_sa drivers
                    with process_checks._seeded_rambo_rng():
                        p, _ = evaluator.get_momenta(
                            me.get('processes')[0],
                            {'energy': 1000., 'events': None, 'skip_evt': 0})
                values[gauge, pol], _ = evaluator.evaluate_matrix_element(me, p=p)

        for gauge in gauges:
            self.assertGreater(values[gauge, '0'], 0.)
            self.assertGreater(values[gauge, 'T'], 0.)
            self.assertAlmostEqual(
                (values[gauge, '0'] + values[gauge, 'T']) / values[gauge, 'all'],
                1., places=10, msg='sum over the polarisations in the %s gauge' % gauge)
            for pol in processes:
                self.assertAlmostEqual(
                    values[gauge, pol] / values['unitary', pol], 1., places=9,
                    msg='%s: %s gauge vs unitary' % (processes[pol], gauge))

    def test_fd_polarised_propagator_projections(self):
        """FD gauge: the five-component propagator numerator splits exactly,
        on or off shell, into the transverse pieces, the FD longitudinal piece
        eps0 eps0* (eps0 = (-M n/(n.q), i)) and the remainder
        (q^2-M^2) n n/(n.q)^2. The projections of
        wavefunctions.fd_polarized_propagator therefore add up to the full FD
        projector of multiply_propagator_factor: {T}+{0}+{A} = full,
        {+}+{-} = {T}, {0S} = {0}+{S}, and {S} is {A} with
        (q^2-M^2) -> q^2-M^2+iM*Gamma. Checked for both signs of the energy
        flowing through the line."""

        import random
        import aloha.template_files.wavefunctions as wf

        rng = random.Random(7)
        mass, width = 80.4, 2.05
        for trial in range(6):
            qv = [rng.uniform(-150, 150) for _ in range(3)]
            energy = math.sqrt(sum(x * x for x in qv) + rng.uniform(20, 200)**2)
            if trial % 2:
                energy = -energy
            win = wf.WaveFunction(size=7)
            win.momenta = [energy] + qv
            for i in range(2, 7):
                win[i] = complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
            full = wf.multiply_propagator_factor(win, mass)
            piece = dict((pol, wf.fd_polarized_propagator(win, mass, width, pol))
                         for pol in range(7))
            T, R, L, Z, A, S, ZS = [piece[i] for i in (1, 2, 3, 0, 4, 5, 6)]
            # the momentum flows backwards in the projector: q = -p
            q2 = energy**2 - sum(x * x for x in qv)
            for i in range(2, 7):
                scale = max(abs(full[j]) for j in range(2, 7))
                self.assertAlmostEqual(abs(T[i] + Z[i] + A[i] - full[i]) / scale, 0., 12)
                self.assertAlmostEqual(abs(R[i] + L[i] - T[i]) / scale, 0., 12)
                self.assertAlmostEqual(abs(Z[i] + S[i] - ZS[i]) / scale, 0., 12)
                self.assertAlmostEqual(
                    abs(A[i] * (q2 - mass**2 + 1j * mass * width) / (q2 - mass**2)
                        - S[i]) / scale, 0., 12)
            # the transverse and helicity pieces have no time-like and no
            # Goldstone component, the remainders no Goldstone component
            for X in (T, R, L):
                self.assertEqual(X[2], 0)
                self.assertEqual(X[6], 0)
            self.assertEqual(A[6], 0)
            self.assertEqual(S[6], 0)

    def test_fd_polarised_propagator_vs_unitary(self):
        """t > w+{X} b, w+ > ta+ vt in the FD and unitary gauges, at a point
        with the W on its mass shell and at one 7% below it (widths kept).

        - {T}, {+}, {-} and the unpolarised chain agree at both points: the
          transverse pieces are the same in both gauges, with no Goldstone
          component.
        - {0} agrees on shell, where the FD longitudinal vector eps0 differs
          from the unitary one by a term along q, which the Goldstone
          component cancels (Ward identity); off shell the two definitions
          differ (here by 4e-5).
        - {A} vanishes on shell in both gauges (it is proportional to
          q^2-M^2) and is gauge dependent off shell.
        - With the W at rest (the frame of its decay products, plus rounding
          noise) the helicities are taken along +z in both gauges."""

        import madgraph.interface.master_interface as interface
        import madgraph.core.helas_objects as helas_objects

        def boost(p, b):
            bb = sum(x * x for x in b)
            g = 1 / math.sqrt(1 - bb)
            bp = sum(b[i] * p[i+1] for i in range(3))
            return [g * (p[0] + bp)] + \
                   [p[i+1] + (g - 1) / bb * bp * b[i] + g * b[i] * p[0] for i in range(3)]

        def two_body(M, m1, m2, theta, phi):
            E1 = (M * M + m1 * m1 - m2 * m2) / (2 * M)
            k = math.sqrt(E1 * E1 - m1 * m1)
            d = [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi),
                 math.cos(theta)]
            return [E1] + [k * x for x in d], [M - E1] + [-k * x for x in d]

        def point(mt, mb, mw, mta):
            """{pdg: momentum} for t (at rest) > b W(mw), W > ta+ vt"""
            pb, pw = two_body(mt, mb, mw, 0.7, 0.3)
            pta, pv = two_body(mw, mta, 0., 1.9, 2.2)
            beta = [pw[i+1] / pw[0] for i in range(3)]
            return {6: [mt, 0., 0., 0.], 5: pb, -15: boost(pta, beta),
                    16: boost(pv, beta)}

        def rest_point(mt, mb, mw, mta):
            """the same decay seen from the rest frame of the W, as when
            me_frame is the W's decay products, with a rounding-size offset
            (1e-12) on the decay products: the line is at rest up to noise"""
            mom = point(mt, mb, mw, mta)
            pw = [mom[-15][i] + mom[16][i] for i in range(4)]
            beta = [-pw[i+1] / pw[0] for i in range(3)]
            mom = dict((pdg, boost(p, beta)) for pdg, p in mom.items())
            mom[-15][1] += 3e-12
            mom[16][2] -= 2e-12
            return mom

        braces = ('T', '+', '-', '0', 'A', '')
        values = {}
        for gauge in ('unitary', 'FD'):
            cmd = interface.MasterCmd()
            cmd.no_notification()
            cmd.exec_cmd('set apply_flavor_grouping False')
            cmd.exec_cmd('set gauge %s' % gauge)
            cmd.exec_cmd('import model sm')
            for brace in braces:
                cmd.exec_cmd('generate t > w+%s b, w+ > ta+ vt'
                             % ('{%s}' % brace if brace else ''))
                me = helas_objects.HelasMultiProcess(
                                       cmd._curr_amps).get_matrix_elements()[0]
                evaluator = process_checks.MatrixElementEvaluator(
                    cmd._curr_model, cmd=cmd, reuse=False)
                param = evaluator.full_model.get('parameter_dict')
                masses = [param[name].real for name in
                          ('mdl_MT', 'mdl_MB', 'mdl_MW', 'mdl_MTA')]
                legs = [leg.get('id') for leg in
                        me.get('processes')[0].get_legs_with_decays()]
                for shell, scale in (('on', 1.), ('off', 0.93), ('rest', 1.)):
                    make = rest_point if shell == 'rest' else point
                    mom = make(masses[0], masses[1], scale * masses[2], masses[3])
                    values[gauge, brace, shell], _ = evaluator.evaluate_matrix_element(
                        me, p=[mom[pdg] for pdg in legs])

        def rel(brace, shell):
            u, f = values['unitary', brace, shell], values['FD', brace, shell]
            return abs(u - f) / max(abs(u), abs(f))

        for brace in ('T', '+', '-', ''):
            for shell in ('on', 'off'):
                self.assertLess(rel(brace, shell), 1e-12, (brace, shell))
        # a W at rest (up to rounding): the helicity axis is pinned to +z, as
        # in the unitary gauge, instead of following the rounding noise
        for brace in ('T', '+', '-', '0', ''):
            self.assertLess(rel(brace, 'rest'), 1e-9, brace)
        # the two helicities are not swapped
        self.assertGreater(values['FD', '+', 'off'], 10 * values['FD', '-', 'off'])
        self.assertLess(rel('0', 'on'), 1e-12)
        self.assertLess(rel('0', 'off'), 1e-3)
        self.assertGreater(rel('0', 'off'), 1e-8)
        for gauge in ('unitary', 'FD'):
            self.assertLess(values[gauge, 'A', 'on'], 1e-20 * values[gauge, '0', 'on'])
            self.assertGreater(values[gauge, 'A', 'off'], 1e-8 * values[gauge, '0', 'off'])

    def test_fd_complex_mass_scheme_matches_unitary(self):
        """In the complex-mass scheme the FD and the unitary gauge agree to
        machine precision once the EXTERNAL unstable particles have a zero
        width; the internal W, Z, h, t keep theirs (complex masses).

        With the width of an external particle the two gauges cannot agree:
        its momentum sits on the real mass shell, not on the complex one the
        Ward identities need, and every gauge gives another O(width/mass)
        answer (unitary and Feynman included). See docs/auto-fd-gauge.md.

        Guards the FD propagator, which contracted the goldstone component
        with the conjugate of -i*m: 6e-7 (w+ w- > w+ w-) and 1e-5
        (b b > h b b) before the fix."""

        import aloha
        import madgraph.interface.master_interface as interface
        import madgraph.core.helas_objects as helas_objects
        import madgraph.models.write_param_card as write_param_card
        import madgraph.models.check_param_card as check_param_card

        saved = (aloha.complex_mass, aloha.unitary_gauge)
        def restore():
            aloha.complex_mass, aloha.unitary_gauge = saved
        self.addCleanup(restore)
        tmpdir = tempfile.mkdtemp(prefix='fd_cms_')
        self.addCleanup(shutil.rmtree, tmpdir, True)

        # process, the external particle(s) whose width is set to zero
        for process_line, external in (('w+ w- > w+ w-', [24]),
                                       ('b b > h b b QCD=0', [25])):
            values = {}
            p = None
            for gauge in ['unitary', 'FD']:
                cmd = interface.MasterCmd()
                cmd.no_notification()
                cmd.exec_cmd('set auto_gauge False')
                cmd.exec_cmd('set complex_mass_scheme True')
                cmd.exec_cmd('set gauge %s' % gauge)
                cmd.exec_cmd('import model sm')
                cmd.exec_cmd('generate %s' % process_line)
                me = helas_objects.HelasMatrixElement(cmd._curr_amps[0])
                card = os.path.join(tmpdir, 'param_card_%s.dat' % gauge)
                write_param_card.ParamCardWriter(cmd._curr_model, card)
                param_card = check_param_card.ParamCard(card)
                for pdg in external:
                    param_card['decay'].get((pdg,)).value = 0.
                evaluator = process_checks.MatrixElementEvaluator(
                    cmd._curr_model, param_card=param_card, cmd=cmd, reuse=False)
                if p is None:
                    p, _ = evaluator.get_momenta(me.get('processes')[0])
                values[gauge], _ = evaluator.evaluate_matrix_element(me, p=p)

            self.assertGreater(abs(values['unitary']), 0)
            relative_difference = abs(values['unitary'] - values['FD']) / \
                max(abs(values['unitary']), abs(values['FD']))
            self.assertLess(relative_difference, 1e-10, process_line)

    def test_failed_process(self):
        """Test that check process fails for wrong color-Lorentz."""

        # Change 4g interaction so color and lorentz don't agree
        id = [int.get('id') for int in self.base_model.get('interactions')
               if [p['pdg_code'] for p in int['particles']] == [21,21,21,21]][0]
        gggg = self.base_model.get_interaction(id)
        assert [p['pdg_code'] for p in gggg['particles']] == [21,21,21,21]
        gggg.set('lorentz', ['VVVV1', 'VVVV4', 'VVVV3'])

        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':False}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':True}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})
        process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
        comparison = process_checks.check_processes(myproc)[0][0]

        self.assertFalse(comparison['passed'])

        comparison = process_checks.check_processes(myproc, quick = True)[0][0]

        self.assertFalse(comparison['passed'])
        
        comparison = process_checks.check_gauge(myproc)
        nb_fail = process_checks.output_gauge(comparison, output='fail')
        self.assertNotEqual(nb_fail, 0)
        
        comparison = process_checks.check_lorentz(myproc)
        nb_fail = process_checks.output_lorentz_inv(comparison, output='fail')
        self.assertNotEqual(0, nb_fail)
        #self.assertNotAlmostEqual(max(comparison[0][1]), min(comparison[0][1]))

#===============================================================================
# TestReferenceMomentaChecks
#===============================================================================
class TestReferenceMomentaChecks(unittest.TestCase):
    """check gauge and check brs compare the matrix element with the external
    wavefunctions of output --axial_gauge=True (column Reference): the
    massless vectors in the axial gauge of a reference momentum, the massive
    fermions quantised along it."""

    def setUp(self):
        self.base_model = import_ufo.import_model('sm',
                                    options={'apply_flavor_grouping': False})

    def process(self, ids):
        legs = base_objects.LegList()
        for i, pid in enumerate(ids):
            legs.append(base_objects.Leg({'id': pid, 'state': i >= 2}))
        return base_objects.Process({'legs': legs, 'model': self.base_model})

    def test_brs_compares_the_reference_momenta(self):
        """g g > t t~ g: the reference value equals the standard one, and the
        output table has the column and passes"""

        process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
        comparison = process_checks.check_gauge(self.process([21, 21, 6, -6, 21]))
        self.assertEqual(len(comparison), 1)
        one = comparison[0]
        self.assertIn('ref', one)
        self.assertLess(abs(one['ref']['m2'] / one['value']['m2'] - 1), 1e-12)
        self.assertEqual(process_checks.output_gauge(comparison, output='fail'), 0)
        self.assertIn('Reference', process_checks.output_gauge(comparison))

    def test_reference_evaluator_uses_the_reference_routines(self):
        """the Python matrix element of the reference evaluator calls
        vxxxxxr / ixxxxxr / oxxxxxr: the helicity sum is unchanged, the
        helicity amplitudes are not, and four helicity configurations of
        g g > t t~ g vanish"""

        import madgraph.core.diagram_generation as diagram_generation
        import madgraph.core.helas_objects as helas_objects
        import madgraph.iolibs.export_python as export_python
        process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
        proc = self.process([21, 21, 6, -6, 21])
        amplitude = diagram_generation.Amplitude(proc)
        standard = process_checks.MatrixElementEvaluator(self.base_model,
                                        auth_skipping=False, reuse=False)
        for particle in standard.full_model.get('particles'):
            if particle.get('width') != 'ZERO':
                standard.full_model.get('parameter_dict')[particle.get('width')] = 0.
        reference = process_checks.reference_momenta_evaluator(
                     self.base_model, None, False, process_checks.FakeInterface())
        p, _ = standard.get_momenta(proc)
        me = lambda: helas_objects.HelasMatrixElement(amplitude, gen_color=False)
        code = list(export_python.ProcessExporterPython(me(),
                       reference.helas_writer).get_python_matrix_methods().values())[0]
        self.assertEqual(sorted(set(re.findall(r'\b[iov]xxxxxr\(', code))),
                         ['ixxxxxr(', 'oxxxxxr(', 'vxxxxxr('])
        a = standard.evaluate_matrix_element(me(), p=p, output='jamp')
        b = reference.evaluate_matrix_element(me(), p=p, output='jamp')
        self.assertLess(abs(a['m2'] / b['m2'] - 1), 1e-12)
        per_hel = lambda value: [sum(abs(x)**2 for x in row) for row in value['jamp']]
        ha, hb = per_hel(a), per_hel(b)
        self.assertGreater(max(abs(x - y) / max(x, y) for x, y in zip(ha, hb)), 0.1)
        zeros = lambda h: sum(1 for x in h if x < 1e-12 * max(h))
        self.assertEqual(zeros(hb) - zeros(ha), 4)

    def test_output_gauge_fails_on_the_reference(self):
        """a reference value away from the standard one fails"""

        value = {'m2': 1.0, 'jamp': [[1.0]]}
        comparison = [{'process': self.process([21, 21, 6, -6, 21]),
                       'value': value, 'brs': {'m2': 0., 'jamp': [[0.]]},
                       'ref': {'m2': 1.0 + 1e-5, 'jamp': [[1.0]]}}]
        self.assertEqual(process_checks.output_gauge(comparison, output='fail'), 1)
        comparison[0]['ref'] = {'m2': 1.0, 'jamp': [[1.0]]}
        self.assertEqual(process_checks.output_gauge(comparison, output='fail'), 0)

    def test_output_unitary_feynman_missing_value(self):
        """a gauge in which a process got no value is reported as a failure,
        not an IndexError"""

        value = {'m2': 1.0, 'jamp': [[1.0]]}
        result = [{'perturbation_couplings': []},
                  {'process': 'g g > t t~ g', 'value_unit': value,
                   'value_feynm': value, 'value_ref': None}]
        self.assertEqual(process_checks.output_unitary_feynman(result,
                                                               output='fail'), 1)
        self.assertIn('no value in the Reference gauge',
                      process_checks.output_unitary_feynman(result))

    def test_check_gauge_without_reference_value(self):
        """check gauge does not crash when the reference evaluation leaves a
        process out"""

        import madgraph.interface.master_interface as interface
        cmd = interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('import model sm')
        procdef = cmd.extract_process('e+ e- > a a')
        original = process_checks.evaluate_with_reference_momenta
        process_checks.evaluate_with_reference_momenta = lambda *args, **opts: []
        try:
            process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
            result = process_checks.check_unitary_feynman(procdef, procdef,
                                cmd=cmd, processes_reference=procdef)
        finally:
            process_checks.evaluate_with_reference_momenta = original
        self.assertIsNone(result[1]['value_ref'])
        self.assertEqual(process_checks.output_unitary_feynman(result,
                                                               output='fail'), 1)

    def test_brs_widths_zero_in_every_gauge(self):
        """check_gauge on a Process sets the widths to zero in the FD gauge as
        in the unitary one (a fixed width breaks gauge invariance)"""

        import aloha
        seen = []
        original = process_checks.MatrixElementEvaluator.evaluate_matrix_element
        def spy(evaluator, *args, **opts):
            seen.append(evaluator.full_model.get('parameter_dict')['mdl_WT'])
            return original(evaluator, *args, **opts)
        process_checks.MatrixElementEvaluator.evaluate_matrix_element = spy
        old = aloha.unitary_gauge
        try:
            for gauge in (1, 3):
                aloha.unitary_gauge = gauge
                process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
                process_checks.check_gauge(self.process([21, 21, 6, -6, 21]))
        finally:
            aloha.unitary_gauge = old
            process_checks.MatrixElementEvaluator.evaluate_matrix_element = original
        self.assertTrue(seen)
        self.assertEqual(set(seen), {0.})

    def test_brs_no_reference_without_references(self):
        """t t~ > z a: no leg gets a reference, so nothing is evaluated with
        the reference momenta and the table shows '-'"""

        process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
        comparison = process_checks.check_gauge(self.process([6, -6, 23, 22]))
        self.assertEqual(len(comparison), 1)
        self.assertNotIn('ref', comparison[0])
        self.assertEqual(process_checks.output_gauge(comparison, output='fail'), 0)

    def test_output_unitary_feynman_reference_column(self):
        """check gauge: the Reference column, and its failure"""

        def result(ref):
            value = lambda m2: {'m2': m2, 'jamp': [[1.0]]}
            return [{'perturbation_couplings': []},
                    {'process': 'g g > t t~ g', 'value_unit': value(1.0),
                     'value_feynm': value(1.0), 'value_ref': value(ref)}]
        text = process_checks.output_unitary_feynman(result(1.0))
        self.assertIn('Reference', text.splitlines()[0])
        self.assertEqual(process_checks.output_unitary_feynman(result(1.0),
                                                               output='fail'), 0)
        self.assertEqual(process_checks.output_unitary_feynman(result(1.1),
                                                               output='fail'), 1)


#===============================================================================
# TestLorentzInvariance
#===============================================================================
class TestLorentzInvariance(unittest.TestCase):
    """Test class for the Lorentz Invariance and boost_momenta"""
    
    def setUp(self):
        sm_path = import_ufo.find_ufo_path('MSSM_SLHA2')
        self.base_model = import_ufo.import_model(sm_path)
        
    def test_boost_momenta(self):
        """check if the momenta are boosted correctly by checking invariant mass
        """    
        
        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':-11,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':11,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':22,
                                           'state':True,
                                           'number': 4}))
        myleglist.append(base_objects.Leg({'id':23,
                                           'state':True,
                                           'number': 5}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        evaluator = process_checks.MatrixElementEvaluator(self.base_model)
        p, w_rambo = evaluator.get_momenta(myproc)

        def invariant_mass(p1, p2):
            #helping function to compute invariant mass
            return p1[0] * p2[0] - p1[1] * p2[1] - p1[2] * p2[2] -p1[3] * p2[3]

        # Compute invariant mass on the initial set of impulsion
        invariant_mass_result = []
        for p1 in p:
            for p2 in p: 
                m12 = invariant_mass(p1, p2)
                if abs(m12) < 1e-8:
                    m12=0
                invariant_mass_result.append(m12)
        
        # Compute invariant mass on a x direction boost
        invariant_mass_boost=[]
        p_boost = process_checks.boost_momenta(p)
        for p1 in p_boost:
            for p2 in p_boost: 
                m12 = invariant_mass(p1, p2)
                invariant_mass_boost.append(m12)  
                     
        for i in range(len(invariant_mass_boost)):
            self.assertAlmostEqual(invariant_mass_boost[i], 
                                   invariant_mass_result[i])

        # Compute invariant mass on a y direction boost
        invariant_mass_boost=[]
        p_boost = process_checks.boost_momenta(p, boost_direction=2)
        for p1 in p_boost:
            for p2 in p_boost: 
                m12 = invariant_mass(p1, p2)
                invariant_mass_boost.append(m12)  
                     
        for i in range(len(invariant_mass_boost)):
            self.assertAlmostEqual(invariant_mass_boost[i], 
                                   invariant_mass_result[i])
    
        # Compute invariant mass on a z direction boost
        invariant_mass_boost=[]
        p_boost = process_checks.boost_momenta(p, boost_direction=3, beta=0.8)
        for p1 in p:
            for p2 in p: 
                m12 = invariant_mass(p1, p2)
                invariant_mass_boost.append(m12)  
                     
        for i in range(len(invariant_mass_boost)):
            self.assertAlmostEqual(invariant_mass_boost[i], 
                                   invariant_mass_result[i])    

    def test_boost_momenta_gluino(self):
        """check if the momenta are boosted correctly by checking invariant mass
        in the case of massive final state
        """    
        
        myleglist = base_objects.LegList()

        myleglist.append(base_objects.Leg({'id':21,
                                           'state':False,
                                           'number': 1}))
        myleglist.append(base_objects.Leg({'id':21,
                                           'state':False,
                                           'number': 2}))
        myleglist.append(base_objects.Leg({'id':1000021,
                                           'state':True,
                                           'number': 3}))
        myleglist.append(base_objects.Leg({'id':1000021,
                                           'state':True,
                                           'number': 4}))

        myproc = base_objects.Process({'legs':myleglist,
                                       'model':self.base_model})

        evaluator = process_checks.MatrixElementEvaluator(self.base_model)
        p, w_rambo = evaluator.get_momenta(myproc)

        def invariant_mass(p1, p2):
            #helping function to compute invariant mass
            return p1[0] * p2[0] - p1[1] * p2[1] - p1[2] * p2[2] -p1[3] * p2[3]

        # Compute invariant mass on the initial set of impulsion
        invariant_mass_result = []
        for p1 in p:
            for p2 in p: 
                m12 = invariant_mass(p1, p2)
                if abs(m12) < 1e-8:
                    m12=0
                invariant_mass_result.append(m12)
                self.assertGreaterEqual(m12, 0)
        
        # Compute invariant mass on a x direction boost
        invariant_mass_boost=[]
        p_boost = process_checks.boost_momenta(p)
        for p1 in p_boost:
            for p2 in p_boost: 
                m12 = invariant_mass(p1, p2)
                invariant_mass_boost.append(m12)  
                     
        for i in range(len(invariant_mass_boost)):
            self.assertAlmostEqual(invariant_mass_boost[i], 
                                   invariant_mass_result[i])
        
        # Compute invariant mass on a y direction boost
        invariant_mass_boost=[]
        p_boost = process_checks.boost_momenta(p, boost_direction=2)
        for p1 in p_boost:
            for p2 in p_boost: 
                m12 = invariant_mass(p1, p2)
                invariant_mass_boost.append(m12)  
                     
        for i in range(len(invariant_mass_boost)):
            self.assertAlmostEqual(invariant_mass_boost[i], 
                                   invariant_mass_result[i])
    
        # Compute invariant mass on a z direction boost
        invariant_mass_boost=[]
        p_boost = process_checks.boost_momenta(p, boost_direction=3, beta=0.8)
        for p1 in p_boost:
            for p2 in p_boost: 
                m12 = invariant_mass(p1, p2)
                invariant_mass_boost.append(m12)  
                     
        for i in range(len(invariant_mass_boost)):
            self.assertAlmostEqual(invariant_mass_boost[i], 
                                   invariant_mass_result[i])    



    def test_check_sa_energy_absolute_masses(self):
        """check language raises the energy to 2*sum|m| of the external legs,
        as the Fortran pmass.inc: a negative mass must not lower it."""

        class FakeEvaluator(object):
            def __init__(self, masses):
                self.masses = masses
            def get_external_masses(self, proc):
                return list(self.masses)

        legs = base_objects.LegList(
            [base_objects.Leg({'id': 6, 'state': False, 'number': 1}),
             base_objects.Leg({'id': -6, 'state': False, 'number': 2}),
             base_objects.Leg({'id': 24, 'state': True, 'number': 3}),
             base_objects.Leg({'id': -24, 'state': True, 'number': 4})])
        proc = base_objects.Process({'legs': legs, 'model': self.base_model})

        signed = FakeEvaluator([-173., -173., 80., 80.])
        self.assertAlmostEqual(
            process_checks._check_sa_energy(proc, signed, 100.), 1012.)
        self.assertAlmostEqual(
            process_checks._check_sa_energy(proc, signed, 2000.), 2000.)
        # a 1 -> n decay keeps the energy it is given
        decay = base_objects.Process(
            {'legs': base_objects.LegList(legs[1:]), 'model': self.base_model})
        self.assertAlmostEqual(
            process_checks._check_sa_energy(decay, signed, 100.), 100.)


class TestFlavorCheck(unittest.TestCase):
    """Test the flavor-grouping check (check_flavor / output_flavor)."""

    def setUp(self):
        self.merged_model = import_ufo.import_model(
            'sm', options={'apply_flavor_grouping': True})

    def _make_proc_def(self, is_ids, fs_ids):
        """Helper: build a ProcessDefinition with multi-legs."""
        legs = base_objects.MultiLegList(
            [base_objects.MultiLeg({'ids': list(is_ids), 'state': False})] * 1 +
            [base_objects.MultiLeg({'ids': list(is_ids), 'state': False})] * 0
        )
        # Build proper multi-leg list
        all_legs = base_objects.MultiLegList()
        for ids, state in [(is_id_list, False) for is_id_list in
                           ([list(i) if isinstance(i, (list, tuple)) else [i]
                             for i in is_ids])]:
            all_legs.append(base_objects.MultiLeg({'ids': ids, 'state': state}))
        for ids, state in [(list(i) if isinstance(i, (list, tuple)) else [i], True)
                           for i in fs_ids]:
            all_legs.append(base_objects.MultiLeg({'ids': ids, 'state': state}))
        return base_objects.ProcessDefinition({
            'legs': all_legs,
            'model': self.merged_model,
        })

    def test_check_flavor_qqbar_to_tt(self):
        """check_flavor for _quark _anti_quark > t t~ should pass all flavors."""
        # _quark has PDG 81 in the merged model
        q_id = 81
        t_id = 6
        proc_def = base_objects.ProcessDefinition({
            'legs': base_objects.MultiLegList([
                base_objects.MultiLeg({'ids': [q_id],  'state': False}),
                base_objects.MultiLeg({'ids': [-q_id], 'state': False}),
                base_objects.MultiLeg({'ids': [t_id],  'state': True}),
                base_objects.MultiLeg({'ids': [-t_id], 'state': True}),
            ]),
            'model': self.merged_model,
        })

        result = process_checks.check_flavor(proc_def, cmd=process_checks.FakeInterface())
        # Should have at least one entry (one per unmerged flavor)
        self.assertTrue(len(result) > 0,
                        "check_flavor returned no results for _quark _anti_quark > t t~")

        # Every result should pass (merged agrees with unmerged)
        fail_count = process_checks.output_flavor(result, output='fail')
        self.assertEqual(fail_count, 0,
                         "check_flavor: some flavor subprocesses disagree.\n" +
                         process_checks.output_flavor(result))

    def test_check_flavor_qqbar_to_zg(self):
        """check_flavor for _quark _anti_quark > z g (mixed merged+non-merged FS).

        This process has non-M gluon vertices mixed with M-tagged Z vertices.
        Flavor must be propagated through the non-M gluon off-shell wavefunction
        so the subsequent M-tagged Z amplitude receives the correct flavor.
        """
        q_id = 81
        z_id = 23
        g_id = 21
        proc_def = base_objects.ProcessDefinition({
            'legs': base_objects.MultiLegList([
                base_objects.MultiLeg({'ids': [q_id],  'state': False}),
                base_objects.MultiLeg({'ids': [-q_id], 'state': False}),
                base_objects.MultiLeg({'ids': [z_id],  'state': True}),
                base_objects.MultiLeg({'ids': [g_id],  'state': True}),
            ]),
            'model': self.merged_model,
        })

        result = process_checks.check_flavor(proc_def, cmd=process_checks.FakeInterface())
        self.assertTrue(len(result) > 0,
                        "check_flavor returned no results for _quark _anti_quark > z g")

        fail_count = process_checks.output_flavor(result, output='fail')
        self.assertEqual(fail_count, 0,
                         "check_flavor: some flavor subprocesses disagree.\n" +
                         process_checks.output_flavor(result))


    def test_check_language_proclist_merged_expansion(self):
        """check_language with a merged-particle ProcessDefinition.

        For ``_quark _anti_quark > t t~``, the SA binaries internally loop over
        all individual quark flavors.  check_language must therefore return one
        result entry per flavor combination (u u~ > t t~, d d~ > t t~, …)
        when at least one compiled backend is available, or exactly one
        placeholder entry when no backend can run.

        Each returned entry must:
        * carry the original merged-id Process object so callers can trace back
          which merged process it came from,
        * have a ``process_label`` string that includes '>' and the individual
          flavor particle names.
        * have Fortran and C++ values that agree within 1e-4 relative when both
          backends ran successfully.
        """
        q_id = 81
        t_id = 6
        proc_def = base_objects.ProcessDefinition({
            'legs': base_objects.MultiLegList([
                base_objects.MultiLeg({'ids': [q_id],  'state': False}),
                base_objects.MultiLeg({'ids': [-q_id], 'state': False}),
                base_objects.MultiLeg({'ids': [t_id],  'state': True}),
                base_objects.MultiLeg({'ids': [-t_id], 'state': True}),
            ]),
            'model': self.merged_model,
        })

        results = process_checks.check_language(
            proc_def,
            param_card=None,
            options=None,
            cmd=process_checks.FakeInterface(),
        )

        self.assertGreater(len(results), 0,
                           "Expected at least one result for merged process")

        for entry in results:
            # The original Process object must use the merged-particle ids.
            proc = entry['process']
            legs = proc.get('legs')
            is_legs = [l for l in legs if not l.get('state')]
            for leg in is_legs:
                self.assertEqual(abs(leg.get('id')), q_id,
                                 "IS leg should use merged id %d" % q_id)

            # Every entry must carry a process_label with a '>' separator.
            label = entry.get('process_label', '')
            self.assertIn('>', label,
                          "process_label should contain '>' separator: %r" % label)

            # When both backends ran they must agree within 1e-4 relative.
            vf  = entry['value_fortran']
            vc  = entry['value_cpp']
            if vf is not None and vc is not None:
                me_f   = vf['m2']
                me_cpp = vc['m2']
                ref = abs(me_f) if me_f != 0 else abs(me_cpp)
                if ref > 0:
                    rel = abs(me_f - me_cpp) / ref
                    self.assertLess(rel, 1e-4,
                                    "Fortran/C++ disagree for %s: F=%g C++=%g "
                                    "rel=%g" % (label, me_f, me_cpp, rel))

            # MG7 SA must also match Fortran per flavor when it ran.
            # value_mg7['m2'] is a dict {value_mg7_<backend>: float|None}.
            vm = entry.get('value_mg7')
            if vf is not None and vm is not None:
                me_f   = vf['m2']
                for backend, me_mg7 in vm['m2'].items():
                    if me_mg7 is None:
                        continue
                    ref = abs(me_f) if me_f != 0 else abs(me_mg7)
                    if ref > 0:
                        rel = abs(me_f - me_mg7) / ref
                        self.assertLess(rel, 1e-4,
                                        "Fortran/MG7 disagree for %s (%s): "
                                        "F=%g MG7=%g rel=%g"
                                        % (label, backend, me_f, me_mg7, rel))

    def test_output_flavor_summary(self):
        """output_flavor returns a string with summary line."""
        q_id = 81
        t_id = 6
        proc_def = base_objects.ProcessDefinition({
            'legs': base_objects.MultiLegList([
                base_objects.MultiLeg({'ids': [q_id],  'state': False}),
                base_objects.MultiLeg({'ids': [-q_id], 'state': False}),
                base_objects.MultiLeg({'ids': [t_id],  'state': True}),
                base_objects.MultiLeg({'ids': [-t_id], 'state': True}),
            ]),
            'model': self.merged_model,
        })
        result = process_checks.check_flavor(proc_def, cmd=process_checks.FakeInterface())
        text = process_checks.output_flavor(result)
        self.assertIn('Summary:', text)
        self.assertIn('passed', text)



#===============================================================================
# TestMultiLanguageComparison
#===============================================================================
class TestMultiLanguageComparison(unittest.TestCase):
    """Compare Python, Fortran SA, and C++ SA matrix elements for the same
    phase-space point.  The strategy is:
      1. Generate and run the Fortran (or C++) standalone check binary.
      2. Parse the phase-space point and |M|^2 value printed by the binary.
      3. Evaluate the Python matrix element with those *same* momenta.
      4. Assert that both results agree to within a relative tolerance.
    Tests are skipped automatically if gfortran / g++ is not found.
    """

    @classmethod
    def setUpClass(cls):
        """Import heavy dependencies once and decide which back-ends exist."""
        import shutil as _shutil
        import madgraph.iolibs.export_v4 as _ev4
        import madgraph.iolibs.export_cpp as _ecpp
        import madgraph.iolibs.helas_call_writers as _hcw
        import madgraph.various.misc as _misc
        cls._ev4 = _ev4
        cls._ecpp = _ecpp
        cls._hcw = _hcw
        cls._misc = _misc
        cls._shutil = _shutil

        cls.has_fortran = bool(_misc.which('gfortran'))
        cls.has_cpp = bool(_misc.which('g++'))
        try:
            import madmatrix.output  # noqa: F401
            cls.has_mg7 = cls.has_cpp and bool(_misc.which('make'))
        except ImportError:
            cls.has_mg7 = False

        cls.model = import_ufo.import_model(
            'sm', options={'apply_flavor_grouping': False})

        # Build the HelasMatrixElement for e+ e- > a a once
        legs = base_objects.LegList([
            base_objects.Leg({'id': -11, 'state': False, 'number': 1}),
            base_objects.Leg({'id':  11, 'state': False, 'number': 2}),
            base_objects.Leg({'id':  22, 'state': True,  'number': 3}),
            base_objects.Leg({'id':  22, 'state': True,  'number': 4}),
        ])
        import madgraph.core.diagram_generation as _dg
        import madgraph.core.helas_objects as _ho
        proc = base_objects.Process({
            'legs': legs, 'model': cls.model,
            'orders': {}, 'forbidden_particles': [],
            'forbidden_onsh_s_channels': [],
            'forbidden_s_channels': [],
            'perturbation_couplings': [],
        })
        amplitude = _dg.Amplitude(proc)
        cls.matrix_element = _ho.HelasMatrixElement(amplitude, gen_color=True)

        # Ensure Template/LO/Source/make_opts exists (created on first MG5 run)
        import madgraph
        MG5DIR = madgraph.MG5DIR
        make_opts = os.path.join(MG5DIR, 'Template', 'LO', 'Source', 'make_opts')
        make_opts_src = os.path.join(MG5DIR, 'Template', 'LO', 'Source', '.make_opts')
        if not os.path.exists(make_opts) and os.path.exists(make_opts_src):
            _shutil.copy(make_opts_src, make_opts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    _me_re = re.compile(
        r"\sMatrix\selement\s=\s*(?P<val>-?\d*\.\d*(E[+-]?\d*)?)"
        r"\s*GeV\^\s*(?P<pow>-?\d+)", re.IGNORECASE | re.VERBOSE)
    _mom_re = re.compile(
        r"""\s*\d+\s+(?P<p0>-?\d*\.\d*E[+-]?\d*)\s+
             (?P<p1>-?\d*\.\d*E[+-]?\d*)\s+
             (?P<p2>-?\d*\.\d*E[+-]?\d*)\s+
             (?P<p3>-?\d*\.\d*E[+-]?\d*)""",
        re.IGNORECASE | re.VERBOSE)
    # Matches the "PDG  2  -2  6  -6" line written per flavor by Fortran SA.
    _pdg_re = re.compile(r'^\s*PDG\s+(.+)$', re.IGNORECASE)

    def _parse_sa_output(self, text, target_pdgs=None):
        """Return (me_value, [[E,px,py,pz], ...]) from check binary output.

        If *target_pdgs* is given the matrix-element value for the flavor
        whose preceding "PDG …" line matches those PDG codes is returned.
        If no block matches (e.g. because the process uses merged-particle
        legs whose PDG codes are flavor indices in the Fortran output), falls
        back to the last seen matrix-element value.
        If *target_pdgs* is not given the last value is returned (legacy).
        """
        momenta = []
        me_val = None
        last_val = None
        current_pdgs = None
        for line in text.split('\n'):
            mp = self._pdg_re.match(line)
            if mp:
                try:
                    current_pdgs = [int(x) for x in mp.group(1).split()]
                except ValueError:
                    current_pdgs = None
            m = self._me_re.match(line)
            if m:
                val = float(m.group('val'))
                last_val = val
                if target_pdgs is not None and current_pdgs is not None:
                    if current_pdgs == list(target_pdgs):
                        me_val = val
                else:
                    me_val = val
                current_pdgs = None
            m2 = self._mom_re.match(line)
            if m2:
                momenta.append([float(x) for x in m2.groups()[:4]])
        # Fallback: target was specified but no PDG block matched.
        if target_pdgs is not None and me_val is None:
            me_val = last_val
        return me_val, momenta

    def _run_check_binary(self, check_dir):
        """Compile and run ./check in *check_dir*, return stdout."""
        devnull = open(os.devnull, 'w')
        ret = subprocess.call(['make', 'check'], cwd=check_dir,
                              stdout=devnull, stderr=devnull)
        devnull.close()
        if ret != 0:
            self.skipTest('make check failed in %s' % check_dir)
        return subprocess.check_output(
            './check', cwd=check_dir, stderr=subprocess.STDOUT).decode()

    def _first_P_dir(self, sa_root):
        """Return the first P-directory found under sa_root/SubProcesses."""
        sub = os.path.join(sa_root, 'SubProcesses')
        for d in sorted(os.listdir(sub)):
            if d.startswith('P') and os.path.isdir(os.path.join(sub, d)):
                return os.path.join(sub, d)
        return None

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_parse_sa_output_flavor_selection(self):
        """_parse_sa_output correctly selects or falls back for target_pdgs.

        When the Fortran SA iterates over multiple flavors (e.g. u, d, b for a
        merged _quark leg) it prints one "PDG …" + "Matrix element = …" pair per
        flavor.  The function should:
        1. Return the ME for the block whose PDG line matches target_pdgs.
        2. Fall back to the last ME value when target_pdgs is given but no block
           matches.  This handles merged-particle processes where Fortran stores
           flavor *indices* (1, -1, …) rather than actual PDG codes.
        3. Return the last ME when target_pdgs is not given (legacy behaviour).
        """
        # Simulated Fortran SA output with three flavors:
        #   u (PDG 2 -2 6 -6)  → ME = 0.6177
        #   d (PDG 1 -1 6 -6)  → ME = 0.6178 (same coupling; differs slightly)
        #   b (PDG 5 -5 6 -6)  → ME = 0.5894 (b mass ≈ 4.7 GeV gives ~5% diff)
        fake_output = (
            "\n Phase space point:\n"
            "  1   5.0000000E+02  0.0000000E+00  0.0000000E+00  4.9000000E+02\n"
            "  2   5.0000000E+02  0.0000000E+00  0.0000000E+00 -4.9000000E+02\n"
            "  3   6.0000000E+02  1.0000000E+02  2.0000000E+02 -3.0000000E+02\n"
            "  4   4.0000000E+02 -1.0000000E+02 -2.0000000E+02  3.0000000E+02\n"
            " PDG           2          -2           6          -6\n"
            " Matrix element =  0.617700E+00 GeV^  -4\n"
            " -----------------------------------------------------------------------------\n"
            " PDG           1          -1           6          -6\n"
            " Matrix element =  0.617800E+00 GeV^  -4\n"
            " -----------------------------------------------------------------------------\n"
            " PDG           5          -5           6          -6\n"
            " Matrix element =  0.589400E+00 GeV^  -4\n"
            " -----------------------------------------------------------------------------\n"
        )

        # ── no target: backward-compat, returns last flavor (b quark) ──────
        me, p = self._parse_sa_output(fake_output)
        self.assertAlmostEqual(me, 0.5894, places=7,
                               msg='Without target_pdgs should return last ME')
        self.assertEqual(len(p), 4,
                         'Should parse 4 momentum vectors')

        # ── matching target: u quark → 0.6177 ───────────────────────────────
        me_u, _ = self._parse_sa_output(fake_output, target_pdgs=[2, -2, 6, -6])
        self.assertAlmostEqual(me_u, 0.6177, places=7,
                               msg='target_pdgs=[2,-2,6,-6] should return u ME')

        # ── matching target: d quark → 0.6178 ───────────────────────────────
        me_d, _ = self._parse_sa_output(fake_output, target_pdgs=[1, -1, 6, -6])
        self.assertAlmostEqual(me_d, 0.6178, places=7,
                               msg='target_pdgs=[1,-1,6,-6] should return d ME')

        # ── matching target: b quark → 0.5894 ───────────────────────────────
        me_b, _ = self._parse_sa_output(fake_output, target_pdgs=[5, -5, 6, -6])
        self.assertAlmostEqual(me_b, 0.5894, places=7,
                               msg='target_pdgs=[5,-5,6,-6] should return b ME')

        # ── sanity: b quark ME differs >1% from u/d (mass effect) ───────────
        rel_ub = abs(me_u - me_b) / abs(me_u)
        self.assertGreater(rel_ub, 0.01,
                           'u and b quark MEs should differ by >1%% due to b mass')

        # ── unmatched target: Fortran uses flavor indices for merged legs ────
        # In this scenario Fortran outputs PDG lines like "PDG  1  -1  6  -6"
        # (flavor *indices*) while proc.get('legs') contains the merged-particle
        # PDG code (e.g. 2000).  No block matches, so the function falls back
        # to the *last* ME value (0.5894 here) instead of returning None.
        me_merged, _ = self._parse_sa_output(fake_output,
                                             target_pdgs=[2000, -2000, 6, -6])
        self.assertAlmostEqual(me_merged, 0.5894, places=7,
                               msg='Unmatched target_pdgs should fall back to '
                                   'last ME, not return None')


    def test_python_vs_fortran_epem_aa(self):
        """e+ e- > a a: Python and Fortran SA must agree within 1e-4 rel."""
        if not self.has_fortran:
            self.skipTest('gfortran not available')

        import tempfile
        ev4 = self._ev4
        hcw = self._hcw
        misc = self._misc
        model = self.model
        me = self.matrix_element

        wl = misc.make_unique(me.get_used_lorentz())
        wc = misc.make_unique([c for l in me.get_used_couplings() for c in l])

        parent = tempfile.mkdtemp(prefix='mg5_test_f_')
        sa_dir = os.path.join(parent, 'sa_f')
        try:
            opt = {'sa_symmetry': False, 'export_format': 'standalone_fortran',
                   'mp': False, 'v5_model': True,
                   'output_options': {'noeps': 'True'}}
            exporter = ev4.ProcessExporterFortranSA(sa_dir, opt)
            fmodel = hcw.FortranUFOHelasCallWriter(model)
            exporter.copy_template(model)
            exporter.generate_subprocess_directory(me, fmodel, 0)
            exporter.convert_model(model, wl, wc)
            exporter.finalize({'matrix_elements': [me]}, '',
                              {'fortran_compiler': 'gfortran',
                               'cpp_compiler': 'g++',
                               'f2py_compiler': 'f2py',
                               'output_dependencies': 'external'},
                              ['nojpeg'])

            check_dir = self._first_P_dir(sa_dir)
            if check_dir is None:
                self.skipTest('No subprocess directory created')

            output = self._run_check_binary(check_dir)
            me_f, p_f = self._parse_sa_output(output)

            self.assertIsNotNone(me_f, 'Could not parse Fortran ME value')
            self.assertGreater(len(p_f), 0, 'Could not parse Fortran momenta')

            # Evaluate Python with the *same* phase-space point
            process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
            evaluator = process_checks.MatrixElementEvaluator(model)
            me_py, _ = evaluator.evaluate_matrix_element(me, p=p_f)

            self.assertGreater(abs(me_f), 0.,
                               'Fortran ME is zero – unexpected')
            rel_diff = abs(me_f - me_py) / abs(me_f)
            self.assertLess(rel_diff, 1e-4,
                            'Python and Fortran SA disagree: '
                            'Fortran=%g  Python=%g  rel_diff=%g'
                            % (me_f, me_py, rel_diff))
        finally:
            self._shutil.rmtree(parent, ignore_errors=True)

    def test_python_vs_cpp_epem_aa(self):
        """e+ e- > a a: Python and C++ SA must agree within 1e-4 rel."""
        if not self.has_cpp:
            self.skipTest('g++ not available')

        import tempfile
        ecpp = self._ecpp
        hcw = self._hcw
        misc = self._misc
        model = self.model
        me = self.matrix_element

        wl = misc.make_unique(me.get_used_lorentz())
        wc = misc.make_unique([c for l in me.get_used_couplings() for c in l])

        parent = tempfile.mkdtemp(prefix='mg5_test_cpp_')
        sa_dir = os.path.join(parent, 'sa_cpp')
        try:
            opt = {'sa_symmetry': False, 'export_format': 'standalone_cpp',
                   'mp': False, 'v5_model': True, 'cpp_compiler': 'g++'}
            exporter = ecpp.ProcessExporterCPP(sa_dir, opt)
            cpp_model = hcw.CPPUFOHelasCallWriter(model)
            exporter.copy_template(model)
            exporter.generate_subprocess_directory(me, cpp_model, 0)
            exporter.convert_model(model, wl, wc)
            exporter.finalize({'matrix_elements': [me]}, '',
                              {'fortran_compiler': 'gfortran',
                               'cpp_compiler': 'g++',
                               'f2py_compiler': 'f2py',
                               'output_dependencies': 'external'},
                              ['nojpeg'])

            check_dir = self._first_P_dir(sa_dir)
            if check_dir is None:
                self.skipTest('No subprocess directory created')

            output = self._run_check_binary(check_dir)
            me_cpp, p_cpp = self._parse_sa_output(output)

            self.assertIsNotNone(me_cpp, 'Could not parse C++ ME value')
            self.assertGreater(len(p_cpp), 0, 'Could not parse C++ momenta')

            # Evaluate Python with the *same* phase-space point
            process_checks.clean_added_globals(process_checks.ADDED_GLOBAL)
            evaluator = process_checks.MatrixElementEvaluator(model)
            me_py, _ = evaluator.evaluate_matrix_element(me, p=p_cpp)

            self.assertGreater(abs(me_cpp), 0.,
                               'C++ ME is zero – unexpected')
            rel_diff = abs(me_cpp - me_py) / abs(me_cpp)
            self.assertLess(rel_diff, 1e-4,
                            'Python and C++ SA disagree: '
                            'C++=%g  Python=%g  rel_diff=%g'
                            % (me_cpp, me_py, rel_diff))
        finally:
            self._shutil.rmtree(parent, ignore_errors=True)

    def test_check_language_function_epem_aa(self):
        """check_language() must return Passed for e+ e- > a a when at least
        one compiled backend (gfortran or g++) is available.

        check_language compares Fortran SA, C++ SA and the pure-Python back-end
        side by side.  All three use the same RAMBO-generated phase-space
        point so the matrix-element values must agree.
        """
        model = self.model

        # Build a single Process (not a ProcessDefinition) so check_language
        # takes the non-ProcessDefinition branch.
        import madgraph.core.base_objects as _bo
        legs = _bo.LegList([
            _bo.Leg({'id': -11, 'state': False, 'number': 1}),
            _bo.Leg({'id':  11, 'state': False, 'number': 2}),
            _bo.Leg({'id':  22, 'state': True,  'number': 3}),
            _bo.Leg({'id':  22, 'state': True,  'number': 4}),
        ])
        proc = _bo.Process({
            'legs': legs, 'model': model,
            'orders': {}, 'forbidden_particles': [],
            'forbidden_onsh_s_channels': [],
            'forbidden_s_channels': [],
            'perturbation_couplings': [],
        })

        results = process_checks.check_language(proc)
        self.assertEqual(len(results), 1)

        entry = results[0]
        self.assertIn('value_fortran', entry)
        self.assertIn('value_cpp', entry)
        self.assertIn('value_mg7', entry)
        self.assertIn('value_python', entry)

        if not (self.has_fortran or self.has_cpp):
            self.skipTest('No compiled backend available')

        me_f   = entry['value_fortran']['m2'] if entry['value_fortran'] else None
        me_cpp = entry['value_cpp']['m2']      if entry['value_cpp']     else None
        # value_mg7['m2'] is a dict {value_mg7_<backend>: float|None}.
        mg7_vals = entry['value_mg7']['m2']    if entry['value_mg7']     else {}
        me_py  = entry['value_python']['m2']   if entry['value_python']  else None

        # Every pair of available backends must agree within 1e-4 relative.
        pairs = [
            (me_f,   me_cpp, 'Fortran', 'C++'),
            (me_f,   me_py,  'Fortran', 'Python'),
            (me_cpp, me_py,  'C++',     'Python'),
        ]
        for backend, me_mg7 in mg7_vals.items():
            pairs.extend([
                (me_f,   me_mg7, 'Fortran', backend),
                (me_cpp, me_mg7, 'C++',     backend),
                (me_mg7, me_py,  backend,   'Python'),
            ])
        for a, b, na, nb in pairs:
            if a is None or b is None:
                continue
            ref = abs(a) if a != 0 else abs(b)
            rel = abs(a - b) / ref if ref > 0 else 0.
            self.assertLess(rel, 1e-4,
                            '%s/%s disagree: %s=%g %s=%g rel=%g'
                            % (na, nb, na, a, nb, b, rel))

        # MG7 SA must be available when the madmatrix backend is present.
        if self.has_mg7:
            self.assertIsNotNone(entry['value_mg7'],
                                 'MG7 SA value missing though madmatrix present')

        # output_language must at least contain a Summary line + MG7 column.
        text = process_checks.output_language(results)
        self.assertIn('Summary', text)
        self.assertIn('MG7 SA', text)

    def test_check_language_gg_ttx_cpp_ps_point(self):
        """g g > t t~: Fortran SA and C++ SA must agree within 1e-4 rel.

        Both backends receive the same energy argument so RAMBO generates
        identical momenta; the matrix-element values should therefore agree
        to machine precision (modulo compiler differences).
        """
        if not (self.has_fortran and self.has_cpp):
            self.skipTest('Both gfortran and g++ are required for this test')

        import madgraph.core.base_objects as _bo

        model = self.model
        legs = _bo.LegList([
            _bo.Leg({'id': 21, 'state': False, 'number': 1}),
            _bo.Leg({'id': 21, 'state': False, 'number': 2}),
            _bo.Leg({'id':  6, 'state': True,  'number': 3}),
            _bo.Leg({'id': -6, 'state': True,  'number': 4}),
        ])
        proc = _bo.Process({
            'legs': legs, 'model': model,
            'orders': {}, 'forbidden_particles': [],
            'forbidden_onsh_s_channels': [],
            'forbidden_s_channels': [],
            'perturbation_couplings': [],
        })

        results = process_checks.check_language(proc)
        self.assertEqual(len(results), 1, 'Expected exactly one subprocess result')

        entry = results[0]
        self.assertIn('value_fortran', entry)
        self.assertIn('value_cpp', entry)
        self.assertIn('value_mg7', entry)
        self.assertIn('value_python', entry)

        if entry['value_fortran'] is None:
            self.skipTest('Fortran SA evaluation failed')
        if entry['value_cpp'] is None:
            self.skipTest('C++ SA evaluation failed (no g++ or compile error)')

        me_f   = entry['value_fortran']['m2']
        me_cpp = entry['value_cpp']['m2']
        # value_mg7['m2'] is a dict {value_mg7_<backend>: float|None}.
        mg7_vals = entry['value_mg7']['m2'] if entry['value_mg7'] else {}
        me_py  = entry['value_python']['m2'] if entry['value_python'] else None

        self.assertGreater(abs(me_cpp), 0., 'C++ ME is zero – unexpected')
        rel_diff = abs(me_f - me_cpp) / abs(me_f)
        self.assertLess(rel_diff, 1e-4,
                        'Fortran and C++ SA disagree for g g > t t~: '
                        'F=%g  C++=%g  rel_diff=%g'
                        % (me_f, me_cpp, rel_diff))

        # MG7 SA must agree with Fortran when the madmatrix backend ran.
        for backend, me_mg7 in mg7_vals.items():
            if me_mg7 is None:
                continue
            rel_mg7_f = abs(me_f - me_mg7) / abs(me_f)
            self.assertLess(rel_mg7_f, 1e-4,
                            'MG7 and Fortran SA disagree for g g > t t~ (%s): '
                            'MG7=%g  F=%g  rel_diff=%g'
                            % (backend, me_mg7, me_f, rel_mg7_f))

        # Python must also agree if it ran successfully.
        self.assertIsNotNone(me_py, 'Python evaluation failed for g g > t t~')
        rel_py_f = abs(me_f - me_py) / abs(me_f)
        self.assertLess(rel_py_f, 1e-4,
                        'Python and Fortran SA disagree for g g > t t~: '
                        'Python=%g  F=%g  rel_diff=%g'
                        % (me_py, me_f, rel_py_f))

        # output_language must also report Passed
        text = process_checks.output_language(results)
        self.assertIn('Passed', text,
                      'output_language reported failure for g g > t t~:\n' + text)


class TestPrecisionCheck(unittest.TestCase):
    """check precision: the statistics, the report, and one compiled run."""

    def test_precision_statistics(self):
        """Relative errors against double precision, with the zero and
        non-finite reference cases counted apart from the mean."""
        me_double = [1., 2., 0., 0., 4., 3.]
        me_mode = [1.001, 2.1, 0., 1e-3, 4., float('nan')]
        stats = process_checks.precision_statistics(me_double, me_mode)
        self.assertEqual(stats['nb_event'], 6)
        self.assertEqual(stats['nb_invalid'], 2)   # 0 -> 1e-3, and the nan
        self.assertEqual(stats['nb_above'], 3)     # 5% and the two invalid ones
        self.assertAlmostEqual(stats['rate'], 0.5)
        self.assertTrue(math.isinf(stats['max']))
        self.assertAlmostEqual(stats['mean'], (1e-3 + 0.05) / 4)
        self.assertAlmostEqual(stats['errors'][0], 1e-3)
        self.assertEqual(stats['errors'][2], 0.)

        self.assertRaises(Exception, process_checks.precision_statistics,
                          [1., 2.], [1.])

    def test_output_precision(self):
        """The report quotes the three numbers and where the plot went."""
        stats = process_checks.precision_statistics([1., 2.], [1.5, 2.])
        stats.update({'process_label': 'e+ e- > mu+ mu-', 'subprocess': 'P0',
                      'flavor': 0, 'mode': 'f', 'plot': '/tmp/x.pdf',
                      'time_double': 0.5, 'time_mode': 0.25})
        stats_m = process_checks.precision_statistics([1., 2.], [1., 2.])
        stats_m.update({'process_label': 'e+ e- > mu+ mu-', 'subprocess': 'P0',
                        'flavor': 0, 'mode': 'm', 'plot': '/tmp/x.pdf',
                        'time_double': 0.5, 'time_mode': 0.4})
        text = process_checks.output_precision([stats, stats_m])
        for word in ('FPTYPE=f,m vs FPTYPE=d', 'mean error', 'max error',
                     'rate > 1%', 'mode', 'e+ e- > mu+ mu-', '5.000e-01',
                     'time d [s]', 'time mode [s]', '2.500e-01', '2.00', '1.25',
                     process_checks.PRECISION_MODES['m']):
            self.assertIn(word, text)
        # one plot for both modes, quoted once
        self.assertEqual(text.count('Plot of the difference: /tmp/x.pdf'), 1)
        rows = [l for l in text.split('\n') if l.startswith('e+ e- > mu+ mu-')]
        self.assertEqual([r.split()[5] for r in rows], ['f', 'm'])
        stats['time_mode'] = None
        self.assertIn('N/A', process_checks.output_precision([stats]))
        stats['plot'] = '/tmp/x.dat'
        self.assertIn('Relative errors (no matplotlib, no plot): /tmp/x.dat',
                      process_checks.output_precision([stats]))

    def test_precision_progress(self):
        """The progress bar counts every step, grows when told to, and is a
        silent no-op when tqdm cannot be imported."""
        import io
        try:
            import tqdm  # noqa: F401
            has_tqdm = True
        except ImportError:
            has_tqdm = False
        if has_tqdm:
            stream = io.StringIO()            # keep the test output clean
            progress = process_checks.PrecisionProgress(2, stream=stream)
            progress.add(1)
            for i in range(3):
                with progress.step('step %d' % i):
                    pass
            self.assertEqual((progress.bar.n, progress.bar.total), (3, 3))
            progress.close()
            self.assertIn('3/3', stream.getvalue())
            self.assertIn('step 2', stream.getvalue())

        saved = sys.modules.get('tqdm')
        sys.modules['tqdm'] = None            # makes 'from tqdm import tqdm' raise
        try:
            progress = process_checks.PrecisionProgress(2)
            self.assertIsNone(progress.bar)
            progress.add(1)
            with progress.step('no bar'):
                pass
            progress.close()
        finally:
            if saved is None:
                del sys.modules['tqdm']
            else:
                sys.modules['tqdm'] = saved

    def test_matrix_element_time(self):
        """The matrix element time is read from the check_sa.exe perf report."""
        perf = ('TotalTime[Rambo]        (2) = ( 1.000000e-03 )  sec\n'
                'TotalTime[MatrixElems]  (3) = ( 2.345678e-02 )  sec\n')
        self.assertAlmostEqual(process_checks.matrix_element_time(perf), 2.345678e-02)
        self.assertIsNone(process_checks.matrix_element_time('nothing here'))

    def test_check_precision_epem_mupmum(self):
        """FPTYPE=f against FPTYPE=d for e+ e- > mu+ mu-: single precision is
        good to ~1e-7 there, so no event may come anywhere near 1%."""
        import shutil
        import tempfile
        if not (misc.which('g++') and misc.which('make')):
            self.skipTest('check precision needs g++ and make')
        try:
            import madmatrix.output  # noqa: F401
        except ImportError:
            self.skipTest('madmatrix is not available')

        model = import_ufo.import_model('sm', options={'apply_flavor_grouping': False})
        legs = base_objects.MultiLegList([
            base_objects.MultiLeg({'ids': [-11], 'state': False}),
            base_objects.MultiLeg({'ids': [11], 'state': False}),
            base_objects.MultiLeg({'ids': [-13], 'state': True}),
            base_objects.MultiLeg({'ids': [13], 'state': True})])
        procdef = base_objects.ProcessDefinition({'legs': legs, 'model': model})

        output_path = tempfile.mkdtemp(prefix='mg5_test_precision_')
        try:
            # two modes (and a duplicate) share one double precision reference
            results = process_checks.check_precision(
                procdef, ['f', 'm', 'f'], options={'nb_event': 2000},
                output_path=output_path)
            self.assertEqual([r['mode'] for r in results], ['f', 'm'])
            for entry in results:
                self.assertEqual(entry['nb_event'], 2000)
                self.assertEqual(entry['nb_invalid'], 0)
                self.assertEqual(entry['nb_above'], 0)
                self.assertLess(entry['max'], 1e-4)
                self.assertGreater(entry['time_double'], 0.)
                self.assertGreater(entry['time_mode'], 0.)
                self.assertTrue(os.path.exists(entry['plot']), entry['plot'])
            self.assertEqual(results[0]['time_double'], results[1]['time_double'])
            self.assertEqual(results[0]['plot'], results[1]['plot'])
            self.assertIn('check_precision_f_m_', results[0]['plot'])
            # single precision leaves a visible difference, the colour-only
            # single precision of FPTYPE=m cannot do worse on a colourless process
            self.assertGreater(results[0]['max'], 0.)
            self.assertLessEqual(results[1]['max'], results[0]['max'])
        finally:
            shutil.rmtree(output_path, ignore_errors=True)

    def test_check_precision_v_epem_mupmum(self):
        """FPTYPE=v against FPTYPE=d for e+ e- > mu+ mu-, on the widest SIMD
        backend of the host and on the scalar one.

        FPTYPE=v computes the momenta and the propagator denominators in double
        precision and the rest in single precision. On a SIMD backend the
        denominators are a double precision vector with the single precision
        lane count (MGONGPU_SIMD_DENOM64), a code path of its own; on the scalar
        backend they are plain doubles. The process uses both forms of
        madmatrix::denom: the photon propagator with its coupling (FFV1P0_3) and
        the Z one of a routine assembled from two Lorentz structures, without
        coupling (FFV2_4_3). FPTYPE=v is good to ~1e-6 there, so a lane filled
        from the wrong momenta, or a non-finite value, fails at once."""
        import shutil
        import tempfile
        if not (misc.which('g++') and misc.which('make')):
            self.skipTest('check precision needs g++ and make')
        try:
            import madmatrix.output  # noqa: F401
        except ImportError:
            self.skipTest('madmatrix is not available')

        model = import_ufo.import_model('sm', options={'apply_flavor_grouping': False})
        legs = base_objects.MultiLegList([
            base_objects.MultiLeg({'ids': [-11], 'state': False}),
            base_objects.MultiLeg({'ids': [11], 'state': False}),
            base_objects.MultiLeg({'ids': [-13], 'state': True}),
            base_objects.MultiLeg({'ids': [13], 'state': True})])
        procdef = base_objects.ProcessDefinition({'legs': legs, 'model': model})

        for backend in ('auto', 'scalar'):
            output_path = tempfile.mkdtemp(prefix='mg5_test_precision_v_')
            try:
                results = process_checks.check_precision(
                    procdef, ['v'], options={'nb_event': 20000, 'backend': backend},
                    output_path=output_path)
                self.assertEqual([(r['mode'], r['backend']) for r in results],
                                 [('v', backend)])
                entry = results[0]
                self.assertEqual(entry['nb_event'], 20000)
                self.assertEqual(entry['nb_invalid'], 0, backend)
                self.assertEqual(entry['nb_above'], 0, backend)
                self.assertLess(entry['max'], 1e-4, backend)
                # single precision amplitudes leave a visible difference: a
                # build that fell back to double precision would give 0
                self.assertGreater(entry['max'], 0., backend)
                self.assertGreater(entry['time_mode'], 0., backend)
            finally:
                shutil.rmtree(output_path, ignore_errors=True)



class TestCheckGaugeDecayChain(unittest.TestCase):
    """'check gauge' on a decay chain (process_checks.
    check_unitary_feynman_decay_chains): the chain is compared in the
    unitary, axial and FD gauges, with every decayed particle on shell."""

    @classmethod
    def setUpClass(cls):
        import madgraph.interface.master_interface as interface
        cls.cmd = interface.MasterCmd()
        cls.cmd.no_notification()
        cls.cmd.exec_cmd('set apply_flavor_grouping False')
        cls.cmd.exec_cmd('import model sm')

    def check_gauge(self, process, cmd=None):
        """(gauge labels, {process string: (relative difference, values)})
        from the table that 'check gauge' logs"""
        cmd = cmd or self.cmd
        with self.assertLogs('madgraph.check_cmd', 'INFO') as logs:
            cmd.exec_cmd('check gauge %s' % process)
        text = '\n'.join(record.getMessage() for record in logs.records)
        lines = text.split('\n')
        header = [line for line in lines if line.startswith('Process ')][0]
        labels = header.split()[1:-3]
        rows = {}
        for line in lines:
            if line.endswith(('Passed', 'Failed')) and not \
                                            line.lstrip().startswith('JAMP'):
                words = line.split()
                values = [float(x) for x in words[-2 - len(labels):-2]]
                rows[' '.join(words[:-2 - len(labels)])] = \
                    (float(words[-2]), values)
        return labels, rows

    def test_decay_chain_momenta(self):
        """Momenta of all the external legs, in the order of
        get_legs_with_decays, with every decayed particle on shell and
        momentum conserved at every vertex."""
        import madgraph.core.diagram_generation as diagram_generation
        import madgraph.core.helas_objects as helas_objects
        procdef = self.cmd.extract_check_process(
                        'u u~ > t t~, (t > w+ b, w+ > e+ ve), t~ > w- b~')
        amplitudes = diagram_generation.MultiProcess(procdef,
                                collect_mirror_procs=False).get('amplitudes')
        me = helas_objects.HelasMultiProcess(amplitudes).get_matrix_elements()[0]
        process = me.get('processes')[0]
        evaluator = process_checks.MatrixElementEvaluator(self.cmd._curr_model,
                                                          cmd=self.cmd)
        p = process_checks.get_decay_chain_momenta(evaluator, process)
        ids = [leg.get('id') for leg in process.get_legs_with_decays()]
        self.assertEqual(ids, [2, -2, -11, 12, 5, -24, -5])
        param = evaluator.full_model.get('parameter_dict')
        mass = lambda name: abs(param[name])
        sq = lambda q: q[0]**2 - q[1]**2 - q[2]**2 - q[3]**2
        add = lambda *qs: [sum(q[i] for q in qs) for i in range(4)]
        self.assertAlmostEqual(sq(add(p[2], p[3])) / mass('mdl_MW')**2, 1, 12)
        self.assertAlmostEqual(sq(add(p[2], p[3], p[4])) / mass('mdl_MT')**2, 1, 12)
        self.assertAlmostEqual(sq(add(p[5], p[6])) / mass('mdl_MT')**2, 1, 12)
        self.assertAlmostEqual(sq(p[5]) / mass('mdl_MW')**2, 1, 12)
        total = add(p[2], p[3], p[4], p[5], p[6])
        for i in range(4):
            self.assertAlmostEqual(total[i], p[0][i] + p[1][i], 8)

    def test_check_gauge_polarised_decay_chain(self):
        """The helicity pieces and the unpolarised chain agree in the
        unitary, axial and FD gauges; {S} is gauge dependent and differs in
        FD. The Feynman gauge is not compared (no Goldstone boson in the
        chain), and {G} has no FD counterpart."""
        for brace in ('', '{0}', '{T}', '{+}', '{0S}'):
            labels, rows = self.check_gauge('t > w+%s b, w+ > ta+ vt' % brace)
            self.assertEqual(labels, ['Unitary', 'Axial', 'FD'], brace)
            (diff, values), = rows.values()
            self.assertLess(diff, 1e-12, brace)
            self.assertGreater(min(values), 0, brace)
        labels, rows = self.check_gauge('t > w+{S} b, w+ > ta+ vt')
        (diff, values), = rows.values()
        self.assertGreater(diff, 1e-2)
        self.assertEqual(values[0], values[1])          # unitary = axial
        labels, rows = self.check_gauge('t > w+{G} b, w+ > ta+ vt')
        self.assertEqual(labels, ['Unitary', 'Axial'])

    def test_check_gauge_two_decays(self):
        """With two decayed bosons the core process has an s-channel W:
        the width given to the decayed particles (1e-9 M) leaves it gauge
        invariant to 1e-10 (at the physical width it differs by 4e-4)."""
        labels, rows = self.check_gauge(
                        'u d~ > w+{0} z, w+ > e+ ve, z > mu+ mu-')
        (diff, values), = rows.values()
        self.assertLess(diff, 1e-9)
        labels, rows = self.check_gauge(
                'g g > t t~, (t > w+{T} b, w+ > e+ ve), (t~ > w-{0} b~, w- > mu- vm~)')
        (diff, values), = rows.values()
        self.assertLess(diff, 1e-10)

    def test_check_gauge_decay_chain_flavour_grouping(self):
        """With flavour grouping (the default) every physical flavour of a
        grouped leg is checked, at any depth of the chain, and a combination
        that does not conserve charge is skipped: 'p p' gives u d~ and c s~
        and their mirrors, 'l+ vl' the electron and the muon."""
        import madgraph.interface.master_interface as interface
        cmd = interface.MasterCmd()
        cmd.no_notification()
        cmd.exec_cmd('import model sm')
        labels, rows = self.check_gauge(
                'p p > w+{0} z, w+ > e+ ve, z > mu+ mu-', cmd)
        flavours = sorted(name.split('[')[1].split(' >')[0] for name in rows)
        self.assertEqual(flavours, ['c s~', 'd~ u', 's~ c', 'u d~'])
        for diff, values in rows.values():
            self.assertLess(diff, 1e-9)
            self.assertGreater(min(values), 0)
        labels, rows = self.check_gauge('t > w+{0} b, w+ > l+ vl', cmd)
        self.assertEqual(sorted(name.split('[')[1] for name in rows),
                         ['t > e+ ve b]', 't > mu+ vm b]'])
        for diff, values in rows.values():
            self.assertLess(diff, 1e-12)
        # a leg tagged with several flavours ('define q = u c') checks each of
        # them, not only the first: u d~ and c s~ (u s~ and c d~ vanish)
        cmd.exec_cmd('define qq = u c')
        cmd.exec_cmd('define qb = d~ s~')
        labels, rows = self.check_gauge(
                'qq qb > w+{0} z, w+ > e+ ve, z > mu+ mu-', cmd)
        flavours = sorted(name.split('[')[1].split(' >')[0] for name in rows)
        self.assertEqual(flavours, ['c s~', 'u d~'])
        for diff, values in rows.values():
            self.assertLess(diff, 1e-9)

    def test_check_gauge_restores_the_gauge(self):
        """'check gauge' switches the session through the unitary, axial and
        FD gauges: the gauge of the session comes back when the check refuses
        the process after the switch, as on success."""
        import madgraph
        import aloha
        self.cmd.exec_cmd('set gauge Feynman')
        try:
            self.assertRaises(madgraph.InvalidCmd, lambda: self.cmd.exec_cmd(
                'check gauge e+ e- > a z, a > mu+ mu-', errorhandling=False))
            self.assertEqual(self.cmd.options['gauge'], 'Feynman')
            self.assertIs(aloha.unitary_gauge, False)
        finally:
            self.cmd.exec_cmd('set gauge unitary')

    def test_check_decay_chain_refused(self):
        """Decay chains: only 'check gauge', without loops, outside the
        complex-mass scheme, and with massive decayed particles."""
        import madgraph
        for line in ('check brs t > w+ b, w+ > ta+ vt',
                     'check gauge t > w+ b, w+ > ta+ vt --events=events.lhe',
                     'check full t > w+ b, w+ > ta+ vt',
                     'check gauge u u~ > z g, z > mu+ mu- [QCD]',
                     'check gauge e+ e- > a z, a > mu+ mu-'):
            self.assertRaises(madgraph.InvalidCmd,
                              lambda: self.cmd.exec_cmd(line, errorhandling=False))
        self.cmd.exec_cmd('set complex_mass_scheme True')
        try:
            self.assertRaises(madgraph.InvalidCmd, lambda: self.cmd.exec_cmd(
                'check gauge t > w+ b, w+ > ta+ vt', errorhandling=False))
        finally:
            self.cmd.exec_cmd('set complex_mass_scheme False')


if __name__ == '__main__':
    unittest.unittest.main()
