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
"""The export of GPD processes (madgraph/iolibs/gpd_export.py): the leg
bookkeeping, the x-flow tagging of the propagators (a propagator carries x
when its external legs hold exactly one of the two GPD legs), the patched
HELAS calls (wave functions of the pair built on P, momentum slots rescaled,
amplitudes masked per pole group), the makefile, and the consistency of the
gpd_card with the Fortran readers."""

from __future__ import absolute_import

import os
import re
import sys
import tempfile

root_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
sys.path.append(os.path.join(root_path, os.path.pardir, os.path.pardir))

import tests.unit_tests as unittest

import madgraph
import madgraph.core.base_objects as base_objects
import madgraph.core.helas_objects as helas_objects
import madgraph.interface.master_interface as MGCmd
import madgraph.iolibs.gpd_export as gpd_export
import madgraph.iolibs.gpd_madevent as gpd_madevent
import madgraph.iolibs.helas_call_writers as helas_call_writers

MG7 = madgraph.MG5DIR


class GPDExportTest(unittest.TestCase):

    interface = None

    def setUp(self):
        if GPDExportTest.interface is None:
            interface = MGCmd.MasterCmd()
            interface.exec_cmd('import model sm', printcmd=False, precmd=True)
            GPDExportTest.interface = interface
        self.cmd = GPDExportTest.interface

    def matrix_element(self, line):
        self.cmd.do_generate(line)
        return helas_objects.HelasMatrixElement(self.cmd._curr_amps[0])

    def test_info(self):
        me = self.matrix_element('a gpd_u > a a')
        self.assertTrue(gpd_export.has_gpd(me))
        info = gpd_export.GPDInfo(me)
        self.assertEqual((info.n_in, info.n_out), (2, 5))
        self.assertEqual((info.channel, info.parton, info.is_gluon), ('gpd_u', 2, False))
        self.assertEqual((info.nphys, info.ninitial), (4, 2))
        self.assertEqual(info.denominator(), 8)
        self.assertFalse(gpd_export.has_gpd(self.matrix_element('a u > u a a')))
        info = gpd_export.GPDInfo(self.matrix_element('a gpd_g > t t~'))
        self.assertEqual((info.channel, info.parton, info.is_gluon), ('gpd_g', 21, True))

    def test_x_flow_poles(self):
        """gamma q -> gamma gamma q: the six orderings of the photons on the
        quark line, two x-carrying propagators each; every pole holds one GPD
        leg, and every diagram enters two poles."""
        info = gpd_export.GPDInfo(self.matrix_element('a gpd_u > a a'))
        poles = info.pole_list()
        self.assertEqual(len(poles), 6)
        gpd_legs = set([info.n_in, info.n_out])
        count = dict((d, 0) for d in range(1, 7))
        for pole in poles:
            legs, mass, diagrams = pole[0], pole[2], pole[5]
            self.assertEqual(len(gpd_legs & set(legs)), 1, legs)
            self.assertEqual(mass, 'ZERO')
            for d in diagrams:
                count[d] += 1
        self.assertEqual(set(count.values()), set([2]))
        # the momenta K_i of the poles are distinct
        self.assertEqual(len(set((tuple(p[0]), p[1]) for p in poles)), 6)

    def test_no_x_pole_for_one_body_final_state(self):
        """gamma q -> e+ e- q through the photon: the x-carrying propagators
        are the quark propagators between the photons."""
        info = gpd_export.GPDInfo(self.matrix_element('a gpd_u > e+ e- / z'))
        for pole in info.pole_list():
            self.assertEqual(len(set([info.n_in, info.n_out]) & set(pole[0])), 1)

    def test_structural_poles(self):
        """A propagator next to a GPD leg whose other legs are one external
        leg of its own mass has its pole exactly at -sigma xi: it is placed
        there in GPD_POLES (in a strongly boosted frame K^2 loses the digits
        that locate it). gamma q -> gamma gamma q has only such poles; in
        gamma q -> gamma e+ e- q the propagators with two other legs have
        kinematics-dependent poles and stay computed."""
        info = gpd_export.GPDInfo(self.matrix_element('a gpd_u > a a'))
        poles = info.pole_list()
        self.assertTrue(all(info.is_structural_pole(p[0], p[2], p[3]) for p in poles))
        lines = info.gpd_pole_lines()
        self.assertEqual(lines.count('AP(NP) = DCMPLX(-XI, 0D0)'),
                         len([p for p in poles if p[1] > 0]))
        self.assertEqual(lines.count('AP(NP) = DCMPLX(XI, 0D0)'),
                         len([p for p in poles if p[1] < 0]))
        self.assertEqual(lines.count('NP0 = NP'), len(poles))
        info = gpd_export.GPDInfo(self.matrix_element('a gpd_u > a e+ e- / z'))
        kinds = [info.is_structural_pole(p[0], p[2], p[3]) for p in info.pole_list()]
        self.assertIn(True, kinds)
        self.assertIn(False, kinds)
        for p, structural in zip(info.pole_list(), kinds):
            others = [j for j in p[0] if j not in (info.n_in, info.n_out)]
            self.assertEqual(structural, len(others) == 1)

    def test_skip_matrix_reshuffle(self):
        """The matrix element of a GPD channel does not reshuffle the kernel
        momenta again (GPD_SET_POINT did it on the physical legs)."""
        text = ("      PMASS_RESH = 0.D0\n"
                "      CALL C2_RESHUFFLE_MOMENTA(P,PMASS_RESH,P_RESH)\n"
                "      IF (RESH_FAILED) THEN\n")
        out = gpd_export.skip_matrix_reshuffle(text, 'C2_')
        self.assertNotIn('CALL C2_RESHUFFLE_MOMENTA', out)
        self.assertIn('      P_RESH = P\n      RESH_FAILED = .FALSE.\n', out)
        self.assertEqual(gpd_export.skip_matrix_reshuffle(text, 'C1_'), text)

    def calls(self, line):
        me = self.matrix_element(line)
        writer = helas_call_writers.FortranUFOHelasCallWriter(self.cmd._curr_model)
        return me, writer.get_matrix_element_calls(me)

    def test_patch_helas_calls(self):
        me, calls = self.calls('a gpd_u > a a')
        info = gpd_export.GPDInfo(me)
        out = gpd_export.patch_helas_calls(calls, info)
        self.assertIn('CALL IXXXXX(PGPD(0),ZERO,NHEL(2),+1, FLAVOR(2),W(2))', out)
        self.assertEqual(out[out.index('CALL IXXXXX(PGPD(0),ZERO,NHEL(2),+1, FLAVOR(2),W(2))') + 1],
                         'W(2)%P = W(2)%P*CGPD_IN')
        self.assertIn('CALL OXXXXX(PGPD(0),ZERO,NHEL(5),+1, FLAVOR(5),W(5))', out)
        self.assertIn('W(5)%P = W(5)%P*CGPD_OUT', out)
        # the other legs are untouched
        self.assertIn('CALL VXXXXX(P(0,1),ZERO,NHEL(1),-1,W(1))', out)
        self.assertFalse(any('P(0,2)' in c or 'P(0,5)' in c for c in out))
        # every amplitude is masked
        amps = [c for c in calls if re.match(r'\s*CALL\s+\w+\(.*,AMP\(\d+\)\)', c)]
        masks = [c for c in out if 'GPD_AMASK' in c]
        self.assertEqual(len(amps), 6)
        self.assertEqual(len(masks), 6)
        for c in masks:
            n = re.match(r'AMP\((\d+)\) = AMP\((\d+)\)\*GPD_AMASK\((\d+)\)', c).groups()
            self.assertEqual(len(set(n)), 1)

    def test_patch_helas_calls_dual(self):
        """P-wave onia (dual numbers): the pair takes the dual copy PGPDD
        (blank before '(0)' against the later P(0 rewrite of the onia
        exporter)."""
        me, calls = self.calls('a gpd_u > a a')
        out = gpd_export.patch_helas_calls(calls, gpd_export.GPDInfo(me), dual=True)
        self.assertEqual(out[:4], ["DO IGPD = 0, 3", "PGPDD(IGPD)%COMP (:) = 0D0",
                                   "PGPDD(IGPD)%COMP (0) = PGPD(IGPD)", "ENDDO"])
        self.assertIn('CALL IXXXXX(PGPDD(0),ZERO,NHEL(2),+1, FLAVOR(2),W(2))', out)

    def test_patch_helas_calls_needs_the_pair(self):
        me, calls = self.calls('a gpd_u > a a')
        calls = [c for c in calls if 'P(0,2)' not in c]
        self.assertRaises(base_objects.PhysicsObject.PhysicsObjectError,
                          gpd_export.patch_helas_calls, calls, gpd_export.GPDInfo(me))

    def test_process_keys(self):
        """The coherent standalone output and madevent group the channels by
        the same physical process."""
        self.cmd.do_generate('a gpd > mu+ mu- / z @3')
        for amp in self.cmd._curr_amps:
            proc = amp.get('process')
            self.assertEqual(gpd_export.process_key(proc), gpd_madevent.physical_key(amp))
            self.assertEqual(gpd_export.physical_string(proc).split(' WEIGHTED')[0],
                             'a gpd > mu+ mu-')


class GPDExportTextTest(unittest.TestCase):
    """Text helpers, no model."""

    def test_makefile_lines(self):
        base = ("PROCESS=  matrix.o $(patsubst %.f,%.o,$(wildcard polynomial.f))\n"
                "LINKLIBS =  -L$(LIBDIR) -ldhelas -lmodel\n"
                "F_INCLUDE = -I$(ROOT_DIR)/Source/DHELAS\n")
        text = gpd_export.makefile_lines(base)
        self.assertIn("PROCESS=  matrix.o gpd_driver.o", text)
        self.assertIn("LINKLIBS =  -L$(LIBDIR) -lgpd -ldhelas -lmodel", text)
        self.assertIn("-I$(ROOT_DIR)/Source/GPD -I$(ROOT_DIR)/Source/DHELAS", text)
        for target in ('check_gpd:', 'mc_gpd:', 'dump_gpd:', 'libgpdme.$(dylibext):', 'gpd2py.so:'):
            self.assertIn(target, text)
        # P-wave onia link -ldhelas<n>
        text = gpd_export.makefile_lines(base.replace('-ldhelas', '-ldhelas2'))
        self.assertIn("LINKLIBS =  -L$(LIBDIR) -lgpd -ldhelas2 -lmodel", text)

    def test_gpd_card_defaults(self):
        """Standalone outputs: toy model and the x quadrature (deterministic
        CFFs); madevent outputs: the dd model and the two-replica Monte Carlo
        x integration. Otherwise the two cards are the same."""
        def values(card):
            out = {}
            for line in card.splitlines():
                w = line.split('!')[0].split('#')[0].split('=')
                if len(w) == 2:
                    out[w[1].strip()] = w[0].strip()
            return out
        sa, me = values(gpd_export.GPD_CARD), values(gpd_madevent.GPD_CARD_MADEVENT)
        self.assertEqual((sa['gpd_model'], sa['gpd_x_integration']), ('toy', 'quadrature'))
        self.assertEqual((me['gpd_model'], me['gpd_x_integration']), ('dd', 'mc'))
        self.assertEqual(set(sa), set(me))
        for k in sa:
            if k not in ('gpd_model', 'gpd_x_integration'):
                self.assertEqual(sa[k], me[k], k)

    def test_gpd_decl(self):
        text = gpd_export.gpd_decl('PROC1_', 7)
        self.assertNotIn('PGPDD', text)
        self.assertIn('PROC1_', text)
        self.assertIn('PGPDD', gpd_export.gpd_decl('PROC1_', 7, dual=True))

    def test_inline_files(self):
        with tempfile.NamedTemporaryFile('w', suffix='.inc', delete=False) as f:
            f.write("      INTEGER N\n")
        try:
            text = "      INCLUDE 'a.inc'\n      INCLUDE 'b.inc'\n"
            out = gpd_export.inline_files(text, {'a.inc': f.name})
            self.assertEqual(out, "      INTEGER N\n\n      INCLUDE 'b.inc'\n")
        finally:
            os.remove(f.name)

    def card_keys(self):
        keys = []
        for line in gpd_export.GPD_CARD.splitlines():
            w = line.split('!')[0].split('#')[0].split('=')
            if len(w) == 2:
                keys.append(w[1].strip())
        return keys

    def test_card_matches_the_fortran_readers(self):
        """Every entry of the default gpd_card is read by the GPD library
        (GPD_INIT) or the photon flux of lpp = 5 (gpd_photon_flux), and every
        name GPD_INIT reads is in the card."""
        models = open(os.path.join(MG7, 'Template', 'LO', 'Source', 'GPD', 'gpd_models.f')).read()
        flux = open(os.path.join(MG7, 'Template', 'LO', 'Source', 'PDF', 'PhotonFlux.f')).read()
        read = set(re.findall(r"\bNAME\.EQ\.'(\w+)'", models)) | \
            set(re.findall(r"\.(?:eq|ne)\.'(gpd_\w+)'", flux))
        keys = self.card_keys()
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), read)
        self.assertIn('gpd_gap', keys)
        self.assertIn('gpd_flux', keys)

    def test_reset_fixed_beam_pdfs(self):
        """The densities of an lpp = 0 beam are reset to 1 in both the scalar
        (with its nested scale IF) and the vector PDF lines, and nothing else
        changes."""
        text = "\n".join([
            "      IF (ABS(LPP(IB(1))).GE.1) THEN",
            "C       LP=SIGN(1,LPP(IB(1)))",
            "        IF (DSQRT(Q2FACT(IB(1))).EQ.0D0) THEN",
            "          QSCALE=0D0",
            "        ENDIF",
            "        EM1=PDG2PDF(LPP(IB(1)),11, IB(1),XBK(IB(1)), QSCALE)",
            "      ENDIF",
            "          IF (ABS(LPP(IB(2))).GE.1) THEN",
            "            Z2(IVEC)=PDG2PDF(LPP(IB(2)),23, IB(2),ALL_XBK(IB(2),IVEC)",
            "     $       ,DSQRT(ALL_Q2FACT(2, IVEC)))",
            "          ENDIF",
            "      PD(1)=EM1*Z2"])
        with tempfile.NamedTemporaryFile('w', suffix='.f', delete=False) as f:
            f.write(text)
        try:
            gpd_madevent.reset_fixed_beam_pdfs(f.name)
            out = open(f.name).read().split("\n")
        finally:
            os.remove(f.name)
        lines = text.split("\n")
        self.assertEqual(out[:6], lines[:6])
        self.assertEqual(out[6:9], ["      ELSE", "        EM1=1D0", "      ENDIF"])
        self.assertEqual(out[9:12], lines[7:10])
        self.assertEqual(out[12:15], ["          ELSE", "            Z2(IVEC)=1D0", "          ENDIF"])
        self.assertEqual(out[15:], lines[11:])


if __name__ == '__main__':
    unittest.main()
