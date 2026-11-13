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
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""CommonRunCmd.copy_lep_densities puts the chosen dressed-lepton density into
Source/PDF before Source is compiled. Source/makefile builds libpdf.a from the
PDF *directory*, whose mtime does not move when an existing file is
overwritten, so switching pdlabel between two launches of one process
directory used to keep the previous density linked: a fcce365ll run after a
fcce240ll one gave 1.324 pb instead of 1.195 pb for e- e+ > mu- mu+."""

from __future__ import absolute_import

import os
import shutil
import tempfile
import unittest

import madgraph
import madgraph.interface.common_run_interface as common_run
import madgraph.various.py8_bstrahl as py8_bstrahl

pjoin = os.path.join


class _Cmd(object):
    """copy_lep_densities needs no state"""

    copy_lep_densities = common_run.CommonRunCmd.copy_lep_densities


class TestCopyLepDensities(unittest.TestCase):

    # density file -> its object. bstrahl_pars.inc carries the parameters of
    # the density's beamstrahlung/ISR split, compiled into the shared bstrahl.f
    OBJECTS = {'eepdf.f': 'eepdf.o', 'gridpdfaux.f': 'gridpdfaux.o',
               'bstrahl_pars.inc': 'bstrahl.o'}
    DENSITY_FILES = tuple(OBJECTS)

    def setUp(self):
        self.me_dir = tempfile.mkdtemp(prefix='copy_lep_densities_')
        self.source = pjoin(self.me_dir, 'Source')
        self.pdf = pjoin(self.source, 'PDF')
        self.lib = pjoin(self.me_dir, 'lib', 'libpdf.a')
        for name in ('setA', 'setB'):
            os.makedirs(pjoin(self.pdf, 'lep_densities', name))
            for filename in self.DENSITY_FILES:
                self.write(pjoin(self.pdf, 'lep_densities', name, filename),
                           '      ! %s %s\n' % (name, filename))
        os.makedirs(os.path.dirname(self.lib))
        self.cmd = _Cmd()

    def tearDown(self):
        shutil.rmtree(self.me_dir)

    @staticmethod
    def write(path, text):
        with open(path, 'w') as stream:
            stream.write(text)

    def compile(self):
        """stand in for 'make': the objects and the library of what is there"""
        for filename in self.DENSITY_FILES:
            self.write(pjoin(self.pdf, self.OBJECTS[filename]), 'object')
        self.write(self.lib, 'library')

    def read(self, filename):
        with open(pjoin(self.pdf, filename)) as stream:
            return stream.read()

    def test_switching_density_invalidates_the_build(self):
        self.cmd.copy_lep_densities('setA', self.source)
        self.compile()

        self.cmd.copy_lep_densities('setB', self.source)

        for filename in self.DENSITY_FILES:
            self.assertIn('setB', self.read(filename))
            self.assertFalse(os.path.exists(pjoin(self.pdf, self.OBJECTS[filename])),
                             filename)
        # without this, Source/makefile never even enters PDF
        self.assertFalse(os.path.exists(self.lib))

    def test_same_density_keeps_the_build(self):
        self.cmd.copy_lep_densities('setA', self.source)
        self.compile()
        mtimes = dict((path, os.path.getmtime(path)) for path in
                      [self.lib] + [pjoin(self.pdf, f) for f in self.DENSITY_FILES])

        self.cmd.copy_lep_densities('setA', self.source)

        for filename in self.DENSITY_FILES:
            self.assertTrue(os.path.exists(pjoin(self.pdf, self.OBJECTS[filename])))
        for path, mtime in mtimes.items():
            self.assertEqual(os.path.getmtime(path), mtime, path)

    def test_first_copy_without_a_build(self):
        self.cmd.copy_lep_densities('setA', self.source)
        for filename in self.DENSITY_FILES:
            self.assertIn('setA', self.read(filename))

    def test_incomplete_density(self):
        """a density without its bstrahl_pars.inc would leave the
        no-beamstrahlung default in place, i.e. events written without their
        fractions"""
        self.cmd.copy_lep_densities('setA', self.source)
        self.compile()
        os.remove(pjoin(self.pdf, 'lep_densities', 'setB', 'bstrahl_pars.inc'))
        with self.assertRaises(Exception):
            self.cmd.copy_lep_densities('setB', self.source)
        # refused before anything is copied: setA stays whole, and built
        for filename in self.DENSITY_FILES:
            self.assertIn('setA', self.read(filename))
            self.assertTrue(os.path.exists(pjoin(self.pdf, self.OBJECTS[filename])))
        self.assertTrue(os.path.exists(self.lib))

    def test_unknown_density(self):
        with self.assertRaises(Exception):
            self.cmd.copy_lep_densities('nosuchset', self.source)


class TestShippedDensities(unittest.TestCase):
    """The sets in Template/Common/Source/PDF/lep_densities, as shipped."""

    SOURCE = pjoin(madgraph.MG5DIR, 'Template', 'Common', 'Source')
    # the densities that fold a CIRCE beamstrahlung spectrum into their grid
    WITH_BEAMSTRAHLUNG = set(['cepc240ll', 'clic3000ll', 'fcce240ll',
                              'fcce365ll', 'ilc500ll'])

    def sets(self):
        base = pjoin(self.SOURCE, 'PDF', 'lep_densities')
        return sorted(name for name in os.listdir(base)
                      if os.path.isdir(pjoin(base, name)))

    def test_every_set_is_complete(self):
        """what copy_lep_densities refuses to run without"""
        self.assertTrue(self.sets())
        for name in self.sets():
            for filename in TestCopyLepDensities.DENSITY_FILES:
                self.assertTrue(os.path.exists(pjoin(
                    self.SOURCE, 'PDF', 'lep_densities', name, filename)),
                    (name, filename))

    def test_density_has_bstrahl(self):
        """read from bstrahl_pars.inc, the same switch the Fortran uses"""
        self.assertTrue(self.WITH_BEAMSTRAHLUNG <= set(self.sets()))
        for name in self.sets():
            self.assertEqual(
                py8_bstrahl.density_has_bstrahl(name, self.SOURCE),
                name in self.WITH_BEAMSTRAHLUNG, name)
        self.assertFalse(py8_bstrahl.density_has_bstrahl('', self.SOURCE))
        self.assertFalse(py8_bstrahl.density_has_bstrahl('nosuchset',
                                                         self.SOURCE))

    def test_default_has_no_beamstrahlung(self):
        """the include next to the shared bstrahl.f, before any density is
        copied over it"""
        with open(pjoin(self.SOURCE, 'PDF', 'bstrahl_pars.inc')) as stream:
            text = stream.read().lower()
        self.assertIn('bs_has=.false.', text)


class TestBeamstrahlungSetup(unittest.TestCase):
    """py8_bstrahl.beamstrahlung_setup_error, shared by madevent and mg7."""

    def test_accepted(self):
        self.assertIsNone(py8_bstrahl.beamstrahlung_setup_error(
            'fcce240ll', 120.0, 120.0, False))
        # the run card may hold the energies as strings or ints
        self.assertIsNone(py8_bstrahl.beamstrahlung_setup_error(
            'fcce240ll', '120', 120, False))

    def test_mlm_refused(self):
        """no Pythia8 main both vetoes and degrades the beams"""
        error = py8_bstrahl.beamstrahlung_setup_error(
            'fcce240ll', 120.0, 120.0, True)
        self.assertIn('MLM', error)
        self.assertIn('fcce240ll', error)

    def test_asymmetric_beams_refused(self):
        error = py8_bstrahl.beamstrahlung_setup_error(
            'ilc500ll', 250.0, 200.0, False)
        self.assertIn('ebeam1 = ebeam2', error)
        self.assertNotIn('MLM', error)

    def test_both_reported(self):
        error = py8_bstrahl.beamstrahlung_setup_error(
            'ilc500ll', 250.0, 200.0, True)
        self.assertIn('MLM', error)
        self.assertIn('ebeam1 = ebeam2', error)


if __name__ == '__main__':
    unittest.main()
