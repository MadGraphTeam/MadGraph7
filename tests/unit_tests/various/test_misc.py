################################################################################
#
# Copyright (c) 2012 The MadGraph7 Development team and Contributors
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
"""Test the validity of the LHE parser"""

from __future__ import absolute_import
import os
import unittest
import subprocess
from unittest import mock
import madgraph.various.misc as misc
class TEST_misc(unittest.TestCase):
    
    def test_equal(self):
        
        eq = misc.equal
        
        self.assertFalse(eq(1,2,1))
        self.assertTrue(eq(1,1.2,1))
        self.assertFalse(eq(1, 1.2, 2))
        
        self.assertFalse(eq(10,20,1))
        self.assertTrue(eq(10,12,1))
        self.assertFalse(eq(10, 12, 2))        
        
        self.assertTrue(eq(100,1e2))
        self.assertFalse(eq(100,1e2 + 1e-3))
        self.assertTrue(eq(100,1e2 + 1e-4))
        self.assertTrue(eq(100,1e2 + 1e-5))

        self.assertFalse(eq(0.1,0.2,1))
        self.assertTrue(eq(0.1,0.12,1))
        self.assertFalse(eq(0.10, 0.12, 2))         

        self.assertFalse(eq(-0.1,-0.2,1))
        self.assertTrue(eq(-0.1,-0.12,1))
        self.assertFalse(eq(-0.10, -0.12, 2))         
        
        self.assertFalse(eq(-10,-20,1))
        self.assertTrue(eq(-10,-12,1))
        self.assertFalse(eq(-10, -12, 2)) 

        self.assertTrue(eq(-100,-1e2))
        self.assertFalse(eq(-100,-1e2 + 1e-3))
        self.assertTrue(eq(-100,-1e2 + 1e-4))
        self.assertTrue(eq(-100,-1e2 + 1e-5))
        
        self.assertTrue(eq(1,1.0))
        self.assertTrue(eq(1,1.0, 1))
        self.assertTrue(eq(1,1.0, 7))
        self.assertTrue(eq(1,1.0 + 2e-8, 7))
        self.assertTrue(eq(1,1.0 - 2e-8, 7))
        self.assertTrue(eq(1,1.0 + 2e-8, 8))
        self.assertFalse(eq(1,1.0 + 2e-7, 8))
        self.assertTrue(eq(9,9.0 + 2e-8, 7))
        self.assertTrue(eq(9,9.0 - 2e-8, 7))
        self.assertTrue(eq(9,9.0 + 2e-8, 8))
        self.assertFalse(eq(9,9.0 + 2e-8, 9))
        self.assertFalse(eq(1,-1.0))
        self.assertTrue(eq(0 ,0e-5))
        

        self.assertTrue(eq(80.419, 80.419002))
        for i in range(1,4):
            self.assertTrue(eq(81.966005, 81.891469,i))
        for i in range(4,7):
            self.assertFalse(eq(81.966005, 81.891469,i))
        
        # Check negative number
        self.assertTrue(eq(-1,-1))
        self.assertTrue(eq(-1,-1.0 + 1e-8, 7))
        self.assertTrue(eq(-1,-1.0 + 2e-8, 7))
        self.assertTrue(eq(-1,-1.0 - 2e-8, 7))
        self.assertTrue(eq(-1,-1.0 + 1e-8, 8))
        self.assertTrue(eq(-1,-1.0 + 2e-8, 8))
        self.assertTrue(eq(-1,-1.0 - 2e-8, 8))
        self.assertFalse(eq(-1,-1.0 + 1e-8, 9))
        self.assertFalse(eq(-1,-1.0 + 2e-8, 9))
        self.assertFalse(eq(-1,-1.0 - 2e-8, 9))

        self.assertFalse(eq(-1,1.0))
        self.assertTrue(eq(-0 ,-0e-5))
        
        self.assertTrue(eq(-100,-1e2))
        self.assertTrue(eq(-100,-1e2 + 1e-6))
        self.assertFalse(eq(-100,-1e2 + 1e-3))
        self.assertTrue(eq(-80.419, -80.419002))
        
        # check with 0
        self.assertTrue(eq(0 ,1e-11))  
        self.assertTrue(eq(0 ,1e-8))
        self.assertTrue(eq(0 ,1e-7))
        self.assertFalse(eq(0 ,1e-6))          
        self.assertFalse(eq(0 ,1e-1))

        self.assertTrue(eq(1e-11, 0))  
        self.assertTrue(eq(1e-8, 0))
        self.assertTrue(eq(1e-7, 0))
        self.assertFalse(eq(1e-6, 0))          
        self.assertFalse(eq(1e-1, 0))
        
        self.assertFalse(eq(0 ,1e-11, zero_limit=False))  
        self.assertFalse(eq(0 ,1e-8, zero_limit=False))
        self.assertFalse(eq(0 ,1e-7, zero_limit=False))
        self.assertFalse(eq(0 ,1e-6, zero_limit=False))          
        self.assertFalse(eq(0 ,1e-1, zero_limit=False))
        
        self.assertFalse(eq(1e-11, 0, zero_limit=False))  
        self.assertFalse(eq(1e-8, 0, zero_limit=False))
        self.assertFalse(eq(1e-7, 0, zero_limit=False))
        self.assertFalse(eq(1e-6, 0, zero_limit=False))          
        self.assertFalse(eq(1e-1, 0, zero_limit=False))         

    def test_ordered_set(self):

        set = misc.OrderedSet

        a = set(['a'])
        self.assertEqual(a.pop(), 'a')
        self.assertEqual(len(a), 0)

        a = set(['a', 'b'])
        self.assertEqual(a.pop(), 'a')
        self.assertEqual(len(a), 1)

    def test_popen_with_closed_sys_stdout_and_stderr_stdout(self):
        class ClosedStdout(object):
            def fileno(self):
                raise ValueError('I/O operation on closed file')

        with mock.patch('sys.__stdout__', ClosedStdout()):
            proc = misc.Popen(['echo', 'ok'], stdout=None, stderr=subprocess.STDOUT)
            self.assertEqual(proc.wait(), 0)


class TEST_pythia8_main164(unittest.TestCase):
    """main164, the Pythia8 driver of the LO shower, is found or compiled with
    the Makefile of the Pythia8 examples directory (a stub here)."""

    # stands in for the Pythia8 examples Makefile: 'make main164' copies the
    # source to an executable, 'make mainMG' (the HEPToolsInstaller rule) too
    makefile = ("main164: main164.cc\n\tcp main164.cc main164; chmod +x main164\n"
                "%s")

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.py8 = os.path.join(self.tmpdir, 'pythia8')
        self.examples = os.path.join(self.py8, 'share', 'Pythia8', 'examples')
        os.makedirs(self.examples)

    def tearDown(self):
        import shutil
        os.chmod(self.examples, 0o755)
        shutil.rmtree(self.tmpdir)

    def write_sources(self, mainMG=False, hepmc=2):
        with open(os.path.join(self.examples, 'main164.cc'), 'w') as fsock:
            fsock.write('#!/bin/sh\necho main164\n')
        rule = "mainMG: main164.cc\n\techo mainMG > main164; chmod +x main164\n" \
                                                               if mainMG else ''
        with open(os.path.join(self.examples, 'Makefile'), 'w') as fsock:
            fsock.write(self.makefile % rule)
        # the HepMC setup written by the Pythia8 configure script
        with open(os.path.join(self.examples, 'Makefile.inc'), 'w') as fsock:
            fsock.write('CXX_COMMON=-O2 -DHEPMC2HACK -DHEPMC2 -DGZIP\n'
                        'HEPMC2_USE=%s\nHEPMC2_INCLUDE=-I/old/include\n'
                        'HEPMC2_LIB=-L/old/lib -lHepMC\nHEPMC3_USE=%s\n'
                        'HEPMC3_INCLUDE=\nHEPMC3_LIB=\n' % (
                            'true' if hepmc == 2 else 'false',
                            'true' if hepmc == 3 else 'false'))

    def make_hepmc(self, version):
        """a fake HepMC<version> installation, as HEPToolsInstaller lays it out"""
        prefix = os.path.join(self.tmpdir, 'hepmc3' if version == 3 else 'hepmc')
        include = os.path.join(prefix, 'include', 'HepMC3' if version == 3 else 'HepMC')
        os.makedirs(include)
        os.makedirs(os.path.join(prefix, 'lib'))
        open(os.path.join(include, 'GenEvent.h'), 'w').close()
        open(os.path.join(prefix, 'lib', 'libHepMC%s.dylib' % ('3' if version == 3 else '')),
             'w').close()
        return prefix

    def test_find_main164(self):
        self.assertEqual(misc.find_pythia8_main164(self.py8), (None, None))
        self.write_sources()
        self.assertEqual(misc.find_pythia8_main164(self.py8), (None, self.examples))
        self.assertRaises(misc.MadGraph5Error, misc.get_pythia8_main164,
                          os.path.join(self.tmpdir, 'nopythia8'))

    def test_compile_in_examples(self):
        self.write_sources()
        executable = misc.get_pythia8_main164(self.py8)
        self.assertEqual(executable, os.path.join(self.examples, 'main164'))
        self.assertEqual(misc.find_pythia8_main164(self.py8),
                         (executable, self.examples))

    def test_compile_prefers_mainMG_rule(self):
        self.write_sources(mainMG=True)
        executable = misc.get_pythia8_main164(self.py8)
        self.assertEqual(open(executable).read().strip(), 'mainMG')

    def test_compile_in_fallback_for_readonly_examples(self):
        self.write_sources()
        os.chmod(self.examples, 0o555)
        if os.access(self.examples, os.W_OK):
            self.skipTest('running as a user that can write read-only directories')
        self.assertRaises(misc.MadGraph5Error, misc.get_pythia8_main164, self.py8)

        fallback = os.path.join(self.tmpdir, 'proc', 'lib', 'PY8_main164')
        executable = misc.get_pythia8_main164(self.py8, fallback_dir=fallback)
        self.assertEqual(executable, os.path.join(fallback, 'main164'))
        self.assertFalse(os.path.exists(os.path.join(self.examples, 'main164')))
        # reused as long as it was built against the same Pythia8
        mtime = os.path.getmtime(executable)
        self.assertEqual(misc.get_pythia8_main164(self.py8, fallback_dir=fallback),
                         executable)
        self.assertEqual(os.path.getmtime(executable), mtime)
        # rebuilt when made against another Pythia8, even if make would
        # consider the old executable up to date
        with open(os.path.join(fallback, 'BUILD_STAMP'), 'w') as fsock:
            fsock.write('/another/pythia8')
        with open(executable, 'w') as fsock:
            fsock.write('stale')
        misc.get_pythia8_main164(self.py8, fallback_dir=fallback)
        self.assertIn('echo main164', open(executable).read())
        self.assertEqual(open(os.path.join(fallback, 'BUILD_STAMP')).read(),
                         os.path.realpath(self.py8))

    def test_hepmc_version_of_the_pythia8_build(self):
        self.write_sources(hepmc=2)
        self.assertEqual(misc.pythia8_hepmc_version(self.examples), 2)
        self.write_sources(hepmc=3)
        self.assertEqual(misc.pythia8_hepmc_version(self.examples), 3)
        self.write_sources(hepmc=None)
        self.assertEqual(misc.pythia8_hepmc_version(self.examples), None)

    def test_find_hepmc(self):
        self.assertEqual(misc.find_hepmc(3, [self.tmpdir]), None)
        prefix = self.make_hepmc(3)
        self.assertEqual(misc.find_hepmc(3, [None, self.tmpdir, prefix]),
                         (os.path.realpath(prefix),
                          os.path.realpath(os.path.join(prefix, 'lib'))))
        # a HepMC3 installation is not a HepMC2 one
        self.assertEqual(misc.find_hepmc(2, [prefix]), None)

    def test_same_hepmc_version_uses_the_default_main164(self):
        self.write_sources(hepmc=2)
        executable = misc.get_pythia8_main164(self.py8, hepmc_version=2)
        self.assertEqual(executable, os.path.join(self.examples, 'main164'))

    def test_other_hepmc_version_gets_its_own_main164(self):
        self.write_sources(hepmc=2)
        fallback = os.path.join(self.tmpdir, 'proc', 'lib', 'PY8_main164')
        # no HepMC3 installation: clear error
        self.assertRaises(misc.MadGraph5Error, misc.get_pythia8_main164,
                          self.py8, fallback_dir=fallback, hepmc_version=3)
        # found next to Pythia8, as HEPToolsInstaller installs it
        prefix = os.path.realpath(self.make_hepmc(3))
        self.assertEqual(prefix, os.path.realpath(os.path.join(self.py8, os.pardir, 'hepmc3')))
        executable = misc.get_pythia8_main164(self.py8, fallback_dir=fallback,
                                              hepmc_version=3)
        # shared by all processes, next to the default main164
        shared = os.path.join(self.examples, 'main164_hepmc3')
        self.assertEqual(executable, os.path.join(shared, 'main164'))
        self.assertFalse(os.path.exists(os.path.join(self.examples, 'main164')))
        makefile_inc = open(os.path.join(shared, 'Makefile.inc')).read()
        self.assertIn('HEPMC2_USE=false', makefile_inc)
        self.assertIn('HEPMC3_USE=true', makefile_inc)
        self.assertIn('HEPMC3_INCLUDE=-I%s/include' % prefix, makefile_inc)
        self.assertIn('HEPMC3_LIB=-L%s/lib -Wl,-rpath,%s/lib -lHepMC3' % (prefix, prefix),
                      makefile_inc)
        self.assertIn('CXX_COMMON=-O2 -DGZIP', makefile_inc)
        self.assertNotIn('/old/', makefile_inc)
        # the default one is not affected
        self.assertEqual(misc.get_pythia8_main164(self.py8),
                         os.path.join(self.examples, 'main164'))

    def test_other_hepmc_version_in_readonly_pythia8(self):
        self.write_sources(hepmc=2)
        prefix = self.make_hepmc(3)
        fallback = os.path.join(self.tmpdir, 'proc', 'lib', 'PY8_main164')
        os.chmod(self.examples, 0o555)
        if os.access(self.examples, os.W_OK):
            self.skipTest('running as a user that can write read-only directories')
        self.assertRaises(misc.MadGraph5Error, misc.get_pythia8_main164,
                          self.py8, hepmc_version=3, hepmc_paths=[prefix])
        executable = misc.get_pythia8_main164(self.py8, fallback_dir=fallback,
                                              hepmc_version=3, hepmc_paths=[prefix])
        self.assertEqual(executable, fallback + '_hepmc3/main164')

    def test_pythia8_hepmc_flags(self):
        # a pythia8-config that only knows HepMC2 (Pythia8 configured with it)
        bindir = os.path.join(self.py8, 'bin')
        os.makedirs(bindir)
        with open(os.path.join(bindir, 'pythia8-config'), 'w') as fsock:
            fsock.write('#!/bin/sh\n[ "$1" = "--hepmc2" ] && echo "-I/h2/include -lHepMC"\n'
                        'exit 0\n')
        os.chmod(os.path.join(bindir, 'pythia8-config'), 0o755)
        self.assertEqual(misc.get_pythia8_hepmc_flags(self.py8, 2), '-I/h2/include -lHepMC')
        # HepMC3 is taken from its installation next to Pythia8
        self.assertRaises(misc.MadGraph5Error, misc.get_pythia8_hepmc_flags, self.py8, 3)
        prefix = os.path.realpath(self.make_hepmc(3))
        self.assertEqual(misc.get_pythia8_hepmc_flags(self.py8, 3),
                         '-I%s/include -L%s/lib -Wl,-rpath,%s/lib -lHepMC3' % ((prefix,)*3))

    def test_hepmc_file_version(self):
        import gzip
        hepmc2 = os.path.join(self.tmpdir, 'events.hepmc')
        with open(hepmc2, 'w') as fsock:
            fsock.write('\nHepMC::Version 2.06.09\n'
                        'HepMC::IO_GenEvent-START_EVENT_LISTING\nE 0 -1\n')
        hepmc3 = os.path.join(self.tmpdir, 'events3.hepmc.gz')
        with gzip.open(hepmc3, 'wt') as fsock:
            fsock.write('HepMC::Version 3.02.07\n'
                        'HepMC::Asciiv3-START_EVENT_LISTING\nW Weight\nE 0 3 5\n')
        self.assertEqual(misc.hepmc_file_version(hepmc2), 2)
        self.assertEqual(misc.hepmc_file_version(hepmc3), 3)
        self.assertEqual(misc.hepmc_file_version(os.path.join(self.tmpdir, 'none')), None)
