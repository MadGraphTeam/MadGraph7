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
"""The python side of madmatrix.density (no library is built here: see
test_madmatrix_density_matches_fortran in tests/acceptance_tests/test_cmd.py)."""

import json
import os
import shutil
import tempfile

from unittest import skipIf

import tests.unit_tests as unittest

import madmatrix.density as density

# importing madmatrix.density works without numpy (most CI runners have none);
# loading and evaluating a library does not
needs_numpy = skipIf(density.np is None, 'madmatrix.density needs numpy')


class _Sub(object):
    def __init__(self, index, legs, ninitial):
        self.index, self.legs, self.ninitial = index, tuple(legs), ninitial


def _library(table):
    """A MadDensityLibrary whose process table is `table` ({pdgs: (sub,
    flavor)}), without any library behind it."""
    lib = density.MadDensityLibrary.__new__(density.MadDensityLibrary)
    lib._by_pdgs = dict(table)
    lib._dispatch_cache = {}
    lib._merged = {}
    for pdgs, (sub, _) in lib._by_pdgs.items():
        for leg, pdg in zip(sub.legs, pdgs):
            lib._merged[abs(pdg)] = abs(leg)
    return lib


@needs_numpy
class TestMadMatrixDensityDispatch(unittest.TestCase):

    def setUp(self):
        # q q' > z q q', the quarks merged as 81: one leg ordering per flavor
        self.sub = _Sub(0, [81, 81, 23, 81, 81], 2)
        self.lib = _library({(1, 2, 23, 1, 2): (self.sub, 0),
                             (2, 2, 23, 2, 2): (self.sub, 1)})

    def test_tabulated_order(self):
        self.assertEqual(self.lib.dispatch([1, 2, 23, 1, 2]), (self.sub, 0, None))
        self.assertEqual(self.lib.dispatch([2, 2, 23, 2, 2]), (self.sub, 1, None))

    def test_repair_swaps_within_blocks(self):
        """`d u > z d u` tabulated as `u d > z u d`: both blocks swapped."""
        sub, iflavor, perm = self.lib.dispatch([2, 1, 23, 2, 1])
        self.assertEqual((sub, iflavor), (self.sub, 0))
        self.assertEqual(perm, [1, 0, 2, 4, 3])
        # the permuted legs are the tabulated ones
        self.assertEqual([[2, 1, 23, 2, 1][i] for i in perm], [1, 2, 23, 1, 2])

    def test_repair_keeps_frozen_legs(self):
        """A leg whose helicity is open in the density cannot move."""
        self.assertRaises(density.MadDensityError, self.lib.dispatch,
                          [2, 1, 23, 2, 1], (4, 5))
        # moving the initial state only is fine with the final legs frozen
        sub, iflavor, perm = self.lib.dispatch([2, 1, 23, 1, 2], (4, 5))
        self.assertEqual(perm, [1, 0, 2, 3, 4])

    def test_never_across_initial_final(self):
        lib = _library({(1, 21, 23, 1): (_Sub(0, [81, 21, 23, 81], 2), 0)})
        self.assertRaises(density.MadDensityError, lib.dispatch, [21, 1, 23, 1])

    def test_unknown_process(self):
        self.assertRaises(density.MadDensityError, self.lib.dispatch, [3, 3, 23, 3, 3])


@needs_numpy
class TestMadMatrixDensityFiles(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='mmdensity')
        os.makedirs(os.path.join(self.tmp, 'lib'))
        os.makedirs(os.path.join(self.tmp, 'SubProcesses'))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def touch(self, name):
        open(os.path.join(self.tmp, 'lib', name), 'w').close()

    def test_make_variables(self):
        self.assertEqual(density.make_variables(),
                         ['PGOAL=lib', 'BACKEND=scalar', 'DENSITY=1'])
        self.assertEqual(density.build_command(4, 'simd_128', '_ms2d'),
                         ['make', '-j4', 'PGOAL=lib', 'BACKEND=simd_128', 'DENSITY=1',
                          'LIBTAG=_ms2d'])

    def test_find_libtag(self):
        self.assertRaises(density.MadDensityError, density.find_libtag, self.tmp)
        self.touch('libmadmatrix_common_ms1p_scalar.so')
        self.touch('libmadmatrix_common_ms1p_simd_128.so')
        self.assertEqual(density.find_libtag(self.tmp), '_ms1p')
        self.assertEqual(density.find_libtag(self.tmp, 'simd_128'), '_ms1p')
        self.touch('libmadmatrix_common_scalar.so')
        self.assertRaises(density.MadDensityError, density.find_libtag, self.tmp)

    def test_is_output(self):
        self.assertFalse(density.is_output(self.tmp))
        with open(os.path.join(self.tmp, 'SubProcesses', density.PROCESSES_FILE), 'w') as f:
            json.dump({'subprocesses': []}, f)
        self.assertTrue(density.is_output(self.tmp))

    def test_offshell_any_backend(self):
        """Every backend evaluates a '*' leg at the invariant mass of its
        momentum (per SIMD lane, per GPU thread): a SIMD build of such a
        process is not refused, it only has to exist."""
        with open(os.path.join(self.tmp, 'SubProcesses', density.PROCESSES_FILE), 'w') as f:
            json.dump({'subprocesses': [{'path': 'P1_t_bLxN', 'legs': [6, 5, -82, 83],
                                         'ninitial': 1, 'offshell': [1], 'processes': [],
                                         'library': 'libmadmatrix{libtag}_P1_t_bLxN_{backend}.so',
                                         'flavors': [[[6, 5, -11, 12]]]}]}, f)
        try:
            density.MadDensityLibrary(self.tmp, 'simd_128', '_none%d' % os.getpid())
        except density.MadDensityError as error:
            # refused for the missing build, not for the backend
            self.assertIn('missing', str(error))
        else:
            self.fail('no library was built, yet it loaded')

    def test_same_name_twice_is_refused(self):
        """Two libraries of one name: the loader would hand back the first."""
        other = tempfile.mkdtemp(prefix='mmdensity')
        try:
            name = 'libmadmatrix_common_clash%d_scalar.so' % os.getpid()
            density._loaded[name] = os.path.realpath(os.path.join(other, name))
            self.assertRaisesRegex(density.MadDensityError, 'LIBTAG',
                                   density._load, os.path.join(self.tmp, 'lib', name))
        finally:
            density._loaded.pop(name, None)
            shutil.rmtree(other)
