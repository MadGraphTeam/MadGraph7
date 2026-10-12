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
"""MG7 custom cut, scale, PDF and bias functions ([custom_functions]).

Each custom function is created with bin/init_custom_func in a copy of one
``p p > t t~`` output, given a body that reproduces a built-in setting, and
compared with a run using that built-in setting:

  * cut: pT > 150 GeV for both tops against ``top-pt.min = 150``, and every
    event passes the cut;
  * scale: H_T/2 against ``dynamical_scale_choice = half_transverse_mass``,
    and the LHE scale of every event is H_T/2 of its momenta;
  * pdf: the LHAPDF set of the run card called through the LHAPDF C++ API
    against the built-in interpolation of the same set;
  * bias: b = max((pT / 100 GeV)^2, 1) keeps the cross section and the
    weighted pT tail, while putting more events into the tail;
  * every function in every language with a toolchain here builds, loads and
    returns its default value when evaluated through madspace.

Run locally with::

    ./tests/test_manager.py test_.*custom.*_mg7 -pA -t0

``MG7_CUSTOM_EVENTS`` sets the events per run (default 5000).
"""

from __future__ import absolute_import
from __future__ import division

import glob
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import madgraph.interface.master_interface as MGCmd
import madgraph.various.misc as misc
from tests.acceptance_tests.test_check_xsec_processes_mg7 import (
    _REFERENCE_PDF, _edit_run_card, _require_mg7_runtime, _tail)

pjoin = os.path.join

_PROCESS = 'p p > t t~'
_EVENTS = int(os.environ.get('MG7_CUSTOM_EVENTS', 5000))
_NSIGMA = 3
_SEED = 1234
_TOP_PT_CUT = 150.

# bodies of the custom functions, in C++
_CUT_BODY = r'''
#include <cmath>
extern "C" double custom_cut(int particle_count, const int* pdg_ids, const double momenta[][4]) {
    for (int i = 2; i < 4; ++i) {
        if (std::hypot(momenta[i][1], momenta[i][2]) < %(cut)r) {
            return 0.;
        }
    }
    return 1.;
}
''' % {'cut': _TOP_PT_CUT}

_SCALE_BODY = r'''
#include <cmath>
extern "C" double custom_scale(int particle_count, const int* pdg_ids, const double momenta[][4]) {
    double ht = 0.;
    for (int i = 2; i < particle_count; ++i) {
        double e = momenta[i][0], pz = momenta[i][3];
        ht += std::sqrt(std::fmax(e * e - pz * pz, 0.));
    }
    return ht / 2.;
}
'''

_PDF_BODY = r'''
#include <mutex>
#include "LHAPDF/LHAPDF.h"
extern "C" double custom_pdf(int beam_index, int pdg_id, double x, double q) {
    static LHAPDF::PDF* pdf = LHAPDF::mkPDF("%(pdf)s", 0);
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    return pdf->xfxQ(pdg_id, x, q);
}
''' % {'pdf': _REFERENCE_PDF}

_BIAS_BODY = r'''
#include <cmath>
extern "C" double custom_bias(int particle_count, const int* pdg_ids, const double momenta[][4]) {
    double pt2 = momenta[2][1] * momenta[2][1] + momenta[2][2] * momenta[2][2];
    return std::fmax(pt2 / (100. * 100.), 1.);
}
'''


def _set(text, key, value):
    """Set ``key = value`` (first occurrence at the start of a line)."""
    text, count = re.subn(r'(?m)^%s = .*$' % re.escape(key),
                          '%s = %s' % (key, value), text, count=1)
    assert count == 1, 'no %s in the run card' % key
    return text


def _read_lhe(path):
    """(weights, scales, momenta) of an LHE file; momenta[i] lists the
    (E, px, py, pz) of every particle of event i."""
    weights, scales, momenta = [], [], []
    with gzip.open(path, 'rt') as f:
        lines = iter(f)
        for line in lines:
            if not line.startswith('<event'):
                continue
            fields = next(lines).split()
            count = int(fields[0])
            weights.append(float(fields[2]))
            scales.append(float(fields[3]))
            event = []
            for _ in range(count):
                p = next(lines).split()
                event.append((float(p[9]), float(p[6]), float(p[7]), float(p[8])))
            momenta.append(event)
    return weights, scales, momenta


def _pt(p):
    return math.hypot(p[1], p[2])


def _import_madspace():
    """The madspace of this MadGraph, as bin/generate_events imports it, or
    None. Done before anything imports another madspace from the path."""
    try:
        from madgraph.iolibs.template_files.mg7.bootstrap import (
            ensure_madspace, drop_install_path)
        ensure_madspace(interactive=False)
        import madspace
        drop_install_path()
        return madspace
    except Exception:
        return None


class MG7CustomFunctionsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ms = _import_madspace()
        cls.path = tempfile.mkdtemp(prefix='mg7_custom_')
        cls.base = pjoin(cls.path, 'BASE')
        cls.reference = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.path, ignore_errors=True)

    def base_process(self):
        """The ``p p > t t~`` output shared by the tests: fixed scales, the
        PDF set of the xsec references, LHE output and a fixed seed. The first
        call also generates (and compiles) it with a reference run."""
        _require_mg7_runtime(self)
        cls = type(self)
        if cls.reference is None:
            mg = MGCmd.MasterCmd()
            mg.no_notification()
            mg.exec_cmd('set automatic_html_opening False --no_save')
            mg.exec_cmd('generate %s' % _PROCESS)
            mg.exec_cmd('output mg7 %s' % self.base)
            toml = pjoin(self.base, 'Cards', 'run_card.toml')
            _edit_run_card(toml, _EVENTS, disable_jet_cuts=False)
            text = open(toml).read()
            text = _set(text, 'output_format', '"lhe"')
            text = _set(text, 'seed', _SEED)
            text = text.replace('[multiparticles]\n', '[multiparticles]\ntop = [6, -6]\n')
            open(toml, 'w').write(text)
            cls.reference = self.run_process(self.base)
        return self.base

    def copy_process(self, name):
        target = pjoin(self.path, name)
        shutil.copytree(self.base_process(), target, symlinks=True)
        # the copy starts without the events of the reference run
        shutil.rmtree(pjoin(target, 'Events'), ignore_errors=True)
        os.mkdir(pjoin(target, 'Events'))
        return target

    def edit_card(self, proc_dir, edit):
        toml = pjoin(proc_dir, 'Cards', 'run_card.toml')
        text = open(toml).read()
        open(toml, 'w').write(edit(text))

    def init_custom_function(self, proc_dir, kind, body):
        """bin/init_custom_func for ``kind`` in C++, with ``body``; the run
        builds it."""
        log = pjoin(proc_dir, 'init_custom_func.log')
        with open(log, 'w') as logfh:
            ret = subprocess.call(
                [sys.executable, pjoin(proc_dir, 'bin', 'init_custom_func'),
                 '-f', kind, '-l', 'cpp', '-y', '--no-build'],
                cwd=proc_dir, stdout=logfh, stderr=subprocess.STDOUT)
        self.assertEqual(ret, 0, 'bin/init_custom_func failed:\n%s' % _tail(log))
        with open(pjoin(proc_dir, 'custom_%s' % kind, 'custom_%s.cpp' % kind), 'w') as f:
            f.write(body)

    def run_process(self, proc_dir):
        """Run bin/generate_events: (cross section, error, LHE path)."""
        log = pjoin(proc_dir, 'mg7_gen.log')
        with open(log, 'w') as logfh:
            ret = subprocess.call(
                [sys.executable, pjoin(proc_dir, 'bin', 'generate_events'), '-f'],
                cwd=proc_dir, stdout=logfh, stderr=subprocess.STDOUT)
        self.assertEqual(ret, 0, 'mg7 generate_events failed:\n%s' % _tail(log))
        infos = sorted(glob.glob(pjoin(proc_dir, 'Events', '*', 'info.json')))
        self.assertTrue(infos, 'no info.json produced:\n%s' % _tail(log))
        with open(infos[-1]) as f:
            info = json.load(f)['process']
        lhe = pjoin(os.path.dirname(infos[-1]), 'events.lhe.gz')
        self.assertTrue(os.path.exists(lhe), 'no LHE file produced')
        return float(info['mean']), float(info['error']), lhe

    def assertCompatible(self, got, ref, what):
        """Cross sections (mean, error) agree within _NSIGMA."""
        sigma = math.hypot(got[1], ref[1])
        self.assertLessEqual(
            abs(got[0] - ref[0]), _NSIGMA * sigma,
            '%s: %.5g +- %.3g pb against %.5g +- %.3g pb'
            % (what, got[0], got[1], ref[0], ref[1]))

    def test_custom_cut_mg7(self):
        builtin = self.copy_process('CUT_BUILTIN')
        self.edit_card(builtin, lambda t: t.replace(
            '[cuts]\n', '[cuts]\ntop-pt.min = %r\n' % _TOP_PT_CUT))
        ref = self.run_process(builtin)

        custom = self.copy_process('CUT_CUSTOM')
        self.init_custom_function(custom, 'cut', _CUT_BODY)
        got = self.run_process(custom)
        self.assertCompatible(got, ref, 'custom cut')

        _, _, momenta = _read_lhe(got[2])
        for event in momenta:
            for p in event[2:4]:
                # up to the precision of the LHE file
                self.assertGreaterEqual(_pt(p), _TOP_PT_CUT * (1 - 1e-6))

    def test_custom_scale_mg7(self):
        def dynamic(text):
            text = _set(text, 'fixed_ren_scale', 'false')
            text = _set(text, 'fixed_fact_scale', 'false')
            return _set(text, 'dynamical_scale_choice', '"half_transverse_mass"')

        builtin = self.copy_process('SCALE_BUILTIN')
        self.edit_card(builtin, dynamic)
        ref = self.run_process(builtin)

        custom = self.copy_process('SCALE_CUSTOM')
        # partonic_energy differs from H_T/2, so the custom function must win
        self.edit_card(custom, lambda t: _set(
            dynamic(t), 'dynamical_scale_choice', '"partonic_energy"'))
        self.init_custom_function(custom, 'scale', _SCALE_BODY)
        got = self.run_process(custom)
        self.assertCompatible(got, ref, 'custom scale')

        _, scales, momenta = _read_lhe(got[2])
        for scale, event in zip(scales, momenta):
            ht = sum(math.sqrt(max(p[0] ** 2 - p[3] ** 2, 0.)) for p in event[2:])
            self.assertAlmostEqual(scale / (ht / 2), 1., places=5)

    def test_custom_pdf_mg7(self):
        lhapdf_config = misc.resolve_lhapdf().config
        if not lhapdf_config:
            self.skipTest('lhapdf-config not found')
        flags = {}
        for flag in ('cflags', 'libs', 'libdir'):
            flags[flag] = subprocess.check_output(
                [lhapdf_config, '--%s' % flag], text=True).strip()

        custom = self.copy_process('PDF_CUSTOM')
        self.init_custom_function(custom, 'pdf', _PDF_BODY)
        build = subprocess.run(
            ['make', 'CFLAGS=%s' % flags['cflags'],
             'LIBS=%s -Wl,-rpath,%s' % (flags['libs'], flags['libdir'])],
            cwd=pjoin(custom, 'custom_pdf'), capture_output=True, text=True)
        self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
        got = self.run_process(custom)
        self.assertCompatible(got, type(self).reference, 'custom pdf')

    def test_custom_bias_mg7(self):
        custom = self.copy_process('BIAS_CUSTOM')
        ref = type(self).reference
        self.init_custom_function(custom, 'bias', _BIAS_BODY)
        got = self.run_process(custom)
        self.assertCompatible(got, ref, 'biased cross section')

        def tail(lhe, threshold=250.):
            weights, _, momenta = _read_lhe(lhe)
            n = len(weights)
            in_tail = [w for w, event in zip(weights, momenta)
                       if _pt(event[2]) > threshold]
            mean = sum(weights) / n
            error = math.sqrt(sum((w - mean) ** 2 for w in weights) / n / (n - 1))
            xsec = sum(in_tail) / n
            xsec_error = math.sqrt(sum(w ** 2 for w in in_tail)) / n
            return weights, (mean, error), (xsec, xsec_error), len(in_tail) / n

        ref_weights, _, ref_tail, ref_fraction = tail(ref[2])
        weights, mean, tail_xsec, fraction = tail(got[2])
        self.assertGreater(max(weights) / min(weights), 2.,
                           'the biased events should not have a constant weight')
        self.assertCompatible(mean, ref[:2], 'mean biased event weight')
        self.assertCompatible(tail_xsec, ref_tail, 'pT > 250 GeV cross section')
        self.assertGreater(fraction, 2 * ref_fraction,
                           'the bias should put more events into the tail')

    def test_init_custom_func_languages_mg7(self):
        ms = type(self).ms
        if ms is None or not hasattr(ms, 'CustomFunction'):
            self.skipTest('madspace with custom functions unavailable')
        import numpy as np
        compilers = {'c': 'cc', 'cpp': 'c++', 'fortran': 'gfortran',
                     'cuda': 'nvcc', 'hip': 'hipcc'}
        from madgraph.iolibs.template_files.mg7 import init_custom_func

        expected = {'cut': 1., 'bias': 1.}
        momenta = np.array([[[500., 0., 0., 500.], [500., 0., 0., -500.],
                              [500., 100., 50., 200.], [500., -100., -50., -200.]],
                             [[300., 0., 0., 300.], [200., 0., 0., -200.],
                              [250., 30., 0., 100.], [250., -30., 0., 0.]]])
        tested = []
        for language, compiler in compilers.items():
            if shutil.which(compiler) is None:
                continue
            me_dir = pjoin(self.path, 'LANG_%s' % language)
            os.makedirs(me_dir)
            for kind in ('cut', 'scale', 'pdf', 'bias'):
                ret = init_custom_func.main(me_dir=me_dir, argv=[
                    '-f', kind, '-l', language, '-y', '--build', '--no-set-run-card'])
                self.assertEqual(ret, 0, '%s %s did not build' % (language, kind))
                self.assertTrue(os.path.exists(pjoin(me_dir, 'lib', 'custom_%s.so' % kind)))
            tested.append(language)
            if language in ('cuda', 'hip'):
                continue
            context = ms.Context(device=ms.cpu_device(), thread_count=1)
            for kind in ('cut', 'scale', 'bias'):
                api = context.load_matrix_element(
                    pjoin(me_dir, 'lib', 'custom_%s.so' % kind), '')
                function = ms.CustomFunction(
                    api, getattr(ms.CustomFunction, kind), 4, [[21, 21, 6, -6]])
                values = ms.FunctionRuntime(function.function(), context)(momenta)
                for value, event in zip(np.array(values), momenta):
                    if kind == 'scale':
                        p = [a + b for a, b in zip(event[0], event[1])]
                        target = math.sqrt(p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2)
                    else:
                        target = expected[kind]
                    self.assertAlmostEqual(value, target, places=9)
            api = context.load_matrix_element(pjoin(me_dir, 'lib', 'custom_pdf.so'), '')
            pdf = ms.PartonDensity(ms.CustomPdf(api), [21, 2], 0)
            values = ms.FunctionRuntime(pdf.function(), context)(
                np.array([0.1, 0.5]), np.array([100., 100.]))
            for x, row in zip([0.1, 0.5], np.array(values)):
                toy = math.sqrt(x) * (1 - x) ** 3
                self.assertAlmostEqual(row[0], 3 * toy, places=12)
                self.assertAlmostEqual(row[1], 0.5 * toy, places=12)
        if not tested:
            self.skipTest('no compiler found')
