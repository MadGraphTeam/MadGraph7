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
"""The two reweighting backends of an MG7 output give the same weights, in
the same place.

A small MG7 process (u u~ > t t~, 500 events, LHE output, systematics on)
is generated once; the run is copied, and one copy is reweighted with
MadtRex, the other with the classic module (launch.run_reweighting, as the
launch runs them). The card uses the shared vocabulary: 'change helicity
False' and a parameter given by its param_card name ('set MT 175'). Both
add their weights to the run's own event file (Events/<run>/events.lhe.gz),
after the systematics weights, and agree event by event. A second process
directory with output_format = lhe_npy checks that MadtRex reweights
events.npy in place (generate_events --reweight=CARD) and that npy_to_lhe
carries the new weight.

Skipped unless madspace and madtrex are installed and numpy.f2py runs with
this Python (the classic module compiles its matrix elements with f2py, by
default the one installed with this Python: misc.default_f2py()).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

MG_ROOT = Path(__file__).resolve().parents[2]


def both_backends_available():
    if not (MG_ROOT / 'madspace' / 'install' / 'madspace').is_dir():
        return False
    if not (MG_ROOT / 'madtrex' / 'install' / 'madtrex').is_dir():
        return False
    if shutil.which('gfortran') is None:
        return False
    result = subprocess.run([sys.executable, '-m', 'numpy.f2py', '-v'],
                            capture_output=True, text=True)
    return result.returncode == 0


CARD = "change helicity False\nlaunch --rwgt_name=mt175\nset MT 175\n"

# reweight run <run> of the process directory (the cwd) with <backend>
REWEIGHT = textwrap.dedent("""
    import os, sys
    sys.path.insert(0, {root!r})
    from madgraph.iolibs.template_files.mg7 import bootstrap
    bootstrap.ensure_madspace(interactive=False)
    from madgraph.iolibs.template_files.mg7 import launch
    class Process:
        run_path = os.path.join("Events", {run!r})
        events_path = os.path.join(run_path, "events.lhe.gz")
        run_card = {{"run": {{"output_format": "lhe"}}}}
    launch.run_reweighting(Process(), launch.REWEIGHT_CARD, requested={backend!r})
""")


@unittest.skipUnless(both_backends_available(),
                     'needs madspace, madtrex, gfortran and numpy.f2py')
class ReweightBackendsAgreeTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def run_checked(self, args, **kwargs):
        result = subprocess.run(args, capture_output=True, text=True, **kwargs)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        return result

    def make_process(self, name, output_format):
        proc = os.path.join(self.tmp, name)
        script = os.path.join(self.tmp, name + '.mg5')
        with open(script, 'w') as f:
            f.write('import model sm\ngenerate u u~ > t t~\noutput mg7 %s -f\n' % proc)
        self.run_checked([sys.executable, str(MG_ROOT / 'bin' / 'madgraph'), script])
        run_card = os.path.join(proc, 'Cards', 'run_card.toml')
        with open(run_card) as f:
            text = f.read()
        text = text.replace('events = 100000', 'events = 500', 1)
        text = text.replace('output_format = "lhe_npy"', 'output_format = "%s"' % output_format, 1)
        with open(run_card, 'w') as f:
            f.write(text)
        with open(os.path.join(proc, 'Cards', 'reweight_card.dat'), 'w') as f:
            f.write(CARD)
        return proc

    def weights(self, path):
        sys.path.insert(0, str(MG_ROOT))
        import madgraph.various.lhe_parser as lhe_parser
        return [(event.wgt, event.parse_reweight())
                for event in lhe_parser.EventFile(path)]

    def test_classic_and_madtrex_write_the_same_file(self):
        proc = self.make_process('PROC', 'lhe')
        self.run_checked([sys.executable, 'bin/generate_events', '-f'], cwd=proc)
        events = os.path.join(proc, 'Events')
        shutil.copytree(os.path.join(events, 'run_01'), os.path.join(events, 'run_02'))

        for run, backend in (('run_01', 'madtrex'), ('run_02', 'classic')):
            self.run_checked([sys.executable, '-c', REWEIGHT.format(
                root=str(MG_ROOT), run=run, backend=backend)], cwd=proc)
            with open(os.path.join(events, run, 'reweighting.json')) as f:
                record = json.load(f)
            self.assertEqual(record['backend'], backend)
            self.assertEqual(record['events'], os.path.join('Events', run, 'events.lhe.gz'))
            # no separate output file, no stale decompressed copy
            self.assertNotIn('events_reweighted.lhe', os.listdir(os.path.join(events, run)))
            self.assertNotIn('events.lhe', os.listdir(os.path.join(events, run)))

        madtrex = self.weights(os.path.join(events, 'run_01', 'events.lhe.gz'))
        classic = self.weights(os.path.join(events, 'run_02', 'events.lhe.gz'))
        self.assertEqual(len(madtrex), 500)
        self.assertEqual(len(classic), 500)
        for (orig_m, rw_m), (orig_c, rw_c) in zip(madtrex, classic):
            self.assertAlmostEqual(orig_m, orig_c, delta=1e-6 * abs(orig_c))
            # the systematics weights are still there, the new one follows them
            self.assertEqual(list(rw_m)[:109], [str(i) for i in range(1, 110)])
            self.assertEqual(list(rw_m)[-1], 'mt175')
            self.assertIn('1', rw_c)
            self.assertEqual(rw_m['1'], rw_c['1'])
            # the LHE files carry about 7 significant digits
            self.assertAlmostEqual(rw_m['mt175'], rw_c['mt175'], delta=1e-6 * abs(rw_c['mt175']))
        # MT 173 -> 175 changes the weights
        total = sum(w for w, _ in madtrex)
        self.assertNotAlmostEqual(sum(r['mt175'] for _, r in madtrex), total,
                                  delta=1e-3 * abs(total))
        # the shower & co. read unweighted_events.lhe.gz: the same events
        tool_copy = os.path.join(events, 'run_02', 'unweighted_events.lhe.gz')
        self.assertEqual([r['mt175'] for _, r in self.weights(tool_copy)],
                         [r['mt175'] for _, r in classic])

    def test_madtrex_reweights_npy_events_in_place(self):
        import numpy as np
        proc = self.make_process('NPY', 'lhe_npy')
        card = os.path.join(proc, 'Cards', 'reweight_card.dat')
        result = self.run_checked([sys.executable, 'bin/generate_events', '-f',
                                   '--reweight=%s' % card, '--reweight-backend=madtrex'],
                                  cwd=proc)
        self.assertIn('Reweighting with madtrex (--reweight-backend)', result.stdout)
        run = os.path.join(proc, 'Events', 'run_01')
        events = np.load(os.path.join(run, 'events.npy'), mmap_mode='r')
        names = [n for n in events.dtype.names if n.startswith('rwgt_')]
        self.assertEqual(names, ['rwgt_%d' % i for i in range(1, 110)] + ['rwgt_mt175'])
        self.assertFalse(any(n.startswith('.madtrex') for n in os.listdir(run)))
        lhe = self.run_checked([sys.executable, '-c', textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(MG_ROOT)!r})
            from madgraph.iolibs.template_files.mg7 import npy_to_lhe
            print(npy_to_lhe.convert({run!r}, compress=False))
        """)], cwd=proc).stdout.split()[-1]
        converted = self.weights(lhe)
        self.assertEqual(len(converted), 500)
        for (_, weights), value in zip(converted, events['rwgt_mt175']):
            self.assertEqual(list(weights)[-1], 'mt175')
            self.assertAlmostEqual(weights['mt175'], value, delta=1e-6 * abs(value))


if __name__ == '__main__':
    unittest.main()
