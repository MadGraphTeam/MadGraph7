################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Test the pyamplicol worker handling of export_amplicol, with fake workers
instead of pyamplicol."""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import madgraph
import madgraph.iolibs.export_amplicol as export_amplicol


class TestRunWorker(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def fake_worker(self, code):
        path = os.path.join(self.tmp, 'worker.py')
        with open(path, 'w') as stream:
            stream.write('import sys, time\n' + code)
        patcher = mock.patch.object(export_amplicol, 'WORKER', path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_success_prints_the_output(self):
        self.fake_worker('print("generated")\n')
        with mock.patch('sys.stdout', new=io.StringIO()) as out:
            export_amplicol.run_worker('job.json')
        self.assertIn('generated', out.getvalue())

    def test_failure_raises(self):
        self.fake_worker('sys.exit(1)\n')
        with self.assertRaises(madgraph.InvalidCmd):
            export_amplicol.run_worker('job.json')

    def test_busy_symbolica_is_retried_then_given_up(self):
        self.fake_worker('print(%r); sys.exit(1)\n' % export_amplicol.SYMBOLICA_BUSY)
        with mock.patch.object(export_amplicol, 'SYMBOLICA_RETRY', 0), \
                mock.patch.object(export_amplicol, 'SYMBOLICA_MAX_WAIT', 0.3), \
                mock.patch.object(export_amplicol, 'run_worker_once',
                                  wraps=export_amplicol.run_worker_once) as once:
            with self.assertRaisesRegex(madgraph.InvalidCmd, 'another process'):
                export_amplicol.run_worker('job.json')
        self.assertGreater(once.call_count, 1)

    def test_interrupt_kills_the_worker(self):
        self.fake_worker('print("started", flush=True); time.sleep(60)\n')
        workers = []
        popen = subprocess.Popen

        def record(*args, **kwargs):
            workers.append(popen(*args, **kwargs))
            return workers[-1]

        with mock.patch('subprocess.Popen', side_effect=record), \
                mock.patch('sys.stdout.write', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                export_amplicol.run_worker('job.json')
        self.assertIsNotNone(workers[0].returncode)


class TestCheckPyamplicol(unittest.TestCase):

    def test_stale_install_is_an_error(self):
        from madgraph.mg7 import bootstrap
        with mock.patch.object(bootstrap, 'ensure_pyamplicol'), \
                mock.patch.object(bootstrap, 'check_pyamplicol_source_hash',
                                  side_effect=RuntimeError('mismatch')):
            with self.assertRaisesRegex(madgraph.InvalidCmd, 'mismatch'):
                export_amplicol.check_pyamplicol()

    def test_check_script_rejects_another_pyamplicol(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        export_amplicol.write_rusticol_config(
            tmp, {'version': '0.0.0', 'source_hash': 'stale'})
        result = subprocess.run([os.path.join(tmp, 'check-pyamplicol')],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('generated with pyamplicol 0.0.0', result.stderr)


if __name__ == '__main__':
    unittest.main()
