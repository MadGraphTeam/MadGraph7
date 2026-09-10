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
"""`launch` on an mg7 output.

The mg7 branch of :meth:`MadGraphCmd.do_launch` used to shell out to
``bin/generate_events`` and guess, from a hand-written prefix list, which of the
lines following ``launch`` in the user's command file belonged to the run. It
now builds the run interface in process and registers it through
``define_child_cmd_interface``, exactly like a madevent output, so the run reads
MG5's own script and hands back what it does not understand.

The wiring tests here stub the launcher module, so they run without a compiled
madspace; the tests that exercise the real :class:`MG7Cmd` are skipped when
madspace is not installed.
"""

from __future__ import absolute_import

import os
import subprocess
import sys
import tempfile
import types
import unittest

import madgraph.interface.extended_cmd as ext_cmd
import madgraph.interface.master_interface as mgcmd
from madgraph.iolibs.template_files.mg7 import bootstrap as mg7_bootstrap

LAUNCH_MODULE = 'madgraph.iolibs.template_files.mg7.launch'


def make_mg7_dir(root):
    """The minimum that makes find_output_type() answer 'mg7'."""
    os.makedirs(os.path.join(root, 'SubProcesses'))
    os.makedirs(os.path.join(root, 'Cards'))
    with open(os.path.join(root, 'Cards', 'run_card.toml'), 'w') as stream:
        stream.write('[run]\nrun_name = "run"\n')
    return root


class FakeMG7Cmd(ext_cmd.Cmd):
    """Stands in for the real MG7Cmd: records what it was asked to run."""

    def __init__(self, me_dir='.', options=None):
        super(FakeMG7Cmd, self).__init__()
        self.me_dir = me_dir
        # the real MG7Cmd layers these over the output's own configuration;
        # keep the raw argument so the wiring can be checked
        self.raw_options = options
        self.options = options or {}
        self.commands = []
        self.raise_on_run = None

    def do_generate_events(self, line):
        self.commands.append(line)
        if self.raise_on_run is not None:
            raise self.raise_on_run

    def do_quit(self, line):
        self.commands.append('quit')
        return True


class MG7LaunchWiringTest(unittest.TestCase):
    """do_launch's mg7 branch: what it builds and what it hands the child."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.me_dir = make_mg7_dir(os.path.join(self.tmpdir.name, 'PROC'))

        self.cmd = mgcmd.MasterCmd()
        self.cmd.use_rawinput = False

        # MG5's error handling writes a 'debug' file into the working
        # directory; keep that (and anything else a test provokes) inside the
        # temporary directory instead of the checkout.
        self._saved_cwd = os.getcwd()
        os.chdir(self.tmpdir.name)

        # Stub the launcher module: importing the real one needs madspace.
        self.built = []
        stub = types.ModuleType(LAUNCH_MODULE)

        def MG7Cmd(me_dir='.', options=None):
            child = FakeMG7Cmd(me_dir, options)
            self.built.append(child)
            return child

        stub.MG7Cmd = MG7Cmd
        self.stub = stub
        self._saved_module = sys.modules.get(LAUNCH_MODULE)
        sys.modules[LAUNCH_MODULE] = stub
        # `from pkg import launch` resolves through the parent package's
        # attribute when the real module has already been imported (which
        # MG7CmdTest does), so sys.modules alone is not enough to stub it.
        import madgraph.iolibs.template_files.mg7 as mg7_pkg
        self.mg7_pkg = mg7_pkg
        self._saved_attr = getattr(mg7_pkg, 'launch', None)
        mg7_pkg.launch = stub

        # Never run the real madspace bootstrap from a unit test.
        self.bootstrap_calls = []
        self._saved_ensure = mg7_bootstrap.ensure_madspace
        mg7_bootstrap.ensure_madspace = \
            lambda interactive=None: self.bootstrap_calls.append(interactive)

    def tearDown(self):
        mg7_bootstrap.ensure_madspace = self._saved_ensure
        if self._saved_module is None:
            sys.modules.pop(LAUNCH_MODULE, None)
        else:
            sys.modules[LAUNCH_MODULE] = self._saved_module
        if self._saved_attr is None:
            del self.mg7_pkg.launch
        else:
            self.mg7_pkg.launch = self._saved_attr
        os.chdir(self._saved_cwd)
        self.tmpdir.cleanup()

    def launch(self, line, script_lines=()):
        """Run `launch <line>` with ``script_lines`` following it in the file."""
        self.cmd.inputfile = iter(list(script_lines))
        self.cmd.do_launch('%s %s' % (self.me_dir, line) if line else self.me_dir)
        return self.built[-1] if self.built else None

    # -- the bug this change fixes ------------------------------------------
    def test_set_line_is_left_for_the_question_not_scavenged(self):
        """The lines after `launch` must stay in MG5's inputfile.

        The subprocess branch consumed every following line that did not start
        with one of a fixed list of MG5 keywords -- `set` deliberately among the
        consumed ones -- and piped them to the child's stdin. An MG5-level
        `set gauge Feynman` after `launch` was therefore swallowed with no
        message from either side. In process the child shares MG5's inputfile,
        so nothing is consumed up front.
        """
        remaining = ['set nevents 500\n', 'set gauge Feynman\n']
        self.launch('', remaining)
        # the child was handed the *same* iterator, still holding both lines
        child = self.built[-1]
        self.assertIs(child.inputfile, self.cmd.inputfile)
        self.assertEqual([l.strip() for l in child.inputfile],
                         ['set nevents 500', 'set gauge Feynman'])

    def test_child_is_registered_with_mg5(self):
        """define_child_cmd_interface is what gives the child MG5's script."""
        self.launch('')
        child = self.built[-1]
        self.assertIs(child.mother, self.cmd)
        self.assertIs(self.cmd.child, child)

    def test_child_runs_generate_events_then_quits(self):
        child = self.launch('')
        # do_generate_events is handed the arguments, so a bare launch is ''
        self.assertEqual(child.commands, ['', 'quit'])

    def test_me_dir_and_options_are_passed(self):
        child = self.launch('')
        # check_launch normalises the path with realpath
        self.assertEqual(child.me_dir, os.path.realpath(self.me_dir))
        self.assertIs(child.raw_options, self.cmd.options)

    # -- launch options, all of which the subprocess branch dropped ---------
    def test_force_flag_is_forwarded(self):
        child = self.launch('-f')
        self.assertEqual(child.commands[0], '-f')

    def test_run_name_is_forwarded(self):
        child = self.launch('-n myrun')
        self.assertEqual(child.commands[0], '--name=myrun')

    def test_laststep_is_forwarded(self):
        child = self.launch('--laststep=parton')
        self.assertEqual(child.commands[0], '--laststep=parton')

    def test_options_combine(self):
        child = self.launch('-f -n myrun --laststep=parton')
        self.assertEqual(child.commands[0],
                         '-f --name=myrun --laststep=parton')

    def test_interactive_hands_over_the_prompt(self):
        """`launch -i` must give the user the child's own command loop."""
        self.cmd.inputfile = iter([])
        self.cmd.do_launch('%s -i' % self.me_dir)
        child = self.built[-1]
        self.assertIs(self.cmd.child, child)
        # not driven with a canned command: the user drives it
        self.assertEqual(child.commands, [])

    # -- interrupts ---------------------------------------------------------
    def test_keyboard_interrupt_reaches_the_error_handling(self):
        """Ctrl-C used to be swallowed by `except KeyboardInterrupt: pass`.

        Going through the child's run_cmd means it now reaches the standard
        cmd error handling: stop_on_keyboard_stop runs and the interrupt then
        stops MG5, like every other interface, rather than being dropped so the
        rest of the script runs on as if nothing happened.
        """
        stopped = []

        def MG7Cmd(me_dir='.', options=None):
            child = FakeMG7Cmd(me_dir, options)
            child.raise_on_run = KeyboardInterrupt()
            child.stop_on_keyboard_stop = lambda: stopped.append(True)
            self.built.append(child)
            return child

        self.stub.MG7Cmd = MG7Cmd
        self.cmd.inputfile = iter([])
        self.assertRaises(SystemExit, self.cmd.do_launch, self.me_dir)
        self.assertEqual(stopped, [True])

    # -- the environment the subprocess branch used to build ---------------
    def test_environment_is_exported_in_process(self):
        """MADGRAPH_HEPTOOLS_DIR / LHAPDF_DATA_PATH / MADGRAPH_LHAPDF_CONFIG
        used to be put in the subprocess env; in process they have to reach
        os.environ, which is the only channel the tools the run shells out to
        (the madspace build, the real LHAPDF library) can read."""
        saved = {k: os.environ.get(k) for k in
                 ('MADGRAPH_HEPTOOLS_DIR', 'LHAPDF_DATA_PATH',
                  'MADGRAPH_LHAPDF_CONFIG')}
        for key in saved:
            os.environ.pop(key, None)
        try:
            self.cmd.options['heptools_install_dir'] = self.tmpdir.name
            self.cmd.setup_mg7_environment()
            self.assertEqual(os.environ.get('MADGRAPH_HEPTOOLS_DIR'),
                             os.path.abspath(self.tmpdir.name))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_bootstrap_is_told_the_interactivity_explicitly(self):
        """The bootstrap must not guess from sys.argv/sys.stdin in process."""
        self.launch('')
        self.assertEqual(self.bootstrap_calls, [False])

        self.bootstrap_calls[:] = []
        self.cmd.use_rawinput = True
        self.cmd.inputfile = iter([])
        self.cmd.do_launch(self.me_dir)
        self.assertEqual(self.bootstrap_calls, [True])


class MG7BootstrapTest(unittest.TestCase):
    """The one-off madspace install that used to be a launch.py import side
    effect."""

    def setUp(self):
        self.saved = (mg7_bootstrap.INSTALL_DIR, mg7_bootstrap.MADSPACE_DIR,
                      subprocess.run)
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        (mg7_bootstrap.INSTALL_DIR, mg7_bootstrap.MADSPACE_DIR,
         subprocess.run) = self.saved
        sys.path[:] = [p for p in sys.path if self.tmpdir.name not in p]
        self.tmpdir.cleanup()

    def point_at_empty_install(self):
        from pathlib import Path
        root = Path(self.tmpdir.name)
        mg7_bootstrap.MADSPACE_DIR = root / 'madspace'
        mg7_bootstrap.INSTALL_DIR = root / 'madspace' / 'install'
        return root

    def record_runs(self):
        calls = []

        def fake_run(cmd, **opts):
            calls.append((cmd, opts))
            return types.SimpleNamespace(returncode=0)

        subprocess.run = fake_run
        return calls

    @staticmethod
    def quiet():
        """The bootstrap narrates the install on stdout; keep it out of the
        test output."""
        import contextlib
        import io
        return contextlib.redirect_stdout(io.StringIO())

    def test_no_install_when_already_present(self):
        root = self.point_at_empty_install()
        (root / 'madspace' / 'install' / 'madspace').mkdir(parents=True)
        calls = self.record_runs()
        with self.quiet():
            mg7_bootstrap.ensure_madspace(interactive=False)
        self.assertEqual(calls, [])
        self.assertIn(str(mg7_bootstrap.INSTALL_DIR), sys.path)

    def test_non_interactive_install_is_unattended(self):
        self.point_at_empty_install()
        calls = self.record_runs()
        with self.quiet():
            mg7_bootstrap.ensure_madspace(interactive=False)
        self.assertEqual(len(calls), 1)
        cmd, opts = calls[0]
        self.assertIn('--source', cmd)
        self.assertIn('--yes', cmd)
        # must not eat the caller's stdin, which carries the run's script
        self.assertEqual(opts['stdin'], subprocess.DEVNULL)

    def test_interactive_install_keeps_the_terminal(self):
        self.point_at_empty_install()
        calls = self.record_runs()
        with self.quiet():
            mg7_bootstrap.ensure_madspace(interactive=True)
        cmd, opts = calls[0]
        self.assertNotIn('--yes', cmd)
        self.assertIsNone(opts['stdin'])

    def test_explicit_interactivity_ignores_sys_argv(self):
        """In process, sys.argv is MG5's command line, not the run's: a `-f`
        meant for `mg5_aMC -f` must not be read as this run's force flag."""
        self.point_at_empty_install()
        calls = self.record_runs()
        saved_argv = sys.argv
        sys.argv = ['mg5_aMC', '-f']
        try:
            with self.quiet():
                mg7_bootstrap.ensure_madspace(interactive=True)
        finally:
            sys.argv = saved_argv
        cmd, opts = calls[0]
        self.assertNotIn('--yes', cmd)

    def test_drop_install_path_leaves_no_shadowing_entry(self):
        """The install dir also holds madspace's build dependencies (yaml,
        packaging, pathspec, ...), so leaving it on sys.path would shadow the
        caller's copies for the rest of what is now MG5's own session."""
        root = self.point_at_empty_install()
        (root / 'madspace' / 'install' / 'madspace').mkdir(parents=True)
        calls = self.record_runs()
        mg7_bootstrap.ensure_madspace(interactive=False)
        self.assertIn(str(mg7_bootstrap.INSTALL_DIR), sys.path)
        mg7_bootstrap.drop_install_path()
        self.assertNotIn(str(mg7_bootstrap.INSTALL_DIR), sys.path)
        self.assertEqual(calls, [])

    def test_failure_is_reported(self):
        self.point_at_empty_install()

        def failing_run(cmd, **opts):
            return types.SimpleNamespace(returncode=1)

        subprocess.run = failing_run
        with self.quiet():
            self.assertRaises(RuntimeError,
                              mg7_bootstrap.ensure_madspace, interactive=False)


@unittest.skipUnless(mg7_bootstrap.madspace_is_installed(),
                     'madspace is not installed')
class MG7CmdTest(unittest.TestCase):
    """The real run interface (needs a compiled madspace to import)."""

    def setUp(self):
        from madgraph.iolibs.template_files.mg7 import launch as mg7_launch
        self.launch = mg7_launch
        self.tmpdir = tempfile.TemporaryDirectory()
        self.me_dir = make_mg7_dir(os.path.join(self.tmpdir.name, 'PROC'))

    def tearDown(self):
        self.tmpdir.cleanup()

    def make_cmd(self):
        return self.launch.MG7Cmd(me_dir=self.me_dir)

    def test_option_parsing(self):
        cmd = self.make_cmd()
        self.assertEqual(cmd._parse_run_options('-f'),
                         {'force': True, 'name': '', 'laststep': ''})
        self.assertEqual(cmd._parse_run_options('--name=abc')['name'], 'abc')
        self.assertEqual(cmd._parse_run_options('--laststep=parton')['laststep'],
                         'parton')
        self.assertEqual(
            cmd._parse_run_options('-f --name=abc --laststep=parton'),
            {'force': True, 'name': 'abc', 'laststep': 'parton'})
        # space-separated forms, as typed at the MG7> prompt
        self.assertEqual(cmd._parse_run_options('-n abc'),
                         {'force': False, 'name': 'abc', 'laststep': ''})
        self.assertEqual(cmd._parse_run_options('-s parton')['laststep'],
                         'parton')
        # a bare word is the run name
        self.assertEqual(cmd._parse_run_options('myrun')['name'], 'myrun')

    def test_laststep_parton_switches_every_tool_off(self):
        cmd = self.make_cmd()
        switch = {'shower': 'Pythia8', 'madspin': 'ON', 'reweight': 'OFF'}
        out = cmd._apply_laststep(switch, {'laststep': 'parton'})
        self.assertEqual(out, {'shower': 'OFF', 'madspin': 'OFF',
                               'reweight': 'OFF'})

    def test_laststep_empty_leaves_the_switch_alone(self):
        cmd = self.make_cmd()
        switch = {'shower': 'Pythia8'}
        self.assertEqual(cmd._apply_laststep(switch, {'laststep': ''}), switch)

    def test_run_name_is_written_to_the_run_card(self):
        cmd = self.make_cmd()
        cwd = os.getcwd()
        os.chdir(self.me_dir)
        try:
            cmd._set_run_name('myrun')
            with open(os.path.join('Cards', 'run_card.toml')) as stream:
                self.assertIn('myrun', stream.read())
        finally:
            os.chdir(cwd)

    def test_options_come_from_the_output_dir_not_the_cwd(self):
        """Cards/me5_configuration.txt of the *output* has the last word.

        bin/generate_events chdirs into the output first, so reading the config
        relative to the working directory was right there. In process MG5 has
        not moved when the interface is built, so the directory has to be
        passed explicitly or the output's own configuration is ignored.
        """
        with open(os.path.join(self.me_dir, 'Cards',
                               'me5_configuration.txt'), 'w') as stream:
            stream.write('delphes_path = /somewhere/delphes\n')
        cmd = self.make_cmd()
        self.assertNotEqual(os.getcwd(), self.me_dir)
        self.assertEqual(cmd.options['delphes_path'], '/somewhere/delphes')

    def test_status_file_path_is_absolute(self):
        """The status file must not depend on the working directory.

        ms.StatusFile finishes its write from a C++ destructor, which runs when
        Python collects the process object -- for an interrupted run, that is
        after the launch command has restored the working directory. A relative
        path there throws a filesystem_error out of a destructor, i.e. aborts
        the whole of MG5.
        """
        recorded = []
        ms = self.launch.ms
        saved = ms.StatusFile
        ms.StatusFile = lambda path: recorded.append(path)
        cwd = os.getcwd()
        os.chdir(self.me_dir)
        try:
            process = self.launch.MadgraphProcess.__new__(
                self.launch.MadgraphProcess)
            process.run_card = {"run": {"run_name": "run"}}
            process.init_event_dir()
        finally:
            os.chdir(cwd)
            ms.StatusFile = saved
        self.assertEqual(len(recorded), 1)
        self.assertTrue(os.path.isabs(recorded[0]), recorded[0])

    def test_setup_logging_defers_to_an_existing_handler(self):
        """In process MG5 owns the loggers; adding a second handler would print
        every line twice."""
        import logging
        lg = logging.getLogger('madgraph')
        handler = logging.NullHandler()
        lg.addHandler(handler)
        saved = self.launch._TOOL_LOGGING_READY
        self.launch._TOOL_LOGGING_READY = False
        try:
            before = list(lg.handlers)
            self.launch._setup_logging()
            self.assertEqual(list(lg.handlers), before)
        finally:
            self.launch._TOOL_LOGGING_READY = saved
            lg.removeHandler(handler)


if __name__ == '__main__':
    unittest.main()
