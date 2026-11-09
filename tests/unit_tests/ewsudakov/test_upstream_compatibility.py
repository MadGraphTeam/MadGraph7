"""Upstream behaviour kept by the EW Sudakov changes outside the reweighting."""

import ast
import io
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from madgraph import MG5DIR
from madgraph.various import misc
from madgraph.interface import common_run_interface
from madgraph.iolibs.export_fks import ProcessExporterFortranFKS


class TestAlphaScheme(unittest.TestCase):
    def write_conversion(self, model, tagged=True):
        matrix_element = SimpleNamespace(
            born_me={
                "processes": [
                    {
                        "model": model,
                        "legs": [{"id": 22, "is_tagged": tagged}],
                    }
                ]
            }
        )
        output = io.StringIO()
        value = ProcessExporterFortranFKS.write_a0gmuconv_file(
            None, output, matrix_element
        )
        return value, output.getvalue()

    def test_alpha0_metadata_does_not_depend_on_model_directory(self):
        value, output = self.write_conversion(
            {"modelpath": "/models/renamed_model", "startfromalpha0": True}
        )
        self.assertTrue(value)
        self.assertIn("startfroma0=.true.", output)

    def test_directory_name_does_not_override_metadata(self):
        value, output = self.write_conversion(
            {"modelpath": "/models/loop_qcd_qed_sm_a0", "startfromalpha0": False}
        )
        self.assertFalse(value)
        self.assertIn("startfroma0=.false.", output)

    def test_untagged_photon_does_not_enable_conversion(self):
        value, _ = self.write_conversion({"startfromalpha0": True}, tagged=False)
        self.assertFalse(value)


class TestCollierRuntimePath(unittest.TestCase):
    """The collier rpath belongs to make_opts, which links the madfks executables."""

    exporter = SimpleNamespace(mgme_dir=MG5DIR)
    collier = ["-L/opt/collier/lib -lcollier"]

    def test_make_opts_carries_collier_rpath(self):
        text = ProcessExporterFortranFKS.write_make_opts(
            self.exporter, None, self.collier, []
        )
        line = [row for row in text.splitlines() if "-lcollier" in row]
        self.assertEqual(len(line), 1)
        self.assertTrue(line[0].strip().startswith("libcuttools=-lcts"))
        self.assertIn("-Wl,-rpath,/opt/collier/lib", line[0])

    def test_make_opts_without_collier_has_no_rpath(self):
        text = ProcessExporterFortranFKS.write_make_opts(
            self.exporter, None, ["-L/opt/ninja/lib -lninja"], []
        )
        self.assertNotIn("-Wl,-rpath", text)

    def test_loop_makefile_is_not_given_the_rpath(self):
        text = ProcessExporterFortranFKS.write_makefile_TIR(
            self.exporter, None, self.collier, [], []
        )
        self.assertIn("-lcollier", text)
        self.assertNotIn("-Wl,-rpath,/opt/collier/lib", text)


class TestFirstAddProcessToGenerate(unittest.TestCase):
    @staticmethod
    def convert(commands):
        from madgraph.interface.reweight_interface import first_add_process_to_generate

        return first_add_process_to_generate(commands)

    def test_definitions_before_the_first_process(self):
        commands = [
            "define pert_QCD = g u d; add process p p > z j pert_QCD;",
            "add process p p > z;",
        ]
        self.assertEqual(
            self.convert(commands),
            [
                "define pert_QCD = g u d; generate p p > z j pert_QCD;",
                "add process p p > z;",
            ],
        )

    def test_only_the_first_occurrence_is_converted(self):
        commands = [
            "add process p p > z; add process p p > z j;",
            "add process p p > w;",
        ]
        self.assertEqual(
            self.convert(commands),
            ["generate p p > z; add process p p > z j;", "add process p p > w;"],
        )

    def test_commands_without_process_are_kept(self):
        commands = ["set group_subprocesses False", "add process p p > z;"]
        self.assertEqual(
            self.convert(commands),
            ["set group_subprocesses False", "generate p p > z;"],
        )
        self.assertEqual(
            self.convert(["output standalone_rw x"]), ["output standalone_rw x"]
        )
        self.assertEqual(self.convert([]), [])

    def test_input_is_not_modified(self):
        commands = ["add process p p > z;"]
        self.convert(commands)
        self.assertEqual(commands, ["add process p p > z;"])


class TestFxFxSudakovReweightOptions(unittest.TestCase):
    def options(self, card, ickkw=3):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reweight_card.dat")
            if card is not None:
                with open(path, "w") as fsock:
                    fsock.write(card)
            return common_run_interface.fxfx_sudakov_reweight_options(
                {"ickkw": ickkw}, path
            )

    def test_active_line_allows_madspin_input(self):
        for card in (
            "change include_sudakov True\nlaunch\n",
            "  change   include_sudakov True\n",
            "change include_sudakov True # EW logs\n",
            "change rwgt_dir ./rw\nchange include_sudakov True",
            "change rwgt_dir ./rw; change include_sudakov True\n",
            "change include_sudakov True;launch\n",
        ):
            with self.subTest(card=card):
                self.assertEqual(self.options(card), {"allow_madspin": True})

    def test_inactive_or_rejected_values_keep_the_madspin_guard(self):
        for card in (
            "#change include_sudakov True\nlaunch\n",
            "launch # change include_sudakov True\n",
            "launch # see notes; change include_sudakov True\n",
            "change include_sudakov False\n# true\n",
            "change include_sudakov true\n",
            "change include_sudakov Truex\n",
            "set include_sudakov True\n",
            "",
        ):
            with self.subTest(card=card):
                self.assertEqual(self.options(card), {})

    def test_only_fxfx_runs(self):
        self.assertEqual(self.options("change include_sudakov True\n", ickkw=0), {})
        self.assertEqual(self.options("change include_sudakov True\n", ickkw=1), {})

    def test_missing_card_or_run_card(self):
        self.assertEqual(self.options(None), {})
        self.assertEqual(
            common_run_interface.fxfx_sudakov_reweight_options(None, "/nonexistent"), {}
        )
        self.assertEqual(
            common_run_interface.fxfx_sudakov_reweight_options({}, "/nonexistent"), {}
        )


class TestLaunchModelReload(unittest.TestCase):
    """do_launch reloads the model only for EW Sudakov reweighting."""

    class Stop(Exception):
        pass

    def launch(self, inc_sudakov):
        from madgraph.interface import reweight_interface

        interface = reweight_interface.ReweightInterface.__new__(
            reweight_interface.ReweightInterface
        )
        interface.exitted = True  # nothing to clean up in __del__
        interface.inc_sudakov = inc_sudakov
        interface.has_standalone_dir = False
        interface.mg5cmd = None
        interface.options = {}
        interface.banner = mock.Mock()
        interface.banner.get.side_effect = lambda card, key: {
            "full_model_line": "import model sm",
            "model": "loop_qcd_qed_sm_Gmu_forSudakov",
        }[key]
        with mock.patch.object(
            interface,
            "check_launch",
            return_value={"rwgt_name": None, "rwgt_info": None},
        ), mock.patch.object(interface, "load_model") as load_model, mock.patch.object(
            interface, "_ensure_fxfx_cluster_catalogue"
        ), mock.patch.object(
            interface, "_collect_sudakov_weight_sources"
        ), mock.patch.object(interface, "setup_f2py_interface", side_effect=self.Stop):
            with self.assertRaises(self.Stop):
                interface.do_launch("")
        return load_model

    def test_sudakov_launch_loads_the_model(self):
        self.launch(True).assert_called_once_with(
            "loop_qcd_qed_sm_Gmu_forSudakov", True, False
        )

    def test_other_launches_keep_their_model(self):
        self.launch(False).assert_not_called()


class TestReweightSetupPlugin(unittest.TestCase):
    """The no-argument setup hook and explicit launch names remain compatible."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def interface(self, cls=None, sudakov=False):
        from madgraph.interface.reweight_interface import ReweightInterface

        cls = cls or ReweightInterface
        interface = cls.__new__(cls)
        interface.exitted = True
        interface.inc_sudakov = sudakov
        interface.has_standalone_dir = False
        interface.multicore = False
        interface.me_dir = str(self.root)
        interface.rwgt_dir = None
        interface.mg5cmd = None
        interface.lhe_input = True
        interface.options = {'rwgt_name': None, 'rwgt_info': None}
        interface.banner = SimpleNamespace(get=lambda *args: 'sm')
        interface.rwgt_mode = 'LO'
        interface._ensure_fxfx_cluster_catalogue = mock.Mock()
        interface._collect_sudakov_weight_sources = mock.Mock()
        interface._prepare_sudakov_weight_ids = mock.Mock(return_value=['2001'])
        interface.load_model = mock.Mock()
        interface.load_module = mock.Mock()
        interface.load_density_modules = mock.Mock()
        interface.handle_param_card = mock.Mock(
            side_effect=lambda *args: ([], interface.options['rwgt_name']))
        interface.launch_actual_reweighting = mock.Mock(return_value='completed')
        return interface

    def plugin(self, stop):
        from madgraph.interface.reweight_interface import ReweightInterface

        class Plugin(ReweightInterface):
            def setup_f2py_interface(self):
                self.name_seen_by_plugin = self.options['rwgt_name']
                return stop

        return self.interface(Plugin)

    def test_original_plugin_hook_can_stop_the_launch(self):
        interface = self.plugin(stop=True)
        self.assertIsNone(interface.do_launch('--rwgt_name=requested'))
        self.assertEqual(interface.name_seen_by_plugin, 'requested')
        interface.handle_param_card.assert_not_called()
        interface.launch_actual_reweighting.assert_not_called()

    def test_original_plugin_hook_can_continue_the_launch(self):
        interface = self.plugin(stop=False)
        self.assertEqual(interface.do_launch('--rwgt_name=requested'), 'completed')
        self.assertEqual(interface.name_seen_by_plugin, 'requested')
        self.assertEqual(interface.launch_actual_reweighting.call_args.args[1], 'requested')

    def write_cached_settings(self):
        from madgraph.iolibs import save_load_object

        directory = self.root / 'rw_me'
        directory.mkdir(exist_ok=True)
        save_load_object.save_to_file(str(directory / 'rwgt.pkl'), {
            'id_to_path': {}, 'id_to_path_second': {}, 'all_cross_section': {},
            'processes': [], 'second_process': None, 'second_model': None,
            'merged_map': {(21, 21): 21}, 'merged_particles': None,
            'rwgt_dir': str(self.root), 'has_nlo': False, 'rwgt_mode': 'LO',
            'rwgt_name': 'cached_name', 'allow_missing_finalstate': False,
            'identical_particle_in_prod_and_decay': 'average', 'nb_library': 1,
            'sudakov_source_ids': {'1001'},
        })

    def test_explicit_name_survives_cached_setup_and_worker_setup(self):
        self.write_cached_settings()
        for sudakov in (False, True):
            for worker in (False, True):
                with self.subTest(sudakov=sudakov, worker=worker):
                    interface = self.interface(sudakov=sudakov)
                    interface._sudakov_source_ids = {'1001'}
                    if worker:
                        interface.multicore = 'wait'
                    else:
                        interface.rwgt_dir = str(self.root)
                    self.assertEqual(interface.do_launch(
                        '--rwgt_name=requested --rwgt_info=description'), 'completed')
                    self.assertTrue(interface.has_standalone_dir)
                    self.assertEqual(interface.options['rwgt_name'], 'requested')
                    self.assertEqual(interface.options['rwgt_info'], 'description')
                    self.assertEqual(interface.launch_actual_reweighting.call_args.args[1], 'requested')
                    interface.load_module.assert_called_once_with()

    def test_cached_name_is_not_reused_without_an_explicit_name(self):
        self.write_cached_settings()
        interface = self.interface()
        interface.rwgt_dir = str(self.root)
        self.assertEqual(interface.do_launch(''), 'completed')
        self.assertIsNone(interface.options['rwgt_name'])
        self.assertIsNone(interface.launch_actual_reweighting.call_args.args[1])


class TestClusteringBuffers(unittest.TestCase):
    """The <clustering> line of a long history fits the Fortran buffers."""

    subprocesses = Path(MG5DIR) / "Template/NLO/SubProcesses"

    @staticmethod
    def worst_case_line(nsteps, nlegs):
        # Two-digit leg labels, step k merges k+1 legs (at most nlegs);
        # ES20.10 scales; 7-digit PDG.
        steps = " ".join(
            'step_%d="%s;%s;mom=-1000021"'
            % (
                k,
                ",".join(str(10 + i) for i in range(min(k + 1, nlegs))),
                "-1.2345678901E+03",
            )
            for k in range(1, nsteps + 1)
        )
        return (
            '<clustering type="real" iproc="512" %s final_scale="1.2345678901E+03">'
            "</clustering>" % steps
        )

    def declared_length(self, filename, pattern, routine=None):
        text = (self.subprocesses / filename).read_text()
        if routine is not None:
            # body of 'subroutine <routine>(' up to the next subroutine/function
            start = re.search(r"^\s+subroutine %s\(" % routine, text, re.I | re.M)
            self.assertIsNotNone(start, routine)
            end = re.search(
                r"^\s+(subroutine|\w+\s+function)\s", text[start.end() :], re.I | re.M
            )
            text = text[
                start.start() : start.end() + (end.start() if end else len(text))
            ]
        return [int(n) for n in re.findall(pattern, text, re.I | re.M)]

    def test_writer_and_reader_buffers_hold_twenty_steps(self):
        line = self.worst_case_line(20, 14)
        pairstr = self.declared_length("cluster.f", r"^\s+CHARACTER\*(\d+) pairstr$")
        self.assertEqual(len(pairstr), 1)
        self.assertLessEqual(len(",".join(str(10 + i) for i in range(14))), pairstr[0])
        temp = self.declared_length("cluster.f", r"^\s+CHARACTER\*(\d+) temp$")
        clusterstrings = self.declared_length(
            "cluster.f", r"^\s+CHARACTER\*\((\d+)\) clusterstrings\(0:MAXFKSLOC\)$"
        )
        readers = [
            self.declared_length(
                "handling_lhe_events.f", r"^\s+character\*(\d+) string$", routine
            )
            for routine in ("read_lhef_event", "read_lhef_event_catch")
        ]
        self.assertEqual(len(temp), 1)
        self.assertEqual(len(clusterstrings), 1)
        self.assertEqual([len(r) for r in readers], [1, 1])
        readers = [r[0] for r in readers]
        self.assertLess(len(line), temp[0])
        self.assertLess(len(line), clusterstrings[0])
        for length in readers:
            self.assertLess(len(line), length)


class TestCallSites(unittest.TestCase):
    """The callers use the helpers above, so the tested behaviour is the one run."""

    @staticmethod
    def tree(module):
        return ast.parse((Path(MG5DIR) / module).read_text())

    @staticmethod
    def function(tree, name):
        found = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        assert len(found) == 1, name
        return found[0]

    @staticmethod
    def calls(node, name):
        return [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and getattr(call.func, "id", getattr(call.func, "attr", None)) == name
        ]

    def test_reweight_interface_options_come_from_the_card_check(self):
        do_reweight = self.function(
            self.tree("madgraph/interface/common_run_interface.py"), "do_reweight"
        )
        self.assertEqual(
            len(self.calls(do_reweight, "fxfx_sudakov_reweight_options")), 1
        )
        (call,) = self.calls(do_reweight, "rwgt_interface")
        names = [kw.arg for kw in call.keywords]
        self.assertNotIn("allow_madspin", names)
        self.assertIn(None, names)  # **options

    def test_generation_commands_use_the_helper(self):
        tree = self.tree("madgraph/interface/reweight_interface.py")
        self.assertEqual(len(self.calls(tree, "first_add_process_to_generate")), 3)
        # the single-prefix test that skipped 'define ...; add process' commands
        for call in self.calls(tree, "startswith"):
            for arg in call.args:
                try:
                    value = ast.literal_eval(arg)
                except ValueError:
                    continue
                self.assertNotEqual(value, "add process")


class TestOptionalNumpy(unittest.TestCase):
    """Modules importing the reweight interface do not need numpy."""

    blocker = (
        "import importlib, importlib.abc, sys\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name == 'numpy' or name.startswith('numpy.'):\n"
        "            raise ImportError('numpy unavailable')\n"
        "sys.meta_path.insert(0, Block())\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "for name in sys.argv[2:]:\n"
        "    importlib.import_module(name)\n"
    )

    def test_import_without_numpy(self):
        import subprocess

        modules = [
            "madgraph.various.fxfx_ewsudakov",
            "madgraph.interface.reweight_interface",
        ]
        if (Path(MG5DIR) / "MadSpin" / "decay.py").exists():
            modules.append("MadSpin.decay")
        result = subprocess.run(
            [sys.executable, "-c", self.blocker, MG5DIR] + modules,
            cwd=MG5DIR,
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_discovery_without_numpy(self):
        """Discovery imports every module, including unselected NumPy tests."""
        import subprocess

        script = self.blocker + (
            "import unittest\n"
            "from tests.test_manager import TestFinder\n"
            "finder = TestFinder(package='tests/unit_tests', expression=[\n"
            "    'test_upper_triangle_is_not_conjugated',\n"
            "    'test_pure_states_give_squared_amplitude',\n"
            "    'test_complete_input_covers_all_channels_and_xis',\n"
            "    'test_scalar_fxfx_central_factor_controls_all_variants',\n"
            "    'test_launch_reset_allows_event_ids_to_restart',\n"
            "    'test_source_scan_keeps_input_position',\n"
            "    'test_processes_load_distinct_native_modules',\n"
            "    'test_numeric_allocation_is_deterministic',\n"
            "    'test_directory_name_does_not_override_metadata',\n"
            "    'test_no_event_is_lost_after_leaving_the_loop',\n"
            "])\n"
            "names = list(finder)\n"
            "assert len(names) == 10, names\n"
            "suite = unittest.defaultTestLoader.loadTestsFromNames(names)\n"
            "result = unittest.TestResult()\n"
            "suite.run(result)\n"
            "assert result.wasSuccessful(), (result.errors, result.failures)\n"
            "assert result.testsRun == 10, result.testsRun\n"
            "assert len(result.skipped) == 8, result.skipped\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, MG5DIR],
            cwd=MG5DIR,
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])


class TestSudakovF2pyBuild(unittest.TestCase):
    """The ewsudpy f2py module builds with the meson backend (Python >= 3.12).

    meson compiles in its own build directory and resolves the
    sub_f2py_ewsudakov.f link to SubProcesses/, so the process directory, which
    holds the included files, must be passed to f2py explicitly.
    """

    subprocesses = Path(MG5DIR) / "Template/NLO/SubProcesses"

    def test_process_directory_is_an_include_path(self):
        recipe = (self.subprocesses / "makefile_fks_dir").read_text()
        lines = [
            row
            for row in recipe.splitlines()
            if "sub_f2py_ewsudakov.f" in row and "$(F2PY)" in row
        ]
        self.assertEqual(len(lines), 1)
        command = lines[0].split("$(F2PY)", 1)[1].split()
        self.assertIn("-I$(HERE)", command)
        self.assertLess(
            command.index("-I$(HERE)"), command.index("sub_f2py_ewsudakov.f")
        )

    def test_includes_are_process_directory_files(self):
        source = (self.subprocesses / "sub_f2py_ewsudakov.f").read_text()
        includes = set(re.findall(r"^\s+include\s+'([^']+)'", source, re.I | re.M))
        self.assertTrue(includes)
        # files written into each P* directory of an EW Sudakov output
        self.assertLessEqual(includes, {"nexternal.inc", "coupl.inc", "orders.inc"})


class TestSudakovDispatcherImport(unittest.TestCase):
    def setUp(self):
        try:
            import numpy
        except ImportError:
            self.skipTest("NumPy is required for the generated Sudakov dispatcher")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ("P0_uux_zz", "P0_ddx_zz"):
            (self.root / "SubProcesses" / name).mkdir(parents=True)
        template = (
            Path(MG5DIR) / "madgraph/iolibs/template_files/ewsudakov_pydispatcher.inc"
        ).read_text()
        self.namespace = {
            "__file__": str(self.root / "bin/internal/ewsud_pydispatcher.py")
        }
        old_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), old_path))
        exec(  # noqa: S102 - execute the generated dispatcher
            compile(
                misc.apply_template(
                    template,
                    {"path": "", "pdir_list": "", "pdg2sud": "", "pdgsorted": ""},
                ),
                self.namespace["__file__"],
                "exec",
            ),
            self.namespace,
        )

    def test_processes_load_distinct_native_modules(self):
        modules = {}

        def load(name):
            process = Path.cwd().name
            self.assertEqual(name, "ewsudpy_" + process)
            self.assertEqual(Path(sys.path[0]), Path.cwd())
            return modules.setdefault(name, object())

        before_cwd, before_path = os.getcwd(), sys.path[:]
        with mock.patch("importlib.import_module", side_effect=load):
            first = self.namespace["import_lib"]("P0_uux_zz")
            second = self.namespace["import_lib"]("P0_ddx_zz")
        self.assertIsNot(first, second)
        self.assertEqual(os.getcwd(), before_cwd)
        self.assertEqual(sys.path, before_path)

    def test_failed_import_restores_working_directory_and_search_path(self):
        before_cwd, before_path = os.getcwd(), sys.path[:]
        with mock.patch("importlib.import_module", side_effect=ImportError):
            with self.assertRaises(ImportError):
                self.namespace["import_lib"]("P0_uux_zz")
        self.assertEqual(os.getcwd(), before_cwd)
        self.assertEqual(sys.path, before_path)
