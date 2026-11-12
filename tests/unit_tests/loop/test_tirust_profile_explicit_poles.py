"""Exercise the actual stock profile methods without compilers or reducers."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from madgraph.interface.madevent_interface import MadLoopInitializer
from madgraph.various import banner, process_checks

ROOT = Path(__file__).resolve().parents[3] if 'tests' in Path(__file__).parts else Path.cwd()
SOURCE = ROOT/'madgraph/various/process_checks.py'
HELPER = ROOT/'tests/unit_tests/loop/test_ninjars_profile_timing_metadata.py'


def load_helper():
    spec = importlib.util.spec_from_file_location('profile_metadata_fixture', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.OLD = ROOT/'tests/input_files/ninjars_profile_timing_legacy.py'
    module.NEW = SOURCE
    return module


def actual_setup():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'LoopMatrixElementTimer')
    method = copy.deepcopy(next(n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == 'setup_process'))
    namespace = dict(process_checks.__dict__)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['setup_process']


class ProfileExplicitPoles(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='tirust-profile-')
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)

    def fixture(self, *, mode=0, executable=False):
        export = self.work/'export'
        directory = export/'SubProcesses/P0_fixture'
        resources = directory/'MadLoop5_resources'
        resources.mkdir(parents=True)
        (export/'SubProcesses/MadLoop5_resources').symlink_to('P0_fixture/MadLoop5_resources')
        (export/'Cards').mkdir()
        (export/'Cards/MadLoopParams.dat').symlink_to('../SubProcesses/MadLoopParams.dat')
        card = banner.MadLoopParam()
        card['LoopPoleMode'] = mode
        card['HelicityFilterLevel'] = 2
        card['UseLoopFilter'] = True
        card.write(str(export/'SubProcesses/MadLoopParams.dat'))
        (directory/'proc_prefix.txt').write_text('ML5_0_')
        (directory/'loop_matrix.f').write_text('mock generated matrix; not executable')
        (directory/'check_sa.f').write_text('mock driver; not compiled')
        if executable:
            (directory/'check').write_text('mock executable identity; never run')
            (directory/'check').chmod(0o700)
        return export, directory

    def setup(self, export, directory, options):
        class Candidate(process_checks.LoopMatrixElementTimer):
            setup_process = actual_setup()
            def __init__(self):
                pass
        attempts = []
        def initialize(run_dir, subprocess_dir, infos, **kwargs):
            attempts.append(copy.deepcopy(kwargs))
            # This is the compile boundary: no compiler or numerical process runs.
            (directory/'check').write_text('mock compiled executable; never run')
            (directory/'check').chmod(0o700)
            infos.update(Process_compilation=.01, Initialization=.02)
            return 1
        original = copy.deepcopy(options)
        with mock.patch.object(MadLoopInitializer, 'fix_PSPoint_in_check'), \
                mock.patch.object(MadLoopInitializer, 'run_initialization', side_effect=initialize), \
                mock.patch.object(process_checks.subprocess, 'Popen', side_effect=AssertionError('native forbidden')):
            info = Candidate().setup_process(None, str(export), reusing=True, MLOptions=options)
        self.assertEqual(options, original)
        return info, attempts, banner.MadLoopParam(str(export/'SubProcesses/MadLoopParams.dat'))

    def test_fresh_explicit_mode_compiles_without_waiting_for_filters(self):
        export, directory = self.fixture()
        info, attempts, card = self.setup(export, directory, {'LoopPoleMode':0, 'MLReductionLib':[9]})
        self.assertEqual(attempts, [dict(req_files=[], attempts=[1])])
        self.assertEqual(info['Initialization'], .02)
        self.assertTrue(os.access(directory/'check', os.X_OK))
        self.assertEqual(card['HelicityFilterLevel'], 0)
        self.assertFalse(card['UseLoopFilter'])
        self.assertEqual(list((directory/'MadLoop5_resources').glob('*Filter.dat')), [])

    def test_reuse_reads_effective_card_mode_when_option_is_omitted(self):
        export, directory = self.fixture(executable=True)
        info, attempts, card = self.setup(export, directory, {'MLReductionLib':[9]})
        self.assertEqual(attempts, [])
        self.assertIsNone(info['Initialization'])
        self.assertEqual(card['HelicityFilterLevel'], 0)
        self.assertFalse(card['UseLoopFilter'])

    def test_lowercase_override_uses_effective_explicit_pole_policy(self):
        export, directory = self.fixture(mode=-1)
        _, attempts, card = self.setup(export, directory, {'looppolemode':0})
        self.assertEqual(attempts, [dict(req_files=[], attempts=[1])])
        self.assertEqual(card['LoopPoleMode'], 0)
        self.assertEqual(card['HelicityFilterLevel'], 0)
        self.assertFalse(card['UseLoopFilter'])

    def test_explicit_legacy_override_preserves_normal_filter_learning(self):
        export, directory = self.fixture()
        _, attempts, card = self.setup(export, directory, {'LoopPoleMode':-1})
        self.assertEqual(attempts, [dict(req_files=['HelFilter.dat','LoopFilter.dat'], attempts=[3,15])])
        self.assertEqual(card['HelicityFilterLevel'], 2)
        self.assertTrue(card['UseLoopFilter'])

    def test_default_legacy_mode_preserves_normal_filter_learning(self):
        export, directory = self.fixture(mode=-1)
        _, attempts, card = self.setup(export, directory, {})
        self.assertEqual(attempts, [dict(req_files=['HelFilter.dat','LoopFilter.dat'], attempts=[3,15])])
        self.assertEqual(card['HelicityFilterLevel'], 2)

    def test_full_timing_method_uses_all_configurations_without_a_filter(self):
        helper = load_helper()
        for tool in (8, 9):
            for optimized in (True, False):
                with self.subTest(tool=tool, optimized=optimized):
                    result, calls, _ = helper.exercise('all-%d-%s'%(tool, optimized), SOURCE,
                        tool=tool, optimized=optimized, real_card=True, pole_mode=0,
                        helicity_filter_file=False, base_root=self.work)
                    metadata = json.loads(Path(result['timing_metadata']['path']).read_text())
                    selected = metadata['selected_timing_helicity']
                    self.assertEqual(selected['filter_policy'], 'all-helicities-explicit-poles')
                    self.assertFalse(selected['learned_filter_used'])
                    self.assertEqual(selected['one_based_index'], 1)
                    self.assertEqual(selected['physical_tuple'], [1,-1,-1])
                    self.assertEqual(selected['filter_tokens'], [('1' if optimized else 'T')]*3)
                    self.assertEqual(selected['contributing_count'], 3)
                    self.assertFalse(selected['filter'].get('available', True))
                    self.assertEqual(metadata['backend_labels'], ['TIRust' if tool == 9 else 'NinjaRS'])
                    self.assertFalse(metadata['per_evaluation_statuses_available'])
                    for stage in metadata['stages']:
                        effective = {k.lower():v for k,v in stage['effective_madloop_parameters'].items()}
                        self.assertEqual(effective['looppolemode'], 0)
                        self.assertTrue(effective['tirustcomputefinite'])
                        self.assertTrue(effective['useqpintegrandfortirust'])

    def test_stale_filter_is_preserved_as_evidence_but_not_used(self):
        helper = load_helper()
        result, _, _ = helper.exercise('stale', SOURCE, tool=9,
            real_card=True, pole_mode=2, base_root=self.work)
        metadata = json.loads(Path(result['timing_metadata']['path']).read_text())
        selected = metadata['selected_timing_helicity']
        self.assertFalse(selected['learned_filter_used'])
        self.assertEqual(selected['filter_tokens'], ['1']*3)
        self.assertEqual(selected['filter']['text'], '-10000 1 -10000\n')
        self.assertEqual(selected['total_count'], 3)

    def test_legacy_timing_calls_and_results_stay_unchanged(self):
        helper = load_helper()
        old, old_calls, _ = helper.exercise('legacy-old', helper.OLD, tool=9, base_root=self.work)
        new, new_calls, _ = helper.exercise('legacy-new', SOURCE, tool=9, base_root=self.work)
        self.assertEqual(old_calls, new_calls)
        metadata = json.loads(Path(new.pop('timing_metadata')['path']).read_text())
        old['dir_path'] = new['dir_path']
        self.assertEqual(old, new)
        selected = metadata['selected_timing_helicity']
        self.assertEqual(selected['filter_policy'], 'learned-helicity-filter')
        self.assertTrue(selected['learned_filter_used'])
        self.assertEqual(selected['one_based_index'], 2)


if __name__ == '__main__':
    unittest.main()
