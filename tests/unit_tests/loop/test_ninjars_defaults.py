"""Private default-policy regression controls; no exports, builds or reducers."""
import ast
import copy
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from madgraph.various import banner

CANDIDATE = ROOT
spec = importlib.util.spec_from_file_location('private_default_ninjars_install', CANDIDATE / 'madgraph/various/ninjars_install.py')
install = importlib.util.module_from_spec(spec); spec.loader.exec_module(install)
export_tree = ast.parse((CANDIDATE / 'madgraph/loop/loop_exporters.py').read_text())
export_class = next(n for n in export_tree.body if isinstance(n, ast.ClassDef) and n.name == 'LoopProcessExporterFortranSA')
methods = [n for n in export_class.body if isinstance(n, ast.FunctionDef) and n.name in ('finalize', 'set_default_reduction_library')]
optimized_class = next(n for n in export_tree.body if isinstance(n, ast.ClassDef) and n.name == 'LoopProcessOptimizedExporterFortranSA')
methods.append(next(n for n in optimized_class.body if isinstance(n, ast.FunctionDef) and n.name == 'link_TIR'))
namespace = {'ninjars_install': install, 'getattr': getattr, 'pjoin': os.path.join, 'banner_mod': banner,
    'os': os, 'logger': logging.getLogger('private-default-test'),
    'LoopProcessExporterFortranSA': object, 'super': lambda *args: SimpleNamespace(finalize=lambda *args: None)}
exec(compile(ast.Module(body=methods, type_ignores=[]), '<private-exporter-methods>', 'exec'), namespace)


class NinjaRSDefault(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT, prefix='.test-nrs-default-')
        self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        self.lib = self.root / 'installed/lib'; self.lib.mkdir(parents=True)
        self.config = {'abi_version': install.ABI_VERSION, 'native_libs': ['-ldl', '-lm'], 'link_order_version': 1,
            'link_before': ['-Wl,--push-state,--no-as-needed', '-lquadmath', '-Wl,--pop-state']}
        (self.lib / 'libninjars.a').write_bytes(b'!<arch>\nsynthetic archive metadata')
        (self.lib / 'mninjars.mod').write_bytes(b'synthetic nonempty module')
        self.write_config()

    def write_config(self):
        (self.lib / 'ninjars-config.json').write_text(json.dumps(self.config))
        for name, key in (('ninjars-native-libs.txt', 'native_libs'), ('ninjars-link-before.txt', 'link_before')):
            (self.lib / name).write_text(' '.join(self.config[key]) + '\n')

    def owner(self, has_loop_induced=False, available=True, directory=None):
        owner = SimpleNamespace(has_loop_induced=has_loop_induced,
            tir_available_dict={'ninjars': available}, ninjars_dir=str(self.lib), dir_path=str(directory or self.root / 'output'))
        owner.set_default_reduction_library = MethodType(namespace['set_default_reduction_library'], owner)
        return owner

    def template(self, directory=None, selection=None):
        directory = directory or self.root / 'output'; (directory / 'Cards').mkdir(parents=True, exist_ok=True)
        (directory / 'SubProcesses').mkdir(exist_ok=True)
        text = (ROOT / 'Template/loop_material/StandAlone/Cards/MadLoopParams.dat').read_text()
        if selection is not None: text = text.replace('!6|7|1\n', selection + '\n', 1)
        for name in ('MadLoopParams.dat', 'MadLoopParams_default.dat'): (directory / 'Cards' / name).write_text(text)
        return directory

    def test_complete_installation_and_relative_configured_path_are_usable(self):
        self.assertTrue(install.is_available(self.lib))
        self.assertTrue(install.is_available(str(self.lib.relative_to(ROOT))))

    def test_module_in_standard_include_directory_is_usable(self):
        include = self.lib.parent / 'include'; include.mkdir()
        (self.lib / 'mninjars.mod').rename(include / 'mninjars.mod')
        self.assertTrue(install.is_available(self.lib))

    def test_missing_disabled_or_corrupt_archive_paths_are_not_usable(self):
        for path in (None, '', False, self.root / 'absent'):
            with self.subTest(path=path): self.assertFalse(install.is_available(path))
        (self.lib / 'libninjars.a').write_bytes(b'not an archive')
        self.assertFalse(install.is_available(self.lib))
        (self.lib / 'libninjars.a').write_bytes(b'!<arch>\n')
        self.assertFalse(install.is_available(self.lib))

    def test_archive_only_or_missing_module_is_not_an_installation(self):
        (self.lib / 'ninjars-config.json').unlink(); self.assertFalse(install.is_available(self.lib))
        self.write_config(); (self.lib / 'mninjars.mod').unlink()
        self.assertFalse(install.is_available(self.lib))

    def test_invalid_abi_or_native_metadata_are_not_usable(self):
        for config in ({**self.config, 'abi_version': 3}, {**self.config, 'native_libs': 'invalid'},
                {**self.config, 'link_before': []}):
            (self.lib / 'ninjars-config.json').write_text(json.dumps(config))
            with self.subTest(config=config): self.assertFalse(install.is_available(self.lib))
        self.write_config(); (self.lib / 'ninjars-native-libs.txt').write_text('-ldifferent\n')
        self.assertFalse(install.is_available(self.lib))

    def test_unusable_optional_installation_is_skipped_before_export_link_setup(self):
        (self.lib / 'ninjars-config.json').unlink()
        owner = self.owner()
        with self.assertLogs('private-default-test', level='WARNING') as logged:
            self.assertEqual(namespace['link_TIR'](owner, str(self.root / 'target'), str(self.lib),
                'libninjars.a', tir_name='ninjars'), '')
        self.assertFalse(owner.tir_available_dict['ninjars'])
        self.assertIn('install ninjars --update', '\n'.join(logged.output))

    def test_untouched_default_is_active_ninjars_first_with_legacy_fallbacks(self):
        card = banner.MadLoopParam(); self.assertEqual(card['MLReductionLib'], '6|7|1')
        self.assertTrue(self.owner().set_default_reduction_library(card))
        self.assertEqual(card['MLReductionLib'], '8|6|7|1')
        stream = io.StringIO(); card.write(stream)
        self.assertIn('#MLReductionLib\n8|6|7|1\n', stream.getvalue())

    def test_explicit_backend_and_explicit_legacy_order_are_preserved(self):
        for value in ('6', '8', '1|6', '6|7|1', '7|6|1'):
            card = banner.MadLoopParam(); card.set('MLReductionLib', value, user=True)
            with self.subTest(value=value):
                self.assertFalse(self.owner().set_default_reduction_library(card))
                self.assertEqual(card['MLReductionLib'], value)

    def test_absent_or_unusable_library_and_nonoptimized_export_keep_legacy_default(self):
        for owner in (self.owner(available=False), self.owner()):
            owner.ninjars_dir = str(self.root / 'missing')
            card = banner.MadLoopParam(); self.assertFalse(owner.set_default_reduction_library(card))
            self.assertEqual(card['MLReductionLib'], '6|7|1')
        card = banner.MadLoopParam(); self.assertFalse(self.owner(available=False).set_default_reduction_library(card))
        self.assertEqual(card['MLReductionLib'], '6|7|1')

    def test_ordinary_standalone_finalization_writes_both_default_and_runtime_cards(self):
        output = self.template(); owner = self.owner(directory=output)
        namespace['finalize'](owner, None, None, {}, [])
        for name in ('MadLoopParams.dat', 'MadLoopParams_default.dat'):
            card = banner.MadLoopParam(str(output / 'Cards' / name))
            self.assertEqual(card['MLReductionLib'], '8|6|7|1')
            self.assertTrue(card['COLLIERComputeUVpoles']); self.assertTrue(card['COLLIERComputeIRpoles'])

    def test_standalone_explicit_card_and_loop_induced_three_defaults_are_unchanged(self):
        output = self.template(selection='6|7|1'); owner = self.owner(directory=output)
        namespace['finalize'](owner, None, None, {}, [])
        self.assertEqual(banner.MadLoopParam(str(output / 'Cards/MadLoopParams.dat'))['MLReductionLib'], '6|7|1')
        output = self.template(directory=self.root / 'loop'); owner = self.owner(has_loop_induced=True, directory=output)
        namespace['finalize'](owner, None, None, {}, [])
        card = banner.MadLoopParam(str(output / 'Cards/MadLoopParams.dat'))
        self.assertEqual(card['MLReductionLib'], '7|6|1')
        self.assertFalse(card['COLLIERComputeUVpoles']); self.assertFalse(card['COLLIERComputeIRpoles'])

    def fks_hooks(self):
        tree = ast.parse((CANDIDATE / 'madgraph/iolibs/export_fks.py').read_text())
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef) and cls.name in ('ProcessExporterFortranFKS', 'ProcessOptimizedExporterFortranFKS'):
                method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'copy_fkstemplate')
                blocks = [n for n in method.body if isinstance(n, ast.If) and any(
                    isinstance(x, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'MadLoopparam' for t in x.targets)
                    for x in ast.walk(n))]
                self.assertEqual(len(blocks), 1)
                yield cls.name, blocks[0]

    def test_both_fks_card_copy_routes_propagate_default_to_all_three_cards(self):
        for name, block in self.fks_hooks():
            output = self.template(directory=self.root / name); owner = self.owner(directory=output)
            scope = {'self': owner, 'os': os, 'pjoin': os.path.join, 'banner_mod': banner}
            exec(compile(ast.Module(body=[block], type_ignores=[]), '<private-fks-card-hook>', 'exec'), scope)
            for path in ('Cards/MadLoopParams.dat', 'Cards/MadLoopParams_default.dat', 'SubProcesses/MadLoopParams.dat'):
                with self.subTest(route=name, path=path):
                    self.assertEqual(banner.MadLoopParam(str(output / path))['MLReductionLib'], '8|6|7|1')

    def test_both_fks_card_copy_routes_preserve_explicit_user_order(self):
        for name, block in self.fks_hooks():
            output = self.template(directory=self.root / name, selection='1|6')
            scope = {'self': self.owner(directory=output), 'os': os, 'pjoin': os.path.join, 'banner_mod': banner}
            exec(compile(ast.Module(body=[block], type_ignores=[]), '<private-fks-card-hook>', 'exec'), scope)
            self.assertEqual(banner.MadLoopParam(str(output / 'SubProcesses/MadLoopParams.dat'))['MLReductionLib'], '1|6')

    def cli_default(self, args, has_born, selected):
        tree = ast.parse((CANDIDATE / 'madgraph/interface/madgraph_interface.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MadGraphCmd')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'do_check')
        assignment = next(n for n in method.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == 'automatic_reduction' for t in n.targets))
        condition = next(n for n in method.body if isinstance(n, ast.If) and 'automatic_reduction' in ast.unparse(n.test))
        scope = {'args': args, 'self': SimpleNamespace(options={'ninjars': str(self.lib)}),
            'myprocdef': {'has_born': has_born}, 'MLoptions': {'MLReductionLib': selected[:]}}
        with mock.patch('madgraph.various.ninjars_install.is_available', create=True, side_effect=install.is_available):
            exec(compile(ast.Module(body=[assignment, condition], type_ignores=[]), '<private-cli-default>', 'exec'), scope)
        return scope['MLoptions']['MLReductionLib']

    def test_cli_timing_profile_default_to_sole_ninjars_when_usable(self):
        for mode in ('timing', 'profile'):
            self.assertEqual(self.cli_default([mode, 'u u~ > z [virt=QCD]'], True, [1,3,5,6]), [8])
        (self.lib / 'ninjars-config.json').unlink()
        self.assertEqual(self.cli_default(['profile', 'u u~ > z [virt=QCD]'], True, [1,3,5,6]), [6])

    def test_cli_explicit_lists_loop_induced_and_comparison_checks_are_preserved(self):
        for mode, born, args, selection in (
                ('profile', True, ['--reduction=6'], [6]),
                ('timing', True, ['--reduction=8|6'], [8,6]),
                ('profile', False, [], [1,3,5,6]),
                ('stability', True, [], [1,3,5,6])):
            with self.subTest(mode=mode, born=born, selection=selection):
                self.assertEqual(self.cli_default([mode, 'process', *args], born, selection), selection)

    def test_explicit_cli_ninjars_with_invalid_metadata_is_actionable(self):
        (self.lib / 'ninjars-config.json').unlink()
        tree = ast.parse((CANDIDATE / 'madgraph/interface/madgraph_interface.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MadGraphCmd')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'do_check')
        branch = next(n for n in method.body if isinstance(n, ast.If) and
            ast.unparse(n.test).startswith("isinstance(self.options.get('ninjars'), str)"))
        scope = {'self': SimpleNamespace(options={'ninjars': str(self.lib)}, InvalidCmd=ValueError),
            'MLoptions': {'MLReductionLib': [8]}, 'TIR_dir': {}}
        with mock.patch('madgraph.various.ninjars_install.is_available', create=True, side_effect=install.is_available):
            with self.assertRaisesRegex(ValueError, 'install ninjars --update'):
                exec(compile(ast.Module(body=[branch], type_ignores=[]), '<private-explicit-cli>', 'exec'), scope)
        self.assertEqual(scope['MLoptions']['MLReductionLib'], [8])

    def direct_cli_selection(self, name, has_born=True, reusing=False, selected=None):
        tree = ast.parse((CANDIDATE / 'madgraph/various/process_checks.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        class Process(dict): pass
        process = Process(has_born=has_born)
        matrix = process if reusing else {'processes': [process]}
        observed = []
        class Timer:
            loop_optimized_output = True
            def __init__(self, **kwargs): pass
            def time_matrix_element(self, *args, **kwargs):
                observed.append(copy.deepcopy(kwargs['MLOptions']['MLReductionLib']))
                return {}
            def check_matrix_element_stability(self, *args, **kwargs):
                observed.append(copy.deepcopy(kwargs['MLOptions']['MLReductionLib']))
                return {}
        scope = {'FakeInterface': lambda: None, 'base_objects': SimpleNamespace(Process=Process),
            'generate_loop_matrix_element': lambda *a, **k: ({}, matrix),
            'LoopMatrixElementTimer': Timer, 'ninjars_install': install, 'copy': copy}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<private-direct-cli>', 'exec'), scope)
        mloptions = {} if selected is None else {'MLReductionLib': selected[:]}
        scope[name]({'model': object(), 'has_born': not has_born}, options={},
            tir={'ninjars_dir': str(self.lib)}, MLOptions=mloptions)
        return observed

    def test_direct_timing_and_profile_use_actual_generated_born_status_on_reuse(self):
        for name in ('check_timing', 'check_profile'):
            for reuse in (False, True):
                with self.subTest(name=name, reuse=reuse):
                    self.assertTrue(all(value == [8] for value in self.direct_cli_selection(name, reusing=reuse)))
        (self.lib / 'ninjars-config.json').unlink()
        for name in ('check_timing', 'check_profile'):
            self.assertTrue(all(value == [6] for value in self.direct_cli_selection(name)))

    def test_direct_cli_explicit_lists_and_loop_induced_timing_are_unchanged(self):
        for name in ('check_timing', 'check_profile'):
            for selection in ([6], [8], [6, 8], [7, 6, 1]):
                with self.subTest(name=name, selection=selection):
                    self.assertTrue(all(value == selection for value in self.direct_cli_selection(name, selected=selection)))
            self.assertTrue(all(value == [7, 6, 1] for value in self.direct_cli_selection(name,
                has_born=False, selected=[7, 6, 1])))
        self.assertEqual(self.direct_cli_selection('check_timing', has_born=False), [[6]])


if __name__ == '__main__': unittest.main()
