"""First-use prompts respect process defaults and supported host platforms."""

from pathlib import Path
from types import SimpleNamespace
import json
import os
import tempfile
import unittest
from unittest import mock

from madgraph.interface import loop_interface


class NinjaRSInstallPrompt(unittest.TestCase):
    def owner(self, born=True, *, processes=True):
        return SimpleNamespace(
            options={'ninja': '/not/installed/ninja', 'collier': None,
                     'ninjars': '/installed/ninjars/lib'},
            _curr_amps=[{'process': {'has_born': born}}] if processes else [],
            ask=mock.Mock(return_value={}))

    def run_prompt(self, owner, available=True, force=False):
        with mock.patch.object(loop_interface.ninjars_install, 'is_available',
                               return_value=available), \
                mock.patch.object(loop_interface.logger, 'info'):
            loop_interface.LoopInterface.install_reduction_library(owner, force=force)

    def test_usable_ninjars_satisfies_born_processes_only(self):
        born = self.owner()
        self.run_prompt(born)
        born.ask.assert_not_called()
        induced = self.owner(born=False)
        self.run_prompt(induced)
        induced.ask.assert_called_once()

    def test_unknown_and_mixed_processes_keep_legacy_install_offer(self):
        for owner in (self.owner(processes=False), self.owner()):
            if owner._curr_amps:
                owner._curr_amps.append({'process': {'has_born': False}})
            self.run_prompt(owner)
            owner.ask.assert_called_once()

    def test_nlo_process_definitions_can_use_installed_ninjars(self):
        owner = self.owner(processes=False)
        owner._fks_multi_proc = {'process_definitions': [{'has_born': True}]}
        self.run_prompt(owner)
        owner.ask.assert_not_called()

    def test_unusable_ninjars_does_not_suppress_born_offer(self):
        owner = self.owner()
        self.run_prompt(owner, available=False)
        owner.ask.assert_called_once()

    def test_relative_installed_path_is_resolved_from_mg5dir_not_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'mg7'
            outside = Path(directory) / 'outside'
            outside.mkdir()
            lib = root / 'HEPTools' / 'ninjars' / 'lib'
            lib.mkdir(parents=True)
            (lib / 'libninjars.a').write_bytes(b'!<arch>\nnonempty archive')
            (lib / 'mninjars.mod').write_bytes(b'nonempty module')
            (lib / 'ninjars-config.json').write_text(json.dumps({
                'abi_version': loop_interface.ninjars_install.ABI_VERSION,
                'native_libs': ['-lm'], 'link_order_version': 1,
                'link_before': ['-lgcc_s']}))
            (lib / 'ninjars-native-libs.txt').write_text('-lm\n')
            (lib / 'ninjars-link-before.txt').write_text('-lgcc_s\n')
            owner = self.owner()
            owner.options['ninjars'] = 'HEPTools/ninjars/lib'
            previous = os.getcwd()
            try:
                os.chdir(outside)
                with mock.patch.object(loop_interface.ninjars_install, 'MG5DIR', str(root)):
                    loop_interface.LoopInterface.install_reduction_library(owner)
            finally:
                os.chdir(previous)
            owner.ask.assert_not_called()

    def test_existing_legacy_backend_or_explicit_opt_out_is_respected(self):
        with tempfile.TemporaryDirectory() as directory:
            lib = Path(directory)
            for backend, archive in (('ninja', 'libninja.a'), ('collier', 'libcollier.a')):
                with self.subTest(backend=backend):
                    (lib / archive).write_bytes(b'present')
                    owner = self.owner(born=False)
                    owner.options[backend] = str(lib)
                    self.run_prompt(owner)
                    owner.ask.assert_not_called()
        owner = self.owner(born=False)
        owner.options['ninja'] = None
        self.run_prompt(owner)
        owner.ask.assert_not_called()

    def test_force_still_opens_the_installer_for_born_processes(self):
        owner = self.owner()
        self.run_prompt(owner, force=True)
        owner.ask.assert_called_once()

    def menu(self, platform):
        with mock.patch.object(loop_interface.sys, 'platform', platform), \
                mock.patch('urllib.request.urlopen'), \
                mock.patch.object(loop_interface.misc, 'which', return_value='/bin/tool'), \
                mock.patch.object(loop_interface.cmd.OneLinePathCompletion, '__init__',
                                  return_value=None):
            return loop_interface.AskLoopInstaller('install')

    def test_menu_offers_ninjars_only_on_linux(self):
        for platform in ('darwin', 'win32', 'freebsd14'):
            with self.subTest(platform=platform):
                menu = self.menu(platform)
                question, allowed = menu.create_question()
                self.assertNotIn('ninjars', menu.code)
                self.assertNotIn('ninjars', menu.order)
                self.assertNotIn('ninjars', question.lower())
                self.assertNotIn('ninjars', allowed)
                with mock.patch.object(loop_interface.logger, 'warning'):
                    menu.default('ninjars install')
                self.assertNotIn('ninjars', menu.code)
        menu = self.menu('linux')
        self.assertIn('ninjars', menu.code)
        self.assertIn('ninjars', menu.create_question()[1])
