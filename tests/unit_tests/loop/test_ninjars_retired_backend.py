"""Retiring PJFry's return-code slot also rejects its live backend ID."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from madgraph import MadGraph5Error
from madgraph.interface.master_interface import MasterCmd
from madgraph.loop.loop_exporters import LoopProcessOptimizedExporterFortranSA
from madgraph.various.banner import MadLoopParam
from madgraph.various.process_checks import LoopMatrixElementTimer
from madgraph.various import process_checks


class TestRetiredPJFryCard(unittest.TestCase):
    def test_assignment_rejects_retired_backend_without_changing_card(self):
        card = MadLoopParam()
        previous = card['MLReductionLib']
        for value in ('2', '6|2|1', ' 8 | 02 | 1 ', '+2|8', 2):
            with self.subTest(value=value):
                with self.assertRaisesRegex(MadGraph5Error, r'PJFry.*retired.*MLReductionLib=8'):
                    card.set('mlreductionlib', value, user=True)
                self.assertEqual(card['MLReductionLib'], previous)
                self.assertNotIn('mlreductionlib', card.user_set)

    def test_import_rejects_retired_backend(self):
        with self.assertRaisesRegex(MadGraph5Error, r'PJFry.*retired'):
            MadLoopParam('#MLReductionLib\n6|2|1\n')

    def test_card_editor_quoted_and_assignment_forms_are_checked(self):
        card = MadLoopParam()
        card.set('MLReductionLib', '= "8|6"')
        self.assertEqual(card['MLReductionLib'], '8|6')
        for value in ('"2"', "'6|2'", '= "+02|8"', [8, 2], (8, '02')):
            with self.subTest(value=value):
                with self.assertRaisesRegex(MadGraph5Error, r'PJFry.*retired'):
                    card.set('MLReductionLib', value)
                self.assertEqual(card['MLReductionLib'], '8|6')

    def test_live_backend_selection_remains_eight(self):
        for value in ('8', '8|6|7|1', '1|6'):
            with self.subTest(value=value):
                card = MadLoopParam('#MLReductionLib\n' + value + '\n')
                stream = io.StringIO()
                card.write(stream)
                self.assertIn('#MLReductionLib\n' + value + '\n', stream.getvalue())
                self.assertEqual(card['MLReductionLib'], value)

    def test_serialization_catches_bypassed_setter_before_writing(self):
        for value in ('8|2|6', 2, [8, 2], ('8', '02')):
            with self.subTest(value=value):
                card = MadLoopParam()
                card.update({'mlreductionlib': value})
                stream = io.StringIO()
                with self.assertRaisesRegex(MadGraph5Error, r'PJFry.*retired'):
                    card.write(stream)
                self.assertEqual(stream.getvalue(), '')


class TestRetiredPJFryEntryPoints(unittest.TestCase):
    def test_export_refuses_even_an_existing_pjfry_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / 'library'
            library.mkdir()
            archive = library / 'libpjfry.a'
            archive.write_bytes(b'!<arch>\n')
            for dependency in ('internal', 'external'):
                with self.subTest(dependency=dependency):
                    exporter = LoopProcessOptimizedExporterFortranSA.__new__(
                        LoopProcessOptimizedExporterFortranSA)
                    exporter.dependencies = dependency
                    exporter.tir_available_dict = {'pjfry': True}
                    target = root / dependency
                    self.assertEqual(exporter.link_TIR(
                        str(target), str(library), archive.name,
                        tir_name='pjfry'), '')
                    self.assertFalse(exporter.tir_available_dict['pjfry'])
                    self.assertFalse(target.exists())
                    self.assertEqual(archive.read_bytes(), b'!<arch>\n')

    def test_check_cli_rejects_before_timing_or_export(self):
        command = MasterCmd()
        # Use the real loop command dispatcher and model validation. Retirement
        # must still be detected before diagram generation or backend checks.
        command.do_import('model loop_sm')
        with patch('madgraph.various.process_checks.check_timing') as timing, \
                patch('madgraph.various.process_checks.check_profile') as profile, \
                patch('madgraph.various.process_checks.check_stability') as stability:
            for check in ('timing', 'profile', 'stability'):
                with self.subTest(check=check):
                    with self.assertRaisesRegex(command.InvalidCmd,
                                                r'PJFry.*retired.*8.*NinjaRS'):
                        command.do_check(check + ' u u~ > z [virt=QCD] --reduction=2')
            timing.assert_not_called()
            profile.assert_not_called()
            stability.assert_not_called()

    def test_direct_stability_rejects_before_matrix_validation(self):
        timer = LoopMatrixElementTimer.__new__(LoopMatrixElementTimer)
        with self.assertRaisesRegex(MadGraph5Error, r'PJFry.*retired.*8.*NinjaRS'):
            timer.check_matrix_element_stability(
                None, MLOptions={'MLReductionLib': [6, 2]})

    def test_direct_timing_rejects_before_matrix_validation(self):
        timer = LoopMatrixElementTimer.__new__(LoopMatrixElementTimer)
        with self.assertRaisesRegex(MadGraph5Error, r'PJFry.*retired.*8.*NinjaRS'):
            timer.time_matrix_element(None, MLOptions={'MLReductionLib': [6, 2]})

    def test_module_checks_reject_before_process_generation(self):
        for check in (process_checks.check_profile, process_checks.check_timing,
                      process_checks.check_stability):
            with self.subTest(check=check.__name__):
                with self.assertRaisesRegex(MadGraph5Error, r'PJFry.*retired.*8.*NinjaRS'):
                    check(None, MLOptions={'MLReductionLib': [6, 2]})


if __name__ == '__main__':
    unittest.main()
