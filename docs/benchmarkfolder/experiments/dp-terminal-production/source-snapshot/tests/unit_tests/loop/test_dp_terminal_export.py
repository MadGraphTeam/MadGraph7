"""DP private call proposals must preserve dense output when proof is missing."""
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from madgraph.loop.loop_exporters import LoopProcessOptimizedExporterFortranSA as Exporter
from madgraph.iolibs.helas_call_writers import FortranUFOHelasCallWriterOptimized as Writer
from madgraph.various.q_polynomial import FortranPolynomialRoutines, PolynomialError


EXPECTED = {3: dict(loop_mother_rank=1, vertex_rank=1, loop_mother_number=2,
                   lcut_size=4, in_size=4, out_size=4, out=3)}


def batch(prefix='TEST_'):
    return ('      SUBROUTINE '+prefix+'COEF_CONSTRUCTION_1()\n'
            '      USE '+prefix+'POLYNOMIAL_CONSTANTS\n'
            '      IMPLICIT NONE\n'
            '      INTEGER I\n'
            '      CALL VERTEX(COEFS)\n'
            '      CALL '+prefix+'UPDATE_WL_1_1(WL(1,0,1,2),4,COEFS,4,4,\n'
            '     $ WL(1,0,1,3))\n'
            '      CALL '+prefix+'CREATE_LOOP_COEFS(WL(1,0,1,3),2,4,1)\n'
            '      END\n')


class DPTerminalProposalTest(unittest.TestCase):
    def test_capture_follows_vertex_and_directly_precedes_private_call(self):
        for prefix in ('TEST_', ''):
            source = batch(prefix)
            candidate, found = Exporter.prepare_dp_terminal_batch(source, prefix, EXPECTED)
            self.assertEqual(found, {3})
            statements = []
            for line in candidate.splitlines():
                if line[5] not in ' 0':
                    statements[-1] += line[6:].strip()
                else:
                    statements.append(line[6:].strip())
            private = next(i for i, text in enumerate(statements) if text.startswith('CALL '+prefix+'DP_TERMINAL_'))
            self.assertEqual(statements[private-2:private],
                             ['CALL VERTEX(COEFS)', 'CALL IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)'])
            self.assertIn('USE, INTRINSIC :: IEEE_EXCEPTIONS, ONLY: IEEE_GET_FLAG,IEEE_INEXACT', statements)
            self.assertIn('LOGICAL WL_ENTRY_INEXACT', statements)
            self.assertIn('      CALL '+prefix+'CREATE_LOOP_COEFS(WL(1,0,1,3),2,4,1)\n', candidate)

    def test_unrelated_batch_is_byte_preserved(self):
        source = batch().replace('WL(1,0,1,3)', 'WL(1,0,1,4)')
        self.assertEqual(Exporter.prepare_dp_terminal_batch(source, 'TEST_', EXPECTED), (source, set()))

    def test_labeled_duplicate_or_ambiguous_scope_refused(self):
        original = batch()
        call = '      CALL TEST_UPDATE_WL_1_1(WL(1,0,1,2),4,COEFS,4,4,\n     $ WL(1,0,1,3))\n'
        for source in (original.replace('      CALL TEST_UPDATE', '  400 CALL TEST_UPDATE'),
                       original.replace(call, call+call),
                       original.replace('      IMPLICIT NONE\n', ''),
                       original.replace('      INTEGER I\n', '      LOGICAL WL_ENTRY_INEXACT\n')):
            with self.subTest(source=source), self.assertRaises(ValueError):
                Exporter.prepare_dp_terminal_batch(source, 'TEST_', EXPECTED)

    def test_unsupported_shape_refused(self):
        for field, value in [('lcut_size', 3), ('in_size', 5), ('out_size', 3),
                             ('loop_mother_rank', 2), ('vertex_rank', 0)]:
            info = dict(EXPECTED[3], **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                Exporter.prepare_dp_terminal_batch(batch(), 'TEST_', {3: info})


class DPTerminalSelectionTest(unittest.TestCase):
    def exporter(self, cls=Exporter):
        exporter = object.__new__(cls)
        exporter.opt = {}
        exporter.get_context = lambda matrix: {'MadEventOutput': False}
        matrix = Mock()
        matrix.rep_dict = {'proc_prefix': 'TEST_', 'max_lwf_size': 4}
        matrix.get.return_value = [{'has_born': True}]
        return exporter, matrix

    def test_only_qualified_shape_is_selected(self):
        exporter, matrix = self.exporter()
        choices = dict(EXPECTED)
        for number, (field, value) in enumerate(
                [('loop_mother_rank', 2), ('vertex_rank', 0), ('lcut_size', 3),
                 ('in_size', 3), ('out_size', 3)], start=4):
            choices[number] = dict(EXPECTED[3], out=number, **{field: value})
        with patch.object(Writer, 'get_terminal_wavefunctions', return_value=choices):
            self.assertEqual(exporter.get_dp_terminal_wavefunctions(matrix), EXPECTED)

    def test_custom_vector_nlo_and_unknown_shapes_stay_dense(self):
        class CustomExporter(Exporter):
            pass
        for setting in ('subclass', 'vector', 'nlo', 'loopinduced', 'shape', 'identifier'):
            with self.subTest(setting=setting):
                exporter, matrix = self.exporter(CustomExporter if setting == 'subclass' else Exporter)
                if setting == 'vector':
                    exporter.opt['vector_size'] = 4
                elif setting == 'nlo':
                    exporter.get_context = lambda matrix: {'MadEventOutput': True}
                elif setting == 'loopinduced':
                    matrix.get.return_value = [{'has_born': False}]
                elif setting == 'shape':
                    matrix.rep_dict['max_lwf_size'] = 16
                elif setting == 'identifier':
                    matrix.rep_dict['proc_prefix'] = 'A'*64
                with patch.object(Writer, 'get_terminal_wavefunctions') as graph:
                    self.assertEqual(exporter.get_dp_terminal_wavefunctions(matrix), {})
                    graph.assert_not_called()


class DPTerminalGeneratorTest(unittest.TestCase):
    def test_unrecognized_dense_structure_cannot_emit_unguarded_private_code(self):
        writer = FortranPolynomialRoutines(2, updater_max_rank=1,
            sub_prefix='TEST_', coef_format='complex*16')
        dense = writer.write_expanded_wl_updater(1, 1, _sparse=False)
        for malformed in (dense.replace('DO I=1,LCUT_SIZE\n', 'DO I = 1,LCUT_SIZE\n'),
                          dense.replace('  INTEGER I,J,K\n', '  INTEGER I,J,K\n'*2),
                          dense[:-3]+'END SUBROUTINE'):
            with self.subTest(source=malformed), patch.object(
                    writer, 'write_expanded_wl_updater', return_value=malformed):
                with self.assertRaises(PolynomialError):
                    writer.write_dp_terminal_wl_updater()


class DPTerminalFinalizeTest(unittest.TestCase):
    def finalize(self, directory, validator):
        exporter = object.__new__(Exporter)
        exporter._dp_terminal_pending = {str(directory): ('TEST_', EXPECTED)}
        module = types.ModuleType('madgraph.loop.dp_terminal_proof')
        module.validate_dp_terminal_wavefunctions = validator
        with patch.dict(sys.modules, {'madgraph.loop.dp_terminal_proof': module}):
            exporter.finalize_dp_terminal_updates()
        self.assertEqual(exporter._dp_terminal_pending, {})

    def test_independent_validator_sees_complete_proposal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root/'coef_construction_1.f'
            path.write_text(batch())
            seen = []
            def validate(directory, prefix, expected):
                seen.append((directory, prefix, expected))
                self.assertIn('DP_TERMINAL_WL_1_1', path.read_text())
                return True
            self.finalize(root, validate)
            self.assertEqual(len(seen), 1)
            self.assertIn('DP_TERMINAL_WL_1_1', path.read_text())

    def test_failed_proof_restores_every_original_byte(self):
        for behavior in ('false', 'error'):
            with self.subTest(behavior=behavior), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                path = root/'coef_construction_1.f'
                original = batch().encode()
                path.write_bytes(original)
                def validate(*args):
                    self.assertIn(b'DP_TERMINAL_WL_1_1', path.read_bytes())
                    if behavior == 'error':
                        raise ValueError('unknown generated consumer')
                    return False
                self.finalize(root, validate)
                self.assertEqual(path.read_bytes(), original)

    def test_duplicate_across_batches_never_promotes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            paths = [root/'coef_construction_1.f', root/'coef_construction_2.f']
            for path in paths:
                path.write_text(batch())
            self.finalize(root, lambda *args: self.fail('Incomplete proposal reached validator'))
            for path in paths:
                self.assertEqual(path.read_text(), batch())

    def test_symlinked_batch_never_writes_external_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            external = root/'external.f'
            external.write_text(batch())
            process = root/'process'
            process.mkdir()
            (process/'coef_construction_1.f').symlink_to(external)
            self.finalize(process, lambda *args: self.fail('Aliased batch reached validator'))
            self.assertEqual(external.read_text(), batch())
