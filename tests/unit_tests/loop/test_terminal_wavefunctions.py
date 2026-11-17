"""The private diagonal updater must fail closed on unknown consumers."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

from madgraph.iolibs.helas_call_writers import FortranUFOHelasCallWriterOptimized as Writer
from madgraph.loop.loop_exporters import LoopProcessOptimizedExporterFortranSA as Exporter
from madgraph.loop.loop_helas_objects import LoopHelasMatrixElement


class Wavefunction(dict):
    def __init__(self, number, mother=None, is_loop=True):
        super().__init__(number=number, me_id=number, is_loop=is_loop,
                         mothers=[] if mother is None else [mother])

    def get_loop_mother(self):
        return next((m for m in self['mothers'] if m['is_loop']), None)

    def get_helas_call_dict(self, **kwargs):
        return dict(out=self['number'], loop_mother_number=self.get_loop_mother()['number'],
                    loop_mother_rank=1, vertex_rank=1, lcut_size=4, in_size=4, out_size=4)


def matrix():
    seed = Wavefunction(0)
    first = Wavefunction(1, seed)
    final = Wavefunction(2, first)
    amp = Mock()
    amp.get_final_loop_wavefunction.return_value = final
    diagram = Mock()
    diagram.get.side_effect = {'loop_wavefunctions': [first, final], 'number': 1}.get
    diagram.get_loop_amplitudes.return_value = [amp]
    me = Mock(spec=LoopHelasMatrixElement)
    me.get_loop_diagrams.return_value = [diagram]
    me.get_all_wavefunctions.return_value = []
    return me, diagram, first, final, amp


class TerminalWavefunctionGraphTest(unittest.TestCase):
    def test_only_final_unconsumed_roots(self):
        me, diagram, first, final, amp = matrix()
        self.assertEqual(set(Writer.get_terminal_wavefunctions(me)), {2})
        another = Mock()
        another.get_final_loop_wavefunction.return_value = first
        diagram.get_loop_amplitudes.return_value.append(another)
        self.assertEqual(set(Writer.get_terminal_wavefunctions(me)), {2})

    def test_unknown_producer_alias_and_identifier_mismatch_disable(self):
        for problem in ('mother', 'duplicate', 'alias', 'id', 'root'):
            with self.subTest(problem=problem):
                me, diagram, first, final, amp = matrix()
                if problem == 'mother':
                    first['mothers'] = [Wavefunction(99)]
                elif problem == 'duplicate':
                    diagram.get('loop_wavefunctions').append(final)
                elif problem == 'alias':
                    final['mothers'] = [final]
                elif problem == 'id':
                    final['me_id'] = 3
                else:
                    amp.get_final_loop_wavefunction.return_value = Wavefunction(99)
                self.assertEqual(Writer.get_terminal_wavefunctions(me), {})

    def test_non_loop_consumer_disables(self):
        me, _, _, final, _ = matrix()
        me.get_all_wavefunctions.return_value = [Wavefunction(5, final, False)]
        self.assertEqual(Writer.get_terminal_wavefunctions(me), {})

    def test_shared_terminal_becomes_full(self):
        me, diagram, _, final, _ = matrix()
        diagram.get('loop_wavefunctions').append(Wavefunction(3, final))
        self.assertEqual(Writer.get_terminal_wavefunctions(me), {})

    def test_explicit_selection_preserves_cache_dp_and_labels(self):
        me, _, first, final, amp = matrix()
        me.get.side_effect = {'processes': [{'has_born': True}]}.get
        born = Mock()
        born.get.side_effect = {'number': 1}.get
        amp.get.side_effect = {'loop_group_id': 0, 'amplitudes': [born],
                              'loopsymmetryfactor': 1, 'multiplier': 1}.get
        amp.get_analytic_info.return_value = 2
        amp.get_lcut_size.return_value = 4
        writer = Mock(spec=Writer)
        writer.get_sqso_target_skip_code.return_value = [' 4000 CONTINUE']
        calls = {wf['number']:
            ('CALL VERTEX(PL(0,{loop_mother_number}),COEFS)\n'
             'CALL %(proc_prefix)sUPDATE_WL_1_1(WL(1,0,1,{loop_mother_number}),'
             '4,COEFS,4,4,WL(1,0,1,{out}))').format(**wf.get_helas_call_dict())
                 for wf in (first, final)}
        writer.get_wavefunction_call.side_effect = lambda wf: calls[wf['number']]
        before = Writer.get_coef_construction_calls(writer, me)[0]
        selected = Writer.get_coef_construction_calls(writer, me,
            terminal_wavefunctions=Writer.get_terminal_wavefunctions(me))[0]
        after = Writer.get_coef_construction_calls(writer, me)[0]
        self.assertEqual(before, after)
        self.assertEqual(sum('TERMINAL_WL_' in call for call in selected), 1)
        self.assertEqual(selected[-1], ' 4000 CONTINUE')
        calls[2] = calls[2].replace('CALL %(proc_prefix)sUPDATE', ' 4000 CALL %(proc_prefix)sUPDATE')
        rejected = Writer.get_coef_construction_calls(writer, me,
            terminal_wavefunctions=Writer.get_terminal_wavefunctions(me))[0]
        self.assertFalse(any('TERMINAL_WL_' in call for call in rejected))


class RenderedTerminalGraphTest(unittest.TestCase):
    def write_ieee_batches(self, work, prefix):
        (work/'batch.inc').write_text(
            '      SUBROUTINE %(bunch_name)s_%(bunch_number)d()\n'
            '%(terminal_ieee_use)s      IMPLICIT NONE\n'
            '%(terminal_ieee_declarations)s%(helas_calls)s\n      END\n')
        exporter = object.__new__(Exporter)
        exporter.template_dir = str(work)
        replacement = {'proc_prefix': prefix,
            'terminal_ieee_use': '      USE, INTRINSIC :: IEEE_EXCEPTIONS\n',
            'terminal_ieee_declarations': '      LOGICAL WL_ENTRY_INEXACT\n'}
        before = os.getcwd()
        try:
            os.chdir(work)
            exporter.split_HELASCALLS(True, replacement, 'batch.inc', '',
                ['CALL FULL()', 'CALL IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)\n'
                 'CALL MP_%sTERMINAL_WL_1_1()' % prefix, 'CALL FULL()'],
                'entry', 'batch', n_helas=1)
        finally:
            os.chdir(before)

    def test_ieee_boundary_only_in_batches_with_private_calls(self):
        for prefix in ('TEST_', ''):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                self.write_ieee_batches(work, prefix)
                self.assertNotIn('IEEE_', (work/'batch_1.f').read_text())
                self.assertIn('IEEE_EXCEPTIONS', (work/'batch_2.f').read_text())
                self.assertIn('LOGICAL WL_ENTRY_INEXACT', (work/'batch_2.f').read_text())
                self.assertIn('CALL IEEE_GET_FLAG', (work/'batch_2.f').read_text())
                self.assertNotIn('IEEE_', (work/'batch_3.f').read_text())

    @unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
    def test_empty_nlo_prefix_and_prefixed_batches_compile_with_static_locals(self):
        for prefix in ('TEST_', ''):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                self.write_ieee_batches(work, prefix)
                built = subprocess.run(['gfortran', '-fno-automatic',
                    '-ffixed-line-length-132', '-c', 'batch_1.f',
                    'batch_2.f', 'batch_3.f'], cwd=work, capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stdout+built.stderr)

    def fixture(self, work):
        (work/'mp_coef_construction_1.f').write_text(
            '      CALL MP_TEST_UPDATE_WL_1_1(WL(1,0,1,0),4,COEFS,4,4,\n'
            '     $ WL(1,0,1,1))\n'
            '      CALL MP_TEST_TERMINAL_WL_1_1(WL(1,0,1,1),4,COEFS,4,4,\n'
            '     $ WL(1,0,1,2),WL_ENTRY_INEXACT)\n'
            '      CALL MP_TEST_CREATE_LOOP_COEFS(WL(1,0,1,2),2,4,1)\n')
        (work/'polynomial.f').write_text(
            '      SUBROUTINE MP_TEST_MERGE_WL(WL,R,LCUT_SIZE,CONST,OUT)\n'
            '      COMPLEX*32 WL(4,0:34,4)\n'
            '      OUT(J)=OUT(J)+WL(I,J,I)*CONST\n'
            '      END\n'
            '      SUBROUTINE MP_TEST_CREATE_LOOP_COEFS(LOOP_WF,R,LCUT)\n'
            '      COMPLEX*32 LOOP_WF(4,0:34,4)\n'
            '      CALL MP_TEST_MERGE_WL(LOOP_WF,R,LCUT,CONST,OUT)\n'
            '      END\n')
        return {2: dict(loop_mother_rank=1, vertex_rank=1, loop_mother_number=1,
                        lcut_size=4, in_size=4, out_size=4, out=2)}

    def test_known_diagonal_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            expected = self.fixture(work)
            self.assertTrue(Exporter.validate_terminal_wavefunctions(tmp, 'TEST_', expected))
            batch = work/'mp_coef_construction_1.f'
            original = batch.read_text()
            batch.write_text(' 4000 CONTINUE\n'+original+' 1001 CONTINUE\n')
            self.assertTrue(Exporter.validate_terminal_wavefunctions(tmp, 'TEST_', expected))
            self.assertEqual(batch.read_text(), ' 4000 CONTINUE\n'+original+' 1001 CONTINUE\n')

    def test_hidden_common_consumer_rejected(self):
        for consumer in ('X=WL(1,0,2,2)', 'CALL HIDDEN(WL)', 'CALL HIDDEN(WL(1,0,1,2))',
                         'X=WL', 'EQUIVALENCE(WL,X)'):
            with self.subTest(consumer=consumer), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                expected = self.fixture(work)
                (work/'hidden.f').write_text(
                    '      COMMON/TEST_MP_WL/WL,PL\n      '+consumer+'\n')
                self.assertFalse(Exporter.validate_terminal_wavefunctions(tmp, 'TEST_', expected))

    def test_common_alias_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            expected = self.fixture(work)
            (work/'hidden.f').write_text('      COMMON/TEST_MP_WL/X,Y\n      CALL HIDDEN(X)\n')
            self.assertFalse(Exporter.validate_terminal_wavefunctions(tmp, 'TEST_', expected))

    def test_bare_array_batch_consumer_rejected(self):
        for consumer in ('CALL HIDDEN(WL)', 'EQUIVALENCE(WL,X)'):
            with self.subTest(consumer=consumer), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                expected = self.fixture(work)
                with (work/'mp_coef_construction_1.f').open('a') as stream:
                    stream.write('      '+consumer+'\n')
                self.assertFalse(Exporter.validate_terminal_wavefunctions(tmp, 'TEST_', expected))

    def test_later_update_and_changed_merge_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            expected = self.fixture(work)
            with (work/'mp_coef_construction_1.f').open('a') as stream:
                stream.write('      CALL MP_TEST_UPDATE_WL_1_1(WL(1,0,1,2),4,COEFS,4,4,\n'
                             '     $ WL(1,0,1,3))\n')
            self.assertFalse(Exporter.validate_terminal_wavefunctions(tmp, 'TEST_', expected))
            self.fixture(work)
            polynomial = work/'polynomial.f'
            polynomial.write_text(polynomial.read_text().replace('WL(I,J,I)', 'WL(I,J,1)'))
            self.assertFalse(Exporter.validate_terminal_wavefunctions(tmp, 'TEST_', expected))

    def test_extra_create_argument_or_replaced_consumer_rejected(self):
        for substitution in ('extra', 'replace', 'merge'):
            with self.subTest(substitution=substitution), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                expected = self.fixture(work)
                if substitution == 'extra':
                    batch = work/'mp_coef_construction_1.f'
                    batch.write_text(batch.read_text().replace('),2,4,1)', '),2,4,WL)'))
                else:
                    polynomial = work/'polynomial.f'
                    text = polynomial.read_text()
                    if substitution == 'replace':
                        text = text.replace('CALL MP_TEST_MERGE_WL(', 'CALL HIDDEN(')
                    else:
                        text = text.replace('      OUT(J)=', '      CALL HIDDEN(WL)\n      OUT(J)=')
                    polynomial.write_text(text)
                self.assertFalse(Exporter.validate_terminal_wavefunctions(tmp, 'TEST_', expected))
