"""Private tensor contractions require explicit shapes and preserved consumers."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

from madgraph.loop.loop_exporters import LoopProcessOptimizedExporterFortranSA as Exporter
from tests.unit_tests.loop import test_terminal_wavefunctions as terminal_tests


class FactorizedWavefunctionExportTest(unittest.TestCase):
    @staticmethod
    def block(prefix='TEST_', mode='UPDATE', rank=2, vertex='FFV1L2_1',
              origin='1', target='2', cut=4, ins=4, outs=4, flag=None):
        if flag is None:
            flag = mode == 'TERMINAL'
        return ('CALL MP_%s(PL(0,1),V(1,1),GC_1,MDL_MT,ZERO,PL(0,2),COEFS)\n'
                'CALL MP_%s%s_WL_%d_1(WL(1,0,1,%s),%d,COEFS,%d,%d,'
                'WL(1,0,1,%s)%s)\n\n' %
                (vertex, prefix, mode, rank, origin, cut, ins, outs, target,
                 ',WL_ENTRY_INEXACT' if flag else ''))

    def test_selected_full_and_terminal_preserve_arguments_and_line_endings(self):
        for prefix in ('TEST_', ''):
            for mode in ('UPDATE', 'TERMINAL'):
                for vertex in ('FFV1L1_2', 'FFV1L2_1'):
                    with self.subTest(prefix=prefix, mode=mode, vertex=vertex):
                        before = self.block(prefix, mode, vertex=vertex)
                        after = Exporter.factorize_mp_coef_calls([before], prefix)[0]
                        name = 'MP_%s%s_WL_2_1' % (prefix, mode)
                        expected = before.replace(name+'(', name+'_FACTOR_'+vertex+'(')
                        if mode == 'UPDATE':
                            expected = expected.replace('WL(1,0,1,2))',
                                'WL(1,0,1,2),WL_ENTRY_INEXACT)')
                        self.assertEqual(after, expected)

    def test_unknown_shapes_ranks_aliases_flags_and_vertices_keep_original(self):
        controls = [dict(ins=3), dict(outs=3), dict(cut=5), dict(rank=1),
            dict(rank=5), dict(origin='2'), dict(origin='02', target='2'),
            dict(target='0'), dict(vertex='VVV1L2P0_1'), dict(flag=True),
            dict(mode='TERMINAL', flag=False), dict(prefix='LONG_'*8)]
        for options in controls:
            with self.subTest(options=options):
                before = self.block(**options)
                self.assertEqual(Exporter.factorize_mp_coef_calls(
                    [before], options.get('prefix', 'TEST_')), [before])

    def test_intervening_statement_or_separate_block_disables_selection(self):
        vertex, updater = self.block().splitlines()[:2]
        for middle in ('CALL OTHER(COEFS)', 'COEFS=0', 'X=1'):
            before = vertex+'\n'+middle+'\n'+updater+'\n'
            self.assertEqual(Exporter.factorize_mp_coef_calls([before], 'TEST_'), [before])
        blocks = [vertex+'\n', updater+'\n']
        self.assertEqual(Exporter.factorize_mp_coef_calls(blocks, 'TEST_'), blocks)

    def test_comments_between_vertex_and_updater_preserve_selection(self):
        before = self.block().replace('\nCALL MP_TEST_', '\n# retained comment\nCALL MP_TEST_')
        after = Exporter.factorize_mp_coef_calls([before], 'TEST_')[0]
        self.assertIn('# retained comment\n', after)
        self.assertIn('UPDATE_WL_2_1_FACTOR_FFV1L2_1(', after)

    @unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
    def test_factored_batches_compile_with_prefixed_and_empty_nlo_names(self):
        for prefix in ('TEST_', ''):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                (work/'batch.inc').write_text(
                    '      SUBROUTINE %(bunch_name)s_%(bunch_number)d()\n'
                    '%(terminal_ieee_use)s      IMPLICIT NONE\n'
                    '%(terminal_ieee_declarations)s%(helas_calls)s\n      END\n')
                exporter = object.__new__(Exporter)
                exporter.template_dir = str(work)
                replacement = {'proc_prefix': prefix,
                    'terminal_ieee_use': '      USE, INTRINSIC :: IEEE_EXCEPTIONS\n',
                    'terminal_ieee_declarations': '      LOGICAL WL_ENTRY_INEXACT\n'}
                calls = ['CALL FULL()'] + [
                    'CALL IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)\n'
                    'CALL MP_%s%s_WL_2_1_FACTOR_FFV1L2_1(WL_ENTRY_INEXACT)' %
                    (prefix, mode) for mode in ('UPDATE', 'TERMINAL')]
                before = os.getcwd()
                try:
                    os.chdir(work)
                    exporter.split_HELASCALLS(True, replacement, 'batch.inc', '',
                        calls, 'entry', 'batch', n_helas=1)
                finally:
                    os.chdir(before)
                self.assertNotIn('IEEE_', (work/'batch_1.f').read_text())
                for number in (2, 3):
                    self.assertIn('LOGICAL WL_ENTRY_INEXACT',
                                  (work/f'batch_{number}.f').read_text())
                built = subprocess.run(['gfortran', '-fno-automatic',
                    '-ffixed-line-length-132', '-c', 'batch_1.f', 'batch_2.f',
                    'batch_3.f'], cwd=work, capture_output=True, text=True)
                self.assertEqual(built.returncode, 0, built.stdout+built.stderr)

    def rendered_fixture(self, work):
        expected = terminal_tests.RenderedTerminalGraphTest().fixture(work)
        batch = work/'mp_coef_construction_1.f'
        text = batch.read_text().replace('UPDATE_WL_1_1(',
            'UPDATE_WL_2_1_FACTOR_FFV1L2_1(').replace('WL(1,0,1,1))',
            'WL(1,0,1,1),WL_ENTRY_INEXACT)')
        text = text.replace('TERMINAL_WL_1_1(', 'TERMINAL_WL_3_1_FACTOR_FFV1L1_2(')
        batch.write_text(text)
        expected[2]['loop_mother_rank'] = 3
        return expected

    def test_rendered_factored_producers_preserve_terminal_graph_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            expected = self.rendered_fixture(work)
            self.assertTrue(Exporter.validate_terminal_wavefunctions(tmp, 'TEST_', expected))
            batch = work/'mp_coef_construction_1.f'
            with batch.open('a') as stream:
                stream.write('      CALL MP_TEST_UPDATE_WL_2_1_FACTOR_FFV1L2_1(\n'
                    '     $ WL(1,0,1,2),4,COEFS,4,4,WL(1,0,1,3),WL_ENTRY_INEXACT)\n')
            self.assertFalse(Exporter.validate_terminal_wavefunctions(tmp, 'TEST_', expected))

    def test_missing_flag_unknown_factor_and_unsupported_shape_fail_proof(self):
        for change in ('flag', 'vertex', 'shape'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                expected = self.rendered_fixture(work)
                batch = work/'mp_coef_construction_1.f'
                text = batch.read_text()
                if change == 'flag':
                    text = text.replace(',WL_ENTRY_INEXACT)', ')', 1)
                elif change == 'vertex':
                    text = text.replace('_FACTOR_FFV1L2_1(', '_FACTOR_UNKNOWN(')
                else:
                    text = text.replace(',4,COEFS,4,4,', ',4,COEFS,3,4,', 1)
                batch.write_text(text)
                self.assertFalse(Exporter.validate_terminal_wavefunctions(tmp, 'TEST_', expected))


if __name__ == '__main__':
    unittest.main()
