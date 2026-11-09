"""In-memory source mutations against the complete retained export; no edits."""
from pathlib import Path
import importlib.util, sys, unittest
from unittest import mock
HERE=Path(__file__).resolve().parent
PROCESS=Path(sys.argv.pop(1)).resolve()
spec=importlib.util.spec_from_file_location('higher_proof',HERE/'dp_terminal_proof.py')
proof=importlib.util.module_from_spec(spec);spec.loader.exec_module(proof)

class FullClosureMutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.expected=proof.collect_expected(PROCESS)
    def check(self): return proof.validate_dp_terminal_wavefunctions(PROCESS,'ML5_0_',self.expected)
    def test_actual_full_closure(self):
        self.assertEqual(len(self.expected),147)
        self.assertTrue(self.check())
    def test_table_write_alias_escape_and_shadow_in_other_units(self):
        real=proof._expanded
        for name,text in (
            ('loop_matrix.f','COMB_COEF_POS(0,0)=1'),
            ('MadLoopCommons.f','CALL MUTATE(COMB_COEF_POS)'),
            ('MadLoopCommons.f','USE ML5_0_POLYNOMIAL_CONSTANTS,ONLY:ALIAS=>COMB_COEF_POS'),
            ('loop_matrix.f','INTEGER COMB_COEF_POS(0:69,0:4)'),
        ):
            with self.subTest(name=name,text=text):
                def changed(unit,export,roots):
                    rows=real(unit,export,roots)
                    if unit.name==name: rows.append(proof._Row(unit,1,text,''))
                    return rows
                with mock.patch.object(proof,'_expanded',side_effect=changed): self.assertFalse(self.check())
    def test_missing_capture_and_extra_wavefunction_consumer(self):
        real=proof._expanded
        for mode in ('capture','consumer'):
            with self.subTest(mode=mode):
                def changed(unit,export,roots):
                    rows=real(unit,export,roots)
                    if unit.name=='coef_construction_1.f':
                        if mode=='capture':
                            rows=[r._replace(text='CONTINUE') if proof._compact(r.text)=='CALLIEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)' else r for r in rows]
                        else: rows.insert(-1,proof._Row(unit,1,'CALL UNKNOWN(WL(1,0,1,7))',''))
                    return rows
                with mock.patch.object(proof,'_expanded',side_effect=changed): self.assertFalse(self.check())
    def test_wrong_rank_or_missing_expected_producer(self):
        expected=dict(self.expected);expected.pop(next(iter(expected)))
        self.assertFalse(proof.validate_dp_terminal_wavefunctions(PROCESS,'ML5_0_',expected))
        expected={k:dict(v) for k,v in self.expected.items()}
        expected[next(iter(expected))]['loop_mother_rank']=4
        self.assertFalse(proof.validate_dp_terminal_wavefunctions(PROCESS,'ML5_0_',expected))

if __name__=='__main__': unittest.main()
