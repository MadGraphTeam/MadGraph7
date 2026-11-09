"""Source-only mutation tests for the private higher-rank table proof."""
from pathlib import Path
import importlib.util
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('private_higher_proof', HERE / 'dp_terminal_proof.py')
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)

class CombinationTableProof(unittest.TestCase):
    def setUp(self):
        self.rows = proof._expanded(HERE / 'table-fixture.f', HERE, [])

    def mutate(self, old, new):
        found = False
        rows = []
        for row in self.rows:
            if proof._compact(row.text) == old:
                row = row._replace(text=new)
                found = True
            rows.append(row)
        self.assertTrue(found)
        return rows

    def test_complete_literal_table_and_last_consumed_row(self):
        certified = proof._canonical_comb(self.rows, 'ML5_0_')
        self.assertEqual(len(certified), 71)
        self.assertTrue(any(proof._compact(r.text) == 'DATACOMB_COEF_POS(34,0:4)/34,66,67,68,69/' for r in certified))
        proof._readonly_comb({'unit': self.rows}, certified)

    def test_mutated_table_extent_value_and_counts_rejected(self):
        changes = [
            ('INTEGERCOMB_COEF_POS(0:LOOPMAXCOEFS-1,0:4)', 'INTEGERCOMB_COEF_POS(1:LOOPMAXCOEFS,0:4)'),
            ('INTEGERCOMB_COEF_POS(0:LOOPMAXCOEFS-1,0:4)', 'INTEGERCOMB_COEF_POS(0:LOOPMAXCOEFS-1,1:5)'),
            ('DATACOMB_COEF_POS(0,0:4)/0,1,2,3,4/', 'DATACOMB_COEF_POS(0,0:4)/1,1,2,3,4/'),
            ('DATACOMB_COEF_POS(34,0:4)/34,66,67,68,69/', 'DATACOMB_COEF_POS(34,0:4)/34,66,67,68,70/'),
            ('DATACOMB_COEF_POS(69,0:4)/69,122,123,124,125/', 'DATACOMB_COEF_POS(69,0:4)/69,122,123,124,124/'),
            ('INTEGERNCOEF_R(0:4)', 'INTEGERNCOEF_R(1:5)'),
            ('DATANCOEF_R/1,5,15,35,70/', 'DATANCOEF_R/1,5,15,36,70/'),
            ('PARAMETER(LOOPMAXCOEFS=70)', 'PARAMETER(LOOPMAXCOEFS=71)'),
        ]
        for old,new in changes:
            with self.subTest(old=old):
                with self.assertRaises(ValueError): proof._canonical_comb(self.mutate(old,new), 'ML5_0_')
        with self.assertRaises(ValueError): proof._canonical_comb(self.rows[:-1] + [next(r for r in self.rows if 'DATA COMB_COEF_POS' in r.text)] + self.rows[-1:], 'ML5_0_')

    def test_writes_aliases_shadowing_and_argument_escapes_rejected(self):
        certified = proof._canonical_comb(self.rows, 'ML5_0_')
        for text in (
            'COMB_COEF_POS(0,0)=1', 'CALL EVIL(COMB_COEF_POS)',
            'USE ML5_0_POLYNOMIAL_CONSTANTS, ONLY: ALIAS=>COMB_COEF_POS',
            'INTEGER COMB_COEF_POS(0:69,0:4)', 'EQUIVALENCE(COM B_COEF_POS,ALIAS)',
            'P=C_LOC(COMB_COEF_POS)', 'COMMON/EVIL/COMB_COEF_POS',
            'NEW_POSITION=COMB_COEF_POS(L,EVIL())',
        ):
            with self.subTest(text=text):
                row=proof._Row(HERE/'other.f', 1, text, '')
                with self.assertRaises(ValueError): proof._readonly_comb({'unit':self.rows+[row]}, certified)
        for text in ('NEW_POSITION=COMB_COEF_POS(L,M)', 'NEW_POSITION=COMB_COEF_POS(L,4)'):
            proof._readonly_comb({'unit':self.rows+[proof._Row(HERE/'other.f',1,text,'')]}, certified)

if __name__ == '__main__': unittest.main()
