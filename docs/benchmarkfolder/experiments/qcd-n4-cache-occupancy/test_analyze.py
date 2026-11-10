"""Synthetic parser checks only; these are not reducer qualification."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('occupancy_analyze', HERE/'analyze.py')
analyze=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


def fixture(root):
    text='OCC_PUBLIC 1\n'
    for i in range(2):
        text+=f'CALL {i} 219\n'
        for label,n in [('INPUT',25),('INPUT_AFTER',25),('ANS',8),('ACC',2)]:
            text+=label+' 0000000000000000'*n+'\n'
        text+='STATE'+' 0'*22+'\n'
    for arm in ('original','disabled','enabled'):
        folder=root/arm
        folder.mkdir()
        (folder/'public.txt').write_text(text)
        (folder/'occupancy.txt').write_text('OCCUPANCY 1 0 0 0 0\n')
    rows=[]
    # Existing clear establishes the context; successful tensor then fills it.
    for seq,tag,generation,query in [(1,20,-1,0),(2,21,1,1),(3,10,1,1),(4,20,1,1),(5,21,1,1)]:
        meta=[seq,0,tag,0,0,0,0,-999,-999,0,0,0,generation,query,0 if query else -999,0 if query else -999]
        counters=[0]*15
        if tag==10:
            meta[3:12]=[53,1,6,5,0,0,1,0,1]
        if seq in (3,4):
            counters[:9]=[3,8,4,9,5,10,11,67108860,2]
        rows.append(meta+counters+[0]*22)
    second=[row[:] for row in rows]
    for row in second:
        row[0]+=len(rows); row[1]=1
        if row[2]==20 and row[12]<0:
            row[12:16]=[1,1,0,0]
    rows+=second
    write_rows(root,rows)
    return rows


def write_rows(root,rows):
    (root/'enabled/occupancy.txt').write_text(f'OCCUPANCY 1 1 {len(rows)} 0 0\n'+
        ''.join(' '.join(map(str,row))+'\n' for row in rows))


class TestParser(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.rows=fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_success_and_quota_refusal_are_distinct(self):
        report=analyze.summarize(self.root)
        self.assertTrue(report['second_call_accepted'])
        self.assertEqual(report['full_dp_observed_peak']['not_stored'],2)
        self.assertEqual(report['full_dp_observed_peak']['estimated_bytes'],67108860)
        self.assertEqual(report['failed_evaluations_without_statistics'],0)

    def test_failure_is_recorded_without_query(self):
        row=self.rows[2]
        row[7]=1;row[13]=0;row[14:31]=[-999,-999]+[0]*15
        write_rows(self.root,self.rows)
        self.assertEqual(analyze.summarize(self.root)['failed_evaluations_without_statistics'],1)

    def test_qp_contributions_subtracted(self):
        for row in self.rows[2:4]:
            qp=[1,2,3,100,4,5]
            row[25:31]=qp
            row[20:25]=[v+q for v,q in zip(row[20:25],qp)]
        write_rows(self.root,self.rows)
        self.assertEqual(analyze.summarize(self.root)['full_dp_observed_peak']['estimated_bytes'],67108860)

    def test_tampering_rejected(self):
        variants={
            'query_failure': (2,14,2),
            'queried_failed_eval': (2,7,1),
            'created_context': (2,12,-1),
            'ieee_changed': (2,42,1),
            'wrong_finite_mode': (2,9,0),
            'cache_disabled': (2,11,0),
            'check_not_independent': (2,10,1),
            'clear_not_empty': (4,23,1),
            'negative_derived_dp_bytes': (2,28,67108861),
            'sequence': (2,0,20),
            'wrong_precision_direction': (2,3,113),
            'clear_not_sentinel': (0,3,53),
        }
        for name,(row,column,value) in variants.items():
            with self.subTest(name=name):
                changed=[r[:] for r in self.rows]
                changed[row][column]=value
                write_rows(self.root,changed)
                with self.assertRaises(ValueError): analyze.summarize(self.root)

    def test_missing_and_reordered_boundaries(self):
        variants=[self.rows[:5], self.rows[2:3]+self.rows[7:8],
                  self.rows[:3]+[self.rows[4],self.rows[3]]+self.rows[5:],
                  self.rows[:1]+[self.rows[2],self.rows[1]]+self.rows[3:]]
        for rows in variants:
            with self.subTest(rows=len(rows)):
                changed=[row[:] for row in rows]
                for seq,row in enumerate(changed,1): row[0]=seq
                write_rows(self.root,changed)
                with self.assertRaises(ValueError): analyze.summarize(self.root)

    def test_public_ieee_parity_is_required(self):
        path=self.root/'enabled/public.txt'
        path.write_text(path.read_text().replace('STATE 0','STATE 1',1))
        with self.assertRaises(ValueError): analyze.summarize(self.root)


if __name__=='__main__': unittest.main()
