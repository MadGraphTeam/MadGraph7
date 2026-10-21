"""Synthetic direct-mode parser checks only; no process/model/native imports."""
import ast,math,tempfile,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parent
source=(BASE/'prepare.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='parse_direct')
namespace={'math':math};exec(compile(ast.Module(body=[node],type_ignores=[]),'retained-parse-direct','exec'),namespace);parse=namespace['parse_direct']
class DirectParser(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=BASE);self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
    def put(self,dims=(1,1,0),mode=4):
        code=110 if mode in(1,2) else 120
        (self.path/'dimensions.dat').write_text(' '.join(map(str,dims)))
        (self.path/'values.dat').write_text(''.join(f'{p} {o} {code} 0 1 -2 3 -1\n' for p in range(1,11) for o in range(dims[1]+1)))
        (self.path/'timed_status.dat').write_text(f'{code} 10\n')
    def test_real_hh_complex_dimensions_and_both_directions_precisions(self):
        for dims in((1,1,0),(1,1,1)):
            for mode in(1,2,4,5):
                self.put(dims,mode);result=parse(self.path,dims,mode)
                self.assertEqual(result['answer_dimensions']['born_orders'],dims[2]);self.assertEqual(len(result['rows']),20)
                self.assertFalse(result['stability_measurement']);self.assertFalse(result['automatic_rescue']);self.assertFalse(result['performance_acceptance_claim'])
    def test_complex_born_dimension_cannot_be_dropped(self):
        self.put((1,1,0))
        with self.assertRaisesRegex(ValueError,'dimensions'):parse(self.path,(1,1,1),4)
    def test_exact_order_and_complete_rows_required(self):
        for mutate in(lambda lines:lines[:-1],lambda lines:list(reversed(lines)),lambda lines:lines+[lines[-1]]):
            self.put();p=self.path/'values.dat';p.write_text(''.join(mutate(p.read_text().splitlines(True))))
            with self.assertRaisesRegex(ValueError,'physical rows'):parse(self.path,(1,1,0),4)
    def test_nonfinite_wrong_code_and_fake_accuracy_rejected(self):
        for old,new in((' 1 -2 ',' nan -2 '),(' 120 ',' 218 '),(' 3 -1\n',' 3 0\n')):
            self.put();p=self.path/'values.dat';p.write_text(p.read_text().replace(old,new,1))
            with self.assertRaisesRegex(ValueError,'code/accuracy or nonfinite'):parse(self.path,(1,1,0),4)
    def test_exact_all_evaluations_histogram_and_mode_required(self):
        for histogram in('120 9\n','120 5\n120 5\n','110 10\n'):
            self.put();(self.path/'timed_status.dat').write_text(histogram)
            with self.assertRaisesRegex(ValueError,'histogram'):parse(self.path,(1,1,0),4)
        with self.assertRaisesRegex(ValueError,'direct mode'):parse(self.path,(1,1,0),3)
if __name__=='__main__':unittest.main()
