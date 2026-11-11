import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('stress_report_prepare',HERE/'prepare.py')
P=importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def fixture_rows():
    return [{'worker':worker,'phase':phase,'point':point,'order':order,'value':'1.25'}
            for worker in P.WORKERS for phase,point in ((0,1),(1,20)) for order in (0,1)]


class SelectionTests(unittest.TestCase):
    def test_orders_collapse_once(self):
        rows=P.duplicate_order_pairs(fixture_rows(),('value',))
        self.assertEqual(len(rows),4)
        self.assertTrue(all(r['order']==0 for r in rows))

    def test_missing_order_rejected(self):
        rows=fixture_rows()[:-1]
        with self.assertRaises(ValueError):
            P.duplicate_order_pairs(rows,('value',))

    def test_unequal_duplicate_rejected(self):
        rows=fixture_rows()
        rows[1]['value']='1.26'
        with self.assertRaises(ValueError):
            P.duplicate_order_pairs(rows,('value',))

    def test_other_worker_cannot_replace_intended_population(self):
        rows=fixture_rows()
        rows[0]['worker']='cms-p20-mode4-lib8-reference'
        with self.assertRaises(ValueError):
            P.duplicate_order_pairs(rows,('value',))

    def test_json_duplicate_and_nonfinite_tokens_rejected(self):
        for text in ('{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'):
            with self.assertRaises(ValueError):
                P.parsed(text)

    def test_unsafe_paths_rejected(self):
        for name in ('../source.json','/source.json','x/../y','x\\y',''):
            with self.assertRaises(ValueError):
                P.safe(name)

    def test_index_identity_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            (p/'evidence-index.json').write_text('{"files":{}}\n')
            with self.assertRaises(ValueError):
                P.Packet(p,'0'*64)

    def test_no_native_execution_import_or_call_in_generator(self):
        import ast
        tree=ast.parse((HERE/'prepare.py').read_text())
        names={alias.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))
               for alias in (n.names if isinstance(n,ast.Import) else [])}
        self.assertFalse(names & {'ctypes','subprocess','os'})
        forbidden={'eval','exec','compile','system','popen','spawn','CDLL','PyDLL'}
        self.assertFalse(any(isinstance(n,ast.Call) and
                             (isinstance(n.func,ast.Name) and n.func.id in forbidden
                              or isinstance(n.func,ast.Attribute) and n.func.attr in forbidden)
                             for n in ast.walk(tree)))


class BoundDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.value=P.derive(P.ROOT/'docs/benchmarkfolder/experiments')

    def test_data_reproduces_exactly(self):
        self.assertEqual(P.encoded(self.value),(HERE/'data.json').read_text())

    def test_ineligible_legacy_buffers_absent_from_derived_data(self):
        blocked=self.value['blocked_legacy_references']
        self.assertEqual([r['backend'] for r in blocked],[6,8])
        for row in blocked:
            self.assertFalse(row['eligible'])
            self.assertFalse(row['coefficient_arithmetic_performed'])
            self.assertFalse(any(k in row for k in ('raw_rows','coefficients','words','finite')))

    def test_failed_and_unqualified_gates_remain_false(self):
        stage=self.value['stage_status']
        self.assertFalse(stage['candidate_comparison_gate_passed'])
        self.assertFalse(stage['full_stress_passed'])
        self.assertEqual(stage['stress_workers_unlaunched'],7)
        for key in ('runtime_coupling_bits_observed','new_candidate_external_word_capture',
                    'UV_cancellation_separately_measured','finite_term_adjudicated',
                    'normal_host_stability_qualified','performance_qualified'):
            self.assertFalse(self.value['assumptions'][key])

    def test_only_eligible_fixed_qp_rows_plotted(self):
        rows=self.value['pole_comparisons']
        self.assertEqual(len(rows),8)
        self.assertEqual({r['worker'] for r in rows},set(P.WORKERS))
        self.assertEqual({(r['point'],r['pole']) for r in rows},{(1,-2),(20,-2),(1,-1),(20,-1)})
        self.assertTrue(all(r['return_code']==120 and r['eligible_fixed_mode']
                            and not r['normal_host_stability_qualified'] for r in rows))


if __name__=='__main__':
    unittest.main()
