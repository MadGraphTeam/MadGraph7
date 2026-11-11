"""Meaningful source-only contracts for the current private experiment."""
import json
from pathlib import Path
import re
import tempfile
import unittest

from admission import HERE, OLD, ROOT, historical_modules, read_json, sha
from prepare_inputs import card_derivative, card_values, point_slice
from trace_current import derive
from parity import fixed_mode_contract, trace_modes
from native_job import completion_ok
from run import source_inventory, exact_inventory


class CurrentSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='current-source-test-',dir=HERE.parent)
        cls.destination=Path(cls.temp.name)/'trace'
        binding=HERE/'ew-source-bindings.json'
        cls.report=derive(binding,sha(binding.read_bytes()),[5],cls.destination)
        cls.saved,cls.emitter=historical_modules()

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_current_insertions_strip_to_exact_qualified_source(self):
        for item in self.report['records']:
            filename=Path(item['input']).name
            result=(self.destination/filename).read_text()
            self.assertEqual(self.emitter.strip_trace(result).encode(),(ROOT/item['input']).read_bytes())
            blocks=re.findall(re.escape(self.emitter.BEGIN)+r'(.*?)'+re.escape(self.emitter.END),result,re.S)
            self.assertTrue(blocks)
            for block in blocks:
                self.assertTrue(all(len(line)<=72 for line in block.splitlines()))
                self.assertNotRegex(block,r'\b(?:PL|MP_PL|M2L|SHIFTS|INVARIANTS|S_MAT|DIFFSQ)\s*(?:\([^\n]*\))?\s*=')
                self.assertNotRegex(block,r'\b(?:RAW|TIRCOEFS|LOOPRES)\(')

    def test_current_group_and_complex_words_are_explicit(self):
        target=self.report['target_identities'][0]
        self.assertEqual(target['reduction_identifier'],5)
        self.assertEqual(tuple(target['source_comment_numbers']),(9,11))
        self.assertEqual(target['identity_kind'],'group')
        source=(self.destination/'CT_interface.f').read_text()
        self.assertIn('KT_ACTIVE=KT_ENABLED.AND.(',source)
        self.assertIn('LOOPNUM.EQ.5)',source)
        self.assertIn('TRANSFER(W(KT_EXT)%P,0_8,8)',source)
        self.assertIn('TRANSFER(MP_W(KT_EXT)%P,0_8,16)',source)
        tensor=(self.destination/'TIRUST_interface.f').read_text()
        self.assertIn('TRANSFER(P_TMP(:,0),0_8,4)',tensor)
        self.assertIn('TRANSFER(P_TMP(:,0),0_8,8)',tensor)
        self.assertIn("'KT_STATUS_DP'",tensor)
        self.assertIn("'KT_STATUS_QP'",tensor)

    def test_card_overrides_leave_every_other_field_identical(self):
        inputs=read_json(OLD/'input-bindings.json')
        for key in ('cms','ew'):
            raw=(ROOT/inputs[key]['numerical_card']['path']).read_bytes()
            _,before=card_values(raw)
            for mode in (1,4):
                output,changes=card_derivative(raw,mode);_,after=card_values(output)
                for name,entry in before.items():
                    expected=changes['overrides'].get(name,entry['value'])
                    self.assertEqual(after[name]['value'],expected)
                self.assertEqual(after['looppolemode']['value'],'0')
                self.assertEqual(after['osthres']['value'],before['osthres']['value'])
        with self.assertRaises(ValueError):card_derivative(raw,5)

    def test_point_slices_preserve_exact_source_decimal_bytes(self):
        inputs=read_json(OLD/'input-bindings.json')
        for key,nlegs,point in (('cms',4,20),('cms',4,21),('ew',5,2)):
            raw=(ROOT/inputs[key]['points']['path']).read_bytes()
            lines=raw.splitlines(keepends=True)
            self.assertEqual(point_slice(raw,nlegs,point),
                             b'1\n'+b''.join(lines[1+(point-1)*nlegs:1+point*nlegs]))
        with self.assertRaises(ValueError):point_slice(b'1\n1 2 3\n',1,1)

    def test_restored_init_flags_do_not_invent_readiness_failure(self):
        def row(stage,phase,call,code,actual):
            return f'STATE {stage} {phase} {call} 1 4 {code} {actual} 1 0 0 0 00000001 00000000 00000000 00000000\n'
        raw=row('B',0,1,-987654321,-999)+row('A',0,1,120,4)+row('B',1,2,-987654321,-999)+row('A',1,2,120,4)
        self.assertEqual(len(fixed_mode_contract(raw,1,4)),4)
        with self.assertRaises(ValueError):fixed_mode_contract(raw.replace('120 4','110 1'),1,4)
        with self.assertRaises(ValueError):fixed_mode_contract(raw+row('A',1,2,120,4),1,4)

    def test_tensor_success_code_is_not_complete_trace_alone(self):
        path=Path(self.temp.name)/'partial-trace.dat'
        path.write_text('KT_SCHEMA_V2 2 64 128 64 32\nKT_STATUS_DP 2 1 5 1 2 4 0 0 0 1 0\n')
        with self.assertRaises(ValueError):trace_modes(path,2,1,[5])
        path.write_text('KT_SCHEMA_V2 2 64 128 64 32\nKT_STATUS_DP 2 1 5 2 2 4 0 0 0 1 0\n')
        with self.assertRaises(ValueError):trace_modes(path,2,1,[5])

    def test_stdout_overflow_after_last_protocol_reply_still_rejects(self):
        captures=[{'boundary':'READY'},{'boundary':'DONE'}]
        self.assertTrue(completion_ok(None,None,True,captures,[],[]))
        self.assertFalse(completion_ok(None,None,True,captures,[33554433],[]))
        self.assertFalse(completion_ok(None,None,True,captures[:1],[],[]))
        self.assertFalse(completion_ok(None,None,True,captures,[],[{'file':'trace.dat'}]))

    def test_extra_unreviewed_generated_source_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='inventory-test-',dir=HERE.parent) as tmp:
            export=Path(tmp)
            (export/'Source').mkdir();(export/'SubProcesses').mkdir()
            (export/'Source'/'original.f').write_text('      END\n')
            expected=source_inventory(export)
            exact_inventory(export,expected)
            (export/'Source'/'unreviewed.f').write_text('      END\n')
            with self.assertRaises(ValueError):exact_inventory(export,expected)


if __name__=='__main__':unittest.main()
