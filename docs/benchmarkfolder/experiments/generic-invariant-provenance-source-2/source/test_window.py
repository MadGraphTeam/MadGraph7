"""Pure ordered-window regressions; no scientific execution."""
from pathlib import Path
import sys,unittest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import madgraph.loop
madgraph.loop.__path__.insert(0,str(HERE/'src/madgraph/loop'))
from madgraph.loop.invariant_environment_source import Library,fixed
from madgraph.loop.invariant_window import momentum_window

def f(s):return '\n'.join('      '+line for line in s.strip().splitlines())+'\n'

def fixture(before='',after='',extra=''):
 text=f('SUBROUTINE B(P,W)\nUSE OBJ\nREAL P(0:3),X\nTYPE(NODE) W(1)\n'+before+'\nCALL MAKE(P,W(1))\n'+after+'\nEND')
 defs=f('MODULE OBJ\nTYPE NODE\nREAL P(0:3),W(4)\nEND TYPE\nEND MODULE\nSUBROUTINE MAKE(P,O)\nUSE OBJ\nREAL P(0:3)\nTYPE(NODE) O\nO%P(:)=P(:)\nEND\n'+extra)
 raw={'Source/DHELAS/a.f':defs,'SubProcesses/P0/b.f':text}
 receipt=[{'caller':'b.f','line':line,'call':s} for _,line,s in fixed(text,'x') if s.startswith('CALLMAKE')]
 return raw,[('b.f',text)],receipt

class Window(unittest.TestCase):
 def test_safe_prefix_write_after_final_momentum(self):
  args=fixture(after="WRITE(*,*) 'diagnostic'",before='X=SQRT(2.)')
  got=momentum_window(*args);self.assertTrue(got['complete']);self.assertFalse(got['suffix']['complete']);self.assertTrue(got['suffix']['reference_complete']);self.assertFalse(got['scientific_admission'])
 def test_interleaved_write_is_unknown(self):
  got=momentum_window(*fixture(before="WRITE(*,*) 'diagnostic'"));self.assertFalse(got['complete']);self.assertIn('prefix',got['reason'])
 def test_prefix_function_setter_not_hidden(self):
  got=momentum_window(*fixture(before='X=F()',extra='REAL FUNCTION F()\nCALL FESETROUND(0)\nF=0\nEND'));self.assertFalse(got['complete']);self.assertIn('mutator',str(got))
 def test_unknown_later_effects_not_ignored(self):
  got=momentum_window(*fixture(after='X=FOREIGN()'));self.assertFalse(got['complete']);self.assertIn('later',got['reason'])
 def test_later_function_momentum_overwrite(self):
  got=momentum_window(*fixture(after='X=F(W(1))',extra='REAL FUNCTION F(A)\nUSE OBJ\nTYPE(NODE) A\nA%P(0)=1.\nF=0\nEND'));self.assertFalse(got['complete']);self.assertTrue(got['later_writes'])
 def test_later_whole_object_overwrite(self):
  got=momentum_window(*fixture(after='W(1)=W(1)'));self.assertFalse(got['complete']);self.assertTrue(got['later_writes'])
 def test_numeric_dummy_receives_momentum_component(self):
  got=momentum_window(*fixture(after='CALL MUTATE(W(1)%P)',extra='SUBROUTINE MUTATE(P)\nREAL P(0:3)\nP(0)=1.\nEND'));self.assertFalse(got['complete']);self.assertTrue(got['later_writes'])
 def test_forwarded_whole_object_to_numeric_component(self):
  extra='SUBROUTINE FORWARD(A)\nUSE OBJ\nTYPE(NODE) A\nCALL MUTATE(A%P)\nEND\nSUBROUTINE MUTATE(P)\nREAL P(0:3)\nP(0)=1.\nEND'
  got=momentum_window(*fixture(after='CALL FORWARD(W(1))',extra=extra));self.assertFalse(got['complete']);self.assertTrue(got['later_writes'])
 def test_function_numeric_dummy_receives_momentum(self):
  extra='REAL FUNCTION F(P)\nREAL P(0:3)\nP(0)=1.\nF=0\nEND'
  got=momentum_window(*fixture(after='X=F(W(1)%P)',extra=extra));self.assertFalse(got['complete']);self.assertTrue(got['later_writes'])
 def test_proved_readonly_numeric_dummy_is_allowed(self):
  extra='REAL FUNCTION F(P)\nREAL P(0:3)\nF=P(0)\nEND'
  got=momentum_window(*fixture(after='X=F(W(1)%P)',extra=extra));self.assertTrue(got['complete'],got)
 def test_numeric_common_reinterpretation_unknown(self):
  raw,batches,receipt=fixture(after='CALL MUTATE()',extra='SUBROUTINE MUTATE()\nREAL P(8)\nCOMMON/BLOCK/P\nP(1)=1.\nEND')
  old=batches[0][1];new=old.replace('REAL P(0:3),X','REAL P(0:3),X\n      COMMON/BLOCK/W')
  raw['SubProcesses/P0/b.f']=new;batches=[('b.f',new)];receipt[0]['line']+=1
  got=momentum_window(raw,batches,receipt);self.assertFalse(got['complete']);self.assertIn('COMMON storage',got['reason'])
 def test_loop_counter_dummy_is_a_write(self):
  extra='SUBROUTINE MUTATE(P)\nREAL P\nDO P=1,2\nENDDO\nEND'
  got=momentum_window(*fixture(after='CALL MUTATE(W(1)%P(0))',extra=extra));self.assertFalse(got['complete']);self.assertTrue(got['later_writes'])
 def test_later_numerator_component_preserved(self):
  got=momentum_window(*fixture(after='W(1)%W(1)=2.'));self.assertTrue(got['complete'])
 def test_malformed_receipt_or_unknown_batch(self):
  raw,batches,receipt=fixture();receipt[0]['line']+=1
  self.assertFalse(momentum_window(raw,batches,receipt)['complete'])
  self.assertFalse(momentum_window(raw,batches,[])['complete'])
 def test_later_io_function_is_followed(self):
  got=momentum_window(*fixture(after="WRITE(*,*) F(W(1))",extra='REAL FUNCTION F(A)\nUSE OBJ\nTYPE(NODE) A\nA%P(0)=1.\nF=0\nEND'));self.assertFalse(got['complete']);self.assertFalse(got['suffix']['reference_complete']);self.assertIn('derived-object I/O unsupported',str(got))
if __name__=='__main__':unittest.main(verbosity=2)
