import unittest,json,copy
from pathlib import Path
import prepare,analyze
P=Path(__file__).resolve().parent
class ProtocolTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.c=json.loads((P/'cases.json').read_text());cls.e=json.loads((P/'expected.json').read_text());cls.t=json.loads((P/'tokens.json').read_text());cls.raw=analyze.synthetic(cls.e,cls.t)
 def test_exact_generation(self):
  c,e,t=prepare.corpus();self.assertEqual(json.loads(json.dumps([c,e,t])),[self.c,self.e,self.t])
  a,b=prepare.render(c,t);self.assertEqual(a,(P/'inputs.txt').read_text());self.assertEqual(b,(P/'tokens.txt').read_text())
 def test_all104_and_negative_scope(self):
  c=self.c[:104];self.assertEqual({(x['route'],x['mode']) for x in c},{(r,m) for r in range(1,27) for m in (1,2,4,5)})
  self.assertEqual(len(self.c),174);self.assertTrue(any(not x['chords'][0]['have'] for x in self.e[104:]));self.assertTrue(any(x['chords'][0]['have'] for x in self.e[104:]))
 def test_positive_mock(self):self.assertEqual(analyze.parse(self.raw,self.e,self.t)['cases'],174)
 def reject(self,b):
  with self.assertRaises(ValueError):analyze.parse(b,self.e,self.t)
 def test_short_extra_and_duplicate(self):
  for b in (self.raw[:100],self.raw+b'DONE 174 11\n',self.raw.replace(b'C 1 1 1\n',b'C 1 1 1\nC 1 1 1\n')):self.reject(b)
 def test_bad_bound_class_or_route_gate(self):
  for old,new in [(b'C 1 1 1',b'C 1 0 1'),(b'P 1 1 1 1 0',b'P 1 1 1 2 0'),(b'-1000000',b'-999999')]:
   bad=self.raw.replace(old,new,1)
   self.assertTrue(bad!=self.raw,old.decode())
   self.reject(bad)
 def test_flags_and_environment(self):
  for old,new in [(b'E 1 T T FFFFT FFFFT',b'E 1 T T FFFFT FFFFF'),(b'E 1 T T',b'E 1 T F'),(b'ENV_BEGIN 4B4500001F80033F T',b'ENV_BEGIN 0000000000000000 T')]:
   bad=self.raw.replace(old,new,1)
   self.assertTrue(bad!=self.raw,old.decode())
   self.reject(bad)
 def test_environment_token_ftz_daz_rounding(self):
  good=analyze.MARK|(0x1f80<<16)|0x033f;self.assertTrue(analyze.environment_ok(good))
  for bit in (0x40<<16,0x8000<<16,0x2000<<16,0x400,0x800):self.assertFalse(analyze.environment_ok(good|bit))
 def test_lost_token_result(self):self.reject(self.raw.replace(b'T 1 T\n',b'T 1 F\n',1))
if __name__=='__main__':unittest.main()
