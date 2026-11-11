"""Pure semantic closure tests: no compiler, subprocess, loader or numerical work."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src/madgraph/loop'))
from invariant_environment_source import Library

def fixed(s):return '\n'.join('      '+line for line in s.strip().splitlines())+'\n'
def check(s,roots=('ROOT',),extra=None):
 raw={'source.f':fixed(s),**(extra or {})};return Library(raw).closure(roots)
class Closure(unittest.TestCase):
 def test_implicit_undeclared_foreign_function(self):
  r=check('SUBROUTINE ROOT()\nREAL X\nX=FOREIGN()\nEND');self.assertFalse(r['reference_complete']);self.assertIn('unresolved function FOREIGN',str(r))
 def test_implicit_wrapper_reaches_setter(self):
  r=check('SUBROUTINE ROOT()\nREAL X,F\nX=F()\nEND\nREAL FUNCTION F()\nCALL FESETROUND(0)\nF=0\nEND');self.assertFalse(r['complete']);self.assertIn('environment mutator',str(r));self.assertIn('F',r['procedures']['ROOT']['edges'])
 def test_user_function_no_external_needed(self):
  r=check('SUBROUTINE ROOT()\nREAL X,F\nX=F(2.)\nEND\nREAL FUNCTION F(X)\nREAL X\nF=SQRT(ABS(X))\nEND');self.assertTrue(r['complete']);self.assertEqual(r['procedures']['F']['intrinsics'],['ABS','SQRT'])
 def test_array_name_intrinsic_is_data(self):
  r=check('SUBROUTINE ROOT()\nREAL SQRT(2),X\nX=SQRT(1)\nEND');self.assertTrue(r['complete']);self.assertEqual(r['procedures']['ROOT']['intrinsics'],[])
 def test_scalar_declaration_is_not_array(self):
  r=check('SUBROUTINE ROOT()\nREAL X,ALIEN\nX=ALIEN(1)\nEND');self.assertFalse(r['complete'])
 def test_function_in_array_subscript(self):
  r=check('SUBROUTINE ROOT()\nREAL A(2),X\nX=A(FOREIGN())\nEND');self.assertFalse(r['complete']);self.assertIn('FOREIGN',str(r))
 def test_component_type_and_nested_subscript(self):
  s='MODULE OBJECT\nTYPE NODE\nREAL P(0:3)\nEND TYPE\nEND MODULE\nSUBROUTINE ROOT()\nUSE OBJECT\nTYPE(NODE) W(2)\nREAL X\nX=W(1)%P(INT(1.))\nEND'
  self.assertTrue(check(s)['complete']);self.assertFalse(check(s.replace('%P(INT(1.))','%OTHER(1)'))['complete']);self.assertFalse(check(s.replace('INT(1.)','HIDDEN()'))['complete'])
 def test_pointer_component_unknown_when_used(self):
  s='MODULE OBJECT\nTYPE NODE\nREAL,POINTER::P\nEND TYPE\nEND MODULE\nSUBROUTINE ROOT()\nUSE OBJECT\nTYPE(NODE) W\nREAL X\nX=W%P\nEND'
  self.assertFalse(check(s)['complete'])
 def test_contained_module_function(self):
  s='MODULE M\nCONTAINS\nREAL FUNCTION F(X) RESULT(Y)\nREAL X,Y\nY=SQRT(X)\nEND FUNCTION\nEND MODULE\nSUBROUTINE ROOT()\nUSE M,ONLY:F\nREAL X\nX=F(4.)\nEND'
  r=check(s);self.assertTrue(r['complete']);self.assertEqual(r['procedures']['ROOT']['edges'],['M::F'])
 def test_contained_internal_function_host_array(self):
  s='SUBROUTINE ROOT()\nREAL A(2),X\nX=F(1)\nCONTAINS\nREAL FUNCTION F(I)\nINTEGER I\nF=A(I)\nEND FUNCTION\nEND'
  r=check(s);self.assertTrue(r['complete']);self.assertEqual(r['procedures']['ROOT']['edges'],['ROOT::F'])
 def test_contained_wrapper_hidden_mutator(self):
  s='SUBROUTINE ROOT()\nREAL X\nX=F()\nCONTAINS\nREAL FUNCTION F()\nCALL UNKNOWN_ENV()\nF=0\nEND FUNCTION\nEND'
  self.assertFalse(check(s)['complete'])
 def test_external_dummy_and_procedure_actual_unknown(self):
  s='SUBROUTINE ROOT(F)\nREAL F,X\nEXTERNAL F\nX=F()\nEND\nREAL FUNCTION F()\nF=0\nEND'
  self.assertFalse(check(s)['complete'])
  s='SUBROUTINE ROOT()\nEXTERNAL F\nCALL S(F)\nEND\nSUBROUTINE S(X)\nREAL X\nEND\nREAL FUNCTION F()\nF=0\nEND'
  self.assertFalse(check(s)['complete'])
 def test_intrinsic_shadow_ambiguity(self):
  s='SUBROUTINE ROOT()\nREAL X\nX=SQRT(4.)\nEND\nREAL FUNCTION SQRT(X)\nREAL X\nSQRT=X\nEND'
  self.assertFalse(check(s)['complete']);self.assertTrue(check(s.replace('REAL X\nX=SQRT','REAL X\nEXTERNAL SQRT\nX=SQRT'))['complete'])
 def test_unknown_intrinsic_signature(self):
  self.assertFalse(check('SUBROUTINE ROOT()\nREAL F(2)\nEXTERNAL F\nEND')['complete'])
  self.assertFalse(check('SUBROUTINE ROOT()\nEXTERNAL SQRT\nINTRINSIC SQRT\nEND')['complete'])
  self.assertFalse(check('SUBROUTINE ROOT()\nREAL X\nX=SQRT(1.,2.)\nEND')['complete'])
 def test_include_array_and_executable_function(self):
  s="SUBROUTINE ROOT()\nINCLUDE 'decl.inc'\nA(1)=SQRT(2.)\nEND"
  self.assertTrue(check(s,extra={'decl.inc':fixed('REAL A(2)')})['complete'])
  self.assertFalse(check(s,extra={'decl.inc':fixed('REAL A(2)\nA(2)=FOREIGN()')})['complete'])
 def test_include_ambiguity(self):
  s="SUBROUTINE ROOT()\nINCLUDE 'a.inc'\nEND"
  self.assertFalse(check(s,extra={'a/a.inc':fixed('REAL A'),'b/a.inc':fixed('REAL B')})['complete'])
 def test_statement_function_and_alias_unknown(self):
  for middle in ('REAL F,X\nF(X)=X','REAL A,B\nEQUIVALENCE(A,B)','REAL,POINTER::P\nP=2.'):
   self.assertFalse(check('SUBROUTINE ROOT()\n'+middle+'\nEND')['complete'])
 def test_intrinsic_module_explicit_no_setters(self):
  s='SUBROUTINE ROOT()\nUSE,INTRINSIC::IEEE_EXCEPTIONS,ONLY:IEEE_GET_FLAG,IEEE_INEXACT\nLOGICAL F\nCALL IEEE_GET_FLAG(IEEE_INEXACT,F)\nEND'
  self.assertTrue(check(s)['complete']);self.assertFalse(check(s.replace('USE,INTRINSIC::','USE '))['complete']);self.assertFalse(check(s.replace('IEEE_GET_FLAG','IEEE_SET_FLAG'))['complete'])
 def test_readonly_value_association(self):
  s='SUBROUTINE ROOT()\nINTEGER*8,PARAMETER::ONE(2)=TRANSFER(1.0_16,(/0_8,0_8/))\nREAL*16 X\nINTEGER*8 Y\nASSOCIATE(WORDS=>TRANSFER(X,ONE))\nY=WORDS(1)\nEND ASSOCIATE\nEND'
  self.assertTrue(check(s)['complete']);self.assertFalse(check(s.replace('Y=WORDS(1)','WORDS(1)=2'))['complete']);self.assertFalse(check(s.replace('TRANSFER(X,ONE)','X'))['complete'])
 def test_association_name_does_not_escape_scope(self):
  s='SUBROUTINE ROOT()\nINTEGER*8 ONE(2),Y\nREAL*16 X\nASSOCIATE(WORDS=>TRANSFER(X,ONE))\nY=WORDS(1)\nEND ASSOCIATE\nY=WORDS(1)\nEND'
  r=check(s);self.assertFalse(r['complete']);self.assertIn('unresolved function WORDS',str(r))
 def test_numeric_dot_operator_boundary(self):
  self.assertTrue(check('SUBROUTINE ROOT()\nINTEGER I\nIF(I.NE.4.OR.I.EQ.2)I=3\nEND')['complete'])
 def test_io_graph_complete_but_environment_unknown(self):
  r=check("SUBROUTINE ROOT()\nREAL X,F\nWRITE(*,*) F()\nEND\nREAL FUNCTION F()\nF=1.\nEND");self.assertTrue(r['reference_complete']);self.assertFalse(r['complete']);self.assertEqual(r['procedures']['ROOT']['edges'],['F']);self.assertTrue(r['runtime_limitations'])
 def test_unknown_preprocessor_custom_operator(self):
  self.assertFalse(Library({'a.f':'#include "x"\n'+fixed('SUBROUTINE ROOT()\nEND')}).closure(['ROOT'])['complete'])
  self.assertFalse(check('SUBROUTINE ROOT()\nREAL X\nX=1.CUSTOM.2\nEND')['complete'])
 def test_defined_derived_type_io_is_not_ignored(self):
  s='MODULE OBJ\nTYPE NODE\nREAL X\nCONTAINS\nPROCEDURE::OUTPUT\nGENERIC::WRITE(FORMATTED)=>OUTPUT\nEND TYPE\nEND MODULE\nSUBROUTINE ROOT()\nUSE OBJ\nTYPE(NODE) A\nWRITE(*,*) A\nEND'
  self.assertFalse(check(s)['reference_complete'])
 def test_scientific_gate_never_implied(self):
  r=check('SUBROUTINE ROOT()\nEND');self.assertTrue(r['complete']);self.assertIs(r['scientific_admission'],False)
if __name__=='__main__':unittest.main(verbosity=2)
