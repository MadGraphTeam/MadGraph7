"""Pure flag-policy tests; no make, compiler, loader or native invocation."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('o3_policy',HERE/'policy.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


class Compiler:
    @staticmethod
    def _signature(row):return (row['compiler'],row['flags'])


def fixture(candidate=False):
    recipe={'compiler':['frozen-GNU15'],'flags':p.ORIGINAL_FLAGS[:]}
    original={'environment':{},'scopes':{
        'process':{'recipes':{'TIRUST_interface.f':copy.deepcopy(recipe),'TIR_interface.f':copy.deepcopy(recipe),'check_sa.f':copy.deepcopy(recipe),
                              'polynomial.f':{'compiler':['frozen-GNU15'],'flags':p.ORIGINAL_FLAGS+['-O3']}}},
        'MODEL':{'recipes':{'couplings.f':copy.deepcopy(recipe)}},'DHELAS':{'recipes':{'helas.f':copy.deepcopy(recipe)}}}}
    actual=copy.deepcopy(original)
    if candidate:actual['scopes']['process']['recipes']['TIRUST_interface.f']['flags']+=p.CANDIDATE_ADDITIONS
    return original,actual


class PolicyTests(unittest.TestCase):
    def test_original_and_one_file_candidate_policies(self):
        self.assertEqual(p.verify_policy(*fixture(),False,Compiler)['changed_recipes'],[])
        self.assertEqual(p.verify_policy(*fixture(True),True,Compiler)['changed_recipes'],['process/TIRUST_interface.f'])

    def test_other_recipe_compiler_or_environment_change_rejected(self):
        original,actual=fixture(True)
        for scope,name in [('process','check_sa.f'),('process','TIR_interface.f'),('process','polynomial.f'),('MODEL','couplings.f')]:
            wrong=copy.deepcopy(actual);wrong['scopes'][scope]['recipes'][name]['flags']+=['-O3']
            with self.assertRaises(ValueError):p.verify_policy(original,wrong,True,Compiler)
        wrong=copy.deepcopy(actual);wrong['environment']={'NIX_CFLAGS_COMPILE':'-Ofast'}
        with self.assertRaises(ValueError):p.verify_policy(original,wrong,True,Compiler)

    def test_fastmath_or_extra_flag_rejected(self):
        original,actual=fixture(True)
        actual['scopes']['process']['recipes']['TIRUST_interface.f']['flags']+=['-ffast-math']
        with self.assertRaises(ValueError):p.verify_policy(original,actual,True,Compiler)

    def test_explicit_recipe_has_no_variable_inheritance(self):
        base='\t$(FC) $(FFLAGS) -o $(PROG) check_sa.o $(PROCESS) $(LINKLIBS)\n'
        original=p.private_makefile(base);candidate=p.private_makefile(base,True)
        self.assertEqual(candidate.replace(' -O3 -fno-fast-math',''),original)
        delta=candidate[len(base):]
        self.assertNotIn('FFLAGS +=',delta)
        self.assertIn('$(POLYNOMIAL) $(OLP_POLYNOMIAL)',candidate)
        with self.assertRaises(ValueError):p.private_makefile(candidate,True)

    def test_selected_two_compiles_and_one_link_only(self):
        text='/compiler -w -c check_sa.f -o check_sa.o\n/compiler -w -O3 -fno-fast-math -c TIRUST_interface.f -o TIRUST_interface.o\nln -s libcollier || echo existing\n/compiler -w -o check check_sa.o TIRUST_interface.o -ltirust\n'
        with patch.object(p.shutil,'which',return_value='/compiler'):
            recipes,link=p.select_commands(text,'/compiler',{})
            self.assertEqual([n for n,_ in recipes],list(p.NEEDED));self.assertIn('-ltirust',link)
            with self.assertRaises(ValueError):p.select_commands(text.replace('-w -c check_sa.f','-w ; -c check_sa.f'),'/compiler',{})
            with self.assertRaises(ValueError):p.select_commands(text.replace(' -ltirust',' mtirust.o -ltirust'),'/compiler',{})


if __name__=='__main__':unittest.main()
