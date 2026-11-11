"""Pure controller tests; no compiler, loader, Fortran or reducer is invoked."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('qp_private_controller',HERE/'controller.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def policy():
    common={'compiler':[{'path':'/compiler','sha256':'compiler','version':'GNU15'}],
            'flags':['-w','-fPIC','-ffixed-line-length-132']}
    left={'environment':{},'scopes':{'process':{'recipes':{'TIRUST_interface.f':common}},
            'MODEL':{'recipes':{'model.f':common}},'DHELAS':{'recipes':{'helas.f':common}}}}
    right=copy.deepcopy(left)
    for name in ('qp_profile.f90','mtirust.f90'):
        right['scopes']['process']['recipes'][name]={'compiler':copy.deepcopy(common['compiler']),'flags':['-O3','-fPIC']}
    return left,right


class ControllerTests(unittest.TestCase):
    def test_private_recipes_all_sources(self):
        c.verify_recipes(*policy())

    def test_missing_f90_recipe_rejected(self):
        a,b=policy();del b['scopes']['process']['recipes']['qp_profile.f90']
        with self.assertRaisesRegex(ValueError,'Missing or extra'):c.verify_recipes(a,b)

    def test_original_flags_not_relabeled(self):
        a,b=policy();b['scopes']['process']['recipes']['TIRUST_interface.f']['flags']=['-O3']
        with self.assertRaisesRegex(ValueError,'Changed original'):c.verify_recipes(a,b)

    def test_new_module_recipe_must_match_install_policy(self):
        a,b=policy();b['scopes']['process']['recipes']['mtirust.f90']['flags'].append('-ffast-math')
        with self.assertRaisesRegex(ValueError,'private module recipe'):c.verify_recipes(a,b)

    def test_compiler_environment_must_match(self):
        a,b=policy();b['environment']={'NIX_CFLAGS_COMPILE':'-ffast-math'}
        with self.assertRaisesRegex(ValueError,'environment'):c.verify_recipes(a,b)

    def test_link_additions_precede_archive(self):
        original='\t$(FC) $(FFLAGS) -o $(PROG) check_sa.o $(PROCESS) $(LINKLIBS)\n'
        actual=c.private_makefile(original)
        self.assertLess(actual.index('mtirust.o $(PROCESS)'),actual.index('$(LINKLIBS)'))
        self.assertIn('qp_profile.o: qp_profile.f90',actual)
        self.assertIn('mtirust.o: mtirust.f90 qp_profile.o',actual)
        with self.assertRaises(ValueError):c.private_makefile(original+original)

    def test_unique_mtirust_definitions_and_map(self):
        table='00000 T __mtirust_MOD_tensor_mp\n00001 b __mtirust_MOD_context\n'
        self.assertFalse(c.verify_symbols(table,table,'LOAD mtirust.o\n')['original_archive_member_loaded'])
        with self.assertRaisesRegex(ValueError,'missing/duplicated'):c.verify_symbols(table,table+table,'LOAD mtirust.o\n')
        with self.assertRaisesRegex(ValueError,'archive member'):c.verify_symbols(table,table,'LOAD mtirust.o\n/path/libtirust.a(mtirust.o)\n')

    def test_only_five_compile_recipes_selected(self):
        names=['qp_profile.f90','mtirust.f90','TIR_interface.f','TIRUST_interface.f','check_sa.f','polynomial.f']
        text='\n'.join(f'gfortran -c {n} -o {Path(n).stem}.o' for n in names)+'\n'
        text+='gfortran -o check check_sa.o qp_profile.o mtirust.o polynomial.o -ltirust\n'
        with patch.object(c,'commands',return_value=text):
            selected,link=c.expanded_commands(Path('.'),{},Path('.'))
        self.assertEqual([x[0] for x in selected],names[:-1])
        self.assertNotIn('polynomial.f',[x[0] for x in selected])
        with patch.object(c,'commands',return_value=text.replace('mtirust.o polynomial.o -ltirust','polynomial.o -ltirust mtirust.o')):
            with self.assertRaisesRegex(ValueError,'precedence'):c.expanded_commands(Path('.'),{},Path('.'))

    def test_clone_sources_private_and_filters_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'original';source.mkdir();(source/'lib').mkdir()
            (source/'lib/a.a').write_bytes(b'archive');(source/'external.f').write_text('original')
            (source/'source.f').symlink_to(source/'external.f');(source/'fooHelFilter.dat').write_text('old')
            dest=root/'copy';c.clone(source,dest)
            self.assertTrue((dest/'lib').is_symlink());self.assertFalse((dest/'source.f').is_symlink())
            self.assertFalse((dest/'fooHelFilter.dat').exists());(dest/'source.f').write_text('private')
            self.assertEqual((source/'external.f').read_text(),'original')

    def test_flags_shape_and_unknown_observations(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'flags.dat'
            rows=[[2,point,phase,-1]+[-1]*10 for point in (1,16) for phase in (1,2)]
            p.write_text('\n'.join(' '.join(map(str,r)) for r in rows))
            self.assertEqual(len(c.read_flags(p,2)),4)
            with self.assertRaisesRegex(ValueError,'Malformed'):c.read_flags(p,0)
            p.write_text('\n'.join(' '.join(map(str,r)) for r in rows[:3]))
            with self.assertRaisesRegex(ValueError,'Incomplete'):c.read_flags(p,2)

    def test_cleanup_budget_uses_original_deadline(self):
        class Owned:
            def stop(self,**kw):self.kw=kw;return {'survivors':[]}
        owned=Owned()
        with patch.object(c.time,'monotonic',return_value=99.):c.cleanup_owned(owned,100.)
        self.assertAlmostEqual(sum(owned.kw.values()),1.)
        with patch.object(c.time,'monotonic',return_value=101.):c.cleanup_owned(owned,100.)
        self.assertEqual(sum(owned.kw.values()),0.)

    def test_source_pin_is_explicit(self):
        with patch.object(c,'EXPECTED_SOURCE',None):
            with self.assertRaisesRegex(ValueError,'Explicit reviewed source'):c.verify_seal()

    def test_run_requires_reviewed_closed_build_hash(self):
        with patch.object(c,'read',return_value={}),patch.object(c,'REVIEWED_BUILD',None):
            with self.assertRaisesRegex(ValueError,'reviewed closed build'):c.verify_build()

    def test_limits_set_on_owned_child_only(self):
        with patch.object(c.resource,'getrlimit',return_value=(-1,-1)),patch.object(c.resource,'setrlimit') as setter:
            c.set_limits()
        self.assertEqual(setter.call_count,3)
        self.assertEqual(setter.call_args_list[-1].args,(c.resource.RLIMIT_CORE,(0,0)))

    def test_profile_clock_scopes_and_finite_counters(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'point-16';timer=Path(str(p)+'.timers.tsv');calls=Path(str(p)+'.calls.tsv')
            timer.write_text('# mode 1\n# clock_rate 1000\n4 4 3 2 2 1 12\n')
            calls.write_text('# raw snapshots\n')
            result=c.validate_profile(p,1);self.assertEqual(result['timers'][0]['seconds'],.012)
            timer.write_text('# mode 1\n# clock_rate 1000\n4 4 3 2 2 1 -1\n')
            with self.assertRaisesRegex(ValueError,'Invalid stage'):c.validate_profile(p,1)
            timer.write_text('# mode 1\n# clock_rate 0\n')
            with self.assertRaisesRegex(ValueError,'clock/mode'):c.validate_profile(p,1)

    def test_ldd_query_interpreter_alias_is_not_amplitude_provider(self):
        old={'libraries':{'/target/ld-linux.so':'a','/query/ld-linux.so':'b','/lib/a.so':'c'},
             'ldd':'/target/ld-linux.so => /query/ld-linux.so (0x01)\n'}
        class Inner:
            def interpreter(self,_):return Path('/target/ld-linux.so')
        class Helper:
            def _support(self,*_):return Inner()
        with patch.object(c,'read',return_value={'runtime':old}),patch.object(c,'loader_support',return_value=Helper()):
            files,reason=c.expected_actual_libraries()
        self.assertEqual(files,{'/target/ld-linux.so':'a','/lib/a.so':'c'})
        self.assertEqual(reason['historical_ldd_only_interpreter_aliases'],{'/query/ld-linux.so':'b'})

    def test_caps_and_no_automatic_case_expansion(self):
        self.assertEqual(c.CAPS,{'build':180.,'mode-0':300.,'mode-1':300.,'mode-2':300.})
        self.assertEqual(sum(c.CAPS.values()),1080.)
        self.assertEqual(c.CLEANUP,60.)
        self.assertEqual(c.CASE_ID,'qcd_n3_uux_zgg')


if __name__=='__main__':unittest.main()
