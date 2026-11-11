"""Pure controller tests; no compiler, loader, Fortran or reducer is invoked."""
import copy
import os
import json
import shlex
import shutil
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

    def recorded_commands(self):
        text=(HERE/'failed-expanded-check.txt').read_text()
        compiler=shlex.split(next(line for line in text.splitlines() if '-c qp_profile.f90' in line))[0]
        return text,compiler

    def test_actual_failed_dryrun_selects_exact_five_plus_link(self):
        text,driver=self.recorded_commands()
        self.assertIn('|| echo "libcolier already linked"',text)
        selected,link=c.select_commands(text,driver,os.environ)
        self.assertEqual([row[0] for row in selected],['qp_profile.f90','mtirust.f90','TIR_interface.f','TIRUST_interface.f','check_sa.f'])
        self.assertEqual(len(selected),5);self.assertEqual(link[link.index('-o')+1],'check')
        self.assertTrue(all('ln' not in cmd and 'echo' not in cmd for _,cmd in selected))
        self.assertNotIn('-c',link)
        with patch.object(c,'commands',return_value=text):
            self.assertEqual(c.expanded_commands(Path('.'),os.environ,Path('.'),driver),(selected,link))

    def test_actual_selected_compound_rejected(self):
        text,driver=self.recorded_commands()
        for token in (';','&&','||','|','>','>>','<','&'):
            mutated=text.replace('-c qp_profile.f90 -o qp_profile.o','-c qp_profile.f90 -o qp_profile.o '+token+' echo unsafe')
            with self.subTest(token=token),self.assertRaisesRegex(ValueError,'Compound selected'):
                c.select_commands(mutated,driver,os.environ)

    def test_actual_selected_duplicate_missing_rejected(self):
        text,driver=self.recorded_commands();row=next(line for line in text.splitlines() if '-c qp_profile.f90' in line)
        with self.assertRaisesRegex(ValueError,'Duplicate selected'):c.select_commands(text+'\n'+row,driver,os.environ)
        with self.assertRaisesRegex(ValueError,'complete selected'):c.select_commands(text.replace(row,''),driver,os.environ)
        link=next(line for line in text.splitlines() if ' -o check ' in line)
        with self.assertRaisesRegex(ValueError,'exactly one selected'):c.select_commands(text+'\n'+link,driver,os.environ)
        with self.assertRaisesRegex(ValueError,'exactly one selected'):c.select_commands(text.replace(link,''),driver,os.environ)

    def test_actual_unknown_prefix_new_source_rejected(self):
        text,driver=self.recorded_commands();row=next(line for line in text.splitlines() if '-c qp_profile.f90' in line)
        bad='env '+row
        with self.assertRaisesRegex(ValueError,'Unknown selected compiler'):c.select_commands(text.replace(row,bad),driver,os.environ)
        bad=row.replace('-c qp_profile.f90','-c qp_profile.f90 injected.f')
        with self.assertRaisesRegex(ValueError,'Unexpected source'):c.select_commands(text.replace(row,bad),driver,os.environ)
        bad=row.replace('-o qp_profile.o','-o outside.o')
        with self.assertRaisesRegex(ValueError,'target differs'):c.select_commands(text.replace(row,bad),driver,os.environ)

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

    def test_actual_failed_three_to_four_card_inventory(self):
        value=json.loads((HERE/'failed-card-inventory.json').read_text())
        expected=value['expected_original_card_inventory'];actual=value['actual_dereferenced_private_card_inventory']
        self.assertEqual(len(expected),3);self.assertEqual(len(actual),4)
        c.verify_private_cards(actual,expected)
        with self.assertRaisesRegex(ValueError,'four-card'):c.verify_private_cards(expected,expected)
        for path in actual:
            mutated=dict(actual);mutated[path]='0'*64
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'four-card'):c.verify_private_cards(mutated,expected)
            missing=dict(actual);del missing[path]
            with self.subTest(missing=path),self.assertRaisesRegex(ValueError,'four-card'):c.verify_private_cards(missing,expected)
        extra=dict(actual);extra['SubProcesses/foreign/MadLoopParams.dat']=next(iter(actual.values()))
        with self.assertRaisesRegex(ValueError,'four-card'):c.verify_private_cards(extra,expected)
        changed=dict(expected);changed['Cards/MadLoopParams.dat']='0'*64
        with self.assertRaisesRegex(ValueError,'three-card'):c.verify_private_cards(actual,changed)

    def test_build_entrypoints_disabled(self):
        with self.assertRaisesRegex(ValueError,'RUN-only'):c.stage('build')
        with self.assertRaisesRegex(ValueError,'RUN-only'):c.build_child()

    def test_source_compatibility_uses_actual_reviewed_five_sources(self):
        self.assertEqual(c.source_compatibility(),json.loads((HERE/'source-compatibility.json').read_text()))
        self.assertEqual(c.BUILD_SOURCE,'c5e6eeb2614ef2b0d99abc6621281479ac847add01e1019262b4c7fe97654f1d')
        self.assertEqual(c.BUILD_TERMINAL,'90ef4a7b47c0a795b9f2bd80a4c4b059d1930af4a5ad05e2e68ba70b3c516a11')

    def test_caps_and_no_automatic_case_expansion(self):
        self.assertEqual(c.CAPS,{'mode-0':300.,'mode-1':300.,'mode-2':300.})
        self.assertEqual(sum(c.CAPS.values()),900.)
        self.assertEqual(c.CLEANUP,60.)
        self.assertEqual(c.CASE_ID,'qcd_n3_uux_zgg')


if __name__=='__main__':unittest.main()
