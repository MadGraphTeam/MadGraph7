"""Pure environment/ELF parsing and mocked execve; no ELF execution or probes."""
from pathlib import Path
from unittest import mock
import copy
import hashlib
import json
import struct
import tempfile
import unittest
import exec_target
import run as control


def fake_elf(interp):
    encoded=interp.encode()+b'\0'
    ident=b'\x7fELF\x02\x01\x01'+bytes(9)
    header=struct.pack('<16sHHIQQQIHHHHHH',ident,2,62,1,0,64,0,0,64,56,1,64,0,0)
    return header+struct.pack('<IIQQQQQQ',3,4,120,0,0,len(encoded),len(encoded),1)+encoded


class EnvironmentTests(unittest.TestCase):
    def binding(self):return json.loads((control.HERE/'bindings.json').read_text())
    def test_clean_outer_environment_has_no_loader_variables(self):
        dirty={'LD_LIBRARY_PATH':'/old-target','LD_PRELOAD':'/inject','LD_AUDIT':'/audit',
               'LD_TRACE_LOADED_OBJECTS':'1','LD_DEBUG':'all','LD_ORIGIN_PATH':'/other'}
        with mock.patch.dict(control.os.environ,dirty):
            result=control.env(self.binding())
            self.assertFalse(any(k.startswith('LD_') for k in result))
            self.assertEqual(control.os.environ['LD_LIBRARY_PATH'],'/old-target')
    def test_target_environment_replaces_all_loader_overrides(self):
        original={'LD_PRELOAD':'/bad','LD_AUDIT':'/bad','LD_TRACE_LOADED_OBJECTS':'1','LC_ALL':'C'}
        result=exec_target.target_environment(original,'/one:/two',False)
        self.assertEqual(result,{'LC_ALL':'C','LD_LIBRARY_PATH':'/one:/two'})
        self.assertIn('LD_PRELOAD',original)
    def test_only_loader_stage_adds_trace(self):
        normal=exec_target.target_environment({},'/target',False)
        trace=exec_target.target_environment({},'/target',True)
        self.assertEqual(trace,{**normal,'LD_TRACE_LOADED_OBJECTS':'1'})
    def test_relative_or_empty_native_path_rejected(self):
        for path in ('','relative','/absolute:','/a:relative'):
            with self.subTest(path=path),self.assertRaises(ValueError):
                exec_target.target_environment({},path,False)
    def test_execve_preserves_exact_target_argv_and_env(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'not-executed';target.write_bytes(fake_elf('/bound/ld.so'))
            fake=mock.Mock(return_value='captured')
            args=['run',str(target),hashlib.sha256(target.read_bytes()).hexdigest(),'/bound/ld.so','/bound/lib','9','5','1']
            self.assertEqual(exec_target.dispatch(args,environ={'LD_PRELOAD':'/evil','LC_ALL':'C'},execve=fake),'captured')
            fake.assert_called_once_with(str(target),[str(target),'9','5','1'],{'LC_ALL':'C','LD_LIBRARY_PATH':'/bound/lib'})
    def test_loader_trace_execs_target_not_ld(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'not-executed';target.write_bytes(fake_elf('/bound/ld.so'))
            fake=mock.Mock()
            exec_target.dispatch(['trace',str(target),hashlib.sha256(target.read_bytes()).hexdigest(),'/bound/ld.so','/bound/lib'],environ={},execve=fake)
            self.assertEqual(fake.call_args.args[0],str(target))
            self.assertEqual(fake.call_args.args[1],[str(target)])
            self.assertEqual(fake.call_args.args[2]['LD_TRACE_LOADED_OBJECTS'],'1')
    def test_wrong_hash_or_interpreter_never_executes(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'not-executed';target.write_bytes(fake_elf('/bound/ld.so'))
            checksum=hashlib.sha256(target.read_bytes()).hexdigest()
            for sha,interp in [('0'*64,'/bound/ld.so'),(checksum,'/wrong/ld.so')]:
                fake=mock.Mock()
                with self.assertRaises(ValueError):exec_target.dispatch(['run',str(target),sha,interp,'/bound/lib','1','4','0'],environ={},execve=fake)
                fake.assert_not_called()
    def test_invalid_arm_and_trace_arguments_never_executes(self):
        fake=mock.Mock()
        for args in (['run','/never','hash','/ld','/lib','7','4','0'],['trace','/never','hash','/ld','/lib','extra']):
            with self.assertRaises(ValueError):exec_target.dispatch(args,environ={},execve=fake)
        fake.assert_not_called()
    def test_truncated_or_wrong_elf_rejected(self):
        for raw in (b'notELF',fake_elf('/ld')[:64],fake_elf('/ld')[:-1]):
            with self.assertRaises(ValueError):exec_target.interpreter(raw)
    def test_exact_single_host_loader_exclusion(self):
        binding=self.binding();control.validate_target_closure(binding)
        self.assertEqual(len(binding['historical_ldd_runtime'])-len(binding['runtime']),1)
        bad=copy.deepcopy(binding);bad['runtime'].pop(next(p for p in bad['runtime'] if p.endswith('libc.so.6')))
        with self.assertRaisesRegex(ValueError,'closure changed'):control.validate_target_closure(bad)
        bad=copy.deepcopy(binding);bad['excluded_host_ldd_interpreter']['/extra']='0'*64
        with self.assertRaisesRegex(ValueError,'loader exclusion'):control.validate_target_closure(bad)
    def test_closure_or_native_path_disagreement_rejected(self):
        binding=self.binding()
        with self.assertRaisesRegex(ValueError,'closure differs'):control.validate_loader('/unexpected/library.so',binding)
        binding['native_library_path']+=':/injected'
        with self.assertRaisesRegex(ValueError,'library path differs'):control.validate_target_closure(binding)
    def test_target_command_delays_native_environment(self):
        binding=self.binding();target=Path('/private/diagnostic')
        command=control.target_command(binding,target,'0'*64,[6,4,0])
        self.assertEqual(command[:len(control.prefix(binding))],control.prefix(binding))
        self.assertIn(binding['programs']['python'],command)
        self.assertEqual(command[-3:],['6','4','0'])
        self.assertIn(binding['native_library_path'],command)
        self.assertFalse(any(k.startswith('LD_') for k in control.env(binding)))
    def test_build_and_run_namespaces_are_fresh(self):
        source=(control.HERE/'run.py').read_text()
        self.assertIn("WORK/'build2/build.json'",source);self.assertIn("WORK/'run3'",source)
        self.assertNotIn("WORK/'build3'",source);self.assertNotIn("WORK/'run2'",source)

if __name__=='__main__':unittest.main(verbosity=2)
