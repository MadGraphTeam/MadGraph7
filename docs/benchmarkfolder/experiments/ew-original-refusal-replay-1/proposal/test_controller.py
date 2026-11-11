"""Pure negative admission tests; fake loaders only, no child launches."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import run as r


class RuntimeAdmissionTests(unittest.TestCase):
    def fixture(self,temp,wrong_loader=False,changed_before=False,changed_after=False,changed_executable=False):
        base=Path(temp);(base/'bindings.json').write_text('{}');(base/'exe').write_bytes(b'closed executable')
        declared=base/'declared-loader';calls=[]
        closed={'loader':str(declared),'loader_sha256':'loaderhash','files':{'/provider':'old'}}
        def query(exe):
            calls.append(str(exe))
            if changed_executable:Path(exe).write_bytes(b'changed executable')
            return {**closed,'files':{'/provider':'new' if changed_after else 'old'}}
        def verify(files):
            if changed_before and files==closed['files']:raise ValueError('closed provider bytes changed')
        def save(path,data):
            with Path(path).open('x') as f:json.dump(data,f)
        fakeG=NS(compiler_pins=lambda b:declared,providers=lambda *args:None,save=save)
        fakeR=NS(L=NS(interpreter=lambda p:base/'wrong' if wrong_loader else declared,
                      q=NS(verify_hashes=verify),runtime=query))
        return base,closed,calls,fakeG,fakeR

    def exercise(self,values,error=None):
        base,closed,calls,G,R=values
        with patch.object(r,'BUILD_GUARD',base),patch.object(r,'RESULTS',base), \
             patch.object(r,'G',G),patch.object(r,'R',R), \
             patch.object(r,'B',{'fixed_supplemental_providers':{}}):
            if error:
                with self.assertRaisesRegex(ValueError,error):r.runtime(base/'exe',{'/input':'hash'},'test',closed)
            else:r.runtime(base/'exe',{'/input':'hash'},'test',closed)
        return json.loads((base/'test-runtime.json').read_bytes()),calls

    def test_wrong_interpreter_fails_before_query_and_is_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt,calls=self.exercise(self.fixture(temp,wrong_loader=True),'PT_INTERP differs')
            self.assertFalse(receipt['loader_query_started']);self.assertEqual(calls,[])

    def test_changed_closed_provider_fails_before_query(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt,calls=self.exercise(self.fixture(temp,changed_before=True),'closed provider bytes changed')
            self.assertFalse(receipt['loader_query_started']);self.assertEqual(calls,[])

    def test_changed_returned_provider_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt,calls=self.exercise(self.fixture(temp,changed_after=True),'Runtime differs')
            self.assertTrue(receipt['loader_query_started']);self.assertFalse(receipt['passed'])
            self.assertEqual(receipt['runtime']['files'],{'/provider':'new'});self.assertEqual(len(calls),1)

    def test_same_closed_set_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt,calls=self.exercise(self.fixture(temp))
            self.assertTrue(receipt['passed']);self.assertEqual(len(calls),1)

    def test_executable_changed_during_loader_query_is_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt,calls=self.exercise(self.fixture(temp,changed_executable=True),'Executable changed')
            self.assertTrue(receipt['loader_query_started']);self.assertFalse(receipt['passed'])
            self.assertIn('runtime',receipt);self.assertEqual(len(calls),1)

    def test_actual_retained_capture_accepts_run_after_owned_setup(self):
        path=r.ROOT/'docs/benchmarkfolder/experiments/current-cms-replay-supervisor-1/ownership/run.py'
        raw=path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         'be6212d7e93f698b551eee35ec577132c0ebb6db73d07cc203262be6319ce861')
        capture=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='capture')
        owner=ModuleType('retained_capture_test');calls=[]
        def command(argv,label,cwd,timeout,cpu=True):
            calls.append((argv,label,cwd,timeout,cpu));return 'retained query path'
        oldstages=('generate','derive','build','ordinary','stress')
        owner.__dict__.update(ROOT=r.ROOT,require=r.require,QUERY=0,CURRENT_STAGE='run',
                              STAGES=oldstages,command=command)
        exec(compile(ast.Module(body=[capture],type_ignores=[]),str(path),'exec'),owner.__dict__)
        with self.assertRaisesRegex(ValueError,'explicit stage'):owner.capture(['/loader','--list','/exe'])
        fake=NS(L=owner,setup_owned_helpers=lambda:setattr(owner,'STAGES',oldstages))
        with patch.object(r,'R',fake):
            r.setup_owned()
            self.assertEqual(owner.STAGES,('build','run'))
            self.assertEqual(owner.capture(['/loader','--list','/exe']),'retained query path')
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][1],'query-run-0001')
        self.assertEqual(calls[0][0][-3:],['/loader','--list','/exe'])


if __name__=='__main__':unittest.main()
