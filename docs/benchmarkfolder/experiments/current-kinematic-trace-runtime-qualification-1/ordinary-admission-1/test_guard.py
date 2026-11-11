"""Pure ordinary admission tests with fake runtime callbacks only."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import guard as ordinary

p=Path(__file__).resolve().parents[1]/'build-admission-1/guard.py'
spec=importlib.util.spec_from_file_location('pure_build_guard_helpers',p)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
ordinary.G=g


class OrdinaryTests(unittest.TestCase):
    def fixture(self,t,wrong_loader=False,new_provider=False,wrong_executable=False,changed_provider=False,variant='enabled'):
        base=Path(t);out=base/'receipts';out.mkdir()
        request={'id':'cms-point1-mode4-'+variant,'case':'cms','variant':variant,'requested_mode':4}
        exe=base/'work/workers'/request['id']/'SubProcesses/P/check'
        exe.parent.mkdir(parents=True);exe.write_bytes(b'closed-executable')
        loader=base/'loader';loader.write_bytes(b'closed-loader')
        declaration=base/'dynamic-linker';declaration.write_text(str(loader)+'\n')
        compiler=base/'compiler';compiler.write_bytes(b'closed-compiler')
        provider=base/'provider';provider.write_bytes(b'closed-provider')
        files={str(p):g.record(p)['sha256'] for p in (loader,provider)}
        runtime={'loader':str(loader),'loader_sha256':files[str(loader)],'files':files}
        build={'executable_pin':g.record(exe),'prelink_bound':{str(provider):files[str(provider)]},'runtime':runtime}
        toolchain={'toolchain':{str(p):g.record(p) for p in (loader,declaration,compiler)},
                   'dynamic_linker_declaration':str(declaration),'loader':str(loader)}
        if wrong_executable:exe.write_bytes(b'altered-executable')
        if changed_provider:provider.write_bytes(b'altered-provider')
        calls=[]
        def original(exe,deadline):
            calls.append(exe)
            returned={**runtime,'files':dict(files)}
            if new_provider:returned['files'][str(base/'new-provider')]='unadmitted'
            return returned
        r=NS(CASES={'cms':('case','P',4)},L=NS(runtime=original,
            interpreter=lambda p:base/'wrong-loader' if wrong_loader else loader))
        key='cms-unmodified' if variant=='unmodified' else 'cms-instrumented'
        return base,out,exe,request,toolchain,{key:build},r,calls

    def exercise(self,fixture,expected_error=None):
        base,out,exe,req,tc,builds,r,calls=fixture
        with patch.object(ordinary,'BASE',base):
            seen=ordinary.install_hook(r,tc,builds,[req],out)
            if expected_error:
                with self.assertRaisesRegex(ValueError,expected_error):r.L.runtime(exe)
            else:r.L.runtime(exe)
        receipt=json.loads((out/(req['id']+'-runtime.json')).read_bytes())
        self.assertEqual(seen,{req['id']})
        return receipt,r,calls,exe

    def test_success_and_exact_instrumented_variant(self):
        with tempfile.TemporaryDirectory() as t:
            receipt,r,calls,exe=self.exercise(self.fixture(t))
            self.assertTrue(receipt['passed']);self.assertTrue(receipt['loader_query_started'])
            self.assertEqual(calls,[exe])

    def test_unmodified_variant_uses_unmodified_binding(self):
        with tempfile.TemporaryDirectory() as t:
            receipt,_,calls,_=self.exercise(self.fixture(t,variant='unmodified'))
            self.assertTrue(receipt['passed']);self.assertEqual(len(calls),1)

    def test_wrong_pt_interp_prevents_query(self):
        with tempfile.TemporaryDirectory() as t:
            receipt,_,calls,_=self.exercise(self.fixture(t,wrong_loader=True),'PT_INTERP differs')
            self.assertFalse(receipt['loader_query_started']);self.assertEqual(calls,[])

    def test_modified_provider_prevents_query(self):
        with tempfile.TemporaryDirectory() as t:
            receipt,_,calls,_=self.exercise(self.fixture(t,changed_provider=True),'Closed input bytes differ')
            self.assertFalse(receipt['loader_query_started']);self.assertEqual(calls,[])

    def test_modified_executable_prevents_query(self):
        with tempfile.TemporaryDirectory() as t:
            receipt,_,calls,_=self.exercise(self.fixture(t,wrong_executable=True),'Pinned bytes differ')
            self.assertFalse(receipt['loader_query_started']);self.assertEqual(calls,[])

    def test_changed_runtime_retained_and_no_retry(self):
        with tempfile.TemporaryDirectory() as t:
            receipt,r,calls,exe=self.exercise(self.fixture(t,new_provider=True),'runtime differs')
            self.assertFalse(receipt['passed']);self.assertTrue(receipt['loader_query_started'])
            self.assertIn(str(Path(t)/'new-provider'),receipt['runtime']['files'])
            with self.assertRaisesRegex(ValueError,'Repeated runtime'):r.L.runtime(exe)
            with self.assertRaisesRegex(ValueError,'outside ordinary'):r.L.runtime(Path(t)/'stress/check')
            self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
