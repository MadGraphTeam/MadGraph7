"""Pure mutation/admission tests: no compiler, loader, executable or amplitude."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import guard as g


class GuardTests(unittest.TestCase):
    def test_hash_before_parse(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad.json';p.write_text('not-json')
            with self.assertRaisesRegex(ValueError,'Pinned bytes differ'):
                g.admit_generation(p,{'generated_source_receipt':{'bytes':0,'sha256':'bad'}},{})

    def test_receipt_must_equal_reviewed_generation(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'g.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Unreviewed generation'):
                g.admit_generation(p,{'generated_source_receipt':g.record(p)},{})

    def test_raw_sources_must_equal_derived(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'g.json';p.write_text(json.dumps({'raw_sources':{'x':1}}))
            with patch.object(g,'GENERATION_SHA',g.record(p)['sha256']):
                with self.assertRaisesRegex(ValueError,'CMS source bindings differ'):
                    g.admit_generation(p,{'generated_source_receipt':g.record(p)},{'files':{}})
                self.assertEqual(g.admit_generation(p,{'generated_source_receipt':g.record(p)},
                    {'files':{'x':1}})['raw_sources'],{'x':1})

    def test_provider_missing_and_hash_mismatch(self):
        for runtime in ({'/abs/new':'hash'},{'/abs/old':'changed'}):
            with self.assertRaisesRegex(ValueError,'Unadmitted runtime'):
                g.providers({'files':runtime},{'/abs/old':'old'},{})

    def test_supplement_is_exact_not_basename(self):
        supplement={'/old/libquadmath.so':{'sha256':'hash'}}
        g.providers({'files':{'/old/libquadmath.so':'hash'}},{},supplement)
        with self.assertRaisesRegex(ValueError,'Unadmitted runtime'):
            g.providers({'files':{'/new/libquadmath.so':'hash'}},{},supplement)
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            g.providers({'files':{}},{'/old/libquadmath.so':'wrong'},supplement)

    def fixture(self, directory, actual_loader=True, unknown=False):
        base=Path(directory);out=base/'guard';out.mkdir()
        executable=base/'work/builds/cms/unmodified/SubProcesses/P/check'
        executable.parent.mkdir(parents=True);executable.write_bytes(b'fake-executable')
        compiler=base/'compiler';compiler.write_bytes(b'fake-compiler')
        loader=base/'loader';loader.write_bytes(b'fake-loader')
        declaration=base/'dynamic-linker';declaration.write_text(str(loader)+'\n')
        boundfile=base/'provider';boundfile.write_bytes(b'fake-provider')
        prelink=base/'results/builds/cms-unmodified/prelink.json'
        prelink.parent.mkdir(parents=True)
        prelink.write_text(json.dumps({'completed_before_link':True,'bound':{str(boundfile):g.record(boundfile)['sha256']}}))
        binding={'compiler':str(compiler),'dynamic_linker_declaration':str(declaration),'loader':str(loader),
            'toolchain':{str(p):g.record(p) for p in (compiler,loader,declaration)},
            'supplemental_providers':{str(loader):g.record(loader)}}
        calls=[]
        def original(path,deadline):
            calls.append(path)
            return {'loader':str(loader),'loader_sha256':g.record(loader)['sha256'],
                'files':{str(loader):g.record(loader)['sha256'],
                         str(boundfile if not unknown else base/'unadmitted'):'wrong' if unknown else g.record(boundfile)['sha256']}}
        def hashes(bound):
            for name,checksum in bound.items():
                g.require(g.record(name)['sha256']==checksum,'Changed prelink input')
        r=NS(CASES={'cms':('case','P',4)},L=NS(runtime=original,
            interpreter=lambda p:loader if actual_loader else base/'wrong-loader',q=NS(verify_hashes=hashes)))
        return base,out,executable,binding,r,calls

    def test_wrong_interpreter_fails_before_query(self):
        with tempfile.TemporaryDirectory() as t:
            base,out,exe,b,r,calls=self.fixture(t,actual_loader=False)
            with patch.object(g,'BASE',base),patch.object(g,'RESULTS',base/'results'):
                g.install_hook(r,b,out)
                with self.assertRaisesRegex(ValueError,'PT_INTERP differs'):r.L.runtime(exe)
            self.assertEqual(calls,[])
            self.assertFalse(json.loads((out/'cms-unmodified-runtime.json').read_text())['passed'])

    def test_unknown_provider_retained_without_retry(self):
        with tempfile.TemporaryDirectory() as t:
            base,out,exe,b,r,calls=self.fixture(t,unknown=True)
            with patch.object(g,'BASE',base),patch.object(g,'RESULTS',base/'results'):
                g.install_hook(r,b,out)
                with self.assertRaisesRegex(ValueError,'Unadmitted runtime'):r.L.runtime(exe)
                with self.assertRaisesRegex(ValueError,'Repeated loader query'):r.L.runtime(exe)
            self.assertEqual(len(calls),1)
            receipt=json.loads((out/'cms-unmodified-runtime.json').read_text())
            self.assertFalse(receipt['passed']);self.assertIn(str(base/'unadmitted'),receipt['runtime']['files'])

    def test_bound_provider_and_loader_pass(self):
        with tempfile.TemporaryDirectory() as t:
            base,out,exe,b,r,calls=self.fixture(t)
            with patch.object(g,'BASE',base),patch.object(g,'RESULTS',base/'results'):
                seen=g.install_hook(r,b,out);r.L.runtime(exe)
                with self.assertRaisesRegex(ValueError,'outside four'):r.L.runtime(base/'other')
            self.assertEqual(len(calls),1);self.assertEqual(seen,{'cms-unmodified'})
            self.assertTrue(json.loads((out/'cms-unmodified-runtime.json').read_text())['passed'])


if __name__=='__main__':unittest.main()
