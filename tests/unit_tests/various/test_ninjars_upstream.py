"""Upstream checks report changes without moving or concealing source pins."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('ninjars_upstream',ROOT/'tools/ninjars_upstream.py')
upstream = None
if sys.version_info >= (3, 11):
    upstream = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(upstream)


@unittest.skipIf(upstream is None, 'The optional upstream checker requires Python 3.11+')
class TestNinjaRSUpstream(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.manifest, self.lock = self.work/'Cargo.toml', self.work/'Cargo.lock'
        self.pin, self.master = '1'*40, '2'*40
        self.manifest.write_text('[dependencies]\n'+''.join(
            '%s = {git="https://github.com/alphal00p/ninjars", rev="%s"}\n' % (name,self.pin)
            for name in ('ninjars','ninjars-symbolica')))
        self.lock.write_text('version = 4\n'+''.join(
            '\n[[package]]\nname = "%s"\nversion = "0.1.0"\nsource = "git+%s?rev=%s#%s"\n' % (name,url,pin,pin)
            for name,url,pin in [('ninjars',upstream.REPOSITORIES['ninjars'],self.pin),
                ('ninjars-symbolica',upstream.REPOSITORIES['ninjars'],self.pin),
                ('oneloop',upstream.REPOSITORIES['oneloopmaster']+'.git',self.master)]))

    def remote(self,pin):
        return upstream.parse_remote('ref: refs/heads/main\tHEAD\n%s\tHEAD\n%s\trefs/heads/main\n' % (pin,pin))

    def test_current_and_pending_main_preserve_both_inputs(self):
        before = (self.manifest.read_bytes(),self.lock.read_bytes())
        def query(url):
            return self.remote(self.pin if url.endswith('/ninjars') else '3'*40)
        result = upstream.check(self.manifest,self.lock,query)
        self.assertTrue(result['update_pending'])
        self.assertFalse(result['repositories']['ninjars']['update_pending'])
        self.assertTrue(result['repositories']['oneloopmaster']['update_pending'])
        self.assertEqual(before,(self.manifest.read_bytes(),self.lock.read_bytes()))

    def test_manifest_disagreement_rejected_before_network(self):
        self.manifest.write_text(self.manifest.read_text().replace(
            'rev="'+self.pin+'"','rev="'+'3'*40+'"',1))
        query = mock.Mock()
        with self.assertRaisesRegex(ValueError,'pins disagree'):
            upstream.check(self.manifest,self.lock,query)
        query.assert_not_called()

    def test_lock_resolution_and_missing_or_duplicate_packages_reject(self):
        valid = self.lock.read_text()
        for changed in (valid.replace('#'+self.pin,'#'+'3'*40,1),
                        valid.replace('name = "oneloop"','name = "absent"'),
                        valid+valid[valid.rfind('[[package]]'):],
                        'version = 4\npackage = "bad"\n'):
            self.lock.write_text(changed)
            with self.subTest(lock=changed), self.assertRaises(ValueError):
                upstream.read_pins(self.manifest,self.lock)

    def test_lock_and_manifest_mismatch_and_unknown_repository_reject(self):
        valid = self.lock.read_text()
        self.lock.write_text(valid.replace(self.pin,'3'*40))
        with self.assertRaisesRegex(ValueError,'disagree'):
            upstream.read_pins(self.manifest,self.lock)
        self.lock.write_text(valid.replace('/oneloopmaster.git','/different.git'))
        with self.assertRaisesRegex(ValueError,'Unexpected oneloopmaster'):
            upstream.read_pins(self.manifest,self.lock)

    def test_annotated_tag_is_inventory_not_release_order(self):
        text = 'ref: refs/heads/main\tHEAD\n'+self.pin+'\tHEAD\n'+self.pin+'\trefs/heads/main\n'
        text += '3'*40+'\trefs/tags/v2\n'+'4'*40+'\trefs/tags/v2^{}\n'
        data = upstream.parse_remote(text)
        self.assertEqual(data['tags']['v2'],{'object':'3'*40,'commit':'4'*40})
        with self.assertRaisesRegex(ValueError,'no tag object'):
            upstream.parse_remote(text.replace('3'*40+'\trefs/tags/v2\n',''))
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            upstream.parse_remote(text+self.pin+'\tHEAD\n')

    def test_query_is_read_only_and_bounded(self):
        output = 'ref: refs/heads/main\tHEAD\n'+self.pin+'\tHEAD\n'+self.pin+'\trefs/heads/main\n'
        with mock.patch.object(upstream.subprocess,'run',return_value=mock.Mock(stdout=output)) as run:
            upstream.query(upstream.REPOSITORIES['ninjars'])
        self.assertEqual(run.call_args.args[0],['git','ls-remote','--symref',upstream.REPOSITORIES['ninjars'],
            'HEAD','refs/heads/main','refs/tags/*'])
        self.assertEqual(run.call_args.kwargs['timeout'],30)

    def test_failure_is_structured_and_snapshot_is_never_overwritten(self):
        target = self.work/'checkpoint.json'
        args = ['--manifest',str(self.manifest),'--lockfile',str(self.lock),'--output',str(target)]
        with mock.patch.object(upstream,'check',side_effect=subprocess.TimeoutExpired(['git'],30)), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(upstream.main(args),2)
        self.assertFalse(json.loads(target.read_text())['complete'])
        before = target.read_bytes()
        with mock.patch.object(upstream,'check') as check, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(upstream.main(args),2)
        check.assert_not_called()
        self.assertEqual(target.read_bytes(),before)

    def test_repository_pins_parse_without_network(self):
        pins = upstream.read_pins(ROOT/'vendor/ninjars/Cargo.toml',ROOT/'vendor/ninjars/Cargo.lock')
        self.assertEqual(set(pins),{'ninjars','oneloopmaster'})
        self.assertTrue(all(upstream.SHA.fullmatch(pin) for pin in pins.values()))

    def test_transitive_git_sources_require_explicit_matching_full_revisions(self):
        valid = self.lock.read_text()
        for query, resolved in [('branch=main', '4'*40), ('rev=abcd', '4'*40),
                                ('rev='+'4'*40, '5'*40),
                                ('rev='+'4'*40+'&branch=main', '4'*40)]:
            self.lock.write_text(valid + '\n[[package]]\nname="transitive"\nversion="1.0.0"\n'
                'source="git+https://example.invalid/dependency?%s#%s"\n' % (query,resolved))
            query_remote = mock.Mock()
            with self.subTest(query=query), self.assertRaises(ValueError):
                upstream.check(self.manifest,self.lock,query_remote)
            query_remote.assert_not_called()

    def test_transitive_inventory_retains_repository_revision_and_package_names(self):
        result = upstream.check(self.manifest,self.lock,
            lambda url: self.remote(self.pin if url.endswith('/ninjars') else self.master))
        self.assertFalse(result['update_pending'])
        self.assertEqual(result['git_dependencies'],[
            {'repository':upstream.REPOSITORIES['ninjars'], 'revision':self.pin,
             'packages':['ninjars','ninjars-symbolica']},
            {'repository':upstream.REPOSITORIES['oneloopmaster'], 'revision':self.master,
             'packages':['oneloop']}])


if __name__ == '__main__':
    unittest.main()
