#!/usr/bin/env python3
"""Portable, offline integrity/semantic regression checks. No native execution."""
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('preparation_verifier', HERE / 'verify.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def seal(folder):
    files = {p.name: {'sha256': v.sha(p.read_bytes()), 'bytes': p.stat().st_size}
             for p in folder.iterdir() if p.is_file() and p.name != 'index.json'}
    (folder / 'index.json').write_text(json.dumps({'files': files}, sort_keys=True))


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='tirust-preparation-portable-')
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / 'relocated'
        shutil.copytree(HERE, self.folder)

    def mutate_record(self, logical_path, change):
        path = self.folder / 'evidence.json.gz'
        bundle = json.loads(gzip.decompress(path.read_bytes()))
        record = bundle['files'][logical_path]
        data = json.loads(bundle['objects'][record['sha256']])
        change(data)
        rendered = json.dumps(data, sort_keys=True, indent=2) + '\n'
        digest = v.sha(rendered.encode())
        bundle['objects'][digest] = rendered
        # Mutate every logical reference to shared content, preserving the
        # deduplicated count so the test reaches the semantic guard.
        for name, value in bundle['files'].items():
            if value['sha256'] == record['sha256']:
                bundle['files'][name] = {'sha256': digest, 'bytes': len(rendered.encode())}
        used = {r['sha256'] for r in bundle['files'].values()}
        bundle['objects'] = {k: value for k, value in bundle['objects'].items() if k in used}
        path.write_bytes(gzip.compress(json.dumps(bundle).encode(), mtime=0))
        inventory = json.loads((self.folder / 'inventory.json').read_text())
        inventory['files'] = bundle['files']
        inventory['unique_text_contents'] = len(bundle['objects'])
        (self.folder / 'inventory.json').write_text(json.dumps(inventory))
        seal(self.folder)

    def test_relocated_complete_packet(self):
        result = v.verify(self.folder)
        self.assertEqual((result['stages'], result['physical_cards'], result['cases']), (5, 22, 11))
        self.assertEqual(result['amplitude_evaluations'], 0)

    def test_unindexed_file_rejected(self):
        (self.folder / 'extra.txt').write_text('unexpected')
        with self.assertRaisesRegex(ValueError, 'unindexed'):
            v.verify(self.folder)

    def test_corrupted_container_rejected(self):
        with (self.folder / 'evidence.json.gz').open('ab') as f:
            f.write(b'!')
        with self.assertRaisesRegex(ValueError, 'packet bytes'):
            v.verify(self.folder)

    def test_failed_stage_rejected_after_container_reseal(self):
        self.mutate_record(v.PREP + 'status.json', lambda d: d['stages'][0].update(returncode=1))
        with self.assertRaisesRegex(ValueError, 'stage outcome'):
            v.verify(self.folder)

    def test_surviving_child_rejected_after_container_reseal(self):
        self.mutate_record(v.PREP + 'status.json', lambda d: d['stages'][0]['cleanup'].update(survivors=[123]))
        with self.assertRaisesRegex(ValueError, 'child process cleanup'):
            v.verify(self.folder)

    def test_missing_fresh_build_rejected_after_container_reseal(self):
        self.mutate_record(v.BASE + 'inputs/candidate/qcd_n1_uux_z/compilation.json', lambda d: d.update(full_numerical_rebuild_this_invocation=False))
        with self.assertRaisesRegex(ValueError, 'object build policy'):
            v.verify(self.folder)

    def test_ancestral_card_mismatch_rejected_after_container_reseal(self):
        self.mutate_record(v.BASE + 'inputs/candidate/qcd_n1_uux_z/generation.json', lambda d: d.update(parameter_card_sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'physical card ancestry'):
            v.verify(self.folder)

    def test_incomplete_card_check_rejected_after_container_reseal(self):
        self.mutate_record(v.PREP + 'build-candidate/cards-after.json', lambda d: d.pop('candidate/qcd_n1_uux_z'))
        with self.assertRaisesRegex(ValueError, 'all 22 cards'):
            v.verify(self.folder)


if __name__ == '__main__':
    unittest.main(verbosity=2)
