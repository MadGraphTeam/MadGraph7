"""Offline integrity/completeness tests against the closed actual proof."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
PACKET = HERE.with_name('subprocess-cache-lifecycle-current-qualification-2')

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

VERIFY = load('portable_lifecycle_test', HERE / 'verify_evidence.py').verify
BASE_INDEX = json.loads((PACKET / 'evidence-index.json').read_bytes())
BASE = {name: (PACKET / name).read_bytes() for name in BASE_INDEX['files']}

def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

class Packet:
    def __init__(self):
        self.index = copy.deepcopy(BASE_INDEX)
        self.data = dict(BASE)
    def raw(self, name):
        packed = self.index['compressed_text'].get(name)
        return gzip.decompress(self.data[packed['path']]) if packed else self.data[name]
    def json(self, name):
        return json.loads(self.raw(name))
    def put(self, name, raw):
        old = self.index['compressed_text'].pop(name, None)
        if old:
            self.index['files'].pop(old['path'])
            self.data.pop(old['path'])
        self.data[name] = raw
        self.index['files'][name] = {'kind': 'file', 'sha256': sha(raw), 'size': len(raw)}
    def set_json(self, name, value):
        self.put(name, encoded(value))
    def reader(self, name):
        return encoded(self.index) if name == 'evidence-index.json' else self.data[name]
    def rebind_originals(self):
        """Let semantic tests reach their gate after deliberate self-consistent forgery."""
        binding = self.json('retention-binding.json')
        binding['closed_original_files'] = {name: sha(self.raw(name)) for name in binding['closed_original_files']}
        self.set_json('retention-binding.json', binding)
    def check(self, strict=False):
        return VERIFY(read_bytes=self.reader, derive_summary=not strict)

class TestPortableProof(unittest.TestCase):
    def test_actual_closed_raw_derivation(self):
        result = Packet().check()
        self.assertTrue(result['passed'])
        self.assertEqual((result['physical_workers'], result['measured_physical_calls'], result['tensor_key_calls']), (4, 76, 96))
        self.assertEqual(result['summary']['parity']['physical_status'], 219)
    def test_strict_summary_and_portable_only_relocation(self):
        packet = Packet()
        packet.put('verify_evidence.py', (HERE / 'verify_evidence.py').read_bytes())
        packet.set_json('qualification-summary.json', packet.check()['summary'])
        self.assertTrue(packet.check(strict=True)['passed'])
        with tempfile.TemporaryDirectory(prefix='lifecycle-relocation-') as temporary:
            root = Path(temporary)
            for name, raw in packet.data.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            (root / 'evidence-index.json').write_bytes(encoded(packet.index))
            self.assertTrue(VERIFY(root)['passed'])
            self.assertFalse((root / 'execution/lifecycle_plain.map').exists())
    def test_unindexed_file_fails(self):
        packet = Packet(); packet.index['files'].pop(packet.index['compressed_text']['results.json']['path'])
        with self.assertRaisesRegex(ValueError, 'Unindexed'):
            packet.check()
    def test_corrupted_compressed_payload_fails(self):
        packet = Packet(); name = next(iter(packet.index['compressed_text']))
        item = packet.index['compressed_text'][name]
        packet.put(item['path'], gzip.compress(packet.raw(name) + b' ', mtime=0))
        with self.assertRaisesRegex(ValueError, 'Compressed text binding'):
            packet.check()
    def test_path_traversal_rejected(self):
        packet = Packet(); packet.put('../escape.txt', b'text\n')
        with self.assertRaisesRegex(ValueError, 'Nonportable'):
            packet.check()
    def test_binary_even_if_unused_rejected(self):
        packet = Packet(); packet.put('unused-native.so', b'\x7fELF\x00contents')
        with self.assertRaisesRegex(ValueError, 'Binary in portable text'):
            packet.check()
    def test_retained_generated_source_hash_is_semantic(self):
        packet = Packet()
        name = next(name for name in packet.json('retained-generated-index.json') if name.endswith('.f'))
        packet.put('retained-generated/' + name, packet.raw('retained-generated/' + name) + b'! changed\n')
        with self.assertRaisesRegex(ValueError, 'Retained generated source differs'):
            packet.check()
    def test_missing_generated_source_coverage_rejected(self):
        packet = Packet(); table = packet.json('retained-generated-index.json'); table.pop(next(iter(table)))
        packet.set_json('retained-generated-index.json', table)
        with self.assertRaisesRegex(ValueError, 'Generated retention coverage'):
            packet.check()
    def test_closed_original_log_change_rejected(self):
        packet = Packet(); name = 'execution/native-plain-cache0.txt'
        packet.put(name, packet.raw(name) + b'edited\n')
        with self.assertRaisesRegex(ValueError, 'Original closed evidence changed'):
            packet.check()
    def test_w03_console_rejected_even_after_metadata_rebinding(self):
        packet = Packet(); name = 'execution/native-plain-cache0.txt'
        packet.put(name, packet.raw(name) + b'##W03: omitted contribution\n')
        metadata = packet.json('execution/native-plain-cache0.json'); metadata['log_sha256'] = sha(packet.raw(name))
        packet.set_json('execution/native-plain-cache0.json', metadata); packet.rebind_originals()
        with self.assertRaisesRegex(ValueError, 'Changed/incomplete native console'):
            packet.check()
    def test_unexpected_runtime_library_rejected(self):
        packet = Packet(); name = 'runs/plain-cache0/runtime-maps-close.txt'
        packet.put(name, packet.raw(name) + b'1000-2000 r-xp 00000000 00:21 999 /nix/store/unknown/libbad.so\n')
        outputs = packet.json('runs/plain-cache0/outputs.json')
        outputs['declared_outputs']['runtime-maps-close.txt'].update(sha256=sha(packet.raw(name)), size=len(packet.raw(name)))
        packet.set_json('runs/plain-cache0/outputs.json', outputs); packet.rebind_originals()
        with self.assertRaisesRegex(ValueError, 'Unknown/non-data runtime mapping'):
            packet.check()
    def test_saved_summary_cannot_promote_scope(self):
        packet = Packet(); summary = packet.check()['summary']; summary['performance_eligible'] = True
        packet.set_json('qualification-summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'Saved qualification summary'):
            packet.check(strict=True)
    def test_source_identity_cannot_be_substituted(self):
        packet = Packet(); freeze = packet.json('source-freeze.json'); freeze['implementation_revision'] = '0'*40
        packet.set_json('source-freeze.json', freeze); packet.rebind_originals()
        with self.assertRaisesRegex(ValueError, 'Source/campaign identity'):
            packet.check()
    def test_report_loader_accepts_only_expected_contract(self):
        report = load('retained_report_loader', HERE.parents[1] / 'campaign_report_lifecycle.py')
        verify = report.retained_verifier((HERE / 'verify_evidence.py').read_bytes())
        packet = Packet(); packet.set_json('qualification-summary.json', packet.check()['summary'])
        self.assertTrue(verify(read_bytes=packet.reader)['passed'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
