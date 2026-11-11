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
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PACKET = HERE.parent if (HERE.parent/'evidence-index.json').is_file() else HERE.parent/'subprocess-cache-successor-retention-trial-1/capsule'
VERIFIER_PATH = HERE/'verify_evidence.py' if (HERE/'verify_evidence.py').is_file() else HERE.parent/'verify_evidence.py'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

VERIFY = load('portable_lifecycle_test', VERIFIER_PATH).verify
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
        self.assertEqual((result['physical_workers'], result['measured_physical_calls'], result['tensor_key_calls']), (4, 76, 112))
        self.assertEqual(result['summary']['parity']['physical_status'], 219)
    def test_strict_summary_and_portable_only_relocation(self):
        packet = Packet()
        packet.put('verify_evidence.py', (VERIFIER_PATH).read_bytes())
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
        with self.assertRaisesRegex(ValueError, 'Closed command log changed|Original closed evidence changed'):
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

class SuccessorPortableGuards(unittest.TestCase):
    def test_all_83_command_closures_are_required(self):
        packet = Packet()
        name = next(name for name in packet.index['files']
                    if name.startswith('execution/query-') and name.endswith('.json'))
        packet.index['files'].pop(name)
        with self.assertRaisesRegex(ValueError, 'Incomplete closed command registry'):
            packet.check()

    def test_failed_nonphysics_command_is_not_hidden_by_rebinding(self):
        packet = Packet()
        name = next(name for name in packet.index['files']
                    if name.startswith('execution/query-') and name.endswith('.json'))
        value = packet.json(name)
        value['returncode'] = 1
        packet.set_json(name, value)
        packet.rebind_originals()
        with self.assertRaisesRegex(ValueError, 'Unclosed/failed external command'):
            packet.check()

    def test_unexecuted_source1_cannot_be_substituted(self):
        packet = Packet()
        name = 'source1-unexecuted/driver.f.in'
        packet.put(name, packet.raw(name) + b'C altered history\n')
        with self.assertRaisesRegex(ValueError, 'Unexecuted proposal source changed'):
            packet.check()

    def test_failed_latch_recovery_cannot_pass_raw_replay(self):
        packet = Packet()
        name = 'runs/plain-cache0/records.txt'
        lines = packet.raw(name).decode().splitlines(True)
        matched = False
        for i,line in enumerate(lines):
            fields = line.split()
            if fields[:2] == ['LATCH','302']:
                self.assertEqual(fields[-2:], ['F','F'])
                fields[-1] = 'T'
                lines[i] = ' '.join(fields)+'\n'
                matched = True
                break
        self.assertTrue(matched)
        packet.put(name, ''.join(lines).encode())
        outputs = packet.json('runs/plain-cache0/outputs.json')
        outputs['declared_outputs']['records.txt'].update(sha256=sha(packet.raw(name)),size=len(packet.raw(name)))
        packet.set_json('runs/plain-cache0/outputs.json',outputs)
        packet.rebind_originals()
        with self.assertRaisesRegex(ValueError,'Raw lifecycle analysis failed'):
            packet.check()

    def test_absolute_and_parent_paths_rejected_before_external_read(self):
        for name in ('/tmp/outside-cache-sentinel', '../outside-cache-sentinel'):
            packet = Packet()
            packet.put(name,b'outside sentinel\n')
            requested=[]
            def reader(rel):
                requested.append(rel)
                if rel == name:
                    self.fail('Unsafe evidence name reached the physical reader')
                return packet.reader(rel)
            with self.assertRaisesRegex(ValueError,'Nonportable evidence name'):
                VERIFY(read_bytes=reader,derive_summary=True)
            self.assertNotIn(name,requested)

    def test_actual_packet_replay_blocks_live_paths_and_native_apis(self):
        import ctypes
        import encodings.ascii
        import subprocess
        allowed = PACKET.resolve()
        original = Path.open
        attempts=[]
        def guard(path,*a,**kw):
            if not Path(path).absolute().is_relative_to(allowed):
                attempts.append(str(path))
                raise AssertionError('Outside packet read: '+str(path))
            return original(path,*a,**kw)
        def forbidden(*a,**kw):
            raise AssertionError('Native/subprocess API called')
        with patch.object(Path,'open',guard), patch.object(subprocess,'Popen',forbidden), \
                patch.object(subprocess,'run',forbidden),patch.object(ctypes,'CDLL',forbidden), \
                patch.object(ctypes,'PyDLL',forbidden):
            result=VERIFY(PACKET)
        self.assertTrue(result['passed'])
        self.assertEqual(attempts,[])
        self.assertEqual(result['summary']['software_failure_latches']['poisoned_physical_calls'],72)


if __name__ == '__main__':
    unittest.main(verbosity=2)
