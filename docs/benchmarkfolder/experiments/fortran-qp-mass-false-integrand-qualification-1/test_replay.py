"""Portable pure replay and adversarial file-only regressions; no native work."""
import ctypes
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('false_portable_replay',HERE/'replay.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
def save(root,index):
    (root/'index.json').write_text(json.dumps(index,indent=2,sort_keys=True)+'\n')
def mutate(root,name,change):
    index=json.loads((root/'index.json').read_text());row=index['logical_files'][name];p=root/row['path'];raw=p.read_bytes();raw=gzip.decompress(raw) if row['compression']=='gzip' else raw;raw=change(raw)
    stored=gzip.compress(raw,mtime=0) if row['compression']=='gzip' else raw;p.write_bytes(stored)
    row.update(raw_bytes=len(raw),raw_sha256=r.sha(raw));index['files'][row['path']]={'bytes':len(stored),'sha256':r.sha(stored)};save(root,index)
class Replay(unittest.TestCase):
    def clone(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);root=Path(temporary.name)/'relocated';shutil.copytree(HERE,root);return root
    def test_complete_replay_and_returned_slot_semantics(self):
        result=r.verify(HERE,draft=True);self.assertEqual(result['status'],'passed');self.assertEqual(result['effects_attempted'],0)
        self.assertEqual(result['coefficient_word_differences'],0);self.assertEqual(result['accuracy_word_differences'],0)
        for arm in result['arms'].values():
            self.assertEqual((arm['physical_points'],arm['returned_point_order_slots'],arm['returned_coefficient_slots'],arm['accuracy_word_slots']),(100,200,800,200));self.assertTrue(arm['overall_and_sole_split_order_duplicate'])
    def test_relocated_packet_requires_no_workspace_or_parent(self):
        result=r.verify(self.clone(),draft=True);self.assertTrue(result['False_False_comparison_passed'])
    def test_corrupt_payload_rejected(self):
        root=self.clone();index=json.loads((root/'index.json').read_text());name=next(iter(index['logical_files'].values()))['path'];(root/name).write_bytes(b'bad')
        with self.assertRaises(ValueError):r.verify(root,draft=True)
    def test_extra_and_alias_rejected(self):
        root=self.clone();(root/'extra').write_text('extra')
        with self.assertRaises(ValueError):r.verify(root,draft=True)
        (root/'extra').unlink();(root/'alias').symlink_to(root/'index.json')
        with self.assertRaises(ValueError):r.verify(root,draft=True)
    def test_resigned_numerical_word_mutation_rejected(self):
        root=self.clone()
        def change(raw):
            rows=raw.decode().splitlines();fields=rows[0].split();fields[3]='3FF0000000000000';rows[0]=' '.join(fields);return ('\n'.join(rows)+'\n').encode()
        mutate(root,'run/candidate/exact-words.dat',change)
        with self.assertRaises(ValueError):r.verify(root,draft=True)
    def test_resigned_failed_startup_status_retag_rejected(self):
        root=self.clone()
        def change(raw):
            value=json.loads(raw);value['status']='passed';return json.dumps(value).encode()
        mutate(root,'failed-source2/terminal.json',change)
        with self.assertRaises(ValueError):r.verify(root,draft=True)
    def test_missing_required_identity_or_source_rejected(self):
        for name in ('worker-inputs/original/Cards/ident_card.dat','source2/controller.py'):
            root=self.clone();index=json.loads((root/'index.json').read_text());row=index['logical_files'].pop(name);index['files'].pop(row['path']);(root/row['path']).unlink();save(root,index)
            with self.assertRaises(ValueError):r.verify(root,draft=True)
    def test_resigned_scope_and_excluded_identity_rejected(self):
        for field in ('scope','excluded_binaries'):
            root=self.clone();index=json.loads((root/'index.json').read_text())
            if field=='scope':index[field]['production_promotion']=True
            else:
                key=next(k for k in index[field] if k.endswith('.o'));index[field][key]['sha256']='0'*64
            save(root,index)
            with self.assertRaises(ValueError):r.verify(root,draft=True)
    def test_indexed_native_payload_rejected(self):
        root=self.clone();index=json.loads((root/'index.json').read_text());raw=b'\x7fELFnot-a-program';(root/'native.bin').write_bytes(raw);index['files']['native.bin']={'bytes':len(raw),'sha256':r.sha(raw)};save(root,index)
        with self.assertRaises(ValueError):r.verify(root,draft=True)
    def test_nondeterministic_gzip_rejected_even_if_resigned(self):
        root=self.clone();index=json.loads((root/'index.json').read_text());row=next(x for x in index['logical_files'].values() if x['compression']=='gzip');p=root/row['path'];raw=gzip.decompress(p.read_bytes());stored=gzip.compress(raw,mtime=1);p.write_bytes(stored);index['files'][row['path']]={'bytes':len(stored),'sha256':r.sha(stored)};save(root,index)
        with self.assertRaises(ValueError):r.verify(root,draft=True)
    def test_guard_blocks_external_io_native_process_network_and_limits(self):
        with r.blocked_apis() as attempts:
            for action in (lambda:Path('/outside').read_bytes(),lambda:subprocess.Popen(['forbidden']),lambda:ctypes.CDLL('forbidden'),lambda:socket.socket(),lambda:resource.setrlimit(resource.RLIMIT_CORE,(0,0)),lambda:os.system('forbidden')):
                with self.assertRaises(RuntimeError):action()
            self.assertEqual(len(attempts),6)
if __name__=='__main__':unittest.main()
