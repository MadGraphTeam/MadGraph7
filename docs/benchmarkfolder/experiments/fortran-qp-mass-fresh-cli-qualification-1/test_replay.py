"""Portable relocation, integrity and effect-guard regressions; no native tools."""
import ctypes
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fresh_cli_portable_replay_test',HERE/'replay.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)


def save_index(root,index):
    (root/'index.json').write_text(json.dumps(index,indent=2,sort_keys=True)+'\n')


def mutate_logical(root,logical,mutator):
    index=json.loads((root/'index.json').read_text());row=index['logical_files'][logical]
    path=root/row['path'];raw=path.read_bytes()
    if row['compression']=='gzip':raw=gzip.decompress(raw)
    raw=mutator(raw)
    stored=gzip.compress(raw,mtime=0) if row['compression']=='gzip' else raw
    path.write_bytes(stored)
    row['raw_sha256']=hashlib.sha256(raw).hexdigest();row['raw_bytes']=len(raw)
    index['files'][row['path']]={'sha256':hashlib.sha256(stored).hexdigest(),'bytes':len(stored)}
    save_index(root,index)


class Portable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=HERE
        cls.seal=hashlib.sha256((HERE/'index.json').read_bytes()).hexdigest()

    def clone(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name)/'relocated capsule';shutil.copytree(self.root,root)
        return root

    def test_clean_full_replay(self):
        result=replay.verify(self.root,self.seal)
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['source'],{'public_raw_exact':25,'private_statement_exact':9,
            'private_helpers':7,'caller_groups':116})
        self.assertEqual(result['build']['compiles'],7)
        self.assertEqual(result['build']['numeric_bindings'],144)
        self.assertEqual((result['coefficient_words'],result['accuracy_words']),(800,200))
        self.assertEqual(result['forbidden_API_attempts'],0)
        self.assertFalse(result['predecessor_full_campaign_replayed'])

    def test_clean_relocation_needs_no_workspace_or_sibling(self):
        result=replay.verify(self.clone(),self.seal)
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['GEN2_status'],'failed-startup-unqualified')

    def test_corrupted_owned_payload_rejected(self):
        root=self.clone();index=json.loads((root/'index.json').read_text())
        row=index['logical_files']['run/candidate/exact-words.dat']
        path=root/row['path'];path.write_bytes(path.read_bytes()+b'corrupt')
        with self.assertRaises(ValueError):replay.verify(root)

    def test_exact_inventory_rejects_extra_and_symlink(self):
        root=self.clone();(root/'extra.txt').write_text('unindexed')
        with self.assertRaises(ValueError):replay.verify(root)
        (root/'extra.txt').unlink();(root/'alias').symlink_to('index.json')
        with self.assertRaises(ValueError):replay.verify(root)

    def test_resigned_promotion_or_broader_scope_claim_rejected(self):
        for key in ('candidate_promoted','performance_acceptance','independent_physical_finite_truth_qualified'):
            root=self.clone();index=json.loads((root/'index.json').read_text());index[key]=True;save_index(root,index)
            with self.assertRaises(ValueError):replay.verify(root)
        root=self.clone();index=json.loads((root/'index.json').read_text())
        index['scope']['all_five_backends_qualified']=True;save_index(root,index)
        with self.assertRaises(ValueError):replay.verify(root)

    def test_resigned_final_word_mutation_rejected(self):
        root=self.clone()
        def change(raw):
            text=raw.decode();lines=text.splitlines(keepends=True)
            fields=lines[0].split();fields[3]='0000000000000000' if fields[3]!='0000000000000000' else '3FF0000000000000'
            lines[0]=' '.join(fields)+'\n';return ''.join(lines).encode()
        mutate_logical(root,'run/candidate/exact-words.dat',change)
        with self.assertRaises(ValueError):replay.verify(root)

    def test_resigned_source_omission_rejected(self):
        root=self.clone();index=json.loads((root/'index.json').read_text())
        del index['logical_files']['fresh-cli/SubProcesses/P0_uux_zgg/coef_construction_1.f']
        save_index(root,index)
        with self.assertRaises((ValueError,KeyError)):replay.verify(root)

    def test_resigned_historical_generation_status_rejected(self):
        root=self.clone()
        mutate_logical(root,'generation/result.json',lambda raw:raw.replace(
            b'generated_pending_source_review',b'passed'))
        with self.assertRaises(ValueError):replay.verify(root)

    def test_resigned_raw_generation_evidence_rejected(self):
        root=self.clone();mutate_logical(root,'generation/generation.log',lambda raw:raw+b'changed\n')
        with self.assertRaises(ValueError):replay.verify(root)

    def test_resigned_predecessor_scope_or_subset_rejected(self):
        root=self.clone();index=json.loads((root/'index.json').read_text())
        index['predecessor']['whole_parent_numerical_replay_claimed']=True;save_index(root,index)
        with self.assertRaises(ValueError):replay.verify(root)
        root=self.clone();index=json.loads((root/'index.json').read_text())
        index['predecessor']['selected_logical_files'].pop();save_index(root,index)
        with self.assertRaises(ValueError):replay.verify(root)

    def test_indexed_native_payload_forbidden(self):
        for filename,data in (('hidden.txt',b'\x7fELF' + b'\0'*32),
                              ('hidden.txt',b'!<arch>\n' + b'\0'*32),
                              ('concealed.o',b'not an executable; extension is still forbidden\n')):
            root=self.clone();index=json.loads((root/'index.json').read_text())
            (root/filename).write_bytes(data)
            digest=hashlib.sha256(data).hexdigest()
            index['files'][filename]={'sha256':digest,'bytes':len(data)}
            index['logical_files'][filename]={'path':filename,'compression':'none',
                'raw_sha256':digest,'raw_bytes':len(data)}
            save_index(root,index)
            with self.assertRaises(ValueError):replay.verify(root)
        root=self.clone();index=json.loads((root/'index.json').read_text());data=b'nonlogical object placeholder\n'
        (root/'physical.o').write_bytes(data)
        index['files']['physical.o']={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        save_index(root,index)
        with self.assertRaises(ValueError):replay.verify(root)

    def test_resigned_excluded_native_identity_mutation_rejected(self):
        root=self.clone();index=json.loads((root/'index.json').read_text())
        index['excluded_binaries'][0]['sha256']='0'*64;save_index(root,index)
        with self.assertRaises(ValueError):replay.verify(root)

    def test_resigned_frozen_source_or_predecessor_mutation_rejected(self):
        root=self.clone();mutate_logical(root,'generation-source/compare.py',lambda raw:raw+b'\n# changed\n')
        with self.assertRaises(ValueError):replay.verify(root)
        root=self.clone();index=json.loads((root/'index.json').read_text());path=root/index['predecessor']['index_path']
        path.write_bytes(path.read_bytes()+b' ')
        index['files'][str(path.relative_to(root))]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
        save_index(root,index)
        with self.assertRaises(ValueError):replay.verify(root)

    def test_resigned_load_or_policy_mutation_rejected(self):
        root=self.clone();mutate_logical(root,'build/commands/policy-before-01.stdout.txt',
            lambda raw:raw.replace(b'-ffixed-line-length-132',b'-ffast-math',1))
        with self.assertRaises(ValueError):replay.verify(root)
        root=self.clone();mutate_logical(root,'build/private-link.map',lambda raw:raw.replace(b'LOAD ',b'FAIL ',1))
        with self.assertRaises(ValueError):replay.verify(root)

    def test_nondeterministic_gzip_rejected_even_if_resigned(self):
        root=self.clone();index=json.loads((root/'index.json').read_text())
        name=next(n for n,r in index['logical_files'].items() if r['compression']=='gzip')
        row=index['logical_files'][name];path=root/row['path'];raw=gzip.decompress(path.read_bytes())
        stored=gzip.compress(raw,mtime=1);path.write_bytes(stored)
        index['files'][row['path']]={'sha256':hashlib.sha256(stored).hexdigest(),'bytes':len(stored)};save_index(root,index)
        with self.assertRaises(ValueError):replay.verify(root)

    def test_guard_denies_external_reads_process_load_network_and_writes(self):
        controls=(lambda:Path('/etc/passwd').read_bytes(),lambda:open('/etc/passwd','rb'),
            lambda:os.open('/etc/passwd',os.O_RDONLY),lambda:io.FileIO('/etc/passwd','r'),
            lambda:os.read(0,1),lambda:os.write(1,b'forbidden'),lambda:subprocess.run(['true']),
            lambda:os.system('true'),lambda:ctypes.CDLL(None),lambda:socket.socket(),
            lambda:Path('/tmp/portable-replay-forbidden').write_text('no'),
            lambda:shutil.copyfile('/etc/passwd','/tmp/portable-replay-forbidden'))
        for control in controls:
            with replay.blocked_apis() as attempts:
                with self.assertRaises(RuntimeError):control()
                self.assertEqual(len(attempts),1)


if __name__=='__main__':unittest.main()
