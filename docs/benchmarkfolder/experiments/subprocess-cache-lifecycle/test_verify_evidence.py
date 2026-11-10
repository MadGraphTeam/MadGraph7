"""Synthetic path-guard tests only; no model, compiler or native execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location('portable',Path(__file__).with_name('verify_evidence.py'))
VERIFY=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(VERIFY)


class PortableGuards(unittest.TestCase):
    def test_callback_never_reads_unindexed_or_noncanonical_map(self):
        for path in ('/tmp/outside','../outside','a/./b','unindexed.txt'):
            with self.subTest(path=path):
                index={'files':{},'historical_symlinks':[],'compressed_maps':[{'path':path}]}
                requested=[]
                def read(name):
                    requested.append(name)
                    if name!='evidence-index.json': self.fail('Unsafe callback read')
                    return json.dumps(index).encode()
                with self.assertRaises(ValueError):VERIFY.verify(read_bytes=read)
                self.assertEqual(requested,['evidence-index.json'])

    def test_default_reader_rejects_file_and_ancestor_symlinks(self):
        for ancestor in (False,True):
            with self.subTest(ancestor=ancestor),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);(root/'real').mkdir();(root/'real/value').write_bytes(b'x')
                if ancestor:
                    (root/'alias').symlink_to('real',target_is_directory=True);name='alias/value'
                else:
                    (root/'alias').symlink_to('real/value');name='alias'
                index={'files':{name:{'kind':'file','size':1,'sha256':hashlib.sha256(b'x').hexdigest()}},
                       'historical_symlinks':[],'compressed_maps':[]}
                (root/'evidence-index.json').write_text(json.dumps(index))
                with self.assertRaisesRegex(ValueError,'symlink'):VERIFY.verify(root)


if __name__=='__main__':unittest.main(verbosity=2)
