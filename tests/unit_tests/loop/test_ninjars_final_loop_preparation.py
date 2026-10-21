"""Protect original exports from the loop campaign's directory-copy aliases."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/loop-execution/prepare.py'
spec = importlib.util.spec_from_file_location('final_loop_preparation', SOURCE)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


class TestFinalLoopArchiveLinks(unittest.TestCase):
    def test_internal_links_redirect_without_changing_external_inputs(self):
        (ROOT / '.ninjars-build/tmp').mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / '.ninjars-build/tmp') as directory:
            base = Path(directory)
            original, target = base / 'original', base / 'archive'
            original.mkdir(); target.mkdir()
            (original / 'source.f').write_text('original-source\n')
            (target / 'source.f').write_text('original-source\n')
            external = base / 'external.f'; external.write_text('external-source\n')
            (target / 'internal.f').symlink_to(original / 'source.f')
            (target / 'external.f').symlink_to(external)
            changes = preparation.redirect_internal_links(original, target)
            self.assertEqual([item['path'] for item in changes], ['internal.f'])
            self.assertEqual((target / 'internal.f').resolve(), target / 'source.f')
            self.assertEqual((target / 'external.f').resolve(), external)
            (target / 'internal.f').write_text('worker-private-change\n')
            self.assertEqual((original / 'source.f').read_text(), 'original-source\n')
            self.assertEqual(external.read_text(), 'external-source\n')

    def test_top_level_source_and_target_symlink_aliases_are_rejected(self):
        (ROOT / '.ninjars-build/tmp').mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / '.ninjars-build/tmp') as directory:
            base = Path(directory)
            original, target = base / 'original', base / 'archive'
            original.mkdir(); target.mkdir()
            alias = base / 'top-alias'; alias.symlink_to(original, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'never top-level symlink'):
                preparation.redirect_internal_links(original, alias)
            with self.assertRaisesRegex(ValueError, 'never top-level symlink'):
                preparation.redirect_internal_links(alias, target)


if __name__ == '__main__':
    unittest.main()
