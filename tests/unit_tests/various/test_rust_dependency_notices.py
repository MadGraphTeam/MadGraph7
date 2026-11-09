"""Original Cargo notices survive installation/export without interpretation."""

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from madgraph import MadGraph5Error
from madgraph.various import rust_dependency_notices as notices


class TestRustDependencyNotices(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bridge, self.lib = self.root / 'bridge', self.root / 'lib'
        self.bridge.mkdir()
        (self.bridge / 'Cargo.lock').write_text('version = 4\n')
        self.source = self.root / 'checkout'
        self.crate = self.source / 'crates/symbolica'
        self.crate.mkdir(parents=True)
        (self.source / 'Cargo.toml').write_text('[workspace]\nmembers=["crates/symbolica"]\n')
        (self.source / 'License.md').write_bytes(b'Full original runtime license\r\n\xff\x00\n')
        (self.source / 'README.md').write_bytes(b'Upstream source caveat.\n')
        (self.crate / 'Cargo.toml').write_text('[package]\nname="symbolica"\nversion="1.0.0"\n')
        (self.crate / 'AUTHORS').write_text('Original author names\n')
        (self.crate / 'LICENSES').mkdir()
        (self.crate / 'LICENSES/custom.txt').write_bytes(b'Full additional terms\n')
        (self.crate / 'src').mkdir()
        (self.crate / 'src/README.md').write_bytes(b'Nested source readme\n')
        (self.crate / 'src/lib.rs').write_bytes(b'not a notice\n')
        self.metadata = {'packages': [{
            'id': 'git+https://example.test/symbolica#symbolica@1.0.0',
            'name': 'symbolica', 'version': '1.0.0',
            'manifest_path': str(self.crate / 'Cargo.toml'),
            'license': None, 'license_file': '../../License.md',
            'authors': ['Original authors'], 'repository': 'https://example.test/symbolica',
            'source': 'git+https://example.test/symbolica?rev=abc#abc'}]}

    def package(self, **kwargs):
        response = SimpleNamespace(stdout=json.dumps(self.metadata), stderr='')
        with mock.patch.object(notices.subprocess, 'run', return_value=response) as run:
            result = notices.package(self.bridge, self.lib, 'tirust', **kwargs)
        return result, run

    def test_exact_original_bytes_declarations_and_deterministic_archive(self):
        original = {str(path): path.read_bytes() for path in self.source.rglob('*') if path.is_file()}
        result, run = self.package(environment={'CARGO_NET_OFFLINE': 'true'})
        self.assertEqual(run.call_args.args[0], ['cargo', 'metadata', '--format-version=1', '--locked'])
        self.assertEqual(run.call_args.kwargs['env'], {'CARGO_NET_OFFLINE': 'true'})
        self.assertEqual(result['package_count'], 1)
        self.assertEqual(notices.export_files(self.lib, 'tirust', result), result['files'])
        provenance = json.loads((self.lib / result['files'][1]).read_text())
        package = provenance['packages'][0]
        self.assertIsNone(package['license'])
        self.assertEqual(package['license_file'], '../../License.md')
        self.assertEqual(package['source'], self.metadata['packages'][0]['source'])
        with tarfile.open(self.lib / result['files'][0], 'r:gz') as archive:
            members = archive.getmembers()
            self.assertTrue(all(member.isfile() for member in members))
            self.assertTrue(all(not member.name.startswith('/') and '..' not in Path(member.name).parts
                                for member in members))
            by_name = {entry['archive_path']: entry for entry in package['files']}
            for member in members:
                contents = archive.extractfile(member).read()
                entry = by_name[member.name]
                self.assertEqual(contents, original[entry['source_path']])
                self.assertEqual(hashlib.sha256(contents).hexdigest(), entry['sha256'])
            relative = {entry['source_relative_path'] for entry in package['files']}
            self.assertIn('License.md', relative)
            self.assertIn('README.md', relative)
            self.assertIn('crates/symbolica/LICENSES/custom.txt', relative)
            self.assertIn('crates/symbolica/src/README.md', relative)
            self.assertNotIn('crates/symbolica/src/lib.rs', relative)
        again, _ = self.package()
        self.assertEqual(result, again)
        self.assertEqual(original, {str(path): path.read_bytes() for path in self.source.rglob('*')
                                    if path.is_file()})
        self.assertIn('Symbolica runtime rights are not included or transferred',
                      (self.lib / result['files'][2]).read_text())

    def test_distinct_package_ids_cannot_overwrite_same_name_and_version(self):
        another = dict(self.metadata['packages'][0])
        another['id'] = 'git+https://example.test/another#symbolica@1.0.0'
        self.metadata['packages'].append(another)
        result, _ = self.package()
        with tarfile.open(self.lib / result['files'][0], 'r:gz') as archive:
            names = archive.getnames()
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(len({name.split('/')[0] for name in names}), 2)

    def test_registry_boundary_does_not_include_ancestor_notices(self):
        (self.crate / '.cargo-checksum.json').write_text('{}')
        self.metadata['packages'][0]['license_file'] = None
        result, _ = self.package()
        package = json.loads((self.lib / result['files'][1]).read_text())['packages'][0]
        self.assertEqual(package['source_root'], str(self.crate))
        self.assertNotIn('License.md', {entry['source_relative_path'] for entry in package['files']})

    def test_source_symlink_bytes_preserved_only_inside_boundary(self):
        (self.crate / 'NOTICE').symlink_to('../../License.md')
        result, _ = self.package()
        package = json.loads((self.lib / result['files'][1]).read_text())['packages'][0]
        notice = next(entry for entry in package['files'] if entry['source_relative_path'].endswith('/NOTICE'))
        self.assertEqual(notice['sha256'], hashlib.sha256((self.source / 'License.md').read_bytes()).hexdigest())
        (self.crate / 'NOTICE').unlink()
        outside = self.root / 'outside'
        outside.write_text('must not package outside contents')
        (self.crate / 'NOTICE').symlink_to(outside)
        with self.assertRaisesRegex(MadGraph5Error, 'inside its source'):
            self.package()

    def test_declared_license_escape_and_notice_directory_symlink_rejected(self):
        self.metadata['packages'][0]['license_file'] = '../../../outside'
        (self.root / 'outside').write_text('outside source')
        with self.assertRaisesRegex(MadGraph5Error, 'inside its source'):
            self.package()
        self.metadata['packages'][0]['license_file'] = None
        (self.crate / 'COPYING').symlink_to(self.crate / 'src', target_is_directory=True)
        with self.assertRaisesRegex(MadGraph5Error, 'directory is a symlink'):
            self.package()

    def test_export_preserves_legacy_but_rejects_partial_or_tampered_notices(self):
        self.assertEqual(notices.export_files(self.lib, 'tirust'), [])
        with self.assertRaisesRegex(MadGraph5Error, 'Incomplete'):
            notices.export_files(self.lib, 'tirust', {'files': []})
        result, _ = self.package()
        (self.lib / result['files'][0]).write_bytes(b'replaced')
        with self.assertRaisesRegex(MadGraph5Error, 'modified'):
            notices.export_files(self.lib, 'tirust', result)
        (self.lib / result['files'][0]).unlink()
        with self.assertRaisesRegex(MadGraph5Error, 'Incomplete'):
            notices.export_files(self.lib, 'tirust')

    def test_metadata_failures_raise_without_publishing(self):
        for failure in (FileNotFoundError('cargo missing'),
                        subprocess.CalledProcessError(1, ['cargo'], stderr='locked graph unavailable')):
            with mock.patch.object(notices.subprocess, 'run', side_effect=failure):
                with self.assertRaisesRegex(MadGraph5Error, 'Cannot resolve'):
                    notices.package(self.bridge, self.lib, 'tirust')
        with mock.patch.object(notices.subprocess, 'run', return_value=SimpleNamespace(stdout='invalid')):
            with self.assertRaisesRegex(MadGraph5Error, 'Cannot resolve'):
                notices.package(self.bridge, self.lib, 'tirust')
        self.assertFalse(self.lib.exists())

    def test_backend_cannot_escape_output_directory(self):
        with self.assertRaisesRegex(MadGraph5Error, 'backend name'):
            notices.package(self.bridge, self.lib, '../escape')


if __name__ == '__main__':
    unittest.main()
