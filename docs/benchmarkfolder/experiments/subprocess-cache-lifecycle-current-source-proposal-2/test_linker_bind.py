"""Pure file fixtures for pre-link dependency discovery; never execute a linker."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('tested_linker_binding', HERE / 'linker_bind.py')
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)


class LinkerBindingTests(unittest.TestCase):
    def test_fortran_implicit_roots_include_wrapper_quadmath(self):
        names = L.compiler_libraries('-ltirust -lstdc++ -lgcc_s')
        self.assertEqual(set(names), {'tirust', 'stdc++', 'gcc_s', 'gfortran', 'quadmath', 'gcc', 'm', 'c'})
        self.assertEqual(names.count('gcc_s'), 1)

    def test_recursive_group_resolves_relative_soname_and_named_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library, compiler = root / 'wrapper-lib', root / 'compiler-lib'
            library.mkdir(); compiler.mkdir()
            (library / 'libgcc_s.so').write_text('/* GNU ld script */ GROUP ( child.ld -lgcc )\n')
            (library / 'child.ld').write_text('INPUT ( AS_NEEDED ( libgcc_s.so.1 ) )\n')
            (library / 'libgcc_s.so.1').write_bytes(b'\x7fELFfixture')
            (compiler / 'libgcc.a').write_bytes(b'!<arch>\nfixture')
            result = L.prebind({}, [library, compiler], ['gcc_s'])
            self.assertEqual(len(result['files']), 4)
            self.assertEqual(len(result['scripts']), 2)
            for name in result['files']:
                L.require_prebound_load(name, result['files'])

    def test_missing_transitive_input_fails_before_link(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'libfoo.so').write_text('GROUP ( missing.so )\n')
            with self.assertRaisesRegex(ValueError, 'Unresolved pre-link'):
                L.prebind({}, [root], ['foo'])

    def test_thin_archive_without_member_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'libfoo.a').write_bytes(b'!<thin>\n')
            with self.assertRaisesRegex(ValueError, 'Thin archives require'):
                L.prebind({}, [root], ['foo'])

    def test_unknown_actual_load_is_not_added_after_link(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'unbound.so'
            path.write_bytes(b'\x7fELFfixture')
            bound = {}
            with self.assertRaisesRegex(ValueError, 'absent or changed'):
                L.require_prebound_load(path, bound)
            self.assertEqual(bound, {})

    def test_bound_load_mutation_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bound.so'
            path.write_bytes(b'\x7fELFbefore')
            bound = {str(path): L.digest(path)}
            path.write_bytes(b'\x7fELFafter')
            with self.assertRaisesRegex(ValueError, 'absent or changed'):
                L.require_prebound_load(path, bound)

    def test_unsupported_script_syntax_fails_closed(self):
        for source in ('SECTIONS { .text : { *(.text) } }', 'GROUP ( missing.so',
                       'INCLUDE other.ld', 'SEARCH_DIR(=sysroot) GROUP(libfoo.so)',
                       'GROUP(--plugin=untrusted.so)'):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    members, paths = L.script_members(source)
                    # Member options are rejected during resolution.
                    with tempfile.TemporaryDirectory() as temp:
                        path = Path(temp) / 'libfoo.so'
                        path.write_text(source)
                        L.prebind({}, [Path(temp)], ['foo'])

    def test_wrapper_flags_are_bound_and_only_link_flags_used(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'bin').mkdir(); (root / 'nix-support').mkdir()
            compiler = root / 'bin/gfortran'
            compiler.write_text('never executed')
            (root / 'nix-support/add-flags.sh').write_text('shell body is retained, never executed')
            (root / 'nix-support/cc-ldflags').write_text('-L/first -L /second -lgcc_s')
            value = L.wrapper_flags(compiler, {'NIX_LDFLAGS_x86_64_unknown_linux_gnu': '-L/third',
                                             'PRIVATE_TOKEN': 'not a flag'})
            self.assertEqual(len(value['inputs']), 2)
            self.assertNotIn('never executed', value['flags'])
            self.assertNotIn('PRIVATE_TOKEN', value['environment_flags'])
            self.assertEqual(L.search_paths('', 'libraries: =/fourth/:/fifth/', value['flags'], root),
                             list(map(Path, ('/first', '/second', '/third', '/fourth', '/fifth'))))
            self.assertEqual(L.requested_libraries(value['flags']), ['gcc_s'])

    def test_two_explicit_candidates_are_both_prebound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dirs = [root / 'first', root / 'second']
            for path in dirs:
                path.mkdir(); (path / 'libx.so').write_bytes(b'\x7fELFfixture')
            result = L.prebind({}, dirs, ['x'])
            self.assertEqual(len(result['requested_roots']['-lx']), 2)

    def test_script_cycle_is_finite_and_all_nodes_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'libx.so').write_text('INPUT(y.ld)')
            (root / 'y.ld').write_text('INPUT(libx.so)')
            result = L.prebind({}, [root], ['x'])
            self.assertEqual(len(result['scripts']), 2)


if __name__ == '__main__':
    unittest.main()
