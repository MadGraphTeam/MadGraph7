"""Pure fixtures; fake callbacks never run a compiler, loader, or executable."""
import importlib.util
from pathlib import Path
import struct

import pytest

SPEC = importlib.util.spec_from_file_location('qp_loader_checks', Path(__file__).with_name('loader_checks.py'))
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


def elf(path, loader):
    name = str(loader).encode() + b'\0'
    header = struct.pack('<16sHHIQQQIHHHHHH', b'\x7fELF\x02\x01\x01' + b'\0'*9,
                         2, 62, 1, 0, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    ph = struct.pack('<IIQQQQQQ', 3, 4, 120, 0, 0, len(name), len(name), 1)
    path.write_bytes(header + ph + name)


@pytest.fixture
def runtime(tmp_path):
    loader = tmp_path / 'ld-linux.so'; loader.write_bytes(b'loader fixture')
    library = tmp_path / 'library.so'; library.write_bytes(b'library fixture')
    exe = tmp_path / 'check'; elf(exe, loader)
    expected = {str(p): checks.digest(p) for p in (loader, library)}
    return exe, loader, library, expected


def test_actual_interpreter_and_exact_provider_closure(runtime):
    exe, loader, library, expected = runtime
    calls = []
    def callback(argv, label):
        calls.append((argv, label))
        return f'library.so => {library} (0x123)\n{loader} (0x456)\n'
    result = checks.query_runtime(exe, expected, {'bound': expected}, callback)
    assert calls == [([str(loader), '--list', str(exe)], 'actual-pt-interp-runtime')]
    assert result['files'] == expected and result['loader'] == str(loader)


@pytest.mark.parametrize('missing', ['loader', 'library'])
def test_unbound_provider_rejected_before_callback(runtime, missing):
    exe, loader, library, expected = runtime
    bound = dict(expected); bound.pop(str(loader if missing == 'loader' else library))
    with pytest.raises(ValueError, match='prebound'):
        checks.query_runtime(exe, expected, bound, lambda *_: pytest.fail('loader invoked'))


def test_changed_provider_rejected_before_callback(runtime):
    exe, loader, library, expected = runtime
    library.write_bytes(b'changed')
    with pytest.raises(ValueError, match='missing/changed'):
        checks.query_runtime(exe, expected, expected, lambda *_: pytest.fail('loader invoked'))


@pytest.mark.parametrize('text', ['not found', ''])
def test_missing_runtime_provider_rejected(runtime, text):
    exe, loader, library, expected = runtime
    with pytest.raises(ValueError, match='Unresolved|Empty runtime'):
        checks.query_runtime(exe, expected, expected, lambda *_: text)


def test_executable_mutation_during_query_rejected(runtime):
    exe, loader, library, expected = runtime
    def callback(*_):
        exe.write_bytes(exe.read_bytes() + b'changed')
        return f'{library} (0x123)'
    with pytest.raises(ValueError, match='Executable changed'):
        checks.query_runtime(exe, expected, expected, callback)


def test_missing_extra_provider_cannot_disappear_from_closure(runtime):
    exe, loader, library, expected = runtime
    missing = exe.parent / 'missing-extra.so'
    with pytest.raises(ValueError, match='Missing listed runtime provider'):
        checks.query_runtime(exe, expected, expected,
                             lambda *_: f'{loader} (0x1)\n{library} (0x2)\n{missing} (0x3)\n')


@pytest.mark.parametrize('extra', ['extra.so => ./extra.so (0x123)', 'unknown diagnostic text'])
def test_relative_or_unknown_provider_line_cannot_be_ignored(runtime, extra):
    exe, loader, library, expected = runtime
    with pytest.raises(ValueError, match='Relative/unrecognized'):
        checks.query_runtime(exe, expected, expected,
                             lambda *_: f'{loader} (0x1)\n{library} (0x2)\n{extra}\n')


def test_actual_loads_and_interp_must_be_prebound(runtime):
    exe, loader, library, expected = runtime
    assert checks.requireactualmap('LOAD library.so\n', exe.parent, expected, exe) == {str(library): expected[str(library)]}
    with pytest.raises(ValueError, match='absent or changed'):
        checks.requireactualmap('LOAD library.so\n', exe.parent, {str(loader): expected[str(loader)]}, exe)
    with pytest.raises(ValueError, match='No actual'):
        checks.requireactualmap('', exe.parent, expected, exe)


def test_pinned_helper_tampering_rejected(tmp_path, monkeypatch):
    (tmp_path / 'linker_bind.py').write_text('raise RuntimeError("must not run")')
    monkeypatch.setattr(checks, 'SUPPORT_ROOT', tmp_path)
    with pytest.raises(ValueError, match='Changed pinned helper'):
        checks._support('linker_bind.py')


@pytest.mark.parametrize('tamper', [None, 'compiler', 'wrapper'])
def test_prelink_binds_objects_scripts_crt_wrapper_and_loader(tmp_path, tamper):
    bin_dir = tmp_path / 'bin'; bin_dir.mkdir()
    compiler = bin_dir / 'gfortran'; compiler.write_text('# fake compiler, never executed')
    compiler.chmod(0o700)
    support = tmp_path / 'nix-support'; support.mkdir()
    lib = tmp_path / 'lib'; lib.mkdir()
    loader = lib / 'ld-linux-x86-64.so.2'; loader.write_bytes(b'\x7fELFloader')
    for name in ('gfortran', 'quadmath', 'gcc', 'gcc_s', 'm'):
        (lib / ('lib' + name + '.so')).write_bytes(b'\x7fELFfixture')
    (lib / 'libc-real.so').write_bytes(b'\x7fELFlibc')
    (lib / 'libc.so').write_text(f'GROUP ( {lib / "libc-real.so"} AS_NEEDED ( {loader} ) )')
    (support / 'libc-ldflags').write_text('-L' + str(lib))
    (support / 'add-flags.sh').write_text('# declarative fixture')
    obj = tmp_path / 'unit.o'; obj.write_bytes(b'object')
    crt = lib / 'crtbegin.o'; crt.write_bytes(b'crt object')
    output = tmp_path / 'check'; output.write_bytes(b'old output must not be prebound')
    queries = []
    def callback(argv, label):
        queries.append((argv, label))
        if len(queries) == 1 and tamper:
            changed = compiler if tamper == 'compiler' else support / 'add-flags.sh'
            changed.write_text('replacement during compiler query')
        if argv[-1] == '-print-search-dirs': return 'libraries: =' + str(lib)
        name = argv[-1].split('=', 1)[1]
        path = lib / name
        return str(path) if path.exists() else name
    command = [str(compiler), '-o', str(output), str(obj), '-L'+str(lib), '-lc']
    if tamper:
        with pytest.raises(ValueError, match='missing/changed'):
            checks.prelink(command, tmp_path, {'PATH': str(bin_dir)}, callback)
        return
    result = checks.prelink(command, tmp_path, {'PATH': str(bin_dir)}, callback)
    assert result['completed_before_link'] and queries
    for path in (compiler, obj, crt, loader, lib / 'libc.so', support / 'libc-ldflags', support / 'add-flags.sh'):
        assert result['bound'][str(path)] == checks.digest(path)
    assert str(output) not in result['bound']
    assert all(argv[0] == str(compiler) and argv[-1].startswith('-print-') for argv, _ in queries)
