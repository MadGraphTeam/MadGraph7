"""Pinned CMS pre-link/ELF helpers; executable calls belong to the callback.

The controller must retain prelink() before linking, call requireactualmap()
afterwards, and only then query_runtime(). No subprocess API is used here.
"""
import ast
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import types

SUPPORT_ROOT = Path(__file__).resolve().parent.parent / 'current-cms-replay-supervisor-1/ownership'
SUPPORT_PINS = {
    'linker_bind.py': 'b54528ebf6b9847f134a7d8ad927c7394607e927397e45169cb644479c8452c6',
    'run.py': 'be6212d7e93f698b551eee35ec577132c0ebb6db73d07cc203262be6319ce861',
    'legacy_qualify.py': 'a408dc73d880a9471defaf6cf4fa9b426e616780b4861c595d608826b6b0f89a',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def _support(name, functions=None, **extra):
    path = SUPPORT_ROOT / name
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == SUPPORT_PINS[name], 'Changed pinned helper: ' + name)
    tree = ast.parse(raw, filename=str(path))
    if functions is not None:
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions]
        require({node.name for node in tree.body} == set(functions), 'Missing pure helper definitions')
    namespace = dict(Path=Path, struct=struct, require=require, digest=digest, os=os, re=re, shlex=shlex, **extra)
    exec(compile(tree, str(path), 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


def _bound(value):
    return value['bound'] if 'bound' in value else value


def _verify(files):
    for name, checksum in files.items():
        require(Path(name).is_absolute() and Path(name).is_file() and digest(name) == checksum,
                'Prebound input missing/changed: ' + name)


def prelink(command, cwd, environment, command_callback):
    """Bind objects, libraries/scripts, wrapper files and compiler CRT queries."""
    cwd = Path(cwd).resolve(); command = list(map(str, command))
    require(command and '-o' in command, 'Expected expanded compiler link command')
    require(not any('@' in arg for arg in command), 'Response files require explicit provenance')
    output = command.index('-o')
    require(output + 1 < len(command), 'Missing link output')
    inputs = command[:output] + command[output + 2:]
    compiler = shutil.which(command[0], path=environment.get('PATH'))
    if '/' in command[0]:
        compiler = str((cwd / command[0]).resolve())
    require(compiler and Path(compiler).is_file(), 'Missing compiler executable')
    compiler = str(Path(compiler).resolve()); inputs[0] = compiler
    queries = []
    def capture(argv, **unused):
        label = 'prelink-query-' + str(len(queries)).zfill(3)
        queries.append({'argv': list(map(str, argv)), 'label': label})
        return command_callback(argv, label)
    linker = _support('linker_bind.py')
    wrapper = linker.wrapper_flags(compiler, environment)
    tools = {compiler: digest(compiler), **{name: item['sha256'] for name, item in wrapper['inputs'].items()}}
    legacy = _support('legacy_qualify.py', {'external_inputs'}, capture=capture, remaining=lambda *args: 10)
    info = shlex.join(inputs)
    initial = legacy.external_inputs(cwd, info, [compiler], None)
    _verify(tools)
    initial.update(tools)
    for token in inputs[1:]:
        if not token.startswith('-') and re.search(r'(\.o|\.a|\.so(?:\.\d+)*|\.ld)$', token):
            path = (cwd / token).resolve()
            require(path.is_file(), 'Missing explicit link input: ' + token)
            initial[str(path)] = digest(path)
    search = capture([compiler, '-print-search-dirs'])
    _verify(tools)
    require(linker.wrapper_flags(compiler, environment) == wrapper, 'Compiler wrapper changed during queries')
    paths = linker.search_paths(info, search, wrapper['flags'], cwd)
    closure = linker.prebind(initial, paths, linker.compiler_libraries(info + '\n' + wrapper['flags']))
    _verify(closure['files'])
    return {'completed_before_link': True, 'command': command, 'cwd': str(cwd), 'bound': closure['files'],
            'closure': closure, 'wrapper': wrapper, 'queries': queries, 'support': dict(SUPPORT_PINS)}


def requireactualmap(map_text, cwd, prebound, executable=None):
    """Fail before a loader call if any actual LOAD/PT_INTERP was not prebound."""
    bound = _bound(prebound); _verify(bound)
    linker = _support('linker_bind.py'); loads = {}
    for spelling in re.findall(r'^LOAD\s+(.+)$', map_text, re.M):
        path = (Path(cwd) / spelling).resolve()
        linker.require_prebound_load(path, bound)
        loads[str(path)] = digest(path)
    require(loads, 'No actual linker LOAD records')
    if executable is not None:
        loader = _support('run.py', {'interpreter'}).interpreter(executable)
        linker.require_prebound_load(loader, bound)
    return loads


def query_runtime(exe, expectedlibraries, prebound, callback):
    """Query the actual prebound PT_INTERP; require exact expected providers."""
    exe = Path(exe).resolve(); checksum = digest(exe)
    bound = _bound(prebound); _verify(bound); _verify(expectedlibraries)
    loader = _support('run.py', {'interpreter'}).interpreter(exe)
    loader_name = str(loader)
    require(loader_name in bound and expectedlibraries.get(loader_name) == bound[loader_name],
            'Actual PT_INTERP is not the expected prebound loader')
    require(all(bound.get(name) == value for name, value in expectedlibraries.items()),
            'Expected runtime provider was not prebound')
    listing = callback([loader_name, '--list', str(exe)], 'actual-pt-interp-runtime')
    require('not found' not in listing, 'Unresolved runtime provider')
    providers = []
    for line in listing.splitlines():
        line = line.strip()
        if not line or re.fullmatch(r'linux-vdso\.so\.1\s+\(0x[0-9a-fA-F]+\)', line):
            continue
        match = re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s+\(0x[0-9a-fA-F]+\)', line)
        require(match is not None, 'Relative/unrecognized runtime provider line: ' + line)
        providers.append(match.group(1))
    require(providers, 'Empty runtime provider listing')
    require(all(Path(p).is_file() for p in providers), 'Missing listed runtime provider')
    paths = {str(Path(p).resolve()) for p in providers}
    paths.add(loader_name)
    files = {p: digest(p) for p in sorted(paths)}
    require(files == expectedlibraries, 'Actual runtime provider closure differs')
    require(digest(exe) == checksum, 'Executable changed during loader query')
    _verify(bound); _verify(expectedlibraries)
    return {'loader': loader_name, 'loader_sha256': files[loader_name], 'listing': listing, 'files': files}
