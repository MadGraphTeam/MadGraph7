"""Read-only pre-link closure for explicit libraries and small GNU ld scripts.

No compiler/linker is run here and no post-link map is consulted. The caller
still rejects every actual LOAD absent from the completed pre-link inventory.
"""
import hashlib
from pathlib import Path
import re
import shlex


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def wrapper_flags(compiler, environment):
    """Bind declarative flags from this compiler wrapper; never infer from a map."""
    support = Path(compiler).resolve().parent.parent / 'nix-support'
    inputs, flags = {}, []
    for name in ('add-flags.sh', 'cc-ldflags', 'libc-ldflags', 'libcxx-ldflags'):
        path = support / name
        if path.is_file():
            raw = path.read_bytes()
            text = raw.decode('utf-8')
            inputs[str(path.resolve())] = {'sha256': hashlib.sha256(raw).hexdigest(),
                                         'bytes': len(raw), 'text': text}
            if name != 'add-flags.sh':
                flags.append(text)
    env_flags = {name: value for name, value in environment.items()
                 if name.startswith(('NIX_LDFLAGS', 'NIX_CFLAGS_LINK'))}
    flags.extend(env_flags.values())
    return {'inputs': inputs, 'environment_flags': env_flags, 'flags': '\n'.join(flags)}


def search_paths(info, compiler_search, wrapper_text, cwd):
    result = []
    def add(text):
        path = Path(text)
        if not path.is_absolute():
            path = Path(cwd) / path
        path = path.resolve()
        if path not in result:
            result.append(path)
    tokens = shlex.split(info + '\n' + wrapper_text)
    iterator = iter(tokens)
    for token in iterator:
        if token == '-L':
            value = next(iterator, None)
            require(value is not None, 'Missing library search directory')
            add(value)
        elif token.startswith('-L'):
            add(token[2:])
        elif token.startswith('-Wl,-L,'):
            add(token[len('-Wl,-L,'):])
        elif token.startswith('-Wl,-L'):
            add(token[len('-Wl,-L'):])
    for line in compiler_search.splitlines():
        if line.startswith('libraries: ='):
            for name in line.split('=', 1)[1].split(':'):
                if name:
                    add(name)
    return result


def requested_libraries(text):
    result = []
    iterator = iter(shlex.split(text))
    for token in iterator:
        if token == '-l':
            name = next(iterator, None)
            require(name is not None, 'Missing library name')
        elif token.startswith('-l'):
            name = token[2:]
        else:
            continue
        require(re.fullmatch(r':?[A-Za-z0-9_.+\-]+', name), 'Unsupported library name')
        if name not in result:
            result.append(name)
    return result


def compiler_libraries(text):
    # Same implicit GNU Fortran roots queried by the inherited prebinding code,
    # now resolved against the compiler wrapper directories as well.
    return list(dict.fromkeys(requested_libraries(text) +
                              ['gfortran', 'quadmath', 'gcc', 'gcc_s', 'm', 'c']))


def script_members(text):
    """Strict subset used by the pinned toolchain; unsupported syntax fails closed."""
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    lex = shlex.shlex(text, posix=True, punctuation_chars='(),;')
    lex.whitespace_split = True
    lex.commenters = '#'
    tokens = []
    for token in lex:
        tokens.extend(token if token and set(token) <= set('(),;') else [token])
    cursor, members, paths = 0, [], []
    def take(expected=None):
        nonlocal cursor
        require(cursor < len(tokens), 'Truncated linker script')
        value = tokens[cursor]
        cursor += 1
        require(expected is None or value == expected, 'Unexpected linker script token')
        return value
    def group():
        take('(')
        while cursor < len(tokens) and tokens[cursor] != ')':
            value = take()
            if value == ',':
                continue
            if value in ('AS_NEEDED', 'GROUP', 'INPUT'):
                group()
            else:
                require(value not in ('(', ';') and not value.startswith('='),
                        'Unsupported linker script member')
                members.append(value)
        take(')')
    while cursor < len(tokens):
        directive = take()
        if directive == ';':
            continue
        if directive in ('INPUT', 'GROUP'):
            group()
        elif directive == 'OUTPUT_FORMAT':
            take('(')
            while cursor < len(tokens) and tokens[cursor] != ')':
                require(re.fullmatch(r'[A-Za-z0-9_+.,\-]+', take()), 'Unsupported output format')
            take(')')
        elif directive == 'SEARCH_DIR':
            take('(')
            name = take()
            require(Path(name).is_absolute(), 'Relative/sysroot script SEARCH_DIR unsupported')
            paths.append(name)
            take(')')
        else:
            raise ValueError('Unsupported linker script directive: ' + directive)
    require(members, 'Empty linker script')
    return members, paths


def prebind(initial, search_dirs, requested):
    """Bind candidate inputs before link, then recursively bind script references.

    Multiple explicit search directories can provide the same requested name.
    All such candidates are bound; the actual link map later determines which
    were selected and must remain a strict subset of these pre-bound bytes.
    """
    files = dict(initial)
    queue, scripts, roots = list(files), {}, {}
    directories = [Path(path).resolve() for path in search_dirs]
    def candidates(spelling, parent=None):
        if spelling.startswith('-l'):
            name = spelling[2:]
            require(re.fullmatch(r':?[A-Za-z0-9_.+\-]+', name), 'Unsupported script library name')
            names = [name[1:]] if name.startswith(':') else ['lib' + name + '.so', 'lib' + name + '.a']
            bases = directories
        else:
            require(not spelling.startswith('-') and '\x00' not in spelling, 'Unsupported script member option')
            path = Path(spelling)
            if path.is_absolute():
                names, bases = [str(path)], [Path('/')]
            else:
                names, bases = [spelling], ([parent] if parent is not None else []) + directories
        found = sorted({str((base / name).resolve()) for base in bases for name in names if (base / name).is_file()})
        require(found, 'Unresolved pre-link script/library input: ' + spelling)
        for name in found:
            checksum = digest(name)
            require(name not in files or files[name] == checksum, 'Prebound linker input changed: ' + name)
            if name not in files:
                files[name] = checksum
                queue.append(name)
        return found
    for name in requested:
        roots['-l' + name] = candidates('-l' + name)
    seen = set()
    while queue:
        name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        path = Path(name)
        require(digest(path) == files[name], 'Prebound linker input changed: ' + name)
        if not re.search(r'(\.so(?:\.\d+)*|\.a|\.ld)$', path.name):
            continue
        with path.open('rb') as stream:
            raw = stream.read(65537)
        require(not raw.startswith(b'!<thin>\n'), 'Thin archives require separate member provenance')
        if raw.startswith((b'\x7fELF', b'!<arch>\n')):
            continue
        require(len(raw) <= 65536 and b'\0' not in raw, 'Unsupported non-ELF link input: ' + name)
        members, extra_dirs = script_members(raw.decode('utf-8'))
        for directory in extra_dirs:
            if Path(directory).resolve() not in directories:
                directories.append(Path(directory).resolve())
        edges = [{'spelling': member, 'candidates': candidates(member, path.parent)} for member in members]
        scripts[name] = {'sha256': files[name], 'text': raw.decode('utf-8'),
                         'references': edges, 'search_dirs': extra_dirs}
        require(len(scripts) <= 128 and len(files) <= 4096, 'Unexpected linker script closure size')
    return {'files': files, 'requested_roots': roots, 'scripts': scripts,
            'policy': 'Pre-link candidate closure only; actual LOAD inputs must already be bound.'}


def require_prebound_load(path, files):
    name = str(Path(path).resolve())
    require(name in files and digest(path) == files[name],
            'Actual linker input absent or changed before link: ' + str(path))
