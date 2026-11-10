"""Lossless deterministic gzip chunks for an explicitly supplied closed journal.

The descriptor contains no source path. Chunk paths are relative to the supplied
destination directory; decoding opens each regular file once, without following
symlinks. No native program, journal parser, or campaign discovery is involved.
"""
import gzip
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import zlib

SCHEMA = 'madloop-journal-gzip-v1'
MAX_RAW_CHUNK = 32 * 1024 * 1024
MAX_GZIP_CHUNK = 40 * 1024 * 1024
DEFAULT_RAW_BUDGET = 128 * 1024 * 1024
_BLOCK = 64 * 1024


def _size(value, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError('Invalid or excessive byte count')
    return value


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid SHA-256')
    return value


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value or '\0' in value:
        raise ValueError('Unsafe chunk path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != value or value == '.':
        raise ValueError('Unsafe chunk path')
    return path


def _directory_fd(path, *, create=False):
    """Walk from / with O_NOFOLLOW, including every caller-supplied component."""
    path = Path(path).absolute()
    if '..' in path.parts:
        raise ValueError('Unsafe directory path')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _regular_file(root_fd, relative):
    descriptor = os.dup(root_fd)
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                          dir_fd=descriptor)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            os.close(file_fd)
            raise ValueError('A regular file is required')
        return os.fdopen(file_fd, 'rb')
    finally:
        os.close(descriptor)


def encode_journal(source: Path, destination_directory: Path,
                   raw_chunk_bytes=MAX_RAW_CHUNK, total_raw_budget=DEFAULT_RAW_BUDGET):
    """Capture one closed source once; exclusively create deterministic chunks.

    Existing directories are allowed; existing chunk files are never replaced.
    Failed attempts remove only the chunk files created by this call. An empty
    source has one valid empty gzip member. The returned JSON-compatible schema
    has raw_sha256/raw_bytes and ordered chunks, each with path, raw_sha256,
    raw_bytes, gzip_sha256 and gzip_bytes.
    """
    _size(total_raw_budget)
    if _size(raw_chunk_bytes, MAX_RAW_CHUNK) == 0:
        raise ValueError('Chunk size must be positive')
    source = Path(source)
    source_directory = _directory_fd(source.parent)
    try:
        captured = _regular_file(source_directory, _relative(source.name))
    finally:
        os.close(source_directory)
    with captured:
        before = os.fstat(captured.fileno())
        _size(before.st_size, total_raw_budget)
        destination = _directory_fd(destination_directory, create=True)
        created, chunks = [], []
        raw_hash, raw_bytes = hashlib.sha256(), 0
        try:
            while True:
                raw = captured.read(min(raw_chunk_bytes, total_raw_budget - raw_bytes + 1))
                if not raw and chunks:
                    break
                raw_bytes += len(raw)
                _size(raw_bytes, total_raw_budget)
                name = f'chunk-{len(chunks):06d}.gz'
                file_fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                  0o600, dir_fd=destination)
                created.append(name)
                with os.fdopen(file_fd, 'wb') as output:
                    with gzip.GzipFile(filename='', mode='wb', compresslevel=9,
                                       mtime=0, fileobj=output) as compressed:
                        compressed.write(raw)
                raw_hash.update(raw)
                with _regular_file(destination, _relative(name)) as compressed:
                    size = _size(os.fstat(compressed.fileno()).st_size, MAX_GZIP_CHUNK)
                    gzip_hash = hashlib.sha256()
                    for block in iter(lambda: compressed.read(_BLOCK), b''):
                        gzip_hash.update(block)
                chunks.append({'path': name, 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                               'raw_bytes': len(raw), 'gzip_sha256': gzip_hash.hexdigest(),
                               'gzip_bytes': size})
                if not raw:
                    break
            after = os.fstat(captured.fileno())
            identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if raw_bytes != before.st_size or identity(before) != identity(after):
                raise ValueError('Journal changed during capture')
            return {'schema': SCHEMA, 'raw_sha256': raw_hash.hexdigest(),
                    'raw_bytes': raw_bytes, 'chunks': chunks}
        except BaseException:
            for name in created:
                os.unlink(name, dir_fd=destination)
            raise
        finally:
            os.close(destination)


def decode_journal(descriptor, root: Path, total_raw_budget=DEFAULT_RAW_BUDGET) -> bytes:
    """Validate all declared budgets first, then stream each single gzip member.

    Actual compressed bytes and decompressed bytes are bounded independently;
    truncated, corrupt, concatenated or trailing gzip data are rejected.
    """
    _size(total_raw_budget)
    if not isinstance(descriptor, dict) or set(descriptor) != {'schema', 'raw_sha256', 'raw_bytes', 'chunks'}:
        raise ValueError('Invalid journal descriptor')
    if descriptor['schema'] != SCHEMA:
        raise ValueError('Unexpected journal schema')
    _sha(descriptor['raw_sha256'])
    _size(descriptor['raw_bytes'], total_raw_budget)
    chunks = descriptor['chunks']
    if not isinstance(chunks, list) or not chunks:
        raise ValueError('At least one gzip chunk is required')
    total, paths = 0, set()
    for chunk in chunks:
        if not isinstance(chunk, dict) or set(chunk) != {'path', 'raw_sha256', 'raw_bytes', 'gzip_sha256', 'gzip_bytes'}:
            raise ValueError('Invalid chunk descriptor')
        path = _relative(chunk['path'])
        if path in paths:
            raise ValueError('Repeated chunk path')
        paths.add(path)
        total += _size(chunk['raw_bytes'], MAX_RAW_CHUNK)
        _size(total, total_raw_budget)
        if _size(chunk['gzip_bytes'], MAX_GZIP_CHUNK) == 0:
            raise ValueError('An empty file is not gzip')
        _sha(chunk['raw_sha256'])
        _sha(chunk['gzip_sha256'])
    if total != descriptor['raw_bytes']:
        raise ValueError('Declared total does not equal chunk sizes')
    directory = _directory_fd(root)
    result, full_hash = bytearray(), hashlib.sha256()
    try:
        for chunk in chunks:
            raw_bytes, gzip_bytes = 0, 0
            raw_hash, gzip_hash = hashlib.sha256(), hashlib.sha256()
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
            with _regular_file(directory, _relative(chunk['path'])) as source:
                if os.fstat(source.fileno()).st_size != chunk['gzip_bytes']:
                    raise ValueError('Compressed size mismatch')
                for block in iter(lambda: source.read(_BLOCK), b''):
                    gzip_bytes += len(block)
                    if gzip_bytes > chunk['gzip_bytes']:
                        raise ValueError('Compressed size changed during reading')
                    gzip_hash.update(block)
                    pending = block
                    while pending:
                        try:
                            raw = decompressor.decompress(pending, chunk['raw_bytes'] - raw_bytes + 1)
                        except zlib.error as error:
                            raise ValueError('Corrupt gzip stream') from error
                        raw_bytes += len(raw)
                        if raw_bytes > chunk['raw_bytes']:
                            raise ValueError('Decompressed size exceeds declaration')
                        if decompressor.unused_data:
                            raise ValueError('Trailing or concatenated gzip data')
                        raw_hash.update(raw)
                        full_hash.update(raw)
                        result.extend(raw)
                        pending = decompressor.unconsumed_tail
                if not decompressor.eof or raw_bytes != chunk['raw_bytes'] or gzip_bytes != chunk['gzip_bytes']:
                    raise ValueError('Truncated gzip or byte count mismatch')
            if raw_hash.hexdigest() != chunk['raw_sha256'] or gzip_hash.hexdigest() != chunk['gzip_sha256']:
                raise ValueError('Chunk checksum mismatch')
        if len(result) != descriptor['raw_bytes'] or full_hash.hexdigest() != descriptor['raw_sha256']:
            raise ValueError('Original journal checksum mismatch')
        return bytes(result)
    finally:
        os.close(directory)
