"""Read ELF64 dependency declarations without invoking a loader or executable."""
from pathlib import Path
import struct


def inspect(path):
    data = Path(path).read_bytes()
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('Expected little-endian ELF64')
    offset = struct.unpack_from('<Q', data, 32)[0]
    size, count = struct.unpack_from('<HH', data, 54)
    headers = [struct.unpack_from('<IIQQQQQQ', data, offset + i * size) for i in range(count)]
    interpreters = [data[h[2]:h[2]+h[5]].rstrip(b'\0').decode('ascii') for h in headers if h[0] == 3]
    if len(interpreters) > 1:
        raise ValueError('Multiple ELF interpreters')
    tags = []
    for h in headers:
        if h[0] == 2:
            for i in range(h[2], h[2]+h[5], 16):
                key, value = struct.unpack_from('<QQ', data, i)
                if key == 0:
                    break
                tags.append((key, value))
    needed_offsets = [value for key, value in tags if key == 1]
    if not needed_offsets:
        return {'interpreter': interpreters[0] if interpreters else None, 'needed': []}
    strings = [value for key, value in tags if key == 5]
    if len(strings) != 1:
        raise ValueError('Missing or ambiguous ELF string table')
    maps = [h[2]+strings[0]-h[3] for h in headers if h[0] == 1 and h[3] <= strings[0] < h[3]+h[5]]
    if len(maps) != 1:
        raise ValueError('Cannot map ELF string table')
    needed = [data[maps[0]+n:data.index(b'\0', maps[0]+n)].decode('ascii') for n in needed_offsets]
    if len(set(needed)) != len(needed):
        raise ValueError('Duplicate ELF dependencies')
    return {'interpreter': interpreters[0] if interpreters else None, 'needed': needed}
