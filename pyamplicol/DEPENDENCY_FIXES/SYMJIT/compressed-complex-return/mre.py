"""Tiny saved-MIR examples through SymJIT's C API; no compilation is needed.

Usage: python mre.py /path/to/libsymjit.dylib [/path/to/other-library.so ...]
Use --expect-bug only to verify the affected pre-fix libraries.
The four storage-v3 payloads encode inline/called versions of:
  return 3+4i after a real temporary; return 7 after a complex temporary.
They contain portable instructions, no native code or custom function pointers.
The real value is parameter zero, so this reproducer does not require the
separate real-input-location repair.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import tempfile

PROGRAMS = {
    (False, False): "e9080d419587564003000000000000005d4d8efbae44b2c2702e42c7773c2c806d12e0000000000004000000000000000200000000000000020000000000000000000000000000000000000000000000bc27e6ab251ef212000000000000000050000000000000000000000000000000000000000000000001000000000000009e0cb023639b6a870c0000000000000003c0810003c0410204c04104000000000000000001000000000000000000000003000000",
    (True, False): "e9080d419587564003000000000000005d4d8efbae44b2c2702e42c7773c2c806d12e0000000000004000000000000000200000000000000020000000000000000000000000000000000000000000000bc27e6ab251ef212000000000000000050000000000000000000000000000000000000000000000001000000000000009e0cb023639b6a870c0000000000000003c0410203c0810004c04104000000000000000001000000000000000000000003000000",
    (False, True): "e9080d419587564003000000000000005d4d8efbae44b2c2702e42c7773c2c806d12e0000000000004000000000000000200000000000000020000000000000000000000000000000000000000000000bc27e6ab251ef212000000000000000050000000000000000000000000000000000000000000000001000000000000009e0cb023639b6a87370000000000000003c081000b000668656c70657204c04104080866696e69736865640c0668656c70657203c0410208042e7265740c0866696e6973686564000000000000000001000000000000000000000003000000",
    (True, True): "e9080d419587564003000000000000005d4d8efbae44b2c2702e42c7773c2c806d12e0000000000004000000000000000200000000000000020000000000000000000000000000000000000000000000bc27e6ab251ef212000000000000000050000000000000000000000000000000000000000000000001000000000000009e0cb023639b6a87370000000000000003c041020b000668656c70657204c04104080866696e69736865640c0668656c70657203c0810008042e7265740c0866696e6973686564000000000000000001000000000000000000000003000000",
}


def bind(path):
    lib = C.CDLL(str(path))
    for name, args, result in (
        ("load", [C.c_char_p, C.c_void_p], C.c_void_p),
        ("check_status", [C.c_void_p], C.c_char_p),
        ("finalize", [C.c_void_p], None),
        ("create_matrix", [], C.c_void_p),
        ("ptr_params", [C.c_void_p], C.POINTER(C.c_double)),
        ("finalize_matrix", [C.c_void_p], None),
        ("add_row", [C.c_void_p, C.POINTER(C.c_double), C.c_size_t], None),
        ("execute_matrix", [C.c_void_p, C.c_void_p, C.c_void_p], C.c_bool),
    ):
        function = getattr(lib, name)
        function.argtypes = args
        function.restype = result
    return lib


def check(lib, file, count):
    handle = lib.load(str(file).encode(), None)
    assert handle and lib.check_status(handle) == b"Success"
    # Parameter zero has the same location under both historical marker conventions.
    params = lib.ptr_params(handle)
    params[0], params[1] = 7., 0.
    states, outputs = lib.create_matrix(), lib.create_matrix()
    buffers = [(C.c_double * count)(*([value] * count))
               for value in (7., 0., 3., 4., float("nan"), float("nan"))]
    try:
        for row in buffers[:4]:
            lib.add_row(states, row, count)
        for row in buffers[4:]:
            lib.add_row(outputs, row, count)
        assert lib.execute_matrix(handle, states, outputs)
        return [[buffers[4][i], buffers[5][i]] for i in range(count)]
    finally:
        lib.finalize_matrix(outputs)
        lib.finalize_matrix(states)
        lib.finalize(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("libraries", nargs="+", type=Path)
    parser.add_argument("--expect-bug", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="symjit-complex-return-") as temporary:
        for path in args.libraries:
            lib = bind(path.resolve())
            bad = []
            for (real, compressed), hex_data in PROGRAMS.items():
                file = Path(temporary) / f"return-{real}-{compressed}.symjit"
                file.write_bytes(bytes.fromhex(hex_data))
                expected = [7., 0.] if real else [3., 4.]
                for count in (1, 2):
                    values = check(lib, file, count)
                    correct = values == [expected] * count
                    print(json.dumps({"library": str(path), "real_return": real,
                                      "compressed": compressed, "points": count,
                                      "result": values, "expected": expected,
                                      "correct": correct}), flush=True)
                    if not correct:
                        bad.append((real, compressed, count))
            if args.expect_bug:
                assert bad and all(compressed for _, compressed, _ in bad), bad
            else:
                assert not bad, bad


if __name__ == "__main__":
    main()
