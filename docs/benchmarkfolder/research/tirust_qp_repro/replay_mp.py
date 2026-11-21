import ctypes, glob, sys
sys.path.insert(0, "/shared/external/madloop_onia/madgraph7_madloop_onia/tools/tirust_capture")
import tirust_trace as t
lib = ctypes.CDLL('/shared/external/madloop_onia/madgraph7_madloop_onia/HEPTools/tirust/lib/libmadloop_tirust.so')
ctx = ctypes.c_void_p()
assert lib.ml_tirust_context_create(ctypes.c_uint32(1), ctypes.byref(ctx)) == 0
def err():
    buf = ctypes.create_string_buffer(4096)
    lib.ml_tirust_last_error(buf, ctypes.c_size_t(4096))
    return buf.value.decode(errors='replace')
for path in sys.argv[1:]:
    for rec in t.records(glob.glob(path)[0]):
        if rec.get('kind') != 5:
            continue
        n, rank = rec['n'], rec['rank']
        count = t.coefficient_count(rank)
        out = ctypes.create_string_buffer(8 * count * 64)
        args = []
        for key in ['shifts', 'invariants', 'masses', 'scale']:
            b = rec[key]
            args += [ctypes.c_char_p(b), ctypes.c_size_t(len(b))]
        lib.ml_tirust_register_plan(ctx, n, rank)
        st = lib.ml_tirust_evaluate_mp(ctx, ctypes.c_int32(n), ctypes.c_int32(rank), *args,
                                       out, ctypes.c_size_t(len(out)), ctypes.c_int32(rec['poles']),
                                       ctypes.c_int32(rec['finite']), ctypes.c_int32(rec['check']),
                                       ctypes.c_int32(0))
        if st != 0:
            print('n=%d rank=%d status=%d (trace %d): %s' % (n, rank, st, rec['status'], err()))
