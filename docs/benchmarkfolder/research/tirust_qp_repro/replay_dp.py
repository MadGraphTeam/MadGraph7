import ctypes, glob, sys
sys.path.insert(0,'/shared/external/madloop_onia/madgraph7_madloop_onia/tools/tirust_capture')
import tirust_trace as t
recs=list(t.records(glob.glob(sys.argv[1])[0]))
fails=[r for r in recs if r.get('kind')==1 and r.get('status',0)!=0]
lib = ctypes.CDLL('/shared/external/madloop_onia/madgraph7_madloop_onia/HEPTools/tirust/lib/libmadloop_tirust.so')
ctx = ctypes.c_void_p(); lib.ml_tirust_context_create(ctypes.c_uint32(1), ctypes.byref(ctx))
def err():
    buf = ctypes.create_string_buffer(4096); lib.ml_tirust_last_error(buf, ctypes.c_size_t(4096)); return buf.value.decode()
class C(ctypes.Structure): _fields_=[('re',ctypes.c_double),('im',ctypes.c_double)]
for r in fails:
    n,rank=r['n'],r['rank']
    sh=(ctypes.c_double*len(r['shifts']))(*r['shifts']); iv=(ctypes.c_double*len(r['invariants']))(*r['invariants'])
    m=r['masses']
    ms=(C*n)(*[C(m[2*i],m[2*i+1]) for i in range(n)])
    cnt=t.coefficient_count(rank); out=(C*(4*cnt))()
    lib.ml_tirust_register_plan(ctx,n,rank)
    st=lib.ml_tirust_evaluate_dp(ctx,n,rank,sh,ctypes.c_size_t(len(r['shifts'])),iv,ctypes.c_size_t(len(r['invariants'])),ms,ctypes.c_size_t(n),ctypes.c_double(r['mu2']),out,ctypes.c_size_t(4*cnt),r['poles'],r['finite'],r['check'],0)
    print('n=%d r=%d st=%d %s'%(n,rank,st,err()[:260]))
    print('   masses^2', [round(m[2*i],4) for i in range(n)], ' shifts', [[round(x,4) for x in r['shifts'][4*i:4*i+4]] for i in range(n)])
