"""Per-call DP vs QP deviation of the TIRust arrays (relative to the call's
largest QP entry), for kind 0 (plain) and 1 (powers)."""
import struct, sys, json
def h2d(h):
    return struct.unpack('>d', bytes.fromhex(h))[0]
def read(path):
    calls, cur = [], None
    for line in open(path):
        if line.startswith('C '):
            f = line.split(); cur = dict(kind=int(f[3]), n=int(f[4]), rank=int(f[5]), nc=int(f[6]), st=int(f[7]), v=[]); calls.append(cur)
        else:
            x = [h2d(h) for h in line.split()]
            cur['v'].append([complex(x[2*j], x[2*j+1]) for j in range(4)])
    return calls
base = sys.argv[1]
dp, qp = read(base + '_dp.dump'), read(base + '_qp.dump')
rows = []
for i, (a, b) in enumerate(zip(dp, qp)):
    if a['st'] or b['st']:
        rows.append((i, a['kind'], a['n'], a['rank'], None, None)); continue
    fin = max(abs(r[3]) for r in b['v']) or 1.
    dfin = max(abs(x[3] - y[3]) for x, y in zip(a['v'], b['v']))
    rows.append((i, a['kind'], a['n'], a['rank'], dfin / fin, fin))
for kind in (0, 1):
    sel = [r for r in rows if r[1] == kind and r[4] is not None]
    sel.sort(key=lambda r: -r[4])
    print('kind', kind, 'calls', len(sel), 'worst:', ['%d N%d r%d %.1e' % (r[0], r[2], r[3], r[4]) for r in sel[:6]])
    import statistics
    print('   median %.1e' % statistics.median(r[4] for r in sel))
