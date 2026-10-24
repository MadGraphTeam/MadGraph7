import sys, gzip, os
D = sys.argv[1]
for tag in sys.argv[2:]:
    ev = os.path.join(D, 'Events', tag, 'events.lhe.gz')
    c = {}
    with gzip.open(ev, 'rt') as f:
        inev = False
        for l in f:
            if l.startswith('<event'):
                inev = True; continue
            if inev:
                v = l.split(); npart = int(v[0]); w = float(v[2])
                k = c.setdefault(npart, [0, 0])
                k[0] += 1; k[1] += (w < 0)
                inev = False
    tot = sum(a[0] for a in c.values()); neg = sum(a[1] for a in c.values())
    print(tag, 'r=%.4f' % (neg / tot), ' '.join('n%d: N=%d r=%.4f' % (k, a[0], a[1] / a[0]) for k, a in sorted(c.items())))
