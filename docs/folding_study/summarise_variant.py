import sys, glob, gzip, re, os
D, tag, wall = sys.argv[1], sys.argv[2], int(sys.argv[3])
cpu = {0: 0., 1: 0., 2: 0.}
treal = {0: 0., 1: 0., 2: 0.}
for f in glob.glob(os.path.join(D, 'vlogs', tag, '*log_MINT*.txt')):
    m = int(re.search(r'log_MINT(\d)', f).group(1))
    for l in open(f, errors='replace'):
        if 'Time spent in Total' in l:
            cpu[m] += float(l.split()[-1])
        if 'Time spent in Reals_evaluation' in l:
            treal[m] += float(l.split()[-1])
npos = nneg = 0
wpos = wneg = 0.
ev = os.path.join(D, 'Events', tag, 'events.lhe.gz')
with gzip.open(ev, 'rt') as f:
    inev = False
    for l in f:
        if l.startswith('<event'):
            inev = True
            continue
        if inev:
            w = float(l.split()[2])
            if w < 0:
                nneg += 1; wneg += -w
            else:
                npos += 1; wpos += w
            inev = False
xs = ''
for l in open(os.path.join(D, 'launch_%s.log' % tag), errors='replace'):
    if 'Total cross section' in l:
        xs = l.split(':')[1].strip()
n = npos + nneg
r = nneg / n
neff = n * (1 - 2 * r) ** 2
print('%-10s xs=%-22s N=%d r=%.4f Neff=%.0f CPU(M0,M1,M2)=%.0f,%.0f,%.0f real(M1,M2)=%.0f,%.0f wall=%ds CPU12/Neff=%.4f' % (
    tag, xs, n, r, neff, cpu[0], cpu[1], cpu[2], treal[1], treal[2], wall, (cpu[1] + cpu[2]) / neff))
