"""Summary of the comparison: per case, over the points MadLoop reports
stable (and separately the unstable ones), the largest relative difference
of the finite part, of the 1/eps coefficient (absolute), the Born ratio."""
import glob, json, os
SP = '/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad'
for f in sorted(glob.glob(os.path.join(SP, 'swj', 'cmp_*.json'))):
    case = os.path.basename(f)[4:-5]
    rows = json.load(open(f))
    opt = sorted(rows[0]['ref'])[-1] if '3' not in rows[0]['ref'] else '3'
    st = [r for r in rows if int(r['ml']['code']) // 100 in (1, 2, 3)]
    un = [r for r in rows if r not in st]
    def stats(rs):
        if not rs:
            return '-'
        df = max(abs(r['ml']['fin'] - r['ref'][opt]['fin']) / abs(r['ref'][opt]['fin']) for r in rs)
        dc = max(abs(r['ml']['c1'] - r['ref'][opt]['c1']) for r in rs)
        d2 = max(abs(r['ml']['c2'] - r['ref'][opt]['c2']) for r in rs)
        br = sorted(r['ml']['born'] / r['ref'][opt]['born'] for r in rs)
        rs_ = sorted(r['sqrts'] for r in rs)
        return 'n=%d rs %.3f-%.3f  fin %.1e  c1 %.1e  c2 %.1e  born ratio %.6f-%.6f' % (
            len(rs), rs_[0], rs_[-1], df, dc, d2, br[0], br[-1])
    print('%-10s stable:   %s' % (case, stats(st)))
    if un:
        print('%-10s unstable: %s' % ('', stats(un)))
