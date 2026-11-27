"""Compare the results of this campaign (TIRust 6ba4b53 + onia_loop bridge,
the pin commit of MadGraph7 PR #557 cherry-picked) with the committed
da7902b + v16 records (docs/benchmarkfolder/data/raw/v16all; the 2.5.5
cases with b and t excluded: mg255excl_*). Per file: the largest relative
change of Born and finite parts, the largest absolute change of the
normalised single poles (units of Born alpha_s/(2 pi)), and changed return
codes."""
import glob
import json
import os

W = os.path.dirname(os.path.abspath(__file__))
REF = '/shared/external/madloop_onia/madgraph7_madloop_onia/docs/benchmarkfolder/data/raw/v16all'
REL = ('born', 'finite', 'finite_norm', 'fin', 'c0')
ABS = ('single_norm', 'c1', 'c2', 'double_norm')
SKIP = ('time', 'work', 'path', 'label', 'tirust', 'lib', 'date', 'seconds', 'elapsed', 'old_ml')


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b and not any(s in k.lower() for s in SKIP):
                walk(a[k], b[k], path + [k], out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out['len'].append('/'.join(map(str, path)))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + [i], out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and path:
        key = str(path[-1])
        where = '/'.join(map(str, path))
        if 'code' in key.lower() and a != b:
            out['codes'].append((where, b, a))
        elif key in REL:
            d = abs(a - b) / abs(b) if b else abs(a)
            if d > out['rel'][0]:
                out['rel'] = (d, where, b, a)
        elif key in ABS:
            d = abs(a - b)
            if d > out['abs'][0]:
                out['abs'] = (d, where, b, a)


def pairs():
    for f in sorted(glob.glob(os.path.join(W, 'results', 'cmp_v16_*.json'))):
        yield f, os.path.join(REF, 'mg255excl_%s' % os.path.basename(f)[8:])
    for f in sorted(glob.glob(os.path.join(W, 'results', 'pw_check_*.json'))):
        yield f, os.path.join(REF, os.path.basename(f))
    for name in ('pol_v16.json',):
        yield os.path.join(W, name), os.path.join(REF, 'pol.json')
    for f in sorted(glob.glob(os.path.join(W, '*.json'))):
        b = os.path.basename(f)
        if b not in ('pol_v16.json', 'validate_v16.json'):
            yield f, os.path.join(REF, b)
    yield (os.path.join(W, 'validate_v16.json'),
           os.path.join(os.path.dirname(REF), 'validate_tirustda7902b.json'))


for f, g in pairs():
    rel = os.path.relpath(f, W)
    if not (os.path.isfile(f) and os.path.isfile(g)):
        print('%-34s missing %s' % (rel, g if not os.path.isfile(g) else f))
        continue
    a, b = json.load(open(f)), json.load(open(g))
    out = dict(rel=(0.0, '', None, None), abs=(0.0, '', None, None), codes=[], len=[])
    walk(a, b, [], out)
    print('%-34s rel %.1e%s | abs c1 %.1e%s | %d code changes%s' % (
        rel, out['rel'][0], (' at %s' % out['rel'][1]) if out['rel'][0] > 1e-9 else '',
        out['abs'][0], (' at %s' % out['abs'][1]) if out['abs'][0] > 1e-7 else '',
        len(out['codes']), ('' if not out['len'] else '; lengths differ at %s' % out['len'][:2])))
    for c in out['codes'][:8]:
        print('      code %s: %s -> %s' % c)
