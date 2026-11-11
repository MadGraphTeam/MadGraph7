"""Emit immutable Fortran DATA constants from derived source metadata."""
import json
from pathlib import Path


def emit(d):
    routes=sorted(d['routes'],key=lambda x:x['route'])
    assert [r['route'] for r in routes]==list(range(1,27))
    modes=('1','2','4','5')
    out=['! Generated from authenticated source expressions; no event inputs.']
    def array(name,shape,values,logical=False):
        dtype='logical' if logical else 'integer'
        vals=[('.true.' if x else '.false.') if logical else str(x) for x in values]
        out.append(f'  {dtype}, parameter :: {name}({",".join(map(str,shape))}) = reshape([ &')
        for i in range(0,len(vals),12):
            out.append('    '+', '.join(vals[i:i+12])+(', &' if i+12<len(vals) else ' &'))
        out.append('  ], ['+','.join(map(str,shape))+'])')
    array('route_n',(26,),[r['n'] for r in routes])
    array('route_nw',(26,),[r['nw'] for r in routes])
    array('route_we',(4,26),[r['we'][e] if e<r['nw'] else 0 for r in routes for e in range(4)])
    array('route_pairing',(4,26),[r['pairing'][e] if e<r['n'] else 0 for r in routes for e in range(4)])
    array('route_supported',(26,4),[r['modes'][m]['supported'] for m in modes for r in routes],True)
    array('route_ops',(4,26,4),[
        r['modes'][m]['chords'][s]['expanded_add_sub_operations']
        if r['modes'][m]['supported'] and s<r['n'] else 0
        for m in modes for r in routes for s in range(4)])
    array('route_mult',(4,4,26,4),[
        r['modes'][m]['chords'][s]['unsigned_leaf_multiplicities'][e]
        if r['modes'][m]['supported'] and s<r['n'] else 0
        for m in modes for r in routes for s in range(4) for e in range(4)])
    array('route_class',(4,4,26,4),[
        r['modes'][m]['pair_class'][i][j]
        if r['modes'][m]['supported'] and i<r['n'] and j<r['n'] else 0
        for m in modes for r in routes for j in range(4) for i in range(4)])
    return '\n'.join(out)+'\n'


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    print(emit(json.loads((root/'dag-metadata.json').read_text())),end='')
