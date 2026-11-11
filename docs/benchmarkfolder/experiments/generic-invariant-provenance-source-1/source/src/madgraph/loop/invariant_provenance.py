"""Private generic momentum-provenance proposal; no scientific correction.

Register at the actual optimized reduction call, then admit only rendered
same-kind momentum producers. Failed/unknown joins produce all-zero support.
This module does not inspect numerical inputs, change masses, or choose a reducer.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from madgraph.loop.invariant_source import (blocks, call, external_rule,
    internal_rule, statements, wf_number, require)

MODES = (1, 2, 4, 5)

@dataclass(frozen=True)
class Expr:
    signed: tuple
    mult: tuple
    ops: int

    def neg(self):
        return Expr(tuple(-x for x in self.signed), self.mult, self.ops)

    def add(self, other):
        require(len(self.signed) == len(other.signed), 'DAG dimension')
        return Expr(tuple(a+b for a,b in zip(self.signed,other.signed)),
                    tuple(a+b for a,b in zip(self.mult,other.mult)),
                    self.ops+other.ops+1)

    def record(self):
        return dict(signed=list(self.signed), mult=list(self.mult), ops=self.ops)


def zero(e):
    return Expr((0,)*e, (0,)*e, 0)


def seeded(nodes, e):
    x=zero(e)
    for node in nodes: x=x.add(node)
    return x


def proportional(a, c):
    require(len(a)==len(c) and any(c), 'conservation shape')
    k=next(i for i,x in enumerate(c) if x)
    q,r=divmod(a[k],c[k])
    return not r and all(x==q*y for x,y in zip(a,c))


def classify(a, conservation, protected):
    if proportional(a,conservation): return 1
    for e in range(len(a)):
        for sign in (-1,1):
            b=list(a);b[e]-=sign
            if proportional(b,conservation): return 1 if protected[e] else 0
    return 2


def geometry(corners, mode, conservation, protected):
    """Exact SWITCH_ORDER and zero-seeded closure/prefix value DAGs."""
    n=len(corners);e=len(conservation)
    require(1<=n<=6 and mode in MODES, 'geometry range')
    routed=list(corners)
    if mode in (2,5) and n>2: routed=[x.neg() for x in reversed(routed)]
    chords=[seeded(routed,e)]
    prefix=zero(e)
    for item in routed[:-1]:
        prefix=prefix.add(item);chords.append(prefix)
    classes=[[classify(tuple(a-b for a,b in zip(x.signed,y.signed)),
                       conservation,protected) for y in chords] for x in chords]
    return dict(supported=True, corners=[x.record() for x in routed],
                chords=[x.record() for x in chords], pair_class=classes)


def capture_graph(matrix_element):
    """Graph provenance, not an assertion about custom writer arithmetic."""
    e,ni=matrix_element.get_nexternal_ninitial()
    process=matrix_element.get('processes')[0]
    leaves={}; conflicts=[]
    unsupported=[]
    for lorentz in process.get('model').get('lorentz'):
        if getattr(lorentz,'formfactors',None):
            unsupported.append('unqualified custom form-factor source '+lorentz.name)
    seen=set()
    def walk(w):
        if id(w) in seen:return
        seen.add(id(w))
        if w.get('is_loop'): return
        if w.get('mothers'):
            for m in w.get('mothers'): walk(m)
        else:
            d=w.get_external_helas_call_dict()
            leg=d['number_external'];slot=d['number']
            item=dict(slot=slot,leg=leg,mass=d['mass'],
                      offshell=w.get('offshell'),state=w.get('state'),
                      state_id=d['state_id'],is_part=w.get_with_flow('is_part'),
                      spin=w.get('spin'))
            if slot in leaves and leaves[slot]!=item: conflicts.append(slot)
            leaves[slot]=item
    for d in matrix_element.get('diagrams'):
        for w in d.get('wavefunctions'): walk(w)
        if 'loop_wavefunctions' in d:
            for w in d.get('loop_wavefunctions'):
                for m in w.get('mothers'): walk(m)
    # Actual external leg policy, distinct from internal onshell flags.
    legs={x.get('number'):dict(offshell=x.get('offshell'),state=x.get('state'),
                              pdg=x.get('id')) for x in process.get('legs')}
    return dict(nexternal=e,ninitial=ni,has_born=process.get('has_born'),
                process=process.base_string(),legs=legs,
                leaves=leaves,conflicting_slots=sorted(set(conflicts)),
                unsupported_source_features=unsupported)


def start(matrix_element):
    matrix_element._invariant_provenance = dict(schema=1,
        graph=capture_graph(matrix_element), routes=[])


def register(matrix_element, amp, emitted):
    """Called while temporary LOOPNUM/group ID is the emitted one."""
    data=getattr(matrix_element,'_invariant_provenance',None)
    if data is None:return emitted
    d=amp.get_helas_call_dict(OptimizedOutput=True)
    route=len(data['routes'])+1
    n=len(amp.get('wavefunctions'))-2
    item=dict(route=route,loopnum=d['loop_group_id'],amplitude=amp.get('number'),
              n=n,we=[m.get('number') for m in amp.get('mothers')],
              pairing=list(amp.get('pairing')),rank=d['LoopRank'],
              masses=[d['LoopMass%d'%i] for i in range(1,n+1)],call=emitted)
    data['routes'].append(item)
    # Rewrite only this live call, never a cached call-key closure.
    m=re.fullmatch(r'(CALL %\(proc_prefix\)sLOOP_[0-9]+(?:_[0-9]+)?)(\(.*)\)',emitted)
    if not m:
        item['source_unknown']='unsupported optimized reduction call'
        return emitted
    return m[1]+'_PROVENANCE'+m[2]+',%d,GP_ENV,GP_FRESH)'%route


def kind(block, formal, width):
    expected='TYPE('+('MP_ALOHA' if width==128 else 'ALOHA')+')'+formal
    require(any(s==expected for _,s,_ in block['statements']),
            'unsupported/mixed momentum storage '+formal)


def validate_momentum_writes(block):
    """Admit direct unconditional momentum writes without aliases/foreign calls."""
    output=block['arguments'][-1]
    require(len(set(block['arguments']))==len(block['arguments']),'aliased formal arguments')
    depth=0
    for _,s,_ in block['statements']:
        require(not s.startswith(('CALL','EQUIVALENCE','POINTER','SAVE','COMMON')),
                'custom/aliased momentum routine')
        if s.startswith(('ENDIF','ENDDO','ENDSELECT','ENDWHERE')): depth-=1
        require(depth>=0,'unrecognized producer control flow')
        lhs=re.match(r'([A-Z][A-Z0-9_]*)%P\([^)]*\)=',s)
        if lhs:
            require(lhs[1]==output and depth==0,'conditional/input momentum write')
        if (s.startswith('IF(') and s.endswith('THEN')) or s.startswith(('DO','SELECTCASE','WHERE(')):
            # DOUBLEPRECISION declarations are not DO loops.
            if not s.startswith(('DOUBLEPRECISION','DOUBLECOMPLEX')):depth+=1
    require(depth==0,'unclosed producer control flow')


def prove_wavefunctions(graph, batches, routines, width):
    """Read actual emitted assignments, preserving repeated unsigned leaves."""
    e=graph['nexternal'];out={};references={};proof=[];unknown={}
    require(3<=e<=6 and not graph['conflicting_slots'], 'external graph range/conflict')
    require(not graph.get('unsupported_source_features'), 'custom producer source family')
    for filename,text in batches:
        for line,s,_ in statements(text):
            require(not re.match(r'.*%P\([^)]*\)=',s),'custom batch momentum write')
            c=call(s)
            if not c or not c[1] or not re.fullmatch(r'W\([1-9][0-9]*\)',c[1][-1]):continue
            name,args=c;dest=wf_number(args[-1]);node=None
            try:
                b=routines[name]
                require(len(b['arguments'])==len(args),'producer arity')
                validate_momentum_writes(b)
                for formal,actual in zip(b['arguments'],args):
                    if re.fullmatch(r'W\([1-9][0-9]*\)',actual):kind(b,formal,width)
                writes=[s for _,s,_ in b['statements'] if re.match(re.escape(b['arguments'][-1])+r'%P\(',s)]
                if re.fullmatch(r'(?:MP_)?[IOVST]XXXXX',name):
                    leg,sign,mass=external_rule(b,args)
                    leaf=graph['leaves'].get(str(dest),graph['leaves'].get(dest))
                    require(leaf is not None and leaf['leg']==leg,'graph/external slot join')
                    require(1<=leg<=e and len(writes)==4,'external write extent')
                    real=('REAL*16',) if width==128 else ('REAL*8','DOUBLEPRECISION')
                    require(any((s==t+b['arguments'][0]+'(0:3)' or s.startswith(t+b['arguments'][0]+'(0:3),')) for _,s,_ in b['statements'] for t in real),'external input kind')
                    expected=leaf['mass'].upper()
                    # MP batches use mp_coupl_same_name.inc; this is a symbol
                    # identity join, never a claim about numerical mass equality.
                    allowed={expected} if width==64 else {expected,'MP__'+expected}
                    require(mass is None or mass in allowed,'external mass expression differs')
                    node=Expr(tuple(sign if i==leg else 0 for i in range(1,e+1)),
                              tuple(1 if i==leg else 0 for i in range(1,e+1)),0)
                    references.setdefault(leg,[]).append((dest,sign))
                else:
                    require(len(writes)==1,'internal momentum write extent')
                    terms=internal_rule(b,args)
                    children=[out[j] if sign==1 else out[j].neg() for sign,j in terms]
                    require(all(x is not None for x in children),'unknown parent')
                    node=children[0]
                    for child in children[1:]:node=node.add(child)
                require(dest not in out or out[dest]==node,'overwritten W lineage')
                proof.append(dict(caller=filename,line=line,call=s,slot=dest,width=width,
                                  expression=node.record()))
            except (ValueError,KeyError,AttributeError,TypeError) as exc:
                unknown[dest]=str(exc);node=None
            # Once a slot is ambiguous it stays unknown even if later rewritten.
            out[dest]=None if dest in unknown else node
    require(set(references)==set(range(1,e+1)), 'missing physical external source: '+str(unknown))
    slots=[]
    for leg in range(1,e+1):
        choices=[slot for slot,sgn in references[leg] if out.get(slot) is not None and
                 out[slot].ops==0 and out[slot].mult[leg-1]==1]
        require(bool(choices),'no live external reference')
        slots.append(min(choices))
    return out,slots,proof,unknown


def derive(data, dp_batches, qp_batches, routines, repair, pmass):
    """No global sources or implicit per-event IDs. Caller owns all text inputs."""
    graph=data['graph'];e=graph['nexternal'];routes=[];proof={};per_kind={}
    conservation=[-1 if i<=graph['ninitial'] else 1 for i in range(1,e+1)]
    protected=[]
    for leg in range(1,e+1):
        leafs=[x for x in graph['leaves'].values() if x['leg']==leg]
        policy=graph['legs'].get(str(leg),graph['legs'].get(leg))
        masses={x['mass'].upper() for x in leafs}
        protected.append(bool(policy is not None and policy['offshell'] is False and
            leafs and all(x['offshell'] is False for x in leafs) and
            repair.get(leg) is False and len(masses)==1 and
            pmass.get(leg) in masses|{'ABS('+x+')' for x in masses}))
    for width,batches in ((64,dp_batches),(128,qp_batches)):
        try:per_kind[width]=prove_wavefunctions(graph,batches,routines,width)
        except (ValueError,KeyError,AttributeError,TypeError) as exc:
            per_kind[width]=None;proof[str(width)]={'unknown':str(exc)}
    for raw in data['routes']:
        r=dict(raw);r['modes']={}
        for mode in MODES:
            width=64 if mode<4 else 128;entry=per_kind[width]
            try:
                require(not r.get('source_unknown'),r.get('source_unknown',''))
                require(entry is not None,'producer proof unavailable')
                nodes,slots,receipts,unknown=entry
                require(1<=r['n']<=6 and len(r['pairing'])==r['n'] and
                        all(type(x) is int and x>0 for x in r['pairing']) and
                        sum(r['pairing'])==len(r['we']), 'route shape')
                require(all(nodes.get(i) is not None for i in r['we']),'unknown route mother')
                corners=[];offset=0
                for count in r['pairing']:
                    corners.append(seeded([nodes[i].neg() for i in r['we'][offset:offset+count]],e));offset+=count
                item=geometry(corners,mode,conservation,protected)
                require(all(c['ops']<=1000000 and max(c['mult'],default=0)<=1000000 for c in item['chords']),'integer bound metadata overflow')
                item['external_slots']=slots;r['modes'][str(mode)]=item
                proof[str(width)]=dict(producers=receipts,unsupported=unknown)
            except (ValueError,KeyError,AttributeError,TypeError) as exc:
                r['modes'][str(mode)]=dict(supported=False,reason=str(exc))
        routes.append(r)
    return dict(schema=1,graph=graph,conservation=conservation,
                protected=protected,routes=routes,source_proof=proof)


def emit_tables(d):
    routes=d['routes'];nr=len(routes);e=d['graph']['nexternal']
    require([r['route'] for r in routes]==list(range(1,nr+1)) and nr>0,'route IDs')
    nmax=max(r['n'] for r in routes);nw=max(len(r['we']) for r in routes)
    out=['! Immutable per-subprocess geometry, emitted after source proof.',
         'integer,parameter::route_count=%d,external_count=%d'%(nr,e),
         '! CALL graph is diagnostic only; expression-function closure not qualified.',
         'logical,parameter,public::source_environment_supported=.false.']
    def array(name,shape,values,logical=False):
        values=list(values)
        product=1
        for v in shape:product*=v
        require(product==len(values),'table orientation/extent')
        vals=[('.true.' if x else '.false.') if logical else str(x) for x in values]
        out.append(('logical' if logical else 'integer')+',parameter::'+name+'('+','.join(map(str,shape))+')=reshape([ &')
        for i in range(0,len(vals),10):out.append('  '+','.join(vals[i:i+10])+(', &' if i+10<len(vals) else ' &'))
        out.append('],['+','.join(map(str,shape))+'])')
    array('route_n',(nr,),(r['n'] for r in routes))
    array('route_nw',(nr,),(len(r['we']) for r in routes))
    array('route_we',(nw,nr),(r['we'][i] if i<len(r['we']) else 0 for r in routes for i in range(nw)))
    array('route_pairing',(nmax,nr),(r['pairing'][i] if i<r['n'] else 0 for r in routes for i in range(nmax)))
    def item(r,m):return r['modes'][str(m)]
    array('route_supported',(nr,4),(item(r,m)['supported'] for m in MODES for r in routes),True)
    array('route_external',(e,nr,4),(item(r,m)['external_slots'][i] if item(r,m)['supported'] else 0 for m in MODES for r in routes for i in range(e)))
    array('route_ops',(nmax,nr,4),(item(r,m)['chords'][i]['ops'] if item(r,m)['supported'] and i<r['n'] else 0 for m in MODES for r in routes for i in range(nmax)))
    array('route_mult',(e,nmax,nr,4),(item(r,m)['chords'][i]['mult'][j] if item(r,m)['supported'] and i<r['n'] else 0 for m in MODES for r in routes for i in range(nmax) for j in range(e)))
    array('route_class',(nmax,nmax,nr,4),(item(r,m)['pair_class'][i][j] if item(r,m)['supported'] and i<r['n'] and j<r['n'] else 0 for m in MODES for r in routes for j in range(nmax) for i in range(nmax)))
    return '\n'.join(out)+'\n'


def routing_contract(ct_source,tir_source,tirust_source,prefix):
    """Validate the implemented zero seeds and left-associated routing.

    Strict complete snippets plus exclusive assignments are required; a
    modified helper cannot inherit a descriptor from graph metadata alone.
    """
    prefix=prefix.upper();ct=blocks(ct_source);tir=blocks(tir_source);ti=blocks(tirust_source)
    corner=('DOI=0,3','TEMP=1','DOJ=1,NLOOPLINE','PL(I,J)=0.D0',
            'IF(DOING_QP)THEN','MP_PL(I,J)=0.0E+0_16','ENDIF',
            'DOK=TEMP,(TEMP+PAIRING(J)-1)','PL(I,J)=PL(I,J)-W(WE(K))%P(I)',
            'IF(DOING_QP)THEN','MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)',
            'ENDIF','ENDDO','TEMP=TEMP+PAIRING(J)','ENDDO','ENDDO')
    names=[n for n in ct if re.fullmatch(re.escape(prefix)+r'LOOP_[1-9][0-9]*(?:_[1-9][0-9]*)?',n)]
    require(bool(names),'no legacy corner helper')
    for name in names:
        seq=[s for _,s,_ in ct[name]['statements']]
        require(any(tuple(seq[i:i+len(corner)])==corner for i in range(len(seq))),'unknown corner arithmetic: '+name)
        writes=[s for s in seq if re.match(r'(?:MP_)?PL\([^)]*\)=',s)]
        require(writes==[corner[i] for i in (3,5,8,10)],'extra corner assignment')
    for qp in (False,True):
        name=prefix+('MP_' if qp else '')+'SWITCH_ORDER';seq=[s for _,s,_ in tir[name]['statements']]
        require(seq.count('IF(CTMODE.NE.2.AND.CTMODE.NE.5)THEN')==1 and
                seq.count('IF(NLOOPLINE.LE.2)THEN')==1,'unknown reversal guard')
        require([s for s in seq if re.match(r'(?:NEW_)?PL\([^)]*\)=',s)]==
                ['NEW_PL(J,I)=-PL(J,NLOOPLINE+1-I)','PL(J,I)=NEW_PL(J,I)'],'unknown reversal arithmetic')
        name=prefix+('MP_' if qp else '')+'TIRUST_TENSOR_DIAGNOSTIC';seq=[s for _,s,_ in ti[name]['statements']]
        kind='KI_QTIR' if qp else 'KI_TIR'
        pieces=[('P_TMP=0','ABSP=0','DOI=0,3','DOJ=1,NLOOPLINE',
                 'P_TMP(I,0)=P_TMP(I,0)+PL(I,J)','ABSP(I)=ABSP(I)+ABS(PL(I,J))','ENDDO','ENDDO'),
                ('DOI=0,3','PREFIX_SUM=REAL(0,KIND='+kind+')','DOJ=1,NLOOPLINE-1',
                 'PREFIX_SUM=PREFIX_SUM+PL(I,J)','P_TMP(I,J)=PREFIX_SUM','ENDDO','ENDDO'),
                ('DOJ=1,NLOOPLINE','DOI=0,3','SHIFTS(I,J)=P_TMP(I,J-1)',
                 'P_S_MAT(J,I)=SHIFTS(I,J)','ENDDO','ENDDO')]
        for piece in pieces:
            require(any(tuple(seq[i:i+len(piece)])==piece for i in range(len(seq))),'unknown closure/prefix arithmetic')
        require([s for s in seq if re.match(r'P_TMP(?:\([^)]*\))?=',s)]==
                ['P_TMP=0','P_TMP(I,0)=P_TMP(I,0)+PL(I,J)','P_TMP(I,J)=PREFIX_SUM'],'extra chord assignment')
    return dict(corner_helpers=names,modes=list(MODES),first_shift='actual zero-seeded closure, not replaced by zero',
                scope='TIRust routing only; other adapter arithmetic requires a separate join')
