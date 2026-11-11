"""Ordered producer prefix proof; no token insertion or scientific activation."""
import copy,re
from madgraph.loop.invariant_environment_source import Library,Unknown,need,close,split


def momentum_window(raw,batches,producer_receipts):
    """Use the existing same-kind DAG receipts, never a routine-name allowlist.

    All proved W producers are included (a conservative superset of each route's
    mothers). The source prefix ends after the last such CALL. The suffix must
    have complete CALL/FUNCTION references and no possible momentum/whole-object
    write; I/O there remains recorded, not labelled environment-pure.
    """
    answer={'schema':'source-momentum-environment-window-v1','complete':False,
            'scientific_admission':False,'required_runtime_contract':[
                'fresh token before all included batches; same control state at use',
                'actual same-kind producer path completed, no skip/reuse/widening',
                'bound source/ABI/compiler contract; no external hidden alias writes'],
            'reason':'no proof'}
    try:
        need(producer_receipts,'no same-kind DAG producer receipts')
        lib=Library(raw);need(not lib.file_errors,'source family indexing incomplete')
        ordered=[]
        for filename,text in batches:
            units=[k for k,u in lib.units.items() if u.file.endswith('/'+filename) or u.file==filename]
            need(len(units)==1,'batch unit ambiguity');ordered.append(units[0])
        locations={}
        for i,key in enumerate(ordered):
            for j,row in enumerate(lib.units[key].body):locations[(row[0].split('/')[-1],row[1],row[2])]=(i,j,key)
        points=[]
        for p in producer_receipts:
            loc=(p['caller'],p['line'],p['call'])
            need(loc in locations,'DAG receipt not joined to actual batch statement');points.append(locations[loc])
        bi,ri,key=max(points)
        prefix=copy.deepcopy(lib);prefix.scope_cache={}
        prefix.units[key].body=prefix.units[key].body[:ri+1]
        before=prefix.closure(ordered[:bi+1])
        suffix=copy.deepcopy(lib);suffix.scope_cache={}
        # Keep all declarations/imports for reference classification, but remove
        # earlier executable rows. The split is analysis-only, not emitted code.
        _,_,_,_,executable=lib.scope(key);late={r for r in executable if r in lib.units[key].body[ri+1:]}
        declared={r for r in lib.units[key].body if r not in executable}
        suffix.units[key].body=[r for r in lib.units[key].body if r in late or r in declared]
        after=suffix.closure(ordered[bi:])
        writes=[]
        for proc in after['procedures']:
            symbols,_,_,_,rows=suffix.scope(proc)
            for file,line,s in rows:
                # Direct or scalar-IF assignment to any momentum field / entire
                # derived object is conservative rejection. Component W-only
                # writes remain ordinary numerator arithmetic.
                tail=s
                if tail.startswith('IF('):tail=tail[close(tail,2)+1:]
                if '=' not in tail or tail.startswith(('CALL','DO','ASSOCIATE')):continue
                lhs=tail.split('=',1)[0]
                base=re.match(r'([A-Z][A-Z0-9_]*)',lhs)
                if '%P' in lhs or (base and symbols.get(base[1]) and symbols[base[1]].kind.startswith('TYPE(') and '%' not in lhs):
                    writes.append({'procedure':proc,'file':file,'line':line,'statement':s})
        if after['reference_complete']:
            writes.extend(suffix_alias_writes(suffix,after['procedures']))
        need(before['complete'],'prefix source environment unresolved')
        need(after['reference_complete'],'later CALL/FUNCTION effects unresolved')
        need(not writes,'possible later momentum/object overwrite')
        answer.update(complete=True,reason=None)
    except (Unknown,ValueError,KeyError,TypeError) as exc:answer['reason']=str(exc)
    if 'ri' in locals():answer['last_producer']={'batch':key,'file':lib.units[key].file,'line':lib.units[key].body[ri][1],'statement':lib.units[key].body[ri][2]}
    if 'before' in locals():answer['prefix']=before
    if 'after' in locals():answer['suffix']=after
    if 'writes' in locals():answer['later_writes']=writes
    return answer


def suffix_alias_writes(lib,procedure_names):
    """Conservative formal-write summary, including calls inside expressions.

    Fields are ALL/P/W/etc, with ALL absorbing narrower fields. Any mutation
    through a numeric dummy receiving a momentum actual therefore remains a P
    mutation; forwarding a whole object cannot erase that effect.
    """
    names=set(procedure_names);effects={k:{a:set() for a in lib.units[k].args} for k in names}
    direct=[];scopes={k:lib.scope(k)[0] for k in names}
    # COMMON is explicit storage association, not a hidden ABI exemption. All
    # views of a derived-object block must have the same exact declared layout;
    # a numeric reinterpretation of W storage remains unknown.
    common={}
    for key in names:
        for _,_,row in lib.expanded(lib.units[key].body):
            if not row.startswith('COMMON'):continue
            views=re.findall(r'/([A-Z][A-Z0-9_]*)/([^/]+)',row[6:])
            need(views and ''.join('/'+n+'/'+v for n,v in views)==row[6:],'unsupported COMMON view')
            for block,entries in views:
                layout=[]
                for entry in split(entries):
                    m=re.match(r'([A-Z][A-Z0-9_]*)',entry);need(m is not None and m[1] in scopes[key],'unknown COMMON symbol')
                    a=scopes[key][m[1]];layout.append((a.kind,a.shape))
                common.setdefault(block,[]).append(tuple(layout))
    for block,views in common.items():
        if any(kind.startswith('TYPE(') and kind!='TYPE(IEEE_ROUND_TYPE)' for view in views for kind,shape in view):
            need(len(set(views))==1,'ambiguous derived-object COMMON storage '+block)
    def ref(text):
        # Only a simple definable scalar/array/component actual can be mapped.
        m=re.match(r'([A-Z][A-Z0-9_]*)',text)
        if not m:return None
        base=m[1];pos=len(base);field='ALL'
        if pos<len(text) and text[pos]=='(':pos=close(text,pos)+1
        if pos<len(text) and text[pos]=='%':
            m=re.match(r'%([A-Z][A-Z0-9_]*)',text[pos:]);need(m is not None,'component actual syntax');field=m[1];pos+=len(m[0])
            if pos<len(text) and text[pos]=='(':pos=close(text,pos)+1
        return (base,field) if pos==len(text) else None
    def written_path(actual,fields):
        a=ref(actual)
        if not a:return None
        base,field=a
        return base,({field} if field!='ALL' else fields)
    def add(k,base,fields):
        if base not in effects[k]:return False
        old=set(effects[k][base]);effects[k][base].update(fields)
        if 'ALL' in effects[k][base]:effects[k][base]={'ALL'}
        return old!=effects[k][base]
    for key in names:
        _,_,_,_,rows=lib.scope(key)
        for file,line,s in rows:
            tail=s
            if tail.startswith('IF('):tail=tail[close(tail,2)+1:]
            loop=re.fullmatch(r'DO(?:[0-9]+)?([A-Z][A-Z0-9_]*)=.*',tail)
            if loop:add(key,loop[1],{'ALL'});continue
            if '=' not in tail or tail.startswith(('CALL','DO','ASSOCIATE')):continue
            lhs=tail.split('=',1)[0];a=ref(lhs)
            if a:add(key,a[0],{a[1]})
    # Finite monotone fixed point over the bound procedure/formal/field set.
    limit=1+sum(len(u.args) for k,u in lib.units.items() if k in names)*8
    for iteration in range(limit):
        changed=False
        for key in sorted(names):
            for target,args in lib.invocation_cache.get(key,[]):
                if target.startswith('@'):
                    out={'@IEEE_GET_FLAG':1,'@IEEE_GET_HALTING_MODE':1,'@IEEE_GET_ROUNDING_MODE':0}.get(target)
                    need(out is not None,'unknown intrinsic subroutine write contract')
                    writes=[(out,{'ALL'})]
                else:
                    need(target in effects,'missing callee write summary')
                    writes=[(i,effects[target][a]) for i,a in enumerate(lib.units[target].args)]
                for i,fields in writes:
                    if not fields:continue
                    actual=args[i];mapped=written_path(actual,fields)
                    if mapped:changed=add(key,*mapped) or changed
                    else:
                        # An unclassifiable write-capable actual must not hide a
                        # formal/derived-object alias inside a conversion/expression.
                        touched=[n for n in set(effects[key])|{n for n,s in scopes[key].items() if s.kind.startswith('TYPE(') and s.kind!='TYPE(IEEE_ROUND_TYPE)'} if re.search(r'(?<![A-Z0-9_])'+re.escape(n)+r'(?![A-Z0-9_])',actual)]
                        need(not touched,'unproved write-capable expression actual '+actual)
        if not changed:break
    else:raise Unknown('write-effect fixed point exceeded bound')
    for key in sorted(names):
        for target,args in lib.invocation_cache.get(key,[]):
            if target.startswith('@'):
                out={'@IEEE_GET_FLAG':1,'@IEEE_GET_HALTING_MODE':1,'@IEEE_GET_ROUNDING_MODE':0}[target];writes=[(out,{'ALL'})]
            else:writes=[(i,effects[target][a]) for i,a in enumerate(lib.units[target].args)]
            for i,fields in writes:
                if not fields:continue
                mapped=written_path(args[i],fields)
                if not mapped:continue
                base,path=mapped;symbol=scopes[key].get(base)
                # Any explicit momentum component is protected, including numeric
                # actuals and formal-to-formal forwarding. For a whole derived
                # actual, ALL or P effects are writes to momentum storage.
                if 'P' in path or symbol and symbol.kind.startswith('TYPE(') and symbol.kind!='TYPE(IEEE_ROUND_TYPE)' and 'ALL' in path:
                    direct.append({'procedure':key,'callee':target,'argument':i+1,'actual':args[i],'written_fields':sorted(path),'reason':'write-capable actual/dummy alias'})
    return direct
