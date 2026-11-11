"""Finalize generic immutable tables after model/ALOHA generation, no compiler."""
import hashlib
import json
from pathlib import Path
import re
from madgraph.loop.invariant_provenance import derive,emit_tables,routing_contract
from madgraph.loop.invariant_source import blocks,call,statements,require

ENV_WRITE=re.compile(r'IEEE_SET_(?:ROUNDING|UNDERFLOW|HALTING)_MODE|FESET(?:ROUND|ENV)|FEHOLD(?:EXCEPT|ENV)|FEUPDATEENV|F[DE]?(?:ENABLE|DISABLE)EXCEPT|LDMXCSR|FLDCW',re.I)


def source_inputs(directory,prefix):
    p=Path(directory).resolve();root=next((a for a in p.parents if (a/'Source/DHELAS').is_dir()),None)
    require(root is not None,'no generated DHELAS source root')
    raw={}
    for folder in (root/'Source/DHELAS',root/'Source/MODEL'):
        require(not any(q.suffix.lower() in ('.c','.cc','.cpp','.f90','.f95') for q in folder.iterdir()),
                'unqualified custom/foreign source family')
    def read(path):
        path=path.resolve();b=path.read_bytes();raw[str(path.relative_to(root))]=b
        return b.decode('utf8')
    routines={}
    for path in sorted((root/'Source/DHELAS').glob('*.f')):
        for name,b in blocks(read(path)).items():
            require(name not in routines,'ambiguous DHELAS routine '+name);routines[name]=b
    def batches(qp):
        caller=read(p/('mp_compute_loop_coefs.f' if qp else 'loop_matrix.f'))
        out=[];seen=set()
        for _,s,_ in statements(caller):
            c=call(s)
            if not c:continue
            m=re.fullmatch(re.escape(prefix.upper())+('MP_' if qp else '')+
                r'((?:HELAS_CALLS_(?:AMPB|UVCT)|COEF_CONSTRUCTION)_[0-9]+)',c[0])
            if not m:continue
            n=('mp_' if qp else '')+m[1].lower()+'.f'
            require(n not in seen,'repeated producer batch');seen.add(n)
            out.append((n,read(p/n)))
        require(bool(out),'no producer batches')
        return out
    dp=batches(False);qp=batches(True)
    routing=routing_contract(read(p/'CT_interface.f'),read(p/'TIR_interface.f'),read(p/'TIRUST_interface.f'),prefix)
    repair={};pmass={}
    for _,s,_ in statements(read(p/'loop_matrix.f')):
        m=re.fullmatch(r'KEEP_OFFSHELL_MASS\(([0-9]+)\)=\.(TRUE|FALSE)\.',s)
        if m:
            value=m[2]=='TRUE';i=int(m[1]);require(i not in repair or repair[i]==value,'repair conflict');repair[i]=value
    for _,s,_ in statements(read(p/'pmass.inc')):
        m=re.fullmatch(r'PMASS\(([0-9]+)\)=(.*)',s)
        require(m is not None,'pmass form');pmass[int(m[1])]=m[2]
    # No setters in generated producer/model/value-path sources. Unknown custom
    # source still requires a separate source-contract review before promotion.
    for folder in (p,root/'Source/MODEL'):
        for path in sorted(folder.glob('*.f')):read(path)
    for name,b in raw.items():
        code='\n'.join(s for _,s,_ in statements(b.decode('utf8')))
        require(not ENV_WRITE.search(code),'environment setter: '+name)
    return dp,qp,routines,repair,pmass,raw,routing



def environment_closure(raw,batches,model_present):
    """Generated CALL-closure diagnostic; scientific admission stays false.

    This proves only the retained generated Fortran source contract. Compiler
    flags, actual control words and native parity still require qualification.
    """
    library={}
    try:
        require(model_present,'model source missing')
        for filename,b in raw.items():
            for name,body in blocks(b.decode('utf8')).items():
                require(name not in library,'ambiguous generated callee '+name)
                library[name]=body
        queue=[]
        for filename,text in batches:
            queue.extend(blocks(text))
        seen=set();safe={'IEEE_GET_FLAG'}
        while queue:
            name=queue.pop()
            if name in seen or name in safe:continue
            require(name in library,'unresolved producer callee '+name)
            seen.add(name)
            for _,s,_ in library[name]['statements']:
                require(not s.startswith(('EXTERNAL','POINTER','ENTRY')),'unqualified indirect/custom callee')
                c=call(s)
                if c:queue.append(c[0])
        return dict(complete=False,call_graph_complete=True,
                    function_reference_closure_qualified=False,
                    subroutines=sorted(seen),intrinsics=sorted(safe),
                    reason='expression function-reference closure is unqualified',
                    scope='generated direct CALL diagnostic only; scientific admission disabled')
    except (ValueError,KeyError,UnicodeError) as exc:
        return dict(complete=False,call_graph_complete=False,
                    function_reference_closure_qualified=False,reason=str(exc))

def finalize_one(exporter,directory,me):
    p=Path(directory);data=me._invariant_provenance;prefix=me.rep_dict['proc_prefix']
    try:
        dp,qp,routines,repair,pmass,raw,routing=source_inputs(directory,prefix)
        result=derive(data,dp,qp,routines,repair,pmass)
        result['routing_contract']=routing
        result['environment_contract']=environment_closure(raw,dp+qp,
            any(n.startswith('Source/MODEL/') for n in raw))
        result['source_bindings']={n:dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)) for n,b in sorted(raw.items())}
    except (ValueError,OSError,KeyError,TypeError,UnicodeError) as exc:
        result=dict(schema=1,graph=data['graph'],routes=[dict(r,modes={str(m):dict(supported=False,reason=str(exc)) for m in (1,2,4,5)}) for r in data['routes']],
                    source_proof={'unknown':str(exc)},source_bindings={})
    (p/'invariant_provenance.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    (p/'invariant_bounds_data.inc').write_text(emit_tables(result))
    t=Path(exporter.template_dir)
    module=(t/'invariant_bounds.f90').read_text().replace('%(proc_prefix)s',prefix)
    # Namespace the C token symbol with this matrix-element prefix.
    module=module.replace("name='ml_private_kinematic_environment_token'", "name='"+prefix+"kinematic_environment_token'")
    (p/'invariant_bounds.f90').write_text(module)
    c=(t/'invariant_environment.c').read_text().replace('ml_private_kinematic_environment_token',prefix+'kinematic_environment_token')
    (p/'invariant_environment.c').write_text(c)
    return result


def finalize_pending(exporter):
    pending=getattr(exporter,'_invariant_pending',{})
    exporter._invariant_pending={}
    for path,me in sorted(pending.items()):finalize_one(exporter,path,me)
