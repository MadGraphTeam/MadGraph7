#!/usr/bin/env python3
"""Extract actual AVH classifier prefixes and add private checked overloads."""
import hashlib
import json
from pathlib import Path
import re
import sys

if sys.flags.optimize:raise RuntimeError('Source checks require assertions')
ROOT=Path.cwd().resolve()
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'vendor/CutTools/src/avh/avh_olo.f90'
NAMES=('a0_c','a0_r','b11rc','b11rr','c0rc','c0rr','d0rc','d0rr')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def logical_lines(text):
    """Keep exact physical bytes and source offsets; join free-form continuations."""
    lines=text.splitlines(keepends=True);i=0
    while i<len(lines):
        start=i;chunk=lines[i];i+=1
        if chunk.lstrip().startswith('!') or not chunk.strip():
            yield start,i,chunk,None;continue
        joined=chunk.strip()
        while joined.rstrip().endswith('&'):
            joined=joined.rstrip()[:-1]+' '+lines[i].strip().lstrip('&');chunk+=lines[i];i+=1
        yield start,i,chunk,joined


def extract(source,precision,name):
    offset=source.index('module avh_olo_forCutTools_'+precision+'\n')
    match=re.search(r'^  subroutine '+name+r'\s*\(.*?^  end subroutine\s*$',source[offset:],re.M|re.S)
    assert match
    start=offset+match.start();full=match.group(0)
    return full,source[:start].count('\n')+1


def generate_routine(source,precision,name,number):
    original,startline=extract(source,precision,name)
    signature=re.match(r'  subroutine \w+\s*\(([^\n]+)\)',original)[1]
    arguments=[s.strip() for s in signature.split(',')]
    outputs=arguments[:4] if name.startswith('b11') else arguments[:1]
    args=arguments[len(outputs):]
    topology={'a':1,'b':2,'c':3,'d':4}[name[0]]
    n=topology;np={1:0,2:1,3:3,4:6}[n]
    masses=['mm'] if n==1 else ['m'+str(i) for i in range(1,n+1)]
    invariants=[] if n==1 else ['pp'] if n==2 else ['p1','p2','p3'] if n==3 else ['p1','p2','p3','p4','p12','p23']
    complex_mass=name in ('a0_c','b11rc','c0rc','d0rc')
    executable=original.index('  if (initz) call init')
    dispatch=original.index('  call tadp(') if n==1 else original.index('  call bub11(') if n==2 else original.index('  if     (icase.eq.'+str(n)+') then')
    prefix=original[executable:dispatch]
    declaration=original[original.index('\n')+1:executable]
    declaration=re.sub(r'  complex\(kindr2\) &[^\n]*\n\s*,intent\(out\) ::[^\n]+\n','',declaration)
    assert 'intent(out)' not in declaration
    # Prefixes require only the exact permutation constants, not analytic kernels.
    declaration=re.sub(r'^  use [^\n]+\n','',declaration,flags=re.M)
    uses='  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite\n'
    if n>=3:uses+='  use avh_olo_forCutTools_'+precision+('_tri' if n==3 else '_box')+', only: base, permtable, casetable\n'
    declaration=re.sub(r'  character\(\d+\+99\) ,parameter :: warning=&\n\s*[^\n]+\n','',declaration)
    declaration+='  type(cms_scalar_report), intent(out) :: report\n'
    declaration+='  real(kindr2) :: cms_scale\n  complex(kindr2) :: cms_original_p(6),cms_original_m(4)\n'
    declaration+='  integer :: cms_pi(6),cms_mi(4)\n'
    initialization=f'  call cms_reset(report,{8 if precision=="dp" else 16},{n},{number})\n'
    initialization+='  if (initz) then\n    report%status=CMS_UNINITIALIZED\n    return\n  endif\n'
    initialization+='  if (.not.nonzerothrs) then\n    report%status=CMS_DEFAULT_UNSUPPORTED\n    return\n  endif\n'
    initialization+='  cms_original_p=0;cms_original_m=0\n  cms_pi=(/1,2,3,4,5,6/);cms_mi=(/1,2,3,4/)\n'
    for i,arg in enumerate(invariants,1):initialization+=f'  cms_original_p({i})={arg}\n'
    for i,arg in enumerate(masses,1):initialization+=f'  cms_original_m({i})={arg}\n'
    initialization+='  if (.not.all(ieee_is_finite(real(cms_original_p,kindr2))).or. &\n'
    initialization+='      .not.all(ieee_is_finite(real(cms_original_m,kindr2))).or. &\n'
    initialization+='      .not.all(ieee_is_finite(aimag(cms_original_m))).or. &\n'
    initialization+='      .not.ieee_is_finite(onshellthrs).or..not.ieee_is_finite(muscale)) then\n'
    initialization+='    report%status=CMS_BAD_INPUT\n    return\n  endif\n'
    initialization+='  if (onshellthrs.lt.RZRO) then\n    report%status=CMS_BAD_INPUT\n    return\n  endif\n'
    body=[];events=[]
    prefix_line=startline+original[:executable].count('\n')
    for first,last,physical,statement in logical_lines(prefix):
        if statement is None:body.append(physical);continue
        compact=re.sub(r'\s+','',statement).lower()
        if compact=='if(initz)callinit':body.append(initialization);continue
        if 'write(' in compact:continue  # diagnostics only; no observer I/O
        if compact=='rslt(0)=0;rslt(1)=0;rslt(2)=0':
            body.append('    report%status=CMS_ZERO_TOPOLOGY_UNSUPPORTED\n');continue
        if compact=='if(nonzerothrs)then':
            if not events:
                metrics={1:'am,mulocal2',2:'app,am1,am2,mulocal2',3:'ap,am,mulocal2',4:'ap,am,small,mulocal2'}[n]
                scale='max(am,mulocal2)' if n==1 else 'max(app,am1,am2,mulocal2)' if n==2 else 'smax'
                body.append('  cms_scale='+scale+'\n')
            else:
                metrics='s1r2,s2r3,s3r3' if n==3 else 's1r2,s2r2,s2r3,s3r4,s4r4'
            body.append('  if (.not.all(ieee_is_finite((/'+metrics+'/)))) then\n')
            body.append('    report%status=CMS_BAD_INPUT\n    return\n  endif\n')
        erase=re.fullmatch(r'if\(([^)]+(?:\([^)]*\))?)\.lt\.hh\)(.+)=0',compact)
        # Native erasures have scalar or indexed metrics; require lhs identity.
        if '.lt.hh)' in compact and re.search(r'=0$',compact):
            match=re.fullmatch(r'if\((.+)\.lt\.hh\)(.+)=0',compact)
            assert match and match[1]==match[2],statement
            metric=match[1]
            pindex=mindex='0';kind='CMS_INPUT';norm='CMS_ABS_REAL'
            if metric=='app':pindex='1';original_value='cms_original_p(1)';classified='ss'
            elif metric.startswith('ap('):
                pindex='ii';original_value='cms_original_p(ii)';classified='cmplx(pp(ii),kind=kindr2)'
            elif metric=='am':
                kind='CMS_MASS';mindex='1';original_value='cms_original_m(1)';classified='cmplx(mm,kind=kindr2)'
                norm='CMS_ABS_COMPLEX' if complex_mass else 'CMS_ABS_REAL'
            elif metric in ('am1','am2'):
                kind='CMS_MASS';mindex=metric[-1];original_value=f'cms_original_m({mindex})';classified='r'+mindex
                norm='CMS_L1_COMPLEX' if complex_mass else 'CMS_ABS_REAL'
            elif metric=='am(ii)':
                kind='CMS_MASS';mindex='ii';original_value='cms_original_m(ii)';classified='cmplx(mm(ii),kind=kindr2)'
                norm='CMS_L1_COMPLEX' if complex_mass else 'CMS_ABS_REAL'
            else:
                delta=re.fullmatch(r's([1-6])r([1-4])',metric);assert delta,metric
                i,j=delta.groups();kind='CMS_DIFFERENCE';pindex=f'cms_pi({i})';mindex=f'cms_mi({j})'
                norm='CMS_ABS_COMPLEX' if n==3 else 'CMS_L1_COMPLEX'
                original_value=f'cms_original_p({pindex})-cms_original_m({mindex})'
                classified=f'ss({i})-rr({j})'
            line=prefix_line+first
            original_invariant=f'cms_original_p({pindex})' if pindex!='0' else 'cmplx(0,0,kind=kindr2)'
            original_mass=f'cms_original_m({mindex})' if mindex!='0' else 'cmplx(0,0,kind=kindr2)'
            events.append({'line':line,'metric':metric,'kind':kind,'norm':norm,'p_index':pindex,'mass_index':mindex,
                           'original_value':original_value,'classified_value':classified,'original_statement':statement,
                           'original_invariant':original_invariant,'original_mass':original_mass})
            body.append(f'    if ({metric}.lt.hh) then\n')
            body.append(f'      call cms_record(report,{line},{kind},{norm},{pindex},{mindex}, &\n')
            body.append(f"        '{metric}',{metric},{original_value}, &\n        {classified},hh,cms_scale, &\n")
            body.append(f'        {original_invariant},{original_mass})\n')
            body.append(f'      {metric}=0\n    endif\n')
            continue
        body.append(physical)
        # Source indices follow the exact same permutation expression before
        # icase is replaced by casetable(icase).
        for target,index,expression in re.findall(r'(ss|rr)\((\d+)\)=(?:pp|mm)\((.+?)\)\s*;',statement):
            body.append(f'  {"cms_pi" if target=="ss" else "cms_mi"}({index})={expression}\n')
    assert len(events)=={1:1,2:3,3:5,4:7}[n],(name,len(events))
    ending='  report%p_sources=cms_pi\n  report%mass_sources=cms_mi\n'
    if n>=3:ending+='  report%classification_case=icase\n'
    if n==4:ending+='  report%soft_permutation=jj\n'
    ending+='  end subroutine cms_preflight_'+name+'\n'
    preflight='  subroutine cms_preflight_'+name+'(report,'+','.join(args)+')\n'+uses+declaration+''.join(body)+ending
    # No copy of the original analytic body: accepted calls delegate directly
    # to its untouched original entrypoint and original actual arguments.
    checked='  subroutine cms_checked_'+name+'('+','.join(arguments)+',report)\n'
    checked+='  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite\n'
    outdecl='  complex(kindr2), intent(inout) :: '+','.join(x+'(0:2)' for x in outputs)+'\n'
    indecl=''
    if invariants:indecl+='  real(kindr2), intent(in) :: '+','.join(invariants)+'\n'
    indecl+=('  complex(kindr2)' if complex_mass else '  real(kindr2)')+', intent(in) :: '+','.join(masses)+'\n'
    temps=['cms_result_'+str(i) for i in range(1,len(outputs)+1)]
    checked+=outdecl+indecl+'  type(cms_scalar_report), intent(out) :: report\n'
    checked+='  complex(kindr2) :: '+','.join(t+'(0:2)' for t in temps)+'\n'
    checked+='  call cms_preflight_'+name+'(report,'+','.join(args)+')\n'
    checked+='  if (report%status.ne.CMS_OK) return\n  report%native_called=.true.\n'
    checked+='  call '+name+'('+','.join(temps+args)+')\n'
    checked+='  if (.not.all(ieee_is_finite(real((/'+','.join(temps)+'/),kindr2))).or. &\n'
    checked+='      .not.all(ieee_is_finite(aimag((/'+','.join(temps)+'/))))) then\n'
    checked+='    report%status=CMS_NONFINITE_RESULT\n    return\n  endif\n'
    for result,temp in zip(outputs,temps):checked+='  '+result+'='+temp+'\n'
    checked+='  report%result_valid=.true.\n'
    checked+='  end subroutine cms_checked_'+name+'\n'
    return {'precision':precision,'name':name,'topology':n,'complex_masses':complex_mass,'line':startline,
            'original':original,'preflight':preflight,'checked':checked,'events':events,
            'max_events':{1:1,2:3,3:9,4:13}[n],
            'original_sha256':hashlib.sha256(original.encode()).hexdigest()}


def main():
    assert ROOT.name=='madgraph7_madloop_tirust'
    assert not (HERE/'generated').exists()
    output=HERE/'generated';output.mkdir()
    source=SOURCE.read_text();patched=source;records=[]
    for precision in ('dp','qp'):
        routines=[generate_routine(source,precision,name,i) for i,name in enumerate(NAMES,1)]
        for record in routines:
            for label in ('original','preflight','checked'):
                (output/f'{precision}_{record["name"]}_{label}.f90').write_text(record[label]+'\n')
        start=patched.index('module avh_olo_forCutTools_'+precision+'\n')
        end=patched.index('\nend module',start)
        host=patched[start:end]
        host=host.replace('\n','\n  use cms_scalar_observer\n',1)
        declarations='\n'.join('  public :: cms_preflight_'+name+', cms_checked_'+name for name in NAMES)+'\n'
        host=host.replace('\ncontains\n','\n'+declarations+'\ncontains\n',1)
        host+='\n! Additive bounded scalar-classification prototype.\n'
        host+='\n'.join(record['preflight']+record['checked'] for record in routines)
        patched=patched[:start]+host+patched[end:]
        records+=routines
    # Existing routines and all original public generic definitions remain exact.
    for record in records:
        actual,_=extract(patched,record['precision'],record['name'])
        assert actual==record['original']
    (output/'avh_olo_checked.f90').write_text(patched)
    metadata={'schema':'avh-absolute-classifier-prefix-v1','source':str(SOURCE),'source_sha256':sha(SOURCE),
        'observer_sha256':sha(HERE/'cms_scalar_observer.f90'),'generator_sha256':sha(__file__),
        'original_public_routines_byte_identical':True,'global_setters_called':False,
        'supported_overloads':[{k:v for k,v in record.items() if k not in ('original','preflight','checked')} for record in records],
        'unsupported':['complex-invariant overloads','explicit-rmu overloads','AN','B0','dB0','BN','MPFUN precision',
                       'uninitialized state','default/non-absolute policy','all-zero C0/D0'],
        'scope':'Every explicit nonzero metric-to-zero assignment in these no-rmu wrappers only; no certification of deeper analytic branches or physical accuracy',
        'generated_sha256':{str(p.relative_to(HERE)):sha(p) for p in sorted(output.iterdir())}}
    (HERE/'classification.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'overloads':len(records),'assignment_sites':sum(len(r['events']) for r in records),
                      'max_events_per_call':13,'compiler_or_native_started':False}))


if __name__=='__main__':main()
