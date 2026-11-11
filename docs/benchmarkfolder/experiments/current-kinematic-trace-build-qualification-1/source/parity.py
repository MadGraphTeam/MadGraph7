"""Exact observational parity only; fixed-mode output is not an accuracy oracle."""
from pathlib import Path
import re

from admission import read_json, require, sha

EQUAL_FILES=('bootstrap-values.dat','driver-state.dat','driver-bits.dat',
             'qualification.dat','dimensions.dat','values.dat','timed_status.dat')


def states(raw):
    out=[]
    for line in raw.splitlines():
        if not line or line.startswith('#'):continue
        fields=line.split()
        require(len(fields)==16 and fields[0]=='STATE' and fields[1] in ('A','B'),
                'Driver state row schema differs')
        numbers=list(map(int,fields[2:12]));bits=fields[12:]
        require(all(re.fullmatch('[0-9A-F]{8}',value) for value in bits),'Logical raw bits malformed')
        out.append({'stage':fields[1],'phase':numbers[0],'call':numbers[1],
                    'point':numbers[2],'requested_mode':numbers[3],'code':numbers[4],
                    'actual_mode':numbers[5],'checkphase':numbers[6],'helicity_checked':numbers[7],
                    'fpe_dp':numbers[8],'fpe_qp':numbers[9],'bits':bits})
    return out


def fixed_mode_contract(raw,point,mode):
    rows=states(raw)
    expected=[(phase,call,stage) for phase,calls in ((0,[1]),(1,[2]))
              for call in calls for stage in ('B','A')]
    require([(r['phase'],r['call'],r['stage']) for r in rows]==expected,
            'Incomplete or duplicate driver calls')
    end=rows[-1]
    # LoopPoleMode=0 temporarily bypasses filters and restores these flags.
    # Their post-call values are observations, never a readiness predicate.
    require(end['point']==point and end['requested_mode']==end['actual_mode']==mode,
            'Measured actual mode differs from request')
    require(end['code']=={1:110,4:120}[mode], 'Fixed-mode unchecked return code differs')
    return rows


def output_coverage(directory,point,mode):
    """Retain both total and sole split-order columns, including bootstrap."""
    directory=Path(directory)
    require((directory/'dimensions.dat').read_text().split()==['1','1','1'],
            'Current CMS/EW split-order dimensions differ')
    rows=[]
    for line in (directory/'driver-bits.dat').read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        values=line.split()
        require(len(values)==10 and all(re.fullmatch('[0-9A-F]{16}',v) for v in values[5:]),
                'Raw answer-bit schema differs')
        head=list(map(int,values[:5]));rows.append(head)
        if head[0]==1:
            require(head[4]=={1:110,4:120}[mode],'Per-order fixed return code differs')
            require(all(int(word,16)&0x7ff0000000000000!=0x7ff0000000000000 for word in values[5:9]),
                    'Nonfinite measured coefficient retained')
    expected=[(0,1,1,0),(0,1,1,1),(1,2,point,0),(1,2,point,1)]
    require([tuple(row[:4]) for row in rows]==expected,'Missing/duplicate answer split-order coverage')
    values=[line.split() for line in (directory/'values.dat').read_text().splitlines() if line.strip()]
    require(len(values)==2 and all(len(row)==8 for row in values) and
            [tuple(map(int,row[:3])) for row in values]==[(1,0,{1:110,4:120}[mode]),(1,1,{1:110,4:120}[mode])],
            'Decimal answer split-order/status coverage differs')
    return {'orders':[0,1],'bootstrap_columns':2,'measurement_columns':2}


def trace_modes(path,point,mode,groups):
    count=0;seen=set();schema=0;serials={}
    with Path(path).open() as stream:
        for line in stream:
            fields=line.split()
            require(fields,'Empty trace record')
            if fields[0]=='KT_SCHEMA_V2':
                schema+=1;require(fields[1:]==[str(point),'64','128','64','32'],'Native storage schema differs')
                continue
            require(fields[0].startswith('KT_') and len(fields)>=11,'Malformed trace record')
            header=list(map(int,fields[1:11]))
            require(header[0]==point and header[2] in groups and header[3]==mode,
                    'Trace point/group/mode differs from admitted request')
            tags=serials.setdefault(header[1],set());tags.add(fields[0])
            if fields[0].startswith('KT_STATUS_'):
                require(len(fields)==12,'Tensor evaluation status schema differs')
                require(int(fields[11])==0,'Tensor evaluation returned failure; retained as incomplete')
            else:
                require(all(re.fullmatch('[0-9A-F]{16}',v) for v in fields[11:]),
                        'Trace bit payload malformed')
            seen.add(header[2]);count+=1
    require(schema==1 and count>0 and seen==set(groups),'Missing requested group trace evidence')
    suffix='DP' if mode==1 else 'QP'
    required={'KT_LOOP_DP','KT_MATRIX_'+suffix,'KT_RESIDUAL_'+suffix,
              'KT_TIRUST_'+suffix,'KT_STATUS_'+suffix}
    require(all(required<=tags for tags in serials.values()),
            'Missing terminal tensor stage; an early return is not a complete reduction trace')
    return {'records':count,'groups':sorted(seen),'actual_mode':mode,
            'evaluation_status_is_not_host_stability':True}


def compare_stage(root,requests):
    root=Path(root);groups={};pairs=[];failures=[]
    for r in requests:groups.setdefault((r['case'],r['point'],r['requested_mode']),[]).append(r)
    for (case,point,mode),group in groups.items():
        try:
            require({r['variant'] for r in group}=={'unmodified','disabled','enabled'},'Incomplete variants')
            dirs={r['variant']:root/r['id'] for r in group}
            results={v:read_json(p/'result.json') for v,p in dirs.items()}
            require(all(x['orchestration_complete'] and x['runtime_map_verified'] and
                        not x['stdout_overflow'] and x['trace_complete'] and not x['collection_errors']
                        and x['native_returncode']==0 and x['close_receipts']
                        and all(close['complete'] and close['worker_closed'] for close in x['close_receipts'])
                        for x in results.values()),
                    'Incomplete worker/runtime evidence')
            for filename in EQUAL_FILES:
                raw={v:(p/filename).read_bytes() for v,p in dirs.items()}
                require(len(set(raw.values()))==1,'Variant observations differ in '+filename)
            raw=(dirs['unmodified']/'driver-state.dat').read_text()
            observations=fixed_mode_contract(raw,point,mode)
            coverage=output_coverage(dirs['unmodified'],point,mode)
            # Status/flags and every raw row are retained even for equal failure.
            final=observations[-1]
            numerical_failure=bool(final['fpe_dp'] or final['fpe_qp'] or final['code']<0 or final['code']>=400)
            require(not numerical_failure,'Equal variants retain a failed contribution; interpretation blocked')
            warnings=[]
            for v,p in dirs.items():
                console=(p/'stdout.txt').read_text(errors='replace')
                warnings.append([line for line in console.splitlines() if '##W03' in line])
            require(warnings[0]==warnings[1]==warnings[2],'Incomplete-contribution warning mismatch')
            require(not warnings[0],'Incomplete-contribution warning retained; interpretation blocked')
            require(not (dirs['unmodified']/'trace.dat').read_bytes() and
                    not (dirs['disabled']/'trace.dat').read_bytes(),'Disabled trace produced records')
            trace=trace_modes(dirs['enabled']/'trace.dat',point,mode,
                              [3,4,8,12,24] if case=='cms' else [5])
            pairs.append({'case':case,'point':point,'mode':mode,'exact_observation_parity':True,
                          'trace':trace,'coverage':coverage,'independent_accuracy_qualified':False})
        except (ValueError,KeyError,OSError) as exc:
            failures.append({'case':case,'point':point,'mode':mode,'failure':str(exc)})
    return {'passed':not failures and len(pairs)==len(groups),'comparisons':pairs,
            'failures':failures,'interpretation_gate':'Exact U/D/E observations, explicit full-pole bypass contract and actual fixed mode',
            'independent_accuracy_qualified':False,'performance_measurement':False}
