"""Pure universal QED IR single-pole calculation using closed binary128 inputs."""
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

HERE=Path(__file__).resolve().parent
TRACE_SEAL='730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4'
DOUBLE_SEAL='5748eeb64f53f4a201ae0e5a355221c03e5dac39790d55531e52b0d290e057e3'
CLOSURES={1:('results/ordinary-terminal-closure.json','56d4249509bbda4d1aa3c328b18b42176c7419f4805a1d4454a73a7fa3608fcf'),
          20:('results/stress-terminal-closure.json','df0987b3f0c20dd26c4a903ff7c4acc7fb6cce4e433a50071a82918194553e9e')}


def require(ok,message):
    if not ok:raise ValueError(message)


def pin(raw):
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def binary128(words):
    require(len(words)==2 and all(len(w)==16 for w in words),'Invalid binary128 words')
    bits=int(words[0],16)+(int(words[1],16)<<64)
    exponent=(bits>>112)&32767
    require(exponent!=32767,'Nonfinite captured input')
    mantissa=(bits&((1<<112)-1))+((1<<112) if exponent else 0)
    power=(exponent or 1)-16383-112
    return (-1 if bits>>127 else 1)*mantissa*(F(2**power) if power>=0 else F(1,2**-power))


def dot(a,b):
    return a[0]*b[0]-sum(a[i]*b[i] for i in (1,2,3))


def decimal(x):
    return D(x.numerator)/D(x.denominator)


def wire(x):
    if isinstance(x,F):
        return {'numerator':str(x.numerator),'denominator':str(x.denominator),'decimal_70_digits':str(decimal(x))}
    if isinstance(x,list):return [wire(v) for v in x]
    if isinstance(x,dict):return {k:wire(v) for k,v in x.items()}
    return x


def load_trace_payloads():
    raw=(HERE/'trace-parent-index.json').read_bytes()
    require(pin(raw)['sha256']==TRACE_SEAL,'Published trace seal changed')
    index=json.loads(raw)
    retained=json.loads((HERE/'trace-retention.json').read_bytes())
    data={}
    for logical,binding in retained.items():
        stored=(HERE/binding['local']).read_bytes()
        require(pin(stored)==index['files'][binding['parent_storage']],'Retained stored trace payload changed')
        content=gzip.decompress(stored) if binding['local'].endswith('.gz') else stored
        require(pin(content)=={'bytes':binding['raw_bytes'],'sha256':binding['raw_sha256']},'Raw trace payload changed')
        origin=index['origins'][logical]
        require(pin(content)=={k:origin[k] for k in ('bytes','sha256')} and origin['path']==binding['origin_path'],
                'Published trace origin differs')
        data[logical]=content
    require(len(data)==6,'Unexpected retained trace scope')
    return data


def external_inputs(raw,point):
    selected=[]
    for lineno,line in enumerate(raw.decode().splitlines(),1):
        fields=line.split()
        if fields[0]=='KT_SCHEMA_V2':
            require(list(map(int,fields[1:]))==[point,64,128,64,32],'Trace ABI/schema differs')
            continue
        require(len(fields)>=11,'Malformed trace header')
        header=list(map(int,fields[1:11]))
        if header[2]!=3:continue
        if fields[0] not in ('KT_EXTERNAL_QP','KT_TIRUST_QP'):continue
        require(header[0]==point and header[3:6]==[4,2,4] and header[9]>0,'Unexpected input call')
        selected.append((fields[0],header,fields[11:],lineno))
    context={(h[0],h[1],h[2],h[3],h[4],h[5],h[9]) for _,h,_,_ in selected}
    require(len(context)==1,'Ambiguous captured input context')
    leaves=[r for r in selected if r[0]=='KT_EXTERNAL_QP']
    native=[r for r in selected if r[0]=='KT_TIRUST_QP']
    require(len(leaves)==4 and len(native)==1,'Missing/duplicate leaf or scale inputs')
    leaves.sort(key=lambda r:r[1][6])
    require([r[1][6] for r in leaves]==[1,2,3,4],'External leaf order differs')
    vectors=[]
    for _,_,words,_ in leaves:
        require(len(words)==16,'External complex-vector shape differs')
        require(all(binary128(words[j+2:j+4])==0 for j in range(0,16,4)),'Nonreal external input')
        vectors.append([binary128(words[j:j+2]) for j in range(0,16,4)])
    require(all(v[0]<0 for v in vectors[:2]) and all(v[0]>0 for v in vectors[2:]),'Crossed momentum signs differ')
    words=native[0][2]
    require(len(words)==82,'Native input width differs')
    mu2=binary128(words[-2:]);require(mu2>0,'Nonpositive native mu squared')
    return vectors,mu2,{'context':list(next(iter(context))),
                        'external_lines':[r[3] for r in leaves],'native_input_line':native[0][3],
                        'external_complex_words':[r[2] for r in leaves],'native_mu_squared_words':words[-2:]}


def verify_numerical_card(raw):
    lines=raw.decode().splitlines()
    def value(name):
        found=[i for i,s in enumerate(lines) if s.strip()=='#'+name]
        require(len(found)==1,'Ambiguous numerical parameter '+name)
        return lines[found[0]+1].strip().upper()
    require(value('LoopPoleMode')=='0','UV+IR bare-loop pole setting required')
    require(value('TIRustComputeFinite')=='.TRUE.','Complete finite-mode amplitude setting required')
    require(value('UseQPIntegrandForTIRust')=='.TRUE.','Genuine QP integrand setting required')


def calculate():
    # Reuse the prior immutable model/card derivation without editing its seal.
    old_raw=(HERE/'double-pole/index.json').read_bytes()
    require(pin(old_raw)['sha256']==DOUBLE_SEAL,'Prior double-pole packet changed')
    old_index=json.loads(old_raw)
    for name,identity in old_index['files'].items():
        require(pin((HERE/'double-pole'/name).read_bytes())==identity,'Prior retained text changed')
    spec=importlib.util.spec_from_file_location('retained_double_pole_calculation',HERE/'double-pole/calculate.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    base=old.calculate()
    require(base==json.loads((HERE/'double-pole/result.json').read_bytes()),'Prior model replay differs')
    observed=json.loads((HERE/'double-pole/observations.json').read_bytes())['rows']
    raw_card=(HERE/'MadLoopParams.dat').read_bytes();verify_numerical_card(raw_card)
    trial=json.loads((HERE/'trial-specification.json').read_bytes())
    physical=pin((HERE/'double-pole/param_card.dat').read_bytes())
    require(trial['physical_card']==physical and trial['numerical_card']==pin(raw_card),'Trial cards differ')
    require(trial['point_pin']==pin((HERE/'point20.dat').read_bytes()) and
            trial['bootstrap_pin']==pin((HERE/'point1.dat').read_bytes()),'Trial point inputs differ')
    data=load_trace_payloads()
    results=[]
    with localcontext() as context:
        context.prec=70
        alpha=D(base['alpha_reconstructed']);pi=D('3.1415926535897932384626433832795')
        physical_charges=[1,-1,1,-1]
        signs=[-1,-1,1,1]
        charges=[q*s for q,s in zip(physical_charges,signs)]
        require(sum(charges)==0 and sum(charges[i]*charges[j] for i in range(4) for j in range(i+1,4))==-2,
                'Charge conservation/pair sum differs')
        for point,phase in ((1,0),(20,1)):
            label=f'cms-point{point}-mode4-enabled'
            closure_name,closure_sha=CLOSURES[point]
            require(pin(data[closure_name])['sha256']==closure_sha,'Closed trace receipt changed')
            closure=json.loads(data[closure_name]);worker=closure['workers'][label]
            logical='results/runs/'+label+'/trace.dat'
            require(pin(data[logical])==worker['trace_pin'],'Trace not joined to its closed worker')
            specification=json.loads(data['results/runs/'+label+'/specification.json'])
            require(specification['physical_card']==physical and specification['numerical_card']==pin(raw_card),
                    'Trace model/selection differs from the tested trial')
            require(specification['point_pin']==pin((HERE/f'point{point}.dat').read_bytes()),'Trace physical point differs')
            vectors,mu2,input_record=external_inputs(data[logical],point)
            pairs=[];soft=D(0)
            for i in range(4):
                for j in range(i+1,4):
                    invariant=abs(2*dot(vectors[i],vectors[j]))
                    require(invariant>0,'Zero pair invariant: fixed noncollinear formula unavailable')
                    logarithm=(decimal(invariant/mu2)).ln()
                    term=-2*charges[i]*charges[j]*logarithm;soft+=term
                    pairs.append({'legs':[i+1,j+1],'all_outgoing_charge_product':charges[i]*charges[j],
                                  'positive_pair_invariant':invariant,'log_pair_over_mu2':str(logarithm),
                                  'single_soft_factor_term':str(term)})
            factor=-D(6)+soft
            inv={(r['legs'][0],r['legs'][1]):r['positive_pair_invariant'] for r in pairs}
            factored=inv[1,2]*inv[1,3]*inv[2,4]*inv[3,4]/(inv[1,4]*inv[2,3]*mu2**2)
            factored_factor=-D(6)+2*decimal(factored).ln()
            require(abs(factored_factor-factor)<D('1e-60'),'Four-leg factored logarithm differs')
            # Reproduce the FKS intermediate QES convention and its final scale
            # expansion; varying auxiliary QES must cancel algebraically.
            qes=mu2*F(13,7)
            fks=-D(6)-2*sum(D(charges[i]*charges[j])*(decimal(abs(2*dot(vectors[i],vectors[j]))/qes)).ln()
                            for i in range(4) for j in range(i+1,4))
            fks-=4*(decimal(mu2/qes)).ln()
            require(abs(fks-factor)<D('1e-60'),'Auxiliary FKS scale failed to cancel')
            comparisons=[]
            for row in [r for r in observed if r['phase']==phase]:
                born,single=[D.from_float(struct.unpack('>d',bytes.fromhex(row['words'][k]))[0]) for k in (0,2)]
                expected=born*alpha/(2*pi)*factor
                comparisons.append({'worker':row['worker'],'order':row['order'],'Born':str(born),
                    'observed_single_pole':str(single),'expected_IR_single_pole':str(expected),
                    'relative_residual':str(abs((single-expected)/expected))})
            results.append(wire({'point':point,'phase':phase,'captured_inputs':input_record,
                'all_outgoing_vectors':vectors,'native_mu_squared':mu2,'pair_terms':pairs,
                'external_shell_residuals':[dot(v,v) for v in vectors],
                'external_conservation_residual':[sum(v[a] for v in vectors) for a in range(4)],
                'collinear_factor':'-6','soft_factor':str(soft),'single_factor':str(factor),
                'factored_charge_log_ratio':factored,'factored_single_factor':str(factored_factor),
                'auxiliary_QES_squared':qes,'FKS_form_factor':str(fks),
                'trace_parent_stage_passed':closure['stage']['passed'],
                'trace_parent_terminal_exit_code':closure['terminal_exit_code'],'comparisons':comparisons}))
        return {'schema':'private-cms-four-massless-leg-qed-IR-single-pole-audit-v1',
            'native_executions':0,'decimal_working_digits':70,'pair_dot_arithmetic':'Exact rational binary128 input words',
            'alpha_reconstructed':str(alpha),'physical_charges':physical_charges,'all_outgoing_charges':charges,
            'formula':'A[-1]=Born*alpha/(2*pi)*[-6-2*sum(i<j,Q_i*Q_j*ln(abs(2*k_i.k_j)/mu^2))]',
            'source_normalization':'FKS massless fermion gamma=3/2; charge link crossing; -2 link weight; final double*ln(mu^2/QES^2)',
            'amplitude_scope':'Renormalized full ANS with UV counterterms and UV+IR loop selection; not bare tensor IR1 alone',
            'model_runtime_coupling_bits_observed':False,'current_trial_new_external_word_capture':False,
            'trace_input_ancestry':'Earlier closed actual QP leaves; exact physical point/card identity and unchanged momentum-source path',
            'points':results,'finite_truth_qualified':False,'full_amplitude_truth_qualified':False,
            'checked_stability_qualified':False,'performance_qualified':False,
            'UV_cancellation_independently_measured':False,'failed_tensor_outputs_read':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-live-sources',action='store_true');args=parser.parse_args()
    if args.verify_live_sources:
        repo=HERE.parents[1]
        for name,identity in json.loads((HERE/'source-pins.json').read_bytes()).items():
            require(pin((repo/name).read_bytes())==identity,'Live source changed: '+name)
    print(json.dumps(calculate(),indent=2,sort_keys=True))
