"""Read retained inputs only; reconstruct conditional q words, never run reducers."""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import argparse
import gzip
import hashlib
import importlib.util
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CAP=ROOT/'docs/benchmarkfolder/experiments/current-kinematic-trace-runtime-qualification-1'
FAC=ROOT/'.tirust-build/kinematic-invariant-caller-metadata-1'
CAP_SHA='730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4'
FAC_SHA='aba903450f49e0e18ba3fff40dbf6074e1cd67e9af1c67cd83bbd6fd583c1047'
DEC_SHA='6a44cfd35294692f401b9be0d28458b2631f15fca1d80021d7e72a7fabbe2a7b'
DAG_SHA='29cfececfd12f6ded67fa6f89ab6de02d2991dfbe41adf0288fc655cc50c50c9'


def require(value,message):
    if not value:
        raise ValueError(message)


def pinned(path,sha,size=None):
    raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==sha,'Changed retained input: '+str(path))
    require(size is None or len(raw)==size,'Changed retained input size')
    return raw


def nearest(value,p,emin):
    if not value:
        return Fraction(0)
    sign=-1 if value<0 else 1
    value=abs(value)
    exponent=value.numerator.bit_length()-value.denominator.bit_length()
    if value<Fraction(2)**exponent:
        exponent-=1
    quantum=Fraction(2)**max(exponent-p+1,emin-p+1)
    scaled=value/quantum
    lower,remainder=divmod(scaled.numerator,scaled.denominator)
    lower+=int(2*remainder>scaled.denominator or
               (2*remainder==scaled.denominator and lower%2))
    return sign*lower*quantum


def exact_word(value,width):
    """Encode an exactly representable dyadic; canonicalize reconstructed zero."""
    nf,bias,emin=(52,1023,-1022) if width==64 else (112,16383,-16382)
    if not value:
        return 0
    sign=int(value<0)
    value=abs(value)
    exponent=value.numerator.bit_length()-value.denominator.bit_length()
    if value<Fraction(2)**exponent:
        exponent-=1
    stored=max(0,exponent+bias)
    scaled=value/Fraction(2)**(max(exponent,emin)-nf)
    require(scaled.denominator==1,'Reconstructed q is not representable')
    significand=int(scaled)
    if stored:
        significand-=1<<nf
    require(0<=significand<1<<nf,'Invalid reconstructed fraction')
    return (sign<<(width-1))|(stored<<nf)|significand


def word(record):
    return sum(int(chunk,16)<<(64*i) for i,chunk in enumerate(record['words']))


def build():
    index=json.loads(pinned(CAP/'evidence-index.json',CAP_SHA))
    factory_index=json.loads(pinned(FAC/'index.json',FAC_SHA))
    require(factory_index['files']['dag-metadata.json']['sha256']==DAG_SHA,'DAG not joined')
    dag=json.loads(pinned(FAC/'dag-metadata.json',DAG_SHA))
    decoder_path=CAP/'analysis/decoder/decode_trace.py'
    require(index['files']['analysis/decoder/decode_trace.py']['sha256']==DEC_SHA,'Decoder not joined')
    pinned(decoder_path,DEC_SHA)
    spec=importlib.util.spec_from_file_location('geometry_witness_retained_decoder',decoder_path)
    decoder=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    routes={row['route']:row for row in dag['routes']}
    cases=[]
    for point in (20,21):
        for mode in (1,4):
            logical=f'results/runs/cms-point{point}-mode{mode}-enabled/trace.dat'
            compressed=index['compressed_text'][logical]
            entry=index['files'][compressed['path']]
            data=pinned(CAP/compressed['path'],entry['sha256'],entry['bytes'])
            raw=gzip.decompress(data)
            require(hashlib.sha256(raw).hexdigest()==compressed['raw_sha256'] and
                    len(raw)==compressed['raw_bytes'],'Changed uncompressed trace')
            _,records=decoder.parse(raw,'low-first')
            calls=defaultdict(list)
            for record in records:
                key=tuple(record[k] for k in ('section','point','serial','id','mode','rank','n','split_order'))
                calls[key].append(record)
            selected=[(key,rows) for key,rows in calls.items() if key[3]==3]
            require(len(selected)==1,'Expected one retained group3 call')
            key,rows=selected[0]
            width=64 if mode==1 else 128
            p,emin=(53,-1022) if width==64 else (113,-16382)
            suffix='DP' if width==64 else 'QP'
            def one(tag,i=None,j=None):
                return decoder.one(rows,'KT_'+tag+'_'+suffix,i,j)
            external=[one('EXTERNAL',leg)['data']['complex_momentum'] for leg in range(1,5)]
            proof=dag['wavefunction_proofs']['dp' if width==64 else 'qp']
            signs=[proof[str(leg)][0]['sign'] for leg in range(1,5)]
            require(signs==[-1,-1,1,1],'Retained source-sign convention changed')
            n=key[6]
            shifts=one('MATRIX')['data']['shifts_component_major']
            i,j=1,3
            q=[nearest(shifts[mu*n+i]['value']-shifts[mu*n+j]['value'],p,emin) for mu in range(4)]
            raw_parts=one('RAW',i+1,j+1)['data']['diffsq']
            raw_replay=nearest(q[0]*q[0],p,emin)
            for value in q[1:]:
                raw_replay=nearest(raw_replay-nearest(value*value,p,emin),p,emin)
            require(raw_replay==raw_parts[0]['value'] and raw_parts[1]['value']==0,
                    'Conditional nearest raw arithmetic differs from captured DIFFSQ')
            chords=routes[3]['modes'][str(mode)]['chords']
            coefficients=[a-b for a,b in zip(chords[i]['signed_coefficients'],chords[j]['signed_coefficients'])]
            external_w=[[[word(external[leg][2*mu+part]) for part in range(2)]
                         for mu in range(4)] for leg in range(4)]
            # Signed graph coefficients already include these external signs.
            # Undo W=sign*P to recover source P bits, including signed zero.
            external_p=[[[pair[0]^((1<<(width-1)) if signs[leg]<0 else 0),pair[1]]
                         for pair in external_w[leg]] for leg in range(4)]
            cases.append({'point':point,'mode':mode,'width':width,'route':3,'pair':[2,4],
                'source_call_identity':dict(zip(('section','point','serial','id','mode','rank','n','split_order'),key)),
                'trace_logical_path':logical,'trace_gzip':compressed['path'],
                'trace_gzip_sha256':entry['sha256'],'trace_raw_sha256':compressed['raw_sha256'],
                'trace_raw_bytes':compressed['raw_bytes'],'external_source_signs':signs,
                'coefficients_relative_to_repaired_P':coefficients,
                'captured_external_W_bits':external_w,'inferred_external_P_bits':external_p,
                'captured_shift_pair_bits':[[word(shifts[mu*n+k]) for mu in range(4)] for k in (i,j)],
                'reconstructed_q_bits':[[exact_word(value,width),0] for value in q],
                'captured_raw_dot_bits':[word(v) for v in raw_parts],
                'conditional_nearest_raw_replay_exact':True,'owned_snapshot_observed':False,
                'q_component_ABI_observed':False,'q_zero_signs_observed':False})
    return {'schema':'private-exact-geometry-prospective-witnesses-v1',
        'retained_trace_capsule_sha256':CAP_SHA,'retained_factory_index_sha256':FAC_SHA,
        'retained_decoder_sha256':DEC_SHA,'retained_DAG_sha256':DAG_SHA,
        'scope':'Retained input bits only. q is reconstructed by exact nearest-even subtraction; '
                'original traces do not observe q temporary words or an owned fresh snapshot. '
                'Raw DIFFSQ arithmetic agrees under that explicit final-arithmetic assumption. '
                'Reconstructed zero q words use positive zero. No new native or ownership qualification.',
        'cases':cases}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true')
    arguments=parser.parse_args()
    result=build()
    path=HERE/'witnesses.json'
    if arguments.write:
        with path.open('x') as output:
            json.dump(result,output,sort_keys=True,indent=2)
            output.write('\n')
    else:
        require(result==json.loads(path.read_text()),'Frozen witnesses differ from sealed retained traces')
    print(json.dumps({'cases':len(result['cases']),'new_native_executions':0,
                      'q_component_ABI_observed':False,'owned_snapshot_observed':False},sort_keys=True))
