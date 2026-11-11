"""Hash-bound CMS momentum DAGs; parser reuse, no generated-code execution."""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from saved_source import Sources, blocks, call, external_rule, internal_rule, source_call_order, wf_number, statements

DERIVED = Path('.tirust-build/kinematic-provenance-current-executor-1/results/derived')
DERIVED_SHA = '25c2114756ed3eb968c311945a27d175c5c5a0af54d4ae21158abc8ed42a4184'
ROUTING_SHA = 'ca9ee4363075a15a9f0d81e3dc744331064da6f724c449b9125d8d64f2a1d499'


@dataclass(frozen=True)
class Expr:
    signed: tuple
    occurrences: tuple
    operations: int

    def neg(self):
        return Expr(tuple(-x for x in self.signed), self.occurrences, self.operations)

    def add(self, other):
        return Expr(tuple(x+y for x,y in zip(self.signed, other.signed)),
                    tuple(x+y for x,y in zip(self.occurrences, other.occurrences)),
                    self.operations+other.operations+1)

    def record(self):
        return {'signed_coefficients':list(self.signed),
                'unsigned_leaf_multiplicities':list(self.occurrences),
                'expanded_add_sub_operations':self.operations}


def zero():
    return Expr((0,)*4,(0,)*4,0)


def sum_seeded(nodes):
    out=zero()
    for node in nodes:
        out=out.add(node)
    return out


def require(test, message):
    if not test:
        raise ValueError(message)


def momentum_kind(block, name, width):
    kind='MP_ALOHA' if width==128 else 'ALOHA'
    require(any(s==f'TYPE({kind}){name}' for _,s,_ in block['statements']),
            'unknown or mixed momentum kind '+name)


def wavefunction_dags(sources, case, precision):
    width=64 if precision=='dp' else 128
    expected_real='REAL*8' if width==64 else 'REAL*16'
    external_text=sources.text(case['dhelas']+'/aloha_functions.f')
    storage=[s for _,s,_ in statements(external_text)]
    dtype='MP_ALOHA' if width==128 else 'ALOHA'
    start=storage.index('TYPE'+dtype);end=storage.index('ENDTYPE'+dtype,start)
    required=('COMPLEX*32::P(0:3)' if width==128 else 'DOUBLECOMPLEX::P(0:3)')
    require('SEQUENCE' in storage[start:end] and required in storage[start:end], 'external P storage kind')
    external_blocks=blocks(external_text)
    out, proof, unsupported = {}, {}, {}
    for filename in source_call_order(sources,case,precision):
        text=sources.text(filename)
        outer=list(blocks(text).values())
        require(len(outer)==1,'unexpected caller block count')
        require(any(s.startswith(expected_real+'P(0:3,NEXTERNAL)')
                    for _,s,_ in outer[0]['statements']), 'caller input kind')
        for line,statement,_ in outer[0]['statements']:
            parsed=call(statement)
            if not parsed or not parsed[1] or not re.fullmatch(r'W\([1-9][0-9]*\)',parsed[1][-1]):
                continue
            name,args=parsed; dest=wf_number(args[-1])
            receipt={'caller':filename,'line':line,'call':statement,'width':width}
            try:
                if re.fullmatch(r'(?:MP_)?[IOVST]XXXXX',name):
                    block=external_blocks[name]
                    leg,sign,_=external_rule(block,args)
                    require(leg==dest and 1<=leg<=4,'external slot mismatch')
                    momentum_kind(block,block['arguments'][-1],width)
                    pname=block['arguments'][0]
                    real_types=(expected_real,'DOUBLEPRECISION') if width==64 else (expected_real,)
                    require(any(s.startswith(t+pname+'(0:3)')
                                for _,s,_ in block['statements'] for t in real_types), 'external real input kind')
                    node=Expr(tuple(sign if i==leg else 0 for i in range(1,5)),
                              tuple(1 if i==leg else 0 for i in range(1,5)),0)
                    receipt.update({'kind':'same-kind external sign/copy','leg':leg,'sign':sign,
                                    'rounding_operations':0,'conversion_error':'exact real-to-complex same-kind; imaginary zero'})
                else:
                    path=case['dhelas']+'/'+name+'.f'
                    block=blocks(sources.text(path))[name]
                    terms=internal_rule(block,args)
                    formal=dict(zip(block['arguments'],args))
                    for param,actual in formal.items():
                        if re.fullmatch(r'W\([1-9][0-9]*\)',actual):
                            momentum_kind(block,param,width)
                    output=block['arguments'][-1]
                    writes=[(l,s) for l,s,_ in block['statements']
                            if re.match(re.escape(output)+r'%P\([^)]*\)=',s)]
                    require(len(writes)==1,'extra momentum write')
                    children=[out[i] if sign==1 else out[i].neg() for sign,i in terms]
                    require(all(x is not None for x in children),'unsupported dependency')
                    node=children[0]
                    for child in children[1:]:node=node.add(child)
                    receipt.update({'kind':'linear same-kind ALOHA sum','routine':path,
                                    'assignment_line':writes[0][0],'assignment':writes[0][1],
                                    'terms':[list(t) for t in terms],'new_add_sub_operations':len(terms)-1,
                                    'conversions':[]})
                require(dest not in out or out[dest]==node,'reused W changes expression')
                out[dest]=node; proof.setdefault(str(dest),[]).append(receipt)
            except (ValueError, KeyError, AttributeError) as exc:
                out[dest]=None;unsupported[str(dest)]=str(exc)
    require(all(i in out for i in range(1,9)),'missing expected CMS wavefunction slot')
    return out,proof,unsupported


def proportional(a,c):
    k=next(i for i,v in enumerate(c) if v)
    return a[k]%c[k]==0 and all(x==(a[k]//c[k])*y for x,y in zip(a,c))


def classify(a,c,fixed):
    if proportional(a,c):return 1
    for e in range(4):
        for sign in (-1,1):
            residual=list(a);residual[e]-=sign
            if proportional(residual,c):return 1 if fixed[e] else 0
    return 2


def assert_value_path(sources, case):
    """Recognize the exact admitted loops; all unrecognized variants reject."""
    wrapper_blocks=blocks(sources.text(case['subprocess']+'/CT_interface.f'))
    for name,block in wrapper_blocks.items():
        if not re.fullmatch(r'ML5_0_LOOP_[1-4](?:_[1-4])?',name):continue
        ss=[s for _,s,_ in block['statements']]
        for s in ('DOI=0,3','TEMP=1','DOJ=1,NLOOPLINE','PL(I,J)=0.D0',
                  'MP_PL(I,J)=0.0E+0_16','DOK=TEMP,(TEMP+PAIRING(J)-1)',
                  'PL(I,J)=PL(I,J)-W(WE(K))%P(I)',
                  'MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)',
                  'TEMP=TEMP+PAIRING(J)'):
            require(ss.count(s)==1,'corner operation sequence '+name+': '+s)
    tir=blocks(sources.text(case['subprocess']+'/TIRUST_interface.f'))
    for qp in (False,True):
        name='ML5_0_'+('MP_' if qp else '')+'TIRUST_TENSOR_DIAGNOSTIC'
        ss=[s for _,s,_ in tir[name]['statements']]
        for s in ('P_TMP=0','DOI=0,3','P_TMP(I,0)=P_TMP(I,0)+PL(I,J)',
                  'PREFIX_SUM=REAL(0,KIND=KI_'+('QTIR' if qp else 'TIR')+')',
                  'DOJ=1,NLOOPLINE-1','PREFIX_SUM=PREFIX_SUM+PL(I,J)',
                  'P_TMP(I,J)=PREFIX_SUM','SHIFTS(I,J)=P_TMP(I,J-1)',
                  'P_S_MAT(J,I)=SHIFTS(I,J)'):
            require(s in ss,'prefix operation missing '+name+': '+s)
        require(ss.count('P_TMP(I,0)=P_TMP(I,0)+PL(I,J)')==1,'closure repeated')
        require(ss.count('PREFIX_SUM=PREFIX_SUM+PL(I,J)')==1,'prefix repeated')
    order=blocks(sources.text(case['subprocess']+'/TIR_interface.f'))
    for name in ('ML5_0_SWITCH_ORDER','ML5_0_MP_SWITCH_ORDER'):
        ss=[s for _,s,_ in order[name]['statements']]
        for s in ('IF(CTMODE.NE.2.AND.CTMODE.NE.5)THEN','IF(NLOOPLINE.LE.2)THEN',
                  'NEW_PL(J,I)=-PL(J,NLOOPLINE+1-I)','PL(J,I)=NEW_PL(J,I)'):
            require(s in ss,'unsupported reversal '+name+': '+s)


def derive():
    raw=(DERIVED/'index.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==DERIVED_SHA,'derived index')
    idx=json.loads(raw)
    for name in ('cms-source-bindings.json','cms/routing.json'):
        data=(DERIVED/name).read_bytes();rec=idx['files'][name]
        require(len(data)==rec['bytes'] and hashlib.sha256(data).hexdigest()==rec['sha256'],name)
    sources=Sources('.',DERIVED/'cms-source-bindings.json',idx['files']['cms-source-bindings.json']['sha256'])
    setters=[]
    pattern=re.compile(r'\b(?:IEEE_SET_ROUNDING_MODE|IEEE_SET_UNDERFLOW_MODE|IEEE_SET_STATUS|FESETROUND|FESETENV|FEUPDATEENV|_CONTROLFP|_MM_SETCSR|LDMXCSR|FLDCW)\b',re.I)
    for filename,data in sources.raw.items():
        for line,text in enumerate(data.decode('ascii').splitlines(),1):
            if pattern.search(text):setters.append({'source':filename,'line':line,'text':text})
    require(not setters,'unreviewed floating-environment setter in bound source')
    case=next(iter(sources.binding['cases'].values()))
    routing=json.loads((DERIVED/'cms/routing.json').read_text())['cases']['cms_ew_n2_epem_mupmum']
    assert_value_path(sources,case)
    wf,proof,unsupported={},{},{}
    for precision in ('dp','qp'):
        wf[precision],proof[precision],unsupported[precision]=wavefunction_dags(sources,case,precision)
    fixed=[e['generated_keep_offshell_mass'] is False and e['pmass_expression']=='ZERO'
           and e['external_call_mass_expression']=='ZERO' for e in routing['external_provenance']]
    rows=[]
    for route in routing['reductions']:
        groups=route['corner_wavefunctions']; n=len(groups)
        record={'route':route['reduction_identifier'],'n':n,'nw':sum(map(len,groups)),
                'we':[x for group in groups for x in group],'pairing':list(map(len,groups)),
                'source':route['source'],'line':route['line'],'modes':{}}
        for mode in (1,2,4,5):
            precision='dp' if mode<4 else 'qp'; width=64 if mode<4 else 128
            if any(wf[precision][w] is None for g in groups for w in g):
                record['modes'][str(mode)]={'supported':False,'width':width};continue
            corners=[sum_seeded([wf[precision][w].neg() for w in g]) for g in groups]
            if mode in (2,5) and n>2:corners=[v.neg() for v in reversed(corners)]
            shifts=[sum_seeded(corners)]
            for j in range(1,n):shifts.append(sum_seeded(corners[:j]))
            expected=route['modes'][str(mode)]
            require([list(v.signed) for v in corners]==expected['corners'],'corner signed mismatch')
            require([list(v.signed) for v in shifts]==expected['shifts'],'chord signed mismatch')
            classes=[[1 if i==j else 0 for j in range(n)] for i in range(n)]
            differences=[]
            for j in range(1,n):
                for i in range(j):
                    diff=shifts[i].add(shifts[j].neg())
                    kind=classify(diff.signed,route['conservation_coefficients'],fixed)
                    classes[i][j]=classes[j][i]=kind
                    differences.append({'pair':[i+1,j+1],**diff.record()})
            record['modes'][str(mode)]={'supported':True,'width':width,
                'corners':[v.record() for v in corners],'chords':[v.record() for v in shifts],
                'differences':differences,'pair_class':classes,
                'kind_conversions':'same-kind only; complex real projection exact for zero imaginary linear path'}
        rows.append(record)
    require(len(rows)==26 and sorted(r['route'] for r in rows)==list(range(1,27)),'route inventory')
    return {'schema':'cms-source-dag-bounds-v1','source_index_sha256':DERIVED_SHA,
            'routing_sha256':ROUTING_SHA,'input_binding_sha256':idx['files']['cms-source-bindings.json']['sha256'],
            'source_files_verified':len(sources.raw),'source_files_consumed':sorted(sources.consumed),
            'wavefunction_proofs':proof,'unsupported_wavefunctions':unsupported,'routes':rows,
            'environment_source_scan':{'source_files':len(sources.raw),'pattern':pattern.pattern,'matches':setters,
                'scope':'Source spelling audit only; compile/runtime no-unsafe-flags and entry/use controls remain separate preconditions.'},
            'count_definition':'expanded source-expression add/sub count; duplication overcounts shared executed DAG nodes conservatively',
            'external_reference':'actual postrepair external W/MP_W leaves, source-proved same-kind sign copies of numerator P/MP_P',
            'native_executions':0}


if __name__=='__main__':
    print(json.dumps(derive(),sort_keys=True,indent=2))
