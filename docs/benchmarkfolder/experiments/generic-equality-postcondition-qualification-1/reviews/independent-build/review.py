"""Independent fixture mathematics and closed BUILD review; no native actions."""
import hashlib,json,math,re,time
from collections import Counter
from datetime import datetime
from fractions import Fraction
from pathlib import Path
ROOT=Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
SOURCE=ROOT/'.tirust-build/generic-equality-postcondition-qualification-proposal-1'
BUILD=ROOT/'.tirust-build/generic-equality-postcondition-qualification-work-1/build'
HERE=Path(__file__).resolve().parent
SOURCE_SHA='4e3878690b366992d1dad8e43ae650a62362220d7bfc21e85850228d2aeef12d'
TERMINAL='a12fd704ab54e935028258299239beb62f6d366620f990e0130db4f9df4fa506'
RESULT='13b896d86270ca9da75404046d0a1de655a0adaabcf8987cbf2846f54b2f5f25'
hashed={}
def sha(b):return hashlib.sha256(b).hexdigest()
def match(p,want):
    p=Path(p).resolve()
    if str(p) not in hashed:hashed[str(p)]=sha(p.read_bytes())
    assert hashed[str(p)]==want,str(p)
def read(p):return json.loads(Path(p).read_text())
def decode(bits,width):
    if width not in (64,128) or not 0<=bits<1<<width:raise ValueError('width or high storage bits')
    nf,eb,bias={64:(52,11,1023),128:(112,15,16383)}[width]
    exponent=(bits>>nf)&((1<<eb)-1);fraction=bits&((1<<nf)-1)
    if exponent==(1<<eb)-1:raise ValueError('nonfinite')
    mantissa=fraction+(1<<nf if exponent else 0)
    shift=(exponent if exponent else 1)-bias-nf
    sign=-1 if bits>>(width-1) else 1
    rational=Fraction(sign*mantissa)*(Fraction(2)**shift)
    if mantissa:
        z=(mantissa&-mantissa).bit_length()-1;mantissa>>=z;shift+=z
    return rational,mantissa,shift
def expectation(header,coeff,external,q):
    ident,width,reference,n,shape=header
    if width not in (64,128) or reference!=width or not 1<=n<=64 or shape:return False,None,None
    try:
        # Validate every component, including coefficient-zero leaves.
        leaves=[[decode(pair[0],width) for pair in vec] for vec in external]
        target=[decode(pair[0],width) for pair in q]
        if any(decode(pair[1],width)[0] for vec in external for pair in vec) or any(decode(pair[1],width)[0] for pair in q):return False,None,None
    except ValueError:return False,None,None
    equal=all(sum((c*leaves[j][mu][0] for j,c in enumerate(coeff)),Fraction(0))==target[mu][0] for mu in range(4))
    capacity=[]
    for mu in range(4):
        terms=[]
        for j,c in enumerate(coeff):
            magnitude=leaves[j][mu][1]
            if not c or not magnitude:continue
            absolute=abs(c);zeros=(absolute&-absolute).bit_length()-1
            product=magnitude*(absolute>>zeros)
            if product.bit_length()>127:return False,equal,None
            terms.append((product,leaves[j][mu][2]+zeros))
        if target[mu][1]:terms.append((target[mu][1],target[mu][2]))
        if not terms:capacity.append(0);continue
        base=min(e for _,e in terms);magnitudes=[]
        for m,e in terms:
            d=e-base
            if d>126 or m.bit_length()+d>127:return False,equal,None
            magnitudes.append(m<<d)
        total=sum(magnitudes)
        if total>(1<<127)-1:return False,equal,None
        capacity.append(total.bit_length())
    return equal,equal,capacity

def main():
    start=time.monotonic();match(SOURCE/'source-index.json',SOURCE_SHA);seal=read(SOURCE/'source-index.json')
    assert {str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file()}==set(seal['files'])|{'source-index.json'}
    for n,w in seal['files'].items():
        p=SOURCE/n;assert not p.is_symlink() and p.stat().st_size==w['bytes'];match(p,w['sha256'])
    pins=read(SOURCE/'input-pins.json');ancestor=read(SOURCE/'ancestry.json');tools=read(SOURCE/'toolchain-pins.json')
    for name,want in {**ancestor['source_files'],**pins['support'],**pins['closed_standalone_build'],**pins['files'],**tools['donor_lineage']}.items():
        match(Path(name) if Path(name).is_absolute() else ROOT/name,want)
    parent=ROOT/ancestor['helper_parent'];match(parent/'source-index.json',ancestor['helper_index_sha256'])
    for n,w in read(parent/'source-index.json')['files'].items():match(parent/n,w['sha256'])
    for n,w in pins['additional_compiler_files'].items():match(n,w['sha256']);assert Path(n).stat().st_size==w['bytes']
    for n in pins['additional_compiler_absences']:assert not Path(n).exists() and not Path(n).is_symlink()
    expected=read(SOURCE/'expectations.json');lines=(SOURCE/'fixtures.hex').read_text().splitlines();cursor=1
    assert int(lines[0])==expected['case_count']==1698
    counts=Counter();equal_capacity_refusals=0
    for identifier in range(1,1699):
        begin=cursor;header=list(map(int,lines[cursor].split()));cursor+=1
        assert len(header)==5 and header[0]==identifier
        _,width,reference,n,shape=header
        coeff=list(map(int,lines[cursor].split()));cursor+=1;assert len(coeff)==n and all(-(1<<31)<=c<(1<<31) for c in coeff)
        external=[]
        for leg in range(n):
            vector=[]
            for mu in range(4):
                words=lines[cursor].split();cursor+=1;assert len(words)==2 and all(re.fullmatch('[0-9A-F]{32}',v) for v in words)
                vector.append([int(v,16) for v in words])
            external.append(vector)
        q=[]
        for mu in range(4):
            words=lines[cursor].split();cursor+=1;assert len(words)==2 and all(re.fullmatch('[0-9A-F]{32}',v) for v in words)
            q.append([int(v,16) for v in words])
        row=expected['cases'][str(identifier)]
        assert [identifier,row['width'],row['reference_width'],row['n'],row['shape']]==header
        assert sha(('\n'.join(lines[begin:cursor])+'\n').encode())==row['input_case_sha256']
        accept,equal,capacity=expectation(header,coeff,external,q)
        assert accept==row['expected_accept'] and equal==row['mathematical_equality'] and capacity==row['component_absolute_sum_bits'],row['tag']
        counts['accept' if accept else 'refuse']+=1
        equal_capacity_refusals+=bool(equal and not accept)
    assert cursor==len(lines) and counts=={'accept':526,'refuse':1172} and equal_capacity_refusals==10
    match(BUILD/'terminal.json',TERMINAL);match(BUILD/'result.json',RESULT)
    term=read(BUILD/'terminal.json');result=read(BUILD/'result.json');admission=read(BUILD/'admission.json')
    assert term['status']==result['status']=='passed' and term['returncode']==0 and not term['cleanup']['survivors']
    assert term['cap_seconds']==180 and term['cleanup_reserve_seconds']==60 and term['result_sha256']==RESULT
    elapsed=(datetime.fromisoformat(term['finished_utc'])-datetime.fromisoformat(term['started_utc'])).total_seconds();assert elapsed<180
    assert result['source_index_sha256']==admission['source_index_sha256']==SOURCE_SHA
    assert result['helper_main_calls']==result['scientific_reducer_calls']==0
    assert read(BUILD/'limits.json')['actual']=={'address_space_bytes':[4294967296]*2,'file_bytes':[268435456]*2,'core_bytes':[0,0]}
    match(SOURCE/'exact_geometry.f90',result['helper_sha256']);match(BUILD/'fenv_observer.o',result['fenv_object_sha256'])
    command_counts=Counter();command_names=set()
    for file in (BUILD/'commands').glob('*.command.json'):
        label=file.name.removesuffix('.command.json');command_names.add(label);command=read(file);closed=read(file.with_name(label+'.terminal.json'))
        assert closed['returncode']==0;match(file.with_name(label+'.stdout.txt'),closed['stdout_sha256'])
        argv=command['argv']
        if label.endswith('-compile'):kind='compile'
        elif label.endswith('-link'):kind='link'
        elif 'prelink-query-' in label:kind='prelink'
        elif label.endswith('-version'):kind='version'
        elif label.endswith('-symbols') or label.endswith('-undefined'):kind='nm'
        else:kind='loader';assert '--list' in argv
        command_counts[kind]+=1
    assert command_counts=={'compile':5,'link':2,'prelink':42,'version':2,'nm':4,'loader':2}
    variants={}
    for name,row in result['variants'].items():
        case=BUILD/name;match(row['path'],row['sha256'])
        for n,h in row['objects'].items():match(case/n,h)
        for n,k in [('prelink.json','prelink_sha256'),('link.map','link_map_sha256'),('static-runtime.json','static_runtime_sha256')]:match(case/n,row[k])
        pre=read(case/'prelink.json');assert pre['completed_before_link'] and pre['explicit_standalone_runtime_join_before_link']==pins['runtime_before_link_augmentation']
        for p,h in pre['bound'].items():match(p,h)
        loads={}
        for spelling in re.findall(r'^LOAD\s+(.+)$',(case/'link.map').read_text(),re.M):
            path=Path(spelling);path=(case/path).resolve() if not path.is_absolute() else path.resolve()
            assert str(path) in pre['bound'];loads[str(path)]=pre['bound'][str(path)]
        assert loads==row['actual_link_loads'] and len(loads)==20
        providers=row['runtime']['files'];assert len(providers)==5 and providers==pins['files']
        assert all(pre['bound'].get(n)==h for n,h in providers.items())
        runtime_cmd=read(BUILD/'commands'/(name+'-actual-pt-interp-runtime.command.json'))
        assert runtime_cmd['argv']==[row['runtime']['loader'],'--list',row['path']]
        assert (BUILD/'commands'/(name+'-actual-pt-interp-runtime.stdout.txt')).read_text()==row['runtime']['listing']
        listed={}
        for line in row['runtime']['listing'].splitlines():
            if 'linux-vdso' in line:continue
            found=re.search(r'(/[^\s()]+)\s+\(0x',line);assert found
            path=str(Path(found[1]).resolve());assert path in providers;listed[path]=providers[path]
        assert listed==providers
        undefined=(BUILD/'commands'/(name+'-helper-undefined.stdout.txt')).read_text()
        assert re.findall(r'\bU\s+(\S+)',undefined)==row['helper_undefined_symbols']
        assert all(symbol.startswith(('__addv','__subv','__negv','__mulv','__divti3','_gfortran_runtime_error_at')) for symbol in row['helper_undefined_symbols'])
        symbols=(BUILD/'commands'/(name+'-symbols.stdout.txt')).read_text()
        assert len(re.findall(r'^\S+\s+\S\s+__ml_private_exact_geometry_MOD_exact_geometry_difference$',symbols,re.M))==1
        command=read(BUILD/'commands'/(name+'-helper-compile.command.json'))
        assert command['argv'][1:-5]==row['flags'] or command['argv'][1:1+len(row['flags'])]==row['flags']
        assert row['flags']==({'checked':['-O0','-fcheck=all','-ftrapv'],'optimized':['-O3','-fno-fast-math']}[name])
        variants[name]={'executable_sha256':row['sha256'],'runtime_providers':5,'link_LOAD_count':20,'flags':row['flags'],'undefined_symbols':row['helper_undefined_symbols']}
    assert result['variants']['checked']['objects']['driver.o']==result['variants']['optimized']['objects']['driver.o']
    receipt={'schema':'independent-exact-geometry-source-build-audit-v1','status':'PASS','source_index_sha256':SOURCE_SHA,
        'build_terminal_sha256':TERMINAL,'build_result_sha256':RESULT,'independent_fixture_cases':1698,'independent_accepts':526,
        'independent_refusals':1172,'exact_equal_capacity_refusals':10,'all_case_echo_hashes_checked':True,
        'independent_oracle':'Direct IEEE integer-field decoding to Fraction plus signed128 product/alignment/absolute-sum capacity, without calling supplied corpus/reference code.',
        'source_review':'Helper bounds products and every signed-sum ordering; zero coefficient leaves still validate. Actual fenv requests/masks/raw status/before-after/restoration are checked. I/O follows restoration; failure uses integer-only raw exit. Unsupported optional16-18 excluded; remaining cohorts mandatory.',
        'planned_fenv_cohorts':20,'fenv_native_outcomes_observed_by_this_audit':0,
        'recorded_commands':dict(command_counts),'recorded_command_total':57,'build_elapsed_UTC_seconds':elapsed,'recorded_cleanup_survivors':0,
        'variants':variants,'live_files_hashed':len(hashed),'separate_RUN_source_build_barrier_clear':True,
        'scope':'Standalone integer-helper readiness only; no physical geometry, owned snapshot, caller lifetime, scientific reducer or performance qualification.',
        'audit_native_calls':0,'audit_compiler_calls':0,'audit_loader_calls':0,'audit_source_sha256':sha(Path(__file__).read_bytes()),'elapsed_seconds':time.monotonic()-start}
    out=HERE/'receipt.json';assert not out.exists();out.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':'PASS','receipt':str(out),'sha256':sha(out.read_bytes()),'elapsed':receipt['elapsed_seconds']}))

if __name__=='__main__':main()
