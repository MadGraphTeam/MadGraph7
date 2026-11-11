#!/usr/bin/env python3
"""Replay closed recording-provider parser tests using indexed text reads only."""
import argparse
import ast
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import signal
import types

SOURCE_PINS={
 'proposal':('index.json','600fb55cae10c8f105998c6ae689c09992526765734f33036dd79936261f5150'),
 'source2':('source-index.json','ad63b2e1128c70085fb282b1a74e1b84aa282c7e5be95514a79a558845817fe4'),
 'source3':('source-index.json','511225ea5bb4523cbf84012ea424a0710ef6f86b3ec6964ba8e8b41ea08a76d6'),
 'source4':('source-index.json','9d159331439f106df0a7571c69633c40b5900cb5c4162d6f9850521a3db8f219'),
}
TERMINAL_PINS={
 'build2':'874557356d762b3b51fa5bdb7f33df23375d14a46f6ea38f4a6b991e972de64c',
 'build3':'d4eb1386ce923a16d70666637ff3677091449b30a812b6ebcb90b421c0b3cf9f',
 'run3':'4e70b14200b037f0acf05723f05a29078f26e95bf2871b3be4fc97a4421a815f',
 'run4':'272399199071f0ab510c32ff830a41f0e2db0a3a1e8c44cd5d464a67a8cc1de0',
}


def verify(root=None, *, read_bytes=None, derive_summary=False, expected_index_sha256=None):
    def require(value,message):
        if not value:raise ValueError(message)
    def sha(raw):return hashlib.sha256(raw).hexdigest()
    def identity(raw):return {'bytes':len(raw),'sha256':sha(raw)}
    def safe(name):
        require(isinstance(name,str) and name and '\\' not in name,'Invalid evidence name')
        p=PurePosixPath(name)
        require(not p.is_absolute() and '..' not in p.parts and str(p)==name,'Unsafe relative evidence name')
        return name
    local=read_bytes is None
    if local:
        root=Path(root or Path(__file__).parent).absolute()
        require(all(not p.is_symlink() for p in (root,*root.parents)),'Symlink evidence root')
        def local_read(name):
            safe(name);path=root
            for part in PurePosixPath(name).parts:
                path/=part;require(not path.is_symlink(),'Symlink evidence path')
            require(path.is_file(),'Missing evidence file')
            return path.read_bytes()
        read_bytes=local_read
    index_raw=read_bytes('evidence-index.json')
    if expected_index_sha256 is not None:require(sha(index_raw)==expected_index_sha256,'Different evidence index')
    index=json.loads(index_raw);require(index.get('schema_version')==1 and index.get('kind')=='madloop-tirust-bulk-unit-qualification-v1','Wrong packet schema')
    files=index['files'];packed=index.get('compressed_text',{})
    for name in files:
        safe(name);require(name!='evidence-index.json','Self-indexed payload')
    for name,item in packed.items():safe(name);safe(item['path'])
    packed_paths=[x['path'] for x in packed.values()]
    require(len(set(packed_paths))==len(packed_paths),'Aliased packed storage')
    require(not set(packed)&set(files),'Raw/compressed logical collision')
    def physical(name):
        safe(name);require(name in files and files[name]['kind']=='file','Unindexed evidence read')
        raw=read_bytes(name);meta=files[name]
        require(len(raw)==meta['size'] and sha(raw)==meta['sha256'],'Changed evidence payload: '+name)
        return raw
    def read(name):
        safe(name)
        if name in packed:
            item=packed[name];raw=gzip.decompress(physical(item['path']))
            require(len(raw)==item['raw_bytes'] and sha(raw)==item['raw_sha256'],'Changed compressed text')
        else:raw=physical(name)
        require(b'\0' not in raw,'Binary bytes in logical text evidence');raw.decode('utf-8')
        return raw
    def invalid(value):raise ValueError('Nonfinite JSON '+value)
    def load(name):return json.loads(read(name),parse_constant=invalid)
    logical=(set(files)-set(packed_paths))|set(packed)
    for name in files:physical(name)
    for name in logical:read(name)
    if local:
        paths=list(root.rglob('*'));require(all(not p.is_symlink() for p in paths),'Symlink packet member')
        require({str(p.relative_to(root)) for p in paths if p.is_file()}==set(files)|{'evidence-index.json'},'Packet inventory changed')
    retention=load('retention.json');require(retention['schema']=='madloop-tirust-bulk-unit-retention-v1','Wrong retention owner')
    groups=retention['groups']
    for group,row in groups.items():
        expected={group+'/'+safe(name) for name in row['files']}
        require({name for name in logical if name.startswith(group+'/')}==expected,'Retained group inventory differs: '+group)
        for name,want in row['files'].items():require(identity(read(group+'/'+name))==want,'Retained original bytes differ')
        for name,want in row['binary_hashes'].items():
            safe(name);require(group+'/'+name not in logical and set(want)=={'bytes','sha256'},'Binary retained as text')
            require(name.endswith(('.o','.mod')) or PurePosixPath(name).name=='check','Unknown omitted binary')
    for group,(index_name,pin) in SOURCE_PINS.items():
        require(sha(read(group+'/'+index_name))==pin,'Scientific/source seal differs: '+group)
        seal=load(group+'/'+index_name)
        for name,want in seal['files'].items():require(identity(read(group+'/'+safe(name)))==want,'Sealed source payload differs')
    seven=['provider.c','driver.f90','corpus.py','corpus.json','analyze.py','observer-mtirust.f90','forced-mtirust.f90']
    for name in seven:require(read('source2/'+name)==read('source3/'+name)==read('source4/'+name),'Scientific file changed across continuation')
    for slot in ('original','candidate'):
        raw=read('proposal/'+slot+'-mtirust.f90')
        require(identity(raw)==load('proposal/index.json')['files'][slot+'-mtirust.f90'],'Compared module differs')
    pins=load('source4/input-pins.json')['files']
    require(sha(read('production/bridge-lib.rs'))==pins['vendor/tirust/src/lib.rs'] and
            sha(read('production/mtirust.f90'))==pins['vendor/tirust/mtirust.f90'],'Native rejection basis differs')
    bridge=read('production/bridge-lib.rs').decode()
    require('if finite == 0 && poles == 3 {' in bridge and 'fn guard(' in bridge and
            'return Err("pole-only evaluation requires at least one pole origin".into());' in bridge,'Native error source basis absent')
    stages={}
    for group,pin in TERMINAL_PINS.items():
        require(sha(read(group+'/terminal.json'))==pin,'Closed stage terminal differs')
        t=load(group+'/terminal.json');stages[group]=t
        require(not t['cleanup']['survivors'] and t['cleanup_reserve_seconds']==60.,'Unclosed original stage')
        wanted='passed' if group in ('build3','run4') else 'failed'
        require(t['status']==wanted and t['returncode']==(0 if wanted=='passed' else 1),'Historical failure/success changed')
        require(t['cap_seconds']==(180. if group.startswith('build') else 600.),'Stage cap differs')
        limits=load(group+'/limits.json')['actual']
        require(limits=={'address_space_bytes':[16*1024**3]*2,'file_bytes':[512*1024**2]*2,'core_bytes':[0,0]},'Actual child limits differ')
        admission=load(group+'/admission.json')
        require(admission['command']==t['command'] and t['command'][1:3]==['-c','12'],'Stage command/CPU admission differs')
        source_group='source2' if group=='build2' else 'source4' if group=='run4' else 'source3'
        require(admission['source_index_sha256']==SOURCE_PINS[source_group][1],'Stage scientific source admission differs')
        if group.startswith('run'):require(admission['reviewed_build_sha256']==TERMINAL_PINS['build3'],'RUN uses a different build')
        if group=='run4':require(admission['reviewed_run_sha256']==TERMINAL_PINS['run3'],'Continuation failed-RUN ancestry differs')
    for group,count in [('build2',27),('build3',107)]:
        commands=sorted(name for name in logical if name.startswith(group+'/commands/') and name.endswith('.terminal.json'))
        require(len(commands)==count,'Closed compiler/query command count differs')
        for name in commands:
            term=load(name);command=load(name.replace('.terminal.json','.command.json'))
            require(term['returncode']==0 and not term.get('timed_out'),'Failed original compiler/query command')
            require(all(term.get(k)==v for k,v in command.items()),'Compiler command/terminal join differs')
            require(sha(read(name.replace('.terminal.json','.stdout.txt')))==term['stdout_sha256'],'Compiler log differs')
    build=load('build3/result.json');require(sha(read('build3/result.json'))==stages['build3']['result_sha256'],'Successful build result differs')
    for name,checksum in build['objects'].items():require(groups['build3']['binary_hashes'][name]['sha256']==checksum,'Provider object hash differs')
    require(build['source_index_sha256']==SOURCE_PINS['source3'][1] and set(build['executables'])=={'original','candidate','observer','forced'},'Wrong compiled variants')
    resolution=load('link-runtime-resolution.json')
    allowed={'ld-linux-x86-64.so.2','libgfortran.so.5','libm.so.6','libgcc_s.so.1','libc.so.6'}
    for variant,row in build['executables'].items():
        base='build3/'+variant+'/'
        binaries=groups['build3']['binary_hashes']
        require(binaries[variant+'/check']['sha256']==row['sha256'],'Executable hash proof differs')
        for name,checksum in row['objects'].items():require(binaries[variant+'/'+name]['sha256']==checksum,'Compiled object hash differs')
        module='proposal/'+variant+'-mtirust.f90' if variant in ('original','candidate') else 'source4/'+variant+'-mtirust.f90'
        require(row['sources']=={'module':sha(read(module)),'driver':sha(read('source4/driver.f90')),'provider':sha(read('source4/provider.c'))},'Compiled source identity differs')
        symbols=read('build3/commands/'+variant+'-symbols.stdout.txt').decode()
        names=re.findall(r'^[0-9a-fA-F]+\s+[A-Za-z]\s+(\S+)$',symbols,re.M)
        require(names.count('ml_tirust_evaluate_mp')==1 and names.count('__mtirust_MOD_tensor_mp')==1,'Ambiguous public provider/module symbols')
        pre=load(base+'prelink.json');proof=load(base+'unit-runtime-prequery.json');runtime=row['runtime']
        require(sha(read(base+'prelink.json'))==row['prelink_sha256'] and sha(read(base+'link.map'))==row['link_map_sha256'] and
                sha(read(base+'unit-runtime-prequery.json'))==row['unit_runtime_derivation_sha256'],'Closed link/runtime proof changed')
        aliases=resolution[variant]['map_aliases'];spellings=re.findall(r'^LOAD\s+(.+)$',read(base+'link.map').decode(),re.M)
        require(spellings==[a['spelling'] for a in aliases],'Actual linker LOAD spellings differ')
        require({a['resolved']:a['sha256'] for a in aliases}==row['actual_link_loads'],'Actual LOAD resolution differs')
        for a in aliases:require(pre['bound'].get(a['resolved'])==a['sha256'],'Actual linker input was not prebound')
        require(proof['files']==runtime['files'] and len(runtime['files'])==5 and set(proof['sonames'])==allowed,'Unit runtime closure differs')
        require(proof['actual_interpreter_query_still_required'] is True and proof['compiler_or_loader_invocations']==0,'Static derivation scope differs')
        for name,checksum in runtime['files'].items():require(pre['bound'].get(name)==checksum,'Runtime provider not prebound')
        query='build3/commands/'+variant+'-actual-pt-interp-runtime.'
        term=load(query+'terminal.json');require(term['argv']==[runtime['loader'],'--list',row['path']] and term['returncode']==0,'Wrong actual interpreter query')
        listing=read(query+'stdout.txt').decode();require(listing==runtime['listing'],'Actual runtime listing differs')
        runtime_spellings=[]
        for line in listing.splitlines():
            line=line.strip()
            if not line or re.fullmatch(r'linux-vdso\.so\.1\s+\(0x[0-9a-fA-F]+\)',line):continue
            match=re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s+\(0x[0-9a-fA-F]+\)',line)
            require(match is not None,'Malformed actual runtime listing');runtime_spellings.append(match[1])
        aliases=resolution[variant]['runtime_aliases']
        require(runtime_spellings==[a['spelling'] for a in aliases] and {a['resolved']:a['sha256'] for a in aliases}==runtime['files'],'Runtime alias proof differs')
        objects=proof['objects'];root_obj=next(x for x in objects if x['path']==row['path'])
        require(root_obj['sha256']==row['sha256'] and root_obj['interpreter']==runtime['loader'],'ELF/PT_INTERP recorded proof differs')
        for obj in objects:
            require(not obj['rpath'] and obj['needed']==[x for x in obj['needed'] if x in allowed],'Unsupported recorded dynamic dependency')
            if obj is not root_obj:require(runtime['files'].get(obj['path'])==obj['sha256'] and proof['sonames'].get(obj['soname'])==obj['path'],'Recorded dependency provider differs')
        require(Counter((a['requester'],a['needed']) for a in proof['resolutions'])==Counter((obj['path'],need) for obj in objects for need in obj['needed']),'Dependency resolution coverage differs')
        for route in proof['resolutions']:require(route['selected']==proof['sonames'][route['needed']],'Dependency resolution target differs')
    # Execute only indexed, exact-pinned pure parser/corpus functions. The
    # analyzer's original importlib/Path bootstrap is deliberately not executed.
    corpus_ns={'__name__':'portable_bulk_corpus'}
    exec(compile(read('source4/corpus.py'),'indexed source4/corpus.py','exec'),corpus_ns)
    oracle=types.SimpleNamespace(**corpus_ns)
    cases=load('source4/corpus.json')['cases'];require(oracle.build_corpus()==load('source4/corpus.json'),'Frozen corpus differs')
    def analyzer(correct):
        raw=read('source4/analyze.py');text=raw.decode()
        require(sha(raw)=='a84cee0c224b024419d3496ac28f6f9392f3359bdce3501133b00131d2c43cb8','Original analyzer differs')
        if correct:
            replacements=[('status=3 if unsupported else 2 if bad else 0','status=2 if unsupported else 2 if bad else 0'),
                          ("error==(ERROR if status==2 else 'mock native recording provider')","error==(ERROR if status==2 and not unsupported else 'mock native recording provider')")]
            overlay=read('source4/analyze_rejection.py').decode()
            for old,new in replacements:
                require(text.count(old)==1 and old in overlay and new in overlay,'Oracle correction scope differs');text=text.replace(old,new)
        tree=ast.parse(text);tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ERROR' for t in n.targets)]
        ns={'oracle':oracle,'signal':signal};exec(compile(tree,'indexed exact parser oracle','exec'),ns)
        return types.SimpleNamespace(**ns)
    original=analyzer(False);corrected=analyzer(True)
    continuation=load('source4/continuation-inputs.json');prefix=continuation['prefix_case_ids'];pending=continuation['pending_case_ids']
    end=next(i for i,c in enumerate(cases) if c['id']=='options-f0-p3')+1
    require(prefix==[c['id'] for c in cases[:end] if c['execution']!='lexical-only'] and len(prefix)==88,'Retained prefix differs')
    require(pending==[c['id'] for c in cases[end:] if c['execution']!='lexical-only'] and len(pending)==47 and not set(prefix)&set(pending),'Suffix overlaps/gaps')
    lexical=[c['id'] for c in cases if c['execution']=='lexical-only'];require(len(lexical)==14 and lexical==continuation['lexical_only_case_ids'],'Lexical-only coverage differs')
    for group,expected in [('run3',prefix),('run4',pending)]:
        actual={name.split('/')[1] for name in logical if name.startswith(group+'/') and name.endswith('/case.txt')}
        require(actual==set(expected),'Missing/duplicate/unexpected executed case')
    observations=[];retained=[];fresh=[];byid={c['id']:c for c in cases};native_returns=Counter();abi_calls=post_calls=word_records=0
    markers=Counter();old_fail_observed=False;trap_details={}
    for group,names in [('run3',prefix),('run4',pending)]:
        origin=groups[group]['origin']
        for name in names:
            case=byid[name];base=group+'/'+name+'/'
            fields=['id','n','rank','poles','finite','check','cache','clear_fails','rounding','sticky_mask','halt_flag']
            expected=(' '.join(str(case[k]) for k in fields)+'\n'+'\n'.join(oracle.records(case))+'\n').encode()
            require(read(base+'case.txt')==expected,'Executed case input differs from frozen corpus')
            variants=['forced'] if case['execution']=='forced-hook' else ['original','candidate','observer'];pair={}
            for variant in variants:
                row=build['executables'][variant];command=load(base+variant+'.command.json');term=load(base+variant+'.terminal.json')
                require(command['argv']==[row['path'],origin+'/'+name+'/case.txt'] and command['cwd']==origin+'/'+name and
                        command['timeout_seconds']==10. and command['subprocess_new_session'] is False,'Executed command/input/build join differs')
                require(all(term.get(k)==v for k,v in command.items()) and not term.get('timed_out'),'Incomplete command terminal')
                raw=read(base+variant+'.stdout.txt');require(sha(raw)==term['stdout_sha256'],'Raw output hash differs')
                require(term['returncode']==0 or case['halt_flag'] and term['returncode']==-signal.SIGFPE,'Unexpected exit/signal')
                native_returns[term['returncode']]+=1;pair[variant]={'text':raw.decode(),'terminal':term}
                rows,seen=corrected.parse(raw.decode());abi_calls+=seen.count('ABI_ENTER');post_calls+=len(rows.get('POST',[]));word_records+=len(rows.get('WORD',[]))
                if variant=='observer':markers.update(seen)
                if term['returncode']!=0:trap_details[variant]={'markers':seen,'PRE':len(rows.get('PRE',[])),'POST':len(rows.get('POST',[]))}
            result=corrected.compare(case,pair);observations.append(result)
            if group=='run3':
                retained.append(result)
                if name==prefix[-1]:
                    require(base+'comparison.json' not in logical,'Failed original case rewritten')
                    try:original.compare(case,pair)
                    except ValueError as error:require('Public status/round/halting mismatch' in str(error),'Original failure reason differs');old_fail_observed=True
                    else:raise ValueError('Original oracle failure disappeared')
                else:require(load(base+'comparison.json')==result,'Earlier passing comparison changed')
            else:fresh.append(result);require(load(base+'comparison.json')==result,'New case comparison differs')
    saved_replay=load('run4/retained-replay.json');saved_new=load('run4/result.json')
    require(saved_replay['replayed_cases']==retained and saved_replay['old_failed_run_status_preserved']=='failed' and old_fail_observed,'Retained replay ledger differs')
    require(saved_new['cases']==fresh and saved_new['case_count']==47 and saved_new['retained_cases']==88 and saved_new['combined_executed_cases']==135,'Continuation result differs')
    require(sha(read('run4/result.json'))==stages['run4']['result_sha256'] and sha(read('run4/retained-replay.json'))==saved_new['retained_replay_sha256'],'New stage result/replay binding differs')
    require(len(observations)==135 and len({x['case'] for x in observations})==135 and sum(native_returns.values())==401,'Total executed coverage differs')
    trap=[x for x in observations if any(v!=0 for v in x['returncodes'].values())]
    require(len(trap)==1 and trap[0]['case']=='ieee-halt5-inexact' and set(trap[0]['returncodes'].values())=={-8} and
            trap[0]['halting_output_phase_observed'],'Expected preserved IEEE trap differs')
    require(set(trap_details)=={'original','candidate','observer'},'Trap variant coverage differs')
    for variant,row in trap_details.items():
        expected=['ABI_ENTER','SCALAR_ENTER']*3 if variant=='observer' else ['ABI_ENTER']*3
        require(row=={'markers':expected,'PRE':3,'POST':2},'Trap did not enter third output decode after two returned calls')
    branch={v:{k:sum(o['observations'].get(v,{}).get(k,0) for o in observations) for k in
               ['public_calls','binary128_word_records','bulk_calls','scalar_calls','forced_calls']}
            for v in ['original','candidate','observer','forced']}
    require(branch['observer']['bulk_calls']==104 and branch['observer']['scalar_calls']==551 and branch['forced']['forced_calls']==4,'Actual decoder coverage differs')
    diagnostics=[{'case':o['case'],'variant':v,'difference_records':len(row['non_nearest_directional_reference_differences'])}
                 for o in observations for v,row in o['observations'].items() if row.get('non_nearest_directional_reference_differences')]
    summary={'schema':'madloop-tirust-bulk-unit-portable-summary-v1','status':'pass',
             'corpus_cases':149,'retained_replayed_cases':88,'new_executed_cases':47,'lexical_only_cases':14,
             'executed_distinct_cases':135,'native_unit_invocations':401,'retained_invocations':264,'new_invocations':137,
             'actual_native_return_counts':{str(k):v for k,v in sorted(native_returns.items())},
             'observed_mock_ABI_entries':abi_calls,'returned_public_calls':post_calls,'emitted_binary128_word_records':word_records,
             'completed_case_branch_counts':branch,'observer_all_call_markers':dict(sorted(markers.items())),
             'preserved_trap_case':'ieee-halt5-inexact','old_build2_status':'failed','old_run3_status':'failed',
             'non_nearest_directional_reference_diagnostics':diagnostics,
             'non_nearest_scope':'Diagnostic only; default internal-file READ ROUND= semantics unqualified; all-mode exact differential words and IEEE gates remain hard',
             'candidate_promoted':False,'physical_amplitude_calls':0,'actual_tirust_reducer_calls':0,
             'compiler_or_native_executions_by_verifier':0,'measured_benefit':None,'performance_acceptance':False,
             'remaining':'Real wire corpus, unchanged-full100-learning amplitude parity and measured benefit; final precise acceptance local to Hua-Sheng'}
    if not derive_summary:require(load('qualification-summary.json')==summary,'Saved qualification summary differs')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',nargs='?',default=str(Path(__file__).parent));p.add_argument('--index-sha256',required=True)
    a=p.parse_args();print(json.dumps(verify(a.root,expected_index_sha256=a.index_sha256),sort_keys=True,indent=2))


if __name__=='__main__':main()
