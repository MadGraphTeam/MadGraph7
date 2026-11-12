"""Portable file-only analytic Fortran repair replay; historical controllers are never run."""
import argparse
import builtins
from contextlib import contextmanager
import ctypes
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import types

sys.dont_write_bytecode=True

def require(value,message):
    if not value:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()


@contextmanager
def blocked_apis(memory):
    """Only preloaded logical text reads are available during semantic replay."""
    saved=[];attempts=[]
    def deny(*args,**kwargs):
        attempts.append(repr(args[:1]));raise RuntimeError('Forbidden API during portable replay')
    def memory_bytes(path,*args,**kwargs):
        if str(path) not in memory:return deny(str(path))
        return memory[str(path)]
    def memory_text(path,encoding=None,errors=None,*args,**kwargs):
        return memory_bytes(path).decode(encoding or 'utf-8',errors or 'strict')
    def patch(obj,name,value=deny):
        if hasattr(obj,name):saved.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    try:
        patch(Path,'read_bytes',memory_bytes);patch(Path,'read_text',memory_text)
        for name in ('open','stat','lstat','resolve','exists','is_file','is_dir','is_symlink','iterdir','glob','rglob',
                     'write_bytes','write_text','mkdir','unlink','rename','replace','touch','chmod','symlink_to','hardlink_to'):
            patch(Path,name)
        for name in ('open',):patch(builtins,name)
        for name in ('open','open_code'):patch(io,name)
        for name in ('open','stat','lstat','access','listdir','scandir','readlink','system','popen','posix_spawn','posix_spawnp',
                     'execv','execve','execvp','execvpe','spawnv','spawnve','spawnvp','spawnvpe','fork','forkpty',
                     'mkdir','makedirs','unlink','remove','rename','replace','rmdir','chmod','chown','truncate','utime',
                     'link','symlink','kill','killpg'):
            patch(os,name)
        for name in ('Popen','run','call','check_call','check_output'):patch(subprocess,name)
        for name in ('CDLL','PyDLL','_dlopen'):patch(ctypes,name)
        if '_ctypes' in sys.modules:patch(sys.modules['_ctypes'],'dlopen')
        for name in ('socket','create_connection'):patch(socket,name)
        for name in ('signal','setitimer','alarm','pthread_kill'):patch(signal,name)
        yield attempts
    finally:
        for obj,name,value in reversed(saved):setattr(obj,name,value)


def load_source(name,raw,path):
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__);return module



SOURCE='bac02ef904383e8215e40c6e8306f00b0b3ca46f9ea5233e94b4ac0e76e125cf'
SCIENCE='84e8caf51078c3fb5817582e94e9a6a62f1ad22fc4bf7909f5034fce23dbc31c'
PHASES={
 'build':('79443cd96935e9b0f0768a07870e0d5d5d0805a08a675c67a45aaa7cb2ef3a44','f7f630cb59bac81b7f103dd87e1562c6b6ae98a095cd840471749f59df82ff08'),
 'repair':('e153961127a0d4ee8a5ca4f2345a0ef273b9f18b0e2720dcb4641ff60532a6f9','4991ace1e441e451f3508c6a067de981c3d9dec61aa952c2748f1209d48ae41d'),
 'amplitude':('82446c2e6a2ccc46ca16b42c7b4c44c42cbc04fa21b5115141f1c0d09db8f286','0a4fb37e649b0e38ac56672714ede31da4e10dccbe0777eabc90cc93731e3edf'),
 'timing':('46b5b2fa0fef80b785b11e068c17c73d1716a3e2861add4fe80a5e4ac55187c1','323e8ffaaf4b70a1419c46f31ee305f85bb7ec4701495a8cc87be186dea4f18d')}

def normalized(value):return json.loads(json.dumps(value,sort_keys=True))

def verify(root,expected_index_sha256=None):
    import ast
    import fractions
    import math
    import statistics
    import struct
    from collections import Counter
    root=Path(root).resolve();index_raw=(root/'index.json').read_bytes()
    if expected_index_sha256:require(sha(index_raw)==expected_index_sha256,'Portable index seal changed')
    index=json.loads(index_raw);require(index['schema']=='analytic-Fortran-repair-portable-v1','Wrong capsule schema')
    actual=set()
    for path in root.rglob('*'):
        require(not path.is_symlink(),'Portable symlink refused')
        if path.is_file():actual.add(str(path.relative_to(root)))
    require(actual==set(index['files'])|{'index.json'},'Portable exact inventory changed')
    physical={}
    for name,pin in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe physical path')
        raw=(root/name).read_bytes();require(sha(raw)==pin['sha256'] and len(raw)==pin['bytes'],'Changed physical member: '+name)
        require(not raw.startswith(b'\x7fELF'),'Native binary retained');physical[name]=raw
    blobs={}
    for name,pin in index['logical_files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe logical path')
        raw=physical[pin['path']]
        if pin['compression']=='gzip':
            require(raw[:3]==b'\x1f\x8b\x08' and raw[4:8]==b'\0'*4,'Nondeterministic gzip');raw=gzip.decompress(raw)
        else:require(pin['compression']=='none','Unknown compression')
        require(sha(raw)==pin['raw_sha256'] and len(raw)==pin['raw_bytes'],'Changed logical text: '+name)
        raw.decode('utf-8');require(not raw.startswith(b'\x7fELF'),'Native logical member');blobs[name]=raw
    def data(name,wanted=None):
        require(name in blobs,'Missing logical member: '+name);raw=blobs[name]
        if wanted:require(sha(raw)==wanted,'Changed frozen closure: '+name)
        return raw
    def record(name,wanted=None):return json.loads(data(name,wanted))
    source=record('source/source-index.json',SOURCE)
    require({n[7:] for n in blobs if n.startswith('source/')}==set(source['files'])|{'source-index.json'},'Qualification exact source inventory changed')
    for name,pin in source['files'].items():require(len(data('source/'+name,pin['sha256']))==pin['bytes'],'Qualification source size changed')
    old=record('history/qualification-source1/source-index.json','f41765c3546a3e73561966e9c0dd46e1236edf29178cfb680d2d32c5eb8d361b')
    require({n[len('history/qualification-source1/'):] for n in blobs if n.startswith('history/qualification-source1/')}==set(old['files'])|{'source-index.json'},'Unexecuted qualification source1 inventory changed')
    for name,pin in old['files'].items():data('history/qualification-source1/'+name,pin['sha256'])
    pins=record('source/input-pins.json')
    def external(path):
        name=index['external_text_members'][path];pin=pins['external_files'][path]
        require(len(data(name,pin['sha256']))==pin['bytes'],'External text dependency size changed');return name
    for path in index['external_text_members']:external(path)
    binary_parents={path:pin for path,pin in pins['external_files'].items() if path not in index['external_text_members']}
    require(len(binary_parents)==1 and all(Path(p).suffix=='.o' for p in binary_parents),'Exact native-only parent scope changed')
    for path,pin in binary_parents.items():
        key='parents/'+path.split('/madgraph7_madloop_tirust/',1)[1]
        require(index['excluded_binaries'][key]['sha256']==pin['sha256'] and index['excluded_binaries'][key]['bytes']==pin['bytes'],'Excluded native parent identity changed')
    data('scientific-source2/candidate-improve_ps.f',SCIENCE)
    data('scientific-source2/original-improve_ps.f','d511a2b32b286067c387fc2fc9b1e3d53600f8f7aaf107a53b98e18ec9a53260')
    data('scientific-source2/analytic-axial.patch','54ea102d701354e5d06e2ba06ac6618c3a6670a76fdcf0a8696e7a09bbb08c96')
    repair_code=data('source/analyze.py');amplitude_name=external(pins['analysis']);learning_name=external(pins['collector'])
    profile=record(external(pins['profile_parent']+'/source-index.json'),'97cad155079138419c198599adbbbcc00bb8df0a3ac799b8ef9ec9dd9858bace')
    benchmark=data('parents/tools/tirust_benchmark.py',profile['support']['tools/tirust_benchmark.py'])
    # Execute only the exact authenticated comparator definition, not project
    # imports or a historical benchmark/controller entrypoint.
    functions=[node for node in ast.parse(benchmark).body if isinstance(node,ast.FunctionDef) and node.name=='compare_rows']
    require(len(functions)==1,'Frozen comparator definition missing/duplicate')
    memory={str(root/name):raw for name,raw in blobs.items()}
    with blocked_apis(memory) as attempts:
        repair=load_source('analytic_frozen_repair_parser',repair_code,root/'source/analyze.py')
        amplitude=load_source('analytic_frozen_amplitude_parser',data(amplitude_name),root/amplitude_name)
        learning=load_source('analytic_frozen_learning_parser',data(learning_name),root/learning_name)
        namespace={};exec(compile(ast.Module(body=functions,type_ignores=[]),'contained_compare_rows','exec'),namespace)
        comparator=namespace['compare_rows']
        phases={};terminals={}
        for phase,(termsha,resultsha) in PHASES.items():
            term=record(phase+'/terminal.json',termsha);result=record(phase+'/result.json',resultsha)
            require(term['status']==result['status']=='passed' and term['returncode']==0 and not term['cleanup']['survivors'],'Phase not closed')
            require(term['result_sha256']==resultsha and result['source_index_sha256']==SOURCE,'Phase source/result binding changed')
            cap=600 if phase=='build' else 300
            started=record(phase+'/started.json');limits={'address_space_bytes':17179869184,'file_bytes':536870912,'core_bytes':0}
            require(term['cap_seconds']==cap and term['cleanup_reserve_seconds']==60 and term['deadline_monotonic']==started['deadline_monotonic'] and
                term['resource_limits']==started['resource_limits']==limits and not term['cpu_reservation'],'Original phase cap changed')
            actual=record(phase+'/actual-limits.json')
            require(actual['resource_limits']==limits and actual['pid']==term['pid'] and actual['cpu']==term['cpu']==12,'Actual phase limits changed')
            phases[phase]=result;terminals[phase]=term
        for phase,parents in [('repair',['build']),('amplitude',['build','repair']),('timing',['build','repair','amplitude'])]:
            for parent in parents:require(phases[phase][parent+'_terminal_sha256']==PHASES[parent][0],'Separately reviewed predecessor chain changed')
        built=phases['build'];excluded=index['excluded_binaries']
        require(built['compiler_flags']==pins['flags']==['-w','-fPIC','-ffixed-line-length-132'] and built['compiler_invocations']==2 and built['amplitude_or_model_calls']==0,'BUILD original flags or scope changed')
        commands=sorted(n for n in blobs if n.startswith('build/') and '/commands/' in n and n.endswith('.command.json'))
        require(len(commands)==218,'BUILD command coverage changed')
        compiles=0;links=0
        for name in commands:
            stem=name[:-len('.command.json')];row=record(stem+'.terminal.json');argv=record(name)['argv']
            require(row['returncode']==0 and row['stdout_sha256']==sha(data(stem+'.stdout.txt')),'BUILD command/log failure')
            compiles+=int('-c' in argv);links+=int(any(a.startswith('-Wl,-Map,') for a in argv) and '-c' not in argv)
        require(compiles==2 and links==4,'Actual compiler/link command count changed')
        for role,row in built['variants'].items():
            require(len(row['sources'])==111 and len(row['objects'])==60,'Exact source/object binding count changed')
            for name,wanted in row['objects'].items():require(excluded['build/'+role+'/'+name]['sha256']==wanted,'Excluded numerical object changed')
            for kind,mapname in [('amplitude','private-link.map'),('repair','repair-link.map')]:
                elf=row[kind];require(excluded['build/'+role+'/'+kind+'-ELF']['sha256']==elf['executable_sha256'],'Excluded ELF identity changed')
                data('build/'+role+'/'+mapname,elf['map_sha256']);pre=record('build/'+role+'/'+kind+'-prelink.json',elf['prelink_sha256'])
                require(len(elf['actual_link_loads'])==52 and len(elf['runtime']['files'])==15 and all(pre['bound'].get(n)==h for n,h in
                    {**elf['actual_link_loads'],**elf['runtime']['files']}.items()),'Recorded prebound LOAD/runtime set changed')
        require(excluded['build/common/repair_driver.o']['sha256']==built['driver_object_sha256'],'Standalone driver object identity changed')
        expected_excluded={'build/common/repair_driver.o'}|{'parents/'+p.split('/madgraph7_madloop_tirust/',1)[1] for p in binary_parents}
        for role,row in built['variants'].items():
            expected_excluded|={'build/'+role+'/'+name for name in row['objects']}|{'build/'+role+'/'+kind+'-ELF' for kind in ('amplitude','repair')}
        require(set(excluded)==expected_excluded,'Excluded native identity inventory changed')
        corpus=record('source/repair-corpus.json');oldcorpus=record('history/qualification-source1/repair-corpus.json')
        require(len(corpus['cases'])==136 and Counter(c['path_expected_from_source'] for c in corpus['cases'])==
            {'eligible_physical':106,'local_guard_predicate_refusal':26,'caller_sanity_refusal':4},'Source-label corpus coverage changed')
        zero=next(c for c in corpus['cases'] if c['tag']=='ZERO_POINT');oldzero=next(c for c in oldcorpus['cases'] if c['tag']=='ZERO_POINT')
        require(zero['path_expected_from_source']=='caller_sanity_refusal' and zero['refused'] and oldzero['path_expected_from_source']=='eligible_close_refusal_then_legacy_rescue','Pre-native ZERO_POINT correction changed')
        require(not pins['source_history']['source1_native_stages_executed'],'Source1 must remain unexecuted')
        points=data('inputs/saved-points.dat','2ed64c5cba477ce0e7ed3a7bbd642521061b251573ed20957f8e3bc396fa5b35').decode().splitlines()
        require(int(points[0])==100 and len(points)==301,'Saved physical point set changed')
        for i,case in enumerate(corpus['cases'][:100]):
            vectors=[[float(v.replace('D','E').replace('d','e')) for v in line.split()] for line in points[1+3*i:4+3*i]]
            require(len(vectors)==3 and all(len(v)==4 for v in vectors),'Saved point vector shape changed')
            require([pair[0] for pair in case['words']]==[struct.pack('>d',vectors[leg][mu]).hex().upper() for leg in range(3) for mu in range(4)],'Saved DP fixture-word join changed')
            require(all(repair.number(pair[0])==repair.number(pair[1]) for pair in case['words']),'Saved QP input is not its documented DP lift')
        model=pins['model_initialization'];require(repair.number(model['expected_dp_MZ_word'])==repair.number(model['expected_mp_MZ_word'])>0,'Initialized model53-bit lift changed')
        journal={};diagnostics={}
        for role in ('original','candidate'):
            journal[role]=repair.journal(data('repair/'+role+'/repair-journal.dat').decode(),corpus)
            diagnostics[role]=repair.diagnostics(data('repair/'+role+'/commands/standalone-repair.stdout.txt').decode(),corpus)
            require(normalized(journal[role])==record('repair/'+role+'/parsed-journal.json') and normalized(diagnostics[role])==record('repair/'+role+'/parsed-diagnostics.json'),'Standalone raw replay changed')
        compared=repair.compare(journal['original'],journal['candidate'],diagnostics['original'],diagnostics['candidate'],pins)
        require(normalized(compared)==phases['repair']['comparison'] and compared['DP_and_QP_calls']==544,'Standalone physical/diagnostic/state qualification changed')
        require(all(a['output']==b['output'] for a,b in zip(journal['original']['rows'],journal['candidate']['rows']) if a['precision']==1),'Observed DP repair words changed')
        require(len(compared['word_differences'])==94 and all(r['precision']==2 for r in compared['word_differences']) and
            sum(len(r['components']) for r in compared['word_differences'])==233,'Retained QP word-difference diagnostics changed')
        codes={i:219 for i in range(1,101)};fresh={};map_snapshots=0
        for phase in ('amplitude','timing'):
            fresh[phase]={}
            for role in ('original','candidate'):
                prefix=phase+'/'+role;learned=learning.learning(data(prefix+'/warmup.dat').decode())
                rows=amplitude.exact(data(prefix+'/exact/exact-words.dat').decode(),codes);states=amplitude.ieee(data(prefix+'/exact/exact-ieee.dat').decode())
                require(len(learned['history'])==101 and len(rows)==200 and len(states)==200,'Fresh learning/final100 scope changed')
                info={'learning':normalized(learned),'exact':{'rows':normalized(rows),'ieee':states},'ready':record(prefix+'/ready.json')};fresh[phase][role]=info
                require(amplitude.compare_variants(learned['rows'],rows,comparator)['passed'],'Fresh learning/final coefficient gate changed')
                if phase=='amplitude':require(info==phases['amplitude']['variants'][role],'Amplitude raw/retained result changed')
                else:
                    own=amplitude.compare_own(phases['amplitude']['variants'][role]['exact']['rows'],rows)
                    require(own==record(prefix+'/own-qualification-comparison.json') and own['passed'] and
                        learned['history']==phases['amplitude']['variants'][role]['learning']['history'] and states==phases['amplitude']['variants'][role]['exact']['ieee'],'Timing fresh own-qualification gate changed')
                for edge in ('before','after'):
                    mapped=record(prefix+'/maps-'+edge+'.json');map_snapshots+=1
                    runtime=built['variants'][role]['amplitude'];check=str(Path(built['variants'][role]['output']).parent.parent/phase/role/'SubProcesses/P0_uux_z/check')
                    require(mapped['files']=={**runtime['runtime']['files'],check:runtime['executable_sha256']},'Fresh actual provider/ELF mapping changed')
                    paths={os.path.normpath(line.split(None,5)[5]) for line in mapped['maps'].splitlines() if len(line.split(None,5))==6 and line.split(None,5)[5].startswith('/')}
                    require(paths==set(mapped['files']),'Each raw map snapshot must match its recorded file closure')
                    require(any(p['pid']==mapped['pid'] for p in terminals[phase]['cleanup']['observed_owned_processes']),'Mapped worker missing from recorded ownership closure')
                require(record(prefix+'/maps-before.json')['pid']==record(prefix+'/maps-after.json')['pid'],'Worker PID changed between snapshots')
        a=fresh['amplitude']['original'];c=fresh['amplitude']['candidate']
        final=amplitude.compare_variants(a['exact']['rows'],c['exact']['rows'],comparator)
        learned=amplitude.compare_variants(a['learning']['rows'],c['learning']['rows'],comparator)
        require(final==phases['amplitude']['final_comparison'] and learned==phases['amplitude']['learning_comparison'] and
            final['all_coefficient_words_equal'] and final['all_accuracy_words_equal'] and a['exact']['ieee']==c['exact']['ieee'],'Full100 amplitude parity changed')
        pairs=[];timed_calls=0;batch_rows=0;batch_states=0;accuracy_drift=0;durations=[]
        for i in range(1,8):
            pair=record('timing/pair-'+str(i)+'.json');require(pair==phases['timing']['pairs'][i-1],'Retained pair/result join changed')
            require(pair['order']==(['original','candidate'] if i%2 else ['candidate','original']),'Alternating timing order changed')
            for role in ('original','candidate'):
                prefix='timing/'+role+'/pair-'+str(i);entry=pair[role]
                rows=amplitude.exact(data(prefix+'/timed-exact-words.dat').decode(),codes)
                require(amplitude.decimal_exact(data(prefix+'/values.dat').decode(),codes)==rows==entry['rows'],'Batch-final100 decimal/raw/retained join changed')
                own=amplitude.compare_own(fresh['timing'][role]['exact']['rows'],rows)
                require(own==entry['own_comparison']==record(prefix+'/own-comparison.json') and own['passed'],'Own timed coefficient/status gate changed')
                states=amplitude.ieee(data(prefix+'/timed-ieee.dat').decode(),[0]);reference=fresh['timing'][role]['exact']['ieee']
                require(states==entry['ieee'] and states[0][2:]==reference[0][2:] and states[1][2:]==reference[-1][2:],'Observed whole-batch pre/post state changed')
                counts=amplitude.status_histogram(data(prefix+'/timed_status.dat').decode(),codes,128)
                require(counts==entry['timed_status_counts']=={'219':12800},'All timed accepted-status counts changed')
                fields=data(prefix+'/timing.dat').decode().split();require(len(fields)==2,'Raw timing format changed')
                cold,seconds=map(float,fields)
                require(seconds==entry['seconds'] and seconds>1 and cold==phases['timing']['initialization'][role]['cold_seconds'],'Raw batch/cold timer join changed')
                timed_calls+=sum(counts.values());batch_rows+=len(rows);batch_states+=len(states);accuracy_drift+=own['accuracy_drift_count'];durations.append(seconds)
            require(amplitude.compare_variants(pair['original']['rows'],pair['candidate']['rows'],comparator)==pair['comparison'],'Timed retained cross-variant comparison changed')
            pairs.append(pair)
        summary=amplitude.timing_summary(pairs)
        require(summary==phases['timing']['summary'] and timed_calls==179200 and batch_rows==2800 and batch_states==28 and map_snapshots==8,'Descriptive timing/result totals changed')
        require(not summary['performance_acceptance'] and not phases['timing']['candidate_promoted'],'Formal runtime acceptance/promotion must remain false')
        reviews={
          'fortran-orig-axial-source-independent-review-1':'fb7cc9ca7e34b2890d3239a660bc894119bd03b128aeb132d79407096676059b',
          'fortran-orig-axial-source-independent-review-2':'484846320a17c94c12608608ba1b908543011c3ed4a91839c6aaf92dc072a88b',
          'fortran-orig-axial-qualification-source-independent-review-1':'f45b590a445f67d0d963cac271642410973851bc072d96b5e2ec7cebc7894095',
          'fortran-orig-axial-qualification-source-independent-review-2':'a1dc8012a8b4fec3aa9101865d1ffeba36d07248f92ed0a1d1e35e97391f4870',
          'fortran-orig-axial-qualification-build-independent-review-2':'80e5514a8b6e1a5df96a37bd27d49e3ae939ffce0d0c7c2487261e57d83aad8a',
          'fortran-orig-axial-qualification-repair-independent-review-2':'c6e47819bca4ebf4f013cee10d050641c1f5449b4ad3579a893630932306a23c',
          'fortran-orig-axial-qualification-amplitude-independent-review-2':'81535b9f4a09c18fa3e99d7233acae62943c747b8d0c0dcaacb7e300691e26ab',
          'fortran-orig-axial-qualification-timing-independent-review-2':'9e10f47d4599235efef6fb7908c5a710282a9ffbb89cbe53793e0e2cfa780306'}
        require(index['review_receipt_sha256']==reviews,'Independent review inventory/pins changed')
        for name,wanted in reviews.items():record('reviews/'+name+'/receipt.json',wanted)
        timing_review=record('reviews/fortran-orig-axial-qualification-timing-independent-review-2/receipt.json','9e10f47d4599235efef6fb7908c5a710282a9ffbb89cbe53793e0e2cfa780306')
        data('reviews/fortran-orig-axial-qualification-timing-independent-review-2/report.md',timing_review['report_sha256'])
        require(timing_review['summary']==summary and not timing_review['coefficients_observed_at_all_timed_calls'],'Independent timing scope changed')
        data('owner/fortran-orig-axial-qualification-timing-owner-readout-2.json','485c86738d7ba1bd7213abd2f74d455c17ed4cd4fe5c4de4f4d82e4f150f2d21')
        require(data('owner/fortran-orig-axial-qualification-timing-owner-readout-2.empty-owner-write')==b'','Empty administrative artifact changed')
        facts={'standalone_mains':2,'source_label_cases':136,'outer_repair_wrapper_invocations':544,'standalone_IEEE_snapshots_per_arm':545,
            'source_label_counts':{'eligible_physical':106,'local_guard_predicate_refusal':26,'caller_sanity_refusal':4},
            'DP_output_word_differences':0,'QP_fixture_results_with_word_differences':94,'QP_component_word_differences':233,
            'physical_DP_residual_limit':'1e-12','physical_QP_residual_limit':'1e-25','model_MP_mass_is_binary64_lift':True,
            'source1_corrections_unexecuted':True,'compiler_flags_unchanged':True,'compile_commands':2,'link_commands':4,'build_commands':218,
            'full100_amplitude_returned_rows':200,'full100_amplitude_coefficient_words':800,'full100_amplitude_ACC_words':200,
            'fresh_workers_amplitude':2,'fresh_workers_timing':2,'learning_calls_per_fresh_worker':101,'final_exact_points_per_fresh_worker':100,
            'timing_pairs':7,'timing_batches':14,'timed_accepted219_calls':timed_calls,'timed_coefficients_recorded_for_every_call':False,
            'retained_batch_final100_sets':14,'retained_coefficient_rows_per_batch':200,'retained_coefficient_words_per_batch':800,
            'retained_ACC_words_per_batch':200,'retained_accuracy_drift_rows':accuracy_drift,'timing_whole_batch_IEEE_snapshots':28,
            'paired_median_candidate_over_original':summary['median_ratio'],'minimum_batch_seconds':min(durations),'maximum_batch_seconds':max(durations),
            'empty_administrative_write_retained':True,'native_binaries_packaged':0,'new_native_executions':0,'excluded_native_identities':len(excluded),
            'actual_fast_branch_coverage_instrumented':False,'massive_incoming_model_qualified':False,'software_latch_observed':False,'trap_freedom_qualified':False,
            'independent_full_amplitude_truth':False,'formal_9228_default_chain_eight_process_acceptance':False,'local_runtime_acceptance':False,'production_promotion':False}
        require(index['claims']==facts,'Published claims differ from actual closed evidence');require(not attempts,'Forbidden portable replay API attempted')
    return {**facts,'verified':True,'forbidden_API_attempts':0}

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--index-sha256');args=cli.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent,args.index_sha256),sort_keys=True,indent=2))
