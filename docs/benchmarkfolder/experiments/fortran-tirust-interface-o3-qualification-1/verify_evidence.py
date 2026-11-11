"""Pure portable replay. Recorded absolute paths are identity strings only."""
import argparse
from collections import Counter
from collections.abc import Mapping
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import struct
import types

SOURCE1 = '841c90f5a2aa49c74e42fd1a64d56ffe659008d4f80d4272fd348a7f9579657e'
SOURCE2 = '83721ec7d56313807d5ba2a282411b2ea2623ed0810158bed8a0b0b108afcb0b'
PARSER = '60bd375b02950e33b4e15851a5359d86b1d9308180eff454725f164b01606adb'
WARMUP_PARSER = 'a85ea55dc9f432e533189369c96ce0fc503fc859fe047085ad1d57de2b4ef1fe'
DELIVERY = 'efc5e18329e95fe12ea6ddaf3b0dc0b072782babcd3bfd535150edbabc398a5c'
STAGES = {
 'build':('29dabbb9f0ab7f8e8eebdba6e56dca095b6fcc630dd561cd90f81d7accd56502','2494684a388bdfa0c0bb481d1eccd6b9a28f197551606ac1dec16d5283d2e734'),
 'n1parity':('558849b4ce6eb99da75c36d613c9828b4a5b96ecbec88e35109ee1a4dab69a1f','d87ce9312707595382a27846f694031c3cde9086c2b23d23420fcbd87cdbd8e9'),
 'n1timing':('d10ecb2613cd406fc72ee11dea79ddc1ff48a5d93bdb9bdcca28c9cbda5a3fd1','98a13096cbfb1d68607b5b81f387d8521d2cf9a1bad471a2b2c73a67c598f1e5'),
 'n3parity':('af82c067c35dbc4895b974060492cf17365a5e962adf33f10e2468c5a44b71dc','7a4c65ae89036fe97940eda3d63a5d4a4bf924c16b94d61e6c9b24b361f89372'),
 'n3anchortiming':('ddabc42a0bb1368a94103c21bee2b387c872e1c55740fd5b2dd2120a201d224e','c5585510d4128f15562a36bbdcf8fa96f73d19a34e96cf89a53d7040c82b7d8e')}
AUDITS = {'build':'cb7c42ef8db9195fdfa06af9b212db6d459bc767e2ac3f4d083a0491c1195b08',
 'n1parity':'ba160b095927b08ed639a61ef6cfb037f1ac89d52f1c36113fd8e4f5eedbce6c',
 'n1timing':'212c4eeb32aba1eac0dfdc6caf696ae99bb9e05a869b381dce1fdaca51d8dfcb',
 'n3parity':'1207ad501fe4f12c8792eb08bb0e5d415a396a0c403c6f3c3b0cabfb44192e36',
 'n3anchortiming':'88b0aeed3a40f5accb44180a2a9bd3d5b1e1c85a05a48d3f560c4627c12caafc'}
PRIMARY = {'build':'631911b7a6ef41834e41ea4254b452379883c5b163084219b9c8f862b5d0817b',
 'n1parity':'3ad7b6166417a7507ab4eeb8d8946d1a0e20772f19b32c343354d987e6d264fe',
 'n1timing':'472402b73c8754dcfb729ef85682aaef59e353cecf091c2d40344b9016aad5df',
 'n3anchor':'e2573b5e0fcc21c4dc8bc6d151b41b6930affdb5dff223e8e88c375a7e4182b8'}
ROLES=('original','candidate')


def replay_own_reports(reference, previous, actual, stored_reports, analysis):
    """Exact coefficient/status gates, with finite accuracy drift diagnostics."""
    reports={'qualified_parity':analysis.compare_own(reference,actual),
             'previous_batch':analysis.compare_own(previous,actual)}
    if json.loads(json.dumps(reports)) != stored_reports:
        raise ValueError('Own-reference diagnostic reports disagree with raw words')
    if not all(row['passed'] for row in reports.values()):
        raise ValueError('Own-reference coefficient/status gate failed')
    return reports


def verify(root=None, *, read_bytes=None, derive_summary=False, expected_index_sha256=None):
    def require(value, message):
        if not value:
            raise ValueError(message)
    def sha(raw):
        return hashlib.sha256(raw).hexdigest()
    def safe(name):
        require(isinstance(name, str) and name and '\\' not in name, 'Invalid capsule name')
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts and str(path) == name,
                'Unsafe capsule-relative name')
        require('__pycache__' not in path.parts, 'Bytecode is not text evidence')
        return name
    def decode(raw):
        def invalid(value):
            raise ValueError('Nonfinite JSON ' + value)
        def pairs(items):
            result = {}
            for key, value in items:
                require(key not in result, 'Duplicate JSON key')
                result[key] = value
            return result
        return json.loads(raw, parse_constant=invalid, object_pairs_hook=pairs)
    local = read_bytes is None
    if local:
        root = Path(root or Path(__file__).parent).absolute()
        require(all(not p.is_symlink() for p in (root, *root.parents)), 'Symlink capsule root')
        def read_local(name):
            safe(name)
            path = root
            for part in PurePosixPath(name).parts:
                path /= part
                require(not path.is_symlink(), 'Symlink capsule member')
            require(path.is_file(), 'Missing capsule member')
            return path.read_bytes()
        read_bytes = read_local
    raw_index = read_bytes('evidence-index.json')
    if expected_index_sha256 is not None:
        require(sha(raw_index) == expected_index_sha256, 'Different capsule index')
    index = decode(raw_index)
    require(index['schema'] == 'o3-portable-experiment-evidence-v1', 'Wrong capsule schema')
    require(all(index[name] is False for name in ('native_bytes_retained', 'performance_acceptance',
                                                 'candidate_promoted', 'optimization_benefit_claimed')), 'Scope changed')
    files, payloads = index['files'], index['payloads']
    for name, row in files.items():
        safe(name)
        require(name != 'evidence-index.json' and set(row) == {'sha256', 'bytes'}, 'Self/invalid physical index')
        require(type(row['bytes']) is int and 0 <= row['bytes'] <= 32 * 1024**2 and
                re.fullmatch('[0-9a-f]{64}', row['sha256']), 'Invalid physical identity')
    physical_names = []
    for name, row in payloads.items():
        safe(name)
        safe(row['path'])
        require(name != 'evidence-index.json' and row['path'] in files and
                set(row) == {'path', 'sha256', 'bytes', 'encoding'}, 'Unindexed logical payload')
        require(row['encoding'] in ('utf8', 'gzip+utf8') and type(row['bytes']) is int and
                0 <= row['bytes'] <= 64 * 1024**2 and re.fullmatch('[0-9a-f]{64}', row['sha256']),
                'Invalid decoded identity')
        require(PurePosixPath(name).suffix.lower() not in ('.o', '.mod', '.a', '.so', '.dll', '.dylib', '.exe'),
                'Native file suffix')
        physical_names.append(row['path'])
    require(len(set(physical_names)) == len(physical_names) and set(physical_names) == set(files),
            'Aliased, extra or missing physical payload')
    require(sum(row['bytes'] for row in payloads.values()) <= 512 * 1024**2, 'Decoded capsule too large')
    if local:
        paths = list(root.rglob('*'))
        require(all(not p.is_symlink() for p in paths), 'Symlink inventory')
        require({str(p.relative_to(root)) for p in paths if p.is_file()} == set(files) | {'evidence-index.json'},
                'Physical capsule inventory differs')
    else:
        require(hasattr(read_bytes, 'paths') and set(read_bytes.paths) == set(files) | {'evidence-index.json'},
                'Callback physical inventory differs')
    decoded = {}
    for name, row in payloads.items():
        raw = read_bytes(row['path'])
        require(len(raw) == files[row['path']]['bytes'] and sha(raw) == files[row['path']]['sha256'],
                'Changed stored payload: ' + name)
        if row['encoding'] == 'gzip+utf8':
            require(raw[:4] == b'\x1f\x8b\x08\x00' and raw[4:8] == b'\0' * 4, 'Nondeterministic gzip header')
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                raw = stream.read(row['bytes'] + 1)
        require(len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'Changed decoded payload: ' + name)
        require(b'\0' not in raw and not raw.startswith((b'\x7fELF', b'!<arch>')), 'Native/NUL bytes retained')
        raw.decode('utf8')
        decoded[name] = raw
    def raw(name):
        safe(name)
        require(name in decoded, 'Unindexed logical read: ' + name)
        return decoded[name]
    def text(name):
        return raw(name).decode('utf8')
    def load(name):
        return decode(raw(name))
    def norm(value):
        return json.loads(json.dumps(value))
    def prefix_inventory(prefix):
        return {name[len(prefix):] for name in decoded if name.startswith(prefix)}
    retention=load('retention.json')
    require(retention['schema']=='o3-private-experiment-retention-v1' and retention['patches_applied']==0 and
            retention['native_evaluations_during_capture']==0 and all(retention[k] is False for k in
            ('production_changes','public_writes','native_bytes_retained','candidate_promoted','performance_acceptance','optimization_benefit_claimed')),
            'Retention scope changed')
    for name,row in retention['origins'].items():
        require(sha(raw(name))==row['sha256'] and len(raw(name))==row['bytes'],'Retained origin changed')
    for number,expected in ((1,SOURCE1),(2,SOURCE2)):
        prefix='source/source'+str(number)+'/'
        require(sha(raw(prefix+'source-index.json'))==expected,'Source seal differs')
        seal=load(prefix+'source-index.json')
        require(prefix_inventory(prefix)==set(seal['files'])|{'source-index.json'},'Source inventory differs')
        for name,row in seal['files'].items():
            require(sha(raw(prefix+safe(name)))==row['sha256'] and len(raw(prefix+name))==row['bytes'],'Sealed source differs')
        for name,value in seal['support'].items():
            require(sha(raw('support/'+safe(name)))==value,'Sealed support differs')
    predecessor=load('source/source2/predecessor.json')
    require(predecessor['predecessor_source_index_sha256']==SOURCE1 and predecessor['predecessor_native_admitted'] is False and
            predecessor['predecessor_native_stages_executed']==[] and len(predecessor['predecessor_files'])==35,'Unadmitted source1 scope differs')
    require(prefix_inventory('source/source1/')==set(predecessor['predecessor_files']),'Predecessor inventory differs')
    for name,value in predecessor['predecessor_files'].items():
        require(sha(raw('source/source1/'+safe(name)))==value,'Predecessor bytes differ')
    for name in ('drivers.py','policy.py','tool_environment.py','toolchain-pins.json','makefile-n1.diff','makefile-n3.diff'):
        require(raw('source/source1/'+name)==raw('source/source2/'+name),'Scientific/policy delta differs')
    for prefix in ('original-source/','source-preview/'):
        for name in prefix_inventory('source/source1/'):
            if name.startswith(prefix):
                require(raw('source/source1/'+name)==raw('source/source2/'+name),'Fortran/source preview changed')
    pins=load('source/source2/input-pins.json');tools=load('source/source2/toolchain-pins.json')
    require(tools['original_compiler_environment']=={} and len(tools['required_tools'])==11 and
            all(v['executable'] is True for v in tools['required_tools'].values()),'Frozen tool policy differs')
    for name,value in {**pins['ancestry'],**tools['donor_lineage']}.items():
        require(sha(raw('ancestry/'+safe(name)))==value,'Retained ancestry differs')
    for name,value in AUDITS.items():
        path='audits/'+name+'/report.json';audit=load(path)
        require(sha(raw(path))==value and audit['status']=='PASS' and audit['compiler_invocations_by_auditor']==
                audit['loader_queries_by_auditor']==audit['native_numerical_evaluations_by_auditor']==0,'Independent audit differs')
    for name,value in PRIMARY.items():
        prefix='primary-audits/'+name+'/';audit=load(prefix+'receipt.json')
        require(sha(raw(prefix+'receipt.json'))==value and audit['status'].lower()=='pass' and audit['audit_native_calls']==0 and
                sha(raw(prefix+'review.py'))==audit['audit_source_sha256'],'Primary independent audit differs')
    require(sha(raw('primary-audits/root-preflight-2.json'))=='c7f61e5e9fe473d9c2463e7bd3113fbed7b53e0e91a731183466b29fcfc11fa3' and
            load('primary-audits/root-preflight-2.json')['source_index_sha256']==SOURCE2,'Root preflight differs')
    require(sha(raw('prospective/delivery/index.json'))==DELIVERY,'Prospective delivery index differs')
    delivery=load('prospective/delivery/index.json')
    require(delivery['schema']=='private-unapplied-wrapper-delivery-v1' and delivery['status']=='experimental-unpromoted' and
            delivery['patches_applied']==0 and delivery['production_changes'] is False and delivery['public_writes'] is False and
            retention['prospective_delivery_is_separate'] is True,'Prospective patches were relabeled as applied')
    require(prefix_inventory('prospective/delivery/')==set(delivery['files'])|{'index.json'},'Delivery inventory differs')
    for name,row in delivery['files'].items():
        require(sha(raw('prospective/delivery/'+safe(name)))==row['sha256'] and
                len(raw('prospective/delivery/'+name))==row['bytes'],'Unapplied delivery bytes differ')
    require(raw('prospective/delivery/IMPACT_AUDIT.txt') and raw('prospective/delivery/backend9-wrapper-opt-in.patch') and
            raw('prospective/delivery/backend9-wrapper-enabled-default.patch'),'Prospective impact/patch evidence absent')
    order=('build','n1parity','n1timing','n3parity','n3anchortiming');stages={}
    for index,name in enumerate(order):
        prefix='results/'+name+'/';terminal,result=STAGES[name]
        require(sha(raw(prefix+'terminal.json'))==terminal and sha(raw(prefix+'result.json'))==result,'Closed stage pin differs')
        t,r,a=load(prefix+'terminal.json'),load(prefix+'result.json'),load(prefix+'admission.json')
        require(t['status']==r['status']=='passed' and t['returncode']==0 and not t['cleanup']['survivors'],'Stage/cleanup did not close')
        require(t['result_sha256']==result and t['source_index_sha256']==r['source_index_sha256']==a['source_index_sha256']==SOURCE2 and
                t['stage']==a['stage']==name and a['cpu']==12 and a['cpu_reservation'] is False,'Stage source/admission join differs')
        require(t['cap_seconds']=={'build':600,'n1parity':300,'n1timing':300,'n3parity':600,'n3anchortiming':600}[name] and
                t['cleanup_reserve_seconds']==60 and load(prefix+'actual-limits.json')['actual']==
                {k:[v,v] for k,v in t['resource_limits'].items()},'Stage limits differ')
        if name!='build':
            expected={key:STAGES[key][0] for key in order[:index]}
            require(r['reviewed']==expected and {k:v for k,v in a['reviewed'].items() if v is not None}==expected and
                    r['candidate_promoted'] is False and r['performance_acceptance'] is False,'Reviewed chain/promotion differs')
        stages[name]=r
    require(sha(raw('source/source2/analysis.py'))==PARSER and sha(raw('replay_warmup.py'))==WARMUP_PARSER,'Pure parser changed')
    analysis=types.ModuleType('indexed_o3_analysis');exec(compile(text('source/source2/analysis.py'),'<indexed:pure-analysis>','exec'),analysis.__dict__)
    learning_parser=types.ModuleType('indexed_o3_learning');exec(compile(text('replay_warmup.py'),'<indexed:pure-learning>','exec'),learning_parser.__dict__)
    import ast
    comparator_source='support/tools/tirust_benchmark.py';node=[n for n in ast.parse(text(comparator_source)).body if isinstance(n,ast.FunctionDef) and n.name=='compare_rows']
    require(len(node)==1,'Ambiguous existing comparator');namespace={};exec(compile(ast.Module(body=node,type_ignores=[]),'<indexed:existing-comparator>','exec'),namespace);compare=namespace['compare_rows']
    def float_rows(rows):
        return [{'point':r['point'],'order':r['order'],'status':r['status'],
                 'words':[struct.pack('>d',v).hex().upper() for v in r['coefficients']+[r['accuracy']]]} for r in rows]
    originals={}
    for case,p in pins['cases'].items():
        require(p['compiler_flags']['environment']=={},'Original compiler injection differs')
        for name,value in p['generated_sources'].items():
            require(sha(raw('donor/'+case+'/'+safe(name)))==value,'Original generated source differs')
        require(load('donor/'+case+'/compilation.json')['effective_compiler_flags']==p['compiler_flags'],'Original compiler receipt differs')
        for name,value in p['numerical_cards'].items():
            require(sha(raw('donor/'+case+'/'+name))==value,'Original card differs')
        require(sha(raw('donor/'+case+'/Cards/param_card.dat'))==p['param_card_sha256'],'Original physical card differs')
        closed=load('inputs/'+case+'/closed-observation.json')
        require(text('inputs/'+case+'/closed-warmup.dat')==closed['raw_outputs']['warmup.dat'],'Original learning raw join differs')
        for logical,recorded in (('closed-observation.json',p['closed_observation']),('closed-input.json',p['closed_input']),
                                 ('closed-warmup.dat',p['closed_warmup']),('points.dat',p['points'])):
            key=next((name for name in pins['files'] if recorded.endswith(name)),None)
            require(key is not None and sha(raw('inputs/'+case+'/'+logical))==pins['files'][key]['sha256'],'Original textual input pin differs')
        originals[case]=closed
    require(sha(raw('inputs/stock-mtirust.f90'))==pins['files']['vendor/tirust/mtirust.f90']['sha256'],'Stock wrapper source differs')
    basis=load('source/source2/validation-plan.json')['optional_n3_anchors']['DP_loop_basis']
    historical_timing=load('ancestry/'+basis['path'])
    require(historical_timing['anchors']['1']['summary']['original']['median_seconds_per_call']==basis['original_median_seconds_per_call']==0.070471572 and
            basis['fixed_loops']==15 and basis['estimated_batch_seconds_from_closed_original']==1.05707358,'Fixed DP-anchor planning basis differs')
    build=stages['build'];aliases=load('link-runtime-resolution.json');counts=Counter()
    require(build['native_amplitude_calls']==0 and build['compiled_units_per_variant']==2 and
            set(build['variants'])=={c+'/'+r for c in ('n1','n3') for r in ROLES},'BUILD scope differs')
    for key,row in build['variants'].items():
        case,role=key.split('/');p=pins['cases'][case];compiled='compiled/'+key+'/';prefix='results/build/'+key+'/'
        for name in ('check_sa.f','TIRUST_interface.f'):
            require(raw(compiled+name)==raw('source/source2/source-preview/'+case+'/'+name),'Compiled source differs')
        require(raw(compiled+'makefile')==raw('source/source2/source-preview/'+case+'/makefile.'+role),'Compiled makefile differs')
        for name,value in row['sources'].items():
            base=name.removeprefix('SubProcesses/'+p['subprocess']+'/')
            logical=compiled+base if name.startswith('SubProcesses/'+p['subprocess']+'/') and base in ('makefile','check_sa.f','TIRUST_interface.f') else 'donor/'+case+'/'+name
            require(sha(raw(logical))==value,'Compiled/reused source identity differs')
        old_objects={n:v for n,v in p['numerical_objects'].items() if not n.startswith('lib/')}
        changed={'SubProcesses/'+p['subprocess']+'/'+n for n in ('check_sa.o','TIRUST_interface.o')}
        require(set(old_objects)==set(row['objects']) and {n:v for n,v in old_objects.items() if n not in changed}==
                {n:v for n,v in row['objects'].items() if n not in changed},'Other numerical objects changed')
        actual=row['compiler_flags'];require(actual['environment']=={},'Compiler injection differs')
        for scope,left in p['compiler_flags']['scopes'].items():
            right=actual['scopes'][scope]['recipes'];require(set(right)==set(left['recipes']),'Compiler recipe inventory differs')
            for name,value in left['recipes'].items():
                extra=['-O3','-fno-fast-math'] if role=='candidate' and scope=='process' and name=='TIRUST_interface.f' else []
                require(right[name]['compiler']==value['compiler'] and right[name]['flags']==value['flags']+extra,'Candidate compile delta differs')
        for scope,number in (('process','01'),('MODEL','03'),('DHELAS','04')):
            require(text(prefix+'commands/policy-before-'+number+'.stdout.txt')==text(prefix+'commands/policy-after-'+number+'.stdout.txt')==
                    actual['scopes'][scope]['expanded_make_output'],'Before/after policy capture differs')
        require(sha(raw(prefix+'prelink.json'))==row['prelink_sha256'] and sha(raw(compiled+'private-link.map'))==row['link_map_sha256'],'Prelink/map pin differs')
        pre=load(prefix+'prelink.json');mappings=aliases[key]['map_aliases']
        require(re.findall(r'^LOAD\s+(.+)$',text(compiled+'private-link.map'),re.M)==[m['spelling'] for m in mappings] and
                {m['resolved']:m['sha256'] for m in mappings}==row['actual_link_loads'] and len(row['actual_link_loads'])==(52 if case=='n1' else 54) and
                all(pre['bound'].get(m['resolved'])==m['sha256'] for m in mappings),'Actual LOAD proof differs')
        require(re.search(r'libtirust\.a\([^)]*mtirust[^)]*\)',text(compiled+'private-link.map')),'Stock wrapper archive absent')
        runtime=aliases[key]['runtime_aliases'];spellings=[]
        for line in row['runtime']['listing'].splitlines():
            line=line.strip()
            if not line or line.startswith('linux-vdso.so.1 '):continue
            match=re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s+\(0x[0-9a-fA-F]+\)',line)
            require(match is not None,'Malformed retained loader listing');spellings.append(match[1])
        require(spellings==[m['spelling'] for m in runtime] and {m['resolved']:m['sha256'] for m in runtime}==row['runtime']['files'] and
                len(row['runtime']['files'])==15 and all(pre['bound'].get(n)==v for n,v in row['runtime']['files'].items()),'Runtime provider closure differs')
        definitions=Counter()
        for line in text(prefix+'commands/03-executable-symbols.stdout.txt').splitlines():
            words=line.split()
            if len(words)>=3 and '__mtirust_MOD_' in words[-1]:definitions[words[-1]]+=1
        require(dict(definitions)==row['mtirust_definitions']=={n:1 for n in pins['expected_mtirust_symbols']} and len(definitions)==23,'Stock module symbols differ')
        local_counts=Counter()
        for name in sorted(n for n in decoded if n.startswith(prefix+'commands/') and n.endswith('.command.json')):
            c=load(name);stem=name.removesuffix('.command.json');t=load(stem+'.terminal.json');label=PurePosixPath(stem).name
            require(t['returncode']==0 and sha(raw(stem+'.stdout.txt'))==t['stdout_sha256'],'Recorded BUILD command failed/changed')
            if 'stderr_sha256' in t:require(sha(raw(stem+'.stderr.txt'))==t['stderr_sha256'],'Policy stderr changed')
            argv=c['argv']
            if '-c' in argv:kind='object_compiles'
            elif label=='02-link':
                kind='links';require('mtirust.o' not in argv and 'qp_profile.o' not in argv and '-ltirust' in argv,'Private wrapper leaked')
            elif label.startswith('prelink-query-'):kind='prelink_queries'
            elif label.startswith(('policy-before','policy-after')):kind='version_queries' if '--version' in argv else 'make_queries'
            elif label=='expanded-check':kind='make_queries'
            elif label=='03-executable-symbols':kind='nm_queries'
            else:kind='loader_queries';require('--list' in argv,'Unexpected BUILD action')
            local_counts[kind]+=1
        require(local_counts=={'object_compiles':2,'links':1,'prelink_queries':51,'make_queries':7,'version_queries':2,'nm_queries':1,'loader_queries':1},'BUILD action counts differ')
        counts.update(local_counts)
    require(sum(counts.values())==260 and retention['query_count_scope']['recorded_BUILD_actions']==260 and
            retention['query_count_scope']['all_make_internal_children_individually_traced'] is False,'Query-count claim differs')
    public_learning=0;steady_learning=0;untimed_exact=0;runtime_queries=4;cross_accuracy_differences=0;own_drift=0;worker_gates={}
    def history(value):
        return [list(map(int,line.split()[:4])) for line in value.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    for phase,case in (('n1parity','n1'),('n1timing','n1'),('n3parity','n3'),('n3anchortiming','n3')):
        for role in ROLES:
            prefix='results/'+phase+'/'+role+'/';p=pins['cases'][case];row=build['variants'][case+'/'+role]
            ready,warm=load(prefix+'ready.json'),load(prefix+'warmup.json')
            require(learning_parser.replay_warmup(warm,warm,ready)==warm,'Raw full learning parser differs')
            actual=text(prefix+'warmup.dat');historic=text('inputs/'+case+'/closed-warmup.dat');learning=load(prefix+'learning-comparison.json')
            require(actual==warm['raw_outputs']['warmup.dat'] and history(actual)==history(historic)==learning['actual_call_point_order_status']==
                    learning['historical_call_point_order_status'] and learning['history_matches'] is True and learning['raw_byte_identity']==(actual==historic),
                    'Complete learning history/raw joins differ')
            science=analysis.compare_variants(float_rows(originals[case]['rows']),float_rows(warm['rows']),compare)
            require(norm(science)==learning['steady_full100_comparison'] and science['passed'],'Learning scientific comparison differs')
            require(warm['warmup_first_call']==1 and warm['warmup_last_call']==100 and warm['status_counts']==({'219':100} if case=='n1' else {'219':87,'323':13}),
                    'Accepted learning cohort differs')
            public_learning+=len({r[0] for r in history(actual)});steady_learning+=100
            gate=prefix+('exact/' if phase.endswith('parity') else 'exact-before-timing/')
            rows=analysis.exact(text(gate+'exact-words.dat'),p['expected_codes']);flags=analysis.ieee(text(gate+'exact-ieee.dat'))
            historical=analysis.compare_variants(float_rows(originals[case]['rows']),rows,compare)
            require(norm(historical)==load(gate+'historical-comparison.json') and historical['passed'],'Historical full100 science differs')
            untimed_exact+=100
            if phase.endswith('parity'):
                require({'rows':rows,'ieee':flags,'historical_comparison':norm(historical)}==stages[phase]['lanes'][role],'Parity parser/result join differs')
            else:
                reference=stages['n1parity' if case=='n1' else 'n3parity']['lanes'][role]
                own=analysis.compare_own(reference['rows'],rows)
                require(norm(own)==load(gate+'own-qualified-comparison.json')==stages[phase]['fresh_own_comparisons'][role] and
                        own['passed'] and flags==reference['ieee'],'Fresh own coefficient/status/IEEE gate differs')
                own_drift+=own['accuracy_drift_count']
            worker_gates[phase,role]={'rows':rows,'ieee':flags}
            before,after=load(prefix+'private-before.json'),load(prefix+'private-after.json')
            alias='SubProcesses/'+p['subprocess']+'/MadLoop5_resources/MadLoopParams.dat'
            require(before==after and before['executable_sha256']==row['executable_sha256'] and before['physical_sha256']==p['param_card_sha256'] and
                    before['cards']=={**p['numerical_cards'],alias:next(iter(p['numerical_cards'].values()))},'Worker card/executable binding differs')
            for when in ('before','after'):
                mapped=load(prefix+'maps-'+when+'.json');providers=row['runtime']['files']
                require(all(mapped['files'].get(n)==v for n,v in providers.items()),'Actual mapped provider differs')
                allowed={**providers,**load('source/source2/source-index.json')['allowed_extra_mappings']}
                extra={n:v for n,v in mapped['files'].items() if allowed.get(n)!=v}
                require(len(extra)==1 and next(iter(extra.values()))==row['executable_sha256'],'Actual mapped executable/extra differs')
                names={line.split(None,5)[5] for line in mapped['maps'].splitlines() if len(line.split(None,5))==6 and line.split(None,5)[5].startswith('/')}
                require(names==set(mapped['files']),'Raw map paths differ')
            queries=[n for n in decoded if n.startswith(prefix+'commands/') and n.endswith('.command.json')]
            require(len(queries)==2,'Worker loader-query count differs')
            for name in queries:
                stem=name.removesuffix('.command.json');c,t=load(name),load(stem+'.terminal.json')
                require(c['argv'][0]==row['runtime']['loader'] and c['argv'][1]=='--list' and t['returncode']==0 and
                        sha(raw(stem+'.stdout.txt'))==t['stdout_sha256'],'Runtime loader receipt differs')
            runtime_queries+=2
        if phase.endswith('parity'):
            left,right=worker_gates[phase,'original'],worker_gates[phase,'candidate']
            science=analysis.compare_variants(left['rows'],right['rows'],compare)
            require(norm(science)==stages[phase]['comparison']==load('results/'+phase+'/cross-variant-comparison.json') and science['passed'] and
                    left['ieee']==right['ieee'] and load('results/'+phase+'/ieee-comparison.json')['observed_states_equal'],'Crossvariant parity differs')
            cross_accuracy_differences+=science['accuracy_difference_count']
    timed_calls=Counter();timed_rows=0;timed_batches=0;summaries={}
    for phase,case,parity_name,specs in (('n1timing','n1','n1parity',(('full100',None,128),)),
            ('n3anchortiming','n3','n3parity',(('DP',1,15),('QP',16,1)))):
        require(set(stages[phase]['scopes'])=={label for label,point,loops in specs} and stages[phase]['pairs']==7,'Timing scopes differ')
        for label,point,loops in specs:
            record=stages[phase]['scopes'][label]
            require(record['point']==point and record['loops']==loops and len(record['pairs'])==7,'Fixed timing batch differs')
            previous={role:[r for r in worker_gates[phase,role]['rows'] if point is None or r['point']==point] for role in ROLES}
            for index,pair in enumerate(record['pairs'],1):
                require(pair['pair']==index and pair['order']==(['original','candidate'] if index%2 else ['candidate','original']) and
                        load('results/'+phase+'/'+label+'/pair-'+str(index)+'.json')==pair,'Alternating pair receipt differs')
                for role in ROLES:
                    prefix='results/'+phase+'/'+role+'/'+label+'/pair-'+str(index)+'/'
                    actual=pair[role];p=pins['cases'][case]
                    if point is None:
                        rows=analysis.exact(text(prefix+'timed-exact-words.dat'),p['expected_codes'])
                        require(analysis.decimal_exact(text(prefix+'values.dat'),p['expected_codes'])==rows==actual['rows'],'Same-event five-word join differs')
                        histogram=analysis.status_histogram(text(prefix+'timed_status.dat'),p['expected_codes'],loops)
                        require(histogram==actual['status_counts'],'Full timed histogram differs')
                        flags=analysis.ieee(text(prefix+'timed-ieee.dat'),points=[0]);require(flags==actual['ieee'],'Timed IEEE raw join differs')
                        timing=list(map(float,text(prefix+'timing.dat').split()))
                        require(len(timing)==2 and all(math.isfinite(v) and v>0 for v in timing) and timing[1]==actual['seconds'],'Raw timing join differs')
                    else:
                        parsed=analysis.anchor_outputs({n:text(prefix+n) for n in ('anchor-timing.dat','anchor-status.dat','anchor-exact.dat','anchor-ieee.dat','anchor-values.dat')},p['expected_codes'],point=point,loops=loops)
                        require(all(norm(value)==actual[key] for key,value in parsed.items()),'Anchor raw parser/result differs')
                        rows,flags,histogram=parsed['rows'],parsed['ieee'],parsed['timed_status_counts']
                    wanted=[r for r in stages[parity_name]['lanes'][role]['rows'] if point is None or r['point']==point]
                    reports=replay_own_reports(wanted,previous[role],rows,load(prefix+'own-comparisons.json'),analysis)
                    require(norm(reports)==actual['own_comparisons'] and actual['previous_batch_reference']==
                            ('fresh_full100_exact' if index==1 else 'pair-'+str(index-1)),'Own diagnostic/reference metadata differs')
                    own_drift+=sum(v['accuracy_drift_count'] for v in reports.values());previous[role]=rows
                    timed_calls.update({int(k):v for k,v in histogram.items()});timed_rows+=len(rows);timed_batches+=1
                require(pair['original']['ieee']==pair['candidate']['ieee'],'Timed paired IEEE differs')
                science=analysis.compare_variants(pair['original']['rows'],pair['candidate']['rows'],compare)
                require(norm(science)==pair['comparison']==load('results/'+phase+'/'+label+'/pair-'+str(index)+'-comparison.json') and science['passed'],
                        'Paired scientific comparison differs')
                cross_accuracy_differences+=science['accuracy_difference_count']
            recomputed=analysis.timing_summary(record['pairs']);require(norm(recomputed)==record['summary'],'Timing median replay differs')
            summaries[phase+'/'+label]=recomputed
    require(public_learning==808 and steady_learning==800 and untimed_exact==800 and timed_batches==42 and timed_rows==2856 and
            dict(timed_calls)=={219:179410,323:14} and runtime_queries==20,'Full experiment counts differ')
    summary={'schema':'o3-portable-experiment-summary-v1','status':'PASS','candidate_promoted':False,'performance_acceptance':False,
        'optimization_benefit_claimed':False,'source1_unadmitted':True,'prospective_delivery_unapplied':True,
        'delivery_registration_and_formal_compiler_policy_qualified':False,'production_scientific_source':'0fcd2cd873d61d134837e1d9e87afb13a47824fd',
        'recorded_BUILD_actions':260,'make_internal_children_individually_traced':False,'recorded_loader_queries':20,
        'fresh_workers':8,'learning_calls_including_bootstrap':808,'bootstrap_calls_unqualified':8,'steady_learning_calls':800,
        'untimed_full100_qualification_calls':800,'timed_physical_calls':sum(timed_calls.values()),
        'timed_status_histogram':{str(k):v for k,v in timed_calls.items()},'total_public_calls_including_bootstrap':public_learning+untimed_exact+sum(timed_calls.values()),
        'timed_batches':42,'final_timed_answer_rows':2856,'own_accuracy_diagnostic_drift_records':own_drift,
        'crossvariant_accuracy_diagnostic_differences':cross_accuracy_differences,'timing_summaries':summaries,
        'limitations':['Own-reference gates require exact coefficient/status words; finite nonnegative accuracy drift is diagnostic, never a cross-history identity gate.',
            'Same-event decimal/raw joins require all five words, including accuracy.',
            'Only each n1 batch final100 traversal and each anchor batch final call supplies timed words; all timed calls supply statuses.',
            'Bootstrap call0 is retained but scientifically unqualified.',
            'Shared-host timings scatter across1; no optimization benefit, n3 full-cohort runtime gain, promotion or precise local acceptance is established.',
            'Whole-amplitude automatic-QP timings include DP/stability/rescue work.',
            'IEEE pre/post sticky observations do not prove trap freedom or identify newly raised true-to-true flags.',
            'Prospective delivery patches remain separate and unapplied; registration/compiler-policy delivery blockers remain open.',
            'Native hashes/maps are historical identities, not receiving-host file checks.']}
    return {'status':'PASS','summary':summary,'indexed_text_payloads':len(payloads),'index_sha256':sha(raw_index),
            'compiler_invocations':0,'loader_queries':0,'native_evaluations':0}


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--expected-index');args=cli.parse_args()
    print(json.dumps(verify(expected_index_sha256=args.expected_index),indent=2,sort_keys=True))
