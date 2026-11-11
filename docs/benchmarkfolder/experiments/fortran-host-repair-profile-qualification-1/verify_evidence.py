"""Pure portable replay; historical absolute names are never opened."""
import argparse
import builtins
import math
import statistics
import struct
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import re
import types

SOURCE='97cad155079138419c198599adbbbcc00bb8df0a3ac799b8ef9ec9dd9858bace'
MEASUREMENT_REPLAY_SHA256='8771e5b0b405980d661b22076f84dc6283bafa8d340e7d7db9ccd0eb2e739db2'
STAGES={
 'build':('663b3947123be881864b78179bb8caaa19870277ed654c1232219699a992e5b2','7c16ffb23011d40075b6782ded8105da6521c2117d86d0fbc3a516abec041e86'),
 'parity':('5218212507e983ae8400c6396c32da84d4aed180660addac7436297d5ec761d8','98c2fd869ac47a4f60e075f82f5cfc783f917abbb412f122dc576da2fdd3ebc8'),
 'profile':('39fb1129fa6417ddf9b7950e0eedab3a4484c4a23a314538630a46c4ee33797d','b2b12b6cd71aaa91e510fead3fc02fb85361b602c95d81704fb3a7c9a29baa5a')}
AUDITS={
 'source/final-seal-review.json':'fa25c373e2d50644dfdd7f1b4fd8973fb28ef2ea6be31e6e2c494ce95d0f35cf',
 'build/receipt.json':'bec02c56a364da1d39d16807e340246e295f94af9e617d4e41cf0e76d670d1b1',
 'parity/receipt.json':'872cf3aa1643ce461d11d2593a145555aa930e30e7502df0932d9a7bc4aefe4d',
 'profile/receipt.json':'49dac514264a74af3353bdb582fb4b1a4d49cede21a86946ce892a7ad9d75e4e',
 'primary/receipt.json':'db5410b97d151c621121c8612637a14fdfe4f016fbf0536183c80960f150f69a'}
LANES=('baseline','observed-0','observed-1','observed-2')

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
    require(index['schema'] == 'host-repair-profile-portable-v1', 'Wrong capsule schema')
    require(all(index[name] is False for name in ('native_bytes_retained','performance_acceptance','speedup_claimed','production_promoted',
                                                 'all_five_backends_qualified','general_stress_qualified')), 'Scope changed')
    files, payloads = index['files'], index['payloads']
    for name, row in files.items():
        safe(name)
        require(name != 'evidence-index.json' and set(row) == {'sha256', 'bytes'}, 'Self/invalid physical index')
        require(type(row['bytes']) is int and 0 <= row['bytes'] <= 64 * 1024**2 and
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
    require(sum(row['bytes'] for row in payloads.values()) <= 128 * 1024**2, 'Decoded capsule too large')
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
    require(retention['schema']=='host-repair-profile-retention-v1' and retention['source_index_sha256']==SOURCE,'Retention/source scope differs')
    require(retention['native_executions_during_capture']==0 and all(retention[k] is False for k in
            ('binary_bytes_retained','performance_acceptance','speedup_claimed','production_promoted','all_five_backends_qualified','general_stress_qualified')),'Retention claims changed')
    require(set(retention['origins'])==set(decoded)-{'retention.json'},'Capture origin inventory differs')
    for name,row in retention['origins'].items():
        require(sha(raw(name))==row['sha256'] and len(raw(name))==row['bytes'],'Captured origin differs:'+name)
    require(sha(raw('source/source-index.json'))==SOURCE,'Frozen source seal differs')
    source=load('source/source-index.json')
    require(prefix_inventory('source/')==set(source['files'])|{'source-index.json'},'Frozen source inventory differs')
    for name,row in source['files'].items():
        require(sha(raw('source/'+safe(name)))==row['sha256'] and len(raw('source/'+name))==row['bytes'],'Frozen source bytes differ')
    for name,wanted in source['support'].items():require(sha(raw('support/'+safe(name)))==wanted,'Pinned support differs')
    predecessor=load('source/predecessor.json')
    require(prefix_inventory('scaffold/')==set(predecessor['donor_proposal_files']),'Scaffold exact inventory differs')
    for name,wanted in predecessor['donor_proposal_files'].items():require(sha(raw('scaffold/'+safe(name)))==wanted,'Scaffold bytes differ')
    require(sha(raw('scaffold/source-index.json'))==predecessor['donor_source_index_sha256'],'Scaffold source seal differs')
    for name,wanted in AUDITS.items():require(sha(raw('audits/'+name))==wanted,'Independent audit differs:'+name)
    for phase in ('build','parity','profile'):
        audited=load('audits/'+phase+'/receipt.json')['report_file_sha256']
        require(prefix_inventory('results/'+phase+'/')==set(audited),'Independent audited stage inventory differs:'+phase)
        for name,wanted in audited.items():require(sha(raw('results/'+phase+'/'+safe(name)))==wanted,'Independent audited stage bytes differ:'+phase+'/'+name)
    tools=load('source/toolchain-pins.json')
    require(len(tools['required_tools'])==11 and tools['original_compiler_environment']=={},'Toolchain scope differs')
    for name,wanted in tools['donor_lineage'].items():require(sha(raw('environment-donor/'+safe(name)))==wanted,'Closed tool environment lineage differs')
    stages={}
    for phase,(terminal_hash,result_hash) in STAGES.items():
        prefix='results/'+phase+'/'
        require(sha(raw(prefix+'terminal.json'))==terminal_hash and sha(raw(prefix+'result.json'))==result_hash,'Closed stage seal differs')
        terminal,result,admission=load(prefix+'terminal.json'),load(prefix+'result.json'),load(prefix+'admission.json')
        require(terminal['status']==result['status']=='passed' and terminal['returncode']==0 and not terminal['cleanup']['survivors'],'Stage failed or cleanup incomplete')
        require(terminal['result_sha256']==result_hash and terminal['source_index_sha256']==result['source_index_sha256']==admission['source_index_sha256']==SOURCE,'Stage source/result binding differs')
        require(terminal['stage']==admission['stage']==phase and terminal['cap_seconds']=={'build':600.,'parity':300.,'profile':600.}[phase] and terminal['cleanup_reserve_seconds']==60. and terminal['cpu']==12 and terminal['cpu_reservation'] is False,'Stage scope/bounds differ')
        limits={'address_space_bytes':16*1024**3,'file_bytes':512*1024**2,'core_bytes':0}
        require(terminal['resource_limits']==admission['resource_limits']==limits and load(prefix+'actual-limits.json')['actual']=={k:[v,v] for k,v in limits.items()},'Actual resource limits differ')
        actual=load(prefix+'actual-limits.json')
        require(actual['pid']==actual['session']==terminal['pid'] and any(row['pid']==terminal['pid'] and row['parent_pid']==admission['parent_pid'] and row['session']==terminal['pid'] for row in terminal['cleanup']['observed_owned_processes']),'Actual child session/cleanup identity differs')
        if phase!='build':require(admission['reviewed_build_sha256']==result['build_terminal_sha256']==STAGES['build'][0],'Reviewed BUILD join differs')
        if phase=='profile':require(admission['reviewed_parity_sha256']==result['parity_terminal_sha256']==STAGES['parity'][0],'Reviewed PARITY join differs')
        stages[phase]=result
    pins=load('source/input-pins.json');donor=load('donor/generation.json')
    require(retention['production_numerical_source']=='0fcd2cd873d61d134837e1d9e87afb13a47824fd' and retention['actual_export_revision']==donor['source_identity']['revision'],'Actual/export production provenance differs')
    require(load('donor/compilation.json')['effective_compiler_flags']==pins['compiler_flags'],'Original compiler receipt differs')
    for name,wanted in pins['generated_sources'].items():require(sha(raw('donor/'+safe(name)))==wanted,'Original generated source differs')
    for name,wanted in pins['numerical_cards'].items():require(sha(raw('donor/'+safe(name)))==wanted,'Original numerical card differs')
    excluded=load('excluded-native-identities.json');aliases=load('link-runtime-resolution.json');build=stages['build']
    require(set(build['variants'])=={'baseline','observed'},'Wrong executable variants')
    def module_symbols(value):
        return dict(Counter(line.split()[-1] for line in value.splitlines() if len(line.split())>=3 and line.split()[-1].startswith('__mtirust_MOD_')))
    original_symbols=module_symbols(text('results/build/original-symbols/original-symbols.stdout.txt'))
    require(len(original_symbols)==23 and set(original_symbols.values())=={1},'Original MTIRUST ownership differs')
    total_commands=0;compile_sources={};total_links=0
    commands=sorted(name for name in decoded if name.startswith('results/') and name.endswith('.command.json'))
    for name in commands:
        command=load(name);stem=name.removesuffix('.command.json');terminal=load(stem+'.terminal.json')
        require(terminal['returncode']==0 and sha(raw(stem+'.stdout.txt'))==terminal['stdout_sha256'],'Command result/stdout differs')
        if 'stderr_sha256' in terminal:require(sha(raw(stem+'.stderr.txt'))==terminal['stderr_sha256'],'Command stderr differs')
        if name.startswith('results/build/'):
            total_commands+=1;argv=command['argv']
            require(not argv[0].endswith('/check'),'Unexpected model executable call in BUILD')
            if '-c' in argv:
                inputs=[a for a in argv if a.endswith(('.f','.f90'))]
                require(len(inputs)==1 and inputs[0] in ('repair_profile.f90','improve_ps.f','check_sa.f'),'Unexpected compiled scientific source')
                variant=name.split('/')[2];compile_sources.setdefault(variant,[]).append(inputs[0])
            if PurePosixPath(stem).name=='link':
                total_links+=1
                require(argv.count('repair_profile.o')==1 and '-ltirust' in argv and 'mtirust.o' not in argv and 'qp_profile.o' not in argv,'Installed wrapper/profile link ownership differs')
        else:require('--list' in command['argv'],'Unexpected nonloader stage command')
    require(total_commands==retention['recorded_BUILD_commands']==133 and total_links==2 and
            all(sorted(compile_sources.get(v,[]))==['check_sa.f','improve_ps.f','repair_profile.f90'] for v in ('baseline','observed')),'Build action/compile closure differs')
    for variant,row in build['variants'].items():
        compiled='compiled/'+variant+'/';source_prefix='source/'+variant+'-source/'
        for name in ('check_sa.f','improve_ps.f'):require(raw(compiled+name)==raw(source_prefix+name),'Actually compiled source differs')
        require(raw(compiled+'repair_profile.f90')==raw('source/repair_profile.f90'),'Compiled profile module differs')
        for name,wanted in row['sources'].items():
            basename=name.removeprefix('SubProcesses/P0_uux_z/')
            logical=compiled+basename if name.startswith('SubProcesses/P0_uux_z/') and basename in ('check_sa.f','improve_ps.f','repair_profile.f90','makefile') else 'donor/'+name
            require(sha(raw(logical))==wanted,'Compiled/reused source identity differs')
        prefix='SubProcesses/P0_uux_z/';allowed={prefix+'check_sa.o',prefix+'improve_ps.o'};added={prefix+'repair_profile.o',prefix+'host_repair_profile_state.mod'}
        require(set(row['objects'])==set(pins['numerical_objects'])|added and len(row['objects'])==62,'Canonical object inventory differs')
        require(all(row['objects'].get(name)==wanted for name,wanted in pins['numerical_objects'].items() if name not in allowed),'Unselected physics object changed')
        if variant=='baseline':require(row['objects'][prefix+'improve_ps.o']==pins['numerical_objects'][prefix+'improve_ps.o'],'Baseline repair object differs')
        for name in ('libcts.a','libdhelas.a','libiregi.a','libmodel.a','mpmodule.mod'):
            require(row['objects']['lib/'+name]==pins['numerical_objects']['lib/'+name],'Mandatory shared library/module differs')
        require(excluded[variant]=={'executable':{'path':row['output']+'/SubProcesses/P0_uux_z/check','sha256':row['executable_sha256']},'objects_modules_archives':row['objects'],'runtime_providers':row['runtime']['files']},'Excluded native identity inventory differs')
        before=pins['compiler_flags'];after=row['compiler_flags']
        require(before['environment']==after['environment']=={} and set(before['scopes'])==set(after['scopes']),'Compiler injection/scopes differ')
        for scope,left in before['scopes'].items():
            right=after['scopes'][scope]['recipes'];extra={'repair_profile.f90'} if scope=='process' else set()
            require(set(right)==set(left['recipes'])|extra,'Compiler recipe inventory differs')
            for name,value in left['recipes'].items():require(value==right[name],'Original compiler recipe changed')
            if extra:require(right['repair_profile.f90']==right['improve_ps.f'],'New module ordinary compile policy differs')
        require(row['policy']['repair_flags']==['-w','-fPIC','-ffixed-line-length-132'] and row['policy']['all_original_recipes_preserved'] is True and row['policy']['optimization_experiment'] is False,'Repair flags/scope differ')
        label='results/build/'+variant+'/';pre=load(label+'prelink.json');resolution=aliases[variant]
        require(sha(raw(label+'prelink.json'))==row['prelink_sha256'] and sha(raw(compiled+'private-link.map'))==row['link_map_sha256'],'Link proof seal differs')
        loads=resolution['map_aliases']
        require(re.findall(r'^LOAD\s+(.+)$',text(compiled+'private-link.map'),re.M)==[a['spelling'] for a in loads] and {a['resolved']:a['sha256'] for a in loads}==row['actual_link_loads'] and len(row['actual_link_loads'])==53,'Actual LOAD closure differs')
        require(all(pre['bound'].get(a['resolved'])==a['sha256'] for a in loads),'Link input not prebound')
        runtime=resolution['runtime_aliases'];spellings=[]
        for line in row['runtime']['listing'].splitlines():
            line=line.strip()
            if not line or line.startswith('linux-vdso.so.1 '):continue
            match=re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s+\(0x[0-9a-fA-F]+\)',line)
            require(match is not None,'Malformed historical runtime listing');spellings.append(match[1])
        require(spellings==[a['spelling'] for a in runtime] and {a['resolved']:a['sha256'] for a in runtime}==row['runtime']['files'] and len(row['runtime']['files'])==15,'Actual runtime closure differs')
        require(row['runtime']['files'][row['runtime']['loader']]==row['runtime']['loader_sha256'],'Actual interpreter identity differs')
        require(module_symbols(text(label+'commands/linked-symbols.stdout.txt'))==row['MTIRUST_definitions']==original_symbols,'Raw stock MTIRUST symbols differ')
        private={'ml5_0_host_repair_dp_body_','ml5_0_host_repair_mp_body_'} if variant=='observed' else set()
        names=[line.split()[-1] for line in text(label+'commands/linked-symbols.stdout.txt').splitlines() if len(line.split())>=3]
        require({n for n in names if n.startswith('ml5_0_host_repair_')}==set(row['private_repair_definitions'])==private and all(names.count(n)==1 for n in private),'Private repair ownership differs')
        require(all(names.count(n)==1 for n in ('ml5_0_improve_ps_point_precision_','ml5_0_mp_improve_ps_point_precision_')),'Public repair ABI ownership differs')
        require('libtirust.a(' in text(compiled+'private-link.map'),'Installed tensor wrapper archive missing')
    for phase in ('parity','profile'):
        for lane in LANES:
            variant='baseline' if lane=='baseline' else 'observed';row=build['variants'][variant];prefix='results/'+phase+'/'+lane+'/'
            before,after=load(prefix+'private-before.json'),load(prefix+'private-after.json')
            require(before==after and before['executable_sha256']==row['executable_sha256'],'Worker private bindings differ')
            expected_cards={**pins['numerical_cards'],'SubProcesses/P0_uux_z/MadLoop5_resources/MadLoopParams.dat':next(iter(pins['numerical_cards'].values()))}
            require(before['cards']==expected_cards and before['physical_sha256']==load('inputs/closed-input.json')['param_card_sha256']==sha(raw('donor/Cards/param_card.dat')),'Worker physical/numerical cards differ')
            for which in ('before','after'):
                maps=load(prefix+'maps-'+which+'.json');providers=row['runtime']['files']
                require(all(maps['files'].get(name)==wanted for name,wanted in providers.items()),'Actual mapped provider differs')
                allowed={**providers,**source['allowed_extra_mappings']};extra={name:value for name,value in maps['files'].items() if allowed.get(name)!=value}
                require(len(extra)==1 and next(iter(extra.values()))==row['executable_sha256'],'Unbound mapped executable/extra provider')
                paths={line.split(None,5)[5] for line in maps['maps'].splitlines() if len(line.split(None,5))==6 and line.split(None,5)[5].startswith('/')}
                require(paths==set(maps['files']),'Raw process map paths differ')
            queries=[name for name in commands if name.startswith(prefix+'commands/')]
            require(len(queries)==2,'Wrong worker runtime query coverage')
    require(sha(raw('replay_measurements.py'))==MEASUREMENT_REPLAY_SHA256,'Reviewed pure measurement replay changed')
    measurement=types.ModuleType('capsule_pure_host_repair_measurement')
    exec(compile(text('replay_measurements.py'),'<indexed:pure-repair-measurement>','exec'),measurement.__dict__)
    scientific=measurement.replay(decoded)
    return {'status':'PASS','index_sha256':sha(raw_index),'indexed_text_payloads':len(payloads),'summary':scientific,
            'recorded_BUILD_commands':total_commands,'compiler_invocations':0,'loader_queries':0,'native_evaluations':0,
            'performance_acceptance':False,'speedup_claimed':False,'production_promoted':False,'all_five_backends_qualified':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--expected-index');args=parser.parse_args()
    print(json.dumps(verify(expected_index_sha256=args.expected_index),indent=2,sort_keys=True))
