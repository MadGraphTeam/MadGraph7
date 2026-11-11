"""Pure verification of retained fresh-CLI text; no historical program is run."""
import argparse
import ast
import builtins
from contextlib import contextmanager
import ctypes
import decimal
import difflib
import fractions
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import posixpath
import re
import shlex
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import types
from collections import Counter

sys.dont_write_bytecode=True
PREDECESSOR='893c51f75dedbb6bbb1a70f78bed84b00d97b958ac678c194dc28cbb3aaa6b0b'
GEN_SOURCE='044c90afa64f3c65a6f96a46d67c781b8d643aec39e1e9506611299b3e59021b'
QUAL_SOURCE='c2acf316159ab0c12ab775b4ca92ff0d0eaa122a29db602deb629bae93da4450'
GEN_TERMINAL='3cfd82b4437220bfa21aae773af318457c86ff405bfb69b5ec570a9e8bf7c3c8'
GEN_RESULT='9b295b787f2d0cd4d221bb41c81163e3cfaf06bdd55a1ac1597275b394f69160'
BUILD_TERMINAL='6357ea74762033e7f5b00c897dc415179ce1b2f10c079f60333169ab1512ee41'
BUILD_RESULT='f6af9e96508160be42afc3e88d1b6f559a2fdbfb795834c29cde1637b1dc671d'
RUN_TERMINAL='3984b95e5b981b9b4dabbb7d50a00494a9020b560664433c701ba8fc5b4cba79'
RUN_RESULT='fc79a0217f1e851fa71b9b7faea2731c55592d3bbf93fd1aaa389c74f62fb0f3'
SOURCE_REVIEW='a739240069767725a55cbd143e0c28eeaf62da83fff93a5c1bbaf570412895dd'
ALIASES='177361c2193b67ea9365c0b2fb04e497d0f3a4ac6e969e50dd1d1f2bcef9f712'
REVIEW_PINS={
 'all-source-audit/receipt.json':'a8ac131c6081638d0ff31c9bccc9b08d4ea4e45709838e616a19c66ce96ebcb9',
 'source-review/receipt.json':'21084caa8a058306577b80aed4eeccf5e95c18f289f7397ebf77158e3e118d44',
 'reviews/generation-source-3/receipt.json':'b926246d391faea549360070551876072b5453af744ff1c82e4739d367fa6e8d',
 'reviews/qualification-source-1/receipt.json':'e5df034422c90772bf2b03ee16bd15de1c1eff585cae39a4fd65821a3e51a0ef',
 'reviews/build-primary-1/receipt.json':'f776fb1fa15a754fc08dabfa98185407f12c3cd21eff3f394b7ce01403732e6a',
 'reviews/build-independent-1/receipt.json':'3c5ac77973bdfa1fe7aca0cdea9d9f7b6567f189ba2a36bc1db02e45fdbfef5c',
 'reviews/run-primary-1/receipt.json':'cbcb0da465e4c4c735beb7bb66e95e60569abda7e2a19e3c287fccc80b2b1910',
 'reviews/run-independent-1/receipt.json':'0dbcfde9b49538167c828c7a4f75b1a1fdd3cf396d7a59a8140cba6da9b59909',
 'reviews/run-independent-1/chain-bindings.json':'5d65c4556e3764b85cdcb200f4a091978efb89d3a73de0d4279a2d81ff626d60'}
SCOPE={'model':'loop_sm','process':'u u~ > z g g [virt=QCD]',
       'UseQPIntegrandForTIRust':True,'cache':0,'normal_checked_mode':True,
       'all_five_backends_qualified':False,'EW_qualified':False,
       'loop_induced_qualified':False,'CMS_qualified':False}
SELECTED=('CT_interface.f','TIRUST_interface.f','loop_CT_calls_1.f',
          'coef_construction_1.f','helas_calls_ampb_1.f','helas_calls_uvct_1.f','check_sa.f')
LIMITS={'address_space_bytes':17179869184,'file_bytes':536870912,'core_bytes':0}


def require(value,message):
    if not value:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()
def normalized(value):return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def load_source(name,data,logical):
    module=types.ModuleType(name);module.__file__=logical
    exec(compile(data,logical,'exec'),module.__dict__)
    return module


@contextmanager
def blocked_apis():
    """All mathematical replay consumes memory; effects are always failures."""
    saved=[];attempts=[]
    def deny(*args,**kwargs):
        attempts.append(repr(args[:1]));raise RuntimeError('Forbidden effect in portable replay')
    def patch(obj,name):
        if hasattr(obj,name):saved.append((obj,name,getattr(obj,name)));setattr(obj,name,deny)
    try:
        for name in ('open','stat','lstat','resolve','exists','is_file','is_dir','is_symlink',
                     'iterdir','glob','rglob','read_bytes','read_text','write_bytes','write_text',
                     'mkdir','unlink','rename','replace','touch','chmod','symlink_to','hardlink_to'):
            patch(Path,name)
        patch(builtins,'open');patch(io,'open');patch(io,'open_code')
        patch(io,'FileIO')
        for name in ('open','stat','lstat','access','listdir','scandir','readlink','system','popen',
                     'posix_spawn','posix_spawnp','execv','execve','execvp','execvpe','spawnv','spawnve',
                     'spawnvp','spawnvpe','fork','forkpty','mkdir','makedirs','unlink','remove','rename',
                     'replace','rmdir','removedirs','chmod','chown','truncate','utime','link','symlink',
                     'read','readv','pread','preadv','write','writev','pwrite','pwritev','ftruncate','fchmod'):
            patch(os,name)
        for name in ('Popen','run','call','check_call','check_output'):patch(subprocess,name)
        for name in ('CDLL','PyDLL','_dlopen'):patch(ctypes,name)
        if '_ctypes' in sys.modules:patch(sys.modules['_ctypes'],'dlopen')
        patch(socket,'socket');patch(socket,'create_connection')
        for name in ('copyfile','copy','copy2','copytree','move','rmtree'):patch(shutil,name)
        yield attempts
    finally:
        for obj,name,value in reversed(saved):setattr(obj,name,value)


def preload(root,expected_index_sha256=None):
    root=Path(root).resolve();raw=(root/'index.json').read_bytes()
    if expected_index_sha256:require(sha(raw)==expected_index_sha256,'Capsule index seal changed')
    index=json.loads(raw)
    require(index['schema']=='qp-mass-fresh-cli-portable-qualification-v1' and index['sealed'] is True,
            'Wrong or unsealed capsule')
    expected=set(index['files'])|{'index.json'};actual=set();physical={}
    for path in root.rglob('*'):
        require(not path.is_symlink(),'Capsule aliases forbidden')
        if path.is_file():actual.add(str(path.relative_to(root)))
    require(actual==expected,'Capsule exact file inventory changed')
    for name,row in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe owned path')
        data=(root/name).read_bytes()
        require(len(data)==row['bytes'] and sha(data)==row['sha256'],'Owned payload changed: '+name)
        require(not data.startswith((b'\x7fELF',b'!<arch>\n')),'Binary payload forbidden')
        require(posixpath.splitext(name)[1].lower() not in ('.o','.mod','.a','.so','.dll','.dylib'),
                'Native physical payload suffix forbidden')
        physical[name]=data
    blobs={}
    for name,row in index['logical_files'].items():
        data=physical[row['path']]
        require(row['compression'] in ('none','gzip'),'Unknown text encoding')
        if row['compression']=='gzip':
            require(data[:3]==b'\x1f\x8b\x08' and data[4:8]==b'\0'*4,'Non-deterministic gzip')
            data=gzip.decompress(data)
        require(len(data)==row['raw_bytes'] and sha(data)==row['raw_sha256'],'Decoded payload changed: '+name)
        require(not data.startswith((b'\x7fELF',b'!<arch>\n')),'Decoded native binary forbidden')
        require(posixpath.splitext(name)[1].lower() not in ('.o','.mod','.a','.so','.dll','.dylib'),
                'Native object/module/library payload forbidden')
        data.decode('utf-8');blobs[name]=data
    predecessor=index['predecessor']
    require(predecessor['mode']=='authenticated-subset' and predecessor['index_sha256']==PREDECESSOR,
            'Wrong predecessor/subset policy')
    require(predecessor['whole_parent_numerical_replay_claimed'] is False
            and predecessor['logical_prefix']=='predecessor/'
            and predecessor['index_path']=='predecessor-index.json', 'Predecessor replay scope changed')
    previous_raw=physical[predecessor['index_path']]
    require(sha(previous_raw)==PREDECESSOR,'Pinned predecessor index changed')
    previous=json.loads(previous_raw);selected=predecessor['selected_logical_files']
    require(len(selected)==len(set(selected)),'Duplicate predecessor selection')
    selected_expected={name for name in previous['logical_files'] if name.startswith(
        ('original-cli/','implementation/','source-2/','support/','auxiliary-cli/','stages/cache0/'))}
    selected_expected.update('inputs/docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/points/'+
                             'qcd_n3_uux_zgg.'+extension for extension in ('dat','json'))
    require(len(selected)==394 and set(selected)==selected_expected,'Exact predecessor subset changed')
    prefix=predecessor['logical_prefix'];pred={}
    require({n[len(prefix):] for n in blobs if n.startswith(prefix)}==set(selected),
            'Predecessor subset inventory differs')
    for name in selected:
        old=previous['logical_files'][name];own=index['logical_files'][prefix+name]
        require(own['compression']==old['compression'] and own['raw_bytes']==old['raw_bytes']
                and own['raw_sha256']==old['raw_sha256'],'Predecessor logical identity differs')
        require(sha(physical[own['path']])==previous['files'][old['path']]['sha256']
                and len(physical[own['path']])==previous['files'][old['path']]['bytes'],
                'Predecessor physical encoding differs')
        pred[name]=blobs[prefix+name]
    for old,own in predecessor['physical_files'].items():
        require(sha(physical[own])==previous['files'][old]['sha256']
                and len(physical[own])==previous['files'][old]['bytes'],'Predecessor code identity changed')
    require(set(predecessor['physical_files'])=={'replay.py','math_replay.py'},'Unexpected predecessor code surface')
    return index,physical,blobs,previous,pred


def nested(blobs,prefix,checksum):
    raw=blobs[prefix+'source-index.json'];require(sha(raw)==checksum,'Frozen source seal changed: '+prefix)
    index=json.loads(raw);names={n[len(prefix):] for n in blobs if n.startswith(prefix)}
    require(names==set(index['files'])|{'source-index.json'},'Frozen source inventory differs: '+prefix)
    for name,row in index['files'].items():
        value=blobs[prefix+name]
        require(sha(value)==row['sha256'] and len(value)==row['bytes'],'Frozen source payload changed')
    return index


def source_review(blobs,pred,variants):
    raw=blobs['source-review/source_review.py'];require(sha(raw)==SOURCE_REVIEW,'Independent source parser changed')
    tree=ast.parse(raw);wanted={'require','canon','statements','units','arguments','signature','review'}
    functions={node.name:ast.get_source_segment(raw.decode(),node) for node in tree.body
               if isinstance(node,ast.FunctionDef) and node.name in wanted}
    require(set(functions)==wanted,'Independent parser function inventory differs')
    review=functions.pop('review')
    head=review.index("    report={'schema'")
    review='def review(actual_sources,original_sources,expected_sources):\n'+review[head:]
    changes={"units((actual/name).read_bytes().decode())":"units(actual_sources[name].decode())",
             "units((old/name).read_bytes().decode())":"units(original_sources[name].decode())",
             "units((expected/name).read_bytes().decode())":"units(expected_sources[name].decode())",
             "units((expected/FILES[2]).read_bytes().decode())":"units(expected_sources[FILES[2]].decode())"}
    for before,after in changes.items():
        require(before in review,'Independent parser adapter site changed');review=review.replace(before,after)
    source=('import re\nimport collections\nFILES=("CT_interface.f","TIRUST_interface.f","loop_CT_calls_1.f")\n'+
            '\n\n'.join(functions.values())+'\n\n'+review).encode()
    parser=load_source('fresh_portable_fixed_form',source,'source-review/source_review.py::pure-functions')
    prefix='SubProcesses/P0_uux_zgg/'
    actual={name:blobs['fresh-cli/'+prefix+name] for name in parser.FILES}
    original={name:pred['original-cli/'+prefix+name] for name in parser.FILES}
    expected={name:variants['candidate'][name].encode() for name in parser.FILES}
    report=parser.review(actual,original,expected)
    require(normalized(report)==json.loads(blobs['source-review/source-findings.json']),
            'Explicit fixed-form source review no longer reproduces')
    return report


def parse_recipes(text,identity):
    recipes={}
    for line in text.replace('\\\n',' ').splitlines():
        argv=shlex.split(line)
        if '-c' not in argv:continue
        sources=[a for a in argv if not a.startswith('-') and posixpath.splitext(a)[1].lower()
                 in ('.f','.for','.f90','.f95','.f03','.f08')]
        require(len(sources)==1 and not any(a in {';','&&','||','|','>','>>','<','&'} or a.startswith('@') for a in argv),
                'Ambiguous saved compiler recipe')
        require(argv[0]==identity['path'],'Saved compiler identity differs')
        source=sources[0];flags=[];i=1
        while i<len(argv):
            value=argv[i];i+=1
            if value==source or value=='-c':continue
            if value in ('-o','-I','-J','-isystem','-iquote'):
                require(i<len(argv),'Incomplete compiler option');i+=1;continue
            if value.startswith(('-I','-J','-isystem=','-iquote=')) or (value.startswith('-o') and len(value)>2):continue
            flags.append(value)
        name=posixpath.basename(source);row={'compiler':[identity],'flags':flags}
        require(name not in recipes or recipes[name]==row,'Conflicting repeated recipe')
        recipes[name]=row
    return recipes


def closed_stage(blobs,prefix,terminal_sha,result_sha,source,cap,reviewed=None):
    raw=blobs[prefix+'terminal.json'];require(sha(raw)==terminal_sha,'Frozen terminal changed')
    terminal=json.loads(raw);value=blobs[prefix+'result.json']
    require(sha(value)==result_sha==terminal['result_sha256'],'Frozen result changed')
    result=json.loads(value)
    require(terminal['status']=='passed' and terminal['returncode']==0 and not terminal['cleanup']['survivors']
            and not terminal['cleanup']['events'] and terminal['cap_seconds']==cap
            and terminal['cleanup_reserve_seconds']==60 and terminal['cpu']==12
            and terminal['cpu_reservation'] is False and terminal['resource_limits']==LIMITS,
            'Stage is not successfully and boundedly closed')
    key='source_sha256' if prefix=='generation/' else 'source_index_sha256'
    require(terminal[key]==source and result[key]==source,'Stage source differs')
    if reviewed is not None:require(terminal['reviewed_build']==reviewed,'Reviewed BUILD differs')
    return terminal,result


def raw_files(blobs,prefix,result,extras=()):
    for name,wanted in result['raw_files'].items():
        require(sha(blobs[prefix+name])==wanted,'Raw stage evidence changed: '+prefix+name)
    actual={n[len(prefix):] for n in blobs if n.startswith(prefix)}
    require(actual==set(result['raw_files'])|{'terminal.json','result.json'}|set(extras),
            'Stage raw inventory incomplete or extended')


def build_replay(blobs,pred,qpin,gen,build,alias_bytes):
    raw_files(blobs,'build/',build,extras=('private-link.map',))
    selected=tuple(qpin['selected_sources']);require(selected==SELECTED,'Seven changed units/driver not selected')
    sources={name:row['sha256'] for name,row in gen['source_inventory'].items()}
    prefix='SubProcesses/P0_uux_zgg/'
    for name in (prefix+'check_sa.f',prefix+'makefile'):
        sources[name]=sha(blobs['build-source/'+name])
    require(sources==build['sources'],'BUILD source inventory differs from fresh195 plus two controlled overrides')
    for name in sources:
        value=blobs['build-source/'+name] if name in (prefix+'check_sa.f',prefix+'makefile') else blobs['fresh-cli/'+name]
        require(sha(value)==sources[name],'BUILD source bytes missing')
    require(blobs['build-source/'+prefix+'check_sa.f']==blobs['qualification-source/check_sa.f'],
            'Qualified benchmark driver changed')
    policy=load_source('fresh_portable_build_policy',blobs['qualification-source/policy.py'],
                       'qualification-source/policy.py')
    require(blobs['build-source/'+prefix+'makefile']==policy.private_makefile(
        blobs['fresh-cli/'+prefix+'makefile'].decode()).encode(), 'Controlled BUILD makefile no longer derives')
    oldcase=qpin['case'];fresh=qpin['fresh_objects'];reuse=qpin['reuse_subprocess_objects']
    replaced={prefix+posixpath.splitext(name)[0]+'.o' for name in selected}
    require(len(fresh)==122 and len(reuse)==15 and len(replaced)==7 and len(build['objects'])==144,
            'Numerical object closure counts differ')
    require(set(fresh)|set(reuse)|replaced==set(build['objects'])==set(oldcase['numerical_objects']),
            'Numerical object keys differ')
    for name,wanted in oldcase['numerical_objects'].items():
        if name not in replaced:require(build['objects'][name]==wanted,'Unselected numeric object changed')
    require(all(oldcase['numerical_objects'][name]==wanted for name,wanted in fresh.items()),
            'Actual GEN3 fresh numeric object identities differ')
    generated_snapshot=json.loads(blobs['source-review/output-file-hashes.json'])['files']
    require(all(generated_snapshot[name]['sha256']==wanted for name,wanted in fresh.items()),
            'Fresh object identities differ from retained actual GEN3 snapshot')
    require(sum(n.startswith('Source/') and n.endswith('.o') for n in fresh)==112,'Fresh Source object scope changed')
    for name,row in reuse.items():
        require(row['sha256']==oldcase['numerical_objects'][name] and row['source_sha256']==
                oldcase['generated_sources'][row['source']]==qpin['fresh_sources'][row['source']]
                and posixpath.basename(row['source'])==row['recipe_name'], 'Unmodified object source/recipe join differs')
    for phase in ('before','after'):
        version=blobs['build/commands/policy-'+phase+'-02.stdout.txt'].decode().strip()
        tool=json.loads(blobs['qualification-source/toolchain-pins.json'])['required_tools']['gfortran']
        identity={'path':tool['resolved'],'sha256':tool['sha256'],'version':version}
        for number,scope,count in ((1,'process',23),(3,'MODEL',14),(4,'DHELAS',99)):
            recipes=parse_recipes(blobs[f'build/commands/policy-{phase}-{number:02d}.stdout.txt'].decode(),identity)
            require(len(recipes)==count and recipes==build['compiler_flags']['scopes'][scope]['recipes']
                    ==oldcase['compiler_flags']['scopes'][scope]['recipes'], 'Saved effective compilation policy differs')
    require(build['compiler_flags']['environment']==oldcase['compiler_flags']['environment']=={},'Compiler injection changed')
    commands={n[len('build/commands/'):-len('.command.json')]:json.loads(v) for n,v in blobs.items()
              if n.startswith('build/commands/') and n.endswith('.command.json')}
    require(len(commands)==70,'Recorded BUILD command scope differs')
    for name,command in commands.items():
        terminal=json.loads(blobs['build/commands/'+name+'.terminal.json'])
        require(terminal['returncode']==0,'Recorded BUILD command failed')
        require(sha(blobs['build/commands/'+name+'.stdout.txt'])==terminal['stdout_sha256'],'Recorded stdout changed')
        if 'stderr_sha256' in terminal:
            require(sha(blobs['build/commands/'+name+'.stderr.txt'])==terminal['stderr_sha256'],'Recorded stderr changed')
    require([name for name,_ in build['selected_recipes']]==list(selected),'Selected compiler order differs')
    for i,(name,argv) in enumerate(build['selected_recipes']):
        require(commands[f'{i:02d}-compile-'+name]['argv']==argv,'Actual selected compile changed')
        recipe=parse_recipes(shlex.join(argv),identity)
        require(recipe[name]['flags']==['-w','-fPIC','-ffixed-line-length-132'],'Selected original flags changed')
    pre=json.loads(blobs['build/prelink.json'])
    require(sha(blobs['build/prelink.json'])==build['prelink_sha256'] and pre['completed_before_link'] is True,
            'Prelink binding differs')
    require(commands['link']['argv']==build['link_command']==pre['command'],'Actual link differs')
    require(len(pre['bound'])==87 and pre['explicit_current_runtime_before_link']==build['runtime']['files']
            ==qpin['expected_runtime']['files'] and len(build['runtime']['files'])==15,'Runtime/prelink closure differs')
    require(all(pre['bound'].get(name)==wanted for name,wanted in build['runtime']['files'].items())
            and build['runtime']['loader_sha256']==build['runtime']['files'][build['runtime']['loader']],
            'Runtime providers/interpreter are not all prebound')
    require(all(pre['bound'].get(name)==wanted for name,wanted in build['actual_link_loads'].items())
            and len(build['actual_link_loads'])==54,'Actual LOAD closure escaped prebinding')
    require(sha(blobs['build/private-link.map'])==build['link_map_sha256'],'Raw linker map changed')
    require(sha(alias_bytes)==ALIASES,'Read-only path-alias corroboration changed')
    aliases=json.loads(alias_bytes)
    # Aliases are authenticated retained file-only corroboration, not live resolution.
    aliasrows=aliases.get('aliases',aliases)
    loads=[]
    for line in blobs['build/private-link.map'].decode().splitlines():
        if line.startswith('LOAD '):
            spelling=line[5:].strip();absolute=posixpath.normpath(spelling if spelling.startswith('/') else posixpath.join(pre['cwd'],spelling))
            row=aliasrows.get(spelling,aliasrows.get(absolute));require(row and row['absolute']==absolute,'LOAD spelling alias differs')
            resolved=row['resolved']
            require(pre['bound'].get(resolved)==row['sha256'],'LOAD alias not prebound');loads.append(resolved)
    require(len(loads)==80 and set(loads)==set(build['actual_link_loads']),'Raw LOAD derivation differs')
    for name,wanted in build['actual_link_loads'].items():require(pre['bound'][name]==wanted,'LOAD hash changed')
    runtime=build['runtime'];listing=blobs['build/commands/actual-pt-interp-runtime.stdout.txt'].decode()
    require(listing==runtime['listing'],'Saved runtime listing changed')
    observed={}
    for line in listing.splitlines():
        line=line.strip()
        if not line or line.startswith('linux-vdso.so.1 '):continue
        match=re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s*\(0x[0-9a-fA-F]+\)',line)
        require(match,'Malformed loader listing')
        row=aliasrows[posixpath.normpath(match[1])];observed[row['resolved']]=row['sha256']
    require(observed==runtime['files'],'Saved loader provider derivation differs')
    require(commands['actual-pt-interp-runtime']['argv']==[runtime['loader'],'--list',pre['cwd']+'/check']
            and runtime['loader']==qpin['expected_runtime']['actual_interpreter'], 'Actual PT_INTERP query differs')
    symbols=blobs['build/commands/symbols.stdout.txt'].decode().splitlines();mt=Counter();private=Counter()
    for line in symbols:
        words=line.split()
        if not words:continue
        if '__mtirust_MOD_' in words[-1]:mt[words[-1]]+=1
        if '_qpmass_' in words[-1]:private[words[-1]]+=1
    require(mt==Counter({name:1 for name in qpin['expected_mtirust_symbols']}) and len(mt)==23,
            'Installed wrapper definitions changed/duplicated')
    require(private==Counter({name:1 for name in qpin['expected_private_symbols']}) and len(private)==9,
            'Private QP definitions changed/duplicated')
    require(build['native_amplitude_calls']==0 and build['candidate_promoted'] is False,
            'BUILD numerical/publication scope changed')
    return {'sources':195,'numeric_bindings':144,'compiles':7,'recorded_commands':70,
            'raw_LOAD_records':80,'actual_LOADs':54,'prebound_files':87,'runtime_providers':15,'MTIRUST_definitions':23,'private_definitions':9}


def verify(root,expected_index_sha256=None):
    index,physical,blobs,previous,pred=preload(root,expected_index_sha256)
    require(index['scope']==SCOPE,'Qualification scope changed')
    for name,value in (('candidate_promoted',False),('performance_acceptance',False),
        ('independent_physical_finite_truth_qualified',False),('native_binaries_packaged',0),
        ('new_native_executions',0),('public_writes',False)):
        require(index[name] is value if isinstance(value,bool) else index[name]==value,'Contradictory claim: '+name)
    with blocked_apis() as attempts:
        oldcode=index['predecessor']['physical_files']
        old=load_source('fresh_portable_predecessor',physical[oldcode['replay.py']],'predecessor-replay.py')
        mathcode=physical[oldcode['math_replay.py']]
        math_replay=load_source('fresh_portable_mass_math',mathcode,'predecessor-math_replay.py')
        nested(blobs,'generation-source/',GEN_SOURCE);nested(blobs,'qualification-source/',QUAL_SOURCE)
        bootstrap=json.loads(blobs['generation-source/bootstrap-static-report.json'])
        gpins=json.loads(blobs['generation-source/input-pins.json'])
        tracked=json.loads(blobs['generation-source/tracked-bootstrap-tree.json'])
        require(bootstrap['visited_count']==129 and bootstrap['missing_count']==0 and not bootstrap['errors']
                and bootstrap['nonliteral_import_count']==29,'Historical static bootstrap scope changed')
        owned_bootstrap={name[len('bootstrap-sources/'):]:value for name,value in blobs.items()
                         if name.startswith('bootstrap-sources/')}
        require(set(owned_bootstrap)==set(bootstrap['visited_files'])|{'PLUGIN/__init__.py'},
                'Retained bootstrap source body set differs')
        for name,value in owned_bootstrap.items():
            row=(bootstrap['visited_files'].get(name) or gpins['bootstrap_files'][name])
            require(sha(value)==row['sha256'] and len(value)==row['bytes'],'Bootstrap source body changed')
            ast.parse(value)
        require(len(gpins['bootstrap_files'])==12,'Additional bootstrap inventory changed')
        for name,row in gpins['bootstrap_files'].items():
            value=owned_bootstrap[name]
            require(hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest()==row['git_blob_sha1']
                    ==tracked[name]['git_blob_sha1'] and row['anchor']==
                    '0fcd2cd873d61d134837e1d9e87afb13a47824fd','Frozen Git bootstrap binding changed')
        require(previous['auxiliary_CLI_status']=='failed-startup-unqualified'
                and previous['fresh_CLI_correspondence_qualified'] is False,'Historical failure scope relabelled')
        failure=json.loads(pred['auxiliary-cli/execution-2/terminal.json'])
        require(failure['status']=='failed' and failure['returncode']==1 and not failure['cleanup']['survivors']
                and sha(pred['auxiliary-cli/execution-2/terminal.json'])==
                '61b000fbf56ab27dd32f37accf37ce69fc73635254ceb76fcc9a0c5616a47304', 'GEN2 failure changed')
        require(b"ModuleNotFoundError: No module named 'mg5decay'" in pred['auxiliary-cli/execution-2/generation.log'],
                'GEN2 failure witness absent')
        variants,_,proof=old.source_derivation(pred)
        require(blobs['qualification-source/check_sa.f']==variants['candidate']['check_sa.f'].encode(),
                'Benchmark driver derivation differs')
        _,gen=closed_stage(blobs,'generation/',GEN_TERMINAL,GEN_RESULT,GEN_SOURCE,300)
        require(gen['status']=='generated_pending_source_review' and gen['amplitude_evaluations']==0
                and gen['actual_CLI_invocations']==1 and gen['stock_Model_DHELAS_build_objects']==112,
                'Historical generation scope/status changed')
        source_receipt=json.loads(blobs['source-review/receipt.json'])
        generation_raw=source_receipt['raw_execution_files']
        require(len(generation_raw)==11 and {n[len('generation/'):] for n in blobs
                if n.startswith('generation/')}==set(generation_raw), 'Actual CLI raw inventory differs')
        for name,row in generation_raw.items():
            value=blobs['generation/'+name]
            require(sha(value)==row['sha256'] and len(value)==row['bytes'], 'Pinned actual CLI evidence changed')
        generation_command=json.loads(blobs['generation/generation.command.json'])
        generation_command_terminal=json.loads(blobs['generation/generation.command-terminal.json'])
        historical_root=generation_command['cwd'].split('/.tirust-build/')[0]
        generation_tools=json.loads(blobs['generation-source/toolchain-pins.json'])['required_tools']
        require(generation_command['argv']==[generation_tools['python3']['path'],'-B','./bin/madgraph','--plain',
                historical_root+'/.tirust-build/qp-mass-cli-generation-proposal-3/generate.mg5']
                and generation_command['cwd']==historical_root+'/.tirust-build/qp-mass-cli-generation-work-3/mg-current-qpmass'
                and generation_command_terminal['returncode']==0
                and generation_command_terminal['log_sha256']==sha(blobs['generation/generation.log'])
                and generation_command['make_internal_children_individually_recorded'] is False,
                'Actual CLI command/closure differs')
        qpin=json.loads(blobs['qualification-source/input-pins.json']);oldcase=qpin['case']
        all_fresh={n[len('fresh-cli/'):]:data for n,data in blobs.items() if n.startswith('fresh-cli/')}
        require(set(all_fresh)==set(gen['source_inventory'])|{'Cards/MadLoopParams.dat','Cards/param_card.dat'},
                'Fresh CLI source/card inventory changed')
        require(sha(all_fresh['Cards/param_card.dat'])==gen['physical_card_sha256']==oldcase['param_card_sha256']
                and sha(all_fresh['Cards/MadLoopParams.dat'])==gen['numerical_card_sha256']
                and all_fresh['Cards/MadLoopParams.dat']==blobs['qualification-source/cards/cache0.dat'],
                'Pinned actual physical/cache0 cards differ')
        fresh={n:all_fresh[n] for n in gen['source_inventory']}
        require(len(fresh)==195 and set(fresh)==set(gen['source_inventory'])==set(oldcase['generated_sources']),
                'Complete195-source scope changed')
        compiler_settings={}
        for line in blobs['generation-source/generate.mg5'].decode().splitlines():
            words=shlex.split(line)
            if len(words)==4 and words[0]=='set' and words[-1]=='--no_save':compiler_settings[words[1]]=words[2]
        make_options=pred['original-cli/Source/make_opts'].decode()
        for key,before,setting in (('DEFAULT_CPP_COMPILER','g++','cpp_compiler'),
                                    ('DEFAULT_F_COMPILER','gfortran','fortran_compiler')):
            old=key+'='+before+'\n';require(make_options.count(old)==1,'Original compiler selector is ambiguous')
            make_options=make_options.replace(old,key+'='+compiler_settings[setting]+'\n',1)
        make_policy=json.loads(blobs['generation/make-opts-policy.json'])
        require(make_options.encode()==fresh['Source/make_opts'] and make_policy['all_other_bytes_exact'] is True
                and make_policy['original_sha256']==sha(pred['original-cli/Source/make_opts'])
                and make_policy['actual_sha256']==sha(fresh['Source/make_opts']), 'Actual compiler selector derivation differs')
        changed=[];inventory=json.loads(blobs['all-source-audit/inventory.json'])
        inventory_receipt=json.loads(blobs['all-source-audit/receipt.json'])
        require(set(inventory)==set(fresh),'All195 delta keys differ')
        for name,row in inventory_receipt['files'].items():
            value=blobs['all-source-audit/'+name]
            require(sha(value)==row['sha256'] and len(value)==row['bytes'],
                    'Pinned all-source audit artifact changed')
        for name,data in fresh.items():
            require(sha(data)==gen['source_inventory'][name]['sha256']==qpin['fresh_sources'][name]
                    and len(data)==gen['source_inventory'][name]['bytes'],'Actual generated source changed')
            before=pred['original-cli/'+name];row=inventory[name]
            base={'actual_bytes':len(data),'actual_sha256':sha(data),'original_bytes':len(before),
                  'original_sha256':sha(before),'raw_equal':data==before}
            require(all(row.get(k)==v for k,v in base.items()),'All195 delta inventory differs')
            if data!=before:
                changed.append(name);diff=row['raw_diff'];value=blobs['all-source-audit/'+diff['path']]
                require(sha(value)==diff['sha256'] and len(value)==diff['bytes'], 'Raw source diff hash differs')
                regenerated=''.join(difflib.unified_diff(before.decode().splitlines(keepends=True),
                    data.decode().splitlines(keepends=True),fromfile='original/'+name,tofile='GEN3/'+name)).encode()
                require(value==regenerated,'Raw source diff no longer derives from actual bytes')
            else:require(row==base,'Unexpected metadata for unchanged source')
        expected_changed=['Source/make_opts']+['SubProcesses/P0_uux_zgg/'+name for name in
            ('CT_interface.f','TIRUST_interface.f','check_sa.f','coef_construction_1.f',
             'helas_calls_ampb_1.f','helas_calls_uvct_1.f','loop_CT_calls_1.f','makefile')]+['SubProcesses/makefile']
        require(sorted(changed)==sorted(expected_changed),'Changed translation-unit scope differs')
        for name in SELECTED[3:6]:
            path='SubProcesses/P0_uux_zgg/'+name;before=pred['original-cli/'+path];after=fresh[path]
            extra=b"      INCLUDE 'coupl.inc'\n      REAL*8 LSCALE\n      INTEGER CTMODE\n      COMMON/ML5_0_CT/LSCALE,CTMODE\n"
            require(after.count(extra)==1 and after.replace(extra,b"      INCLUDE 'coupl.inc'\n",1)==before,
                    'Additional split unit changed beyond exact three COMMON declarations')
        review=source_review(blobs,pred,variants)
        compare=load_source('fresh_portable_compare',blobs['generation-source/compare.py'],'generation-source/compare.py')
        p='SubProcesses/P0_uux_zgg/'
        fields=('CT_interface.f','TIRUST_interface.f','loop_CT_calls_1.f')
        actual_compare=compare.compare_sources({n:pred['original-cli/'+p+n].decode() for n in fields},
            {n:fresh[p+n].decode() for n in fields},
            {n:variants['original'][n] for n in fields},{n:variants['candidate'][n] for n in fields})
        require(normalized(actual_compare)==json.loads(blobs['generation/source-comparison.json'])
                and not actual_compare['strict_pass'],'Raw source-comparison history changed')
        _,build=closed_stage(blobs,'build/',BUILD_TERMINAL,BUILD_RESULT,QUAL_SOURCE,600)
        build_summary=build_replay(blobs,pred,qpin,gen,build,physical['build-path-aliases.json'])
        build_directory=json.loads(blobs['build/prelink.json'])['cwd']
        build_root=posixpath.dirname(posixpath.dirname(build_directory))
        excluded=[{'path':posixpath.join(build_root,name),'role':'actual BUILD numerical object/module/archive',
                   'sha256':wanted} for name,wanted in build['objects'].items()]
        excluded.append({'path':build_directory+'/check','role':'actual BUILD/RUN executable',
                         'sha256':build['executable_sha256']})
        excluded.extend({'path':name,'role':'observed runtime provider','sha256':wanted}
                        for name,wanted in build['runtime']['files'].items())
        require(len(index['excluded_binaries'])==160 and sorted(index['excluded_binaries'],key=lambda r:(r['path'],r['role']))
                ==sorted(excluded,key=lambda r:(r['path'],r['role'])), 'Excluded native identities differ from frozen BUILD')
        _,run=closed_stage(blobs,'run/',RUN_TERMINAL,RUN_RESULT,QUAL_SOURCE,600,BUILD_TERMINAL)
        raw_files(blobs,'run/',run);require(len(run['raw_files'])==20,'RUN raw scope changed')
        baseline=json.loads(pred['stages/cache0/result.json']);codes=oldcase['expected_codes']
        analysis=load_source('fresh_portable_base_analysis',pred['source-2/base_analysis.py'],'source-2/base_analysis.py')
        exact=analysis.exact(blobs['run/candidate/exact-words.dat'].decode(),codes)
        flags=analysis.ieee(blobs['run/candidate/exact-ieee.dat'].decode())
        software=math_replay._software(blobs['run/candidate/software-flags.dat'].decode())
        learning,_,_=math_replay._learning_rows(blobs['run/candidate/warmup.dat'].decode(),codes)
        learning_words=math_replay._words(learning);history=math_replay._history(blobs['run/candidate/warmup.dat'].decode())
        require(len(history)==202 and history==baseline['learning']['candidate']['history']==run['learning']['history'],
                'Full100 learning/bootstrap history changed')
        require(learning_words==run['learning']['rows']==baseline['learning']['candidate']['rows'],
                'Steady full100 learning words changed')
        require(exact==run['lane']['rows']==baseline['lanes']['candidate']['rows'] and len(exact)==200,
                'All final coefficient/accuracy/status words no longer exact')
        require(flags==run['lane']['ieee']==baseline['lanes']['candidate']['ieee'] and
                software==run['lane']['software']==baseline['lanes']['candidate']['software'],
                'Raw IEEE/software observations changed')
        for name,oldname in (('exact-words.dat','exact-words.dat'),('exact-ieee.dat','exact-ieee.dat'),
                             ('software-flags.dat','software-flags.dat')):
            require(blobs['run/candidate/'+name]==pred['stages/cache0/candidate/exact/'+oldname],
                    'Final baseline raw bytes changed')
        require(sum(row['status']==219 for row in exact)==174 and sum(row['status']==323 for row in exact)==26,
                'Checked87DP/13QP cohort changed')
        require(run['comparison']['passed'] and run['comparison']['DP_exact'] and
                run['comparison']['QP_word_accuracy_differences']==[],'Scientific gate summary changed')
        comparator_source=pred['support/tools/tirust_benchmark.py'].decode()
        comparator_nodes=[node for node in ast.parse(comparator_source).body
                          if isinstance(node,ast.FunctionDef) and node.name=='compare_rows']
        require(len(comparator_nodes)==1,'Missing existing coefficient criterion')
        comparator=load_source('fresh_existing_coefficient_criterion',
            ast.get_source_segment(comparator_source,comparator_nodes[0]).encode(),
            'support/tools/tirust_benchmark.py::compare_rows').compare_rows
        metric=analysis.compare_variants(baseline['lanes']['candidate']['rows'],exact,comparator)
        require(metric['passed'],'Existing3e-6 or hard1e-8 criterion failed on raw final rows')
        for phase in ('before','after'):
            maps=json.loads(blobs['run/candidate/maps-'+phase+'.json']);expected=dict(build['runtime']['files'])
            exe=posixpath.join(qpin['fresh_output'].split('/.tirust-build/')[0],
                '.tirust-build/qp-mass-fresh-cli-qualification-work-1/run/candidate/SubProcesses/P0_uux_zgg/check')
            expected[exe]=build['executable_sha256']
            require(maps['files']==expected and len(expected)==16,'Observed initialized mapping closure changed')
            observed=set()
            for line in maps['maps'].splitlines():
                fields=line.split(None,5)
                if len(fields)==6 and fields[-1].startswith('/'):
                    require(not fields[-1].endswith(' (deleted)'),'Deleted file mapping')
                    observed.add(posixpath.normpath(fields[-1]))
            require(observed==set(expected),'Raw initialized mapping names differ')
        require(run['candidate_promoted'] is False and run['performance_acceptance'] is False
                and run['independent_physical_finite_truth_qualified'] is False,'RUN scope relabelled')
        require(all(index.get('review_pins',{}).get(name)==wanted for name,wanted in REVIEW_PINS.items()),
                'Required review/chain references missing or retagged')
        for name,wanted in index['review_pins'].items():require(sha(blobs[name])==wanted,'Review/chain pin changed')
        source_receipt=json.loads(blobs['source-review/receipt.json'])
        for name,wanted in source_receipt['artifacts'].items():
            require(sha(blobs['source-review/'+name])==wanted,'Pinned source-review artifact changed')
        require(not attempts,'Replay attempted a forbidden effect')
    return {'status':'passed','scope':SCOPE,'predecessor_mode':'authenticated-subset',
        'predecessor_full_campaign_replayed':False,'GEN2_status':'failed-startup-unqualified',
        'generation_historical_status':'generated_pending_source_review','source':review['counts'],
        'all_source_inputs':195,'raw_identical_inputs':185,'build':build_summary,
        'learning_history_rows':202,'learning_bootstrap_rows_unqualified':2,'steady_learning_rows':200,
        'final_rows':200,'DP_points':87,'QP_points':13,'coefficient_words':800,'accuracy_words':200,
        'coefficient_word_differences':0,'accuracy_word_differences':0,'status_differences':0,
        'observed_map_files':16,'runtime_providers':15,'locale_archive_observed':False,
        'bootstrap_source_bodies':130,'bootstrap_additional_Git_bindings':12,
        'bootstrap_scope':'Retained report/source hashes and AST syntax; dynamic import edges are not executed or independently rederived',
        'forbidden_API_attempts':0,'new_native_executions':0,'candidate_promoted':False,
        'performance_acceptance':False,'independent_physical_finite_truth_qualified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--index-sha256')
    args=parser.parse_args();result=verify(Path(__file__).resolve().parent,args.index_sha256)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':main()
