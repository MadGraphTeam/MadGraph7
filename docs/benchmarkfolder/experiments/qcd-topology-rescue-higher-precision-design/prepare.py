#!/usr/bin/env python3
"""Derive a source-only proposal for reusing an existing private512 DSO."""
from pathlib import Path
import hashlib
import json
import re

from elf_metadata import inspect

ROOT=Path.cwd().resolve()
HERE=Path(__file__).resolve().parent
NATIVE=ROOT/'docs/benchmarkfolder/experiments/qcd-topology-rescue-native'
EW=ROOT/'docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/higher-precision'
NATIVE_INDEX='c7d185c4991fdfacf5c76710d1790b7ccf31c5c2ec0f9270cc4f72ec3dfdfc32'
BUILD512='adad13cb1e0f8664468dc6afa9f3776b23de37938b202f309d5281cfb93d2873'
OLD_DSO='49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12'
NEW_DSO='e8b65f86b78f77d8090e7c406274da32920d89279ee4d5d3ef40e518fb05afdc'
SOURCE256='1eab09d61652e4dfade900a22f8c63a071b4fecee2e7454e8144f934f17e34e0'
SOURCE512='59184358f8c2f91b1ac38397e76f8accc6e4ffce8aaa529bdde2acbaee985871'


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    path=Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read(path):
    return json.loads(Path(path).read_text())


def derive():
    pins={}

    def pin(path,digest=None):
        path=Path(path)
        actual=sha(path)
        require(digest is None or actual==digest,'Frozen input changed: '+str(path))
        name=rel(path)
        require(name not in pins or pins[name]==actual,'Conflicting frozen pin: '+name)
        pins[name]=actual
        return actual

    pin(NATIVE/'evidence-index.json',NATIVE_INDEX)
    entries=read(NATIVE/'evidence-index.json')['files']
    require(len(entries)==73,'Unexpected native inventory.')
    for name,entry in entries.items():
        path=NATIVE/name
        require(not path.is_symlink() and not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe native inventory.')
        require(path.stat().st_size==entry['bytes'],'Native evidence size changed.')
        pin(path,entry['sha256'])
    pin(EW/'build.json',BUILD512)
    build512=read(EW/'build.json')
    require(build512['complete'] and build512['returncode']==0 and build512['protected_inputs_unchanged'] and
            build512['protected_inputs_before']==build512['protected_inputs_after'],'Existing512 build is incomplete.')
    require(build512['library_sha256']==NEW_DSO and build512['source_before_sha256']==SOURCE256 and
            build512['source_after_sha256']==SOURCE512 and build512['input_bits']==113 and build512['working_bits']==512 and
            build512['gate']=='2^-392','Existing512 build policy differs.')
    source256=ROOT/'.tirust-build/bridge-poles/src/lib.rs'
    source512=ROOT/'.tirust-build/ew-threshold-512/src/lib.rs'
    pin(source256,SOURCE256);pin(source512,SOURCE512)
    original=source256.read_text()
    require(original.count('const MP_BITS: u32 = 256;')==1 and original.count('Float::with_val(MP_BITS, 2).pow(136).inv()')==2,'Unexpected historical precision policy.')
    changed=original.replace('const MP_BITS: u32 = 256;','const MP_BITS: u32 = 512;')
    changed=changed.replace('Float::with_val(MP_BITS, 2).pow(136).inv()',
                            'Float::with_val(MP_BITS, 2).pow(u64::from(MP_BITS - 120)).inv()')
    changed=changed.replace('// 256-bit arithmetic with substantial headroom above binary128.',
                            '// 512-bit diagnostic arithmetic; input binary128 semantics unchanged.')
    require(changed==source512.read_text(),'Private512 source has changes beyond the declared arithmetic/guard/comment.')
    pin(EW/'build.log',build512['build_log_sha256'])
    # Bind exact direct dependencies and the source files governing the guard;
    # unrelated historical release-cache outputs are retained in build.json,
    # not incorrectly assumed to remain immutable forever.
    direct_dependencies={}
    for token in build512['command']:
        if token.startswith(('iregi_rust=','symbolica=')):
            path=Path(token.split('=',1)[1])
            expected=build512['protected_inputs_before'].get(rel(path))
            require(expected is not None,'Direct dependency absent from historical build pins.')
            pin(path,expected);direct_dependencies[rel(path)]=expected
    require(len(direct_dependencies)==2,'Unexpected direct Rust dependencies.')
    guard=read(NATIVE/'analysis1/guard-source-audit.json')
    for entry in guard['source_identities'].values():
        pin(ROOT/entry['path'],entry['sha256'])
    original_lib=ROOT/'HEPTools/tirust/lib/libmadloop_tirust.so'
    private_lib=ROOT/build512['runtime_library']
    pin(original_lib,OLD_DSO);pin(private_lib,NEW_DSO)
    records={}
    for backend in (9,8):
        directory=NATIVE/'attempt1/topology-c'/f'backend-{backend}'
        record=read(directory/'execution.json')
        require(record['complete'] and record['protected_inputs_unchanged'] and record['runtime_sha256_unchanged'],'Incomplete topology-c control.')
        require(record['protected_inputs_before']==record['protected_inputs_after'] and record['runtime_sha256_before']==record['runtime_sha256_after'],'Historical control drift.')
        for field in ('protected_inputs_before','runtime_sha256_before','resource_sha256_before'):
            for name,digest in record[field].items():
                pin(ROOT/name,digest)
        records[backend]=record
    record=records[9]
    executable=Path(record['command'][3])
    # command = taskset -c 11 executable param points resources trace.
    require(record['command'][:3]==['taskset','-c','11'],'Control CPU/command shape changed.')
    require(executable.name=='amplitude-private','Unexpected control executable.')
    original_metadata=inspect(original_lib)
    private_metadata=inspect(private_lib)
    executable_metadata=inspect(executable)
    require(original_metadata['sha256']==OLD_DSO and private_metadata['sha256']==NEW_DSO,'ELF hash mismatch.')
    old=original_metadata['metadata'];new=private_metadata['metadata'];exe=executable_metadata['metadata']
    require(old['exported_ml_tirust_symbols']==new['exported_ml_tirust_symbols']==build512['abi_exports'],'Public ABI differs.')
    require(set(new['version_requirements']) <= set(old['version_requirements']), 'Private DSO needs a new versioned runtime library.')
    for library,versions in new['version_requirements'].items():
        require(set(versions) <= set(old['version_requirements'][library]),'Private DSO needs a new runtime symbol version.')
    removed_versions={library:sorted(set(versions)-set(new['version_requirements'].get(library,[])))
                      for library,versions in old['version_requirements'].items()}
    removed_versions={library:versions for library,versions in removed_versions.items() if versions}
    require(removed_versions=={'libc.so.6':['GLIBC_2.32']},'Unexpected version-requirement difference.')
    require(new['soname']=='libmadloop_tirust.so' and 'libmadloop_tirust.so' in exe['needed'],'Loader SONAME/NEEDED differs.')
    require(exe['runpath'] and not exe['rpath'],'Executable does not permit controlled RUNPATH substitution.')
    pin(executable,executable_metadata['sha256'])
    next_work=ROOT/'.tirust-build/qcd-topology-rescue-higher-precision/attempt1'
    next_out=ROOT/'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-native/attempt1'
    env=dict(record['execution_environment'])
    old_prefix=str(original_lib.parent)
    require(env['LD_LIBRARY_PATH'].split(':')[0]==old_prefix,'Unexpected historical loader environment.')
    env['LD_LIBRARY_PATH']=str(private_lib.parent)+':'+env['LD_LIBRARY_PATH']
    env['TMPDIR']=str(next_work/'tmp')
    resources=Path(record['command'][6])
    resource_pins={name:sha(resources/name) for name in ('MadLoopParams.dat','ML5_0_ColorNumFactors.dat','ML5_0_ColorDenomFactors.dat','ML5_0_HelConfigs.dat')}
    parameters=dict(re.findall(r'^#(\w+)\s*\n([^!#\n]+)',(resources/'MadLoopParams.dat').read_text(),flags=re.M))
    parameters={name:value.strip() for name,value in parameters.items()}
    require(parameters['TIRustGlobalCache']=='-1' and parameters['TIRustCacheMaxBytes']=='268435456','Unexpected unchanged cache policy.')
    expected_runtime=dict(record['runtime_sha256_before'])
    require(expected_runtime.pop(str(original_lib.resolve()))==OLD_DSO,'Missing original runtime dependency.')
    expected_runtime[str(private_lib.resolve())]=NEW_DSO
    command=list(record['command'])
    command[6]=str(next_work/'MadLoop5_resources');command[7]=str(next_out/'trace.dat')
    return {
        'schema':1,'scope':'Source-only bounded diagnostic proposal; not execution approval or physical acceptance.',
        'native_authorized':False,'native_executed':False,'compiler_or_linker_required':False,'production_changes':False,
        'proposed_processes':1,'backend':9,'arm':'topology-c','sequence':'ordinary/stress/ordinary',
        'cpu':11,'thread_count':1,'seconds_per_process_cap':180,'automatic_retry_count':0,
        'shared_lock':'.tirust-build/numerical-tests.lock','formal_performance_eligible':False,
        'source_changes':{'before_sha256':SOURCE256,'after_sha256':SOURCE512,'working_bits':[256,512],
                          'input_bits_unchanged':113,'guard_exponents':[-136,-392],
                          'functional_replacements':3,'comment_replacements':1,
                          'lightlike_gate_unchanged':'2^-100','affine_relation_tolerances_unchanged':'fixed binary_tolerance40/33/80',
                          'all_other_bridge_source_bytes_unchanged':True},
        'elf':{'historical256':original_metadata,'private512':private_metadata,'unchanged_executable':executable_metadata,
               'additional_private_runtime_versions':{},'historical_only_runtime_versions':removed_versions},
        'existing_build':{'path':rel(EW/'build.json'),'sha256':BUILD512,'historical_protected_count':len(build512['protected_inputs_before']),
                          'historical_before_equals_after':True,'direct_dependencies':direct_dependencies},
        'controls':{str(backend):{'path':rel(NATIVE/'attempt1/topology-c'/f'backend-{backend}'/'execution.json'),
                                'sha256':sha(NATIVE/'attempt1/topology-c'/f'backend-{backend}'/'execution.json'),
                                'host_stress_code':records[backend]['point_summaries'][1]['host_code'],
                                'trace_summary':records[backend]['trace_summary']} for backend in (9,8)},
        'runtime':{'command':command,'environment':env,'expected_loaded_files_sha256':expected_runtime,
                   'resource_source':str(resources),'resource_copies_sha256':resource_pins,'numerical_parameters':parameters,
                   'executable_bytes_unchanged':True,'fresh_process_context':True,'cache_policy_changed':False,
                   'old512_tensor_replays_used_cache_disabled':True,'actual_process_maps_required':True,
                   'no_preload_or_loader_audit_hook':True,'caller_fpe_policy_unchanged':True},
        'required_preflight':['Independent controller review and explicit root execution authorization.',
                              'Verify this frozen design plus both source/DSO/executable/runtime/card/input hashes.',
                              'Use a fresh one-shot result/work directory; copy only pinned resource bytes.',
                              'Acquire shared numerical lock; resolve dependencies and require exactly one DSO substitution.'],
        'required_capture':['Retain command/environment/CPU/timeout, loaded /proc/PID/maps paths, pre/post hashes and all raw stdout/trace.',
                            'Require all ordinary rows/raw words/status/IEEE parity and all zero observer health acknowledgements.',
                            'Require unchanged exact original input, repaired C, Born, numerator, masses, scales, shifts, matrix/clamp sequence for every group/direction reached.',
                            'Record every CT4/5 group visit, success/refusal, exact error, all defined tensor outputs and aggregate finite/single/double values.',
                            'If host flow changes group coverage, retain it explicitly; do not synthesize missing calls or discard surplus calls.',
                            'Retain all failures/timeouts/drift without automatic retry; kill/reap process group on interruption.'],
        'offline_adjudication':['Compare256 and512 per-group status and defined tensors, finite/single/double amplitude values and independent direction checks.',
                                'Compare aggregate512 with retained NinjaRS topology-c, using the identical C and existing parameter transport.',
                                'Recompute universal double pole from actual captured Born/coupling; preserve all original failures and external3e-6 criterion.',
                                'Separate39massless,6bottom and6top historical refusals. Bottom mass-square mismatch is unchanged and is not attributed to precision.',
                                'Record working precision512 and changed guard jointly; this single replay cannot isolate arithmetic from admission policy.',
                                'Do not promote host success or agreement to full accuracy/production/performance qualification.'],
        'limitations':['The existing512 DSO previously failed EW tensor qualification; loader suitability is not numerical certification.',
                       'The source threshold is shared by Gram, modified-Cayley and other guarded paths; this is not a Gram-only change.',
                       'Higher working precision does not recover input information lost before the113-bit transport.',
                       'The retained finite/single NinjaRS values are a comparator, not independently proven truth.',
                       'No new mass transport, shell policy, coordinate repair, coupling update, cache setting or256-bit guard change is included.'],
        'protected_inputs':pins}


def main():
    path=HERE/'proposal.json'
    require(not path.exists(),'Preserve the frozen proposal.')
    proposal=derive()
    path.write_text(json.dumps(proposal,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'protected_inputs':len(proposal['protected_inputs']),'proposal_sha256':sha(path),
                      'native_executed':False,'build_required':False},sort_keys=True))


if __name__=='__main__':
    main()
