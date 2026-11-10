#!/usr/bin/env python3
"""Copy frozen text-only control evidence. No compiler/native calls or overwrites."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
SOURCE=ROOT/'.tirust-build/born-effective-runtime/control-stub'
PARENT=ROOT/'.tirust-build/born-effective-runtime/attempt1'


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    copied={}
    def put(name,data):
        data.decode('utf-8')
        assert b'\x00' not in data,name
        path=HERE/name
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert path.read_bytes()==data,name
        else:path.write_bytes(data)
    def copy(source,name):
        assert source.is_file() and not source.is_symlink(),source
        data=source.read_bytes();put(name,data)
        key=str(source.relative_to(ROOT))
        assert key not in copied
        copied[key]={'copy':name,'sha256':digest(data),'bytes':len(data)}
    top=('README.md','born_driver.f90.in','repair_driver.f90','repair_stubs.f90',
         'prepare.py','validate_sources.py','run.py','manifest.json',
         'compiler-jobs.json','expected.json','native.json')
    for name in top:copy(SOURCE/name,'original/'+name)
    for directory in ['sources','native-runs']:
        for path in sorted((SOURCE/directory).rglob('*')):
            if path.is_file():copy(path,'original/'+str(path.relative_to(SOURCE)))
    for group,names in {
        'learning-transaction':('generate.py','inputs/source.json','inputs/matrix_standalone_v4.inc',
            'inputs/matrix_standalone_splitOrders_v4.inc','inputs/file_writers.py','inputs/formatter.json',
            'artifacts/manifest.json','history/revision-1/manifest.json'),
        'checked-repair':('generate.py','inputs/source.json','inputs/improve_ps.inc',
            'inputs/loop_matrix_standalone.inc','artifacts/manifest.json'),
    }.items():
        directory=ROOT/'.tirust-build/born-effective-implementation'/group
        for name in names:copy(directory/name,'generators/'+group+'/'+name)
    copy(ROOT/'madgraph/iolibs/file_writers.py','support/file_writers.py')
    helper=ROOT/'docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/attempt2/control.py'
    copy(helper,'support/bounded_owner.py')
    copy(PARENT/'build.py','build/original_build.py')
    native=json.loads((SOURCE/'native.json').read_text())
    build_bytes=(PARENT/'build.json').read_bytes()
    build=json.loads(build_bytes)
    assert build['state']=='complete' and not build['changed_inputs']
    assert digest(build_bytes)==native['bindings_before']['build_report_sha256']
    commands=[x for x in build['commands'] if x['name'].startswith('control-stub/')]
    assert len(commands)==5 and all(x['returncode']==0 and not x['timed_out'] and not x.get('error') for x in commands)
    prefix=str((SOURCE/'work').relative_to(ROOT))+'/'
    artifacts={k:v for k,v in build['artifacts'].items() if k.startswith(prefix)}
    assert artifacts==native['bindings_before']['artifact_bindings']
    source_prefix=str((SOURCE/'sources').relative_to(ROOT))+'/'
    inputs_bytes=(PARENT/'build-inputs.json').read_bytes()
    assert digest(inputs_bytes)==build['build_inputs_sha256']
    inputs={k:v for k,v in json.loads(inputs_bytes).items() if k.startswith(source_prefix)}
    assert inputs==native['bindings_before']['compiled_source_bindings']
    log_bytes=(PARENT/'build.log').read_bytes()
    assert digest(log_bytes)==build['build_log_sha256']
    offset=0;first=None
    for row in log_bytes.splitlines(keepends=True):
        if row.startswith(b'{'):
            try:record=json.loads(row)
            except json.JSONDecodeError:record={}
            if record.get('name')=='control-stub/scalar_n3':
                first=offset;break
        offset+=len(row)
    assert first is not None
    log_subset=log_bytes[first:]
    put('build/compile.log',log_subset)
    proof={
        'schema':1,'scope':'Exact five successful control compiler events and their source/artifact bindings; unrelated process builds excluded.',
        'parent_build_root_relative':str((PARENT/'build.json').relative_to(ROOT)),
        'parent_build_sha256':digest(build_bytes),
        'parent_build_controller_root_relative':str((PARENT/'build.py').relative_to(ROOT)),
        'parent_build_controller_sha256':build['controller_sha256'],
        'parent_build_log_root_relative':str((PARENT/'build.log').relative_to(ROOT)),
        'parent_build_log_sha256':build['build_log_sha256'],
        'parent_build_inputs_root_relative':str((PARENT/'build-inputs.json').relative_to(ROOT)),
        'parent_build_inputs_sha256':build['build_inputs_sha256'],
        'compile_log_exact_suffix':{'copy':'build/compile.log','parent_byte_offset':first,
             'bytes':len(log_subset),'sha256':digest(log_subset)},
        'state':build['state'],'cpu':build['cpu'],'max_compiler_jobs':build['max_compiler_jobs'],
        'helper_sha256':build['helper_sha256'],'environment':build['environment'],
        'commands':commands,'compiled_source_bindings':inputs,'artifact_hashes_only':artifacts,
        'binary_artifacts_copied':False,
        'provenance_limit':'The full parent report/log/input inventory are identified by original hashes. Their selected fields are preserved here and cross-bound to the complete control native report; unrelated parent bytes are not reconstructible from this subset.'}
    put('build/compile-proof.json',(json.dumps(proof,indent=2,sort_keys=True)+'\n').encode())
    put('source-map.json',(json.dumps({'schema':1,'original_repository_root':str(ROOT),
        'copied_text':copied,'binary_artifact_policy':'Only original path/hash identities in compile-proof/native.json; no executable, object, archive, module, pickle or pycache is copied.'},indent=2,sort_keys=True)+'\n').encode())
    print('Copied',len(copied),'exact text files and two derived compile proof files; no native/build work.')


if __name__=='__main__':main()
