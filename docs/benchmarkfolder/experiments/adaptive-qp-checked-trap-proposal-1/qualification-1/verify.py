"""Portable receipt consistency and frozen Python replay; no native execution."""
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import tempfile

HERE=Path(__file__).resolve().parent
SOURCE_HASH='32909d9a7b4b2102f35eeddc547d2f45c7560f903253b0571bf77b5628220a29'


def sha(data):return hashlib.sha256(data).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    index=json.loads((HERE/'evidence-index.json').read_text())
    actual_files={str(path.relative_to(HERE)) for path in HERE.rglob('*') if path.is_file()}
    assert actual_files==set(index['files'])|{'evidence-index.json'}
    for name,info in index['files'].items():
        relative=Path(name);assert not relative.is_absolute() and '..' not in relative.parts
        path=HERE/relative;assert not path.is_symlink()
        data=path.read_bytes();assert len(data)==info['bytes'] and sha(data)==info['sha256'],name
    inventory=json.loads((HERE/'payload-inventory.json').read_text());blobs={}
    for name,info in inventory.items():
        compressed=(HERE/info['archive']).read_bytes();assert sha(compressed)==info['archive_sha256']
        data=gzip.decompress(compressed);data.decode('utf-8')
        assert len(data)==info['bytes'] and sha(data)==info['sha256'];blobs[name]=data
    native=json.loads((HERE/'native-identities.json').read_text())
    assert all(info['retained_as']=='native_or_dependency_identity_only' for info in native.values())
    source=json.loads(blobs['source/source-index.json']);prepared=json.loads(blobs['source/prepared.json'])
    assert sha(blobs['source/source-index.json'])==SOURCE_HASH
    for name,digest in source['sources'].items():assert sha(blobs['source/'+name])==digest,name
    assert source['dependencies']==prepared['dependencies']
    for name,digest in prepared['dependencies'].items():assert native['dependency/'+name]['sha256']==digest,name
    build=json.loads(blobs['build1/build.json']);run=json.loads(blobs['run1/execution.json'])
    assert build['complete'] and build['protected_inputs_after']=='unchanged'
    assert build['source_index_sha256']==SOURCE_HASH and run['source_index_sha256']==SOURCE_HASH
    assert run['build_sha256']==sha(blobs['build1/build.json'])
    assert not any(k in build for k in ('failure','closure_failure'))
    assert len(build['commands'])==2
    assert [row['argv'] for row in build['commands']]==[prepared['compile_argv'],prepared['link_argv']]
    assert all(row['exit_code']==0 and row['progress']['process_reaped'] and row['progress']['returncode']==0 for row in build['commands'])
    assert build['unchanged_numerical_objects']==prepared['unchanged_numerical_objects']
    assert len(build['unchanged_numerical_objects'])==25
    for name,digest in build['unchanged_numerical_objects'].items():assert build['actual_link_inputs'][name]==digest
    assert prepared['original_main'] not in build['actual_link_inputs']
    for name,digest in build['actual_link_inputs'].items():assert native['dependency/'+name]['sha256']==digest
    for name,digest in build['artifacts'].items():
        key='build1/'+name;assert (sha(blobs[key]) if key in blobs else native[key]['sha256'])==digest,name
    # Map literal paths may be aliases. Their observed resolved identities are
    # checked against recorded native/dependency identities, never host files.
    origin_root=str(Path(prepared['compile_argv'][prepared['compile_argv'].index('-c')+1]).parents[4])
    def origin(path):return path[len(origin_root)+1:] if path.startswith(origin_root+'/') else path
    by_origin={info['origin']:info for info in native.values()}
    for path,digest in build['actual_link_inputs'].items():assert by_origin[origin(path)]['sha256']==digest
    loads=re.findall(r'^LOAD\s+(.+)$',blobs['build1/checked.map'].decode(),re.M)
    lineage=json.loads((HERE/'link-map-path-lineage.json').read_text())
    assert lineage['build_sha256']==sha(blobs['build1/build.json'])
    assert [row['literal'] for row in lineage['LOAD_rows']]==loads
    resolved={}
    for row in lineage['LOAD_rows']:
        assert by_origin[origin(row['resolved'])]['sha256']==row['sha256']
        if row['resolved'] in resolved:assert resolved[row['resolved']]==row['sha256']
        resolved[row['resolved']]=row['sha256']
    assert resolved==build['actual_link_inputs']
    assert run['execution_complete'] and not run['candidate_checked_and_trap_coverage_qualified']
    assert run['protected_inputs_after']=='unchanged' and not run['automatic_retry']
    assert not any(k in run for k in ('failure','closure_failure'))
    for receipt,limit in ((build,180),(run,600)):
        assert math.isfinite(receipt['elapsed_seconds']) and 0<=receipt['elapsed_seconds']<=limit
        assert not receipt['post_deadline_closure_is_administrative_only']
        assert receipt['cpu']==12 and not receipt['cpu_reservation']
    assert run['rlimit_as']==[4294967296]*2
    assert set(run['workers'])=={'fault-invalid','fault-divide_by_zero','fault-overflow','original-masked','adaptive-masked','original-trapped','adaptive-trapped'}
    original_runtime={Path(name).name:info['sha256'] for name,info in prepared['runtime_inputs'].items()}
    for arm,row in run['loaders'].items():
        expected=row['closure']['actual_runtime_sha256']
        assert len(expected)==15 and row['closure']['exact'] and row['exit_code']==0 and row['progress']['process_reaped']
        paths={name for name in re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(0x[0-9a-f]+\)',blobs['run1/loader-'+arm+'.log'].decode(),re.M)}
        assert paths==set(expected) and expected==run['runtime_sha256_after'][arm]
        after={Path(path).name:digest for path,digest in expected.items()}
        assert set(after)==set(original_runtime)
        assert all(after[name]==digest for name,digest in original_runtime.items() if name!='libmadloop_tirust.so')
        assert after['libmadloop_tirust.so']==prepared['bridge_metadata'][arm]['sha256']
    with tempfile.TemporaryDirectory(prefix='checked-trap-receipts-') as directory:
        source_dir=Path(directory)/'portable/repository/docs/benchmarkfolder/experiments/packet/source'
        source_dir.mkdir(parents=True)
        for name in ('controller.py',): (source_dir/name).write_bytes(blobs['source/'+name])
        controller=load('checked_trap_portable_transport',source_dir/'controller.py')
        assert controller.transport_complete(run['workers'])
        for name,row in run['workers'].items():
            expected=row['expected_runtime_sha256'];mapping=row['actual_mapping'];prefix='run1/'+name+'/'
            assert expected==run['runtime_sha256_after'][row['arm']]
            assert mapping['cpu_allowed_list']=='12' and mapping['executable']==row['argv'][0]
            assert expected[mapping['selected_library']]==prepared['bridge_metadata'][row['arm']]['sha256']
            assert sha(blobs[prefix+'native-maps.txt'])==mapping['maps_sha256']==sha(mapping['maps_text'].encode())
            for path,digest in expected.items():assert by_origin[origin(path)]['sha256']==digest
            for rel,digest in row['inputs_before'].items():assert sha(blobs[prefix+rel])==digest
            for rel,digest in row['files_after'].items():assert sha(blobs[prefix+rel])==digest
            actual_closure=controller.output_closure(row['inputs_before'],row['files_after'],row['kind'],row['exit_code'])
            assert actual_closure==row['native_output_closure'] and actual_closure['matched']
            for path,digest in row['complete_case_files_sha256'].items():
                relative=path.split('/run1/'+name+'/',1)[1]
                assert sha(blobs[prefix+relative])==digest
    analysis=load('checked_trap_portable_analysis',HERE/'analyze.py').reanalyze(blobs)
    assert analysis==json.loads((HERE/'offline-analysis.json').read_text())
    assert analysis['masked_adaptive_checked_outputs_qualified'] and analysis['masked_adaptive_QP_three_pass_coverage']
    assert not analysis['combined_checked_trap_coverage_qualified']
    assert not any(analysis[k] for k in ('per_call_working_precision_available','independent_physical_accuracy_qualified','performance_qualified'))
    symbols=json.loads((HERE/'symbolization.json').read_text())
    assert not symbols['exact_hardware_exception_identified']
    for arm,item in symbols['arms'].items():
        prefix='run1/'+arm+'-trapped/'
        assert item['maps_sha256']==sha(blobs[prefix+'native-maps.txt']) and item['stdout_sha256']==sha(blobs[prefix+'stdout.log'])
        frame=item['frames'][0];assert frame['number']==3 and frame['bytes_at_pc'].startswith('f20f5ec1')
        assert 'cayley_cancellation::<f64>' in frame['addr2line_stdout'] and 'divsd' in frame['objdump_stdout']
        assert frame['object_sha256']==prepared['bridge_metadata'][arm]['sha256']
        row=frame['mapping_line'].split(None,5);pc=int(frame['pc'],16)
        assert frame['file_offset']==pc-int(row[0].split('-')[0],16)+int(row[2],16)
    assert sha((HERE/'reference-gram.rs').read_bytes())==symbols['reference_source']['sha256']
    assert sha(blobs['source/source-index.json'])==SOURCE_HASH
    print(json.dumps({'portable_verified':True,'masked_adaptive_checked_QP_qualified':True,
                      'combined_trap_coverage_qualified':False,'formerly_served_outputs_preserved':489,
                      'trap_path':'Shared bridge DPf64 Cayley-conditioning divide',
                      'exact_trap_exception_code_available':False,'native_or_loader_executed':False,
                      'independent_physics_or_performance_qualified':False},sort_keys=True))


if __name__=='__main__':main()
