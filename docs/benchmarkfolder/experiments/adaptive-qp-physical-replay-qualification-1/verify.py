"""Verify retained text, native identities and replay only frozen Python parsing."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BRIDGE='c5de24e6c41a90caf9e34684392d0193588522f26e6bbac646bdf74e5f151cb7'


def sha(data):return hashlib.sha256(data).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    assert not sys.flags.optimize
    index=json.loads((HERE/'evidence-index.json').read_text())
    for name,info in index['files'].items():
        path=HERE/name;assert path.is_relative_to(HERE) and not path.is_symlink()
        data=path.read_bytes();assert len(data)==info['bytes'] and sha(data)==info['sha256'],name
    inventory=json.loads((HERE/'payload-inventory.json').read_text());blobs={}
    for name,info in inventory.items():
        compressed=(HERE/info['archive']).read_bytes();assert sha(compressed)==info['archive_sha256']
        data=gzip.decompress(compressed);data.decode('utf-8')
        assert len(data)==info['bytes'] and sha(data)==info['sha256'];blobs[name]=data
    native=json.loads((HERE/'native-identities.json').read_text())
    record=json.loads(blobs['attempt/execution.json']);failed=json.loads(blobs['attempt/analysis.json'])
    assert record['complete'] is False and record['native_launched'] is True
    assert failed['failures']==[[1,'point_input_or_model_bits'],[2,'point_input_or_model_bits']]
    assert not record['automatic_retry'] and record['cpu']==12 and not record['cpu_reservation']
    assert record['execution_seconds_max']==180 and record['elapsed_seconds']<180
    assert record['rlimit_as']==[4294967296]*2 and record['protected_inputs_after']=='unchanged'
    assert record['source_index_sha256']==sha(blobs['proposal/source-index.json'])
    assert record['prepared_sha256']==sha(blobs['proposal/prepared.json'])
    assert record['analysis_sha256']==sha(blobs['attempt/analysis.json'])
    assert len(record['commands'])==2
    assert [c['kind'] for c in record['commands']]==['loader-list','one-physical-worker']
    assert all(c['exit_code']==0 and c['progress']['process_reaped'] and c['progress']['returncode']==0 for c in record['commands'])
    expected=record['expected_runtime_sha256'];mapping=record['actual_native_mapping']
    assert len(expected)==15 and record['loader_closure']['full_closure_exact']
    assert record['loader_closure']['actual_listed_runtime_sha256']==expected
    assert mapping['expected_runtime_sha256']==mapping['mapped_files_sha256_after_exit']==record['runtime_sha256_after']==expected
    assert set(mapping['mapped_elf_paths'])==set(expected)|{mapping['executable']}
    assert mapping['cpu_allowed_list']=='12' and expected[mapping['selected_library']]==BRIDGE
    assert record['bridge_byte_identity']==BRIDGE
    assert mapping['maps_sha256']==sha(blobs['attempt/native-maps.txt'])==sha(mapping['maps_text'].encode())
    listed={str(Path(p)) for p in re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(0x[0-9a-f]+\)',blobs['attempt/loader.log'].decode(),re.M)}
    assert listed==set(expected)
    native_by_origin={v['origin']:v for v in native.values()}
    for path,digest in expected.items():
        if path.startswith('/shared/external/madloop_tirust/madgraph7_madloop_tirust/'):
            origin=path.split('/shared/external/madloop_tirust/madgraph7_madloop_tirust/',1)[1]
        else:origin=path
        assert native_by_origin[origin]['sha256']==digest
    assert native['attempt/guarded-qp']['sha256']==record['executable_sha256']
    prepared=json.loads(blobs['proposal/prepared.json'])
    assert record['executable_sha256']==prepared['original_executable_sha256']
    before={Path(name).name:info['sha256'] for name,info in prepared['runtime_inputs'].items()}
    after={Path(name).name:digest for name,digest in expected.items()}
    assert set(before)==set(after)
    assert [name for name in before if before[name]!=after[name]]==['libmadloop_tirust.so']
    assert after['libmadloop_tirust.so']==BRIDGE
    proposal=json.loads(blobs['proposal/source-index.json'])
    for name,digest in proposal['sources'].items():assert sha(blobs['proposal/'+name])==digest
    for name,digest in record['all_case_files_after'].items():assert sha(blobs['attempt/case/'+name])==digest
    assert set(record['all_case_files_after'])-set(record['input_sha256'])==set(record['output_sha256'])
    for name,digest in record['input_sha256'].items():
        assert sha(blobs['attempt/case/'+name])==sha(blobs['original/'+name])==digest
    with tempfile.TemporaryDirectory(prefix='adaptive-physical-offline-') as directory:
        # Preserve enough ancestry for the unchanged frozen parser's module
        # constants. Only its pure parse/check_run functions are called.
        base=Path(directory)/'portable/replay/input'
        for name,data in blobs.items():
            path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        parser=load('adaptive_physical_frozen_parser',base/'frozen/journal-parser.py')
        completion=load('adaptive_physical_frozen_completion',base/'frozen/completion-validator.py')
        job=dict(prepared['original_job']);job['cwd']=str(base/'attempt/case')
        job['input_sha256']={str(base/'attempt/case'/name):digest for name,digest in record['input_sha256'].items()}
        codes=completion.check_run(job)
        old=parser.parse(base/'original/mass-observer.hex','candidate',9)
        new=parser.parse(base/'attempt/case/mass-observer.hex','candidate',9)
        assert codes==[int(new['points'][n]['data']['PUBLIC_STATUS'][0][1]) for n in (1,2)]
        analyze=load('adaptive_physical_offline_corrected',HERE/'offline-successor/analyze.py')
        result=analyze.compare(old,new,blobs['original/stdout.log'].decode(),blobs['attempt/case/stdout.log'].decode())
        saved=json.loads((HERE/'corrected-analysis.json').read_text())
        for key,value in result.items():assert saved[key]==value,key
        assert saved['offline_analyzer_sha256']==sha((HERE/'offline-successor/analyze.py').read_bytes())
        assert saved['offline_input_sha256']=={key:sha(blobs[name]) for key,name in {
            'old_journal':'original/mass-observer.hex','new_journal':'attempt/case/mass-observer.hex',
            'old_console':'original/stdout.log','new_console':'attempt/case/stdout.log','parser':'frozen/journal-parser.py'}.items()}
        assert result['invariants_passed'] and result['refusals_now_served']==37
        assert len(result['previously_served_calls'])==363 and all(r['old_served_output_bits_preserved'] for r in result['previously_served_calls'])
        assert len(result['old_W03'])==74 and not result['new_W03']
        assert result['observed_sums_have_no_recorded_native_omissions']
        assert len(result['Born_bit_checks'])==2 and all(r['bits_preserved'] for r in result['Born_bit_checks'])
        assert len(result['point_FPE_state_differences'])==4
        assert not result['physical_accuracy_qualified'] and not result['stability_or_performance_qualified']
        assert not result['actual_per_call_working_precision_available']
    print(json.dumps({'verified':True,'native_processes_complete':True,'original_controller_acceptance':False,
                      'additive_offline_invariants_passed':True,'formerly_refused_calls_now_served':37,
                      'previously_served_calls_preserved':363,'native_executed_by_verifier':False,
                      'physical_accuracy_or_stability_or_performance_qualified':False},sort_keys=True))


if __name__=='__main__':main()
