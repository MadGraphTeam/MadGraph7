#!/usr/bin/env python3
"""Lightweight receipt-only qualification admission; no binaries or numerics read.

Final-artifact qualification is separate from root's final packet admission and
from every performance measurement. This module launches no programs.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

DELIVERY='88f6b75392dcb9cb5e03df6487d0ade2802c919d37e9aac6e8b16e54f2673b64'
DSO='d427667a0f2f55d0bd96778c1259ffcf52e3925341890b81c38f5f94e1eb7052'
ARCHIVE='37dbb061cb4391dbacea1d7f6b85d2fa116f005806b5b37054c16071dde3775a'
MODULE='471fe1e5e4c5bbb31490608ddc59c15c1dbdd3ba0bdc9dc723c18e3d8a7f1bfa'
SOURCE='0c2c1b078fad7f40caccb91661e24db0e18982ad4a1948bb4d3bac6cfb44584f'
PROTOTYPE='607e7f58842b50a07df7bbdbcbd00ee399f6baad0b670745b45cbf807de7ff07'
UPSTREAM='10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1'
PROOF='fdb60858ab40ea25fdfb45bbb1b2281fee8f4756a5221f3b7ec50d0f7baab4e2'
HERE=Path(__file__).resolve().parent


def require(value,message):
    if not value:raise ValueError(message)


def encoded(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def raw_small(path):
    path=Path(path);require(path.stat().st_size<=65536,'Admission reads small metadata only')
    return path.read_bytes()


def identity(path):
    raw=raw_small(path);return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def read(path):
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'Duplicate JSON key');out[k]=v
        return out
    def invalid(value):raise ValueError('Nonfinite JSON '+value)
    return json.loads(raw_small(path),object_pairs_hook=pairs,parse_constant=invalid)


def inside(root,name):
    require(isinstance(name,str) and name and '\\' not in name,'Invalid input path')
    part=PurePosixPath(name);require(not part.is_absolute() and '..' not in part.parts and str(part)==name,'Unsafe input path')
    path=Path(root)
    for component in part.parts:path/=component;require(not path.is_symlink(),'Symlink admission input')
    return path


def admit(root,manifest):
    require(manifest['schema']=='madloop-final-native-tiers-lightweight-admission-inputs-v1','Wrong admission input schema')
    for name,expected in manifest['files'].items():require(identity(inside(root,name))==expected,'Changed admission receipt '+name)
    folder=inside(root,manifest['final_root']);delivery=read(folder/'delivery-receipt.json')
    require(identity(folder/'delivery-receipt.json')['sha256']==DELIVERY,'Delivery pin differs')
    expected={'shared_library_sha256':DSO,'fortran_archive_sha256':ARCHIVE,'fortran_module_sha256':MODULE,
              'rust_bridge_source_sha256':SOURCE,'validated_prototype_rust_source_sha256':PROTOTYPE,'tirust_revision':UPSTREAM}
    require(all(delivery[k]==v for k,v in expected.items()),'Final source/artifact identity differs')
    require(delivery['status']=='qualified-final-private-installation-published' and delivery['abi_version']==1
            and delivery['release_tests_passed']==16 and delivery['release_tests_failed']==delivery['release_tests_ignored']==0
            and delivery['native_fortran_route_genuine_qp_poles_smoke_passed'] is True
            and delivery['live_abi_capability_and_error_preservation_passed'] is True
            and delivery['all_compilers_and_sdk_jobs_closed'] is True and not delivery['owned_survivors']
            and delivery['old85ee_and_prototype470_preserved'] is True
            and delivery['performance_eligible'] is False,'Final native qualification scope differs')
    require(all(delivery[k] is True for k in ('dp_route_statistics','finite_terms','quad_precision','poles_only')),'Required final capabilities absent')
    compiled=read(folder/'compiled.json');qualified=read(folder/'qualified.json');published=read(folder/'published.json');live=read(folder/'live-abi.json');prepared=read(folder/'source-prepared.json')
    for name,value in (('compiled',compiled),('qualified',qualified),('published',published)):
        require(identity(folder/(name+'.json'))['sha256']==delivery[name+'_receipt_sha256'] and value['candidate_source_sha256']==SOURCE,'Final stage/source chain differs')
        require(value['artifacts']['lib/libmadloop_tirust.so']==DSO and value['artifacts']['lib/libtirust.a']==ARCHIVE
                and value['artifacts']['lib/mtirust.mod']==MODULE,'Final stage artifact chain differs')
    require(published['qualified_receipt_sha256']==delivery['qualified_receipt_sha256']
            and qualified['compiled_receipt_sha256']==delivery['compiled_receipt_sha256']
            and qualified['live_abi_receipt_sha256']==delivery['live_abi_receipt_sha256']
            and qualified['qualification_log_sha256']==delivery['qualification_log_sha256']
            and qualified['release_tests_passed']==16 and qualified['sdk_smoke_executed'] is True,'Qualification linkage differs')
    require(identity(folder/'source-prepared.json')['sha256']==compiled['source_prepared_sha256']
            and prepared['validated_prototype_source_sha256']==PROTOTYPE
            and prepared['state_numeric_identity_proof_sha256']==delivery['numerical_identity_proof_sha256']==PROOF,'Prepared numerical identity linkage differs')
    proof_path=inside(root,manifest['identity_proof']);proof=read(proof_path)
    require(identity(proof_path)['sha256']==PROOF and proof['status']=='source-identity-passed'
            and proof['validated_prototype_source_sha256']==PROTOTYPE and proof['proposed_source_sha256']==SOURCE
            and set(proof['sections'])=={'complete_numerical_state','dp_c_abi','exact_qp_cache_settings_modes','genuine_qp_c_abi','input_batch_transport_packing'}
            and all(section['unchanged'] is True for section in proof['sections'].values()),'Numerical body identity proof differs')
    require(identity(folder/'live-abi.json')['sha256']==delivery['live_abi_receipt_sha256'] and live['status']=='passed'
            and live['abi_version']==1 and live['count']==20 and live['dp_route_statistics'] is True
            and live['preserves_prior_error_on_success'] is True and live['full_and_poles_rank6_supported'] is True
            and live['mapped_library_sha256']==DSO and live['source_sha256']==SOURCE
            and live['artifact_sha256']=={'libmadloop_tirust.so':DSO,'libtirust.a':ARCHIVE,'mtirust.mod':MODULE},'Live ABI/error preservation differs')
    for stage,receipt in (('compile',compiled),('qualify',qualified),('publish',published)):
        terminal_path=folder/'execution'/stage/'terminal.json';terminal=read(terminal_path)
        require(identity(terminal_path)['sha256']==delivery['stage_terminal_sha256'][stage]
                and terminal['status']=='passed' and terminal['returncode']==0 and not terminal['cleanup']['survivors']
                and terminal['phase_receipt_sha256']==delivery[{'compile':'compiled','qualify':'qualified','publish':'published'}[stage]+'_receipt_sha256'],
                'Final stage did not close cleanly')
    log_path=inside(root,manifest['root_integration_log']);log=raw_small(log_path).decode('utf-8')
    require(re.search(r'^Ran 67 tests in [0-9.]+s$',log,re.M) and re.search(r'^OK$',log,re.M)
            and 'skipped' not in log.lower() and 'FAILED' not in log and 'ERROR' not in log,'Root integration log not67 OK without skips')
    return {'schema':'madloop-final-native-tiers-artifact-admission-v1','final_artifact_qualified':True,
        'artifact':expected,'c_header_sha256':delivery['c_header_sha256'],'fortran_source_sha256':delivery['fortran_module_source_sha256'],
        'delivery_receipt':identity(folder/'delivery-receipt.json'),'native_release_tests':{'passed':16,'failed':0,'ignored':0},
        'native_genuine113_to256_transport_and_poles_smoke_passed':True,'live_error_preservation_passed':True,
        'numerical_identity_proof':identity(proof_path),'root_integration_tests':{'passed':67,'skips':0,'log':identity(log_path)},
        'root_configuration_selected_at_delivery':delivery['root_configuration_selected'],
        'performance_eligible_from_qualification':False,'prototype_measurements_relabelled':False,
        'prototype_timing_DSO':'470d46df9c8fcb76dc5448a04187dfc23a757481b3d4ef0384e13bd6e7e9e709',
        'final_unit_route_observation':delivery['unit_fixture_route_observation'],'final_unit_route_scope':delivery['route_observation_scope'],
        'root_source_commit_reported':manifest['root_source_commit_reported'],'tools_commit':manifest['tools_commit'],
        'final_packet_admitted':False,'canonical_publication_admitted':False,'upcoming_performance_records':manifest['upcoming_performance_records'],
        'large_binary_hashes_recomputed':False,'render_or_numerical_replay_performed':False,'new_native_physics_evaluations':0}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    value=admit(a.root.resolve(),read(HERE/'admission-inputs.json'));require(not a.output.exists(),'Fresh admission output required');a.output.write_bytes(encoded(value))
    print(encoded({'qualified':True,'root_integration_passes':67,'canonical_publication_admitted':False,'heavy_work_performed':False}).decode(),end='')

if __name__=='__main__':main()
