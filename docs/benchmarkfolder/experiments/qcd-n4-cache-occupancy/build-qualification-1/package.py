#!/usr/bin/env python3
"""Retain text-only build evidence once; default mode verifies portable bytes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
BUILD=ROOT/'.tirust-build/qcd-n4-cache-occupancy/attempt1/build/build.json'
BUILD_SHA='40dedfe9db3793d20f21fc97f6f7d1f4507268619d968ce0c64392bf5be58d8a'


def digest(data): return hashlib.sha256(data).hexdigest()


def validate(record):
    assert record['status']=='complete'
    assert record['protected_before']==record['protected_after']
    assert record['numerical_execution_performed'] is False
    assert record['numerical_runtime_mapping_validated'] is False
    assert record['runtime_provenance_validated'] is True
    assert (record['cpu'],record['jobs'],record['seconds_cap'])==(12,1,180)
    commands=record['commands']
    assert len(commands)==7
    assert all(c['returncode']==0 and c['process_reaped'] for c in commands)
    assert sum('-c' in c['argv'] for c in commands)==3
    assert sum('--list' in c['argv'] for c in commands)==2
    assert sum(any(x.startswith('-Wl,-Map,') for x in c['argv']) for c in commands)==2
    a,b=record['arms']['original'],record['arms']['observed']
    assert a['runtime']==b['runtime'] and len(a['runtime'])==15
    assert len(a['link_inputs'])==len(b['link_inputs'])==59
    assert a['objects'].keys()==b['objects'].keys()
    differences=[name for name in a['objects'] if a['objects'][name]['sha256']!=b['objects'][name]['sha256']]
    assert differences==['TIRUST_interface.o']
    assert a['common_objects']==b['common_objects']
    assert record['installed_dso']['sha256']=='eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9'
    return {'status':'compile-and-link-qualified','commands':7,'protected_inputs_unchanged':True,
            'ordinary_numerical_objects_changed':['TIRUST_interface.o'],
            'common_private_objects':['observer.o','driver.o'],'dependency_libraries_per_arm':15,
            'verified_link_inputs_per_arm':59,'native_evaluation':False,
            'original_disabled_enabled_parity':'unlaunched','budget_pressure':'unmeasured',
            'performance_eligible':False,
            'executables':{name:value['executable'] for name,value in record['arms'].items()}}


def package():
    assert not (HERE/'index.json').exists()
    raw=BUILD.read_bytes();assert digest(raw)==BUILD_SHA
    record=json.loads(raw);summary=validate(record)
    inventory={}
    def compressed(name,path,expected):
        content=Path(path).read_bytes();assert digest(content)==expected
        content.decode('utf8')
        output=HERE/name;assert not output.exists()
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('xb') as stream:
            with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as encoded:
                encoded.write(content)
        inventory[name]={'sha256':digest(output.read_bytes()),'size_bytes':output.stat().st_size,
                         'compression':'gzip','raw_sha256':expected,'raw_bytes':len(content),
                         'original_path':str(path)}
    compressed('build.json.gz',BUILD,BUILD_SHA)
    for name,arm in record['arms'].items():
        compressed(name+'-link.map.gz',arm['link_map']['path'],arm['link_map']['sha256'])
    for i,command in enumerate(record['commands'],1):
        compressed(f'logs/{i:02}.txt.gz',command['log'],command['log_pin']['sha256'])
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    for name in ('package.py','README.md','summary.json'):
        path=HERE/name
        inventory[name]={'sha256':digest(path.read_bytes()),'size_bytes':path.stat().st_size}
    result={'schema':1,'scope':'Private Fortran build only; numerical replay not launched',
            'files':inventory,'artifact_count':len(inventory),'source_build_manifest_sha256':BUILD_SHA}
    (HERE/'index.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')


def verify():
    index=json.loads((HERE/'index.json').read_text())
    assert set(index['files'])=={p.relative_to(HERE).as_posix() for p in HERE.rglob('*')
                                if p.is_file() and p.name!='index.json'}
    for name,expected in index['files'].items():
        path=HERE/name;assert not path.is_symlink()
        data=path.read_bytes()
        assert digest(data)==expected['sha256'] and len(data)==expected['size_bytes']
        if expected.get('compression')=='gzip':
            raw=gzip.decompress(data);raw.decode('utf8')
            assert digest(raw)==expected['raw_sha256'] and len(raw)==expected['raw_bytes']
    raw=gzip.decompress((HERE/'build.json.gz').read_bytes())
    assert digest(raw)==BUILD_SHA
    assert validate(json.loads(raw))==json.loads((HERE/'summary.json').read_text())
    print(json.dumps({'portable_verification':'pass','files':len(index['files']),
                      'index_sha256':digest((HERE/'index.json').read_bytes()),
                      'native_evaluation':False},sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--package',action='store_true')
    if parser.parse_args().package: package()
    verify()
