#!/usr/bin/env python3
"""Verify the retained interpreter listing without executing a program."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import struct

ROOT=Path.cwd().resolve()
HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    entries=json.loads((HERE/'evidence-index.json').read_text())['files']
    paths=list(HERE.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(HERE)) for p in paths if p.is_file()}==set(entries)|{'evidence-index.json'}
    for name,entry in entries.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        p=HERE/name;raw=p.read_bytes()
        assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
        assert not raw.startswith(b'\x7fELF') and p.suffix not in ('.pyc','.o','.a','.so','.mod'),name
    record=json.loads((HERE/'inspection1/inspection.json').read_text())
    assert record['complete'] and not record['amplitude_executed'] and not record['compiler_executed']
    assert record['protected_inputs_unchanged'] and record['protected_inputs_before']==record['protected_inputs_after']
    assert (record['cpu'],record['seconds_cap'],record['automatic_retry_count'])==(11,30,0)
    progress=record['launch_progress']
    assert progress['process_spawned'] and progress['process_reaped'] and progress['returncode']==record['returncode']==0
    assert 0<progress['process_wall_seconds']<=30
    interp=record['interpreter']
    assert record['command'][:5]==['taskset','-c','11',interp['resolved_path'],'--list']
    assert len(record['command'])==6
    failed=HERE.with_name('qcd-topology-rescue-higher-precision-native')
    assert sha(failed/'evidence-index.json')=='d5b67551d3c9353dbd472a0d272710b63bf220bdcab6e186944713ed07e9b14c'
    audit=json.loads((failed/'failure-audit.json').read_text())
    assert sha(failed/'failure-audit.json')==record['protected_inputs_before'][str((failed/'failure-audit.json').relative_to(ROOT))]
    assert interp==audit['interpreter']
    assert record['command'][5]==str(ROOT/audit['executable']['path'])
    expected=dict(record['ldd_tool_runtime_sha256'])
    loaders=[p for p in expected if Path(p).name=='ld-linux-x86-64.so.2']
    assert len(loaders)==1 and loaders[0]!=interp['resolved_path']
    del expected[loaders[0]]
    expected[interp['resolved_path']]=interp['sha256']
    assert record['expected_native_runtime_sha256']==record['listed_runtime_sha256']==expected
    listing=HERE/'inspection1/interpreter-list.txt'
    assert sha(listing)==record['listing_sha256'] and listing.stat().st_size==record['listing_bytes']
    parsed={}
    for line in listing.read_text().splitlines():
        match=re.search(r'(?:=>\s*)?(/\S+)\s+\(',line)
        if match:
            p=Path(match[1]).resolve();parsed[str(p)]=sha(p)
    assert parsed==expected
    # Independently decode the pinned executable's sole interpreter segment.
    executable=Path(record['command'][5]);raw=executable.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==audit['executable']['sha256']
    assert raw[:6]==b'\x7fELF\x02\x01'
    header=struct.unpack_from('<16sHHIQQQIHHHHHH',raw)
    assert header[9]==56 and header[5]+header[9]*header[10]<=len(raw)
    found=[]
    for i in range(header[10]):
        ph=struct.unpack_from('<IIQQQQQQ',raw,header[5]+i*header[9])
        if ph[0]==3:
            assert ph[5]>1 and ph[2]+ph[5]<=len(raw)
            payload=raw[ph[2]:ph[2]+ph[5]]
            assert payload[-1:]==b'\0' and b'\0' not in payload[:-1]
            found.append(payload[:-1].decode())
    assert found==[interp['pt_interp']]
    assert str(Path(found[0]).resolve())==interp['resolved_path'] and sha(found[0])==interp['sha256']
    if args.live:
        for name,digest in record['protected_inputs_before'].items():
            assert sha(ROOT/name)==digest,name
    print(json.dumps({'verification':'pass','portable_files':len(entries),'listed_files':len(parsed),
                      'live_inputs_checked':len(record['protected_inputs_before']) if args.live else 0,
                      'amplitude_executed_by_verifier':False},sort_keys=True))


if __name__=='__main__':
    main()
