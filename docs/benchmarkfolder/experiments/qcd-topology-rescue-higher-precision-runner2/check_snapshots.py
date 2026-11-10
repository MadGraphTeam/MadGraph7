#!/usr/bin/env python3
"""Exercise durable mapping evidence with synthetic files; never load an ELF."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
from unittest.mock import patch

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def compute():
    from supervise import capture_maps
    original=Path.read_text
    current={'maps':'','status':'Cpus_allowed_list:\t11\n'}
    def fake_read(path,*args,**kwargs):
        if str(path)=='/proc/123/maps':
            return current['maps']
        if str(path)=='/proc/123/status':
            return current['status']
        return original(path,*args,**kwargs)
    passed=[]
    with tempfile.TemporaryDirectory(prefix='map-snapshot-test-',dir=Path.cwd()/'.tirust-build') as directory:
        work=Path(directory)
        exe=work/'program';bridge=work/'libmadloop_tirust.so';loader=work/'ld-linux-x86-64.so.2'
        for path in (exe,bridge,loader):
            path.write_bytes(b'\x7fELFsynthetic-not-executable')
        expected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (bridge,loader)}
        def maps(paths):
            return ''.join(f'{4096+i*4096:x}-{8192+i*4096:x} r-xp 00000000 00:00 1 {p}\n' for i,p in enumerate(paths))
        out=work/'maps.txt';state={};snap=work/'maps.snapshots'
        with patch.object(Path,'read_text',fake_read):
            current['maps']=maps((exe,bridge))
            first=current['maps']
            assert capture_maps(123,expected,str(exe),str(bridge),out,state) is None
            assert original(out)==first and original(snap/'first-nonempty.txt')==first
            assert state['last_incomplete_reason']['missing_expected_paths']==[str(loader)]
            passed.append('missing-loader-preserves-first-nonempty')
            current['maps']=''
            assert capture_maps(123,expected,str(exe),str(bridge),out,state) is None
            assert original(out)==first and original(snap/'last-nonempty.txt')==first
            assert original(snap/'last-read.txt')=='' and original(snap/'first-read.txt')==first
            passed.append('empty-exit-read-cannot-erase-live-snapshot')
            current['maps']=maps((exe,bridge,loader))
            result=capture_maps(123,expected,str(exe),str(bridge),out,state)
            assert set(result['mapped_elf_paths'])==set(expected)|{str(exe)}
            assert original(snap/'complete.txt')==current['maps'] and original(snap/'first-read.txt')==first
            passed.append('complete-native-closure-snapshot-retained')
            current['maps']=maps((str(bridge)+' (deleted)',))
            try:
                capture_maps(123,expected,str(exe),str(bridge),out,state)
            except RuntimeError as exc:
                assert 'deleted' in str(exc)
            else:
                raise AssertionError('Deleted mapping accepted.')
            assert original(snap/'mismatch.txt')==current['maps']
            assert original(snap/'first-nonempty.txt')==first
            passed.append('failed-map-retained-before-exception')
    return {'schema':1,'passed':passed,'synthetic_cases':len(passed),'native_executed':False,
            'library_loaded':False,'scope':'Synthetic parser/evidence-retention checks; not runtime or numerical qualification.'}


if __name__=='__main__':
    result=compute()
    if sys.argv[1:]==['--write']:
        target=HERE/'snapshot-checks.json'
        assert not target.exists()
        target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    else:
        assert not sys.argv[1:]
        assert result==json.loads((HERE/'snapshot-checks.json').read_text())
        print(json.dumps(result,sort_keys=True))
