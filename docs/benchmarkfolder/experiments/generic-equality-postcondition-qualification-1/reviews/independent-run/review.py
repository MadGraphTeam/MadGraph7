"""Pure replay of closed helper transcripts and observed environment cohorts."""
import builtins,ctypes,hashlib,io,json,os,subprocess,time,types
from collections import Counter
from datetime import datetime
from pathlib import Path
ROOT=Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'.tirust-build/generic-equality-postcondition-qualification-proposal-1'
WORK=ROOT/'.tirust-build/generic-equality-postcondition-qualification-work-1'
EARLIER=ROOT/'.tirust-build/generic-equality-postcondition-independent-build-review-1'
SOURCE_SHA='4e3878690b366992d1dad8e43ae650a62362220d7bfc21e85850228d2aeef12d'
BUILD_SHA='a12fd704ab54e935028258299239beb62f6d366620f990e0130db4f9df4fa506'
TERMINAL='7a67bf86babae2497c65c2c421995c7038edd1e19186c1d5ed0ecc24f707f93e'
RESULT='49b2cf6c15c6cc9b4ef43519c8f21d1a6126901c28ba140199d0f9c295f843db'
RAW='a40826258c4842ac4ec21fa477a28f6b75670484387ea8c9c30ddfdafdbb4be7'
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
seen={}
def match(p,h):
    p=Path(p).resolve()
    if str(p) not in seen:seen[str(p)]=sha(p.read_bytes())
    assert seen[str(p)]==h,str(p)

def main():
    start=time.monotonic();run=WORK/'run';build=WORK/'build'
    match(EARLIER/'receipt.json','1524bc3595e3bff40e0b6007aa56d8a08f47b375a3df3447a89c5ddc8a0edd99')
    previous=read(EARLIER/'receipt.json');match(EARLIER/'review.py',previous['audit_source_sha256'])
    assert previous['independent_fixture_cases']==1698 and previous['independent_accepts']==526 and previous['independent_refusals']==1172
    match(SOURCE/'source-index.json',SOURCE_SHA);seal=read(SOURCE/'source-index.json')
    for n,v in seal['files'].items():match(SOURCE/n,v['sha256'])
    assert {str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file()}==set(seal['files'])|{'source-index.json'}
    match(build/'terminal.json',BUILD_SHA);bt=read(build/'terminal.json');match(build/'result.json',bt['result_sha256']);built=read(build/'result.json')
    match(run/'terminal.json',TERMINAL);match(run/'result.json',RESULT)
    terminal=read(run/'terminal.json');result=read(run/'result.json');admission=read(run/'admission.json')
    assert terminal['status']==result['status']=='passed' and terminal['returncode']==0 and not terminal['cleanup']['survivors']
    assert terminal['result_sha256']==RESULT and terminal['cap_seconds']==180 and terminal['cleanup_reserve_seconds']==60
    assert admission['source_index_sha256']==result['source_index_sha256']==SOURCE_SHA
    assert admission['reviewed_build_sha256']==result['reviewed_build_sha256']==BUILD_SHA
    assert result['scientific_reducer_calls']==0 and not result['physical_or_performance_qualification'] and not result['snapshot_ownership_qualified']
    assert read(run/'limits.json')['actual']=={'address_space_bytes':[4294967296]*2,'file_bytes':[268435456]*2,'core_bytes':[0,0]}
    elapsed=(datetime.fromisoformat(terminal['finished_utc'])-datetime.fromisoformat(terminal['started_utc'])).total_seconds();assert elapsed<180
    pins=read(SOURCE/'input-pins.json');ancestor=read(SOURCE/'ancestry.json');tools=read(SOURCE/'toolchain-pins.json')
    for name,want in {**ancestor['source_files'],**pins['support'],**pins['closed_standalone_build'],**pins['files'],**tools['donor_lineage']}.items():
        match(Path(name) if Path(name).is_absolute() else ROOT/name,want)
    match(build/'fenv_observer.o',built['fenv_object_sha256'])
    expected=read(SOURCE/'expectations.json');raw={};stored={};maps={};closure=[]
    for variant,row in built['variants'].items():
        folder=run/variant;match(row['path'],row['sha256'])
        for n,h in row['objects'].items():match(build/variant/n,h)
        for n,k in [('prelink.json','prelink_sha256'),('link.map','link_map_sha256'),('static-runtime.json','static_runtime_sha256')]:match(build/variant/n,row[k])
        for n,h in {**row['actual_link_loads'],**row['runtime']['files']}.items():match(n,h)
        closed=read(folder/'terminal.json');cmd=read(folder/'command.json');started=read(folder/'started.json')
        assert closed['returncode']==0 and not closed['stdout_tail'] and cmd['new_session'] is False
        assert cmd['argv']==[row['path'],str(SOURCE/'fixtures.hex'),str(folder/'observations.txt')] and cmd['cwd']==str(folder)
        pid=started['pid'];assert (folder/'ready.txt').read_text().strip()=='READY '+str(pid) and (folder/'done.txt').read_text().strip()=='DONE '+str(pid)
        assert (folder/'stderr.txt').read_bytes()==b'';match(folder/'stderr.txt',closed['stderr_sha256'])
        raw[variant]=(folder/'observations.txt').read_bytes();assert len(raw[variant])==6698235 and sha(raw[variant])==RAW==closed['observations_sha256']
        stored[variant]=read(folder/'analysis.json');maps[variant]={}
        for edge in ('before','after'):
            file=folder/('maps-'+edge+'.json');mapped=read(file)
            assert mapped['ELF']=={**row['runtime']['files'],str(Path(row['path']).resolve()):row['sha256']}
            assert len(mapped['ELF'])==6 and mapped['affinity']==[12] and not mapped['exclusive_CPU']
            assert all(pins['allowed_data_mappings'].get(n)==h for n,h in mapped['data'].items())
            paths={str(Path(line.split(None,5)[5]).resolve()) for line in mapped['raw'].splitlines() if len(line.split(None,5))==6 and line.split(None,5)[5].startswith('/')}
            assert paths==set(mapped['ELF'])|set(mapped['data']) and not any(n.endswith(' (deleted)') for n in paths)
            for n,h in {**mapped['ELF'],**mapped['data']}.items():match(n,h)
            assert mapped['pid']==mapped['identity']['pid']==pid
            assert all(mapped['identity'][k]==started['identity'][k] for k in ('pid','parent_pid','process_group','session','start_time_ticks'))
            assert any(all(x[k]==mapped['identity'][k] for k in ('pid','process_group','session','start_time_ticks')) for x in terminal['cleanup']['observed_owned_processes'])
            maps[variant][edge]=sha(file.read_bytes())
        closure.append((datetime.fromisoformat(cmd['started_utc']),datetime.fromisoformat(closed['finished_utc'])) )
    assert raw['checked']==raw['optimized'] and closure[0][1]<=closure[1][0]
    analyzer={};exec(compile((SOURCE/'analyze.py').read_bytes(),str(SOURCE/'analyze.py'),'exec'),analyzer)
    saved=[];attempts=[]
    def deny(*a,**kw):attempts.append(str(a[:1]));raise RuntimeError('Forbidden API during pure helper replay')
    try:
        for obj,n in ((builtins,'open'),(io,'open'),(io,'open_code'),(os,'open'),(os,'stat'),(os,'lstat'),(os,'scandir'),(os,'system'),(os,'popen'),(Path,'read_bytes'),(Path,'read_text'),(Path,'resolve'),(subprocess,'Popen'),(subprocess,'run'),(subprocess,'call'),(subprocess,'check_call'),(subprocess,'check_output'),(ctypes,'CDLL'),(ctypes,'PyDLL')):
            saved.append((obj,n,getattr(obj,n)));setattr(obj,n,deny)
        replayed={v:json.loads(json.dumps(analyzer['analyze'](b.decode(),expected))) for v,b in raw.items()}
        assert replayed==stored==result['variants']
        assert replayed['checked']==replayed['optimized']
        # Independent token-level cohort/decision/state check, without invoking
        # the frozen parser's request generator or result implementation.
        counts=Counter();modes=Counter();case_sequence=[];flags=[1,4,8,16,32];rounds=[0,1024,2048,3072]
        for line in raw['checked'].decode().splitlines():
            fields=line.split()
            if not fields or fields[0]!='CASE':continue
            x=list(map(int,fields[1:]));assert len(x)==38
            ident,mode,status,accepted,restored=x[:5];req=x[5:10];baseline=x[10:17];before=x[17:24];after=x[24:31];last=x[31:38]
            assert status==0 and restored==1 and baseline==last and before==after
            assert accepted==int(expected['cases'][str(ident)]['expected_accept'])
            expected_request=[-1]*5 if mode==0 else [rounds[mode-1] if 1<=mode<=4 else 0,
                flags[mode-5] if 5<=mode<=9 else 61 if mode==10 else 0,
                flags[mode-11] if 11<=mode<=15 else 61 if mode==19 else 0,
                1 if mode in (16,18) else 0 if mode==17 else -1,
                1 if mode in (17,18) else 0 if mode==16 else -1]
            assert req==expected_request
            if mode:
                rounding,sticky,halts=before[:3];control,cw,csr,sw=before[3:]
                assert [rounding,sticky,halts]==req[:3]
                assert ((cw>>10)&3)==((csr>>13)&3)==rounding//1024
                assert ((~cw)&61)==((~(csr>>7))&61)==halts
                assert sticky==((csr|sw)&61) and control==(csr&~63)
                if mode in (16,17,18):assert [(csr>>15)&1,(csr>>6)&1]==req[3:]
            else:assert before==baseline
            case_sequence.append((ident,mode));modes[mode]+=1;counts['accept' if accepted else 'refuse']+=1
        assert case_sequence==[(i,m) for i in range(1,1699) for m in range(20)]
        assert modes=={m:1698 for m in range(20)} and counts=={'accept':10520,'refuse':23440}
        assert all(x['native_INTEGER128_input_words_joined']==66320 for x in replayed.values()) and not attempts
    finally:
        for obj,n,v in reversed(saved):setattr(obj,n,v)
    match(run/'terminal.json',TERMINAL);match(run/'result.json',RESULT)
    receipt={'schema':'independent-exact-geometry-closed-run-audit-v1','status':'PASS','source_index_sha256':SOURCE_SHA,'build_terminal_sha256':BUILD_SHA,
        'run_terminal_sha256':TERMINAL,'run_result_sha256':RESULT,'independent_expectation_audit_sha256':'1524bc3595e3bff40e0b6007aa56d8a08f47b375a3df3447a89c5ddc8a0edd99',
        'raw_per_variant_sha256':RAW,'raw_per_variant_bytes':6698235,'raw_variants_byte_identical':True,'pure_parser_replay_exact':True,
        'cases_per_variant':1698,'executed_fenv_cohorts_per_case':20,'helper_calls_per_variant':33960,'total_helper_calls':67920,
        'per_variant_decisions':dict(counts),'explicitly_skipped_cohorts':0,'per_variant_integer128_input_word_joins':66320,'total_input_word_joins':132640,
        'independent_token_cohort_state_replay':True,'observed_before_after_and_restored_fields_equal':True,
        'maps':maps,'runtime_providers_per_variant':5,'worker_executable_maps_additional':1,'serial_workers':True,'recorded_cleanup_survivors':0,
        'recorded_elapsed_UTC_seconds':elapsed,'live_files_hashed':len(seen),'forbidden_API_attempts':len(attempts),
        'scope':'Standalone INTEGER128 helper and observed executed fenv cohorts only; no physical geometry, owned/current snapshot, generated caller lifetime, scientific reducer or performance qualification. Refusal reasons remain offline metadata; native API is logical only.',
        'audit_native_calls':0,'audit_compiler_calls':0,'audit_loader_calls':0,'audit_source_sha256':sha(Path(__file__).read_bytes()),'elapsed_seconds':time.monotonic()-start}
    out=HERE/'receipt.json';assert not out.exists();out.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':'PASS','receipt':str(out),'sha256':sha(out.read_bytes()),'elapsed':receipt['elapsed_seconds']}))

if __name__=='__main__':main()
