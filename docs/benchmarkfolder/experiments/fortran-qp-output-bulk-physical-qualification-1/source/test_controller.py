"""Pure source/ownership guards and a fake-worker timing-history integration."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import types
import pytest

HERE=Path(__file__).resolve().parent
def load(name):
    spec=importlib.util.spec_from_file_location('bulk_physical_test_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_reviewed_ownership_definitions_are_unchanged():
    own=(HERE/'ownership.py').read_text();ancestry=json.loads((HERE/'ownership-provenance.json').read_text())
    original=Path(ancestry['source']).read_text()
    assert hashlib.sha256(original.encode()).hexdigest()==ancestry['source_sha256']
    for name,pin in ancestry['functions'].items():
        current=next(n for n in ast.parse(own).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        old=next(n for n in ast.parse(original).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        assert ast.get_source_segment(own,current)==ast.get_source_segment(original,old)
        assert hashlib.sha256(ast.get_source_segment(own,current).encode()).hexdigest()==pin


def test_source_files_parse_and_phase_deadline_contains_preflight():
    for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    s=(HERE/'controller.py').read_text()
    stage=s[s.index('def stage():'):s.index('def main():')]
    assert stage.index('ownership.StageDeadline(CAPS[PHASE])')<stage.index('preflight()')
    assert 'owned=process=raw_record=None' in stage
    assert stage.index('process=subprocess.Popen')<stage.index('raw_record=cleanup.identity')<stage.index('owned=cleanup.OwnedChild')
    assert "CAPS={'build':300.,'parity':600.,'timing':480.}" in s


def test_frozen_runtime_join_rejects_conflict_and_changed_provider(tmp_path):
    controller=load('controller');provider=tmp_path/'libfrozen.so';provider.write_bytes(b'closed provider bytes')
    name=str(provider);checksum=controller.digest(provider);expected={name:checksum}
    pre={'bound':{}};controller.bind_frozen_runtime(pre,expected)
    assert pre['bound']==expected and pre['explicit_frozen_runtime_before_link']==expected
    with pytest.raises(ValueError,match='conflict'):
        controller.bind_frozen_runtime({'bound':{name:'0'*64}},expected)
    provider.write_bytes(b'changed provider bytes')
    with pytest.raises(ValueError,match='changed before link'):
        controller.bind_frozen_runtime({'bound':{}},expected)


def test_runtime_join_and_fixed_shell_precede_link():
    text=(HERE/'controller.py').read_text();tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_locked')
    body=ast.get_source_segment(text,node)
    assert body.index('loader.prelink(')<body.index('bind_frozen_runtime(')<body.index("dump(proc/'prelink.json'")<body.index('command(link,')<body.index('loader.query_runtime(')
    assert "'SHELL='+MAKE_SHELL" in body
    pin=json.loads((HERE/'input-pins.json').read_text())
    controller=load('controller')
    assert pin['tools']['make_shell']==controller.MAKE_SHELL
    assert controller.digest(controller.MAKE_SHELL)==pin['files'][controller.MAKE_SHELL]


def test_compiler_environment_reconstructed_without_loader_injection(monkeypatch):
    controller=load('controller')
    pin={'environment':{'NIX_REVIEWED':'fixed','FFLAGS':None},'dso_path':'/frozen/tirust/lib/libmadloop_tirust.so'}
    monkeypatch.setattr(controller,'read',lambda path:pin)
    for key in ('LD_AUDIT','LD_DEBUG','LD_PRELOAD','NIX_UNREVIEWED','BASH_ENV','ENV','FFLAGS'):
        monkeypatch.setenv(key,'unreviewed')
    result=controller.environment()
    assert not any(k in result for k in ('LD_AUDIT','LD_DEBUG','NIX_UNREVIEWED','BASH_ENV','ENV','FFLAGS'))
    assert result['LD_PRELOAD']=='' and result['NIX_REVIEWED']=='fixed'
    assert result['OMP_NUM_THREADS']==result['RAYON_NUM_THREADS']=='1'


def test_matched_calibration_and_seven_pair_histories(tmp_path,monkeypatch):
    controller=load('controller');analysis=load('analyze')
    monkeypatch.setattr(controller,'local',lambda name:analysis)
    monkeypatch.setattr(controller,'verify_source',lambda:None)
    monkeypatch.setattr(controller,'protected',lambda:None)
    monkeypatch.setattr(controller,'verify_build',lambda:{'executables':{name+'-timing':{} for name in ('original','candidate')}})
    reference={str(i):dict(point=point,phase=i,status=code,ans_words={'0:'+str(j):'0000000000000000' for j in range(4)},accuracy_words={'0':'0000000000000000'})
               for i,point,code in [(1,1,219),(2,16,323)]}
    monkeypatch.setattr(controller,'verify_parity',lambda:{'variants':{'original-parity':{'observations':{'values':reference}}}})
    monkeypatch.setattr(controller,'mapped',lambda worker,row:{'fake_source_fixture':True})
    histories={'original':[],'candidate':[]}
    ready=dict(points=100,ndim=0,nso=0,nborn=0,cold_seconds=1.)
    def start(variant,row,directory,stack):
        name=variant.split('-')[0];proc=directory/'proc';proc.mkdir()
        return types.SimpleNamespace(name=name),proc,{},ready
    monkeypatch.setattr(controller,'start_worker',start)
    def custom(worker,proc,directory,line,expected,files):
        command,seq,point,loops=line.split();seq,point,loops=map(int,(seq,point,loops))
        assert command=='ANCHORRUN' and expected==['ANCHORDONE',str(seq),str(point),str(loops)]
        histories[worker.name].append((seq,point,loops))
        directory.mkdir(parents=True)
        status=219 if point==1 else 323
        meta=f'META {seq} {point} {loops} 0 0 0\n'
        per_call=1000 if point==1 else 1200000
        if worker.name=='candidate':per_call=per_call*11//10
        (directory/'anchor-timing.dat').write_text(meta+f'TIMING {seq} {point} {loops} 100 {100+loops*per_call} 1000000\n')
        rows=[f'ANS {seq} {loops} 4 {point} 0 {component} {status} 0000000000000000' for component in range(4)]
        rows.append(f'ACC {seq} {loops} 4 {point} 0 {status} 0000000000000000')
        (directory/'anchor-words.dat').write_text(meta+'\n'.join(rows)+'\n')
        (directory/'anchor-status.dat').write_text(f'META {seq} {point} {loops}\nSTATUS {seq} {point} {loops} {status} {loops}\n')
    monkeypatch.setattr(controller,'custom',custom)
    controller.timing_locked(tmp_path)
    assert histories['original']==histories['candidate']
    assert len(histories['original'])==18
    assert [x[0] for x in histories['original']]==list(range(1,19))
    result=json.loads((tmp_path/'result.json').read_text())
    for point in ('1','16'):
        row=result['anchors'][point]
        assert len(row['pairs'])==7
        assert [p['order'] for p in row['pairs']]==[['original','candidate'] if i%2==0 else ['candidate','original'] for i in range(7)]
        assert row['summary']['median_paired_ratio']>1
        assert not row['summary']['candidate_promoted'] and not result['performance_acceptance']
