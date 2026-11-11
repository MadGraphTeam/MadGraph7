"""Portable, read-only hash/algebra replay. No processes or native libraries."""
import argparse,hashlib,io,json,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent

def identity(p):
    b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}

def verify(expected=None):
    index=HERE/'source-index.json'
    if expected:assert identity(index)['sha256']==expected,'unexpected index'
    data=json.loads(index.read_text())
    assert data['schema']=='rank6-rational-pole-oracle-source-index-v1'
    actual={str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}
    assert actual==set(data['files'])|{'source-index.json'},'package inventory differs'
    for rel,want in data['files'].items():
        p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
        assert identity(HERE/rel)==want,rel
    sys.path.insert(0,str(HERE))
    import analyze,test_oracle
    result=analyze.analyze()
    assert result==json.loads((HERE/'retrospective-analysis.json').read_text())
    assert result['all_coefficient_oracle_gates_pass']
    expected_future=json.loads(json.dumps(analyze.future()))
    assert expected_future==json.loads((HERE/'future-triangle-fixture.json').read_text())
    assert len(expected_future['components'])==210
    assert all(c['IR2']!=['0','0'] for c in expected_future['components'])
    log=io.StringIO()
    run=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromModule(test_oracle))
    assert run.wasSuccessful(),log.getvalue()
    assert run.testsRun==9
    return {'verified':True,'payload_files':len(data['files']),'pure_test_methods':run.testsRun,
      'retrospective_rows':len(result['rows']),'components':210,'pole_complex_coefficients':6300,
      'original_IEEE_failure_unchanged':True,'new_native_calls':0,'new_compiler_calls':0,
      'physical_stress_qualified':False,'performance_acceptance':False}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index-sha256');a=p.parse_args()
    print(json.dumps(verify(a.index_sha256),indent=2,sort_keys=True))
