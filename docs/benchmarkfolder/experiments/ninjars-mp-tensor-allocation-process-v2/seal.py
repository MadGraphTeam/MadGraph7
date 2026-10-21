#!/usr/bin/env python3
"""Create or verify a complete text/source/raw-data seal, excluding build products."""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inventory():
    answer = {}
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or path.name == 'manifest.json' or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        assert path.suffix not in ['.a','.o','.mod','.so','.rlib'],path
        content = path.read_bytes()
        assert not content.startswith(b'\x7fELF'),path
        answer[str(path.relative_to(HERE))] = {'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
    return answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--create',action='store_true')
    args = parser.parse_args()
    path = HERE / 'manifest.json'
    files = inventory()
    identity = hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if args.create:
        measurement = json.loads((HERE / 'timings/attempt-1/result.json').read_text())
        analysis = json.loads((HERE / 'timings/attempt-1/analysis.json').read_text())
        assert measurement['status'] == 'complete-unanalysed' and len(measurement['samples']) == 56
        assert analysis['promotion_gates_passed']
        audits = [p for p in HERE.glob('independent-*.json') if 'readiness' not in p.name and 'recovery' not in p.name]
        assert audits, 'Independent measurement audit must be present before sealing'
        assert all(json.loads(p.read_text()).get('passed') for p in audits)
        record = {'schema':'allocation-final-process-retention-v2','status':'complete-decision-gates-passed',
            'identity':identity,'file_count':len(files),'total_bytes':sum(r['bytes'] for r in files.values()),
            'files':files,'production_changed':False,'final_acceptance_eligible':False,
            'qcd3_pairs':21,'ew2_pairs':7,'all56_workers_exit_zero':True,
            'exact_coefficients_accuracy_status_and_raw_output':True,
            'physical_component_comparisons':44800,'historical_or_pause_spanning_samples_used':False,
            'optimization_decision':{'qcd3_ratio':analysis['cases']['qcd-3']['ratio'],
                'qcd3_interval95':analysis['cases']['qcd-3']['confidence_interval_95'],
                'ew2_ratio':analysis['cases']['ew-2']['ratio'],
                'ew2_interval95':analysis['cases']['ew-2']['confidence_interval_95']},
            'excluded':['Python bytecode','all compiled products/caches/static archives and generated executable trees under .benchmark-work'],
            'required_followup':'Final combined regression suite, actual installation/export checks, full original-Ninja acceptance benchmark and report remain root responsibilities.'}
        with path.open('x') as stream:
            json.dump(record,stream,indent=2,sort_keys=True)
            stream.write('\n')
    record = json.loads(path.read_text())
    assert record['files'] == files and record['identity'] == identity
    assert record['file_count'] == len(files)
    assert record['total_bytes'] == sum(r['bytes'] for r in files.values())
    print(json.dumps({'passed':True,'manifest_sha256':sha(path),'identity':identity,
        'files':len(files),'bytes':record['total_bytes']}),flush=True)

if __name__ == '__main__':
    main()
