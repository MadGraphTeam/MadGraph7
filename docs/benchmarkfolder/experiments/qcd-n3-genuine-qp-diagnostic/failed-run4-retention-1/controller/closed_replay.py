#!/usr/bin/env python3
"""Pure replay of both complete closed run3 CT4 workers with the proposed parser."""
from pathlib import Path
import ast
import gzip
import hashlib
import importlib.util
import json
import struct
import analyze

HERE = Path(__file__).resolve().parent
RETENTION = HERE.parent / 'failed-run3-retention-1'
RETENTION_SHA = '50fcb62f7d2b8b725e409755caf396f5a5f682e08edb07640d90a246a3201b0c'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def replay(retention=RETENTION):
    retention = Path(retention)
    require(digest((retention / 'evidence-index.json').read_bytes()) == RETENTION_SHA,
            'Closed failure-retention index differs')
    spec = importlib.util.spec_from_file_location('closed_run3_failure_replay', retention / 'verify.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original = module.verify(retention)
    require(original['overall_native_attempt_passed'] is False and original['adjudication_executed'] is False,
            'Original failed attempt must remain failed')
    inventory = json.loads((retention / 'retention-inventory.json').read_bytes())

    def raw(name):
        record = inventory['original_files'][name]
        value = (retention / record['member']).read_bytes()
        value = gzip.decompress(value) if record['codec'] == 'gzip' else value
        require(digest(value) == record['raw_sha256'], 'Original raw bytes differ')
        return value

    selected = json.loads(raw('controller/fixtures/selected-inputs.json'))
    old_parser=raw('controller/analyze.py').decode()
    new_parser=(HERE/'analyze.py').read_text()
    functions=lambda text:{node.name:ast.dump(node,include_attributes=False) for node in ast.parse(text).body
                           if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    before,after=functions(old_parser),functions(new_parser)
    require(set(after)-set(before)=={'widen_binary64'},'Unexpected analyzer helper addition')
    require(set(before)<=set(after),'Analyzer function removed')
    for name in before:
        if name!='observed_inputs':
            require(before[name]==after[name],'Other analyzer gate changed: '+name)
    start="    require(payload(b0)==payload(b1)[:176],'Born input mutation')"
    old_adjacent=old_parser[old_parser.index(start):old_parser.index("    return {'shared_input_sha256'")]
    new_adjacent=new_parser[new_parser.index(start):new_parser.index('    transitions=')]
    require(old_adjacent==new_adjacent,'Subsequent observed-input gates/common digest changed')
    new_controller=(HERE/'run.py').read_text().replace('run4','run3').replace('Source4','Source3').replace('source4','source3')
    require(new_controller==raw('controller/run.py').decode(),'Controller changed beyond namespace labels')
    science=json.loads(raw('controller/mechanical-change.json'))['unchanged_scientific_payloads']
    unchanged_science=[]
    for name in science:
        if name!='analyze.py':
            require((HERE/name).read_bytes()==raw('controller/'+name),'Scientific payload changed: '+name)
            unchanged_science.append(name)
    points = {p: b''.join(struct.pack('<d', float.fromhex(x)) for row in
                          selected['points'][str(p)]['binary64_hex'] for x in row) for p in (1, 93, 39)}
    workers = []
    for observed in (0, 1):
        prefix = 'actual-run3/b1-m4-o' + str(observed)
        trace, stdout = raw(prefix + '/trace.txt'), raw(prefix + '/worker.log')
        parsed = analyze.parse_worker(trace.decode(), stdout.decode(), 1, 4, observed, points)
        workers.append({'worker': 'b1-m4-o' + str(observed), 'trace_sha256': digest(trace),
                        'stdout_sha256': digest(stdout), 'parsed': parsed})
    require(workers[0]['parsed']['public_raw'] == workers[1]['parsed']['public_raw'],
            'Actual plain/observed public input/output/status/IEEE parity differs')
    return {'schema': 'source4-closed-run3-two-worker-pure-replay-v1',
            'scope': 'Two completed CTmode4 arms only; no new native evaluations or sixteen-worker qualification',
            'passed': True, 'parser_sha256': digest((HERE / 'analyze.py').read_bytes()),
            'delta_guard':{'unchanged_analyzer_functions':[name for name in before if name!='observed_inputs'],
                'unchanged_adjacent_gate_source_sha256':digest(old_adjacent.encode()),
                'unchanged_scientific_payloads':unchanged_science,'controller_only_namespace_and_label_changes':True},
            'closed_failed_run3_retention_index_sha256': RETENTION_SHA,
            'original_source3_attempt_remains_failed': True,
            'original_source3_parser_failure': original['overall_failure'],
            'candidate_parser_full_workers_passed': 2, 'matrix_workers_required': 16,
            'matrix_workers_not_replayed': 14, 'adjudication_executed': False,
            'coefficient_comparison_qualified': False, 'scientific_acceptance': False,
            'public_plain_observed_bytes_equal': True, 'workers': workers}


if __name__ == '__main__':
    print(json.dumps(replay(), sort_keys=True, indent=2, allow_nan=False))
