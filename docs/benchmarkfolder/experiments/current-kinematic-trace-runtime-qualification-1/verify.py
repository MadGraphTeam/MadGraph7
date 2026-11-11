"""Portable closed-trace replay: text and pure decoding only, never native code."""
import ast
from decimal import Decimal, localcontext
from fractions import Fraction as F
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BUILD_SEAL = '89697913341ecc5fe765fb023cc09bced783eb3eedb85436afe1ca7d4dc6c033'
WORK = '.tirust-build/kinematic-provenance-current-executor-1/'
SOURCE_SEAL = '7ea709b48d8de462b754fabfb47dcbd5d21d7847458382f5844190028e8c1bed'
DERIVED_SEAL = '25c2114756ed3eb968c311945a27d175c5c5a0af54d4ae21158abc8ed42a4184'
CLOSURES = {
    'ordinary': '56d4249509bbda4d1aa3c328b18b42176c7419f4805a1d4454a73a7fa3608fcf',
    'stress': 'df0987b3f0c20dd26c4a903ff7c4acc7fb6cce4e433a50071a82918194553e9e',
}
ANALYSES = {
    'decoder': 'df0c1113bee8a88bc6e9a85b6770faa17820e66707ff9d33e45776e69a939040',
    'ordinary': '5842fea99e508654bf9f032f371530943cc80cb4cd188aab0bb914281537a453',
    'stress': '66fab0dde170020c83840364845d4413f7130f1c09dab1105fd2e95a0f13611d',
}
GUARDS = {
    'ordinary': '325eb3205b7990e5e8ec367b25ef0af48b41d7271e227dbbd15bdc878be560cf',
    'stress': '9c81ae01f16c4e9a02493a3fac49198c59f0540c2f079c0a4d2aafee2a004735',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def packet(root, expected_seal=None):
    raw = (root / 'evidence-index.json').read_bytes()
    if expected_seal:
        require(sha(raw) == expected_seal, 'Build sibling seal changed')
    index = json.loads(raw)
    paths = list(root.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'Packet contains a symlink')
    require({str(p.relative_to(root)) for p in paths if p.is_file()} ==
            set(index['files']) | {'evidence-index.json'}, 'Packet file set changed')
    for name, pin in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe stored path')
        content = (root / name).read_bytes()
        require(sha(content) == pin['sha256'] and len(content) == pin['bytes'], 'Stored bytes changed: ' + name)
        decoded = gzip.decompress(content) if name.endswith('.gz') else content
        require(b'\0' not in decoded and not decoded.startswith((b'\x7fELF', b'!<arch>')), 'Native payload retained')
        decoded.decode('utf-8')

    def data(name):
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe logical path')
        compressed = index['compressed_text'].get(name)
        if compressed:
            require(compressed['path'] in index['files'], 'Unindexed compressed path')
            content = gzip.decompress((root / compressed['path']).read_bytes())
            require(len(content) == compressed['raw_bytes'] and sha(content) == compressed['raw_sha256'],
                    'Raw compressed identity changed')
            return content
        require(name in index['files'], 'Unindexed logical path')
        return (root / name).read_bytes()

    for name, pin in index['origins'].items():
        content = data(name)
        require(len(content) == pin['bytes'] and sha(content) == pin['sha256'], 'Original copy identity changed')
    return index, data, lambda name: json.loads(data(name))


def main():
    index, data, document = packet(HERE)
    build_index, build_data, build_document = packet(HERE.parent / 'current-kinematic-trace-build-qualification-1', BUILD_SEAL)
    summary = document('qualification-summary.json')
    require(summary['build_sibling_sha256'] == BUILD_SEAL == index['build_dependency_sha256'] and
            summary['build_sibling'] == '../current-kinematic-trace-build-qualification-1', 'Dependency differs')
    require(sha(build_data('source/index.json')) == SOURCE_SEAL and
            sha(build_data('results/derived/index.json')) == DERIVED_SEAL, 'Original source/derived seal differs')
    for name, checksum in summary['closed_inputs'].items():
        require(sha(data(name)) == checksum, 'Closed input changed')
    closures = {}
    selected_origins = {WORK + name for name in summary['closed_inputs']}
    for stage, checksum in CLOSURES.items():
        name = 'results/' + stage + '-terminal-closure.json'
        require(sha(data(name)) == checksum, 'Closed runtime stage changed')
        closures[stage] = document(name)
        guard_name = stage + '-admission-1/index.json'
        require(sha(data(guard_name)) == GUARDS[stage], 'Runtime admission source changed')
        guard = document(guard_name)
        for relative, pin in guard['files'].items():
            logical = stage + '-admission-1/' + relative
            require({'sha256': sha(data(logical)), 'bytes': len(data(logical))} == pin, 'Guard source payload changed')
            selected_origins.add(WORK + logical)
        closed = closures[stage]
        for original, pins in closed['commands'].items():
            require(original.startswith(WORK), 'Foreign command origin')
            selected_origins.update([original, str(Path(original).with_suffix('.txt'))])
        selected_origins.update(closed['closed_receipts'])
        for original, pin in closed['closed_receipts'].items():
            require(original.startswith(WORK), 'Foreign closed receipt')
            raw = data(original[len(WORK):])
            require({'bytes': len(raw), 'sha256': sha(raw)} == pin, 'Closed stage receipt differs')
        admission = 'results/' + stage + '-admission-1/'
        require(document(admission + 'before.json') == document(admission + 'after.json'),
                'Protected runtime inputs changed')
        require(document(admission + 'result.json') == closed['guard_result'] and
                closed['guard_result']['guard_index_sha256'] == GUARDS[stage] and
                closed['guard_result']['postcheck_failure'] is None and
                closed['stage']['source_index_sha256'] == SOURCE_SEAL and
                closed['stage']['derived_index_sha256'] == DERIVED_SEAL,
                'Runtime source/guard closure mismatch')
        for worker, item in closed['workers'].items():
            selected_origins.update(item['files'])
            request = document('results/runs/' + worker + '/specification.json')['request']
            for key in ['numerical_card', 'physical_card', 'bootstrap', 'measurement']:
                logical = 'inputs/' + request[key]
                require(data(logical) == build_data('source/' + logical), 'Literal source input differs')
                selected_origins.add(WORK + 'source/' + logical)

    # This is our pure retained-data checker, loaded only after its packet bytes
    # have been verified. No original native controller is imported or executed.
    spec = importlib.util.spec_from_file_location('closed_kinematic_worker_checks', HERE / 'worker_checks.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    worker_summary = checks.verify_workers(data, document, build_data, build_document, closures)

    for label, checksum in ANALYSES.items():
        directory = 'analysis/' + label
        require(sha(data(directory + '/index.json')) == checksum, 'Analysis source/result seal changed')
        analysis_index = document(directory + '/index.json')
        original_directory = '.tirust-build/kinematic-provenance-analysis-' + ('' if label == 'decoder' else label + '-') + '1'
        selected_origins.add(original_directory + '/index.json')
        for name, pin in analysis_index['files'].items():
            raw = data(directory + '/' + name)
            require({'bytes': len(raw), 'sha256': sha(raw)} == pin, 'Analysis payload changed')
            selected_origins.add(original_directory + '/' + name)
    require({x['path'] for x in index['origins'].values()} == selected_origins,
            'Unexpected or missing closed-origin selection')
    decoder_raw = data('analysis/decoder/decode_trace.py')
    require(sha(decoder_raw) == '6a44cfd35294692f401b9be0d28458b2631f15fca1d80021d7e72a7fabbe2a7b', 'Decoder changed')
    names = {'require', 'digest', 'ieee', 'wire', 'payload', 'parse', 'values', 'vectors',
             'subtract', 'squared', 'norm2', 'one', 'analyze_call', 'analyze'}
    functions = [node for node in ast.parse(decoder_raw).body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in functions} == names, 'Pure decoder function set changed')
    namespace = {'Decimal': Decimal, 'localcontext': localcontext, 'F': F, 'hashlib': hashlib, 'json': json, 're': re,
                 'HEADER': ('point', 'serial', 'id', 'mode', 'rank', 'n', 'i', 'j', 'branch', 'split_order'),
                 'WORD': re.compile(r'[0-9A-Fa-f]{16}\Z')}
    # source_binding and main are deliberately excluded: only in-memory analysis
    # sees the captured raw bytes and the already verified sibling derivation.
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'sealed-pure-kinematic-decoder', 'exec'), namespace)
    decoded_counts, nonzero_statuses = {}, {}
    derived = build_document('results/derived/index.json')
    for stage, expected_count in [('ordinary', 4), ('stress', 6)]:
        directory = 'analysis/' + stage
        analysis_index = document(directory + '/index.json')
        require(analysis_index['decoder_source_index_sha256'] == ANALYSES['decoder'] and
                analysis_index['root_verified_run_closure_sha256'] == CLOSURES[stage] and
                analysis_index.get('independent_accuracy_qualified', analysis_index.get('accuracy_qualified')) is False and
                analysis_index['new_native_executions'] == 0,
                'Analysis authority/scope mismatch')
        receipt = document(directory + '/receipt.json')
        require(receipt['source_index_sha256'] == ANALYSES['decoder'] and
                receipt['root_verified_run_closure_sha256'] == CLOSURES[stage] and
                receipt['native_calls'] == 0 and len(receipt['runs']) == expected_count, 'Analysis receipt mismatch')
        count = nonzero = 0
        for run in receipt['runs']:
            name = run['name']
            require(name in closures[stage]['workers'] and name.endswith('-enabled') and run['returncode'] == 0,
                    'Analysis selected an unclosed/non-enabled trace')
            original = run['trace']['path'] if isinstance(run['trace'], dict) else run['trace']
            checksum = run['trace']['sha256'] if isinstance(run['trace'], dict) else run['trace_sha256']
            require(original.startswith(WORK), 'Analysis trace path outside retained executor')
            raw = data(original[len(WORK):])
            require(sha(raw) == checksum == closures[stage]['workers'][name]['trace_pin']['sha256'], 'Analysis trace mismatch')
            saved = document(directory + '/' + run['output'])
            case = saved['case']
            derivation = build_document('results/derived/' + case + '/derivation.json')
            routing = build_document('results/derived/' + case + '/routing.json')
            result = namespace['analyze'](raw, case, 'low-first', derivation, next(iter(routing['cases'].values())))
            selected = [case + '/' + n for n in ['CT_interface.f', 'TIRUST_interface.f', 'derivation.json', 'routing.json']]
            result.update(derived_index_sha256=DERIVED_SEAL, derived_files={n: derived['files'][n] for n in selected})
            require(namespace['wire'](result) == saved, 'Decoded analysis does not replay exactly: ' + name)
            count += 1
            nonzero += sum(row['data']['native_status'] != 0 for row in result['records'] if row['tag'].startswith('KT_STATUS_'))
        decoded_counts[stage] = count
        nonzero_statuses[stage] = nonzero
    require(decoded_counts == {'ordinary': 4, 'stress': 6} and nonzero_statuses == {'ordinary': 0, 'stress': 22},
            'Actual decoded status observations differ')
    ordinary_review = document('results/root-ordinary-review.json')
    stress_review = document('results/root-stress-review.json')
    require(ordinary_review['closure_sha256'] == CLOSURES['ordinary'] and ordinary_review['independent_parity_replay_matches'] and
            stress_review['closure_sha256'] == CLOSURES['stress'] and stress_review['independent_failed_parity_replay_matches'] and
            stress_review['failed_numerical_gate_preserved'], 'Root closure review mismatch')
    decoder_review = document('results/root-ordinary-decoder-review.json')
    require(decoder_review['analysis_index_sha256'] == ANALYSES['ordinary'] and
            decoder_review['decoder_sha256'] == sha(decoder_raw) and decoder_review['recomputed_outputs_exact'],
            'Root ordinary decoder review mismatch')
    require(not summary['independent_accuracy_qualified'] and not summary['stability_qualified'] and
            not summary['performance_qualified'] and not summary['stress_equal_observation_bytes_promoted_to_success'],
            'Qualification scope broadened')
    print(json.dumps({'portable_runtime_evidence_verified': True, 'worker_checks': worker_summary,
                      'exact_pure_decoder_replays': decoded_counts, 'nonzero_tensor_status_records': nonzero_statuses,
                      'external_binary_reads': 0, 'new_native_executions': 0,
                      'independent_accuracy_stability_performance_qualified': False}, indent=2))


if __name__ == '__main__':
    main()
