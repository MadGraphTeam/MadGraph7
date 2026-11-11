"""Pure standalone snapshot expectation/state replay from authenticated bytes.

The only corpus substitution removes its filesystem/import bootstrap and
captures generate()'s two writes in memory.  Exact algorithm source segments
and ASTs are preserved.  Execution, ABI, origin and scientific authorization
remain the outer owner's responsibility, never an inference of this parser.
"""
import ast
import dataclasses  # Preload the frozen reference's sole nonbuiltin dependency.
import hashlib
import json


SOURCE = '41797bbf9e25eb1836932307f66ad74f413714dd38cf1485e07cc8f2d74160d9'
PARENT = 'faae6af3bb71d490fde1ab7bb46a77c43e99991ec314d4a6a9eeb921ff9a1015'
REFERENCE_RECEIPT = 'a953bc69888029ceae35d576f9c5557cf046d283b89daa6cad3cfce24aec7011'
REFERENCE = '7e75a1b06528461161fc6176cb06ce61e05e629d073d2f3d8bd0efc797625155'
RAW = '6b75d55910c8f00b0ae10d9a52544913ea2dfa7cf20a477326eda69b32e2c31b'
EXTERNAL = 'external/.tirust-build/generic-owned-snapshot-source-1/'
FUNCTIONS = ('need', 'dirty', 'export', 'expect', 'cases', 'render', 'generate')
CONSTANTS = ('DIRTY', 'SCENARIOS', 'DEFAULT_DP', 'DEFAULT_QP')
FALSE_ANALYSIS_FLAGS = (
    'execution_authorization_checked', 'native_snapshot_transport_qualified',
    'canonical_column_origin_qualified', 'actual_constructor_completion_qualified',
    'q_at_use_lowering_qualified', 'final_dot_runtime_qualified',
    'full_caller_qualified', 'scientific_dispatch_enabled',
)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


class MemoryOutput:
    """Two exact write_text destinations, without any filesystem operation."""
    def __init__(self):
        self.files = {}

    def __truediv__(self, name):
        require(name in ('fixtures.hex', 'expectations.json'),
                'Corpus attempted an unexpected output destination')
        owner = self

        class TextFile:
            def write_text(self, value, *args, **kwargs):
                require(type(value) is str and not args and not kwargs,
                        'Corpus output API changed')
                require(name not in owner.files, 'Corpus wrote a destination twice')
                owner.files[name] = value.encode('utf-8')
                return len(value)

        return TextFile()


def corpus_algorithm(source):
    """Select exact algorithm nodes; omit only known I/O/import bootstrap."""
    text = source.decode('utf-8')
    tree = ast.parse(text)
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    require(tuple(functions) == FUNCTIONS, 'Frozen corpus function coverage changed')
    selected = []
    constants = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            selected.append(node)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1 and
              isinstance(node.targets[0], ast.Name) and node.targets[0].id in CONSTANTS):
            selected.append(node)
            constants.append(node.targets[0].id)
    require(tuple(constants) == CONSTANTS, 'Frozen corpus constant coverage changed')
    segments = [ast.get_source_segment(text, node) for node in selected]
    require(all(type(value) is str for value in segments), 'Algorithm source segment missing')
    rendered = ('\n\n'.join(segments) + '\n').encode()
    reparsed = ast.parse(rendered)
    require(ast.dump(ast.Module(body=selected, type_ignores=[]), include_attributes=False) ==
            ast.dump(reparsed, include_attributes=False),
            'Corpus algorithm AST changed while removing I/O bootstrap')
    return rendered


def replay(blobs, load_source):
    """Replay the 37-case snapshot corpus and both observed state transcripts.

    load_source(name, bytes, logical_path) loads already authenticated pure
    source.  No historical controller, import loader or native program runs.
    """
    def data(name, checksum=None):
        value = blobs[name]
        require(type(value) is bytes, 'Logical input must be bytes: ' + name)
        if checksum is not None:
            require(sha(value) == checksum, 'Frozen input changed: ' + name)
        return value

    def record(name, checksum=None):
        return json.loads(data(name, checksum))

    record('source/source-index.json', SOURCE)
    parent = record('parent/source-receipt.json', PARENT)
    reference_receipt = record(EXTERNAL + 'source-receipt.json', REFERENCE_RECEIPT)
    reference_bytes = data(EXTERNAL + 'snapshots_spec.py', REFERENCE)
    require(reference_receipt['payloads']['snapshots_spec.py'] ==
            {'bytes': len(reference_bytes), 'sha256': REFERENCE},
            'Independent reference bytes do not join their frozen receipt')
    for name in ('corpus.py', 'analyze.py', 'fixtures.hex', 'expectations.json'):
        value = data('parent/' + name)
        require(parent['payloads'][name] == {'bytes': len(value), 'sha256': sha(value)} and
                data('source/' + name) == value,
                'Qualified corpus/parser source differs from frozen parent: ' + name)
    reference = load_source('portable_owned_snapshot_reference', reference_bytes,
                            EXTERNAL + 'snapshots_spec.py')
    algorithm = corpus_algorithm(data('source/corpus.py'))
    corpus = load_source('portable_owned_snapshot_corpus', algorithm,
                         'source/corpus.py::exact_algorithm_nodes')
    output = MemoryOutput()
    corpus.reference = reference
    corpus.hashlib = hashlib
    corpus.json = json
    corpus.HERE = output
    expected = corpus.generate()
    require(set(output.files) == {'fixtures.hex', 'expectations.json'},
            'Corpus generated incomplete output')
    for name, value in output.files.items():
        require(value == data('source/' + name), 'Regenerated corpus text differs: ' + name)
    require(normalized(expected) == record('source/expectations.json'),
            'Regenerated expectations differ')
    require(expected['case_count'] == 37 and expected['mode_count'] == 20 and
            expected['native_snapshot_transport_qualified'] is False and
            expected['no_runtime_precision_tag'] is True and
            expected['canonical_column_and_ownership_are_caller_preconditions'] is True,
            'Corpus scope or coverage changed')
    analyzer = load_source('portable_owned_snapshot_analyzer', data('source/analyze.py'),
                           'source/analyze.py')
    run = record('run/result.json')
    require(run['status'] == 'passed' and run['source_index_sha256'] == SOURCE and
            set(run['variants']) == {'checked', 'optimized'} and
            run['scientific_reducer_calls'] == 0,
            'Standalone run identity, scope or variant coverage changed')
    for key in ('actual_constructor_completion_qualified', 'q_at_use_lowering_qualified',
                'final_dot_runtime_qualified', 'full_caller_qualified',
                'scientific_dispatch_enabled', 'physical_or_performance_qualification',
                'snapshot_ownership_qualified'):
        require(run[key] is False, 'Run scope inflated: ' + key)
    variants = {}
    raw_bytes = {}
    for variant in ('checked', 'optimized'):
        prefix = 'run/' + variant + '/'
        raw = data(prefix + 'observations.txt', RAW)
        parsed = analyzer.analyze(raw.decode('utf-8'), expected)
        require(normalized(parsed) == record(prefix + 'analysis.json') == run['variants'][variant],
                'Observed state analysis differs: ' + variant)
        require(parsed['status'] == 'pass' and parsed['analysis_only'] is True and
                parsed['case_count'] == 37 and parsed['cohorts'] == 20 and
                parsed['observations'] == 740 and parsed['module_calls'] == 1840 and
                parsed['complete_integer_state_words'] == 47360,
                'Observed state coverage changed: ' + variant)
        require(all(parsed[key] is False for key in FALSE_ANALYSIS_FLAGS),
                'Parser-only scope inflated: ' + variant)
        variants[variant] = parsed
        raw_bytes[variant] = raw
    require(raw_bytes['checked'] == raw_bytes['optimized'] and
            variants['checked'] == variants['optimized'],
            'Checked and optimized state observations differ')
    return {
        'status': 'pass', 'analysis_only': True,
        'source_index_sha256': SOURCE, 'parent_receipt_sha256': PARENT,
        'independent_reference_sha256': REFERENCE,
        'case_count': 37, 'cohorts_per_variant': 20, 'variant_results': variants,
        'total_observations': 1480, 'total_module_call_outcomes': 3680,
        'total_complete_integer_state_words': 94720,
        'complete_raw_transcripts_identical': True, 'raw_sha256': RAW,
        'algorithm_source_sha256': sha(algorithm), 'algorithm_AST_preserved': True,
        'I_O_only_substitution': 'Authenticated reference/import bootstrap replaced; generate writes captured in memory.',
        'new_native_executions': 0,
        'execution_authorization_checked': False, 'native_snapshot_transport_qualified': False,
        'snapshot_ownership_qualified': False,
        'canonical_column_origin_qualified': False,
        'actual_constructor_completion_qualified': False,
        'q_at_use_lowering_qualified': False, 'final_dot_runtime_qualified': False,
        'full_caller_qualified': False, 'runtime_precision_tag_or_same_buffer_qualified': False,
        'scientific_dispatch_enabled': False, 'physical_or_performance_qualification': False,
        'physical_amplitude_qualified': False, 'performance_acceptance': False,
    }
