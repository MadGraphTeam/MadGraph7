"""Keep held higher-Higgs controls bound before any numerical data is read."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/higher-higgs-controls/control.py'
SUPPLEMENT = ROOT / 'tools/ninjars_supplement_v2.py'


def selected_functions(path, names, scope):
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise AssertionError('Missing real source functions')
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)


class LaterValidation(Exception):
    pass


class HigherHiggsControlBindingsTest(unittest.TestCase):
    def setUp(self):
        parent = ROOT / '.ninjars-build/tmp'
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent, prefix='higher-higgs-software-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / 'run'; self.run.mkdir()
        self.held = self.root / 'held.json'
        self.cases = [{'id': 'synthetic-hhh', 'seed': 101}, {'id': 'synthetic-hhhh', 'seed': 102}]
        self.configuration = {'cases': self.cases, 'points_per_case': 2, 'energy_gev': 1000}
        self.sources = [self.root / ('source-%d.py' % index) for index in range(9)]
        self.checked_scope = {
            'policy': SimpleNamespace(path=lambda name: self.root / name),
            'digest': self.digest,
        }
        selected_functions(SUPPLEMENT, {'checked'}, self.checked_scope)
        self.checked = mock.Mock(wraps=self.checked_scope['checked'])
        self.generation = SimpleNamespace(CASES=self.cases, source_inventory=mock.Mock())
        self.scope = {'ROOT': self.root, 'RUN': self.run, 'SPECIFICATION': self.held,
                      'SOURCE_PATHS': self.sources, 'CONFIGURATION': self.configuration,
                      'generation': self.generation, 'workers': SimpleNamespace(digest=self.digest),
                      'supplement': SimpleNamespace(checked=self.checked), 'json': json, 'Path': Path,
                      'pair': SimpleNamespace(validate_origin_records=mock.Mock(), actual_runtime=mock.Mock())}
        selected_functions(SOURCE, {'read', 'check_inputs', 'validate'}, self.scope)
        self.spec = {
            'schema': 'higher-higgs-fixed-two-point-proposal-v1', 'configuration': self.configuration,
            'targets': [6, 8], 'loops': 1, 'fresh_process_invocations': 4,
            'evaluation_wall_budget_seconds': 300, 'per_process_timeout_seconds': 120,
            'sources': {path.name: self.write(path.name, 'source\n') for path in self.sources},
            'production_inventory': self.write_json('inventory.json', {'files': {}}),
            'origins': {}, 'points': {},
        }
        for case in self.cases:
            cid = case['id']; point = self.write(cid + '.dat', '2\nsynthetic stream\n')
            self.spec['points'][cid] = {'data': point, 'metadata': self.write_json(cid + '-points.json',
                {'points': 2, 'energy_gev': 1000, 'seed': case['seed'], 'sha256': point['sha256']})}
            proof = {'case': case}
            for name in ('generation', 'compilation', 'compiled_manifest', 'physical_card',
                         'runtime_signature', 'madloop_origin_card'):
                proof[name] = self.write_json(cid + '-' + name + '.json', {})
            self.spec['origins'][cid] = self.write_json(cid + '-proof.json', proof)
        text = json.dumps(self.spec, sort_keys=True) + '\n'
        self.held.write_text(text); (self.run / 'input-spec.json').write_text(text)
        self.authorization = {'root_authorized': True, 'cpu': 190,
                              'purpose': 'higher-higgs-two-point-correctness',
                              'specification_sha256': self.digest(self.held)}
        self.write_authorization(self.authorization)

    @staticmethod
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write(self, name, text):
        path = self.root / name; path.write_text(text)
        return {'path': name, 'sha256': self.digest(path), 'bytes': path.stat().st_size}

    def write_json(self, name, value):
        return self.write(name, json.dumps(value, sort_keys=True) + '\n')

    def write_authorization(self, value):
        (self.run / 'root-handoff.json').write_text(json.dumps(value) + '\n')

    def test_changed_retained_spec_is_rejected_before_handoff_or_physics_reads(self):
        (self.run / 'input-spec.json').write_text(self.held.read_text() + ' ')
        read = mock.Mock(side_effect=AssertionError('No JSON or physics reads are allowed yet'))
        checks = mock.Mock(side_effect=AssertionError('No numerical-input reads are allowed yet'))
        with mock.patch.dict(self.scope, read=read, check_inputs=checks):
            with self.assertRaisesRegex(ValueError, 'Retained input spec differs'):
                self.scope['validate']()
        read.assert_not_called(); checks.assert_not_called()

    def test_handoff_requires_exact_spec_purpose_cpu_and_strict_root_true(self):
        changes = [('root_authorized', None), ('root_authorized', 1), ('cpu', None), ('cpu', 6),
                   ('purpose', None), ('purpose', 'other'), ('specification_sha256', None),
                   ('specification_sha256', '0' * 64)]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                handoff = copy.deepcopy(self.authorization)
                if value is None:
                    handoff.pop(key)
                else:
                    handoff[key] = value
                self.write_authorization(handoff)
                checks = mock.Mock(side_effect=AssertionError('No numerical-input reads are allowed yet'))
                with mock.patch.dict(self.scope, check_inputs=checks):
                    with self.assertRaisesRegex(ValueError, 'original root handoff'):
                        self.scope['validate']()
                checks.assert_not_called()

    def test_matching_handoff_reaches_later_validation_without_reading_results(self):
        checks = mock.Mock(side_effect=LaterValidation('reached checked inputs'))
        with mock.patch.dict(self.scope, check_inputs=checks):
            with self.assertRaises(LaterValidation):
                self.scope['validate']()
        checks.assert_called_once_with(self.spec)
        self.assertFalse((self.run / 'status.json').exists())

    def test_exact_nine_source_keys_and_matching_key_paths_precede_physics_reads(self):
        for kind in ('missing', 'extra', 'key-path'):
            with self.subTest(kind=kind):
                spec = copy.deepcopy(self.spec)
                if kind == 'missing':
                    spec['sources'].pop(self.sources[0].name)
                elif kind == 'extra':
                    spec['sources']['unexpected.py'] = self.write('unexpected.py', 'extra\n')
                else:
                    spec['sources'][self.sources[0].name]['path'] = self.sources[1].name
                checks = mock.Mock(side_effect=AssertionError('Source closure must reject first'))
                reader = mock.Mock(side_effect=AssertionError('No physics reads are allowed yet'))
                with mock.patch.dict(self.scope, supplement=SimpleNamespace(checked=checks), read=reader):
                    with self.assertRaisesRegex(ValueError, 'collector dependency|key/path mismatch'):
                        self.scope['check_inputs'](spec)
                checks.assert_not_called(); reader.assert_not_called()

    def test_correct_source_and_point_bindings_reach_end_of_retained_input_guard(self):
        self.assertIsNone(self.scope['check_inputs'](self.spec))
        self.assertEqual(self.checked.call_count, 9 + 1 + 2 * (1 + 2 + 6))
        self.generation.source_inventory.assert_not_called()
        self.scope['pair'].validate_origin_records.assert_not_called()
        self.scope['pair'].actual_runtime.assert_not_called()

    def test_changed_authorized_point_bytes_or_rebound_count_are_rejected(self):
        cid = self.cases[0]['id']; original = self.spec['points'][cid]
        data = self.root / original['data']['path']; retained = data.read_bytes()
        data.write_bytes(retained + b'changed\n')
        with self.assertRaisesRegex(ValueError, 'Changed supplement evidence'):
            self.scope['check_inputs'](self.spec)
        spec = copy.deepcopy(self.spec)
        spec['points'][cid]['data'] = self.write(data.name, 'different authorized data\n')
        with self.assertRaisesRegex(ValueError, 'Changed authorized two-point stream'):
            self.scope['check_inputs'](spec)
        data.write_bytes(retained)
        spec = copy.deepcopy(self.spec)
        metadata = json.loads((self.root / original['metadata']['path']).read_text())
        metadata['points'] = 3
        spec['points'][cid]['metadata'] = self.write_json(original['metadata']['path'], metadata)
        with self.assertRaisesRegex(ValueError, 'Changed authorized two-point stream'):
            self.scope['check_inputs'](spec)


class HigherHiggsFinalizerAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = SOURCE.with_name('finalize_v1.py')
        spec = importlib.util.spec_from_file_location('higher_higgs_finalizer_regression', path)
        cls.finalizer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.finalizer)
        sys.path.insert(0, str(ROOT / 'tools'))
        import ninjars_benchmark_policy_v2 as policy2
        import ninjars_benchmark_policy_v3 as policy3
        cls.policy2, cls.policy3 = policy2, policy3

    @staticmethod
    def groups(delta):
        left = [{'point': 1, 'order': 0, 'coefficients': [0., 1., -2., 3.]},
                {'point': 1, 'order': 1, 'coefficients': [0., 1., -2., 3.]}]
        right = copy.deepcopy(left)
        right[0]['coefficients'][1] += delta
        return {'candidate-6': left, 'candidate-8': right}

    def test_real_comparator_accepts_difference_just_within_tolerance(self):
        groups = self.groups(1e-7)
        result = self.finalizer.compare_rows(groups)
        self.assertEqual(result, self.policy2.compare_rows(groups))
        self.assertEqual(result, {'passed': True, 'pairs': 1, 'failures': []})

    def test_real_comparator_records_difference_above_tolerance(self):
        groups = self.groups(2e-7)
        result = self.finalizer.compare_rows(groups)
        self.assertEqual(result, self.policy2.compare_rows(groups))
        self.assertFalse(result['passed'])
        self.assertEqual(result['pairs'], 1)
        self.assertEqual(len(result['failures']), 1)
        failure = result['failures'][0]
        self.assertEqual((failure['point'], failure['order'], failure['coefficient']), (1, 0, 1))
        self.assertEqual(failure['targets'], ['candidate-6', 'candidate-8'])
        self.assertGreater(abs(failure['values'][0] - failure['values'][1]), failure['tolerance'])

    def test_real_reader_uses_v2_function_without_mutating_imported_v3_module(self):
        before = vars(self.policy3).copy()
        reader = self.finalizer.load_reader()
        self.assertIs(reader.policy3, self.policy2)
        self.assertIs(reader.policy3.compare_rows, self.policy2.compare_rows)
        self.assertFalse(hasattr(self.policy3, 'compare_rows'))
        self.assertEqual(set(vars(self.policy3)), set(before))
        for name, value in before.items():
            self.assertIs(vars(self.policy3)[name], value)

    def test_changed_expected_frozen_controller_hash_is_rejected(self):
        with mock.patch.object(self.finalizer, 'CONTROLLER_SHA256', '0' * 64):
            with self.assertRaisesRegex(ValueError, 'original measured controller must remain unchanged'):
                self.finalizer.load_reader()


if __name__ == '__main__':
    unittest.main()
