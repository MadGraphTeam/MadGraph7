"""Synthetic filesystem-only clone tests; no production/native/loader query."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('four_lepton_pure_preparation', Path(__file__).with_name('preparation.py'))
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)


def forbidden(*args, **kwargs):
    raise AssertionError('Native/compiler/loader/source-query path called')


class Fixture:
    def __init__(self, root):
        self.root = root
        self.origin = root / prep.ORIGIN
        report, work = self.origin / 'report', self.origin / 'work'
        self.bench = SimpleNamespace(ROOT=root, REPORT=report, WORK=work, MODEL_ROOT=work/'models', digest=digest,
            runtime_identity=forbidden, environment=forbidden, source_identity=forbidden, campaign_guard=forbidden,
            CAMPAIGN_ID=None, CAMPAIGN_MANIFEST=None, campaign_binding=lambda: None, TIRUST_PREFIX=root/'native')
        self.bench.generated_sources = lambda output: {
            path.relative_to(output).as_posix(): digest(path) for scope in ('Source', 'SubProcesses')
            for path in sorted((output/scope).rglob('*')) if path.is_file() and path.suffix in ('.f', '.f90')}
        self.bench.numerical_card_inputs = lambda output: {
            path.relative_to(output).as_posix(): digest(path) for path in
            [output/'Cards/MadLoopParams.dat', *(output/'SubProcesses').rglob('MadLoopParams.dat')] if path.is_file()}
        self.bench.output_dir = lambda source, case: work/'outputs'/source/case['id']
        self.bench.resolved_parameter_card = lambda directory: directory.parents[1]/'Cards/param_card.dat'
        self.bench.dependency_inputs = self.dependencies
        self.bench.verify_worker_template = self.verify_template
        self.case = {'id': prep.CASES[0], 'nfinal': 4, 'model': 'loop_sm', 'process': 'u u~ > e+ e- mu+ mu- [virt=QCD]', 'role': 'diagnostic'}
        self.config = {'cases': [self.case], 'seed': 20261109, 'points_per_case': 100, 'energy_gev': 1000.}
        self.bench.load_config = lambda: self.config
        write(report/'cases.json', self.config)
        write(root/'models/loop_sm/particles.py', 'frozen model\n')
        write(root/'tools/tirust_benchmark_worker.f', 'frozen worker\n')
        files = {'models/loop_sm/particles.py': digest(root/'models/loop_sm/particles.py'), 'absent-generator.py': None}
        self.source_sha = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        source = {'revision': prep.SOURCE_REVISION, 'files': files, 'files_sha256': self.source_sha}
        manifest = {'candidate': {'exporter_identity': source, 'numerical_identity_sha256': prep.NUMERICAL_IDENTITY}}
        write(root/prep.MANIFEST, manifest)
        write(self.origin/'status.json', {'status': 'prepared', 'amplitude_evaluations': 0, 'stages': [
            {'id': name, 'status': 'passed', 'returncode': 0, 'cleanup': {'survivors': []}}
            for name in ('generate', 'build')]})
        write(self.origin/'prepared.json', {'synthetic': True})
        write(self.origin/'case-plan.json', {'synthetic': True})
        for folder, value in (('native/one', 'first loader'), ('native/two', 'second loader')):
            write(root/folder/'ld-linux.so.2', value)
        write(root/'native/lib/libmadloop_tirust.so', 'synthetic eb3 provider')
        self.library_sha = digest(root/'native/lib/libmadloop_tirust.so')
        write(root/'input/mg7_configuration.txt', 'frozen ignored configuration')
        write(self.origin/'case-plan.json', {'configuration_files': {'input/mg7_configuration.txt': digest(root/'input/mg7_configuration.txt')},
                                           'tirust_prefix': 'native'})
        manifest['candidate']['native'] = {'prefix': str(root/'native'), 'files': {'lib/libmadloop_tirust.so': self.library_sha}}
        write(root/prep.MANIFEST, manifest)
        write(root/'native/libreducer.a', 'synthetic archive')
        write(root/'native/tirust-config.json', {'synthetic': True})
        output = self.bench.output_dir('candidate', self.case)
        directory = output/'SubProcesses/P0_fake'
        write(output/'Cards/param_card.dat', 'physical card\n')
        write(output/'Cards/MadLoopParams.dat', 'backend9\n')
        write(output/'Source/model.f', 'model source\n')
        write(output/'Source/model.o', 'model object\n')
        write(directory/'check_sa.f', 'frozen worker\n')
        write(directory/'check_sa.o', 'worker object\n')
        write(directory/'check', 'SYNTHETIC NOT EXECUTABLE\n')
        write(directory/'warmup.dat', 'old scratch preserved\n')
        write(output/'SubProcesses/MadLoop5_resources/P0_HelFilter.dat', 'original filter\n')
        (directory/'MadLoopParams.dat').symlink_to('../../Cards/MadLoopParams.dat')
        (directory/'MadLoop5_resources').symlink_to('../MadLoop5_resources', target_is_directory=True)
        (output/'lib').mkdir()
        for name in ('libreducer.a', 'tirust-config.json'):
            (output/'lib'/name).symlink_to(root/'native'/name)
        generation = {'source': 'candidate', 'case': self.case, 'source_identity': source,
                      'parameter_card_sha256': digest(output/'Cards/param_card.dat')}
        dependencies = self.dependencies(directory)
        compilation = {'sources': self.bench.generated_sources(output), 'numerical_build_from_sources': True,
            'full_numerical_rebuild_this_invocation': True, 'previous_compilation_sha256': None,
            'build_logs': {name: 'synthetic' for name in ('MODEL','DHELAS','process')},
            'executable_sha256': digest(directory/'check'), 'worker_template_sha256': digest(root/'tools/tirust_benchmark_worker.f'),
            'masses': [0.]*6, 'link_inputs_before': dependencies, 'link_inputs_after': dependencies,
            'compiler': 'synthetic compiler', 'effective_compiler_flags': {'recorded_only': True}}
        write(output/'generation.json', generation); write(output/'compilation.json', compilation)
        points = report/'points'/(prep.CASES[0]+'.dat')
        write(points, '100\nsynthetic saved stream\n')
        write(points.with_suffix('.json'), {'sha256': digest(points), 'inputs': {
            'count': 100, 'seed': 20261109, 'energy': 1000., 'incoming_masses': [0.,0.], 'masses': [0.]*4}})
        libraries = {str(root/folder/'ld-linux.so.2'): digest(root/folder/'ld-linux.so.2') for folder in ('native/one','native/two')}
        libraries[str(root/'native/lib/libmadloop_tirust.so')] = self.library_sha
        runtime = {'libraries': libraries, **dependencies, 'ldd': 'borrowed historical loader text\n'}
        donor_compilation = {'compiler': compilation['compiler'], 'executable_sha256': 'different-original-executable'}
        inputs = {'generation': {'source_identity': source}, 'compilation': donor_compilation,
                  'runtime': runtime, 'points_sha256': 'donor-point-hash', 'numerical_cards': {'donor': 'card'}}
        after = {'runtime': runtime, 'executable_sha256': donor_compilation['executable_sha256'],
                 'points_sha256': inputs['points_sha256'], 'numerical_cards': inputs['numerical_cards']}
        write(root/prep.BORROWED/'candidate-inputs.json', inputs)
        write(root/prep.BORROWED/'inputs-after.json', after)
        self.pins = {name: digest(root/name) for name in prep.PINS}

    def dependencies(self, directory):
        library = directory.parents[1]/'lib'
        return {'archives': {str((library/'libreducer.a').resolve()): digest(library/'libreducer.a')},
                'metadata': {str((library/'tirust-config.json').resolve()): digest(library/'tirust-config.json')}}

    @staticmethod
    def verify_template(compilation, template):
        if digest(template) != compilation['worker_template_sha256']:
            raise ValueError('worker changed')


class PreparationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.fixture = Fixture(self.root); self.bench = self.fixture.bench
        self.patch = patch.multiple(prep, PINS=self.fixture.pins, SOURCE_FILES_SHA=self.fixture.source_sha, LIBRARY_SHA=self.fixture.library_sha)
        self.patch.start(); self.addCleanup(self.patch.stop)

    def prepare(self):
        return prep.prepare(self.bench, self.root/'private', self.root/'records')

    def test_five_private_clones_and_explicit_unverified_runtime(self):
        original = prep.collect_originals(self.bench); result = self.prepare()
        self.assertEqual(result['clone_count'], 5); self.assertFalse(result['native_queries_performed'])
        self.assertEqual(original, prep.collect_originals(self.bench))
        self.assertIsNone(result['campaign_binding'])
        for row in result['clones'].values():
            output = Path(row['private_output'])
            self.assertTrue((output/'lib').is_symlink())
            self.assertTrue((output/'SubProcesses/P0_fake/warmup.dat').is_file())
            self.assertTrue((output/'SubProcesses/MadLoop5_resources/P0_HelFilter.dat').is_file())
            self.assertFalse(row['runtime_verified_for_original'])
            self.assertEqual(prep.verify_clone(self.bench, output, row)['status'], 'verified')
        row = original[prep.CASES[0]]
        self.assertEqual(len(row['runtime_files']['libraries']), 3)
        self.assertEqual(row['runtime']['archives'], self.fixture.dependencies(Path(row['original_output'])/'SubProcesses/P0_fake')['archives'])

    def test_private_card_changes_do_not_modify_original(self):
        result = self.prepare(); row = result['clones'][prep.CASES[0]+'/1']; output = Path(row['private_output'])
        (output/'Cards/MadLoopParams.dat').write_text('backend1\n')
        prep.verify_clone(self.bench, output, row)
        self.assertEqual(prep.collect_originals(self.bench), result['originals'])

    def test_runtime_change_rejected_without_loader(self):
        (self.root/'native/one/ld-linux.so.2').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'runtime file changed'): prep.collect_originals(self.bench)

    def test_expected_absence_rejects_new_source(self):
        write(self.root/'absent-generator.py', 'new source')
        with self.assertRaisesRegex(ValueError, 'Registered current source changed'): prep.shared_bindings(self.bench)

    def test_private_object_change_rejected(self):
        row = self.prepare()['clones'][prep.CASES[0]+'/9']; output = Path(row['private_output'])
        (output/'Source/model.o').write_text('changed object')
        with self.assertRaisesRegex(ValueError, 'objects changed'): prep.verify_clone(self.bench, output, row)

    def test_external_mutable_link_retains_failure(self):
        original = self.bench.output_dir('candidate', self.fixture.case)
        link = original/'SubProcesses/P0_fake/MadLoopParams.dat'; link.unlink(); link.symlink_to(original/'Cards/MadLoopParams.dat')
        with self.assertRaisesRegex(ValueError, 'Mutable card/resource escapes'): self.prepare()
        self.assertTrue((self.root/'records/preparation-failure.json').exists())
        self.assertFalse((self.root/'records/preparation.json').exists())

    def test_active_original_refused(self):
        output = self.bench.output_dir('candidate', self.fixture.case)
        os.mkfifo(output/'SubProcesses/P0_fake/worker-protocol.fifo')
        with self.assertRaisesRegex(ValueError, 'worker is active'): self.prepare()
        self.assertFalse((self.root/'private').exists())

    def test_append_once(self):
        self.prepare(); before = (self.root/'records/preparation.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'already exists'): self.prepare()
        self.assertEqual(before, (self.root/'records/preparation.json').read_bytes())

    def test_registered_namespace_forbidden(self):
        self.bench.CAMPAIGN_ID = 'not-private'
        with self.assertRaisesRegex(ValueError, 'outside registered campaigns'): prep.collect_originals(self.bench)

    def test_closed_preparation_changes_refused(self):
        write(self.fixture.origin/'status.json', {'status': 'running'})
        with self.assertRaisesRegex(ValueError, 'Closed preparation input changed'): prep.collect_originals(self.bench)

    def test_borrowed_runtime_postcheck_is_required(self):
        path = self.root/prep.BORROWED/'inputs-after.json'; after = prep.read(path)
        after['executable_sha256'] = 'mismatch'; write(path, after)
        self.fixture.pins[path.relative_to(self.root).as_posix()] = digest(path)
        with self.assertRaisesRegex(ValueError, 'Borrowed runtime postcheck differs'): prep.collect_originals(self.bench)

    def test_production_configuration_set_and_bytes_are_frozen(self):
        write(self.root/'input/mg7_configuration.txt', 'changed ignored configuration')
        with self.assertRaisesRegex(ValueError, 'Production configuration set/bytes changed'): prep.shared_bindings(self.bench)

    def test_canonical_provider_prefix_and_bytes_are_frozen(self):
        self.bench.TIRUST_PREFIX = self.root/'wrong-prefix'
        with self.assertRaisesRegex(ValueError, 'Canonical TIRust prefix/provider differs'): prep.shared_bindings(self.bench)
        self.bench.TIRUST_PREFIX = self.root/'native'
        write(self.root/'native/lib/libmadloop_tirust.so', 'changed provider')
        with self.assertRaisesRegex(ValueError, 'Canonical TIRust file changed'): prep.shared_bindings(self.bench)

    def test_point_seed_and_count_must_remain_fixed(self):
        self.fixture.config['seed'] = 7
        with self.assertRaisesRegex(ValueError, 'case/policy differs'): prep.collect_originals(self.bench)


if __name__ == '__main__': unittest.main()
