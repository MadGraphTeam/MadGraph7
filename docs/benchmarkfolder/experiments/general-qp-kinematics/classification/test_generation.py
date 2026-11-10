"""Private Python source-generation checks. No compiler or amplitude execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import unittest

ROOT = Path.cwd()
HERE = ROOT / '.tirust-build/general-kinematics-classification'
sys.path[:0] = [str(ROOT), str(HERE / 'models')]
from madgraph.core import base_objects, helas_objects
from madgraph.loop import loop_diagram_generation, loop_helas_objects
from madgraph.iolibs import helas_call_writers
from madgraph.models import import_ufo
import madgraph.loop


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


KI = load('madgraph.loop.kinematic_identity', HERE / 'src/madgraph/loop/kinematic_identity.py')
madgraph.loop.kinematic_identity = KI
NEW_E = load('private_identity_exporter', HERE / 'src/madgraph/loop/loop_exporters.py')
OLD_E = load('original_identity_exporter', HERE / 'base/madgraph/loop/loop_exporters.py')


def generate(pdgs, induced=False):
    # Fresh conversion/generation avoids reliance on an opaque graph pickle.
    import_ufo.ReadWrite = False
    model_path = HERE / 'models/loop_sm'
    if not model_path.exists():
        shutil.copytree(ROOT / 'models/loop_sm', model_path,
                       ignore=shutil.ignore_patterns('*.pkl', '__pycache__', '*.pyc', '*.log'))
    for path in (ROOT / 'models/loop_sm').rglob('*'):
        if path.is_file() and path.suffix in ('.py', '.dat'):
            assert path.read_bytes() == (model_path / path.relative_to(ROOT / 'models/loop_sm')).read_bytes()
    model = import_ufo.import_model(str(model_path), complex_mass_scheme=False)
    legs = base_objects.LegList([base_objects.Leg({'id': pdg, 'number': i, 'state': i > 2})
                                for i, pdg in enumerate(pdgs, 1)])
    process = base_objects.Process({'legs': legs, 'model': model,
              'perturbation_couplings': ['QCD'], 'NLO_mode': 'noborn' if induced else 'virt',
              'has_born': not induced})
    amplitude = loop_diagram_generation.LoopAmplitude(process)
    result = loop_helas_objects.LoopHelasMatrixElement(amplitude, optimized_output=True)
    result.compute_all_analytic_information()
    return result


def emitted_ids(matrix, grouped):
    writer = helas_call_writers.FortranUFOHelasCallWriterOptimized(
        matrix.get('processes')[0].get('model'))
    calls = writer.get_loop_CT_calls(matrix, group_loops=grouped)
    return [int(re.search(r',I_SO,(\d+)\)$', call).group(1))
            for call in calls if re.match(r'CALL .*LOOP_', call)]


class GenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qcd = generate([2, -2, 23, 21, 21])
        cls.model = cls.qcd.get('processes')[0].get('model')
        cls.table = KI.build(cls.qcd)
        cls.frozen = json.loads((ROOT / 'docs/benchmarkfolder/experiments/'
            'qcd-loop-kinematics-observer/routing-modes.json').read_text())
        cls.notes = {}

    def check_rows(self, table):
        self.assertTrue(all(row['known'] for row in table['reductions']))
        size = len(table['external'])
        for row in table['reductions']:
            for direction in row['directions'].values():
                for pair in direction['pairs']:
                    if pair['class'] in (1, 2):
                        self.assertEqual(pair['vector'], [pair['closure'] +
                            (pair['sign'] if i + 1 == pair['leg'] else 0) for i in range(size)])

    def test_actual_qcd_all_pairs_both_directions_match_frozen_source_proof(self):
        self.check_rows(self.table)
        self.assertEqual(len(self.table['reductions']), 116)
        for ct, name in [('4', 'primary'), ('5', 'reverse')]:
            frozen = self.frozen['groups_by_mode'][ct]
            self.assertEqual(len(frozen), 116)
            for row, expected in zip(self.table['reductions'], frozen):
                self.assertEqual(row['reduction_id'], expected['group'])
                direction = row['directions'][name]
                self.assertEqual(direction['corners'], expected['corner_coefficients_k'])
                self.assertEqual(direction['shifts'], expected['shift_coefficients_k'])
                self.assertEqual(direction['mass_order'], expected['mass_source_denominators'])
                for pair, old_pair in zip(direction['pairs'], expected['pairs']):
                    self.assertEqual([pair['i'], pair['j']], old_pair['propagators'])
                    self.assertEqual(pair['vector'], old_pair['literal_coefficients_k'])
                    self.assertEqual(pair['leg'] or None, old_pair['single_leg'])
                    self.assertEqual(pair['class'], {'direct_single_leg': 1,
                        'complement_of_single_leg': 2, 'composite': 3}[old_pair['classification']])
        self.notes['qcd'] = {'groups': 116, 'pairs_per_direction': sum(
            len(row['directions']['primary']['pairs']) for row in self.table['reductions'])}

    def test_actual_tiny_composites_remain_composites_in_both_modes(self):
        for ct, name in [('4', 'primary'), ('5', 'reverse')]:
            targets = self.frozen['target_pairs'][ct]
            self.assertEqual(len(targets), 19)
            for target in targets:
                row = next(r for r in self.table['reductions'] if r['reduction_id'] == target['group'])
                pair_ids = target['pair']['propagators']
                pair = next(p for p in row['directions'][name]['pairs']
                            if [p['i'], p['j']] == pair_ids)
                self.assertEqual(pair['class'], 3)
                self.assertFalse(pair['declared_onshell'])
        self.notes['tiny_composite_pairs_per_direction'] = 19

    def test_actual_grouped_and_ungrouped_ids_match_call_writer(self):
        self.assertEqual([row['reduction_id'] for row in self.table['reductions']], emitted_ids(self.qcd, True))
        separate = KI.build(self.qcd, False)
        self.check_rows(separate)
        self.assertEqual([row['reduction_id'] for row in separate['reductions']], emitted_ids(self.qcd, False))
        self.assertGreater(len(separate['reductions']), len(self.table['reductions']))
        self.notes['ungrouped_reductions'] = len(separate['reductions'])

    def test_actual_crossed_qcd_graph_and_momentum_signs(self):
        matrix = generate([2, 21, 23, 2, 21])
        table = KI.build(matrix)
        self.check_rows(table)
        self.assertEqual([r['reduction_id'] for r in table['reductions']], emitted_ids(matrix, True))
        self.assertEqual([leg['outgoing'] for leg in table['external']], [False, False, True, True, True])
        # The crossed final up quark must now have W momentum +P, not the incoming -P.
        incidence = KI.Incidence(matrix.get('processes')[0])
        external_up = next(w for w in matrix.get_all_wavefunctions()
                           if not w.get('mothers') and not w.get('is_loop') and w.get('number_external') == 4)
        self.assertEqual(incidence.wavefunction(external_up), (0, 0, 0, 1, 0))
        self.notes['crossed_groups'] = len(table['reductions'])

    def test_actual_loop_induced_uses_ungrouped_reduction_ids(self):
        matrix = generate([21, 21, 25, 25], True)
        table = KI.build(matrix, True)
        self.check_rows(table)
        self.assertFalse(table['has_born'])
        self.assertFalse(table['grouped'])
        self.assertEqual([r['reduction_id'] for r in table['reductions']], emitted_ids(matrix, True))
        self.notes['loop_induced_reductions'] = len(table['reductions'])

    def test_one_and_two_denominator_boundaries(self):
        # One-denominator boundary uses genuine HELAS objects but is a constructed
        # graph, not a claimed generated physical SM tadpole process.
        process = self.qcd.get('processes')[0]
        reference = self.qcd.get('loop_groups')[0][1][0]
        cut = copy.deepcopy(reference.get_starting_loop_wavefunction())
        end = copy.deepcopy(cut)
        final = copy.deepcopy(cut)
        externals = [helas_objects.HelasWavefunction(leg, 0, self.model)
                     for leg in process.get('legs')]
        final.set('mothers', helas_objects.HelasWavefunctionList([cut] + externals))
        closing = helas_objects.HelasAmplitude({'mothers': helas_objects.HelasWavefunctionList([final, end])})
        lamp = loop_helas_objects.LoopHelasAmplitude({
            'wavefunctions': helas_objects.HelasWavefunctionList([cut, final, end]),
            'amplitudes': helas_objects.HelasAmplitudeList([closing]), 'number': 1, 'loop_group_id': 0})
        lamp.set_mothers_and_pairing()
        result = KI.Incidence(process).amplitude(lamp)
        self.assertEqual(result['n'], 1)
        self.assertEqual(result['corners'], [(1, 1, 1, 1, 1)])
        self.assertEqual(KI.direction(result['corners']), KI.direction(result['corners'], True))
        self.assertEqual(KI.direction(result['corners'])['pairs'], [])
        one = loop_helas_objects.LoopHelasMatrixElement()
        one['processes'] = self.qcd.get('processes')
        one['loop_groups'] = [(lamp.get_denominators(), helas_objects.HelasAmplitudeList([lamp]))]
        single_table = KI.build(one)
        self.assertTrue(single_table['reductions'][0]['known'])
        single_source = KI.render(single_table, 'BOUNDARY_')
        self.assertIn('INTEGER CLASSES(1),LEGS(1),SIGNS(1)', single_source)
        self.assertIn('IF(I.EQ.J) RETURN', single_source)
        bubbles = [r for r in self.table['reductions'] if r['n'] == 2]
        self.assertTrue(bubbles)
        for bubble in bubbles:
            self.assertEqual(bubble['directions']['primary'], bubble['directions']['reverse'])
        self.notes['n1_scope'] = 'constructed HELAS boundary; no physical amplitude claim'
        self.notes['actual_n2_groups'] = len(bubbles)

    def test_group_member_disagreement_is_unknown_not_single_shell(self):
        denominator, members = next(g for g in self.qcd.get('loop_groups') if len(g[1]) > 1)
        broken = copy.deepcopy(members[1])
        broken['mothers'] = helas_objects.HelasWavefunctionList(list(broken.get('mothers'))[::-1])
        isolated = loop_helas_objects.LoopHelasMatrixElement()
        isolated['processes'] = self.qcd.get('processes')
        isolated['loop_groups'] = [(denominator, helas_objects.HelasAmplitudeList([members[0], broken]))]
        row = KI.build(isolated)['reductions'][0]
        self.assertFalse(row['known'])
        self.assertIn('reason', row)
        self.assertNotIn('directions', row)
        self.assertIn('GVALID', KI.render(KI.build(isolated), 'TEST_'))

    def test_offshell_descriptor_disables_declared_shell_only(self):
        matrix = copy.deepcopy(self.qcd)
        matrix.get('processes')[0].get('legs')[3]['offshell'] = True
        table = KI.build(matrix)
        self.check_rows(table)
        selected = [pair for row in table['reductions'] for mode in row['directions'].values()
                    for pair in mode['pairs'] if pair['leg'] == 4]
        self.assertTrue(selected)
        self.assertTrue(all(not pair['declared_onshell'] for pair in selected))
        self.assertTrue(all(pair['class'] in (1, 2) for pair in selected))

    def test_generated_integer_data_and_exporter_append_only_hook(self):
        source = KI.render(self.table, 'TEST_')
        self.assertNotRegex(source, r'(?m)^      (REAL|COMPLEX|COMMON)\b')
        self.assertTrue(all(len(line) <= 72 for line in source.splitlines() if not line.startswith('C')))
        arrays = {}
        for name, first, last, values in re.findall(
                r'DATA \((\w+)\(L\),L=(\d+),(\d+)\) /\n     \$ ([\d,\-]+)/', source):
            chunk = list(map(int, values.split(',')))
            self.assertEqual(len(chunk), int(last) - int(first) + 1)
            arrays.setdefault(name, []).extend(chunk)
        index = 0
        for row in self.table['reductions']:
            self.assertEqual(arrays['GOFF'][row['reduction_id'] - 1], index)
            for name in ('primary', 'reverse'):
                for pair in row['directions'][name]['pairs']:
                    cls, leg = arrays['CLASSES'][index], arrays['LEGS'][index]
                    if cls in (1, 2):
                        self.assertEqual(pair['vector'], [arrays['CLOSURES'][index] +
                            (arrays['SIGNS'][index] if i + 1 == leg else 0) for i in range(5)])
                    else:
                        self.assertEqual(leg, 0)
                    index += 1
        self.assertEqual(index, 954)
        template = ROOT / 'madgraph/iolibs/template_files/loop_optimized/TIR_interface.inc'
        replacements = {key: (6 if kind == 'd' else 'TEST_') for key, kind in
                        re.findall(r'%\((\w+)\)([sd])', template.read_text())}
        replacements.update(maxrank=6, complex_dp_format='COMPLEX*16', proc_prefix='TEST_')
        self.qcd.rep_dict = replacements
        outputs = []
        for module in (OLD_E, NEW_E):
            exporter = object.__new__(module.LoopProcessOptimizedExporterFortranSA)
            exporter.opt = {'vector_size': False}
            exporter.template_dir = str(template.parent)
            exporter.tir_available_dict = {'pjfry': False, 'iregi': False}
            exporter.group_loops = True
            outputs.append(exporter.write_TIR_interface(None, self.qcd))
        self.assertEqual(outputs[1], outputs[0] + '\n' + source)
        (HERE / 'generated').mkdir(exist_ok=True)
        (HERE / 'generated/qcd_n3_identity.f').write_text(source)
        (HERE / 'generated/qcd_n3_identity.json').write_text(json.dumps(self.table, sort_keys=True, indent=2) + '\n')


if __name__ == '__main__':
    protected = [ROOT / p for p in ('madgraph/loop/loop_exporters.py',
        'madgraph/loop/loop_helas_objects.py', 'madgraph/core/helas_objects.py',
        'madgraph/iolibs/helas_call_writers.py', 'aloha/aloha_writers.py')]
    protected += [p for p in (ROOT / 'models/loop_sm').rglob('*') if p.is_file()]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GenerationTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    unchanged = before == after
    report = {'source_only': True, 'compiler_executed': False, 'amplitude_executed': False,
              'tests_run': result.testsRun, 'successful': result.wasSuccessful() and unchanged,
              'protected_inputs_unchanged': unchanged, 'protected_inputs': before,
              'failures': [str(test) for test, _ in result.failures],
              'errors': [str(test) for test, _ in result.errors], 'details': GenerationTest.notes}
    report['private_inputs'] = {str(path.relative_to(HERE)): digest(path) for path in
        [Path(__file__), HERE / 'prepare.py',
         HERE / 'src/madgraph/loop/kinematic_identity.py', HERE / 'src/madgraph/loop/loop_exporters.py']}
    report['frozen_routing_sha256'] = digest(ROOT / 'docs/benchmarkfolder/experiments/'
        'qcd-loop-kinematics-observer/routing-modes.json')
    (HERE / 'source-tests.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    raise SystemExit(0 if report['successful'] else 1)
