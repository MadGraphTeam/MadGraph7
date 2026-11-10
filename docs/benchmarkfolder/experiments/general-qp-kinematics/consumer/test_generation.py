"""Private source-generation checks; all subprocess/compiler launch forbidden."""
import copy
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path.cwd()
HERE = ROOT / '.tirust-build/general-kinematics-consumer'
OLD = ROOT / '.tirust-build/general-kinematics-classification'
MASS = ROOT / '.tirust-build/qp-mass-general-implementation'
TEMPLATE = Path('madgraph/iolibs/template_files/loop_optimized')
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


graph = load('consumer_graph_source', OLD / 'test_generation.py')
from madgraph.iolibs import file_writers
from madgraph.models import import_ufo
import madgraph.loop
KC = load('madgraph.loop.kinematic_consumer', HERE / 'src/madgraph/loop/kinematic_consumer.py')
madgraph.loop.kinematic_consumer = KC
NEW = load('consumer_exporter', HERE / 'src/madgraph/loop/loop_exporters.py')
BASE = load('consumer_base_exporter', HERE / 'base/loop_exporters.py')


def exporter(module):
    obj = object.__new__(module.LoopProcessOptimizedExporterFortranSA)
    obj.template_dir = str(HERE / 'src' / TEMPLATE)
    obj.opt = {'vector_size': 0}
    obj.group_loops = True
    return obj


def mapping():
    all_text = ''.join(p.read_text() for p in (HERE / 'src' / TEMPLATE).iterdir() if p.is_file())
    result = {key: 2 if fmt == 'd' else '' for key, fmt in re.findall(r'%\((\w+)\)([ds])', all_text)}
    result.update(proc_prefix='KC_', real_dp_format='REAL*8', real_mp_format='REAL*16',
        complex_dp_format='COMPLEX*16', complex_mp_format='COMPLEX*32',
        mass_dp_format='COMPLEX*16', mass_mp_format='COMPLEX*32', maxrank=6,
        nexternal=5, nloop_groups=116, nSquaredSO=1, loopmaxcoefs=210)
    return result


def routines(source):
    starts = list(re.finditer(r'(?mi)^\s*SUBROUTINE\s+(\w+)\(', source))
    return {m.group(1).upper(): source[m.start():starts[k + 1].start() if k + 1 < len(starts) else len(source)]
            for k, m in enumerate(starts)}


class SourceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qcd = graph.generate([2, -2, 23, 21, 21])
        cls.qcd.rep_dict = mapping()
        cls.base = exporter(BASE)
        cls.new = exporter(NEW)
        cls.notes = {}

    def test_01_original_adapter_entries_remain_exact_prefix(self):
        for name in ('NINJARS_qp_interface.inc', 'TIRUST_qp_loop_interface.inc',
                     'TIRUST_qp_tensor_interface.inc'):
            old = self.base.qp_mass_interfaces(name, mapping())
            new = self.new.qp_mass_interfaces(name, mapping())
            self.assertTrue(new.startswith(old), name)
            extra = new[len(old):]
            self.assertIn('_QPMASS_KIN(', extra)
            self.assertNotRegex(extra, r'COMMON/\w+CT/')
            self.assertIn('GROUP_ID', extra)
            self.assertIn('KIN_POLICY', extra)
            if name != 'TIRUST_qp_loop_interface.inc':
                self.assertIn('MP_SWITCH_ORDER(CTMODE,', extra)
                self.assertIn('MP_BUILD_KINEMATIC_MATRIX_KIN(GROUP_ID,CTMODE,KIN_POLICY,', extra)
            if name == 'TIRUST_qp_tensor_interface.inc':
                self.assertIn('CHECK_PASS=CTMODE.EQ.2.OR.CTMODE.EQ.5', extra)
            if name == 'TIRUST_qp_loop_interface.inc':
                self.assertIn('DIAGNOSTIC_QPMASS_KIN(GROUP_ID,KIN_POLICY,CTMODE,', extra)

    def test_02_generated_qcd_helpers_only_select_kin_in_paired_qp(self):
        old = BASE.LoopProcessExporterFortranSA.write_CT_interface(self.base, None, self.qcd, optimized_output=True)
        new = NEW.LoopProcessExporterFortranSA.write_CT_interface(self.new, None, self.qcd, optimized_output=True)
        original, derived = routines(old), routines(new)
        count = 0
        for name in original:
            if re.fullmatch(r'KC_LOOP_[0-9_]+_QPMASS', name):
                self.assertEqual(derived[name].strip(), KC.connect_paired_helper(original[name]).strip())
                count += 1
            elif re.fullmatch(r'KC_LOOP_[0-9_]+', name):
                self.assertEqual(derived[name].strip(), original[name].strip())
                self.assertNotIn('_KIN(', derived[name])
        self.assertGreater(count, 0)
        self.assertEqual(new.count('SUBROUTINE KC_KINEMATIC_IDENTITY('), 1)
        self.assertEqual(new.count('SUBROUTINE KC_MP_BUILD_KINEMATIC_MATRIX_KIN('), 1)
        self.notes['qcd_helper_families'] = count
        self.notes['rendered_qcd_source_sha256'] = hashlib.sha256(new.encode()).hexdigest()
        self.raw = new
        (HERE / 'generated').mkdir(exist_ok=True)
        (HERE / 'generated/CT_interface.raw').write_text(new)

    def test_03_real_physical_external_mass_and_zero(self):
        rows = KC.external_mass_rows(self.qcd)
        self.assertEqual(rows[2]['mass_symbol'].upper(), 'MDL_MZ')
        self.assertEqual(rows[2]['mp_symbol'].upper(), 'MP__MDL_MZ')
        self.assertTrue(all(r['supported_real'] for r in rows))
        self.assertTrue(all(rows[i]['zero'] for i in (0, 1, 3, 4)))
        source = KC.render(self.qcd, True, 'KC_')
        self.assertIn('MASS=MP__MDL_MZ', source.upper())
        self.assertNotIn('MASS=MP__CMASS', source)
        self.assertIn("INCLUDE 'mp_coupl.inc'", source)
        self.notes['external_mass_rows'] = rows

    def test_04_cms_uses_physical_mass_symbol(self):
        import_ufo.ReadWrite = False
        model = import_ufo.import_model(str(OLD / 'models/loop_sm'), complex_mass_scheme=True)
        process = copy.copy(self.qcd.get('processes')[0])
        process['model'] = model
        matrix = mock.Mock()
        matrix.get.return_value = [process]
        rows = KC.external_mass_rows(matrix)
        self.assertEqual(rows[2]['mp_symbol'].upper(), 'MP__MDL_MZ')
        self.assertTrue(rows[2]['supported_real'])
        self.notes['cms_source_mass_rows'] = rows

    def test_05_unsupported_mass_types_are_not_projected(self):
        model = copy.deepcopy(self.qcd.get('processes')[0].get('model'))
        for group in model.get('parameters').values():
            for parameter in group:
                if parameter.name.upper() == 'MDL_MZ':
                    parameter.type = 'complex'
        process = copy.copy(self.qcd.get('processes')[0])
        process['model'] = model
        matrix = mock.Mock()
        matrix.get.return_value = [process]
        self.assertFalse(KC.external_mass_rows(matrix)[2]['supported_real'])

    def test_06_topology_matrix_policy_and_legacy_dispatch(self):
        source = KC.render(self.qcd, True, 'KC_')
        start = source.index('SUBROUTINE KC_MP_BUILD_KINEMATIC_MATRIX_KIN(')
        body = source[start:]
        self.assertLess(body.index('IF(POLICY.EQ.0)'), body.index('TOL=REAL(OSThres'))
        self.assertIn('CALL KC_MP_BUILD_KINEMATIC_MATRIX(N,P_LOOP,M2L,S_MAT)', body)
        for guard in ('IF(.NOT.KNOWN.OR..NOT.DECLARED_ONSHELL) CYCLE',
                      'IF(CLASS.NE.1.AND.CLASS.NE.2) CYCLE',
                      'IF(CLOSURE_SIZE/PAIR_SCALE.GT.TOL) CYCLE',
                      'IF(.NOT.TARGET_VALID) CYCLE', 'IF(RESIDUAL.LT.TOL) THEN'):
            self.assertIn(guard, body)
            self.assertLess(body.index(guard), body.index('S_MAT(I,J)=CMPLX(TARGET'))
        self.assertNotIn('DIFFSQ-M2L(I))/M2L(I)', body)
        self.assertNotIn('DIFFSQ-M2L(J))/M2L(J)', body)
        self.assertNotRegex(body, r'IEEE_SET|COMMON|MP_DONE|SET_MP_PS')
        self.assertLess(body.index('P_FINITE=ALL(IEEE_IS_FINITE(P_LOOP))'),
                        body.index('CLOSURE_SIZE=MAXVAL'))
        self.assertLess(body.index('IF(CLOSURE_SIZE.GT.PAIR_SCALE) CYCLE'),
                        body.index('IF(CLOSURE_SIZE/PAIR_SCALE.GT.TOL) CYCLE'))
        self.assertLess(body.index('IF(PAIR_SCALE.GE.SQRT(HUGE(PAIR_SCALE))) CYCLE'),
                        body.index('REF_NORMALIZATION=PAIR_SCALE**2'))
        self.assertLess(body.index('IF(PAIR_SCALE.LT.SQRT(TINY(PAIR_SCALE))) CYCLE'),
                        body.index('REF_NORMALIZATION=PAIR_SCALE**2'))
        self.assertLess(body.index('IF(.NOT.IEEE_IS_FINITE(REF_NORMALIZATION)) CYCLE'),
                        body.index('RESIDUAL=ABS(DIFFSQ/REF_NORMALIZATION)'))
        self.assertNotIn('ABS(P_LOOP(I,K))+ABS(P_LOOP(J,K))', body)

    def test_07_source_render_no_compiler_all_three_adapters(self):
        raw = NEW.LoopProcessExporterFortranSA.write_CT_interface(self.new, None, self.qcd, optimized_output=True)
        for name in ('TIRUST_qp_loop_interface.inc', 'TIRUST_qp_tensor_interface.inc'):
            raw += self.new.qp_mass_interfaces(name, mapping())
        context = dict(AmplitudeReduction=False, samurai_available=False,
            ninja_available=True, ninja_supports_quad_prec=True,
            ninjars_available=True, ninjars_supports_quad_prec=True,
            tirust_available=True, tirust_supports_quad_prec=True)
        path = HERE / 'generated/consumer.f'
        with file_writers.FortranWriter(str(path)) as writer:
            writer.writelines(raw, context=context)
        rendered = path.read_text()
        self.assertNotIn('%(', rendered)
        for line in rendered.splitlines():
            self.assertLessEqual(len(line), 132)
            if line.strip() and line[0] not in 'Cc*!#':
                self.assertTrue(all(c in ' 0123456789' for c in line[:5]), line)
        self.notes['fixed_form_sha256'] = sha(path)

    def test_08_vector_scope_refuses(self):
        private = exporter(NEW)
        private.opt['vector_size'] = 1
        with self.assertRaisesRegex(ValueError, 'requires scalar export'):
            NEW.LoopProcessExporterFortranSA.write_CT_interface(private, None, self.qcd, optimized_output=True)

    def test_09_retained_small_composites_never_have_shell_identity(self):
        table = graph.KI.build(self.qcd)
        by_id = {r['reduction_id']: r for r in table['reductions']}
        oracle = json.loads((ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer/routing-modes.json').read_text())
        counts = {}
        for ct, mode in [('4', 'primary'), ('5', 'reverse')]:
            count = 0
            for row in oracle['groups_by_mode'][ct]:
                for pair in row['pairs']:
                    if not pair['target_p1_minus_p4']:
                        continue
                    i, j = pair['propagators']
                    actual = next(p for p in by_id[row['group']]['directions'][mode]['pairs']
                                  if (p['i'], p['j']) == (i, j))
                    self.assertEqual(actual['class'], 3)
                    self.assertFalse(actual['declared_onshell'])
                    count += 1
            self.assertEqual(count, 19)
            counts[ct] = count
        self.notes['excluded_small_composites_by_mode'] = counts

    def test_10_ungrouped_and_induced_export_metadata(self):
        ungrouped = KC.render(self.qcd, False, 'UG_')
        ids = graph.emitted_ids(self.qcd, False)
        self.assertEqual(len(ids), 224)
        self.assertIn('INTEGER GOFF(%d)' % max(ids), ungrouped)
        induced = graph.generate([21, 21, 25, 25], induced=True)
        induced.rep_dict = mapping()
        table = graph.KI.build(induced)
        ids = graph.emitted_ids(induced, True)
        self.assertEqual([r['reduction_id'] for r in table['reductions']], ids)
        raw = NEW.LoopProcessExporterFortranSA.write_CT_interface(self.new, None, induced, optimized_output=True)
        self.assertIn('MP_NINJARS_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1,', raw)
        self.assertIn('MP_TIRUST_LOOP_QPMASS_KIN(LOOPNUM,CTMODE,1,', raw)
        self.assertIn('INTEGER GOFF(%d)' % max(ids), raw)
        self.notes['ungrouped_reductions'] = len(graph.emitted_ids(self.qcd, False))
        self.notes['induced_reductions'] = len(ids)

    def test_11_discriminating_exact_soft_leg_policy_examples(self):
        # Exact rational design checks, not a native floating-point evaluation.
        tolerance = Fraction(1, 10**8)
        hard = 10**8
        pair = [1, 0, 0, 0]  # Declared massless leg with timelike defect.
        defect = pair[0]**2 - sum(x*x for x in pair[1:])
        historical = Fraction(defect, 1) / (Fraction(2*hard + 1, 2)**2)
        local = Fraction(defect, max(abs(x) for x in pair)**2)
        self.assertLess(historical, tolerance)
        self.assertGreater(local, tolerance)
        closure, soft = Fraction(1, 10**4), Fraction(1, 10**5)
        self.assertLess(closure / hard, tolerance)
        self.assertGreater(closure / soft, tolerance)
        true_massless = [1, 1, 0, 0]
        self.assertEqual(true_massless[0]**2 - sum(x*x for x in true_massless[1:]), 0)
        self.notes['exact_policy_examples'] = {
            'scope': 'Rational policy examples only; no native floating-point qualification.',
            'timelike_soft_leg_historical_residual': str(historical),
            'timelike_soft_leg_local_residual': str(local),
            'global_closure_ratio': str(closure / hard),
            'local_closure_ratio': str(closure / soft), 'tolerance': str(tolerance)}


def main():
    preparation = json.loads((HERE / 'preparation.json').read_text())
    before = dict(preparation['inputs'])
    for name, entry in json.loads((OLD / 'source-index.json').read_text())['files'].items():
        before[str((OLD / name).relative_to(ROOT))] = entry['sha256']
    for path in (ROOT / 'models/loop_sm').rglob('*'):
        if path.is_file():
            before[str(path.relative_to(ROOT))] = sha(path)
    assert all(sha(ROOT / name) == digest for name, digest in before.items())
    log = io.StringIO()
    with mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('No subprocess allowed')):
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(SourceChecks))
    print(log.getvalue(), end='')
    unchanged = all(sha(ROOT / name) == digest for name, digest in before.items())
    record = {'source_only': True, 'compiler_executed': False, 'native_executed': False,
        'tests_run': result.testsRun, 'successful': result.wasSuccessful() and unchanged,
        'protected_inputs_unchanged': unchanged, 'protected_inputs': before,
        'errors': [(str(test), error) for test, error in result.errors],
        'failures': [(str(test), error) for test, error in result.failures],
        'details': getattr(SourceChecks, 'notes', {}),
        'test_sha256': sha(Path(__file__)),
        'consumer_sha256': sha(HERE / 'src/madgraph/loop/kinematic_consumer.py')}
    (HERE / 'source-tests.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    (HERE / 'source-tests.log').write_text(log.getvalue())
    assert record['successful']


if __name__ == '__main__':
    main()
