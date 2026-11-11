"""Pure source and receipt tests. No compiler, library or amplitude executes."""
from pathlib import Path
import ast
import copy
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT1 = ROOT / '.tirust-build/generic-invariant-provenance-proposal-1'
PARENT2 = ROOT / '.tirust-build/generic-invariant-provenance-proposal-2'
sys.path.insert(0, str(ROOT))
import madgraph.loop
for path in (PARENT1, PARENT2, HERE):
    madgraph.loop.__path__.insert(0, str(path / 'src/madgraph/loop'))
from madgraph.loop.invariant_environment_source import fixed
from madgraph.loop.invariant_window import momentum_window
from madgraph.loop.invariant_geometry_window import (derive_geometry_contract,
    geometry_support, source_bindings)
from madgraph.loop import invariant_render as render


def f(s):
    return '\n'.join('      ' + line for line in s.strip().splitlines()) + '\n'


def fixture(before='', after='', extra=''):
    text = f('SUBROUTINE B(P,W)\nUSE OBJ\nREAL P(0:3),X\nTYPE(NODE) W(1)\n' +
             before + '\nCALL MAKE(P,W(1))\n' + after + '\nEND')
    defs = f('MODULE OBJ\nTYPE NODE\nREAL P(0:3),W(4)\nEND TYPE\nEND MODULE\n'
             'SUBROUTINE MAKE(P,O)\nUSE OBJ\nREAL P(0:3)\nTYPE(NODE) O\n'
             'O%P(:)=P(:)\nEND\n' + extra)
    raw = {'Source/DHELAS/a.f': defs, 'SubProcesses/P0/b.f': text}
    receipts = [{'caller': 'b.f', 'line': line, 'call': statement}
                for _, line, statement in fixed(text, 'x')
                if statement.startswith('CALLMAKE')]
    batches = [('b.f', text)]
    window = momentum_window(raw, batches, receipts)
    result = {'graph': {'nexternal': 3}, 'routes': [], 'source_proof': {
                  width: {'producers': copy.deepcopy(receipts)} for width in ('64', '128')},
              'source_bindings': source_bindings(raw),
              'momentum_environment_contract': {width: copy.deepcopy(window)
                                                for width in ('64', '128')}}
    return result, raw, {'64': batches, '128': batches}


def joined(**kwargs):
    d, raw, batches = fixture(**kwargs)
    d['geometry_window_source_contract'] = derive_geometry_contract(d, raw, batches)
    return d, raw, batches


class GeometryWindow(unittest.TestCase):
    def test_plain_source_window(self):
        d, _, _ = joined()
        self.assertEqual(geometry_support(d), (True, True))
        self.assertFalse(d['geometry_window_source_contract']['runtime_abi_qualified'])
        self.assertFalse(d['geometry_window_source_contract']['scientific_dispatch_enabled'])

    def test_post_momentum_scalar_io_does_not_claim_whole_purity(self):
        d, _, _ = joined(after="WRITE(*,*) 'diagnostic'")
        self.assertEqual(geometry_support(d), (True, True))
        self.assertFalse(d['momentum_environment_contract']['64']['suffix']['complete'])

    def test_prefix_io_remains_rejected(self):
        d, _, _ = joined(before="WRITE(*,*) 'diagnostic'")
        self.assertEqual(geometry_support(d), (False, False))

    def test_prefix_function_setter_remains_rejected(self):
        d, _, _ = joined(before='X=F()', extra='REAL FUNCTION F()\nCALL FESETROUND(0)\nF=0\nEND')
        self.assertEqual(geometry_support(d), (False, False))

    def test_unknown_suffix_function_remains_rejected(self):
        d, _, _ = joined(after='X=FOREIGN()')
        self.assertEqual(geometry_support(d), (False, False))

    def test_numeric_alias_overwrite_remains_rejected(self):
        d, _, _ = joined(after='CALL MUTATE(W(1)%P)',
                         extra='SUBROUTINE MUTATE(P)\nREAL P(0:3)\nP(0)=1.\nEND')
        self.assertEqual(geometry_support(d), (False, False))

    def test_whole_object_overwrite_remains_rejected(self):
        d, _, _ = joined(after='W(1)=W(1)')
        self.assertEqual(geometry_support(d), (False, False))

    def test_missing_contract_false(self):
        d, _, _ = fixture()
        self.assertEqual(geometry_support(d), (False, False))

    def test_missing_precision_rejected(self):
        d, raw, batches = fixture()
        del batches['128']
        with self.assertRaises(ValueError): derive_geometry_contract(d, raw, batches)

    def test_changed_source_receipt_rejected(self):
        d, raw, batches = fixture()
        raw['Source/DHELAS/a.f'] += f('!changed')
        with self.assertRaisesRegex(ValueError, 'source bindings'):
            derive_geometry_contract(d, raw, batches)

    def test_changed_batch_bytes_rejected(self):
        d, raw, batches = fixture()
        batches['64'] = [('b.f', batches['64'][0][1] + '\n')]
        with self.assertRaisesRegex(ValueError, 'source bytes'):
            derive_geometry_contract(d, raw, batches)

    def test_ambiguous_batch_rejected(self):
        d, raw, batches = fixture()
        raw['Other/b.f'] = raw['SubProcesses/P0/b.f']
        d['source_bindings'] = source_bindings(raw)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            derive_geometry_contract(d, raw, batches)

    def test_repeated_batch_rejected(self):
        d, raw, batches = fixture()
        batches['64'] = batches['64'] * 2
        with self.assertRaisesRegex(ValueError, 'repeated'):
            derive_geometry_contract(d, raw, batches)

    def test_stale_window_receipt_rejected(self):
        d, raw, batches = fixture()
        d['momentum_environment_contract']['64']['complete'] = False
        with self.assertRaisesRegex(ValueError, 'window receipt'):
            derive_geometry_contract(d, raw, batches)

    def test_changed_producer_coordinate_rejected(self):
        d, raw, batches = fixture()
        d['source_proof']['64']['producers'][0]['line'] += 1
        with self.assertRaisesRegex(ValueError, 'window receipt'):
            derive_geometry_contract(d, raw, batches)

    def test_serialized_result_mutation_falls_back(self):
        d, _, _ = joined()
        for key in ['graph', 'routes', 'source_proof', 'source_bindings']:
            changed = copy.deepcopy(d)
            changed[key] = {}
            self.assertEqual(geometry_support(changed), (False, False), key)

    def test_contract_mutation_falls_back(self):
        d, _, _ = joined()
        c = d['geometry_window_source_contract']
        for key, value in [('schema', 'unknown'), ('slots', {}),
                           ('runtime_abi_qualified', True), ('contract_sha256', '0' * 64)]:
            changed = copy.deepcopy(d)
            changed['geometry_window_source_contract'][key] = value
            self.assertEqual(geometry_support(changed), (False, False), key)

    def test_parser_subset_must_join_full_source_closure(self):
        d, raw, batches = fixture()
        parser = dict(raw)
        parser['Source/DHELAS/a.f'] += '\n'
        with self.assertRaisesRegex(ValueError, 'parser sources'):
            derive_geometry_contract(d, raw, batches, parser)

    def test_whole_source_inventory_can_include_unreachable_program(self):
        d, raw, batches = fixture()
        parser = dict(raw)
        raw['Source/MODEL/unreachable.f'] = f('PROGRAM UNREACHABLE\nSTOP\nEND')
        d['source_bindings'] = source_bindings(raw)
        d['geometry_window_source_contract'] = derive_geometry_contract(d, raw, batches, parser)
        self.assertEqual(geometry_support(d), (True, True))

    def test_precision_admissions_are_independent(self):
        d, raw, batches = fixture()
        qp = raw['SubProcesses/P0/b.f'].replace('SUBROUTINE B(', 'SUBROUTINE Q(')
        qp = qp.replace('      END\n', '      X=FOREIGN()\n      END\n')
        raw['SubProcesses/P0/qp.f'] = qp
        receipts = [{'caller': 'qp.f', 'line': line, 'call': statement}
                    for _, line, statement in fixed(qp, 'x')
                    if statement.startswith('CALLMAKE')]
        batches['128'] = [('qp.f', qp)]
        d['source_proof']['128']['producers'] = receipts
        d['source_bindings'] = source_bindings(raw)
        for width in ('64', '128'):
            d['momentum_environment_contract'][width] = momentum_window(
                raw, batches[width], d['source_proof'][width]['producers'])
        d['geometry_window_source_contract'] = derive_geometry_contract(d, raw, batches)
        self.assertEqual(geometry_support(d), (True, False))


class HookConfinement(unittest.TestCase):
    def test_lifetime_transformations_unchanged(self):
        def funcs(path):
            text = path.read_text()
            return {n.name: ast.get_source_segment(text, n)
                    for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
        old = funcs(PARENT1 / 'src/madgraph/loop/invariant_render.py')
        new = funcs(HERE / 'src/madgraph/loop/invariant_render.py')
        for name in ['once', 'main_template', 'mp_template', 'batch_template']:
            self.assertEqual(old[name], new[name], name)

    def test_main_entry_rotation_skip_narrowing_invalidation(self):
        original = (PARENT1 / 'src/madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc').read_text()
        got = render.main_template(original)
        self.assertIn('208 CONTINUE\nGP_ENV_DP=0_C_INT64_T\nGP_ENV_QP=0_C_INT64_T\nGP_FRESH_DP=.FALSE.\nGP_FRESH_QP=.FALSE.', got)
        self.assertEqual(got.count('GP_FRESH_DP=.FALSE.\n  CALL %(proc_prefix)sMP_COMPUTE_LOOP_COEFS_PROVENANCE'), 2)
        self.assertLess(got.index('  GP_FRESH_DP=.FALSE.\n  GP_ENV_DP=0_C_INT64_T'), got.index('  IF (SKIP_LOOPNUM_COEFS_CONSTRUCTION)'))
        self.assertIn('GP_FRESH_DP=.NOT.FILTER_SO', got)
        self.assertIn('5001 CONTINUE\nENDDO\n## }\nGP_FRESH_DP=.FALSE.\nGP_FRESH_QP=.FALSE.', got)

    def test_qp_capture_only_inside_genuine_producer_branch(self):
        original = (PARENT1 / 'src/madgraph/iolibs/template_files/loop_optimized/mp_compute_loop_coefs.inc').read_text()
        got = render.mp_template(original)
        self.assertIn('GP_ENV_OUT=0_C_INT64_T\nGP_FRESH_OUT=.FALSE.\nCOMPUTE_INTEGRAND_IN_QP =', got)
        self.assertIn('IF (COMPUTE_INTEGRAND_IN_QP) THEN\n    GP_ENV=READ_KINEMATIC_ENVIRONMENT_TOKEN()\n    GP_FRESH=.NOT.FILTER_SO\n    %(mp_born_ct_helas_calls)s', got)
        self.assertEqual(got.count('GP_ENV=READ_KINEMATIC_ENVIRONMENT_TOKEN()'), 1)

    def test_window_mode_query_is_typed_and_no_global_gate_flip(self):
        bounds = (HERE / 'src/madgraph/iolibs/template_files/loop_optimized/invariant_bounds.f90').read_text()
        self.assertIn('if((mode==1.or.mode==2).and.width==64)', bounds)
        self.assertIn('if((mode==4.or.mode==5).and.width==128)', bounds)
        source = (HERE / 'src/madgraph/loop/invariant_render.py').read_text()
        self.assertIn('GEOMETRY_WINDOW_SOURCE_OK(CTMODE,GP_WIDTH)', source)
        self.assertIn('GP_OK=GP_OK.AND.GP_FRESH.AND.GP_ENV_OK', source)
        self.assertIn('GP_OK=GP_OK.AND.ROUTE_MATCHES', source)
        self.assertIn('GP_OK=GP_OK.AND.PRODUCER_KIND_OK', source)
        self.assertIn('METADATA-ONLY SEAM', source)
        provenance = (HERE / 'src/madgraph/loop/invariant_provenance.py').read_text()
        self.assertIn('source_environment_supported=.false.', provenance)
        self.assertNotIn('source_environment_supported=.true.', provenance)


if __name__ == '__main__':
    unittest.main(verbosity=2)
