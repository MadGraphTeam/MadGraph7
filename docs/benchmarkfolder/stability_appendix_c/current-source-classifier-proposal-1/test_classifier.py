"""Pure fixtures only: no native jobs, source edits or historical reclassification."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location('proposed_current_classifier', HERE / 'classifier.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
PILOT = (HERE.parent / 'source5/pilot.py').read_bytes()
fixture_blob = (HERE.parent / 'source5/test_pilot.py').read_bytes()
assert hashlib.sha256(fixture_blob).hexdigest() == 'aac3ef54ca0e731bcd6c17d137381a7f26856595c150ea48d5180420bf974b8d'
fixture_tree = ast.parse(fixture_blob)
fixture_namespace = {}
exec(compile(ast.Module(body=[node for node in fixture_tree.body
    if isinstance(node, ast.FunctionDef) and node.name in ('diagnostic_text', 'group_text')],
    type_ignores=[]), 'frozen-source5/test_pilot.py', 'exec'), fixture_namespace)
LAYOUT = {'nsquaredso': 1, 'nloopgroups': 1, 'ncoefs': [1]}


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def binding_inputs():
    manifest = (ROOT / 'docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/manifest.json').read_bytes()
    template = (ROOT / c.TEMPLATE_KEY).read_bytes()
    worker = (HERE.parent / 'source5/stability_worker.f').read_bytes()
    identity = json.loads(manifest)['candidate']['exporter_identity']
    campaign = {'id': 'entry-latch-cache-guard-1', 'manifest_sha256': c.MANIFEST_SHA}
    # Mock build receipts test joins; this is not an actual qualified new worker.
    matrix = b'''      SUBROUTINE ML5_0_SLOOPMATRIX(P_USER,ANS)
      REF=0.0D0
      DO I=1,NSQSO_BORN
      REF=REF+ANS(0,I)
      ENDDO
      FPE_IN_DP_REDUCTION=.FALSE.
      FPE_IN_QP_REDUCTION=.FALSE.
      MP_DONE=.FALSE.
      MP_DONE_ONCE=.FALSE.
      MP_PS_SET=.FALSE.
      STAB_INDEX=0
      DOING_QP_EVALS=.FALSE.
      END
'''
    source_key = 'SubProcesses/P0_uux_zgg/loop_matrix.f'
    params = {'prefix': 'ML5_0_', 'nexternal': 5, 'nborn': 1}
    driver = worker.decode()
    for token, key in (('@PREFIX@', 'prefix'), ('@NEXTERNAL@', 'nexternal'), ('@NBORN@', 'nborn')):
        driver = driver.replace(token, str(params[key]))
    driver = driver.encode()
    generation = {'source': 'candidate', 'source_identity': identity, 'campaign_binding': campaign}
    compilation = {'campaign_binding': campaign, 'full_numerical_rebuild_this_invocation': True,
                   'numerical_build_from_sources': True, 'sources': {source_key: c.sha(matrix)},
                   'worker_render_parameters': params}
    private = {'driver_sha256': c.sha(driver), 'executable_sha256': 'a' * 64,
               'donor': {'compiled': compilation, 'subprocess': source_key.rsplit('/', 1)[0]},
               'group_layout': {'prefix': params['prefix'],
                                'layout_source_hashes': {'loop_matrix.f': c.sha(matrix)}},
               'command': {'process_started': True, 'returncode': 0, 'error': None,
                           'cleanup': {'survivors': []}}}
    return dict(manifest=manifest, template=template, generation=encoded(generation),
                compilation=encoded(compilation), matrix=matrix, worker_template=worker,
                driver=driver, private_build=encoded(private), matrix_source_key=source_key,
                worker_parameters=params)


def journal(*, before=(0, 0), after=(0, 0), failed=False, coverage=False, nonfinite=False):
    raw = fixture_namespace['group_text'](failed_call=2 if failed else None,
        coverage_call=2 if coverage else None, nonfinite_call=2 if nonfinite else None)
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        fields = line.split()
        if fields[:5] == ['B', '1', '2', '1', '1']:
            fields[-2:] = map(str, before)
            lines[index] = ' '.join(fields)
        if fields[:5] == ['C', '1', '2', '1', '1']:
            fields[-2:] = map(str, after)
            lines[index] = ' '.join(fields)
    return '\n'.join(lines) + '\n'


class Classification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qualified = c.qualify_sources(**binding_inputs())
        cls.new = c.parsers(PILOT, qualification=cls.qualified)
        cls.old = c.parsers(PILOT)

    def group(self, **kwargs):
        return self.new['parse_groups'](journal(**kwargs), LAYOUT)['calls'][(1, 2, 1, 1)]

    def test_incoming_dp_qp_or_both_cleared_can_qualify(self):
        for before in ((1, 0), (0, 1), (1, 1)):
            with self.subTest(before=before):
                row = self.group(before=before)
                self.assertTrue(row['eligible'])
                self.assertFalse(row['legacy_source5_eligible'])
                self.assertTrue(row['incoming_software_latch_cleared'])
                self.assertEqual((row['fpe_dp_before'], row['fpe_qp_before']), tuple(map(bool, before)))
                self.assertFalse(row['hardware_ieee_state_observed'])

    def test_outgoing_dp_qp_or_both_still_rejected(self):
        for before in ((0, 0), (1, 0), (0, 1), (1, 1)):
            for after in ((1, 0), (0, 1), (1, 1)):
                with self.subTest(before=before, after=after):
                    row = self.group(before=before, after=after)
                    self.assertFalse(row['eligible'])
                    self.assertIn(c.LATCH_REASON, row['reason'])
                    self.assertEqual((row['fpe_dp_set_during_current_call'], row['fpe_qp_set_during_current_call']),
                                     tuple(map(bool, after)))

    def test_failed_group_not_rescued_by_clear_flags(self):
        row = self.group(before=(1, 0), failed=True)
        self.assertFalse(row['eligible'])
        self.assertEqual(row['failed_groups'], 1)
        self.assertIn('Active loop group', row['reason'])
        self.assertTrue(row['invalid_groups'])

    def test_nonfinite_group_not_rescued(self):
        row = self.group(before=(1, 0), nonfinite=True)
        self.assertFalse(row['eligible'])
        self.assertIn('Nonfinite', row['reason'])

    def test_missing_coverage_not_rescued(self):
        row = self.group(before=(1, 0), coverage=True)
        self.assertFalse(row['eligible'])
        self.assertIn('coverage unavailable', row['reason'])

    def test_malformed_journal_never_adjusted(self):
        raw = journal(before=(1, 0)) + 'G bad\n'
        result = self.new['parse_groups'](raw, LAYOUT)
        self.assertTrue(result['parse_issues'])
        row = result['calls'][(1, 2, 1, 1)]
        self.assertFalse(row['eligible'])
        self.assertFalse(row['incoming_software_latch_cleared'])

    def test_wrapper_runs_before_row_and_fixed_outcome_derivation(self):
        raw = fixture_namespace['diagnostic_text']()
        old = self.old['parse_diagnostics']('1 100 1 1 1', raw, '', journal(before=(1, 0)), LAYOUT)
        new = self.new['parse_diagnostics']('1 100 1 1 1', raw, '', journal(before=(1, 0)), LAYOUT)
        self.assertFalse(old['rows'][2]['raw_DP_diagnostic_eligible'])
        self.assertTrue(new['rows'][2]['raw_DP_diagnostic_eligible'])
        self.assertEqual(old['primary_call_outcomes'][0]['state'], 'rejected')
        self.assertEqual(new['primary_call_outcomes'][0]['state'], 'diagnostic_available')
        self.assertFalse(new['host_accuracy_qualified'])
        self.assertEqual(new['rows'][2]['status'], 110)

    def test_status_warning_nonfinite_and_completeness_still_reject(self):
        samples = ((fixture_namespace['diagnostic_text'](status=120), ''),
                   (fixture_namespace['diagnostic_text'](), '##W03 unattributed\n'),
                   (fixture_namespace['diagnostic_text']().replace('110 1 2 3 4', '110 1 NaN 3 4'), ''),
                   (fixture_namespace['diagnostic_text']().replace('# COMPLETE 1 300', ''), ''))
        for raw, console in samples:
            with self.subTest(console=console, prefix=raw[:30]):
                parsed = self.new['parse_diagnostics']('1 100 1 1 1', raw, console, journal(before=(1, 0)), LAYOUT)
                self.assertFalse(parsed['rows'][2]['raw_DP_diagnostic_eligible'])

    def test_historical_default_exactly_unchanged(self):
        result = c.parsers(PILOT)['parse_groups'](journal(before=(1, 1)), LAYOUT)
        row = result['calls'][(1, 2, 1, 1)]
        self.assertFalse(row['eligible'])
        self.assertEqual(row['reason'], c.LATCH_REASON)
        self.assertTrue(row['fpe_dp_before'] and row['fpe_qp_before'])
        self.assertFalse(row['fpe_dp'] or row['fpe_qp'])
        self.assertNotIn('entry_reset_source_qualified', row)


class Binding(unittest.TestCase):
    def test_modified_or_historical_template_rejected(self):
        args = binding_inputs()
        args['template'] += b'\n'
        with self.assertRaisesRegex(ValueError, 'qualified entry-reset'):
            c.qualify_sources(**args)

    def test_actual_generated_source_must_match_compiled_receipt(self):
        args = binding_inputs()
        args['matrix'] += b'changed'
        with self.assertRaisesRegex(ValueError, 'Compiled generated'):
            c.qualify_sources(**args)

    def test_historical_generation_identity_rejected(self):
        args = binding_inputs()
        value = json.loads(args['generation'])
        value['source_identity']['files'][c.TEMPLATE_KEY] = '0' * 64
        args['generation'] = encoded(value)
        with self.assertRaisesRegex(ValueError, 'Generated source inventory'):
            c.qualify_sources(**args)

    def test_unqualified_driver_or_failed_private_build_rejected(self):
        for target in ('driver', 'private_build'):
            args = binding_inputs()
            if target == 'driver':
                args[target] += b'changed'
            else:
                value = json.loads(args[target]); value['command']['returncode'] = 1
                args[target] = encoded(value)
            with self.subTest(target=target), self.assertRaises(ValueError):
                c.qualify_sources(**args)

    def test_boolean_flag_cannot_enable_new_classification(self):
        result = c.parsers(PILOT)['parse_groups'](journal(), LAYOUT)
        with self.assertRaisesRegex(ValueError, 'qualified generated-source'):
            c.adjust_group_calls(result, True)

    def test_mixed_subprocess_receipts_rejected(self):
        args = binding_inputs()
        value = json.loads(args['private_build'])
        value['donor']['subprocess'] = 'SubProcesses/P9_other'
        args['private_build'] = encoded(value)
        with self.assertRaisesRegex(ValueError, 'different compiled subprocess'):
            c.qualify_sources(**args)

    def test_mixed_compilation_or_worker_dimensions_rejected(self):
        for target in ('donor', 'dimensions'):
            args = binding_inputs()
            value = json.loads(args['private_build'])
            if target == 'donor':
                value['donor']['compiled']['numerical_build_from_sources'] = False
            else:
                value['group_layout']['prefix'] = 'ML5_9_'
            args['private_build'] = encoded(value)
            with self.subTest(target=target), self.assertRaises(ValueError):
                c.qualify_sources(**args)

    def test_stale_matrix_rejected_even_if_receipt_digests_are_recomputed(self):
        args = binding_inputs()
        args['matrix'] = args['matrix'].replace(b'      FPE_IN_DP_REDUCTION=.FALSE.\n', b'')
        compiled = json.loads(args['compilation'])
        compiled['sources'][args['matrix_source_key']] = c.sha(args['matrix'])
        private = json.loads(args['private_build'])
        private['donor']['compiled'] = compiled
        private['group_layout']['layout_source_hashes']['loop_matrix.f'] = c.sha(args['matrix'])
        args.update(compilation=encoded(compiled), private_build=encoded(private))
        with self.assertRaisesRegex(ValueError, 'lacks the qualified entry-reset'):
            c.qualify_sources(**args)

    def test_conditional_entry_reset_is_rejected(self):
        args = binding_inputs()
        matrix = args['matrix'].replace(b'      REF=0.0D0', b'      IF(X) THEN\n      REF=0.0D0')
        with self.assertRaisesRegex(ValueError, 'conditional'):
            c.generated_entry_reset(matrix, 'ML5_0_', 5, args['template'])

    def test_actual_closed_lifecycle_candidate_and_original_source_fixtures(self):
        base = ROOT / 'docs/benchmarkfolder/experiments/reduction-latch-lifecycle/source-proposal1'
        candidate = (base / 'loop_matrix.candidate.f').read_bytes()
        self.assertEqual(c.sha(candidate), 'b37926358e421ba1de350eb0a6cb0ce9eab5127b171365eeb682439e0b6c9c5f')
        template = binding_inputs()['template']
        c.generated_entry_reset(candidate, 'ML5_0_', 5, template)
        with self.assertRaisesRegex(ValueError, 'lacks the qualified entry-reset'):
            c.generated_entry_reset((base / 'loop_matrix.original.f').read_bytes(), 'ML5_0_', 5, template)


if __name__ == '__main__':
    unittest.main()
