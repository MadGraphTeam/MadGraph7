"""Pure parsing, mocked admission, and temporary-filesystem controller tests.

No test starts a CLI, compiler, executable, loader, or numerical worker.
"""
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('appendix_c_pilot_tests', HERE / 'pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def diagnostic_text(*, status=110, accuracy=-1., bootstrap=1, missing=None):
    rows = []
    for call in range(1, bootstrap + 1):
        for order in (0, 1):
            rows.append('0 %d %d 0 %d %d 1 2 3 4 -1' %
                        (call, call, order, 110 if call == bootstrap else 130))
    rows.append('# BOOTSTRAP_COMPLETE %d' % bootstrap)
    call = bootstrap
    for point in range(1, 101):
        for method in (1, 2, 3):
            call += 1
            for order in (0, 1):
                if (point, method, order) == missing:
                    continue
                rows.append('1 %d %d %d %d %d 1 2 3 4 %.17g' %
                            (call, point, method, order, status, accuracy))
    rows.append('# COMPLETE %d 300' % bootstrap)
    return '\n'.join(rows) + '\n'


LAYOUT = {'nsquaredso': 1, 'nloopgroups': 1, 'ncoefs': [1]}


def group_text(*, failed_call=None, coverage_call=None, nonfinite_call=None,
               fpe_call=None, before_set=False, bootstrap=1):
    lines = ['# GROUP_SCHEMA 1 1 1']
    identities = [(0, call, call, 0) for call in range(1, bootstrap + 1)]
    identities += [(1, bootstrap + (point - 1) * 3 + method, point, method)
                   for point in range(1, 101) for method in (1, 2, 3)]
    for phase, call, point, method in identities:
        prefix = '%d %d %d %d' % (phase, call, point, method)
        lines.append('B ' + prefix + (' 1 0' if before_set else ' 0 0'))
        if call == coverage_call:
            lines.append('C ' + prefix + ' 0 0 0 0 0 0 0')
            continue
        failed = int(call == failed_call)
        nonfinite = int(call == nonfinite_call)
        fpe = int(call == fpe_call or before_set)
        words = ['7ff0000000000000' if nonfinite else '0000000000000001', '0000000000000000']
        words += ['0000000000000001'] * 6
        lines.append('G ' + prefix + ' 1 1 1 1 %d %d 0 ' % (nonfinite, 1 - failed) + ' '.join(words))
        lines.append('C ' + prefix + ' 1 1 %d %d 0 %d 0' % (failed, nonfinite, fpe))
    lines.append('# COMPLETE %d 300' % bootstrap)
    return '\n'.join(lines) + '\n'


def parse(raw, console='', groups=None):
    return pilot.parse_diagnostics('1 100 1 1 1', raw, console,
                                   group_text() if groups is None else groups, LAYOUT)


def parsed(**kwargs):
    return parse(diagnostic_text(**kwargs))


class InertPlanning(unittest.TestCase):
    def test_default_plan_launches_nothing(self):
        with (mock.patch.object(pilot.subprocess, 'Popen', side_effect=AssertionError('native')),
              mock.patch.object(pilot, 'verify_production', side_effect=AssertionError('active input')),
              contextlib.redirect_stdout(io.StringIO()) as output):
            self.assertEqual(pilot.main([]), 0)
        self.assertEqual(json.loads(output.getvalue())['commands_executed'], 0)

    def test_dryrun_is_inert(self):
        for argv in ([], ['plan', '--execute'], ['prepare'], ['run']):
            with (self.subTest(argv=argv), mock.patch.object(pilot.subprocess, 'Popen', side_effect=AssertionError('native')),
                  mock.patch.object(pilot, 'load_helpers', side_effect=AssertionError('execute helper')),
                  mock.patch.object(pilot, 'verify_production', side_effect=AssertionError('active input')),
                  contextlib.redirect_stdout(io.StringIO()) as output):
                self.assertEqual(pilot.main(argv), 0)
                result = json.loads(output.getvalue())
                self.assertEqual(result['commands_executed'], 0)
                self.assertEqual(result['native_calls'], 0)
                self.assertEqual(len(result['requests']), 20)

    def test_source_seal_precedes_helper_execution(self):
        with (mock.patch.object(pilot, 'verify_seal', side_effect=ValueError('unsealed')),
              mock.patch.object(pilot, 'load_helpers', side_effect=AssertionError('helper executed'))):
            with self.assertRaisesRegex(ValueError, 'unsealed'):
                pilot.main(['prepare', '--execute'])

    def test_only_closed_four_case_schedule(self):
        value = pilot.protocol()
        schedule = pilot.requests(value)
        self.assertEqual({item['case'] for item in schedule}, set(pilot.CASE_IDS))
        self.assertFalse(any(item['case'] == 'ew_n4_epem_mupmumaa' for item in schedule))
        self.assertEqual(sum(item['primary_calls_expected'] for item in schedule), 6000)
        self.assertEqual(sum(item['lock_required'] for item in schedule), 8)

    def test_unknown_donor_is_rejected_before_filesystem_read(self):
        with mock.patch.object(pilot, 'digest', side_effect=AssertionError('donor read')):
            with self.assertRaisesRegex(ValueError, 'potentially active'):
                pilot.donor({}, {}, 'ew_n4_epem_mupmumaa')

    def test_render_only_three_known_placeholders(self):
        self.assertEqual(pilot.render('@PREFIX@ @NEXTERNAL@ @NBORN@',
                         {'prefix': 'ML5_0_', 'nexternal': 4, 'nborn': 1}), 'ML5_0_ 4 1')
        with self.assertRaisesRegex(ValueError, 'Unexpanded'):
            pilot.render('@UNKNOWN@', {'prefix': '', 'nexternal': 4, 'nborn': 1})

    def test_make_target_has_no_numerical_dependencies(self):
        self.assertIn('pilot-build:\n', pilot.MAKEFILE)
        lines = [line for line in pilot.MAKEFILE.splitlines() if line.startswith('\t')]
        self.assertEqual(len(lines), 2)
        self.assertIn('-c check_sa.f -o check_sa.o', lines[0])
        self.assertIn('-o stability-check check_sa.o $(PROCESS) $(LINKLIBS)', lines[1])
        self.assertNotIn('$(MAKE)', pilot.MAKEFILE)


class Parsing(unittest.TestCase):
    def test_full_stream_has_exact_population(self):
        result = parsed()
        self.assertTrue(result['structurally_complete'])
        self.assertEqual(len(result['rows']), 602)
        self.assertEqual(len(result['primary_call_outcomes']), 300)
        self.assertTrue(all(row['state'] == 'diagnostic_available' for row in result['primary_call_outcomes']))
        self.assertFalse(result['host_accuracy_qualified'])

    def test_unchecked_accuracy_is_not_a_checked_gate(self):
        result = parsed(accuracy=-1.)
        self.assertTrue(result['rows'][2]['raw_DP_diagnostic_eligible'])
        self.assertFalse(result['rows'][2]['host_accuracy_qualified'])
        self.assertEqual(result['rows'][2]['backend_observation'], 'not_available_in_unchecked_retcode')

    def test_positive_120_is_retained_and_excluded_from_dp(self):
        result = parsed(status=120)
        self.assertEqual(result['rows'][2]['status'], 120)
        self.assertFalse(result['rows'][2]['raw_DP_diagnostic_eligible'])
        self.assertEqual(result['primary_call_outcomes'][0]['state'], 'rejected')

    def test_warning_is_attributed_to_call(self):
        console = 'STABILITY_CALL_BEGIN 1 2 1 1 1\n##W03 numerical warning\nSTABILITY_CALL_END 1 2 1 1 1 110\n'
        result = parse(diagnostic_text(), console)
        self.assertFalse(result['rows'][2]['raw_DP_diagnostic_eligible'])
        self.assertTrue(result['rows'][4]['raw_DP_diagnostic_eligible'])
        self.assertEqual(result['warning_calls'], [[1, 2, 1, 1]])

    def test_unattributed_warning_conservatively_excludes_primary(self):
        result = parse(diagnostic_text(), '##W03 outside a call\n')
        self.assertTrue(result['unattributed_warning'])
        self.assertFalse(any(row['raw_DP_diagnostic_eligible'] for row in result['rows']))

    def test_missing_row_is_not_dropped_from_population(self):
        result = parsed(missing=(100, 3, 1))
        self.assertFalse(result['structurally_complete'])
        self.assertEqual(len(result['primary_call_outcomes']), 300)
        self.assertEqual(result['primary_call_outcomes'][-1]['state'], 'rejected')

    def test_missing_whole_call_is_unexecuted(self):
        raw = diagnostic_text()
        raw = '\n'.join(line for line in raw.splitlines() if not line.startswith('1 301 '))
        result = parse(raw)
        self.assertEqual(result['primary_call_outcomes'][-1]['state'], 'unexecuted')

    def test_duplicate_row_is_parse_failure(self):
        raw = diagnostic_text()
        raw += raw.splitlines()[3] + '\n'
        result = parse(raw)
        self.assertFalse(result['structurally_complete'])
        self.assertTrue(result['parse_issues'])

    def test_wrong_call_sequence_is_incomplete(self):
        result = parse(diagnostic_text().replace('1 2 1 1 0', '1 999 1 1 0'))
        self.assertFalse(result['structurally_complete'])

    def test_layout_and_observed_loop_orders_must_match(self):
        with self.assertRaisesRegex(ValueError, 'frozen initialized loop layout'):
            pilot.parse_diagnostics('1 100 2 2 1', '', '', '', LAYOUT)

    def test_bootstrap_requires_final_110_and_completion_marker(self):
        raw = diagnostic_text().replace('0 1 1 0 0 110', '0 1 1 0 0 130')
        result = parse(raw)
        self.assertFalse(result['bootstrap_readiness_verified'])
        self.assertFalse(result['rows'][2]['raw_DP_diagnostic_eligible'])
        raw = '\n'.join(line for line in diagnostic_text().splitlines() if not line.startswith('# BOOTSTRAP_COMPLETE'))
        self.assertFalse(parse(raw)['bootstrap_readiness_verified'])

    def test_bad_split_order_invalidates_aggregate_order_zero(self):
        raw = diagnostic_text().replace('1 2 1 1 1 110 1 2', '1 2 1 1 1 110 1 NaN', 1)
        result = parse(raw)
        self.assertFalse(result['rows'][2]['raw_DP_diagnostic_eligible'])
        self.assertIn('split-order', result['rows'][2]['reason'])

    def test_mismatched_status_across_orders_invalidates_call(self):
        raw = diagnostic_text().replace('1 2 1 1 1 110', '1 2 1 1 1 120', 1)
        result = parse(raw)
        self.assertFalse(result['rows'][2]['raw_DP_diagnostic_eligible'])

    def test_nonfinite_values_are_json_safe_and_ineligible(self):
        result = parse(diagnostic_text().replace('110 1 2 3 4', '110 1 NaN 3 4'))
        self.assertFalse(result['rows'][2]['raw_DP_diagnostic_eligible'])
        json.dumps(result, allow_nan=False)

    def test_missing_dimensions_fails_explicitly(self):
        for header in ('', '1 99 1 1 1', '2 100 1 1 1', '1 100 0 1 1'):
            with self.subTest(header=header), self.assertRaises(ValueError):
                pilot.parse_diagnostics(header, '', '')

    def test_unknown_group_coverage_cannot_qualify_finite_remainder(self):
        result = pilot.parse_diagnostics('1 100 1 1 1', diagnostic_text(), '')
        self.assertFalse(any(row['raw_DP_diagnostic_eligible'] for row in result['rows']))
        self.assertFalse(result['loop_groups_structurally_complete'])

    def test_active_tensor_failure_rejects_110_remainder(self):
        result = parse(diagnostic_text(), groups=group_text(failed_call=2))
        self.assertEqual(result['rows'][2]['status'], 110)
        self.assertFalse(result['rows'][2]['raw_DP_diagnostic_eligible'])
        self.assertIn('Active loop group', result['rows'][2]['reason'])
        self.assertTrue(result['rows'][4]['raw_DP_diagnostic_eligible'])

    def test_unknown_initialized_coverage_rejects_affected_call(self):
        result = parse(diagnostic_text(), groups=group_text(coverage_call=2))
        self.assertFalse(result['rows'][2]['raw_DP_diagnostic_eligible'])
        self.assertTrue(result['loop_groups_structurally_complete'])

    def test_nonfinite_tensor_bits_reject_point(self):
        result = parse(diagnostic_text(), groups=group_text(nonfinite_call=2))
        self.assertFalse(result['rows'][2]['raw_DP_diagnostic_eligible'])
        self.assertIn('Nonfinite tensor', result['rows'][2]['reason'])

    def test_sticky_fpe_is_not_claimed_new(self):
        result = parse(diagnostic_text(), groups=group_text(before_set=True))
        self.assertFalse(result['rows'][2]['raw_DP_diagnostic_eligible'])
        self.assertTrue(result['loop_group_calls'][0]['fpe_dp_before'])
        self.assertFalse(result['loop_group_calls'][0]['fpe_dp_became_set'])

    def test_new_fpe_transition_recorded_without_reset(self):
        result = parse(diagnostic_text(), groups=group_text(fpe_call=2))
        self.assertTrue(result['loop_group_calls'][1]['fpe_dp_became_set'])
        self.assertFalse(result['rows'][2]['raw_DP_diagnostic_eligible'])

    def test_wrong_coefficient_extent_is_not_admitted(self):
        raw = group_text().replace('G 1 2 1 1 1 1 1 ', 'G 1 2 1 1 1 1 2 ', 1)
        result = parse(diagnostic_text(), groups=raw)
        self.assertTrue(result['loop_group_parse_issues'])
        self.assertFalse(result['rows'][2]['raw_DP_diagnostic_eligible'])

    def test_retained_result_words_validate_nonfinite_counter(self):
        raw = group_text()
        tokens = raw.splitlines()
        parts = tokens[2].split()
        self.assertEqual(parts[0], 'G')
        parts[14] = '7ff0000000000000'
        tokens[2] = ' '.join(parts)
        result = parse(diagnostic_text(), groups='\n'.join(tokens))
        self.assertTrue(result['loop_group_parse_issues'])
        self.assertFalse(result['rows'][2]['raw_DP_diagnostic_eligible'])


class Metrics(unittest.TestCase):
    @staticmethod
    def spread(values):
        span, mean = max(values) - min(values), math.fsum(item / 3 for item in values)
        return (0. if span == 0 else math.inf) if mean == 0 else span / abs(mean)

    def lane(self):
        return {'request': {'id': pilot.CASE_IDS[0] + '-backend1'}, 'diagnostics': parsed(),
                'runtime_binding': {'verified': True}, 'readonly_guard_passed': True}

    def test_full_intended_population_even_when_nineteen_requests_unexecuted(self):
        dataset = pilot.point_metrics(pilot.protocol(), [self.lane()], self.spread)
        self.assertEqual(len(dataset['cases']), 4)
        self.assertEqual(sum(len(case['lanes']) for case in dataset['cases']), 20)
        self.assertTrue(all(len(lane['values']) == 100 for case in dataset['cases'] for lane in case['lanes']))
        self.assertEqual(dataset['cases'][0]['lanes'][0]['values'], [0.] * 100)
        self.assertEqual(dataset['cases'][0]['lanes'][1]['values'], [None] * 100)

    def test_unknown_runtime_is_not_plotted(self):
        lane = self.lane(); lane.pop('runtime_binding')
        result = pilot.point_metrics(pilot.protocol(), [lane], self.spread)
        self.assertEqual(result['cases'][0]['lanes'][0]['values'], [None] * 100)

    def test_failed_readonly_guard_is_not_plotted(self):
        lane = self.lane(); lane['readonly_guard_passed'] = False
        result = pilot.point_metrics(pilot.protocol(), [lane], self.spread)
        self.assertEqual(result['cases'][0]['lanes'][0]['values'], [None] * 100)

    def test_zero_born_is_unavailable(self):
        lane = self.lane(); lane['diagnostics']['rows'][2]['coefficients'][0] = 0.
        result = pilot.point_metrics(pilot.protocol(), [lane], self.spread)
        self.assertIsNone(result['cases'][0]['lanes'][0]['values'][0])

    def test_explicit_infinity_and_zero_are_preserved(self):
        lane = self.lane()
        for row in lane['diagnostics']['rows']:
            if row['phase'] == 1 and row['point'] == 1 and row['order'] == 0:
                row['coefficients'][1] = {1: -1., 2: 0., 3: 1.}[row['method']]
        result = pilot.point_metrics(pilot.protocol(), [lane], self.spread)
        self.assertEqual(result['cases'][0]['lanes'][0]['values'][:2], ['inf', 0.])


class PrivateFiles(unittest.TestCase):
    def test_card_tokens_independently_verify_all_control_types(self):
        values = {'MLReductionLib': '9', 'CTModeRun': 1, 'TIRustComputeFinite': True,
                  'COLLIERRequiredAccuracy': 1e-13, 'TIRustGlobalCache': 0}
        text = '''#MLReductionLib
9
#CTModeRun
1
#TIRustComputeFinite
.TRUE.
#COLLIERRequiredAccuracy
1.0D-13
#TIRustGlobalCache
0
'''
        self.assertEqual(pilot.verify_card_tokens(text, values), values)
        for bad in (text.replace('.TRUE.', '.FALSE.'), text.replace('1.0D-13', '1.0D-3'),
                    text.replace('\n9\n', '\n9|1\n'), text + '#MLReductionLib\n9\n'):
            with self.subTest(bad=bad[-30:]), self.assertRaises(ValueError):
                pilot.verify_card_tokens(bad, values)

    @staticmethod
    def layout_fixture(path, tail='2,I_SO,1'):
        source = '''      PARAMETER(NSQUAREDSO=1,NLOOPGROUPS=1,NBORNAMPS=1)
      COMPLEX*16 LOOPCOEFS(0:LOOPMAXCOEFS-1,NSQUAREDSO,
     $ NLOOPGROUPS)
      COMPLEX*16 LOOPRES(3,NSQUAREDSO,NLOOPGROUPS)
      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)
      COMMON/ML5_0_LCOEFS/LOOPCOEFS
      COMMON/ML5_0_LOOPRES/LOOPRES,S
      LOGICAL UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,CT_REQ_SO_DONE,
     $ MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,MP_LOOP_REQ_SO_DONE,
     $ CTCALL_REQ_SO_DONE,FILTER_SO
      DATA UVCT_REQ_SO_DONE/.FALSE./
      DATA MP_UVCT_REQ_SO_DONE/.FALSE./
      DATA CT_REQ_SO_DONE/.FALSE./
      DATA MP_CT_REQ_SO_DONE/.FALSE./
      DATA LOOP_REQ_SO_DONE/.FALSE./
      DATA MP_LOOP_REQ_SO_DONE/.FALSE./
      DATA CTCALL_REQ_SO_DONE/.FALSE./
      DATA FILTER_SO/.FALSE./
      COMMON/ML5_0_SO_REQS/UVCT_REQ_SO_DONE,MP_UVCT_REQ_SO_DONE,
     $ CT_REQ_SO_DONE,MP_CT_REQ_SO_DONE,LOOP_REQ_SO_DONE,
     $ MP_LOOP_REQ_SO_DONE,CTCALL_REQ_SO_DONE,FILTER_SO
      INTEGER SQSO_TARGET
      COMMON/ML5_0_SOCHOICE/SQSO_TARGET
      LOGICAL FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION
      DATA FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION/.FALSE.,.FALSE./
      COMMON/ML5_0_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,
     $ FPE_IN_QP_REDUCTION
      PARAMETER(SKIPLOOPEVAL=.FALSE.)
      DATA SQSO_TARGET/-1/
      DO I=1,NLOOPGROUPS
      DO J=0,LOOPMAXCOEFS-1
      DO K=1,NSQUAREDSO
      LOOPCOEFS(J,K,I)=(0.0D0,0.0D0)
      ENDDO
      ENDDO
      ENDDO
      DO I=1,NLOOPGROUPS
      DO J=1,3
      DO K=1,NSQUAREDSO
      LOOPRES(J,K,I)=(0.0D0,0.0D0)
      ENDDO
      ENDDO
      ENDDO
      DO I_SO=1,NSQUAREDSO
      DO J=1,NLOOPGROUPS
      S(I_SO,J)=.TRUE.
      ENDDO
'''
        (path / 'loop_matrix.f').write_text(source)
        (path / 'nsqso_born.inc').write_text('      PARAMETER(NSQSO_BORN=1)\n')
        (path / 'loop_max_coefs.inc').write_text('      PARAMETER(LOOPMAXCOEFS=35)\n')
        (path / 'loop_CT_calls_1.f').write_text('      CALL ML5_0_LOOP_2_3(1,2,' + tail + ')\n')

    def test_group_mapping_includes_grouped_topology_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); self.layout_fixture(path)
            result = pilot.group_layout(path)
            self.assertEqual(result['ranks'], [2])
            self.assertEqual(result['ncoefs'], [15])
            self.assertTrue(all(result['source_prerequisites'].values()))
            self.assertIn('     $ 15/', result['include'])

    def test_missing_canonical_initialization_rejects_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); self.layout_fixture(path)
            source = path / 'loop_matrix.f'
            source.write_text(source.read_text().replace('S(I_SO,J)=.TRUE.', 'S(I_SO,J)=.FALSE.'))
            with self.assertRaisesRegex(ValueError, 'prerequisites'):
                pilot.group_layout(path)

    def test_nonconstant_or_missing_group_mapping_rejected(self):
        for tail in ('RANK,I_SO,1', '2,I_SO,2'):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp); self.layout_fixture(path, tail)
                with self.assertRaises(ValueError): pilot.group_layout(path)

    def test_common_member_order_and_types_are_admitted_explicitly(self):
        changes = [('COMMON/ML5_0_LOOPRES/LOOPRES,S', 'COMMON/ML5_0_LOOPRES/S,LOOPRES'),
                   ('COMPLEX*16 LOOPRES', 'COMPLEX*32 LOOPRES'),
                   ('      DATA FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION/.FALSE.,.FALSE./\n', ''),
                   ('      LOGICAL S(NSQUAREDSO,NLOOPGROUPS)\n', ''),
                   ('COMMON/ML5_0_SOCHOICE/SQSO_TARGET', 'COMMON/ML5_0_SOCHOICE/OTHER,SQSO_TARGET')]
        for old, new in changes:
            with self.subTest(old=old), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp); self.layout_fixture(path)
                source = path / 'loop_matrix.f'; source.write_text(source.read_text().replace(old, new))
                with self.assertRaises(ValueError): pilot.group_layout(path, 'ML5_0_')

    def test_actual_process_prefix_must_match_compiled_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); self.layout_fixture(path)
            with self.assertRaisesRegex(ValueError, 'prefix/common layout differs'):
                pilot.group_layout(path, 'OTHER_')

    def test_exclusive_json_does_not_overwrite_previous_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.json'
            pilot.save(path, {'x': 1})
            with self.assertRaises(FileExistsError): pilot.save(path, {'x': 2})
            self.assertEqual(pilot.read(path), {'x': 1})

    def test_input_parser_requires_exact_count_and_finite_values(self):
        good = '100\n' + ('500 0 0 500\n' * 400)
        self.assertEqual(len(pilot.point_array(good, 4)), 100)
        for bad in (good + '1', good.replace('100\n', '99\n', 1), good.replace('500', 'NaN', 1)):
            with self.subTest(bad=bad[:20]), self.assertRaises(ValueError): pilot.point_array(bad, 4)

    def test_cut_boundaries_are_strict(self):
        point = [[500, 0, 0, 500], [500, 0, 0, -500], [500, 50, 0, 0], [500, -50, 0, 0]]
        self.assertEqual(pilot.cut_report([point])['passed'], 0)
        point[2][1] = 51; point[3][1] = -51
        self.assertEqual(pilot.cut_report([point])['passed'], 1)

    def test_runtime_without_application_sample_is_unqualified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); executable = path / 'app'; executable.write_bytes(b'test')
            shared = path / 'libmadloop_tirust.so'; shared.write_bytes(b'\x7fELFtest')
            with self.assertRaisesRegex(ValueError, 'No authenticated application'):
                pilot.runtime_binding(path, executable, {str(shared): pilot.digest(shared)})

    @staticmethod
    def map_line(path):
        info = path.stat()
        return '00100000-00101000 r-xp 00000000 %x:%x %d %s\n' % (
            pilot.os.major(info.st_dev), pilot.os.minor(info.st_dev), info.st_ino, path)

    def test_runtime_requires_production_dso_in_actual_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); executable = path / 'app'; executable.write_bytes(b'\x7fELFtest')
            shared = path / 'libmadloop_tirust.so'; shared.write_bytes(b'\x7fELFshared')
            maps = path / 'runtime-maps-00.txt'; maps.write_text(self.map_line(executable))
            with self.assertRaisesRegex(ValueError, 'exact target closure'):
                pilot.runtime_binding(path, executable, {str(shared): pilot.digest(shared)})
            maps.write_text(self.map_line(executable) + self.map_line(shared))
            result = pilot.runtime_binding(path, executable, {str(shared): pilot.digest(shared)})
            self.assertTrue(result['verified'])
            self.assertFalse(result['backend_observed'])


class Lifecycle(unittest.TestCase):
    class Requests:
        def __init__(self): self.requested = []
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def check(self):
            if self.requested: raise InterruptedError('signal')

    class Process:
        def __init__(self):
            self.stdin = io.BytesIO(); self.returncode = 0; self.pid = 123
        def poll(self): return 0

    def test_local_hup_is_deferred_and_previous_handler_restored(self):
        old = pilot.signal.getsignal(pilot.signal.SIGHUP)
        cleanup = types.SimpleNamespace(SignalRequests=self.Requests)
        with pilot.signal_requests(cleanup) as guard:
            pilot.signal.getsignal(pilot.signal.SIGHUP)(pilot.signal.SIGHUP, None)
            with self.assertRaises(InterruptedError): guard.check()
        self.assertEqual(pilot.signal.getsignal(pilot.signal.SIGHUP), old)

    def test_terminal_elapsed_includes_cleanup_in_wall_cap(self):
        process = self.Process()
        owner = types.SimpleNamespace(stop=lambda **kw: {'survivors': []})
        cleanup = types.SimpleNamespace(OwnedChild=lambda p: owner)
        with tempfile.TemporaryDirectory() as tmp:
            with (mock.patch.object(pilot.subprocess, 'Popen', return_value=process),
                  mock.patch.object(pilot.time, 'monotonic', side_effect=[0., 61.])):
                record = pilot.command(['true'], Path(tmp), Path(tmp) / 'command', 60., cleanup, self.Requests())
            self.assertEqual(record['returncode'], 0)
            self.assertEqual(record['error']['class'], 'TimeoutError')
            self.assertEqual(record['elapsed_seconds'], 61.)

    def test_setup_environment_is_clean_and_native_ld_only_in_exec_payload(self):
        process = self.Process(); observed = {}
        def spawn(argv, **kwargs):
            observed.update(argv=argv, kwargs=kwargs)
            return process
        owner = types.SimpleNamespace(stop=lambda **kw: {'survivors': []})
        cleanup = types.SimpleNamespace(OwnedChild=lambda p: owner)
        with tempfile.TemporaryDirectory() as tmp:
            with (mock.patch.dict(pilot.os.environ, {'LD_LIBRARY_PATH': '/old-glibc', 'LD_DEBUG': 'libs',
                    'GCONV_PATH': '/bad-gconv', 'GLIBC_TUNABLES': 'bad'}, clear=False),
                  mock.patch.object(pilot.subprocess, 'Popen', side_effect=spawn),
                  mock.patch.object(pilot.time, 'monotonic', side_effect=[0., 1.])):
                result = pilot.command(['/private/stability-check', 'points', '9'], Path(tmp),
                    Path(tmp) / 'native', 60., cleanup, self.Requests(),
                    target_environment={'LD_LIBRARY_PATH': '/bound-native'})
            self.assertNotIn('LD_LIBRARY_PATH', observed['kwargs']['env'])
            self.assertNotIn('LD_DEBUG', observed['kwargs']['env'])
            self.assertNotIn('GCONV_PATH', observed['kwargs']['env'])
            payload = json.loads(observed['argv'][-1])
            self.assertEqual(payload['environment'], {'LD_LIBRARY_PATH': '/bound-native'})
            self.assertEqual(payload['argv'][0], '/private/stability-check')
            self.assertIn('os.execvpe', observed['argv'][-2])
            self.assertTrue(result['process_started'])

    def test_private_driver_and_donor_must_have_same_interpreter(self):
        with mock.patch.object(pilot, 'elf_interpreter', side_effect=[Path('/new/ld.so'), Path('/old/ld.so')]):
            with self.assertRaisesRegex(ValueError, 'changed the target interpreter'):
                pilot.qualify_loader_closure('/new/app', '/old/app', {}, '')

    def test_fresh_target_closure_excludes_only_verified_host_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'target').mkdir(); (root / 'host').mkdir()
            target = root / 'target/ld-linux-x86-64.so.2'; target.write_bytes(b'target-loader')
            host = root / 'host/ld-linux-x86-64.so.2'; host.write_bytes(b'host-loader')
            shared = root / 'libmadloop_tirust.so'; shared.write_bytes(b'target-shared')
            historical = {str(path): pilot.digest(path) for path in (target, host, shared)}
            listing = str(shared) + ' (0x0)\n' + str(target) + ' (0x0)\n'
            with (mock.patch.object(pilot, 'elf_interpreter', return_value=target),
                  mock.patch.object(pilot, 'HISTORICAL_LDD_HOST_LOADER',
                    {'path': str(host), 'sha256': pilot.digest(host), 'binding': 'fixture historical metadata'})):
                value = pilot.qualify_loader_closure(root / 'new', root / 'old', historical, listing)
                self.assertEqual(set(value['files']), {str(target), str(shared)})
                self.assertEqual(set(value['verified_historical_host_loader_exclusions']), {str(host)})
                with self.assertRaisesRegex(ValueError, 'not the verified non-target host loader'):
                    pilot.qualify_loader_closure(root / 'new', root / 'old', historical, str(target))

    def test_elf_interpreter_parser_reads_no_utility(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'elf'
            interpreter = str(Path(tmp) / 'loader').encode() + b'\0'
            header = bytearray(64); header[:6] = b'\x7fELF\x02\x01'
            pilot.struct.pack_into('<Q', header, 32, 64)
            pilot.struct.pack_into('<HH', header, 54, 56, 1)
            program = bytearray(56)
            pilot.struct.pack_into('<I', program, 0, 3)
            pilot.struct.pack_into('<Q', program, 8, 120)
            pilot.struct.pack_into('<Q', program, 32, len(interpreter))
            path.write_bytes(header + program + interpreter)
            with mock.patch.object(pilot.subprocess, 'Popen', side_effect=AssertionError('ELF utility')):
                self.assertEqual(pilot.elf_interpreter(path), Path(tmp) / 'loader')

    def test_unregistered_runtime_elf_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); executable = path / 'app'; executable.write_bytes(b'\x7fELFtest')
            shared = path / 'libmadloop_tirust.so'; shared.write_bytes(b'\x7fELFshared')
            unknown = path / 'unknown.so'; unknown.write_bytes(b'\x7fELFunknown')
            (path / 'runtime-maps-00.txt').write_text(PrivateFiles.map_line(executable) + PrivateFiles.map_line(shared)
                                                    + PrivateFiles.map_line(unknown))
            with self.assertRaisesRegex(ValueError, 'Unregistered mapped ELF'):
                pilot.runtime_binding(path, executable, {str(shared): pilot.digest(shared)})


class RecordedGenerationIdentity(unittest.TestCase):
    @staticmethod
    def fixture():
        binding = {'id': 'invariant-symmetry-1', 'manifest_sha256': pilot.MANIFEST_SHA}
        files = {'generator.py': 'a' * 64, 'frozen-link': None}
        identity = {'revision': pilot.REVISION, 'files_sha256': pilot.GENERATOR_SHA, 'files': files}
        generated = {'source': 'candidate', 'case': {'id': pilot.CASE_IDS[0]},
                     'campaign_binding': dict(binding), 'source_identity': {
                         'revision': pilot.GENERATION_REVISION, 'files_sha256': pilot.GENERATOR_SHA,
                         'files': dict(files)}}
        compiled = {'campaign_binding': dict(binding), 'numerical_build_from_sources': True,
                    'full_numerical_rebuild_this_invocation': True}
        return generated, compiled, identity

    def test_recorded_generation_revision_and_production_anchor_have_distinct_roles(self):
        generated, compiled, identity = self.fixture()
        result = pilot.verify_generation_identity(generated, compiled, identity, pilot.CASE_IDS[0])
        self.assertNotEqual(result['generation_receipt_revision'], result['production_source_revision'])
        self.assertEqual(result['production_source_revision'], pilot.REVISION)
        self.assertEqual(result['generation_receipt_revision'], '584dae7875d20e881a4f9dc27bdb67f5e0fa5512')
        self.assertTrue(result['complete_exporter_inventory_matches'])

    def test_production_anchor_cannot_substitute_for_generation_receipt_revision(self):
        generated, compiled, identity = self.fixture()
        generated['source_identity']['revision'] = pilot.REVISION
        with self.assertRaisesRegex(ValueError, 'Recorded generation revision'):
            pilot.verify_generation_identity(generated, compiled, identity, pilot.CASE_IDS[0])

    def test_generation_revision_cannot_replace_production_anchor(self):
        generated, compiled, identity = self.fixture()
        identity['revision'] = pilot.GENERATION_REVISION
        with self.assertRaisesRegex(ValueError, 'production exporter identity'):
            pilot.verify_generation_identity(generated, compiled, identity, pilot.CASE_IDS[0])

    def test_digest_string_cannot_hide_changed_or_incomplete_exporter_files(self):
        for change in ('modify', 'remove', 'add'):
            with self.subTest(change=change):
                generated, compiled, identity = self.fixture()
                files = generated['source_identity']['files']
                if change == 'modify': files['generator.py'] = 'b' * 64
                elif change == 'remove': del files['frozen-link']
                else: files['unexpected.py'] = 'c' * 64
                with self.assertRaisesRegex(ValueError, 'complete exporter inventory'):
                    pilot.verify_generation_identity(generated, compiled, identity, pilot.CASE_IDS[0])

    def test_both_full_numerical_build_flags_are_required(self):
        for key in ('numerical_build_from_sources', 'full_numerical_rebuild_this_invocation'):
            for wrong in (False, 1, None):
                with self.subTest(key=key, wrong=wrong):
                    generated, compiled, identity = self.fixture(); compiled[key] = wrong
                    with self.assertRaisesRegex(ValueError, 'full numerical source build'):
                        pilot.verify_generation_identity(generated, compiled, identity, pilot.CASE_IDS[0])

    def test_both_campaign_bindings_and_case_source_must_match(self):
        for field in ('generation_campaign', 'compilation_campaign', 'case', 'source'):
            with self.subTest(field=field):
                generated, compiled, identity = self.fixture()
                if field == 'generation_campaign': generated['campaign_binding']['manifest_sha256'] = '0' * 64
                elif field == 'compilation_campaign': compiled['campaign_binding']['id'] = 'other'
                elif field == 'case': generated['case']['id'] = pilot.CASE_IDS[1]
                else: generated['source'] = 'baseline'
                with self.assertRaisesRegex(ValueError, 'case/source/campaign bindings'):
                    pilot.verify_generation_identity(generated, compiled, identity, pilot.CASE_IDS[0])


class ReadonlyPrerequisites(unittest.TestCase):
    @staticmethod
    def fixture(root):
        directory = root / 'SubProcesses/P0_test'; directory.mkdir(parents=True)
        (root / 'Source').mkdir()
        definitions = 'DOTO = %.o\nDOTF = %.f\n'
        (root / 'SubProcesses/MadLoop_makefile_definitions').write_text(definitions)
        (root / 'Source/make_opts').write_text('FFLAGS = -w -fPIC\n')
        (directory / 'makefile').write_text(
            'PROCESS= fixed.o ' + chr(92) + chr(10)
            + ' $(patsubst $(DOTF),$(DOTO),$(wildcard numerical*.f))' + chr(10))
        (directory / 'fixed.o').write_bytes(b'fixed-object')
        (directory / 'numerical1.f').write_bytes(b'      END\n')
        (directory / 'numerical1.o').write_bytes(b'numerical-object')
        (directory / 'unused.f').write_bytes(b'      END\n')
        for name in ('coupl.inc', 'MadLoopParams.inc', 'loop_max_coefs.inc', 'nsqso_born.inc'):
            (directory / name).write_text('C     fixture\n')
        archive = root / 'frozen.a'; archive.write_bytes(b'archive')
        inputs = {'archives': {str(archive): pilot.digest(archive)}, 'metadata': {}}
        compiled = {'link_inputs_before': inputs, 'link_inputs_after': dict(inputs), 'settings': {
            'Source/make_opts': 'FFLAGS = -w -fPIC\n', 'SubProcesses/MadLoop_makefile_definitions': definitions}}
        return directory, compiled

    def test_canonical_process_objects_are_expanded_without_make_or_unused_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, compiled = self.fixture(Path(tmp))
            with mock.patch.object(pilot.subprocess, 'Popen', side_effect=AssertionError('make invoked')):
                result = pilot.process_prerequisites(directory, compiled)
            self.assertEqual(set(result['numerical_objects']), {'fixed.o', 'numerical1.o'})
            self.assertEqual(len(result['driver_includes']), 4)

    def test_missing_object_or_changed_link_input_fails_before_any_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory, compiled = self.fixture(root)
            (directory / 'fixed.o').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing frozen numerical object'):
                pilot.process_prerequisites(directory, compiled)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory, compiled = self.fixture(root)
            (root / 'frozen.a').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed original link input'):
                pilot.process_prerequisites(directory, compiled)

    def test_new_make_expression_requires_explicit_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, compiled = self.fixture(Path(tmp))
            (directory / 'makefile').write_text('PROCESS=$(shell unexpected-command)\n')
            with mock.patch.object(pilot.subprocess, 'Popen', side_effect=AssertionError('shell invoked')):
                with self.assertRaisesRegex(ValueError, 'PROCESS object expression'):
                    pilot.process_prerequisites(directory, compiled)


class FrozenSourceInventory(unittest.TestCase):
    @staticmethod
    def fixture(root):
        files = {}
        for name, target in pilot.FROZEN_DANGLING_LINKS.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)
            files[name] = None
        ordinary = root / 'source/ordinary.f'
        ordinary.parent.mkdir()
        ordinary.write_bytes(b'      END\n')
        link = root / 'source/file_link.f'
        link.symlink_to('ordinary.f')
        files['source/ordinary.f'] = pilot.digest(ordinary)
        files['source/file_link.f'] = pilot.digest(link)
        return files

    def test_exact_two_dangling_links_and_nonnull_files_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = self.fixture(root)
            pilot.verify_source_inventory(root, files)
            self.assertEqual({name for name, value in files.items() if value is None},
                             set(pilot.FROZEN_DANGLING_LINKS))

    def test_missing_nonnull_regular_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = self.fixture(root)
            (root / 'source/ordinary.f').unlink()
            with self.assertRaises(FileNotFoundError):
                pilot.verify_source_inventory(root, files)

    def test_missing_null_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = self.fixture(root)
            (root / next(iter(pilot.FROZEN_DANGLING_LINKS))).unlink()
            with self.assertRaisesRegex(ValueError, 'dangling source link differs'):
                pilot.verify_source_inventory(root, files)

    def test_null_target_materialized_as_regular_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = self.fixture(root)
            link = root / next(iter(pilot.FROZEN_DANGLING_LINKS))
            target = link.resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'unexpected generated target')
            with self.assertRaisesRegex(ValueError, 'target was materialized'):
                pilot.verify_source_inventory(root, files)

    def test_null_target_materialized_as_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = self.fixture(root)
            link = root / next(iter(pilot.FROZEN_DANGLING_LINKS))
            link.resolve().mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'target was materialized'):
                pilot.verify_source_inventory(root, files)

    def test_new_or_missing_null_inventory_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = self.fixture(root)
            for changed in ({**files, 'unregistered-null.f': None},
                            {name: value for name, value in files.items()
                             if name != next(iter(pilot.FROZEN_DANGLING_LINKS))}):
                with self.subTest(nulls={name for name, value in changed.items() if value is None}),\
                        self.assertRaisesRegex(ValueError, 'null source names differ'):
                    pilot.verify_source_inventory(root, changed)

    def test_changed_raw_null_link_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = self.fixture(root)
            link = root / next(iter(pilot.FROZEN_DANGLING_LINKS))
            link.unlink(); link.symlink_to('a-different-dangling-target')
            with self.assertRaisesRegex(ValueError, 'dangling source link differs'):
                pilot.verify_source_inventory(root, files)

    def test_null_symlink_replaced_by_ordinary_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = self.fixture(root)
            name = next(iter(pilot.FROZEN_DANGLING_LINKS)); link = root / name
            link.unlink(); link.write_text(pilot.FROZEN_DANGLING_LINKS[name])
            with self.assertRaisesRegex(ValueError, 'dangling source link differs'):
                pilot.verify_source_inventory(root, files)

    def test_nonnull_content_and_valid_file_symlink_hashing_are_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); files = self.fixture(root)
            alternate = root / 'source/alternate.f'; alternate.write_bytes(b'      END\n')
            link = root / 'source/file_link.f'
            link.unlink(); link.symlink_to('alternate.f')
            pilot.verify_source_inventory(root, files)
            alternate.write_bytes(b'changed contents\n')
            with self.assertRaisesRegex(ValueError, 'Production source differs: source/file_link.f'):
                pilot.verify_source_inventory(root, files)


if __name__ == '__main__':
    unittest.main()
