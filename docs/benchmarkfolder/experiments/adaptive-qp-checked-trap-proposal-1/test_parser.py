#!/usr/bin/env python3
"""Synthetic protocol/qualification controls only; no compiler or native work."""
import copy
from pathlib import Path
import struct
import tempfile
import unittest

import parser


def word(value):
    return struct.pack('>d', value).hex().upper()


def synthetic(*, passes=(4, 5, 4), expected_groups=2, refusal=None,
              early=None, status=323, snapshots=True):
    rows = [parser.FORMAT]

    def array(label, kind, size, value=0.0):
        for index in range(1, size+1):
            words = [word(value)] if kind == 'R8' else ['0000000000000000']*parser.COUNTS[kind]
            rows.append(f'{kind} {label} {index} ' + ' '.join(words))

    def scalar(label, kind, value):
        rows.append(f'{kind} {label} {value} FFFFF')

    for point in (1, 2):
        rows.append(f'BEGIN {point} FFFFF')
        array('INPUT_P', 'R8', 20)
        for label in ('FPE_DP_BEGIN', 'FPE_QP_BEGIN'):
            scalar(label, 'LOGICAL', 'F')
        ordinal = 0
        for mode in passes:
            if snapshots:
                scalar('MODEL_UPDATE', 'INTEGER', 1)
                scalar('GENUINE_QP_INTEGRAND', 'LOGICAL', 'T')
                for label, (kind, size) in parser.MODEL_ARRAYS.items():
                    array(label, kind, size)
            for gid in range(1, expected_groups+1):
                ordinal += 1
                rows.append(f'GROUP {point} {ordinal} {gid} 1 {mode} 1 0 FFFFF')
                example = {'key': [ordinal, gid, 1, mode, 1, 0]}
                specs = parser.group_arrays(example)
                for label, (kind, size) in specs.items():
                    if label in parser.RAW or label == 'GROUP_RESULT':
                        continue
                    if ordinal == early and (label in parser.NATIVE_ARRAYS or
                                               label in ('ORDERED_PL', 'ORDERED_MASSES')):
                        continue
                    array(label, kind, size)
                scalar('ACTUAL_BACKEND', 'INTEGER', 9)
                if ordinal != early:
                    scalar('CHECK_PASS', 'LOGICAL', 'T' if mode == 5 else 'F')
                    scalar('CACHE_ENABLED', 'LOGICAL', 'T')
                    rows.append('NATIVE 9 ' + ('2' if ordinal == refusal else '0') + ' FFFFF')
                    if ordinal == refusal:
                        rows.append('ERROR_BYTES 47 72 61 6D')
                    else:
                        for label in parser.RAW:
                            array(label, 'C32', 1)
                array('GROUP_RESULT', 'C16', 3)
                scalar('GROUP_STABLE', 'LOGICAL', 'F' if ordinal in (refusal, early) else 'T')
        array('RETURN_P', 'R8', 20)
        array('AMPLITUDE', 'R8', 8, 1.0)
        array('ACCURACY', 'R8', 2, 1e-9)
        scalar('PUBLIC_STATUS', 'INTEGER', status)
        for label in ('FPE_DP_END', 'FPE_QP_END'):
            scalar(label, 'LOGICAL', 'F')
        rows.append(f'END {point} {ordinal} {expected_groups if ordinal else 0} FFFFF')
    return seal(rows)


def seal(rows):
    rows = [line for line in rows if not line.startswith('FOOTER ')]
    count = sum(line.split()[0] not in ('FORMAT', 'BEGIN', 'END') for line in rows)
    return '\n'.join(rows + [f'FOOTER {count} 0 0 1000000']) + '\n'


def controls(status=323):
    return {k: {'H': status//100, 'T': status//10 % 10, 'U': status % 10,
                'accuracy': [1e-9, 1e-9]} for k in (1, 2)}


class Parser(unittest.TestCase):
    def parse(self, text, *, partial=False, expected_groups=2):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'synthetic.hex'
            path.write_text(text)
            return parser.parse(path, allow_partial=partial, expected_groups=expected_groups)

    def reject(self, text):
        with self.assertRaises(ValueError):
            self.parse(text)

    def test_repeated_ids_passes_snapshots_are_retained(self):
        capture = self.parse(synthetic())
        self.assertTrue(capture['complete'])
        self.assertEqual([g['key'][0] for g in capture['points'][1]['groups']], list(range(1, 7)))
        self.assertEqual([g['key'][1] for g in capture['points'][1]['groups']], [1, 2]*3)
        self.assertEqual(len(capture['points'][1]['model_snapshots']), 3)
        self.assertEqual(parser.summarize(capture)['points'][0]['pass_modes'], [4, 5, 4])
        qualified = parser.qualify_checked(capture, controls())
        self.assertTrue(qualified['checked_outputs_qualified'])
        self.assertTrue(qualified['qp_coverage_observed'])
        self.assertFalse(qualified['rotation_identity_qualified'])
        self.assertFalse(qualified['independent_physical_accuracy_qualified'])

    def test_physical_default_is_200_not_synthetic_override(self):
        with self.assertRaises(ValueError):
            self.parse(synthetic(), expected_groups=200)

    def test_dp_only_is_valid_output_without_qp_truth(self):
        capture = self.parse(synthetic(passes=(), status=219))
        qualified = parser.qualify_checked(capture, controls(219))
        self.assertTrue(qualified['checked_outputs_qualified'])
        self.assertFalse(qualified['qp_coverage_observed'])
        self.assertEqual(capture['points'][1]['model_snapshots'], [])

    def test_partial_trapped_group_retained_never_qualifies(self):
        text = synthetic().split('C32 RAW_UV1', 1)[0]
        self.reject(text)
        capture = self.parse(text, partial=True)
        self.assertTrue(capture['partial'])
        self.assertEqual(len(capture['points'][1]['groups']), 1)
        self.assertFalse(parser.qualify_checked(capture, controls())['checked_outputs_qualified'])
        fragment = self.parse(text+'C32 RAW_U', partial=True)
        self.assertTrue(fragment['trailing_fragment'])

    def test_partial_cannot_hide_closed_group_damage(self):
        text = synthetic().replace('C16 ARG_M1 1 0000000000000000 0000000000000000\n', '', 1)
        with self.assertRaises(ValueError):
            self.parse(text.split('FOOTER', 1)[0], partial=True)

    def test_refusal_and_early_return_are_explicit(self):
        for kwargs in ({'refusal': 1}, {'early': 1}):
            capture = self.parse(synthetic(**kwargs))
            summary = parser.summarize(capture)['points'][0]
            self.assertEqual(summary['native_refusals'], int('refusal' in kwargs))
            self.assertEqual(summary['early_adapter_returns'], int('early' in kwargs))
            # Parsing a refusal is not native QP truth; final driver output is separate.
            self.assertTrue(parser.qualify_checked(capture, controls())['checked_outputs_qualified'])

    def test_unknown_labels_and_scalar_kind_duplicates_rejected(self):
        original = synthetic()
        for old, new in [('MODEL_MDL_MT', 'MODEL_UNKNOWN'), ('ARG_M1', 'ARG_M2'),
                         ('R8 INPUT_P', 'C16 INPUT_P'), ('INTEGER ACTUAL_BACKEND', 'LOGICAL ACTUAL_BACKEND'),
                         ('MODEL_UPDATE 1', 'MODEL_UPDATE 2'), ('GENUINE_QP_INTEGRAND T', 'GENUINE_QP_INTEGRAND F')]:
            with self.subTest(old=old):
                self.reject(seal(original.replace(old, new, 1).splitlines()))
        duplicate = original.replace('INTEGER PUBLIC_STATUS 323 FFFFF',
                                     'INTEGER PUBLIC_STATUS 323 FFFFF\nINTEGER PUBLIC_STATUS 323 FFFFF', 1)
        self.reject(seal(duplicate.splitlines()))

    def test_native_store_policy_status_and_full_raw_shapes(self):
        original = synthetic()
        cases = [('CHECK_PASS F', 'CHECK_PASS T'), ('CACHE_ENABLED T', 'CACHE_ENABLED F'),
                 ('NATIVE 9 0', 'NATIVE 8 0'), ('NATIVE 9 0', 'NATIVE 9 2'),
                 ('C32 RAW_FINITE 1', 'C32 RAW_FINITE 2'),
                 ('INTEGER ACTUAL_BACKEND 9', 'INTEGER ACTUAL_BACKEND 8')]
        for old, new in cases:
            with self.subTest(old=old):
                self.reject(seal(original.replace(old, new, 1).splitlines()))
        self.reject(seal([line for line in original.splitlines() if 'RAW_FINITE' not in line]))

    def test_bad_shape_indices_group_sequence_and_mode_switch(self):
        original = synthetic()
        for old, new in [('GROUP 1 2 2 1 4 1 0', 'GROUP 1 3 2 1 4 1 0'),
                         ('GROUP 1 2 2 1 4 1 0', 'GROUP 1 2 1 1 4 1 0'),
                         ('GROUP 1 2 2 1 4 1 0', 'GROUP 1 2 2 1 5 1 0'),
                         ('GROUP 1 1 1 1 4 1 0', 'GROUP 1 1 1 1 4 6 0'),
                         ('R8 INPUT_P 2 ', 'R8 INPUT_P 1 '),
                         ('R16 ENTRY_PL 1 ', 'R16 ENTRY_PL 0 ')]:
            with self.subTest(old=old):
                self.reject(seal(original.replace(old, new, 1).splitlines()))

    def test_completed_partial_traversal_rejected(self):
        text = synthetic()
        start = text.index('GROUP 1 2 2 1 4')
        end = text.index('GROUP 1 3 1 1 5')
        text = text[:start] + text[end:]
        self.reject(seal(text.splitlines()))

    def test_footers_flags_counts_endings_and_extra_records(self):
        original = synthetic()
        for text in [original.replace(' 0 0 1000000', ' 1 0 1000000'),
                     original.replace(' 0 0 1000000', ' 0 1 1000000'),
                     original.replace(' 0 0 1000000', ' 0 0 999999'),
                     original.replace('END 1 6 2', 'END 1 5 2'),
                     original+'BEGIN 3 FFFFF\n',
                     original.replace('BEGIN 1 FFFFF', 'BEGIN 1 FTFF'),
                     original.replace('BEGIN 1 FFFFF', 'BEGIN 1 FFFFF\nOBSERVER_FLAG_CHANGE 1'),
                     original.replace('BEGIN 2 FFFFF', 'BEGIN 3 FFFFF')]:
            with self.subTest(text=text[-80:]):
                self.reject(text)

    def test_snapshot_required_but_not_arbitrary_model_prefix(self):
        self.reject(synthetic(snapshots=False))
        original = synthetic()
        self.reject(seal(original.replace('INTEGER MODEL_UPDATE 1 FFFFF\n', '', 1).splitlines()))

    def test_checked_outputs_reject_unchecked_wrong_backend_or_omissions(self):
        capture = self.parse(synthetic())
        self.assertFalse(parser.qualify_checked(capture, controls(), omissions=1)['checked_outputs_qualified'])
        for code in (120, 219, 320, 329, 423):
            changed = self.parse(synthetic(status=code))
            self.assertFalse(parser.qualify_checked(changed, controls(code))['checked_outputs_qualified'])
        self.assertFalse(parser.qualify_checked(capture, controls(219))['checked_outputs_qualified'])

    def test_accuracy_nonfinite_unknown_threshold_and_driver_mismatch(self):
        original = synthetic()
        for bad in (float('inf'), float('nan'), -1.0, 1e-6):
            changed = self.parse(original.replace(word(1e-9), word(bad)))
            self.assertFalse(parser.qualify_checked(changed, controls())['checked_outputs_qualified'])
        wrong = controls(); wrong[1]['accuracy'][0] = 2e-9
        self.assertFalse(parser.qualify_checked(self.parse(original), wrong)['checked_outputs_qualified'])

    def test_qp_success_requires_three_observed_complete_passes(self):
        for passes in ((4,), (4, 5), (4, 4, 5)):
            capture = self.parse(synthetic(passes=passes))
            self.assertFalse(parser.qualify_checked(capture, controls())['checked_outputs_qualified'])

    def test_inputs_born_compare_detects_perturbation_and_preserves_outputs(self):
        capture = self.parse(synthetic())
        self.assertTrue(parser.compare_inputs_and_born(capture, capture)['inputs_and_born_exact'])
        changed = copy.deepcopy(capture)
        changed['points'][1]['data']['AMPLITUDE'][1][-1] = word(2.)
        self.assertTrue(parser.compare_inputs_and_born(capture, changed)['inputs_and_born_exact'])
        changed['points'][1]['data']['AMPLITUDE'][0][-1] = word(2.)
        self.assertFalse(parser.compare_inputs_and_born(capture, changed)['inputs_and_born_exact'])
        changed = copy.deepcopy(capture)
        changed['points'][1]['groups'][2]['data']['NATIVE_SCALE'][0][-1] = '0000000000000001'
        self.assertFalse(parser.compare_inputs_and_born(capture, changed)['inputs_and_born_exact'])

    def test_flag_only_changes_remain_observations_not_inputs(self):
        capture = self.parse(synthetic())
        changed = copy.deepcopy(capture)
        changed['points'][1]['model_snapshots'][0]['data']['MODEL_UPDATE'][0][-1] = 'TFFFF'
        changed['points'][1]['model_snapshots'][0]['data']['GENUINE_QP_INTEGRAND'][0][-1] = 'FTFFF'
        changed['points'][1]['groups'][0]['data']['ACTUAL_BACKEND'][0][-1] = 'FFTFF'
        changed['points'][1]['groups'][0]['native'][0]['flags'] = 'FFFTF'
        changed['points'][1]['begin_flags'] = 'FFFFT'
        changed['points'][1]['data']['FPE_QP_END'][0][1] = 'T'
        comparison = parser.compare_inputs_and_born(capture, changed)
        self.assertTrue(comparison['inputs_and_born_exact'])
        self.assertTrue(comparison['pass_sequence_comparable'])
        self.assertFalse(comparison['event_flags_equal'])
        self.assertEqual(comparison['sequence_differences'], [])
        self.assertNotEqual(comparison['event_flags']['left'], comparison['event_flags']['right'])
        self.assertNotEqual(comparison['software_reduction_flags']['left'],
                            comparison['software_reduction_flags']['right'])

    def test_pass_count_is_coverage_gap_and_shared_model_bits_still_checked(self):
        short = self.parse(synthetic(passes=(4,)))
        full = self.parse(synthetic())
        comparison = parser.compare_inputs_and_born(short, full)
        self.assertTrue(comparison['inputs_and_born_exact'])
        self.assertFalse(comparison['pass_sequence_comparable'])
        self.assertFalse(comparison['all_input_records_comparable'])
        self.assertTrue(comparison['sequence_differences'])
        altered = copy.deepcopy(full)
        altered['points'][1]['model_snapshots'][0]['data']['MODEL_MDL_MT'][0][-1] = word(2.)
        comparison = parser.compare_inputs_and_born(short, altered)
        self.assertFalse(comparison['inputs_and_born_exact'])
        self.assertTrue(any(x['scope'] == 'model_input' for x in comparison['failures']))

    def test_wrong_flag_word_width_rejected(self):
        original = synthetic()
        for bad in ('FFFF', 'FFFFFF'):
            self.reject(original.replace('BEGIN 1 FFFFF', 'BEGIN 1 ' + bad, 1))

    def test_ieee_exact_values(self):
        self.assertEqual(parser.finite_value([word(1.)]), 1)
        self.assertEqual(parser.finite_value(['0000000000000001']), parser.Fraction(2)**-1074)
        self.assertEqual(parser.finite_value(['0000000000000000', '3FFF000000000000'], 128), 1)
        self.assertIsNone(parser.finite_value([word(float('inf'))]))


if __name__ == '__main__':
    unittest.main()
