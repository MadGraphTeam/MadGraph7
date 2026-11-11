"""Pure report joins against admitted replay records; no fabricated timings."""
import csv
import io
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'prepare_timing_report.py'
owner=types.ModuleType('timing_report_prepare');owner.__file__=str(SOURCE)
exec(compile(SOURCE.read_bytes(),str(SOURCE),'exec'),owner.__dict__)


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((HERE/'timing-report-data.json').read_bytes())
        cls.replays=json.loads((HERE/'timing-replay.json').read_bytes())['replays']

    def test_unadmitted_fourth_capsule_fails_before_any_replay(self):
        with patch.object(owner,'SELECTIONS',((owner.FIRST,owner.FIRST_SHA,owner.ORDER[:3]),
                                             (owner.FOURTH,'PENDING',owner.ORDER[3:]))), \
             patch.object(owner,'selected_replay',side_effect=AssertionError('No partial preparation allowed')):
            with self.assertRaisesRegex(ValueError,'closed and pinned'):
                owner.prepare(Path('/unused'))

    def test_envelope_keeps_admitted_analyzer_summaries_exact(self):
        self.assertEqual([c['summary'] for c in self.data['cases']],
                         [c['summary'] for r in self.replays for c in r['cases']])

    def test_csv_contains_every_admitted_lane_and_exact_unrounded_statistics(self):
        rows=list(csv.DictReader(io.StringIO(owner.csv_bytes(self.data).decode())))
        self.assertEqual(len(rows),20)
        indexed={(row['case_id'],int(row['backend_id'])):row for row in rows}
        for case in self.data['cases']:
            for backend in owner.IDS:
                row=indexed[case['case_id'],backend];lane=case['summary']['backends'][str(backend)]
                self.assertEqual(float(row['median_ms_per_point']),lane['median_seconds_per_point']*1000)
                self.assertEqual(float(row['unscaled_MAD_ms_per_point']),lane['mad_seconds_per_point']*1000)
                self.assertEqual(float(row['median_within_cycle_ratio_to_8']),lane['median_ratio_to_8'])
                self.assertEqual(float(row['READY_seconds']),case['summary']['initialization'][str(backend)]['cold_seconds'])
                self.assertEqual(int(row['accepted_DP_output_calls'])+int(row['accepted_QP_output_calls']),
                                 sum(sample['evaluations'] for sample in lane['samples']))
                self.assertEqual(row['source_packet_index_sha256'],case['source_packet']['index_sha256'])


if __name__=='__main__':unittest.main()
