"""Inclusive hadronic MLM samples with several jet multiplicities and flavors.

Run with test_manager.py test_mg7_mlm_multijet -pA -t0. Requires source-built
MadSpace and NNPDF23_lo_as_0130_qed. The test-only UMAMI proxy checks each real
matrix-element call against the diagram supplied by MLM and the exported color
mask. No shower is run: this tests the hard-event clustering/color handoff.
"""

import collections
import glob
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import madgraph.interface.master_interface as MGCmd
import madgraph.various.misc as misc
from madgraph import MG5DIR
from madgraph.various.banner import RunCardMG7

_PDF = 'NNPDF23_lo_as_0130_qed'
_HISTORIES = ('all_diagrams', 'diagram', 'madevent')
_EVENTS = 200


def _launch_with_probe():
    """Child entry point, invoked in a generated process directory."""
    from madgraph.mg7.madgraph_process import MadgraphProcess
    from madgraph.mg7.launch import main
    original = MadgraphProcess.load_matrix_element
    template = (Path(MG5DIR) / 'tests/input_files/mlm_diagram_probe.cpp').read_text()

    def load(process, path_format, card, parameters):
        index, meta = next((i, m) for i, m in enumerate(process.subprocess_data)
                           if m['me_path'] == path_format)
        assert process.backends == ['scalar']
        target = Path(path_format.format(device='scalar', fptype=process.fptype)).resolve()
        directory = Path('mlm_probes')
        directory.mkdir(exist_ok=True)
        proxy = (directory / ('probe_%d.so' % index)).resolve()
        if not proxy.exists():
            allowed = [set() for _ in range(meta['diagram_count'])]
            for channel in meta['channels']:
                for diagram in channel['diagrams']:
                    allowed[diagram['diagram']].update(diagram['active_colors'])
            ncolors = len(meta['color_flows'])
            mask = '{' + ','.join('{' + ','.join(
                'true' if c in row else 'false' for c in range(ncolors)) + '}'
                for row in allowed) + '}'
            source = template
            for key, value in {'SUBPROCESS': str(index), 'NDIAGRAMS': str(len(allowed)),
                               'NCOLORS': str(ncolors), 'ALLOWED': mask,
                               'LIBRARY': json.dumps(str(target))}.items():
                source = source.replace('@%s@' % key, value)
            cpp = proxy.with_suffix('.cpp')
            cpp.write_text(source)
            command = ['c++', '-shared', '-fPIC', '-std=c++17', '-O2',
                       '-I' + str(Path(MG5DIR) / 'madspace/include'), str(cpp), '-o', str(proxy)]
            if sys.platform != 'darwin':
                command.append('-ldl')
            subprocess.run(command, check=True)
        return original(process, str(proxy), card, parameters)

    MadgraphProcess.load_matrix_element = load
    main()


class MG7MLMMultijetTest(unittest.TestCase):
    def missing_dependency(self, reason):
        if os.environ.get('MG7_MLM_REQUIRE_DEPS') == '1':
            self.fail(reason)
        self.skipTest(reason)

    def setUp(self):
        from madgraph.mg7 import bootstrap
        if not bootstrap.madspace_is_installed():
            self.missing_dependency('source-built madspace unavailable')
        pdf = misc.resolve_lhapdf().find_set(_PDF)
        if not pdf:
            self.missing_dependency('%s LHAPDF data not found' % _PDF)
        self.datadir = str(Path(pdf))
        self.path = Path(tempfile.mkdtemp(prefix='mg7_mlm_multijet_'))
        if not os.environ.get('MG7_MLM_KEEP_OUTPUT'):
            self.addCleanup(shutil.rmtree, str(self.path))
        print('MLM inclusive artifacts: %s' % self.path, flush=True)

    def test_mlm_inclusive_ttbar_mg7(self):
        self.check_sample('ttbar', ['p p > t t~ QED=0', 'p p > t t~ j QED=0'], 2, {0, 1})

    def test_mlm_inclusive_zjets_mg7(self):
        self.check_sample('zjets', ['p p > z', 'p p > z j QED=1', 'p p > z j j QED=1'], 1, {0, 1, 2})

    def check_sample(self, name, processes, core_particles, jet_counts):
        import numpy as np
        run_dir = self.path / name
        mg = MGCmd.MasterCmd()
        mg.no_notification()
        mg.exec_cmd('set automatic_html_opening False --no_save')
        for i, process in enumerate(processes):
            mg.exec_cmd(('%s %s' % ('generate' if i == 0 else 'add process', process)))
        mg.exec_cmd('output mg7 %s -f' % run_dir)
        with (run_dir / 'SubProcesses/subprocesses.json').open() as stream:
            metadata = json.load(stream)
        self.assertEqual({len(m['outgoing']) - core_particles for m in metadata}, jet_counts)
        self.assertGreater(len(metadata), len(jet_counts))
        self.assertTrue(any(len(d['active_colors']) < len(m['color_flows'])
                            for m in metadata for c in m['channels'] for d in c['diagrams']))
        settings = {
            'run.seed': 445, 'run.cpu_mode': 'scalar', 'run.madspace_cpu_mode': 'scalar',
            'run.precision': 'all64', 'run.cpu_thread_pool_size': 1,
            'run.combine_thread_pool_size': 1, 'run.output_format': 'compact_npy',
            'run.weighted_histograms': False, 'run.postprocessing_histograms': False,
            'run.make_plots': False, 'beam.pdf': _PDF, 'beam.dynamical_scale_choice': 'mlm',
            'beam.fixed_ren_scale': False, 'beam.fixed_fact_scale': False,
            'phasespace.xqcut': 30., 'generation.events': _EVENTS,
            'generation.cpu_batch_size': 256, 'generation.survey_min_iters': 1,
            'generation.survey_max_iters': 1, 'vegas.start_batch_size': 256,
            'vegas.max_batch_size': 256, 'madnis.enable': False,
            'systematics.enable': False, 'postprocessing.systematics': False,
        }
        summaries = []
        union_events = None
        for history in _HISTORIES:
            with self.subTest(sample=name, history=history):
                card_path = run_dir / 'Cards/run_card.toml'
                card = RunCardMG7(str(card_path))
                for key, value in dict(settings, **{'run.run_name': history,
                                                   'beam.clustering_history': history}).items():
                    card.set(key, value, user=True)
                card.write(str(card_path))
                probe = run_dir / ('probe_%s.jsonl' % history)
                if probe.exists():
                    probe.unlink()
                log = run_dir / ('launch_%s.log' % history)
                env = dict(os.environ, LHAPDF_DATA_PATH=self.datadir, MG7_MLM_PROBE_LOG=str(probe))
                env['PYTHONPATH'] = MG5DIR + os.pathsep + env.get('PYTHONPATH', '')
                command = [sys.executable, '-c',
                    'from tests.acceptance_tests.test_mg7_mlm_multijet import _launch_with_probe; '
                    '_launch_with_probe()', '-f']
                with log.open('w') as stream:
                    result = subprocess.run(command, cwd=run_dir, env=env,
                                            stdout=stream, stderr=subprocess.STDOUT, timeout=600)
                tail = log.read_text(errors='replace')[-6000:]
                self.assertEqual(result.returncode, 0, tail)
                infos = sorted((run_dir / 'Events').glob(history + '_*/info.json'))
                self.assertTrue(infos, tail)
                info = json.loads(infos[-1].read_text())
                self.assertEqual(info['status'], 'done', tail)
                self.assertTrue(info['process']['done'], tail)
                self.assertEqual(info['process']['count_unweighted'], _EVENTS, tail)
                self.assertTrue(math.isfinite(info['process']['mean']))
                self.assertTrue(math.isfinite(info['process']['error']))
                self.assertGreater(info['process']['mean'], 0.)
                self.assertEqual({c['subprocess'] for c in info['channels']}, set(range(len(metadata))))
                self.assertTrue(probe.exists(), tail)
                records = [json.loads(line) for line in probe.read_text().splitlines()]
                counts = collections.defaultdict(collections.Counter)
                for record in records:
                    index = record.pop('subprocess')
                    counts[index].update(record)
                self.assertEqual(set(counts), set(range(len(metadata))))
                for index, values in counts.items():
                    self.assertEqual(values['errors'], 0, (history, index, values))
                    self.assertGreater(values['colored'], 0, (history, index, values))
                    self.assertGreater(values['sampled'], 0, (history, index, values))
                    if history == 'all_diagrams':
                        self.assertEqual(values['forced'], 0)
                self.assertTrue(any(values['diagrams'] > 1 for values in counts.values()))
                events = np.load(infos[-1].parent / 'events.npy')
                self.assertEqual(len(events), _EVENTS)
                self.assertTrue(np.isfinite(events['weight']).all())
                for event in events:
                    meta = metadata[event['subprocess_index']]
                    mask = {d['diagram']: d['active_colors']
                            for c in meta['channels'] for d in c['diagrams']}
                    self.assertIn(event['color_index'], mask[event['diagram_index']])
                if history == 'all_diagrams':
                    union_events = (infos[-1].parent / 'events.npy').read_bytes()
                summary = {'sample': name, 'history': history, 'subprocesses': len(metadata),
                           'jet_counts': sorted(jet_counts), 'events': len(events),
                           'cross_section': info['process']['mean'], 'error': info['process']['error'],
                           'forced_events': sum(c['forced'] for c in counts.values()),
                           'subprocess_counts': dict(counts)}
                summaries.append(summary)
                print(json.dumps({k: v for k, v in summary.items()
                                  if k != 'subprocess_counts'}), flush=True)
        # The observer must preserve the production run, including its random
        # choices and weights. Repeat the union history through the ordinary
        # launcher, with the proxy disabled and the same seed/cards.
        card = RunCardMG7(str(card_path))
        card.set('run.run_name', 'unobserved', user=True)
        card.set('beam.clustering_history', 'all_diagrams', user=True)
        card.write(str(card_path))
        log = run_dir / 'launch_unobserved.log'
        with log.open('w') as stream:
            result = subprocess.run([sys.executable, str(run_dir / 'bin/generate_events'), '-f'],
                                    cwd=run_dir, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=600)
        tail = log.read_text(errors='replace')[-6000:]
        self.assertEqual(result.returncode, 0, tail)
        infos = sorted((run_dir / 'Events').glob('unobserved_*/info.json'))
        self.assertTrue(infos, tail)
        self.assertTrue(json.loads(infos[-1].read_text())['process']['done'], tail)
        self.assertEqual((infos[-1].parent / 'events.npy').read_bytes(), union_events,
                         'the probe changed the seeded production events')
        (run_dir / 'mlm_validation.json').write_text(json.dumps(summaries, indent=2) + '\n')
