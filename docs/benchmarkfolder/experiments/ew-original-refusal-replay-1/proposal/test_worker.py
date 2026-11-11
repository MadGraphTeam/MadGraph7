"""Nine pure replay-worker regressions; Popen/execv are always replaced.

The fake executable, maps and zero input words below test orchestration only.
They are not numerical fixtures, runtime qualifications or native executions.
"""
import contextlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import limit_exec as limiter
import worker


WORDS = ['0000000000000000'] * 41
REQUEST = {'id': 'ew-point2-mode1-group5-stock', 'point': 2,
           'mode': 1, 'variant': 'stock'}


def transcript(status=1, changed=False):
    post = list(WORDS)
    if changed:
        post[0] = '0000000000000001'
    return ['INPUT_ID 2 1 4 3 41',
            'OPTIONS pole=0 finite=1 check=0 cache=0 budget=268435456',
            'INPUT_WORDS ' + ' '.join(WORDS), 'READY KPROJ_V1',
            f'FORTRAN_STATUS {status}', 'FORTRAN_EVALUATION_CALLS 1',
            'CONTEXT_GENERATION 1', 'ERROR_BUFFER_FULL 0', 'ERROR_BEGIN',
            'original error retained', 'ERROR_END',
            'POST_INPUT_WORDS ' + ' '.join(post), 'INPUTS_UNCHANGED 1',
            'COEFFICIENT_BUFFERS_NOT_INSPECTED', 'DONE KPROJ_V1',
            'FINALIZE_STATUS 0', 'CLOSED KPROJ_V1']


class WorkerTests(unittest.TestCase):
    def test_status_zero_through_three_are_diagnostic_outcomes(self):
        for status in range(4):
            protocol = worker.Protocol(REQUEST, WORDS)
            events = [event for line in transcript(status)
                      if (event := protocol.line(line))]
            self.assertEqual(events, ['READY', 'DONE', 'CLOSED'])
            self.assertEqual(protocol.error_lines, ['original error retained'])

    def test_bad_pre_echo_or_options_prevents_ready(self):
        for key, replacement in [('INPUT_WORDS', 'INPUT_WORDS BAD'),
                                 ('OPTIONS', 'OPTIONS cache=1')]:
            protocol = worker.Protocol(REQUEST, WORDS)
            with self.assertRaises(ValueError):
                for line in transcript():
                    protocol.line(replacement if line.startswith(key + ' ') else line)
            self.assertEqual(protocol.events, [])

    def test_post_echo_call_count_and_context_fail_before_exit(self):
        mutations = [('POST_INPUT_WORDS', 'POST_INPUT_WORDS BAD'),
                     ('FORTRAN_EVALUATION_CALLS', 'FORTRAN_EVALUATION_CALLS 2'),
                     ('CONTEXT_GENERATION', 'CONTEXT_GENERATION 2'),
                     ('INPUTS_UNCHANGED', 'INPUTS_UNCHANGED 0'),
                     ('ERROR_BUFFER_FULL', 'ERROR_BUFFER_FULL 1'),
                     ('FORTRAN_STATUS', 'FORTRAN_STATUS BROKEN')]
        for key, replacement in mutations:
            protocol = worker.Protocol(REQUEST, WORDS)
            with self.assertRaises(ValueError):
                for line in transcript():
                    protocol.line(replacement if line.startswith(key + ' ') else line)
            self.assertEqual(protocol.events, ['READY'])

    def test_duplicates_and_error_framing(self):
        protocol = worker.Protocol(REQUEST, WORDS)
        for line in transcript()[:4]:
            protocol.line(line)
        with self.assertRaises(ValueError):
            protocol.line('READY KPROJ_V1')
        with self.assertRaises(ValueError):
            protocol.line('ERROR_END')

    def test_limiter_is_fixed_and_mocked(self):
        with patch.object(limiter.sys, 'argv', ['limit', '/absolute/executable', '--argument']), \
             patch.object(limiter.resource, 'setrlimit') as caps, \
             patch.object(limiter.os, 'execv') as execute:
            limiter.main()
            self.assertEqual(caps.call_args_list[0].args,
                             (limiter.resource.RLIMIT_CORE, (0, 0)))
            self.assertEqual(caps.call_args_list[1].args,
                             (limiter.resource.RLIMIT_AS, (4294967296, 4294967296)))
            self.assertEqual(caps.call_args_list[2].args,
                             (limiter.resource.RLIMIT_FSIZE, (16777216, 16777216)))
            execute.assert_called_once_with('/absolute/executable',
                                            ['/absolute/executable', '--argument'])

    def simulated(self, changed=False, status='1', cancelled=False):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            here, executor = base / 'proposal', base / 'executor'
            here.mkdir()
            executable = executor / 'work/build/replay'
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b'FAKE-NOT-EXECUTABLE')
            cwd = executor / 'work/workers' / REQUEST['id']
            cwd.mkdir(parents=True)
            evidence = executor / 'results/runs' / REQUEST['id']
            evidence.mkdir(parents=True)
            source = cwd / 'input.words'
            source.write_text('KPROJ_WORDS_V1\n2 1 4 3 41\n' + '\n'.join(WORDS) + '\n')
            request = {**REQUEST, 'input': {'path': 'unused', **worker.record(source)}}
            (here / 'input-certificate.json').write_text(json.dumps({'requests': [request]}))
            specification = {
                'schema': worker.SCHEMA, 'proposal_index_sha256': 'mocked',
                'request': request, 'cwd': str(cwd), 'evidence': str(evidence),
                'executable': str(executable), 'executable_pin': worker.record(executable),
                'input_file': str(source), 'input_pin': worker.record(source),
                'expected_runtime': {str(executable): worker.record(executable)['sha256']}}
            specpath = evidence / 'specification.json'
            specpath.write_text(json.dumps(specification))
            raw_stdout = ('\n'.join(transcript(status, changed)) + '\n').encode()
            chunks, sent = [raw_stdout, b''], []

            class Stream:
                def fileno(self): return 42
                def close(self): pass
                def write(self, value): sent.append(value); return len(value)
                def flush(self): pass

            class Process:
                pid = 777777777
                returncode = None
                stdin, stdout = Stream(), Stream()
                def poll(self): return self.returncode
                def wait(self, timeout=None): self.returncode = 0; return 0
                def kill(self): self.returncode = -9

            process = Process()

            class Owner:
                def __init__(self, process): self.process = process
                def sample(self): pass
                def stop(self, **kwargs):
                    return {'survivors': [], 'observed_owned_processes': [], 'events': []}

            class Signals(contextlib.AbstractContextManager):
                def __enter__(self): return self
                def __exit__(self, *args): return None
                def check(self):
                    if cancelled:
                        raise InterruptedError('parent cancellation before launch')

            class Poller(contextlib.AbstractContextManager):
                def __enter__(self): return self
                def __exit__(self, *args): return None
                def register(self, *args): pass
                def unregister(self, *args): pass
                def select(self, timeout): return [(NS(fd=42), 1)]

            def read(fd, count):
                chunk = chunks.pop(0) if chunks else b''
                if not chunk:
                    process.returncode = 0
                return chunk

            original_read_text = Path.read_text

            def read_text(path, *args, **kwargs):
                if str(path) == '/proc/777777777/maps':
                    return 'fake maps for mocked parser\n'
                return original_read_text(path, *args, **kwargs)

            helpers = (NS(SignalRequests=Signals, OwnedChild=Owner),
                       lambda raw, expected: {'elf': expected, 'data': {}})
            with patch.object(worker, 'HERE', here), \
                 patch.object(worker, 'EXECUTOR', executor), \
                 patch.object(worker.Path, 'cwd', return_value=cwd), \
                 patch.object(worker, 'admitted_helpers', return_value=helpers), \
                 patch.object(worker.subprocess, 'Popen', return_value=process) as launch, \
                 patch.object(worker.selectors, 'DefaultSelector', Poller), \
                 patch.object(worker.os, 'set_blocking'), \
                 patch.object(worker.os, 'read', read), \
                 patch.object(Path, 'read_text', read_text):
                if changed or status == 'BROKEN' or cancelled:
                    with self.assertRaisesRegex(ValueError, 'evidence retained'):
                        worker.execute(specpath, worker.record(specpath)['sha256'])
                else:
                    worker.execute(specpath, worker.record(specpath)['sha256'])
                self.assertEqual(launch.call_count, 0 if cancelled else 1)
            result = json.loads((evidence / 'result.json').read_bytes())
            if cancelled:
                self.assertIsNone(result['native_pid'])
                self.assertFalse(result['orchestration_complete'])
                self.assertIn('parent cancellation before launch', result['failure'])
                self.assertEqual(sent, [])
                self.assertFalse((evidence / 'stdout.txt').exists())
                return
            self.assertEqual((evidence / 'stdout.txt').read_bytes(), raw_stdout)
            self.assertEqual(result['orchestration_complete'], not (changed or status == 'BROKEN'))
            self.assertEqual(result['cleanup']['survivors'], [])
            self.assertEqual(result['fortran_status'], None if status == 'BROKEN' else int(status))
            self.assertEqual(sent, [b'EVALUATE\n'] if changed or status == 'BROKEN'
                             else [b'EVALUATE\n', b'EXIT\n'])

    def test_mocked_full_worker_status1_and_maps(self):
        self.simulated()

    def test_mocked_changed_post_words_retains_failure_and_raw_stdout(self):
        self.simulated(changed=True)

    def test_mocked_malformed_status_retains_failure(self):
        self.simulated(status='BROKEN')

    def test_cancellation_before_launch_retains_failure_without_popen(self):
        self.simulated(cancelled=True)


if __name__ == '__main__':
    unittest.main()
