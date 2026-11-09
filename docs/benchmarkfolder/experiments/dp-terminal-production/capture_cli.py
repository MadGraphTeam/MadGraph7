#!/usr/bin/env python3
"""Run the actual CLI with read-only before/after finalizer observation."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from madgraph.loop.loop_exporters import LoopProcessOptimizedExporterFortranSA as Exporter
from madgraph.loop import dp_terminal_proof as proof

record = Path(sys.argv[1])
arguments = sys.argv[2:]
original = Exporter.finalize_dp_terminal_updates
observations = []
verdicts = []
original_validator = proof.validate_dp_terminal_wavefunctions


def validate(*args, **kwargs):
    result = original_validator(*args, **kwargs)
    reason = None
    if not result:
        # The public validator intentionally hides parser diagnostics. Read
        # the same proposal again only to retain its refusal explanation.
        try:
            options = {'source_manifest': None, 'include_roots': ()}
            options.update(kwargs)
            proof._proof(*args, **options)
        except Exception as error:
            reason = type(error).__name__+': '+str(error)
    verdicts.append({'directory': str(args[0]), 'accepted': result, 'refusal': reason})
    return result


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(directory).glob('coef_construction_*.f'))}


def capture(self):
    pending = dict(getattr(self, '_dp_terminal_pending', {}))
    before = {directory: hashes(directory) for directory in pending}
    result = original(self)
    for directory, (prefix, expected) in sorted(pending.items()):
        source = ''.join(p.read_text() for p in sorted(Path(directory).glob('coef_construction_*.f')))
        count = source.upper().count('CALL '+prefix.upper()+'DP_TERMINAL_WL_1_1(')
        observations.append({'directory': directory, 'prefix': prefix,
            'expected': expected, 'proposed': len(expected), 'applied': count,
            'declined': len(expected)-count, 'before_sha256': before[directory],
            'after_sha256': hashes(directory)})
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(observations, indent=2, sort_keys=True)+'\n')
    return result


Exporter.finalize_dp_terminal_updates = capture
proof.validate_dp_terminal_wavefunctions = validate
sys.argv = [str(ROOT/'bin/madgraph'), *arguments]
try:
    runpy.run_path(str(ROOT/'bin/madgraph'), run_name='__main__')
finally:
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(observations, indent=2, sort_keys=True)+'\n')
    record.with_name('proof-verdicts.json').write_text(json.dumps(verdicts, indent=2, sort_keys=True)+'\n')
