"""Source-only geometry-window admission; no numerical/scientific dispatch.

This joins a freshly computed prefix/suffix contract to the exact generated
sources, same-kind DAG receipts and route data. It never substitutes endpoint
FENV equality for the source proof, nor labels the complete producer pure.
The hashes are reproducibility/stale-receipt checks, not an adversarial proof.
"""
import copy
import hashlib
import json
from pathlib import PurePosixPath

from madgraph.loop.invariant_window import momentum_window


SCHEMA = 'generated-geometry-window-source-contract-v1'
WIDTHS = ('64', '128')
BOUND_FIELDS = ('graph', 'routes', 'source_proof', 'routing_contract',
                'protected', 'conservation', 'source_bindings',
                'momentum_environment_contract')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf8')).hexdigest()


def source_bindings(raw):
    result = {}
    for name, data in sorted(raw.items()):
        path = PurePosixPath(name)
        if not isinstance(name, str) or path.is_absolute() or '..' in path.parts:
            raise ValueError('nonrelative generated source name')
        if isinstance(data, str):
            data = data.encode('utf8')
        if not isinstance(data, bytes):
            raise ValueError('source bytes required')
        result[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    if not result:
        raise ValueError('empty generated source closure')
    return result


def _batch_bindings(raw, batches):
    result = []
    seen = set()
    for filename, text in batches:
        matches = [name for name in raw
                   if name == filename or name.endswith('/' + filename)]
        if len(matches) != 1 or matches[0] in seen:
            raise ValueError('ambiguous/repeated producer batch')
        name = matches[0]
        data = raw[name]
        if isinstance(data, bytes):
            data = data.decode('utf8')
        if data != text:
            raise ValueError('producer batch not joined to source bytes')
        seen.add(name)
        result.append(dict(file=name, bytes=len(text.encode('utf8')),
                           sha256=hashlib.sha256(text.encode('utf8')).hexdigest()))
    if not result:
        raise ValueError('empty producer batches')
    return result


def _source_inputs(d):
    return {name: copy.deepcopy(d.get(name)) for name in BOUND_FIELDS}


def _window_supported(window):
    """Full-producer I/O limitations may remain after momentum is complete."""
    return (window.get('schema') == 'source-momentum-environment-window-v1'
            and window.get('complete') is True
            and window.get('prefix', {}).get('complete') is True
            and window.get('suffix', {}).get('reference_complete') is True
            and window.get('later_writes') == []
            and window.get('reason') is None)


def derive_geometry_contract(d, raw, batches_by_width, parser_sources=None):
    """Reparse bound sources; serialized past window receipts cannot admit."""
    if set(batches_by_width) != set(WIDTHS):
        raise ValueError('both precision batch sets are required')
    bindings = source_bindings(raw)
    if d.get('source_bindings') != bindings:
        raise ValueError('stale/incomplete generated source bindings')
    windows = d.get('momentum_environment_contract', {})
    if set(windows) != set(WIDTHS):
        raise ValueError('missing precision window receipts')
    parser_sources = raw if parser_sources is None else parser_sources
    parser_bindings = source_bindings(parser_sources)
    if any(bindings.get(name) != pin for name, pin in parser_bindings.items()):
        raise ValueError('parser sources not joined to full source closure')
    slots = {}
    for width in WIDTHS:
        batch_pins = _batch_bindings(parser_sources, batches_by_width[width])
        producers = d.get('source_proof', {}).get(width, {}).get('producers', [])
        fresh = momentum_window(parser_sources, batches_by_width[width], producers)
        if fresh != windows[width]:
            raise ValueError('stale/unjoined momentum-window receipt')
        slots[width] = dict(source_window_supported=_window_supported(fresh),
                            batch_bindings=batch_pins,
                            window_sha256=digest(fresh),
                            reason=fresh.get('reason'))
    result = dict(schema=SCHEMA, slots=slots,
                  parser_source_bindings=parser_bindings,
                  source_input_sha256=digest(_source_inputs(d)),
                  source_scope='momentum prefix and reference-complete nonwriting suffix only',
                  runtime_abi_qualified=False, scientific_dispatch_enabled=False)
    result['contract_sha256'] = digest(result)
    return result


def geometry_support(d):
    """Emitter's conservative check of the fresh finalizer join.

    Unknown, missing, stale, malformed or retagged contracts emit two false
    constants. Only the finalizer reparses live generated sources; this check
    does not claim to reparse sources that a serialized result no longer owns.
    """
    try:
        c = copy.deepcopy(d['geometry_window_source_contract'])
        checksum = c.pop('contract_sha256')
        if checksum != digest(c) or c['schema'] != SCHEMA:
            return False, False
        if c['source_input_sha256'] != digest(_source_inputs(d)):
            return False, False
        if c['runtime_abi_qualified'] is not False or c['scientific_dispatch_enabled'] is not False:
            return False, False
        if set(c['slots']) != set(WIDTHS):
            return False, False
        if not c['parser_source_bindings'] or any(
                d['source_bindings'].get(name) != pin
                for name, pin in c['parser_source_bindings'].items()):
            return False, False
        answer = []
        for width in WIDTHS:
            slot = c['slots'][width]
            window = d['momentum_environment_contract'][width]
            value = slot['source_window_supported']
            if type(value) is not bool or value != _window_supported(window):
                return False, False
            if slot['window_sha256'] != digest(window) or not slot['batch_bindings']:
                return False, False
            for pin in slot['batch_bindings']:
                if d['source_bindings'].get(pin['file']) != {
                        'bytes': pin['bytes'], 'sha256': pin['sha256']}:
                    return False, False
            answer.append(value)
        return tuple(answer)
    except (KeyError, ValueError, TypeError, AttributeError):
        return False, False
