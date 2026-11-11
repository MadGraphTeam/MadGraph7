"""Read-only CURRENT source/input admission checks; no child or native calls."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PINS = {'context-bindings.json':'082f5df9baa6973940914ce9d0d8b89cf14ae31c8cc5b2cb4340f3a71d227366',
        'input-bindings.json':'6f801d535de58b77ae8158315d0e686ea8575bfdc61c668181b4a7cb3d703a4c'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def content_digest(value):
    return digest(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def read_record(record):
    path=Path(record['path'])
    require(not path.is_absolute() and '..' not in path.parts,'unsafe record path')
    raw=(ROOT/path).read_bytes()
    require(len(raw)==record['bytes'] and digest(raw)==record['sha256'],
            'input changed: '+str(path))
    return raw


def records(value):
    if isinstance(value,dict):
        if set(('path','bytes','sha256')) <= value.keys():
            yield value
        else:
            for child in value.values():
                yield from records(child)
    elif isinstance(value,list):
        for child in value:
            yield from records(child)


def main():
    inputs={}
    for name,h in PINS.items():
        raw=(HERE/name).read_bytes()
        require(digest(raw)==h,'proposal binding changed')
        inputs[name]=json.loads(raw)
    context,science=inputs['context-bindings.json'],inputs['input-bindings.json']
    blobs={r['path']:read_record(r) for r in records(inputs)}
    old_record=context['historical_packet']
    old=json.loads(blobs[old_record['path']])
    old_root=(ROOT/old_record['path']).parent
    for path,pin in old['files'].items():
        raw=(old_root/path).read_bytes()
        require(len(raw)==pin['bytes'] and digest(raw)==pin['sha256'],
                'sealed historical packet changed: '+path)
    checkpoint=json.loads(blobs[context['ew_checkpoint']['path']])
    retained={}
    for name,pin in checkpoint['files'].items():
        raw=(ROOT/name).read_bytes()
        require(len(raw)==pin['bytes'] and digest(raw)==pin['sha256'],
                'EW checkpoint payload changed: '+name)
        retained[name]=raw
    require(len(retained)==checkpoint['file_count'] and
            sum(map(len,retained.values()))==checkpoint['total_bytes'], 'checkpoint count/size')
    manifest=json.loads(blobs[context['campaign_manifest']['path']])
    identity=manifest['candidate']['exporter_identity']
    require(identity['revision']==context['production_anchor'] and
            identity['files_sha256']==context['exporter_inventory']==content_digest(identity['files']),
            'wrong source anchor/inventory')
    require(manifest['candidate']['numerical_identity_sha256']==context['numerical_identity'],
            'wrong numerical identity')
    for name,h in identity['files'].items():
        path=ROOT/name
        require((digest(path.read_bytes()) if path.is_file() else None)==h,
                'live exporter differs: '+name)
    model=json.loads(blobs[science['model']['path']])
    require(model['files']==manifest['shared_inputs']['model_tree'], 'model/campaign mismatch')
    require(content_digest(model['files'])==model['tree_sha256'], 'model tree digest mismatch')
    for name,h in model['files'].items():
        require(digest((ROOT/context['model_donor']/name).read_bytes())==h,
                'model donor changed: '+name)
    compiled=json.loads(blobs[context['ew_compilation']['path']])
    lane=json.loads(blobs[context['ew_candidate_inputs']['path']])
    generation=lane['generation']; donor=ROOT/context['ew_donor']
    require(generation['source_identity']['files']==identity['files'] and
            generation['source_identity']['files_sha256']==context['exporter_inventory'],
            'actual generation inventory differs')
    require(json.loads((donor/'generation.json').read_bytes())==generation and
            json.loads((donor/'compilation.json').read_bytes())==compiled,
            'donor receipt changed')
    require(compiled['full_numerical_rebuild_this_invocation'] is True and
            compiled['numerical_build_from_sources'] is True, 'not a fresh compiled export')
    for name,h in compiled['sources'].items():
        require(digest((donor/name).read_bytes())==h,'generated source changed: '+name)
    for name,text in compiled['settings'].items():
        require((donor/name).read_text()==text,'compiler settings changed: '+name)
    require(digest((donor/'SubProcesses/P0_epem_ttxa/check').read_bytes())==compiled['executable_sha256'],
            'qualified donor executable changed')
    # Only the unchanged compiled numerical archives are candidates for reuse;
    # future process objects/driver will be rebuilt in private destinations.
    private_archives={}
    for name,h in compiled['link_inputs_after']['archives'].items():
        p=Path(name)
        if p.is_relative_to(donor):
            require(digest(p.read_bytes())==h,'private donor archive changed')
            private_archives[str(p.relative_to(donor))]=h
    native=manifest['candidate']['native']; native_prefix=Path(native['prefix'])
    for name,h in native['files'].items():
        require(digest((native_prefix/name).read_bytes())==h,'installed TIRust input changed: '+name)
    require(not (HERE.parent/'work').exists() and not (HERE.parent/'results').exists(),
            'work/results already exists; this is only the initial source-only preflight')
    return {'schema':'current-kinematic-trace-readonly-preflight-v1',
            'historical_packet_unchanged':True,'historical_payloads_verified':len(old['files']),
            'ew_checkpoint_payloads_verified':len(retained),'exporter_files_verified':len(identity['files']),
            'exporter_inventory':context['exporter_inventory'],'source_anchor':context['production_anchor'],
            'actual_ew_generation_revision':generation['source_identity']['revision'],
            'ew_generated_source_files_verified':len(compiled['sources']),
            'ew_compiler':compiled['compiler'],'ew_executable_sha256':compiled['executable_sha256'],
            'ew_numerical_archives_verified':private_archives,'model_files_verified':len(model['files']),
            'model_tree_sha256':model['tree_sha256'],'native_files_hashed_only':len(native['files']),
            'cli_executions':0,'compiler_executions':0,'native_executions':0,
            'scope':'Source/input admission only; no runtime closure or new numerical result.'}


if __name__=='__main__':
    print(json.dumps(main(),sort_keys=True,indent=2))
