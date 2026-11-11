"""Read-only exact dormant-unit payload/input verification; no compilation/run."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def identity(path):
    data=path.read_bytes();return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def main():
    receipt_path=HERE/'source-receipt.json';receipt=json.loads(receipt_path.read_bytes())
    sha=(HERE/'source-receipt.sha256').read_text().strip()
    if identity(receipt_path)['sha256']!=sha:raise ValueError('changed dormant unit receipt')
    actual={str(path.relative_to(HERE)) for path in HERE.rglob('*') if path.is_file()}
    if actual!=set(receipt['payloads'])|{'source-receipt.json','source-receipt.sha256'}:raise ValueError('unit payload inventory differs')
    for base,entries in ((HERE,receipt['payloads']),(ROOT,receipt['external_inputs'])):
        for name,pin in entries.items():
            path=base/name
            if path.is_symlink() or not path.resolve().is_relative_to(base) or identity(path)!=pin:
                raise ValueError('changed source input: '+name)
    if receipt['BUILD_admitted'] or receipt['RUN_admitted'] or receipt['compiler_calls'] or receipt['native_calls']:
        raise ValueError('dormant source receipt claims execution')
    for flag in ('native_snapshot_transport_qualified','actual_constructor_completion_qualified',
                 'q_at_use_lowering_qualified','final_dot_runtime_qualified','full_caller_qualified','scientific_dispatch_enabled'):
        if receipt[flag] is not False:raise ValueError('unadmitted unit qualification: '+flag)
    print(json.dumps(dict(status='pass',source_receipt_sha256=sha,payloads=len(receipt['payloads']),
        external_inputs=len(receipt['external_inputs']),compiler_calls=0,native_calls=0,
        BUILD_admitted=False,RUN_admitted=False,scientific_dispatch_enabled=False),sort_keys=True))


if __name__=='__main__':main()
