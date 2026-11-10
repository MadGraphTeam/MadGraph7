#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
manifest=json.loads((here/'manifest.json').read_text())
for name,digest in manifest['retained_files'].items():
 p=here/name
 if not p.resolve().is_relative_to(here) or p.is_symlink(): raise ValueError('Nonportable evidence path: '+name)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=digest: raise ValueError('Changed retained file: '+name)
print(json.dumps({'retained_files_verified':len(manifest['retained_files']),'native_artifacts_hash_only':len(manifest['native_artifacts']),'compiler_or_native_launched':False}))
