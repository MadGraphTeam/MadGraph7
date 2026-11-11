"""Explicit read-only inputs for the source-only generation/check scripts."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PARENT=ROOT/'.tirust-build/generic-invariant-provenance-proposal-1'
def bound(path,expected,size=None):
 data=path.read_bytes()
 if hashlib.sha256(data).hexdigest()!=expected or size is not None and len(data)!=size:raise ValueError('source input changed: '+str(path))
 return data

def verify_parents(live=False):
 pins=json.loads((HERE/'parent-pins.json').read_bytes())
 for name,h in pins.items():bound(ROOT/name,h)
 idx=json.loads((PARENT/'source-index.json').read_bytes())
 for name,m in idx['files'].items():bound(PARENT/name,m['sha256'],m['bytes'])
 current=json.loads((PARENT/'source-pins.json').read_bytes())['current_source_files']
 if live:
  for name,m in current.items():bound(ROOT/name,m['sha256'],m['bytes'])
  for name,m in json.loads((HERE/'model-source-pins.json').read_bytes()).items():bound(ROOT/name,m['sha256'],m['bytes'])
 return {'parent_payloads':len(idx['files']),'current_sources':len(current) if live else None}
