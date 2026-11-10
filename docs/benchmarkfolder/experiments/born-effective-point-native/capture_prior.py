#!/usr/bin/env python3
"""Preserve earlier isolated build failures as text, with native hashes only."""
import gzip,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];BASE=ROOT/'.tirust-build/born-effective-build'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 index=BASE/'build-index.json';assert sha(index)=='c28a90390f08baeeb8d375d365c3de1c1098e2db5c8d603beefb1d8a4c014f9f'
 entries=json.loads(index.read_text())['files'];files={};blobs={};omitted={}
 for name,r in entries.items():
  p=BASE/name
  if r['kind']!='file':omitted[name]=r;continue
  data=p.read_bytes();assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
  native=data.startswith((b'\x7fELF',b'!<arch>\n')) or b'\0' in data or p.suffix.lower() in ('.o','.a','.mod','.so','.smod','.pyc','.pkl') or p.name=='check'
  try:text=None if native else data.decode('utf8')
  except UnicodeDecodeError:text=None
  if text is None:omitted[name]=r;continue
  files[name]=r;blobs.setdefault(r['sha256'],text)
 out=HERE/'prior-build-history.json.gz';assert not out.exists();out.write_bytes(gzip.compress(json.dumps({'base':str(BASE.relative_to(ROOT)),'files':files,'blobs':blobs,'native_or_alias_hash_only':omitted},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode(),9,mtime=0));shutil.copyfile(index,HERE/'prior-build-index.json')
 print(json.dumps({'text_paths':len(files),'unique_blobs':len(blobs),'omitted_native_or_aliases':len(omitted),'compressed_bytes':out.stat().st_size}))
if __name__=='__main__':main()
