"""Reconstruct the exact selected software image; never compile physics code."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,tarfile
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);args=p.parse_args()
work=(ROOT/args.work).resolve() if not args.work.is_absolute() else args.work.resolve()
if work.parent!=ROOT/'.benchmark-work' or work.exists():raise ValueError('Choose a new direct .benchmark-work child; no overwrite')
work.mkdir(parents=True);source=work/'source-v4';source.mkdir();prep=json.loads((HERE/'preparation-v4.json').read_text())
commands=[prep['git_archive_command'],prep['closure_addition']['command'],prep['second_closure_addition']['command']]
for i,command in enumerate(commands):
 archive=work/f'source-{i}.tar'
 with archive.open('xb') as out:subprocess.run(command,cwd=ROOT,stdout=out,check=True)
 with tarfile.open(archive) as stream:stream.extractall(source,filter='data')
def info(path):return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
for name,record in prep['overlays'].items():
 origin=ROOT/name
 if info(origin)!=record:raise ValueError('Use repository source matching overlay hash: '+name)
 target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists() or target.is_symlink():target.unlink()
 target.write_bytes(origin.read_bytes())
state={'files':{str(x.relative_to(source)):info(x) for x in sorted(source.rglob('*')) if x.is_file() and not x.is_symlink()},'symlinks':{str(x.relative_to(source)):os.readlink(x) for x in source.rglob('*') if x.is_symlink()}}
if state!=json.loads((HERE/'source-state-before-v4.json').read_text()):raise ValueError('Reconstructed source differs from observed exact image')
for name in ['run-module.py','run-suite.py','modules.json','preparation-v4.json','source-state-before-v4.json']:shutil.copy2(HERE/name,work/name)
raise SystemExit(subprocess.run(['taskset','-c','4',sys.executable,'-B',str(work/'run-suite.py')],cwd=ROOT).returncode)
