import datetime, hashlib, io, json, posixpath, shutil, subprocess, tarfile
from pathlib import Path
root=Path.cwd(); work=root/'.tirust-build/current-install'
revision=subprocess.check_output(['git','rev-parse','606603e720'],text=True).strip()
archive=subprocess.check_output(['git','archive',revision])
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    entries={entry.name:entry for entry in tar.getmembers()}
    def contents(name,seen=()):
        if name in seen: raise RuntimeError('Symlink cycle '+name)
        entry=entries[name]
        if entry.issym(): return contents(posixpath.normpath(posixpath.join(posixpath.dirname(name),entry.linkname)),seen+(name,))
        if not entry.isfile(): raise RuntimeError('Non-file target '+name)
        return tar.extractfile(entry).read()
    sources={}; symlinks={}; dangling={}
    for name,entry in sorted(entries.items()):
        if name.split('/')[0] in {'docs','doc','tests','tools','HEPTools'} or Path(name).suffix in {'.pyc','.o','.a','.so','.mod','.pdf','.png','.jpg'}: continue
        if not (entry.isfile() or entry.issym()): continue
        try: data=contents(name)
        except KeyError:
            if not entry.issym(): raise
            dangling[name]=entry.linkname
            continue
        target=work/'mg'/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data); target.chmod(entry.mode if entry.isfile() else 0o755)
        sources[name]=hashlib.sha256(data).hexdigest()
        if entry.issym(): symlinks[name]=entry.linkname

def write(name,value): (work/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
write('source-inventory.json',sources)
write('source-revision.json',{'revision':revision,'source':'git archive of exact committed revision','copied_file_count':len(sources),'materialized_symlinks':symlinks,'excluded_dangling_symlinks':dangling,'source_overrides':False,'excluded_top_directories':['docs','doc','tests','tools','HEPTools'],'excluded_suffixes':['.pyc','.o','.a','.so','.mod','.pdf','.png','.jpg']})
config=work/'mg/input/mg7_configuration.txt'; config.write_text((work/'mg/input/.mg7_configuration_default.txt').read_text()+'\nauto_update = 0\nautomatic_html_opening = False\n')
protected=['input/mg7_configuration.txt','HEPTools/tirust/lib/tirust-config.json','HEPTools/tirust/lib/libtirust.a','HEPTools/tirust/lib/libmadloop_tirust.so','HEPTools/tirust/lib/mtirust.mod']
write('production-before.json',{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in protected})
(work/'observer').mkdir(exist_ok=True); shutil.copy2(root/'.tirust-build/clean-install/observer/sitecustomize.py',work/'observer/sitecustomize.py')
for name in ['xdg-config','xdg-cache']: (work/name).mkdir(exist_ok=True)
(work/'install.mg5').write_text('set automatic_html_opening False --no_save\nset auto_update 0 --no_save\nset nb_core 4 --no_save\ninstall tirust --jobs=4\nquit\n')
(work/'run.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
source tools/tirust_env.sh
export TIRUST_CLEAN_WORK="$PWD/.tirust-build/current-install"
export TIRUST_NUMERICAL_LOCK="$PWD/.tirust-build/numerical-tests.lock"
export CARGO_HOME="$PWD/.tirust-build/cargo-home"
export CARGO_TARGET_DIR="$PWD/.tirust-build/target"
export CARGO_BUILD_BUILD_DIR="$PWD/.tirust-build/cargo-build"
export XDG_CONFIG_HOME="$TIRUST_CLEAN_WORK/xdg-config"
export XDG_CACHE_HOME="$TIRUST_CLEAN_WORK/xdg-cache"
export PYTHONPATH="$TIRUST_CLEAN_WORK/observer"
export PYTHONDONTWRITEBYTECODE=1
unset CARGO_NET_OFFLINE MADGRAPH_BASE RUSTC_WRAPPER RUSTC_WORKSPACE_WRAPPER
python3 "$TIRUST_CLEAN_WORK/mg/bin/madgraph" --plain "$TIRUST_CLEAN_WORK/install.mg5"
'''); (work/'run.sh').chmod(0o755)
write('install-start.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'revision':revision,'cwd':str(root),'cache_policy':'Reuse existing common Cargo download and release dependency caches; this is not a clean-cache installation. Ordinary installer staging and compilation remain unchanged.','cache_paths':{name:str(root/'.tirust-build'/name) for name in ['cargo-home','target','cargo-build']},'source_overrides':False,'source_inventory':'source-inventory.json','env_script_sha256':hashlib.sha256((root/'tools/tirust_env.sh').read_bytes()).hexdigest(),'bridge_sha256':sources['vendor/tirust/src/lib.rs'],'fortran_source_sha256':sources['vendor/tirust/mtirust.f90'],'patch_sha256':sources['vendor/tirust/patches/0001-direct-poles-only.patch']})
print(json.dumps({'revision':revision,'copied':len(sources),'symlinks':len(symlinks),'omitted_dangling_symlinks':dangling,'bridge_sha256':sources['vendor/tirust/src/lib.rs']},indent=2))
