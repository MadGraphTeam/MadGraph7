#!/usr/bin/env python3
"""Isolated compensated integration prototype over the pinned fast engine."""
from pathlib import Path
import argparse,decimal,difflib,hashlib,json,os,shutil,subprocess
ROOT=Path(__file__).resolve().parents[4];DOC=Path(__file__).resolve().parent
WORK=ROOT/'.benchmark-work/experiments/qcd3-point67-compensated'
MASTERS=DOC.parent/'qcd3-point67-masters';INPUT=DOC.parent/'qcd3-point67-dp/oracle-inputs.json'
REV='0d0d7e7a8033431b60a8580731084742e0655cda'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def prepare(source):
 if WORK.exists():raise RuntimeError('refusing an existing prototype directory')
 actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip();assert actual==REV
 crate=WORK/'bridge';native=crate/'ninjars';shutil.copytree(source/'crates/ninjars/src',native/'src')
 (native/'Cargo.toml').write_text('''[package]
name = "ninjars"
version = "0.1.0"
edition = "2024"
rust-version = "1.96"
license = "GPL-3.0-or-later"
[dependencies]
feynkit-kinematics = { git = "https://github.com/alphal00p/gammaloop.git", rev = "1e8063a763d3aa515749c33616ee59aa827705be", default-features = false, features = ["native"] }
symbolica = { version = "3.0.1", default-features = false, features = ["integer-gmp", "float-mpfr", "bincode", "serde", "tracing_max_level_info"] }
thiserror = "2"
''')
 engine=native/'src/fast/engine.rs';original=engine.read_text();start=original.index('    fn integrate<M:');end=original.index('    /// Ninja\'s rank n+1 choice',start);body=original[start:end]
 new=body.replace('fn integrate<','fn integrate_compensated<',1).replace('        let n = self.topo.n;','        let products = std::env::var("NRS_DIAGNOSTIC_SUM_MODE").as_deref() == Ok("2");\n        let n = self.topo.n;',1)
 new=new.replace('let mut res = [C64::ZERO; 3];','let mut res = [AccurateComplex::default(); 3];').replace('let mut rational = C64::ZERO;','let mut rational = AccurateComplex::default();')
 for old,replacement,count in [
  ('res[k] += c0 * integral[k];','res[k].add_product(c0, integral[k], products);',2),
  ('res[k] += c[0] * integral[k];','res[k].add_product(c[0], integral[k], products);',1),
  ('res[k] += data.c0 * integral[k];','res[k].add_product(data.c0, integral[k], products);',1),
  ('''res[j] += c[0] * out[0][j]
                                + c1 * out[1][j]
                                + c2 * out[2][j]
                                + c13 * out[3][j];''','''res[j].add_product(c[0], out[0][j], products);
                            res[j].add_product(c1, out[1][j], products);
                            res[j].add_product(c2, out[2][j], products);
                            res[j].add_product(c13, out[3][j], products);''',1),
  ('res[j] += c[0] * out[0][j] + c1 * out[1][j] + c2 * out[2][j];','''res[j].add_product(c[0], out[0][j], products);
                            res[j].add_product(c1, out[1][j], products);
                            res[j].add_product(c2, out[2][j], products);''',1),
  ('            let result = FastResult {','            let res = res.map(|x| x.value());\n            let rational = rational.value();\n            let result = FastResult {',1)]:
  assert new.count(old)==count,(old,new.count(old));new=new.replace(old,replacement)
 wrapper='''    fn integrate<M: FastMasters + ?Sized>(&mut self, input: &FastInput, masters: &mut M) -> Result<()> {
        match std::env::var("NRS_DIAGNOSTIC_SUM_MODE").as_deref() {
            Ok("1" | "2") => self.integrate_compensated(input, masters),
            _ => self.integrate_original(input, masters),
        }
    }

'''
 changed=original[:start]+wrapper+body.replace('fn integrate<','fn integrate_original<',1)+new+original[end:]+'\n'+(DOC/'accumulator.rs').read_text();engine.write_text(changed)
 (DOC/'engine.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile='a/src/fast/engine.rs',tofile='b/src/fast/engine.rs')))
 base=(MASTERS/'Cargo.toml').read_text();begin=base.index('[patch.crates-io]');patch=base[begin:]
 (crate/'Cargo.toml').write_text('''[package]
name = "point67-compensated"
version = "0.1.0"
edition = "2024"
[workspace]
[dependencies]
ninjars = { path = "ninjars" }
serde_json = { version = "1", features = ["float_roundtrip"] }
'''+patch)
 (crate/'src').mkdir();shutil.copy2(DOC/'main.rs',crate/'src/main.rs');shutil.copy2(MASTERS/'Cargo.lock',crate/'Cargo.lock')
 save(DOC/'source.json',{'upstream_revision':actual,'upstream_engine_sha256':hashlib.sha256(original.encode()).hexdigest(),'prototype_engine_sha256':sha(engine),'accumulator_sha256':sha(DOC/'accumulator.rs'),'driver_sha256':sha(DOC/'main.rs'),'source_files':{str(p.relative_to(native)):sha(p) for p in sorted(native.rglob('*')) if p.is_file()}})
def build():
 crate=WORK/'bridge';env=os.environ.copy();env['CARGO_TARGET_DIR']=str(WORK/'target')
 update=['cargo','metadata','--offline','--format-version=1','--manifest-path',str(crate/'Cargo.toml')];subprocess.run(update,cwd=ROOT,env=env,stdout=subprocess.DEVNULL,check=True)
 command=['cargo','build','--release','--locked','--manifest-path',str(crate/'Cargo.toml')]
 with (DOC/'build.log').open('w') as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 for f in ['Cargo.toml','Cargo.lock']:shutil.copy2(crate/f,DOC/f)
 save(DOC/'build.json',{'commands':[update,command],'executable_sha256':sha(WORK/'target/release/point67-compensated'),'manifest_sha256':sha(crate/'Cargo.toml'),'lock_sha256':sha(crate/'Cargo.lock'),'source_sha256':sha(DOC/'source.json')})
def run():
 exe=WORK/'target/release/point67-compensated';runs=[]
 for mode in range(3):
  env=os.environ.copy();env['NRS_DIAGNOSTIC_SUM_MODE']=str(mode);out=DOC/f'mode-{mode}.json'
  cmd=[str(exe),str(INPUT),str(MASTERS/'results.json'),str(out)];subprocess.run(cmd,cwd=ROOT,env=env,check=True);runs.append({'command':cmd,'mode':mode,'output_sha256':sha(out)})
 save(DOC/'run.json',{'executable_sha256':sha(exe),'input_sha256':sha(INPUT),'masters_capture_sha256':sha(MASTERS/'results.json'),'runs':runs});analyze()
def analyze():
 decimal.getcontext().prec=90;D=decimal.Decimal
 oracle={(r['loop'],r['direction']):r for r in json.loads((DOC.parent/'qcd3-point67-dp/oracle-results.json').read_text())['rows']};rows=[]
 for mode in range(3):
  for row in json.loads((DOC/f'mode-{mode}.json').read_text()):
   ref=oracle[row['loop'],row['direction']];err=[str(D.from_float(x)-D(y)) for x,y in zip([v for z in row['laurent'] for v in z],ref['oracle_laurent_components'])];rows.append({'mode':mode,'loop':row['loop'],'direction':row['direction'],'errors':err});print(mode,row['loop'],row['direction'],err[0])
 save(DOC/'summary.json',{'modes':{'0':'unchanged pinned integration','1':'Neumaier accumulation of rounded complex products','2':'Neumaier accumulation with FMA product residuals'},'rows':rows})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build','run','analyze']);p.add_argument('--source',type=Path,default=ROOT.parent/'references/ninjars');a=p.parse_args()
 if a.action=='prepare':prepare(a.source)
 else:globals()[a.action]()
