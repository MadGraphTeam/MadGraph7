"""Curate the finite actual-figure assets; no native qualification is rerun."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/"bin/madgraph").is_file())
PLOTS=HERE/"attempt-3-print"
OUT=ROOT/"docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/paper-comparison/lightweight-stability-figures-v1"
SCIENCE=OUT.parent/"lightweight-final-science-v1"


def sha(filename): return hashlib.sha256(filename.read_bytes()).hexdigest()
def write(filename,value): filename.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def bound(filename): return {"path":str(filename.relative_to(ROOT)),"sha256":sha(filename),"bytes":filename.stat().st_size}


def main():
    if OUT.exists(): raise ValueError("Preserve the curated actual capsule")
    receipt=json.loads((PLOTS/"render-receipt.json").read_text())
    records={Path(record["path"]).name:record for record in receipt["artifacts"]}
    def copy(name,subdir):
        source=PLOTS/name; record=records[name]
        if bound(source)!=record: raise ValueError("Changed actual figure artifact: "+name)
        target=OUT/subdir/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    OUT.mkdir(parents=True)
    for name in sorted(records):
        if name.endswith('.svg'): copy(name,'assets')
        elif name.startswith('figure-data-') or name in ('counts.json','hardware-diagnostics.json'): copy(name,'data')
    evidence=OUT/'evidence';evidence.mkdir()
    authorities={}
    for name,expected in {
        '0000-manifest.json':'de92bfae90b1068ba0fcb22034163d0b97ec9e4ab416ad17146bfa605e87f1a2',
        '0001-summary.json':'c46368e08f039d39d4acb6369f067407785315cf5a591099db97f1e9adb6b12e',
        '0002-cross-backend-discrepancy.json':'392f3ff2360df961aa1cb2170b08ca2aea446e1db5dc55c27b710bd76b13d94d',
        '1172-light-eight-arithmetic-audit-v1.json':'99e0f508d32c2c668fe0a54a9c2e4cf3361e8915197ad54782137ef4c027baa7',
        '1173-light8-final-raw-custody-ownership-audit-v1.json':'7a99193f3970403891d1e2dd5285843779c8063274dac4966a206ec4489811d4',
        '1174-light-eight-final-independent-composite-audit-v1.json':'64ac6c50cb1712272e11dbca41e797e7389d893df88ca67c1189e8f1bdf14e33',
    }.items():
        filename=SCIENCE/'data'/name
        if sha(filename)!=expected: raise ValueError('Changed public science copy: '+name)
        authorities[name]=bound(filename)
    write(evidence/'scientific-authorities.json',authorities)
    proofs={
        HERE/'attempt-1/render-receipt.json':'original-live-reader-replay-receipt.json',
        PLOTS/'render-receipt.json':'materialized-render-receipt.json',
        PLOTS/'preview-attempt-2/visual-reproduction-receipt.json':'preview-visual-reproduction-receipt.json',
        HERE/'render-attempt-1.py':'first-live-render-source.py',
        HERE/'render-print.py':'print-layout-source.py',
        HERE/'replot-print.py':'offline-materialization-source.py',
    }
    for source,name in proofs.items(): shutil.copyfile(source,evidence/name)
    source=(HERE/'render-print.py').read_text();tree=ast.parse(source)
    constants={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
        and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('PAIRS','LABELS','COLORS')}
    node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='draw')
    draw=ast.get_source_segment(source,node).replace('paper.require(', 'require(').replace('if estimator == "combined":','if INSPECTION_PNG and estimator == "combined":')
    header='''"""Offline vector reproduction of the materialized actual N100 DP study.

This is a rendering recipe, not a source/runtime qualification authority.
No native MadLoop/Rust archive, compiler or generated process is used.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import textwrap

HERE=Path(__file__).resolve().parent
INSPECTION_PNG=False
def require(condition,message):
    if not condition: raise ValueError(message)
'''
    declarations='\n'.join(name+'='+repr(constants[name]) for name in ('PAIRS','LABELS','COLORS'))+'\n\n'
    footer='''
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); output=args.output.resolve()
    require(not output.exists(),'Preserve previous vector reproductions')
    manifest=json.loads((HERE/'manifest.json').read_text())
    for name,record in manifest['files'].items():
        filename=(HERE/name).resolve()
        require(filename.is_relative_to(HERE),'Invalid capsule path')
        raw=filename.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==record['sha256'] and len(raw)==record['bytes'],'Changed figure capsule input: '+name)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot,font_manager
    require(matplotlib.__version__==manifest['rendering']['matplotlib'],'Use the recorded Matplotlib version for byte-identical vectors')
    font=Path(font_manager.findfont('Libertinus Serif'))
    require(hashlib.sha256(font.read_bytes()).hexdigest()==manifest['rendering']['font_sha256'],'Use the recorded Libertinus Serif font for byte-identical vectors')
    output.mkdir(parents=True)
    with pyplot.rc_context({'font.family':'Libertinus Serif','font.size':10.3,'text.color':'#1f3b63',
            'axes.labelcolor':'#1f3b63','axes.titlecolor':'#1f3b63','mathtext.fontset':'stix',
            'svg.fonttype':'none','svg.hashsalt':'actual-lightweight-dp-v1'}):
        for estimator in ('combined','direction','lorentz'):
            prepared=json.loads((HERE/'data'/('figure-data-'+estimator+'.json')).read_text())
            require(set(prepared)==set(LABELS),'Unexpected materialized case selection')
            for case in prepared.values():
                require(case['N']==100 and case['requested_tools']==[6,8] and set(case['tools'])=={'6','8'},'Unexpected materialized point/tool scope')
            for group,pair in PAIRS.items(): draw(prepared,pair,estimator,output/(estimator+'-'+group),pyplot)
    expected={name:record['sha256'] for name,record in manifest['files'].items() if name.startswith('assets/')}
    for name,sha256 in expected.items():
        require(hashlib.sha256((output/Path(name).name).read_bytes()).hexdigest()==sha256,'Reproduced vector bytes differ: '+name)
    print(json.dumps({'vectors':len(expected),'byte_identical':True,'native_artifacts_opened':0,'new_scientific_qualification':False}))

if __name__=='__main__': main()
'''
    (OUT/'plot.py').write_text(header+declarations+draw+'\n'+footer)
    readme='''These are actual, audited Ninja6/NinjaRS8 native-double stability figures for eight selected processes, 100 points per tool. They contain no synthetic values. The three figure families show the combined, loop-direction and Lorentz self-consistency spreads. Each panel retains the true sample, sampling authority, seed, cuts, invariant energy and helicity selection. The loop-induced panels are single all-minus; the six Born processes use full helicity/color sums.

All 1600 combined scores are finite. The empirical resolution is 1%; zero events have an exact one-sided 95% binomial reference upper bound of about 2.95%. Boosted on-shell 2→1 frames have one physical invariant and are descriptive. The other points follow the declared massive-RAMBO proposal and cuts, rather than a claimed uniform massive integration. Ninja-zero tail ratios stay undefined/unbounded gaps. No epsilon or smoothing is used, and these transformation estimates are not error to an independent finite-part oracle.

Hardware diagnostics remain distinct. Saved IEEE-invalid is recorded on all 300 transformed observations in each of five NinjaRS streams: QCD2, EW2, QCD3, EW3 and gg→HH; the plain-prefix observations match. Saved divide-by-zero/overflow flags and algorithmic FPE/rejections/nonfinite/MP-comp counts are zero. The frozen observer did not treat the saved-invalid flag alone as automatic rejection. See data/hardware-diagnostics.json and the independent audit bindings.

The materialized arrays and SVGs are portable offline report assets. To reproduce the vectors, run plot.py --output NEW_DIRECTORY under the recorded Matplotlib/Libertinus environment (Python -B and one thread are recommended); the script checks every local payload and verifies all 12 output SVG byte hashes. Set MPLCONFIGDIR to a private cache if needed. It does not open a native archive or rerun numerical physics. The source images and preview receipt are historical rendering evidence, not entry points for fresh full runtime qualification.

The full native/source/runtime verification was performed once against the retained exact installation and generated targets. Its receipt, exact public scientific summary, original-path raw manifest and independent arithmetic/raw-custody/composite receipts are bound under evidence/. The public science copies preserve their original embedded paths; do not pass the relocated raw manifest to a live validator without an explicit resolver. Full custody verification needs the retained native installation; regeneration from pins may produce different compiler identities. Offline Typst rendering from these committed vector/data assets is independent of that installation.

Typography/colors follow the NinjaRS benchmark at ca1009e89ecb03d3ff477f57b170241f7a9f8669; figure methodology follows Hirschi–Peraro, arXiv:1604.01363. These assets are separate from stock-CLI stage timing, the six-case full-sum timing study, and normal rescue statistics. Private PNGs, intermediate preview PDFs and layout attempts are intentionally excluded from this finite publication capsule.
'''
    (OUT/'README.md').write_text(readme)
    font=json.loads((HERE/'attempt-1/render-receipt.json').read_text())['font']
    write(OUT/'manifest.json',{'schema':'actual-lightweight-stability-vector-assets-v1','scientific_data':'actual selected-eight N100; audited source/runtime/raw-custody/arithmetic',
        'requested_tools':[6,8],'points_per_case_per_tool':100,'rendering':{'matplotlib':receipt['matplotlib'],'font_sha256':font['sha256'],'font_family':'Libertinus Serif','minimum_axes_font_pt':10.3,'sampling_metadata_font_pt':9.5},
        'files':{str(p.relative_to(OUT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file()}})
    print(OUT)


if __name__=='__main__': main()
