#!/usr/bin/env python3
"""Append the sealed five-backend timing tail while preserving the previous39 pages."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import types

ROOT=Path(__file__).resolve().parent
SCHEMA='madloop-tirust-timing-composite-report-v1'
BASE='history/report-before-five-backend-timing-1'
BASE_SHA='1e26d042e133120a68b48787a6aae75907fb3769c1d70fac4aa3648606d6830d'
OWNER_SHA='f9df20e66a8457eb492941da67426ad2af1174d0ed4593065ff119697219803b'
REPORT='experiments/current-source-five-backend-timing-report-1/rendered-1'
REPORT_SHA='da73ae6af1770efedde48602c2b0dc44426b125b018e808d944aa62f03f5df28'
CASES={'qcd_n1_uux_z','qcd_n2_gg_ttx','ew_n2_epem_mupmum','qcd_n4_uux_epem_mupmum'}


def require(value,message):
    if not value:raise ValueError(message)


def encoded(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def identity(path):
    raw=Path(path).read_bytes();return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def digest(value):return hashlib.sha256(encoded(value)).hexdigest()


def read(path):
    def pairs(items):
        out={}
        for key,value in items:require(key not in out,'Duplicate JSON key');out[key]=value
        return out
    def bad(value):raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,parse_constant=bad)


def archived_owner(archive):
    archive=Path(archive)
    require(identity(archive/'index.json')['sha256']==BASE_SHA,'Archived39 index changed')
    index=read(archive/'index.json')
    require({p.name for p in archive.iterdir()}==set(index['files'])|{'index.json'},'Archived39 inventory changed')
    for name,expected in index['files'].items():
        require(Path(name).name==name and not (archive/name).is_symlink(),'Unsafe archived input')
        require(identity(archive/name)==expected,'Archived39 input changed')
    path=archive/'build_stress_report.py'
    require(identity(path)['sha256']==OWNER_SHA,'Archived39 owner changed')
    # Execute only the authenticated source bytes, without import-loader cache
    # writes inside this strictly inventoried archive (also without python -B).
    module=types.ModuleType('frozen39_owner');module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module


def selected(root,archive,report):
    owner=archived_owner(archive)
    owner.verify(root,Path(archive)/'benchmark.pdf',Path(archive)/'report-build.json')
    packet=owner.packet(report,REPORT_SHA)
    require(packet['page_count']>0 and packet['page_count']<=4,'Unexpected timing appendix length')
    selection=read(Path(report)/'scientific-selection.json')
    require(selection['closed'] is True and len(selection['case_ids'])==4 and set(selection['case_ids'])==CASES,
            'Timing evidence is not closed for the four requested cases')
    # Exact paths and seals only: first-three and fresh2->4 timing evidence can
    # have separate owners. Do not discover a latest attempt or change cohorts.
    for item in selection['packets']:
        require(set(item)=={'path','index_name','index_sha256'},'Unexpected scientific packet selector')
        require(item['index_name'] in ('index.json','evidence-index.json'),'Unsafe or unsupported explicit packet index name')
        owner.packet(owner.inside(root,item['path']),item['index_sha256'],item['index_name'])
    return owner,{'archive_index_sha256':BASE_SHA,'report_index_sha256':REPORT_SHA,
                  'old39_PDF':identity(Path(archive)/'benchmark.pdf'),
                  'old39_ledger':identity(Path(archive)/'report-build.json'),
                  'old39_owner':identity(Path(archive)/'build_stress_report.py'),
                  'appendix_PDF':identity(Path(report)/'timing-supplement.pdf'),
                  'scientific_selection':identity(Path(report)/'scientific-selection.json'),
                  'appendix_pages':packet['page_count']}


def prove(owner,tools,archive,report,merged,scratch,pages):
    base=owner.inspect_document(tools,Path(archive)/'benchmark.pdf',scratch,'old39',39)
    tail=owner.inspect_document(tools,Path(report)/'timing-supplement.pdf',scratch,'timings',pages)
    combined=owner.inspect_document(tools,merged,scratch,'combined',39+pages)
    require(combined[:39]==base and combined[39:]==tail,'Timing composite page preservation failed')
    return {'base_pages':39,'tail_pages':pages,'total_pages':39+pages,'pages':combined,
            'prefix_preserved':True,'tail_preserved':True,'PDF':identity(merged),
            'dpi':144,'tool_versions':tools.versions,
            'scope':'Paired RGB pixels, per-page text/geometry/fonts; not object-graph or new annotation/link equivalence.'}


def location(root,path):
    path=Path(path).resolve();return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def verify(root,output,ledger,verify_pages=False):
    record=read(ledger);require(record['schema']==SCHEMA,'Wrong report owner; use the matching append builder')
    archive=Path(record['archive']);archive=archive if archive.is_absolute() else root/archive
    report=Path(record['report']);report=report if report.is_absolute() else root/report
    owner,inputs=selected(root,archive,report)
    require(inputs==record['selected_inputs'],'Timing report inputs changed')
    require(identity(output)==record['PDF'] and identity(__file__)==record['builder'],'Timing report or builder changed')
    require(digest(record['page_proof'])==record['page_proof_sha256'],'Page-proof receipt changed')
    proof=record['page_proof']
    require(proof['PDF']==record['PDF'] and proof['prefix_preserved'] and proof['tail_preserved']
            and proof['base_pages']==39 and proof['tail_pages']==inputs['appendix_pages']
            and len(proof['pages'])==39+inputs['appendix_pages'],'Incomplete bound page proof')
    if verify_pages:
        with tempfile.TemporaryDirectory(prefix='madloop-timing-page-verify-') as name:
            fresh=prove(owner,owner.DocumentTools(),archive,report,output,Path(name),inputs['appendix_pages'])
            require(fresh['pages']==proof['pages'],'Current page check differs from the recorded proof')
    return {'verified':True,'pages':len(proof['pages']),'mode':'document pages' if verify_pages else 'file hashes and bound page proof','new_native_physics_evaluations':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--verify-pages',action='store_true')
    parser.add_argument('--rebuild-supplement',action='store_true')
    args=parser.parse_args();root=args.root.resolve();output=(args.output or root/'benchmark.pdf').resolve()
    ledger=root/'report-build.json' if output==root/'benchmark.pdf' else output.with_suffix('.build.json')
    require(not args.verify_pages or args.verify_only,'--verify-pages requires --verify-only')
    require(not (args.verify_only and args.rebuild_supplement),'Rebuild and verification are separate operations')
    if args.verify_only:print(encoded(verify(root,output,ledger,args.verify_pages)).decode(),end='');return
    archive=(args.archive or root/BASE).resolve();report=(args.report or root/REPORT).resolve()
    if output==root/'benchmark.pdf':require(archive.is_relative_to(root) and report.is_relative_to(root),'Canonical dependencies must be retained in docs/benchmarkfolder')
    owner,inputs=selected(root,archive,report)
    output.parent.mkdir(parents=True,exist_ok=True)
    lockdir=root.parents[1]/'.tirust-build';lockdir.mkdir(parents=True,exist_ok=True)
    with (lockdir/'stress-report-publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if output.exists() or ledger.exists():
            require(output.exists() and ledger.exists(),'Incomplete previous publication; inspect and restore before retrying')
            if read(ledger).get('schema')==SCHEMA:verify(root,output,ledger)
            else:require(output==root/'benchmark.pdf' and identity(output)==inputs['old39_PDF'] and identity(ledger)==inputs['old39_ledger'],'Existing output is not the frozen39 or a valid timing composite')
        before={str(p):identity(p) if p.exists() else None for p in (output,ledger)}
        with tempfile.TemporaryDirectory(prefix='.timing-report-',dir=output.parent) as name:
            scratch=Path(name);tools=owner.DocumentTools();appendix=report/'timing-supplement.pdf'
            if args.rebuild_supplement:
                typst=shutil.which('typst');require(typst is not None,'Missing Typst')
                require(tools.run([typst,'--version'])['stdout'].startswith('typst 0.15.1 '),'Rebuild requires Typst0.15.1')
                appendix=scratch/'timing-supplement.pdf'
                tools.run([typst,'compile',str(report/'timings.typ'),str(appendix),'--input','include-plots=true','--creation-timestamp','1794355200'])
            merged=scratch/'combined.pdf';tools.run([tools.paths['pdfunite'],str(archive/'benchmark.pdf'),str(appendix),str(merged)])
            proof=prove(owner,tools,archive,report,merged,scratch,inputs['appendix_pages'])
            require(selected(root,archive,report)[1]==inputs,'Source inputs changed while preparing the timing appendix')
            require({str(p):identity(p) if p.exists() else None for p in (output,ledger)}==before,'Existing report changed while preparing this append')
            record={'schema':SCHEMA,'archive':location(root,archive),'report':location(root,report),'selected_inputs':inputs,
                    'builder':identity(__file__),'PDF':identity(merged),'page_proof':proof,'page_proof_sha256':digest(proof),
                    'commands':tools.commands,'rebuild_supplement':args.rebuild_supplement,'performance_acceptance':False,
                    'publication':'Individually atomic renames under exclusive lock, ledger last and exception rollback; not a crash-atomic two-path transaction.',
                    'new_native_physics_evaluations':0}
            owner.publish_pair(merged,encoded(record),output,ledger)
    print(encoded({'PDF':str(output),'sha256':record['PDF']['sha256'],'pages':proof['total_pages'],'new_native_physics_evaluations':0}).decode(),end='')


if __name__=='__main__':main()
