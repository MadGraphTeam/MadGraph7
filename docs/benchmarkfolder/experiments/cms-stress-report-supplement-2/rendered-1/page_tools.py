#!/usr/bin/env python3
"""Append the sealed CMS supplement without rerendering the original report.

Document tools run only when building or explicitly requesting --verify-pages.
They never evaluate amplitudes. File-only --verify-only launches no programs.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
SCHEMA = 'madloop-tirust-composite-report-v1'
ARCHIVE = 'history/report-before-cms-stress-supplement-1'
ARCHIVE_SHA = '3ff7ffb3c19032301c79747b587751795e194ebf6e79c4ac23955303c5864210'
SUPPLEMENT = 'experiments/cms-stress-report-supplement-1/rendered-1'
SUPPLEMENT_SHA = '26ebae5025032f9aa9451da754ce2e4ac0f789fcfd411881fafcc2c5e6559c25'
SOURCE_SHA = 'cdb111b82c60978ffd63a2d1ecd6992b2c9b6ec030d780a409d76931dfde5973'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def identity(path):
    raw = Path(path).read_bytes()
    return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read(path):
    def unique(items):
        out = {}
        for key,value in items:
            require(key not in out,'Duplicate JSON key')
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(Path(path).read_bytes(),object_pairs_hook=unique,parse_constant=invalid)


def inside(root, name):
    require(isinstance(name,str) and name and '\\' not in name,'Invalid relative input path')
    part = PurePosixPath(name)
    require(not part.is_absolute() and '..' not in part.parts and str(part)==name,'Unsafe relative input path')
    path = Path(root)
    for component in part.parts:
        path /= component
        require(not path.is_symlink(),'Symlink report input')
    return path


def packet(path, expected, index_name='index.json'):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path,*path.parents)),'Symlink package root')
    require(identity(path/index_name)['sha256']==expected,'Pinned package index changed')
    index = read(path/index_name)
    actual = {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()}
    require(actual==set(index['files'])|{index_name},'Pinned package inventory changed')
    for name,value in index['files'].items():
        require(identity(inside(path,name))==value,'Pinned package payload changed: '+name)
    return index


def selected(root, archive, supplement):
    base = packet(archive,ARCHIVE_SHA)
    addition = packet(supplement,SUPPLEMENT_SHA)
    require(base['page_count']==35 and addition['page_count']==3,'Unexpected page counts')
    require(addition['source_index_sha256']==SOURCE_SHA,'Supplement has another source')
    require(identity(Path(supplement)/'source-index.json')['sha256']==SOURCE_SHA,'Supplement source seal differs')
    old = read(Path(archive)/'report-build.json')
    require(old['pdf_sha256']==identity(Path(archive)/'benchmark.pdf')['sha256'],'Archived PDF/ledger disagree')
    # The original builder is the sole old input changed by this delivery.
    # Its original bytes stay in the immutable archive, not silently repinned.
    for name,value in old['inputs'].items():
        path = Path(archive)/'build_report.py' if name=='build_report.py' else inside(root,name)
        require(identity(path)==value,'Original report input changed: '+name)
    return {'archived_ledger':identity(Path(archive)/'report-build.json'),
            'old_input_count':len(old['inputs']),'original_pdf':identity(Path(archive)/'benchmark.pdf'),
            'supplement_pdf':identity(Path(supplement)/'supplement.pdf'),
            'source_index_sha256':SOURCE_SHA,'archive_index_sha256':ARCHIVE_SHA,
            'supplement_index_sha256':SUPPLEMENT_SHA}


class DocumentTools:
    def __init__(self):
        self.commands = []
        self.paths = {}
        self.versions = {}
        for name in ('pdfinfo','pdftoppm','pdftotext','pdffonts','pdfunite'):
            path = shutil.which(name)
            require(path is not None,'Missing document tool: '+name)
            self.paths[name] = path
            self.versions[name] = self.run([path,'-v'])['stdout'].strip() or self.commands[-1]['stderr'].strip()

    def run(self, argv, timeout=600):
        result = subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=False)
        record = {'argv':list(map(str,argv)),'returncode':result.returncode,
                  'stdout':result.stdout.decode('utf-8','replace'),'stderr':result.stderr.decode('utf-8','replace')}
        self.commands.append(record)
        require(result.returncode==0,'Document command failed: '+str(argv))
        return record


def geometries(text, count):
    pages = {n:{} for n in range(1,count+1)}
    for line in text.splitlines():
        match = re.fullmatch(r'Page\s+(\d+)\s+(size|rot|MediaBox|CropBox|BleedBox|TrimBox|ArtBox):\s*(.*)',line)
        if match and int(match[1]) in pages:
            pages[int(match[1])][match[2]] = ' '.join(match[3].split())
    require(all(set(page)=={'size','rot','MediaBox','CropBox','BleedBox','TrimBox','ArtBox'}
                for page in pages.values()),'Incomplete page geometry')
    return pages


def text_pages(raw, count):
    parts = raw.decode('utf-8').split('\f')
    if parts and not parts[-1].strip():
        parts.pop()
    require(len(parts)==count,'Extracted text page count differs')
    return [hashlib.sha256(p.encode()).hexdigest() for p in parts]


def font_rows(text):
    lines = text.splitlines()
    require(len(lines)>=2 and lines[0].startswith('name'),'Unexpected font inventory')
    # Ignore only PDF object/generation numbers, which change on merging.
    rows = [line.rsplit(None,2)[0] for line in lines[2:] if line.strip()]
    return sorted(rows)


def inspect_document(tools, pdf, scratch, label, expected_pages):
    from PIL import Image
    pdf, scratch = Path(pdf),Path(scratch)
    prefix = scratch/label
    info = tools.run([tools.paths['pdfinfo'],'-box','-f','1','-l',str(expected_pages),str(pdf)])['stdout']
    match = re.search(r'^Pages:\s+(\d+)\s*$',info,re.M)
    require(match and int(match[1])==expected_pages,'Unexpected document page count')
    geometry = geometries(info,expected_pages)
    text = scratch/(label+'.txt')
    tools.run([tools.paths['pdftotext'],'-layout','-enc','UTF-8',str(pdf),str(text)])
    texts = text_pages(text.read_bytes(),expected_pages)
    tools.run([tools.paths['pdftoppm'],'-r','144','-png',str(pdf),str(prefix)])
    images = sorted(scratch.glob(label+'-*.png'),key=lambda p:int(p.stem.rsplit('-',1)[1]))
    require(len(images)==expected_pages,'Raster page count differs')
    pages = []
    for n,image in enumerate(images,1):
        with Image.open(image) as opened:
            pixels = opened.convert('RGB')
            size = list(pixels.size)
            checksum = hashlib.sha256(pixels.tobytes()).hexdigest()
        fonts = tools.run([tools.paths['pdffonts'],'-f',str(n),'-l',str(n),str(pdf)])['stdout']
        pages.append({'pixel_size':size,'RGB_sha256':checksum,'text_sha256':texts[n-1],
                      'geometry':geometry[n],'fonts':font_rows(fonts)})
        image.unlink()
    return pages


def compare_pages(base, addition, merged):
    require(len(base)==35 and len(addition)==3 and len(merged)==38,'Wrong document lengths')
    require(merged[:35]==base,'Original 35-page visual/text/geometry/font preservation failed')
    require(merged[35:]==addition,'Appended three-page preservation failed')
    return True


def page_proof(tools, archive, supplement, merged, scratch):
    base = inspect_document(tools,Path(archive)/'benchmark.pdf',scratch,'original',35)
    addition = inspect_document(tools,Path(supplement)/'supplement.pdf',scratch,'supplement',3)
    combined = inspect_document(tools,merged,scratch,'combined',38)
    compare_pages(base,addition,combined)
    return {'schema':'madloop-paired-document-page-proof-v1','dpi':144,'colour':'RGB',
            'original_pdf':identity(Path(archive)/'benchmark.pdf'),
            'supplement_pdf':identity(Path(supplement)/'supplement.pdf'),'composite_pdf':identity(merged),
            'original_pages':35,'appended_pages':3,'total_pages':38,'prefix_preserved':True,
            'supplement_preserved':True,'pages':combined,'tool_versions':tools.versions,
            'scope':'Same-version paired raster pixels, per-page text/geometry/fonts; not PDF object-graph equivalence or a fresh annotation/link proof.',
            'scientific_evaluations':0}


def publish_pair(pdf, ledger_bytes, output, ledger, replace=os.replace):
    """Atomic individual renames, exclusive caller lock and exception rollback.

    POSIX cannot atomically replace two independent file paths. The ledger is
    the final commit marker; readers fail closed on any transient hash mismatch.
    """
    output, ledger = Path(output),Path(ledger)
    require(output.parent==ledger.parent,'PDF and ledger must share a publication directory')
    before = {p:p.read_bytes() if p.exists() else None for p in (output,ledger)}
    with tempfile.TemporaryDirectory(prefix='.stress-publish-',dir=output.parent) as name:
        stage = Path(name)
        shutil.copyfile(pdf,stage/'report.pdf')
        (stage/'ledger.json').write_bytes(ledger_bytes)
        for p in (stage/'report.pdf',stage/'ledger.json'):
            with p.open('rb') as stream:
                os.fsync(stream.fileno())
        changed = []
        try:
            replace(stage/'report.pdf',output)
            changed.append(output)
            replace(stage/'ledger.json',ledger)
            changed.append(ledger)
            fd = os.open(output.parent,os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except BaseException:
            for p in reversed(changed):
                if before[p] is None:
                    p.unlink(missing_ok=True)
                else:
                    restore = stage/('restore-'+p.name)
                    restore.write_bytes(before[p])
                    os.replace(restore,p)
            raise


def location(root, path):
    path = Path(path).resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def located(root, value):
    return Path(value) if Path(value).is_absolute() else inside(root,value)


def verify(root, output, ledger, verify_pages=False):
    recorded = read(ledger)
    require(recorded.get('schema')==SCHEMA,'Not a composite stress-report ledger')
    archive,supplement = (located(root,recorded[k]) for k in ('archive','supplement'))
    require(selected(root,archive,supplement)==recorded['selected_inputs'],'Composite inputs differ')
    require(identity(output)==recorded['pdf'],'Composite PDF changed')
    require(identity(Path(__file__))==recorded['builder'],'Composite builder changed')
    require(digest(recorded['page_proof'])==recorded['page_proof_sha256'],'Page-proof record changed')
    proof = recorded['page_proof']
    require(proof['composite_pdf']==recorded['pdf'] and proof['prefix_preserved'] is True
            and proof['supplement_preserved'] is True and len(proof['pages'])==38
            and proof['original_pdf']==recorded['selected_inputs']['original_pdf']
            and proof['supplement_pdf']==recorded['selected_inputs']['supplement_pdf']
            and (proof['original_pages'],proof['appended_pages'],proof['total_pages'])==(35,3,38),
            'Incomplete page proof')
    if verify_pages:
        with tempfile.TemporaryDirectory(prefix='madloop-stress-page-verify-') as name:
            fresh = page_proof(DocumentTools(),archive,supplement,output,Path(name))
            require(fresh['pages']==proof['pages'],'Current document-tool page proof differs from recorded proof')
    return {'verified':True,'mode':'document pages' if verify_pages else 'file hashes and bound page-proof receipt',
            'pages':38,'native_physics_evaluations':0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--supplement',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--verify-pages',action='store_true')
    parser.add_argument('--rebuild-supplement',action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root/'benchmark.pdf').resolve()
    ledger = root/'report-build.json' if output==root/'benchmark.pdf' else output.with_suffix('.build.json')
    require(not args.verify_pages or args.verify_only,'--verify-pages requires --verify-only')
    require(not (args.verify_only and args.rebuild_supplement),'Rebuild and verify are separate operations')
    if args.verify_only:
        print(encoded(verify(root,output,ledger,args.verify_pages)).decode(),end='')
        return
    archive = (args.archive or root/ARCHIVE).resolve()
    supplement = (args.supplement or root/SUPPLEMENT).resolve()
    if output==root/'benchmark.pdf':
        require(archive.is_relative_to(root) and supplement.is_relative_to(root),
                'Canonical publication requires retained report dependencies inside docs/benchmarkfolder')
    inputs = selected(root,archive,supplement)
    output.parent.mkdir(parents=True,exist_ok=True)
    lock_dir = root.parents[1]/'.tirust-build'
    lock_dir.mkdir(parents=True,exist_ok=True)
    with (lock_dir/'stress-report-publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if output.exists() or ledger.exists():
            require(output.exists() and ledger.exists(),'Incomplete previous report publication; inspect and restore the retained original pair')
            if read(ledger).get('schema')==SCHEMA:
                verify(root,output,ledger)
            else:
                require(output==root/'benchmark.pdf' and identity(output)==inputs['original_pdf']
                        and identity(ledger)==inputs['archived_ledger'],
                        'Existing output is not the pinned original report or a valid composite')
        before_publication = {str(p):identity(p) if p.exists() else None for p in (output,ledger)}
        with tempfile.TemporaryDirectory(prefix='.stress-build-',dir=output.parent) as name:
            scratch = Path(name)
            tools = DocumentTools()
            addition = supplement/'supplement.pdf'
            if args.rebuild_supplement:
                typst = shutil.which('typst')
                require(typst is not None,'Missing Typst')
                version = tools.run([typst,'--version'])['stdout'].strip()
                require(version.startswith('typst 0.15.1 '),'Rebuild requires Typst 0.15.1')
                addition = scratch/'supplement.pdf'
                tools.run([typst,'compile',str(supplement/'supplement.typ'),str(addition),
                           '--input','include-pole-plot=true','--creation-timestamp','1794355200'])
            merged = scratch/'combined.pdf'
            tools.run([tools.paths['pdfunite'],str(archive/'benchmark.pdf'),str(addition),str(merged)])
            proof = page_proof(tools,archive,supplement,merged,scratch)
            require(selected(root,archive,supplement)==inputs,'Report inputs changed during rendering')
            require({str(p):identity(p) if p.exists() else None for p in (output,ledger)}==before_publication,
                    'Existing report changed while this append was prepared')
            record = {'schema':SCHEMA,'archive':location(root,archive),'supplement':location(root,supplement),
                      'selected_inputs':inputs,'builder':identity(Path(__file__)),'pdf':identity(merged),
                      'page_proof':proof,'page_proof_sha256':digest(proof),'document_commands':tools.commands,
                      'rebuild_supplement':args.rebuild_supplement,'numerical_campaign_rerun':False,
                      'publication':'Atomic individual renames under exclusive lock with exception rollback; ledger published last. Two paths are not one POSIX atomic transaction.'}
            publish_pair(merged,encoded(record),output,ledger)
    print(encoded({'pdf':str(output),'ledger':str(ledger),'sha256':record['pdf']['sha256'],
                   'pages':38,'native_physics_evaluations':0}).decode(),end='')


if __name__=='__main__':
    main()
