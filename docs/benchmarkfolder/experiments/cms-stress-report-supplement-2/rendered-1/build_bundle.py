#!/usr/bin/env python3
"""Join the separately rendered chronology pages, preserving each page."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-index-sha256',required=True)
    args=parser.parse_args()
    raw=(HERE/'source-index.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=args.source_index_sha256:raise ValueError('Source index changed')
    index=json.loads(raw)
    for name,expected in index['files'].items():
        source=(HERE/name).read_bytes()
        if {'sha256':hashlib.sha256(source).hexdigest(),'bytes':len(source)}!=expected:raise ValueError('Source changed: '+name)
    helper=HERE/'page_tools.py'
    if hashlib.sha256(helper.read_bytes()).hexdigest()!='cddab2f6055b96a1524713d5d624c9a2a67219482933635f7bb7ead985907ff3':raise ValueError('Proof helper changed')
    spec=importlib.util.spec_from_file_location('frozen_document_proof',helper)
    tools_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools_module)
    tools_module.packet(HERE/'earlier','26ebae5025032f9aa9451da754ce2e4ac0f789fcfd411881fafcc2c5e6559c25')
    tools_module.packet(HERE/'later','19bfe2e61cb5077db5bd833425b148afbe39c021727e911c8fe5a0e4b501fc68')
    if (HERE/'supplement.pdf').exists():raise FileExistsError('Fresh four-page output required')
    tools=tools_module.DocumentTools()
    tools.run([tools.paths['pdfunite'],str(HERE/'earlier/supplement.pdf'),str(HERE/'later/matched-page.pdf'),str(HERE/'supplement.pdf')])
    with tempfile.TemporaryDirectory(prefix='madloop-four-page-proof-') as name:
        scratch=Path(name)
        earlier=tools_module.inspect_document(tools,HERE/'earlier/supplement.pdf',scratch,'earlier',3)
        later=tools_module.inspect_document(tools,HERE/'later/matched-page.pdf',scratch,'later',1)
        combined=tools_module.inspect_document(tools,HERE/'supplement.pdf',scratch,'combined',4)
        if combined[:3]!=earlier or combined[3:]!=later:raise ValueError('Four-page preservation proof failed')
    proof={'schema':'madloop-four-page-supplement-preservation-v1','passed':True,
           'source_index_sha256':args.source_index_sha256,'page_count':4,'dpi':144,
           'earlier_three_pages_preserved':True,'later_page_preserved':True,'pages':combined,
           'tool_versions':tools.versions,'commands':tools.commands,
           'PDF':tools_module.identity(HERE/'supplement.pdf'),
           'scope':'Paired RGB pixels, text, geometry and fonts; no object-graph or annotation/link claim',
           'native_physics_evaluations':0,'canonical_or_tracked_writes':False}
    (HERE/'four-page-proof.json').write_bytes(tools_module.encoded(proof))
    print(json.dumps({'passed':True,'pages':4,'PDF_sha256':proof['PDF']['sha256'],'commands':len(tools.commands)}))


if __name__=='__main__':main()
