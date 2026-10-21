"""Reject extra tools and incomplete/unalternated loop timing dispatches."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest import mock

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/loop-execution/loop_pair.py'
spec=importlib.util.spec_from_file_location('final_loop_pair',SOURCE)
loop=importlib.util.module_from_spec(spec);spec.loader.exec_module(loop)


def actual_dispatch_shape():
    events=[]
    def stage(cid,phase,source,backend,repetition=None):
        for kind in ('start','complete'):
            events.append({'index':len(events),'case':cid,'phase':phase,'event':kind,
                           'source':source,'backend':backend,'repetition':repetition})
    for cid in loop.CASE_IDS:
        for source,backend in loop.TARGETS:stage(cid,'prepare',source,backend)
        for source,backend in loop.TARGETS:
            stage(cid,'initialize',source,backend)
            stage(cid,'calibration',source,backend,'calibration')
        for repetition in range(7):
            order=loop.TARGETS if repetition%2==0 else tuple(reversed(loop.TARGETS))
            for source,backend in order:stage(cid,'measurement',source,backend,repetition)
        for source,backend in reversed(loop.TARGETS):stage(cid,'cleanup',source,backend)
    return events


class TestFinalLoopPairDispatch(unittest.TestCase):
    def test_complete_explicit_pair_protocol(self):
        loop.check_dispatch(actual_dispatch_shape(),loop.CASE_IDS)

    def test_cuttools_and_collier_cannot_be_added(self):
        for backend in (1,7):
            with self.subTest(backend=backend):
                events=actual_dispatch_shape();events[0]['backend']=backend
                with self.assertRaisesRegex(ValueError,'Unexpected dispatch target'):
                    loop.check_dispatch(events,loop.CASE_IDS)

    def test_repeated_even_order_on_an_odd_pair_is_rejected(self):
        events=actual_dispatch_shape()
        group=[i for i,e in enumerate(events) if e['case']==loop.CASE_IDS[0] and
               e['phase']=='measurement' and e['repetition']==1]
        for i in group:
            e=events[i];e['source']='baseline' if e['source']=='candidate' else 'candidate'
            e['backend']=6 if e['source']=='baseline' else 8
        with self.assertRaisesRegex(ValueError,'complete adjacent alternating'):
            loop.check_dispatch(events,loop.CASE_IDS)


class TestFinalLoopSourceAndOwnership(unittest.TestCase):
    def test_replaced_report_points_fail_even_if_the_record_hash_is_refreshed(self):
        temporary=ROOT/'.ninjars-build/tmp';temporary.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as directory:
            copied,authorized=Path(directory)/'copied',Path(directory)/'authorized'
            original=b'fixed point bytes\n';authorized.write_bytes(original);copied.write_bytes(original)
            with mock.patch.dict(loop.POINT_SHA256,{'unit-case':hashlib.sha256(original).hexdigest()}):
                loop.check_points('unit-case',copied,authorized)
                copied.write_bytes(b'replaced point bytes\n')
                with self.assertRaisesRegex(ValueError,'authorized fixed100'):
                    loop.check_points('unit-case',copied,authorized)
                authorized.write_bytes(copied.read_bytes())
                with self.assertRaisesRegex(ValueError,'authorized fixed100'):
                    loop.check_points('unit-case',copied,authorized)

    def test_changed_numerator_cannot_be_hidden_in_a_refreshed_manifest(self):
        temporary=ROOT/'.ninjars-build/tmp';temporary.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as directory:
            path=Path(directory)/'worker.f';prefix='ML5_0_';case={'nfinal':1}
            fresh=(ROOT/'tools/ninjars_benchmark_driver.f').read_text().replace('@PREFIX@',prefix).replace(
                '@NEXTERNAL@','3').replace('@NBORN@','0').encode()
            worker=(ROOT/'tools/ninjars_benchmark_worker.f').read_text().replace('@PREFIX@',prefix).replace(
                '@NEXTERNAL@','3').replace('@NBORN@','0').encode();path.write_bytes(worker)
            driver='SubProcesses/P0_gg_h/check_sa.f';numerator='SubProcesses/P0_gg_h/loop_num.f'
            files={driver:{'sha256':hashlib.sha256(fresh).hexdigest(),'bytes':len(fresh)},
                   numerator:{'sha256':'1'*64,'bytes':31}}
            anchor={'compiled_manifest':{'files':files},'compilation':{'driver_sha256':hashlib.sha256(fresh).hexdigest()}}
            changed=copy.deepcopy(files);changed[driver]={'sha256':hashlib.sha256(worker).hexdigest(),'bytes':len(worker)}
            build={'driver_sha256':hashlib.sha256(worker).hexdigest()}
            manifest={'files':changed,'files_sha256':loop.bench.content_digest(changed)}
            loop.check_worker_delta(case,anchor,build,manifest,path)
            changed[numerator]['sha256']='2'*64;manifest['files_sha256']=loop.bench.content_digest(changed)
            with self.assertRaisesRegex(ValueError,'exact CLI sources'):
                loop.check_worker_delta(case,anchor,build,manifest,path)

    def test_missing_or_mismatched_terminal_process_receipt_is_rejected(self):
        spawn={'event':'spawn','pid':12345,'start_ticks':555,'pgid':12345}
        reaped={**spawn,'event':'reaped','reaped':True,'returncode':0}
        loop.check_ownership([spawn,reaped])
        for events in ([spawn],[spawn,{**reaped,'start_ticks':556}],
                       [spawn,{**reaped,'returncode':-15}]):
            with self.subTest(events=events),self.assertRaises(ValueError):
                loop.check_ownership(events)

    def test_unknown_live_start_identity_cannot_claim_cleanup(self):
        with self.assertRaisesRegex(ValueError,'unknown live owned identity'):
            loop.check_ownership([{'event':'spawn','pid':12345,'pgid':12345,'start_ticks':None}])

    def test_missing_cleanup_is_not_a_complete_series(self):
        events=actual_dispatch_shape()[:-2]
        with self.assertRaisesRegex(ValueError,'complete adjacent alternating'):
            loop.check_dispatch(events,loop.CASE_IDS)

    def test_duplicate_or_missing_dispatch_index_is_rejected(self):
        events=actual_dispatch_shape();events[1]['index']=0
        with self.assertRaisesRegex(ValueError,'indices are missing or duplicated'):
            loop.check_dispatch(events,loop.CASE_IDS)

    def test_injected_complete_initialization_pair_is_rejected(self):
        events=actual_dispatch_shape();events.extend(copy.deepcopy(events[:2]))
        for i,event in enumerate(events):event['index']=i
        with self.assertRaisesRegex(ValueError,'complete adjacent alternating'):
            loop.check_dispatch(events,loop.CASE_IDS)


if __name__=='__main__':unittest.main()
