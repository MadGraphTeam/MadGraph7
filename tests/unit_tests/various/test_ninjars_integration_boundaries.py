"""Failure/precision boundaries across installed capability, export and reports.

These exercise publication rollback and consumer behavior without compiling or
calling a reduction engine. Numerical ABI/precision tests live in the existing
Fortran and Rust suites.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from xml.dom.minidom import parseString

from madgraph import MadGraph5Error
from madgraph.core import base_objects
from madgraph.loop import loop_exporters
from madgraph.madevent.sum_html import RunStatistics
from madgraph.various import ninjars_install, process_checks


class TestNinjaRSIntegrationBoundaries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.lib = self.work/'installed/lib'
        self.lib.mkdir(parents=True)
        self.config = {'abi_version':ninjars_install.ABI_VERSION,
            'native_libs':['-lgmp','-lmpfr'], 'link_before':['-lgcc_s'],
            'link_order_version':1, 'quad_precision':True}

    def metadata(self, **changes):
        config = dict(self.config, **changes)
        (self.lib/ninjars_install.CONFIG_NAME).write_text(json.dumps(config))
        return config

    def exporter(self, available):
        exporter = object.__new__(loop_exporters.LoopProcessOptimizedExporterFortranSA)
        exporter.tir_available_dict = dict.fromkeys(exporter.all_tir,False)
        exporter.tir_available_dict['ninjars'] = available
        exporter.ninjars_dir = str(self.lib)
        return exporter

    def context(self, exporter):
        with mock.patch.object(loop_exporters.LoopProcessExporterFortranSA,
                               'get_context',return_value={'caller_setting':17}):
            return exporter.get_context(None)

    def test_absent_backend_does_not_read_stale_precision_metadata(self):
        (self.lib/ninjars_install.CONFIG_NAME).write_text('{broken JSON')
        exporter = self.exporter(False)
        with mock.patch.object(ninjars_install,'read_config',side_effect=AssertionError('absent backend read')):
            context = self.context(exporter)
        self.assertFalse(context['ninjars_available'])
        self.assertFalse(context['ninjars_supports_quad_prec'])
        self.assertEqual(context['caller_setting'],17)

    def test_export_requires_literal_quad_capability_and_archive(self):
        exporter = self.exporter(True)
        for advertised, expected in [(True,True),(False,False),('true',False),(1,False),(None,False)]:
            self.metadata(quad_precision=advertised)
            (self.lib/'libninjars.a').write_bytes(b'archive')
            with self.subTest(advertised=advertised):
                self.assertEqual(self.context(exporter)['ninjars_supports_quad_prec'],expected)
        (self.lib/'libninjars.a').unlink()
        self.metadata()
        self.assertFalse(self.context(exporter)['ninjars_supports_quad_prec'])

    def test_available_backend_rejects_incompatible_metadata_instead_of_claiming_qp(self):
        (self.lib/'libninjars.a').write_bytes(b'archive')
        for changes in ({'abi_version':2}, {'native_libs':['-lgmp\nBAD=1']},
                        {'link_before':[]}, {'link_before':['-lgcc_s\rBAD=1']},
                        {'link_order_version':0}):
            self.metadata(**changes)
            with self.subTest(changes=changes), self.assertRaises(MadGraph5Error):
                self.context(self.exporter(True))

    def test_publication_failure_restores_complete_previous_install(self):
        prefix = self.work/'HEPTools'
        previous = prefix/'ninjars'
        (previous/'lib').mkdir(parents=True)
        (previous/'lib/libninjars.a').write_bytes(b'old archive')
        (previous/'lib/mninjars.mod').write_bytes(b'old module')
        (previous/'install.log').write_bytes(b'old validated log')
        before = {str(p.relative_to(previous)):p.read_bytes() for p in previous.rglob('*') if p.is_file()}
        bridge = self.work/'vendor/ninjars'
        bridge.mkdir(parents=True)
        for name in ('Cargo.lock','Cargo.toml','smoke.f90'):
            (bridge/name).write_text('fixture\n')
        (bridge/'Cargo.lock').write_text('version = 4\npackage = []\n')
        calls = []
        def build(command,cwd,log,environment):
            calls.append(command)
            log.write_text('build and smoke successful\n')
            if command[0] == 'make':
                installed = Path(next(x[7:] for x in command if x.startswith('PREFIX=')))
                (installed/'lib').mkdir(parents=True)
                for name,content in {'libninjars.a':'new archive','mninjars.mod':'new module',
                    'ninjars-native-libs.txt':'-lgmp -lmpfr','ninjars-link-before.txt':'-lgcc_s'}.items():
                    (installed/'lib'/name).write_text(content)
        replace = ninjars_install.os.replace
        def fail_publish(source,destination):
            if Path(source).name == 'install' and Path(destination) == previous:
                raise OSError('simulated publication failure')
            return replace(source,destination)
        with mock.patch.object(ninjars_install.shutil,'which',return_value='/bin/tool'), \
                mock.patch.object(ninjars_install,'_run',side_effect=build), \
                mock.patch.object(ninjars_install.os,'replace',side_effect=fail_publish):
            with self.assertRaisesRegex(OSError,'publication failure'):
                ninjars_install.install(self.work,prefix,['--force'])
        self.assertEqual(len(calls),3)  # Both compilation and installed smoke ran.
        self.assertEqual(before,{str(p.relative_to(previous)):p.read_bytes() for p in previous.rglob('*') if p.is_file()})
        self.assertFalse((prefix/'lib').exists())

    def test_partial_existing_install_requires_update_without_starting_build(self):
        prefix = self.work/'HEPTools'
        directory = prefix/'ninjars/lib'
        directory.mkdir(parents=True)
        (directory/ninjars_install.CONFIG_NAME).write_text(json.dumps(self.config))
        (directory/'libninjars.a').write_bytes(b'existing archive without module')
        with mock.patch.object(ninjars_install,'_run') as run:
            with self.assertRaisesRegex(MadGraph5Error,'Incomplete.*--update'):
                ninjars_install.install(self.work,prefix)
        run.assert_not_called()
        self.assertEqual((directory/'libninjars.a').read_bytes(),b'existing archive without module')

    def test_explicit_unavailable_backend_has_actionable_error_before_generation(self):
        evaluator = object.__new__(process_checks.LoopMatrixElementTimer)
        options = {'reuse':True, 'npoints':1, 'split_orders':-1}
        configurations = ({}, {'ninjars_dir':None},
                          {'ninjars_dir':str(self.work/'missing')},
                          {'ninjars_dir':str(self.lib)})
        for configuration in configurations:
            evaluator.tir_dir = configuration
            selection = {'MLReductionLib':[8]}
            with self.subTest(configuration=configuration), self.assertRaisesRegex(
                    MadGraph5Error, "NinjaRS.*unavailable.*install ninjars"):
                evaluator.check_matrix_element_stability(
                    base_objects.Process(), options=options, MLOptions=selection)
            self.assertEqual(selection['MLReductionLib'],[8])

    def test_failed_smoke_then_retry_publishes_only_validated_install(self):
        prefix = self.work/'HEPTools'
        bridge = self.work/'vendor/ninjars'
        bridge.mkdir(parents=True)
        for name in ('Cargo.lock','Cargo.toml','smoke.f90'):
            (bridge/name).write_text('fixture\n')
        (bridge/'Cargo.lock').write_text('version = 4\npackage = []\n')
        smoke_attempts, build_directories = [], []
        def build(command,cwd,log,environment):
            log.write_text('retained installer attempt\n')
            if command[0] == 'make':
                build_directories.append(Path(cwd))
                installed = Path(next(x[7:] for x in command if x.startswith('PREFIX=')))
                (installed/'lib').mkdir(parents=True)
                for name,content in {'libninjars.a':'validated archive','mninjars.mod':'module',
                    'ninjars-native-libs.txt':'-lgmp -lmpfr','ninjars-link-before.txt':'-lgcc_s'}.items():
                    (installed/'lib'/name).write_text(content)
            elif command[0].endswith('/smoke'):
                smoke_attempts.append(command[0])
                if len(smoke_attempts) == 1:
                    raise MadGraph5Error('first installed precision smoke failed')
        with mock.patch.object(ninjars_install.shutil,'which',return_value='/bin/tool'), \
                mock.patch.object(ninjars_install,'_run',side_effect=build):
            with self.assertRaisesRegex(MadGraph5Error,'precision smoke failed'):
                ninjars_install.install(self.work,prefix)
            self.assertFalse((prefix/'ninjars').exists())
            self.assertFalse((prefix/'lib').exists())
            failed_stage = build_directories[0].parent
            self.assertTrue((failed_stage/'install.log').is_file())
            result = ninjars_install.install(self.work,prefix)
        self.assertEqual(len(smoke_attempts),2)
        self.assertNotEqual(build_directories[0],build_directories[1])
        self.assertTrue(failed_stage.exists())  # Failure diagnostics survive retry.
        self.assertFalse(build_directories[1].parent.exists())
        self.assertEqual(result,str(prefix/'ninjars/lib'))
        self.assertTrue(ninjars_install.supports_quad_precision(result))
        self.assertEqual((prefix/'lib/libninjars.a').read_text(),'validated archive')

    def test_combined_source_overrides_are_isolated_before_cargo_resolution(self):
        ninja, masters = self.work/'ninja-source', self.work/'masters-source'
        for root, names in ((ninja,('Cargo.toml','crates/ninjars/Cargo.toml',
                                   'crates/ninjars-symbolica/Cargo.toml')),
                            (masters,('Cargo.toml',))):
            for name in names:
                path = root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('original '+name+'\n')
            (root/'target').mkdir()
            (root/'target/cached').write_text('never copy')
            (root/'.git').mkdir()
        before = {str(path):path.read_bytes() for root in (ninja,masters)
                  for path in root.rglob('*') if path.is_file()}
        stage = self.work/'stage'
        bridge = stage/'bridge'
        bridge.mkdir(parents=True)
        options = ninjars_install.build_parser().parse_args(
            ['--source',str(ninja),'--oneloop-source',str(masters)])
        with mock.patch.object(ninjars_install.subprocess,'check_output',
                               side_effect=['1'*40+'\n','2'*40+'\n']):
            provenance = ninjars_install._prepare_overrides(bridge,stage,options)
        config = (bridge/'.cargo/config.toml').read_text()
        self.assertIn('[patch."https://github.com/alphal00p/ninjars"]',config)
        self.assertIn('[patch."https://github.com/alphal00p/oneloopmaster.git"]',config)
        for name in ('ninjars','oneloopmaster'):
            copied = stage/'sources'/name
            self.assertIn(str(copied),config)
            self.assertFalse((copied/'target').exists())
            (copied/'Cargo.toml').write_text('changed only in isolated build\n')
        self.assertEqual(provenance['ninjars']['revision'],'1'*40)
        self.assertEqual(provenance['oneloopmaster']['revision'],'2'*40)
        self.assertEqual(before,{path:Path(path).read_bytes() for path in before})

    @staticmethod
    def statistics(pjfry,ninja_qp,nrs_dp,nrs_qp,legacy=False):
        count = pjfry+ninja_qp+nrs_dp+nrs_qp
        extra = '' if legacy else '<ninjars_dp_usage>%s</ninjars_dp_usage><ninjars_qp_usage>%s</ninjars_qp_usage>' % (nrs_dp,nrs_qp)
        xml = '''<statistics><u_return_code>0,0,%s,0,0,0,0,0,%s,0</u_return_code>
          <t_return_code>0,%s,%s,0,0</t_return_code><h_return_code>0,0,%s,%s,0</h_return_code>
          <average_time>0.001</average_time><cumulated_time>%s</cumulated_time>
          <max_prec>1e-8</max_prec><min_prec>1e-12</min_prec><n_evals>%s</n_evals>%s</statistics>''' % (
            pjfry+nrs_qp,ninja_qp+nrs_dp,pjfry+nrs_dp,ninja_qp+nrs_qp,
            pjfry+nrs_dp,ninja_qp+nrs_qp,count*.001,count,extra)
        value = RunStatistics()
        value.load_statistics(parseString(xml).documentElement)
        return value

    def test_mixed_legacy_and_contextual_status_aggregation_is_order_independent(self):
        shards = [(9,5,0,0,True),(0,0,12,4,False),(3,2,7,6,False)]
        keys = ['PJFry_usage','Ninja_QP_usage','NinjaRS_DP_usage','NinjaRS_QP_usage']
        for order in (shards,list(reversed(shards)),[shards[1],shards[0],shards[2]]):
            total = self.statistics(*order[0])
            for shard in order[1:]:
                total.aggregate_statistics(self.statistics(*shard))
            self.assertEqual([total[key] for key in keys],[12,7,19,10])


if __name__ == '__main__':
    unittest.main()
