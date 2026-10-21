"""Prove the update alias leaves the retained generation paths unchanged."""
import ast
import copy
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PRIOR=HERE.parent/'regressions'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(node):
    return ast.dump(node,include_attributes=False)

class InstallerAliasToPrior(ast.NodeTransformer):
    def __init__(self):
        self.aliases=0
        self.messages=0

    def visit_Call(self,node):
        self.generic_visit(node)
        if (isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument'
                and [a.value for a in node.args if isinstance(a,ast.Constant)]==['--update','--force']):
            keywords={k.arg:k.value for k in node.keywords}
            assert isinstance(keywords['dest'],ast.Constant) and keywords['dest'].value=='force'
            assert isinstance(keywords['action'],ast.Constant) and keywords['action'].value=='store_true'
            assert keywords['help'].value==(
                "rebuild using this MadGraph checkout's pinned dependencies; "
                'replace an existing installation after validation')
            node.args=node.args[1:]
            node.keywords=[k for k in node.keywords if k.arg!='dest']
            for k in node.keywords:
                if k.arg=='help':
                    k.value=ast.Constant('replace an existing NinjaRS installation after validation')
            self.aliases+=1
        return node

    def visit_Constant(self,node):
        values={
            "Cannot read NinjaRS installation metadata in %s: %s. Run 'install ninjars --update'.":
                "Cannot read NinjaRS installation metadata in %s: %s. Run 'install ninjars'.",
            "Unsupported NinjaRS bridge ABI in %s; run 'install ninjars --update'.":
                "Unsupported NinjaRS bridge ABI in %s; run 'install ninjars --force'.",
            "Invalid NinjaRS native link metadata in %s; run 'install ninjars --update'.":
                'Invalid NinjaRS native link metadata in %s',
            "Missing or invalid NinjaRS compiler-runtime link metadata in %s; run 'install ninjars --update'.":
                "Missing or invalid NinjaRS compiler-runtime link metadata in %s; run 'install ninjars --force'.",
            "Incomplete NinjaRS installation: run 'install ninjars --update'.":
                "Incomplete NinjaRS installation: run 'install ninjars --force'.",
            "NinjaRS is already installed in %s; use --update to rebuild with this checkout's pinned dependencies.":
                'NinjaRS is already installed in %s; use --force to rebuild.',
        }
        if isinstance(node.value,str) and node.value in values:
            self.messages+=1
            return ast.Constant(values[node.value])
        return node

def check():
    prior_installer=HERE/'before-update-ninjars_install.py'
    assert sha(prior_installer)=='dfc74ce517f45fdd05e20e52a1f79954a80f19b444bb271ff4337289d782a743'
    old=ast.parse(prior_installer.read_text())
    current=ast.parse((ROOT/'madgraph/various/ninjars_install.py').read_text())
    transformer=InstallerAliasToPrior()
    normalized=transformer.visit(copy.deepcopy(current))
    assert (transformer.aliases,transformer.messages)==(1,6)
    assert dump(normalized)==dump(old)
    revision=json.loads((HERE/'summary.json').read_text())['tested_revision']
    prior_interface_bytes=subprocess.check_output(['git','show',f'{revision}:madgraph/interface/madgraph_interface.py'],cwd=ROOT)
    assert hashlib.sha256(prior_interface_bytes).hexdigest()=='ce879b2eaf4f569ed02bbfa125a1f24bf00c10c87c10eb3552fc63ed0933a001'
    old_interface=ast.parse(prior_interface_bytes.decode())
    new_interface=ast.parse((ROOT/'madgraph/interface/madgraph_interface.py').read_text())
    old_help=[node for node in ast.walk(old_interface) if isinstance(node,ast.FunctionDef) and node.name=='help_install']
    new_help=[node for node in ast.walk(new_interface) if isinstance(node,ast.FunctionDef) and node.name=='help_install']
    assert len(old_help)==len(new_help)==1
    old_help=old_help[0]
    new_help=new_help[0]
    replacement=ast.parse('''logger.info("     --update              Rebuild with this checkout's committed dependency pins (after git pull).")
logger.info("                           Replace the installation after DP and QP smoke tests pass.")
logger.info("     --force               Equivalent backward-compatible alias for --update.")
''').body
    assert [dump(s) for s in new_help.body[-3:]]==[dump(s) for s in replacement]
    assert [dump(s) for s in new_help.body[:-3]]==[dump(s) for s in old_help.body[:-1]]
    new_help.body=copy.deepcopy(old_help.body)
    assert dump(new_interface)==dump(old_interface)
    authority=json.loads(gzip.decompress((PRIOR/'source-hashes.json.gz').read_bytes()))['repository_files']
    changes={p:{'before':d,'after':sha(ROOT/p)} for p,d in authority.items() if sha(ROOT/p)!=d}
    expected={'madgraph/various/ninjars_install.py','madgraph/interface/madgraph_interface.py',
        'tests/unit_tests/various/test_ninjars_install.py',
        'tests/unit_tests/various/test_ninjars_integration_boundaries.py',
        'vendor/ninjars/UPSTREAM.md'}
    assert set(changes)==expected,changes
    production_changes={p for p in changes if p.startswith(('madgraph/','aloha/','Template/'))}
    assert production_changes=={'madgraph/various/ninjars_install.py','madgraph/interface/madgraph_interface.py'}
    record={'schema':'ninjars-update-generation-reuse-review-v1',
        'passed':True,'prior_manifest_sha256':sha(PRIOR/'manifest.json'),
        'changed_prior_bound_files':changes,
        'installer_ast_change_scope':{'one_new_update_alias':True,'force_destination_preserved':True,
            'six_diagnostic_strings_changed':True,'all_other_module_ast_identical':True},
        'madgraph_interface_ast_change_scope':{'help_install_three_logging_lines_replace_one':True,
            'all_other_module_ast_identical':True},
        'all_fortran_templates_exporters_fks_modules_models_and_fixtures_in_prior_scope_unchanged':True,
        'prior_scope_note':'Only files actually inventoried by the original software capsule are claimed. Scientific model/runtime provenance is separate.',
        'reused_suite_tests':{'exporters-2':2,'fks-1':19},
        'reused_known_skips':1,
        'new_engine_calls':0,
        'limits':'This proves source-level compatibility for retained software generation observations; it does not replay numerical physics, stability, convergence or performance.'}
    return record

if __name__=='__main__':
    print(json.dumps(check(),sort_keys=True))
