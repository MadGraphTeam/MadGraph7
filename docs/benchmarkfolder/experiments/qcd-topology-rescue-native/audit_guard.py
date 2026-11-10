#!/usr/bin/env python3
"""Bind recorded refusals to the historical source; no native execution."""
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path
import json
import re

from common import ROOT, HERE, sha, rel, require


def main():
    output = HERE/'analysis1/guard-source-audit.json'
    excerpts = HERE/'analysis1/guard-source-excerpts.md'
    require(not output.exists() and not excerpts.exists(), 'Preserve the previous audit.')
    old_audit_path = ROOT/'docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/guard-audit/audit.json'
    old_audit = json.loads(old_audit_path.read_text())
    install_path = ROOT/'docs/benchmarkfolder/qualification/tirust-install/check-installation.json'
    install = json.loads(install_path.read_text())
    expected_dso = '49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12'
    require(install['actual_artifact_sha256']['libmadloop_tirust.so'] == expected_dso, 'Historical DSO binding changed.')
    source_specs = {
        'historical_bridge': ('.tirust-build/bridge-poles/src/lib.rs', [(20,29),(43,52),(79,87),(152,184),(346,354),(490,505)]),
        'patched_executor': ('.tirust-build/pole-development/crates/iregi-rust/src/direct/exec.rs',
                             [(404,426),(610,627),(1458,1475),(1510,1526),(1625,1695),(3439,3493)]),
        'patched_plan': ('.tirust-build/pole-development/crates/iregi-rust/src/direct/plan.rs',
                         [(230,234),(698,735),(749,821),(879,909)]),
        'patched_numeric': ('.tirust-build/pole-development/crates/iregi-rust/src/direct/numeric.rs',[(269,290),(341,354)]),
        'patched_direct_api': ('.tirust-build/pole-development/crates/iregi-rust/src/direct.rs',[(188,220)]),
        'patched_types': ('.tirust-build/pole-development/crates/iregi-rust/src/types.rs',[(175,195)]),
    }
    identities, text = {}, []
    for name, (path, ranges) in source_specs.items():
        p = ROOT/path
        digest = sha(p)
        if name in old_audit['source_identities']:
            require(digest == old_audit['source_identities'][name]['sha256'], 'Earlier source audit mismatch: '+name)
        if name == 'historical_bridge':
            expected = install['production_source_validation']['bridge_sources_sha256']['src/lib.rs']
            require(digest == expected, 'Installed bridge source binding mismatch.')
        elif name in ('patched_executor', 'patched_numeric', 'patched_direct_api'):
            short = path.split('/pole-development/')[1]
            expected = install['production_source_validation']['unchanged_production_source_sha256'][short]
            require(digest == expected, 'Installed core source binding mismatch.')
        identities[name] = {'path':path, 'sha256':digest, 'bytes':p.stat().st_size, 'line_ranges':ranges}
        lines = p.read_text().splitlines()
        text.append(f'## {name}\n\n`{path}`\n\nSHA256 `{digest}`\n')
        for a,b in ranges:
            text.append('```text\n'+'\n'.join(f'{i}: {lines[i-1]}' for i in range(a,b+1))+'\n```\n')
    bridge = (ROOT/source_specs['historical_bridge'][0]).read_text()
    mp = bridge[bridge.index('    fn mp('):bridge.index('    fn mp_records(')]
    require('workspace(Float::new(MP_BITS))' in mp and 'set_input_precision(113)' in mp and
            'set_stability_threshold(Float::with_val(MP_BITS, 2).pow(136).inv())' in mp and
            'enable_' not in mp, 'Historical full-QP policy changed.')
    gram_path = HERE/'analysis1/matrices/topology-c/gram.json'
    grams = json.loads(gram_path.read_text())
    rows = []
    with localcontext() as context:
        context.prec = 100
        threshold = Decimal(2)**-136
        for g in grams:
            message = g['native_error']
            parsed = re.fullmatch(r'TIRust QP: Unstable\("ill-conditioned Gram matrix for (\[[0-9, ]+\]): conditioning ([0-9.e+-]+)"\)', message)
            require(parsed and json.loads(parsed[1]) == g['members_zero_based'] and parsed[2] == g['captured_condition'], 'Unexpected native refusal.')
            n, rank, size = g['n'], g['rank'], len(g['members_zero_based'])
            require(n in (4,5) and rank in (3,4) and size in (3,4), 'Unhandled plan domain.')
            condition = Decimal(g['captured_condition'])
            # No dotted propagators, dimension shifts, or top-child boost.
            # Relations propagate moment rank without incrementing it; all
            # possible coefficient ranks are <= the original tensor rank.
            feasible = [r for r in range(1,rank+1) if condition**r < threshold]
            # Every reported PV box has coefficient rank R here. N4 top:
            # moment_need=R. N5: absent relations, DD=min(R,5)=R; with
            # relations, relation_for(top) propagates unchanged moment R
            # to children before the DD path. A contained box relation is
            # also contained in its N5 parent (only size>=6 is excluded).
            if size == 4:
                feasible = [r for r in feasible if r == rank]
            require(feasible, 'Printed condition cannot satisfy source guard.')
            rows.append({
                'ctmode':g['ctmode'], 'group':g['group'], 'n':n, 'rank':rank,
                'reported_subset_zero_based':g['members_zero_based'],
                'native_error_verbatim':message, 'printed_unpowered_condition':str(condition),
                'subset_coefficient_rank_possibilities':feasible,
                'subset_rank_uniquely_deduced':len(feasible)==1,
                'rank_is_runtime_observed':False,
                'powered_guard_values':{str(r):str(condition**r) for r in feasible},
                'powered_guard_over_threshold':{str(r):str(condition**r/threshold) for r in feasible},
                'all_possible_exponents_refuse':all(condition**r < threshold for r in feasible),
                'supplied_determinant_nonzero':not g['variants']['supplied']['determinant_is_zero'],
                'cartesian_determinant_nonzero':not g['variants']['exact_cartesian']['determinant_is_zero'],
                'source_rank_reason':('PV box has coefficient rank R: direct top moment, rank-R DD box, or unchanged rank through a parent relation.' if size==4 else
                    'Unrecorded relation profiles can raise a pinch rank; fixed gate inequality plus coefficient_rank<=R bounds the possibilities.')})
    require(len(rows)==51 and all(r['supplied_determinant_nonzero'] and r['cartesian_determinant_nonzero'] for r in rows), 'Unexpected remaining failures.')
    bindings = {rel(p):sha(p) for p in (Path(__file__),old_audit_path,install_path,gram_path)}
    bindings.update({e['path']:e['sha256'] for e in identities.values()})
    result = {
        'schema':1,'native_executed':False,'physics_accepted':False,
        'historical_library_sha256':expected_dso,
        'policy':{'carrier':'symbolica::domains::float::Float (MPFR)', 'input_bits':113,'working_bits':256,
                  'workspace':'DirectWorkspace<Float>, full finite mode, DirectMasters::Native',
                  'threshold_expression':'Float::with_val(256,2).pow(136).inv()',
                  'threshold_decimal':str(threshold),'threshold_exponent':-136,
                  'small_gram_rescue_enabled':False,'partial_double_double_rescue_enabled':False,
                  'internal_method_choice_or_precision_ladder_enabled':False,
                  'denominator_relation_variants_enabled':True,'lightlike_bubble_variants_enabled':True,
                  'dimension_shift':0,'propagator_powers':'all1','top_child_boost':0,
                  'condition_formula':'min(abs(det(Z))^(1/m)/scale, abs(det(Z))/product(row_norm_2))',
                  'decision':'condition^subset.coefficient_rank < threshold',
                  'error_prints':'condition before exponentiation',
                  'scale':'max(abs(m0_squared), abs(f_i), abs(Z_ij))',
                  'status_path':'GeometryFailure::Singular attempts relation_support; if unavailable/already present, LaneStatus::Unstable; bridge status1.'},
        'source_identities':identities,'bindings':bindings,'calls':rows,
        'summary':{'calls':len(rows),'unique_exponent_calls':sum(r['subset_rank_uniquely_deduced'] for r in rows),
                   'exponent_possibility_counts':dict(Counter(','.join(map(str,r['subset_coefficient_rank_possibilities'])) for r in rows)),
                   'all_reported_conditions_are_unpowered':True,'all_powered_candidates_below_gate':True,
                   'all_supplied_and_cartesian_determinants_nonzero':True},
        'limitations':[
            'Relation profiles and subset.coefficient_rank were not instrumented. Possibility sets are source/data deductions, not runtime observations.',
            'The source condition is a heuristic. A guard refusal does not prove that a tensor is inaccurate, or that more working precision is necessary.',
            'The fixed threshold is independent of MP_BITS. Raising working precision alone does not automatically relax it.',
            'No threshold change, higher-precision replay, native evaluation, or production change was made.',
            'No finite/single-pole truth is inferred from same-backend direction agreement or this source audit.']}
    excerpts.write_text('\n'.join(text))
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result['summary'],sort_keys=True))


if __name__=='__main__':
    main()
