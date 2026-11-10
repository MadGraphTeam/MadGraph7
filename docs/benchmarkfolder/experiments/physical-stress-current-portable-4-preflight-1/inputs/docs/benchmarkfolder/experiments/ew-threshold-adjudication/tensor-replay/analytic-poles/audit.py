#!/usr/bin/env python3
"""Source-specialized MadFKS QED pole prediction; no reducer evaluation."""
import ast
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
TRACE = HERE.parents[1]
OLD_EXPORT = ROOT / '.tirust-build/physical-stress/unchecked/ew_n3_epem_ttxa/candidate-9-qp-init-dp'
D = Decimal
spec = importlib.util.spec_from_file_location('exact_binary128_inputs', HERE.parent / 'input-consistency/audit.py')
exact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(token):
    value = exact.binary128(token)
    return D(value.numerator) / D(value.denominator)


def dot(left, right):
    return left[0] * right[0] - sum(left[k] * right[k] for k in range(1, 4))


def rational_expression(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -rational_expression(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return rational_expression(node.left) / rational_expression(node.right)
    raise ValueError('Unexpected UFO charge expression: ' + ast.dump(node))


def main():
    assert Path.cwd().resolve() == ROOT
    sources = {
        'fks_poles_and_photon_gamma': ROOT / 'Template/NLO/SubProcesses/fks_singular.f',
        'charge_linked_born': ROOT / 'madgraph/iolibs/template_files/sborn_sf_fks.inc',
        'default_photon_tag_and_fks_legs': ROOT / 'madgraph/fks/fks_common.py',
        'cli_photon_tag_parser': ROOT / 'madgraph/interface/madgraph_interface.py',
        'fks_export_charge_mapping': ROOT / 'madgraph/iolibs/export_fks.py',
        'tagged_photon_context': ROOT / 'Template/NLO/SubProcesses/BinothLHA.f',
        'actual_generated_flavor_counts': OLD_EXPORT / 'Source/MODEL/coupl.inc',
        'actual_Born_and_QP_kinematic_order': OLD_EXPORT / 'SubProcesses/P0_epem_ttxa/loop_matrix.f',
        'recorded_Born_DP_promotion': TRACE / 'trace.f',
        'physical_card': TRACE / 'param_card.dat',
        'actual_process_card': OLD_EXPORT / 'Cards/proc_card_mg5.dat',
        'actual_momenta_couplings_born': TRACE / 'trace.json',
        'trace_execution': TRACE / 'execution.json',
        'exact_binary128_rounding': HERE.parent / 'input-consistency/audit.py',
        'this_audit': Path(__file__),
    }
    execution = json.loads(sources['trace_execution'].read_text())
    assert execution['complete'] and execution['final_values_identical_to_historical']
    proc = sources['actual_process_card'].read_text()
    commands = [line.strip() for line in proc.splitlines() if line.strip().startswith('generate ')]
    assert commands == ['generate e+ e- > t t~ a [virt=QED]']
    assert '!' not in commands[0]
    assert "self['is_tagged'] = False" in sources['default_photon_tag_and_fks_legs'].read_text()
    imported = [line.strip().removeprefix('import model ') for line in proc.splitlines()
        if line.strip().startswith('import model ')]
    assert len(imported) == 1
    model = Path(imported[0])
    assert model.is_relative_to(ROOT)
    sources['physical_ufo_particles'] = model / 'particles.py'
    if (model / 'restrict_default.dat').exists():
        sources['physical_ufo_restriction'] = model / 'restrict_default.dat'
    ufo_charges = {}
    for node in ast.parse(sources['physical_ufo_particles'].read_text()).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name) or call.func.id != 'Particle':
            continue
        fields = {item.arg: item.value for item in call.keywords}
        pdg = ast.literal_eval(fields['pdg_code'])
        if pdg in (6, 11, 22):
            ufo_charges[pdg] = rational_expression(fields['charge'])
    assert ufo_charges == {6: Fraction(2, 3), 11: Fraction(-1), 22: Fraction(0)}
    identities = {key: {'path': str(path.resolve().relative_to(ROOT)), 'sha256': digest(path)}
        for key, path in sources.items()}
    masses = {}
    in_mass = False
    for line in sources['physical_card'].read_text().splitlines():
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        if fields[0].lower() == 'block':
            in_mass = fields[1].lower() == 'mass'
        elif fields[0].lower() == 'decay':
            in_mass = False
        elif in_mass:
            masses[int(fields[0])] = D(fields[1])
    nf = sum(masses[pdg] == 0 for pdg in range(1, 7))
    nl = sum(masses[pdg] == 0 for pdg in (11, 13, 15))
    assert nf == 5 and nl == 3 and masses[6] > 0
    generated_flavors = dict(re.findall(r'PARAMETER\((N[FL])=(\d+)D0\)',
        sources['actual_generated_flavor_counts'].read_text()))
    assert generated_flavors == {'NF': str(nf), 'NL': str(nl)}
    trace = json.loads(sources['actual_momenta_couplings_born'].read_text())
    records = []
    with localcontext() as context:
        context.prec = 90
        pi = D('3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628')
        # sborn_sf's initial/final sign is incorporated in these all-outgoing
        # charges. Physical e+,e-,t,tbar charges are +1,-1,+2/3,-2/3.
        charges = [-D(1), D(1), D(2) / 3, -D(2) / 3]
        assert sum(charges) == 0
        quark_charge_ninths = sum(4 if pdg % 2 == 0 else 1 for pdg in range(1, nf + 1))
        gamma_ph = -D(2 * (quark_charge_ninths + 3 * nl)) / 9
        assert gamma_ph == -D(40) / 9
        for key in ('1', '2'):
            point = trace[key]
            momenta = [[number(value) for value in row[1:]] for row in point['MP_P']]
            mt = number(point['mt'][1])
            mass = [D(0), D(0), mt, mt]
            scale = number(point['groups']['5']['scale'])
            born = number(point['answers']['0'][0])
            gal_re, gal_im = map(number, point['mp_gal'])
            assert gal_im == 0
            # QES2 is chosen equal to mu2. Therefore the explicit
            # double*log(mu2/QES2) prefactor expansion vanishes.
            diagonal = -D(3) - D(8) / 9 - gamma_ph
            factor = diagonal
            pairs = []
            for i in range(4):
                for j in range(i):
                    product = dot(momenta[i], momenta[j])
                    assert product > 0
                    if mass[i] == 0 and mass[j] == 0:
                        soft = (2 * product / scale).ln()
                        branch = 'massless-massless'
                    elif mass[i] == 0 or mass[j] == 0:
                        massive = mass[i] or mass[j]
                        soft = -(massive * massive / scale).ln() / 2 + (2 * product / scale).ln()
                        branch = 'massive-massless'
                    else:
                        velocity_squared = 1 - (mass[i] * mass[j] / product) ** 2
                        assert velocity_squared >= 0
                        velocity = velocity_squared.sqrt()
                        if velocity <= D('1e-6'):
                            soft = 1 + velocity_squared / 3 + velocity_squared ** 2 / 5
                            branch = 'massive-massive FKS small-velocity series'
                        else:
                            soft = ((1 + velocity) / (1 - velocity)).ln() / (2 * velocity)
                            branch = 'massive-massive logarithm'
                    contribution = -2 * charges[i] * charges[j] * soft
                    factor += contribution
                    pairs.append({'legs': [j + 1, i + 1], 'dot_product': str(product),
                        'soft_factor': str(soft), 'single_pole_factor_contribution': str(contribution), 'branch': branch})
            prefactor = born * gal_re ** 2 / (8 * pi ** 2)
            single, double = prefactor * factor, -2 * prefactor
            actual_single = number(point['answers']['0'][2])
            actual_double = number(point['answers']['0'][3])
            records.append({'point': int(key), 'born_used': str(born), 'mu2': str(scale),
                'gal_real': str(gal_re), 'gamma_ph': str(gamma_ph),
                'single_diagonal_factor': str(diagonal), 'pairs': pairs,
                'single_factor': str(factor), 'single_over_double_ratio': str(-factor / 2),
                'single_pole_prediction': str(single), 'double_pole_prediction': str(double),
                'historical_TIR_single': str(actual_single), 'historical_TIR_double': str(actual_double),
                'historical_TIR_single_relative_difference': str(abs(single - actual_single) / abs(single)),
                'historical_TIR_double_relative_difference': str(abs(double - actual_double) / abs(double)),
                'historical_TIR_complete_tensor_groups': sum(g['status'] == 0 for g in point['groups'].values()),
                'historical_TIR_total_tensor_groups': len(point['groups'])})
    result = {'schema': 1, 'scope': 'Independent source-specialized QED pole formula; no reducer invocation or new Born calculation.',
        'independent_review_status': 'Independently source-reviewed by ct_qp_source_audit; this is not a complete-amplitude acceptance certificate.',
        'sources': identities, 'process_command': commands[0],
        'nf': nf, 'nl': nl, 'external_photon_is_tagged': False,
        'generated_flavor_constants': generated_flavors,
        'ufo_base_particle_charges': {str(pdg): str(charge) for pdg, charge in ufo_charges.items()},
        'charge_convention': 'Physical charges(+1,-1,+2/3,-2/3,0), initial/final sign from sborn_sf; all-outgoing(-1,+1,+2/3,-2/3,0).',
        'precision': 'Exact113-bit round-trip inputs, 90-decimal-digit arithmetic; mathematical pi.',
        'normalization': 'Recorded DP Born times recorded QP real GAL(1)^2/(8*pi^2). No fitted coefficient or reducer pole enters the prediction.',
        'scale_policy': 'QES2=mu2=captured TIRust scale; fksprefact-false explicit logarithm is zero.',
        'limits': [
            'Returned Born was evaluated in DP at the original physical point; the QP momenta include MadLoop repair. This is not an independent QP Born certificate.',
            'Exact input transport and higher arithmetic precision do not remove the supplied mass-shell convention mismatch documented separately.',
            'FKS photon gamma assumes the untagged photon in the frozen command and five massless quarks plus three massless charged leptons in the physical card.',
            'The virt CLI rejects tagged particles. Full-NLO tagged-photon prescriptions include additional terms and cannot reuse this specialization without a separate audit.',
            'The historical threshold TIRust amplitude omits37 refused groups; its coefficients are invalid as a numerical reference.',
            'Pole consistency does not establish the finite coefficient or whole-amplitude stability.',
        ], 'records': records}
    (HERE / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps([{'point': row['point'], 'single': row['single_pole_prediction'],
        'double': row['double_pole_prediction']} for row in records]))


if __name__ == '__main__':
    main()
