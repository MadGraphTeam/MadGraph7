#!/usr/bin/env python3
"""Read frozen DAT decimal tokens; perform no matrix-element/native calls."""
from decimal import Decimal as D, localcontext
from hashlib import sha256
from itertools import combinations, permutations
from pathlib import Path
import json

ROOT = Path.cwd()
if not (ROOT / "docs/benchmarkfolder/cases.json").is_file():
    raise RuntimeError("Run this diagnostic from the MadGraph repository root")
POINTS = ROOT / "docs/benchmarkfolder/points"
CASES = [("qcd_n2_gg_ttx", 1), ("qcd_n3_uux_zgg", 2),
         ("qcd_n4_uux_zzgg", 3), ("ew_n3_epem_ttxa", 1)]


def dot(p, q):
    return p[0] * q[0] - sum((p[j] * q[j] for j in range(1, 4)), D(0))


def vsum(vectors):
    return [sum((v[j] for v in vectors), D(0)) for j in range(4)]


def det(matrix):
    n = len(matrix)
    result = D(0)
    for perm in permutations(range(n)):
        inversions = sum(perm[i] > perm[j] for i in range(n) for j in range(i + 1, n))
        term = D(-1 if inversions % 2 else 1)
        for i, j in enumerate(perm):
            term *= matrix[i][j]
        result += term
    return result


def gram(vectors, s):
    g = [[dot(p, q) for q in vectors] for p in vectors]
    determinant = det(g)
    cartesian_determinant = det(vectors)
    assert determinant == -cartesian_determinant ** 2
    euclidean_norm_product = D(1)
    for p in vectors:
        euclidean_norm_product *= sum(x * x for x in p)
    return {
        "det_pdotp_GeV8": determinant,
        "det_pdotp_over_s4": determinant / s ** 4,
        "abs_det_2pdotp_over_s4": abs(determinant) * 16 / s ** 4,
        "abs_det_over_product_euclidean_norms_squared": abs(determinant) / euclidean_norm_product,
        "cartesian_four_volume_GeV4": cartesian_determinant,
        "nonzero_independent_four_vectors": cartesian_determinant != 0,
    }


def atan_small(x):
    assert abs(x) < D("0.01")
    result = D(0)
    power = x
    n = 0
    tolerance = D(10) ** (-190)
    while True:
        term = power / (2 * n + 1)
        result += term if n % 2 == 0 else -term
        if abs(term) < tolerance:
            return result
        power *= x * x
        n += 1


def case(name, index):
    dat_path = POINTS / f"{name}-stress.dat"
    json_path = POINTS / f"{name}-stress.json"
    metadata = json.loads(json_path.read_text(), parse_float=D)
    point = metadata["points"][index - 1]
    masses = point["incoming_masses"] + point["masses"]
    tokens = dat_path.read_text().splitlines()
    assert int(tokens[0]) == len(metadata["points"])
    start = 1 + (index - 1) * len(masses)
    momenta = [[D(t) for t in line.split()] for line in tokens[start:start + len(masses)]]
    assert all(len(p) == 4 for p in momenta)
    dat_digest = sha256(dat_path.read_bytes()).hexdigest()
    assert dat_digest == metadata["sha256"]
    incoming = vsum(momenta[:2])
    outgoing = vsum(momenta[2:])
    s = dot(incoming, incoming)
    sqrt_s = s.sqrt()
    threshold = sum(point["masses"])
    conservation = [incoming[j] - outgoing[j] for j in range(4)]
    shells = [dot(p, p) - mass ** 2 for p, mass in zip(momenta, masses)]
    row = {
        "case": name, "point_index_one_based": index, "point_id": point["id"],
        "dat_path": str(dat_path.relative_to(ROOT)), "dat_sha256": dat_digest,
        "json_sha256": sha256(json_path.read_bytes()).hexdigest(),
        "dat_first_line_one_based": start + 1, "dat_last_line_one_based": start + len(masses),
        "declared_masses_GeV": masses,
        "sqrt_s_GeV": sqrt_s,
        "on_shell_residuals_p2_minus_m2_GeV2": shells,
        "max_abs_on_shell_residual_GeV2": max(map(abs, shells)),
        "max_abs_on_shell_residual_over_s": max(map(abs, shells)) / s,
        "per_leg_abs_on_shell_residual_over_E2": [abs(r) / p[0] ** 2 for p, r in zip(momenta, shells)],
        "conservation_incoming_minus_outgoing_GeV": conservation,
        "max_abs_conservation_residual_GeV": max(map(abs, conservation)),
        "max_abs_conservation_residual_over_sqrt_s": max(map(abs, conservation)) / sqrt_s,
        "threshold_excess_GeV": sqrt_s - threshold,
        "threshold_excess_over_mass_sum": (sqrt_s - threshold) / threshold,
        "max_abs_DAT_minus_JSON_component_GeV": max(abs(momenta[i][j] - point["momenta"][i][j]) for i in range(len(masses)) for j in range(4)),
    }
    if point["kind"] == "threshold":
        pair = vsum(momenta[2:4])
        pair_mass = dot(pair, pair).sqrt()
        row["pair_mass_GeV"] = pair_mass
        row["pair_excess_above_declared_top_pair_threshold_GeV"] = pair_mass - threshold
        row["pair_threshold_beta_using_declared_masses"] = (1 - threshold ** 2 / pair_mass ** 2).sqrt()
        exact_shell_threshold = sum(dot(p, p).sqrt() for p in momenta[2:4])
        row["pair_excess_above_exact_external_invariant_masses_GeV"] = pair_mass - exact_shell_threshold
        row["external_mass_shell_shift_over_declared_pair_excess"] = (exact_shell_threshold - threshold) / (pair_mass - threshold)
        row["energy_conservation_residual_over_declared_pair_excess"] = conservation[0] / (pair_mass - threshold)
        row["outgoing_kinetic_energy_GeV"] = [p[0] - m for p, m in zip(momenta[2:], point["masses"])]
        if len(momenta) == 5:
            row["photon_energy_over_sqrt_s"] = momenta[4][0] / sqrt_s
            row["photon_energy_over_total_threshold_excess"] = momenta[4][0] / (sqrt_s - threshold)
    if point["kind"] == "soft-collinear":
        soft_index = int(point["soft_leg"])
        soft = momenta[soft_index]
        pt = (soft[1] ** 2 + soft[2] ** 2).sqrt()
        two_dot = 2 * dot(momenta[0], soft)
        two_dot_massless_stable = 2 * momenta[0][0] * pt ** 2 / (soft[0] + soft[3])
        binary = [[D.from_float(float(x)) for x in p] for p in momenta]
        direct_binary64 = 2 * (float(momenta[0][0]) * float(soft[0]) - sum(float(momenta[0][j]) * float(soft[j]) for j in range(1, 4)))
        row["soft_diagnostics"] = {
            "soft_leg_zero_based": soft_index, "energy_GeV": soft[0],
            "energy_over_sqrt_s": soft[0] / sqrt_s,
            "angle_to_positive_z_beam_radians": atan_small(pt / soft[3]),
            "pt_GeV": pt,
            "two_p1_dot_soft_GeV2": two_dot,
            "two_p1_dot_soft_over_s": two_dot / s,
            "two_p1_dot_soft_using_massless_stable_identity_GeV2": two_dot_massless_stable,
            "relative_shift_if_soft_leg_forced_exactly_massless": (two_dot_massless_stable - two_dot) / two_dot,
            "two_p1_dot_soft_exact_binary64_components_GeV2": 2 * dot(binary[0], binary[soft_index]),
            "two_p1_dot_soft_direct_binary64_arithmetic_GeV2": D.from_float(direct_binary64),
            "direct_binary64_relative_error_vs_DAT": (D.from_float(direct_binary64) - two_dot) / two_dot,
            "all_pair_two_dot_over_s": {f"{i+1},{j+1}": 2 * dot(momenta[i], momenta[j]) / s for i, j in combinations(range(len(momenta)), 2)},
        }
    if point["kind"] == "small-gram":
        row["gram_definition"] = "G_ij=p_i dot p_j; metric(+---); normalize det(G) by s^4 with s=(p1+p2)^2; all vectors have positive energies"
        row["gram_final_four_legs_one_based_3_4_5_6"] = gram(momenta[2:], s)
        row["gram_beams_and_Zs_legs_one_based_1_2_3_4"] = gram(momenta[:4], s)
        row["max_abs_final_y_over_max_final_spatial_component"] = max(abs(p[2]) for p in momenta[2:]) / max(abs(x) for p in momenta[2:] for x in p[1:])
        row["all_four_vector_gram_det_over_s4"] = {",".join(str(i+1) for i in inds): gram([momenta[i] for i in inds], s)["det_pdotp_over_s4"] for inds in combinations(range(6), 4)}
    return row


if __name__ == "__main__":
    with localcontext() as ctx:
        ctx.prec = 200
        result = {
            "method": "Python Decimal, 200 decimal digits; exact printed DAT decimal tokens; masses from JSON Decimal; no re-projection and no matrix-element calls",
            "momentum_convention": "E,px,py,pz; incoming and outgoing positive energies; metric +---; GeV",
            "normalization": "s=(p1+p2)^2; mass-shell residual / s; conservation residual / sqrt(s)",
            "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "cases": [case(name, index) for name, index in CASES],
        }
    output = Path(__file__).with_name("diagnostics.json")
    output.write_text(json.dumps(result, default=str, indent=2) + "\n")
    for row in result["cases"]:
        print(row["case"], "sqrt_s", f'{row["sqrt_s_GeV"]:.15E}',
              "max_shell/s", f'{row["max_abs_on_shell_residual_over_s"]:.8E}',
              "max_conservation/sqrt_s", f'{row["max_abs_conservation_residual_over_sqrt_s"]:.8E}')
        for key in ("threshold_excess_GeV", "pair_excess_above_declared_top_pair_threshold_GeV", "pair_threshold_beta_using_declared_masses"):
            if key in row: print(" ", key, f'{row[key]:.15E}')
        if "soft_diagnostics" in row:
            for key, value in row["soft_diagnostics"].items():
                if isinstance(value,D): print(" ",key,f'{value:.15E}')
        for key in ("gram_final_four_legs_one_based_3_4_5_6", "gram_beams_and_Zs_legs_one_based_1_2_3_4"):
            if key in row: print(" ",key,"det/s4",f'{row[key]["det_pdotp_over_s4"]:.15E}')
