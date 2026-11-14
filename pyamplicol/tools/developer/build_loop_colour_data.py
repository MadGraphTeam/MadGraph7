# SPDX-License-Identifier: 0BSD
"""Write the shipped N_c data of MG5_aMC 3.7.0 ``models/loop_sm`` (MQ-NC).

``python tools/developer/build_loop_colour_data.py <loop_sm directory>
[--output src/pyamplicol/models/loop_color_declarations/loop_sm.json]``
compiles the source, computes the structural digest of each of its
counterterm vertices and external-field factors
(``pyamplicol.models.loop_color_declarations``) and writes one data entry per
vertex and factor with the colour factors below. The table is the authoring
input: it names the source's vertices and couplings (this script only; the
shipped data keys every entry by digest) and states each coupling's colour
factor from the literature, independently of how the source writes it:

* UV (textbook QCD renormalisation, MadLoop's mass-dependent decoupling
  scheme, arXiv:1103.0621 §2.4): the strong coupling's counterterm from the
  gluon loops is proportional to C_A (beta_0 = 11 C_A/3 - 4 T_F n_f/3), each
  quark flavour's to T_F; the on-shell mass, Yukawa and heavy-quark field
  counterterms to C_F; the gluon field counterterm of a heavy-quark loop to
  T_F.
* R2 of QCD vertices (Draggiotis, Garzelli, Papadopoulos and Pittau,
  arXiv:0903.0356): the gluon two- and three-point R2 of a gluon loop carry
  N_col, of a closed quark loop T_F (N_f/N_col times N_col); the quark two-
  point and quark-gluon R2 carry (N_col^2-1)/(2 N_col) = C_F. The four-gluon
  R2 (fixed-SU(3) colour tensors in the source) is a whole-rule recipe per
  loop group (gauge loop and fundamental Dirac loop,
  ``loops.qcd_r2_color``).
* R2 with electroweak legs and QCD loops (Garzelli, Malamos and Pittau,
  arXiv:0910.3130): a gluon exchanged on a quark line (q q~ V, q q~ S and
  the Goldstone vertices) carries C_F; a closed quark loop with two or three
  gluons (g g V, g g S, g g V V, g g S S, g g g V) carries T_F.

The script refuses a source vertex or field factor the table does not
state, and a table entry the source does not have, so the data covers the
source exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from pyamplicol.loops.shipped_profiles import LOOP_SM_SCHEME

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "src/pyamplicol/models/loop_color_declarations/loop_sm.json"

UV = (
    "textbook QCD renormalisation in MadLoop's mass-dependent decoupling scheme "
    "(arXiv:1103.0621 §2.4): delta g_s of the gluon loops ~ C_A, of each quark "
    "flavour ~ T_F; on-shell delta m, Yukawa and heavy-quark delta Z ~ C_F; "
    "heavy-quark gluon delta Z ~ T_F"
)
R2_QCD = (
    "general-N_c QCD R2, Draggiotis, Garzelli, Papadopoulos, Pittau "
    "arXiv:0903.0356: gluon loops ~ N_col, closed quark loops ~ T_F "
    "(N_f/N_col x N_col), quark lines ~ (N_col^2-1)/(2 N_col)"
)
R2_EW = (
    "R2 with electroweak legs and QCD loops, Garzelli, Malamos, Pittau "
    "arXiv:0910.3130: a gluon on a quark line ~ C_F, a closed quark loop "
    "with gluons ~ T_F"
)
FOUR_GLUON = (
    "general-N_c four-gluon R2 tensors (arXiv:0903.0356), built and checked "
    "against the source at N_c = 3 by loops.qcd_r2_color (whole rule)"
)


def _quark_line(vertices, couplings, factor="CF", reference=R2_EW):
    return {
        vertex: (reference, dict.fromkeys(couplings, factor)) for vertex in vertices
    }


# vertex name -> (reference, {coupling name: colour factor}) or a whole rule.
TABLE: dict[str, object] = {
    "V_R23G": (R2_QCD, {"R2_3Gq": "TF", "R2_3Gg": "Nc"}),
    "V_R2RGA": (
        FOUR_GLUON,
        [
            {
                "loop_group_index": 0,
                "order": [["QCD", 4]],
                "gauge_coupling": "G",
                "contribution": "fundamental-dirac",
            },
            {
                "loop_group_index": 1,
                "order": [["QCD", 4]],
                "gauge_coupling": "G",
                "contribution": "gauge",
            },
        ],
    ),
    **_quark_line(
        ("V_R2GDD", "V_R2GUU", "V_R2GSS", "V_R2GCC", "V_R2GBB", "V_R2GTT"),
        ("R2_GQQ",),
        reference=R2_QCD,
    ),
    "V_R2GG": (
        R2_QCD,
        {
            "R2_GGq": "TF",
            "R2_GGc": "TF",
            "R2_GGb": "TF",
            "R2_GGt": "TF",
            "R2_GGg_1": "Nc",
            "R2_GGg_2": "Nc",
        },
    ),
    **_quark_line(("V_R2DD", "V_R2UU", "V_R2SS"), ("R2_QQq",), reference=R2_QCD),
    "V_R2CC": (R2_QCD, {"R2_QQq": "CF", "R2_QQc": "CF"}),
    "V_R2BB": (R2_QCD, {"R2_QQq": "CF", "R2_QQb": "CF"}),
    "V_R2TT": (R2_QCD, {"R2_QQq": "CF", "R2_QQt": "CF"}),
    **_quark_line(("V_R2ddA", "V_R2ssA", "V_R2bbA"), ("R2_DDA",)),
    **_quark_line(("V_R2uuA", "V_R2ccA", "V_R2ttA"), ("R2_UUA",)),
    **_quark_line(("V_R2ddZ", "V_R2ssZ", "V_R2bbZ"), ("R2_DDZ_V2", "R2_DDZ_V3")),
    **_quark_line(("V_R2uuZ", "V_R2ccZ", "V_R2ttZ"), ("R2_UUZ_V2", "R2_UUZ_V5")),
    **{
        f"V_R2{down}x{up}W": (R2_EW, {f"R2_{down}x{up}W": "CF"})
        for down in ("d", "s", "b")
        for up in ("u", "c", "t")
    },
    **{
        f"V_R2{up}x{down}W": (R2_EW, {f"R2_{up}x{down}W": "CF"})
        for down in ("d", "s", "b")
        for up in ("u", "c", "t")
    },
    "V_bbH": (R2_EW, {"R2_bbH": "CF"}),
    "V_ttH": (R2_EW, {"R2_ttH": "CF"}),
    "V_ccH": (R2_EW, {"R2_ccH": "CF"}),
    "V_GGH": (R2_EW, dict.fromkeys(("R2_GGHc", "R2_GGHb", "R2_GGHt"), "TF")),
    "V_GGHH": (R2_EW, dict.fromkeys(("R2_GGHHc", "R2_GGHHb", "R2_GGHHt"), "TF")),
    "V_GGZ": (R2_EW, dict.fromkeys(("R2_GGZup", "R2_GGZdown"), "TF")),
    "V_GGZZ": (R2_EW, dict.fromkeys(("R2_GGZZup", "R2_GGZZdown"), "TF")),
    "V_GGAA": (R2_EW, dict.fromkeys(("R2_GGAAup", "R2_GGAAdown"), "TF")),
    "V_GGZA": (R2_EW, dict.fromkeys(("R2_GGZAup", "R2_GGZAdown"), "TF")),
    "V_GGWW": (
        R2_EW,
        dict.fromkeys((f"R2_GGWW{u}{d}" for u in "uct" for d in "dsb"), "TF"),
    ),
    "V_GGGZ": (
        R2_EW,
        dict.fromkeys(
            (
                "R2_GGGZvecUp",
                "R2_GGGZvecDown",
                "R2_GGGZaxialUp",
                "R2_GGGZaxialDown",
            ),
            "TF",
        ),
    ),
    "V_GGGA": (R2_EW, dict.fromkeys(("R2_GGGAvecUp", "R2_GGGAvecDown"), "TF")),
    "V_UV1eps3G": (
        UV,
        {
            "UV_3Gq": "TF",
            "UV_3Gc": "TF",
            "UV_3Gb": "TF",
            "UV_3Gt": "TF",
            "UV_3Gg": "CA",
        },
    ),
    "V_UV1eps4G": (
        UV,
        {
            "UV_4Gq": "TF",
            "UV_4Gc": "TF",
            "UV_4Gb": "TF",
            "UV_4Ggt": "TF",
            "UV_4Gg": "CA",
        },
    ),
    **{
        vertex: (
            UV,
            {
                "UV_GQQq": "TF",
                "UV_GQQc": "TF",
                "UV_GQQb": "TF",
                "UV_GQQt": "TF",
                "UV_GQQg": "CA",
            },
        )
        for vertex in (
            "V_UVGDD",
            "V_UVGUU",
            "V_UVGCC",
            "V_UVGSS",
            "V_UVGBB",
            "V_UVGTT",
        )
    },
    "V_UVcMass": (UV, {"UV_cMass": "CF"}),
    "V_UVbMass": (UV, {"UV_bMass": "CF"}),
    "V_UVtMass": (UV, {"UV_tMass": "CF"}),
    "V_UVHcc": (UV, {"UV_Hcc": "CF"}),
    "V_UVHtt": (UV, {"UV_Htt": "CF"}),
    "V_UVHbb": (UV, {"UV_Hbb": "CF"}),
    "V_R2_GGGpGm": (
        R2_EW,
        dict.fromkeys(
            (
                f"R2_GGGpGm_{pair}"
                for pair in ("ub", "cd", "cs", "cb", "td", "ts", "tb")
            ),
            "TF",
        ),
    ),
    "V_R2_GGG0G0": (
        R2_EW,
        dict.fromkeys((f"R2_GGG0G0_{q}" for q in "cbt"), "TF"),
    ),
}
# The Goldstone-quark R2 and UV vertices: a gluon on the quark line (C_F).
_GOLDSTONE = {
    "txbGp": ("1015", "1021"),
    "bxtGp": ("1027", "1030"),
    "1077": ("1082",),
    "1138": ("1085",),
    "1139": ("1095",),
    "1084": ("1018",),
    "1085": ("1019",),
    "1086": ("1013",),
    "1087": ("1014", "1020"),
    "1116": ("1028",),
    "1117": ("1022",),
    "1118": ("1023",),
    "1119": ("1024", "1029"),
    "1120": ("1025",),
    "1121": ("1026",),
}
for _vertex, _couplings in _GOLDSTONE.items():
    TABLE[f"V_R2_{_vertex}"] = (R2_EW, {f"GC_R2_{c}": "CF" for c in _couplings})
    TABLE[f"V_UV_{_vertex}"] = (UV, {f"GC_UV_{c}": "CF" for c in _couplings})

# particle name -> {loop-group index: colour factor} of its field factors.
FIELDS = {
    "c": {0: "CF"},
    "b": {0: "CF"},
    "t": {0: "CF"},
    "G": {0: "TF", 1: "TF", 2: "TF"},
}
# The scheme of the shipped [loop.model] profile of loop_sm (MQ-MODEL's
# ``loops.shipped_profiles``): the continuation is declared in the scheme
# the stock model runs in, and a request refuses a continuation whose
# scheme differs from the effective one.
SCHEME = asdict(LOOP_SM_SCHEME)
IDENTITIES = {
    "CA": "Nc",
    "CF": "(Nc**2-1)/(2*Nc)",
    "TF": "1/2",
    "Ncol": "Nc",
}


def build(source: Path) -> dict:
    from pyamplicol.models.loading import compile_model_source
    from pyamplicol.models.loop_color_declarations import (
        DATA_SCHEMA,
        field_factors,
        vertex_digest,
    )

    raw = compile_model_source(str(source)).raw_loop_model
    vertices, seen, by_digest = [], set(), {}
    for vertex in raw.collections["counterterm_vertices"]:
        name = str(vertex["name"])
        if name not in TABLE:
            raise SystemExit(f"{name}: no colour data in the table")
        seen.add(name)
        particles = " ".join(p["name"] for p in vertex["fields"]["particles"])
        entries = [
            (tuple(key["items"]), reference["name"])
            for key, reference in vertex["fields"]["couplings"]["entries"]
        ]
        couplings = {name for _indices, name in entries}
        reference, data = TABLE[name]
        entry = {
            "digest": vertex_digest(raw, vertex),
            "label": f"{name} [{particles}]",
            "kind": str(vertex["fields"]["type"]),
            "reference": reference,
        }
        if isinstance(data, list):
            entry.update(continuation="whole-rule", recipes=data)
        else:
            if set(data) != couplings:
                raise SystemExit(
                    f"{name}: table couplings {sorted(data)} "
                    f"!= source {sorted(couplings)}"
                )
            # Per coupling entry (colour, Lorentz, loop-group indices): the
            # shipped data never names a coupling.
            entry.update(
                continuation="identities",
                factors={
                    ",".join(map(str, indices)): data[coupling]
                    for indices, coupling in sorted(entries)
                },
            )
        digest = entry["digest"]
        if digest in by_digest:
            # A vertex of the same structure (e.g. the d and s quark R2): one
            # entry, whose data must agree.
            first = by_digest[digest]
            if {k: v for k, v in first.items() if k != "label"} != {
                k: v for k, v in entry.items() if k != "label"
            }:
                raise SystemExit(
                    f"{name}: same structure as {first['label']}, other data"
                )
            first["label"] += f"; {entry['label']}"
            continue
        by_digest[digest] = entry
        vertices.append(entry)
    if set(TABLE) - seen:
        raise SystemExit(
            f"table vertices absent from the source: {sorted(set(TABLE) - seen)}"
        )
    fields, seen_fields = [], set()
    for factor in field_factors(raw):
        particle = str(raw.collections["particles"][factor.key[0]]["name"])
        group = factor.key[2]
        colour = FIELDS.get(particle, {}).get(group)
        if colour is None or factor.key[1] != "QCD":
            raise SystemExit(f"{factor.label}: no colour data in the table")
        seen_fields.add((particle, group))
        fields.append(
            {
                "digest": factor.digest,
                "label": factor.label,
                "factor": colour,
                "reference": UV,
            }
        )
    expected = {(p, g) for p, groups in FIELDS.items() for g in groups}
    if expected != seen_fields:
        raise SystemExit(f"field table mismatch: {sorted(expected ^ seen_fields)}")
    return {
        "schema": DATA_SCHEMA,
        "source": "MG5_aMC 3.7.0 models/loop_sm (all restrictions)",
        "scheme": SCHEME,
        "formal_symbol": "Nc",
        "parameter_identities": IDENTITIES,
        "scheme_parameter_values": {"lhv": "1"},
        "vertices": vertices,
        "fields": fields,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="the MG5_aMC loop_sm directory")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    payload = build(args.source)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
    print(
        f"{args.output}: {len(payload['vertices'])} vertices, "
        f"{len(payload['fields'])} field factors",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
