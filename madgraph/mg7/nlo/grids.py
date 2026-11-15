"""The PDFs and alpha_s of the fixed-order NLO driver of madspace
(nlo/fks.py), as madspace PdfGrid and AlphaSGrid files.

With pdlabel lhapdf they are those of the LHAPDF set of lhaid. The built-in
sets of MadFKS (BUILTIN) take alpha_s(m_Z) and the loops of its running from
pdfwrap.f, the running of alfas_functions.f on 200 nodes per decade. For the
NNPDF 2.3 grids that MadGraph ships (nn23nlo, nn23lo, nn23lo1) their nodes
are written as an LHAPDF grid (lhagrid1, the flavours fdist of pdf.f takes),
whose interpolation in madspace differs from that of NNPDFDriver.f by up to
3e-3 at large x and low scales (see test_fks_couplings.py of madspace). The
CTEQ6 and CT14qed sets are tabulated by fdist of MadFKS itself (pdf_table.f
of template_files/mg7, linked with the libraries of the output) on 40 nodes
per decade in Q and in x below 0.1 and 90 above, within the range of their
tables (from x = 1e-7 for CT14qed), where the interpolation of madspace
stays within 5e-4 of fdist (at large x, the polynomial interpolation of
their tables in fdist limits it).
"""

import math
import os
import subprocess

import numpy as np

from madgraph import MG5DIR

# the template of the tabulation of the built-in sets
TEMPLATES = os.path.join(MG5DIR, "madgraph", "iolibs", "template_files", "mg7")
NNPDF_GRID = os.path.join(
    MG5DIR, "Template", "LO", "lib", "Pdfdata", "NNPDF23nlo_as_0119_qed_mem0.grid"
)
# alpha_s(m_Z) of nn23nlo (pdfwrap of MadFKS)
NNPDF_ALPHA_S = 0.119
# the built-in sets of MadFKS: alpha_s(m_Z) and the loops of its running
# (pdfwrap.f)
BUILTIN = {
    "nn23nlo": (0.119, 2),
    "nn23lo": (0.119, 2),
    "nn23lo1": (0.130, 2),
    "cteq6_m": (0.118, 2),
    "cteq6_d": (0.118, 2),
    "cteq6_l": (0.118, 2),
    "cteq6l1": (0.130, 1),
    "ct14q00": (0.118, 2),
    "ct14q07": (0.118, 2),
    "ct14q14": (0.118, 2),
    "ct14q21": (0.118, 2),
}
# the NNPDF grids among them
NNPDF_GRIDS = {
    "nn23nlo": NNPDF_GRID,
    "nn23lo": NNPDF_GRID.replace("23nlo_as_0119", "23_lo_as_0119"),
    "nn23lo1": NNPDF_GRID.replace("23nlo_as_0119", "23_lo_as_0130"),
}
# the ranges in x and Q of the tables of the others
TABLE_RANGES = {"cteq6": (1e-6, 1.3, 1e4), "ct14q": (1e-7, 1.295, 1e5)}
# the flavours of the grid and their columns (IPDF + 7 of NNevolvePDF, from 1)
PIDS = [-5, -4, -3, -2, -1, 21, 1, 2, 3, 4, 5, 22]
COLUMNS = {pid: (0 if pid == 21 else 7 if pid == 22 else pid) + 7 for pid in PIDS}


def read_nnpdf_grid(path=NNPDF_GRID):
    """The nodes in x and Q^2 and the values (per node in x and Q^2 the
    columns of the grid) of member 0 of an NNPDF grid of MadGraph."""
    with open(path) as f:
        lines = [line.strip() for line in f]
    k = next(i for i, line in enumerate(lines) if "NNPDF20int" in line) + 2
    nx = int(lines[k])
    x = [float(v) for v in lines[k + 1 : k + 1 + nx]]
    k += 1 + nx
    nq2 = int(lines[k])
    # the record after nq2 is read and dropped, as the one after the grid
    q2 = [float(v) for v in lines[k + 2 : k + 2 + nq2]]
    k += 3 + nq2
    rows = [[float(v) for v in line.split()] for line in lines[k : k + nx * nq2]]
    values = [[rows[ix * nq2 + iq] for iq in range(nq2)] for ix in range(nx)]
    return x, q2, values


def write_lhapdf_grid(path, grid=NNPDF_GRID):
    """The nodes of an NNPDF grid of MadGraph as an LHAPDF grid
    (lhagrid1)."""
    x, q2, values = read_nnpdf_grid(grid)
    with open(path, "w") as f:
        f.write("PdfType: central\nFormat: lhagrid1\n---\n")
        f.write(" ".join(f"{v:.17e}" for v in x) + "\n")
        f.write(" ".join(f"{math.sqrt(v):.17e}" for v in q2) + "\n")
        f.write(" ".join(str(pid) for pid in PIDS) + "\n")
        columns = [COLUMNS[pid] - 1 for pid in PIDS]
        for ix in range(len(x)):
            for iq in range(len(q2)):
                row = values[ix][iq]
                f.write(" ".join(f"{row[c]:.17e}" for c in columns) + "\n")
        f.write("---\n")


def alpha_s_2loop(asmz, q, loops=2):
    """alphas of alfas_functions.f at two loops (or one) with cmass 1.42 and
    bmass 4.7."""
    b0 = {3: 0.716197243913527, 4: 0.66314559621623, 5: 0.61009394851893}
    c1 = {3: 0.565884242104515, 4: 0.49019722472304, 5: 0.40134724779695}

    def newton(t, a_in, nf):
        if loops == 1:
            return a_in / (1.0 + a_in * b0[nf] * t)

        def f2(a):
            return 1.0 / a + c1[nf] * math.log((c1[nf] * a) / (1.0 + c1[nf] * a))

        a_out = a_in / (
            1.0 + b0[nf] * a_in * t + c1[nf] * a_in * math.log(1.0 + a_in * b0[nf] * t)
        )
        while True:
            a = a_out
            f = b0[nf] * t + f2(a_in) - f2(a)
            fp = 1.0 / (a**2 * (1.0 + c1[nf] * a))
            a_out = a - f / fp
            if abs(f / fp / a) <= 5e-4:
                return a_out

    zmass, cmass, bmass = 91.188, 1.42, 4.7
    amb = newton(2.0 * math.log(bmass / zmass), asmz, 5)
    amc = newton(2.0 * math.log(cmass / bmass), amb, 4)
    if q < bmass:
        if q < cmass:
            return newton(2.0 * math.log(q / cmass), amc, 3)
        return newton(2.0 * math.log(q / bmass), amb, 4)
    return newton(2.0 * math.log(q / zmass), asmz, 5)


def write_alpha_s_grid(path, asmz, loops=2):
    """The 2-loop (or 1-loop) running from asmz as an LHAPDF alpha_s grid:
    200 nodes per decade from 1 GeV to 100 TeV, with a doubled node at the c
    and b thresholds (the boundary of two subgrids), below and above it."""
    thresholds = (1.42, 4.7)
    nodes = [10 ** (k / 200) for k in range(1001)]
    nodes = sorted([q for q in nodes if min(abs(q / t - 1) for t in thresholds) > 1e-9])
    qs, values = [], []
    for q in sorted(nodes + list(thresholds)):
        if q in thresholds:
            qs.append(q)
            values.append(alpha_s_2loop(asmz, q * (1 - 1e-15), loops))
        qs.append(q)
        values.append(alpha_s_2loop(asmz, q, loops))
    with open(path, "w") as f:
        f.write(f"AlphaS_Qs: [{', '.join(f'{q:.17e}' for q in qs)}]\n")
        f.write(f"AlphaS_Vals: [{', '.join(f'{v:.17e}' for v in values)}]\n")


def table_nodes(pdlabel):
    """The nodes in x and Q of the tabulation of a CTEQ6 or CT14qed set: 40
    per decade in Q and in x up to 0.1, 90 from there to 1."""
    xmin, qmin, qmax = TABLE_RANGES[pdlabel[:5]]
    decades = round(-math.log10(xmin))
    x = [10 ** (k / 40 - decades) for k in range(40 * (decades - 1))]
    x += [0.1 + 0.9 * k / 90 for k in range(91)]
    count = math.ceil(40 * math.log10(qmax / qmin))
    q = [qmin * (qmax / qmin) ** (k / count) for k in range(count + 1)]
    return x, q


def tabulate(pdlabel, output, directory, x, q):
    """x f of the flavours of PIDS of a built-in set of MadFKS at the nodes x
    and q (an array (len(x), len(q), len(PIDS))), from fdist of the aMC@NLO
    output, whose libraries are built, with pdf_table.f compiled into
    directory."""
    os.makedirs(directory, exist_ok=True)
    source = os.path.join(TEMPLATES, "pdf_table.f")
    executable = os.path.join(directory, "pdf_table")
    subprocess.run(
        [
            "gfortran",
            "-O",
            "-o",
            executable,
            source,
            "-I" + os.path.join(output, "Source", "PDF"),
            "-I" + os.path.join(output, "Source"),
            os.path.join(output, "lib", "libpdf.a"),
            os.path.join(output, "lib", "libgeneric.a"),
        ],
        check=True,
        capture_output=True,
    )
    path = os.path.join(directory, f"{pdlabel}.table")
    nodes = f"{pdlabel}\n{len(x)} {len(q)}\n"
    nodes += " ".join(f"{v:.17e}" for v in x) + "\n"
    nodes += " ".join(f"{v:.17e}" for v in q) + "\n"
    subprocess.run(
        [executable, path],
        input=nodes,
        text=True,
        cwd=os.path.join(output, "lib"),
        check=True,
        capture_output=True,
    )
    values = np.loadtxt(path).reshape(len(x), len(q), -1)
    # the columns of the table: -5 to 5 then the photon, in the order of PIDS
    columns = [
        pid + 5 if pid != 21 and pid != 22 else (5 if pid == 21 else 11) for pid in PIDS
    ]
    return values[..., columns]


def write_builtin_grid(path, pdlabel, output, directory):
    """The densities of a built-in set of MadFKS as an LHAPDF grid
    (lhagrid1): the nodes of its NNPDF grid, or its table of fdist at
    table_nodes for the CTEQ6 and CT14qed sets."""
    if pdlabel in NNPDF_GRIDS:
        write_lhapdf_grid(path, NNPDF_GRIDS[pdlabel])
        return
    x, q = table_nodes(pdlabel)
    values = tabulate(pdlabel, output, directory, x, q)
    with open(path, "w") as f:
        f.write("PdfType: central\nFormat: lhagrid1\n---\n")
        f.write(" ".join(f"{v:.17e}" for v in x) + "\n")
        f.write(" ".join(f"{v:.17e}" for v in q) + "\n")
        f.write(" ".join(str(pid) for pid in PIDS) + "\n")
        for ix in range(len(x)):
            for iq in range(len(q)):
                f.write(" ".join(f"{v:.17e}" for v in values[ix, iq]) + "\n")
        f.write("---\n")


def lhapdf_data_paths():
    """The directories of LHAPDF sets: LHAPDF_DATA_PATH, then the data
    directory of lhapdf-config where it is installed."""
    paths = [p for p in os.environ.get("LHAPDF_DATA_PATH", "").split(":") if p]
    try:
        out = subprocess.run(
            ["lhapdf-config", "--datadir"], capture_output=True, text=True, check=False
        )
        if out.returncode == 0 and out.stdout.strip():
            paths.append(out.stdout.strip())
    except OSError:
        pass
    return paths


def lhapdf_set(lhaid, paths=None):
    """The grid (.dat of the member) and the info file (.info) of the LHAPDF
    set of lhaid, from pdfsets.index of the data directories.

    Raises:
        FileNotFoundError: if no data directory has the set.
    """
    paths = lhapdf_data_paths() if paths is None else paths
    for path in paths:
        index = os.path.join(path, "pdfsets.index")
        if not os.path.isfile(index):
            continue
        best = None
        with open(index) as f:
            for line in f:
                fields = line.split()
                if len(fields) >= 2 and fields[0].isdigit():
                    first, name = int(fields[0]), fields[1]
                    if first <= lhaid and (best is None or first > best[0]):
                        best = (first, name)
        if best is None:
            continue
        first, name = best
        member = lhaid - first
        grid = os.path.join(path, name, f"{name}_{member:04d}.dat")
        info = os.path.join(path, name, f"{name}.info")
        if os.path.isfile(grid) and os.path.isfile(info):
            return grid, info
    raise FileNotFoundError(f"no LHAPDF set with lhaid {lhaid} in {paths}")


def lhapdf_alpha_s_mz(info):
    """AlphaS_MZ of the info file of an LHAPDF set."""
    with open(info) as f:
        for line in f:
            if line.split(":")[0].strip() == "AlphaS_MZ":
                return float(line.split(":")[1])
    raise ValueError(f"no AlphaS_MZ in {info}")
