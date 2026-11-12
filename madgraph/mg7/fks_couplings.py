"""The PDFs and alpha_s of the fixed-order NLO driver of madspace
(fks_driver.py), as madspace PdfGrid and AlphaSGrid files.

With pdlabel lhapdf they are those of the LHAPDF set of lhaid. With pdlabel
nn23nlo, the NNPDF 2.3 grid that MadGraph ships for MadFKS, its nodes are
written as an LHAPDF grid (lhagrid1, the flavours fdist of pdf.f takes) and
alpha_s is the 2-loop running of alfas_functions.f from alpha_s(m_Z), on 200
nodes per decade. The interpolation of madspace on these nodes differs from
that of NNPDFDriver.f by up to 3e-3 at large x and low scales (see
test_fks_couplings.py of madspace).
"""

import math
import os
import subprocess

from madgraph import MG5DIR

NNPDF_GRID = os.path.join(
    MG5DIR, "Template", "LO", "lib", "Pdfdata", "NNPDF23nlo_as_0119_qed_mem0.grid"
)
# alpha_s(m_Z) of nn23nlo (pdfwrap of MadFKS)
NNPDF_ALPHA_S = 0.119
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


def alpha_s_2loop(asmz, q):
    """alphas of alfas_functions.f at two loops with cmass 1.42 and bmass
    4.7."""
    b0 = {3: 0.716197243913527, 4: 0.66314559621623, 5: 0.61009394851893}
    c1 = {3: 0.565884242104515, 4: 0.49019722472304, 5: 0.40134724779695}

    def newton(t, a_in, nf):
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


def write_alpha_s_grid(path, asmz):
    """The 2-loop running from asmz as an LHAPDF alpha_s grid: 200 nodes per
    decade from 1 GeV to 100 TeV, with a doubled node at the c and b
    thresholds (the boundary of two subgrids), below and above it."""
    thresholds = (1.42, 4.7)
    nodes = [10 ** (k / 200) for k in range(1001)]
    nodes = sorted([q for q in nodes if min(abs(q / t - 1) for t in thresholds) > 1e-9])
    qs, values = [], []
    for q in sorted(nodes + list(thresholds)):
        if q in thresholds:
            qs.append(q)
            values.append(alpha_s_2loop(asmz, q * (1 - 1e-15)))
        qs.append(q)
        values.append(alpha_s_2loop(asmz, q))
    with open(path, "w") as f:
        f.write(f"AlphaS_Qs: [{', '.join(f'{q:.17e}' for q in qs)}]\n")
        f.write(f"AlphaS_Vals: [{', '.join(f'{v:.17e}' for v in values)}]\n")


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
