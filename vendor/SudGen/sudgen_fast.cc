// sudgen_fast: rows of the MC@NLO-Delta no-emission tables, with an
// unmodified Pythia8.
//
// A table entry is the probability that one dipole of mass xm does not
// radiate between stupp and st (Merging::generateSingleSudakov, which
// runs the trial showers of the shower's noEmissionProbability). The
// trial showers are Markovian, so for the nodes st_1 < ... < st_n = stupp
//
//   Delta(st_i) = prod_{k=i}^{n-1} P(st_{k+1} -> st_k),
//
// the product of the no-emission probabilities over adjacent intervals.
// One pass down the nodes gives a whole row. Compared with one trial run
// from stupp per node, this needs far fewer trial steps, it is monotone in
// st by construction, and its relative error,
//
//   sqrt( sum_k (1 - P_k) / (P_k N_k) ),
//
// stays small deep into the Sudakov region, where single runs return
// statistical zeros. N_k = nrep x (trials per call: 10^4 for final-state,
// 5x10^5 for initial-state radiators in Pythia 8.3).
//
// Usage:
//   sudgen_fast <settings.cmd> <type> <id> <ixm_first> <ixm_last> <nrep>
//               <seed> <output> [<ist_first> <ist_last> direct]
//   type: 1 II, 2 FF, 3 IF, 4 FI (as Merging::generateSingleSudakov)
//   id:   PDG id of the radiator (1-6, 21)
//   ixm:  mass nodes, 1..50; nrep: calls per interval.
// Output, one line per node: ixm ist st xm Delta relative_error.
// The optional trailing arguments compute single nodes directly from stupp
// (the previous method), for validation.

#include "Pythia8/Pythia.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace Pythia8;

namespace {

const int nnst = 100, nnxm = 50;
const double stlow = 1., stupp = 7000., xmlow = 1., xmupp = 7000.;
// Beam energy squared and momentum fraction of the dummy beams, as in the
// previous generation interface (they enter only initial-state recoilers).
const double sBeams = 7000. * 7000., xBeam = 0.1;

double node(int i, int n, double low, double upp) {
  return low * std::pow(upp / low, double(i - 1) / double(n - 1));
}

}

int main(int argc, char* argv[]) {
  if (argc != 9 && argc != 12) {
    std::fprintf(stderr, "usage: %s <settings.cmd> <type> <id> <ixm_first> "
      "<ixm_last> <nrep> <seed> <output> [<ist_first> <ist_last> direct]\n",
      argv[0]);
    return 1;
  }
  const std::string cmd = argv[1];
  const int type = std::atoi(argv[2]), id = std::atoi(argv[3]);
  const int ixm1 = std::atoi(argv[4]), ixm2 = std::atoi(argv[5]);
  const int nrep = std::atoi(argv[6]), seed = std::atoi(argv[7]);
  const bool direct = (argc == 12);
  const int ist1 = direct ? std::atoi(argv[9]) : 1;
  const int ist2 = direct ? std::atoi(argv[10]) : nnst;
  if (type < 1 || type > 4 || ixm1 < 1 || ixm2 > nnxm || ixm1 > ixm2
    || nrep < 1 || ist1 < 1 || ist2 > nnst || ist1 > ist2) {
    std::fprintf(stderr, "invalid arguments\n");
    return 1;
  }

  Pythia pythia("../share/Pythia8/xmldoc", false);
  if (!pythia.readFile(cmd)) {
    std::fprintf(stderr, "cannot read %s\n", cmd.c_str());
    return 1;
  }
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + std::to_string(seed));
  if (!pythia.init() || !pythia.mergingPtr) {
    std::fprintf(stderr, "Pythia initialisation failed\n");
    return 2;
  }
  // Trials per noEmissionProbability call, for the error estimate.
  const double nTrials = (type == 1 || type == 3) ? 5e5 : 1e4;
  // The coupling actually initialised, for the table metadata.
  const std::string shower = (type == 1 || type == 3) ? "SpaceShower" : "TimeShower";

  FILE* out = std::fopen(argv[8], "w");
  if (!out) return 1;
  std::fprintf(out, "# Pythia %.3f type %d id %d nrep %d seed %d %s\n",
    pythia.settings.parm("Pythia:versionNumber"), type, id, nrep, seed,
    direct ? "direct" : "product");
  std::fprintf(out, "# alphaSvalue %.6f alphaSorder %d alphaSuseCMW %d "
    "alphaSnfmax %d pdfMode %d MEcorrections %d m%d %.6f\n",
    pythia.settings.parm(shower + ":alphaSvalue"),
    pythia.settings.mode(shower + ":alphaSorder"),
    int(pythia.settings.flag(shower + ":alphaSuseCMW")),
    pythia.settings.mode("StandardModel:alphaSnfmax"),
    pythia.settings.mode(shower + ":pdfMode"),
    int(pythia.settings.flag("TimeShower:MEcorrections")), id,
    pythia.particleData.m0(id));
  for (int ixm = ixm1; ixm <= ixm2; ++ixm) {
    const double xm = node(ixm, nnxm, xmlow, xmupp), m2 = xm * xm;
    if (direct) {
      // Previous method: one run from stupp per node.
      for (int ist = ist1; ist <= ist2; ++ist) {
        const double st = node(ist, nnst, stlow, stupp);
        double sum = 0.;
        for (int r = 0; r < nrep; ++r)
          sum += pythia.mergingPtr->generateSingleSudakov(stupp, st, m2,
            id, type, sBeams, xBeam);
        const double delta = sum / nrep;
        const double err = (delta > 0.) ?
          std::sqrt((1. - delta) / (delta * nrep * nTrials)) : -1.;
        std::fprintf(out, "%d %d %.8e %.8e %.10e %.4e\n", ixm, ist, st, xm,
          delta, err);
      }
      continue;
    }
    std::vector<double> delta(nnst + 1, 1.), relvar(nnst + 1, 0.);
    for (int ist = nnst - 1; ist >= 1; --ist) {
      const double upp = node(ist + 1, nnst, stlow, stupp);
      const double low = node(ist, nnst, stlow, stupp);
      double sum = 0.;
      for (int r = 0; r < nrep; ++r)
        sum += pythia.mergingPtr->generateSingleSudakov(upp, low, m2, id,
          type, sBeams, xBeam);
      const double p = sum / nrep;
      delta[ist] = delta[ist + 1] * p;
      relvar[ist] = relvar[ist + 1] + ((p > 0.) ?
        (1. - p) / (p * nrep * nTrials) : 1e30);
    }
    for (int ist = 1; ist <= nnst; ++ist)
      std::fprintf(out, "%d %d %.8e %.8e %.10e %.4e\n", ixm, ist,
        node(ist, nnst, stlow, stupp), xm, delta[ist],
        std::sqrt(relvar[ist]));
    std::fflush(out);
  }
  std::fclose(out);
  return 0;
}
