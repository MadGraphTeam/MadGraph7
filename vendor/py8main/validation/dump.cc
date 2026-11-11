// Dump per-event quantities for the validation plots.
//   ./dump <lhe> <nEvent> <0|1 degrade> <ebeam>
// writes one line per event:
//   y1 y2 x1 x2 eCM sumEfinal ptHard

#include "Pythia8/Pythia.h"
#include "Pythia8/Streams.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace Pythia8;

class Tags {
public:
  bool read(const string& f) {
    igzstream is(f.c_str());
    if (!is.good()) return false;
    string line;
    while (getline(is, line)) {
      size_t p = line.find("<beamstrahlung>");
      if (p == string::npos) continue;
      istringstream ss(line.substr(p + 15));
      double a, b;
      if (ss >> a >> b) frac.push_back(make_pair(a, b));
    }
    return !frac.empty();
  }
  pair<double, double> get(int i) const {
    if (i < 0 || i >= int(frac.size())) return make_pair(1., 1.);
    return frac[i];
  }
  vector<pair<double, double> > frac;
};

class Shape : public BeamShape {
public:
  Shape(const Tags* t) : tags(t), pz(0.), i(0) {}
  void setPz(double p) { pz = p; }
  void pick() override {
    deltaPxA = deltaPyA = deltaPxB = deltaPyB = 0.;
    vertexX = vertexY = vertexZ = vertexT = 0.;
    pair<double, double> y = tags->get(i);
    deltaPzA = (y.first - 1.) * pz;
    deltaPzB = (y.second - 1.) * -pz;
    ++i;
  }
  const Tags* tags; double pz; int i;
};

class Lha : public LHAupLHEF {
public:
  Lha(Info* ip, const char* f, const Tags* t)
    : LHAupLHEF(ip, f, nullptr, false, false), tags(t), i(0) {}
  bool setEvent(int idProc = 0) override {
    if (!LHAupLHEF::setEvent(idProc)) return false;
    pair<double, double> y = tags->get(i);
    ++i;
    double sum = y.first + y.second;
    if (sum <= 0.) return true;
    double beta = (y.first - y.second) / sum;
    if (std::abs(beta) < 1e-15) return true;
    double g = 1. / std::sqrt(1. - beta * beta);
    for (size_t k = 1; k < particles.size(); ++k) {
      double e = particles[k].ePart, pz = particles[k].pzPart;
      particles[k].ePart  = g * (e  - beta * pz);
      particles[k].pzPart = g * (pz - beta * e);
    }
    return true;
  }
  const Tags* tags; int i;
};

int main(int argc, char** argv) {
  string lhe = argv[1];
  int nEvent = atoi(argv[2]);
  bool degrade = atoi(argv[3]) != 0;
  double eBeam = atof(argv[4]);

  Tags tags;
  if (!tags.read(lhe)) { cerr << "no tags\n"; return 1; }

  Pythia pythia;
  pythia.readString("Beams:LHEF = " + lhe);
  pythia.readString("PartonLevel:MPI = off");
  pythia.readString("HadronLevel:all = off");
  pythia.readString("PDF:lepton = on");
  pythia.readString("SpaceShower:QEDshowerByL = on");
  pythia.readString("TimeShower:QEDshowerByL = on");
  pythia.readString("Print:quiet = on");
  pythia.readString("Next:numberShowEvent = 0");

  shared_ptr<Shape> shape;
  shared_ptr<Lha> lha;
  if (degrade) {
    pythia.readString("Beams:allowMomentumSpread = on");
    Info* ip = const_cast<Info*>(&pythia.info);
    lha = make_shared<Lha>(ip, lhe.c_str(), &tags);
    pythia.settings.mode("Beams:frameType", 5);
    pythia.setLHAupPtr(lha);
    shape = make_shared<Shape>(&tags);
    pythia.setBeamShapePtr(shape);
  } else {
    pythia.readString("Beams:frameType = 4");
  }
  if (!pythia.init()) { cerr << "init failed\n"; return 1; }
  if (degrade) shape->setPz(pythia.info.pzA());

  int iEv = -1, nOk = 0;
  for (int i = 0; i < nEvent; ++i) {
    ++iEv;
    if (!pythia.next()) { if (pythia.info.atEndOfFile()) break; continue; }
    pair<double, double> y = tags.get(iEv);
    Vec4 hard;
    double eSum = 0., eGam = 0.;
    for (int j = 0; j < pythia.event.size(); ++j) if (pythia.event[j].isFinal()) {
      eSum += pythia.event[j].e();
      if (pythia.event[j].idAbs() == 13) hard += pythia.event[j].p();
      // photons within ~2.5 degrees of a beam: the ISR the shower built
      if (pythia.event[j].id() == 22 && pythia.event[j].e() > 0. &&
          std::abs(pythia.event[j].pz()) / pythia.event[j].e() > 0.999)
        eGam += pythia.event[j].e();
    }
    ++nOk;
    printf("%.10f %.10f %.10f %.10f %.8f %.8f %.8f %.8f\n",
           y.first, y.second, pythia.info.x1(), pythia.info.x2(),
           pythia.info.eCM(), eSum, hard.pT(), eGam);
  }
  fprintf(stderr, "dumped %d events\n", nOk);
  pythia.stat();
  return 0;
}
