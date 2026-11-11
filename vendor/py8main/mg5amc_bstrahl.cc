// mg5amc_bstrahl.cc -- MadGraph-owned Pythia8 main for dressed-lepton
// collisions whose density includes beamstrahlung.
//
// A dressed-lepton density convolutes beamstrahlung with ISR, so the x
// implied by the LHE initial-state momenta is a product, x = xbs * xisr.
// Handing that straight to Pythia makes its space-like shower attribute
// all of 1-x to collinear QED ISR off a full-energy beam, which gives the
// hard system a transverse recoil that beamstrahlung does not produce.
// MadGraph therefore samples the split at event-writing time and records
// it per event as
//
//     <beamstrahlung> xbs1 xbs2 </beamstrahlung>
//
// This main consumes that tag and does two things with it.
//
// 1. A BeamShape degrades each beam's momentum to xbs*p, so Pythia's
//    beams, eCM and backward evolution all refer to the post-beamstrahlung
//    beam and only xisr is left for the shower to reconstruct.
//    Beams:allowMomentumSpread is not gated on frameType (BeamSetup.cc),
//    unlike Beams:allowVariableEnergy, which is read only for frameType<4
//    and so is silently inert with Les Houches input.
//
// 2. An LHAup boosts each event from the frame MadGraph wrote it in into
//    that event's post-beamstrahlung CM.  This is not optional: Pythia
//    never boosts an incoming Les Houches event inward -- BeamSetup::
//    boostAndVertex is only ever called with toLab=true -- so it reads the
//    LHE momenta as already being in the beams' CM frame.  Once the beams
//    are degraded by different amounts that is a different frame from the
//    one MadGraph wrote, and the x<=1 test in ProcessContainer (the
//    "setting mass failed" error) rejects a large fraction of the events.
//    Measured on a 500 GeV ilc500ll sample: 58% of events rejected without
//    the boost, 0% with it.
//
// Deliberately narrow: no merging, no subruns, no HDF5.  Events must be
// showered one-to-one, because a rejected event makes ProcessLevel read a
// further event from the file while the BeamShape has already committed to
// the previous one's beams -- see the desync guard in the event loop.

#include "Pythia8/Pythia.h"
#include "Pythia8/Streams.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(HEPMC3)
#include "Pythia8Plugins/HepMC3.h"
#elif defined(HEPMC2)
#include "Pythia8Plugins/HepMC2.h"
#endif

using namespace Pythia8;

//==========================================================================

// The per-event beamstrahlung fractions, pre-scanned from the event file.
//
// They have to be known one event ahead of the LHAup, because the
// BeamShape is used in Pythia::next() before ProcessLevel reads the event
// it belongs to.  Pre-scanning is the simplest way to get that ordering
// right; it costs one extra pass over the file and requires the file to be
// rereadable, which rules out a fifo.

class BstrahlTags {

public:

  bool read(string fname) {
    frac.clear();
    nEvent = 0;
    igzstream is(fname.c_str());
    if (!is.good()) return false;
    string line;
    bool inEvent = false;
    while (getline(is, line)) {
      // Only look for the element between <event> and </event>.  The LHEF
      // header embeds the run card and the shower card verbatim, so a line
      // there that happens to contain the element name would otherwise be
      // counted as one event's fractions and shift every later event by
      // one -- and, since the counts are now checked, abort the run.
      if (line.find("</event>") != string::npos) { inEvent = false; continue; }
      if (line.find("<event") != string::npos) {
        ++nEvent;
        inEvent = true;
        continue;
      }
      if (!inEvent) continue;
      size_t pos = line.find("<beamstrahlung>");
      if (pos == string::npos) continue;
      istringstream ss(line.substr(pos + 15));
      double y1, y2;
      if (ss >> y1 >> y2) frac.push_back(make_pair(y1, y2));
    }
    return true;
  }

  // Number of <event> markers seen while scanning.  Zero means the scan
  // read nothing usable -- most likely a compressed file and a Pythia
  // built without GZIP, where igzstream is a plain ifstream (Streams.h).
  // That has to be an error rather than "no beamstrahlung", because
  // silently showering with undegraded beams is the exact failure this
  // main exists to prevent.
  int nEventSeen() const { return nEvent; }

  int size() const { return int(frac.size()); }

  // Out of range returns (1,1), i.e. an undegraded beam, so that a file
  // with no tags at all behaves exactly like a plain shower run.
  pair<double, double> get(int i) const {
    if (i < 0 || i >= int(frac.size())) return make_pair(1., 1.);
    return frac[i];
  }

private:

  vector<pair<double, double> > frac;
  int nEvent = 0;

};

//==========================================================================

// Degrade the beams by the current event's beamstrahlung fractions.
// deltaPA/deltaPB are three-momentum shifts whose energy Pythia recomputes
// on shell, which is exactly what emitting a collinear photon does to the
// beam lepton.

class BstrahlShape : public BeamShape {

public:

  BstrahlShape(const BstrahlTags* tagsIn) : tags(tagsIn), pzA(0.), iEv(0) {}

  // The beam momentum is only known after Pythia::init(); the shape is
  // first used inside next(), so setting it in between is in time.
  void setBeamPz(double pzAin) { pzA = pzAin; }

  void pick() override {
    deltaPxA = deltaPyA = deltaPzA = 0.;
    deltaPxB = deltaPyB = deltaPzB = 0.;
    vertexX = vertexY = vertexZ = vertexT = 0.;
    pair<double, double> y = tags->get(iEv);
    // (y-1)*pz is sign safe for both beams, since pzA > 0 and pzB < 0.
    deltaPzA = (y.first  - 1.) *  pzA;
    deltaPzB = (y.second - 1.) * -pzA;
    ++iEv;
  }

  int nPicked() const { return iEv; }

private:

  const BstrahlTags* tags;
  double pzA;
  int iEv;

};

//==========================================================================

// Read the event as usual, then boost it into its post-beamstrahlung CM.

class BstrahlLHAup : public LHAupLHEF {

public:

  BstrahlLHAup(Pythia8::Info* infoPtrIn, const char* fname,
    const BstrahlTags* tagsIn, bool setScales)
    : LHAupLHEF(infoPtrIn, fname, nullptr, false, setScales),
      tags(tagsIn), iEv(0) {}

  bool setEvent(int idProc = 0) override {
    if (!LHAupLHEF::setEvent(idProc)) return false;
    pair<double, double> y = tags->get(iEv);
    ++iEv;
    double sum = y.first + y.second;
    if (sum <= 0.) return true;
    double beta = (y.first - y.second) / sum;
    if (std::abs(beta) < 1e-15) return true;
    double gamma = 1. / std::sqrt(1. - beta * beta);
    // 'particles' is what LHAup::e()/pz() serve to Pythia; particlesSave
    // is the separate buffer used when writing an LHEF back out, so
    // boosting that one instead is a silent no-op.
    // Entry 0 is the empty placeholder of the Les Houches record.
    for (size_t i = 1; i < particles.size(); ++i) {
      double e  = particles[i].ePart;
      double pz = particles[i].pzPart;
      particles[i].ePart  = gamma * (e  - beta * pz);
      particles[i].pzPart = gamma * (pz - beta * e);
    }
    return true;
  }

  int nRead() const { return iEv; }

private:

  const BstrahlTags* tags;
  int iEv;

};

//==========================================================================

int main(int argc, char* argv[]) {

  // Same command line as the Pythia mains MadGraph already drives.
  string cmndFile;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) cmndFile = argv[++i];
    else if (cmndFile.empty()) cmndFile = argv[i];
  }
  if (cmndFile.empty()) {
    cout << " Usage: mg5amc_bstrahl -c <card>" << endl;
    return 1;
  }

  Pythia pythia;
  // Same settings main164 registers, so that a card written for it works
  // here unchanged.
  pythia.settings.addFlag("Main:HepMC", false);
  pythia.settings.addWord("HepMC:output", "events.hepmc");
  // MadGraph writes Beams:LHEF inside a subrun block (banner.py translates
  // its own card into Pythia's names before we ever see it), so the card
  // has to be read with a subrun index rather than as a flat file.
  pythia.readFile(cmndFile, 0);

  if (pythia.mode("Main:numberOfSubruns") > 1) {
    cout << " mg5amc_bstrahl: ABORT -- this main showers a single subrun,"
         << " but the card declares " << pythia.mode("Main:numberOfSubruns")
         << ".\n   Beamstrahlung fractions belong to one event file, so"
         << " subruns would need one\n   set of fractions each." << endl;
    return 1;
  }

  string lheFile = pythia.word("Beams:LHEF");
  if (lheFile.empty()) {
    cout << " mg5amc_bstrahl: Beams:LHEF is not set" << endl;
    return 1;
  }

  BstrahlTags tags;
  if (!tags.read(lheFile)) {
    cout << " mg5amc_bstrahl: cannot open " << lheFile << endl;
    return 1;
  }
  cout << " mg5amc_bstrahl: " << tags.size() << " <beamstrahlung> tags in "
       << tags.nEventSeen() << " events of " << lheFile << endl;
  if (tags.nEventSeen() == 0) {
    cout << " mg5amc_bstrahl: ABORT -- found no events while scanning "
         << lheFile << ".\n   If the file is compressed, this Pythia was"
         << " built without GZIP support, so the\n   beamstrahlung"
         << " fractions cannot be read and the beams would be left"
         << " undegraded.\n   Rebuild Pythia with --with-gzip, or hand"
         << " this main an uncompressed file." << endl;
    return 1;
  }
  if (tags.size() == 0)
    cout << " mg5amc_bstrahl: no beamstrahlung tags; beams not degraded"
         << endl;
  else if (tags.size() != tags.nEventSeen()) {
    // The fractions are consumed strictly by position, so a file in which
    // only some events carry the tag -- two runs concatenated, or a tool
    // that dropped the element on some events -- would degrade every event
    // after the first gap with another event's beams.  The event loop's
    // counter guard cannot see that, since both counters still advance
    // together.  Refuse, for the same reason that guard refuses.
    cout << " mg5amc_bstrahl: ABORT -- " << tags.size() << " <beamstrahlung>"
         << " tags for " << tags.nEventSeen() << " events in " << lheFile
         << ".\n   The two must correspond one-to-one: the fractions are"
         << " matched to events by\n   position, so a file in which only"
         << " some events carry the tag would shower\n   every later event"
         << " against another event's beams. Refusing to guess." << endl;
    return 1;
  }

  // Our own LHAup, so frameType is 5 rather than 4.  Beams:allowMomentumSpread
  // is read for both.
  bool setScales = pythia.flag("Beams:setProductionScalesFromLHEF")
                || pythia.flag("Beams:setDipoleShowerStartingScalesFromLHEF");
  // Pythia::info is a const reference, so user code that constructs an
  // LHAup has to cast it back; the distribution's own main428 does the same.
  Info* infoPtr = const_cast<Info*>(&pythia.info);
  shared_ptr<BstrahlLHAup> lhaUp = make_shared<BstrahlLHAup>(
    infoPtr, lheFile.c_str(), &tags, setScales);
  pythia.settings.mode("Beams:frameType", 5);
  pythia.setLHAupPtr(lhaUp);

  shared_ptr<BstrahlShape> shape = make_shared<BstrahlShape>(&tags);
  pythia.settings.flag("Beams:allowMomentumSpread", true);
  pythia.setBeamShapePtr(shape);

  if (!pythia.init()) {
    cout << " mg5amc_bstrahl: initialization failed" << endl;
    return 1;
  }
  shape->setBeamPz(pythia.info.pzA());

  // Same two settings main164 uses. MadGraph switches the output off by
  // pointing it at /dev/null rather than by clearing Main:HepMC.
  const string hepmcFile = pythia.word("HepMC:output");
  const bool wantHepMC = pythia.flag("Main:HepMC")
                      && hepmcFile != "/dev/null" && !hepmcFile.empty();
#if defined(HEPMC2) || defined(HEPMC3)
  const bool doHepMC = wantHepMC;
  Pythia8ToHepMC toHepMC(pythia.word("HepMC:output"));
  toHepMC.set_print_inconsistency(false);
  toHepMC.set_free_parton_warnings(false);
#else
  if (wantHepMC)
    cout << " mg5amc_bstrahl: WARNING -- HepMC output was requested but this"
         << " Pythia8 was built without HepMC; no event file will be written."
         << endl;
#endif

  long nEvent = pythia.mode("Main:numberOfEvents");
  if (nEvent < 0) nEvent = tags.size() > 0 ? tags.size() : 1000000000;
  int nAbort = pythia.mode("Main:timesAllowErrors");
  int iAbort = 0;

  for (long iEvent = 0; iEvent < nEvent; ++iEvent) {
    if (!pythia.next()) {
      if (pythia.info.atEndOfFile()) break;
      if (++iAbort < nAbort) continue;
      cout << " mg5amc_bstrahl: too many errors, giving up" << endl;
      break;
    }

    // The beams were degraded before the event was read, so the two must
    // stay in lockstep.  They do not if ProcessLevel rejects an event and
    // reads another within the same next() call, which would silently
    // shower an event against another event's beams.
    if (lhaUp->nRead() != shape->nPicked()) {
      cout << " mg5amc_bstrahl: ABORT -- event and beam counters diverged ("
           << lhaUp->nRead() << " events read, " << shape->nPicked()
           << " beam settings used). The beamstrahlung fractions no longer"
           << " match the events; refusing to write misassigned events."
           << endl;
      return 1;
    }

#if defined(HEPMC2) || defined(HEPMC3)
    if (doHepMC) toHepMC.writeNextEvent(pythia);
#endif
  }

  // MadGraph parses the cross section and the tried/selected/accepted
  // counts out of this, so it has to be the standard Pythia summary.
  pythia.stat();

  return 0;
}
