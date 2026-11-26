#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "CPPProcess.h"
#include "rambo.h"

// ---------------------------------------------------------------------------
// Flavor combinations and corresponding PDG codes.
// %(maxflavor)d flavor combination(s); %(nexternal)d external particles each.
// These arrays are filled by the MadGraph7 code generator.
// ---------------------------------------------------------------------------
static const int maxflavor  = %(maxflavor)d;
static const int _nexternal = %(nexternal)d;
// pdg_arr[iflav][ipart]    : actual PDG code for this particle/flavor
static const int pdg_arr[%(maxflavor)d][%(nexternal)d]    = %(pdg_arr)s;

int main(int argc, char** argv){

  // Create a process object
  CPPProcess process("../../Cards/param_card.dat");

  // Centre-of-mass energy: use argv[1] if provided, else default 1000 GeV
  double energy = 1000.0;
  if(argc > 1) energy = atof(argv[1]);
  double total_mass = 0.0;
  vector<double> masses = process.getMasses();
  for(unsigned int i = 0; i < masses.size(); ++i) total_mass += masses[i];
  if(energy <= 2.0 * total_mass) energy = 2.0 * total_mass;
  double weight;

  // Phase-space points: those of PS.input when it is present (any number of
  // them, nexternal lines of E px py pz each, all evaluated in this run so that
  // the helicity filtering learnt on the first points is used for the others),
  // else one RAMBO point with the fixed seed (reproducible).
  vector<vector<double*> > points;
  {
    ifstream psin("PS.input");
    if(psin){
      while(true){
        vector<double*> point;
        bool complete = true;
        for(int i = 0; i < process.nexternal; i++){
          double* mom = new double[4];
          if(!(psin >> mom[0] >> mom[1] >> mom[2] >> mom[3])){
            delete[] mom;
            complete = false;
            break;
          }
          point.push_back(mom);
        }
        if(!complete){
          for(unsigned int i = 0; i < point.size(); i++) delete[] point[i];
          break;
        }
        points.push_back(point);
      }
      if(points.empty()){
        cerr << "No phase-space point in PS.input." << endl;
        return 1;
      }
    }
    else
      points.push_back(get_momenta(process.ninitial, energy,
                                   process.getMasses(), weight));
  }

  for(unsigned int ipoint = 0; ipoint < points.size(); ipoint++){
  vector<double*>& p = points[ipoint];

  // Print phase space point (same format as the Fortran check_sa.f)
  cout << endl << " Phase space point:" << endl << endl;
  cout << "-----------------------------------------------------------------------------" << endl;
  cout << "n        E             px             py              pz" << endl;
  for(int i = 0; i < process.nexternal; i++)
    cout << setw(4) << i+1
         << setiosflags(ios::scientific) << setprecision(7) << setw(15) << p[i][0]
         << setiosflags(ios::scientific) << setprecision(7) << setw(15) << p[i][1]
         << setiosflags(ios::scientific) << setprecision(7) << setw(15) << p[i][2]
         << setiosflags(ios::scientific) << setprecision(7) << setw(15) << p[i][3] << endl;
  cout << "-----------------------------------------------------------------------------" << endl;

  // Set momenta once (shared across all flavor combinations)
  process.setMomenta(p);

  // Loop over flavor combinations (sigmaKin holds the flavor lookup table)
  for(int iflav = 0; iflav < maxflavor; iflav++){
    // Evaluate matrix element for this flavor combination
    double matrix_element = process.sigmaKin(iflav);

    // Print PDG line (same keyword as Fortran so _parse_sa_output can match it)
    cout << " PDG";
    for(int j = 0; j < _nexternal; j++) cout << " " << pdg_arr[iflav][j];
    cout << endl;

    // Print matrix element value
    cout << " Matrix element = " << setprecision(16) << matrix_element
         << " GeV^" << -(2*process.nexternal-8) << endl;

    cout << " -----------------------------------------------------------------------------" << endl;
  }
  }
  vector<double*>& p = points.back();

%(crossing_example)s

  return 0;
}
