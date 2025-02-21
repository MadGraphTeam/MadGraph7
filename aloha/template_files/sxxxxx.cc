#include <complex>
#include <cmath>
#include "aloha_aux_functions.h"
using namespace std;
void sxxxxx(double p[4],int nss, ALOHAOBJ &sc){
  sc.W[0] = complex<double>(1.00,0.00);
  
for ( int i =0; i<4;i++){
    sc.p[i] = nss*p[i];    
}
  return;
}
