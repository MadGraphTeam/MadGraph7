#include <cmath>
using namespace std;

double Sgn(double a,double b){ 
  return (b<0)?-abs(a):abs(a);
}


std::ostream& operator<<(std::ostream& os, const ALOHAOBJ& obj) {
    os << "p = [ ";
    for (int i = 0; i < 4; ++i) os << obj.p[i] <<  " " ;
    os << " ]\n";
    os << "W = [ ";
    for (int i = 0; i < 4; ++i) os << obj.W[i] << " " ;
    os << " ]\n";
    return os;
}
