#ifndef aloha_aux_functions_guard
#define aloha_aux_functions_guard
#include <iostream>
double Sgn(double e,double f);

struct ALOHAOBJ{
     double p[4];
     std::complex<double> W[4];
     int flv_index =1;

     public:
        //ALOHAOBJ(double p[4], std::complex<double> W[4], int flav):p(p), W(W), flav(flav){};
        inline ALOHAOBJ() {};
};
//ALOHAOBJ::ALOHAOBJ() {}


#endif
