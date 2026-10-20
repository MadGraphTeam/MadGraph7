// Direct callback oracle for the original Ninja implementation, no reduction.
#include <ninja/tensor_ninja.hh>
#include <cstdio>
#include <vector>
int main(){unsigned rows; if(scanf("%u",&rows)!=1)return 2;for(unsigned row=0;row<rows;++row){unsigned n,rank,count; if(scanf("%u%u%u",&n,&rank,&count)!=3)return 3;std::vector<ninja::Complex> coeff(count);double re,im;for(auto&z:coeff){if(scanf("%lf%lf",&re,&im)!=2)return 4;z=ninja::Complex(re,im);}ninja::ComplexMomentum a,e3,e4;for(auto* v:{&a,&e3,&e4})for(unsigned k=0;k<4;++k){if(scanf("%lf%lf",&re,&im)!=2)return 5;(*v)[k]=ninja::Complex(re,im);}if(scanf("%lf%lf",&re,&im)!=2)return 6;ninja::Complex p(re,im),out[4];ninja::TensorNumerator tensor(n,rank,coeff.data());tensor.t3Expansion(a,e3,e4,p,2,0,nullptr,out);printf("%u",row);for(auto z:out)printf(" %.17e %.17e",z.real(),z.imag());puts("");}return 0;}
