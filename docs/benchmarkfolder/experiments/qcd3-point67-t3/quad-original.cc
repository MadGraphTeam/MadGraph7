// Direct callback oracle for the original Ninja implementation, no reduction.
#include <quadninja/tensor_ninja.hh>
#include <cstdio>
#include <vector>
int main(){unsigned rows; if(scanf("%u",&rows)!=1)return 2;for(unsigned row=0;row<rows;++row){unsigned n,rank,count; if(scanf("%u%u%u",&n,&rank,&count)!=3)return 3;std::vector<quadninja::Complex> coeff(count);double re,im;for(auto&z:coeff){if(scanf("%lf%lf",&re,&im)!=2)return 4;z=quadninja::Complex(re,im);}quadninja::ComplexMomentum a,e3,e4;for(auto* v:{&a,&e3,&e4})for(unsigned k=0;k<4;++k){if(scanf("%lf%lf",&re,&im)!=2)return 5;(*v)[k]=quadninja::Complex(re,im);}if(scanf("%lf%lf",&re,&im)!=2)return 6;quadninja::Complex p(re,im),out[4];quadninja::TensorNumerator tensor(n,rank,coeff.data());tensor.t3Expansion(a,e3,e4,p,2,0,nullptr,out);printf("%u",row);for(auto z:out)printf(" %.17e %.17e",double(quadninja::real(z)),double(quadninja::imag(z)));puts("");}return 0;}
