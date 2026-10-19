// SPDX-License-Identifier: 0BSD
#include "rusticol_virtual.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
int main(int argc,char **argv) {
    assert(argc==2);
    rusticol::VirtualRuntime runtime(argv[1]);
    assert(runtime.spin_correlation_classes_json()=="[[3],[4],[3,4]]");
    const std::vector<double> p={500,0,0,500,500,0,0,-500,500,300,0,400,500,-300,0,-400};
    std::vector<std::size_t> legs={3,4};
    std::vector<std::complex<double>> vectors={{0,0},{.8,0},{0,1},{-.6,0},{0,0},{.8,0},{0,1},{-.6,0}};
    auto baseline=runtime.evaluate_correlated_with_sources(p,1,{1e6},legs,vectors,{UINT64_MAX});
    for(std::size_t i=0;i<4;i++) vectors[i]*=std::complex<double>(2,-3);
    for(std::size_t i=4;i<8;i++) vectors[i]*=std::complex<double>(1,2);
    auto scaled=runtime.evaluate_correlated_with_sources(p,1,{1e6},legs,vectors);
    auto a=baseline.virtual_values(),b=scaled.virtual_values();
    assert(a.size()==9 && baseline.successful_ids()[0]==UINT64_MAX);
    for(std::size_t i=0;i<a.size();i++) assert(std::abs(b[i]-65.*a[i])<2e-8*std::max(1.,std::abs(b[i])));
    vectors.clear(); legs.clear(); runtime.clear(); runtime.close();
    assert(baseline.spin_legs_json()=="[3,4]" && baseline.born().size()==3);
    assert(baseline.histories_json()=="[\"identity\",\"ordered\",\"swapped\"]");
    std::cout << "C++ literal joint sources and owned labels passed\n";
}
