// SPDX-License-Identifier: 0BSD
#include "rusticol_virtual.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
int main(int argc,char **argv) {
    assert(argc==2);
    rusticol::VirtualRuntime runtime(argv[1]);
    const std::vector<double> p={500,0,0,500,500,0,0,-500,500,300,0,400,500,-300,0,-400};
    auto result=runtime.evaluate_correlated(p,1,{1e6},{UINT64_MAX});
    assert(result.shape().axes.histories==4 && result.successful_ids()[0]==UINT64_MAX);
    auto values=result.virtual_values();
    assert(std::abs(values[6].imag())>1e-8);
    for(std::size_t e=0;e<3;e++) assert(std::abs(values[6+e]-std::conj(values[9+e]))<1e-8);
    runtime.close();
    assert(result.histories_json()=="[\"identity\",\"casimir\",\"ordered\",\"swapped\"]");
    assert(result.born().size()==4);
    std::cout << "C++ owned complex histories passed\n";
}
