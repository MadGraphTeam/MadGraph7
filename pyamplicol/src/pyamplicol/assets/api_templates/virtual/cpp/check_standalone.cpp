// SPDX-License-Identifier: 0BSD
#include "rusticol_virtual.hpp"
#include "../common/input.h"
#include <iostream>
int main(int argc,char **argv) {
  try {
    double scale; bool resolved=pa_args(argc,argv,&scale)!=0;
    rusticol::VirtualRuntime runtime(".",argv[1]);
    auto axes=runtime.shape(); auto input=pa_read_points(argv[1],axes.external_count);
    std::vector<double> momenta(input.momenta,input.momenta+pa_product(pa_product(input.count,axes.external_count),4));
    free(input.momenta);
    auto result=runtime.evaluate(momenta,input.count,{scale},{},resolved);
    auto shape=result.shape();
    std::cout<<runtime.metadata_json()<<'\n';
    std::cout<<"shape requested="<<shape.requested_points<<" successful="<<shape.successful_points
      <<" born_orders="<<axes.born_orders<<" virtual_orders="<<axes.virtual_orders
      <<" helicities="<<(resolved?axes.helicities:1)<<" born_colours="<<(resolved?axes.born_color_components:1)
      <<" virtual_colours="<<(resolved?axes.virtual_color_components:1)<<" epsilon=3 resolved="<<shape.resolved<<'\n';
    auto ids=result.successful_ids(); pa_ids("successful_ids",ids.data(),ids.size());
    std::cout<<"statuses"; for(auto status:result.statuses()) std::cout<<' '<<status; std::cout<<'\n';
    auto born=result.born(),loop=result.virtual_values();
    pa_values("born",born.data(),born.size()); pa_values("virtual",loop.data(),loop.size());
    std::cout<<result.diagnostics_json()<<'\n';
    return 0;
  } catch(const std::exception &error) { std::cerr<<error.what()<<'\n'; return 1; }
}
