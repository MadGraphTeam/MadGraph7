// SPDX-License-Identifier: 0BSD
// ./client-cpp ARTIFACT [gg|uudd]; the uudd fixture has three squared orders.
#include "rusticol_virtual.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <type_traits>

static void close_value(double a, double b) {
    if (!(std::abs(a-b) <= 1e-8*std::max({std::abs(a),std::abs(b),1e-20}))) {
        std::cerr<<"mismatch "<<a<<" versus "<<b<<"\n"; std::exit(1);
    }
}
int main(int argc,char **argv) {
    if (argc<2 || argc>3) return 2;
    const std::string process=argc==3 ? argv[2] : "gg";
    static_assert(!std::is_copy_constructible_v<rusticol::VirtualRuntime>);
    static_assert(!std::is_copy_constructible_v<rusticol::VirtualEvaluation>);
    rusticol::VirtualRuntime original(argv[1],process,std::nullopt,"model");
    auto runtime=std::move(original);
    const auto axes=runtime.shape();
    assert(axes.external_count==4 && axes.resolved_available && axes.epsilon_count==3);
    assert(axes.born_orders==(process=="gg" ? 1u : 3u));
    assert(runtime.metadata_json().find("\"epsilon_powers\":[0,-1,-2]")!=std::string::npos);
    assert(!runtime.capabilities_json().empty());
    const std::vector<double> point{500,0,0,500,500,0,0,-500,500,300,0,400,500,-300,0,-400};
    std::vector<double> momenta;
    for(int i=0;i<3;++i) momenta.insert(momenta.end(),point.begin(),point.end());
    const std::vector<std::uint64_t> ids{91,4,70};
    auto inclusive=runtime.evaluate(momenta,3,{1e6,9e5,1e6},ids);
    auto resolved=runtime.evaluate(momenta,3,{1e6,9e5,1e6},ids,true);
    assert(inclusive.successful_ids()==ids && resolved.requested_ids()==ids);
    assert(inclusive.statuses()==std::vector<std::uint32_t>(3,RUSTICOL_VIRTUAL_SUCCESS));
    const auto born=inclusive.born(), virt=inclusive.virtual_values();
    const auto rb=resolved.born(), rv=resolved.virtual_values();
    for(std::size_t p=0;p<3;++p) {
        for(std::size_t o=0;o<axes.born_orders;++o) {
            const auto start=(p*axes.born_orders+o)*axes.helicities;
            close_value(std::accumulate(rb.begin()+start,rb.begin()+start+axes.helicities,0.),born[p*axes.born_orders+o]);
        }
        for(std::size_t o=0;o<axes.virtual_orders;++o)
            for(std::size_t e=0;e<3;++e) {
                double sum=0;
                for(std::size_t h=0;h<axes.helicities;++h) sum+=rv[((p*axes.virtual_orders+o)*axes.helicities+h)*3+e];
                close_value(sum,virt[(p*axes.virtual_orders+o)*3+e]);
            }
    }
    if(process=="gg") close_value(virt[2],-12.*0.118/(2*std::acos(-1.))*born[0]);
    else {
        // Independent MadLoop full-CKM fixture, first point, each order separately.
        const std::vector<double> expected_born{0.8013367806451908,-2.0322878122576093,3.0512301482522957};
        const double expected[3][3]={{-0.19094989612756078,-0.28421984016919477,-0.08026310265110026},
            {1.313041990604555,0.7811563156803705,0.20355701776284274},
            {-3.018755956775497,-1.1476469718544264,-0.3056158216076459}};
        // Declared catalogue: Born (2,2),(4,0),(0,4); virtual (2,4),(4,2),(6,0).
        const std::size_t born_order[3]={1,0,2}, virtual_order[3]={2,1,0};
        for(std::size_t o=0;o<3;++o) {
            close_value(born[o],expected_born[born_order[o]]);
            for(std::size_t e=0;e<3;++e) close_value(virt[o*3+e],expected[virtual_order[o]][e]);
        }
    }
    for(std::size_t i=0;i<16;++i) momenta[16+i]=point[i]*1e200;
    auto partial=runtime.evaluate(momenta,3,{1e6},ids,true);
    assert(partial.successful_ids()==std::vector<std::uint64_t>({91,70}));
    assert(partial.statuses()==std::vector<std::uint32_t>({0,3,0}));
    assert(partial.shape().successful_points==2 && !partial.diagnostics_json().empty());
    const std::string verify=R"({"higher":[{"arithmetic":"double-double","precision":null}],"relative_tolerance":"1e-8","absolute_tolerance":"1e-9","verify_successful":true})";
    auto verified=runtime.evaluate(point,1,{1e6},{91},true,std::nullopt,verify);
    assert(verified.statuses()==std::vector<std::uint32_t>({RUSTICOL_VIRTUAL_VERIFIED}));
    auto empty=runtime.evaluate({},0,{}, {},true);
    assert(empty.born().empty() && empty.virtual_values().empty() && empty.statuses().empty());
    try { runtime.evaluate(point,1,{1e6},{91,70}); assert(false); }
    catch(const rusticol::VirtualError &) {}
    try { runtime.evaluate(point,1,{0}); assert(false); }
    catch(const rusticol::VirtualError &error) { assert(std::string(error.what()).size()>0); }
    assert(runtime.update_parameters(R"({"aS":["0.12","0"]})")>0);
    runtime.clear(); runtime.close();
    auto retained=std::move(inclusive);
    assert(retained.born()==born && retained.virtual_values()==virt);
    std::cout<<"native C++ "<<process<<" orders/poles/resolved/failures/verification/lifetime controls passed\n";
}
