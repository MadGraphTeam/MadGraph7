/* SPDX-License-Identifier: 0BSD */
/* Native C and UMAMI grouped complex transport regression. */
#include "rusticol_virtual.h"
#include "pyamplicol_umami_virtual_v1.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void check(int status) { if(status) { fprintf(stderr,"%s\n",rusticol_virtual_last_error_message()); exit(1); } }
static void near(double a,double b) { assert(fabs(a-b)<1e-8*fmax(1.,fmax(fabs(a),fabs(b)))); }
int main(int argc,char **argv) {
    assert(argc==2);
    rusticol_virtual_handle *runtime=NULL;
    check(rusticol_virtual_open(argv[1],NULL,NULL,"model",&runtime));
    rusticol_virtual_correlated_shape_info shape;
    check(rusticol_virtual_correlated_shape(runtime,&shape));
    assert(shape.histories==4 && shape.born_orders==1 && shape.virtual_orders==1 && shape.epsilon_count==3);
    char histories[256]; size_t needed;
    check(rusticol_virtual_correlation_histories_json(runtime,histories,sizeof histories,&needed));
    assert(strcmp(histories,"[\"identity\",\"casimir\",\"ordered\",\"swapped\"]")==0);
    const double point[]={500,0,0,500,500,0,0,-500,500,300,0,400,500,-300,0,-400};
    double momenta[48]; memcpy(momenta,point,sizeof point); memset(momenta+16,0,sizeof point); memcpy(momenta+32,point,sizeof point);
    const uint64_t ids[]={UINT64_MAX,91,0}; double scale=1e6;
    rusticol_virtual_batch_request request={momenta,48,3,ids,&scale,1,0,NULL,NULL};
    rusticol_virtual_correlated_result *result=NULL;
    check(rusticol_virtual_evaluate_correlated(runtime,&request,&result));
    rusticol_virtual_correlated_result_shape_info axes;
    check(rusticol_virtual_correlated_result_shape(result,&axes));
    assert(axes.requested_points==3 && axes.successful_points==2 && axes.virtual_values==24);
    uint32_t statuses[3]; uint64_t successful[2];
    check(rusticol_virtual_correlated_result_copy_statuses(result,statuses,3));
    check(rusticol_virtual_correlated_result_copy_successful_ids(result,successful,2));
    assert(statuses[0]==0 && statuses[1]==3 && statuses[2]==0 && successful[0]==UINT64_MAX && successful[1]==0);
    rusticol_virtual_complex64 values[24];
    for(size_t i=0;i<24;i++) values[i]=(rusticol_virtual_complex64){73.,-19.};
    assert(rusticol_virtual_correlated_result_copy_virtual(result,values,23)==RUSTICOL_VIRTUAL_BUFFER_TOO_SMALL);
    for(size_t i=0;i<24;i++) assert(values[i].real==73. && values[i].imag==-19.);
    check(rusticol_virtual_correlated_result_copy_virtual(result,values,24));
    assert(fabs(values[6].imag)>1e-8);
    for(size_t e=0;e<3;e++) { near(values[6+e].real,values[9+e].real); near(values[6+e].imag,-values[9+e].imag); }
    request.resolved=1;
    rusticol_virtual_correlated_result *rejected=NULL;
    assert(rusticol_virtual_evaluate_correlated(runtime,&request,&rejected)==RUSTICOL_VIRTUAL_INVALID_ARGUMENT && !rejected);
    request.resolved=0;
    /* UMAMI event-contiguous layout; failure and outside-slice sentinels survive. */
    enum {stride=5}; double soa[5*16]={0},scales[5]={1e6,1e6,1e6,1e6,1e6};
    uint64_t event_ids[5]={3,UINT64_MAX,91,0,4};
    for(size_t p=0;p<3;p++) for(size_t l=0;l<4;l++) for(size_t c=0;c<4;c++)
        soa[p+1+stride*(l+4*c)]=momenta[p*16+4*l+c];
    rusticol_virtual_complex64 out[5*12]; uint32_t status_out[5];
    for(size_t i=0;i<60;i++) out[i]=(rusticol_virtual_complex64){73.,-19.};
    for(size_t i=0;i<5;i++) status_out[i]=99;
    int ik[]={UMAMI_IN_MOMENTA,PYAMPLICOL_UMAMI_V1_IN_MU_R2,PYAMPLICOL_UMAMI_V1_IN_POINT_ID};
    const void *inputs[]={soa,scales,event_ids}; const size_t ic[]={80,5,5};
    int ok[]={PYAMPLICOL_UMAMI_V1_OUT_CORRELATED_VIRTUAL,PYAMPLICOL_UMAMI_V1_OUT_STATUS};
    void *outputs[]={out,status_out}; size_t oc[]={60,5};
    rusticol_virtual_correlated_result *diagnostics=NULL;
    assert(pyamplicol_umami_virtual_v1_evaluate_correlated(runtime,3,5,1,3,ik,inputs,ic,2,ok,outputs,oc,NULL,&diagnostics)==UMAMI_SUCCESS);
    assert(status_out[0]==99 && status_out[1]==0 && status_out[2]==3 && status_out[3]==0 && status_out[4]==99);
    for(size_t h=0;h<4;h++) for(size_t e=0;e<3;e++) for(size_t p=0;p<5;p++) {
        rusticol_virtual_complex64 x=out[p+stride*(h+4*e)];
        if(p==1 || p==3) { near(x.real,values[(p==1?0:12)+h*3+e].real); near(x.imag,values[(p==1?0:12)+h*3+e].imag); }
        else assert(x.real==73. && x.imag==-19.);
    }
    rusticol_virtual_complex64 before[60]; memcpy(before,out,sizeof out);
    uint32_t before_status[5]; memcpy(before_status,status_out,sizeof status_out);
    oc[0]=59;
    assert(pyamplicol_umami_virtual_v1_evaluate_correlated(runtime,3,5,1,3,ik,inputs,ic,2,ok,outputs,oc,NULL,NULL)!=UMAMI_SUCCESS);
    assert(memcmp(before,out,sizeof out)==0 && memcmp(before_status,status_out,sizeof status_out)==0);
    assert(pyamplicol_umami_virtual_v1_evaluate_correlated(runtime,0,5,5,0,NULL,NULL,NULL,0,NULL,NULL,NULL,NULL,NULL)==UMAMI_SUCCESS);
    check(rusticol_virtual_correlated_result_free(diagnostics));
    /* Shared transport retains the ordinary real interface and layout. */
    double real_out[15]; for(size_t i=0;i<15;i++) real_out[i]=73.;
    int real_keys[]={PYAMPLICOL_UMAMI_V1_OUT_VIRTUAL,PYAMPLICOL_UMAMI_V1_OUT_STATUS};
    void *real_outputs[]={real_out,status_out}; const size_t real_capacities[]={15,5};
    assert(pyamplicol_umami_virtual_v1_evaluate(runtime,3,5,1,3,ik,inputs,ic,2,real_keys,real_outputs,real_capacities,NULL,NULL)==UMAMI_SUCCESS);
    for(size_t e=0;e<3;e++) for(size_t p=0;p<5;p++) {
        if(p==1 || p==3) near(real_out[p+5*e],values[(p==1?0:12)+e].real);
        else assert(real_out[p+5*e]==73.);
    }
    check(rusticol_virtual_free(runtime));
    check(rusticol_virtual_correlated_result_histories_json(result,histories,sizeof histories,&needed));
    check(rusticol_virtual_correlated_result_free(result));
    puts("C/UMAMI grouped complex axes, sparse IDs, partial rows and sentinels passed");
}
