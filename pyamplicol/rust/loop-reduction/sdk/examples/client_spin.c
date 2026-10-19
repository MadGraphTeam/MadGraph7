/* SPDX-License-Identifier: 0BSD */
/* Saved-source C/UMAMI layout, failure and ownership control. */
#include "pyamplicol_umami_virtual_v1.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void check(int status) {
    if(status) { fprintf(stderr,"%s\n",rusticol_virtual_last_error_message()); exit(1); }
}
static void near(rusticol_virtual_complex64 a,rusticol_virtual_complex64 b) {
    double scale=fmax(1.,fmax(hypot(a.real,a.imag),hypot(b.real,b.imag)));
    assert(fabs(a.real-b.real)<2e-8*scale && fabs(a.imag-b.imag)<2e-8*scale);
}
int main(int argc,char **argv) {
    assert(argc==2);
    rusticol_virtual_handle *runtime=NULL;
    check(rusticol_virtual_open(argv[1],NULL,NULL,"model",&runtime));
    char labels[128]; size_t needed=0;
    check(rusticol_virtual_spin_correlation_classes_json(runtime,labels,sizeof labels,&needed));
    assert(strcmp(labels,"[[3],[4],[3,4]]")==0);
    const double point[]={500,0,0,500,500,0,0,-500,500,300,0,400,500,-300,0,-400};
    size_t leg=3;
    rusticol_virtual_complex64 vector[]={{0,0},{0.8,0},{0,1},{-0.6,0}};
    double scale=1e6; uint64_t id=UINT64_MAX;
    rusticol_virtual_batch_request request={point,16,1,&id,&scale,1,0,NULL,NULL};
    rusticol_virtual_spin_sources sources={&leg,1,vector,4};
    rusticol_virtual_correlated_result *result=NULL;
    check(rusticol_virtual_evaluate_correlated_with_sources(runtime,&request,&sources,&result));
    rusticol_virtual_correlated_result_shape_info shape;
    check(rusticol_virtual_correlated_result_shape(result,&shape));
    assert(shape.axes.histories==3 && shape.axes.virtual_orders==1 && shape.axes.epsilon_count==3);
    rusticol_virtual_complex64 baseline[9];
    check(rusticol_virtual_correlated_result_copy_virtual(result,baseline,9));
    assert(fabs(baseline[3].imag)>1e-8);
    /* Input arrays may be changed immediately; result labels/values are owned. */
    leg=4; vector[1].real=999.;
    check(rusticol_virtual_correlated_result_spin_legs_json(result,labels,sizeof labels,&needed));
    assert(strcmp(labels,"[3]")==0);
    leg=3; vector[1].real=.8;
    double momenta[48]; memcpy(momenta,point,sizeof point); memset(momenta+16,0,sizeof point); memcpy(momenta+32,point,sizeof point);
    uint64_t ids[]={UINT64_MAX,9,0};
    rusticol_virtual_complex64 rows[12];
    for(size_t row=0;row<3;row++) for(size_t c=0;c<4;c++) {
        rusticol_virtual_complex64 v=vector[c];
        rows[4*row+c]=row==2 ? (rusticol_virtual_complex64){2*v.real+3*v.imag,2*v.imag-3*v.real} : v;
    }
    request=(rusticol_virtual_batch_request){momenta,48,3,ids,&scale,1,0,NULL,NULL};
    sources.values=rows; sources.value_count=12;
    rusticol_virtual_correlated_result *partial=NULL;
    check(rusticol_virtual_evaluate_correlated_with_sources(runtime,&request,&sources,&partial));
    uint32_t statuses[3]; rusticol_virtual_complex64 packed[18];
    check(rusticol_virtual_correlated_result_copy_statuses(partial,statuses,3));
    check(rusticol_virtual_correlated_result_copy_virtual(partial,packed,18));
    assert(statuses[0]==0 && statuses[1]==3 && statuses[2]==0);
    for(size_t k=0;k<9;k++) {
        near(packed[k],baseline[k]);
        near(packed[9+k],(rusticol_virtual_complex64){13*baseline[k].real,13*baseline[k].imag});
    }
    /* Real errors null the result before reading output; malformed source length. */
    rusticol_virtual_correlated_result *bad=(void*)(uintptr_t)1;
    sources.value_count=11;
    assert(rusticol_virtual_evaluate_correlated_with_sources(runtime,&request,&sources,&bad)!=0 && !bad);
    sources.value_count=12;
    enum {stride=5,offset=1,count=3};
    double soa[stride*16], scales[stride]; uint64_t event_ids[stride]={8,UINT64_MAX,9,0,7};
    rusticol_virtual_complex64 source_soa[stride*4], output[stride*9]; uint32_t status_out[stride];
    for(size_t i=0;i<stride*16;i++) soa[i]=NAN;
    for(size_t i=0;i<stride*4;i++) source_soa[i]=(rusticol_virtual_complex64){NAN,NAN};
    for(size_t i=0;i<stride;i++) { scales[i]=NAN; status_out[i]=99; }
    for(size_t i=0;i<stride*9;i++) output[i]=(rusticol_virtual_complex64){73,-19};
    for(size_t row=0;row<count;row++) {
        size_t event=offset+row; scales[event]=scale;
        for(size_t l=0;l<4;l++) for(size_t c=0;c<4;c++) soa[event+stride*(l+4*c)]=momenta[row*16+4*l+c];
        for(size_t c=0;c<4;c++) source_soa[event+stride*c]=rows[row*4+c];
    }
    int ik[]={UMAMI_IN_MOMENTA,PYAMPLICOL_UMAMI_V1_IN_MU_R2,PYAMPLICOL_UMAMI_V1_IN_POINT_ID,
        PYAMPLICOL_UMAMI_V1_IN_SPIN_LEGS,PYAMPLICOL_UMAMI_V1_IN_SPIN_VECTORS};
    const void *inputs[]={soa,scales,event_ids,&leg,source_soa}; size_t ic[]={stride*16,stride,stride,1,stride*4};
    int ok[]={PYAMPLICOL_UMAMI_V1_OUT_CORRELATED_VIRTUAL,PYAMPLICOL_UMAMI_V1_OUT_STATUS};
    void *outputs[]={output,status_out}; size_t oc[]={stride*9,stride};
    const int *keys=NULL; size_t nkeys=0;
    assert(pyamplicol_umami_virtual_v1_supported_correlated_inputs(&keys,&nkeys)==UMAMI_SUCCESS && nkeys==7);
    rusticol_virtual_correlated_result *details=NULL;
    assert(pyamplicol_umami_virtual_v1_evaluate_correlated(runtime,count,stride,offset,5,ik,inputs,ic,
        2,ok,outputs,oc,NULL,&details)==UMAMI_SUCCESS);
    for(size_t e=0;e<3;e++) for(size_t h=0;h<3;h++) for(size_t row=0;row<stride;row++) {
        rusticol_virtual_complex64 value=output[row+stride*(h+3*e)];
        if(row==1 || row==3) near(value,packed[(row==1?0:9)+3*h+e]);
        else assert(value.real==73 && value.imag==-19);
    }
    assert(status_out[0]==99 && status_out[1]==0 && status_out[2]==3 && status_out[3]==0 && status_out[4]==99);
    check(rusticol_virtual_correlated_result_free(details)); details=NULL;
    rusticol_virtual_complex64 unchanged[stride*9]; memcpy(unchanged,output,sizeof output);
    uint32_t unchanged_status[stride]; memcpy(unchanged_status,status_out,sizeof status_out);
    ic[4]--;
    assert(pyamplicol_umami_virtual_v1_evaluate_correlated(runtime,count,stride,offset,5,ik,inputs,ic,
        2,ok,outputs,oc,NULL,&details)!=UMAMI_SUCCESS && !details);
    assert(memcmp(unchanged,output,sizeof output)==0 && memcmp(unchanged_status,status_out,sizeof status_out)==0);
    ic[4]++;
    /* Broadcast source has no event stride. */
    ik[4]=PYAMPLICOL_UMAMI_V1_IN_SPIN_VECTORS_SCALAR; inputs[4]=vector; ic[4]=4;
    assert(pyamplicol_umami_virtual_v1_evaluate_correlated(runtime,1,stride,3,5,ik,inputs,ic,
        2,ok,outputs,oc,NULL,NULL)==UMAMI_SUCCESS);
    for(size_t e=0;e<3;e++) for(size_t h=0;h<3;h++) near(output[3+stride*(h+3*e)],baseline[3*h+e]);
    /* Zero-row class selection and ordinary-call rejection keep distinct semantics. */
    int source_keys[]={PYAMPLICOL_UMAMI_V1_IN_SPIN_LEGS,PYAMPLICOL_UMAMI_V1_IN_SPIN_VECTORS};
    const void *empty_sources[]={&leg,NULL}; const size_t empty_capacities[]={1,0};
    assert(pyamplicol_umami_virtual_v1_evaluate_correlated(runtime,0,0,0,2,source_keys,empty_sources,empty_capacities,
        0,NULL,NULL,NULL,NULL,&details)==UMAMI_SUCCESS);
    check(rusticol_virtual_correlated_result_spin_legs_json(details,labels,sizeof labels,&needed)); assert(strcmp(labels,"[3]")==0);
    check(rusticol_virtual_correlated_result_free(details));
    assert(pyamplicol_umami_virtual_v1_evaluate(runtime,0,0,0,2,source_keys,empty_sources,empty_capacities,
        0,NULL,NULL,NULL,NULL,NULL)==UMAMI_ERROR_UNSUPPORTED_INPUT);
    check(rusticol_virtual_clear(runtime)); check(rusticol_virtual_free(runtime));
    check(rusticol_virtual_correlated_result_spin_legs_json(result,labels,sizeof labels,&needed)); assert(strcmp(labels,"[3]")==0);
    check(rusticol_virtual_correlated_result_copy_virtual(result,packed,9));
    for(size_t k=0;k<9;k++) near(packed[k],baseline[k]);
    check(rusticol_virtual_correlated_result_free(result)); check(rusticol_virtual_correlated_result_free(partial));
    puts("C/UMAMI joint-source values, strides, failures, empty class and ownership passed");
}
